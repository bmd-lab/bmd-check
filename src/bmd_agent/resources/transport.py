"""Observation transports for BMD Agent's fixed read-only operations.

Agent's acquisition code expresses every observational read as one value from
the closed operation set in ``observations.py``. A transport turns that
operation into exactly one process invocation; neither transport accepts
command text, a preconstructed command vector, or SSH options from callers.

* SSH (``ssh_host`` is an SSH host alias): the transport builds the whole
  ``ssh`` argv itself. ``BatchMode=yes`` is always the first option, and the
  only other options are Agent-owned: ``ConnectTimeout`` from typed
  configuration and, inside a reusable session, the session's own
  ``ControlMaster``/``ControlPersist``/``ControlPath``. The operation's argv
  is shell-quoted for the remote login shell.
* Local (``ssh_host = "local"``): for running Agent on the POSIX machine it
  observes. The operation's argv runs directly with ``subprocess.run`` and no
  shell; ``ssh`` is never executed. ``"local"`` is a reserved transport
  selector, not a host name, and is unavailable on non-POSIX platforms.

Invocations are list-compatible so injected test runners and the profiler can
inspect them, but the default executor re-renders each invocation from its
typed fields and refuses anything else.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable

from bmd_agent.resources.observations import ObservationOperation, require_operation

LOCAL_TRANSPORT = "local"
SSH_BATCH_MODE_OPTION = "BatchMode=yes"
SSH_CONTROL_PERSIST_SECONDS = 60
MAX_SSH_CONNECT_TIMEOUT_SECONDS = 3600

# Same character set as configuration, but an option-looking leading "-" is
# never a destination.
_SSH_HOST_RE = re.compile(r"^[A-Za-z0-9_.@:][A-Za-z0-9_.@:-]*$")
_EXECUTOR_KWARGS = frozenset({"capture_output", "check", "timeout", "text", "input"})

Runner = Callable[..., subprocess.CompletedProcess[Any]]


class TransportError(ValueError):
    """An observation could not be rendered or executed by a fixed transport."""


def is_local_transport(ssh_host: object) -> bool:
    return ssh_host == LOCAL_TRANSPORT


def local_transport_supported() -> bool:
    return os.name == "posix"


LOCAL_TRANSPORT_UNSUPPORTED_MESSAGE = (
    'ssh_host = "local" selects BMD Agent\'s local observation transport, which '
    "requires a POSIX system (such as the cluster login node); it is not "
    "supported on this platform. Configure an SSH host alias instead."
)


def require_local_transport_supported() -> None:
    if not local_transport_supported():
        raise TransportError(LOCAL_TRANSPORT_UNSUPPORTED_MESSAGE)


def validate_ssh_host(ssh_host: object) -> str:
    if not isinstance(ssh_host, str) or not _SSH_HOST_RE.fullmatch(ssh_host):
        raise TransportError("SSH host alias contains unsafe characters")
    if is_local_transport(ssh_host):
        raise TransportError('"local" is a transport selector, not an SSH host')
    return ssh_host


def _validate_connect_timeout(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TransportError("SSH connection timeout must be an integer number of seconds")
    if value <= 0 or value > MAX_SSH_CONNECT_TIMEOUT_SECONDS:
        raise TransportError(
            f"SSH connection timeout must be between 1 and {MAX_SSH_CONNECT_TIMEOUT_SECONDS} seconds"
        )
    return value


def _validate_control_path(value: object) -> Path:
    if not isinstance(value, Path) or not value.is_absolute() or "\x00" in str(value):
        raise TransportError("SSH control path must be an absolute session-owned path")
    return value


class ObservationInvocation(list[str]):
    """A rendered invocation of one fixed operation. Built only by transports."""

    transport = ""
    operation: ObservationOperation | None = None

    def render(self) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def stdin(self) -> bytes | None:
        return None if self.operation is None else self.operation.stdin

    @property
    def observation_text(self) -> str | None:
        """Shell-quoted operation text, used for profiling classification."""

        return None if self.operation is None else self.operation.remote_command()

    def _freeze(self) -> None:
        super().__init__(self.render())


class LocalInvocation(ObservationInvocation):
    transport = LOCAL_TRANSPORT

    def __init__(self, operation: ObservationOperation) -> None:
        require_local_transport_supported()
        self.operation = require_operation(operation)
        self._freeze()

    def render(self) -> list[str]:
        assert self.operation is not None
        return list(self.operation.argv())


class SshInvocation(ObservationInvocation):
    transport = "ssh"

    def __init__(
        self,
        operation: ObservationOperation,
        *,
        host: str,
        connect_timeout: int | None = None,
        control_path: Path | None = None,
        opens_connection: bool = True,
    ) -> None:
        self.operation = require_operation(operation)
        self.host = validate_ssh_host(host)
        self.connect_timeout = _validate_connect_timeout(connect_timeout)
        self.control_path = None if control_path is None else _validate_control_path(control_path)
        self.control_exit = False
        self.ssh_opens_connection = bool(opens_connection)
        self.ssh_exec_channel = True
        self.ssh_control_operation = False
        self._freeze()

    @classmethod
    def session_exit(cls, *, host: str, control_path: Path) -> "SshInvocation":
        """The session's own ``-O exit`` for its control master."""

        invocation = cls.__new__(cls)
        invocation.operation = None
        invocation.host = validate_ssh_host(host)
        invocation.connect_timeout = None
        invocation.control_path = _validate_control_path(control_path)
        invocation.control_exit = True
        invocation.ssh_opens_connection = False
        invocation.ssh_exec_channel = False
        invocation.ssh_control_operation = True
        invocation._freeze()
        return invocation

    def with_session_control(self, control_path: Path, *, opens_connection: bool) -> "SshInvocation":
        if self.operation is None or self.control_path is not None:
            raise TransportError("SSH invocation already carries session control options")
        return SshInvocation(
            self.operation,
            host=self.host,
            connect_timeout=self.connect_timeout,
            control_path=control_path,
            opens_connection=opens_connection,
        )

    def render(self) -> list[str]:
        argv = ["ssh", "-o", SSH_BATCH_MODE_OPTION]
        if self.control_exit:
            assert self.control_path is not None
            return [*argv, "-S", str(self.control_path), "-O", "exit", self.host]
        if self.control_path is not None:
            argv += [
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPersist={SSH_CONTROL_PERSIST_SECONDS}",
                "-o",
                f"ControlPath={self.control_path}",
            ]
        if self.connect_timeout is not None:
            argv += ["-o", f"ConnectTimeout={self.connect_timeout}"]
        assert self.operation is not None
        return [*argv, self.host, self.operation.remote_command()]


def build_invocation(
    ssh_host: str,
    operation: ObservationOperation,
    *,
    connect_timeout: int | None = None,
) -> ObservationInvocation:
    """Render one fixed operation for the configured transport."""

    if is_local_transport(ssh_host):
        return LocalInvocation(operation)
    return SshInvocation(operation, host=ssh_host, connect_timeout=connect_timeout)


def run_invocation(command: object, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Default executor: runs only transport-rendered fixed invocations."""

    if not isinstance(command, ObservationInvocation):
        raise TransportError("BMD Agent executes only fixed observational operations")
    argv = command.render()
    if list(command) != argv:
        raise TransportError("observational invocation was modified after rendering")
    unexpected = set(kwargs) - _EXECUTOR_KWARGS
    if unexpected:
        raise TransportError(
            "observational transports do not accept: " + ", ".join(sorted(unexpected))
        )
    if kwargs.get("input") != command.stdin:
        raise TransportError("observational input must be the operation's own fixed input")
    if command.stdin is None:
        kwargs.pop("input", None)
    return subprocess.run(argv, **kwargs)


def observe(
    ssh_host: str,
    operation: ObservationOperation,
    *,
    runner: Runner = run_invocation,
    connect_timeout: int | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[Any]:
    """Run one fixed operation through the configured transport."""

    invocation = build_invocation(ssh_host, operation, connect_timeout=connect_timeout)
    if "input" in kwargs:
        raise TransportError("observational input is supplied only by the operation")
    if invocation.stdin is not None:
        kwargs["input"] = invocation.stdin
    return runner(invocation, **kwargs)


class LocalObservationSession:
    """Drop-in for ReusableSshSession when observing this machine.

    Same ``runner(role)`` and context-manager interface; there is no
    connection to open or close. Only local invocations are accepted.
    """

    def __init__(self, *, runner: Runner = run_invocation) -> None:
        require_local_transport_supported()
        self._runner = runner
        self.ssh_host = LOCAL_TRANSPORT

    def __enter__(self) -> "LocalObservationSession":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def close(self) -> None:
        return None

    def runner(self, role: str) -> Runner:
        from bmd_agent.profiling import profiled_runner

        profiled = profiled_runner(self._runner, role=role)

        def run(command: object, **kwargs: Any):
            if not isinstance(command, LocalInvocation):
                raise TransportError("local observation session accepts only local operations")
            return profiled(command, **kwargs)

        return run
