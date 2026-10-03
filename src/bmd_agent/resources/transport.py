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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable
from weakref import ReferenceType, ref

from bmd_agent.resources.observations import (
    ObservationOperation,
    operation_argv,
    operation_remote_command,
    operation_stdin,
    snapshot_operation,
)

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
    return type(ssh_host) is str and ssh_host == LOCAL_TRANSPORT


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
    if type(ssh_host) is not str or not _SSH_HOST_RE.fullmatch(ssh_host):
        raise TransportError("SSH host alias contains unsafe characters")
    if is_local_transport(ssh_host):
        raise TransportError('"local" is a transport selector, not an SSH host')
    return ssh_host


def _validate_connect_timeout(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TransportError("SSH connection timeout must be an integer number of seconds")
    if value <= 0 or value > MAX_SSH_CONNECT_TIMEOUT_SECONDS:
        raise TransportError(
            f"SSH connection timeout must be between 1 and {MAX_SSH_CONNECT_TIMEOUT_SECONDS} seconds"
        )
    return value


def _validate_control_path(value: object) -> Path:
    if type(value) is not type(Path()) or not value.is_absolute() or "\x00" in str(value):
        raise TransportError("SSH control path must be an absolute session-owned path")
    return value


@dataclass(frozen=True, slots=True)
class _InvocationState:
    transport: str
    operation: ObservationOperation | None
    argv: tuple[str, ...]
    stdin: bytes | None
    host: str | None = None
    connect_timeout: int | None = None
    control_path: Path | None = None
    control_exit: bool = False
    opens_connection: bool = False


def _state_store():
    states: dict[int, tuple[ReferenceType[object], _InvocationState]] = {}

    def seal(invocation: object, state: _InvocationState) -> None:
        key = id(invocation)
        existing = states.get(key)
        if existing is not None and existing[0]() is invocation:
            raise AttributeError("observational invocation is already sealed")
        operation = (
            None if state.operation is None else snapshot_operation(state.operation)
        )
        sealed_state = _InvocationState(
            transport=state.transport,
            operation=operation,
            argv=tuple(state.argv),
            stdin=state.stdin,
            host=state.host,
            connect_timeout=state.connect_timeout,
            control_path=state.control_path,
            control_exit=state.control_exit,
            opens_connection=state.opens_connection,
        )

        def discard(reference: object, *, identity: int = key) -> None:
            current = states.get(identity)
            if current is not None and current[0] is reference:
                del states[identity]

        states[key] = (ref(invocation, discard), sealed_state)

    def get(invocation: object) -> _InvocationState:
        entry = states.get(id(invocation))
        if entry is None or entry[0]() is not invocation:
            raise TransportError("observational invocation is not sealed")
        state = entry[1]
        operation = (
            None if state.operation is None else snapshot_operation(state.operation)
        )
        return _InvocationState(
            transport=state.transport,
            operation=operation,
            argv=tuple(state.argv),
            stdin=state.stdin,
            host=state.host,
            connect_timeout=state.connect_timeout,
            control_path=state.control_path,
            control_exit=state.control_exit,
            opens_connection=state.opens_connection,
        )

    return seal, get


_seal_invocation, _invocation_state = _state_store()


class ObservationInvocation(Sequence[str]):
    """Immutable sequence view of one transport-rendered operation.

    The instance deliberately has no execution-bearing attributes. Its sealed
    state is held outside the object so even ``object.__setattr__`` cannot
    synchronize mutated fields with a caller-supplied argv.
    """

    __slots__ = ("__weakref__",)
    transport = ""

    def __len__(self) -> int:
        return len(_invocation_state(self).argv)

    def __getitem__(self, index: int | slice) -> str | list[str]:
        value = _invocation_state(self).argv[index]
        return list(value) if isinstance(index, slice) else value

    def __iter__(self) -> Iterator[str]:
        return iter(_invocation_state(self).argv)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (ObservationInvocation, list, tuple)):
            return tuple(self) == tuple(other)
        return NotImplemented

    __hash__ = None

    def render(self) -> list[str]:
        """Return a copy for diagnostics and injected test runners only."""

        return list(_invocation_state(self).argv)

    @property
    def stdin(self) -> bytes | None:
        return _invocation_state(self).stdin

    @property
    def operation(self) -> ObservationOperation | None:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def observation_text(self) -> str | None:
        """Shell-quoted operation text, used for profiling classification."""

        operation = self.operation
        return None if operation is None else operation_remote_command(operation)


class LocalInvocation(ObservationInvocation):
    __slots__ = ()
    transport = LOCAL_TRANSPORT

    def __init__(self, operation: ObservationOperation) -> None:
        require_local_transport_supported()
        validated = snapshot_operation(operation)
        _seal_invocation(
            self,
            _InvocationState(
                transport=LOCAL_TRANSPORT,
                operation=validated,
                argv=operation_argv(validated),
                stdin=operation_stdin(validated),
            ),
        )

    @property
    def operation(self) -> ObservationOperation:
        operation = _invocation_state(self).operation
        if operation is None:  # pragma: no cover - sealed constructor invariant
            raise TransportError("local invocation has no operation")
        return snapshot_operation(operation)


class SshInvocation(ObservationInvocation):
    __slots__ = ()
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
        if type(opens_connection) is not bool:
            raise TransportError("SSH connection-state metadata must be boolean")
        validated_operation = snapshot_operation(operation)
        validated_host = validate_ssh_host(host)
        validated_timeout = _validate_connect_timeout(connect_timeout)
        validated_control = (
            None if control_path is None else _validate_control_path(control_path)
        )
        argv, stdin = _render_ssh_invocation(
            operation=validated_operation,
            host=validated_host,
            connect_timeout=validated_timeout,
            control_path=validated_control,
            control_exit=False,
            opens_connection=opens_connection,
        )
        _seal_invocation(
            self,
            _InvocationState(
                transport="ssh",
                operation=validated_operation,
                argv=tuple(argv),
                stdin=stdin,
                host=validated_host,
                connect_timeout=validated_timeout,
                control_path=validated_control,
                opens_connection=opens_connection,
            ),
        )

    @classmethod
    def session_exit(cls, *, host: str, control_path: Path) -> "SshInvocation":
        """The session's own ``-O exit`` for its control master."""

        invocation = SshInvocation.__new__(SshInvocation)
        validated_host = validate_ssh_host(host)
        validated_control = _validate_control_path(control_path)
        argv, stdin = _render_ssh_invocation(
            operation=None,
            host=validated_host,
            connect_timeout=None,
            control_path=validated_control,
            control_exit=True,
            opens_connection=False,
        )
        _seal_invocation(
            invocation,
            _InvocationState(
                transport="ssh",
                operation=None,
                argv=tuple(argv),
                stdin=stdin,
                host=validated_host,
                control_path=validated_control,
                control_exit=True,
            ),
        )
        return invocation

    @property
    def operation(self) -> ObservationOperation | None:
        operation = _invocation_state(self).operation
        return None if operation is None else snapshot_operation(operation)

    @property
    def host(self) -> str:
        host = _invocation_state(self).host
        if host is None:  # pragma: no cover - sealed constructor invariant
            raise TransportError("SSH invocation has no host")
        return host

    @property
    def connect_timeout(self) -> int | None:
        return _invocation_state(self).connect_timeout

    @property
    def control_path(self) -> Path | None:
        return _invocation_state(self).control_path

    @property
    def control_exit(self) -> bool:
        return _invocation_state(self).control_exit

    @property
    def ssh_opens_connection(self) -> bool:
        return _invocation_state(self).opens_connection

    @property
    def ssh_exec_channel(self) -> bool:
        return not self.control_exit

    @property
    def ssh_control_operation(self) -> bool:
        return self.control_exit

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

    if type(command) is LocalInvocation:
        argv, fixed_input = _reconstruct_local_invocation(command)
    elif type(command) is SshInvocation:
        argv, fixed_input = _reconstruct_ssh_invocation(command)
    else:
        raise TransportError("BMD Agent executes only fixed observational operations")
    if tuple(command) != tuple(argv):
        raise TransportError("observational invocation was modified after rendering")
    unexpected = set(kwargs) - _EXECUTOR_KWARGS
    if unexpected:
        raise TransportError(
            "observational transports do not accept: " + ", ".join(sorted(unexpected))
        )
    if kwargs.get("input") != fixed_input:
        raise TransportError("observational input must be the operation's own fixed input")
    if fixed_input is None:
        kwargs.pop("input", None)
    return subprocess.run(argv, shell=False, **kwargs)


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
            if type(command) is not LocalInvocation:
                raise TransportError("local observation session accepts only local operations")
            return profiled(command, **kwargs)

        return run


def _reconstruct_local_invocation(
    command: LocalInvocation,
) -> tuple[list[str], bytes | None]:
    require_local_transport_supported()
    state = _invocation_state(command)
    if state.transport != LOCAL_TRANSPORT or state.operation is None:
        raise TransportError("local invocation has inconsistent sealed state")
    operation = snapshot_operation(state.operation)
    argv = list(operation_argv(operation))
    stdin = operation_stdin(operation)
    if state.argv != tuple(argv) or state.stdin != stdin:
        raise TransportError("local invocation has inconsistent sealed state")
    return argv, stdin


def _reconstruct_ssh_invocation(
    command: SshInvocation,
) -> tuple[list[str], bytes | None]:
    state = _invocation_state(command)
    if state.transport != "ssh":
        raise TransportError("SSH invocation has inconsistent sealed state")
    argv, stdin = _render_ssh_invocation(
        operation=state.operation,
        host=state.host,
        connect_timeout=state.connect_timeout,
        control_path=state.control_path,
        control_exit=state.control_exit,
        opens_connection=state.opens_connection,
    )
    if state.argv != tuple(argv) or state.stdin != stdin:
        raise TransportError("SSH invocation has inconsistent sealed state")
    return argv, stdin


def _render_ssh_invocation(
    *,
    operation: ObservationOperation | None,
    host: object,
    connect_timeout: object,
    control_path: object,
    control_exit: object,
    opens_connection: object,
) -> tuple[list[str], bytes | None]:
    """Validate immutable fields and reconstruct one exact SSH invocation."""

    validated_host = validate_ssh_host(host)
    validated_timeout = _validate_connect_timeout(connect_timeout)
    validated_control = (
        None if control_path is None else _validate_control_path(control_path)
    )
    if type(control_exit) is not bool or type(opens_connection) is not bool:
        raise TransportError("SSH invocation metadata must be boolean")

    argv = ["ssh", "-o", SSH_BATCH_MODE_OPTION]
    if control_exit:
        if operation is not None or validated_timeout is not None or validated_control is None:
            raise TransportError("SSH control invocation has inconsistent fields")
        if opens_connection:
            raise TransportError("SSH control invocation has inconsistent fields")
        return [*argv, "-S", str(validated_control), "-O", "exit", validated_host], None

    validated_operation = snapshot_operation(operation)
    if validated_control is not None:
        argv += [
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPersist={SSH_CONTROL_PERSIST_SECONDS}",
            "-o",
            f"ControlPath={validated_control}",
        ]
    if validated_timeout is not None:
        argv += ["-o", f"ConnectTimeout={validated_timeout}"]
    argv += [validated_host, operation_remote_command(validated_operation)]
    return argv, operation_stdin(validated_operation)
