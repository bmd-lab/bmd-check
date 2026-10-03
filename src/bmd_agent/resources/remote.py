from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from bmd_agent.profiling import profiled_runner
from bmd_agent.resources.transport import (
    SshInvocation,
    TransportError,
    run_invocation,
    validate_ssh_host,
)


Runner = Callable[..., subprocess.CompletedProcess[Any]]


class ReusableSshSession:
    """Reuse one invocation-scoped OpenSSH connection for fixed remote commands."""

    def __init__(
        self,
        ssh_host: str,
        *,
        runner: Runner = run_invocation,
        close_timeout: float = 10,
        multiplex: bool | None = None,
    ) -> None:
        if close_timeout <= 0:
            raise ValueError("SSH session close timeout must be positive")
        self.ssh_host = validate_ssh_host(ssh_host)
        self._runner = runner
        self._close_timeout = close_timeout
        self._multiplex = os.name == "posix" if multiplex is None else multiplex
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._control_path: Path | None = None
        self._connected = False
        self._closed = False

    def __enter__(self) -> ReusableSshSession:
        if self._closed:
            raise RuntimeError("SSH session is already closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def runner(self, role: str) -> Runner:
        """Return a role-labelled callable compatible with existing adapters."""

        def run(command: object, **kwargs: object):
            return self._run(command, role=role, **kwargs)

        return run

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._multiplex and self._control_path is not None and (
                self._connected or self._control_path.exists()
            ):
                command = SshInvocation.session_exit(
                    host=self.ssh_host,
                    control_path=self._control_path,
                )
                try:
                    profiled_runner(self._runner, role="ssh_control")(
                        command,
                        capture_output=True,
                        check=False,
                        timeout=self._close_timeout,
                    )
                except (OSError, subprocess.SubprocessError):
                    pass
        finally:
            self._connected = False
            self._closed = True
            if self._temporary_directory is not None:
                self._temporary_directory.cleanup()
                self._temporary_directory = None
                self._control_path = None

    def _run(self, command: object, *, role: str, **kwargs: object):
        if self._closed:
            raise RuntimeError("SSH session is closed")
        invocation = _validated_invocation(command, self.ssh_host, kwargs)
        if not self._multiplex:
            return profiled_runner(self._runner, role=role)(invocation, **kwargs)

        control_path = self._ensure_control_path()
        if self._connected and not control_path.exists():
            self._connected = False
        opens_connection = not self._connected
        multiplexed = invocation.with_session_control(
            control_path,
            opens_connection=opens_connection,
        )
        runner = profiled_runner(self._runner, role=role)
        try:
            result = runner(multiplexed, **kwargs)
        except subprocess.CalledProcessError as exc:
            self._update_connection_state(opens_connection, exc.returncode)
            raise
        except subprocess.TimeoutExpired:
            if control_path.exists():
                self._connected = True
            raise
        except OSError:
            self._discard_control_socket()
            raise
        else:
            self._update_connection_state(
                opens_connection,
                getattr(result, "returncode", 0),
            )
            return result

    def _ensure_control_path(self) -> Path:
        if self._control_path is None:
            temporary_root = "/tmp" if os.name == "posix" else None
            self._temporary_directory = tempfile.TemporaryDirectory(
                prefix="ba-ssh-",
                dir=temporary_root,
            )
            self._control_path = Path(self._temporary_directory.name) / "control"
        return self._control_path

    def _update_connection_state(self, opens_connection: bool, returncode: object) -> None:
        if returncode == 255:
            self._discard_control_socket()
        elif opens_connection:
            self._connected = True

    def _discard_control_socket(self) -> None:
        self._connected = False
        if self._control_path is not None:
            self._control_path.unlink(missing_ok=True)


def _validated_invocation(
    command: object,
    ssh_host: str,
    kwargs: dict[str, object],
) -> SshInvocation:
    if kwargs.get("shell") is True:
        raise TransportError("reusable SSH session does not permit shell=True")
    if not isinstance(command, SshInvocation) or command.operation is None:
        raise TransportError("reusable SSH session accepts only fixed observational operations")
    if command.host != ssh_host:
        raise TransportError("reusable SSH session received an operation for another host")
    if command.control_path is not None:
        raise TransportError("callers may not supply reusable SSH control options")
    return command
