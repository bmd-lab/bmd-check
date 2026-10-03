from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from bmd_agent.profiling import profiled_runner
from bmd_agent.resources.transport import run_ssh_command


Runner = Callable[..., subprocess.CompletedProcess[Any]]

_CONTROL_PERSIST_SECONDS = 60


class _SshCommand(list[str]):
    """List-compatible command carrying transport facts for the profiler."""

    def __init__(
        self,
        parts: Sequence[str],
        *,
        opens_connection: bool,
        exec_channel: bool,
        control_operation: bool = False,
    ) -> None:
        super().__init__(parts)
        self.ssh_opens_connection = opens_connection
        self.ssh_exec_channel = exec_channel
        self.ssh_control_operation = control_operation


class ReusableSshSession:
    """Reuse one invocation-scoped OpenSSH connection for fixed remote commands."""

    def __init__(
        self,
        ssh_host: str,
        *,
        runner: Runner = run_ssh_command,
        close_timeout: float = 10,
        multiplex: bool | None = None,
    ) -> None:
        if close_timeout <= 0:
            raise ValueError("SSH session close timeout must be positive")
        self.ssh_host = ssh_host
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
                command = _SshCommand(
                    [
                        "ssh",
                        "-S",
                        str(self._control_path),
                        "-O",
                        "exit",
                        self.ssh_host,
                    ],
                    opens_connection=False,
                    exec_channel=False,
                    control_operation=True,
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
        parts = _validated_ssh_command(command, self.ssh_host, kwargs)
        if not self._multiplex:
            return profiled_runner(self._runner, role=role)(parts, **kwargs)

        control_path = self._ensure_control_path()
        if self._connected and not control_path.exists():
            self._connected = False
        opens_connection = not self._connected
        multiplexed = _SshCommand(
            [
                "ssh",
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPersist={_CONTROL_PERSIST_SECONDS}",
                "-o",
                f"ControlPath={control_path}",
                *parts[1:],
            ],
            opens_connection=opens_connection,
            exec_channel=True,
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


def _validated_ssh_command(
    command: object,
    ssh_host: str,
    kwargs: dict[str, object],
) -> list[str]:
    if kwargs.get("shell") is True:
        raise ValueError("reusable SSH session does not permit shell=True")
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        raise ValueError("reusable SSH session requires an argument sequence")
    parts = [str(item) for item in command]
    if len(parts) < 3 or parts[0] != "ssh" or parts[-2] != ssh_host or not parts[-1]:
        raise ValueError("reusable SSH session received an unexpected SSH command shape")
    if any(_is_control_option(item) for item in parts[1:-2]):
        raise ValueError("callers may not override reusable SSH control options")
    return parts


def _is_control_option(item: str) -> bool:
    normalized = item.removeprefix("-o")
    return (
        item == "-O"
        or item.startswith("-O")
        or item == "-S"
        or item.startswith("-S")
        or normalized.startswith("ControlMaster=")
        or normalized.startswith("ControlPath=")
        or normalized.startswith("ControlPersist=")
    )
