from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from bmd_agent.profiling import PerformanceProfiler
from bmd_agent.resources.observations import PathTest, ReadFile, ReadFileTail, SacctJob
from bmd_agent.resources.remote import ReusableSshSession
from bmd_agent.resources.transport import SshInvocation

HOST = "powerslurm-bmdguest"


def ssh(operation, **kwargs):
    return SshInvocation(operation, host=kwargs.pop("host", HOST), **kwargs)


class MultiplexingRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.timeouts: list[object] = []
        self.fail_transport = False

    def __call__(
        self,
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        self.commands.append(command)
        self.timeouts.append(kwargs.get("timeout"))
        if getattr(command, "ssh_control_operation", False):
            control_path = Path(command[command.index("-S") + 1])
            control_path.unlink(missing_ok=True)
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

        if self.fail_transport:
            self.fail_transport = False
            return subprocess.CompletedProcess(command, 255, stdout=b"", stderr=b"lost")

        if getattr(command, "ssh_opens_connection", False) and any(
            part.startswith("ControlPath=") for part in command
        ):
            control_path = _control_path(command)
            control_path.touch()
        return subprocess.CompletedProcess(command, 0, stdout=b"ok", stderr=b"")


def _control_path(command: list[str]) -> Path:
    value = next(part for part in command if part.startswith("ControlPath="))
    return Path(value.split("=", 1)[1])


def test_reusable_session_uses_one_connection_for_multiple_exec_channels() -> None:
    base_runner = MultiplexingRunner()
    profiler = PerformanceProfiler()

    with profiler.activate():
        with ReusableSshSession(
            "powerslurm-bmdguest",
            runner=base_runner,
            multiplex=True,
        ) as session:
            scheduler = session.runner("scheduler")
            remote = session.runner("remote")
            scheduler(
                ssh(SacctJob("1"), connect_timeout=10),
                capture_output=True,
                check=True,
                timeout=60,
            )
            remote(
                ssh(ReadFile("/allowed/OSZICAR")),
                capture_output=True,
                check=True,
                timeout=20,
            )

    counts = profiler.snapshot().operations.counts
    assert counts["ssh_connections"] == 1
    assert counts["ssh_exec_channels"] == 2
    assert counts["remote_commands"] == 2
    assert counts["ssh_control_operations"] == 1
    assert counts["ssh_invocations"] == 3
    assert [getattr(command, "ssh_opens_connection", None) for command in base_runner.commands] == [
        True,
        False,
        False,
    ]
    assert not _control_path(base_runner.commands[0]).exists()


def test_reusable_session_closes_connection_when_body_raises() -> None:
    base_runner = MultiplexingRunner()

    with pytest.raises(RuntimeError, match="body failed"):
        with ReusableSshSession(
            "powerslurm-bmdguest",
            runner=base_runner,
            multiplex=True,
        ) as session:
            session.runner("remote")(
                ssh(PathTest("/allowed/INCAR", "file")),
                capture_output=True,
                check=False,
                timeout=20,
            )
            raise RuntimeError("body failed")

    assert getattr(base_runner.commands[-1], "ssh_control_operation", False) is True


def test_reusable_session_preserves_each_command_timeout() -> None:
    base_runner = MultiplexingRunner()

    with ReusableSshSession(
        "powerslurm-bmdguest",
        runner=base_runner,
        close_timeout=7,
        multiplex=True,
    ) as session:
        remote = session.runner("remote")
        remote(
            ssh(ReadFile("/allowed/INCAR")),
            capture_output=True,
            check=True,
            timeout=11,
        )
        remote(
            ssh(ReadFileTail("/allowed/OUTCAR", 10)),
            capture_output=True,
            check=True,
            timeout=23,
        )

    assert base_runner.timeouts == [11, 23, 7]


def test_transport_failure_discards_session_without_retrying_command() -> None:
    base_runner = MultiplexingRunner()
    base_runner.fail_transport = True

    with ReusableSshSession(
        "powerslurm-bmdguest",
        runner=base_runner,
        multiplex=True,
    ) as session:
        remote = session.runner("remote")
        result = remote(
            ssh(PathTest("/allowed/INCAR", "file")),
            capture_output=True,
            check=False,
            timeout=20,
        )
        assert result.returncode == 255
        remote(
            ssh(PathTest("/allowed/OUTCAR", "file")),
            capture_output=True,
            check=False,
            timeout=20,
        )

    exec_commands = [
        command
        for command in base_runner.commands
        if getattr(command, "ssh_exec_channel", False)
    ]
    assert len(exec_commands) == 2
    assert all(getattr(command, "ssh_opens_connection", False) for command in exec_commands)


# Commands are built lazily inside the test: constructing an invocation at
# collection time would validate paths on the collecting platform (a POSIX
# "/tmp/..." control path is not absolute on Windows).
@pytest.mark.parametrize(
    ("make_command", "kwargs"),
    (
        (lambda tmp: ssh(ReadFile("/allowed/INCAR"), host="other-host"), {}),
        (lambda tmp: ssh(ReadFile("/allowed/INCAR")), {"shell": True}),
        # A valid, platform-native absolute path that the caller (not the
        # session) chose: refused because callers may not supply control options.
        (lambda tmp: ssh(ReadFile("/allowed/INCAR"), control_path=tmp / "caller-controlled"), {}),
        # Preconstructed vectors are refused outright, however they look.
        (lambda tmp: ["ssh", "powerslurm-bmdguest", "cat -- /allowed/INCAR"], {}),
        (
            lambda tmp: [
                "ssh",
                f"-oControlPath={tmp / 'caller-controlled'}",
                "powerslurm-bmdguest",
                "cat -- /allowed/INCAR",
            ],
            {},
        ),
    ),
)
def test_reusable_session_rejects_host_shell_and_control_overrides(
    make_command,
    kwargs: dict[str, object],
    tmp_path: Path,
) -> None:
    command = make_command(tmp_path)
    base_runner = MultiplexingRunner()

    with ReusableSshSession(
        "powerslurm-bmdguest",
        runner=base_runner,
        multiplex=True,
    ) as session:
        with pytest.raises(ValueError):
            session.runner("remote")(command, **kwargs)

    assert base_runner.commands == []


def test_session_is_lazy_and_nonmultiplex_fallback_preserves_command() -> None:
    base_runner = MultiplexingRunner()
    command = ssh(ReadFile("/allowed/INCAR"))

    with ReusableSshSession(
        "powerslurm-bmdguest",
        runner=base_runner,
        multiplex=True,
    ):
        assert base_runner.commands == []

    with ReusableSshSession(
        "powerslurm-bmdguest",
        runner=base_runner,
        multiplex=False,
    ) as session:
        assert base_runner.commands == []
        session.runner("remote")(
            command,
            capture_output=True,
            check=True,
            timeout=20,
        )

    assert base_runner.commands == [command]
