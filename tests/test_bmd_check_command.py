"""``bmd-check`` command and the typed observational transports.

``bmd-check`` and ``bmd-agent`` are two console scripts for the same
implementation. Every observational read is one value from a closed set of
typed operations; the SSH transport and the local transport
(``ssh_host = "local"``) each render that same operation themselves. Neither
accepts command text, a preconstructed vector or caller SSH options.
"""

from __future__ import annotations

from dataclasses import replace
import ast
import importlib.metadata
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import tomllib

import pytest

from bmd_agent import cli
from bmd_agent import config as config_module
from bmd_agent.config import (
    ConfigurationError,
    ResourceRegistry,
    SlurmClusterResource,
    load_resources,
    parse_resources,
)
from bmd_agent.profiling import PerformanceProfiler
from bmd_agent.resources import observations as obs
from bmd_agent.resources import transport
from bmd_agent.resources.remote import ReusableSshSession
from bmd_agent.resources.run import inspect_remote_run, inspect_slurm_job
from bmd_agent.resources.slurm import get_job_accounting
from bmd_agent.resources.transport import (
    LOCAL_TRANSPORT,
    LocalInvocation,
    LocalObservationSession,
    SshInvocation,
    TransportError,
    build_invocation,
    observe,
    run_invocation,
)
from bmd_agent.resources.vasp import (
    RemotePathError,
    authorize_remote_path,
    retrieve_remote_file,
)

import test_run_inspection as ri


REPO_ROOT = Path(__file__).resolve().parents[1]
GUEST = "/bmd-db/guest"
HOST = "powerslurm-bmdguest"
CONTROL_PATH = Path(tempfile.gettempdir()) / "ba-ssh-test" / "control"

posix_only = pytest.mark.skipif(
    os.name != "posix",
    reason="the local observation transport is POSIX-only (cluster login nodes)",
)


def cluster_config(ssh_host: str) -> dict:
    return {
        "repositories": {},
        "clusters": {
            "powerslurm": {
                "name": "POWER",
                "ssh_host": ssh_host,
                "partition": "leeburton-pool",
                "access": "observational",
                "allowed_remote_roots": ["/bmd-db"],
            }
        },
    }


# --- console scripts and command names ----------------------------------------------------


def test_pyproject_declares_both_scripts_for_the_same_implementation():
    scripts = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]

    assert scripts == {"bmd-check": "bmd_agent.cli:main", "bmd-agent": "bmd_agent.cli:main"}


def test_installed_entry_points_resolve_to_the_same_callable():
    entry_points = {
        entry.name: entry
        for entry in importlib.metadata.entry_points(group="console_scripts")
        if entry.name in cli.COMMAND_NAMES
    }
    if set(entry_points) != set(cli.COMMAND_NAMES):
        pytest.skip("installed bmd-agent metadata predates the bmd-check entry point; reinstall")

    assert entry_points["bmd-check"].value == entry_points["bmd-agent"].value == "bmd_agent.cli:main"
    assert entry_points["bmd-check"].load() is entry_points["bmd-agent"].load() is cli.main


@pytest.mark.parametrize(
    ("argv0", "expected"),
    [
        ("/bmd/shared/tools/envs/bmd-check/bin/bmd-check", "bmd-check"),
        ("bmd-check", "bmd-check"),
        (r"C:\\env\\Scripts\\bmd-check.exe", "bmd-check"),
        ("/usr/local/bin/bmd-agent", "bmd-agent"),
        ("/usr/bin/pytest", "bmd-agent"),
        ("python", "bmd-agent"),
        ("", "bmd-agent"),
        ("bmd-check\n\x1b[2J", "bmd-agent"),
    ],
)
def test_invoked_command_name(argv0, expected):
    assert cli.invoked_command_name(argv0) == expected


SCENARIOS = [
    ["definitely-not-a-path-or-job"],
    ["structure"],
    ["inspect-run"],
    ["diagnose-run"],
    ["job"],
    ["compare-runs", "/one"],
    ["12345", "--nonsense"],
    ["check-input"],
    ["x", "y", "z"],
]


@pytest.mark.parametrize("argv", SCENARIOS)
def test_bmd_check_matches_bmd_agent_except_for_the_command_name(argv, capsys):
    agent_code = cli.main(list(argv), command_name="bmd-agent")
    agent = capsys.readouterr()
    check_code = cli.main(list(argv), command_name="bmd-check")
    check = capsys.readouterr()

    assert check_code == agent_code
    assert check.out == agent.out.replace("bmd-agent", "bmd-check")
    assert check.err == agent.err.replace("bmd-agent", "bmd-check")


def test_default_invocation_keeps_existing_bmd_agent_wording(capsys):
    cli.main(["x", "y", "z"])
    output = capsys.readouterr().out

    assert "Usage: bmd-agent [TARGET] [--verbose] [--profile]" in output
    assert "bmd-check" not in output


def test_detailed_evidence_hint_follows_the_invoked_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_resources", lambda: (_ for _ in ()).throw(ConfigurationError("none")))
    calculation_dir = tmp_path / "bmd-agent-calculation"
    calculation_dir.mkdir()
    monkeypatch.chdir(calculation_dir)

    cli.main([], command_name="bmd-check")
    check = capsys.readouterr().out
    cli.main([], command_name="bmd-agent")
    agent = capsys.readouterr().out

    check_hint = "Detailed evidence:\n  bmd-check --verbose"
    agent_hint = "Detailed evidence:\n  bmd-agent --verbose"
    assert check_hint in check
    assert agent_hint in agent
    assert str(calculation_dir) in check
    assert str(calculation_dir) in agent
    assert check == agent.replace(agent_hint, check_hint)


def test_command_name_does_not_leak_between_invocations(capsys):
    cli.main(["job"], command_name="bmd-check")
    capsys.readouterr()
    cli.main(["job"])

    assert "Usage: bmd-agent job" in capsys.readouterr().out


@pytest.mark.parametrize(
    "command_name",
    ["evil\n\x1b[2Jrm -rf ~", "bmd-check\n", "bmd-agent ", "", "BMD-CHECK", "bmd"],
)
def test_command_name_override_is_restricted_to_the_allowlist(command_name, capsys):
    # Codex attack: an embedding caller injecting terminal/newline text into hints.
    with pytest.raises(ValueError, match="command_name must be one of"):
        cli.main(["x", "y", "z"], command_name=command_name)

    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert cli._command_name() == cli.DEFAULT_COMMAND_NAME


# --- missing configuration guidance -------------------------------------------------------


def test_missing_config_guidance_from_an_installed_package(tmp_path, monkeypatch):
    nonexistent = tmp_path / "site-packages" / "config" / "resources.example.toml"
    monkeypatch.setattr(config_module, "resources_example_path", lambda: nonexistent)

    with pytest.raises(ConfigurationError) as exc_info:
        load_resources(tmp_path / "missing.toml")
    message = str(exc_info.value)

    assert str(nonexistent) not in message
    assert "config/resources.example.toml in the BMD Agent repository" in message
    assert "BMD_AGENT_RESOURCES" in message


def test_missing_config_guidance_from_a_source_checkout_names_the_example(tmp_path):
    with pytest.raises(ConfigurationError) as exc_info:
        load_resources(tmp_path / "missing.toml")

    assert str(REPO_ROOT / "config" / "resources.example.toml") in str(exc_info.value)


# --- the closed operation set -------------------------------------------------------------


EXPECTED_ARGV = [
    (obs.SqueuePartition("leeburton-pool"), ("squeue", "-p", "leeburton-pool", "--noheader", f"--format={obs._SQUEUE_FORMAT}")),
    (obs.SacctJob("20893681"), ("sacct", "-P", "-n", "-j", "20893681", f"--format={obs._SACCT_FORMAT}")),
    (obs.ReadFile("/allowed/INCAR"), ("cat", "--", "/allowed/INCAR")),
    (obs.ReadFileTail("/allowed/OUTCAR", 10), ("tail", "-c", "10", "--", "/allowed/OUTCAR")),
    (obs.FileSize("/allowed/vasprun.xml"), ("stat", "-c", "%s", "--", "/allowed/vasprun.xml")),
    (obs.PathTest("/allowed/INCAR", "file"), ("test", "-f", "/allowed/INCAR")),
    (obs.PathTest("/allowed", "directory"), ("test", "-d", "/allowed")),
    (
        obs.ProbeErrorArchives("/allowed", 3),
        ("sh", "-c", obs._ARCHIVE_PROBE_PROGRAM, obs._ARCHIVE_PROBE_MARKER, "/allowed", "3"),
    ),
    (
        obs.ExtractOutcarForces("/allowed/OUTCAR", 2),
        ("awk", "-v", "expected=2", obs._OUTCAR_FORCE_EXTRACTOR_AWK, "/allowed/OUTCAR"),
    ),
    (
        obs.AcquireBatch((obs.AcquisitionItem("/allowed/INCAR", "file", 1024),)),
        (
            "sh",
            "-s",
            "--",
            obs._ACQUISITION_MARKER,
            str(obs._MAX_BATCH_TOTAL_BYTES),
            "1",
            "1",
            "0",
            "1",
            "file",
            "1024",
            "/allowed/INCAR",
        ),
    ),
]


@pytest.mark.parametrize(("operation", "argv"), EXPECTED_ARGV)
def test_each_operation_renders_one_fixed_argv(operation, argv):
    assert operation.argv() == argv
    assert type(operation) in obs.OBSERVATION_OPERATION_TYPES


def test_operation_set_is_closed_and_derived_from_existing_builders():
    assert {cls.__name__ for cls in obs.OBSERVATION_OPERATION_TYPES} == {
        "SqueuePartition",
        "SacctJob",
        "ReadFile",
        "ReadFileTail",
        "FileSize",
        "PathTest",
        "ProbeErrorArchives",
        "ExtractOutcarForces",
        "AcquireBatch",
    }
    assert {type(operation) for operation, _argv in EXPECTED_ARGV} == set(
        obs.OBSERVATION_OPERATION_TYPES
    )


def test_shell_programs_are_agent_owned_constants_with_data_as_positional_arguments():
    probe = obs.ProbeErrorArchives("/allowed/x; touch /tmp/p", 2).argv()
    assert probe[2] == obs._ARCHIVE_PROBE_PROGRAM
    assert "/allowed" not in probe[2]
    assert probe[4] == "/allowed/x; touch /tmp/p"

    batch = obs.AcquireBatch((obs.AcquisitionItem("/allowed/$(id)", "file", 10),))
    assert batch.argv()[:4] == ("sh", "-s", "--", obs._ACQUISITION_MARKER)
    assert batch.argv()[-1] == "/allowed/$(id)"
    assert batch.stdin == obs._ACQUISITION_SCRIPT.encode("ascii")


@pytest.mark.parametrize(
    "build",
    [
        lambda: obs.ReadFile("relative/INCAR"),
        lambda: obs.ReadFile("/allowed/../etc/passwd"),
        lambda: obs.ReadFile("/allowed/./INCAR"),
        lambda: obs.ReadFile("/allowed//INCAR"),
        lambda: obs.ReadFile("/allowed/INCAR\x00"),
        lambda: obs.ReadFile(["cat", "/etc/passwd"]),
        lambda: obs.ReadFileTail("/allowed/OUTCAR", 0),
        lambda: obs.ReadFileTail("/allowed/OUTCAR", True),
        lambda: obs.ReadFileTail("/allowed/OUTCAR", "10; id"),
        lambda: obs.PathTest("/allowed", "-e"),
        lambda: obs.ProbeErrorArchives("/allowed", 65),
        lambda: obs.SacctJob("1; scancel 2"),
        lambda: obs.SacctJob("0"),
        lambda: obs.SqueuePartition("pool;scancel 1"),
        lambda: obs.SqueuePartition("-oProxyCommand=x"),
        lambda: obs.ExtractOutcarForces("/allowed/OUTCAR", "2; id"),
        lambda: obs.AcquireBatch(()),
        lambda: obs.AcquireBatch(("cat /etc/passwd",)),
        lambda: obs.AcquisitionItem("/allowed", "exec", 0),
    ],
)
def test_operations_reject_untyped_or_unsafe_values(build):
    with pytest.raises(ValueError):
        build()


def test_transports_refuse_anything_outside_the_closed_set():
    class Lookalike(obs.ReadFile):
        def argv(self):
            return ("sh", "-c", "id")

    for operation in ("cat -- /etc/passwd", ["cat", "/etc/passwd"], Lookalike("/allowed/x"), object()):
        with pytest.raises(ValueError):
            SshInvocation(operation, host=HOST)
        if transport.local_transport_supported():
            with pytest.raises(ValueError):
                LocalInvocation(operation)


def test_operation_subclass_never_reaches_subprocess(monkeypatch):
    calls = recording_subprocess(monkeypatch)

    class Lookalike(obs.ReadFile):
        def argv(self):
            return ("sh", "-c", "echo injected")

    with pytest.raises(ValueError, match="fixed BMD Agent observational operation"):
        observe(HOST, Lookalike("/allowed/INCAR"), capture_output=True)
    assert calls == []


# --- Codex attacks: there is no arbitrary-command primitive -------------------------------


def recording_subprocess(monkeypatch, *, stdout="", returncode=0):
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, **kwargs):
        calls.append((list(command), kwargs))
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(transport.subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize(
    "command",
    [
        ["ssh", "local", "true; touch /tmp/semicolon"],  # semicolon
        ["ssh", "local", "echo $(touch /tmp/substitution)"],  # command substitution
        ["ssh", "local", "true\ntouch /tmp/newline"],  # newline
        ["sh", "-c", "true; touch /tmp/x"],
        "cat /etc/passwd",
        ("cat", "/etc/passwd"),
    ],
)
def test_default_executor_refuses_preconstructed_commands(command, monkeypatch):
    calls = recording_subprocess(monkeypatch)

    with pytest.raises(TransportError):
        run_invocation(command, capture_output=True)
    assert calls == []


def test_no_transport_function_accepts_command_text():
    assert not hasattr(transport, "run_local_command")
    assert not hasattr(transport, "run_ssh_command")
    assert not hasattr(transport, "local_command")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shell": True},
        {"executable": "/bin/sh"},
        {"env": {"PATH": tempfile.gettempdir()}},
        {"cwd": tempfile.gettempdir()},
        {"preexec_fn": print},
        {"input": b"id\n"},
    ],
)
def test_default_executor_refuses_caller_process_options(kwargs, monkeypatch):
    calls = recording_subprocess(monkeypatch)
    invocation = SshInvocation(obs.ReadFile("/allowed/INCAR"), host=HOST)

    with pytest.raises(TransportError):
        run_invocation(invocation, **kwargs)
    assert calls == []


def test_observe_refuses_caller_input():
    with pytest.raises(TransportError):
        observe(HOST, obs.ReadFile("/allowed/INCAR"), runner=pytest.fail, input=b"id\n")


def test_invocation_sequence_and_fields_are_immutable(monkeypatch):
    calls = recording_subprocess(monkeypatch)
    invocation = SshInvocation(
        obs.ReadFile("/allowed/INCAR"),
        host=HOST,
        connect_timeout=10,
    )

    with pytest.raises(TypeError):
        invocation[2] = "BatchMode=no"
    with pytest.raises(AttributeError):
        invocation.operation = obs.ReadFile("/etc/passwd")
    with pytest.raises(AttributeError):
        invocation.host = "-oProxyCommand=sh -c id"
    assert calls == []


def test_executor_rejects_synchronized_post_construction_operation_replacement(monkeypatch):
    calls = recording_subprocess(monkeypatch)

    class FabricatedRead(obs.ReadFile):
        def argv(self):
            return ("sh", "-c", "echo injected")

    invocation = SshInvocation(obs.ReadFile("/allowed/INCAR"), host=HOST)
    with pytest.raises(AttributeError):
        object.__setattr__(invocation, "_operation", FabricatedRead("/allowed/INCAR"))
    with pytest.raises(AttributeError):
        object.__setattr__(
            invocation,
            "_argv",
            ("ssh", "-o", "BatchMode=yes", HOST, "sh -c 'echo injected'"),
        )
    assert calls == []


def test_executor_rejects_synchronized_post_construction_host_replacement(monkeypatch):
    calls = recording_subprocess(monkeypatch)
    invocation = SshInvocation(obs.ReadFile("/allowed/INCAR"), host=HOST)
    hostile_host = "-oProxyCommand=sh -c id"
    with pytest.raises(AttributeError):
        object.__setattr__(invocation, "_host", hostile_host)
    with pytest.raises(AttributeError):
        object.__setattr__(
            invocation,
            "_argv",
            ("ssh", "-o", "BatchMode=yes", hostile_host, "cat -- /allowed/INCAR"),
        )
    assert calls == []


def test_synchronized_mutation_to_other_valid_fields_is_impossible(monkeypatch):
    calls = recording_subprocess(monkeypatch)
    invocation = SshInvocation(obs.ReadFile("/allowed/INCAR"), host=HOST)

    for field, value in (
        ("_operation", obs.ReadFile("/allowed/OTHER")),
        ("_host", "another-host"),
        ("_connect_timeout", 5),
        (
            "_argv",
            ("ssh", "-o", "BatchMode=yes", "another-host", "cat -- /allowed/OTHER"),
        ),
    ):
        with pytest.raises(AttributeError):
            object.__setattr__(invocation, field, value)
    assert calls == []


def test_executor_rejects_fabricated_invocation_subclass(monkeypatch):
    calls = recording_subprocess(monkeypatch)

    class FabricatedInvocation(transport.ObservationInvocation):
        @property
        def operation(self):
            return None

        def render(self):
            return ["sh", "-c", "echo injected"]

    invocation = FabricatedInvocation()

    with pytest.raises(TransportError, match="only fixed observational operations"):
        run_invocation(invocation, capture_output=True)
    assert calls == []


def test_reusable_session_rejects_fabricated_ssh_invocation_subclass(monkeypatch):
    calls = recording_subprocess(monkeypatch)

    class FabricatedSshInvocation(SshInvocation):
        pass

    invocation = FabricatedSshInvocation(obs.ReadFile("/allowed/INCAR"), host=HOST)
    with ReusableSshSession(HOST, runner=transport.run_invocation, multiplex=False) as session:
        with pytest.raises(TransportError, match="only fixed observational operations"):
            session.runner("remote")(invocation, capture_output=True)
    assert calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("_connect_timeout", "1 -oProxyCommand=id"),
        ("_control_path", "-oProxyCommand=id"),
        ("_control_exit", 1),
        ("_opens_connection", 1),
    ],
)
def test_executor_revalidates_every_other_execution_field(
    field,
    value,
    monkeypatch,
):
    calls = recording_subprocess(monkeypatch)
    invocation = SshInvocation(obs.ReadFile("/allowed/INCAR"), host=HOST)
    with pytest.raises(AttributeError):
        object.__setattr__(invocation, field, value)
    with pytest.raises(AttributeError):
        object.__setattr__(invocation, "_argv", tuple(invocation))
    assert calls == []


def test_scalar_subclasses_cannot_override_executable_rendering(monkeypatch):
    calls = recording_subprocess(monkeypatch)

    class Host(str):
        def __format__(self, format_spec):
            return "-oProxyCommand=id"

    class Count(int):
        def __str__(self):
            return "1; id"

    for build in (
        lambda: SshInvocation(obs.ReadFile("/allowed/INCAR"), host=Host(HOST)),
        lambda: build_invocation(Host(LOCAL_TRANSPORT), obs.ReadFile("/allowed/INCAR")),
        lambda: SshInvocation(
            obs.ReadFile("/allowed/INCAR"),
            host=HOST,
            connect_timeout=Count(10),
        ),
        lambda: SshInvocation(obs.ReadFile(Host("/allowed/INCAR")), host=HOST),
        lambda: SshInvocation(
            obs.ReadFileTail("/allowed/OUTCAR", Count(10)),
            host=HOST,
        ),
    ):
        with pytest.raises((TransportError, ValueError)):
            build()
    assert calls == []


def test_private_state_access_returns_a_non_executable_snapshot(monkeypatch):
    calls = recording_subprocess(monkeypatch)
    invocation = SshInvocation(obs.ReadFile("/allowed/INCAR"), host=HOST)
    snapshot = transport._invocation_state(invocation)
    object.__setattr__(snapshot, "host", "another-host")
    object.__setattr__(
        snapshot,
        "argv",
        ("ssh", "-o", "BatchMode=yes", "another-host", "cat -- /allowed/OTHER"),
    )

    run_invocation(invocation, capture_output=True)

    assert calls == [
        (
            ["ssh", "-o", "BatchMode=yes", HOST, "cat -- /allowed/INCAR"],
            {"capture_output": True, "shell": False},
        )
    ]


@pytest.mark.parametrize(
    "connect_timeout",
    ["1 -oProxyCommand=x", "10", 0, -1, 1.5, True, 10**6],
)
def test_malformed_connect_timeout_is_rejected(connect_timeout):
    with pytest.raises(TransportError):
        SshInvocation(obs.ReadFile("/allowed/INCAR"), host=HOST, connect_timeout=connect_timeout)


@pytest.mark.parametrize(
    "host",
    ["-oProxyCommand=sh -c id", "-F/tmp/cfg", "host -oBatchMode=no", "host\nid", "", "local", 7],
)
def test_ssh_host_cannot_carry_options(host):
    with pytest.raises(TransportError):
        SshInvocation(obs.ReadFile("/allowed/INCAR"), host=host)


AGENT_SSH_OPTIONS = ("BatchMode=yes", "ConnectTimeout=", "ControlMaster=auto", "ControlPersist=", "ControlPath=")


def ssh_options(argv: list[str]) -> list[str]:
    return [argv[index + 1] for index, item in enumerate(argv[:-2]) if item == "-o"]


@pytest.mark.parametrize(("operation", "_argv"), EXPECTED_ARGV)
@pytest.mark.parametrize("connect_timeout", [None, 7])
@pytest.mark.parametrize("control_path", [None, CONTROL_PATH])
def test_ssh_invocation_is_built_only_from_agent_owned_options(operation, _argv, connect_timeout, control_path):
    argv = SshInvocation(
        operation,
        host=HOST,
        connect_timeout=connect_timeout,
        control_path=control_path,
    ).render()

    # BatchMode=yes is unconditionally the first option; OpenSSH keeps the
    # first value it obtains, so nothing later could override it.
    assert argv[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert argv[-2:] == [HOST, operation.remote_command()]
    options = ssh_options(argv)
    assert all(option.startswith(AGENT_SSH_OPTIONS) for option in options)
    assert options.count("BatchMode=yes") == 1
    assert not any(option.lower().startswith(("proxycommand", "batchmode=no", "localcommand")) for option in options)
    # Only "-o" pairs precede the host: no other flags.
    assert argv[1:-2] == [item for option in options for item in ("-o", option)]


@pytest.mark.parametrize(("operation", "_argv"), EXPECTED_ARGV)
def test_each_operation_executes_through_the_ssh_boundary(operation, _argv, monkeypatch):
    calls = recording_subprocess(monkeypatch)

    observe(HOST, operation, capture_output=True)

    assert len(calls) == 1
    executed, kwargs = calls[0]
    assert executed[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert executed[-2:] == [HOST, obs.operation_remote_command(operation)]
    assert kwargs["shell"] is False


@pytest.mark.parametrize(
    "make_path",
    [
        lambda tmp: Path("relative") / "control",
        lambda tmp: str(tmp / "control"),  # a string, not a Path
        lambda tmp: tmp / "bad\x00control",
    ],
)
def test_control_path_must_be_an_absolute_native_path(make_path, tmp_path):
    with pytest.raises(TransportError, match="control path"):
        SshInvocation(obs.ReadFile("/allowed/INCAR"), host=HOST, control_path=make_path(tmp_path))
    # A platform-native absolute path is accepted on every platform.
    assert SshInvocation(obs.ReadFile("/allowed/INCAR"), host=HOST, control_path=tmp_path / "control").control_path


@pytest.mark.skipif(os.name == "posix", reason="a drive-less POSIX path is absolute here")
def test_posix_style_control_path_is_not_absolute_on_this_platform():
    with pytest.raises(TransportError, match="control path"):
        SshInvocation(obs.ReadFile("/allowed/INCAR"), host=HOST, control_path=Path("/tmp/caller-controlled"))


def test_session_exit_is_agent_owned_and_batch_mode():
    argv = SshInvocation.session_exit(host=HOST, control_path=CONTROL_PATH).render()

    assert argv == ["ssh", "-o", "BatchMode=yes", "-S", str(CONTROL_PATH), "-O", "exit", HOST]


def test_remote_command_text_is_unchanged_from_the_previous_builders():
    assert obs.ReadFile("/a/project with spaces/POSCAR").remote_command() == "cat -- '/a/project with spaces/POSCAR'"
    assert obs.PathTest("/a/run", "file").remote_command() == "test -f /a/run"
    assert obs.ReadFileTail("/a/OUTCAR", 10).remote_command() == "tail -c 10 -- /a/OUTCAR"
    assert obs.FileSize("/a/v.xml").remote_command() == "stat -c %s -- /a/v.xml"


# --- SSH transport: BatchMode on every real path ------------------------------------------


def test_default_remote_reads_and_scheduler_lookups_use_batch_mode(monkeypatch):
    calls = recording_subprocess(monkeypatch, stdout="")

    retrieve_remote_file(HOST, PurePosixPath("/allowed/INCAR"))
    get_job_accounting(HOST, "123")

    assert calls[0][0][:3] == ["ssh", "-o", "BatchMode=yes"]
    assert calls[1][0][:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def test_reusable_ssh_session_uses_batch_mode(monkeypatch):
    calls = recording_subprocess(monkeypatch)

    with ReusableSshSession(HOST, multiplex=True) as session:
        session.runner("remote")(SshInvocation(obs.PathTest("/allowed/INCAR", "file"), host=HOST), capture_output=True)

    executed = calls[0][0]
    assert executed[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert executed[-2:] == [HOST, "test -f /allowed/INCAR"]
    assert any(item.startswith("ControlMaster=") for item in executed)


@pytest.mark.parametrize(
    "command",
    [
        ["ssh", "-o", "BatchMode=no", HOST, "true"],
        ["ssh", "-o", "ProxyCommand=touch /tmp/x", HOST, "BatchMode=yes"],  # fake BatchMode elsewhere
        ["ssh", "-o", "ProxyCommand=sh -c id", HOST, "true"],
        ["ssh", "-o", "ConnectTimeout=1 -oProxyCommand=x", HOST, "true"],
        ["ssh", HOST, "cat -- /allowed/INCAR"],
    ],
)
def test_ssh_paths_refuse_malformed_vectors(command, monkeypatch):
    calls = recording_subprocess(monkeypatch)

    with pytest.raises(TransportError):
        run_invocation(command, capture_output=True)
    with ReusableSshSession(HOST, multiplex=False) as session:
        with pytest.raises(TransportError):
            session.runner("remote")(command, capture_output=True)
    assert calls == []


def test_reusable_ssh_session_refuses_the_local_selector_and_local_invocations(monkeypatch):
    with pytest.raises(TransportError):
        ReusableSshSession(LOCAL_TRANSPORT)
    if transport.local_transport_supported():
        with ReusableSshSession(HOST, multiplex=False) as session:
            with pytest.raises(TransportError):
                session.runner("remote")(LocalInvocation(obs.ReadFile("/allowed/INCAR")))


# --- platform support for the local transport ---------------------------------------------


def test_local_selector_fails_fast_where_unsupported(monkeypatch):
    # Codex attack: selecting ssh_host="local" on Windows must fail immediately
    # with a concise configuration error, not later with FileNotFoundError.
    monkeypatch.setattr(transport, "local_transport_supported", lambda: False)
    calls = recording_subprocess(monkeypatch)

    with pytest.raises(ConfigurationError, match="requires a POSIX system") as exc_info:
        parse_resources(cluster_config("local"))
    assert "\n" not in str(exc_info.value)
    with pytest.raises(TransportError, match="requires a POSIX system"):
        build_invocation(LOCAL_TRANSPORT, obs.ReadFile("/allowed/INCAR"))
    with pytest.raises(TransportError, match="requires a POSIX system"):
        LocalObservationSession()
    assert calls == []
    # SSH configuration is unaffected.
    assert parse_resources(cluster_config(HOST)).clusters["powerslurm"].ssh_host == HOST


def test_local_selector_from_a_real_config_file_fails_fast_where_unsupported(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(transport, "local_transport_supported", lambda: False)
    path = tmp_path / "resources.toml"
    path.write_text(
        '[repositories]\n\n[clusters.powerslurm]\nname = "POWER"\nssh_host = "local"\n'
        'partition = "leeburton-pool"\naccess = "observational"\nallowed_remote_roots = ["/bmd-db"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("BMD_AGENT_RESOURCES", str(path))

    exit_code = cli.main(["20893681"], command_name="bmd-check")
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "requires a POSIX system" in captured.err
    assert "FileNotFoundError" not in captured.out + captured.err


@pytest.mark.skipif(os.name == "posix", reason="exercises the real non-POSIX platform")
def test_local_selector_is_rejected_on_this_non_posix_platform():
    with pytest.raises(ConfigurationError, match="requires a POSIX system"):
        parse_resources(cluster_config("local"))


@posix_only
def test_local_selector_is_a_valid_posix_deployment_value():
    cluster = parse_resources(cluster_config("local")).clusters["powerslurm"]

    assert isinstance(cli._observation_session(cluster), LocalObservationSession)
    assert isinstance(cli._observation_session(replace(cluster, ssh_host=HOST)), ReusableSshSession)
    assert isinstance(build_invocation(cluster.ssh_host, obs.ReadFile("/bmd-db/x")), LocalInvocation)


def test_config_still_rejects_unsafe_transport_values():
    for value in ("local; rm -rf /", "local host", "$(id)", "-oProxyCommand=id"):
        with pytest.raises(ConfigurationError):
            parse_resources(cluster_config(value))


# --- local transport: argv execution without a shell or SSH -------------------------------


@pytest.mark.parametrize(("operation", "argv"), EXPECTED_ARGV)
def test_local_invocation_is_the_operation_argv(operation, argv, monkeypatch):
    calls = recording_subprocess(monkeypatch)
    monkeypatch.setattr(transport, "require_local_transport_supported", lambda: None)

    observe(LOCAL_TRANSPORT, operation, capture_output=True)

    executed, kwargs = calls[0]
    assert executed == list(argv)
    assert executed[0] != "ssh"
    assert kwargs["shell"] is False


@posix_only
def test_local_transport_never_invokes_ssh(monkeypatch):
    calls = recording_subprocess(monkeypatch, stdout="")

    retrieve_remote_file(LOCAL_TRANSPORT, PurePosixPath("/allowed/INCAR"))
    get_job_accounting(LOCAL_TRANSPORT, "123")
    LocalObservationSession().runner("remote")(LocalInvocation(obs.PathTest("/allowed/INCAR", "file")), capture_output=True)

    assert [command[0] for command, _ in calls] == ["cat", "sacct", "test"]
    assert all("ssh" not in command for command, _ in calls)


@posix_only
@pytest.mark.parametrize(
    "name",
    ["x; touch {marker}", "$(touch {marker})", "`touch {marker}`", "x\ntouch {marker}", "x' ; touch {marker}; '"],
)
def test_hostile_path_text_is_data_never_executed(tmp_path, name):
    # Codex attacks as data: semicolon, command substitution, newline.
    root = tmp_path / "root"
    root.mkdir()
    marker = tmp_path / "pwned"
    hostile = root / name.format(marker=marker).replace("/", "_")
    hostile.write_bytes(b"literal contents")
    remote_path = authorize_remote_path(str(hostile), allowed_roots=(str(root),))

    assert retrieve_remote_file(LOCAL_TRANSPORT, remote_path) == b"literal contents"
    assert obs.ProbeErrorArchives(remote_path, 2) is not None
    assert not marker.exists()


@posix_only
def test_local_probe_and_acquisition_scripts_take_paths_only_as_arguments(tmp_path):
    marker = tmp_path / "pwned"
    directory = tmp_path / f"d; touch {marker}".replace("/", "_")
    directory.mkdir()
    (directory / "error.1.tar.gz").write_bytes(b"")

    result = observe(LOCAL_TRANSPORT, obs.ProbeErrorArchives(directory, 3), capture_output=True, check=False)

    assert result.stdout.decode().splitlines() == [f"{directory}/error.1.tar.gz"]
    observe(
        LOCAL_TRANSPORT,
        obs.AcquireBatch((obs.AcquisitionItem(directory, "directory", 0),)),
        capture_output=True,
        check=False,
    )
    assert not marker.exists()


@posix_only
def test_local_session_refuses_ssh_invocations():
    with pytest.raises(TransportError):
        LocalObservationSession().runner("remote")(SshInvocation(obs.ReadFile("/allowed/INCAR"), host=HOST))


@posix_only
def test_local_observations_are_profiled_without_ssh(tmp_path):
    (tmp_path / "INCAR").write_bytes(b"ENCUT = 520\n")
    profiler = PerformanceProfiler()

    with profiler.activate():
        runner = LocalObservationSession().runner("remote")
        retrieve_remote_file(LOCAL_TRANSPORT, PurePosixPath(f"{tmp_path}/INCAR"), runner=runner)

    counts = profiler.snapshot().operations.counts
    assert counts["remote_commands"] == 1
    assert counts["ssh_invocations"] == 0
    assert counts["ssh_connections"] == 0


# --- local and simulated-SSH observation of the same evidence -----------------------------


def rebase(text: str, root: Path) -> str:
    return text.replace(GUEST, f"{root}{GUEST}")


def materialize_fixture(root: Path) -> tuple[dict[str, bytes], set[str]]:
    files = {
        rebase(path, root): rebase(contents.decode("utf-8"), root).encode("utf-8")
        for path, contents in ri.default_files().items()
    }
    directories = {rebase(path, root) for path in ri.default_directories()}
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    for path, contents in files.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(contents)
    return files, directories


def fixture_cluster(root: Path, ssh_host: str) -> SlurmClusterResource:
    return replace(ri.cluster(), ssh_host=ssh_host, allowed_remote_roots=(PurePosixPath(f"{root}{GUEST}"),))


def any_host_scheduler(root: Path):
    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=rebase(ri.job_sacct_output(work_dir=ri.FLOW_ROOT), root),
            stderr="",
        )

    return runner


@posix_only
def test_local_and_simulated_ssh_observation_parse_identically(tmp_path):
    files, directories = materialize_fixture(tmp_path)
    flow_root = rebase(ri.FLOW_ROOT, tmp_path)

    over_ssh = inspect_remote_run(
        fixture_cluster(tmp_path, HOST),
        flow_root,
        remote_runner=ri.RemoteFixture(files=files, directories=directories),
        slurm_runner=any_host_scheduler(tmp_path),
        scientific_parser=ri.fake_scientific_parser,
        modifier_policies=(ri.modifier_policy(),),
    )
    local = inspect_remote_run(
        fixture_cluster(tmp_path, LOCAL_TRANSPORT),
        flow_root,
        slurm_runner=any_host_scheduler(tmp_path),
        scientific_parser=ri.fake_scientific_parser,
        modifier_policies=(ri.modifier_policy(),),
    )

    assert local == over_ssh
    assert local.scientific.final_formula == "Example2"
    assert local.runtime.packages["pymatgen"] == "2026.8.13"


def write_fake_sacct(tmp_path: Path, root: Path) -> Path:
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    output = tmp_path / "sacct.out"
    output.write_text(rebase(ri.job_sacct_output(work_dir=ri.FLOW_ROOT), root), encoding="utf-8")
    script = bin_dir / "sacct"
    script.write_text(f"#!/bin/sh\ncat {output}\n", encoding="utf-8")
    script.chmod(0o755)
    # A fake ssh that records any call proves local mode never reaches it.
    ssh = bin_dir / "ssh"
    ssh.write_text(f"#!/bin/sh\necho invoked >> {tmp_path / 'ssh-invoked'}\nexit 255\n", encoding="utf-8")
    ssh.chmod(0o755)
    return bin_dir


@posix_only
def test_local_job_inspection_end_to_end_without_ssh(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    materialize_fixture(data)
    bin_dir = write_fake_sacct(tmp_path, data)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    cluster = fixture_cluster(data, LOCAL_TRANSPORT)
    monkeypatch.setattr(cli, "load_resources", lambda: ResourceRegistry({}, {"powerslurm": cluster}))
    monkeypatch.setattr(
        cli,
        "ReusableSshSession",
        lambda *args, **kwargs: pytest.fail("local mode must not open an SSH session"),
    )

    exit_code = cli.main(["20893681", "--verbose"], command_name="bmd-check")
    output = capsys.readouterr().out

    assert exit_code == 0
    assert f"scheduler WorkDir: {rebase(ri.FLOW_ROOT, data)}" in output
    assert "type: BMD Compute" in output
    assert not (tmp_path / "ssh-invoked").exists()


@posix_only
def test_local_job_inspection_matches_simulated_ssh(tmp_path, monkeypatch):
    data = tmp_path / "data"
    files, directories = materialize_fixture(data)
    bin_dir = write_fake_sacct(tmp_path, data)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    def ssh_scheduler(command, **kwargs):
        assert isinstance(command, SshInvocation) and command[-2] == HOST
        return any_host_scheduler(data)(command, **kwargs)

    over_ssh = inspect_slurm_job(
        fixture_cluster(data, HOST),
        "20893681",
        remote_runner=ri.RemoteFixture(files=files, directories=directories),
        slurm_runner=ssh_scheduler,
        scientific_parser=ri.fake_scientific_parser,
        max_vasprun_bytes=0,
    )
    local_session = LocalObservationSession()
    local = inspect_slurm_job(
        fixture_cluster(data, LOCAL_TRANSPORT),
        "20893681",
        remote_runner=local_session.runner("remote"),
        slurm_runner=local_session.runner("scheduler"),
        scientific_parser=ri.fake_scientific_parser,
        max_vasprun_bytes=0,
    )

    assert local == over_ssh
    assert local.scheduler.work_dir == rebase(ri.FLOW_ROOT, data)
    assert not (tmp_path / "ssh-invoked").exists()


@posix_only
def test_path_mode_scheduler_lookup_uses_the_local_transport(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    materialize_fixture(data)
    bin_dir = write_fake_sacct(tmp_path, data)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    cluster = fixture_cluster(data, LOCAL_TRANSPORT)
    executed: list[list[str]] = []
    real_run = subprocess.run

    def recording_run(argv, **kwargs):
        executed.append(list(argv))
        return real_run(argv, **kwargs)

    monkeypatch.setattr(transport.subprocess, "run", recording_run)

    import test_lifecycle as lc

    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    lc.write_inputs(stage)
    lc.write_submission(root, stage_dir=stage, job_id="20893681")

    cli.show_current_directory(root, ResourceRegistry({}, {"powerslurm": cluster}), verbose=True)
    output = capsys.readouterr().out

    assert [argv[:5] for argv in executed] == [["sacct", "-P", "-n", "-j", "20893681"]]
    assert "Scheduler observation:" in output
    assert "state: COMPLETED" in output
    assert not (tmp_path / "ssh-invoked").exists()


# --- local mode keeps path protections ----------------------------------------------------


@posix_only
def test_local_mode_refuses_paths_outside_allowed_roots_before_running_anything(tmp_path, monkeypatch):
    calls = recording_subprocess(monkeypatch)
    cluster = fixture_cluster(tmp_path, LOCAL_TRANSPORT)

    for target in ("/etc", f"{tmp_path}{GUEST}/../../etc"):
        with pytest.raises(RemotePathError):
            inspect_remote_run(cluster, target)
    assert calls == []


@posix_only
def test_local_mode_refuses_an_unauthorized_scheduler_workdir(tmp_path):
    def scheduler(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=ri.job_sacct_output(work_dir="/etc"), stderr="")

    inspection = inspect_slurm_job(
        fixture_cluster(tmp_path, LOCAL_TRANSPORT),
        "20893681",
        slurm_runner=scheduler,
    )

    assert inspection.calculation_directory is None
    assert "not authorized" in inspection.calculation_reason


# --- documented limitation: allowed_remote_roots is lexical, not realpath containment -----


def test_lexical_authorization_rejects_dot_dot_escapes_without_touching_the_filesystem():
    with pytest.raises(RemotePathError):
        authorize_remote_path("/bmd-db/guest/../../etc/passwd", allowed_roots=("/bmd-db/guest",))
    # Accepted purely lexically: nothing is resolved or required to exist.
    assert authorize_remote_path("/bmd-db/guest/does/not/exist", allowed_roots=("/bmd-db/guest",)) == PurePosixPath(
        "/bmd-db/guest/does/not/exist"
    )


@posix_only
def test_symlink_beneath_an_allowed_root_is_followed_outside_it(tmp_path):
    """Current behaviour, kept deliberately in this change and documented in
    SECURITY.md: authorization is lexical, so a symlink under an allowed root
    that points elsewhere is followed with the invoking identity's
    permissions. This is not realpath/symlink containment."""

    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "secret").write_bytes(b"outside the allowed root")
    (allowed / "link").symlink_to(outside / "secret")

    remote_path = authorize_remote_path(f"{allowed}/link", allowed_roots=(str(allowed),))

    assert retrieve_remote_file(LOCAL_TRANSPORT, remote_path) == b"outside the allowed root"


# --- the suite collects on every platform -------------------------------------------------


TESTS_DIR = Path(__file__).resolve().parent


def _collection_time_nodes(tree: ast.Module):
    """Expressions evaluated when a module is imported/collected (not test bodies)."""

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield from node.decorator_list
            if not isinstance(node, ast.ClassDef):
                yield from node.args.defaults
                yield from (d for d in node.args.kw_defaults if d is not None)
        else:
            yield node


def test_no_native_posix_path_literal_is_evaluated_at_collection_time():
    """Regression: test_remote.py once built Path("/tmp/...") in a parametrize
    list, which the control-path validator rejects on Windows, so collection
    failed there. Native ``Path``/``pathlib.Path`` literals beginning with "/"
    must not be evaluated at import time (``PurePosixPath`` is fine)."""

    offenders = []
    for source in sorted(TESTS_DIR.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for root in _collection_time_nodes(tree):
            for node in ast.walk(root):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
                first = node.args[0]
                if (
                    name == "Path"
                    and isinstance(first, ast.Constant)
                    and isinstance(first.value, str)
                    and first.value.startswith("/")
                ):
                    offenders.append(f"{source.name}:{node.lineno}")

    assert offenders == []


_COLLECT_AS_IF_UNSUPPORTED = """
import sys
import pytest
from bmd_agent.resources import transport

def refuse(*args, **kwargs):
    raise RuntimeError("SSH control path validated during collection")

transport._validate_control_path = refuse
transport.local_transport_supported = lambda: False
sys.exit(pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", sys.argv[1]]))
"""


def test_suite_collects_without_platform_dependent_transport_validation():
    """Collect the whole suite with control-path validation poisoned and the
    local transport reported unsupported (as on Windows): collection must not
    construct invocations, validate control paths or select local mode."""

    result = subprocess.run(
        [sys.executable, "-c", _COLLECT_AS_IF_UNSUPPORTED, str(TESTS_DIR)],
        capture_output=True,
        text=True,
        cwd=TESTS_DIR.parent,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "error" not in result.stdout.lower().splitlines()[-1]
