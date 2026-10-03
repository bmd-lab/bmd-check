"""``bmd-check`` command and the local observational transport.

``bmd-check`` and ``bmd-agent`` are two console scripts for the same
implementation. Local mode (``ssh_host = "local"``) runs Agent's existing fixed
observational commands on this machine instead of over SSH; everything else --
path authorization, size limits, parsing and evidence logic -- is shared.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.metadata
import os
from pathlib import Path, PurePosixPath
import subprocess
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
from bmd_agent.resources import transport
from bmd_agent.resources.remote import ReusableSshSession
from bmd_agent.resources.run import inspect_remote_run, inspect_slurm_job
from bmd_agent.resources.slurm import get_job_accounting
from bmd_agent.resources.transport import (
    LOCAL_SHELL,
    LOCAL_TRANSPORT,
    LocalObservationSession,
    TransportError,
    local_command,
    observation_runner,
    run_local_command,
    run_ssh_command,
)
from bmd_agent.resources.vasp import RemotePathError, retrieve_remote_file

import test_run_inspection as ri


REPO_ROOT = Path(__file__).resolve().parents[1]
GUEST = "/bmd-db/guest"


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
    monkeypatch.chdir(tmp_path)

    cli.main([], command_name="bmd-check")
    check = capsys.readouterr().out
    cli.main([], command_name="bmd-agent")
    agent = capsys.readouterr().out

    assert "Detailed evidence:\n  bmd-check --verbose" in check
    assert "Detailed evidence:\n  bmd-agent --verbose" in agent
    assert check == agent.replace("bmd-agent", "bmd-check")


def test_command_name_does_not_leak_between_invocations(capsys):
    cli.main(["job"], command_name="bmd-check")
    capsys.readouterr()
    cli.main(["job"])

    assert "Usage: bmd-agent job" in capsys.readouterr().out


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


# --- SSH transport: BatchMode -------------------------------------------------------------


def recording_subprocess(monkeypatch, *, stdout="", returncode=0):
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, **kwargs):
        calls.append((list(command), kwargs))
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(transport.subprocess, "run", fake_run)
    return calls


def test_ssh_runner_adds_batch_mode(monkeypatch):
    calls = recording_subprocess(monkeypatch)

    run_ssh_command(["ssh", "powerslurm-bmdguest", "cat -- /allowed/INCAR"], capture_output=True)

    assert calls[0][0] == ["ssh", "-o", "BatchMode=yes", "powerslurm-bmdguest", "cat -- /allowed/INCAR"]


def test_default_remote_reads_and_scheduler_lookups_use_batch_mode(monkeypatch):
    calls = recording_subprocess(monkeypatch, stdout="")

    retrieve_remote_file("powerslurm-bmdguest", PurePosixPath("/allowed/INCAR"))
    get_job_accounting("powerslurm-bmdguest", "123")

    assert calls[0][0][:3] == ["ssh", "-o", "BatchMode=yes"]
    assert calls[1][0][:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def test_reusable_ssh_session_uses_batch_mode(monkeypatch):
    calls = recording_subprocess(monkeypatch)

    with ReusableSshSession("powerslurm-bmdguest", multiplex=True) as session:
        session.runner("remote")(["ssh", "powerslurm-bmdguest", "test -f /allowed/INCAR"], capture_output=True)

    executed = calls[0][0]
    assert executed[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert executed[-2:] == ["powerslurm-bmdguest", "test -f /allowed/INCAR"]
    assert any(item.startswith("ControlMaster=") for item in executed)


def test_ssh_runner_refuses_the_local_selector(monkeypatch):
    calls = recording_subprocess(monkeypatch)

    with pytest.raises(TransportError):
        run_ssh_command(["ssh", LOCAL_TRANSPORT, "cat -- /allowed/INCAR"])
    assert calls == []


# --- local transport: shape, safety, no SSH -----------------------------------------------


def test_local_command_runs_the_same_fixed_command_text_without_ssh():
    assert local_command(["ssh", "local", "cat -- /allowed/INCAR"]) == [LOCAL_SHELL, "-c", "cat -- /allowed/INCAR"]
    assert local_command(["ssh", "-o", "ConnectTimeout=10", "local", "sacct -j 1"]) == [
        LOCAL_SHELL,
        "-c",
        "sacct -j 1",
    ]


@pytest.mark.parametrize(
    ("command", "kwargs"),
    [
        ("cat /etc/passwd", {}),  # a string, not an argument vector
        (["cat", "/etc/passwd"], {}),  # not an Agent-built observational vector
        (["ssh", "powerslurm-bmdguest", "cat -- /x"], {}),  # another host
        (["ssh", "-o", "ProxyCommand=sh", "local", "cat -- /x"], {}),  # unexpected option
        (["ssh", "-F", "/tmp/cfg", "local", "cat -- /x"], {}),  # unexpected flag
        (["ssh", "local", "   "], {}),  # empty command
        (["ssh", "local", "cat -- /x"], {"shell": True}),  # shell=True
    ],
)
def test_local_transport_rejects_anything_but_agent_built_commands(command, kwargs):
    with pytest.raises(TransportError):
        local_command(command, kwargs)


def test_local_transport_never_invokes_ssh(monkeypatch):
    calls = recording_subprocess(monkeypatch, stdout="")

    run_local_command(["ssh", "local", "cat -- /allowed/INCAR"], capture_output=True)
    session = LocalObservationSession(runner=transport.subprocess.run)
    session.runner("remote")(["ssh", "local", "test -f /allowed/INCAR"], capture_output=True)

    assert [command[0] for command, _ in calls] == [LOCAL_SHELL, LOCAL_SHELL]
    assert all("ssh" not in command for command, _ in calls)


def test_local_selector_is_a_valid_deployment_value():
    registry = parse_resources(
        {
            "repositories": {},
            "clusters": {
                "powerslurm": {
                    "name": "POWER",
                    "ssh_host": "local",
                    "partition": "leeburton-pool",
                    "access": "observational",
                    "allowed_remote_roots": ["/bmd-db"],
                }
            },
        }
    )

    cluster = registry.clusters["powerslurm"]
    assert observation_runner(cluster.ssh_host) is run_local_command
    assert isinstance(cli._observation_session(cluster), LocalObservationSession)
    assert observation_runner("powerslurm-bmdguest") is run_ssh_command
    assert isinstance(cli._observation_session(replace(cluster, ssh_host="powerslurm-bmdguest")), ReusableSshSession)


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


def test_local_and_simulated_ssh_observation_parse_identically(tmp_path):
    files, directories = materialize_fixture(tmp_path)
    flow_root = rebase(ri.FLOW_ROOT, tmp_path)

    over_ssh = inspect_remote_run(
        fixture_cluster(tmp_path, "powerslurm-bmdguest"),
        flow_root,
        remote_runner=ri.RemoteFixture(files=files, directories=directories),
        slurm_runner=any_host_scheduler(tmp_path),
        scientific_parser=ri.fake_scientific_parser,
        modifier_policies=(ri.modifier_policy(),),
    )
    local = inspect_remote_run(
        fixture_cluster(tmp_path, LOCAL_TRANSPORT),
        flow_root,
        remote_runner=run_local_command,
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
    # A fake ssh that fails loudly proves local mode never reaches it.
    ssh = bin_dir / "ssh"
    ssh.write_text(f"#!/bin/sh\necho invoked >> {tmp_path / 'ssh-invoked'}\nexit 255\n", encoding="utf-8")
    ssh.chmod(0o755)
    return bin_dir


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


def test_local_job_inspection_matches_simulated_ssh(tmp_path, monkeypatch):
    data = tmp_path / "data"
    files, directories = materialize_fixture(data)
    bin_dir = write_fake_sacct(tmp_path, data)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    def ssh_scheduler(command, **kwargs):
        assert command[-2] == "powerslurm-bmdguest"
        return any_host_scheduler(data)(command, **kwargs)

    over_ssh = inspect_slurm_job(
        fixture_cluster(data, "powerslurm-bmdguest"),
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


def test_path_mode_scheduler_lookup_uses_the_local_runner(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    materialize_fixture(data)
    bin_dir = write_fake_sacct(tmp_path, data)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    cluster = fixture_cluster(data, LOCAL_TRANSPORT)
    seen = []
    real = cli.get_job_accounting

    def recording_accounting(*args, **kwargs):
        seen.append(kwargs.get("runner"))
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "get_job_accounting", recording_accounting)

    import test_lifecycle as lc

    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    lc.write_inputs(stage)
    lc.write_submission(root, stage_dir=stage, job_id="20893681")

    cli.show_current_directory(root, ResourceRegistry({}, {"powerslurm": cluster}), verbose=True)
    output = capsys.readouterr().out

    assert seen == [run_local_command]
    assert "Scheduler observation:" in output
    assert "state: COMPLETED" in output
    assert not (tmp_path / "ssh-invoked").exists()


# --- local mode keeps path protections ----------------------------------------------------


def test_local_mode_refuses_paths_outside_allowed_roots_before_running_anything(tmp_path, monkeypatch):
    calls = recording_subprocess(monkeypatch)
    cluster = fixture_cluster(tmp_path, LOCAL_TRANSPORT)

    for target in ("/etc", f"{tmp_path}{GUEST}/../../etc"):
        with pytest.raises(RemotePathError):
            inspect_remote_run(cluster, target, remote_runner=run_local_command)
    assert calls == []


def test_local_mode_refuses_an_unauthorized_scheduler_workdir(tmp_path, monkeypatch):
    def scheduler(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=ri.job_sacct_output(work_dir="/etc"), stderr="")

    inspection = inspect_slurm_job(
        fixture_cluster(tmp_path, LOCAL_TRANSPORT),
        "20893681",
        remote_runner=run_local_command,
        slurm_runner=scheduler,
    )

    assert inspection.calculation_directory is None
    assert "not authorized" in inspection.calculation_reason


def test_shell_metacharacters_in_authorized_paths_are_never_executed(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    marker = tmp_path / "pwned"
    hostile = PurePosixPath(f"{root}/x; touch {marker}")

    with pytest.raises(subprocess.CalledProcessError):
        retrieve_remote_file(LOCAL_TRANSPORT, hostile, runner=run_local_command)
    assert not marker.exists()


def test_config_still_rejects_unsafe_transport_values():
    for value in ("local; rm -rf /", "local host", "$(id)"):
        with pytest.raises(ConfigurationError):
            parse_resources(
                {
                    "repositories": {},
                    "clusters": {
                        "powerslurm": {
                            "name": "POWER",
                            "ssh_host": value,
                            "partition": "leeburton-pool",
                            "access": "observational",
                            "allowed_remote_roots": ["/bmd-db"],
                        }
                    },
                }
            )
