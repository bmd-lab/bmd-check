from pathlib import Path
from types import SimpleNamespace

import pytest

from bmd_agent import cli
from bmd_agent.resources.compute import compute_policy_not_configured
from bmd_agent.config import ConfigurationError, ResourceRegistry, SlurmClusterResource
from bmd_agent.resources.run import (
    InitialStructureComparison,
    QuantityComparison,
    RunComparison,
)


def test_cli_nonexistent_nonnumeric_target_is_clean_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(["frobnicate"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert (
        "Target was not recognized as a SLURM job ID or existing calculation path."
        in captured.out
    )
    assert "Traceback" not in captured.out


def test_bare_numeric_target_and_explicit_job_use_same_job_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_show_job(
        job_id: str,
        registry: ResourceRegistry | None = None,
        *,
        trajectory_json: bool = False,
        **kwargs: object,
    ) -> int:
        calls.append((job_id, trajectory_json))
        return 0

    monkeypatch.setattr(cli, "show_job", fake_show_job)

    assert cli.main(["21853598"]) == 0
    assert cli.main(["job", "21853598"]) == 0
    assert calls == [("21853598", False), ("21853598", False)]


def test_bare_numeric_and_explicit_job_profile_use_same_job_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_show_job(
        job_id: str,
        registry: ResourceRegistry | None = None,
        *,
        trajectory_json: bool = False,
        profile: bool = False,
        **kwargs: object,
    ) -> int:
        assert trajectory_json is False
        calls.append((job_id, profile))
        return 0

    monkeypatch.setattr(cli, "show_job", fake_show_job)

    assert cli.main(["21853598", "--profile"]) == 0
    assert cli.main(["job", "21853598", "--profile"]) == 0
    assert calls == [("21853598", True), ("21853598", True)]


def test_bare_numeric_target_wins_over_same_named_local_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "21853598").mkdir()
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(cli, "show_job", lambda job_id, **kwargs: calls.append(job_id) or 0)
    monkeypatch.setattr(
        cli,
        "show_current_directory",
        lambda *args, **kwargs: pytest.fail("bare numeric target must not dispatch as a path"),
    )

    assert cli.main(["21853598"]) == 0
    assert calls == ["21853598"]


@pytest.mark.parametrize("target_kind", ("relative", "absolute", "numeric_relative"))
def test_existing_path_target_uses_lifecycle_analysis(
    target_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "21853598" if target_kind == "numeric_relative" else "calculation"
    calculation = tmp_path / name
    calculation.mkdir()
    monkeypatch.chdir(tmp_path)
    targets = {
        "relative": "./calculation",
        "absolute": str(calculation.resolve()),
        "numeric_relative": "./21853598",
    }
    analyzed: list[Path] = []

    def fake_show_current_directory(
        directory: Path | None = None,
        registry: ResourceRegistry | None = None,
        **kwargs: object,
    ) -> int:
        assert directory is not None
        analyzed.append(directory.resolve())
        return 0

    monkeypatch.setattr(cli, "show_current_directory", fake_show_current_directory)
    monkeypatch.setattr(
        cli,
        "show_job",
        lambda *args, **kwargs: pytest.fail("explicit path syntax must not dispatch as a job"),
    )

    assert cli.main([targets[target_kind]]) == 0
    assert analyzed == [calculation.resolve()]


def test_normal_job_verbose_and_profile_flags_compose_through_same_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_show_job(job_id: str, **kwargs: object) -> int:
        calls.append({"job_id": job_id, **kwargs})
        return 0

    monkeypatch.setattr(cli, "show_job", fake_show_job)

    assert cli.main(["21853598", "--verbose"]) == 0
    assert cli.main(["21853598", "--verbose", "--profile"]) == 0
    assert calls[0]["verbose"] is True
    assert calls[0]["profile"] is False
    assert calls[1]["verbose"] is True
    assert calls[1]["profile"] is True
    assert calls[0]["detailed_evidence_command"] == "bmd-agent 21853598 --verbose"


def test_explicit_job_verbose_profile_uses_same_job_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli,
        "show_job",
        lambda job_id, **kwargs: calls.append({"job_id": job_id, **kwargs}) or 0,
    )

    assert cli.main(["job", "21853598", "--profile", "--verbose"]) == 0

    assert calls == [
        {
            "job_id": "21853598",
            "trajectory_json": False,
            "verbose": True,
            "profile": True,
            "detailed_evidence_command": "bmd-agent job 21853598 --verbose",
        }
    ]


def test_cwd_and_path_verbose_route_to_existing_lifecycle_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path | None, bool]] = []

    def fake_show_current_directory(
        directory: Path | None = None,
        registry: ResourceRegistry | None = None,
        *,
        verbose: bool = False,
        detailed_evidence_command: str | None = None,
    ) -> int:
        calls.append((directory, verbose))
        return 0

    monkeypatch.setattr(cli, "show_current_directory", fake_show_current_directory)

    assert cli.main(["--verbose"]) == 0
    assert cli.main([str(tmp_path), "--verbose"]) == 0
    assert calls == [(None, True), (tmp_path, True)]


def test_local_path_analysis_does_not_create_reusable_remote_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calculation = tmp_path / "calculation"
    calculation.mkdir()

    class ForbiddenRemoteSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pytest.fail("local path analysis must not create a reusable SSH session")

    monkeypatch.setattr(cli, "ReusableSshSession", ForbiddenRemoteSession)
    monkeypatch.setattr(cli, "load_resources", lambda: ResourceRegistry({}, {}))
    monkeypatch.setattr(
        cli,
        "analyze_calculation_directory",
        lambda directory, **kwargs: SimpleNamespace(
            directory=directory,
            state=SimpleNamespace(value="UNKNOWN"),
            calculation_kind="none",
            message="no calculation",
        ),
    )
    monkeypatch.setattr(cli, "print_lifecycle_analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "enrich_lifecycle_with_bmdex_domain_context",
        lambda *args, **kwargs: None,
    )

    assert cli.show_current_directory(calculation, verbose=True) == 0


def test_cli_structure_requires_directory(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["structure"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Usage: bmd-agent structure <remote-directory>" in captured.out


def test_cli_compare_runs_requires_two_roots(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["compare-runs", "/one"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Usage: bmd-agent compare-runs <flow-a> <flow-b>" in captured.out


def test_cli_diagnose_run_requires_root(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["diagnose-run"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Usage: bmd-agent diagnose-run <remote-flow-root>" in captured.out


def test_cli_job_requires_id(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["job"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Usage: bmd-agent job <SLURM_JOB_ID>" in captured.out


def test_cli_compare_runs_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cluster = SlurmClusterResource(
        key="powerslurm",
        name="PowerSLURM",
        ssh_host="powerslurm-bmdguest",
        partition="leeburton-pool",
        access="observational",
        allowed_remote_roots=(),
    )
    registry = ResourceRegistry(repositories={}, clusters={"powerslurm": cluster})
    baseline = SimpleNamespace(
        flow_root="/flow/a",
        producer_git={"git_commit": "abcdef012345", "state": "clean"},
        scheduler=SimpleNamespace(state="COMPLETED"),
        input_expectations=(),
    )
    comparison_run = SimpleNamespace(
        flow_root="/flow/b",
        producer_git={"git_commit": "abcdef012345", "state": "clean"},
        scheduler=SimpleNamespace(state="COMPLETED"),
        input_expectations=(),
    )
    comparison = RunComparison(
        inspections=(baseline, comparison_run),
        labels={"/flow/a": "PBE", "/flow/b": "PBE + modifier"},
        quantities=(
            QuantityComparison(
                run_label="PBE + modifier",
                flow_root="/flow/b",
                quantity="lattice_c",
                label="lattice c",
                unit="A",
                baseline_value=6.0,
                comparison_value=5.4,
                delta=-0.6,
                percent_delta=-10.0,
                status="available",
            ),
        ),
        initial_structure=InitialStructureComparison("match"),
        energy_warning="energy values are observations, not a ranking of method quality.",
    )

    monkeypatch.setattr(cli, "load_resources", lambda: registry)
    monkeypatch.setattr(
        cli,
        "compute_policy_from_registry",
        lambda registry, **kwargs: compute_policy_not_configured("not configured in test"),
    )
    monkeypatch.setattr(cli, "compare_remote_runs", lambda cluster, flow_roots, **kwargs: comparison)

    exit_code = cli.main(["compare-runs", "/flow/a", "/flow/b"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "BMD Compute Run Comparison" in captured.out
    assert "baseline: PBE" in captured.out
    assert "lattice c: 6.0 -> 5.4 A, delta -0.600000 A, -10.000%" in captured.out
    assert "not a ranking of method quality" in captured.out


def test_cli_reports_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_load_resources() -> None:
        raise ConfigurationError("No BMD Agent resource configuration found.")

    monkeypatch.setattr(cli, "load_resources", fail_load_resources)

    exit_code = cli.main(["status"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "No BMD Agent resource configuration found." in captured.err


def test_status_exposes_resolved_profile_as_context_not_live_observation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cluster = SlurmClusterResource(
        key="powerslurm",
        name="PowerSLURM",
        ssh_host="powerslurm-bmdguest",
        partition="leeburton-pool",
        access="observational",
        allowed_remote_roots=(),
        deployment_profile="power",
    )
    registry = ResourceRegistry(repositories={}, clusters={"powerslurm": cluster})

    assert cli.show_status(registry) == 0

    output = capsys.readouterr().out
    assert "Deployment context:" in output
    assert "profile: power (TAU POWER)" in output
    assert "profile schema: bmd_agent.deployment_profile v1" in output
    assert "expected scheduler: slurm" in output
    assert "live" not in output.lower()


def test_queue_uses_deployment_local_operational_timeouts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cluster = SlurmClusterResource(
        key="powerslurm",
        name="PowerSLURM",
        ssh_host="powerslurm-bmdguest",
        partition="leeburton-pool",
        access="observational",
        allowed_remote_roots=(),
        ssh_connect_timeout_seconds=13,
        remote_command_timeout_seconds=31,
        scheduler_accounting_timeout_seconds=91,
    )
    registry = ResourceRegistry(repositories={}, clusters={"powerslurm": cluster})
    observed: dict[str, object] = {}

    def fake_queue(**kwargs: object) -> list:
        observed.update(kwargs)
        return []

    monkeypatch.setattr(cli, "get_queue", fake_queue)

    assert cli.show_queue(registry) == 0

    assert observed.pop("runner") is cli.observation_runner("powerslurm-bmdguest")
    assert observed == {
        "ssh_host": "powerslurm-bmdguest",
        "partition": "leeburton-pool",
        "timeout": 31,
        "ssh_connect_timeout": 13,
    }
    assert "No jobs in leeburton-pool." in capsys.readouterr().out
