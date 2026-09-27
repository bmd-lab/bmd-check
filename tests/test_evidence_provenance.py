"""Adversarial tests for evidence availability and provenance integrity.

BMD Compute supplies executable policy facts from its *current* checkout, run
artifacts show what executed, and BMDex supplies supporting context. Agent must
keep "evidence unavailable" distinct from "evidence says none", must not imply
that current Compute policy was the policy used for a historical run, and must
present BMDex context as attributed context rather than causal diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from bmd_agent import cli
from bmd_agent.config import ResourceRegistry
from bmd_agent.presentation import (
    build_job_concise_summary,
    render_concise_summary,
)
from bmd_agent.resources.compute import (
    COMPUTE_POLICY_AVAILABLE,
    COMPUTE_POLICY_NOT_CONFIGURED,
    COMPUTE_POLICY_UNAVAILABLE,
    POLICY_ALIGNMENT_DIFFERENT,
    POLICY_ALIGNMENT_SAME,
    POLICY_ALIGNMENT_SAME_UNVERIFIED,
    POLICY_ALIGNMENT_UNKNOWN,
    ComputePolicyObservation,
    compute_policy_alignment,
    observe_compute_policies,
)
from bmd_agent.resources.run import build_run_comparison, inspect_remote_run, inspect_slurm_job

from test_compute import repository, schema_v1_payload
from bmd_agent.resources.custodian import TerminationEvidenceAssessment
from bmd_agent.resources.run import WorkflowStage

from test_presentation import bmd_job, contextual_enrichment, scheduler
from test_run_inspection import (
    FLOW_ROOT,
    RemoteFixture,
    cluster,
    default_directories,
    default_files,
    fake_scientific_parser,
    job_slurm_runner,
    modifier_policy,
    slurm_runner,
)


RUN_COMMIT = "abcdef0123456789"  # producer commit recorded in the fixture submission


def flat(text: str) -> str:
    return " ".join(text.split())


def policy(*, commit: str | None = RUN_COMMIT, dirty: bool | None = False) -> ComputePolicyObservation:
    return ComputePolicyObservation(
        status=COMPUTE_POLICY_AVAILABLE,
        policies=(modifier_policy(),),
        repository="bmd_compute",
        commit=commit,
        dirty=dirty,
        provenance_available=True,
    )


def producer(tmp_path: Path, runner) -> object:
    checkout = tmp_path / "bmd_compute"
    checkout.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    return observe_compute_policies(repository(checkout, python), runner=runner)


# ---------------------------------------------------------------------------
# Compute producer availability
# ---------------------------------------------------------------------------


def test_failing_compute_producer_is_unavailable_evidence(tmp_path: Path) -> None:
    def runner(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, output="", stderr="Traceback: boom")

    observation = producer(tmp_path, runner)

    assert observation.status == COMPUTE_POLICY_UNAVAILABLE
    assert observation.policies == ()
    assert observation.reason


def test_payload_without_modifier_policies_is_unavailable_not_empty(tmp_path: Path) -> None:
    payload = schema_v1_payload()

    observation = producer(
        tmp_path,
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr=""),
    )

    assert observation.status == COMPUTE_POLICY_UNAVAILABLE
    assert "does not declare modifier_policies" in observation.reason
    # Provenance of what was consulted is still retained.
    assert observation.commit == "05eacdb81234567890abcdef"


def test_declared_empty_policies_are_available_evidence_of_none(tmp_path: Path) -> None:
    payload = schema_v1_payload(dirty=True)
    payload["modifier_policies"] = []

    observation = producer(
        tmp_path,
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr=""),
    )

    assert observation.status == COMPUTE_POLICY_AVAILABLE
    assert observation.policies == ()
    assert observation.commit == "05eacdb81234567890abcdef"
    assert observation.dirty is True


def test_missing_compute_configuration_is_reported_not_silent() -> None:
    observation = cli.compute_policy_from_registry(ResourceRegistry({}, {}))

    assert observation.status == COMPUTE_POLICY_NOT_CONFIGURED
    assert observation.reason


def test_unavailable_policy_yields_no_checks_and_says_so(capsys) -> None:
    remote = RemoteFixture(files=default_files(), directories=default_directories())
    unavailable = ComputePolicyObservation(
        status=COMPUTE_POLICY_UNAVAILABLE,
        reason="BMD Compute capability producer failed.",
    )

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=slurm_runner,
        scientific_parser=fake_scientific_parser,
        modifier_policies=(modifier_policy(),),  # ignored: the observation is authoritative
        compute_policy=unavailable,
    )
    cli.print_run_inspection(inspection)
    output = capsys.readouterr().out

    assert inspection.input_expectations == ()
    assert inspection.compute_policy is unavailable
    assert "BMD Compute input-effect policy (producer_capability):" in output
    assert "status: unavailable" in output
    assert "reason: BMD Compute capability producer failed." in output
    assert "missing evidence, not a finding" in output
    assert "requested/executed checks (agent_comparison)" not in output


# ---------------------------------------------------------------------------
# Historical run versus current Compute checkout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy_commit", "policy_dirty", "run_state", "expected"),
    [
        (RUN_COMMIT, False, "clean", POLICY_ALIGNMENT_SAME),
        ("fedcba9876543210", False, "clean", POLICY_ALIGNMENT_DIFFERENT),
        (RUN_COMMIT, True, "clean", POLICY_ALIGNMENT_SAME_UNVERIFIED),
        (RUN_COMMIT, None, "clean", POLICY_ALIGNMENT_SAME_UNVERIFIED),
        (RUN_COMMIT, False, None, POLICY_ALIGNMENT_SAME_UNVERIFIED),
        (None, False, "clean", POLICY_ALIGNMENT_UNKNOWN),
    ],
)
def test_policy_alignment_is_never_assumed(policy_commit, policy_dirty, run_state, expected) -> None:
    run_git = {"git_commit": RUN_COMMIT}
    if run_state is not None:
        run_git["state"] = run_state

    assert compute_policy_alignment(policy(commit=policy_commit, dirty=policy_dirty), run_git) == expected


def test_alignment_is_unknown_without_run_commit_or_policy() -> None:
    assert compute_policy_alignment(policy(), {}) == POLICY_ALIGNMENT_UNKNOWN
    assert compute_policy_alignment(None, {"git_commit": RUN_COMMIT}) == POLICY_ALIGNMENT_UNKNOWN


def test_historical_run_checked_against_newer_compute_exposes_mismatch(capsys) -> None:
    remote = RemoteFixture(files=default_files(), directories=default_directories())
    current = policy(commit="fedcba9876543210")

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=slurm_runner,
        scientific_parser=fake_scientific_parser,
        compute_policy=current,
    )
    cli.print_run_inspection(inspection)
    output = flat(capsys.readouterr().out)

    assert inspection.input_expectations
    assert inspection.compute_policy is current
    assert "source: current checkout fedcba987654 (clean)" in output
    assert "run produced by: abcdef012345 (clean)" in output
    assert "policy alignment: different commit" in output
    assert "not established to be the policy used when this run was produced" in output


def test_matching_compute_commit_is_reported_as_same(capsys) -> None:
    remote = RemoteFixture(files=default_files(), directories=default_directories())

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=slurm_runner,
        scientific_parser=fake_scientific_parser,
        compute_policy=policy(),
    )
    cli.print_run_inspection(inspection)
    output = flat(capsys.readouterr().out)

    assert "policy alignment: same commit as the current BMD Compute checkout" in output
    assert "not established to be the policy used" not in output


def test_compare_runs_reports_per_run_policy_alignment(capsys) -> None:
    remote = RemoteFixture(files=default_files(), directories=default_directories())
    current = policy(commit="fedcba9876543210")
    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=slurm_runner,
        scientific_parser=fake_scientific_parser,
        compute_policy=current,
    )
    comparison = build_run_comparison((inspection, inspection), modifier_policies=current.policies)
    cli.print_run_comparison(comparison, compute_policy=current)
    output = flat(capsys.readouterr().out)

    assert output.count("policy alignment: different commit") == 2
    assert "Modifier policy warning" not in output


def test_compare_runs_distinguishes_unavailable_policy_from_no_checks(capsys) -> None:
    remote = RemoteFixture(files=default_files(), directories=default_directories())
    unavailable = ComputePolicyObservation(status=COMPUTE_POLICY_UNAVAILABLE, reason="timed out")
    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=slurm_runner,
        scientific_parser=fake_scientific_parser,
        compute_policy=unavailable,
    )
    comparison = build_run_comparison((inspection, inspection))
    cli.print_run_comparison(comparison, compute_policy=unavailable)
    output = flat(capsys.readouterr().out)

    assert "not performed: BMD Compute policy evidence is unavailable" in output
    assert "reason: timed out" in output


def test_job_inspection_retains_compute_policy_on_run_evidence() -> None:
    remote = RemoteFixture(files=default_files(), directories=default_directories())
    unavailable = ComputePolicyObservation(status=COMPUTE_POLICY_UNAVAILABLE, reason="failed")

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(work_dir=FLOW_ROOT),
        scientific_parser=fake_scientific_parser,
        max_vasprun_bytes=0,
        compute_policy=unavailable,
    )

    assert inspection.bmd_compute is not None
    assert inspection.bmd_compute.inspection.compute_policy is unavailable


# ---------------------------------------------------------------------------
# Job fallback through the scheduler WorkDir keeps final scientific results
# ---------------------------------------------------------------------------


def test_workdir_fallback_derives_final_scientific_results() -> None:
    remote = RemoteFixture(files=default_files(), directories=default_directories())

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(work_dir=FLOW_ROOT),
        scientific_parser=fake_scientific_parser,
        max_vasprun_bytes=0,
    )

    assert inspection.run_resolution is not None
    assert inspection.run_resolution.resolution_status != "resolved"
    scientific = inspection.bmd_compute.inspection.scientific
    assert scientific.final_formula == "Example2"
    assert scientific.final_energy_ev == -12.5
    assert not any("skipped" in item for item in scientific.unavailable)


# ---------------------------------------------------------------------------
# BMDex context is attributed and non-causal
# ---------------------------------------------------------------------------


HYBRID_STATEMENT = (
    "VASP hybrid-functional calculations include nonlocal exchange, whose evaluation "
    "can be much more expensive than semilocal DFT."
)


def failed_hybrid_job():
    return bmd_job(
        stages=(WorkflowStage(1, "static", "hse06", (), "stage_01"),),
        scheduler_record=scheduler(state="FAILED", exit_code="1:0"),
        termination=TerminationEvidenceAssessment("unknown", "insufficient_evidence"),
    )


def test_bmdex_context_is_attributed_and_not_presented_as_cause() -> None:
    enrichment = contextual_enrichment(HYBRID_STATEMENT)
    record = enrichment.evidence.records[0]
    record.record["sources"] = [
        {"source_type": "VASP Wiki", "authority": "VASP Software GmbH / VASP Wiki"},
    ]

    output = flat(
        render_concise_summary(
            build_job_concise_summary(failed_hybrid_job(), contextual_enrichment=enrichment)
        )
    )

    assert "Why this may have happened" not in output
    assert "Context from BMDex" in output
    assert "It is not a diagnosis of this run and does not establish why it stopped." in output
    assert HYBRID_STATEMENT.split(".")[0] in output
    assert "Source: BMDex record vasp.hybrid.test; cites VASP Software GmbH / VASP Wiki." in output
    # Agent's own cause assessment is unchanged by the presence of context.
    assert "Agent could not determine the cause from the available evidence." in output


def test_bmdex_record_without_sources_is_still_attributed_to_bmdex() -> None:
    output = flat(
        render_concise_summary(
            build_job_concise_summary(
                failed_hybrid_job(),
                contextual_enrichment=contextual_enrichment(HYBRID_STATEMENT),
            )
        )
    )

    assert "Source: BMDex record vasp.hybrid.test." in output
    assert "cites" not in output


def test_no_bmdex_section_without_matching_records() -> None:
    output = render_concise_summary(build_job_concise_summary(failed_hybrid_job()))

    assert "Context from BMDex" not in output
    assert "BMDex" not in output
