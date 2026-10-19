from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from bmd_agent import cli
from bmd_agent.config import ConfigurationError
from bmd_agent.resources.assessment import (
    METHOD_VERSION,
    PROCESS_FAILED,
    PROCESS_SUCCEEDED,
    PROCESS_UNAVAILABLE,
    SCHEMA_ID,
    SCHEMA_VERSION,
    ArtifactEvidence,
    AssessmentStage,
    CompletionAssessmentInput,
    ProducerProvenance,
    ScientificOutputSummary,
    assess_completion,
    build_job_completion_assessment,
    build_lifecycle_completion_assessment,
    serialize_completion_assessment,
)
from bmd_agent.presentation import (
    build_job_concise_summary,
    build_lifecycle_concise_summary,
)
from bmd_agent.resources.completion import (
    STAGE_CONVERGED,
    STAGE_CONVERGENCE_UNDETERMINED,
    STAGE_NOT_CONVERGED,
    StageCompletion,
    StageConvergence,
)
from bmd_agent.resources.lifecycle import (
    BmdWorkflowDiscovery,
    LifecycleAnalysis,
    LifecycleState,
    LocalExecutionDiagnostics,
    LocalFileEvidence,
)
from bmd_agent.resources.run import (
    AttemptStateObservation,
    ComparisonObservation,
    JobInspection,
    LogRuntimeObservation,
    PathObservation,
    RunDiagnosis,
    RunInspection,
    ScientificResult,
    StageTrajectoryObservation,
    StructureObservation,
    TerminationObservation,
    WorkflowStage,
)
from bmd_agent.resources.slurm import SlurmAccountingRecord
from test_lifecycle import write_stage_complete


def stage_completion(
    index: int = 1,
    *,
    status: str = STAGE_CONVERGED,
    terminated: bool | None = True,
) -> StageCompletion:
    return StageCompletion(
        index=index,
        label=f"stage {index}",
        started=True,
        terminated_normally=terminated,
        convergence=StageConvergence(
            status,
            not_converged_scopes=("ionic",) if status == STAGE_NOT_CONVERGED else (),
            undetermined_scopes=("electronic",)
            if status == STAGE_CONVERGENCE_UNDETERMINED
            else (),
        ),
    )


def assessment_input(
    stage_type: str = "static",
    *,
    completions: tuple[StageCompletion, ...] | None = None,
    stages: tuple[AssessmentStage, ...] | None = None,
    artifacts: tuple[ArtifactEvidence, ...] = (
        ArtifactEvidence("vasprun.xml", True, True),
        ArtifactEvidence("contcar", True, True),
    ),
    scientific: ScientificOutputSummary | None = None,
    process_outcome: str = PROCESS_UNAVAILABLE,
) -> CompletionAssessmentInput:
    if stages is None:
        stages = (AssessmentStage(1, "stage 1", stage_type, "pbe"),)
    if completions is None:
        completions = tuple(stage_completion(stage.index) for stage in stages)
    if scientific is None:
        scientific = ScientificOutputSummary(
            final_energy_ev=-10.0,
            final_energy_reliable=True,
            final_energy_source="vasprun",
            final_structure_usable=True,
            dos_available=True,
            dos_reliable=True,
            band_gap_ev=1.2,
            band_kpoints=24,
            bands=16,
            band_structure_reliable=True,
        )
    return CompletionAssessmentInput(
        subject_id="fixture-run",
        calculation_type="BMD Compute",
        scope="bmd_compute_workflow",
        stages=stages,
        stage_completion=completions,
        process_outcome=process_outcome,
        artifacts=artifacts,
        scientific=scientific,
        producer_provenance=ProducerProvenance(
            "available",
            "BMD Compute",
            "bmd_compute",
            "a" * 40,
            "clean",
        ),
    )


@pytest.mark.parametrize(
    ("stage_type", "artifacts"),
    (
        ("static", (ArtifactEvidence("vasprun.xml", True, True),)),
        ("relax", (ArtifactEvidence("contcar", True, True),)),
        ("dos", (ArtifactEvidence("doscar", True, True),)),
        ("band_structure", (ArtifactEvidence("vasprun.xml", True, True),)),
    ),
)
def test_supported_single_stage_outputs_can_complete(
    stage_type: str,
    artifacts: tuple[ArtifactEvidence, ...],
) -> None:
    result = assess_completion(assessment_input(stage_type, artifacts=artifacts))

    assert result.verdict == "completed"
    assert result.assessment_status == "assessed"
    assert any(item.reason_code == "FINAL_OUTPUT_REQUIREMENT_SATISFIED" for item in result.findings)


def test_explicit_ionic_nonconvergence_is_incomplete() -> None:
    evidence = assessment_input(
        "relax",
        completions=(stage_completion(status=STAGE_NOT_CONVERGED),),
    )

    result = assess_completion(evidence)

    assert result.verdict == "incomplete"
    assert any(item.reason_code == "STAGE_CONVERGENCE_NOT_REACHED" for item in result.findings)


def test_explicit_electronic_nonconvergence_is_incomplete() -> None:
    completion = stage_completion(status=STAGE_NOT_CONVERGED)
    completion = replace(
        completion,
        convergence=StageConvergence(
            STAGE_NOT_CONVERGED,
            not_converged_scopes=("electronic",),
        ),
    )

    result = assess_completion(
        assessment_input("static", completions=(completion,))
    )

    assert result.verdict == "incomplete"
    assert any(
        item.reason_code == "STAGE_CONVERGENCE_NOT_REACHED"
        and "Electronic" in item.message
        for item in result.findings
    )


def test_missing_or_truncated_required_output_is_not_positive() -> None:
    missing = assess_completion(
        assessment_input(
            "static",
            artifacts=(ArtifactEvidence("vasprun.xml", False, False),),
            scientific=ScientificOutputSummary(),
        )
    )
    truncated = assess_completion(
        assessment_input(
            "static",
            scientific=ScientificOutputSummary(
                unavailable=("vasprun.xml could not be parsed completely",),
            ),
        )
    )

    assert missing.verdict == "incomplete"
    assert truncated.verdict == "incomplete"
    assert all(result.verdict != "completed" for result in (missing, truncated))


def test_multistage_requires_every_declared_stage() -> None:
    stages = (
        AssessmentStage(1, "PBE relaxation", "relax", "pbe"),
        AssessmentStage(2, "PBE static", "static", "pbe"),
    )
    result = assess_completion(
        assessment_input(
            stages=stages,
            completions=(
                stage_completion(1),
                StageCompletion(
                    2,
                    "stage 2",
                    False,
                    False,
                    StageConvergence(
                        STAGE_CONVERGENCE_UNDETERMINED,
                        undetermined_scopes=("electronic",),
                    ),
                ),
            ),
            process_outcome=PROCESS_SUCCEEDED,
        )
    )

    assert result.verdict == "incomplete"
    assert any(item.reason_code == "STAGE_EXECUTION_UNAVAILABLE" for item in result.findings)


def test_missing_declared_stage_cannot_complete_workflow() -> None:
    stages = (
        AssessmentStage(1, "PBE relaxation", "relax", "pbe"),
        AssessmentStage(2, "PBE static", "static", "pbe"),
    )

    result = assess_completion(
        assessment_input(stages=stages, completions=(stage_completion(1),))
    )

    assert result.verdict == "unknown"
    assert result.assessment_status == "insufficient_evidence"
    assert any(
        item.reason_code == "STAGE_EXECUTION_UNAVAILABLE" and item.stage_index == 2
        for item in result.findings
    )
    assert any(
        item.reason_code == "STAGE_COMPLETION_MODEL_INCONSISTENT"
        for item in result.findings
    )


@pytest.mark.parametrize(
    ("stages", "completions", "reason_code"),
    (
        (
            (AssessmentStage(1, "stage 1", "static", "pbe"),),
            (stage_completion(1), stage_completion(1)),
            "STAGE_COMPLETION_INDEX_DUPLICATE",
        ),
        (
            (AssessmentStage(1, "stage 1", "static", "pbe"),),
            (stage_completion(1), stage_completion(2)),
            "STAGE_COMPLETION_INDEX_UNEXPECTED",
        ),
        (
            (
                AssessmentStage(1, "stage 1a", "static", "pbe"),
                AssessmentStage(1, "stage 1b", "static", "pbe"),
            ),
            (stage_completion(1),),
            "DECLARED_STAGE_INDEX_DUPLICATE",
        ),
    ),
)
def test_duplicate_or_unexpected_stage_indices_fail_closed(
    stages: tuple[AssessmentStage, ...],
    completions: tuple[StageCompletion, ...],
    reason_code: str,
) -> None:
    result = assess_completion(
        assessment_input(stages=stages, completions=completions)
    )

    assert result.verdict == "unknown"
    assert result.assessment_status == "insufficient_evidence"
    assert any(item.reason_code == reason_code for item in result.findings)


@pytest.mark.parametrize(
    ("stage_type", "scientific"),
    (
        (
            "static",
            ScientificOutputSummary(
                final_energy_ev=-10.0,
                final_energy_reliable=False,
                final_energy_source="partial_vasprun",
                unavailable=("vasprun.xml could not be parsed completely",),
            ),
        ),
        (
            "band_structure",
            ScientificOutputSummary(
                band_gap_ev=1.2,
                band_kpoints=24,
                bands=16,
                band_structure_reliable=False,
                unavailable=("vasprun.xml could not be parsed completely",),
            ),
        ),
    ),
)
def test_partial_values_from_malformed_vasprun_cannot_satisfy_required_output(
    stage_type: str,
    scientific: ScientificOutputSummary,
) -> None:
    result = assess_completion(
        assessment_input(stage_type, scientific=scientific)
    )

    assert result.verdict == "incomplete"
    assert any(item.reason_code == "FINAL_OUTPUT_UNUSABLE" for item in result.findings)


def test_remote_doscar_with_unknown_usability_cannot_complete() -> None:
    result = assess_completion(
        assessment_input(
            "dos",
            artifacts=(ArtifactEvidence("doscar", True, None),),
            scientific=ScientificOutputSummary(),
        )
    )

    assert result.verdict == "unknown"
    assert result.assessment_status == "insufficient_evidence"
    assert any(
        item.reason_code == "FINAL_OUTPUT_REQUIREMENT_UNAVAILABLE"
        for item in result.findings
    )


def test_two_byte_doscar_without_scientific_result_cannot_complete() -> None:
    evidence = replace(
        assessment_input(
            "dos",
            artifacts=(ArtifactEvidence("doscar", True, True),),
        ),
        scientific=None,
    )

    result = assess_completion(evidence)

    assert result.verdict == "unknown"
    assert result.assessment_status == "insufficient_evidence"
    assert any(
        item.reason_code == "FINAL_OUTPUT_REQUIREMENT_UNAVAILABLE"
        for item in result.findings
    )


@pytest.mark.parametrize(
    "scientific",
    (
        ScientificOutputSummary(
            unavailable=("vasprun.xml could not be parsed completely",),
        ),
        ScientificOutputSummary(
            error="complete_dos parsing raised an exception",
        ),
        ScientificOutputSummary(
            dos_available=True,
            dos_reliable=False,
            unavailable=("vasprun.xml could not be parsed completely",),
        ),
    ),
)
def test_unusable_or_partial_dos_parser_evidence_cannot_complete(
    scientific: ScientificOutputSummary,
) -> None:
    result = assess_completion(
        assessment_input(
            "dos",
            artifacts=(ArtifactEvidence("doscar", True, True),),
            scientific=scientific,
        )
    )

    assert result.verdict == "incomplete"
    assert any(item.reason_code == "FINAL_OUTPUT_UNUSABLE" for item in result.findings)


@pytest.mark.parametrize(
    "doscar",
    (
        ArtifactEvidence("doscar", False, False),
        ArtifactEvidence("doscar", True, False),
    ),
)
def test_missing_or_empty_doscar_without_parsed_dos_is_incomplete(
    doscar: ArtifactEvidence,
) -> None:
    result = assess_completion(
        assessment_input(
            "dos",
            artifacts=(doscar,),
            scientific=ScientificOutputSummary(),
        )
    )

    assert result.verdict == "incomplete"
    assert any(
        item.reason_code == "FINAL_OUTPUT_REQUIRED_ARTIFACT_MISSING"
        for item in result.findings
    )


def test_reliably_parsed_complete_dos_satisfies_output_requirement() -> None:
    result = assess_completion(
        assessment_input(
            "dos",
            artifacts=(ArtifactEvidence("doscar", True, None),),
            scientific=ScientificOutputSummary(
                dos_available=True,
                dos_reliable=True,
            ),
        )
    )

    assert result.verdict == "completed"
    assert any(
        item.reason_code == "FINAL_OUTPUT_REQUIREMENT_SATISFIED"
        for item in result.findings
    )


def test_unknown_convergence_stays_unknown_even_when_scheduler_succeeded() -> None:
    result = assess_completion(
        assessment_input(
            completions=(
                stage_completion(
                    status=STAGE_CONVERGENCE_UNDETERMINED,
                    terminated=None,
                ),
            ),
            process_outcome=PROCESS_SUCCEEDED,
        )
    )

    assert result.verdict == "unknown"
    assert result.assessment_status == "insufficient_evidence"


def test_confirmed_process_failure_differs_from_unavailable_process_evidence() -> None:
    completion = stage_completion(
        status=STAGE_CONVERGENCE_UNDETERMINED,
        terminated=None,
    )
    failed = assess_completion(
        assessment_input(
            completions=(completion,),
            process_outcome=PROCESS_FAILED,
        )
    )
    unavailable = assess_completion(
        assessment_input(
            completions=(completion,),
            process_outcome=PROCESS_UNAVAILABLE,
        )
    )

    assert failed.verdict == "incomplete"
    assert failed.assessment_status == "assessed"
    assert any(item.reason_code == "PROCESS_TERMINATION_FAILED" for item in failed.findings)
    assert unavailable.verdict == "unknown"
    assert unavailable.assessment_status == "insufficient_evidence"


def test_phonon_validation_is_explicitly_unsupported_not_failed() -> None:
    result = assess_completion(assessment_input("phonons"))
    negative = assess_completion(
        assessment_input(
            "phonons",
            completions=(stage_completion(status=STAGE_NOT_CONVERGED),),
        )
    )

    assert result.verdict == "unknown"
    assert negative.verdict == "unknown"
    assert result.assessment_status == "unsupported"
    assert any(item.reason_code == "PHONON_COMPLETION_UNSUPPORTED_C1" for item in result.findings)
    assert any("imaginary modes" in item for item in result.limitations)


def test_schema_and_method_versions_are_independent_and_bounded() -> None:
    payload = serialize_completion_assessment(assess_completion(assessment_input()))

    assert payload["schema_id"] == SCHEMA_ID
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["method_version"] == METHOD_VERSION
    assert isinstance(payload["schema_version"], int)
    assert payload["subject"]["id"] == "fixture-run"
    encoded = json.dumps(payload, sort_keys=True)
    assert "POTCAR" not in encoded
    assert "environment" not in encoded


def lifecycle_analysis(root: Path) -> LifecycleAnalysis:
    scientific = ScientificResult(
        source_paths=(str(root / "vasprun.xml"),),
        final_energy_ev=-10.0,
        electronic_convergence=True,
        structure=StructureObservation(str(root / "CONTCAR"), site_count=2),
    )
    return LifecycleAnalysis(
        state=LifecycleState.COMPLETED,
        directory=root,
        calculation_kind="direct VASP",
        message="complete",
        output_files={
            "vasprun.xml": LocalFileEvidence("vasprun.xml", root / "vasprun.xml", True, 10),
            "CONTCAR": LocalFileEvidence("CONTCAR", root / "CONTCAR", True, 10),
        },
        normal_completion=True,
        scientific=scientific,
        diagnostics=LocalExecutionDiagnostics(
            trajectories=(
                StageTrajectoryObservation(
                    stage_index=1,
                    stage_label="work_dir",
                    stage_type="direct_vasp",
                    theory="unknown",
                    directory=str(root),
                    oszicar_present=True,
                    vasprun_present=True,
                    converged_electronic=True,
                ),
            ),
        ),
        stage_completion=(stage_completion(),),
    )


def completed_job() -> JobInspection:
    scheduler = SlurmAccountingRecord(
        job_id="12345",
        name="fixture",
        state="COMPLETED",
        elapsed="00:10:00",
        start="2026-01-01T00:00:00",
        end="2026-01-01T00:10:00",
        partition="compute",
        exit_code="0:0",
    )
    trajectory = StageTrajectoryObservation(
        stage_index=1,
        stage_label="stage_01",
        stage_type="static",
        theory="pbe",
        directory="/flow",
        oszicar_present=True,
        vasprun_present=True,
        converged_electronic=True,
    )
    scientific = ScientificResult(
        source_paths=("/flow/vasprun.xml",),
        final_energy_ev=-10.0,
        electronic_convergence=True,
    )
    run = RunInspection(
        flow_root="/flow",
        submission_path="/flow/submission.json",
        workflow_stages=(WorkflowStage(1, "static", "pbe", (), None),),
        stage_directories=(),
        result_directory=PathObservation("result_dir", "/flow", "directory", True, "producer_provenance"),
        log_paths=(),
        final_artifacts=(PathObservation("vasprun", "/flow/vasprun.xml", "file", True, "artifact_observation"),),
        producer_git={"git_commit": "a" * 40, "state": "clean"},
        cluster_request={},
        resources_request={},
        environment_policy={},
        attempt_state=AttemptStateObservation(None, False),
        job_id="12345",
        scheduler=scheduler,
        scheduler_error=None,
        runtime=LogRuntimeObservation("log_observation", (), None, {}, {}, {}),
        scientific=scientific,
        comparison=ComparisonObservation("unavailable", "agent_comparison"),
    )
    return JobInspection(
        job_id="12345",
        scheduler=scheduler,
        scheduler_error=None,
        scheduler_work_dir="/flow",
        calculation_directory="/flow",
        calculation_type="BMD Compute",
        calculation_reason=None,
        bmd_compute=RunDiagnosis(
            run,
            TerminationObservation(scheduler_state="COMPLETED", scheduler_exit_code="0:0"),
            (trajectory,),
        ),
    )


def job_with_final_stage(
    stage_type: str,
    scientific: ScientificResult,
    *,
    artifact_label: str = "vasprun",
    artifact_path: str = "/flow/vasprun.xml",
) -> JobInspection:
    job = completed_job()
    assert job.bmd_compute is not None
    run = replace(
        job.bmd_compute.inspection,
        workflow_stages=(WorkflowStage(1, stage_type, "pbe", (), None),),
        final_artifacts=(
            PathObservation(
                artifact_label,
                artifact_path,
                "file",
                True,
                "artifact_observation",
            ),
        ),
        scientific=scientific,
    )
    return replace(job, bmd_compute=replace(job.bmd_compute, inspection=run))


def test_equivalent_local_and_job_evidence_produces_same_verdict(tmp_path: Path) -> None:
    local = build_lifecycle_completion_assessment(lifecycle_analysis(tmp_path))
    remote = build_job_completion_assessment(completed_job())

    assert local.verdict == remote.verdict == "completed"
    assert local.process_outcome == remote.process_outcome == PROCESS_SUCCEEDED


@pytest.mark.parametrize(
    ("stage_type", "scientific"),
    (
        (
            "static",
            ScientificResult(
                source_paths=("/flow/vasprun.xml",),
                final_energy_ev=-10.0,
                unavailable=("vasprun.xml could not be parsed completely",),
            ),
        ),
        (
            "band_structure",
            ScientificResult(
                source_paths=("/flow/vasprun.xml",),
                band_gap_ev=1.2,
                band_kpoints=24,
                bands=16,
                unavailable=("vasprun.xml could not be parsed completely",),
            ),
        ),
    ),
)
def test_job_adapter_does_not_promote_partial_malformed_vasprun_values(
    stage_type: str,
    scientific: ScientificResult,
) -> None:
    result = build_job_completion_assessment(
        job_with_final_stage(stage_type, scientific)
    )

    assert result.verdict == "incomplete"
    assert any(item.reason_code == "FINAL_OUTPUT_UNUSABLE" for item in result.findings)


def test_job_adapter_does_not_treat_remote_doscar_presence_as_usability() -> None:
    job = job_with_final_stage(
        "dos",
        ScientificResult(source_paths=("/flow/vasprun.xml",)),
        artifact_label="DOSCAR",
        artifact_path="/flow/DOSCAR",
    )

    result = build_job_completion_assessment(job)

    assert result.verdict == "unknown"
    assert result.assessment_status == "insufficient_evidence"
    assert any(
        item.reason_code == "FINAL_OUTPUT_REQUIREMENT_UNAVAILABLE"
        for item in result.findings
    )


def test_failed_process_has_local_job_and_student_focused_parity(tmp_path: Path) -> None:
    timeout = replace(
        completed_job().scheduler,
        state="TIMEOUT",
        exit_code="0:0",
    )
    local = replace(
        lifecycle_analysis(tmp_path),
        state=LifecycleState.INCOMPLETE,
        scheduler=timeout,
        normal_completion=False,
    )
    job = completed_job()
    assert job.bmd_compute is not None
    remote = replace(
        job,
        scheduler=timeout,
        bmd_compute=replace(
            job.bmd_compute,
            inspection=replace(job.bmd_compute.inspection, scheduler=timeout),
            termination=replace(
                job.bmd_compute.termination,
                scheduler_state="TIMEOUT",
                scheduler_exit_code="0:0",
            ),
        ),
    )

    local_assessment = build_lifecycle_completion_assessment(local)
    job_assessment = build_job_completion_assessment(remote)

    assert local_assessment.verdict == job_assessment.verdict == "incomplete"
    assert local_assessment.process_outcome == job_assessment.process_outcome == PROCESS_FAILED
    assert build_lifecycle_concise_summary(local).status == local_assessment.verdict.upper()
    assert build_job_concise_summary(remote).status == job_assessment.verdict.upper()


def test_missing_scheduler_evidence_does_not_become_confirmed_failure(tmp_path: Path) -> None:
    completion = stage_completion(
        status=STAGE_CONVERGENCE_UNDETERMINED,
        terminated=None,
    )
    analysis = replace(
        lifecycle_analysis(tmp_path),
        state=LifecycleState.UNKNOWN,
        scheduler=None,
        normal_completion=None,
        stage_completion=(completion,),
    )

    assessment = build_lifecycle_completion_assessment(analysis)

    assert assessment.process_outcome == PROCESS_UNAVAILABLE
    assert assessment.verdict == "unknown"
    assert build_lifecycle_concise_summary(analysis).status == "UNKNOWN"


def test_student_cli_and_focused_assessment_share_failed_process_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    timeout = replace(
        completed_job().scheduler,
        state="TIMEOUT",
        exit_code="0:0",
    )
    analysis = replace(
        lifecycle_analysis(tmp_path),
        state=LifecycleState.INCOMPLETE,
        scheduler=timeout,
        normal_completion=False,
    )
    monkeypatch.setattr(cli, "analyze_calculation_directory", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        cli,
        "load_resources",
        lambda: (_ for _ in ()).throw(ConfigurationError("not configured")),
    )

    assert cli.main([str(tmp_path)], command_name="bmd-check") == 0
    student_output = capsys.readouterr().out
    assert cli.main(
        ["assess", "completion", str(tmp_path)],
        command_name="bmd-check",
    ) == 0
    focused_output = capsys.readouterr().out

    assert "Status: INCOMPLETE" in student_output
    assert "Verdict: INCOMPLETE" in focused_output


def test_local_cli_and_json_reject_invalid_doscar_without_parsed_dos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "dos-run"
    root.mkdir()
    (root / "submission.json").write_text(
        json.dumps(
            {
                "flow_spec": {
                    "workflow_spec": {
                        "stages": [
                            {
                                "stage_type": "dos",
                                "theory": "pbe",
                                "modifiers": [],
                                "options": {},
                            }
                        ]
                    }
                },
                "paths": {
                    "stage_dirs": {},
                    "result_dir": str(root),
                },
            },
        ),
        encoding="utf-8",
    )
    write_stage_complete(root)
    (root / "DOSCAR").write_bytes(b"x\n")
    monkeypatch.setattr(
        cli,
        "load_resources",
        lambda: (_ for _ in ()).throw(ConfigurationError("not configured")),
    )

    assert cli.main([str(root)], command_name="bmd-check") == 0
    student_output = capsys.readouterr().out
    assert cli.main(
        ["assess", "completion", str(root), "--format", "json"],
        command_name="bmd-check",
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    assert "Status: INCOMPLETE" in student_output
    assert payload["verdict"] == "incomplete"
    assert any(
        item["reason_code"] == "FINAL_OUTPUT_UNUSABLE"
        for item in payload["findings"]
    )


def test_cli_json_is_the_internal_assessment_serialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis = lifecycle_analysis(tmp_path)
    monkeypatch.setattr(cli, "analyze_calculation_directory", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        cli,
        "load_resources",
        lambda: (_ for _ in ()).throw(ConfigurationError("not configured")),
    )

    assert cli.main(
        ["assess", "completion", str(tmp_path), "--format", "json"],
        command_name="bmd-check",
    ) == 0

    output = capsys.readouterr().out
    assert str(tmp_path) not in output
    assert json.loads(output) == serialize_completion_assessment(
        build_lifecycle_completion_assessment(analysis)
    )


def test_stage_path_invocation_is_reassessed_at_workflow_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    workflow = BmdWorkflowDiscovery(
        workflow_root=root,
        submission_path=root / "submission.json",
        submission={},
        workflow_stages=(
            {"stage_type": "static", "theory": "pbe", "modifiers": [], "options": {}},
        ),
        stage_bindings=(),
    )
    stage_view = replace(
        lifecycle_analysis(stage),
        directory=stage,
        calculation_kind="BMD Compute",
        bmd_workflow=workflow,
    )
    root_view = replace(
        lifecycle_analysis(root),
        directory=root,
        calculation_kind="BMD Compute",
        bmd_workflow=workflow,
    )
    calls: list[Path] = []

    def analyze(path: Path | str, **kwargs: object) -> LifecycleAnalysis:
        resolved = Path(path).resolve()
        calls.append(resolved)
        return stage_view if resolved == stage.resolve() else root_view

    monkeypatch.setattr(cli, "analyze_calculation_directory", analyze)
    monkeypatch.setattr(
        cli,
        "load_resources",
        lambda: (_ for _ in ()).throw(ConfigurationError("not configured")),
    )

    assert cli.show_completion_assessment(stage, output_format="json") == 0
    payload = json.loads(capsys.readouterr().out)

    assert calls == [stage.resolve(), root.resolve()]
    assert payload["verdict"] == "completed"
    assert payload["subject"]["scope"] == "bmd_compute_workflow"


def test_human_and_json_renderers_do_not_change_existing_default_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[Path | None] = []
    monkeypatch.setattr(
        cli,
        "show_current_directory",
        lambda directory=None, **kwargs: called.append(directory) or 0,
    )

    assert cli.main([], command_name="bmd-check") == 0
    assert called == [None]


def test_human_cli_uses_the_same_immutable_assessment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis = lifecycle_analysis(tmp_path)
    monkeypatch.setattr(cli, "analyze_calculation_directory", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        cli,
        "load_resources",
        lambda: (_ for _ in ()).throw(ConfigurationError("not configured")),
    )

    assert cli.main(
        ["assess", "completion", str(tmp_path)],
        command_name="bmd-check",
    ) == 0

    output = capsys.readouterr().out
    assert "Scientific Completion Assessment" in output
    assert "Method: scientific_completion 1.0.0-c1" in output
    assert "Verdict: COMPLETED" in output
