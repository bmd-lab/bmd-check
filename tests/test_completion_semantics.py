"""Adversarial tests for workflow completion semantics.

SLURM ``COMPLETED 0:0`` means the batch process exited successfully. It must
never, on its own, imply that VASP converged or that every declared workflow
stage ran. These tests exercise both the local path (lifecycle) and the job
presentation paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bmd_agent.presentation import (
    build_job_concise_summary,
    build_lifecycle_concise_summary,
    render_concise_summary,
)
from bmd_agent.resources.completion import (
    STAGE_CONVERGED,
    STAGE_CONVERGENCE_UNDETERMINED,
    STAGE_NOT_CONVERGED,
    StageCompletion,
    StageConvergence,
    stage_convergence,
    workflow_completion_status,
)
from bmd_agent.resources.lifecycle import LifecycleState, analyze_calculation_directory
from bmd_agent.resources.run import ScientificResult, WorkflowStage

from test_lifecycle import (
    NORMAL_OUTCAR,
    scheduler_record,
    write_inputs,
    write_stage_complete,
    write_stage_partial,
    write_stage_relax_unconverged,
    write_stage_terminated_unverified,
)
from test_presentation import (
    bmd_job,
    rendered_job,
    scheduler,
    scientific,
    trajectory,
)


FORBIDDEN_COMPLETION_CLAIM = "No execution problems were detected."


def slurm_success(job_id: str) -> object:
    return scheduler_record(job_id=job_id, state="COMPLETED", exit_code="0:0")


def write_two_stage_workflow(root: Path) -> tuple[Path, Path]:
    stage_1 = root / "stage_01"
    stage_2 = root / "stage_02"
    payload = {
        "flow_spec": {
            "workflow_spec": {
                "stages": [
                    {"stage_type": "relax", "theory": "pbe", "modifiers": [], "options": {}},
                    {"stage_type": "static", "theory": "hse06", "modifiers": [], "options": {}},
                ]
            }
        },
        "submission": {"job_id": "21153721"},
        "paths": {
            "stage_dirs": {"stage_01": str(stage_1), "stage_02": str(stage_2)},
            "result_dir": str(stage_2),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "submission.json").write_text(json.dumps(payload), encoding="utf-8")
    return stage_1, stage_2


def write_single_relax_workflow(root: Path) -> Path:
    stage = root / "stage_01"
    payload = {
        "flow_spec": {
            "workflow_spec": {
                "stages": [
                    {"stage_type": "relax", "theory": "pbe", "modifiers": [], "options": {}},
                ]
            }
        },
        "submission": {"job_id": "21153721"},
        "paths": {"stage_dirs": {"stage_01": str(stage)}, "result_dir": str(stage)},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "submission.json").write_text(json.dumps(payload), encoding="utf-8")
    return stage


def lifecycle_summary(analysis):
    return build_lifecycle_concise_summary(analysis)


def flat(text: str) -> str:
    """Join wrapped lines so assertions do not depend on terminal wrapping."""

    return " ".join(text.split())


def rendered_lifecycle(analysis) -> str:
    return flat(render_concise_summary(lifecycle_summary(analysis)))


def rendered_job_flat(job) -> str:
    return flat(rendered_job(job))


def invocation_view(directory: Path, lookup) -> tuple[str, str, tuple[str, ...]]:
    analysis = analyze_calculation_directory(directory, scheduler_lookup=lookup)
    summary = lifecycle_summary(analysis)
    return (
        analysis.state.value,
        summary.status,
        tuple(stage.status for stage in summary.stages),
    )


# ---------------------------------------------------------------------------
# Pure completion rules
# ---------------------------------------------------------------------------


def _stage(
    index: int = 1,
    *,
    started: bool = True,
    terminated: bool | None = True,
    convergence: str = STAGE_CONVERGED,
) -> StageCompletion:
    return StageCompletion(
        index=index,
        label=f"stage {index}",
        started=started,
        terminated_normally=terminated,
        convergence=StageConvergence(convergence),
    )


def test_scheduler_success_never_substitutes_for_convergence_evidence() -> None:
    undetermined = (_stage(convergence=STAGE_CONVERGENCE_UNDETERMINED),)

    assert workflow_completion_status(undetermined, execution_succeeded=True) == "UNKNOWN"
    assert workflow_completion_status(undetermined) == "UNKNOWN"


def test_every_required_stage_needs_completion_evidence() -> None:
    stages = (_stage(1), _stage(2, started=False, terminated=False, convergence=STAGE_CONVERGENCE_UNDETERMINED))

    assert workflow_completion_status(stages, execution_succeeded=True) == "INCOMPLETE"
    # Without evidence that execution ended, a missing later stage is not a failure.
    assert workflow_completion_status(stages) == "UNKNOWN"


def test_missing_termination_marker_needs_scheduler_success_to_complete() -> None:
    stages = (_stage(terminated=None),)

    assert workflow_completion_status(stages) == "UNKNOWN"
    assert workflow_completion_status(stages, execution_succeeded=True) == "COMPLETED"


def test_trajectory_without_positive_evidence_is_not_converged_by_default() -> None:
    quiet = trajectory(stage_type="static", completed_ionic_steps=1)

    assert stage_convergence(quiet).status == STAGE_CONVERGENCE_UNDETERMINED
    assert stage_convergence(None).status == STAGE_CONVERGENCE_UNDETERMINED


def test_relaxation_needs_ionic_as_well_as_electronic_convergence() -> None:
    relax = trajectory(
        stage_type="relax",
        completed_ionic_steps=99,
        converged_electronic=True,
        converged_ionic=False,
    )

    result = stage_convergence(relax)

    assert result.status == STAGE_NOT_CONVERGED
    assert result.not_converged_scopes == ("ionic",)


# ---------------------------------------------------------------------------
# Path mode (lifecycle)
# ---------------------------------------------------------------------------


def test_unconverged_single_stage_relaxation_with_slurm_success_is_incomplete(tmp_path: Path) -> None:
    stage = write_single_relax_workflow(tmp_path / "flow")
    write_stage_relax_unconverged(stage)

    analysis = analyze_calculation_directory(stage, scheduler_lookup=slurm_success)
    output = rendered_lifecycle(analysis)

    assert analysis.state == LifecycleState.INCOMPLETE
    assert "Status: INCOMPLETE" in output
    assert "Status: COMPLETED" not in output
    assert "Status: FAILED" not in output
    assert FORBIDDEN_COMPLETION_CLAIM not in output
    assert "ionic convergence was not reached" in output
    assert "SLURM recorded a successful exit, but that alone does not establish" in output


def test_unconverged_manual_relaxation_is_not_completed_by_normal_termination(tmp_path: Path) -> None:
    write_stage_relax_unconverged(tmp_path)

    analysis = analyze_calculation_directory(tmp_path)
    output = rendered_lifecycle(analysis)

    assert analysis.normal_completion is True
    assert analysis.state == LifecycleState.INCOMPLETE
    assert "Status: INCOMPLETE" in output
    assert FORBIDDEN_COMPLETION_CLAIM not in output


@pytest.mark.parametrize("stage_2_state", ["absent", "partial"])
def test_partial_workflow_with_slurm_success_is_incomplete_from_every_directory(
    tmp_path: Path,
    stage_2_state: str,
) -> None:
    root = tmp_path / "flow"
    stage_1, stage_2 = write_two_stage_workflow(root)
    write_stage_complete(stage_1)
    if stage_2_state == "partial":
        write_stage_partial(stage_2)

    directories = [root, stage_1] + ([stage_2] if stage_2.exists() else [])
    views = {directory: invocation_view(directory, slurm_success) for directory in directories}

    assert len(set(views.values())) == 1, views
    state, status, stages = views[root]
    assert state == "INCOMPLETE"
    assert status == "INCOMPLETE"
    assert stages[0] == "COMPLETED"
    assert stages[1] != "COMPLETED"
    output = rendered_lifecycle(analyze_calculation_directory(stage_1, scheduler_lookup=slurm_success))
    assert "Status: COMPLETED" not in output
    assert FORBIDDEN_COMPLETION_CLAIM not in output


def test_partial_workflow_without_scheduler_evidence_stays_unknown_not_failed(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage_1, stage_2 = write_two_stage_workflow(root)
    write_stage_complete(stage_1)
    write_stage_partial(stage_2)

    views = {directory: invocation_view(directory, None) for directory in (root, stage_1, stage_2)}

    assert set(views.values()) == {views[root]}
    assert views[root][0] == "UNKNOWN"
    assert views[root][1] == "UNKNOWN"


def test_scheduler_success_with_normal_termination_but_no_convergence_evidence_is_unknown(
    tmp_path: Path,
) -> None:
    root = tmp_path / "flow"
    stage_1, stage_2 = write_two_stage_workflow(root)
    write_stage_terminated_unverified(stage_1)
    write_stage_terminated_unverified(stage_2)

    analysis = analyze_calculation_directory(root, scheduler_lookup=slurm_success)
    output = rendered_lifecycle(analysis)

    assert analysis.state == LifecycleState.UNKNOWN
    assert "Status: UNKNOWN" in output
    assert "Status: FAILED" not in output
    assert FORBIDDEN_COMPLETION_CLAIM not in output
    assert "VASP terminated normally, but convergence could not be established" in output


def test_scheduler_success_with_unconverged_first_stage_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage_1, stage_2 = write_two_stage_workflow(root)
    write_stage_relax_unconverged(stage_1)
    write_stage_complete(stage_2)

    views = {
        directory: invocation_view(directory, slurm_success)
        for directory in (root, stage_1, stage_2)
    }

    assert set(views.values()) == {views[root]}
    assert views[root][0] == "INCOMPLETE"
    assert views[root][2] == ("INCOMPLETE", "COMPLETED")


def test_scheduler_failure_is_still_reported_as_failed(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage_1, stage_2 = write_two_stage_workflow(root)
    write_stage_complete(stage_1)
    write_stage_partial(stage_2)

    summary = lifecycle_summary(
        analyze_calculation_directory(
            root,
            scheduler_lookup=lambda job_id: scheduler_record(
                job_id=job_id,
                state="TIMEOUT",
                exit_code="0:0",
            ),
        )
    )

    assert summary.status == "FAILED"


@pytest.mark.parametrize("with_scheduler", [True, False])
def test_genuinely_completed_workflow_is_completed_from_every_directory(
    tmp_path: Path,
    with_scheduler: bool,
) -> None:
    root = tmp_path / "flow"
    stage_1, stage_2 = write_two_stage_workflow(root)
    write_stage_complete(stage_1)
    write_stage_complete(stage_2)
    lookup = slurm_success if with_scheduler else None

    views = {directory: invocation_view(directory, lookup) for directory in (root, stage_1, stage_2)}

    assert set(views.values()) == {("COMPLETED", "COMPLETED", ("COMPLETED", "COMPLETED"))}
    output = rendered_lifecycle(analyze_calculation_directory(stage_1, scheduler_lookup=lookup))
    assert "VASP recorded normal termination in every workflow stage." in output
    assert "Convergence criteria were met in every workflow stage." in output
    assert FORBIDDEN_COMPLETION_CLAIM not in output


def test_normal_termination_alone_in_manual_static_is_unknown(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    (tmp_path / "OUTCAR").write_text(NORMAL_OUTCAR, encoding="utf-8")

    output = rendered_lifecycle(analyze_calculation_directory(tmp_path))

    assert "Status: UNKNOWN" in output
    assert "VASP terminated normally, but convergence could not be established" in output
    assert FORBIDDEN_COMPLETION_CLAIM not in output


# ---------------------------------------------------------------------------
# Job mode (presentation over acquired job evidence)
# ---------------------------------------------------------------------------


RELAX = WorkflowStage(1, "relax", "pbe", (), "stage_01")
STATIC_2 = WorkflowStage(2, "static", "hse06", (), "stage_02")


def test_job_unconverged_relaxation_with_slurm_success_is_incomplete() -> None:
    job = bmd_job(
        stages=(RELAX,),
        trajectories=(
            trajectory(
                stage_type="relax",
                completed_ionic_steps=99,
                converged_electronic=True,
                converged_ionic=False,
                criteria={"NSW": 99, "EDIFFG": -0.01},
            ),
        ),
    )

    output = rendered_job_flat(job)

    assert "Status: INCOMPLETE" in output
    assert "Status: COMPLETED" not in output
    assert FORBIDDEN_COMPLETION_CLAIM not in output
    assert "ionic convergence was not reached" in output
    assert "Ionic convergence was not reached." in output


def test_job_partial_workflow_with_slurm_success_is_incomplete() -> None:
    job = bmd_job(
        stages=(RELAX, STATIC_2),
        trajectories=(
            trajectory(
                stage_type="relax",
                completed_ionic_steps=10,
                converged_electronic=True,
                converged_ionic=True,
            ),
        ),
    )

    summary = build_job_concise_summary(job)
    output = flat(render_concise_summary(summary))

    assert summary.status == "INCOMPLETE"
    assert [stage.status for stage in summary.stages] == ["COMPLETED", "NOT STARTED"]
    assert "Stage 2 (HSE06 Static Energy) has no execution evidence." in output
    assert FORBIDDEN_COMPLETION_CLAIM not in output


def test_job_scheduler_success_with_incomplete_science_is_incomplete() -> None:
    job = bmd_job(
        stages=(WorkflowStage(1, "static", "pbe", (), "stage_01"),),
        trajectories=(
            trajectory(stage_type="static", completed_ionic_steps=1, converged_electronic=False),
        ),
        result=scientific(converged=False),
    )

    output = rendered_job_flat(job)

    assert "Status: INCOMPLETE" in output
    assert "electronic convergence was not reached" in output
    assert FORBIDDEN_COMPLETION_CLAIM not in output


def test_job_scheduler_success_without_convergence_evidence_is_unknown_not_failed() -> None:
    job = bmd_job(
        stages=(WorkflowStage(1, "static", "pbe", (), "stage_01"),),
        trajectories=(trajectory(stage_type="static", completed_ionic_steps=1),),
    )

    output = rendered_job_flat(job)

    assert "Status: UNKNOWN" in output
    assert "Status: FAILED" not in output
    assert "SLURM recorded a successful exit, but that alone does not establish" in output
    # The final result's vasprun is in another directory, so it is not borrowed.
    assert "Electronic convergence was reached." not in output
    assert FORBIDDEN_COMPLETION_CLAIM not in output


def test_job_uses_same_directory_final_vasprun_convergence() -> None:
    quiet = trajectory(stage_type="static", completed_ionic_steps=1)
    job = bmd_job(
        stages=(WorkflowStage(1, "static", "pbe", (), "stage_01"),),
        trajectories=(quiet,),
        result=ScientificResult(
            source_paths=(f"{quiet.directory}/vasprun.xml",),
            final_formula="Si",
            electronic_convergence=True,
        ),
    )

    assert build_job_concise_summary(job).status == "COMPLETED"


def test_job_scheduler_success_without_calculation_evidence_is_unknown() -> None:
    job = bmd_job()
    job = type(job)(
        job_id=job.job_id,
        scheduler=scheduler(),
        scheduler_error=None,
        scheduler_work_dir=None,
        calculation_directory=None,
        calculation_type="unknown",
        calculation_reason="scheduler WorkDir was unavailable",
    )

    output = rendered_job_flat(job)

    assert "Status: UNKNOWN" in output
    assert FORBIDDEN_COMPLETION_CLAIM not in output


@pytest.mark.parametrize("converged_ionic", [True, False])
def test_job_completed_with_missing_exit_code_is_not_scheduler_success(
    converged_ionic: bool,
) -> None:
    job = bmd_job(
        stages=(RELAX,),
        trajectories=(
            trajectory(
                stage_type="relax",
                completed_ionic_steps=10,
                converged_electronic=True,
                converged_ionic=converged_ionic,
            ),
        ),
        scheduler_record=scheduler(state="COMPLETED", exit_code=None),
    )

    summary = build_job_concise_summary(job)
    output = flat(render_concise_summary(summary))

    # A missing exit code is missing evidence: not success, and not failure.
    assert summary.status == "UNKNOWN"
    assert "no exit code was available, so successful execution is not established" in output
    assert "SLURM recorded a successful exit" not in output
    assert FORBIDDEN_COMPLETION_CLAIM not in output


def test_path_completed_with_missing_exit_code_is_not_scheduler_success(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage_1, stage_2 = write_two_stage_workflow(root)
    write_stage_complete(stage_1)

    def lookup(job_id: str) -> object:
        return scheduler_record(job_id=job_id, state="COMPLETED", exit_code=None)

    views = {directory: invocation_view(directory, lookup) for directory in (root, stage_1)}

    # Without positive scheduler success, a missing later stage is not INCOMPLETE.
    assert set(views.values()) == {views[root]}
    assert views[root][0] == "UNKNOWN"
    assert views[root][1] == "UNKNOWN"

    write_stage_complete(stage_2)
    # Local normal-termination plus convergence evidence still completes the workflow.
    assert invocation_view(root, lookup)[:2] == ("COMPLETED", "COMPLETED")


def test_job_nonzero_exit_is_failed_not_incomplete() -> None:
    job = bmd_job(scheduler_record=scheduler(state="COMPLETED", exit_code="1:0"))

    assert build_job_concise_summary(job).status == "FAILED"


def test_job_genuinely_completed_workflow_is_completed() -> None:
    job = bmd_job(
        stages=(RELAX, STATIC_2),
        trajectories=(
            trajectory(
                stage_type="relax",
                completed_ionic_steps=10,
                converged_electronic=True,
                converged_ionic=True,
            ),
            trajectory(stage_index=2, stage_type="static", theory="hse06", converged_electronic=True),
        ),
    )

    summary = build_job_concise_summary(job)
    output = flat(render_concise_summary(summary))

    assert summary.status == "COMPLETED"
    assert [stage.status for stage in summary.stages] == ["COMPLETED", "COMPLETED"]
    assert "SLURM recorded a successful exit." in output
    assert "Convergence criteria were met in every workflow stage." in output
    assert FORBIDDEN_COMPLETION_CLAIM not in output
