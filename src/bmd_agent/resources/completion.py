"""Workflow completion semantics shared by path and job diagnosis.

Completion is an evidence claim about every required stage, not a restatement
of process exit status. A successful scheduler exit (SLURM ``COMPLETED`` with
exit code ``0:0``) says only that the batch process ended successfully. It
never establishes, on its own, that VASP converged or that every declared
workflow stage ran.

The functions here are pure: they consume already acquired trajectory and
termination observations and never read files or contact remote resources.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import os

from bmd_agent.resources.run import (
    CONVERGED,
    JobInspection,
    ScientificResult,
    StageTrajectoryObservation,
    assess_convergence_progress,
)


STAGE_CONVERGED = "converged"
STAGE_NOT_CONVERGED = "not_converged"
STAGE_CONVERGENCE_UNDETERMINED = "undetermined"

WORKFLOW_COMPLETED = "COMPLETED"
WORKFLOW_INCOMPLETE = "INCOMPLETE"
WORKFLOW_UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class StageConvergence:
    """Positive, negative, or undetermined convergence evidence for one stage."""

    status: str
    not_converged_scopes: tuple[str, ...] = ()
    undetermined_scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageCompletion:
    """Completion evidence for one producer-declared (or direct VASP) stage.

    ``terminated_normally`` is ``True``/``False`` when VASP normal-termination
    evidence was inspected and ``None`` when that evidence is not acquired by
    the calling path (remote job inspection does not read the marker).
    """

    index: int
    label: str
    started: bool
    terminated_normally: bool | None
    convergence: StageConvergence

    @property
    def complete(self) -> bool:
        return (
            self.started
            and self.terminated_normally is not False
            and self.convergence.status == STAGE_CONVERGED
        )


def stage_convergence(
    trajectory: StageTrajectoryObservation | None,
    *,
    electronic_fallback: bool | None = None,
) -> StageConvergence:
    """Classify stage convergence from the existing conservative assessments.

    Convergence is established only by positive evidence (vasprun convergence
    flags, EDIFF reached in a completed electronic cycle, EDIFFG reached by
    the final force). Missing evidence and the absence of errors remain
    undetermined. Explicit counter-evidence that is not contradicted makes a
    scope not converged.

    ``electronic_fallback`` is a vasprun-derived electronic convergence flag
    for the same stage directory, used only when the trajectory's own vasprun
    enrichment did not supply one (for example because it was size-skipped).
    """

    if trajectory is None:
        return StageConvergence(STAGE_CONVERGENCE_UNDETERMINED, undetermined_scopes=("electronic",))
    if trajectory.converged_electronic is None and electronic_fallback is not None:
        trajectory = replace(trajectory, converged_electronic=electronic_fallback)

    not_converged: list[str] = []
    undetermined: list[str] = []
    for assessment in assess_convergence_progress((trajectory,)):
        if assessment.scope == "stage":
            continue
        if assessment.label == CONVERGED:
            continue
        if assessment.counter_evidence and assessment.sufficiency != "contradictory":
            not_converged.append(assessment.scope)
        else:
            undetermined.append(assessment.scope)

    if not_converged:
        status = STAGE_NOT_CONVERGED
    elif undetermined:
        status = STAGE_CONVERGENCE_UNDETERMINED
    else:
        status = STAGE_CONVERGED
    return StageConvergence(
        status,
        not_converged_scopes=tuple(not_converged),
        undetermined_scopes=tuple(undetermined),
    )


def workflow_completion_status(
    stages: Sequence[StageCompletion],
    *,
    execution_succeeded: bool = False,
) -> str:
    """Return COMPLETED, INCOMPLETE, or UNKNOWN for all required stages.

    ``execution_succeeded`` is true only when the scheduler reports that the
    batch process ended successfully. It allows two things: missing or
    unconverged stages become INCOMPLETE (execution will not continue), and
    stages whose normal-termination marker was not acquired may still
    complete. It never substitutes for convergence evidence.
    """

    if not stages:
        return WORKFLOW_UNKNOWN

    if all(
        stage.complete
        and (stage.terminated_normally is True or execution_succeeded)
        for stage in stages
    ):
        return WORKFLOW_COMPLETED

    for stage in stages:
        # A VASP run that terminated normally without meeting its criteria
        # will not converge by itself; the recorded run is incomplete.
        if stage.terminated_normally is True and stage.convergence.status == STAGE_NOT_CONVERGED:
            return WORKFLOW_INCOMPLETE

    if execution_succeeded and any(
        not stage.started
        or stage.terminated_normally is False
        or stage.convergence.status == STAGE_NOT_CONVERGED
        for stage in stages
    ):
        return WORKFLOW_INCOMPLETE

    return WORKFLOW_UNKNOWN


def job_stage_completion(inspection: JobInspection) -> tuple[StageCompletion, ...]:
    """Normalize already acquired job evidence into shared stage completion records."""

    scientific = _job_scientific(inspection)
    if inspection.bmd_compute is not None:
        trajectories = {
            item.stage_index: item for item in inspection.bmd_compute.trajectories
        }
        stages: list[StageCompletion] = []
        for stage in inspection.bmd_compute.inspection.workflow_stages:
            trajectory = trajectories.get(stage.index)
            started = trajectory is not None and trajectory_has_observations(trajectory)
            stages.append(
                StageCompletion(
                    index=stage.index,
                    label=_job_stage_label(stage),
                    started=started,
                    terminated_normally=None,
                    convergence=stage_convergence(
                        trajectory if started else None,
                        electronic_fallback=same_directory_electronic_convergence(
                            trajectory,
                            scientific,
                        ),
                    ),
                )
            )
        return tuple(stages)
    if inspection.direct_vasp is not None:
        trajectory = inspection.direct_vasp.trajectory
        started = trajectory_has_observations(trajectory)
        return (
            StageCompletion(
                index=1,
                label="Direct VASP calculation",
                started=started,
                terminated_normally=None,
                convergence=stage_convergence(
                    trajectory if started else None,
                    electronic_fallback=same_directory_electronic_convergence(
                        trajectory,
                        scientific,
                    ),
                ),
            ),
        )
    return ()


def trajectory_has_observations(trajectory: StageTrajectoryObservation) -> bool:
    """Return whether a trajectory contains evidence that its stage started."""

    return bool(
        trajectory.oszicar_present
        or trajectory.vasprun_present
        or trajectory.outcar_present
        or trajectory.completed_ionic_steps is not None
        or trajectory.incomplete_electronic_iteration_count is not None
        or trajectory.electronic_cycles
        or trajectory.ionic_steps
    )


def _job_scientific(inspection: JobInspection) -> ScientificResult | None:
    if inspection.bmd_compute is not None:
        return inspection.bmd_compute.inspection.scientific
    if inspection.direct_vasp is not None:
        return inspection.direct_vasp.scientific
    return None


def same_directory_electronic_convergence(
    trajectory: StageTrajectoryObservation | None,
    scientific: ScientificResult | None,
) -> bool | None:
    if trajectory is None or scientific is None or scientific.electronic_convergence is None:
        return None
    directory = os.path.normpath(trajectory.directory)
    for path in scientific.source_paths:
        text = str(path)
        if (
            os.path.basename(text) == "vasprun.xml"
            and os.path.normpath(os.path.dirname(text)) == directory
        ):
            return scientific.electronic_convergence
    return None


def _job_stage_label(stage: object) -> str:
    theory = str(getattr(stage, "theory", "") or "unknown").upper()
    stage_type = str(getattr(stage, "stage_type", "") or "calculation")
    stage_labels = {
        "band_structure": "Band Structure",
        "dos": "DOS",
        "relax": "Geometry Optimisation",
        "static": "Static Energy",
    }
    text = f"{theory} {stage_labels.get(stage_type, stage_type.replace('_', ' ').title())}"
    modifiers = tuple(getattr(stage, "modifiers", ()) or ())
    if modifiers:
        modifier_labels = {
            "soc": "SOC",
            "dft_u": "DFT+U",
        }
        text += " + " + " + ".join(
            modifier_labels.get(str(item).lower(), str(item).replace("_", " ").upper())
            for item in modifiers
        )
    return text


def completion_gap_lines(
    stages: Sequence[StageCompletion],
    *,
    execution_succeeded: bool = False,
) -> tuple[str, ...]:
    """Describe, in plain language, which stages lack completion evidence."""

    lines: list[str] = []
    if execution_succeeded:
        lines.append(
            "SLURM recorded a successful exit, but that alone does not establish "
            "that the calculation reached its goal."
        )
    multi = len(stages) > 1
    for stage in stages:
        if stage.complete and (stage.terminated_normally is True or execution_succeeded):
            continue
        name = f"Stage {stage.index} ({stage.label})" if multi else "The calculation"
        if not stage.started:
            lines.append(f"{name} has no execution evidence.")
            continue
        convergence = stage.convergence
        if convergence.status == STAGE_NOT_CONVERGED:
            scopes = " and ".join(convergence.not_converged_scopes)
            lines.append(f"{name}: {scopes} convergence was not reached.")
        elif convergence.status == STAGE_CONVERGENCE_UNDETERMINED:
            if stage.terminated_normally is True:
                lines.append(
                    f"{name}: VASP terminated normally, but convergence could not be "
                    "established from the available evidence."
                )
            else:
                lines.append(
                    f"{name}: convergence could not be established from the available evidence."
                )
        if stage.terminated_normally is False and convergence.status != STAGE_NOT_CONVERGED:
            lines.append(f"{name}: VASP normal-termination evidence was not found.")
    return tuple(dict.fromkeys(lines))
