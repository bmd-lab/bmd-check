"""Versioned scientific-completion assessment over already acquired evidence.

The verdict function in this module is pure. Local-path and scheduler-job
callers first normalize their observations into ``CompletionAssessmentInput``;
neither rendering path reads calculation artifacts or recomputes convergence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from bmd_agent.resources.completion import (
    STAGE_CONVERGED,
    STAGE_NOT_CONVERGED,
    WORKFLOW_COMPLETED,
    WORKFLOW_INCOMPLETE,
    StageCompletion,
    job_stage_completion,
    workflow_completion_status,
)
from bmd_agent.resources.compute_records import ComputeRecordError, parse_submission_record
from bmd_agent.resources.lifecycle import LifecycleAnalysis
from bmd_agent.resources.run import JobInspection, ScientificResult, StageTrajectoryObservation


SCHEMA_ID = "bmd_check.scientific_completion_assessment"
SCHEMA_VERSION = 1
ASSESSMENT_ID = "scientific_completion"
METHOD_VERSION = "1.0.0-c1"

VERDICT_COMPLETED = "completed"
VERDICT_INCOMPLETE = "incomplete"
VERDICT_UNKNOWN = "unknown"

STATUS_ASSESSED = "assessed"
STATUS_UNSUPPORTED = "unsupported"
STATUS_INSUFFICIENT_EVIDENCE = "insufficient_evidence"

PROCESS_SUCCEEDED = "succeeded"
PROCESS_FAILED = "failed"
PROCESS_UNAVAILABLE = "unavailable"

OUTPUT_SATISFIED = "satisfied"
OUTPUT_MISSING = "missing"
OUTPUT_UNUSABLE = "unusable"
OUTPUT_UNAVAILABLE = "unavailable"
OUTPUT_UNSUPPORTED = "unsupported"

_SUPPORTED_STAGE_TYPES = frozenset({"static", "relax", "dos", "band_structure"})
_PHONON_STAGE_TYPES = frozenset({"phonon", "phonons"})


@dataclass(frozen=True, slots=True)
class AssessmentStage:
    index: int
    label: str
    stage_type: str
    theory: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    name: str
    present: bool | None
    non_empty: bool | None


@dataclass(frozen=True, slots=True)
class ScientificOutputSummary:
    final_energy_ev: float | None = None
    final_energy_reliable: bool = False
    final_energy_source: str | None = None
    final_structure_usable: bool = False
    dos_available: bool | None = None
    dos_reliable: bool = False
    band_gap_ev: float | None = None
    band_kpoints: int | None = None
    bands: int | None = None
    band_structure_reliable: bool = False
    error: str | None = None
    unavailable: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProducerProvenance:
    status: str
    producer: str | None = None
    repository: str | None = None
    git_commit: str | None = None
    git_state: str | None = None


@dataclass(frozen=True, slots=True)
class CompletionAssessmentInput:
    subject_id: str
    calculation_type: str
    scope: str
    stages: tuple[AssessmentStage, ...]
    stage_completion: tuple[StageCompletion, ...]
    process_outcome: str
    artifacts: tuple[ArtifactEvidence, ...]
    scientific: ScientificOutputSummary | None
    producer_provenance: ProducerProvenance
    limitations: tuple[str, ...] = ()
    assessor_version: str = "unknown"


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    evidence_id: str
    evidence_type: str
    source: str
    stage_index: int | None = None


@dataclass(frozen=True, slots=True)
class AssessmentFinding:
    reason_code: str
    status: str
    message: str
    stage_index: int | None = None
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScientificCompletionAssessment:
    subject_id: str
    calculation_type: str
    scope: str
    process_outcome: str
    stages: tuple[AssessmentStage, ...]
    assessment_status: str
    verdict: str
    findings: tuple[AssessmentFinding, ...]
    evidence_references: tuple[EvidenceReference, ...]
    producer_provenance: ProducerProvenance
    assessor_version: str
    limitations: tuple[str, ...] = ()
    schema_id: str = SCHEMA_ID
    schema_version: int = SCHEMA_VERSION
    assessment_id: str = ASSESSMENT_ID
    method_version: str = METHOD_VERSION


def assess_completion(evidence: CompletionAssessmentInput) -> ScientificCompletionAssessment:
    """Produce the sole C1 scientific-completion verdict implementation."""

    findings: list[AssessmentFinding] = []
    references: list[EvidenceReference] = []
    limitations = list(evidence.limitations)
    if len(evidence.stages) > 1:
        limitations.append(
            "C1 validates output usability for the final requested stage; earlier stages "
            "are covered by declared-stage process and convergence evidence."
        )
    completion_by_index = {stage.index: stage for stage in evidence.stage_completion}
    stage_model_findings, stage_model_valid = _stage_model_findings(
        evidence.stages,
        evidence.stage_completion,
    )
    findings.extend(stage_model_findings)

    for stage in evidence.stages:
        completion = completion_by_index.get(stage.index)
        process_ref = f"stage-{stage.index}-process"
        convergence_ref = f"stage-{stage.index}-convergence"
        references.extend(
            (
                EvidenceReference(
                    process_ref,
                    "termination_observation",
                    "VASP normal-termination evidence or successful scheduler process exit",
                    stage.index,
                ),
                EvidenceReference(
                    convergence_ref,
                    "trajectory_observation",
                    "existing electronic/ionic convergence assessment",
                    stage.index,
                ),
            )
        )
        findings.extend(
            _stage_findings(
                stage,
                completion,
                process_ref,
                convergence_ref,
                execution_succeeded=evidence.process_outcome == PROCESS_SUCCEEDED,
            )
        )

    workflow_status = (
        workflow_completion_status(
            evidence.stage_completion,
            execution_succeeded=evidence.process_outcome == PROCESS_SUCCEEDED,
        )
        if stage_model_valid
        else None
    )
    output_finding, output_reference, output_limitation = _final_output_finding(evidence)
    findings.append(output_finding)
    references.append(output_reference)
    if output_limitation:
        limitations.append(output_limitation)

    unsupported = output_finding.status == OUTPUT_UNSUPPORTED
    phonon_unsupported = output_finding.reason_code == "PHONON_COMPLETION_UNSUPPORTED_C1"
    if phonon_unsupported:
        verdict = VERDICT_UNKNOWN
        assessment_status = STATUS_UNSUPPORTED
        reason_code = "SCIENTIFIC_COMPLETION_UNSUPPORTED"
        message = "C1 does not support completion validation for the final requested stage."
    elif evidence.process_outcome == PROCESS_FAILED:
        verdict = VERDICT_INCOMPLETE
        assessment_status = STATUS_ASSESSED
        reason_code = "PROCESS_TERMINATION_FAILED"
        message = "The calculation process is confirmed to have ended unsuccessfully."
    elif unsupported:
        verdict = VERDICT_UNKNOWN
        assessment_status = STATUS_UNSUPPORTED
        reason_code = "SCIENTIFIC_COMPLETION_UNSUPPORTED"
        message = "C1 does not support completion validation for the final requested stage."
    elif not evidence.stages or not evidence.stage_completion:
        verdict = VERDICT_UNKNOWN
        assessment_status = STATUS_INSUFFICIENT_EVIDENCE
        reason_code = "COMPLETION_EVIDENCE_INSUFFICIENT"
        message = "No complete stage model was available for scientific completion assessment."
    elif not stage_model_valid:
        verdict = VERDICT_UNKNOWN
        assessment_status = STATUS_INSUFFICIENT_EVIDENCE
        reason_code = "STAGE_COMPLETION_MODEL_INCONSISTENT"
        message = "Completion evidence does not cover the declared workflow stages exactly."
    elif workflow_status == WORKFLOW_INCOMPLETE:
        verdict = VERDICT_INCOMPLETE
        assessment_status = STATUS_ASSESSED
        reason_code = "SCIENTIFIC_COMPLETION_NOT_REACHED"
        message = "At least one required workflow stage has explicit incomplete evidence."
    elif workflow_status != WORKFLOW_COMPLETED:
        verdict = VERDICT_UNKNOWN
        assessment_status = STATUS_INSUFFICIENT_EVIDENCE
        reason_code = "SCIENTIFIC_COMPLETION_UNDETERMINED"
        message = "Required process or convergence evidence is unavailable or contradictory."
    elif output_finding.status == OUTPUT_SATISFIED:
        verdict = VERDICT_COMPLETED
        assessment_status = STATUS_ASSESSED
        reason_code = "SCIENTIFIC_COMPLETION_ESTABLISHED"
        message = "Every required stage completed and the final required scientific output is usable."
    elif output_finding.status in {OUTPUT_MISSING, OUTPUT_UNUSABLE}:
        verdict = VERDICT_INCOMPLETE
        assessment_status = STATUS_ASSESSED
        reason_code = "REQUIRED_SCIENTIFIC_OUTPUT_NOT_USABLE"
        message = "Execution completed, but a required scientific output is missing or unusable."
    else:
        verdict = VERDICT_UNKNOWN
        assessment_status = STATUS_INSUFFICIENT_EVIDENCE
        reason_code = "REQUIRED_SCIENTIFIC_OUTPUT_UNDETERMINED"
        message = "The required scientific output could not be established from acquired evidence."

    findings.insert(
        0,
        AssessmentFinding(
            reason_code,
            verdict,
            message,
            evidence_refs=tuple(reference.evidence_id for reference in references),
        ),
    )
    return ScientificCompletionAssessment(
        subject_id=evidence.subject_id,
        calculation_type=evidence.calculation_type,
        scope=evidence.scope,
        process_outcome=evidence.process_outcome,
        stages=evidence.stages,
        assessment_status=assessment_status,
        verdict=verdict,
        findings=tuple(findings),
        evidence_references=tuple(references),
        producer_provenance=evidence.producer_provenance,
        assessor_version=evidence.assessor_version,
        limitations=tuple(dict.fromkeys(limitations)),
    )


def completion_input_from_lifecycle(analysis: LifecycleAnalysis) -> CompletionAssessmentInput:
    """Normalize local lifecycle observations without reading any new artifacts."""

    workflow = analysis.bmd_workflow
    trajectories = _lifecycle_trajectories(analysis)
    if workflow is not None:
        stages = tuple(
            AssessmentStage(
                index=index,
                label=_stage_label(index, item),
                stage_type=_stage_type(item.get("stage_type")),
                theory=_optional_text(item.get("theory")),
            )
            for index, item in enumerate(workflow.workflow_stages, start=1)
        )
        subject_id = _submission_subject_id(workflow.submission, workflow.workflow_root.name)
        provenance = _local_producer_provenance(workflow.submission)
        scope = "bmd_compute_workflow"
    else:
        trajectory = trajectories[-1] if trajectories else None
        stages = (
            AssessmentStage(
                index=1,
                label="Direct VASP calculation",
                stage_type=_direct_stage_type(trajectory, analysis.incar_settings),
            ),
        ) if analysis.calculation_kind != "none" else ()
        subject_id = analysis.directory.name or "calculation"
        provenance = ProducerProvenance(status="unavailable")
        scope = "direct_vasp_calculation"

    artifacts = tuple(
        ArtifactEvidence(
            name=name.lower(),
            present=item.present,
            non_empty=(
                None
                if item.present and item.size is None
                else item.non_empty if item.present else False
            ),
        )
        for name, item in {**analysis.input_files, **analysis.output_files}.items()
    )
    limitations: list[str] = []
    if workflow is not None:
        limitations.extend(workflow.record_limitations)
        if workflow.relocated:
            limitations.append(
                "scheduler evidence from the original producer location was not used for this relocated snapshot"
            )
    if analysis.scheduler_error:
        limitations.append("scheduler observation was unavailable")
    return CompletionAssessmentInput(
        subject_id=subject_id,
        calculation_type=analysis.calculation_kind,
        scope=scope,
        stages=stages,
        stage_completion=analysis.stage_completion,
        process_outcome=_lifecycle_process_outcome(analysis),
        artifacts=artifacts,
        scientific=_scientific_summary(
            analysis.scientific,
            trajectories[-1] if trajectories else None,
        ),
        producer_provenance=provenance,
        limitations=tuple(limitations),
        assessor_version=_package_version(),
    )


def completion_input_from_job(inspection: JobInspection) -> CompletionAssessmentInput:
    """Normalize acquired scheduler-job observations without remote reads or parsing."""

    if inspection.bmd_compute is not None:
        run = inspection.bmd_compute.inspection
        stages = tuple(
            AssessmentStage(
                stage.index,
                stage.label or f"{stage.theory.upper()} {stage.stage_type}",
                _stage_type(stage.stage_type),
                stage.theory or None,
            )
            for stage in run.workflow_stages
        )
        scientific = run.scientific
        artifacts = run.final_artifacts
        provenance = _job_producer_provenance(run.producer_git)
        scope = "bmd_compute_workflow"
    elif inspection.direct_vasp is not None:
        direct = inspection.direct_vasp
        stages = (
            AssessmentStage(
                1,
                "Direct VASP calculation",
                _direct_stage_type(direct.trajectory),
            ),
        )
        scientific = direct.scientific
        artifacts = direct.artifacts
        provenance = ProducerProvenance(status="unavailable")
        scope = "direct_vasp_calculation"
    else:
        stages = ()
        scientific = None
        artifacts = ()
        provenance = ProducerProvenance(status="unavailable")
        scope = "unknown_calculation"

    limitations = tuple(
        message
        for condition, message in (
            (inspection.scheduler_error is not None, "scheduler observation was unavailable"),
            (
                inspection.calculation_reason is not None,
                "calculation evidence could not be fully resolved",
            ),
        )
        if condition
    )
    return CompletionAssessmentInput(
        subject_id=f"slurm-job-{inspection.job_id}",
        calculation_type=inspection.calculation_type,
        scope=scope,
        stages=stages,
        stage_completion=job_stage_completion(inspection),
        process_outcome=_scheduler_process_outcome(inspection.scheduler),
        artifacts=tuple(
            ArtifactEvidence(item.label.lower(), item.present, None)
            for item in artifacts
        ),
        scientific=_scientific_summary(
            scientific,
            (
                inspection.bmd_compute.trajectories[-1]
                if inspection.bmd_compute is not None and inspection.bmd_compute.trajectories
                else inspection.direct_vasp.trajectory
                if inspection.direct_vasp is not None
                else None
            ),
        ),
        producer_provenance=provenance,
        limitations=limitations,
        assessor_version=_package_version(),
    )


def build_lifecycle_completion_assessment(
    analysis: LifecycleAnalysis,
) -> ScientificCompletionAssessment:
    return assess_completion(completion_input_from_lifecycle(analysis))


def build_job_completion_assessment(
    inspection: JobInspection,
) -> ScientificCompletionAssessment:
    return assess_completion(completion_input_from_job(inspection))


def serialize_completion_assessment(
    assessment: ScientificCompletionAssessment,
) -> dict[str, Any]:
    """Serialize the immutable C1 result without acquiring or reinterpreting evidence."""

    return {
        "schema_id": assessment.schema_id,
        "schema_version": assessment.schema_version,
        "assessment_id": assessment.assessment_id,
        "method_version": assessment.method_version,
        "assessment_status": assessment.assessment_status,
        "verdict": assessment.verdict,
        "process_outcome": assessment.process_outcome,
        "subject": {
            "id": assessment.subject_id,
            "calculation_type": assessment.calculation_type,
            "scope": assessment.scope,
            "stages": [
                {
                    "index": stage.index,
                    "label": stage.label,
                    "stage_type": stage.stage_type,
                    "theory": stage.theory,
                }
                for stage in assessment.stages
            ],
        },
        "findings": [
            {
                "reason_code": finding.reason_code,
                "status": finding.status,
                "message": finding.message,
                "stage_index": finding.stage_index,
                "evidence_refs": list(finding.evidence_refs),
            }
            for finding in assessment.findings
        ],
        "evidence_references": [
            {
                "id": reference.evidence_id,
                "evidence_type": reference.evidence_type,
                "source": reference.source,
                "stage_index": reference.stage_index,
            }
            for reference in assessment.evidence_references
        ],
        "producer_provenance": {
            "status": assessment.producer_provenance.status,
            "producer": assessment.producer_provenance.producer,
            "repository": assessment.producer_provenance.repository,
            "git_commit": assessment.producer_provenance.git_commit,
            "git_state": assessment.producer_provenance.git_state,
        },
        "assessor_provenance": {
            "name": "BMD Check",
            "package": "bmd-agent",
            "version": assessment.assessor_version,
        },
        "limitations": list(assessment.limitations),
    }


def render_completion_assessment(assessment: ScientificCompletionAssessment) -> str:
    """Render the same immutable result used by the JSON serializer."""

    lines = [
        "Scientific Completion Assessment",
        "================================",
        f"Subject: {assessment.subject_id}",
        f"Scope: {assessment.scope}",
        f"Method: {assessment.assessment_id} {assessment.method_version}",
        f"Assessment status: {assessment.assessment_status}",
        f"Process outcome: {assessment.process_outcome}",
        f"Verdict: {assessment.verdict.upper()}",
        "",
        "Findings:",
    ]
    lines.extend(
        f"  [{finding.status}] {finding.reason_code}: {finding.message}"
        for finding in assessment.findings
    )
    provenance = assessment.producer_provenance
    if provenance.status == "available":
        details = provenance.git_commit or "commit unavailable"
        if provenance.git_state:
            details += f" ({provenance.git_state})"
        lines.extend(("", f"Producer provenance: {details}"))
    if assessment.limitations:
        lines.extend(("", "Limitations:"))
        lines.extend(f"  - {item}" for item in assessment.limitations)
    return "\n".join(lines) + "\n"


def _stage_model_findings(
    stages: tuple[AssessmentStage, ...],
    completion: tuple[StageCompletion, ...],
) -> tuple[tuple[AssessmentFinding, ...], bool]:
    declared = [stage.index for stage in stages]
    observed = [stage.index for stage in completion]
    declared_counts = Counter(declared)
    observed_counts = Counter(observed)
    findings: list[AssessmentFinding] = []

    for index in sorted(item for item, count in declared_counts.items() if count > 1):
        findings.append(
            AssessmentFinding(
                "DECLARED_STAGE_INDEX_DUPLICATE",
                "unavailable",
                f"Declared workflow stage index {index} occurs more than once.",
                index,
            )
        )
    for index in sorted(item for item, count in observed_counts.items() if count > 1):
        findings.append(
            AssessmentFinding(
                "STAGE_COMPLETION_INDEX_DUPLICATE",
                "unavailable",
                f"Completion evidence for stage index {index} occurs more than once.",
                index,
            )
        )
    for index in sorted(set(observed) - set(declared)):
        findings.append(
            AssessmentFinding(
                "STAGE_COMPLETION_INDEX_UNEXPECTED",
                "unavailable",
                f"Completion evidence refers to undeclared stage index {index}.",
                index,
            )
        )

    valid = (
        len(declared) == len(set(declared))
        and len(observed) == len(set(observed))
        and set(declared) == set(observed)
    )
    return tuple(findings), valid


def _stage_findings(
    stage: AssessmentStage,
    completion: StageCompletion | None,
    process_ref: str,
    convergence_ref: str,
    *,
    execution_succeeded: bool,
) -> tuple[AssessmentFinding, ...]:
    if completion is None or not completion.started:
        return (
            AssessmentFinding(
                "STAGE_EXECUTION_UNAVAILABLE",
                "unavailable",
                f"No execution evidence was acquired for stage {stage.index} ({stage.label}).",
                stage.index,
                (process_ref,),
            ),
            AssessmentFinding(
                "STAGE_CONVERGENCE_UNAVAILABLE",
                "unavailable",
                f"Convergence could not be assessed for stage {stage.index} ({stage.label}).",
                stage.index,
                (convergence_ref,),
            ),
        )

    if completion.terminated_normally is True:
        process_status = "supported"
        process_code = "PROCESS_TERMINATION_ESTABLISHED"
        process_message = "VASP normal termination was observed."
    elif completion.terminated_normally is False:
        process_status = "unavailable"
        process_code = "PROCESS_TERMINATION_NOT_ESTABLISHED"
        process_message = "VASP normal-termination evidence was not found."
    elif execution_succeeded:
        process_status = "supported"
        process_code = "PROCESS_TERMINATION_ESTABLISHED"
        process_message = "The scheduler recorded a successful process exit."
    else:
        process_status = "unavailable"
        process_code = "PROCESS_TERMINATION_UNAVAILABLE"
        process_message = "VASP normal-termination evidence was not acquired by this path."

    convergence = completion.convergence
    if convergence.status == STAGE_CONVERGED:
        convergence_status = "supported"
        convergence_code = "STAGE_CONVERGENCE_ESTABLISHED"
        convergence_message = "Required electronic and ionic convergence criteria were reached."
    elif convergence.status == STAGE_NOT_CONVERGED:
        convergence_status = "not_reached"
        convergence_code = "STAGE_CONVERGENCE_NOT_REACHED"
        scopes = ", ".join(convergence.not_converged_scopes) or "required"
        convergence_message = f"{scopes.capitalize()} convergence was not reached."
    else:
        convergence_status = "unavailable"
        convergence_code = "STAGE_CONVERGENCE_UNDETERMINED"
        scopes = ", ".join(convergence.undetermined_scopes) or "required"
        convergence_message = f"{scopes.capitalize()} convergence could not be established."
    return (
        AssessmentFinding(
            process_code,
            process_status,
            process_message,
            stage.index,
            (process_ref,),
        ),
        AssessmentFinding(
            convergence_code,
            convergence_status,
            convergence_message,
            stage.index,
            (convergence_ref,),
        ),
    )


def _final_output_finding(
    evidence: CompletionAssessmentInput,
) -> tuple[AssessmentFinding, EvidenceReference, str | None]:
    reference = EvidenceReference(
        "final-scientific-output",
        "artifact_derived_observation",
        "bounded final-stage VASP artifacts and derived scientific observations",
        evidence.stages[-1].index if evidence.stages else None,
    )
    if not evidence.stages:
        return (
            AssessmentFinding(
                "FINAL_OUTPUT_REQUIREMENT_UNAVAILABLE",
                OUTPUT_UNAVAILABLE,
                "No final stage was available to define a scientific output requirement.",
                evidence_refs=(reference.evidence_id,),
            ),
            reference,
            None,
        )

    stage = evidence.stages[-1]
    stage_type = stage.stage_type
    if stage_type in _PHONON_STAGE_TYPES:
        return (
            AssessmentFinding(
                "PHONON_COMPLETION_UNSUPPORTED_C1",
                OUTPUT_UNSUPPORTED,
                "C1 does not discover or validate nested phonon tasks.",
                stage.index,
                (reference.evidence_id,),
            ),
            reference,
            "Phonon completion validation is unsupported in C1; imaginary modes are not treated as execution failures.",
        )
    if stage_type not in _SUPPORTED_STAGE_TYPES:
        return (
            AssessmentFinding(
                "STAGE_OUTPUT_REQUIREMENT_UNSUPPORTED",
                OUTPUT_UNSUPPORTED,
                f"C1 has no output-usability contract for stage type {stage_type!r}.",
                stage.index,
                (reference.evidence_id,),
            ),
            reference,
            None,
        )

    scientific = evidence.scientific
    artifacts = {item.name: item for item in evidence.artifacts}
    if stage_type == "static":
        status, message = _require_value(
            scientific is not None
            and scientific.final_energy_ev is not None
            and scientific.final_energy_reliable,
            artifacts.get("vasprun.xml") or artifacts.get("vasprun"),
            scientific,
            "final energy",
        )
        limitation = None
    elif stage_type == "relax":
        status, message = _require_value(
            scientific is not None and scientific.final_structure_usable,
            artifacts.get("contcar"),
            scientific,
            "parseable final structure",
        )
        limitation = None
    elif stage_type == "dos":
        doscar = artifacts.get("doscar")
        if (
            scientific is not None
            and scientific.dos_available is True
            and scientific.dos_reliable
        ):
            status = OUTPUT_SATISFIED
            message = "pymatgen recovered a complete DOS from the final vasprun.xml."
        elif scientific is not None and scientific.dos_available is True:
            status = OUTPUT_UNUSABLE
            message = "DOS values were recovered only from incomplete parser evidence."
        elif scientific is not None and scientific.dos_available is False:
            status = OUTPUT_UNUSABLE
            message = "The final vasprun.xml did not provide a complete DOS."
        elif scientific is not None and (scientific.error or scientific.unavailable):
            status = OUTPUT_UNUSABLE
            message = "The final-stage artifacts did not provide a reliably parsed complete DOS."
        elif doscar is not None and (doscar.present is False or doscar.non_empty is False):
            status = OUTPUT_MISSING
            message = "The required DOSCAR is missing or empty."
        else:
            status = OUTPUT_UNAVAILABLE
            message = "A reliably parsed complete DOS could not be established."
        limitation = "C1 requires reliable complete_dos evidence and does not infer DOS usability from DOSCAR presence."
    else:
        usable = bool(
            scientific is not None
            and scientific.band_structure_reliable
            and scientific.band_gap_ev is not None
            and scientific.band_kpoints is not None
            and scientific.band_kpoints > 0
            and scientific.bands is not None
            and scientific.bands > 0
        )
        status, message = _require_value(
            usable,
            artifacts.get("vasprun.xml") or artifacts.get("vasprun"),
            scientific,
            "parseable electronic band structure with gap, k-points, and bands",
        )
        limitation = None

    code = {
        OUTPUT_SATISFIED: "FINAL_OUTPUT_REQUIREMENT_SATISFIED",
        OUTPUT_MISSING: "FINAL_OUTPUT_REQUIRED_ARTIFACT_MISSING",
        OUTPUT_UNUSABLE: "FINAL_OUTPUT_UNUSABLE",
        OUTPUT_UNAVAILABLE: "FINAL_OUTPUT_REQUIREMENT_UNAVAILABLE",
    }[status]
    return (
        AssessmentFinding(code, status, message, stage.index, (reference.evidence_id,)),
        reference,
        limitation,
    )


def _require_value(
    available: bool,
    artifact: ArtifactEvidence | None,
    scientific: ScientificOutputSummary | None,
    label: str,
) -> tuple[str, str]:
    if available:
        return OUTPUT_SATISFIED, f"A {label} was derived from the final-stage artifacts."
    if artifact is not None and (artifact.present is False or artifact.non_empty is False):
        return OUTPUT_MISSING, f"The artifact required for {label} is missing or empty."
    if scientific is not None and (scientific.error or scientific.unavailable):
        return OUTPUT_UNUSABLE, f"The final-stage artifacts did not provide a usable {label}."
    return OUTPUT_UNAVAILABLE, f"A {label} could not be established from acquired evidence."


def _scientific_summary(
    result: ScientificResult | None,
    trajectory: StageTrajectoryObservation | None,
) -> ScientificOutputSummary | None:
    fallback_energy = None
    if trajectory is not None and trajectory.ionic_steps:
        final_step = trajectory.ionic_steps[-1]
        fallback_energy = (
            final_step.free_energy
            if final_step.free_energy is not None
            else final_step.energy_zero
        )
    if result is None:
        return (
            ScientificOutputSummary(
                final_energy_ev=fallback_energy,
                final_energy_reliable=fallback_energy is not None,
                final_energy_source="oszicar" if fallback_energy is not None else None,
            )
            if fallback_energy is not None
            else None
        )
    vasprun_reliable = _vasprun_output_reliable(result)
    result_sources = {
        _source_filename(path)
        for path in result.source_paths
    }
    if result.final_energy_ev is not None and vasprun_reliable:
        final_energy = result.final_energy_ev
        final_energy_reliable = True
        final_energy_source = "vasprun"
    elif fallback_energy is not None:
        final_energy = fallback_energy
        final_energy_reliable = True
        final_energy_source = "oszicar"
    elif result.final_energy_ev is not None and "oszicar" in result_sources:
        final_energy = result.final_energy_ev
        final_energy_reliable = True
        final_energy_source = "oszicar"
    else:
        final_energy = result.final_energy_ev
        final_energy_reliable = False
        final_energy_source = "partial_vasprun" if final_energy is not None else None
    structure = result.structure
    structure_source = _source_filename(
        structure.source_path if structure is not None else None
    )
    structure_reliable = bool(
        structure is not None
        and structure.site_count
        and (structure_source == "contcar" or vasprun_reliable)
    )
    band_values_available = bool(
        result.band_gap_ev is not None
        and result.band_kpoints is not None
        and result.band_kpoints > 0
        and result.bands is not None
        and result.bands > 0
    )
    return ScientificOutputSummary(
        final_energy_ev=final_energy,
        final_energy_reliable=final_energy_reliable,
        final_energy_source=final_energy_source,
        final_structure_usable=structure_reliable,
        dos_available=result.dos_available,
        dos_reliable=result.dos_available is True and vasprun_reliable,
        band_gap_ev=result.band_gap_ev,
        band_kpoints=result.band_kpoints,
        bands=result.bands,
        band_structure_reliable=band_values_available and vasprun_reliable,
        error=result.error,
        unavailable=tuple(result.unavailable),
    )


def _vasprun_output_reliable(result: ScientificResult) -> bool:
    return result.error is None and not any(
        item.startswith("vasprun.xml could not be parsed")
        or item == "vasprun.xml is unavailable"
        for item in result.unavailable
    )


def _source_filename(path: str | None) -> str | None:
    if not path:
        return None
    return path.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _lifecycle_trajectories(analysis: LifecycleAnalysis) -> tuple[StageTrajectoryObservation, ...]:
    diagnostics = analysis.diagnostics
    return diagnostics.trajectories if diagnostics is not None else ()


def _direct_stage_type(
    trajectory: StageTrajectoryObservation | None,
    incar_settings: Mapping[str, Any] | None = None,
) -> str:
    incar_settings = incar_settings or {}
    configured_nsw = incar_settings.get("NSW")
    configured_ediffg = incar_settings.get("EDIFFG")
    if (
        isinstance(configured_nsw, (int, float))
        and not isinstance(configured_nsw, bool)
        and configured_nsw > 0
    ) or configured_ediffg is not None:
        return "relax"
    if trajectory is None:
        return "static" if incar_settings else "direct_vasp"
    nsw = trajectory.criteria.get("NSW")
    ediffg = trajectory.criteria.get("EDIFFG")
    if (isinstance(nsw, (int, float)) and not isinstance(nsw, bool) and nsw > 0) or ediffg is not None:
        return "relax"
    if trajectory.stage_type and trajectory.stage_type != "direct_vasp":
        return _stage_type(trajectory.stage_type)
    return "static" if trajectory.converged_electronic is not None else "direct_vasp"


def _stage_type(value: Any) -> str:
    return str(value or "unknown").strip().lower()


def _stage_label(index: int, stage: Any) -> str:
    label = stage.get("label")
    if isinstance(label, str) and label:
        return label
    theory = _optional_text(stage.get("theory"))
    stage_type = _stage_type(stage.get("stage_type"))
    return f"{theory.upper()} {stage_type}" if theory else f"stage {index}: {stage_type}"


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _submission_subject_id(submission: Any, fallback: str) -> str:
    if isinstance(submission, Mapping):
        run_name = submission.get("run_name")
        if isinstance(run_name, str) and run_name:
            return run_name
    return fallback


def _local_producer_provenance(submission: Any) -> ProducerProvenance:
    try:
        record = parse_submission_record(submission)
    except ComputeRecordError:
        return ProducerProvenance(status="unavailable")
    source = record.provenance.get("bmd_compute", {})
    source = source.get("source", {}) if isinstance(source, Mapping) else {}
    return _job_producer_provenance(source)


def _job_producer_provenance(source: Any) -> ProducerProvenance:
    if not isinstance(source, Mapping) or not source:
        return ProducerProvenance(status="unavailable")
    status = source.get("status")
    return ProducerProvenance(
        status=str(status) if status is not None else "available",
        producer="BMD Compute",
        repository="bmd_compute",
        git_commit=_optional_text(source.get("git_commit") or source.get("commit")),
        git_state=_optional_text(source.get("state")),
    )


def _scheduler_process_outcome(scheduler: Any) -> str:
    if scheduler is None:
        return PROCESS_UNAVAILABLE
    state = str(getattr(scheduler, "state", "") or "").upper().split()
    state = state[0] if state else ""
    exit_code = getattr(scheduler, "exit_code", None)
    if state == "COMPLETED":
        if exit_code == "0:0":
            return PROCESS_SUCCEEDED
        return PROCESS_UNAVAILABLE if not exit_code else PROCESS_FAILED
    if state in {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "RESIZING", "SUSPENDED", "STAGE_OUT"}:
        return PROCESS_UNAVAILABLE
    if state or (exit_code and exit_code != "0:0"):
        return PROCESS_FAILED
    return PROCESS_UNAVAILABLE


def _lifecycle_process_outcome(analysis: LifecycleAnalysis) -> str:
    scheduler_outcome = _scheduler_process_outcome(analysis.scheduler)
    if scheduler_outcome != PROCESS_UNAVAILABLE:
        return scheduler_outcome
    if analysis.stage_completion and all(
        stage.started and stage.terminated_normally is True
        for stage in analysis.stage_completion
    ):
        return PROCESS_SUCCEEDED
    return PROCESS_UNAVAILABLE


def _package_version() -> str:
    try:
        return version("bmd-agent")
    except PackageNotFoundError:
        return "unknown"
