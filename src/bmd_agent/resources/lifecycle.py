from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import json
from pathlib import Path
from pathlib import PurePosixPath
import posixpath
from typing import Any
import warnings

from bmd_agent.resources.completion import (
    WORKFLOW_COMPLETED,
    WORKFLOW_INCOMPLETE,
    StageCompletion,
    stage_convergence,
    workflow_completion_status,
)
from bmd_agent.resources.custodian import (
    CustodianInterventionEvidence,
    CustodianPolicyEvidence,
    TerminationEvidenceAssessment,
    assess_termination_evidence,
    parse_custodian_json,
    parse_custodian_policy_provenance,
)
from bmd_agent.resources.diagnostics import (
    BoundedLogDiagnostic as LocalLogDiagnostic,
    diagnostic_log_messages,
)
from bmd_agent.resources.compute_records import (
    ComputeRecordError,
    SubmissionRecord,
    parse_submission_record,
)
from bmd_agent.resources.oom import OomDiagnosticEvidence, assess_oom_evidence
from bmd_agent.resources.run import (
    ConvergenceProgressAssessment,
    IncarObservation,
    IonicStepObservation,
    ScientificResult,
    StageTrajectoryObservation,
    WorkflowStage,
    assess_convergence_progress,
    parse_incar_contents,
    parse_oszicar_trajectory,
    parse_outcar_force_blocks,
    parse_vasp_output_files,
)
from bmd_agent.resources.slurm import SlurmAccountingRecord, normalize_job_id
from bmd_agent.resources.vasp import StructureInfo, parse_poscar


class LifecycleState(str, Enum):
    PRE_RUN = "PRE_RUN"
    RUNNING = "RUNNING"
    INCOMPLETE = "INCOMPLETE"
    COMPLETED = "COMPLETED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class LocalFileEvidence:
    name: str
    path: Path
    present: bool
    size: int | None = None

    @property
    def non_empty(self) -> bool:
        return self.present and self.size is not None and self.size > 0


@dataclass(frozen=True)
class LocalStageBinding:
    label: str
    path: Path
    stage_index: int | None = None
    producer_path: str | None = None


@dataclass(frozen=True)
class LocalStageEvidence:
    label: str
    path: Path
    stage_index: int | None
    producer_path: str | None
    has_required_inputs: bool
    has_meaningful_execution: bool
    normal_completion: bool


@dataclass(frozen=True)
class BmdWorkflowDiscovery:
    workflow_root: Path
    submission_path: Path
    submission: Mapping[str, Any]
    workflow_stages: tuple[Mapping[str, Any], ...]
    stage_bindings: tuple[LocalStageBinding, ...]
    stage_evidence: tuple[LocalStageEvidence, ...] = ()
    current_stage: LocalStageBinding | None = None
    producer_root: str | None = None
    relocated: bool = False
    job_id: str | None = None
    attempt_state_path: Path | None = None
    attempt_state: Mapping[str, Any] | None = None
    record_contract: str | None = None
    record_limitations: tuple[str, ...] = ()
    custodian_policy: CustodianPolicyEvidence = field(
        default_factory=lambda: CustodianPolicyEvidence(
            available=False,
            reason="submission has no persisted Custodian execution-policy provenance",
        )
    )


@dataclass(frozen=True)
class LifecycleAnalysis:
    state: LifecycleState
    directory: Path
    calculation_kind: str
    message: str
    input_files: Mapping[str, LocalFileEvidence] = field(default_factory=dict)
    output_files: Mapping[str, LocalFileEvidence] = field(default_factory=dict)
    bmd_workflow: BmdWorkflowDiscovery | None = None
    scheduler: SlurmAccountingRecord | None = None
    scheduler_error: str | None = None
    normal_completion: bool | None = None
    structure: StructureInfo | None = None
    incar_settings: Mapping[str, Any] = field(default_factory=dict)
    scientific: ScientificResult | None = None
    diagnostics: LocalExecutionDiagnostics | None = None
    evidence_gaps: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    stage_completion: tuple[StageCompletion, ...] = ()


SchedulerLookup = Callable[[str], SlurmAccountingRecord | None]

_INPUT_FILENAMES = ("POSCAR", "INCAR", "KPOINTS")
_OUTPUT_FILENAMES = ("OUTCAR", "OSZICAR", "vasprun.xml", "CONTCAR", "DOSCAR", "XDATCAR")
_BMD_LOG_KEYS = ("log_out", "log_err", "slurm_out", "slurm_err")
_LOCAL_DIAGNOSTIC_LOG_FILENAMES = (
    "std_err.txt",
    "vasp.out",
    "stdout.txt",
    "stderr.txt",
    "OUTCAR",
)
_LOCAL_DIAGNOSTIC_RECENT_WINDOW = 5
_LOCAL_VASPRUN_MAX_BYTES = 50_000_000
_LOCAL_OUTCAR_DIAGNOSTIC_MAX_BYTES = 2_000_000
_LOCAL_LOG_DIAGNOSTIC_MAX_BYTES = 128_000
_LOCAL_CUSTODIAN_MAX_BYTES = 2_000_000
_DIAGNOSTIC_CRITERIA_KEYS = ("NELM", "EDIFF", "NSW", "EDIFFG", "ISIF")
_ACTIVE_SCHEDULER_STATES = {
    "BOOT_FAIL_REQUEUE_FED",
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "STAGE_OUT",
    "SUSPENDED",
}
_INACTIVE_UNSUCCESSFUL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
_NORMAL_COMPLETION_MARKERS = (
    "General timing and accounting informations for this job",
    "Voluntary context switches",
)
@dataclass(frozen=True)
class LocalExecutionDiagnostics:
    trajectories: tuple[StageTrajectoryObservation, ...] = ()
    assessments: tuple[ConvergenceProgressAssessment, ...] = ()
    logs: tuple[LocalLogDiagnostic, ...] = ()
    custodian: CustodianInterventionEvidence | None = None
    error_archives: tuple[LocalFileEvidence, ...] = ()
    suggested_checks: tuple[str, ...] = ()
    oom: OomDiagnosticEvidence | None = None
    termination: TerminationEvidenceAssessment | None = None


def analyze_calculation_directory(
    directory: Path | str,
    *,
    scheduler_lookup: SchedulerLookup | None = None,
    max_ancestor_levels: int = 4,
) -> LifecycleAnalysis:
    """Classify a calculation directory from local, read-only evidence."""

    current = Path(directory).resolve()
    workflow, rejected_records = _discover_bmd_workflow(
        current,
        max_ancestor_levels=max_ancestor_levels,
    )
    if workflow is not None:
        analysis = _analyze_bmd_workflow(
            current,
            workflow,
            scheduler_lookup=scheduler_lookup,
        )
        return _with_limitations(analysis, (*workflow.record_limitations, *rejected_records))
    return _with_limitations(_analyze_direct_vasp_directory(current), rejected_records)


def _with_limitations(analysis: LifecycleAnalysis, extra: Sequence[str]) -> LifecycleAnalysis:
    if not extra:
        return analysis
    return replace(analysis, limitations=tuple(_ordered_unique((*analysis.limitations, *extra))))


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _analyze_bmd_workflow(
    current: Path,
    workflow: BmdWorkflowDiscovery,
    *,
    scheduler_lookup: SchedulerLookup | None,
) -> LifecycleAnalysis:
    target = workflow.current_stage.path if workflow.current_stage else workflow.workflow_root
    input_files = _observe_files(target, _INPUT_FILENAMES)
    output_files = _observe_files(target, _OUTPUT_FILENAMES)
    log_files = _observe_bmd_logs(workflow)
    # Lifecycle state is a workflow-level claim, so execution evidence is read
    # from every declared stage rather than only the invocation directory.
    meaningful_execution = (
        _has_meaningful_files(output_files)
        or _has_meaningful_files(log_files)
        or any(evidence.has_meaningful_execution for evidence in workflow.stage_evidence)
    )
    if workflow.relocated:
        scheduler, scheduler_error = (
            None,
            "scheduler evidence belongs to the original producer location and was not used for relocated local snapshot",
        )
    else:
        scheduler, scheduler_error = _lookup_scheduler(workflow.job_id, scheduler_lookup)
    normal_completion = _detect_normal_completion(target)
    scientific = _derive_local_scientific(target, workflow.submission)
    structure = _structure_from_inputs(input_files)
    incar_settings = _incar_settings(input_files)
    gaps = _input_gaps(input_files)
    diagnostics = (
        _derive_bmd_execution_diagnostics(workflow, scheduler=scheduler)
        if meaningful_execution
        else None
    )
    stage_completion = _bmd_stage_completion(workflow, diagnostics)
    execution_succeeded = _scheduler_success(scheduler)
    workflow_status = workflow_completion_status(
        stage_completion,
        execution_succeeded=execution_succeeded,
    )

    def result(
        state: LifecycleState,
        message: str,
        *,
        include_diagnostics: bool = True,
        limitations: Sequence[str] = (),
    ) -> LifecycleAnalysis:
        return LifecycleAnalysis(
            state=state,
            directory=current,
            calculation_kind="BMD Compute",
            message=message,
            input_files=input_files,
            output_files=output_files | log_files,
            bmd_workflow=workflow,
            scheduler=scheduler,
            scheduler_error=scheduler_error,
            normal_completion=normal_completion,
            structure=structure,
            incar_settings=incar_settings,
            scientific=scientific,
            diagnostics=diagnostics if include_diagnostics else None,
            evidence_gaps=tuple(gaps),
            limitations=tuple(limitations),
            stage_completion=stage_completion,
        )

    if scheduler is not None and _is_active_scheduler_state(scheduler.state):
        return result(
            LifecycleState.RUNNING,
            "Active scheduler evidence indicates the calculation is running.",
        )

    if workflow_status == WORKFLOW_COMPLETED:
        return result(
            LifecycleState.COMPLETED,
            (
                "Every producer-declared stage has VASP normal-termination evidence "
                "and positive convergence evidence."
            ),
            include_diagnostics=(
                diagnostics is not None
                and diagnostics.custodian is not None
                and bool(diagnostics.custodian.corrections)
            ),
        )

    if scheduler is not None and _is_inactive_unsuccessful_scheduler_state(scheduler):
        return result(
            LifecycleState.INCOMPLETE,
            "Scheduler evidence indicates execution stopped before successful completion.",
        )

    if scheduler is not None and _scheduler_nonzero_exit(scheduler) and not normal_completion:
        return result(
            LifecycleState.INCOMPLETE,
            "Scheduler exit status is non-zero and successful VASP completion is absent.",
        )

    if workflow_status == WORKFLOW_INCOMPLETE:
        return result(
            LifecycleState.INCOMPLETE,
            (
                "Execution ended, but not every producer-declared stage has "
                "normal-termination and convergence evidence."
            ),
        )

    has_inputs = _has_required_inputs(input_files) or any(
        evidence.has_required_inputs for evidence in workflow.stage_evidence
    )
    if has_inputs and not meaningful_execution:
        return result(
            LifecycleState.PRE_RUN,
            "Inputs are present and no meaningful execution output was observed.",
        )

    if meaningful_execution:
        limitations: list[str] = []
        if workflow.job_id and scheduler is None:
            limitations.append(scheduler_error or "scheduler accounting was unavailable")
        limitations.append(
            "partial execution evidence is present, but no reliable active/inactive execution state was established"
        )
        if execution_succeeded:
            limitations.append(
                "a successful scheduler exit does not by itself establish convergence or workflow completion"
            )
        return result(
            LifecycleState.UNKNOWN,
            "BMD Compute evidence was found, but lifecycle state is not deterministic.",
            limitations=limitations,
        )

    return result(
        LifecycleState.UNKNOWN,
        "BMD Compute provenance was found, but calculation inputs or execution evidence are incomplete.",
    )


def _analyze_direct_vasp_directory(directory: Path) -> LifecycleAnalysis:
    input_files = _observe_files(directory, _INPUT_FILENAMES)
    output_files = _observe_files(directory, _OUTPUT_FILENAMES)
    meaningful_execution = _has_meaningful_files(output_files)
    normal_completion = _detect_normal_completion(directory)
    structure = _structure_from_inputs(input_files)
    incar_settings = _incar_settings(input_files)
    scientific = _derive_local_scientific(directory, None)
    gaps = _input_gaps(input_files)
    diagnostics = (
        _derive_direct_execution_diagnostics(directory)
        if meaningful_execution
        else None
    )
    stage_completion = (
        _direct_stage_completion(diagnostics, normal_completion=normal_completion)
        if meaningful_execution
        else ()
    )

    def result(
        state: LifecycleState,
        message: str,
        *,
        include_diagnostics: bool = True,
        limitations: Sequence[str] = (),
        normal: bool = False,
    ) -> LifecycleAnalysis:
        return LifecycleAnalysis(
            state=state,
            directory=directory,
            calculation_kind="direct VASP",
            message=message,
            input_files=input_files,
            output_files=output_files,
            normal_completion=normal,
            structure=structure,
            incar_settings=incar_settings,
            scientific=scientific,
            diagnostics=diagnostics if include_diagnostics else None,
            evidence_gaps=tuple(gaps),
            limitations=tuple(limitations),
            stage_completion=stage_completion,
        )

    if normal_completion:
        status = workflow_completion_status(stage_completion)
        if status == WORKFLOW_COMPLETED:
            return result(
                LifecycleState.COMPLETED,
                "VASP normal-termination evidence and positive convergence evidence were found.",
                include_diagnostics=(
                    diagnostics is not None
                    and diagnostics.custodian is not None
                    and bool(diagnostics.custodian.corrections)
                ),
                normal=True,
            )
        if status == WORKFLOW_INCOMPLETE:
            return result(
                LifecycleState.INCOMPLETE,
                "VASP terminated normally, but its convergence criteria were not reached.",
                normal=True,
            )
        return result(
            LifecycleState.UNKNOWN,
            (
                "VASP normal-termination evidence was found, but convergence could not "
                "be established from the available evidence."
            ),
            limitations=(
                "normal VASP termination does not by itself establish electronic or ionic convergence",
            ),
            normal=True,
        )

    if _has_required_inputs(input_files) and not meaningful_execution:
        return result(
            LifecycleState.PRE_RUN,
            "Inputs are present and no meaningful execution output was observed.",
        )

    if meaningful_execution:
        return result(
            LifecycleState.UNKNOWN,
            (
                "Partial VASP execution evidence is present, but no scheduler/provenance "
                "or durable normal-completion evidence establishes whether it is active or stopped."
            ),
            limitations=(
                "manual VASP directory has no scheduler/provenance evidence for lifecycle status",
            ),
        )

    if any(observation.present for observation in input_files.values()):
        return result(
            LifecycleState.UNKNOWN,
            "Some VASP input evidence is present, but required inputs are incomplete.",
        )

    return LifecycleAnalysis(
        state=LifecycleState.UNKNOWN,
        directory=directory,
        calculation_kind="none",
        message="No recognizable BMD Compute or VASP calculation found in this directory.",
        input_files=input_files,
        output_files=output_files,
        evidence_gaps=("no recognizable BMD Compute or VASP calculation evidence was found",),
    )


def _bmd_stage_completion(
    workflow: BmdWorkflowDiscovery,
    diagnostics: LocalExecutionDiagnostics | None,
) -> tuple[StageCompletion, ...]:
    """Build completion evidence for every producer-declared stage."""

    evidence_by_index = {
        item.stage_index: item for item in workflow.stage_evidence if item.stage_index is not None
    }
    trajectories = {
        item.stage_index: item
        for item in (diagnostics.trajectories if diagnostics is not None else ())
        if item.stage_index is not None
    }
    completion: list[StageCompletion] = []
    for index, payload in enumerate(workflow.workflow_stages, start=1):
        stage = _workflow_stage_from_mapping(index, payload)
        evidence = evidence_by_index.get(index)
        started = bool(evidence is not None and evidence.has_meaningful_execution)
        trajectory = trajectories.get(index) if started else None
        completion.append(
            StageCompletion(
                index=index,
                label=_stage_display_label(stage),
                started=started,
                terminated_normally=bool(evidence.normal_completion) if evidence is not None else False,
                convergence=stage_convergence(trajectory),
            )
        )
    return tuple(completion)


def _direct_stage_completion(
    diagnostics: LocalExecutionDiagnostics | None,
    *,
    normal_completion: bool,
) -> tuple[StageCompletion, ...]:
    trajectory = diagnostics.trajectories[0] if diagnostics and diagnostics.trajectories else None
    return (
        StageCompletion(
            index=1,
            label="direct VASP",
            started=True,
            terminated_normally=normal_completion,
            convergence=stage_convergence(trajectory),
        ),
    )


def _stage_display_label(stage: WorkflowStage) -> str:
    theory = str(stage.theory).upper() if stage.theory else "unknown theory"
    return f"{theory} {stage.stage_type or 'calculation'}"


def _discover_bmd_workflow(
    current: Path,
    *,
    max_ancestor_levels: int,
) -> tuple[BmdWorkflowDiscovery | None, tuple[str, ...]]:
    """Find the nearest BMD Compute workflow and report submission records not read."""

    rejected: list[str] = []
    for root in _bounded_ancestors(current, max_ancestor_levels=max_ancestor_levels):
        submission_path = root / "submission.json"
        if not submission_path.is_file():
            continue
        try:
            submission = _read_json(submission_path)
        except Exception:
            rejected.append(f"{submission_path} could not be read as a JSON object")
            continue
        try:
            record = parse_submission_record(submission)
        except ComputeRecordError as exc:
            rejected.append(f"{submission_path} was not used: {exc}")
            continue
        try:
            workflow = _workflow_from_submission(root, submission_path, submission, record, current)
        except Exception:
            continue
        if workflow is not None:
            return workflow, tuple(rejected)
    return None, tuple(rejected)


def _workflow_from_submission(
    root: Path,
    submission_path: Path,
    submission: Mapping[str, Any],
    record: SubmissionRecord,
    current: Path,
) -> BmdWorkflowDiscovery | None:
    stages = record.workflow_stages
    stage_dirs = {label: path for _index, label, path in record.stage_dirs}
    paths = {"stage_dirs": stage_dirs, "result_dir": record.result_dir}
    stage_bindings = _stage_bindings(root, paths, stage_count=len(stages))
    producer_root = _producer_root_text(_mapping(submission.get("paths")))
    relocated_stage_bindings = _relocated_stage_bindings(
        root,
        paths,
        stage_count=len(stages),
        producer_root=producer_root,
    )
    relocated = _is_relocated_local_snapshot(
        root,
        current,
        stage_bindings,
        relocated_stage_bindings,
    )
    if relocated:
        stage_bindings = relocated_stage_bindings
    related = current == root or any(_is_relative_to(current, binding.path) for binding in stage_bindings)
    if not related:
        return None
    current_stage = _current_stage_binding(current, root, stage_bindings)
    stage_evidence = _stage_evidence(stage_bindings)
    attempt_state_path = _optional_local_path(record.attempt_state, root=root)
    attempt_state = _read_json(attempt_state_path) if attempt_state_path and attempt_state_path.is_file() else None
    return BmdWorkflowDiscovery(
        workflow_root=root,
        submission_path=submission_path,
        submission=submission,
        workflow_stages=stages,
        stage_bindings=stage_bindings,
        stage_evidence=stage_evidence,
        current_stage=current_stage,
        producer_root=producer_root,
        relocated=relocated,
        job_id=_find_job_id(submission, attempt_state),
        attempt_state_path=attempt_state_path,
        attempt_state=attempt_state,
        record_contract=record.label,
        record_limitations=record.limitations,
        custodian_policy=parse_custodian_policy_provenance(submission),
    )


def _stage_bindings(
    root: Path,
    paths: Mapping[str, Any],
    *,
    stage_count: int,
) -> tuple[LocalStageBinding, ...]:
    bindings: list[LocalStageBinding] = []
    seen: set[Path] = set()
    stage_dirs = paths.get("stage_dirs")
    if isinstance(stage_dirs, Mapping):
        for index, (label, value) in enumerate(stage_dirs.items(), start=1):
            path = _optional_local_path(value, root=root)
            if path is None:
                continue
            bindings.append(LocalStageBinding(str(label), path, index, _path_text(value)))
            seen.add(path)
    result_dir = _optional_local_path(paths.get("result_dir"), root=root)
    if result_dir is not None and result_dir not in seen:
        stage_index = 1 if not bindings and stage_count == 1 else stage_count
        bindings.append(LocalStageBinding("result_dir", result_dir, stage_index, _path_text(paths.get("result_dir"))))
    return tuple(bindings)


def _relocated_stage_bindings(
    root: Path,
    paths: Mapping[str, Any],
    *,
    stage_count: int,
    producer_root: str | None,
) -> tuple[LocalStageBinding, ...]:
    if producer_root is None:
        return ()
    bindings: list[LocalStageBinding] = []
    seen: set[Path] = set()
    seen_producer_paths: set[str] = set()
    stage_dirs = paths.get("stage_dirs")
    if isinstance(stage_dirs, Mapping):
        for index, (label, value) in enumerate(stage_dirs.items(), start=1):
            producer_path = _path_text(value)
            local_path = _rebase_producer_path(producer_path, producer_root, root)
            if local_path is None:
                continue
            bindings.append(LocalStageBinding(str(label), local_path, index, producer_path))
            seen.add(local_path)
            normalized = _normalize_posix_path_text(producer_path)
            if normalized is not None:
                seen_producer_paths.add(normalized)

    result_producer_path = _path_text(paths.get("result_dir"))
    normalized_result = _normalize_posix_path_text(result_producer_path)
    result_path = _rebase_producer_path(result_producer_path, producer_root, root)
    if (
        result_path is not None
        and result_path not in seen
        and (normalized_result is None or normalized_result not in seen_producer_paths)
    ):
        stage_index = 1 if not bindings and stage_count == 1 else stage_count
        bindings.append(LocalStageBinding("result_dir", result_path, stage_index, result_producer_path))
    return tuple(bindings)


def _is_relocated_local_snapshot(
    root: Path,
    current: Path,
    bindings: Sequence[LocalStageBinding],
    relocated_bindings: Sequence[LocalStageBinding],
) -> bool:
    if current != root and not _is_relative_to(current, root):
        return False
    if not relocated_bindings:
        return False
    if any(_has_recognizable_local_vasp_evidence(binding.path) for binding in bindings):
        return False
    if not any(_has_recognizable_local_vasp_evidence(binding.path) for binding in relocated_bindings):
        return False
    if current == root:
        return True
    return any(_is_relative_to(current, binding.path) for binding in relocated_bindings)


def _has_recognizable_local_vasp_evidence(directory: Path) -> bool:
    return _has_meaningful_files(_observe_files(directory, _INPUT_FILENAMES + _OUTPUT_FILENAMES))


def _stage_evidence(bindings: Sequence[LocalStageBinding]) -> tuple[LocalStageEvidence, ...]:
    evidence: list[LocalStageEvidence] = []
    for binding in bindings:
        input_files = _observe_files(binding.path, _INPUT_FILENAMES)
        output_files = _observe_files(binding.path, _OUTPUT_FILENAMES)
        evidence.append(
            LocalStageEvidence(
                label=binding.label,
                path=binding.path,
                stage_index=binding.stage_index,
                producer_path=binding.producer_path,
                has_required_inputs=_has_required_inputs(input_files),
                has_meaningful_execution=_has_meaningful_files(output_files),
                normal_completion=_detect_normal_completion(binding.path),
            )
        )
    return tuple(evidence)


def _derive_bmd_execution_diagnostics(
    workflow: BmdWorkflowDiscovery,
    *,
    scheduler: SlurmAccountingRecord | None,
) -> LocalExecutionDiagnostics:
    stage_map = {
        index: _workflow_stage_from_mapping(index, payload)
        for index, payload in enumerate(workflow.workflow_stages, start=1)
    }
    bindings = tuple(
        binding for binding in workflow.stage_bindings
        if _has_meaningful_files(_observe_files(binding.path, _OUTPUT_FILENAMES))
    )
    if not bindings and workflow.current_stage is not None:
        bindings = (workflow.current_stage,)
    trajectories = tuple(
        _observe_local_stage_trajectory(
            binding,
            stage_map.get(binding.stage_index),
        )
        for binding in bindings
    )
    declared_logs = tuple(
        observation.path
        for observation in _observe_bmd_logs(workflow).values()
        if observation.present
    )
    return _local_execution_diagnostics(
        workflow.workflow_root,
        tuple(binding.path for binding in bindings),
        trajectories,
        scheduler=scheduler,
        explicit_log_paths=declared_logs,
    )


def _derive_direct_execution_diagnostics(directory: Path) -> LocalExecutionDiagnostics:
    binding = LocalStageBinding("work_dir", directory, 1)
    stage = WorkflowStage(
        index=1,
        stage_type="direct_vasp",
        theory="unknown",
        modifiers=(),
        label=None,
    )
    trajectory = _observe_local_stage_trajectory(binding, stage)
    return _local_execution_diagnostics(directory, (directory,), (trajectory,))


def _local_execution_diagnostics(
    root: Path,
    directories: Sequence[Path],
    trajectories: Sequence[StageTrajectoryObservation],
    *,
    scheduler: SlurmAccountingRecord | None = None,
    explicit_log_paths: Sequence[Path] = (),
) -> LocalExecutionDiagnostics:
    unique_directories = _unique_paths((root, *directories))
    logs = _observe_local_logs(unique_directories, explicit_paths=explicit_log_paths)
    custodian = _observe_local_custodian(unique_directories)
    archives = _observe_error_archives(unique_directories)
    assessments = assess_convergence_progress(tuple(trajectories))
    log_observations = [
        (f"{log.label} ({log.path})", message)
        for log in logs
        for message in log.messages
    ]
    if custodian is not None:
        log_observations.extend(
            (f"custodian.json ({custodian.path})", event)
            for event in custodian.events
        )
    inspected_log_sources = [
        f"{log.label} ({log.path})"
        for log in logs
        if log.error is None
    ]
    source_limitations = [
        f"{log.label} could not be read through the bounded diagnostic path"
        for log in logs
        if log.error is not None
    ]
    if custodian is not None and custodian.error is None:
        inspected_log_sources.append(f"custodian.json ({custodian.path})")
    elif custodian is not None:
        source_limitations.append("custodian.json could not be parsed for diagnostic evidence")
    return LocalExecutionDiagnostics(
        trajectories=tuple(trajectories),
        assessments=assessments,
        logs=logs,
        custodian=custodian,
        error_archives=archives,
        suggested_checks=_suggested_diagnostic_checks(
            trajectories,
            logs,
            custodian,
            archives,
        ),
        oom=assess_oom_evidence(
            scheduler,
            log_observations=log_observations,
            inspected_log_sources=inspected_log_sources,
            source_limitations=source_limitations,
        ),
        termination=assess_termination_evidence(
            scheduler_state=scheduler.state if scheduler else None,
            custodian_evidence=(custodian,) if custodian is not None else (),
            log_messages=tuple(message for _, message in log_observations),
            error_archive_count=len(archives),
        ),
    )


def _workflow_stage_from_mapping(index: int, payload: Mapping[str, Any]) -> WorkflowStage:
    modifiers = payload.get("modifiers", ())
    if not isinstance(modifiers, Sequence) or isinstance(modifiers, (str, bytes)):
        modifiers = ()
    return WorkflowStage(
        index=index,
        stage_type=str(payload.get("stage_type") or "unknown"),
        theory=str(payload.get("theory") or "unknown"),
        modifiers=tuple(str(item) for item in modifiers),
        label=str(payload["label"]) if payload.get("label") is not None else None,
        options=_mapping(payload.get("options")),
    )


def _observe_local_stage_trajectory(
    binding: LocalStageBinding,
    stage: WorkflowStage | None,
) -> StageTrajectoryObservation:
    directory = binding.path
    unavailable: list[str] = []
    oszicar_path = directory / "OSZICAR"
    oszicar_present = oszicar_path.is_file()
    oszicar_error = None
    oszicar_trajectory = None
    if oszicar_present:
        try:
            oszicar_trajectory = parse_oszicar_trajectory(_read_file_prefix(oszicar_path, limit=2_000_000))
        except Exception as exc:
            oszicar_error = str(exc)
            unavailable.append(f"OSZICAR could not be parsed: {exc}")
    else:
        unavailable.append("OSZICAR is unavailable")

    vasprun = _observe_local_vasprun_trajectory(directory)
    if vasprun["skipped_reason"]:
        unavailable.append(str(vasprun["skipped_reason"]))
    if vasprun["error"]:
        unavailable.append(f"vasprun trajectory enrichment unavailable: {vasprun['error']}")
    if not vasprun["present"]:
        unavailable.append("vasprun.xml is unavailable")

    outcar = _observe_local_outcar_force_trajectory(directory)
    if outcar["error"]:
        unavailable.append(f"OUTCAR force trajectory unavailable: {outcar['error']}")
    if not outcar["present"]:
        unavailable.append("OUTCAR is unavailable")

    ionic_steps = tuple(getattr(oszicar_trajectory, "ionic_steps", ()) if oszicar_trajectory else ())
    outcar_blocks = tuple(outcar["blocks"])
    outcar_complete = tuple(block for block in outcar_blocks if block.complete)
    completed_steps = getattr(oszicar_trajectory, "completed_ionic_steps", None) if oszicar_trajectory else None
    alignment_status = "unavailable"
    alignment_reason = "OSZICAR completed ionic step count unavailable"
    if outcar_complete and completed_steps is not None:
        if len(outcar_complete) == completed_steps:
            ionic_steps = _merge_local_outcar_forces(ionic_steps, outcar_complete)
            alignment_status = "aligned"
            alignment_reason = (
                f"{len(outcar_complete)} OUTCAR force block(s) aligned with OSZICAR completed ionic steps"
            )
        else:
            alignment_status = "discrepancy"
            alignment_reason = (
                f"OSZICAR completed ionic steps {completed_steps} did not align with "
                f"OUTCAR complete force blocks {len(outcar_complete)}"
            )
            unavailable.append(f"OUTCAR force-block alignment discrepancy: {alignment_reason}")
    elif outcar["present"] and not outcar_complete:
        alignment_reason = "OUTCAR contained no complete validated force blocks"

    criteria, source_values, discrepancies = _local_trajectory_criteria(
        binding,
        vasprun["parameters"],
    )
    return StageTrajectoryObservation(
        stage_index=binding.stage_index,
        stage_label=binding.label,
        stage_type=stage.stage_type if stage else None,
        theory=stage.theory if stage else None,
        directory=str(directory),
        oszicar_path=str(oszicar_path),
        oszicar_present=oszicar_present,
        oszicar_error=oszicar_error,
        vasprun_path=str(directory / "vasprun.xml"),
        vasprun_present=bool(vasprun["present"]),
        vasprun_error=vasprun["error"],
        vasprun_skipped_reason=vasprun["skipped_reason"],
        outcar_path=str(directory / "OUTCAR"),
        outcar_present=bool(outcar["present"]),
        outcar_error=outcar["error"],
        outcar_expected_site_count=outcar["expected_site_count"],
        outcar_force_blocks=outcar_blocks,
        outcar_complete_force_blocks=len(outcar_complete) if outcar["present"] and outcar["error"] is None else None,
        outcar_force_alignment_status=alignment_status,
        outcar_force_alignment_reason=alignment_reason,
        criteria=criteria,
        criteria_source_values=source_values,
        criteria_discrepancies=discrepancies,
        ionic_steps_observed=getattr(oszicar_trajectory, "ionic_steps_observed", None),
        electronic_iterations_by_ionic_step=getattr(oszicar_trajectory, "electronic_iterations_by_ionic_step", ()),
        electronic_cycles=getattr(oszicar_trajectory, "electronic_cycles", ()),
        final_electronic_iteration_count=getattr(oszicar_trajectory, "final_electronic_iteration_count", None),
        recent_electronic_iterations=getattr(oszicar_trajectory, "recent_electronic_iterations", ()),
        completed_ionic_steps=completed_steps,
        electronic_iterations_by_completed_ionic_step=getattr(
            oszicar_trajectory,
            "electronic_iterations_by_completed_ionic_step",
            (),
        ),
        incomplete_electronic_iteration_count=getattr(
            oszicar_trajectory,
            "incomplete_electronic_iteration_count",
            None,
        ),
        recent_incomplete_electronic_iterations=getattr(
            oszicar_trajectory,
            "recent_incomplete_electronic_iterations",
            (),
        ),
        ionic_steps=ionic_steps,
        recent_ionic_steps=tuple(ionic_steps[-_LOCAL_DIAGNOSTIC_RECENT_WINDOW:]),
        vasprun_ionic_steps=vasprun["ionic_steps"],
        converged_electronic=vasprun["converged_electronic"],
        converged_ionic=vasprun["converged_ionic"],
        unavailable=tuple(unavailable),
    )


def _observe_local_vasprun_trajectory(directory: Path) -> Mapping[str, Any]:
    path = directory / "vasprun.xml"
    if not path.is_file():
        return _local_vasprun_result(present=False)
    try:
        size = path.stat().st_size
    except OSError as exc:
        return _local_vasprun_result(present=True, error=str(exc))
    if size > _LOCAL_VASPRUN_MAX_BYTES:
        return _local_vasprun_result(
            present=True,
            skipped_reason=(
                f"vasprun.xml skipped because size {size} bytes exceeds limit "
                f"{_LOCAL_VASPRUN_MAX_BYTES}"
            ),
        )
    try:
        from pymatgen.io.vasp.outputs import Vasprun
    except Exception as exc:
        return _local_vasprun_result(present=True, error=str(exc))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            vasprun = Vasprun(str(path), parse_eigenvalues=False)
        if _has_malformed_xml_warning(caught):
            return _local_vasprun_result(present=True, error="file could not be parsed completely")
        ionic_steps = tuple(getattr(vasprun, "ionic_steps", ()) or ())
        return _local_vasprun_result(
            present=True,
            parameters=_vasp_parameter_values(getattr(vasprun, "parameters", None)),
            ionic_steps=len(ionic_steps),
            converged_electronic=_bool_or_none(getattr(vasprun, "converged_electronic", None)),
            converged_ionic=_bool_or_none(getattr(vasprun, "converged_ionic", None)),
        )
    except Exception:
        return _local_vasprun_result(present=True, error="file could not be parsed completely")


def _local_vasprun_result(
    *,
    present: bool,
    error: str | None = None,
    skipped_reason: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    ionic_steps: int | None = None,
    converged_electronic: bool | None = None,
    converged_ionic: bool | None = None,
) -> Mapping[str, Any]:
    return {
        "present": present,
        "error": error,
        "skipped_reason": skipped_reason,
        "parameters": parameters or {},
        "ionic_steps": ionic_steps,
        "converged_electronic": converged_electronic,
        "converged_ionic": converged_ionic,
    }


def _observe_local_outcar_force_trajectory(directory: Path) -> Mapping[str, Any]:
    path = directory / "OUTCAR"
    if not path.is_file():
        return {
            "present": False,
            "error": None,
            "expected_site_count": None,
            "blocks": (),
        }
    try:
        contents = _read_file_tail(path, limit=_LOCAL_OUTCAR_DIAGNOSTIC_MAX_BYTES)
    except OSError as exc:
        return {
            "present": True,
            "error": str(exc),
            "expected_site_count": None,
            "blocks": (),
        }
    expected = _expected_local_site_count(directory)
    try:
        blocks = parse_outcar_force_blocks(
            contents,
            source_path=str(path),
            expected_site_count=expected,
        )
    except Exception as exc:
        return {
            "present": True,
            "error": str(exc),
            "expected_site_count": expected,
            "blocks": (),
        }
    return {
        "present": True,
        "error": None,
        "expected_site_count": expected,
        "blocks": blocks,
    }


def _expected_local_site_count(directory: Path) -> int | None:
    for filename in ("POSCAR", "CONTCAR"):
        path = directory / filename
        if not path.is_file():
            continue
        try:
            structure = parse_poscar(path.read_bytes(), source=str(path))
        except Exception:
            continue
        return structure.sites
    return None


def _merge_local_outcar_forces(
    ionic_steps: Sequence[IonicStepObservation],
    complete_blocks: Sequence[Any],
) -> tuple[IonicStepObservation, ...]:
    by_step = {
        index: block.max_force_eV_per_A
        for index, block in enumerate(complete_blocks, start=1)
        if block.max_force_eV_per_A is not None
    }
    merged: list[IonicStepObservation] = []
    for step in ionic_steps:
        incoming = by_step.get(step.step_index)
        merged.append(
            IonicStepObservation(
                step_index=step.step_index,
                electronic_iterations=step.electronic_iterations,
                free_energy=step.free_energy,
                energy_zero=step.energy_zero,
                dE=step.dE,
                max_force=incoming if step.max_force is None and incoming is not None else step.max_force,
                max_force_source=(
                    "OUTCAR"
                    if step.max_force is None and incoming is not None
                    else step.max_force_source
                ),
            )
        )
    return tuple(merged)


def _local_trajectory_criteria(
    binding: LocalStageBinding,
    vasprun_parameters: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]], tuple[str, ...]]:
    sources: list[tuple[str, Mapping[str, Any]]] = []
    incar = _local_incar_observation(binding)
    if incar.present and not incar.error:
        sources.append((f"retained_incar:{binding.label}", incar.values))
    if vasprun_parameters:
        sources.append(("diagnose_vasprun.parameters", vasprun_parameters))

    criteria: dict[str, Any] = {}
    source_values: dict[str, Mapping[str, Any]] = {}
    discrepancies: list[str] = []
    for key in _DIAGNOSTIC_CRITERIA_KEYS:
        values = {
            source: values[key]
            for source, values in sources
            if key in values
        }
        if not values:
            continue
        source_values[key] = values
        observed = list(values.values())
        criteria[key] = observed[0]
        if not all(_json_equivalent(observed[0], value) for value in observed[1:]):
            discrepancies.append(key)
    return criteria, source_values, tuple(discrepancies)


def _local_incar_observation(binding: LocalStageBinding) -> IncarObservation:
    path = binding.path / "INCAR"
    if not path.is_file():
        return IncarObservation(
            label=binding.label,
            path=str(path),
            present=False,
            stage_index=binding.stage_index,
        )
    try:
        settings, error = parse_incar_contents(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return IncarObservation(
            label=binding.label,
            path=str(path),
            present=True,
            stage_index=binding.stage_index,
            error=str(exc),
        )
    return IncarObservation(
        label=binding.label,
        path=str(path),
        present=True,
        stage_index=binding.stage_index,
        values={} if error else dict(settings),
        error=error,
    )


def _observe_local_logs(
    directories: Sequence[Path],
    *,
    explicit_paths: Sequence[Path] = (),
) -> tuple[LocalLogDiagnostic, ...]:
    diagnostics: list[LocalLogDiagnostic] = []
    seen: set[Path] = set()
    candidates = [
        directory / filename
        for directory in directories
        for filename in _LOCAL_DIAGNOSTIC_LOG_FILENAMES
    ]
    candidates.extend(explicit_paths)
    for path in candidates:
        filename = path.name
        if path in seen or not path.exists():
            continue
        seen.add(path)
        if not path.is_file():
            continue
        try:
            text = _read_file_tail(path, limit=_LOCAL_LOG_DIAGNOSTIC_MAX_BYTES).decode(
                "utf-8",
                "replace",
            )
        except OSError as exc:
            diagnostics.append(LocalLogDiagnostic(filename, path, True, error=str(exc)))
            continue
        messages = _diagnostic_log_messages(text)
        diagnostics.append(LocalLogDiagnostic(filename, path, True, messages=messages))
    return tuple(diagnostics)


def _diagnostic_log_messages(text: str) -> tuple[str, ...]:
    return diagnostic_log_messages(text)


def _observe_local_custodian(
    directories: Sequence[Path],
) -> CustodianInterventionEvidence | None:
    for directory in directories:
        path = directory / "custodian.json"
        if not path.exists():
            continue
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > _LOCAL_CUSTODIAN_MAX_BYTES:
                return CustodianInterventionEvidence(
                    source_path=str(path),
                    present=True,
                    error=(
                        f"custodian.json exceeds diagnostic read limit "
                        f"{_LOCAL_CUSTODIAN_MAX_BYTES} bytes"
                    ),
                )
            contents = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return CustodianInterventionEvidence(
                source_path=str(path),
                present=True,
                error=f"custodian.json could not be read: {type(exc).__name__}",
            )
        return parse_custodian_json(
            contents,
            source_path=str(path),
        )
    return None


def _observe_error_archives(directories: Sequence[Path]) -> tuple[LocalFileEvidence, ...]:
    observations: list[LocalFileEvidence] = []
    seen: set[Path] = set()
    for directory in directories:
        for path in sorted(directory.glob("error.*.tar.gz")):
            if path in seen:
                continue
            seen.add(path)
            try:
                observations.append(LocalFileEvidence(path.name, path, path.is_file(), path.stat().st_size))
            except OSError:
                observations.append(LocalFileEvidence(path.name, path, path.is_file(), None))
    return tuple(observations)


def _suggested_diagnostic_checks(
    trajectories: Sequence[StageTrajectoryObservation],
    logs: Sequence[LocalLogDiagnostic],
    custodian: CustodianInterventionEvidence | None,
    archives: Sequence[LocalFileEvidence],
) -> tuple[str, ...]:
    suggestions: list[str] = []
    if any(trajectory.oszicar_present for trajectory in trajectories):
        suggestions.append(
            "Review the OSZICAR trajectory before interpreting unavailable final-result fields."
        )
    if any(trajectory.vasprun_error for trajectory in trajectories):
        suggestions.append(
            "Treat final-result parsing as incomplete because vasprun.xml trajectory enrichment was unavailable."
        )
    if custodian and custodian.present:
        suggestions.append(
            "Review custodian.json for correction attempts and unresolved handler errors."
        )
    if any(log.messages for log in logs):
        suggestions.append(
            "Inspect the bounded fatal/error log excerpts above in the original calculation context."
        )
    if archives:
        suggestions.append(
            "Error tarballs are present but were not unpacked by BMD Agent."
        )
    return tuple(suggestions)


def _unique_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return tuple(unique)


def _read_file_prefix(path: Path, *, limit: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(limit)


def _vasp_parameter_values(parameters: Any) -> Mapping[str, Any]:
    if parameters is None:
        return {}
    try:
        items = dict(parameters).items()
    except Exception:
        try:
            items = parameters.items()
        except Exception:
            return {}
    return {str(key).upper(): _json_safe_value(value) for key, value in items}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _json_equivalent(left: Any, right: Any) -> bool:
    return _json_safe_value(left) == _json_safe_value(right)


def _current_stage_binding(
    current: Path,
    root: Path,
    bindings: Sequence[LocalStageBinding],
) -> LocalStageBinding | None:
    matches = [binding for binding in bindings if _is_relative_to(current, binding.path)]
    if matches:
        return max(matches, key=lambda binding: len(binding.path.parts))
    if current == root:
        with_outputs = [
            binding for binding in bindings
            if _has_meaningful_files(_observe_files(binding.path, _OUTPUT_FILENAMES))
        ]
        if with_outputs:
            return with_outputs[-1]
        return bindings[0] if bindings else None
    return None


def _bounded_ancestors(current: Path, *, max_ancestor_levels: int) -> tuple[Path, ...]:
    roots = [current]
    roots.extend(current.parents[:max_ancestor_levels])
    return tuple(roots)


def _observe_files(directory: Path, filenames: Sequence[str]) -> dict[str, LocalFileEvidence]:
    observations: dict[str, LocalFileEvidence] = {}
    for name in filenames:
        path = directory / name
        if not path.is_file():
            observations[name] = LocalFileEvidence(name, path, False)
            continue
        try:
            observations[name] = LocalFileEvidence(name, path, True, path.stat().st_size)
        except OSError:
            observations[name] = LocalFileEvidence(name, path, True, None)
    return observations


def _observe_bmd_logs(workflow: BmdWorkflowDiscovery) -> dict[str, LocalFileEvidence]:
    paths = _mapping(workflow.submission.get("paths"))
    observations: dict[str, LocalFileEvidence] = {}
    for key in _BMD_LOG_KEYS:
        path = _optional_local_path(paths.get(key), root=workflow.workflow_root)
        if path is None:
            continue
        if not path.is_file():
            observations[key] = LocalFileEvidence(key, path, False)
            continue
        try:
            observations[key] = LocalFileEvidence(key, path, True, path.stat().st_size)
        except OSError:
            observations[key] = LocalFileEvidence(key, path, True, None)
    return observations


def _has_required_inputs(inputs: Mapping[str, LocalFileEvidence]) -> bool:
    return all(inputs.get(name) is not None and inputs[name].non_empty for name in _INPUT_FILENAMES)


def _input_gaps(inputs: Mapping[str, LocalFileEvidence]) -> list[str]:
    gaps: list[str] = []
    for name in _INPUT_FILENAMES:
        observation = inputs.get(name)
        if observation is None or not observation.present:
            gaps.append(f"{name} is missing")
        elif observation.size == 0:
            gaps.append(f"{name} is empty")
        elif observation.size is None:
            gaps.append(f"{name} size could not be established")
    return gaps


def _has_meaningful_files(files: Mapping[str, LocalFileEvidence]) -> bool:
    return any(observation.non_empty for observation in files.values())


def _detect_normal_completion(directory: Path) -> bool:
    outcar = directory / "OUTCAR"
    try:
        if not outcar.is_file() or outcar.stat().st_size <= 0:
            return False
        text = _read_file_tail(outcar).decode("utf-8", "replace")
    except OSError:
        return False
    return any(marker in text for marker in _NORMAL_COMPLETION_MARKERS)


def _read_file_tail(path: Path, *, limit: int = 256_000) -> bytes:
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > limit:
            handle.seek(size - limit)
        return handle.read(limit)


def _structure_from_inputs(inputs: Mapping[str, LocalFileEvidence]) -> StructureInfo | None:
    poscar = inputs.get("POSCAR")
    if poscar is None or not poscar.non_empty:
        return None
    try:
        return parse_poscar(poscar.path.read_bytes(), source=str(poscar.path))
    except Exception:
        return None


def _incar_settings(inputs: Mapping[str, LocalFileEvidence]) -> Mapping[str, Any]:
    incar = inputs.get("INCAR")
    if incar is None or not incar.non_empty:
        return {}
    try:
        settings, error = parse_incar_contents(incar.path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return {} if error else dict(settings)


def _derive_local_scientific(directory: Path, submission: Mapping[str, Any] | None) -> ScientificResult | None:
    keys = {
        "contcar": directory / "CONTCAR",
        "vasprun": directory / "vasprun.xml",
        "kpoints": directory / "KPOINTS",
    }
    local_paths: dict[str, Path] = {}
    for key, path in keys.items():
        try:
            if path.is_file() and path.stat().st_size > 0:
                local_paths[key] = path
        except OSError:
            continue
    if not any(key in local_paths for key in ("contcar", "vasprun")):
        return None
    display_paths = {key: str(path) for key, path in local_paths.items()}
    workflow_spec = _workflow_spec_from_submission(submission)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            scientific = parse_vasp_output_files(local_paths, display_paths, workflow_spec)
    except Exception as exc:
        return ScientificResult(
            source_paths=tuple(display_paths.values()),
            error=str(exc),
        )
    if _has_malformed_xml_warning(caught):
        reason = "vasprun.xml could not be parsed completely"
        unavailable = tuple(
            item for item in scientific.unavailable
            if "xml is malformed" not in item.lower()
        )
        if reason not in unavailable:
            unavailable = unavailable + (reason,)
        return replace(scientific, unavailable=unavailable)
    return scientific


def _has_malformed_xml_warning(caught_warnings: Sequence[warnings.WarningMessage]) -> bool:
    return any("xml is malformed" in str(item.message).lower() for item in caught_warnings)


def _workflow_spec_from_submission(submission: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if submission is None:
        return {
            "stages": [
                {
                    "stage_type": "direct_vasp",
                    "theory": "unknown",
                    "modifiers": [],
                    "label": None,
                    "options": {},
                }
            ]
        }
    return _mapping(_mapping(submission.get("flow_spec")).get("workflow_spec"))


def _lookup_scheduler(
    job_id: str | None,
    scheduler_lookup: SchedulerLookup | None,
) -> tuple[SlurmAccountingRecord | None, str | None]:
    if job_id is None:
        return None, "No valid SLURM job ID was found in inspected producer artifacts."
    if scheduler_lookup is None:
        return None, "scheduler lookup is unavailable"
    try:
        return scheduler_lookup(job_id), None
    except Exception as exc:
        return None, str(exc)


def _is_active_scheduler_state(state: str | None) -> bool:
    if not state:
        return False
    return str(state).upper().split()[0] in _ACTIVE_SCHEDULER_STATES


def _is_inactive_unsuccessful_scheduler_state(scheduler: SlurmAccountingRecord) -> bool:
    state = str(scheduler.state or "").upper().split()[0]
    return state in _INACTIVE_UNSUCCESSFUL_STATES


def _scheduler_success(scheduler: SlurmAccountingRecord | None) -> bool:
    if scheduler is None:
        return False
    return str(scheduler.state).upper().split()[0] == "COMPLETED" and scheduler.exit_code == "0:0"


def _scheduler_nonzero_exit(scheduler: SlurmAccountingRecord | None) -> bool:
    if scheduler is None:
        return False
    return bool(scheduler.exit_code and scheduler.exit_code != "0:0")


def _find_job_id(
    submission: Mapping[str, Any],
    attempt_payload: Mapping[str, Any] | None,
) -> str | None:
    # submission.json is written before sbatch and never carries a job ID;
    # the submission-attempt state record does.
    candidates: list[Any] = []
    if attempt_payload is not None:
        candidates.extend(
            [
                attempt_payload.get("job_id"),
                _mapping(attempt_payload.get("job_record")).get("job_id"),
            ]
        )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return normalize_job_id(str(candidate))
        except ValueError:
            continue
    return None


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _path_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _producer_root_text(paths: Mapping[str, Any]) -> str | None:
    # BMD Compute records the run directory as paths.run_dir.
    path = _path_text(paths.get("run_dir"))
    if path is not None:
        return path
    stage_dirs = paths.get("stage_dirs")
    if isinstance(stage_dirs, Mapping):
        stage_paths = [
            normalized
            for normalized in (_normalize_posix_path_text(value) for value in stage_dirs.values())
            if normalized is not None
        ]
        if len(set(stage_paths)) > 1:
            try:
                return posixpath.commonpath(stage_paths)
            except ValueError:
                pass
    result_dir = _path_text(paths.get("result_dir"))
    if result_dir is not None:
        return result_dir
    if isinstance(stage_dirs, Mapping):
        for value in reversed(tuple(stage_dirs.values())):
            path = _path_text(value)
            if path is not None:
                return path
    return None


def _rebase_producer_path(
    producer_path: str | None,
    producer_root: str,
    acquisition_root: Path,
) -> Path | None:
    normalized_root = _normalize_posix_path_text(producer_root)
    normalized_path = _normalize_posix_path_text(producer_path)
    if normalized_root is None or normalized_path is None:
        return None
    try:
        relative = PurePosixPath(normalized_path).relative_to(PurePosixPath(normalized_root))
    except ValueError:
        return None
    relative_parts = () if str(relative) == "." else relative.parts
    local_path = acquisition_root.joinpath(*relative_parts)
    try:
        resolved = local_path.resolve()
    except OSError:
        resolved = local_path.absolute()
    if not _is_relative_to(resolved, acquisition_root):
        return None
    if not resolved.exists():
        return None
    return resolved


def _normalize_posix_path_text(value: Any) -> str | None:
    path = _path_text(value)
    if path is None:
        return None
    normalized = posixpath.normpath(path)
    if not PurePosixPath(normalized).is_absolute():
        return None
    return normalized


def _optional_local_path(value: Any, *, root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
