from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import PurePosixPath
import subprocess
from typing import Any, Mapping

from bmd_agent.config import SlurmClusterResource
from bmd_agent.resources.compute_records import (
    CONTRACT_V1,
    UNSUPPORTED_LEGACY,
    UNSUPPORTED_VERSION,
    ComputeRecordError,
    JobRecord,
    SubmissionRecord,
    parse_job_record,
    parse_submission_record,
)
from bmd_agent.deployment import DeploymentContext
from bmd_agent.profiling import profile_phase
from bmd_agent.resources.slurm import normalize_job_id
from bmd_agent.resources.vasp import (
    RemoteAcquisitionRequest,
    RemotePathError,
    authorize_remote_path,
    build_remote_file_path,
    prime_remote_acquisition,
    remote_directory_exists,
    remote_file_exists,
    remote_file_size,
    retrieve_remote_file,
)
from bmd_agent.resources.transport import run_invocation


RESOLVED = "resolved"
NOT_BMD_COMPUTE = "not_bmd_compute"
UNAVAILABLE = "unavailable"
AMBIGUOUS = "ambiguous"
INVALID = "invalid"
UNSUPPORTED_RECORD = "unsupported_record"
PRODUCER_PROVENANCE = "producer_provenance"
PRODUCER_NAME = "BMD Compute"
_MAX_PRODUCER_JSON_BYTES = 1024 * 1024


@dataclass(frozen=True)
class JobRunResolution:
    """Producer-owned evidence resolving one SLURM job to one BMD Compute run."""

    scheduler_job_id: str
    resolution_status: str
    producer: str | None = None
    run_directory: str | None = None
    producer_state_path: str | None = None
    submission_attempt_id: str | None = None
    evidence_type: str = PRODUCER_PROVENANCE
    source: str | None = None
    reason: str | None = None
    limitations: tuple[str, ...] = ()
    _submission_payload: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _attempt_payload: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _attempt_checked: bool = field(default=False, repr=False, compare=False)
    job_record_contract: str | None = None
    submission_record_contract: str | None = None


class _ResolutionInvalid(ValueError):
    pass


class _ResolutionUnsupported(ValueError):
    pass


class _ResolutionConflict(ValueError):
    pass


def resolve_bmd_compute_job(
    cluster: SlurmClusterResource,
    deployment: DeploymentContext | None,
    job_id: str,
    *,
    runner=run_invocation,
    timeout: float | None = None,
) -> JobRunResolution:
    """Resolve an exact producer job record without searching remote filesystems."""

    normalized_job_id = normalize_job_id(job_id)
    timeout = cluster.remote_command_timeout_seconds if timeout is None else timeout
    roots = _producer_roots(cluster, deployment, normalized_job_id)
    if isinstance(roots, JobRunResolution):
        return roots
    logs_root, flows_root = roots
    state_path = build_remote_file_path(
        logs_root,
        f"job_{normalized_job_id}.json",
        allowed_roots=cluster.allowed_remote_roots,
    )
    with profile_phase("producer_submission_provenance"):
        prime_remote_acquisition(
            cluster.ssh_host,
            (RemoteAcquisitionRequest(state_path, read_limit=_MAX_PRODUCER_JSON_BYTES),),
            runner=runner,
            timeout=timeout,
        )
        state_present = remote_file_exists(
            cluster.ssh_host,
            state_path,
            runner=runner,
            timeout=timeout,
        )
    if not state_present:
        return JobRunResolution(
            scheduler_job_id=normalized_job_id,
            resolution_status=NOT_BMD_COMPUTE,
            producer_state_path=str(state_path),
            source="exact BMD Compute remote job record",
            reason="no BMD Compute producer state record was found for this job ID",
        )

    try:
        with profile_phase("producer_submission_provenance"):
            state = _read_bounded_json(
                cluster,
                state_path,
                runner=runner,
                timeout=timeout,
            )
        return _validate_resolution(
            cluster,
            normalized_job_id,
            state_path,
            state,
            flows_root=flows_root,
            logs_root=logs_root,
            runner=runner,
            timeout=timeout,
        )
    except _ResolutionConflict as exc:
        return JobRunResolution(
            scheduler_job_id=normalized_job_id,
            resolution_status=AMBIGUOUS,
            producer=PRODUCER_NAME,
            producer_state_path=str(state_path),
            source="exact BMD Compute remote job record",
            reason=str(exc),
        )
    except _ResolutionUnsupported as exc:
        return JobRunResolution(
            scheduler_job_id=normalized_job_id,
            resolution_status=UNSUPPORTED_RECORD,
            producer=PRODUCER_NAME,
            producer_state_path=str(state_path),
            source="exact BMD Compute remote job record",
            reason=str(exc),
        )
    except (_ResolutionInvalid, RemotePathError) as exc:
        return JobRunResolution(
            scheduler_job_id=normalized_job_id,
            resolution_status=INVALID,
            producer=PRODUCER_NAME,
            producer_state_path=str(state_path),
            source="exact BMD Compute remote job record",
            reason=str(exc),
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        return JobRunResolution(
            scheduler_job_id=normalized_job_id,
            resolution_status=UNAVAILABLE,
            producer=PRODUCER_NAME,
            producer_state_path=str(state_path),
            source="exact BMD Compute remote job record",
            reason=f"BMD Compute producer state could not be read: {type(exc).__name__}",
        )


def _producer_roots(
    cluster: SlurmClusterResource,
    deployment: DeploymentContext | None,
    job_id: str,
) -> tuple[PurePosixPath, PurePosixPath] | JobRunResolution:
    if deployment is None or deployment.profile is None:
        return JobRunResolution(
            scheduler_job_id=job_id,
            resolution_status=UNAVAILABLE,
            source="deployment context",
            reason="no deployment profile supplies BMD Compute logs and flows roots",
        )
    if deployment.cluster is None or deployment.cluster.key != cluster.key:
        return JobRunResolution(
            scheduler_job_id=job_id,
            resolution_status=INVALID,
            source="deployment context",
            reason="deployment context does not match the selected cluster resource",
        )
    if deployment.profile.expected.scheduler != "slurm":
        return JobRunResolution(
            scheduler_job_id=job_id,
            resolution_status=UNAVAILABLE,
            source="deployment context",
            reason="configured deployment profile is not a SLURM deployment",
        )
    try:
        logs_root = authorize_remote_path(
            deployment.profile.paths.logs_root,
            allowed_roots=cluster.allowed_remote_roots,
        )
        flows_root = authorize_remote_path(
            deployment.profile.paths.flows_root,
            allowed_roots=cluster.allowed_remote_roots,
        )
    except RemotePathError as exc:
        return JobRunResolution(
            scheduler_job_id=job_id,
            resolution_status=INVALID,
            source="deployment context",
            reason=f"configured BMD producer root is not authorized: {exc}",
        )
    return logs_root, flows_root


def _validate_resolution(
    cluster: SlurmClusterResource,
    job_id: str,
    state_path: PurePosixPath,
    state: Mapping[str, Any],
    *,
    flows_root: PurePosixPath,
    logs_root: PurePosixPath,
    runner,
    timeout: float,
) -> JobRunResolution:
    record = _parse_record(parse_job_record, state, "producer state")
    _require_equal(job_id, _normalized_job_id(record.job_id, "producer state job_id"), "requested and producer-state job IDs")

    run_directory = _authorized_path_value(
        record.run_dir,
        cluster=cluster,
        label="producer state run_dir",
    )
    if not _is_within(run_directory, flows_root):
        raise _ResolutionInvalid("producer state run_dir is outside the configured BMD flows root")

    if record.remote_state_path is not None:
        declared_state_path = _authorized_path_value(
            record.remote_state_path,
            cluster=cluster,
            label="producer state remote_state_path",
        )
        _require_equal(
            str(state_path),
            str(declared_state_path),
            "resolved and producer-declared state paths",
        )

    limitations: list[str] = [
        "producer job state is a submission-time record and is not a scheduler lifecycle record",
        *record.limitations,
    ]
    state_spec = record.embedded_submission
    if record.contract != CONTRACT_V1:
        if not state_spec:
            limitations.append("producer state has no embedded submission snapshot")
        _cross_check_legacy_embedded_submission(
            record,
            state_spec,
            run_directory,
            cluster=cluster,
        )

    submission_path = build_remote_file_path(
        run_directory,
        "submission.json",
        allowed_roots=cluster.allowed_remote_roots,
    )
    with profile_phase("producer_submission_provenance"):
        prime_remote_acquisition(
            cluster.ssh_host,
            (
                RemoteAcquisitionRequest(run_directory, kind="directory"),
                RemoteAcquisitionRequest(
                    submission_path,
                    read_limit=_MAX_PRODUCER_JSON_BYTES,
                ),
            ),
            runner=runner,
            timeout=timeout,
        )
        run_directory_present = remote_directory_exists(
            cluster.ssh_host,
            run_directory,
            runner=runner,
            timeout=timeout,
        )
    if not run_directory_present:
        return JobRunResolution(
            scheduler_job_id=job_id,
            resolution_status=UNAVAILABLE,
            producer=PRODUCER_NAME,
            run_directory=str(run_directory),
            producer_state_path=str(state_path),
            submission_attempt_id=record.attempt_id,
            source="exact BMD Compute remote job record",
            reason="producer-resolved run directory is not a readable directory",
            limitations=tuple(limitations),
        )

    with profile_phase("producer_submission_provenance"):
        submission_present = remote_file_exists(
            cluster.ssh_host,
            submission_path,
            runner=runner,
            timeout=timeout,
        )
    if not submission_present:
        return JobRunResolution(
            scheduler_job_id=job_id,
            resolution_status=UNAVAILABLE,
            producer=PRODUCER_NAME,
            run_directory=str(run_directory),
            producer_state_path=str(state_path),
            submission_attempt_id=record.attempt_id,
            source="exact BMD Compute remote job record",
            reason="producer-resolved run directory has no readable submission.json",
            limitations=tuple(limitations),
        )

    with profile_phase("producer_submission_provenance"):
        submission = _read_bounded_json(
            cluster,
            submission_path,
            runner=runner,
            timeout=timeout,
        )
    submission_record = _parse_record(parse_submission_record, submission, "submission.json")
    limitations.extend(submission_record.limitations)
    _cross_check_submission(
        record,
        submission_record,
        run_directory,
        cluster=cluster,
    )

    attempt_id = record.attempt_id or submission_record.attempt_id
    attempt_path = (
        _authorized_path_value(
            submission_record.attempt_state,
            cluster=cluster,
            label="submission attempt_state",
        )
        if submission_record.attempt_state is not None
        else None
    )
    if record.contract != CONTRACT_V1:
        legacy_state_attempt = _embedded_attempt_state(state_spec, cluster=cluster)
        if legacy_state_attempt is not None and attempt_path is not None:
            _require_equal(
                str(legacy_state_attempt),
                str(attempt_path),
                "producer-state and submission attempt-state paths",
            )
    attempt_payload: Mapping[str, Any] | None = None
    if attempt_path is None:
        limitations.append("submission attempt-state path is unavailable")
    elif not _is_within(attempt_path, logs_root):
        raise _ResolutionInvalid(
            "submission attempt-state path is outside the configured BMD logs root"
        )
    else:
        with profile_phase("producer_submission_provenance"):
            prime_remote_acquisition(
                cluster.ssh_host,
                (
                    RemoteAcquisitionRequest(
                        attempt_path,
                        read_limit=_MAX_PRODUCER_JSON_BYTES,
                    ),
                ),
                runner=runner,
                timeout=timeout,
            )
            attempt_present = remote_file_exists(
                cluster.ssh_host,
                attempt_path,
                runner=runner,
                timeout=timeout,
            )
        if not attempt_present:
            limitations.append("submission attempt-state record is unavailable")
        else:
            with profile_phase("producer_submission_provenance"):
                attempt_payload = _read_bounded_json(
                    cluster,
                    attempt_path,
                    runner=runner,
                    timeout=timeout,
                )
            attempt_id = _cross_check_attempt(
                attempt_payload,
                record,
                submission_record,
                run_directory,
                job_id,
                cluster=cluster,
            )

    return JobRunResolution(
        scheduler_job_id=job_id,
        resolution_status=RESOLVED,
        producer=PRODUCER_NAME,
        run_directory=str(run_directory),
        producer_state_path=str(state_path),
        submission_attempt_id=attempt_id,
        source="exact BMD Compute remote job record cross-checked against submission provenance",
        limitations=tuple(limitations),
        _submission_payload=submission,
        _attempt_payload=attempt_payload,
        _attempt_checked=True,
        job_record_contract=record.label,
        submission_record_contract=submission_record.label,
    )


def _read_bounded_json(
    cluster: SlurmClusterResource,
    path: PurePosixPath,
    *,
    runner,
    timeout: float,
) -> Mapping[str, Any]:
    size = remote_file_size(
        cluster.ssh_host,
        path,
        runner=runner,
        timeout=timeout,
    )
    if size > _MAX_PRODUCER_JSON_BYTES:
        raise _ResolutionInvalid(
            f"producer JSON exceeds the {_MAX_PRODUCER_JSON_BYTES}-byte read limit"
        )
    contents = retrieve_remote_file(
        cluster.ssh_host,
        path,
        runner=runner,
        timeout=timeout,
    )
    try:
        payload = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _ResolutionInvalid(f"malformed JSON in {path}") from exc
    if not isinstance(payload, Mapping):
        raise _ResolutionInvalid(f"{path} must contain a JSON object")
    return payload


def _parse_record(parser, payload: Mapping[str, Any], label: str):
    try:
        return parser(payload)
    except ComputeRecordError as exc:
        if exc.kind in {UNSUPPORTED_VERSION, UNSUPPORTED_LEGACY}:
            raise _ResolutionUnsupported(f"{label}: {exc}") from exc
        raise _ResolutionInvalid(f"{label}: {exc}") from exc


def _cross_check_legacy_embedded_submission(
    record: JobRecord,
    state_spec: Mapping[str, Any],
    run_directory: PurePosixPath,
    *,
    cluster: SlurmClusterResource,
) -> None:
    """Legacy job records embed a submission snapshot; it must agree."""

    spec_run_dir = _optional_mapping(state_spec, "paths").get("run_dir")
    if spec_run_dir is not None:
        authorized = _authorized_path_value(
            spec_run_dir,
            cluster=cluster,
            label="producer state submission_spec paths.run_dir",
        )
        _require_equal(str(run_directory), str(authorized), "producer-state run directories")
    spec_name = state_spec.get("run_name")
    if spec_name is not None:
        _require_equal(record.run_name, str(spec_name), "producer-state run names")


def _cross_check_submission(
    record: JobRecord,
    submission: SubmissionRecord,
    run_directory: PurePosixPath,
    *,
    cluster: SlurmClusterResource,
) -> None:
    if submission.run_dir is None:
        raise _ResolutionInvalid("submission paths.run_dir must be a non-empty string")
    authorized = _authorized_path_value(
        submission.run_dir,
        cluster=cluster,
        label="submission paths.run_dir",
    )
    _require_equal(str(run_directory), str(authorized), "producer-state and submission run directories")
    if submission.run_name is not None:
        _require_equal(record.run_name, submission.run_name, "producer-state and submission run names")
    if record.attempt_id and submission.attempt_id:
        _require_equal(record.attempt_id, submission.attempt_id, "producer-state and submission attempt IDs")


def _cross_check_attempt(
    attempt: Mapping[str, Any],
    record: JobRecord,
    submission: SubmissionRecord,
    run_directory: PurePosixPath,
    job_id: str,
    *,
    cluster: SlurmClusterResource,
) -> str | None:
    for label, candidate in _job_id_candidates(attempt):
        _require_equal(job_id, _normalized_job_id(candidate, label), f"requested and {label} job IDs")

    attempt_run_dir = attempt.get("run_dir")
    if attempt_run_dir is not None:
        authorized = _authorized_path_value(
            attempt_run_dir,
            cluster=cluster,
            label="attempt run_dir",
        )
        _require_equal(str(run_directory), str(authorized), "producer-state and attempt run directories")
    attempt_record = _optional_mapping(attempt, "job_record")
    record_run_dir = attempt_record.get("run_dir")
    if record_run_dir is not None:
        authorized = _authorized_path_value(
            record_run_dir,
            cluster=cluster,
            label="attempt job_record.run_dir",
        )
        _require_equal(str(run_directory), str(authorized), "producer-state and attempt job-record run directories")
    record_spec = _optional_mapping(attempt_record, "submission_spec")
    record_spec_run_dir = _optional_mapping(record_spec, "paths").get("run_dir")
    if record_spec_run_dir is not None:
        authorized = _authorized_path_value(
            record_spec_run_dir,
            cluster=cluster,
            label="attempt job_record.submission_spec paths.run_dir",
        )
        _require_equal(
            str(run_directory),
            str(authorized),
            "producer-state and attempt job-record submission run directories",
        )

    ids = [
        value
        for value in (
            record.attempt_id,
            submission.attempt_id,
            _string_or_none(attempt.get("attempt_id")),
            _attempt_id(record_spec),
        )
        if value is not None
    ]
    if len(set(ids)) > 1:
        raise _ResolutionConflict("submission attempt IDs conflict across producer artifacts")
    return ids[0] if ids else None


def _embedded_attempt_state(
    state_spec: Mapping[str, Any],
    *,
    cluster: SlurmClusterResource,
) -> PurePosixPath | None:
    submission_block = _optional_mapping(state_spec, "submission")
    paths = _optional_mapping(state_spec, "paths")
    values = tuple(
        value
        for value in (
            submission_block.get("attempt_state"),
            paths.get("submission_attempt_state"),
        )
        if value is not None
    )
    if not values:
        return None
    authorized = tuple(
        _authorized_path_value(
            value,
            cluster=cluster,
            label="producer state submission attempt_state",
        )
        for value in values
    )
    if any(path != authorized[0] for path in authorized[1:]):
        raise _ResolutionConflict("submission attempt-state paths conflict")
    return authorized[0]


def _required_authorized_path(
    payload: Mapping[str, Any],
    key: str,
    *,
    cluster: SlurmClusterResource,
    label: str,
) -> PurePosixPath:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise _ResolutionInvalid(f"{label} must be a non-empty string")
    return _authorized_path_value(value, cluster=cluster, label=label)


def _authorized_path_value(
    value: Any,
    *,
    cluster: SlurmClusterResource,
    label: str,
) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise _ResolutionInvalid(f"{label} must be a non-empty string")
    try:
        return authorize_remote_path(value, allowed_roots=cluster.allowed_remote_roots)
    except RemotePathError as exc:
        raise _ResolutionInvalid(f"{label} is not authorized: {exc}") from exc


def _normalized_job_id(value: Any, label: str) -> str:
    if value is None:
        raise _ResolutionInvalid(f"{label} is missing")
    try:
        return normalize_job_id(str(value))
    except ValueError as exc:
        raise _ResolutionInvalid(f"{label} is invalid") from exc


def _job_id_candidates(attempt: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Job IDs BMD Compute writes in a submission-attempt state record."""

    candidates: list[tuple[str, Any]] = []
    if attempt.get("job_id") is not None:
        candidates.append(("attempt job_id", attempt["job_id"]))
    record = _optional_mapping(attempt, "job_record")
    if record.get("job_id") is not None:
        candidates.append(("attempt job_record.job_id", record["job_id"]))
    return tuple(candidates)


def _attempt_id(payload: Mapping[str, Any]) -> str | None:
    return _string_or_none(_optional_mapping(payload, "submission").get("attempt_id"))


def _optional_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _require_equal(left: str, right: str, label: str) -> None:
    if left != right:
        raise _ResolutionConflict(f"{label} conflict")


def _is_within(path: PurePosixPath, root: PurePosixPath) -> bool:
    return path == root or path.parts[: len(root.parts)] == root.parts
