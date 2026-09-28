"""Readers for BMD Compute run records.

BMD Compute writes two records that Agent uses to locate and describe a run:

* ``<run_dir>/submission.json`` -- ``bmd_compute.submission``: the Prepare-time
  submission specification (what Compute prepared and asked to run);
* ``<logs_dir>/job_<JOB_ID>.json`` -- ``bmd_compute.job_record``: a
  run-resolution record linking a scheduler job ID to a run.

BMD Compute owns both contracts (``backend/run_records.py`` and
``docs/run_records.md`` in bmd_compute). Records written before the contracts
existed have no ``schema`` and are read as legacy unversioned records.

Neither record is scheduler lifecycle authority. Their status/state fields
describe Compute's own preparation or submission step and are never read
here; SLURM accounting and VASP artifacts decide what executed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from typing import Any


SUBMISSION_RECORD_SCHEMA = "bmd_compute.submission"
JOB_RECORD_SCHEMA = "bmd_compute.job_record"
SUPPORTED_SCHEMA_VERSION = 1

CONTRACT_V1 = "v1"
CONTRACT_LEGACY = "legacy_unversioned"

INVALID_RECORD = "invalid"
UNSUPPORTED_VERSION = "unsupported_version"
UNSUPPORTED_LEGACY = "unsupported_legacy"

# Compute-defined stage identifiers: stage_NN or relax_NN name the 1-based
# index in flow_spec.workflow_spec.stages. Key order is never used.
STAGE_DIRECTORY_ID_PATTERN = re.compile(r"^(?:stage|relax)_(\d{2,})$")
LOG_PATH_KEYS = ("log_out", "log_err", "slurm_out", "slurm_err")

LEGACY_SUBMISSION_LIMITATION = (
    "submission.json is a legacy unversioned BMD Compute record; it was read with the "
    "legacy reader and has no declared contract version"
)
LEGACY_JOB_RECORD_LIMITATION = (
    "job record is a legacy unversioned BMD Compute record; it was read with the "
    "legacy reader and has no declared contract version"
)


class ComputeRecordError(ValueError):
    """A Compute record could not be read under any supported contract."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class SubmissionRecord:
    """Contractual view of a BMD Compute submission specification."""

    contract: str
    workflow_stages: tuple[Mapping[str, Any], ...]
    stage_dirs: tuple[tuple[int, str, str], ...]
    result_dir: str
    run_dir: str | None
    run_name: str | None
    attempt_id: str | None
    attempt_state: str | None
    log_paths: Mapping[str, str]
    structure: Mapping[str, Any] | None
    cluster: Mapping[str, Any]
    resources: Mapping[str, Any]
    environment: Mapping[str, Any]
    provenance: Mapping[str, Any]
    limitations: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return describe_contract(SUBMISSION_RECORD_SCHEMA, self.contract)


@dataclass(frozen=True)
class JobRecord:
    """Contractual view of a BMD Compute job (run-resolution) record."""

    contract: str
    job_id: str
    run_name: str
    run_dir: str
    attempt_id: str | None
    remote_state_path: str | None = None
    embedded_submission: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return describe_contract(JOB_RECORD_SCHEMA, self.contract)


def describe_contract(schema: str, contract: str) -> str:
    if contract == CONTRACT_V1:
        return f"{schema} v{SUPPORTED_SCHEMA_VERSION}"
    return f"{schema} (legacy unversioned)"


def record_contract(payload: Any, schema: str) -> str:
    """Classify a record envelope as v1 or legacy, or raise ComputeRecordError."""

    if not isinstance(payload, Mapping):
        raise ComputeRecordError(f"{schema} record must be a JSON object", kind=INVALID_RECORD)
    if "schema" not in payload:
        if "schema_version" in payload:
            raise ComputeRecordError(
                f"{schema} record has schema_version without schema",
                kind=INVALID_RECORD,
            )
        return CONTRACT_LEGACY
    declared = payload.get("schema")
    if declared != schema:
        raise ComputeRecordError(
            f"record declares schema {declared!r}, expected {schema!r}",
            kind=INVALID_RECORD,
        )
    version = payload.get("schema_version")
    if not _strict_int(version):
        raise ComputeRecordError(
            f"{schema} record schema_version must be an integer",
            kind=INVALID_RECORD,
        )
    if version != SUPPORTED_SCHEMA_VERSION:
        raise ComputeRecordError(
            f"{schema} schema_version {version} is not supported by this BMD Agent "
            f"(supported: {SUPPORTED_SCHEMA_VERSION})",
            kind=UNSUPPORTED_VERSION,
        )
    return CONTRACT_V1


def parse_submission_record(payload: Any) -> SubmissionRecord:
    contract = record_contract(payload, SUBMISSION_RECORD_SCHEMA)
    v1 = contract == CONTRACT_V1

    flow_spec = _mapping(payload.get("flow_spec"), "submission flow_spec")
    if "workflow_spec" not in flow_spec and not v1:
        raise ComputeRecordError(
            "unsupported legacy BMD Compute submission record: it has no "
            "flow_spec.workflow_spec (written before 2026-08-11), and workflow stages "
            "are not inferred from flow_spec.workflow",
            kind=UNSUPPORTED_LEGACY,
        )
    workflow_spec = _mapping(flow_spec.get("workflow_spec"), "submission flow_spec.workflow_spec")
    stages = _parse_stages(workflow_spec.get("stages"))

    paths = _mapping(payload.get("paths"), "submission paths")
    result_dir = _text(paths.get("result_dir"), "submission paths.result_dir")
    stage_dirs = _parse_stage_dirs(paths.get("stage_dirs"), len(stages), required=v1)
    log_paths = {
        key: _text(paths[key], f"submission paths.{key}")
        for key in LOG_PATH_KEYS
        if paths.get(key) is not None
    }

    submission_block = payload.get("submission")
    if v1:
        submission_block = _mapping(submission_block, "submission block")
        attempt_id = _text(submission_block.get("attempt_id"), "submission.attempt_id")
        attempt_state = _text(submission_block.get("attempt_state"), "submission.attempt_state")
        run_dir = _text(paths.get("run_dir"), "submission paths.run_dir")
        run_name = _text(payload.get("run_name"), "submission run_name")
        _text(payload.get("created_at"), "submission created_at")
        cluster = _mapping(payload.get("cluster"), "submission cluster")
        resources = _mapping(payload.get("resources"), "submission resources")
        for key in ("partition", "account"):
            _text(cluster.get(key), f"submission cluster.{key}")
        for key in ("nodes", "ntasks", "mem_gb"):
            if not _strict_int(resources.get(key)) or resources[key] < 1:
                raise ComputeRecordError(
                    f"submission resources.{key} must be a positive integer",
                    kind=INVALID_RECORD,
                )
        _text(resources.get("walltime"), "submission resources.walltime")
    else:
        submission_block = submission_block if isinstance(submission_block, Mapping) else {}
        attempt_id = _optional_text(submission_block.get("attempt_id"), "submission.attempt_id")
        attempt_state = _legacy_attempt_state(submission_block, paths)
        run_dir = _optional_text(paths.get("run_dir"), "submission paths.run_dir")
        run_name = _optional_text(payload.get("run_name"), "submission run_name")
        cluster = payload.get("cluster") if isinstance(payload.get("cluster"), Mapping) else {}
        resources = payload.get("resources") if isinstance(payload.get("resources"), Mapping) else {}

    structure = flow_spec.get("structure")
    if structure is not None and not isinstance(structure, Mapping):
        if v1:
            raise ComputeRecordError("submission flow_spec.structure must be an object", kind=INVALID_RECORD)
        structure = None

    environment = payload.get("environment") if isinstance(payload.get("environment"), Mapping) else {}
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), Mapping) else {}

    return SubmissionRecord(
        contract=contract,
        workflow_stages=stages,
        stage_dirs=stage_dirs,
        result_dir=result_dir,
        run_dir=run_dir,
        run_name=run_name,
        attempt_id=attempt_id,
        attempt_state=attempt_state,
        log_paths=log_paths,
        structure=structure,
        cluster=dict(cluster),
        resources=dict(resources),
        environment=dict(environment),
        provenance=dict(provenance),
        limitations=() if v1 else (LEGACY_SUBMISSION_LIMITATION,),
    )


def parse_job_record(payload: Any) -> JobRecord:
    contract = record_contract(payload, JOB_RECORD_SCHEMA)
    job_id = _job_id(payload.get("job_id"))
    run_name = _text(payload.get("run_name"), "job record run_name")
    run_dir = _text(payload.get("run_dir"), "job record run_dir")
    if contract == CONTRACT_V1:
        attempt_id = _text(payload.get("attempt_id"), "job record attempt_id")
        submitted_at = payload.get("submitted_at")
        if submitted_at is not None and not isinstance(submitted_at, str):
            raise ComputeRecordError("job record submitted_at must be a string", kind=INVALID_RECORD)
        # Everything else in a v1 job record is non-contractual and ignored.
        return JobRecord(
            contract=contract,
            job_id=job_id,
            run_name=run_name,
            run_dir=run_dir,
            attempt_id=attempt_id,
        )

    remote_state_path = payload.get("remote_state_path")
    if remote_state_path is not None:
        remote_state_path = _text(remote_state_path, "job record remote_state_path")
    embedded = payload.get("submission_spec")
    if embedded is not None and not isinstance(embedded, Mapping):
        raise ComputeRecordError("job record submission_spec must be a JSON object", kind=INVALID_RECORD)
    embedded = embedded or {}
    embedded_submission = embedded.get("submission") if isinstance(embedded.get("submission"), Mapping) else {}
    return JobRecord(
        contract=contract,
        job_id=job_id,
        run_name=run_name,
        run_dir=run_dir,
        attempt_id=_optional_text(embedded_submission.get("attempt_id"), "job record submission_spec attempt_id"),
        remote_state_path=remote_state_path,
        embedded_submission=embedded,
        limitations=(LEGACY_JOB_RECORD_LIMITATION,),
    )


def stage_index_from_directory_id(identifier: Any) -> int | None:
    if not isinstance(identifier, str):
        return None
    match = STAGE_DIRECTORY_ID_PATTERN.match(identifier)
    if match is None:
        return None
    index = int(match.group(1))
    return index if index >= 1 else None


def _parse_stage_dirs(value: Any, stage_count: int, *, required: bool) -> tuple[tuple[int, str, str], ...]:
    if value is None:
        if required:
            raise ComputeRecordError("submission paths.stage_dirs is required", kind=INVALID_RECORD)
        if stage_count == 1:
            return ()
        raise ComputeRecordError(
            "submission paths.stage_dirs is required for multi-stage runs",
            kind=INVALID_RECORD,
        )
    stage_dirs = _mapping(value, "submission paths.stage_dirs")
    mapped: dict[int, tuple[str, str]] = {}
    for identifier, path in stage_dirs.items():
        index = stage_index_from_directory_id(identifier)
        if index is None or index > stage_count:
            raise ComputeRecordError(
                f"submission paths.stage_dirs identifier {identifier!r} does not name a workflow stage",
                kind=INVALID_RECORD,
            )
        if index in mapped:
            raise ComputeRecordError(
                f"submission paths.stage_dirs names stage {index} more than once",
                kind=INVALID_RECORD,
            )
        mapped[index] = (identifier, _text(path, f"submission paths.stage_dirs.{identifier}"))
    if stage_count > 1 and set(mapped) != set(range(1, stage_count + 1)):
        raise ComputeRecordError(
            "submission paths.stage_dirs must name every stage of a multi-stage workflow",
            kind=INVALID_RECORD,
        )
    return tuple((index, *mapped[index]) for index in sorted(mapped))


def _parse_stages(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ComputeRecordError(
            "submission flow_spec.workflow_spec.stages must be a non-empty list",
            kind=INVALID_RECORD,
        )
    stages = []
    for position, stage in enumerate(value, start=1):
        label = f"submission workflow stage {position}"
        stage = _mapping(stage, label)
        _text(stage.get("stage_type"), f"{label} stage_type")
        _text(stage.get("theory"), f"{label} theory")
        modifiers = stage.get("modifiers", [])
        if modifiers is not None and (
            not isinstance(modifiers, list) or not all(isinstance(item, str) for item in modifiers)
        ):
            raise ComputeRecordError(f"{label} modifiers must be a list of strings", kind=INVALID_RECORD)
        options = stage.get("options", {})
        if options is not None and not isinstance(options, Mapping):
            raise ComputeRecordError(f"{label} options must be a JSON object", kind=INVALID_RECORD)
        if stage.get("label") is not None and not isinstance(stage.get("label"), str):
            raise ComputeRecordError(f"{label} label must be a string or null", kind=INVALID_RECORD)
        stages.append(stage)
    return tuple(stages)


def _legacy_attempt_state(submission_block: Mapping[str, Any], paths: Mapping[str, Any]) -> str | None:
    values = [
        _text(value, "submission attempt_state")
        for value in (submission_block.get("attempt_state"), paths.get("submission_attempt_state"))
        if value is not None
    ]
    if len(set(values)) > 1:
        raise ComputeRecordError("submission attempt-state paths conflict", kind=INVALID_RECORD)
    return values[0] if values else None


def _job_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value).strip() == "":
        raise ComputeRecordError("job record job_id must be a non-empty string", kind=INVALID_RECORD)
    return str(value).strip()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeRecordError(f"{label} must be a JSON object", kind=INVALID_RECORD)
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComputeRecordError(f"{label} must be a non-empty string", kind=INVALID_RECORD)
    return value


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
