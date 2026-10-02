"""Readers for BMD Compute run records.

BMD Compute writes two records that Agent uses to locate and describe a run:

* ``<run_dir>/submission.json`` -- ``bmd_compute.submission``: the Prepare-time
  submission specification (what Compute prepared and asked to run);
* ``<logs_dir>/job_<JOB_ID>.json`` -- ``bmd_compute.job_record``: a
  run-resolution record linking a scheduler job ID to a run;
* ``<run_dir>/runtime_environment.json`` -- ``bmd_compute.runtime_environment``:
  written by the POWER runner before the workflow is built or VASP starts,
  recording Compute's own runtime-parity verdict (``passed``/``failed``).

The ``provenance`` block inside ``submission.json`` is versioned separately by
its own ``schema_version`` (Compute ``docs/provenance.md``).

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

RUNTIME_ENVIRONMENT_SCHEMA = "bmd_compute.runtime_environment"
RUNTIME_ENVIRONMENT_FILENAME = "runtime_environment.json"
RUNTIME_PARITY_PASSED = "passed"
RUNTIME_PARITY_FAILED = "failed"
_RUNTIME_PARITY_STATUSES = (RUNTIME_PARITY_PASSED, RUNTIME_PARITY_FAILED)
_RUNTIME_ENVIRONMENT_KNOWN_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "recorded_at",
        "status",
        "problems",
        "run_name",
        "attempt_id",
        "slurm_job_id",
        "host",
        "python",
        "parity_policy",
        "prepared_packages",
        "runtime_packages",
        "supporting_packages",
        "atomate2_settings",
    }
)

PROVENANCE_SUPPORTED_SCHEMA_VERSION = 1
PROVENANCE_V1 = "v1"
PROVENANCE_ABSENT = "absent"
PROVENANCE_LEGACY = "legacy_unversioned"
PROVENANCE_UNSUPPORTED = "unsupported"

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
    # Additive bmd_compute.submission v1 fields (absent on runs prepared before
    # Compute recorded runtime parity).
    runtime_environment_path: str | None = None
    runtime_parity: Mapping[str, Any] | None = None
    # How the nested provenance block was read; ``provenance`` is empty unless
    # this is PROVENANCE_V1 or PROVENANCE_LEGACY.
    provenance_contract: str = PROVENANCE_ABSENT

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
    provenance_contract, provenance, provenance_limitations = read_submission_provenance(payload)
    if not v1 and provenance_contract == PROVENANCE_LEGACY:
        # A legacy submission's unversioned provenance is already covered by
        # the legacy-record limitation.
        provenance_limitations = ()

    runtime_environment_path = paths.get("runtime_environment")
    if runtime_environment_path is not None:
        runtime_environment_path = _text(runtime_environment_path, "submission paths.runtime_environment")
    runtime_parity = payload.get("runtime_parity")
    if runtime_parity is not None and not isinstance(runtime_parity, Mapping):
        if v1:
            raise ComputeRecordError("submission runtime_parity must be a JSON object", kind=INVALID_RECORD)
        runtime_parity = None

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
        limitations=(() if v1 else (LEGACY_SUBMISSION_LIMITATION,)) + provenance_limitations,
        runtime_environment_path=runtime_environment_path,
        runtime_parity=dict(runtime_parity) if runtime_parity is not None else None,
        provenance_contract=provenance_contract,
    )


def read_submission_provenance(
    payload: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any], tuple[str, ...]]:
    """Return how the submission ``provenance`` block may be interpreted.

    The block is versioned by its own ``schema_version``. Version 1 is read
    intentionally and tolerates additive fields. A block without
    ``schema_version`` is a legacy record and is read best-effort. Any other
    version is not interpreted at all, so nested fields are never silently
    misread under v1 assumptions; the returned mapping is then empty.
    """

    provenance = payload.get("provenance")
    if provenance is None:
        return PROVENANCE_ABSENT, {}, ()
    if not isinstance(provenance, Mapping):
        return (
            PROVENANCE_UNSUPPORTED,
            {},
            ("submission provenance is not a JSON object; producer provenance was not interpreted",),
        )
    if "schema_version" not in provenance:
        return (
            PROVENANCE_LEGACY,
            dict(provenance),
            ("submission provenance has no schema_version; it was read as a legacy record",),
        )
    version = provenance.get("schema_version")
    if _strict_int(version) and version == PROVENANCE_SUPPORTED_SCHEMA_VERSION:
        return PROVENANCE_V1, dict(provenance), ()
    return (
        PROVENANCE_UNSUPPORTED,
        {},
        (
            f"submission provenance schema_version {version!r} is not supported by this "
            f"BMD Agent (supported: {PROVENANCE_SUPPORTED_SCHEMA_VERSION}); producer "
            "provenance (Git source, Custodian policy, automatic treatments) was not interpreted",
        ),
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


@dataclass(frozen=True)
class RuntimeEnvironmentRecord:
    """Contractual view of ``bmd_compute.runtime_environment`` v1.

    ``status`` is BMD Compute's own runtime-parity verdict. Agent reports it
    and never recomputes it from the package versions.
    """

    schema_version: int
    status: str
    problems: tuple[str, ...]
    run_name: str | None = None
    attempt_id: str | None = None
    slurm_job_id: str | None = None
    host: str | None = None
    recorded_at: str | None = None
    python: Mapping[str, Any] = field(default_factory=dict)
    parity_policy: Mapping[str, Any] = field(default_factory=dict)
    prepared_packages: Mapping[str, str | None] | None = None
    runtime_packages: Mapping[str, str | None] = field(default_factory=dict)
    supporting_packages: Mapping[str, str | None] = field(default_factory=dict)
    atomate2_settings: Mapping[str, Any] = field(default_factory=dict)
    # Fields added to the record after this reader was written, preserved
    # verbatim rather than rejected.
    additional_fields: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == RUNTIME_PARITY_PASSED

    @property
    def label(self) -> str:
        return f"{RUNTIME_ENVIRONMENT_SCHEMA} v{self.schema_version}"


def parse_runtime_environment_record(payload: Any) -> RuntimeEnvironmentRecord:
    """Validate a runtime-environment record, or raise ComputeRecordError."""

    if not isinstance(payload, Mapping):
        raise ComputeRecordError(
            f"{RUNTIME_ENVIRONMENT_SCHEMA} record must be a JSON object",
            kind=INVALID_RECORD,
        )
    if "schema" not in payload:
        # The record was introduced with its schema; there is no legacy form.
        raise ComputeRecordError(
            f"runtime environment record has no schema (expected {RUNTIME_ENVIRONMENT_SCHEMA!r})",
            kind=INVALID_RECORD,
        )
    declared = payload.get("schema")
    if declared != RUNTIME_ENVIRONMENT_SCHEMA:
        raise ComputeRecordError(
            f"record declares schema {declared!r}, expected {RUNTIME_ENVIRONMENT_SCHEMA!r}",
            kind=INVALID_RECORD,
        )
    version = payload.get("schema_version")
    if not _strict_int(version):
        raise ComputeRecordError(
            f"{RUNTIME_ENVIRONMENT_SCHEMA} record schema_version must be an integer",
            kind=INVALID_RECORD,
        )
    if version != SUPPORTED_SCHEMA_VERSION:
        raise ComputeRecordError(
            f"{RUNTIME_ENVIRONMENT_SCHEMA} schema_version {version} is not supported by this "
            f"BMD Agent (supported: {SUPPORTED_SCHEMA_VERSION})",
            kind=UNSUPPORTED_VERSION,
        )
    status = payload.get("status")
    if status not in _RUNTIME_PARITY_STATUSES:
        raise ComputeRecordError(
            f"runtime environment status {status!r} is not a v1 value (passed or failed)",
            kind=INVALID_RECORD,
        )
    problems = payload.get("problems")
    if not isinstance(problems, list) or not all(isinstance(item, str) for item in problems):
        raise ComputeRecordError(
            "runtime environment problems must be a list of strings",
            kind=INVALID_RECORD,
        )
    if status == RUNTIME_PARITY_PASSED and problems:
        # Never read a pass from a record that also reports problems.
        raise ComputeRecordError(
            "runtime environment record says passed but lists problems",
            kind=INVALID_RECORD,
        )
    prepared = payload.get("prepared_packages")
    return RuntimeEnvironmentRecord(
        schema_version=version,
        status=status,
        problems=tuple(problems),
        run_name=_optional_record_text(payload, "run_name"),
        attempt_id=_optional_record_text(payload, "attempt_id"),
        slurm_job_id=_optional_record_text(payload, "slurm_job_id"),
        host=_optional_record_text(payload, "host"),
        recorded_at=_optional_record_text(payload, "recorded_at"),
        python=_optional_record_mapping(payload, "python"),
        parity_policy=_optional_record_mapping(payload, "parity_policy"),
        prepared_packages=(
            None if prepared is None else _package_versions(prepared, "prepared_packages")
        ),
        runtime_packages=_package_versions(payload.get("runtime_packages") or {}, "runtime_packages"),
        supporting_packages=_package_versions(
            payload.get("supporting_packages") or {},
            "supporting_packages",
        ),
        atomate2_settings=_optional_record_mapping(payload, "atomate2_settings"),
        additional_fields={
            key: value
            for key, value in payload.items()
            if key not in _RUNTIME_ENVIRONMENT_KNOWN_FIELDS
        },
    )


def _optional_record_text(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ComputeRecordError(f"runtime environment {key} must be a string", kind=INVALID_RECORD)
    return str(value)


def _optional_record_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ComputeRecordError(f"runtime environment {key} must be a JSON object", kind=INVALID_RECORD)
    return dict(value)


def _package_versions(value: Any, key: str) -> Mapping[str, str | None]:
    if not isinstance(value, Mapping):
        raise ComputeRecordError(f"runtime environment {key} must be a JSON object", kind=INVALID_RECORD)
    for name, version in value.items():
        if version is not None and not isinstance(version, str):
            raise ComputeRecordError(
                f"runtime environment {key}[{name!r}] must be a version string or null",
                kind=INVALID_RECORD,
            )
    return dict(value)


AUTOMATIC_TREATMENTS_RECORDED = "recorded"
AUTOMATIC_TREATMENTS_NONE = "none_recorded"
AUTOMATIC_TREATMENTS_NOT_RECORDED = "not_recorded"
AUTOMATIC_TREATMENTS_UNAVAILABLE = "unavailable"
DESIRED_OUTPUT_MODE = "bmd_managed_desired_output"


@dataclass(frozen=True)
class AppliedTreatment:
    modifier: str
    display_name: str | None
    stage_indices: tuple[int, ...]
    consideration_id: str | None = None


@dataclass(frozen=True)
class AutomaticTreatmentsObservation:
    """BMD Compute's automatic-treatment record from ``provenance.execution``.

    Compute writes this record for BMD-managed Desired Output workflows. Agent
    reports what the record says; it never infers a treatment (for example
    automatic DFT+U) from stage modifiers.
    """

    status: str
    mode: str | None = None
    desired_output: str | None = None
    applied: tuple[AppliedTreatment, ...] = ()
    not_applicable: tuple[Mapping[str, Any], ...] = ()
    advisory_consideration_ids: tuple[str, ...] = ()
    dft_u: Mapping[str, Any] | None = None
    reason: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def desired_output_managed(self) -> bool:
        return self.status == AUTOMATIC_TREATMENTS_RECORDED and self.mode == DESIRED_OUTPUT_MODE


def read_automatic_treatments(record: SubmissionRecord) -> AutomaticTreatmentsObservation:
    if record.provenance_contract == PROVENANCE_UNSUPPORTED:
        return AutomaticTreatmentsObservation(
            status=AUTOMATIC_TREATMENTS_UNAVAILABLE,
            reason="submission provenance version is not supported",
        )
    if record.provenance_contract == PROVENANCE_ABSENT:
        return AutomaticTreatmentsObservation(
            status=AUTOMATIC_TREATMENTS_NOT_RECORDED,
            reason="submission has no provenance block",
        )
    execution = record.provenance.get("execution")
    if not isinstance(execution, Mapping) or "automatic_treatments" not in execution:
        return AutomaticTreatmentsObservation(
            status=AUTOMATIC_TREATMENTS_NOT_RECORDED,
            reason="submission provenance does not record automatic treatments (prepared before Compute recorded them)",
        )
    treatments = execution.get("automatic_treatments")
    if treatments is None:
        return AutomaticTreatmentsObservation(
            status=AUTOMATIC_TREATMENTS_NONE,
            reason="no BMD-managed Desired Output automatic-treatment record was written for this run",
        )
    if not isinstance(treatments, Mapping):
        return AutomaticTreatmentsObservation(
            status=AUTOMATIC_TREATMENTS_UNAVAILABLE,
            reason="provenance execution.automatic_treatments is not a JSON object",
        )
    applied: list[AppliedTreatment] = []
    for item in treatments.get("applied_treatments") or ():
        if not isinstance(item, Mapping) or not isinstance(item.get("modifier"), str):
            continue
        indices = item.get("stage_indices")
        applied.append(
            AppliedTreatment(
                modifier=item["modifier"],
                display_name=item.get("display_name") if isinstance(item.get("display_name"), str) else None,
                stage_indices=tuple(
                    index for index in (indices if isinstance(indices, list) else ())
                    if _strict_int(index)
                ),
                consideration_id=(
                    item.get("consideration_id") if isinstance(item.get("consideration_id"), str) else None
                ),
            )
        )
    not_applicable = tuple(
        dict(item) for item in treatments.get("not_applicable_considerations") or () if isinstance(item, Mapping)
    )
    advisory = tuple(
        str(item) for item in treatments.get("advisory_consideration_ids") or () if isinstance(item, str)
    )
    dft_u = treatments.get("dft_u")
    return AutomaticTreatmentsObservation(
        status=AUTOMATIC_TREATMENTS_RECORDED,
        mode=treatments.get("mode") if isinstance(treatments.get("mode"), str) else None,
        desired_output=(
            treatments.get("desired_output") if isinstance(treatments.get("desired_output"), str) else None
        ),
        applied=tuple(applied),
        not_applicable=not_applicable,
        advisory_consideration_ids=advisory,
        dft_u=dict(dft_u) if isinstance(dft_u, Mapping) else None,
        raw=dict(treatments),
    )


PRODUCER_RUNTIME_RECORD = "producer_runtime_record"
RUNTIME_RECORD_RECORDED = "recorded"
RUNTIME_RECORD_NOT_RECORDED = "not_recorded"
RUNTIME_RECORD_UNAVAILABLE = "unavailable"
RUNTIME_RECORD_INVALID = "invalid"
RUNTIME_RECORD_UNSUPPORTED = "unsupported_version"

RUNTIME_RECORD_NOT_DECLARED_REASON = (
    "submission declares no runtime record (prepared before BMD Compute recorded "
    "runtime parity); runtime evidence comes only from the runner log"
)


@dataclass(frozen=True)
class RuntimeEnvironmentObservation:
    """What Agent could learn about a run's ``runtime_environment.json``.

    ``status`` describes Agent's access to the record; the parity verdict is
    ``record.status`` and is only available when ``status`` is ``recorded``.
    """

    status: str
    path: str | None = None
    record: RuntimeEnvironmentRecord | None = None
    reason: str | None = None
    limitations: tuple[str, ...] = ()
    evidence_type: str = PRODUCER_RUNTIME_RECORD

    @property
    def parity_status(self) -> str | None:
        return self.record.status if self.record is not None else None


def runtime_environment_not_declared() -> RuntimeEnvironmentObservation:
    return RuntimeEnvironmentObservation(
        status=RUNTIME_RECORD_NOT_RECORDED,
        reason=RUNTIME_RECORD_NOT_DECLARED_REASON,
    )


def runtime_environment_unavailable(path: str | None, reason: str) -> RuntimeEnvironmentObservation:
    return RuntimeEnvironmentObservation(status=RUNTIME_RECORD_UNAVAILABLE, path=path, reason=reason)


def observe_runtime_environment(
    payload: Any,
    *,
    path: str | None,
    submission: SubmissionRecord | None = None,
    expected_job_id: str | None = None,
) -> RuntimeEnvironmentObservation:
    """Parse a retrieved runtime record and cross-check which execution it names."""

    try:
        record = parse_runtime_environment_record(payload)
    except ComputeRecordError as exc:
        return RuntimeEnvironmentObservation(
            status=(
                RUNTIME_RECORD_UNSUPPORTED
                if exc.kind == UNSUPPORTED_VERSION
                else RUNTIME_RECORD_INVALID
            ),
            path=path,
            reason=str(exc),
        )
    limitations: list[str] = []
    if (
        submission is not None
        and submission.attempt_id
        and record.attempt_id
        and record.attempt_id != submission.attempt_id
    ):
        limitations.append(
            f"runtime record attempt_id {record.attempt_id} differs from the submission "
            f"attempt {submission.attempt_id}; it may describe a different execution"
        )
    if expected_job_id and record.slurm_job_id and record.slurm_job_id != expected_job_id:
        limitations.append(
            f"runtime record SLURM job {record.slurm_job_id} differs from the inspected job "
            f"{expected_job_id}"
        )
    return RuntimeEnvironmentObservation(
        status=RUNTIME_RECORD_RECORDED,
        path=path,
        record=record,
        limitations=tuple(limitations),
    )
