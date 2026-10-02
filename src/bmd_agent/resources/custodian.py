from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from typing import Any


CUSTODIAN_INTERVENTION_EVIDENCE = "custodian_intervention_evidence"
CUSTODIAN_POLICY_EVIDENCE = "producer_provenance"
TERMINATION_ASSESSMENT = "termination_assessment"

_MAX_CONTAINER_ITEMS = 128
_MAX_TEXT_LENGTH = 500
_TERMINAL_FLAG_NAMES = (
    "max_errors",
    "max_errors_per_job",
    "max_errors_per_handler",
    "nonzero_return_code",
)
_FROZEN_HANDLER = "custodian.vasp.handlers.FrozenJobErrorHandler"


@dataclass(frozen=True)
class CustodianActionObservation:
    target: str | None
    operation: str | None
    parameter: str | None
    value: Any
    summary: str


@dataclass(frozen=True)
class CustodianCorrectionObservation:
    attempt_index: int
    correction_index: int
    sequence_index: int
    handler: str
    handler_configuration: Mapping[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    actions: tuple[CustodianActionObservation, ...] = ()

    @property
    def timeout_seconds(self) -> float | int | None:
        value = self.handler_configuration.get("timeout")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        return None


@dataclass(frozen=True)
class CustodianTerminalFlagObservation:
    attempt_index: int
    name: str
    value: Any
    interpretation: str


@dataclass(frozen=True)
class RepeatedCustodianIntervention:
    handler: str
    count: int
    timeout_seconds: float | int | None
    errors: tuple[str, ...]
    action_summaries: tuple[str, ...]
    correction_positions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CustodianInterventionEvidence:
    source_path: str
    present: bool
    attempt_count: int = 0
    corrections: tuple[CustodianCorrectionObservation, ...] = ()
    repeated_interventions: tuple[RepeatedCustodianIntervention, ...] = ()
    terminal_flags: tuple[CustodianTerminalFlagObservation, ...] = ()
    error: str | None = None
    limitations: tuple[str, ...] = ()
    evidence_type: str = CUSTODIAN_INTERVENTION_EVIDENCE

    @property
    def path(self) -> str:
        return self.source_path

    @property
    def events(self) -> tuple[str, ...]:
        """Compatibility view for bounded diagnostic/OOM marker inspection."""

        events: list[str] = []
        for correction in self.corrections:
            errors = ", ".join(correction.errors) or "no reported error"
            events.append(f"{correction.handler}: {errors}")
        for flag in self.terminal_flags:
            if flag.value is True:
                events.append(f"{flag.name}=true")
        return tuple(events)


@dataclass(frozen=True)
class ConfiguredCustodianComponent:
    class_name: str
    configuration: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CustodianStagePolicy:
    stage_index: int
    policy_id: str
    policy_version: int
    stage_type: str
    theory: str
    handlers: tuple[ConfiguredCustodianComponent, ...]
    explicit_handler_exclusions: tuple[str, ...]
    vasp_error_exclusions: tuple[str, ...]
    validators: tuple[ConfiguredCustodianComponent, ...]
    validators_source: str | None
    validators_explicit_override: Any
    walltime_authority: str | None
    walltime_handler: Any
    custodian_version: str | None
    implementation_source: str | None
    rationale: str | None

    @property
    def frozen_job_handler_status(self) -> str:
        if _FROZEN_HANDLER in self.explicit_handler_exclusions:
            return "explicitly excluded"
        if any(handler.class_name == _FROZEN_HANDLER for handler in self.handlers):
            return "configured"
        return "unknown"


@dataclass(frozen=True)
class CustodianPolicyEvidence:
    available: bool
    stages: tuple[CustodianStagePolicy, ...] = ()
    reason: str | None = None
    evidence_type: str = CUSTODIAN_POLICY_EVIDENCE
    native_path: str = "submission.provenance.execution.custodian.stages"


@dataclass(frozen=True)
class TerminationEvidenceAssessment:
    classification: str
    status: str
    basis: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    evidence_type: str = TERMINATION_ASSESSMENT


def parse_custodian_json(text: str, *, source_path: str) -> CustodianInterventionEvidence:
    """Parse bounded Custodian JSON as data without importing serialized classes."""

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, UnicodeError):
        return CustodianInterventionEvidence(
            source_path=source_path,
            present=True,
            error="custodian.json is malformed or incomplete",
        )
    return parse_custodian_payload(payload, source_path=source_path)


def parse_custodian_payload(
    payload: Any,
    *,
    source_path: str,
) -> CustodianInterventionEvidence:
    attempts = _attempt_records(payload)
    if attempts is None:
        return CustodianInterventionEvidence(
            source_path=source_path,
            present=True,
            error="custodian.json has an unsupported top-level structure",
        )

    corrections: list[CustodianCorrectionObservation] = []
    flags: list[CustodianTerminalFlagObservation] = []
    limitations: list[str] = []
    sequence_index = 0
    for attempt_index, attempt in enumerate(attempts[:_MAX_CONTAINER_ITEMS], start=1):
        if not isinstance(attempt, Mapping):
            limitations.append(f"attempt {attempt_index} is not a JSON object")
            continue
        raw_corrections = attempt.get("corrections")
        if raw_corrections is None and _looks_like_correction(attempt):
            raw_corrections = (attempt,)
        if raw_corrections is None:
            raw_corrections = ()
        if not _is_sequence(raw_corrections):
            limitations.append(f"attempt {attempt_index} corrections are not a list")
            raw_corrections = ()
        for correction_index, raw in enumerate(raw_corrections[:_MAX_CONTAINER_ITEMS], start=1):
            sequence_index += 1
            if not isinstance(raw, Mapping):
                limitations.append(
                    f"attempt {attempt_index} correction {correction_index} is not a JSON object"
                )
                continue
            handler, configuration = _handler_record(raw.get("handler"))
            if handler == "unknown":
                limitations.append(
                    f"attempt {attempt_index} correction {correction_index} handler is unavailable"
                )
            corrections.append(
                CustodianCorrectionObservation(
                    attempt_index=attempt_index,
                    correction_index=correction_index,
                    sequence_index=sequence_index,
                    handler=handler,
                    handler_configuration=configuration,
                    errors=_string_tuple(raw.get("errors") or raw.get("error")),
                    actions=_action_observations(raw.get("actions")),
                )
            )
        for name in _TERMINAL_FLAG_NAMES:
            if name not in attempt:
                continue
            value = _bounded_json_value(attempt[name])
            flags.append(
                CustodianTerminalFlagObservation(
                    attempt_index=attempt_index,
                    name=name,
                    value=value,
                    interpretation=_terminal_flag_interpretation(name, value),
                )
            )

    if len(attempts) > _MAX_CONTAINER_ITEMS:
        limitations.append(
            f"only the first {_MAX_CONTAINER_ITEMS} top-level Custodian attempts were parsed"
        )
    return CustodianInterventionEvidence(
        source_path=source_path,
        present=True,
        attempt_count=len(attempts),
        corrections=tuple(corrections),
        repeated_interventions=_repeated_interventions(corrections),
        terminal_flags=tuple(flags),
        limitations=tuple(_ordered_unique(limitations)),
    )


def custodian_policy_from_submission_record(record: Any) -> CustodianPolicyEvidence:
    """Read Custodian policy only from provenance the submission reader accepted."""

    if getattr(record, "provenance_contract", None) == "unsupported":
        return CustodianPolicyEvidence(
            available=False,
            reason=(
                "submission provenance version is not supported; persisted Custodian "
                "execution policy was not interpreted"
            ),
        )
    return parse_custodian_policy_provenance({"provenance": dict(record.provenance)})


def parse_custodian_policy_provenance(
    submission: Mapping[str, Any],
) -> CustodianPolicyEvidence:
    provenance = submission.get("provenance")
    if not isinstance(provenance, Mapping):
        return CustodianPolicyEvidence(
            available=False,
            reason="submission has no persisted Custodian execution-policy provenance",
        )
    execution = provenance.get("execution")
    custodian = execution.get("custodian") if isinstance(execution, Mapping) else None
    stages = custodian.get("stages") if isinstance(custodian, Mapping) else None
    if stages is None:
        return CustodianPolicyEvidence(
            available=False,
            reason="submission has no persisted Custodian execution-policy provenance",
        )
    if not isinstance(stages, list):
        return CustodianPolicyEvidence(
            available=False,
            reason="persisted Custodian policy stages are malformed",
        )

    parsed: list[CustodianStagePolicy] = []
    try:
        for position, raw in enumerate(stages, start=1):
            if not isinstance(raw, Mapping):
                raise ValueError(f"stage policy {position} is not an object")
            stage = raw.get("stage")
            validators = raw.get("validators")
            if not isinstance(stage, Mapping) or not isinstance(validators, Mapping):
                raise ValueError(f"stage policy {position} is missing stage or validators")
            stage_index = raw.get("index")
            policy_id = raw.get("policy_id")
            policy_version = raw.get("policy_version")
            if not isinstance(stage_index, int) or isinstance(stage_index, bool) or stage_index < 1:
                raise ValueError(f"stage policy {position} has invalid index")
            if not isinstance(policy_id, str) or not policy_id:
                raise ValueError(f"stage policy {position} has invalid policy_id")
            if not isinstance(policy_version, int) or isinstance(policy_version, bool):
                raise ValueError(f"stage policy {position} has invalid policy_version")
            parsed.append(
                CustodianStagePolicy(
                    stage_index=stage_index,
                    policy_id=policy_id,
                    policy_version=policy_version,
                    stage_type=_required_policy_text(stage, "stage_type", position),
                    theory=_required_policy_text(stage, "theory", position),
                    handlers=_configured_components(raw.get("handlers"), position, "handlers"),
                    explicit_handler_exclusions=_policy_string_list(
                        raw.get("explicit_handler_exclusions"),
                        position,
                        "explicit_handler_exclusions",
                    ),
                    vasp_error_exclusions=_policy_string_list(
                        raw.get("vasp_error_exclusions"),
                        position,
                        "vasp_error_exclusions",
                    ),
                    validators=_configured_components(
                        validators.get("resolved"),
                        position,
                        "validators.resolved",
                    ),
                    validators_source=_optional_text(validators.get("source")),
                    validators_explicit_override=_bounded_json_value(
                        validators.get("explicit_override")
                    ),
                    walltime_authority=_optional_text(raw.get("walltime_authority")),
                    walltime_handler=_bounded_json_value(raw.get("walltime_handler")),
                    custodian_version=_optional_text(raw.get("custodian_version")),
                    implementation_source=_optional_text(raw.get("implementation_source")),
                    rationale=_optional_text(raw.get("rationale")),
                )
            )
    except ValueError as exc:
        return CustodianPolicyEvidence(
            available=False,
            reason=f"persisted Custodian execution-policy provenance is malformed: {exc}",
        )
    return CustodianPolicyEvidence(available=True, stages=tuple(parsed))


def assess_termination_evidence(
    *,
    scheduler_state: str | None,
    custodian_evidence: Sequence[CustodianInterventionEvidence] = (),
    log_messages: Sequence[str] = (),
    error_archive_count: int = 0,
) -> TerminationEvidenceAssessment:
    """Classify termination only where existing evidence directly supports it."""

    state = (scheduler_state or "").upper()
    if state == "OUT_OF_MEMORY":
        return TerminationEvidenceAssessment(
            classification="slurm_out_of_memory",
            status="supported",
            basis=("SLURM accounting reports OUT_OF_MEMORY",),
        )
    if state == "TIMEOUT":
        return TerminationEvidenceAssessment(
            classification="slurm_timeout",
            status="supported",
            basis=("SLURM accounting reports TIMEOUT",),
            limitations=(
                "scheduler timeout does not by itself describe scientific convergence progress",
            ),
        )

    corrections = tuple(
        correction
        for evidence in custodian_evidence
        for correction in evidence.corrections
    )
    sigterm = any("sigterm" in message.lower() for message in log_messages)
    budget_flag = any(
        flag.value is True and flag.name in {
            "max_errors",
            "max_errors_per_job",
            "max_errors_per_handler",
        }
        for evidence in custodian_evidence
        for flag in evidence.terminal_flags
    )
    nonzero = any(
        flag.name == "nonzero_return_code" and flag.value is True
        for evidence in custodian_evidence
        for flag in evidence.terminal_flags
    )
    repeated_terminating = any(
        item.count > 1
        and (
            item.handler == _FROZEN_HANDLER
            or item.handler.rsplit(".", 1)[-1] == "FrozenJobErrorHandler"
            or any(
                correction.handler == item.handler
                and correction.handler_configuration.get("is_terminating") is True
                for correction in corrections
            )
        )
        for evidence in custodian_evidence
        for item in evidence.repeated_interventions
    )
    if (
        corrections
        and sigterm
        and repeated_terminating
        and (budget_flag or error_archive_count > 1)
    ):
        basis = [
            f"{len(corrections)} Custodian correction record(s) were observed",
            "SIGTERM evidence was observed in bounded execution logs",
        ]
        if budget_flag:
            basis.append("Custodian recorded a terminal correction-budget flag")
        if error_archive_count > 1:
            basis.append(f"{error_archive_count} Custodian error archive(s) were observed")
        return TerminationEvidenceAssessment(
            classification="custodian_triggered_process_termination",
            status="supported",
            basis=tuple(basis),
            limitations=(
                "the evidence does not prove the interrupted VASP operation would eventually converge",
                "the evidence does not establish that every observed Custodian intervention was a false positive",
                "the evidence does not establish scientific success",
            ),
        )
    if nonzero:
        return TerminationEvidenceAssessment(
            classification="vasp_nonzero_exit_observed",
            status="supported",
            basis=("Custodian recorded nonzero_return_code=true",),
            limitations=(
                "a nonzero process exit does not by itself establish its underlying cause",
            ),
        )

    basis = ()
    limitations: list[str] = []
    if corrections:
        basis = (f"{len(corrections)} Custodian correction record(s) were observed",)
        limitations.append(
            "observed Custodian interventions do not establish the final termination cause without corroborating evidence"
        )
    else:
        limitations.append("no decisive termination evidence was available")
    return TerminationEvidenceAssessment(
        classification="unknown",
        status="insufficient_evidence",
        basis=basis,
        limitations=tuple(limitations),
    )


def _attempt_records(payload: Any) -> list[Any] | None:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, Mapping):
        return None
    jobs = payload.get("jobs")
    if isinstance(jobs, list):
        return jobs
    if "corrections" in payload or _looks_like_correction(payload):
        return [payload]
    return None


def _looks_like_correction(value: Mapping[str, Any]) -> bool:
    return "handler" in value and any(key in value for key in ("errors", "error", "actions"))


def _handler_record(value: Any) -> tuple[str, Mapping[str, Any]]:
    if isinstance(value, str):
        return value[:_MAX_TEXT_LENGTH], {}
    if not isinstance(value, Mapping):
        return "unknown", {}
    module = value.get("@module")
    class_name = value.get("@class") or value.get("class") or value.get("name")
    if isinstance(module, str) and isinstance(class_name, str):
        name = f"{module}.{class_name}"
    elif isinstance(class_name, str):
        name = class_name
    else:
        name = "unknown"
    configuration = {
        str(key): _bounded_json_value(item)
        for key, item in list(value.items())[:_MAX_CONTAINER_ITEMS]
        if key not in {"@module", "@class", "class", "name"}
        and not str(key).startswith("_")
        and key != "n_applied_corrections"
    }
    return name[:_MAX_TEXT_LENGTH], configuration


def _action_observations(value: Any) -> tuple[CustodianActionObservation, ...]:
    if value is None:
        return ()
    raw_actions = value if _is_sequence(value) else (value,)
    observations: list[CustodianActionObservation] = []
    for raw in raw_actions[:_MAX_CONTAINER_ITEMS]:
        if not isinstance(raw, Mapping):
            summary = _bounded_text(raw)
            observations.append(CustodianActionObservation(None, None, None, None, summary))
            continue
        target = _optional_text(raw.get("dict") or raw.get("file"))
        action = raw.get("action")
        if not isinstance(action, Mapping):
            observations.append(
                CustodianActionObservation(
                    target,
                    None,
                    None,
                    _bounded_json_value(action),
                    _bounded_text(raw),
                )
            )
            continue
        for operation, operation_value in list(action.items())[:_MAX_CONTAINER_ITEMS]:
            if isinstance(operation_value, Mapping):
                for parameter, parameter_value in list(operation_value.items())[:_MAX_CONTAINER_ITEMS]:
                    summary = _action_summary(
                        target,
                        str(operation),
                        str(parameter),
                        parameter_value,
                    )
                    observations.append(
                        CustodianActionObservation(
                            target,
                            str(operation),
                            str(parameter),
                            _bounded_json_value(parameter_value),
                            summary,
                        )
                    )
            else:
                summary = _action_summary(target, str(operation), None, operation_value)
                observations.append(
                    CustodianActionObservation(
                        target,
                        str(operation),
                        None,
                        _bounded_json_value(operation_value),
                        summary,
                    )
                )
    return tuple(observations)


def _action_summary(
    target: str | None,
    operation: str,
    parameter: str | None,
    value: Any,
) -> str:
    prefix = f"{target}." if target else ""
    if operation == "_set" and parameter:
        return f"{prefix}{parameter} -> {_bounded_text(value)}"
    if parameter:
        return f"{prefix}{operation} {parameter}={_bounded_text(value)}"
    return f"{prefix}{operation} {_bounded_text(value)}".strip()


def _repeated_interventions(
    corrections: Sequence[CustodianCorrectionObservation],
) -> tuple[RepeatedCustodianIntervention, ...]:
    grouped: defaultdict[tuple[Any, ...], list[CustodianCorrectionObservation]] = defaultdict(list)
    for correction in corrections:
        key = (
            correction.handler,
            json.dumps(
                _bounded_json_value(correction.handler_configuration),
                sort_keys=True,
                separators=(",", ":"),
            ),
            correction.timeout_seconds,
            correction.errors,
            tuple(action.summary for action in correction.actions),
        )
        grouped[key].append(correction)
    repeated: list[RepeatedCustodianIntervention] = []
    for (handler, _configuration, timeout, errors, actions), records in grouped.items():
        if len(records) < 2:
            continue
        repeated.append(
            RepeatedCustodianIntervention(
                handler=handler,
                count=len(records),
                timeout_seconds=timeout,
                errors=errors,
                action_summaries=actions,
                correction_positions=tuple(
                    (record.attempt_index, record.correction_index)
                    for record in records
                ),
            )
        )
    return tuple(repeated)


def _terminal_flag_interpretation(name: str, value: Any) -> str:
    if value is not True:
        return (
            "raw flag retained; a non-true value does not rule out a different "
            "Custodian terminal condition"
        )
    return {
        "max_errors": "total Custodian correction limit reached",
        "max_errors_per_job": "per-job Custodian correction limit reached",
        "max_errors_per_handler": "per-handler Custodian correction limit reached",
        "nonzero_return_code": "job returned a nonzero code and Custodian terminated",
    }[name]


def _configured_components(
    value: Any,
    stage_position: int,
    field_name: str,
) -> tuple[ConfiguredCustodianComponent, ...]:
    if not isinstance(value, list):
        raise ValueError(f"stage policy {stage_position} {field_name} is not a list")
    components: list[ConfiguredCustodianComponent] = []
    for component_position, raw in enumerate(value, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"stage policy {stage_position} {field_name}[{component_position}] is not an object"
            )
        class_name = raw.get("class")
        configuration = raw.get("configuration")
        if not isinstance(class_name, str) or not class_name:
            raise ValueError(
                f"stage policy {stage_position} {field_name}[{component_position}] has invalid class"
            )
        if not isinstance(configuration, Mapping):
            raise ValueError(
                f"stage policy {stage_position} {field_name}[{component_position}] has invalid configuration"
            )
        components.append(
            ConfiguredCustodianComponent(
                class_name=class_name,
                configuration=_bounded_json_value(configuration),
            )
        )
    return tuple(components)


def _policy_string_list(value: Any, stage_position: int, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"stage policy {stage_position} {field_name} is not a string list")
    return tuple(value)


def _required_policy_text(value: Mapping[str, Any], key: str, stage_position: int) -> str:
    text = value.get(key)
    if not isinstance(text, str) or not text:
        raise ValueError(f"stage policy {stage_position} has invalid stage.{key}")
    return text


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values = value if _is_sequence(value) else (value,)
    return tuple(_bounded_text(item) for item in values[:_MAX_CONTAINER_ITEMS])


def _bounded_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return "<nested value omitted>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:_MAX_TEXT_LENGTH]
    if isinstance(value, Mapping):
        return {
            str(key)[:_MAX_TEXT_LENGTH]: _bounded_json_value(item, depth=depth + 1)
            for key, item in list(value.items())[:_MAX_CONTAINER_ITEMS]
        }
    if _is_sequence(value):
        return [
            _bounded_json_value(item, depth=depth + 1)
            for item in value[:_MAX_CONTAINER_ITEMS]
        ]
    return _bounded_text(value)


def _bounded_text(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(_bounded_json_value(value), sort_keys=True)
        except (TypeError, ValueError):
            text = type(value).__name__
    return " ".join(text.split())[:_MAX_TEXT_LENGTH]


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value[:_MAX_TEXT_LENGTH]
    return None


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _ordered_unique(values: Sequence[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique
