from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
from typing import TYPE_CHECKING, Any, Callable, Mapping

from bmd_agent.config import GitRepositoryResource, ResourceRegistry
from bmd_agent.resources.context import EvidenceGap, ScientificContext

if TYPE_CHECKING:
    from bmd_agent.resources.lifecycle import LifecycleAnalysis
    from bmd_agent.resources.run import JobInspection, StageTrajectoryObservation


SCHEMA_VERSION = 1
EVIDENCE_TYPE = "composition_context"
PRODUCER_NAME = "BMDex"
PRODUCER_MODULE = "tools.composition.context_producer"
BMDEX_COMPOSITION_CONTEXT = "bmdex_composition_context"
DOMAIN_CONTEXT_SCHEMA_VERSION = 1
DOMAIN_CONTEXT_EVIDENCE_TYPE = "contextual_reference_evidence"
DOMAIN_CONTEXT_SCHEMA = "bmdex.contextual_reference.v1"
DOMAIN_CONTEXT_PRODUCER_MODULE = "tools.domain_context.query"
BMDEX_DOMAIN_CONTEXT = "bmdex_domain_context"
# Observation identifier defined by BMDex's observed-pattern vocabulary
# (vasp/contextual_reference/observed_patterns.json). BMDex owns the identifier
# and its definition; Agent only reports it when its own trajectory evidence
# meets that definition, and never translates other strings into it.
BMDEX_PATTERN_FIRST_CYCLE_ONLY_DAV = "first_electronic_cycle_incomplete_after_only_dav_iterations"

Runner = Callable[..., subprocess.CompletedProcess[str]]


class BmdexCompositionError(RuntimeError):
    """Raised when BMDex composition-context evidence is unavailable."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "producer_failed",
        returncode: int | None = None,
        stderr_summary: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.returncode = returncode
        self.stderr_summary = stderr_summary


@dataclass(frozen=True)
class BmdexCompositionEvidence:
    """Validated BMDex composition-context schema-v1 payload."""

    payload: dict[str, Any]

    @property
    def schema_version(self) -> int:
        return int(self.payload["schema_version"])

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    @property
    def evidence_type(self) -> str:
        return str(self.payload["evidence_type"])

    @property
    def producer(self) -> Mapping[str, Any]:
        return self.payload["producer"]

    @property
    def composition(self) -> Mapping[str, Any]:
        return self.payload["composition"]

    @property
    def datasets(self) -> Mapping[str, Any]:
        return self.payload["datasets"]

    @property
    def missing_evidence(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["missing_evidence"])

    @property
    def limitations(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["limitations"])


@dataclass(frozen=True)
class BmdexContextualReferenceRecord:
    """One producer-owned contextual reference and its deterministic match evidence."""

    record: Mapping[str, Any]
    match: Mapping[str, Any]

    @property
    def record_id(self) -> str:
        return str(self.record["id"])

    @property
    def title(self) -> str:
        return str(self.record["title"])

    @property
    def contextual_statement(self) -> str:
        return str(self.record["contextual_statement"])

    @property
    def applicability(self) -> Mapping[str, Any]:
        return self.record["applicability"]

    @property
    def diagnostic_relevance(self) -> str:
        return str(self.record["diagnostic_relevance"])

    @property
    def limitations(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.record["limitations"])

    @property
    def sources(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.record["sources"])

    @property
    def record_provenance(self) -> Mapping[str, Any]:
        return self.record["record_provenance"]

    @property
    def matched_fields(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.match.get("matched_fields", ()))

    @property
    def matched_observed_patterns(self) -> tuple[str, ...]:
        """Record patterns BMDex reports as matched; empty if none or not reported."""

        value = self.match.get("matched_observed_patterns")
        if not isinstance(value, list):
            return ()
        return tuple(str(item) for item in value)


@dataclass(frozen=True)
class BmdexDomainContextEvidence:
    """Validated BMDex contextual-reference schema-v1 response."""

    payload: dict[str, Any]
    records: tuple[BmdexContextualReferenceRecord, ...]

    @property
    def schema_version(self) -> int:
        return int(self.payload["schema_version"])

    @property
    def evidence_type(self) -> str:
        return str(self.payload["evidence_type"])

    @property
    def producer(self) -> Mapping[str, Any]:
        return self.payload["producer"]

    @property
    def query(self) -> Mapping[str, Any]:
        return self.payload["query"]


@dataclass(frozen=True)
class BmdexContextualAssessment:
    """Qualified Agent assessment linked to producer-owned contextual evidence."""

    basis: tuple[str, ...]
    limitations: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    evidence_type: str = "assessment"


@dataclass(frozen=True)
class BmdexDomainContextEnrichment:
    """Optional contextual evidence acquired after lifecycle classification."""

    query: Mapping[str, Any] | None = None
    evidence: BmdexDomainContextEvidence | None = None
    assessment: BmdexContextualAssessment | None = None
    evidence_gaps: tuple[EvidenceGap, ...] = ()


@dataclass(frozen=True)
class EnrichedScientificContext:
    """ScientificContext plus optional external BMDex composition evidence."""

    base: ScientificContext
    composition_context: BmdexCompositionEvidence | None = None
    evidence_gaps: tuple[EvidenceGap, ...] = ()


class BmdexDomainContextError(RuntimeError):
    """Raised when optional BMDex contextual-reference evidence is unavailable."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "producer_failed",
        returncode: int | None = None,
        stderr_summary: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.returncode = returncode
        self.stderr_summary = stderr_summary


def bmdex_repository(registry: ResourceRegistry) -> GitRepositoryResource | None:
    """Return the explicitly configured BMDex repository, if present."""

    return registry.repositories.get("bmdex")


def enrich_lifecycle_with_bmdex_domain_context(
    analysis: LifecycleAnalysis,
    repository: GitRepositoryResource | None,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> BmdexDomainContextEnrichment:
    """Acquire optional contextual references for an already classified calculation."""

    query = build_bmdex_domain_query(analysis)
    if query is None:
        return BmdexDomainContextEnrichment()

    return _acquire_domain_context(
        query,
        repository,
        runner=runner,
        timeout=timeout,
        assessment_factory=lambda evidence: _domain_context_assessment(analysis, evidence),
    )


def enrich_job_with_bmdex_domain_context(
    inspection: JobInspection,
    repository: GitRepositoryResource | None,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> BmdexDomainContextEnrichment:
    """Acquire the existing contextual references for remote job evidence."""

    query = build_bmdex_domain_query_for_job(inspection)
    if query is None:
        return BmdexDomainContextEnrichment()

    return _acquire_domain_context(
        query,
        repository,
        runner=runner,
        timeout=timeout,
        assessment_factory=lambda evidence: _job_domain_context_assessment(
            inspection,
            evidence,
        ),
    )


def _acquire_domain_context(
    query: Mapping[str, Any],
    repository: GitRepositoryResource | None,
    *,
    runner: Runner,
    timeout: float,
    assessment_factory: Callable[[BmdexDomainContextEvidence], BmdexContextualAssessment | None],
) -> BmdexDomainContextEnrichment:

    if repository is None:
        return BmdexDomainContextEnrichment(
            query=query,
            evidence_gaps=(
                EvidenceGap(
                    BMDEX_DOMAIN_CONTEXT,
                    "missing_bmdex_repository",
                    "BMDex repository configuration is unavailable.",
                ),
            ),
        )

    try:
        evidence = inspect_bmdex_domain_context(
            repository,
            query,
            runner=runner,
            timeout=timeout,
        )
    except BmdexDomainContextError as exc:
        reason = str(exc)
        if exc.returncode is not None:
            reason = f"{reason} exit {exc.returncode}."
        if exc.stderr_summary:
            reason = f"{reason} {exc.stderr_summary}"
        return BmdexDomainContextEnrichment(
            query=query,
            evidence_gaps=(
                EvidenceGap(
                    BMDEX_DOMAIN_CONTEXT,
                    exc.kind,
                    _bounded_text(reason),
                ),
            ),
        )
    except Exception as exc:
        return BmdexDomainContextEnrichment(
            query=query,
            evidence_gaps=(
                EvidenceGap(
                    BMDEX_DOMAIN_CONTEXT,
                    "producer_failed",
                    (
                        "BMDex contextual-reference enrichment failed safely: "
                        f"{type(exc).__name__}."
                    ),
                ),
            ),
        )

    assessment = assessment_factory(evidence)
    return BmdexDomainContextEnrichment(
        query=query,
        evidence=evidence,
        assessment=assessment,
    )


def build_bmdex_domain_query(
    analysis: LifecycleAnalysis,
) -> Mapping[str, Any] | None:
    """Build a factual contextual-reference query from existing lifecycle evidence."""

    diagnostics = analysis.diagnostics
    if diagnostics is None or not diagnostics.trajectories:
        return None
    if not _bool_value(analysis.incar_settings.get("LHFCALC")):
        return None

    trajectory = next(
        (
            item
            for item in diagnostics.trajectories
            if item.completed_ionic_steps == 0
            and item.incomplete_electronic_iteration_count is not None
            and item.incomplete_electronic_iteration_count > 0
        ),
        None,
    )
    if trajectory is None:
        return None

    theory = _observed_stage_theory(analysis, trajectory.stage_index)
    return _hybrid_domain_query(
        trajectory,
        analysis.incar_settings,
        theory=theory,
    )


def build_bmdex_domain_query_for_job(
    inspection: JobInspection,
) -> Mapping[str, Any] | None:
    """Build the same factual hybrid-trajectory query from remote job evidence."""

    if inspection.bmd_compute is not None:
        trajectories = inspection.bmd_compute.trajectories
        inputs = inspection.bmd_compute.inspection.executed_inputs
        stages = inspection.bmd_compute.inspection.workflow_stages
    elif inspection.direct_vasp is not None:
        trajectories = (inspection.direct_vasp.trajectory,)
        inputs = inspection.direct_vasp.executed_inputs
        stages = ()
    else:
        return None

    trajectory = next(
        (
            item
            for item in trajectories
            if item.completed_ionic_steps == 0
            and item.incomplete_electronic_iteration_count is not None
            and item.incomplete_electronic_iteration_count > 0
        ),
        None,
    )
    if trajectory is None:
        return None
    settings = _job_executed_settings(inputs, trajectory.stage_index)
    if not _bool_value(settings.get("LHFCALC")):
        return None
    theory = None
    if trajectory.stage_index is not None and 1 <= trajectory.stage_index <= len(stages):
        theory = stages[trajectory.stage_index - 1].theory
    return _hybrid_domain_query(trajectory, settings, theory=theory)


def _hybrid_domain_query(
    trajectory: StageTrajectoryObservation,
    input_settings: Mapping[str, Any],
    *,
    theory: str | None,
) -> Mapping[str, Any]:
    query: dict[str, Any] = {
        "code": "VASP",
        "calculation_family": "hybrid_functional",
        "topic": "electronic_iteration_behavior",
    }
    if theory and theory.lower() != "unknown":
        query["functional"] = theory

    algorithm = input_settings.get("ALGO")
    if isinstance(algorithm, str) and algorithm.strip():
        query["electronic_algorithm"] = algorithm.strip()

    if _first_cycle_only_dav_iterations_observed(trajectory):
        query["observed_patterns"] = [BMDEX_PATTERN_FIRST_CYCLE_ONLY_DAV]

    input_tags = {
        key: input_settings[key]
        for key in ("LHFCALC", "HFSCREEN", "AEXX", "ALGO", "NELMDL", "LSORBIT")
        if key in input_settings
    }
    if input_tags:
        query["input_tags"] = input_tags
    return query


def _job_executed_settings(inputs: tuple[Any, ...], stage_index: int | None) -> dict[str, Any]:
    observations = tuple(
        item
        for item in inputs
        if item.present
        and item.error is None
        and (stage_index is None or item.stage_index in {None, stage_index})
    )
    values_by_key: dict[str, list[Any]] = {}
    for observation in observations:
        for key, value in observation.values.items():
            values_by_key.setdefault(str(key).upper(), []).append(value)
    return {
        key: values[0]
        for key, values in values_by_key.items()
        if values and all(value == values[0] for value in values[1:])
    }


def _first_cycle_only_dav_iterations_observed(
    trajectory: StageTrajectoryObservation,
) -> bool:
    """Whether the BMDex first-cycle/only-DAV observation definition is met.

    Requires that no ionic step completed, the first electronic cycle is
    incomplete, every iteration written for it was observed, and each of those
    iterations is DAV-labelled. A partial view does not qualify.
    """

    count = trajectory.incomplete_electronic_iteration_count
    iterations = trajectory.recent_incomplete_electronic_iterations
    if trajectory.completed_ionic_steps != 0 or not count or count <= 0:
        return False
    if len(iterations) != count:
        return False
    return all(str(item.algorithm or "").strip().upper() == "DAV" for item in iterations)


def inspect_bmdex_domain_context(
    repository: GitRepositoryResource,
    query: Mapping[str, Any],
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> BmdexDomainContextEvidence:
    """Invoke BMDex's fixed read-only contextual-reference producer."""

    if not repository.path.is_dir():
        raise BmdexDomainContextError(
            f"Configured BMDex checkout does not exist: {repository.path}",
            kind="missing_checkout",
        )
    if repository.capability_python is None:
        raise BmdexDomainContextError(
            "BMDex repository configuration is missing capability_python.",
            kind="missing_interpreter",
        )
    if not repository.capability_python.is_file():
        raise BmdexDomainContextError(
            f"Configured BMDex capability Python does not exist: {repository.capability_python}",
            kind="missing_interpreter",
        )

    command = [
        str(repository.capability_python),
        "-B",
        "-m",
        DOMAIN_CONTEXT_PRODUCER_MODULE,
    ]
    request_text = json.dumps({"query": dict(query)}, sort_keys=True)
    try:
        completed = runner(
            command,
            cwd=repository.path,
            input=request_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BmdexDomainContextError(
            "BMDex contextual-reference producer timed out.",
            kind="timeout",
        ) from exc
    except OSError as exc:
        raise BmdexDomainContextError(
            f"Could not execute BMDex contextual-reference producer: {exc}",
            kind="producer_failed",
        ) from exc

    try:
        evidence = parse_bmdex_domain_context_payload(
            completed.stdout,
            requested_query=query,
        )
    except BmdexDomainContextError as exc:
        if completed.returncode != 0 and exc.kind in {"malformed_json", "malformed_payload"}:
            raise BmdexDomainContextError(
                "BMDex contextual-reference producer failed without a valid contract response.",
                kind="producer_failed",
                returncode=completed.returncode,
                stderr_summary=_stderr_summary(completed.stderr),
            ) from exc
        raise

    if completed.returncode != 0:
        raise BmdexDomainContextError(
            "BMDex contextual-reference producer returned a successful payload with a nonzero exit.",
            kind="producer_failed",
            returncode=completed.returncode,
            stderr_summary=_stderr_summary(completed.stderr),
        )
    return evidence


def parse_bmdex_domain_context_payload(
    output: str,
    *,
    requested_query: Mapping[str, Any] | None = None,
) -> BmdexDomainContextEvidence:
    """Parse and validate BMDex contextual-reference schema v1."""

    try:
        raw_payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise BmdexDomainContextError(
            "BMDex contextual-reference producer emitted malformed JSON.",
            kind="malformed_json",
        ) from exc
    if not isinstance(raw_payload, dict):
        raise BmdexDomainContextError(
            "BMDex contextual-reference payload must be a JSON object.",
            kind="malformed_payload",
        )

    payload = dict(raw_payload)
    _validate_domain_context_payload(payload, requested_query=requested_query)
    if payload["status"] == "error":
        raise _domain_context_structured_error(payload)

    records = tuple(
        BmdexContextualReferenceRecord(
            record=dict(item["record"]),
            match=dict(item["match"]),
        )
        for item in payload["records"]
    )
    return BmdexDomainContextEvidence(payload=payload, records=records)


def _validate_domain_context_payload(
    payload: Mapping[str, Any],
    *,
    requested_query: Mapping[str, Any] | None,
) -> None:
    if payload.get("schema_version") != DOMAIN_CONTEXT_SCHEMA_VERSION:
        raise BmdexDomainContextError(
            "Unsupported BMDex contextual-reference schema_version: "
            f"{payload.get('schema_version')!r}",
            kind="unsupported_schema",
        )
    if payload.get("evidence_type") != DOMAIN_CONTEXT_EVIDENCE_TYPE:
        raise BmdexDomainContextError(
            "BMDex contextual-reference payload has an unexpected evidence_type.",
            kind="malformed_payload",
        )
    status = payload.get("status")
    if status not in {"ok", "error"}:
        raise BmdexDomainContextError(
            f"BMDex contextual-reference payload has unexpected status: {status!r}",
            kind="malformed_payload",
        )
    producer = _domain_required_mapping(payload, "producer")
    if producer.get("name") != PRODUCER_NAME:
        raise BmdexDomainContextError(
            "BMDex contextual-reference producer.name was not BMDex.",
            kind="malformed_payload",
        )
    if status == "error":
        _domain_required_mapping(payload, "error")
        return

    query = _domain_required_mapping(payload, "query")
    if requested_query is not None and dict(query) != dict(requested_query):
        raise BmdexDomainContextError(
            "BMDex contextual-reference response query did not match the request.",
            kind="malformed_payload",
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise BmdexDomainContextError(
            "BMDex contextual-reference payload missing list field: records",
            kind="malformed_payload",
        )
    result_count = payload.get("result_count")
    if not isinstance(result_count, int) or isinstance(result_count, bool):
        raise BmdexDomainContextError(
            "BMDex contextual-reference result_count must be an integer.",
            kind="malformed_payload",
        )
    if result_count != len(records):
        raise BmdexDomainContextError(
            "BMDex contextual-reference result_count did not match records.",
            kind="malformed_payload",
        )
    for item in records:
        _validate_domain_context_match(item)


def _validate_domain_context_match(item: Any) -> None:
    if not isinstance(item, Mapping):
        raise BmdexDomainContextError(
            "BMDex contextual-reference match must be an object.",
            kind="malformed_payload",
        )
    record = _domain_required_mapping(item, "record")
    match = _domain_required_mapping(item, "match")
    required_strings = (
        "id",
        "title",
        "evidence_type",
        "status",
        "contextual_statement",
        "diagnostic_relevance",
    )
    for key in required_strings:
        if not isinstance(record.get(key), str) or not record[key]:
            raise BmdexDomainContextError(
                f"BMDex contextual-reference record missing string field: {key}",
                kind="malformed_payload",
            )
    if record.get("schema_version") != DOMAIN_CONTEXT_SCHEMA_VERSION:
        raise BmdexDomainContextError(
            "BMDex contextual-reference record has unsupported schema_version.",
            kind="unsupported_schema",
        )
    if record.get("evidence_type") != DOMAIN_CONTEXT_EVIDENCE_TYPE:
        raise BmdexDomainContextError(
            "BMDex contextual-reference record has an unexpected evidence_type.",
            kind="malformed_payload",
        )
    _domain_required_mapping(record, "domain")
    _domain_required_mapping(record, "applicability")
    provenance = _domain_required_mapping(record, "record_provenance")
    if provenance.get("machine_readable_schema") != DOMAIN_CONTEXT_SCHEMA:
        raise BmdexDomainContextError(
            "BMDex contextual-reference record has an unsupported machine-readable schema.",
            kind="unsupported_schema",
        )
    if not isinstance(provenance.get("record_version"), int):
        raise BmdexDomainContextError(
            "BMDex contextual-reference record_version must be an integer.",
            kind="malformed_payload",
        )
    limitations = record.get("limitations")
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) and item for item in limitations
    ):
        raise BmdexDomainContextError(
            "BMDex contextual-reference record limitations must be non-empty strings.",
            kind="malformed_payload",
        )
    sources = record.get("sources")
    if not isinstance(sources, list) or not sources:
        raise BmdexDomainContextError(
            "BMDex contextual-reference record must preserve source provenance.",
            kind="malformed_payload",
        )
    for source in sources:
        if not isinstance(source, Mapping) or not all(
            isinstance(source.get(key), str) and source[key]
            for key in (
                "source_type",
                "title",
                "authority",
                "url",
                "retrieved_on",
                "applicability_note",
            )
        ):
            raise BmdexDomainContextError(
                "BMDex contextual-reference source provenance is malformed.",
                kind="malformed_payload",
            )
    matched_fields = match.get("matched_fields")
    if not isinstance(matched_fields, list) or not all(
        isinstance(field, str) and field for field in matched_fields
    ):
        raise BmdexDomainContextError(
            "BMDex contextual-reference match.matched_fields is malformed.",
            kind="malformed_payload",
        )
    matched_patterns = match.get("matched_observed_patterns")
    if matched_patterns is not None:
        if not isinstance(matched_patterns, list) or not all(
            isinstance(pattern, str) and pattern for pattern in matched_patterns
        ):
            raise BmdexDomainContextError(
                "BMDex contextual-reference match.matched_observed_patterns is malformed.",
                kind="malformed_payload",
            )
        if bool(matched_patterns) != ("observed_patterns" in matched_fields):
            raise BmdexDomainContextError(
                "BMDex contextual-reference match reports observed patterns inconsistently.",
                kind="malformed_payload",
            )
    if not isinstance(match.get("match_type"), str) or not match["match_type"]:
        raise BmdexDomainContextError(
            "BMDex contextual-reference match.match_type is malformed.",
            kind="malformed_payload",
        )


def _domain_required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise BmdexDomainContextError(
            f"BMDex contextual-reference payload missing object field: {key}",
            kind="malformed_payload",
        )
    return value


def _domain_context_structured_error(
    payload: Mapping[str, Any],
) -> BmdexDomainContextError:
    error = payload.get("error")
    code = error.get("code") if isinstance(error, Mapping) else None
    message = error.get("message") if isinstance(error, Mapping) else None
    details = [
        _bounded_text(str(item))
        for item in (code, message)
        if isinstance(item, str) and item
    ]
    suffix = f": {' - '.join(details)}" if details else "."
    return BmdexDomainContextError(
        f"BMDex contextual-reference producer returned a structured error{suffix}",
        kind="structured_producer_error",
    )


def _observed_stage_theory(
    analysis: LifecycleAnalysis,
    stage_index: int | None,
) -> str | None:
    workflow = analysis.bmd_workflow
    if workflow is None or not workflow.workflow_stages:
        return None
    index = stage_index
    if index is None and workflow.current_stage is not None:
        index = workflow.current_stage.stage_index
    if index is None or not 1 <= index <= len(workflow.workflow_stages):
        return None
    theory = workflow.workflow_stages[index - 1].get("theory")
    return str(theory) if theory is not None else None


def _domain_context_assessment(
    analysis: LifecycleAnalysis,
    evidence: BmdexDomainContextEvidence,
) -> BmdexContextualAssessment | None:
    termination = (
        analysis.diagnostics.termination
        if analysis.diagnostics is not None
        else None
    )
    custodian_supported = (
        termination is not None
        and termination.classification == "custodian_triggered_process_termination"
        and termination.status == "supported"
    )
    return _contextual_assessment(
        evidence,
        custodian_supported=custodian_supported,
        mentions_sigterm=_analysis_mentions_sigterm(analysis),
    )


def _job_domain_context_assessment(
    inspection: JobInspection,
    evidence: BmdexDomainContextEvidence,
) -> BmdexContextualAssessment | None:
    termination = None
    if inspection.bmd_compute is not None:
        termination = inspection.bmd_compute.termination.assessment
    elif inspection.direct_vasp is not None:
        termination = inspection.direct_vasp.termination_assessment
    custodian_supported = (
        termination is not None
        and termination.classification == "custodian_triggered_process_termination"
        and termination.status == "supported"
    )
    return _contextual_assessment(
        evidence,
        custodian_supported=custodian_supported,
        mentions_sigterm=False,
    )


def _contextual_assessment(
    evidence: BmdexDomainContextEvidence,
    *,
    custodian_supported: bool,
    mentions_sigterm: bool,
) -> BmdexContextualAssessment | None:
    if not evidence.records:
        return None
    basis: list[str] = []
    for record in evidence.records:
        basis.extend(_record_match_basis(record, evidence.query))
    if custodian_supported:
        basis.append(
            "Independently observed Custodian intervention and termination evidence supports "
            "a qualified Custodian-triggered process-termination assessment; the cited BMDex "
            "records remain contextual reference evidence rather than termination evidence."
        )
    limitations = [
        "Contextual-reference applicability does not establish the cause of the calculation's termination.",
        "Contextual reference evidence does not establish a hang or a method incompatibility by itself.",
    ]
    if custodian_supported:
        limitations.extend(
            (
                "The combined evidence does not prove the interrupted VASP operation would eventually converge.",
                "The combined evidence does not establish that every Custodian intervention was a false positive.",
                "The combined evidence does not establish scientific success.",
            )
        )
    elif mentions_sigterm:
        limitations.append("The available evidence does not establish why SIGTERM was issued.")
    return BmdexContextualAssessment(
        basis=tuple(basis),
        limitations=tuple(limitations),
        source_record_ids=tuple(record.record_id for record in evidence.records),
    )


def _record_match_basis(
    record: BmdexContextualReferenceRecord,
    query: Mapping[str, Any],
) -> tuple[str, ...]:
    """Describe a match only as far as BMDex's reported match establishes it."""

    fields = [
        field
        for field in record.matched_fields
        if field not in {"code", "observed_patterns"}
    ]
    lines = []
    if fields:
        lines.append(
            f"BMDex matched cited reference {record.record_id} ({record.title}) "
            f"on query fields: {', '.join(fields)}."
        )
    else:
        lines.append(f"BMDex returned cited reference {record.record_id} ({record.title}).")

    matched_patterns = record.matched_observed_patterns
    sent = query.get("observed_patterns") if isinstance(query, Mapping) else None
    sent_patterns = [sent] if isinstance(sent, str) else list(sent or [])
    if "observed_patterns" in record.matched_fields:
        reported = ", ".join(matched_patterns) if matched_patterns else "unreported pattern identifiers"
        lines.append(
            f"BMDex also matched the observed electronic-trajectory pattern {reported}, "
            "which this reference lists as relevant; this is an observation, not a cause."
        )
    elif sent_patterns:
        lines.append(
            f"The observed electronic-trajectory pattern {', '.join(sent_patterns)} did not "
            "match this reference, so trajectory evidence did not contribute to the match."
        )
    else:
        lines.append(
            "No BMDex-defined electronic-trajectory pattern was observed for this run, "
            "so trajectory evidence did not contribute to the match."
        )
    return tuple(lines)


def _analysis_mentions_sigterm(analysis: LifecycleAnalysis) -> bool:
    diagnostics = analysis.diagnostics
    if diagnostics is None:
        return False
    return any(
        "sigterm" in message.lower()
        for log in diagnostics.logs
        for message in log.messages
    )


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", ".true.", "t", "1", "yes"}
    return False


def enrich_scientific_context_with_bmdex(
    context: ScientificContext,
    repository: GitRepositoryResource | None,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> EnrichedScientificContext:
    """Explicitly enrich an existing ScientificContext with BMDex evidence."""

    formula = context.identity.formula
    if not formula:
        return EnrichedScientificContext(
            base=context,
            composition_context=None,
            evidence_gaps=(
                EvidenceGap(
                    BMDEX_COMPOSITION_CONTEXT,
                    "formula_unavailable",
                    "ScientificContext identity formula is unavailable.",
                ),
            ),
        )

    if repository is None:
        return EnrichedScientificContext(
            base=context,
            composition_context=None,
            evidence_gaps=(
                EvidenceGap(
                    BMDEX_COMPOSITION_CONTEXT,
                    "missing_bmdex_repository",
                    "BMDex repository configuration is unavailable.",
                ),
            ),
        )

    try:
        evidence = inspect_bmdex_composition_context(
            repository,
            formula,
            runner=runner,
            timeout=timeout,
        )
    except BmdexCompositionError as exc:
        return EnrichedScientificContext(
            base=context,
            composition_context=None,
            evidence_gaps=(_gap_from_error(exc),),
        )

    return EnrichedScientificContext(base=context, composition_context=evidence)


def inspect_bmdex_composition_context(
    repository: GitRepositoryResource,
    formula: str,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> BmdexCompositionEvidence:
    """Invoke BMDex's configured read-only composition-context producer."""

    if not repository.path.is_dir():
        raise BmdexCompositionError(
            f"Configured BMDex checkout does not exist: {repository.path}",
            kind="missing_checkout",
        )

    if repository.capability_python is None:
        raise BmdexCompositionError(
            "BMDex repository configuration is missing capability_python.",
            kind="missing_interpreter",
        )

    if not repository.capability_python.is_file():
        raise BmdexCompositionError(
            f"Configured BMDex capability Python does not exist: {repository.capability_python}",
            kind="missing_interpreter",
        )

    command = [str(repository.capability_python), "-B", "-m", PRODUCER_MODULE]
    request_text = json.dumps({"formula": formula}, sort_keys=True)

    try:
        completed = runner(
            command,
            cwd=repository.path,
            input=request_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BmdexCompositionError(
            "BMDex composition-context producer timed out.",
            kind="timeout",
        ) from exc
    except OSError as exc:
        raise BmdexCompositionError(
            f"Could not execute BMDex composition-context producer: {exc}",
            kind="producer_failed",
        ) from exc

    try:
        evidence = parse_bmdex_composition_payload(
            completed.stdout,
            requested_formula=formula,
        )
    except BmdexCompositionError as exc:
        if completed.returncode != 0 and exc.kind in {"malformed_json", "malformed_payload"}:
            raise BmdexCompositionError(
                "BMDex composition-context producer failed without a valid contract response.",
                kind="producer_failed",
                returncode=completed.returncode,
                stderr_summary=_stderr_summary(completed.stderr),
            ) from exc
        raise

    if completed.returncode != 0:
        raise BmdexCompositionError(
            "BMDex composition-context producer returned a successful payload with a nonzero exit.",
            kind="producer_failed",
            returncode=completed.returncode,
            stderr_summary=_stderr_summary(completed.stderr),
        )

    return evidence


def parse_bmdex_composition_payload(
    output: str,
    *,
    requested_formula: str | None = None,
) -> BmdexCompositionEvidence:
    """Parse and validate BMDex composition-context schema v1."""

    try:
        raw_payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise BmdexCompositionError(
            "BMDex composition-context producer emitted malformed JSON.",
            kind="malformed_json",
        ) from exc

    if not isinstance(raw_payload, dict):
        raise BmdexCompositionError(
            "BMDex composition-context payload must be a JSON object.",
            kind="malformed_payload",
        )

    payload = dict(raw_payload)
    _validate_payload(payload, requested_formula=requested_formula)
    if payload["status"] == "error":
        raise _structured_error(payload)
    return BmdexCompositionEvidence(payload=payload)


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    requested_formula: str | None,
) -> None:
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise BmdexCompositionError(
            f"Unsupported BMDex composition-context schema_version: {schema_version!r}",
            kind="unsupported_schema",
        )

    if payload.get("evidence_type") != EVIDENCE_TYPE:
        raise BmdexCompositionError(
            "BMDex composition-context payload has an unexpected evidence_type.",
            kind="malformed_payload",
        )

    status = payload.get("status")
    if status not in {"ok", "error"}:
        raise BmdexCompositionError(
            f"BMDex composition-context payload has unexpected status: {status!r}",
            kind="malformed_payload",
        )

    _validate_producer(_required_mapping(payload, "producer"))

    if status == "error":
        _required_mapping(payload, "error")
        return

    composition = _required_mapping(payload, "composition")
    _required_mapping(payload, "datasets")
    _required_list(payload, "missing_evidence")
    _required_list(payload, "limitations")

    supplied_formula = composition.get("supplied_formula")
    if not isinstance(supplied_formula, str) or not supplied_formula:
        raise BmdexCompositionError(
            "BMDex composition-context composition.supplied_formula must be a non-empty string.",
            kind="malformed_payload",
        )

    if requested_formula is not None and supplied_formula != requested_formula:
        raise BmdexCompositionError(
            "BMDex composition-context payload formula does not match the request.",
            kind="malformed_payload",
        )


def _validate_producer(producer: Mapping[str, Any]) -> None:
    name = producer.get("name")
    if name != PRODUCER_NAME:
        raise BmdexCompositionError(
            "BMDex composition-context producer.name was not BMDex.",
            kind="malformed_payload",
        )


def _structured_error(payload: Mapping[str, Any]) -> BmdexCompositionError:
    error = payload.get("error")
    code = None
    message = None
    if isinstance(error, Mapping):
        code = error.get("code")
        message = error.get("message")

    details = []
    if isinstance(code, str) and code:
        details.append(code)
    if isinstance(message, str) and message:
        details.append(_bounded_text(message))

    suffix = f": {' - '.join(details)}" if details else "."
    return BmdexCompositionError(
        f"BMDex composition-context producer returned a structured error{suffix}",
        kind="structured_producer_error",
    )


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise BmdexCompositionError(
            f"BMDex composition-context payload missing object field: {key}",
            kind="malformed_payload",
        )
    return value


def _required_list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise BmdexCompositionError(
            f"BMDex composition-context payload missing list field: {key}",
            kind="malformed_payload",
        )
    return value


def _gap_from_error(error: BmdexCompositionError) -> EvidenceGap:
    reason = str(error)
    if error.returncode is not None:
        reason = f"{reason} exit {error.returncode}."
    if error.stderr_summary:
        reason = f"{reason} {error.stderr_summary}"
    return EvidenceGap(BMDEX_COMPOSITION_CONTEXT, error.kind, _bounded_text(reason))


def _stderr_summary(stderr: object) -> str | None:
    if stderr is None:
        return None
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith(("traceback", "file \"", "raise ")):
            continue
        return _bounded_text(stripped)
    return None


def _bounded_text(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."
