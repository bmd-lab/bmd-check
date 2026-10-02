from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
from typing import Any, Callable, Mapping

from bmd_agent.config import GitRepositoryResource


# Compatibility with BMD Compute's capability producer is established only from
# machine-readable fields: ``schema_version`` and ``source.repository``. The
# producer's ``scope`` text is a human-readable description that is preserved
# for display but never compared as a contract.
SCHEMA_VERSION = 1
PRODUCER_MODULE = "backend.calculations.capabilities"
PRODUCER_REPOSITORY_ID = "bmd_compute"

Runner = Callable[..., subprocess.CompletedProcess[str]]


class ComputeCapabilityError(RuntimeError):
    """Raised when BMD Compute capability introspection cannot be completed."""


@dataclass(frozen=True)
class ComputeCapabilities:
    """Validated BMD Compute schema-v1 capability payload."""

    payload: dict[str, Any]

    @property
    def source(self) -> Mapping[str, Any]:
        return self.payload["source"]

    @property
    def scope(self) -> str | None:
        """Producer-supplied human-readable scope, if any; not a contract."""

        scope = self.payload.get("scope")
        return scope if isinstance(scope, str) and scope else None

    @property
    def capabilities(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["capabilities"])

    @property
    def base_stage_definitions(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["base_stage_definitions"])

    @property
    def stage_modifier_support(self) -> tuple[Mapping[str, Any], ...] | None:
        """Compute's additive stage-local modifier support, or None if not declared.

        Historical schema-v1 payloads predate this field; None means "not
        declared by this producer", never "no modifiers supported".
        """

        support = self.payload.get("stage_modifier_support")
        return None if support is None else tuple(support)


def inspect_compute_capabilities(
    repository: GitRepositoryResource,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> ComputeCapabilities:
    """Invoke BMD Compute's configured read-only capability producer."""

    if not repository.path.is_dir():
        raise ComputeCapabilityError(
            f"Configured BMD Compute checkout does not exist: {repository.path}"
        )

    if repository.capability_python is None:
        raise ComputeCapabilityError(
            "BMD Compute repository configuration is missing capability_python."
        )

    if not repository.capability_python.is_file():
        raise ComputeCapabilityError(
            f"Configured BMD Compute capability Python does not exist: {repository.capability_python}"
        )

    try:
        completed = runner(
            [str(repository.capability_python), "-B", "-m", PRODUCER_MODULE],
            cwd=repository.path,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )

    except subprocess.TimeoutExpired as exc:
        raise ComputeCapabilityError("BMD Compute capability producer timed out.") from exc

    except subprocess.CalledProcessError as exc:
        details = _first_error_line(exc.stderr)
        message = "BMD Compute capability producer failed."
        if details:
            message = f"{message} {details}"
        raise ComputeCapabilityError(message) from exc

    except OSError as exc:
        raise ComputeCapabilityError(
            f"Could not execute BMD Compute capability producer: {exc}"
        ) from exc

    return parse_capability_payload(completed.stdout)


def parse_capability_payload(output: str) -> ComputeCapabilities:
    """Parse and validate BMD Compute capability schema v1."""

    try:
        raw_payload = json.loads(output)

    except json.JSONDecodeError as exc:
        raise ComputeCapabilityError("BMD Compute capability producer emitted malformed JSON.") from exc

    if not isinstance(raw_payload, dict):
        raise ComputeCapabilityError("BMD Compute capability payload must be a JSON object.")

    payload = dict(raw_payload)
    _validate_payload(payload)
    return ComputeCapabilities(payload=payload)


def supported_capability_pairs(capabilities: ComputeCapabilities) -> tuple[tuple[str, str], ...]:
    """Return producer-declared supported stage/theory pairs."""

    return tuple(
        (str(record["stage_type"]), str(record["theory"]))
        for record in capabilities.capabilities
    )


def supported_modifier_combinations(
    capabilities: ComputeCapabilities,
    stage_type: str,
    theory: str,
) -> tuple[frozenset[str], ...] | None:
    """Return Compute's declared modifier sets for one stage/theory pair.

    Returns None when the producer does not declare stage-local modifier
    support (historical payloads) or does not list the pair. Agent never
    derives modifier support from any other part of the payload, such as the
    automatic-treatment policy text.
    """

    support = capabilities.stage_modifier_support
    if support is None:
        return None
    for record in support:
        if record["stage_type"] == stage_type and record["theory"] == theory:
            return tuple(
                frozenset(combination)
                for combination in record["supported_modifier_combinations"]
            )
    return None


def _validate_payload(payload: Mapping[str, Any]) -> None:
    schema_version = payload.get("schema_version")

    if schema_version != SCHEMA_VERSION:
        raise ComputeCapabilityError(
            f"Unsupported BMD Compute capability schema_version: {schema_version!r}"
        )

    scope = payload.get("scope")
    if scope is not None and not isinstance(scope, str):
        raise ComputeCapabilityError("BMD Compute capability scope must be text when present.")

    _validate_source(_required_mapping(payload, "source"))
    _validate_contract(_required_mapping(payload, "contract"))
    _validate_stage_definitions(_required_list(payload, "base_stage_definitions"))
    _validate_capabilities(_required_list(payload, "capabilities"))
    if payload.get("stage_modifier_support") is not None:
        _validate_stage_modifier_support(
            _required_list(payload, "stage_modifier_support"),
            {
                (record["stage_type"], record["theory"])
                for record in payload["capabilities"]
            },
        )


def _validate_source(source: Mapping[str, Any]) -> None:
    repository = _required_str(source, "repository")
    if repository != PRODUCER_REPOSITORY_ID:
        raise ComputeCapabilityError(
            f"Capability payload is not from BMD Compute: source.repository={repository!r}"
        )

    commit = source.get("commit")
    if commit is not None and not isinstance(commit, str):
        raise ComputeCapabilityError("BMD Compute source.commit must be a string or null.")

    dirty = source.get("dirty")
    if dirty is not None and not isinstance(dirty, bool):
        raise ComputeCapabilityError("BMD Compute source.dirty must be true, false, or null.")

    if not isinstance(source.get("provenance_available"), bool):
        raise ComputeCapabilityError("BMD Compute source.provenance_available must be boolean.")

    reason = source.get("unavailable_reason")
    if reason is not None and not isinstance(reason, str):
        raise ComputeCapabilityError(
            "BMD Compute source.unavailable_reason must be a string or null."
        )


def _validate_contract(contract: Mapping[str, Any]) -> None:
    _required_str(contract, "base_stage_definitions")
    _required_str(contract, "capabilities")


def _validate_stage_definitions(stage_definitions: list[Any]) -> None:
    seen: set[str] = set()

    for index, record in enumerate(stage_definitions):
        mapping = _as_mapping(record, f"base_stage_definitions[{index}]")
        stage_type = _required_str(mapping, "stage_type")

        if stage_type in seen:
            raise ComputeCapabilityError(
                f"Duplicate BMD Compute base stage definition: {stage_type!r}"
            )

        seen.add(stage_type)


def _validate_capabilities(capabilities: list[Any]) -> None:
    seen: set[tuple[str, str]] = set()

    for index, record in enumerate(capabilities):
        mapping = _as_mapping(record, f"capabilities[{index}]")
        stage_type = _required_str(mapping, "stage_type")
        theory = _required_str(mapping, "theory")
        supported = mapping.get("theory_supported_for_stage")

        if supported is not True:
            raise ComputeCapabilityError(
                f"BMD Compute capabilities[{index}] is not an explicitly supported capability."
            )

        key = (stage_type, theory)
        if key in seen:
            raise ComputeCapabilityError(
                f"Duplicate BMD Compute capability record: {stage_type!r}/{theory!r}"
            )

        seen.add(key)


def _validate_stage_modifier_support(
    support: list[Any],
    capability_pairs: set[tuple[str, str]],
) -> None:
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(support):
        mapping = _as_mapping(record, f"stage_modifier_support[{index}]")
        key = (_required_str(mapping, "stage_type"), _required_str(mapping, "theory"))
        if key in seen:
            raise ComputeCapabilityError(
                f"Duplicate BMD Compute stage_modifier_support record: {key[0]!r}/{key[1]!r}"
            )
        if key not in capability_pairs:
            raise ComputeCapabilityError(
                "BMD Compute stage_modifier_support names a stage/theory pair that is not "
                f"a declared capability: {key[0]!r}/{key[1]!r}"
            )
        seen.add(key)
        combinations = mapping.get("supported_modifier_combinations")
        if not isinstance(combinations, list) or not all(
            isinstance(combination, list)
            and all(isinstance(item, str) and item for item in combination)
            for combination in combinations
        ):
            raise ComputeCapabilityError(
                f"BMD Compute stage_modifier_support[{index}].supported_modifier_combinations "
                "must be a list of lists of modifier names."
            )


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)

    if not isinstance(value, Mapping):
        raise ComputeCapabilityError(f"BMD Compute capability payload missing object field: {key}")

    return value


def _required_list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)

    if not isinstance(value, list):
        raise ComputeCapabilityError(f"BMD Compute capability payload missing list field: {key}")

    return value


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeCapabilityError(f"BMD Compute {label} must be an object.")

    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)

    if not isinstance(value, str) or not value:
        raise ComputeCapabilityError(f"BMD Compute field must be a non-empty string: {key}")

    return value


def _first_error_line(stderr: str | None) -> str:
    for line in (stderr or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped

    return ""


PRODUCER_CAPABILITY = "producer_capability"

COMPUTE_POLICY_AVAILABLE = "available"
COMPUTE_POLICY_UNAVAILABLE = "unavailable"
COMPUTE_POLICY_NOT_CONFIGURED = "not_configured"

POLICY_ALIGNMENT_SAME = "same_commit"
POLICY_ALIGNMENT_SAME_UNVERIFIED = "same_commit_worktree_not_verified_clean"
POLICY_ALIGNMENT_DIFFERENT = "different_commit"
POLICY_ALIGNMENT_UNKNOWN = "unknown"


@dataclass(frozen=True)
class ComputePolicyObservation:
    """BMD Compute modifier/input-effect policy as read from the configured checkout.

    ``status`` distinguishes evidence that is unavailable (producer not
    configured, failed, or did not declare policies) from available evidence
    that happens to declare no policies. The source fields describe the
    *current* configured checkout, not the checkout that produced any
    historical run.
    """

    status: str
    policies: tuple[Mapping[str, Any], ...] = ()
    reason: str | None = None
    repository: str | None = None
    commit: str | None = None
    dirty: bool | None = None
    provenance_available: bool | None = None
    evidence_type: str = PRODUCER_CAPABILITY

    @property
    def available(self) -> bool:
        return self.status == COMPUTE_POLICY_AVAILABLE


def compute_policy_not_configured(reason: str) -> ComputePolicyObservation:
    return ComputePolicyObservation(status=COMPUTE_POLICY_NOT_CONFIGURED, reason=reason)


def observe_compute_policies(
    repository: GitRepositoryResource,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> ComputePolicyObservation:
    """Read modifier policies and their source provenance from BMD Compute."""

    try:
        if runner is None:
            capabilities = inspect_compute_capabilities(repository)
        else:
            capabilities = inspect_compute_capabilities(repository, runner=runner)
    except ComputeCapabilityError as exc:
        return ComputePolicyObservation(status=COMPUTE_POLICY_UNAVAILABLE, reason=str(exc))

    source = capabilities.source
    provenance = {
        "repository": source.get("repository"),
        "commit": source.get("commit"),
        "dirty": source.get("dirty"),
        "provenance_available": source.get("provenance_available"),
    }
    policies = capabilities.payload.get("modifier_policies")
    if not isinstance(policies, list):
        return ComputePolicyObservation(
            status=COMPUTE_POLICY_UNAVAILABLE,
            reason="BMD Compute capability payload does not declare modifier_policies",
            **provenance,
        )
    return ComputePolicyObservation(
        status=COMPUTE_POLICY_AVAILABLE,
        policies=tuple(policy for policy in policies if isinstance(policy, Mapping)),
        **provenance,
    )


def compute_policy_alignment(
    policy: ComputePolicyObservation | None,
    run_producer_git: Mapping[str, Any],
) -> str:
    """Compare the current policy checkout with the commit that produced a run."""

    if policy is None or not policy.available:
        return POLICY_ALIGNMENT_UNKNOWN
    run_commit = run_producer_git.get("git_commit")
    if not isinstance(policy.commit, str) or not policy.commit:
        return POLICY_ALIGNMENT_UNKNOWN
    if not isinstance(run_commit, str) or not run_commit:
        return POLICY_ALIGNMENT_UNKNOWN
    if policy.commit != run_commit:
        return POLICY_ALIGNMENT_DIFFERENT
    run_state = str(run_producer_git.get("state") or "").lower()
    if policy.dirty is not False or run_state != "clean":
        # Either checkout may carry uncommitted changes (or its state is unknown).
        return POLICY_ALIGNMENT_SAME_UNVERIFIED
    return POLICY_ALIGNMENT_SAME
