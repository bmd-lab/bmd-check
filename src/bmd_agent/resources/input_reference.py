from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
from typing import Any, Callable, Mapping

from bmd_agent.config import GitRepositoryResource


# Compatibility with BMD Compute's input-reference producer is established only
# from machine-readable fields: ``schema_version``, ``producer.repository`` and
# ``reference_phase``. The producer's ``scope`` text is human-readable and is
# never compared as a contract.
SCHEMA_VERSION = 1
PRODUCER_MODULE = "backend.calculations.input_reference"
PRODUCER_REPOSITORY_ID = "bmd_compute"
REFERENCE_PHASE = "generated_pre_execution"

Runner = Callable[..., subprocess.CompletedProcess[str]]


class InputReferenceError(RuntimeError):
    """Raised when BMD Compute input-reference evidence is unavailable."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "input_reference_unavailable",
        returncode: int | None = None,
        stderr_summary: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.returncode = returncode
        self.stderr_summary = stderr_summary


@dataclass(frozen=True)
class InputReferenceResponse:
    """Validated BMD Compute input-reference schema-v1 payload."""

    payload: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    @property
    def producer(self) -> Mapping[str, Any]:
        return self.payload["producer"]

    @property
    def reference_phase(self) -> str:
        return str(self.payload["reference_phase"])

    @property
    def workflow(self) -> Mapping[str, Any]:
        value = self.payload.get("workflow")
        return value if isinstance(value, Mapping) else {}

    @property
    def stages(self) -> tuple[Mapping[str, Any], ...]:
        reference = self.payload.get("reference")
        if not isinstance(reference, Mapping):
            return ()
        stages = reference.get("stages")
        if not isinstance(stages, list):
            return ()
        return tuple(stage for stage in stages if isinstance(stage, Mapping))

    @property
    def error(self) -> Mapping[str, Any]:
        value = self.payload.get("error")
        return value if isinstance(value, Mapping) else {}


def generate_input_reference(
    repository: GitRepositoryResource,
    request: Mapping[str, Any],
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> InputReferenceResponse:
    """Invoke BMD Compute's configured read-only input-reference producer."""

    if not repository.path.is_dir():
        raise InputReferenceError(
            f"Configured BMD Compute checkout does not exist: {repository.path}",
            kind="missing_checkout",
        )

    if repository.capability_python is None:
        raise InputReferenceError(
            "BMD Compute repository configuration is missing capability_python.",
            kind="missing_interpreter",
        )

    if not repository.capability_python.is_file():
        raise InputReferenceError(
            f"Configured BMD Compute capability Python does not exist: {repository.capability_python}",
            kind="missing_interpreter",
        )

    request_text = json.dumps(_json_safe_value(request), sort_keys=True)
    command = [str(repository.capability_python), "-B", "-m", PRODUCER_MODULE]

    try:
        completed = runner(
            command,
            cwd=repository.path,
            input=request_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise InputReferenceError(
            "BMD Compute input-reference producer timed out.",
            kind="timeout",
        ) from exc
    except subprocess.CalledProcessError as exc:
        try:
            response = _parse_response(exc.stdout)
        except InputReferenceError:
            response = None
        if response is not None:
            return response
        raise InputReferenceError(
            "BMD Compute input-reference producer failed.",
            kind="producer_failed",
            returncode=exc.returncode,
            stderr_summary=_stderr_summary(exc.stderr),
        ) from exc
    except OSError as exc:
        raise InputReferenceError(
            f"Could not execute BMD Compute input-reference producer: {exc}",
            kind="execution_failed",
        ) from exc

    try:
        response = _parse_response(completed.stdout)
    except InputReferenceError as exc:
        if completed.returncode == 0:
            raise
        response = None

    if response is not None:
        return response

    if completed.returncode != 0:
        raise InputReferenceError(
            "BMD Compute input-reference producer failed without a valid contract response.",
            kind="producer_failed",
            returncode=completed.returncode,
            stderr_summary=_stderr_summary(completed.stderr),
        )

    raise InputReferenceError(
        "BMD Compute input-reference producer emitted malformed JSON.",
        kind="malformed_json",
    )


def parse_input_reference_payload(output: str) -> InputReferenceResponse:
    """Parse and validate BMD Compute input-reference schema v1."""

    try:
        raw_payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise InputReferenceError(
            "BMD Compute input-reference producer emitted malformed JSON.",
            kind="malformed_json",
        ) from exc

    if not isinstance(raw_payload, dict):
        raise InputReferenceError(
            "BMD Compute input-reference payload must be a JSON object.",
            kind="malformed_payload",
        )

    payload = dict(raw_payload)
    _validate_payload(payload)
    return InputReferenceResponse(payload=payload)


def _parse_response(output: object) -> InputReferenceResponse:
    if not isinstance(output, str) or not output.strip():
        raise InputReferenceError(
            "BMD Compute input-reference producer emitted no JSON.",
            kind="missing_json",
        )
    return parse_input_reference_payload(output)


def _validate_payload(payload: Mapping[str, Any]) -> None:
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise InputReferenceError(
            f"Unsupported BMD Compute input-reference schema_version: {schema_version!r}",
            kind="unsupported_schema",
        )

    scope = payload.get("scope")
    if scope is not None and not isinstance(scope, str):
        raise InputReferenceError(
            "BMD Compute input-reference scope must be text when present.",
            kind="malformed_payload",
        )

    status = payload.get("status")
    if status not in {"ok", "unsupported", "error"}:
        raise InputReferenceError(
            f"BMD Compute input-reference payload has unexpected status: {status!r}",
            kind="unexpected_status",
        )

    if payload.get("reference_phase") != REFERENCE_PHASE:
        raise InputReferenceError(
            "BMD Compute input-reference payload has an unexpected reference_phase.",
            kind="unexpected_reference_phase",
        )

    _validate_producer(_required_mapping(payload, "producer"))
    _required_mapping(payload, "contract")

    if status == "ok":
        _required_mapping(payload, "request")
        _required_mapping(payload, "workflow")
        reference = _required_mapping(payload, "reference")
        _validate_reference_stages(_required_list(reference, "stages"))
    else:
        _required_mapping(payload, "error")


def _validate_producer(producer: Mapping[str, Any]) -> None:
    repository = _required_str(producer, "repository")
    if repository != PRODUCER_REPOSITORY_ID:
        raise InputReferenceError(
            f"Input-reference payload is not from BMD Compute: producer.repository={repository!r}",
            kind="unexpected_producer",
        )
    source = producer.get("source")
    if source is None:
        return
    if not isinstance(source, Mapping):
        raise InputReferenceError(
            "BMD Compute input-reference producer.source must be an object or null.",
            kind="malformed_payload",
        )


def _validate_reference_stages(stages: list[Any]) -> None:
    if not stages:
        raise InputReferenceError(
            "BMD Compute input-reference payload contains no reference stages.",
            kind="malformed_payload",
        )

    for index, stage in enumerate(stages):
        mapping = _as_mapping(stage, f"reference.stages[{index}]")
        _required_str(mapping, "stage_type")
        _required_str(mapping, "theory")
        incar = _required_mapping(mapping, "incar")
        if not isinstance(incar.get("settings"), Mapping):
            raise InputReferenceError(
                f"BMD Compute reference.stages[{index}].incar.settings must be an object.",
                kind="malformed_payload",
            )
        _required_mapping(mapping, "kpoints")
        _required_mapping(mapping, "poscar")


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise InputReferenceError(
            f"BMD Compute input-reference payload missing object field: {key}",
            kind="malformed_payload",
        )
    return value


def _required_list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise InputReferenceError(
            f"BMD Compute input-reference payload missing list field: {key}",
            kind="malformed_payload",
        )
    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise InputReferenceError(
            f"BMD Compute input-reference field must be a non-empty string: {key}",
            kind="malformed_payload",
        )
    return value


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InputReferenceError(
            f"BMD Compute {label} must be an object.",
            kind="malformed_payload",
        )
    return value


def _stderr_summary(stderr: object) -> str | None:
    if stderr is None:
        return None
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.lower().startswith("traceback"):
            return stripped[:240]
    return None


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
