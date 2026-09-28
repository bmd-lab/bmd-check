import json
from pathlib import Path
import subprocess

import pytest

from bmd_agent.config import GitRepositoryResource, ResourceRegistry
from bmd_agent.resources.compute import (
    ComputeCapabilityError,
    inspect_compute_capabilities,
    parse_capability_payload,
    supported_capability_pairs,
)
from bmd_agent import cli


def schema_v1_payload(*, dirty: bool = False) -> dict:
    return {
        "schema_version": 1,
        "scope": "BMD Compute executable implementation, not a methodology authority",
        "source": {
            "repository": "bmd_compute",
            "commit": "05eacdb81234567890abcdef",
            "dirty": dirty,
            "provenance_available": True,
            "unavailable_reason": None,
        },
        "contract": {
            "base_stage_definitions": "Theory-neutral stage definitions from list_stage_definitions().",
            "capabilities": "Supported stage/theory descriptions from describe_stage(); unsupported combinations are not invented.",
        },
        "base_stage_definitions": [
            {"stage_type": "relax"},
            {"stage_type": "static"},
            {"stage_type": "dos"},
            {"stage_type": "band_structure"},
        ],
        "capabilities": [
            capability("pbe", "band_structure"),
            capability("hse06", "band_structure"),
            capability("pbe", "dos"),
            capability("pbe", "relax"),
            capability("hse06", "relax"),
            capability("pbe", "static"),
            capability("hse06", "static"),
        ],
    }


def capability(theory: str, stage_type: str) -> dict:
    return {
        "scope": "BMD Compute executable implementation, not a methodology authority",
        "stage_type": stage_type,
        "theory": theory,
        "theory_supported_for_stage": True,
        "selected_atomate2": {"maker": "producer.supplied.Maker"},
    }


def repository(path: Path, python: Path | None) -> GitRepositoryResource:
    return GitRepositoryResource(
        key="bmd_compute",
        name="BMD Compute",
        path=path,
        role="compute",
        access="read_only",
        protected=True,
        live=True,
        capability_python=python,
    )


def test_valid_schema_v1_consumption_preserves_payload(tmp_path: Path) -> None:
    checkout = tmp_path / "bmd_compute"
    checkout.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    payload = schema_v1_payload()
    calls: list[tuple[list[str], Path]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs["cwd"]))
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 20
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    capabilities = inspect_compute_capabilities(
        repository(checkout, python),
        runner=runner,
    )

    assert capabilities.payload == payload
    assert calls == [
        ([str(python), "-B", "-m", "backend.calculations.capabilities"], checkout)
    ]


def test_all_advertised_capabilities_are_preserved() -> None:
    capabilities = parse_capability_payload(json.dumps(schema_v1_payload()))

    assert supported_capability_pairs(capabilities) == (
        ("band_structure", "pbe"),
        ("band_structure", "hse06"),
        ("dos", "pbe"),
        ("relax", "pbe"),
        ("relax", "hse06"),
        ("static", "pbe"),
        ("static", "hse06"),
    )


def test_unsupported_combinations_are_not_invented() -> None:
    capabilities = parse_capability_payload(json.dumps(schema_v1_payload()))

    assert ("dos", "hse06") not in supported_capability_pairs(capabilities)


def test_provenance_and_dirty_state_are_preserved() -> None:
    capabilities = parse_capability_payload(json.dumps(schema_v1_payload(dirty=True)))

    assert capabilities.source["repository"] == "bmd_compute"
    assert capabilities.source["commit"] == "05eacdb81234567890abcdef"
    assert capabilities.source["dirty"] is True


def test_malformed_json_is_rejected() -> None:
    with pytest.raises(ComputeCapabilityError, match="malformed JSON"):
        parse_capability_payload("{not json")


def test_unsupported_schema_version_is_rejected() -> None:
    payload = schema_v1_payload()
    payload["schema_version"] = 2

    with pytest.raises(ComputeCapabilityError, match="Unsupported"):
        parse_capability_payload(json.dumps(payload))


def test_missing_required_fields_are_rejected() -> None:
    payload = schema_v1_payload()
    del payload["source"]

    with pytest.raises(ComputeCapabilityError, match="source"):
        parse_capability_payload(json.dumps(payload))


def test_invalid_capability_records_are_rejected() -> None:
    payload = schema_v1_payload()
    payload["capabilities"][0]["theory_supported_for_stage"] = False

    with pytest.raises(ComputeCapabilityError, match="not an explicitly supported"):
        parse_capability_payload(json.dumps(payload))


def test_producer_failure_is_reported(tmp_path: Path) -> None:
    checkout = tmp_path / "bmd_compute"
    checkout.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, command, stderr="boom\n")

    with pytest.raises(ComputeCapabilityError, match="producer failed.*boom"):
        inspect_compute_capabilities(repository(checkout, python), runner=runner)


def test_timeout_is_reported(tmp_path: Path) -> None:
    checkout = tmp_path / "bmd_compute"
    checkout.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout=20)

    with pytest.raises(ComputeCapabilityError, match="timed out"):
        inspect_compute_capabilities(repository(checkout, python), runner=runner)


def test_missing_checkout_is_reported(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")

    with pytest.raises(ComputeCapabilityError, match="checkout does not exist"):
        inspect_compute_capabilities(repository(tmp_path / "missing", python))


def test_missing_interpreter_is_reported(tmp_path: Path) -> None:
    checkout = tmp_path / "bmd_compute"
    checkout.mkdir()

    with pytest.raises(ComputeCapabilityError, match="Python does not exist"):
        inspect_compute_capabilities(repository(checkout, tmp_path / "missing-python"))


def test_cli_summary_uses_declared_capabilities_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = repository(tmp_path, tmp_path / "python")
    registry = ResourceRegistry(repositories={"bmd_compute": repo}, clusters={})
    capabilities = parse_capability_payload(json.dumps(schema_v1_payload()))

    monkeypatch.setattr(cli, "load_resources", lambda: registry)
    monkeypatch.setattr(cli, "inspect_compute_capabilities", lambda repository: capabilities)

    exit_code = cli.main(["compute"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "BMD Compute Capabilities" in captured.out
    assert "repository: bmd_compute" in captured.out
    assert "commit:     05eacdb81234" in captured.out
    assert "state:      clean" in captured.out
    assert "HSE06  Band Structure" in captured.out
    assert "PBE    DOS" in captured.out
    assert "HSE06  DOS" not in captured.out


LEGACY_COMPUTE_SCOPE = "BMD Compute executable implementation, not a methodology authority"
CURRENT_COMPUTE_SCOPE = (
    "BMD Compute executable calculation methodology: the stages, theories, "
    "settings and treatments this BMD Compute checkout implements and can execute"
)


@pytest.mark.parametrize(
    "scope",
    (LEGACY_COMPUTE_SCOPE, CURRENT_COMPUTE_SCOPE, "Any reworded human description"),
)
def test_scope_prose_is_not_a_compatibility_contract(scope: str) -> None:
    payload = schema_v1_payload()
    payload["scope"] = scope
    for record in payload["capabilities"]:
        record["scope"] = scope

    capabilities = parse_capability_payload(json.dumps(payload))

    assert capabilities.scope == scope
    assert supported_capability_pairs(capabilities)


def test_missing_scope_is_accepted_and_not_displayed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = schema_v1_payload()
    del payload["scope"]
    capabilities = parse_capability_payload(json.dumps(payload))
    assert capabilities.scope is None

    repo = repository(tmp_path, tmp_path / "python")
    registry = ResourceRegistry(repositories={"bmd_compute": repo}, clusters={})
    monkeypatch.setattr(cli, "load_resources", lambda: registry)
    monkeypatch.setattr(cli, "inspect_compute_capabilities", lambda repository: capabilities)

    assert cli.main(["compute"]) == 0
    captured = capsys.readouterr()
    assert "Scope:" not in captured.out
    assert "HSE06  Band Structure" in captured.out


def test_reworded_scope_is_displayed_verbatim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = schema_v1_payload()
    payload["scope"] = CURRENT_COMPUTE_SCOPE
    capabilities = parse_capability_payload(json.dumps(payload))
    repo = repository(tmp_path, tmp_path / "python")
    registry = ResourceRegistry(repositories={"bmd_compute": repo}, clusters={})
    monkeypatch.setattr(cli, "load_resources", lambda: registry)
    monkeypatch.setattr(cli, "inspect_compute_capabilities", lambda repository: capabilities)

    assert cli.main(["compute"]) == 0
    assert CURRENT_COMPUTE_SCOPE in capsys.readouterr().out


def test_non_text_scope_is_rejected() -> None:
    payload = schema_v1_payload()
    payload["scope"] = {"authority": "anything"}

    with pytest.raises(ComputeCapabilityError, match="scope must be text"):
        parse_capability_payload(json.dumps(payload))


@pytest.mark.parametrize("schema_version", (None, 0, 2, "1"))
def test_compatibility_requires_machine_readable_schema_version(schema_version: object) -> None:
    payload = schema_v1_payload()
    if schema_version is None:
        del payload["schema_version"]
    else:
        payload["schema_version"] = schema_version

    with pytest.raises(ComputeCapabilityError, match="Unsupported"):
        parse_capability_payload(json.dumps(payload))


def test_compatibility_requires_bmd_compute_repository_identity() -> None:
    payload = schema_v1_payload()
    payload["source"]["repository"] = "bmdex"

    with pytest.raises(ComputeCapabilityError, match="not from BMD Compute"):
        parse_capability_payload(json.dumps(payload))
