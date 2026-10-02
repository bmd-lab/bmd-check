"""check-input reports comparisons with one Compute-generated reference, nothing more.

The cases replay real BMD Compute behaviour: the inputs a Desired Output
workflow generates (including Compute's automatic treatments) and the
input-reference payloads Compute returns for the exact standalone requests
Agent sends. See fixtures/check_input/SOURCE.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest

from bmd_agent import cli
from bmd_agent.config import ResourceRegistry
from bmd_agent.resources.input_check import (
    DIFFERS_FROM_THIS_REFERENCE,
    INSUFFICIENT_INFORMATION,
    MATCH,
    MATCHES_THIS_REFERENCE,
    NO_REFERENCE_FOR_THIS_REQUEST,
    REFERENCE_CONTEXT_LIMITATION,
    check_remote_input_directory,
)

import test_input_check as base


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "check_input"
FIXTURE = FIXTURE_DIR / "compute_generated_cases.json"
DATA = json.loads(FIXTURE.read_text(encoding="utf-8"))
# BMD Compute v1.0.0; generation provenance recorded in each payload.
COMPUTE_COMMIT = "a746155b487f903a167eca6b3c92e860aaf8c7f5"
HISTORICAL_FIXTURE = FIXTURE_DIR / "historical_47546ad" / "compute_generated_cases.json"
HISTORICAL_DATA = json.loads(HISTORICAL_FIXTURE.read_text(encoding="utf-8"))
RETIRED_CLASSIFICATIONS = ("COMPLIANT", "NONSTANDARD", "SUPPORTED BUT", "UNSUPPORTED")


def subdir(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def replay(tmp_path: Path, case: str, *, stage: str, theory: str, modifiers=(), requests=None):
    files = {
        f"{base.REMOTE_DIR}/{name}": text.encode("utf-8")
        for name, text in DATA["cases"][case]["files"].items()
    }
    known = [item for item in DATA["references"] if item["case"] == case]

    def producer(command, **kwargs):
        request = json.loads(str(kwargs["input"]))
        if requests is not None:
            requests.append(request)
        for item in known:
            if item["request"] == request:
                payload = item["payload"]
                return base.subprocess.CompletedProcess(
                    command,
                    0 if payload["status"] != "error" else 2,
                    stdout=json.dumps(payload),
                    stderr="",
                )
        raise AssertionError(f"Agent sent a request Compute fixtures do not cover: {request}")

    return check_remote_input_directory(
        base.cluster(),
        base.repository(tmp_path),
        base.REMOTE_DIR,
        stage=stage,
        theory=theory,
        modifiers=modifiers,
        remote_runner=base.remote_runner_for(files),
        producer_runner=producer,
    )


def differing(observation) -> dict[str, str]:
    return {item.setting: item.status for item in observation.incar_comparisons if item.status != MATCH}


def test_fixture_identity_matches_source_record():
    source = (FIXTURE_DIR / "SOURCE.md").read_text(encoding="utf-8")
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() in source
    assert hashlib.sha256(HISTORICAL_FIXTURE.read_bytes()).hexdigest() in source


def test_current_fixture_was_generated_by_compute_v1():
    commits = {item["payload"]["producer"]["source"]["commit"] for item in DATA["references"]}
    assert commits == {COMPUTE_COMMIT}
    assert {item["payload"]["producer"]["source"]["dirty"] for item in DATA["references"]} == {False}


def test_fixture_cases_are_real_compute_desired_output_inputs():
    cases = DATA["cases"]
    assert cases["graphite_auto_d3"]["applied_treatments"] == ["dispersion"]
    assert cases["pbte_auto_soc"]["applied_treatments"] == ["soc"]
    # Compute v1 applies automatic DFT+U to NiO (O anion, Ni with an MP U value).
    assert cases["nio_auto_spin"]["applied_treatments"] == ["spin_polarized", "dft_u"]
    assert cases["si_pbe_static"]["applied_treatments"] == []


# --- reproduced cases ---------------------------------------------------------------------


def test_plain_si_pbe_static_matches_this_reference(tmp_path):
    observation = replay(tmp_path, "si_pbe_static", stage="static", theory="pbe")

    assert observation.overall_status == MATCHES_THIS_REFERENCE
    assert differing(observation) == {}
    assert observation.kpoints_comparison.status == MATCH


def test_graphite_automatic_d3_input_differs_from_modifier_free_reference_with_limitation(tmp_path):
    observation = replay(tmp_path, "graphite_auto_d3", stage="static", theory="pbe")

    assert observation.overall_status == DIFFERS_FROM_THIS_REFERENCE
    assert differing(observation) == {"IVDW": "supplied_extra"}
    assert REFERENCE_CONTEXT_LIMITATION in observation.limitations
    assert "requested modifiers: none" in observation.reference_description
    assert "Desired Output automatic-treatment resolution: not applied" in observation.reference_description


def test_pbte_automatic_soc_input_differs_from_modifier_free_reference(tmp_path):
    observation = replay(tmp_path, "pbte_auto_soc", stage="static", theory="pbe")

    assert observation.overall_status == DIFFERS_FROM_THIS_REFERENCE
    assert {"LSORBIT", "LNONCOLLINEAR", "ISYM", "SAXIS"} <= set(differing(observation))
    assert REFERENCE_CONTEXT_LIMITATION in observation.limitations


def test_pbte_with_explicit_soc_modifier_matches_this_reference(tmp_path):
    requests = []
    observation = replay(
        tmp_path,
        "pbte_auto_soc",
        stage="static",
        theory="pbe",
        modifiers=("soc",),
        requests=requests,
    )

    assert requests[0]["workflow_spec"]["stages"][0]["modifiers"] == ["soc"]
    assert observation.overall_status == MATCHES_THIS_REFERENCE, differing(observation)
    assert "requested modifiers: soc" in observation.reference_description


def test_nio_automatic_spin_and_dft_u_input_differs_from_modifier_free_reference(tmp_path):
    observation = replay(tmp_path, "nio_auto_spin", stage="static", theory="pbe")

    assert observation.overall_status == DIFFERS_FROM_THIS_REFERENCE
    assert differing(observation) == {
        "ISPIN": "differs_from_reference",
        "MAGMOM": "supplied_extra",
        "LDAU": "supplied_extra",
        "LDAUJ": "supplied_extra",
        "LDAUL": "supplied_extra",
        "LDAUPRINT": "supplied_extra",
        "LDAUTYPE": "supplied_extra",
        "LDAUU": "supplied_extra",
        "LMAXMIX": "supplied_extra",
    }
    # LDAU in the supplied INCAR is reported as a difference only. Agent does
    # not conclude that Compute applied automatic DFT+U from INCAR tags; the
    # standalone reference itself had no automatic-treatment resolution.
    assert "Desired Output automatic-treatment resolution: not applied" in observation.reference_description
    assert not any("dft+u" in item.lower() or "dft_u" in item.lower() for item in observation.limitations)
    assert not any("automatic dft+u" in item.lower() for item in observation.reference_description)


def test_historical_47546ad_reference_payloads_still_parse(tmp_path):
    # Deployed Compute checkouts may predate v1; their schema-v1 payloads stay readable.
    from bmd_agent.resources.input_reference import parse_input_reference_payload

    statuses = {
        parse_input_reference_payload(json.dumps(item["payload"])).status
        for item in HISTORICAL_DATA["references"]
    }
    assert statuses == {"ok", "unsupported", "error"}
    assert HISTORICAL_DATA["cases"]["nio_auto_spin"]["applied_treatments"] == ["spin_polarized"]


@pytest.mark.parametrize(
    ("case", "stage", "reason"),
    [
        ("si_hse06_dos_stage", "dos", "Density of States must follow a converged Static Energy stage."),
        ("si_hse06_band_stage", "band_structure", "Band Structure must follow a converged Static Energy stage."),
    ],
)
def test_standalone_hse06_dos_and_band_refusal_is_no_reference_not_unsupported(tmp_path, case, stage, reason):
    observation = replay(tmp_path, case, stage=stage, theory="hse06")

    assert observation.overall_status == NO_REFERENCE_FOR_THIS_REQUEST
    assert observation.limitations[0] == f"BMD Compute did not generate a reference: {reason}"
    assert any(item.startswith("BMD Compute suggestion:") for item in observation.limitations)
    assert REFERENCE_CONTEXT_LIMITATION in observation.limitations
    assert observation.incar_comparisons == ()


# --- request construction -----------------------------------------------------------------


def test_request_has_no_agent_potcar_functional_or_resources_and_functional_comes_from_compute(tmp_path):
    requests = []
    observation = replay(tmp_path, "si_pbe_static", stage="static", theory="pbe", requests=requests)

    assert "potcar_functional" not in requests[0]
    assert "resources" not in requests[0]
    assert observation.reference.potcar_functional == "PBE_64"
    assert "POTCAR functional reported by BMD Compute: PBE_64" in observation.reference_description


def test_potcar_functional_is_not_invented_when_compute_does_not_report_one(tmp_path):
    payload = base.producer_payload()
    payload["request"] = {}
    observation = base.run_check(tmp_path, payload=payload)

    assert observation.reference.potcar_functional is None
    assert "POTCAR functional reported by BMD Compute: not reported" in observation.reference_description


def test_modifiers_are_forwarded_verbatim_and_compute_reason_is_retained_for_invalid_combinations(tmp_path):
    requests = []
    unsupported = replay(
        subdir(tmp_path, "first"), "pbte_auto_soc", stage="relax", theory="pbe", modifiers=("soc",), requests=requests
    )
    unknown = replay(
        subdir(tmp_path, "second"), "pbte_auto_soc", stage="static", theory="pbe", modifiers=("not_a_modifier",), requests=requests
    )

    assert [request["workflow_spec"]["stages"][0]["modifiers"] for request in requests] == [
        ["soc"],
        ["not_a_modifier"],
    ]
    assert unsupported.overall_status == NO_REFERENCE_FOR_THIS_REQUEST
    assert unsupported.limitations[0] == (
        "BMD Compute did not generate a reference: Geometry Optimisation with PBE is not "
        "available with Spin-Orbit Coupling (SOC)."
    )
    assert unknown.overall_status == INSUFFICIENT_INFORMATION
    assert unknown.limitations[0] == (
        "BMD Compute did not generate a reference: Unsupported Modifier: 'not_a_modifier'"
    )


def test_every_compute_response_carries_a_full_reference_description(tmp_path):
    observation = replay(tmp_path, "graphite_auto_d3", stage="static", theory="pbe")

    assert observation.reference_description == (
        "requested stage/theory: static / pbe",
        "requested modifiers: none",
        f"structure: supplied POSCAR from {base.REMOTE_DIR}/POSCAR",
        "reference phase: generated_pre_execution (pre-execution inputs BMD Compute would generate)",
        "previous-stage context: none (standalone single-stage request)",
        "Desired Output automatic-treatment resolution: not applied",
        f"BMD Compute commit: {COMPUTE_COMMIT}",
        "POTCAR functional reported by BMD Compute: PBE_64",
    )


# --- insufficient-information paths are unchanged ---------------------------------------------


def test_insufficient_paths_do_not_claim_a_reference(tmp_path):
    files = base.remote_files()
    del files[f"{base.REMOTE_DIR}/INCAR"]
    missing = base.run_check(subdir(tmp_path, "missing"), files=files)
    malformed = base.run_check(subdir(tmp_path, "malformed"), payload="{not json")

    for observation in (missing, malformed):
        assert observation.overall_status == INSUFFICIENT_INFORMATION
        assert observation.reference_description == ()
        assert REFERENCE_CONTEXT_LIMITATION not in observation.limitations


# --- user-facing language -------------------------------------------------------------------


def test_classification_vocabulary_is_factual():
    from bmd_agent.resources import input_check

    statuses = {
        value
        for name, value in vars(input_check).items()
        if name.isupper() and isinstance(value, str) and name.endswith(("REFERENCE", "REQUEST", "INFORMATION"))
    }
    assert statuses >= {
        MATCHES_THIS_REFERENCE,
        DIFFERS_FROM_THIS_REFERENCE,
        NO_REFERENCE_FOR_THIS_REQUEST,
        INSUFFICIENT_INFORMATION,
    }
    for retired in ("COMPLIANT", "SUPPORTED_BUT_NONSTANDARD", "UNSUPPORTED"):
        assert not hasattr(input_check, retired)


@pytest.mark.parametrize(
    ("case", "stage", "theory", "modifiers"),
    [
        ("si_pbe_static", "static", "pbe", ()),
        ("graphite_auto_d3", "static", "pbe", ()),
        ("si_hse06_dos_stage", "dos", "hse06", ()),
        ("pbte_auto_soc", "static", "pbe", ("soc",)),
    ],
)
def test_cli_output_uses_no_retired_classification_language(tmp_path, monkeypatch, capsys, case, stage, theory, modifiers):
    observation = replay(tmp_path, case, stage=stage, theory=theory, modifiers=modifiers)
    registry = ResourceRegistry(repositories={}, clusters={})
    monkeypatch.setattr(cli, "load_resources", lambda: registry)
    monkeypatch.setattr(cli, "powerslurm_cluster", lambda registry: base.cluster())
    monkeypatch.setattr(cli, "bmd_compute_repository", lambda registry: object())
    monkeypatch.setattr(cli, "check_remote_input_directory", lambda *args, **kwargs: observation)
    argv = ["check-input", base.REMOTE_DIR, "--stage", stage, "--theory", theory]
    if modifiers:
        argv += ["--modifiers", ",".join(modifiers)]

    assert cli.main(argv) == 0
    output = capsys.readouterr().out

    assert observation.overall_status in output
    assert "Generated reference:" in output
    # Compute's own messages are quoted verbatim and may use any wording;
    # everything Agent adds must avoid the retired classification language.
    agent_text = output
    for message in filter(None, (
        observation.reference.error_code,
        observation.reference.error_message,
        observation.reference.error_suggestion,
    )):
        agent_text = agent_text.replace(message, "")
    for retired in RETIRED_CLASSIFICATIONS:
        assert retired not in agent_text.upper()
    assert not re.search(r"\bSTANDARD\b", agent_text.upper())


def test_cli_forwards_modifiers_verbatim(monkeypatch, capsys):
    received = {}
    registry = ResourceRegistry(repositories={}, clusters={})
    monkeypatch.setattr(cli, "load_resources", lambda: registry)
    monkeypatch.setattr(cli, "powerslurm_cluster", lambda registry: base.cluster())
    monkeypatch.setattr(cli, "bmd_compute_repository", lambda registry: object())

    def capture(*args, **kwargs):
        received.update(kwargs)
        raise ValueError("stop after capturing arguments")

    monkeypatch.setattr(cli, "check_remote_input_directory", capture)
    cli.main(["check-input", base.REMOTE_DIR, "--stage", "static", "--theory", "pbe", "--modifiers", "soc, Dispersion"])

    assert received["modifiers"] == ("soc", "Dispersion")


def test_cli_rejects_empty_modifier_entries(capsys):
    assert cli.main(["check-input", base.REMOTE_DIR, "--stage", "static", "--theory", "pbe", "--modifiers", "soc,,dft_u"]) == 2
    assert "--modifiers must be a comma-separated list" in capsys.readouterr().out
