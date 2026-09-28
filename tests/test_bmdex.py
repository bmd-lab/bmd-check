from __future__ import annotations

from dataclasses import replace
import json
import subprocess
from pathlib import Path

import pytest

from bmd_agent import cli
from bmd_agent.config import GitRepositoryResource, ResourceRegistry, parse_resources
from bmd_agent.resources import run as run_resource
from bmd_agent.resources.bmdex import (
    BMDEX_COMPOSITION_CONTEXT,
    DOMAIN_CONTEXT_EVIDENCE_TYPE,
    DOMAIN_CONTEXT_PRODUCER_MODULE,
    EVIDENCE_TYPE,
    PRODUCER_MODULE,
    BmdexCompositionError,
    BmdexCompositionEvidence,
    BmdexDomainContextError,
    EnrichedScientificContext,
    bmdex_repository,
    build_bmdex_domain_query,
    build_bmdex_domain_query_for_job,
    enrich_lifecycle_with_bmdex_domain_context,
    enrich_scientific_context_with_bmdex,
    inspect_bmdex_domain_context,
    inspect_bmdex_composition_context,
    parse_bmdex_domain_context_payload,
    parse_bmdex_composition_payload,
    _contextual_assessment,
)
from bmd_agent.resources.context import (
    EvidenceGap,
    ScientificContext,
    ScientificIdentitySummary,
    build_scientific_context,
)
from bmd_agent.resources.custodian import (
    assess_termination_evidence,
    parse_custodian_json,
)
from bmd_agent.resources.lifecycle import (
    BmdWorkflowDiscovery,
    LifecycleAnalysis,
    LifecycleState,
    LocalExecutionDiagnostics,
    LocalLogDiagnostic,
    LocalStageBinding,
)
from bmd_agent.resources.run import (
    DirectVaspInspection,
    ElectronicIterationObservation,
    IncarObservation,
    JobInspection,
    ScientificResult,
    StageTrajectoryObservation,
)


def base_context(
    *,
    formula: str | None = "MnCu5",
    gaps: tuple[EvidenceGap, ...] = (),
) -> ScientificContext:
    return ScientificContext(
        job=JobInspection(
            job_id="21153721",
            scheduler=None,
            scheduler_error=None,
            scheduler_work_dir=None,
            calculation_directory=None,
            calculation_type="direct VASP",
            calculation_reason=None,
        ),
        identity=ScientificIdentitySummary(
            job_id="21153721",
            calculation_type="direct VASP",
            formula=formula,
            site_count=24 if formula else None,
            structure_evidence_type=run_resource.PYMATGEN_DERIVED if formula else None,
            executed_input_indicators={
                "spin_polarized": {
                    "evidence_type": run_resource.EXECUTED_INPUT,
                    "parameter": "ISPIN",
                    "value": True,
                    "status": "available",
                    "source_values": {"retained_incar:work_dir": 2},
                },
            } if formula else {},
            executed_input_evidence_type=run_resource.EXECUTED_INPUT if formula else None,
        ),
        evidence_gaps=gaps,
    )


def repository(
    tmp_path: Path,
    *,
    include_checkout: bool = True,
    include_python: bool = True,
    capability_python: bool = True,
) -> GitRepositoryResource:
    checkout = tmp_path / "BMDex"
    python = tmp_path / "env" / "bin" / "python"
    if include_checkout:
        checkout.mkdir()
    if include_python:
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("", encoding="utf-8")
    return GitRepositoryResource(
        key="bmdex",
        name="BMDex",
        path=checkout,
        role="curated_knowledge",
        access="read_only",
        protected=True,
        live=False,
        capability_python=python if capability_python else None,
    )


def domain_analysis(*, hybrid: bool = True) -> LifecycleAnalysis:
    directory = Path("/relocated/calculation")
    trajectory = StageTrajectoryObservation(
        stage_index=1,
        stage_label="result_dir",
        stage_type="static",
        theory="hse06" if hybrid else "pbe",
        directory=str(directory),
        completed_ionic_steps=0,
        incomplete_electronic_iteration_count=4,
        recent_incomplete_electronic_iterations=tuple(
            ElectronicIterationObservation(iteration=index, algorithm="DAV")
            for index in range(1, 5)
        ),
    )
    workflow = BmdWorkflowDiscovery(
        workflow_root=directory,
        submission_path=directory / "submission.json",
        submission={},
        workflow_stages=(
            {
                "stage_type": "static",
                "theory": "hse06" if hybrid else "pbe",
                "modifiers": ["soc"] if hybrid else [],
            },
        ),
        stage_bindings=(LocalStageBinding("result_dir", directory, 1),),
        current_stage=LocalStageBinding("result_dir", directory, 1),
        relocated=True,
    )
    settings = {
        "LHFCALC": hybrid,
        "HFSCREEN": 0.2,
        "AEXX": 0.25,
        "ALGO": "Damped",
        "LSORBIT": hybrid,
    }
    return LifecycleAnalysis(
        state=LifecycleState.UNKNOWN,
        directory=directory,
        calculation_kind="BMD Compute",
        message="partial local snapshot",
        bmd_workflow=workflow,
        incar_settings=settings,
        diagnostics=LocalExecutionDiagnostics(
            trajectories=(trajectory,),
            logs=(
                LocalLogDiagnostic(
                    label="std_err.txt",
                    path=directory / "std_err.txt",
                    present=True,
                    messages=("SIGTERM received",),
                ),
            ),
        ),
    )


def domain_job_inspection() -> JobInspection:
    trajectory = StageTrajectoryObservation(
        stage_index=1,
        stage_label="work_dir",
        stage_type="direct_vasp",
        theory="unknown",
        directory="/remote/calculation",
        completed_ionic_steps=0,
        incomplete_electronic_iteration_count=4,
        recent_incomplete_electronic_iterations=tuple(
            ElectronicIterationObservation(iteration=index, algorithm="DAV")
            for index in range(1, 5)
        ),
    )
    direct = DirectVaspInspection(
        directory="/remote/calculation",
        artifacts=(),
        executed_inputs=(
            IncarObservation(
                label="work_dir",
                path="/remote/calculation/INCAR",
                present=True,
                stage_index=1,
                values={
                    "LHFCALC": True,
                    "HFSCREEN": 0.2,
                    "ALGO": "Damped",
                    "LSORBIT": True,
                },
            ),
        ),
        scientific=ScientificResult(source_paths=()),
        trajectory=trajectory,
        assessments=(),
    )
    return JobInspection(
        job_id="21906221",
        scheduler=None,
        scheduler_error=None,
        scheduler_work_dir="/generic/launcher",
        calculation_directory="/remote/calculation",
        calculation_type="direct VASP",
        calculation_reason=None,
        direct_vasp=direct,
    )
def domain_payload(query: dict, *, include_record: bool = True) -> dict:
    records = []
    if include_record:
        records.append(
            {
                "record": {
                    "schema_version": 1,
                    "id": "vasp.test.context",
                    "title": "Producer-owned test context",
                    "evidence_type": "contextual_reference_evidence",
                    "status": "active",
                    "domain": {"code": "VASP"},
                    "topics": ["electronic_iteration_behavior"],
                    "applicability": {"code": "VASP"},
                    "contextual_statement": "Producer-supplied contextual statement.",
                    "diagnostic_relevance": (
                        "This context is relevant but does not diagnose a specific run."
                    ),
                    "limitations": [
                        "This record is contextual reference evidence only.",
                        "It does not recommend changing calculation inputs.",
                    ],
                    "sources": [
                        {
                            "source_type": "VASP Wiki",
                            "title": "Test reference",
                            "authority": "VASP Software GmbH / VASP Wiki",
                            "url": "https://vasp.at/wiki/Test",
                            "retrieved_on": "2026-09-16",
                            "applicability_note": "Supports the bounded test statement.",
                        }
                    ],
                    "record_provenance": {
                        "record_version": 3,
                        "machine_readable_schema": "bmdex.contextual_reference.v1",
                    },
                    "record_path": "vasp/contextual_reference/records/test.json",
                },
                "match": {
                    "matched_fields": [
                        "code",
                        "calculation_family",
                        "functional",
                        "electronic_algorithm",
                        "input_tags",
                    ],
                    "match_type": "deterministic_structured_field_overlap",
                },
            }
        )
    return {
        "schema_version": 1,
        "status": "ok",
        "evidence_type": "contextual_reference_evidence",
        "producer": {
            "name": "BMDex",
            "contract_module": "tools.domain_context.query",
            "git": {"commit": "b" * 40, "dirty": False, "state": "clean"},
        },
        "query": query,
        "records": records,
        "result_count": len(records),
    }


def mncu5_payload() -> dict:
    return {
        "schema_version": 1,
        "status": "ok",
        "evidence_type": "composition_context",
        "producer": {
            "name": "BMDex",
            "contract_module": "tools.composition.context_producer",
            "contract_command": "python -B -m tools.composition.context_producer",
            "git": {
                "commit": "a" * 40,
                "dirty": False,
                "state": "clean",
            },
        },
        "composition": {
            "supplied_formula": "MnCu5",
            "reduced_formula": "MnCu5",
            "elements": ["Mn", "Cu"],
            "stoichiometric_amounts": [
                {"element": "Mn", "amount": 1},
                {"element": "Cu", "amount": 5},
            ],
        },
        "datasets": {
            "element_abundances": {
                "dataset_id": "bmdex.datasets.element_abundances.earth_abundance",
                "path": "datasets/element_abundances/earth-abundance.yaml",
                "quantity": "crustal abundance",
                "units": "mg/kg",
                "source_reference_status": "not_machine_readable",
                "records": [
                    {
                        "element": "Mn",
                        "status": "present",
                        "abundance": 950.0,
                        "units": "mg/kg",
                    },
                    {
                        "element": "Cu",
                        "status": "present",
                        "abundance": 60.0,
                        "units": "mg/kg",
                    },
                ],
            },
            "element_charges": {
                "dataset_id": "bmdex.datasets.element_charges.oxidation_states_84",
                "path": "datasets/element_charges/oxidation_states_84.yaml",
                "term": "representative oxidation states",
                "source_reference_status": "not_machine_readable",
                "records": [
                    {
                        "element": "Mn",
                        "status": "present",
                        "representative_oxidation_states": [2],
                    },
                    {
                        "element": "Cu",
                        "status": "present",
                        "representative_oxidation_states": [1, 2],
                    },
                ],
            },
        },
        "missing_evidence": [],
        "limitations": [
            {
                "code": "abundance_element_level_only",
                "text": (
                    "Element abundance is contextual element-level evidence and "
                    "does not establish compound viability or sustainability."
                ),
            },
            {
                "code": "oxidation_states_element_level_only",
                "text": (
                    "Representative oxidation-state entries are element-level "
                    "reference data and do not establish oxidation states, charge "
                    "balance, stability, or existence of the supplied compound."
                ),
            },
        ],
    }


def he_payload() -> dict:
    payload = mncu5_payload()
    payload["composition"] = {
        "supplied_formula": "He",
        "reduced_formula": "He",
        "elements": ["He"],
        "stoichiometric_amounts": [{"element": "He", "amount": 1}],
    }
    payload["datasets"]["element_abundances"]["records"] = [
        {
            "element": "He",
            "status": "present",
            "abundance": 0.008,
            "units": "mg/kg",
        }
    ]
    payload["datasets"]["element_charges"]["records"] = [
        {
            "element": "He",
            "status": "missing",
            "representative_oxidation_states": None,
        }
    ]
    payload["missing_evidence"] = [
        {
            "dataset": "element_charges",
            "element": "He",
            "reason": "element_not_present_in_dataset",
        }
    ]
    return payload


def structured_error_payload() -> dict:
    return {
        "schema_version": 1,
        "status": "error",
        "evidence_type": "composition_context",
        "producer": {
            "name": "BMDex",
            "contract_module": "tools.composition.context_producer",
            "contract_command": "python -B -m tools.composition.context_producer",
            "git": {
                "commit": "a" * 40,
                "dirty": False,
                "state": "clean",
            },
        },
        "error": {
            "code": "malformed_formula",
            "message": "Formula syntax is unsupported or malformed.",
        },
    }


def completed(payload: dict, *, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["python"],
        returncode=returncode,
        stdout=json.dumps(payload),
        stderr=stderr,
    )


def test_configured_bmdex_repository_and_capability_python_is_parsed() -> None:
    registry = parse_resources(
        {
            "repositories": {
                "bmdex": {
                    "name": "BMDex",
                    "path": "/home/example/projects/BMDex",
                    "role": "curated_knowledge",
                    "access": "read_only",
                    "protected": True,
                    "live": False,
                    "capability_python": "/home/example/micromamba/envs/bmdex/bin/python",
                }
            },
            "clusters": {
                "powerslurm": {
                    "name": "PowerSLURM",
                    "ssh_host": "powerslurm-bmdguest",
                    "partition": "leeburton-pool",
                    "access": "observational",
                    "allowed_remote_roots": ["/bmd-db/guest/flows"],
                }
            },
        }
    )

    repo = bmdex_repository(registry)

    assert repo is not None
    assert repo.capability_python is not None
    assert repo.capability_python.parts[-3:] == ("bmdex", "bin", "python")


def test_exact_fixed_producer_command_cwd_json_stdin_and_shell_false(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    seen = {}

    def runner(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)
        return completed(mncu5_payload())

    evidence = inspect_bmdex_composition_context(repo, "MnCu5", runner=runner, timeout=7)

    assert evidence.status == "ok"
    assert seen["command"] == [str(repo.capability_python), "-B", "-m", PRODUCER_MODULE]
    assert seen["cwd"] == repo.path
    assert json.loads(seen["input"]) == {"formula": "MnCu5"}
    assert seen["capture_output"] is True
    assert seen["text"] is True
    assert seen["check"] is False
    assert seen["timeout"] == 7
    assert seen["shell"] is False


def test_valid_mncu5_schema_v1_response_is_preserved(tmp_path: Path) -> None:
    payload = mncu5_payload()

    enriched = enrich_scientific_context_with_bmdex(
        base_context(),
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: completed(payload),
    )

    assert isinstance(enriched, EnrichedScientificContext)
    assert isinstance(enriched.composition_context, BmdexCompositionEvidence)
    assert enriched.composition_context.payload == payload
    assert enriched.composition_context.schema_version == 1
    assert enriched.composition_context.evidence_type == EVIDENCE_TYPE
    assert enriched.composition_context.datasets["element_abundances"]["records"][0]["abundance"] == 950.0
    assert enriched.evidence_gaps == ()


def test_he_status_ok_with_missing_evidence_remains_successful_enrichment(tmp_path: Path) -> None:
    context = base_context(formula="He")
    payload = he_payload()

    enriched = enrich_scientific_context_with_bmdex(
        context,
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: completed(payload),
    )

    assert enriched.composition_context is not None
    assert enriched.composition_context.status == "ok"
    assert enriched.composition_context.missing_evidence == (
        {
            "dataset": "element_charges",
            "element": "He",
            "reason": "element_not_present_in_dataset",
        },
    )
    assert enriched.evidence_gaps == ()


def test_formula_unavailable_causes_no_subprocess_call(tmp_path: Path) -> None:
    def runner(*_args, **_kwargs):
        raise AssertionError("BMDex must not be invoked without a safe formula")

    enriched = enrich_scientific_context_with_bmdex(
        base_context(formula=None),
        repository(tmp_path),
        runner=runner,
    )

    assert enriched.composition_context is None
    assert enriched.evidence_gaps == (
        EvidenceGap(
            BMDEX_COMPOSITION_CONTEXT,
            "formula_unavailable",
            "ScientificContext identity formula is unavailable.",
        ),
    )


def test_missing_repository_is_an_enrichment_gap() -> None:
    enriched = enrich_scientific_context_with_bmdex(base_context(), None)

    assert enriched.composition_context is None
    assert enriched.evidence_gaps[0].scope == "missing_bmdex_repository"


def test_missing_checkout_is_an_enrichment_gap(tmp_path: Path) -> None:
    enriched = enrich_scientific_context_with_bmdex(
        base_context(),
        repository(tmp_path, include_checkout=False),
    )

    assert enriched.composition_context is None
    assert enriched.evidence_gaps[0].scope == "missing_checkout"


def test_missing_interpreter_is_an_enrichment_gap(tmp_path: Path) -> None:
    enriched = enrich_scientific_context_with_bmdex(
        base_context(),
        repository(tmp_path, include_python=False),
    )

    assert enriched.composition_context is None
    assert enriched.evidence_gaps[0].scope == "missing_interpreter"


def test_missing_capability_python_is_an_enrichment_gap(tmp_path: Path) -> None:
    enriched = enrich_scientific_context_with_bmdex(
        base_context(),
        repository(tmp_path, capability_python=False),
    )

    assert enriched.composition_context is None
    assert enriched.evidence_gaps[0].scope == "missing_interpreter"


def test_nonzero_producer_exit_without_contract_response_is_an_enrichment_gap(tmp_path: Path) -> None:
    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["python"],
            returncode=1,
            stdout="not json",
            stderr="Traceback hidden\nValueError: boom\n" + ("x" * 500),
        )

    enriched = enrich_scientific_context_with_bmdex(
        base_context(),
        repository(tmp_path),
        runner=runner,
    )

    assert enriched.composition_context is None
    gap = enriched.evidence_gaps[0]
    assert gap.scope == "producer_failed"
    assert "exit 1" in gap.reason
    assert "ValueError: boom" in gap.reason
    assert "Traceback" not in gap.reason
    assert len(gap.reason) <= 240


def test_structured_producer_error_is_an_enrichment_gap(tmp_path: Path) -> None:
    enriched = enrich_scientific_context_with_bmdex(
        base_context(),
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: completed(structured_error_payload(), returncode=2),
    )

    assert enriched.composition_context is None
    gap = enriched.evidence_gaps[0]
    assert gap.scope == "structured_producer_error"
    assert "malformed_formula" in gap.reason


def test_malformed_json_is_reported() -> None:
    with pytest.raises(BmdexCompositionError) as exc_info:
        parse_bmdex_composition_payload("{not-json")

    assert exc_info.value.kind == "malformed_json"


def test_malformed_payload_is_reported_for_missing_required_fields() -> None:
    with pytest.raises(BmdexCompositionError) as exc_info:
        parse_bmdex_composition_payload(
            json.dumps({"schema_version": 1, "status": "ok", "evidence_type": "composition_context"})
        )

    assert exc_info.value.kind == "malformed_payload"


def test_malformed_payload_is_reported_for_formula_mismatch() -> None:
    with pytest.raises(BmdexCompositionError) as exc_info:
        parse_bmdex_composition_payload(
            json.dumps(he_payload()),
            requested_formula="MnCu5",
        )

    assert exc_info.value.kind == "malformed_payload"


def test_malformed_payload_is_reported_for_unexpected_evidence_type() -> None:
    payload = mncu5_payload()
    payload["evidence_type"] = "other_context"

    with pytest.raises(BmdexCompositionError) as exc_info:
        parse_bmdex_composition_payload(json.dumps(payload))

    assert exc_info.value.kind == "malformed_payload"


def test_malformed_payload_is_reported_for_unexpected_status() -> None:
    payload = mncu5_payload()
    payload["status"] = "partial"

    with pytest.raises(BmdexCompositionError) as exc_info:
        parse_bmdex_composition_payload(json.dumps(payload))

    assert exc_info.value.kind == "malformed_payload"


def test_unsupported_schema_version_is_reported() -> None:
    payload = mncu5_payload()
    payload["schema_version"] = 2

    with pytest.raises(BmdexCompositionError) as exc_info:
        parse_bmdex_composition_payload(json.dumps(payload))

    assert exc_info.value.kind == "unsupported_schema"


def test_timeout_is_an_enrichment_gap(tmp_path: Path) -> None:
    def runner(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=20)

    enriched = enrich_scientific_context_with_bmdex(
        base_context(),
        repository(tmp_path),
        runner=runner,
    )

    assert enriched.composition_context is None
    assert enriched.evidence_gaps[0].scope == "timeout"


def test_bmdex_provenance_and_limitations_are_preserved_independently(tmp_path: Path) -> None:
    payload = mncu5_payload()
    context = base_context(
        gaps=(EvidenceGap(run_resource.PRODUCER_PROVENANCE, "producer", "no BMD Compute producer record found"),)
    )

    enriched = enrich_scientific_context_with_bmdex(
        context,
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: completed(payload),
    )

    assert enriched.base is context
    assert enriched.base.evidence_gaps == context.evidence_gaps
    assert enriched.composition_context is not None
    assert enriched.composition_context.producer["git"]["commit"] == "a" * 40
    assert enriched.composition_context.producer["git"]["state"] == "clean"
    assert enriched.composition_context.limitations == tuple(payload["limitations"])
    assert enriched.evidence_gaps == ()


def test_existing_direct_vasp_context_evidence_remains_unchanged(tmp_path: Path) -> None:
    context = base_context(
        gaps=(EvidenceGap(run_resource.TRAJECTORY_OBSERVATION, "stage_1.vasprun", "file could not be parsed completely"),)
    )

    enriched = enrich_scientific_context_with_bmdex(
        context,
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: completed(mncu5_payload()),
    )

    assert enriched.base is context
    assert enriched.base.identity.calculation_type == "direct VASP"
    assert enriched.base.identity.producer_workflow is None
    assert enriched.base.identity.executed_input_indicators["spin_polarized"]["value"] is True
    assert enriched.base.evidence_gaps == context.evidence_gaps


def test_build_scientific_context_remains_zero_io(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("ScientificContext builder must remain pure")

    monkeypatch.setattr(run_resource, "retrieve_remote_file", fail)
    monkeypatch.setattr(run_resource, "remote_file_exists", fail)
    monkeypatch.setattr(run_resource, "remote_directory_exists", fail)
    monkeypatch.setattr(run_resource, "get_job_accounting", fail)

    context = build_scientific_context(
        JobInspection(
            job_id="21153721",
            scheduler=None,
            scheduler_error="scheduler unavailable",
            scheduler_work_dir=None,
            calculation_directory=None,
            calculation_type="unknown",
            calculation_reason="scheduler unavailable",
        )
    )

    assert context.identity.job_id == "21153721"


def test_no_cli_command_was_added(capsys) -> None:
    exit_code = cli.main(["bmdex"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Target was not recognized as a SLURM job ID" in captured.out


def test_agent_adapter_does_not_import_bmdex_or_parse_datasets() -> None:
    source = Path("src/bmd_agent/resources/bmdex.py").read_text(encoding="utf-8")

    assert "from tools" not in source
    assert "import tools" not in source
    assert ".yaml" not in source.lower()


def test_agent_adapter_has_no_network_scheduler_or_write_paths() -> None:
    source = Path("src/bmd_agent/resources/bmdex.py").read_text(encoding="utf-8")

    forbidden = (
        "urllib",
        "requests",
        "socket",
        "sacct",
        "squeue",
        "ssh",
        "open(",
        ".open(",
        "write_text",
        "mkdir",
    )
    assert not [token for token in forbidden if token in source]


def test_domain_query_uses_only_observed_hybrid_input_and_trajectory_context() -> None:
    query = build_bmdex_domain_query(domain_analysis())

    assert query == {
        "code": "VASP",
        "calculation_family": "hybrid_functional",
        "functional": "hse06",
        "electronic_algorithm": "Damped",
        "topic": "electronic_iteration_behavior",
        "observed_patterns": [
            "first_electronic_cycle_incomplete_after_only_dav_iterations",
        ],
        "input_tags": {
            "LHFCALC": True,
            "HFSCREEN": 0.2,
            "AEXX": 0.25,
            "ALGO": "Damped",
            "LSORBIT": True,
        },
    }
    assert "hung" not in json.dumps(query).lower()
    assert "failure" not in json.dumps(query).lower()


def test_unrelated_semilocal_context_does_not_query_bmdex() -> None:
    assert build_bmdex_domain_query(domain_analysis(hybrid=False)) is None


def test_job_domain_query_reuses_remote_executed_input_and_trajectory_evidence() -> None:
    query = build_bmdex_domain_query_for_job(domain_job_inspection())

    assert query == {
        "code": "VASP",
        "calculation_family": "hybrid_functional",
        "topic": "electronic_iteration_behavior",
        "electronic_algorithm": "Damped",
        "observed_patterns": [
            "first_electronic_cycle_incomplete_after_only_dav_iterations",
        ],
        "input_tags": {
            "LHFCALC": True,
            "HFSCREEN": 0.2,
            "ALGO": "Damped",
            "LSORBIT": True,
        },
    }


def test_domain_context_invokes_fixed_external_producer_and_preserves_contract(
    tmp_path: Path,
) -> None:
    query = dict(build_bmdex_domain_query(domain_analysis()) or {})
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return completed(domain_payload(query))

    configured_repository = repository(tmp_path)
    evidence = inspect_bmdex_domain_context(
        configured_repository,
        query,
        runner=runner,
        timeout=9,
    )

    command, kwargs = calls[0]
    assert command[-3:] == ["-B", "-m", DOMAIN_CONTEXT_PRODUCER_MODULE]
    assert kwargs["cwd"] == configured_repository.path
    assert json.loads(kwargs["input"]) == {"query": query}
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 9
    assert evidence.evidence_type == DOMAIN_CONTEXT_EVIDENCE_TYPE
    assert evidence.query == query
    record = evidence.records[0]
    assert record.record_id == "vasp.test.context"
    assert record.contextual_statement == "Producer-supplied contextual statement."
    assert record.applicability == {"code": "VASP"}
    assert record.diagnostic_relevance.startswith("This context is relevant")
    assert record.limitations == (
        "This record is contextual reference evidence only.",
        "It does not recommend changing calculation inputs.",
    )
    assert record.sources[0]["authority"] == "VASP Software GmbH / VASP Wiki"
    assert record.record_provenance["record_version"] == 3
    assert (
        record.record_provenance["machine_readable_schema"]
        == "bmdex.contextual_reference.v1"
    )


def test_domain_context_zero_matches_is_valid_optional_evidence() -> None:
    query = dict(build_bmdex_domain_query(domain_analysis()) or {})
    evidence = parse_bmdex_domain_context_payload(
        json.dumps(domain_payload(query, include_record=False)),
        requested_query=query,
    )

    assert evidence.records == ()


@pytest.mark.parametrize(
    ("output", "kind"),
    [
        ("{not-json", "malformed_json"),
        (
            json.dumps(
                {
                    "schema_version": 2,
                    "status": "ok",
                    "evidence_type": "contextual_reference_evidence",
                }
            ),
            "unsupported_schema",
        ),
    ],
)
def test_domain_context_malformed_or_unknown_schema_is_typed(
    output: str,
    kind: str,
) -> None:
    with pytest.raises(BmdexDomainContextError) as exc_info:
        parse_bmdex_domain_context_payload(output)

    assert exc_info.value.kind == kind


def test_domain_context_unavailability_is_nonfatal_and_does_not_change_lifecycle() -> None:
    analysis = domain_analysis()

    enriched = enrich_lifecycle_with_bmdex_domain_context(analysis, None)

    assert analysis.state == LifecycleState.UNKNOWN
    assert enriched.evidence is None
    assert enriched.evidence_gaps[0].scope == "missing_bmdex_repository"


def test_domain_context_malformed_producer_output_becomes_nonfatal_gap(
    tmp_path: Path,
) -> None:
    analysis = domain_analysis()
    enriched = enrich_lifecycle_with_bmdex_domain_context(
        analysis,
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="{not-json",
            stderr="",
        ),
    )

    assert analysis.state == LifecycleState.UNKNOWN
    assert enriched.evidence is None
    assert enriched.evidence_gaps[0].scope == "malformed_json"


def test_domain_context_producer_failure_is_nonfatal_and_bounded(tmp_path: Path) -> None:
    analysis = domain_analysis()
    enriched = enrich_lifecycle_with_bmdex_domain_context(
        analysis,
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["python"],
            returncode=1,
            stdout="",
            stderr="producer failed safely\nprivate traceback detail\n",
        ),
    )

    assert analysis.state == LifecycleState.UNKNOWN
    assert enriched.evidence is None
    gap = enriched.evidence_gaps[0]
    assert gap.scope == "producer_failed"
    assert "exit 1" in gap.reason
    assert "producer failed safely" in gap.reason
    assert "private traceback detail" not in gap.reason


def test_domain_context_assessment_is_qualified_and_source_linked(tmp_path: Path) -> None:
    analysis = domain_analysis()
    query = dict(build_bmdex_domain_query(analysis) or {})
    enriched = enrich_lifecycle_with_bmdex_domain_context(
        analysis,
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: completed(domain_payload(query)),
    )

    assert enriched.assessment is not None
    assessment_text = " ".join(
        (*enriched.assessment.basis, *enriched.assessment.limitations)
    ).lower()
    assert enriched.assessment.source_record_ids == ("vasp.test.context",)
    assert "consistent with the applicability" not in assessment_text
    assert "on query fields: calculation_family, functional, electronic_algorithm, input_tags" in assessment_text
    assert "did not match this reference" in assessment_text
    assert "trajectory evidence did not contribute to the match" in assessment_text
    assert "does not establish" in assessment_text
    assert "does not establish why sigterm was issued" in assessment_text
    assert "definitely" not in assessment_text
    assert "restart" not in assessment_text


def test_domain_context_remains_independently_sourced_when_custodian_termination_supported(
    tmp_path: Path,
) -> None:
    analysis = domain_analysis()
    evidence = parse_custodian_json(
        (Path(__file__).parent / "fixtures" / "custodian_frozen_repeated.json").read_text(
            encoding="utf-8"
        ),
        source_path="/calculation/custodian.json",
    )
    termination = assess_termination_evidence(
        scheduler_state="FAILED",
        custodian_evidence=(evidence,),
        log_messages=("SIGTERM received",),
        error_archive_count=5,
    )
    analysis = replace(
        analysis,
        diagnostics=replace(
            analysis.diagnostics,
            custodian=evidence,
            termination=termination,
        ),
    )
    query = dict(build_bmdex_domain_query(analysis) or {})

    enriched = enrich_lifecycle_with_bmdex_domain_context(
        analysis,
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: completed(domain_payload(query)),
    )

    assert enriched.evidence is not None
    assert enriched.evidence.evidence_type == DOMAIN_CONTEXT_EVIDENCE_TYPE
    assert enriched.assessment is not None
    assert enriched.assessment.source_record_ids == ("vasp.test.context",)
    text = " ".join(enriched.assessment.basis).lower()
    assert "independently observed custodian intervention" in text
    assert "contextual reference evidence rather than termination evidence" in text
    limitations = " ".join(enriched.assessment.limitations).lower()
    assert "does not prove the interrupted vasp operation would eventually converge" in limitations


def test_lifecycle_cli_renders_contextual_evidence_with_compact_provenance(
    tmp_path: Path,
    capsys,
) -> None:
    analysis = domain_analysis()
    query = dict(build_bmdex_domain_query(analysis) or {})
    enriched = enrich_lifecycle_with_bmdex_domain_context(
        analysis,
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: completed(domain_payload(query)),
    )

    cli.print_lifecycle_analysis(analysis, contextual_enrichment=enriched)

    output = capsys.readouterr().out
    assert "Calculation state: UNKNOWN" in output
    assert "Contextual reference evidence (contextual_reference_evidence):" in output
    assert "record: vasp.test.context" in output
    assert "bmdex.contextual_reference.v1" in output
    assert "Producer-supplied contextual statement." in output
    assert "VASP Software GmbH / VASP Wiki" in output
    assert "https://vasp.at/wiki/Test" in output
    assert "Contextual assessment (assessment):" in output
    assert "does not establish why SIGTERM was issued" in output
    assert "HSE definitely" not in output


def test_completed_unrelated_calculation_has_no_contextual_output(capsys) -> None:
    analysis = LifecycleAnalysis(
        state=LifecycleState.COMPLETED,
        directory=Path("/calculation"),
        calculation_kind="direct VASP",
        message="complete",
        incar_settings={"LHFCALC": False, "ALGO": "Normal"},
    )
    enriched = enrich_lifecycle_with_bmdex_domain_context(analysis, None)

    cli.print_lifecycle_analysis(analysis, contextual_enrichment=enriched)

    assert "Contextual reference evidence" not in capsys.readouterr().out


def test_agent_does_not_copy_the_bmdex_record_or_add_action_paths() -> None:
    source = Path("src/bmd_agent/resources/bmdex.py").read_text(encoding="utf-8")

    assert "vasp.hybrid.exact_exchange_iteration_cost" not in source
    assert "HSE starts at the fifth step" not in source
    assert "sbatch" not in source
    assert "scancel" not in source
    assert "scontrol" not in source


LIVE_JOB_22351669_MATCHED_FIELDS = [
    "code",
    "calculation_family",
    "functional",
    "electronic_algorithm",
    "topic",
    "input_tags",
]
LEGACY_AGENT_PATTERNS = [
    "incomplete_first_electronic_cycle",
    "initial_DAV_iterations_observed",
    "4_initial_DAV_iterations_observed",
]


def payload_with_match(query: dict, *, matched_fields: list, matched_patterns=None) -> dict:
    payload = domain_payload(query)
    match = payload["records"][0]["match"]
    match["matched_fields"] = list(matched_fields)
    if matched_patterns is not None:
        match["matched_observed_patterns"] = list(matched_patterns)
    return payload


def assessment_for(payload: dict) -> str:
    evidence = parse_bmdex_domain_context_payload(json.dumps(payload))
    assessment = _contextual_assessment(evidence, custodian_supported=False, mentions_sigterm=False)
    assert assessment is not None
    return " ".join(assessment.basis)


def test_live_job_22351669_match_without_pattern_does_not_claim_trajectory_support() -> None:
    # The live job's producer query carried Agent's pre-vocabulary pattern
    # strings; BMDex matched only on calculation and input fields.
    query = {
        "code": "VASP",
        "calculation_family": "hybrid_functional",
        "functional": "hse06",
        "electronic_algorithm": "Damped",
        "topic": "electronic_iteration_behavior",
        "observed_patterns": LEGACY_AGENT_PATTERNS,
        "input_tags": {"LHFCALC": True, "ALGO": "Damped"},
    }
    text = assessment_for(payload_with_match(query, matched_fields=LIVE_JOB_22351669_MATCHED_FIELDS))

    assert "consistent with the applicability" not in text
    assert "electronic-trajectory context are consistent" not in text
    assert "did not match this reference" in text
    assert "trajectory evidence did not contribute to the match" in text
    assert "also matched the observed electronic-trajectory pattern" not in text


def test_trajectory_support_is_claimed_only_when_bmdex_reports_the_pattern_match() -> None:
    query = dict(build_bmdex_domain_query(domain_analysis()) or {})
    matched = [*LIVE_JOB_22351669_MATCHED_FIELDS[:-1], "observed_patterns", "input_tags"]
    text = assessment_for(
        payload_with_match(query, matched_fields=matched, matched_patterns=["first_electronic_cycle_incomplete_after_only_dav_iterations"])
    )

    assert "also matched the observed electronic-trajectory pattern first_electronic_cycle_incomplete_after_only_dav_iterations" in text
    assert "observation, not a cause" in text
    assert "did not contribute" not in text


def test_no_observed_pattern_is_stated_as_no_trajectory_contribution() -> None:
    query = dict(build_bmdex_domain_query(domain_analysis()) or {})
    query.pop("observed_patterns")
    text = assessment_for(payload_with_match(query, matched_fields=["code", "topic"]))

    assert "No BMDex-defined electronic-trajectory pattern was observed" in text
    assert "trajectory evidence did not contribute to the match" in text


@pytest.mark.parametrize(
    "match_patch",
    (
        {"matched_fields": ["code", "observed_patterns"], "matched_observed_patterns": []},
        {"matched_fields": ["code", "topic"], "matched_observed_patterns": ["first_electronic_cycle_incomplete_after_only_dav_iterations"]},
        {"matched_fields": ["code", "observed_patterns"], "matched_observed_patterns": "first_electronic_cycle_incomplete_after_only_dav_iterations"},
        {"matched_fields": ["code", "observed_patterns"], "matched_observed_patterns": [""]},
    ),
)
def test_inconsistent_or_malformed_pattern_match_reports_are_rejected(match_patch: dict) -> None:
    payload = domain_payload(dict(build_bmdex_domain_query(domain_analysis()) or {}))
    payload["records"][0]["match"].update(match_patch)

    with pytest.raises(BmdexDomainContextError) as exc_info:
        parse_bmdex_domain_context_payload(json.dumps(payload))
    assert exc_info.value.kind == "malformed_payload"


def _trajectory(algorithms, *, count=None, completed_ionic_steps=0):
    return StageTrajectoryObservation(
        stage_index=1,
        stage_label="result_dir",
        stage_type="static",
        theory="hse06",
        directory="/relocated/calculation",
        completed_ionic_steps=completed_ionic_steps,
        incomplete_electronic_iteration_count=len(algorithms) if count is None else count,
        recent_incomplete_electronic_iterations=tuple(
            ElectronicIterationObservation(iteration=index, algorithm=algorithm)
            for index, algorithm in enumerate(algorithms, start=1)
        ),
    )


@pytest.mark.parametrize(
    ("trajectory", "expected"),
    (
        (_trajectory(["DAV"] * 4), True),
        (_trajectory(["DAV"]), True),
        # Transition out of the initial DAV iterations already written.
        (_trajectory(["DAV"] * 4 + ["DMP"]), False),
        # Only a recent window of a longer cycle was observed.
        (_trajectory(["DAV"] * 5, count=9), False),
        (_trajectory([None] * 4), False),
        (_trajectory(["RMM"] * 4), False),
    ),
)
def test_pattern_is_reported_only_when_its_bmdex_definition_is_met(trajectory, expected: bool) -> None:
    analysis = domain_analysis()
    analysis = replace(
        analysis,
        diagnostics=replace(analysis.diagnostics, trajectories=(trajectory,)),
    )
    query = build_bmdex_domain_query(analysis)

    assert query is not None
    assert ("observed_patterns" in query) is expected
    if expected:
        assert query["observed_patterns"] == ["first_electronic_cycle_incomplete_after_only_dav_iterations"]
    for legacy in LEGACY_AGENT_PATTERNS:
        assert legacy not in json.dumps(query)


def test_cli_lists_bmdex_reported_matched_observed_patterns(tmp_path: Path, capsys) -> None:
    analysis = domain_analysis()
    query = dict(build_bmdex_domain_query(analysis) or {})
    matched = ["code", "calculation_family", "observed_patterns"]
    enriched = enrich_lifecycle_with_bmdex_domain_context(
        analysis,
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: completed(
            payload_with_match(query, matched_fields=matched, matched_patterns=["first_electronic_cycle_incomplete_after_only_dav_iterations"])
        ),
    )

    cli.print_lifecycle_analysis(analysis, contextual_enrichment=enriched)

    output = capsys.readouterr().out
    assert "matched observed patterns: first_electronic_cycle_incomplete_after_only_dav_iterations" in output
    assert "also matched the observed electronic-trajectory pattern" in output
