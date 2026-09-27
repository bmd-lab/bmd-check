from __future__ import annotations

import json
from pathlib import Path

from bmd_agent import cli
from bmd_agent.resources import run as run_resource
from bmd_agent.resources.custodian import (
    assess_termination_evidence,
    parse_custodian_json,
    parse_custodian_payload,
    parse_custodian_policy_provenance,
)
from bmd_agent.resources.lifecycle import analyze_calculation_directory

from test_lifecycle import (
    INCAR,
    KPOINTS,
    NORMAL_OUTCAR,
    OSZICAR_STATIC_CONVERGED,
    POSCAR,
    write_inputs,
    write_single_stage_submission,
)


FIXTURES = Path(__file__).parent / "fixtures"


def one_correction_payload() -> list[dict]:
    return [
        {
            "corrections": [
                {
                    "handler": "VaspErrorHandler",
                    "errors": ["eddrmm"],
                    "actions": [
                        {"dict": "INCAR", "action": {"_set": {"ALGO": "Normal"}}}
                    ],
                }
            ],
            "max_errors": False,
            "max_errors_per_job": False,
            "max_errors_per_handler": False,
            "nonzero_return_code": False,
        }
    ]


def policy_submission() -> dict:
    return {
        "provenance": {
            "execution": {
                "custodian": {
                    "stages": [
                        {
                            "index": 1,
                            "policy_id": "bmd_compute.vasp",
                            "policy_version": 1,
                            "stage": {"stage_type": "static", "theory": "hse06"},
                            "handlers": [
                                {
                                    "class": "custodian.vasp.handlers.VaspErrorHandler",
                                    "configuration": {"errors_subset_to_catch": ["brmix"]},
                                }
                            ],
                            "explicit_handler_exclusions": [
                                "custodian.vasp.handlers.FrozenJobErrorHandler"
                            ],
                            "vasp_error_exclusions": ["auto_nbands"],
                            "validators": {
                                "source": "atomate2.vasp.run._DEFAULT_VALIDATORS",
                                "explicit_override": None,
                                "resolved": [
                                    {
                                        "class": "custodian.vasp.validators.VasprunXMLValidator",
                                        "configuration": {},
                                    }
                                ],
                            },
                            "walltime_authority": "slurm",
                            "walltime_handler": None,
                            "custodian_version": "2025.12.14",
                            "implementation_source": (
                                "backend.calculations.custodian_policy.bmd_custodian_handlers"
                            ),
                            "rationale": (
                                "Output-file inactivity alone is not sufficient evidence for BMD "
                                "to terminate a VASP calculation."
                            ),
                        }
                    ]
                }
            }
        }
    }


def test_one_correction_is_structured_without_deserializing_handler() -> None:
    evidence = parse_custodian_payload(
        one_correction_payload(),
        source_path="/calculation/custodian.json",
    )

    assert evidence.attempt_count == 1
    assert len(evidence.corrections) == 1
    correction = evidence.corrections[0]
    assert correction.handler == "VaspErrorHandler"
    assert correction.errors == ("eddrmm",)
    assert correction.actions[0].summary == "INCAR.ALGO -> Normal"
    assert correction.attempt_index == 1
    assert correction.correction_index == 1


def test_historical_frozen_fixture_preserves_all_five_equivalent_interventions() -> None:
    evidence = parse_custodian_json(
        (FIXTURES / "custodian_frozen_repeated.json").read_text(encoding="utf-8"),
        source_path="/historical/custodian.json",
    )

    assert len(evidence.corrections) == 5
    assert [item.sequence_index for item in evidence.corrections] == [1, 2, 3, 4, 5]
    assert all(
        item.handler == "custodian.vasp.handlers.FrozenJobErrorHandler"
        for item in evidence.corrections
    )
    repeated = evidence.repeated_interventions[0]
    assert repeated.count == 5
    assert repeated.timeout_seconds == 21600
    assert repeated.errors == ("Frozen job",)
    assert repeated.action_summaries == ("INCAR.SYMPREC -> 1e-08",)
    assert repeated.correction_positions == ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5))


def test_terminal_flags_preserve_specific_budget_semantics() -> None:
    evidence = parse_custodian_payload(
        one_correction_payload(),
        source_path="custodian.json",
    )
    false_flag = next(item for item in evidence.terminal_flags if item.name == "max_errors")
    assert false_flag.value is False
    assert "does not rule out a different" in false_flag.interpretation

    historical = parse_custodian_json(
        (FIXTURES / "custodian_frozen_repeated.json").read_text(encoding="utf-8"),
        source_path="custodian.json",
    )
    per_job = next(
        item for item in historical.terminal_flags
        if item.name == "max_errors_per_job"
    )
    assert per_job.value is True
    assert per_job.interpretation == "per-job Custodian correction limit reached"


def test_malformed_partial_custodian_json_is_structured_unavailable_evidence() -> None:
    evidence = parse_custodian_json('[{"corrections":', source_path="custodian.json")

    assert evidence.present is True
    assert evidence.error == "custodian.json is malformed or incomplete"
    assert evidence.corrections == ()


def test_unknown_handler_is_retained_without_class_import() -> None:
    evidence = parse_custodian_payload(
        [{"corrections": [{"errors": ["unknown"], "actions": []}]}],
        source_path="custodian.json",
    )

    assert evidence.corrections[0].handler == "unknown"
    assert "handler is unavailable" in evidence.limitations[0]


def test_historical_submission_does_not_inherit_current_execution_policy() -> None:
    policy = parse_custodian_policy_provenance({"provenance": {"git": {}}})

    assert policy.available is False
    assert policy.stages == ()
    assert "no persisted" in policy.reason


def test_persisted_compute_policy_preserves_exclusions_walltime_and_validators() -> None:
    policy = parse_custodian_policy_provenance(policy_submission())

    assert policy.available is True
    stage = policy.stages[0]
    assert stage.policy_id == "bmd_compute.vasp"
    assert stage.policy_version == 1
    assert stage.frozen_job_handler_status == "explicitly excluded"
    assert stage.walltime_authority == "slurm"
    assert stage.walltime_handler is None
    assert stage.vasp_error_exclusions == ("auto_nbands",)
    assert stage.validators[0].class_name.endswith("VasprunXMLValidator")


def test_persisted_policy_cli_keeps_configuration_and_producer_context(capsys) -> None:
    policy = parse_custodian_policy_provenance(policy_submission())

    cli._print_custodian_policy_context(policy)
    output = capsys.readouterr().out

    assert "FrozenJobErrorHandler: explicitly excluded" in output
    assert "errors_subset_to_catch=['brmix']" in output
    assert "walltime authority: slurm" in output
    assert "internal walltime handler: none" in output
    assert "atomate2.vasp.run._DEFAULT_VALIDATORS" in output
    assert "backend.calculations.custodian_policy.bmd_custodian_handlers" in output


def test_run_submission_parser_consumes_persisted_policy_without_reconstruction() -> None:
    submission = policy_submission()
    submission.update(
        {
            "flow_spec": {
                "workflow_spec": {
                    "stages": [
                        {
                            "stage_type": "static",
                            "theory": "hse06",
                            "modifiers": [],
                            "options": {},
                        }
                    ]
                }
            },
            "paths": {
                "stage_dirs": {},
                "result_dir": "/bmd-db/guest/flows/run",
            },
        }
    )

    parsed = run_resource._parse_submission(
        submission,
        allowed_roots=("/bmd-db/guest/flows",),
    )

    assert parsed["custodian_policy"].available is True
    assert parsed["custodian_policy"].stages[0].walltime_authority == "slurm"


def test_configured_handler_does_not_imply_observed_intervention() -> None:
    submission = policy_submission()
    submission["provenance"]["execution"]["custodian"]["stages"][0]["handlers"].append(
        {
            "class": "custodian.vasp.handlers.MeshSymmetryErrorHandler",
            "configuration": {},
        }
    )
    policy = parse_custodian_policy_provenance(submission)
    evidence = parse_custodian_payload([{"corrections": []}], source_path="custodian.json")

    assert any(
        item.class_name.endswith("MeshSymmetryErrorHandler")
        for item in policy.stages[0].handlers
    )
    assert evidence.corrections == ()


def test_observed_intervention_alone_does_not_establish_final_termination() -> None:
    evidence = parse_custodian_payload(
        one_correction_payload(),
        source_path="custodian.json",
    )

    assessment = assess_termination_evidence(
        scheduler_state="FAILED",
        custodian_evidence=(evidence,),
    )

    assert assessment.classification == "unknown"
    assert assessment.status == "insufficient_evidence"
    assert "do not establish" in assessment.limitations[0]


def test_historical_pattern_supports_qualified_custodian_termination() -> None:
    evidence = parse_custodian_json(
        (FIXTURES / "custodian_frozen_repeated.json").read_text(encoding="utf-8"),
        source_path="custodian.json",
    )

    assessment = assess_termination_evidence(
        scheduler_state="FAILED",
        custodian_evidence=(evidence,),
        log_messages=("SIGTERM received by VASP",),
        error_archive_count=5,
    )

    assert assessment.classification == "custodian_triggered_process_termination"
    assert assessment.status == "supported"
    assert any("correction-budget" in item for item in assessment.basis)
    assert any("does not prove" in item for item in assessment.limitations)


def test_scheduler_timeout_remains_distinct_from_custodian_history() -> None:
    evidence = parse_custodian_payload(
        one_correction_payload(),
        source_path="custodian.json",
    )

    assessment = assess_termination_evidence(
        scheduler_state="TIMEOUT",
        custodian_evidence=(evidence,),
        log_messages=("SIGTERM",),
        error_archive_count=5,
    )

    assert assessment.classification == "slurm_timeout"


def test_scheduler_oom_and_recorded_nonzero_exit_are_distinct_termination_evidence() -> None:
    oom = assess_termination_evidence(scheduler_state="OUT_OF_MEMORY")
    payload = one_correction_payload()
    payload[0]["corrections"] = []
    payload[0]["nonzero_return_code"] = True
    evidence = parse_custodian_payload(payload, source_path="custodian.json")
    nonzero = assess_termination_evidence(
        scheduler_state="FAILED",
        custodian_evidence=(evidence,),
    )

    assert oom.classification == "slurm_out_of_memory"
    assert nonzero.classification == "vasp_nonzero_exit_observed"


def test_historical_bmd_snapshot_renders_decisive_intervention_evidence(
    tmp_path: Path,
    capsys,
) -> None:
    write_single_stage_submission(tmp_path, result_dir=str(tmp_path), job_id="21906221")
    write_inputs(tmp_path)
    (tmp_path / "OUTCAR").write_text("partial", encoding="utf-8")
    (tmp_path / "OSZICAR").write_text("DAV: 1 -1 -1 -1 1 1\n", encoding="utf-8")
    (tmp_path / "std_err.txt").write_text("SIGTERM received by VASP\n", encoding="utf-8")
    (tmp_path / "custodian.json").write_text(
        (FIXTURES / "custodian_frozen_repeated.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for index in range(1, 6):
        (tmp_path / f"error.{index}.tar.gz").write_bytes(b"fixture")

    analysis = analyze_calculation_directory(tmp_path)
    cli.print_lifecycle_analysis(analysis)
    output = capsys.readouterr().out

    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.custodian_policy.available is False
    assert analysis.diagnostics is not None
    assert analysis.diagnostics.termination is not None
    assert (
        analysis.diagnostics.termination.classification
        == "custodian_triggered_process_termination"
    )
    assert "Execution-policy context (producer_provenance):" in output
    assert "no persisted Custodian execution-policy provenance" in output
    assert "FrozenJobErrorHandler:" in output
    assert "interventions: 5" in output
    assert "inactivity timeout: 21600 s (6 h)" in output
    assert "repeated correction: INCAR.SYMPREC -> 1e-08" in output
    assert "classification: custodian_triggered_process_termination" in output


def test_no_custodian_file_remains_uncluttered_for_completed_calculation(
    tmp_path: Path,
    capsys,
) -> None:
    for name, contents in {"POSCAR": POSCAR, "INCAR": INCAR, "KPOINTS": KPOINTS}.items():
        (tmp_path / name).write_text(contents, encoding="utf-8")
    (tmp_path / "OUTCAR").write_text(NORMAL_OUTCAR, encoding="utf-8")
    (tmp_path / "OSZICAR").write_text(OSZICAR_STATIC_CONVERGED, encoding="utf-8")

    analysis = analyze_calculation_directory(tmp_path)
    cli.print_lifecycle_analysis(analysis)

    assert analysis.diagnostics is None
    assert "Custodian intervention evidence" not in capsys.readouterr().out


def test_local_diagnostics_do_not_read_potcar_unpack_archives_or_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name, contents in {"POSCAR": POSCAR, "INCAR": INCAR, "KPOINTS": KPOINTS}.items():
        (tmp_path / name).write_text(contents, encoding="utf-8")
    (tmp_path / "OSZICAR").write_text("DAV: 1 -1 -1 -1 1 1\n", encoding="utf-8")
    (tmp_path / "OUTCAR").write_text("partial", encoding="utf-8")
    (tmp_path / "POTCAR").write_text("secret-potcar-content", encoding="utf-8")
    archive = tmp_path / "error.1.tar.gz"
    archive.write_bytes(b"not-an-archive")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.name == "POTCAR":
            raise AssertionError("POTCAR must not be read")
        mode = args[0] if args else kwargs.get("mode", "r")
        if any(marker in mode for marker in ("w", "a", "+", "x")):
            raise AssertionError("calculation-directory writes are forbidden")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    analysis = analyze_calculation_directory(tmp_path)

    monkeypatch.undo()
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    assert analysis.diagnostics is not None
    assert before == after
    assert not (tmp_path / "error.1").exists()
