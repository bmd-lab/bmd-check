from __future__ import annotations

import json
from pathlib import Path

from bmd_agent import cli
from bmd_agent.config import ConfigurationError
from bmd_agent.resources.bmdex import build_bmdex_domain_query
from bmd_agent.resources.lifecycle import (
    LifecycleState,
    analyze_calculation_directory,
)
from bmd_agent.resources.oom import INSUFFICIENT_OOM_EVIDENCE, OOM_ESTABLISHED
from bmd_agent.resources.slurm import SlurmAccountingRecord


POSCAR = """\
Si
1.0
5.43 0.00 0.00
0.00 5.43 0.00
0.00 0.00 5.43
Si
2
Direct
0.00 0.00 0.00
0.25 0.25 0.25
"""

INCAR = """\
ENCUT = 520
EDIFF = 1E-6
ISIF = 3
"""

KPOINTS = """\
Automatic mesh
0
Gamma
4 4 4
0 0 0
"""

NORMAL_OUTCAR = """\
 some vasp output
 General timing and accounting informations for this job:
 Voluntary context switches
"""

OSZICAR_TWO_STEP = """\
       N       E                     dE             d eps       ncg     rms          rms(c)
DAV:   1   -1.000000000000E+01   -1.00000E+01   -1.00000E+01   10   1.000E+00   2.000E-01
DAV:   2   -1.100000000000E+01   -1.00000E+00   -2.00000E-01   12   1.000E-01   2.000E-02
   1 F= -.11000000E+02 E0= -.10950000E+02  d E =-.110000E+02
       N       E                     dE             d eps       ncg     rms          rms(c)
RMM:   1   -1.200000000000E+01   -1.00000E+00   -1.00000E-01   10   5.000E-02   1.000E-02
RMM:   2   -1.210000000000E+01   -1.00000E-01   -1.00000E-02   12   4.000E-02   8.000E-03
   2 F= -.12100000E+02 E0= -.12050000E+02  d E =-.110000E+01
"""

OUTCAR_TWO_FORCE_BLOCKS = """\
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000000      0.00000000      0.00000000      3.00000000      4.00000000      0.00000000
      0.50000000      0.50000000      0.50000000      0.00000000      0.00000000      1.00000000
 -----------------------------------------------------------------------------------
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000000      0.00000000      0.00000000      0.30000000      0.40000000      0.00000000
      0.50000000      0.50000000      0.50000000      0.00000000      0.00000000      0.10000000
 -----------------------------------------------------------------------------------
"""

OSZICAR_INCOMPLETE_FOUR_DAV = """\
       N       E                     dE             d eps       ncg     rms          rms(c)
DAV:   1   -1.000000000000E+01   -1.00000E+01   -1.00000E+01   10   1.000E+00   2.000E-01
DAV:   2   -1.100000000000E+01   -1.00000E+00   -2.00000E-01   12   1.000E-01   2.000E-02
DAV:   3   -1.110000000000E+01   -1.00000E-01   -2.00000E-02   12   8.000E-02   1.000E-02
DAV:   4   -1.111000000000E+01   -1.00000E-02   -2.00000E-03   12   7.000E-02   9.000E-03
"""


# Static SCF whose final completed electronic cycle reaches EDIFF = 1E-6.
OSZICAR_STATIC_CONVERGED = """\
       N       E                     dE             d eps       ncg     rms          rms(c)
DAV:   1   -1.000000000000E+01   -1.00000E+01   -1.00000E+01   10   1.000E+00   2.000E-01
DAV:   2   -1.080000000000E+01   -8.00000E-01   -2.00000E-02   12   1.000E-01   2.000E-02
DAV:   3   -1.080000010000E+01   -1.00000E-07   -2.00000E-08   12   1.000E-04   2.000E-05
   1 F= -.10800000E+02 E0= -.10800000E+02  d E =-.108000E+02
"""

INCAR_RELAX = """\
ENCUT = 520
EDIFF = 1E-6
EDIFFG = -0.05
IBRION = 2
ISIF = 2
NSW = 50
"""

# Two ionic steps; every completed electronic cycle reaches EDIFF = 1E-6.
OSZICAR_RELAX_TWO_STEP_SCF_CONVERGED = """\
       N       E                     dE             d eps       ncg     rms          rms(c)
DAV:   1   -1.000000000000E+01   -1.00000E+01   -1.00000E+01   10   1.000E+00   2.000E-01
DAV:   2   -1.100000000000E+01   -1.00000E-07   -2.00000E-08   12   1.000E-04   2.000E-05
   1 F= -.11000000E+02 E0= -.10950000E+02  d E =-.110000E+02
RMM:   1   -1.200000000000E+01   -1.00000E+00   -1.00000E-01   10   5.000E-02   1.000E-02
RMM:   2   -1.210000000000E+01   -1.00000E-07   -1.00000E-08   12   4.000E-05   8.000E-06
   2 F= -.12100000E+02 E0= -.12050000E+02  d E =-.110000E+01
"""

OUTCAR_RELAX_FORCES_CONVERGED = """\
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000000      0.00000000      0.00000000      3.00000000      4.00000000      0.00000000
      0.50000000      0.50000000      0.50000000      0.00000000      0.00000000      1.00000000
 -----------------------------------------------------------------------------------
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000000      0.00000000      0.00000000      0.01000000      0.02000000      0.00000000
      0.50000000      0.50000000      0.50000000      0.00000000      0.00000000      0.01000000
 -----------------------------------------------------------------------------------
""" + NORMAL_OUTCAR

# Normal VASP termination, but the final maximum force (0.5 eV/A) is far above
# |EDIFFG| = 0.05 eV/A: NSW was exhausted without ionic convergence.
OUTCAR_RELAX_FORCES_UNCONVERGED = """\
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000000      0.00000000      0.00000000      3.00000000      4.00000000      0.00000000
      0.50000000      0.50000000      0.50000000      0.00000000      0.00000000      1.00000000
 -----------------------------------------------------------------------------------
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000000      0.00000000      0.00000000      0.30000000      0.40000000      0.00000000
      0.50000000      0.50000000      0.50000000      0.00000000      0.00000000      0.10000000
 -----------------------------------------------------------------------------------
""" + NORMAL_OUTCAR


def write_inputs(directory: Path, *, missing: str | None = None) -> None:
    values = {"POSCAR": POSCAR, "INCAR": INCAR, "KPOINTS": KPOINTS}
    for name, contents in values.items():
        if name != missing:
            (directory / name).write_text(contents, encoding="utf-8")


def scheduler_record(
    *,
    job_id: str = "21153721",
    state: str = "RUNNING",
    exit_code: str = "0:0",
    work_dir: str = "/unused",
) -> SlurmAccountingRecord:
    return SlurmAccountingRecord(
        job_id=job_id,
        name="vasp",
        state=state,
        elapsed="00:10:00",
        start="2026-09-16T10:00:00",
        end="",
        partition="leeburton-pool",
        exit_code=exit_code,
        timelimit="01:00:00",
        node_list="compute-0-1",
        allocated_cpus=24,
        work_dir=work_dir,
    )


def write_submission(
    root: Path,
    *,
    job_id: str = "21153721",
    stage_dir: Path | str | None = None,
    result_dir: Path | str | None = None,
    attempt_state: Path | None = None,
) -> None:
    stage_dir = stage_dir or root / "stage_01"
    result_dir = result_dir or stage_dir
    payload = {
        "flow_spec": {
            "workflow_spec": {
                "stages": [
                    {
                        "stage_type": "relax",
                        "theory": "pbe",
                        "modifiers": [],
                        "label": "relax",
                        "options": {},
                    }
                ]
            }
        },
        "submission": {
            "job_id": job_id,
            **({"attempt_state": str(attempt_state)} if attempt_state else {}),
        },
        "paths": {
            "stage_dirs": {"stage_01": str(stage_dir)},
            "result_dir": str(result_dir),
        },
    }
    (root / "submission.json").write_text(json.dumps(payload), encoding="utf-8")


def write_single_stage_submission(
    root: Path,
    *,
    result_dir: str,
    job_id: str = "21153721",
) -> None:
    payload = {
        "flow_spec": {
            "workflow_spec": {
                "stages": [
                    {
                        "stage_type": "static",
                        "theory": "hse06",
                        "modifiers": [],
                        "label": "hse06_static",
                        "options": {},
                    }
                ]
            }
        },
        "submission": {"job_id": job_id},
        "paths": {
            "stage_dirs": {},
            "result_dir": result_dir,
        },
    }
    (root / "submission.json").write_text(json.dumps(payload), encoding="utf-8")


def write_three_stage_submission(
    root: Path,
    *,
    producer_root: str = "/bmd-db/guest/flows/vasp_run_custom_workflow-20260912-170926",
    stage_paths: dict[str, str] | None = None,
    result_dir: str | None = None,
    job_id: str = "21153721",
) -> None:
    stage_paths = stage_paths or {
        "stage_01": f"{producer_root}/stage_01",
        "stage_02": f"{producer_root}/stage_02",
        "stage_03": f"{producer_root}/stage_03",
    }
    result_dir = result_dir or stage_paths["stage_03"]
    payload = {
        "flow_spec": {
            "workflow_spec": {
                "stages": [
                    {"stage_type": "relax", "theory": "pbe", "modifiers": [], "options": {}},
                    {"stage_type": "static", "theory": "hse06", "modifiers": [], "options": {}},
                    {"stage_type": "dos", "theory": "hse06", "modifiers": [], "options": {}},
                ]
            }
        },
        "submission": {"job_id": job_id},
        "paths": {
            "workflow_root": producer_root,
            "stage_dirs": stage_paths,
            "result_dir": result_dir,
        },
    }
    (root / "submission.json").write_text(json.dumps(payload), encoding="utf-8")


def write_stage_inputs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_inputs(directory)


def write_stage_complete(directory: Path) -> None:
    """Write normal termination plus positive electronic and ionic convergence evidence."""

    write_stage_inputs(directory)
    (directory / "INCAR").write_text(INCAR_RELAX, encoding="utf-8")
    (directory / "OUTCAR").write_text(OUTCAR_RELAX_FORCES_CONVERGED, encoding="utf-8")
    (directory / "OSZICAR").write_text(OSZICAR_RELAX_TWO_STEP_SCF_CONVERGED, encoding="utf-8")


def write_stage_terminated_unverified(directory: Path) -> None:
    """Write VASP normal termination without any convergence evidence."""

    write_stage_inputs(directory)
    (directory / "OUTCAR").write_text(NORMAL_OUTCAR, encoding="utf-8")
    (directory / "OSZICAR").write_text(" 1 F= -.1 E0= -.1 d E =0\n", encoding="utf-8")
    (directory / "vasprun.xml").write_text("<modeling></modeling>", encoding="utf-8")


def write_stage_relax_unconverged(directory: Path) -> None:
    """Write a normally terminated relaxation whose final force misses EDIFFG."""

    write_stage_inputs(directory)
    (directory / "INCAR").write_text(INCAR_RELAX, encoding="utf-8")
    (directory / "OUTCAR").write_text(OUTCAR_RELAX_FORCES_UNCONVERGED, encoding="utf-8")
    (directory / "OSZICAR").write_text(OSZICAR_RELAX_TWO_STEP_SCF_CONVERGED, encoding="utf-8")


def write_stage_partial(directory: Path) -> None:
    write_stage_inputs(directory)
    (directory / "OUTCAR").write_text("partial", encoding="utf-8")
    (directory / "OSZICAR").write_text(" 1 F= -.1 E0= -.1 d E =0\n", encoding="utf-8")


def write_partial_diagnostic_outputs(directory: Path) -> None:
    write_inputs(directory)
    (directory / "OSZICAR").write_text(OSZICAR_TWO_STEP, encoding="utf-8")
    (directory / "OUTCAR").write_text(OUTCAR_TWO_FORCE_BLOCKS, encoding="utf-8")
    (directory / "vasprun.xml").write_text("<modeling>", encoding="utf-8")
    (directory / "std_err.txt").write_text("fatal: VASP exited with non-zero status\n", encoding="utf-8")
    (directory / "vasp.out").write_text("ERROR: electronic minimization did not finish\n", encoding="utf-8")
    (directory / "custodian.json").write_text(
        json.dumps(
            [
                {
                    "handler": "VaspErrorHandler",
                    "errors": ["eddrmm"],
                    "actions": [{"dict": "INCAR", "action": {"_set": {"ALGO": "Normal"}}}],
                }
            ]
        ),
        encoding="utf-8",
    )
    (directory / "error.1.tar.gz").write_bytes(b"not-a-real-tar-for-test")


def test_empty_directory_is_unknown_no_calculation(tmp_path: Path) -> None:
    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.UNKNOWN
    assert analysis.calculation_kind == "none"
    assert "No recognizable" in analysis.message


def test_inputs_without_outputs_are_pre_run(tmp_path: Path) -> None:
    write_inputs(tmp_path)

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.PRE_RUN
    assert analysis.calculation_kind == "direct VASP"
    assert analysis.structure is not None
    assert analysis.incar_settings["ENCUT"] == 520


def test_missing_input_does_not_crash_and_reports_gap(tmp_path: Path) -> None:
    write_inputs(tmp_path, missing="KPOINTS")

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.UNKNOWN
    assert "KPOINTS is missing" in analysis.evidence_gaps


def test_non_empty_partial_output_is_not_pre_run(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    (tmp_path / "OSZICAR").write_text(" 1 F= -.1 E0= -.1 d E =0\n", encoding="utf-8")

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.UNKNOWN
    assert analysis.state != LifecycleState.PRE_RUN
    assert "scheduler/provenance" in analysis.message


def test_manual_completed_vasp_without_scheduler_is_completed(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    (tmp_path / "OUTCAR").write_text(NORMAL_OUTCAR, encoding="utf-8")
    (tmp_path / "OSZICAR").write_text(OSZICAR_STATIC_CONVERGED, encoding="utf-8")

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.COMPLETED
    assert analysis.normal_completion is True


def test_manual_normal_termination_without_convergence_evidence_is_unknown(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    (tmp_path / "OUTCAR").write_text(NORMAL_OUTCAR, encoding="utf-8")

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.UNKNOWN
    assert analysis.normal_completion is True
    assert "convergence could not be established" in analysis.message


def test_truncated_vasprun_does_not_imply_completed(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    (tmp_path / "vasprun.xml").write_text("<modeling>", encoding="utf-8")

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.UNKNOWN
    assert analysis.state != LifecycleState.COMPLETED


def test_active_bmd_scheduler_provenance_is_running(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(
        root,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="RUNNING"),
    )

    assert analysis.state == LifecycleState.RUNNING
    assert analysis.calculation_kind == "BMD Compute"
    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.current_stage is not None
    assert analysis.bmd_workflow.current_stage.label == "stage_01"


def test_scheduler_running_before_outputs_is_running(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(
        stage,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="RUNNING"),
    )

    assert analysis.state == LifecycleState.RUNNING
    assert not any(item.non_empty for item in analysis.output_files.values())


def test_failed_scheduler_with_partial_output_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    (stage / "OUTCAR").write_text("partial", encoding="utf-8")
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(
        stage,
        scheduler_lookup=lambda job_id: scheduler_record(
            job_id=job_id,
            state="TIMEOUT",
            exit_code="0:1",
        ),
    )

    assert analysis.state == LifecycleState.INCOMPLETE


def test_nonzero_exit_with_incomplete_output_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    (stage / "OUTCAR").write_text("partial", encoding="utf-8")
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(
        root,
        scheduler_lookup=lambda job_id: scheduler_record(
            job_id=job_id,
            state="COMPLETED",
            exit_code="1:0",
        ),
    )

    assert analysis.state == LifecycleState.INCOMPLETE


def test_successful_scheduler_and_normal_vasp_evidence_is_completed(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_stage_complete(stage)
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(
        stage,
        scheduler_lookup=lambda job_id: scheduler_record(
            job_id=job_id,
            state="COMPLETED",
            exit_code="0:0",
        ),
    )

    assert analysis.state == LifecycleState.COMPLETED


def test_workflow_root_invocation_uses_submission_provenance(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(root)

    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.workflow_root == root.resolve()
    assert analysis.bmd_workflow.relocated is False
    assert analysis.calculation_kind == "BMD Compute"


def test_stage_directory_invocation_binds_to_workflow_root(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(stage)

    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.workflow_root == root.resolve()
    assert analysis.bmd_workflow.current_stage is not None
    assert analysis.bmd_workflow.current_stage.path == stage.resolve()
    assert analysis.bmd_workflow.current_stage.producer_path == str(stage)
    assert analysis.bmd_workflow.relocated is False


def test_relocated_bmd_snapshot_reads_current_local_inputs(tmp_path: Path) -> None:
    original = "/bmd-db/guest/flows/vasp_run_hse_static-20260830"
    write_single_stage_submission(tmp_path, result_dir=original)
    write_inputs(tmp_path)
    (tmp_path / "OSZICAR").write_text(" 1 F= -.1 E0= -.1 d E =0\n", encoding="utf-8")

    analysis = analyze_calculation_directory(
        tmp_path,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="RUNNING"),
    )

    assert analysis.calculation_kind == "BMD Compute"
    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.relocated is True
    assert analysis.bmd_workflow.producer_root == original
    assert analysis.bmd_workflow.current_stage is not None
    assert analysis.bmd_workflow.current_stage.path == tmp_path.resolve()
    assert analysis.bmd_workflow.current_stage.producer_path == original
    assert all(analysis.input_files[name].present for name in ("POSCAR", "INCAR", "KPOINTS"))
    assert analysis.input_files["POSCAR"].path == tmp_path.resolve() / "POSCAR"
    assert "POSCAR is missing" not in analysis.evidence_gaps


def test_relocated_partial_bmd_snapshot_without_current_scheduler_state_is_unknown(tmp_path: Path) -> None:
    write_single_stage_submission(
        tmp_path,
        result_dir="/bmd-db/guest/flows/vasp_run_hse_static-20260830",
    )
    write_inputs(tmp_path)
    (tmp_path / "OUTCAR").write_text("partial", encoding="utf-8")

    analysis = analyze_calculation_directory(
        tmp_path,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="RUNNING"),
    )

    assert analysis.state == LifecycleState.UNKNOWN
    assert analysis.scheduler is None
    assert analysis.scheduler_error is not None
    assert "original producer location" in analysis.scheduler_error


def test_unknown_relocated_partial_snapshot_gets_diagnostic_evidence(tmp_path: Path) -> None:
    write_single_stage_submission(
        tmp_path,
        result_dir="/bmd-db/guest/flows/hse06_soc_failed",
    )
    write_partial_diagnostic_outputs(tmp_path)
    before = sorted(path.name for path in tmp_path.iterdir())

    analysis = analyze_calculation_directory(
        tmp_path,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="FAILED", exit_code="1:0"),
    )

    after = sorted(path.name for path in tmp_path.iterdir())
    assert after == before
    assert analysis.state == LifecycleState.UNKNOWN
    assert analysis.diagnostics is not None
    assert analysis.diagnostics.trajectories
    trajectory = analysis.diagnostics.trajectories[0]
    assert trajectory.oszicar_present is True
    assert trajectory.completed_ionic_steps == 2
    assert trajectory.vasprun_present is True
    assert trajectory.vasprun_error == "file could not be parsed completely"
    assert trajectory.outcar_present is True
    assert trajectory.outcar_complete_force_blocks == 2
    assert analysis.diagnostics.logs
    assert any("fatal" in " ".join(log.messages).lower() for log in analysis.diagnostics.logs)
    assert analysis.diagnostics.custodian is not None
    assert any("VaspErrorHandler" in event for event in analysis.diagnostics.custodian.events)
    assert analysis.diagnostics.custodian.corrections[0].errors == ("eddrmm",)
    assert [archive.name for archive in analysis.diagnostics.error_archives] == ["error.1.tar.gz"]


def test_relocated_failed_hybrid_snapshot_builds_factual_domain_context_query(
    tmp_path: Path,
) -> None:
    write_single_stage_submission(
        tmp_path,
        result_dir="/bmd-db/guest/flows/hse06_soc_failed",
    )
    write_inputs(tmp_path)
    (tmp_path / "INCAR").write_text(
        "\n".join(
            (
                "LHFCALC = .TRUE.",
                "HFSCREEN = 0.2",
                "AEXX = 0.25",
                "ALGO = Damped",
                "LSORBIT = .TRUE.",
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "OSZICAR").write_text(OSZICAR_INCOMPLETE_FOUR_DAV, encoding="utf-8")
    (tmp_path / "OUTCAR").write_text("partial", encoding="utf-8")
    (tmp_path / "std_err.txt").write_text("SIGTERM received by VASP\n", encoding="utf-8")

    analysis = analyze_calculation_directory(tmp_path)
    query = build_bmdex_domain_query(analysis)

    assert analysis.state == LifecycleState.UNKNOWN
    assert analysis.diagnostics is not None
    trajectory = analysis.diagnostics.trajectories[0]
    assert trajectory.completed_ionic_steps == 0
    assert trajectory.incomplete_electronic_iteration_count == 4
    assert analysis.diagnostics.logs[0].messages == ("SIGTERM received by VASP",)
    assert analysis.diagnostics.oom is not None
    assert analysis.diagnostics.oom.assessment == INSUFFICIENT_OOM_EVIDENCE
    assert query is not None
    assert query["functional"] == "hse06"
    assert query["electronic_algorithm"] == "Damped"
    assert query["input_tags"] == {
        "LHFCALC": True,
        "HFSCREEN": 0.2,
        "AEXX": 0.25,
        "ALGO": "Damped",
        "LSORBIT": True,
    }
    assert query["observed_patterns"] == ["first_electronic_cycle_incomplete_after_only_dav_iterations"]


def test_relocated_snapshot_uses_local_cgroup_marker_without_scheduler_state(
    tmp_path: Path,
) -> None:
    write_single_stage_submission(
        tmp_path,
        result_dir="/bmd-db/guest/flows/hse06_soc_failed",
    )
    write_inputs(tmp_path)
    (tmp_path / "OUTCAR").write_text("partial", encoding="utf-8")
    (tmp_path / "std_err.txt").write_text(
        "slurmstepd: error: Detected 1 oom-kill event(s) in step\n",
        encoding="utf-8",
    )

    analysis = analyze_calculation_directory(
        tmp_path,
        scheduler_lookup=lambda job_id: scheduler_record(
            job_id=job_id,
            state="OUT_OF_MEMORY",
            exit_code="0:125",
        ),
    )

    assert analysis.state == LifecycleState.UNKNOWN
    assert analysis.scheduler is None
    assert analysis.diagnostics is not None
    assert analysis.diagnostics.oom is not None
    assert analysis.diagnostics.oom.assessment == OOM_ESTABLISHED
    assert analysis.diagnostics.oom.scheduler_oom_state is False


def test_incomplete_bmd_snapshot_uses_same_diagnostic_evidence(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_partial_diagnostic_outputs(stage)
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(
        root,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="FAILED", exit_code="1:0"),
    )

    assert analysis.state == LifecycleState.INCOMPLETE
    assert analysis.diagnostics is not None
    assert analysis.diagnostics.trajectories[0].oszicar_present is True
    assert analysis.diagnostics.trajectories[0].vasprun_error == "file could not be parsed completely"


def test_running_bmd_snapshot_reports_progress_without_failed_label(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_partial_diagnostic_outputs(stage)
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(
        root,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="RUNNING", exit_code="0:0"),
    )

    assert analysis.state == LifecycleState.RUNNING
    assert analysis.message == "Active scheduler evidence indicates the calculation is running."
    assert analysis.diagnostics is not None
    assert analysis.diagnostics.trajectories[0].completed_ionic_steps == 2


def test_relocated_completed_bmd_snapshot_uses_local_normal_completion(tmp_path: Path) -> None:
    write_single_stage_submission(
        tmp_path,
        result_dir="/bmd-db/guest/flows/vasp_run_hse_static-20260830",
    )
    write_inputs(tmp_path)
    (tmp_path / "OUTCAR").write_text(NORMAL_OUTCAR, encoding="utf-8")
    (tmp_path / "OSZICAR").write_text(OSZICAR_STATIC_CONVERGED, encoding="utf-8")

    analysis = analyze_calculation_directory(
        tmp_path,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="RUNNING"),
    )

    assert analysis.state == LifecycleState.COMPLETED
    assert analysis.normal_completion is True
    assert analysis.scheduler is None


def test_completed_bmd_snapshot_does_not_run_incomplete_diagnostics(tmp_path: Path) -> None:
    write_single_stage_submission(
        tmp_path,
        result_dir="/bmd-db/guest/flows/vasp_run_hse_static-20260830",
    )
    write_inputs(tmp_path)
    (tmp_path / "OUTCAR").write_text(NORMAL_OUTCAR, encoding="utf-8")
    (tmp_path / "OSZICAR").write_text(OSZICAR_STATIC_CONVERGED, encoding="utf-8")

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.COMPLETED
    assert analysis.diagnostics is None


def test_normally_terminated_static_whose_scf_missed_ediff_is_incomplete(tmp_path: Path) -> None:
    # OSZICAR_TWO_STEP ends with dE = -1E-1 against EDIFF = 1E-6.
    write_single_stage_submission(
        tmp_path,
        result_dir="/bmd-db/guest/flows/vasp_run_hse_static-20260830",
    )
    write_inputs(tmp_path)
    (tmp_path / "OUTCAR").write_text(NORMAL_OUTCAR, encoding="utf-8")
    (tmp_path / "OSZICAR").write_text(OSZICAR_TWO_STEP, encoding="utf-8")

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.INCOMPLETE
    assert analysis.stage_completion[0].convergence.not_converged_scopes == ("electronic",)
    assert analysis.diagnostics is not None


def test_pre_run_bmd_snapshot_does_not_run_execution_diagnostics(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(root)

    assert analysis.state == LifecycleState.PRE_RUN
    assert analysis.diagnostics is None


def test_relocated_cli_prints_current_acquisition_and_original_producer_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    original = "/bmd-db/guest/flows/vasp_run_hse_static-20260830"
    write_single_stage_submission(tmp_path, result_dir=original)
    write_inputs(tmp_path)
    (tmp_path / "OSZICAR").write_text(" 1 F= -.1 E0= -.1 d E =0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "load_resources",
        lambda: (_ for _ in ()).throw(ConfigurationError("missing config")),
    )

    exit_code = cli.main(["--verbose"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Calculation type: BMD Compute" in captured.out
    assert f"current acquisition directory: {tmp_path.resolve()}" in captured.out
    assert f"original producer run directory: {original}" in captured.out
    assert f"current stage: result_dir ({tmp_path.resolve()})" in captured.out


def test_relocated_partial_cli_prints_diagnostics_without_vasprun_exception(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    write_single_stage_submission(
        tmp_path,
        result_dir="/bmd-db/guest/flows/hse06_soc_failed",
    )
    write_partial_diagnostic_outputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "load_resources",
        lambda: (_ for _ in ()).throw(ConfigurationError("missing config")),
    )

    exit_code = cli.main(["--verbose"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Calculation state: UNKNOWN" in captured.out
    assert "Progress:" in captured.out
    assert "execution has started" in captured.out
    assert "Trajectory evidence (trajectory_observation):" in captured.out
    assert "completed ionic steps: 2" in captured.out
    assert "vasprun trajectory enrichment unavailable: file could not be parsed completely" in captured.out
    assert "Diagnostic evidence:" in captured.out
    assert "Custodian intervention evidence:" in captured.out
    assert "VaspErrorHandler" in captured.out
    assert "error archives:" in captured.out
    assert "not unpacked by BMD Agent" in captured.out
    assert "Suggested checks:" in captured.out
    assert "list index out of range" not in captured.out
    assert "Scientific observations:" in captured.out
    assert "final-result parsing incomplete" in captured.out


def test_relocated_bmd_snapshot_does_not_write_calculation_directory(tmp_path: Path) -> None:
    write_single_stage_submission(
        tmp_path,
        result_dir="/bmd-db/guest/flows/vasp_run_hse_static-20260830",
    )
    write_inputs(tmp_path)
    (tmp_path / "OUTCAR").write_text("partial", encoding="utf-8")
    before = sorted(path.name for path in tmp_path.iterdir())

    analyze_calculation_directory(tmp_path)

    after = sorted(path.name for path in tmp_path.iterdir())
    assert after == before


def test_relocated_three_stage_workflow_rebases_declared_paths_and_completes(tmp_path: Path) -> None:
    producer_root = "/bmd-db/guest/flows/vasp_run_custom_workflow-20260912-170926"
    write_three_stage_submission(tmp_path, producer_root=producer_root)
    for label in ("stage_01", "stage_02", "stage_03"):
        write_stage_complete(tmp_path / label)

    analysis = analyze_calculation_directory(
        tmp_path,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="RUNNING"),
    )

    assert analysis.state == LifecycleState.COMPLETED
    assert analysis.calculation_kind == "BMD Compute"
    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.relocated is True
    assert analysis.bmd_workflow.producer_root == producer_root
    assert [binding.label for binding in analysis.bmd_workflow.stage_bindings] == [
        "stage_01",
        "stage_02",
        "stage_03",
    ]
    assert [binding.path for binding in analysis.bmd_workflow.stage_bindings] == [
        (tmp_path / "stage_01").resolve(),
        (tmp_path / "stage_02").resolve(),
        (tmp_path / "stage_03").resolve(),
    ]
    assert [binding.producer_path for binding in analysis.bmd_workflow.stage_bindings] == [
        f"{producer_root}/stage_01",
        f"{producer_root}/stage_02",
        f"{producer_root}/stage_03",
    ]
    assert analysis.bmd_workflow.current_stage is not None
    assert analysis.bmd_workflow.current_stage.label == "stage_03"
    assert analysis.input_files["INCAR"].path == (tmp_path / "stage_03" / "INCAR").resolve()
    assert analysis.scheduler is None


def test_relocated_three_stage_invocation_from_declared_stage_directory(tmp_path: Path) -> None:
    write_three_stage_submission(tmp_path)
    for label in ("stage_01", "stage_02", "stage_03"):
        write_stage_complete(tmp_path / label)

    analysis = analyze_calculation_directory(tmp_path / "stage_03")

    assert analysis.state == LifecycleState.COMPLETED
    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.relocated is True
    assert analysis.bmd_workflow.workflow_root == tmp_path.resolve()
    assert analysis.bmd_workflow.current_stage is not None
    assert analysis.bmd_workflow.current_stage.label == "stage_03"
    assert analysis.calculation_kind == "BMD Compute"


def test_relocated_three_stage_partial_terminal_without_scheduler_is_unknown(tmp_path: Path) -> None:
    write_three_stage_submission(tmp_path)
    write_stage_complete(tmp_path / "stage_01")
    write_stage_complete(tmp_path / "stage_02")
    write_stage_partial(tmp_path / "stage_03")

    analysis = analyze_calculation_directory(
        tmp_path,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="RUNNING"),
    )

    assert analysis.state == LifecycleState.UNKNOWN
    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.current_stage is not None
    assert analysis.bmd_workflow.current_stage.label == "stage_03"
    assert analysis.scheduler is None
    assert analysis.normal_completion is False


def test_relocated_cli_lists_multistage_acquisition_and_producer_provenance(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    producer_root = "/bmd-db/guest/flows/vasp_run_custom_workflow-20260912-170926"
    write_three_stage_submission(tmp_path, producer_root=producer_root)
    for label in ("stage_01", "stage_02", "stage_03"):
        write_stage_complete(tmp_path / label)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "load_resources",
        lambda: (_ for _ in ()).throw(ConfigurationError("missing config")),
    )

    exit_code = cli.main(["--verbose"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Calculation state: COMPLETED" in captured.out
    assert f"current acquisition directory: {tmp_path.resolve()}" in captured.out
    assert f"original producer run directory: {producer_root}" in captured.out
    assert "1. stage_01 - completed (VASP normal termination; convergence established)" in captured.out
    assert "2. stage_02 - completed (VASP normal termination; convergence established)" in captured.out
    assert "3. stage_03 - completed (VASP normal termination; convergence established)" in captured.out
    assert f"current stage: stage_03 ({(tmp_path / 'stage_03').resolve()})" in captured.out


def test_relocated_producer_path_traversal_is_not_rebased(tmp_path: Path) -> None:
    producer_root = "/bmd-db/guest/flows/run123"
    write_three_stage_submission(
        tmp_path,
        producer_root=producer_root,
        stage_paths={
            "stage_01": f"{producer_root}/stage_01",
            "stage_02": f"{producer_root}/../outside/stage_02",
            "stage_03": f"{producer_root}/stage_03",
        },
        result_dir=f"{producer_root}/stage_03",
    )
    write_stage_complete(tmp_path / "stage_01")
    write_stage_complete(tmp_path / "stage_02")
    write_stage_complete(tmp_path / "stage_03")

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.relocated is True
    assert [binding.label for binding in analysis.bmd_workflow.stage_bindings] == [
        "stage_01",
        "stage_03",
    ]
    assert all(binding.producer_path != f"{producer_root}/../outside/stage_02" for binding in analysis.bmd_workflow.stage_bindings)
    assert analysis.state != LifecycleState.COMPLETED


def test_valid_producer_stage_outside_root_is_not_treated_as_relocated(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    root.mkdir()
    producer_stage = tmp_path / "producer_stage"
    producer_stage.mkdir()
    write_inputs(root)
    write_inputs(producer_stage)
    write_submission(root, stage_dir=producer_stage)

    analysis = analyze_calculation_directory(root)

    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.relocated is False
    assert analysis.bmd_workflow.current_stage is not None
    assert analysis.bmd_workflow.current_stage.path == producer_stage.resolve()


def test_arbitrary_stage_named_directory_without_provenance_is_manual(tmp_path: Path) -> None:
    stage = tmp_path / "stage_01"
    stage.mkdir()
    write_inputs(stage)

    analysis = analyze_calculation_directory(stage)

    assert analysis.calculation_kind == "direct VASP"
    assert analysis.bmd_workflow is None


def test_arbitrary_stage_03_directory_without_provenance_is_manual(tmp_path: Path) -> None:
    stage = tmp_path / "stage_03"
    stage.mkdir()
    write_inputs(stage)

    analysis = analyze_calculation_directory(stage)

    assert analysis.calculation_kind == "direct VASP"
    assert analysis.bmd_workflow is None


def test_early_stage_failure_with_later_stage_absent_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage_1 = root / "stage_01"
    stage_2 = root / "stage_02"
    stage_1.mkdir(parents=True)
    write_inputs(stage_1)
    (stage_1 / "OUTCAR").write_text("partial", encoding="utf-8")
    payload = {
        "flow_spec": {
            "workflow_spec": {
                "stages": [
                    {"stage_type": "relax", "theory": "pbe", "modifiers": [], "options": {}},
                    {"stage_type": "static", "theory": "hse06", "modifiers": [], "options": {}},
                ]
            }
        },
        "submission": {"job_id": "21153721"},
        "paths": {
            "stage_dirs": {"stage_01": str(stage_1), "stage_02": str(stage_2)},
            "result_dir": str(stage_2),
        },
    }
    (root / "submission.json").write_text(json.dumps(payload), encoding="utf-8")

    analysis = analyze_calculation_directory(
        root,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="FAILED", exit_code="1:0"),
    )

    assert analysis.state == LifecycleState.INCOMPLETE
    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.current_stage is not None
    assert analysis.bmd_workflow.current_stage.label == "stage_01"


def test_potcar_contents_are_not_read(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    (tmp_path / "POTCAR").write_text("secret-potcar", encoding="utf-8")

    analysis = analyze_calculation_directory(tmp_path)

    observed_paths = [item.path.name for item in analysis.input_files.values()]
    observed_paths.extend(item.path.name for item in analysis.output_files.values())
    assert "POTCAR" not in observed_paths


def test_bare_cli_routes_to_cwd_lifecycle_analysis(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    write_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "load_resources",
        lambda: (_ for _ in ()).throw(ConfigurationError("missing config")),
    )

    exit_code = cli.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Status: PRE_RUN" in captured.out
    assert "BMD Agent" in captured.out
    assert "bmd-agent --verbose" in captured.out


def test_explicit_path_and_cwd_produce_equivalent_lifecycle_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    write_inputs(tmp_path)
    monkeypatch.setattr(
        cli,
        "load_resources",
        lambda: (_ for _ in ()).throw(ConfigurationError("missing config")),
    )

    assert cli.main([str(tmp_path)]) == 0
    explicit_output = capsys.readouterr().out

    monkeypatch.chdir(tmp_path)
    assert cli.main([]) == 0
    cwd_output = capsys.readouterr().out

    assert explicit_output.split("Detailed evidence:", 1)[0] == cwd_output.split(
        "Detailed evidence:", 1
    )[0]
    detail_target = f'"{tmp_path}"' if any(char.isspace() for char in str(tmp_path)) else str(tmp_path)
    assert f"bmd-agent {detail_target} --verbose" in explicit_output
    assert "bmd-agent --verbose" in cwd_output


def test_existing_explicit_cli_commands_remain_available(capsys) -> None:
    exit_code = cli.main(["job"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Usage: bmd-agent job <SLURM_JOB_ID>" in captured.out
