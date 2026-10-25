"""Convergence of completed VASP calculations from VASP's own evidence.

Policy: a stage is converged when VASP terminated normally, the final
electronic cycle converged (existing electronic assessment) and, for
relaxations (NSW > 0), ionic convergence is established. VASP's explicit
OUTCAR declaration ``reached required accuracy - stopping structural energy
minimisation`` establishes VASP-reported ionic convergence, including for
ISIF = 3; it is kept distinct from convergence reconstructed from forces.

The calculation directories are SYNTHETIC. The motivating case reproduces
only the evidence of the historical r2SCAN ISIF = 3 relaxation mp-1025387
(normal termination, converged electronic cycles, EDIFFG = -0.02 eV/A, final
maximum atomic force 0.0092 eV/A, and the declaration). The structure,
energies and other values are invented; no research data is copied.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bmd_agent.resources.completion import STAGE_CONVERGED, STAGE_NOT_CONVERGED
from bmd_agent.resources.lifecycle import LifecycleState, analyze_calculation_directory
from bmd_agent.resources.run import (
    CONVERGED,
    INSUFFICIENT_EVIDENCE,
    VASP_STRUCTURAL_CONVERGENCE_MARKER,
    assess_convergence_progress,
    outcar_structural_convergence_declaration,
)
from bmd_agent.resources.slurm import SlurmAccountingRecord


POSCAR = "Si\n1.0\n5.43 0 0\n0 5.43 0\n0 0 5.43\nSi\n2\nDirect\n0 0 0\n0.25 0.25 0.25\n"
KPOINTS = "Automatic mesh\n0\nGamma\n6 6 6\n0 0 0\n"
INCAR_R2SCAN_VC_RELAX = (
    "METAGGA = R2SCAN\nLASPH = .TRUE.\nENCUT = 680\nEDIFF = 1E-6\nEDIFFG = -0.02\n"
    "IBRION = 2\nISIF = 3\nNSW = 99\nNELM = 200\n"
)
INCAR_PBE_RELAX = INCAR_R2SCAN_VC_RELAX.replace("METAGGA = R2SCAN\nLASPH = .TRUE.\n", "").replace("ISIF = 3", "ISIF = 2")
INCAR_PBE_STATIC = "ENCUT = 520\nEDIFF = 1E-6\nNSW = 0\nIBRION = -1\nNELM = 100\n"
INCAR_HSE_STATIC = INCAR_PBE_STATIC + "LHFCALC = .TRUE.\nHFSCREEN = 0.2\nALGO = Damped\n"

HEADER = " vasp.6.4.2 20Jul23 (build Oct 27 2023 10:00:00) complex\n executed on LinuxIFC date 2026.01.01\n"
TIMING = (
    " writing wavefunctions\n"
    "     LOOP+:  cpu time   12.3: real time   12.5\n"
    " General timing and accounting informations for this job:\n"
    " ========================================================\n"
    "                  Total CPU time used (sec):      123.456\n"
    "                            Voluntary context switches:         1234\n"
)
OSZICAR_HEADER = "       N       E                     dE             d eps       ncg     rms          rms(c)\n"
UNFINISHED_STEP = OSZICAR_HEADER + "DAV:   1   -1.000000000000E+01   -1.00000E+01   -1.00000E+01   10   1.000E+00   2.000E-01\n"


def force_block(force: float) -> str:
    return (
        "  FORCE on cell =-STRESS in cart. coord.  units (eV):\n"
        "  Total        0.01      0.01      0.01      0.00      0.00      0.00\n"
        " POSITION                                       TOTAL-FORCE (eV/Angst)\n"
        " -----------------------------------------------------------------------------------\n"
        f"      0.00000000      0.00000000      0.00000000      {force:.8f}      0.00000000      0.00000000\n"
        f"      1.35750000      1.35750000      1.35750000     {-force:.8f}      0.00000000      0.00000000\n"
        " -----------------------------------------------------------------------------------\n"
        "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n"
    )


def oszicar_step(step: int, *, converged: bool = True) -> str:
    final = "-1.00000E-07   -2.00000E-08" if converged else "-1.00000E-02   -2.00000E-03"
    return (
        OSZICAR_HEADER
        + "DAV:   1   -1.000000000000E+01   -1.00000E+01   -1.00000E+01   10   1.000E+00   2.000E-01\n"
        + f"RMM:   2   -1.080000000000E+01   {final}   12   1.000E-04   2.000E-05\n"
        + f"{step:4d} F= -.10800000E+02 E0= -.10800000E+02  d E =-.100000E-02\n"
    )


def oszicar(steps: int, *, last_converged: bool = True) -> str:
    return "".join(oszicar_step(step, converged=last_converged or step < steps) for step in range(1, steps + 1))


def outcar(forces: tuple[float, ...], *, declared: bool = True, terminated: bool = True) -> str:
    text = HEADER + "".join(force_block(force) for force in forces)
    if declared:
        text += f"\n {VASP_STRUCTURAL_CONVERGENCE_MARKER}\n"
    if terminated:
        text += TIMING
    return text


# mp-1025387 evidence: ISIF = 3, EDIFFG = -0.02 eV/A, final maximum force 0.0092 eV/A.
TARGET_FORCES = (0.4100, 0.0815, 0.0092)


def write_calculation(directory: Path, *, incar: str, outcar_text: str, oszicar_text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "POSCAR").write_text(POSCAR)
    (directory / "KPOINTS").write_text(KPOINTS)
    (directory / "INCAR").write_text(incar)
    (directory / "OSZICAR").write_text(oszicar_text)
    (directory / "OUTCAR").write_text(outcar_text)
    return directory


def write_relaxation(
    directory: Path,
    *,
    outcar_text: str | None = None,
    oszicar_text: str | None = None,
    incar: str = INCAR_R2SCAN_VC_RELAX,
) -> Path:
    return write_calculation(
        directory,
        incar=incar,
        outcar_text=outcar(TARGET_FORCES) if outcar_text is None else outcar_text,
        oszicar_text=oszicar(len(TARGET_FORCES)) if oszicar_text is None else oszicar_text,
    )


def assessments(analysis):
    # A COMPLETED analysis does not carry diagnostics; the stage evidence is
    # then re-observed exactly as lifecycle observes it for stage completion.
    if analysis.diagnostics is not None:
        return analysis.diagnostics.assessments
    from bmd_agent.resources import lifecycle

    if analysis.bmd_workflow is not None:
        bindings = analysis.bmd_workflow.stage_bindings
    else:
        bindings = (lifecycle.LocalStageBinding("work_dir", analysis.directory, 1),)
    trajectories = tuple(lifecycle._observe_local_stage_trajectory(binding, None) for binding in bindings)
    return assess_convergence_progress(trajectories)


def scope(analysis, name: str, index: int = 1):
    return next(item for item in assessments(analysis) if item.scope == name and item.stage_index == index)


def has_scope(analysis, name: str) -> bool:
    return any(item.scope == name for item in assessments(analysis))


VASP_REPORTED = "vasp_reported: OUTCAR declares 'reached required accuracy"
FORCE_RECONSTRUCTED = "trajectory_observation: final OUTCAR atomic maximum force reached EDIFFG"


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


def test_declaration_followed_by_normal_termination_is_recognized() -> None:
    assert outcar_structural_convergence_declaration(outcar(TARGET_FORCES)) == (True, None)
    assert outcar_structural_convergence_declaration(outcar(TARGET_FORCES).encode()) == (True, None)
    assert outcar_structural_convergence_declaration(outcar(TARGET_FORCES, declared=False)) == (None, None)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (outcar(TARGET_FORCES, terminated=False), "not followed by VASP normal-termination"),
        (outcar(TARGET_FORCES, terminated=False) + force_block(0.3), "further ionic steps"),
        (outcar(TARGET_FORCES) + HEADER + force_block(0.3) + TIMING, "later VASP run"),
    ],
    ids=["no-termination", "later-ionic-step", "later-vasp-run"],
)
def test_declaration_followed_by_a_later_segment_or_no_termination_is_not_recognized(text: str, reason: str) -> None:
    declared, note = outcar_structural_convergence_declaration(text)
    assert declared is None and reason in note


# ---------------------------------------------------------------------------
# Static calculations: ionic convergence is not applicable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("incar", [INCAR_PBE_STATIC, INCAR_HSE_STATIC], ids=["pbe", "hse"])
def test_completed_static_calculations_are_completed(tmp_path: Path, incar: str) -> None:
    analysis = analyze_calculation_directory(
        write_calculation(tmp_path / "static", incar=incar, outcar_text=HEADER + TIMING, oszicar_text=oszicar(1))
    )
    assert analysis.state == LifecycleState.COMPLETED
    assert not has_scope(analysis, "ionic")
    assert scope(analysis, "electronic").label == CONVERGED


def test_static_calculation_without_electronic_convergence_is_incomplete(tmp_path: Path) -> None:
    analysis = analyze_calculation_directory(
        write_calculation(
            tmp_path / "static",
            incar=INCAR_PBE_STATIC,
            outcar_text=HEADER + TIMING,
            oszicar_text=oszicar(1, last_converged=False),
        )
    )
    assert analysis.state == LifecycleState.INCOMPLETE


# ---------------------------------------------------------------------------
# Relaxations
# ---------------------------------------------------------------------------


def test_mp_1025387_isif3_relaxation_with_vasp_declared_convergence_is_completed(tmp_path: Path) -> None:
    analysis = analyze_calculation_directory(write_relaxation(tmp_path / "mp-1025387-synthetic"))
    assert analysis.state == LifecycleState.COMPLETED
    (stage,) = analysis.stage_completion
    assert stage.terminated_normally is True and stage.convergence.status == STAGE_CONVERGED
    ionic = scope(analysis, "ionic")
    assert ionic.label == CONVERGED
    # VASP-reported and reconstructed evidence stay distinct.
    assert any(item.startswith(VASP_REPORTED) for item in ionic.basis)
    assert FORCE_RECONSTRUCTED in ionic.basis
    assert any("cell convergence is as reported by VASP" in item for item in ionic.limitations)
    assert ionic.features["vasp_reported_ionic_convergence"] is True
    assert ionic.features["final_max_force"] == pytest.approx(0.0092)


def test_isif3_relaxation_is_accepted_from_the_declaration_without_reconstructed_forces(tmp_path: Path) -> None:
    analysis = analyze_calculation_directory(
        write_relaxation(tmp_path / "vc", outcar_text=HEADER + f" {VASP_STRUCTURAL_CONVERGENCE_MARKER}\n" + TIMING)
    )
    assert analysis.state == LifecycleState.COMPLETED
    assert scope(analysis, "ionic").basis == (
        "vasp_reported: OUTCAR declares 'reached required accuracy - stopping structural energy minimisation', "
        "followed by normal termination",
    )


def test_successful_isif2_relaxation_is_completed(tmp_path: Path) -> None:
    analysis = analyze_calculation_directory(write_relaxation(tmp_path / "relax", incar=INCAR_PBE_RELAX))
    assert analysis.state == LifecycleState.COMPLETED
    ionic = scope(analysis, "ionic")
    assert ionic.label == CONVERGED
    assert FORCE_RECONSTRUCTED in ionic.basis and any(item.startswith(VASP_REPORTED) for item in ionic.basis)
    assert not any("cell convergence" in item for item in ionic.limitations)


def test_isif3_relaxation_without_the_declaration_stays_unknown(tmp_path: Path) -> None:
    analysis = analyze_calculation_directory(write_relaxation(tmp_path / "vc", outcar_text=outcar(TARGET_FORCES, declared=False)))
    assert scope(analysis, "ionic").label == INSUFFICIENT_EVIDENCE
    assert analysis.state == LifecycleState.UNKNOWN


@pytest.mark.parametrize("incar", [INCAR_PBE_RELAX, INCAR_R2SCAN_VC_RELAX], ids=["isif2", "isif3"])
def test_relaxation_exhausting_nsw_without_convergence_is_incomplete(tmp_path: Path, incar: str) -> None:
    analysis = analyze_calculation_directory(
        write_relaxation(
            tmp_path / "nsw", incar=incar.replace("NSW = 99", "NSW = 3"), outcar_text=outcar((0.41, 0.30, 0.20), declared=False)
        )
    )
    assert scope(analysis, "ionic").label != CONVERGED
    assert analysis.state == LifecycleState.INCOMPLETE


def test_declared_relaxation_with_unconverged_final_electronic_cycle_is_incomplete(tmp_path: Path) -> None:
    analysis = analyze_calculation_directory(write_relaxation(tmp_path / "scf", oszicar_text=oszicar(3, last_converged=False)))
    assert scope(analysis, "electronic").label != CONVERGED
    assert analysis.state == LifecycleState.INCOMPLETE


def test_missing_normal_termination_is_not_completed(tmp_path: Path) -> None:
    analysis = analyze_calculation_directory(write_relaxation(tmp_path / "killed", outcar_text=outcar(TARGET_FORCES, terminated=False)))
    assert analysis.diagnostics.trajectories[0].vasp_reported_ionic_convergence is None
    (stage,) = analysis.stage_completion
    assert stage.terminated_normally is False
    assert analysis.state != LifecycleState.COMPLETED


def test_declaration_contradicted_by_the_reconstructed_final_force_is_unknown(tmp_path: Path) -> None:
    analysis = analyze_calculation_directory(write_relaxation(tmp_path / "vc", outcar_text=outcar((0.41, 0.0815, 0.05))))
    ionic = scope(analysis, "ionic")
    assert ionic.label == INSUFFICIENT_EVIDENCE and ionic.sufficiency == "contradictory"
    assert analysis.state == LifecycleState.UNKNOWN


# ---------------------------------------------------------------------------
# Obvious stale-file and restarted-run contradictions
# ---------------------------------------------------------------------------


def test_newer_incomplete_run_after_a_converged_outcar_is_contradictory(tmp_path: Path) -> None:
    analysis = analyze_calculation_directory(write_relaxation(tmp_path / "restart", oszicar_text=oszicar(1) + UNFINISHED_STEP))
    ionic = scope(analysis, "ionic")
    assert ionic.label == INSUFFICIENT_EVIDENCE and ionic.sufficiency == "contradictory"
    assert any("unfinished ionic step" in item for item in ionic.counter_evidence)
    assert analysis.state != LifecycleState.COMPLETED


def test_later_vasp_run_in_the_same_outcar_is_judged_on_its_own_evidence(tmp_path: Path) -> None:
    text = outcar(TARGET_FORCES) + HEADER + force_block(0.30) + force_block(0.25) + TIMING
    analysis = analyze_calculation_directory(
        write_relaxation(
            tmp_path / "appended",
            outcar_text=text,
            oszicar_text=oszicar(2),
            incar=INCAR_R2SCAN_VC_RELAX.replace("NSW = 99", "NSW = 2"),
        )
    )
    trajectory = analysis.diagnostics.trajectories[0]
    assert trajectory.vasp_reported_ionic_convergence is None
    assert any("later VASP run" in item for item in trajectory.unavailable)
    assert not any(item.startswith(VASP_REPORTED) for item in scope(analysis, "ionic").basis)
    assert analysis.state != LifecycleState.COMPLETED


def test_more_completed_steps_than_nsw_is_contradictory(tmp_path: Path) -> None:
    analysis = analyze_calculation_directory(
        write_relaxation(tmp_path / "steps", incar=INCAR_R2SCAN_VC_RELAX.replace("NSW = 99", "NSW = 2"))
    )
    ionic = scope(analysis, "ionic")
    assert ionic.sufficiency == "contradictory"
    assert "trajectory_observation: OSZICAR records 3 completed ionic steps, more than NSW = 2" in ionic.counter_evidence
    assert analysis.state != LifecycleState.COMPLETED


# ---------------------------------------------------------------------------
# Multistage workflows: each stage must converge independently
# ---------------------------------------------------------------------------


def write_workflow(root: Path, stages: list[tuple[str, str, str]], stage_types: tuple[str, ...] = ()) -> Path:
    directories = {}
    for index, (incar, outcar_text, oszicar_text) in enumerate(stages, start=1):
        directories[f"stage_{index:02d}"] = write_calculation(
            root / f"stage_{index:02d}", incar=incar, outcar_text=outcar_text, oszicar_text=oszicar_text
        )
    (root / "attempt.json").write_text(json.dumps({"attempt_id": "a", "state": "SUBMITTED", "job_id": "777"}))
    payload = {
        "flow_spec": {
            "workflow_spec": {
                "stages": [
                    {"stage_type": stage_type, "theory": "r2scan", "modifiers": [], "options": {}}
                    for stage_type in (stage_types or ("relax",) * len(stages))
                ]
            }
        },
        "submission": {"attempt_id": "a", "attempt_state": str(root / "attempt.json")},
        "paths": {
            "stage_dirs": {label: str(path) for label, path in directories.items()},
            "result_dir": str(list(directories.values())[-1]),
        },
    }
    (root / "submission.json").write_text(json.dumps(payload))
    return root


def analyze_workflow(root: Path):
    def lookup(job_id: str) -> SlurmAccountingRecord:
        return SlurmAccountingRecord(
            job_id=job_id, name="vasp", state="COMPLETED", elapsed="01:00:00", start="2026-01-01T10:00:00", end="",
            partition="p", exit_code="0:0", timelimit="02:00:00", node_list="n", allocated_cpus=24, work_dir="/unused",
        )

    return analyze_calculation_directory(root, scheduler_lookup=lookup)


CONVERGED_VC_RELAX = (INCAR_R2SCAN_VC_RELAX, outcar(TARGET_FORCES), oszicar(3))


def test_workflow_completes_when_every_stage_converges(tmp_path: Path) -> None:
    analysis = analyze_workflow(write_workflow(tmp_path / "flow", [CONVERGED_VC_RELAX, CONVERGED_VC_RELAX]))
    assert all(stage.convergence.status == STAGE_CONVERGED for stage in analysis.stage_completion)
    assert analysis.state == LifecycleState.COMPLETED


def test_earlier_stage_declaration_never_certifies_a_later_unconverged_stage(tmp_path: Path) -> None:
    unconverged = (INCAR_R2SCAN_VC_RELAX.replace("NSW = 99", "NSW = 3"), outcar((0.41, 0.2, 0.15), declared=False), oszicar(3))
    analysis = analyze_workflow(write_workflow(tmp_path / "flow", [CONVERGED_VC_RELAX, unconverged]))
    stages = {stage.index: stage for stage in analysis.stage_completion}
    assert stages[1].convergence.status == STAGE_CONVERGED
    assert stages[2].convergence.status == STAGE_NOT_CONVERGED
    assert not any(item.startswith(VASP_REPORTED) for item in scope(analysis, "ionic", 2).basis)
    assert analysis.state == LifecycleState.INCOMPLETE


def test_relaxation_then_static_workflow_completes(tmp_path: Path) -> None:
    static = (INCAR_HSE_STATIC, HEADER + TIMING, oszicar(1))
    analysis = analyze_workflow(
        write_workflow(tmp_path / "flow", [CONVERGED_VC_RELAX, static], stage_types=("relax", "static"))
    )
    assert analysis.state == LifecycleState.COMPLETED
