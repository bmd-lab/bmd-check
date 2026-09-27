import base64
import json
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import types
import warnings

import pytest

from bmd_agent import cli
from bmd_agent.resources.compute import compute_policy_not_configured
import bmd_agent.resources.run as run_resource
from bmd_agent.config import ResourceRegistry, SlurmClusterResource
from bmd_agent.deployment import DeploymentContext, load_deployment_profile
from bmd_agent.profiling import PerformanceProfiler
from bmd_agent.resources.bmdex import build_bmdex_domain_query_for_job
from bmd_agent.resources.job_resolution import NOT_BMD_COMPUTE, RESOLVED
from bmd_agent.resources.run import (
    AGENT_COMPARISON,
    ARTIFACT_OBSERVATION,
    CONVERGED,
    EVIDENCE_OF_PROGRESS,
    EXECUTED_INPUT,
    INSUFFICIENT_EVIDENCE,
    LOG_OBSERVATION,
    NO_CLEAR_EVIDENCE_OF_PROGRESS,
    PYMATGEN_DERIVED,
    AttemptStateObservation,
    ComparisonObservation,
    ConvergenceProgressAssessment,
    ElectronicCycleObservation,
    ElectronicIterationObservation,
    IncarObservation,
    InitialStructureObservation,
    InputExpectationObservation,
    IonicStepObservation,
    JobInspection,
    LogRuntimeObservation,
    OutcarForceBlockObservation,
    PathObservation,
    RunDiagnosis,
    RunInspection,
    RunInspectionError,
    ScientificResult,
    StageTrajectoryObservation,
    TRAJECTORY_PROGRESS_EVIDENCE,
    TerminationObservation,
    StructureObservation,
    TRAJECTORY_OBSERVATION,
    WorkflowStage,
    assess_convergence_progress,
    build_run_comparison,
    compare_remote_runs,
    compare_requested_options_to_executed_inputs,
    derive_trajectory_progress_evidence,
    diagnose_remote_run,
    inspect_remote_run,
    inspect_slurm_job,
    parse_incar_contents,
    parse_oszicar_trajectory,
    parse_outcar_force_blocks,
    parse_outcar_force_extraction,
    parse_vasp_output_files,
    run_label_from_provenance,
    serialize_job_trajectory_evidence,
    vasp_reported_parameter_observations,
)
from bmd_agent.resources.oom import (
    INSUFFICIENT_OOM_EVIDENCE,
    NO_OOM_EVIDENCE,
    OOM_ESTABLISHED,
)
from bmd_agent.resources.remote import ReusableSshSession
from bmd_agent.resources.slurm import (
    DEFAULT_SCHEDULER_ACCOUNTING_TIMEOUT_SECONDS,
    SlurmAccountingRecord,
)
from bmd_agent.resources.vasp import RemotePathError, remote_acquisition_cache


FLOW_ROOT = "/bmd-db/guest/flows/validation-run"
DIRECT_DIR = "/bmd-db/guest/flows/direct-vasp"
LOG_ROOT = "/bmd-db/guest/logs"
RESULT_DIR = f"{FLOW_ROOT}/producer-delta"
SACCT_FORMAT = (
    "--format=JobIDRaw,JobName%30,User%20,Account%30,State,ExitCode,Reason%40,Elapsed,"
    "ElapsedRaw,Start,End,Partition%20,Timelimit%20,NodeList%80,NNodes,"
    "AllocCPUS,NTasks,ReqMem,ReqTRES%120,AllocTRES%120,TotalCPU,CPUTimeRAW,"
    "MaxRSS,MaxVMSize,AveRSS,StdOut%160,StdErr%160,WorkDir%160"
)
OSZICAR_TWO_STEP = b"""\
       N       E                     dE             d eps       ncg     rms          rms(c)
DAV:   1   -1.000000000000E+01   -1.00000E+01   -1.00000E+01   10   1.000E+00   2.000E-01
DAV:   2   -1.100000000000E+01   -1.00000E+00   -2.00000E-01   12   1.000E-01   2.000E-02
   1 F= -.11000000E+02 E0= -.10950000E+02  d E =-.110000E+02
       N       E                     dE             d eps       ncg     rms          rms(c)
RMM:   1   -1.200000000000E+01   -1.00000E+00   -1.00000E-01   10   5.000E-02   1.000E-02
RMM:   2   -1.210000000000E+01   -1.00000E-01   -1.00000E-02   12   4.000E-02   8.000E-03
   2 F= -.12100000E+02 E0= -.12050000E+02  d E =-.110000E+01
"""
OSZICAR_NELM_LIMIT = b"""\
       N       E                     dE             d eps       ncg     rms          rms(c)
DAV:   1   -1.000000000000E+01   -1.00000E+01   -1.00000E+01   10   1.000E+00   2.000E-01
DAV:   2   -1.100000000000E+01   -1.00000E+00   -2.00000E-01   12   1.000E-01   2.000E-02
DAV:   3   -1.110000000000E+01   -1.00000E-01   -2.00000E-02   12   9.000E-02   1.000E-02
   1 F= -.11100000E+02 E0= -.11050000E+02  d E =-.111000E+02
"""
POSCAR_TWO_SITE = b"""\
Example
1.0
1 0 0
0 1 0
0 0 1
X
2
Direct
0 0 0
0.5 0.5 0.5
"""
OUTCAR_TWO_FORCE_BLOCKS = b"""\
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000000      0.00000000      0.00000000      3.00000000      4.00000000      0.00000000
      0.50000000      0.50000000      0.50000000      0.00000000      0.00000000      1.00000000
 -----------------------------------------------------------------------------------
 total drift:                               0.00000000      0.00000000      0.00000000
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000000      0.00000000      0.00000000      0.00000000      0.30000000      0.40000000
      0.50000000      0.50000000      0.50000000     -0.10000000     -0.20000000     -0.20000000
 -----------------------------------------------------------------------------------
 total drift:                               0.00000000      0.00000000      0.00000000
"""
OUTCAR_INCOMPLETE_FINAL_BLOCK = b"""\
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000000      0.00000000      0.00000000      0.10000000      0.00000000      0.00000000
      0.50000000      0.50000000      0.50000000      0.00000000      0.20000000      0.00000000
 -----------------------------------------------------------------------------------
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000000      0.00000000      0.00000000      0.30000000      0.00000000      0.00000000
"""
OUTCAR_MALFORMED_FORCE_BLOCK = b"""\
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000000      0.00000000      0.00000000      not-a-force      0.00000000      0.00000000
 -----------------------------------------------------------------------------------
"""
OUTCAR_SCIENTIFIC_NOTATION_BLOCK = b"""\
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000000      0.00000000      0.00000000     -1.00000000E-01      2.00000000E-01     -2.00000000E-01
 -----------------------------------------------------------------------------------
"""


def oszicar_incomplete_cycle(iterations: int = 25) -> bytes:
    lines = [
        "       N       E                     dE             d eps       ncg     rms          rms(c)"
    ]
    for index in range(1, iterations + 1):
        lines.append(
            f"DAV: {index:3d}   {-10 - index / 100:.12E}   -1.00000E-02   "
            "-1.00000E-03   10   1.000E-01   2.000E-02"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def trajectory_observation(
    *,
    stage_type: str = "relax",
    criteria: dict[str, object] | None = None,
    completed_ionic_steps: int | None = None,
    electronic_iterations: tuple[int, ...] = (),
    electronic_cycles: tuple[ElectronicCycleObservation, ...] = (),
    incomplete_electronic_iteration_count: int | None = None,
    recent_incomplete: tuple[ElectronicIterationObservation, ...] = (),
    ionic_steps: tuple[IonicStepObservation, ...] = (),
    converged_electronic: bool | None = None,
    converged_ionic: bool | None = None,
    oszicar_present: bool = True,
    oszicar_error: str | None = None,
    vasprun_present: bool = True,
    vasprun_error: str | None = None,
) -> StageTrajectoryObservation:
    recent_electronic = (
        (electronic_cycles[-1].final_iteration,)
        if electronic_cycles and electronic_cycles[-1].final_iteration is not None
        else ()
    )
    return StageTrajectoryObservation(
        stage_index=1,
        stage_label="result_dir",
        stage_type=stage_type,
        theory="pbe",
        directory=FLOW_ROOT,
        oszicar_path=f"{FLOW_ROOT}/OSZICAR",
        oszicar_present=oszicar_present,
        oszicar_error=oszicar_error,
        vasprun_path=f"{FLOW_ROOT}/vasprun.xml",
        vasprun_present=vasprun_present,
        vasprun_error=vasprun_error,
        criteria=criteria or {},
        ionic_steps_observed=completed_ionic_steps,
        electronic_iterations_by_ionic_step=electronic_iterations,
        electronic_cycles=electronic_cycles,
        final_electronic_iteration_count=(
            electronic_iterations[-1] if electronic_iterations else None
        ),
        recent_electronic_iterations=recent_electronic,
        completed_ionic_steps=completed_ionic_steps,
        electronic_iterations_by_completed_ionic_step=electronic_iterations[
            : completed_ionic_steps or 0
        ],
        incomplete_electronic_iteration_count=incomplete_electronic_iteration_count,
        recent_incomplete_electronic_iterations=recent_incomplete,
        ionic_steps=ionic_steps,
        recent_ionic_steps=ionic_steps[-5:],
        vasprun_ionic_steps=completed_ionic_steps,
        converged_electronic=converged_electronic,
        converged_ionic=converged_ionic,
    )


def electronic_cycle(
    cycle_index: int,
    iterations: int,
    *,
    completed: bool = True,
    dE: float | None = None,
    deps: float | None = None,
) -> ElectronicCycleObservation:
    return ElectronicCycleObservation(
        cycle_index=cycle_index,
        completed_ionic_step=completed,
        iterations=iterations,
        final_iteration=ElectronicIterationObservation(
            iteration=iterations,
            algorithm="RMM",
            energy=-10.0,
            dE=dE,
            deps=deps,
        ),
    )


def assessment_by_scope(
    assessments: tuple[ConvergenceProgressAssessment, ...],
    scope: str,
) -> ConvergenceProgressAssessment:
    for assessment in assessments:
        if assessment.scope == scope:
            return assessment
    raise AssertionError(f"missing {scope} assessment")


def force_progress_trajectory(
    forces: tuple[float | None, ...],
    *,
    criteria: dict[str, object] | None = None,
    electronic_counts: tuple[int, ...] | None = None,
    alignment_status: str | None = "aligned",
    alignment_reason: str | None = None,
) -> StageTrajectoryObservation:
    counts = electronic_counts or tuple(8 for _ in forces)
    ionic_steps = tuple(
        IonicStepObservation(
            index,
            counts[index - 1] if index - 1 < len(counts) else None,
            -100.0 - index,
            -99.9 - index,
            -0.1,
            force,
            "OUTCAR" if force is not None else None,
        )
        for index, force in enumerate(forces, start=1)
    )
    blocks = tuple(
        OutcarForceBlockObservation(
            index,
            24,
            "complete",
            True,
            f"{FLOW_ROOT}/OUTCAR",
            force,
        )
        for index, force in enumerate(forces, start=1)
        if force is not None
    )
    trajectory = trajectory_observation(
        stage_type="relax",
        criteria=(
            {"NELM": 200, "EDIFF": 1e-6, "NSW": 99, "EDIFFG": -0.01}
            if criteria is None
            else criteria
        ),
        completed_ionic_steps=len(forces),
        electronic_iterations=counts,
        electronic_cycles=tuple(
            electronic_cycle(index, count, dE=1e-7)
            for index, count in enumerate(counts, start=1)
        ),
        ionic_steps=ionic_steps,
        converged_electronic=True,
        converged_ionic=False,
    )
    return replace(
        trajectory,
        outcar_path=f"{FLOW_ROOT}/OUTCAR",
        outcar_present=True,
        outcar_expected_site_count=24,
        outcar_force_blocks=blocks,
        outcar_complete_force_blocks=len(blocks),
        outcar_force_alignment_status=alignment_status,
        outcar_force_alignment_reason=(
            alignment_reason
            if alignment_reason is not None
            else f"{len(blocks)} OUTCAR force block(s) aligned with OSZICAR completed ionic steps"
        ),
    )


class RemoteFixture:
    def __init__(self, *, files: dict[str, bytes], directories: set[str]) -> None:
        self.files = files
        self.directories = directories
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        self.commands.append(command)
        assert command[:2] == ["ssh", "powerslurm-bmdguest"]
        assert kwargs["capture_output"] is True
        assert kwargs["timeout"] == 20

        remote_command = command[2]
        parts = shlex.split(remote_command)

        if parts[:4] == ["sh", "-s", "--", "bmd-agent-acquisition-v1"]:
            assert kwargs["check"] is True
            total_limit = int(parts[4])
            request_parts = parts[8:]
            assert len(request_parts) == int(parts[5]) * 4
            used = 0
            lines = ["schema\tbmd-agent-acquisition-v1"]
            for offset in range(0, len(request_parts), 4):
                index = int(request_parts[offset])
                kind = request_parts[offset + 1]
                read_limit = int(request_parts[offset + 2])
                path = request_parts[offset + 3]
                if kind == "archives":
                    count = 0
                    for archive_index in range(1, read_limit + 1):
                        candidate = f"{path}/error.{archive_index}.tar.gz"
                        if candidate not in self.files:
                            break
                        count = archive_index
                    lines.append(f"item\t{index}\tarchives\tpresent\t{count}\t")
                    continue
                if kind == "directory":
                    status = "present" if path in self.directories else "missing"
                    lines.append(f"item\t{index}\tdirectory\t{status}\t\t")
                    continue
                contents = self.files.get(path)
                if contents is None:
                    lines.append(f"item\t{index}\tfile\tmissing\t\t")
                    continue
                size = len(contents)
                if not read_limit or size > read_limit or used + size > total_limit:
                    status = "present" if not read_limit else "deferred"
                    lines.append(f"item\t{index}\tfile\t{status}\t{size}\t")
                    continue
                encoded = base64.b64encode(contents).decode("ascii")
                lines.append(f"item\t{index}\tfile\tread\t{size}\t{encoded}")
                used += size
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=("\n".join(lines) + "\n").encode("ascii"),
                stderr=b"",
            )

        if parts[:2] == ["cat", "--"]:
            assert kwargs["check"] is True
            path = parts[2]
            if path not in self.files:
                raise subprocess.CalledProcessError(1, command, stderr=b"missing")
            return subprocess.CompletedProcess(command, 0, stdout=self.files[path], stderr=b"")

        if parts[:2] == ["tail", "-c"]:
            assert kwargs["check"] is True
            limit = int(parts[2])
            assert parts[3] == "--"
            path = parts[4]
            if path not in self.files:
                raise subprocess.CalledProcessError(1, command, stderr=b"missing")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=self.files[path][-limit:],
                stderr=b"",
            )

        if parts[:2] == ["test", "-f"]:
            assert kwargs["check"] is False
            return subprocess.CompletedProcess(
                command,
                0 if parts[2] in self.files else 1,
                stdout=b"",
                stderr=b"",
            )

        if parts[:2] == ["test", "-d"]:
            assert kwargs["check"] is False
            return subprocess.CompletedProcess(
                command,
                0 if parts[2] in self.directories else 1,
                stdout=b"",
                stderr=b"",
            )

        if parts[:2] == ["sh", "-c"] and parts[3:4] == [
            "bmd-agent-archive-probe-v1"
        ]:
            assert kwargs["check"] is False
            directory = parts[4]
            limit = int(parts[5])
            archives: list[str] = []
            for index in range(1, limit + 1):
                path = f"{directory}/error.{index}.tar.gz"
                if path not in self.files:
                    break
                archives.append(path)
            stdout = "".join(f"{path}\n" for path in archives).encode("utf-8")
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

        if parts[:4] == ["stat", "-c", "%s", "--"]:
            assert kwargs["check"] is True
            path = parts[4]
            if path not in self.files:
                raise subprocess.CalledProcessError(1, command, stderr=b"missing")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{len(self.files[path])}\n".encode("utf-8"),
                stderr=b"",
            )

        if len(parts) >= 5 and parts[0] == "awk" and parts[1] == "-v":
            assert kwargs["check"] is True
            expected_raw = parts[2].removeprefix("expected=")
            expected = int(expected_raw) if expected_raw else None
            path = parts[4]
            if path not in self.files:
                raise subprocess.CalledProcessError(1, command, stderr=b"missing")
            blocks = parse_outcar_force_blocks(
                self.files[path],
                source_path=path,
                expected_site_count=expected,
            )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=outcar_extractor_payload(blocks, expected_site_count=expected),
                stderr=b"",
            )

        raise AssertionError(f"unexpected remote command: {remote_command}")


def outcar_extractor_payload(
    blocks: tuple[OutcarForceBlockObservation, ...],
    *,
    expected_site_count: int | None = None,
) -> bytes:
    lines = [
        "schema\tbmd-agent-outcar-force-v1",
        f"expected_site_count\t{expected_site_count or ''}",
    ]
    for block in blocks:
        value = "" if block.max_force_eV_per_A is None else str(block.max_force_eV_per_A)
        lines.append(
            "\t".join(
                (
                    "block",
                    str(block.block_index),
                    str(block.row_count),
                    block.status,
                    "1" if block.complete else "0",
                    value,
                )
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def cluster() -> SlurmClusterResource:
    return SlurmClusterResource(
        key="powerslurm",
        name="PowerSLURM",
        ssh_host="powerslurm-bmdguest",
        partition="leeburton-pool",
        access="observational",
        allowed_remote_roots=(PurePosixPath("/bmd-db/guest"),),
    )


def power_deployment(resource: SlurmClusterResource | None = None) -> DeploymentContext:
    resource = resource or cluster()
    return DeploymentContext(profile=load_deployment_profile("power"), cluster=resource)


def submission_payload(*, outside_path: bool = False) -> dict:
    stage_dirs = {
        "producer-alpha": f"{FLOW_ROOT}/producer-alpha",
        "producer-beta": f"{FLOW_ROOT}/producer-beta",
        "producer-gamma": f"{FLOW_ROOT}/producer-gamma",
        "producer-delta": RESULT_DIR,
    }
    if outside_path:
        stage_dirs["producer-beta"] = "/etc/not-authorized"

    workflow_spec = {
        "stages": [
            {"stage_type": "relax", "theory": "pbe", "modifiers": [], "label": None, "options": {}},
            {
                "stage_type": "static",
                "theory": "r2scan",
                "modifiers": ["custom_modifier"],
                "label": None,
                "options": {"custom_modifier": {"nested": "kept"}},
            },
            {"stage_type": "dos", "theory": "pbe", "modifiers": [], "label": None, "options": {}},
            {
                "stage_type": "band_structure",
                "theory": "pbe",
                "modifiers": ["dispersion"],
                "label": None,
                "options": {"dispersion": {"method": "dftd3"}},
            },
        ],
        "label": None,
        "recipe": "custom",
    }
    return {
        "flow_spec": {
            "workflow": "custom_workflow",
            "workflow_spec": workflow_spec,
            "structure": {
                "type": "pasted_text",
                "format": "poscar",
                "text": "Example\n1\n1 0 0\n0 1 0\n0 0 1\nX\n1\ndirect\n0 0 0\n",
            },
        },
        "paths": {
            "run_dir": FLOW_ROOT,
            "logs_dir": LOG_ROOT,
            "stage_dirs": stage_dirs,
            "result_dir": RESULT_DIR,
            "log_out": f"{LOG_ROOT}/validation-run.out",
            "log_err": f"{LOG_ROOT}/validation-run.err",
            "slurm_out": f"{LOG_ROOT}/validation-run.slurm.out",
            "slurm_err": f"{LOG_ROOT}/validation-run.slurm.err",
        },
        "cluster": {"partition": "leeburton-pool", "account": "account-name"},
        "resources": {"nodes": 1, "ntasks": 24, "mem_gb": 160, "walltime": "04:00:00"},
        "environment": {
            "VASP_CMD": "srun --mpi=pmi2 -n $SLURM_NTASKS vasp_std",
            "PMG_VASP_PSP_DIR": "/bmd-db/potcars",
        },
        "submission": {
            "attempt_state": f"{LOG_ROOT}/submission_attempts/attempt.json",
        },
        "provenance": {
            "bmd_compute": {
                "source": {
                    "git_commit": "abcdef0123456789",
                    "state": "clean",
                    "dirty": False,
                }
            },
            "execution": {"workflow_spec": workflow_spec},
        },
    }


def default_files(*, include_vasprun: bool = True, invalid_job_id: bool = False) -> dict[str, bytes]:
    files = {
        f"{FLOW_ROOT}/submission.json": json.dumps(submission_payload()).encode("utf-8"),
        f"{LOG_ROOT}/submission_attempts/attempt.json": json.dumps(
            {
                "job_id": "20893681;scancel 1" if invalid_job_id else "20893681",
                "job_record": {
                    "job_id": "20893681;scancel 1" if invalid_job_id else "20893681"
                },
            }
        ).encode("utf-8"),
        f"{LOG_ROOT}/validation-run.out": (
            "[runner] python: 3.12.13 (main)\n"
            "[runner] atomate2 version: 0.1.5\n"
            "[runner] jobflow version: 0.1.19\n"
            "[runner] pymatgen version: 2026.8.13\n"
            "[runner] custodian version: 2025.12.14\n"
            "PMG_VASP_PSP_DIR=/bmd-db/potcars\n"
            "2026-08-21 12:00:00 Starting job - stage_01 (64210872-5626-40c7-a7eb-79f7e49272ba)\n"
            "2026-08-21 12:10:00 Starting job - custom.second (328290de-b493-4d39-a4c7-238eb9055720)\n"
            "2026-08-21 12:20:00 Starting job - final stage (48F52369-D3B9-40B5-9A9F-A87BAE6D007F)\n"
        ).encode("utf-8"),
        f"{LOG_ROOT}/validation-run.err": b"",
        f"{LOG_ROOT}/validation-run.slurm.out": b"",
        f"{LOG_ROOT}/validation-run.slurm.err": b"",
        f"{RESULT_DIR}/CONTCAR": b"contcar",
        f"{RESULT_DIR}/OUTCAR": b"outcar",
        f"{RESULT_DIR}/KPOINTS": b"kpoints",
        f"{FLOW_ROOT}/producer-alpha/INCAR": b"ENCUT = 520\nIVDW = 11\n",
        f"{FLOW_ROOT}/producer-delta/INCAR": b"ENCUT = 600\nIVDW = 11\n",
    }
    if include_vasprun:
        files[f"{RESULT_DIR}/vasprun.xml"] = b"vasprun"
    return files


def default_directories() -> set[str]:
    return {
        FLOW_ROOT,
        f"{FLOW_ROOT}/producer-alpha",
        f"{FLOW_ROOT}/producer-beta",
        f"{FLOW_ROOT}/producer-gamma",
        RESULT_DIR,
    }


def single_stage_submission_payload(*, custom: bool = False) -> dict:
    workflow_spec = {
        "stages": [
            {
                "stage_type": "relax",
                "theory": "pbe",
                "modifiers": ["dispersion"],
                "label": None,
                "options": {"dispersion": {"method": "dftd3"}},
            },
        ],
        "label": None,
        "recipe": "custom" if custom else "relax",
    }
    return {
        "flow_spec": {
            "workflow": "custom_workflow" if custom else "relax",
            "workflow_spec": workflow_spec,
            "structure": {
                "type": "pasted_text",
                "format": "poscar",
                "text": "Example\n1\n1 0 0\n0 1 0\n0 0 1\nX\n1\ndirect\n0 0 0\n",
            },
        },
        "paths": {
            "run_dir": FLOW_ROOT,
            "logs_dir": LOG_ROOT,
            "stage_dirs": {},
            "result_dir": FLOW_ROOT,
            "log_out": f"{LOG_ROOT}/validation-run.out",
            "log_err": f"{LOG_ROOT}/validation-run.err",
        },
        "cluster": {"partition": "leeburton-pool", "account": "account-name"},
        "resources": {"nodes": 1, "ntasks": 24, "mem_gb": 160, "walltime": "04:00:00"},
        "environment": {"VASP_CMD": "mpirun -n $SLURM_NTASKS vasp_std"},
        "provenance": {
            "bmd_compute": {
                "source": {
                    "git_commit": "abcdef0123456789",
                    "state": "clean",
                    "dirty": False,
                }
            },
            "execution": {"workflow_spec": workflow_spec},
        },
    }


def single_stage_files(*, custom: bool = False) -> dict[str, bytes]:
    return {
        f"{FLOW_ROOT}/submission.json": json.dumps(
            single_stage_submission_payload(custom=custom)
        ).encode("utf-8"),
        f"{LOG_ROOT}/validation-run.out": b"",
        f"{LOG_ROOT}/validation-run.err": b"",
        f"{FLOW_ROOT}/POSCAR": POSCAR_TWO_SITE,
        f"{FLOW_ROOT}/CONTCAR": b"contcar",
        f"{FLOW_ROOT}/OUTCAR": OUTCAR_TWO_FORCE_BLOCKS,
        f"{FLOW_ROOT}/vasprun.xml": b"vasprun",
        f"{FLOW_ROOT}/INCAR": b"IVDW = 11\n",
    }


def legacy_static_submission_payload() -> dict:
    workflow_spec = {
        "stages": [
            {
                "stage_type": "static",
                "theory": "pbe",
                "modifiers": [],
                "label": None,
                "options": {},
            },
        ],
        "label": None,
        "recipe": "static",
    }
    return {
        "flow_spec": {
            "workflow": "static",
            "workflow_spec": workflow_spec,
            "structure": {
                "type": "pasted_text",
                "format": "poscar",
                "text": "Example\n1\n1 0 0\n0 1 0\n0 0 1\nX\n1\ndirect\n0 0 0\n",
            },
        },
        "paths": {
            "run_dir": FLOW_ROOT,
            "logs_dir": LOG_ROOT,
            "result_dir": FLOW_ROOT,
            "log_out": f"{LOG_ROOT}/legacy-static.out",
            "log_err": f"{LOG_ROOT}/legacy-static.err",
        },
        "cluster": {"partition": "leeburton-pool", "account": "account-name"},
        "resources": {"nodes": 1, "ntasks": 24, "mem_gb": 128, "walltime": "72:00:00"},
        "environment": {"VASP_CMD": "mpirun -n $SLURM_NTASKS vasp_std"},
    }


def legacy_static_files() -> dict[str, bytes]:
    return {
        f"{FLOW_ROOT}/submission.json": json.dumps(legacy_static_submission_payload()).encode("utf-8"),
        f"{LOG_ROOT}/legacy-static.out": b"[runner] python: 3.12.13 (main)\n",
        f"{LOG_ROOT}/legacy-static.err": b"",
        f"{FLOW_ROOT}/CONTCAR": b"contcar",
        f"{FLOW_ROOT}/OUTCAR": b"outcar",
        f"{FLOW_ROOT}/vasprun.xml": b"vasprun",
        f"{FLOW_ROOT}/INCAR": b"ENCUT = 520\n",
    }


def slurm_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    assert command == [
        "ssh",
        "-o",
        "ConnectTimeout=10",
        "powerslurm-bmdguest",
        (
            "sacct -P -n -j 20893681 "
            f"{SACCT_FORMAT}"
        ),
    ]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is True
    assert kwargs["timeout"] == DEFAULT_SCHEDULER_ACCOUNTING_TIMEOUT_SECONDS
    return subprocess.CompletedProcess(
        command,
        0,
        stdout=(
            "20893681|validation|COMPLETED|02:48:37|2026-08-21T12:14:10|"
            "2026-08-21T15:02:47|leeburton-pool|0:0|72:00:00\n"
        ),
        stderr="",
    )


def fake_scientific_parser(
    local_paths: dict[str, Path],
    display_paths: dict[str, str],
    workflow_spec: dict,
) -> ScientificResult:
    assert set(local_paths) == {"contcar", "vasprun", "kpoints"}
    assert workflow_spec["stages"][3]["stage_type"] == "band_structure"
    return ScientificResult(
        source_paths=tuple(display_paths.values()),
        final_formula="Example2",
        final_energy_ev=-12.5,
        energy_per_atom_ev=-6.25,
        electronic_convergence=True,
        band_gap_ev=1.2,
        band_kpoints=310,
        bands=24,
    )


def fake_direct_scientific_parser(
    local_paths: dict[str, Path],
    display_paths: dict[str, str],
    workflow_spec: dict,
) -> ScientificResult:
    assert set(local_paths).issubset({"contcar", "vasprun"})
    assert "kpoints" not in local_paths
    assert workflow_spec["stages"][0]["stage_type"] == "direct_vasp"
    return ScientificResult(
        source_paths=tuple(display_paths.values()),
        final_formula="Example",
        final_energy_ev=-10.0,
        energy_per_atom_ev=-5.0,
        electronic_convergence=True,
        executed_parameters=(
            IncarObservation(
                "vasprun_xml.parameters",
                display_paths.get("vasprun", f"{DIRECT_DIR}/vasprun.xml"),
                True,
                1,
                source_type="vasprun_xml.parameters",
                values={"NELM": 60, "EDIFF": 1e-6, "NSW": 99, "EDIFFG": -0.01},
            ),
        ),
    )


def direct_vasp_files(
    *,
    include_vasprun: bool = True,
    oszicar: bytes = OSZICAR_TWO_STEP,
    outcar: bytes = OUTCAR_TWO_FORCE_BLOCKS,
) -> dict[str, bytes]:
    files = {
        f"{DIRECT_DIR}/INCAR": b"NELM = 60\nEDIFF = 1E-6\nNSW = 99\nEDIFFG = -0.01\n",
        f"{DIRECT_DIR}/POSCAR": POSCAR_TWO_SITE,
        f"{DIRECT_DIR}/KPOINTS": b"kpoints",
        f"{DIRECT_DIR}/OSZICAR": oszicar,
        f"{DIRECT_DIR}/OUTCAR": outcar,
        f"{DIRECT_DIR}/CONTCAR": b"contcar",
    }
    if include_vasprun:
        files[f"{DIRECT_DIR}/vasprun.xml"] = b"<modeling/>"
    return files


def job_sacct_output(
    *,
    job_id: str = "20893681",
    state: str = "COMPLETED",
    work_dir: str | None = DIRECT_DIR,
    max_rss: str | None = None,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
) -> str:
    return (
        f"{job_id}|direct-vasp|guest|power-leeburton-users_v2|{state}|0:0||06:00:20|"
        "21620|2026-08-30T00:00:00|2026-08-30T06:00:20|leeburton-pool|"
        "06:00:00|compute-0-269|1|24|24|128G|billing=24,cpu=24,mem=128G,node=1|"
        "billing=24,cpu=24,mem=128G,node=1|120:00:00|518880|"
        f"{max_rss or ''}|||{stdout_path or ''}|{stderr_path or ''}|{work_dir or ''}\n"
    )


def job_slurm_runner(
    *,
    job_id: str = "20893681",
    state: str = "COMPLETED",
    work_dir: str | None = DIRECT_DIR,
    max_rss: str | None = None,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
) -> Callable[[list[str]], subprocess.CompletedProcess[str]]:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == [
            "ssh",
            "-o",
            "ConnectTimeout=10",
            "powerslurm-bmdguest",
            f"sacct -P -n -j {job_id} {SACCT_FORMAT}",
        ]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is True
        assert kwargs["timeout"] == DEFAULT_SCHEDULER_ACCOUNTING_TIMEOUT_SECONDS
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=job_sacct_output(
                job_id=job_id,
                state=state,
                work_dir=work_dir,
                max_rss=max_rss,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            ),
            stderr="",
        )

    return runner


class MultiplexedInspectionRunner:
    """Model OpenSSH transport reuse while delegating fixture command behavior."""

    def __init__(
        self,
        remote: RemoteFixture,
        scheduler: Callable[..., subprocess.CompletedProcess[str]],
    ) -> None:
        self.remote = remote
        self.scheduler = scheduler
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        self.commands.append(command)
        if getattr(command, "ssh_control_operation", False):
            control_path = Path(command[command.index("-S") + 1])
            control_path.unlink(missing_ok=True)
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

        if getattr(command, "ssh_opens_connection", False):
            option = next(part for part in command if part.startswith("ControlPath="))
            Path(option.split("=", 1)[1]).touch()

        remote_command = command[-1]
        if remote_command.startswith("sacct "):
            normalized = [
                "ssh",
                "-o",
                "ConnectTimeout=10",
                "powerslurm-bmdguest",
                remote_command,
            ]
            return self.scheduler(normalized, **kwargs)
        normalized = ["ssh", "powerslurm-bmdguest", remote_command]
        return self.remote(normalized, **kwargs)


def test_inspect_slurm_job_rejects_invalid_job_id_without_scheduler_or_remote_reads() -> None:
    remote = RemoteFixture(files=direct_vasp_files(), directories={DIRECT_DIR})
    slurm_calls = 0

    def unused_slurm_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal slurm_calls
        slurm_calls += 1
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(ValueError, match="unsafe"):
        inspect_slurm_job(
            cluster(),
            "20893681;scancel 1",
            remote_runner=remote,
            slurm_runner=unused_slurm_runner,
            scientific_parser=fake_direct_scientific_parser,
        )

    assert slurm_calls == 0
    assert remote.commands == []


def test_inspect_slurm_job_scheduler_timeout_is_finite_and_oom_is_insufficient() -> None:
    remote_calls: list[list[str]] = []
    configured_cluster = replace(
        cluster(),
        ssh_connect_timeout_seconds=7,
        remote_command_timeout_seconds=33,
        scheduler_accounting_timeout_seconds=45,
    )

    def fail_remote(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        remote_calls.append(command)
        raise AssertionError("calculation artifacts must not be read without scheduler accounting")

    def slow_scheduler(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        assert command[:4] == ["ssh", "-o", "ConnectTimeout=7", "powerslurm-bmdguest"]
        assert "sacct -P -n -j 21906221" in command[4]
        assert kwargs["timeout"] == 45
        raise subprocess.TimeoutExpired(command, 45)

    inspection = inspect_slurm_job(
        configured_cluster,
        "21906221",
        remote_runner=fail_remote,
        slurm_runner=slow_scheduler,
    )

    assert inspection.scheduler is None
    assert inspection.scheduler_error is not None
    assert "timed out after 45 seconds" in inspection.scheduler_error
    assert inspection.oom is not None
    assert inspection.oom.assessment == INSUFFICIENT_OOM_EVIDENCE
    assert "scheduler accounting was unavailable" in inspection.oom.limitations
    assert remote_calls == []


def test_inspect_slurm_job_reports_missing_workdir_without_remote_reads() -> None:
    remote = RemoteFixture(files=direct_vasp_files(), directories={DIRECT_DIR})

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(work_dir=None),
        scientific_parser=fake_direct_scientific_parser,
    )

    assert inspection.scheduler is not None
    assert inspection.scheduler.work_dir is None
    assert inspection.scheduler_work_dir is None
    assert inspection.calculation_directory is None
    assert inspection.calculation_type == "unknown"
    assert inspection.calculation_reason == "scheduler WorkDir was unavailable"
    assert remote.commands == []


def test_inspect_slurm_job_rejects_workdir_outside_allowed_roots_without_remote_reads() -> None:
    remote = RemoteFixture(files=direct_vasp_files(), directories={DIRECT_DIR})

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(work_dir="/etc/scratch-job"),
        scientific_parser=fake_direct_scientific_parser,
    )

    assert inspection.scheduler_work_dir == "/etc/scratch-job"
    assert inspection.calculation_directory is None
    assert inspection.calculation_type == "unknown"
    assert inspection.calculation_reason is not None
    assert "scheduler WorkDir is not authorized" in inspection.calculation_reason
    assert remote.commands == []


def test_inspect_slurm_job_reports_unknown_directory_without_direct_vasp_markers() -> None:
    files = {
        f"{DIRECT_DIR}/INCAR": b"ENCUT = 520\n",
        f"{DIRECT_DIR}/POSCAR": b"poscar",
    }
    remote = RemoteFixture(files=files, directories={DIRECT_DIR})

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(),
        scientific_parser=fake_direct_scientific_parser,
    )

    assert inspection.calculation_directory is None
    assert inspection.calculation_type == "unknown"
    assert inspection.calculation_reason is not None
    assert "direct VASP marker set was incomplete" in inspection.calculation_reason
    assert "KPOINTS" in inspection.calculation_reason
    remote_commands = " ".join(" ".join(command) for command in remote.commands)
    assert "find " not in remote_commands
    assert "ls " not in remote_commands
    assert "POTCAR" not in remote_commands


def test_inspect_slurm_job_identifies_direct_vasp_without_submission_json() -> None:
    remote = RemoteFixture(files=direct_vasp_files(), directories={DIRECT_DIR})

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(),
        scientific_parser=fake_direct_scientific_parser,
    )

    assert inspection.calculation_directory == DIRECT_DIR
    assert inspection.calculation_type == "direct VASP"
    assert inspection.bmd_compute is None
    assert inspection.direct_vasp is not None
    direct = inspection.direct_vasp
    assert direct.producer_reason == "no BMD Compute producer record found"
    artifacts = {artifact.label: artifact for artifact in direct.artifacts}
    assert artifacts["incar"].present is True
    assert artifacts["poscar"].present is True
    assert artifacts["kpoints"].present is True
    assert direct.scientific.final_formula == "Example"
    assert {item.source_type for item in direct.executed_inputs} == {
        "retained_incar",
        "vasprun_xml.parameters",
    }
    assert direct.trajectory.stage_label == "work_dir"
    assert direct.trajectory.stage_index == 1
    assert direct.trajectory.completed_ionic_steps == 2
    assert direct.assessments
    remote_commands = " ".join(" ".join(command) for command in remote.commands)
    assert "find " not in remote_commands
    assert "ls " not in remote_commands
    assert "POTCAR" not in remote_commands


def test_missing_bmd_state_preserves_authorized_manual_vasp_workdir_fallback() -> None:
    remote = RemoteFixture(files=direct_vasp_files(), directories={DIRECT_DIR})

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(),
        scientific_parser=fake_direct_scientific_parser,
        deployment=power_deployment(),
    )

    assert inspection.run_resolution is not None
    assert inspection.run_resolution.resolution_status == NOT_BMD_COMPUTE
    assert inspection.scheduler is not None
    assert inspection.calculation_type == "direct VASP"
    assert inspection.direct_vasp is not None
    commands = " ".join(command[2] for command in remote.commands)
    assert f"test -f {LOG_ROOT}/job_20893681.json" in commands
    assert "find " not in commands
    assert "ls " not in commands


def test_acquisition_batch_preserves_manual_vasp_fallback_evidence() -> None:
    files = direct_vasp_files()
    baseline_remote = RemoteFixture(files=files, directories={DIRECT_DIR})
    baseline = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=baseline_remote,
        slurm_runner=job_slurm_runner(),
        scientific_parser=fake_direct_scientific_parser,
        deployment=power_deployment(),
    )

    optimized_remote = RemoteFixture(files=files, directories={DIRECT_DIR})
    with remote_acquisition_cache(
        "powerslurm-bmdguest",
        (PurePosixPath("/bmd-db/guest"),),
    ):
        optimized = inspect_slurm_job(
            cluster(),
            "20893681",
            remote_runner=optimized_remote,
            slurm_runner=job_slurm_runner(),
            scientific_parser=fake_direct_scientific_parser,
            deployment=power_deployment(),
        )

    assert optimized == baseline
    assert len(optimized_remote.commands) < len(baseline_remote.commands)
    optimized_commands = [command[2] for command in optimized_remote.commands]
    assert any("bmd-agent-acquisition-v1" in command for command in optimized_commands)
    assert not any(
        command.startswith(("test -f ", "test -d ", "stat -c "))
        for command in optimized_commands
    )
    assert not any("POTCAR" in command for command in optimized_commands)


def test_missing_bmd_state_keeps_scheduler_evidence_when_workdir_is_not_authorized() -> None:
    remote = RemoteFixture(files={}, directories=set())

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(work_dir="/a/home/cc/tree/taucc/enginer/bmdguest"),
        deployment=power_deployment(),
    )

    assert inspection.run_resolution is not None
    assert inspection.run_resolution.resolution_status == NOT_BMD_COMPUTE
    assert inspection.scheduler is not None
    assert inspection.scheduler.state == "COMPLETED"
    assert inspection.calculation_type == "unknown"
    assert "not authorized" in (inspection.calculation_reason or "")


def test_inspect_slurm_job_reads_bounded_scheduler_stderr_for_explicit_oom() -> None:
    stderr_path = f"{LOG_ROOT}/slurm-20893681.err"
    files = direct_vasp_files()
    files[stderr_path] = b"slurmstepd: error: Detected 1 oom-kill event(s) in step\n"
    remote = RemoteFixture(files=files, directories={DIRECT_DIR})

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(state="FAILED", stderr_path=stderr_path),
        scientific_parser=fake_direct_scientific_parser,
    )

    assert inspection.oom is not None
    assert inspection.oom.assessment == OOM_ESTABLISHED
    assert any(marker.source.startswith("SLURM stderr") for marker in inspection.oom.explicit_evidence)
    assert [command[2] for command in remote.commands if command[2].startswith("tail ")] == [
        f"tail -c 128000 -- {stderr_path}"
    ]
    assert not any(
        token in command[2]
        for command in remote.commands
        for token in ("sbatch", "scancel", "scontrol")
    )


def test_inspect_remote_run_reuses_producer_logs_for_oom_evidence() -> None:
    files = default_files()
    files[f"{LOG_ROOT}/validation-run.slurm.err"] = (
        b"slurmstepd: error: Detected 1 oom-kill event(s) in step\n"
    )
    remote = RemoteFixture(files=files, directories=default_directories())

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=slurm_runner,
        scientific_parser=fake_scientific_parser,
    )

    assert inspection.oom is not None
    assert inspection.oom.assessment == OOM_ESTABLISHED
    assert any(
        marker.source == f"{LOG_ROOT}/validation-run.slurm.err"
        for marker in inspection.oom.explicit_evidence
    )


def test_inspect_slurm_job_can_identify_direct_vasp_with_invalid_submission_json() -> None:
    files = direct_vasp_files()
    files[f"{DIRECT_DIR}/submission.json"] = b"{not-json"
    remote = RemoteFixture(files=files, directories={DIRECT_DIR})

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(),
        scientific_parser=fake_direct_scientific_parser,
    )

    assert inspection.calculation_directory == DIRECT_DIR
    assert inspection.calculation_type == "direct VASP"
    assert inspection.direct_vasp is not None
    assert "no valid BMD Compute producer record" in inspection.direct_vasp.producer_reason


def test_inspect_slurm_job_delegates_bmd_compute_workdir_to_existing_diagnosis() -> None:
    remote = RemoteFixture(files=default_files(), directories=default_directories())

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(work_dir=FLOW_ROOT),
        scientific_parser=fake_scientific_parser,
        max_vasprun_bytes=0,
    )

    assert inspection.calculation_directory == FLOW_ROOT
    assert inspection.calculation_type == "BMD Compute"
    assert inspection.bmd_compute is not None
    assert inspection.direct_vasp is None
    assert inspection.bmd_compute.inspection.workflow_stages[0].stage_type == "relax"
    # The WorkDir fallback derives final scientific results like exact resolution.
    scientific = inspection.bmd_compute.inspection.scientific
    assert scientific.final_formula == "Example2"
    assert scientific.final_energy_ev == -12.5
    assert "skipped" not in " ".join(scientific.unavailable)


def test_authoritative_job_record_resolves_generic_workdir_into_common_bmd_analysis() -> None:
    files = default_files()
    spec = submission_payload()
    spec["run_name"] = "validation-run"
    spec["submission"]["attempt_id"] = "attempt-validation"
    files[f"{FLOW_ROOT}/submission.json"] = json.dumps(spec).encode()
    files[f"{LOG_ROOT}/submission_attempts/attempt.json"] = json.dumps(
        {
            "attempt_id": "attempt-validation",
            "state": "SUBMITTED",
            "job_id": "20893681",
            "run_dir": FLOW_ROOT,
            "job_record": {
                "job_id": "20893681",
                "run_dir": FLOW_ROOT,
                "submission_spec": spec,
            },
        }
    ).encode()
    state_path = f"{LOG_ROOT}/job_20893681.json"
    files[state_path] = json.dumps(
        {
            "job_id": "20893681",
            "run_name": "validation-run",
            "run_dir": FLOW_ROOT,
            "remote_script": f"{FLOW_ROOT}.sbatch.sh",
            "log_paths": {},
            "cluster": {"partition": "leeburton-pool"},
            "resources": {"nodes": 1, "ntasks": 24},
            "submitted_at": "2026-08-21T12:00:00+03:00",
            "status": "submitted",
            "submission_spec": spec,
            "remote_state_path": state_path,
        }
    ).encode()
    remote = RemoteFixture(files=files, directories=default_directories())
    scheduler_calls = 0
    scheduler_runner = job_slurm_runner(work_dir="/a/home/cc/tree/taucc/enginer/bmdguest")

    def counting_scheduler(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal scheduler_calls
        scheduler_calls += 1
        return scheduler_runner(command, **kwargs)

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=counting_scheduler,
        scientific_parser=fake_scientific_parser,
        max_vasprun_bytes=0,
        deployment=power_deployment(),
    )

    assert scheduler_calls == 1
    assert inspection.run_resolution is not None
    assert inspection.run_resolution.resolution_status == RESOLVED
    assert inspection.calculation_directory == FLOW_ROOT
    assert inspection.scheduler_work_dir == "/a/home/cc/tree/taucc/enginer/bmdguest"
    assert inspection.calculation_type == "BMD Compute"
    assert inspection.bmd_compute is not None
    assert inspection.bmd_compute.inspection.scheduler is inspection.scheduler
    assert inspection.bmd_compute.inspection.scientific.final_formula == "Example2"
    commands = " ".join(command[2] for command in remote.commands)
    assert "find " not in commands
    assert "ls " not in commands
    assert "POTCAR" not in commands
    assert "/a/home/cc/tree/taucc/enginer/bmdguest" not in commands
    assert [command[2] for command in remote.commands].count(
        f"cat -- {FLOW_ROOT}/submission.json"
    ) == 1
    assert [command[2] for command in remote.commands].count(
        f"cat -- {LOG_ROOT}/submission_attempts/attempt.json"
    ) == 1


def test_resolved_running_bmd_job_keeps_partial_trajectory_nonfatal() -> None:
    spec = single_stage_submission_payload()
    spec["run_name"] = "running-run"
    attempt_path = f"{LOG_ROOT}/submission_attempts/running-attempt.json"
    spec["submission"] = {
        "attempt_id": "running-attempt",
        "attempt_state": attempt_path,
    }
    spec["paths"]["submission_attempt_state"] = attempt_path
    files = {
        f"{FLOW_ROOT}/submission.json": json.dumps(spec).encode(),
        attempt_path: json.dumps(
            {
                "attempt_id": "running-attempt",
                "state": "SUBMITTED",
                "job_id": "20893681",
                "run_dir": FLOW_ROOT,
            }
        ).encode(),
        f"{LOG_ROOT}/job_20893681.json": json.dumps(
            {
                "job_id": "20893681",
                "run_name": "running-run",
                "run_dir": FLOW_ROOT,
                "status": "submitted",
                "submission_spec": spec,
                "remote_state_path": f"{LOG_ROOT}/job_20893681.json",
            }
        ).encode(),
        f"{FLOW_ROOT}/INCAR": b"NELM = 200\nNSW = 99\nEDIFFG = -0.01\n",
        f"{FLOW_ROOT}/POSCAR": POSCAR_TWO_SITE,
        f"{FLOW_ROOT}/KPOINTS": b"kpoints",
        f"{FLOW_ROOT}/OSZICAR": oszicar_incomplete_cycle(4),
    }
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(
            state="RUNNING",
            work_dir="/a/home/cc/tree/taucc/enginer/bmdguest",
        ),
        max_vasprun_bytes=0,
        deployment=power_deployment(),
    )

    assert inspection.bmd_compute is not None
    assert inspection.scheduler is not None
    assert inspection.scheduler.state == "RUNNING"
    trajectory = inspection.bmd_compute.trajectories[0]
    assert trajectory.completed_ionic_steps == 0
    assert trajectory.incomplete_electronic_iteration_count == 4
    assert inspection.bmd_compute.inspection.scientific.error is None


def test_historical_failed_bmd_fixture_resolves_trajectory_and_custodian_evidence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "job_21906221.json"
    state_bytes = fixture_path.read_bytes()
    state = json.loads(state_bytes)
    spec = state["submission_spec"]
    run_dir = state["run_dir"]
    attempt_path = spec["submission"]["attempt_state"]
    custodian_bytes = (
        Path(__file__).parent / "fixtures" / "custodian_frozen_repeated.json"
    ).read_bytes()
    files = {
        state["remote_state_path"]: state_bytes,
        f"{run_dir}/submission.json": json.dumps(spec).encode(),
        attempt_path: json.dumps(
            {
                "attempt_id": spec["submission"]["attempt_id"],
                "state": "SUBMITTED",
                "job_id": "21906221",
                "run_dir": run_dir,
                "job_record": {
                    "job_id": "21906221",
                    "run_dir": run_dir,
                    "submission_spec": spec,
                },
            }
        ).encode(),
        f"{run_dir}/INCAR": (
            b"LHFCALC = True\nHFSCREEN = 0.2\nLSORBIT = True\n"
            b"NELM = 200\nALGO = Normal\n"
        ),
        f"{run_dir}/OSZICAR": oszicar_incomplete_cycle(4),
        f"{run_dir}/OUTCAR": b"",
        f"{run_dir}/CONTCAR": b"contcar",
        f"{run_dir}/custodian.json": custodian_bytes,
        f"{run_dir}/std_err.txt": (
            b"forrtl: error (78): process killed (SIGTERM)\n"
        ),
        **{
            f"{run_dir}/error.{index}.tar.gz": b"archive contents are not inspected"
            for index in range(1, 6)
        },
        spec["paths"]["log_out"]: b"",
        spec["paths"]["log_err"]: b"",
        spec["paths"]["slurm_out"]: b"",
        spec["paths"]["slurm_err"]: b"",
    }
    unprofiled_remote = RemoteFixture(files=files, directories={run_dir})
    unprofiled_inspection = inspect_slurm_job(
        cluster(),
        "21906221",
        remote_runner=unprofiled_remote,
        slurm_runner=job_slurm_runner(
            job_id="21906221",
            state="FAILED",
            work_dir="/a/home/cc/tree/taucc/enginer/bmdguest",
            max_rss="45045764K",
        ),
        scientific_parser=lambda local, display, workflow: ScientificResult(
            source_paths=tuple(display.values()),
            final_formula="fixture",
        ),
        max_vasprun_bytes=0,
        deployment=power_deployment(),
    )

    profiled_remote = RemoteFixture(files=files, directories={run_dir})
    profiler = PerformanceProfiler()
    transport = MultiplexedInspectionRunner(
        profiled_remote,
        job_slurm_runner(
            job_id="21906221",
            state="FAILED",
            work_dir="/a/home/cc/tree/taucc/enginer/bmdguest",
            max_rss="45045764K",
        ),
    )
    with profiler.activate():
        with ReusableSshSession(
            "powerslurm-bmdguest",
            runner=transport,
            multiplex=True,
        ) as session:
            with remote_acquisition_cache(
                "powerslurm-bmdguest",
                (PurePosixPath("/bmd-db/guest"),),
            ):
                inspection = inspect_slurm_job(
                    cluster(),
                    "21906221",
                    remote_runner=session.runner("remote"),
                    slurm_runner=session.runner("scheduler"),
                    scientific_parser=lambda local, display, workflow: ScientificResult(
                        source_paths=tuple(display.values()),
                        final_formula="fixture",
                    ),
                    max_vasprun_bytes=0,
                    deployment=power_deployment(),
                )
    profile = profiler.snapshot()

    assert inspection == unprofiled_inspection
    assert len(profiled_remote.commands) < len(unprofiled_remote.commands)
    assert profile.operations.counts["scheduler_operations"] == 1
    assert profile.operations.counts["ssh_connections"] == 1
    assert profile.operations.counts["ssh_exec_channels"] == len(profiled_remote.commands) + 1
    assert profile.operations.counts["ssh_control_operations"] == 1
    assert profile.operations.counts["ssh_invocations"] == len(profiled_remote.commands) + 2
    assert profile.operations.counts["archive_probes"] == 1
    assert profile.operations.counts["archive_probe_batches"] == 1
    assert profile.operations.counts["metadata_manifest_operations"] >= 3
    assert profile.operations.counts["batched_file_read_operations"] >= 3
    assert profile.operations.counts["logical_files_described"] >= 15
    assert profile.operations.counts["logical_files_read"] > 0
    assert profile.operations.counts["existence_probes"] == 0
    assert profile.operations.counts["stat_probes"] == 0
    assert profile.operations.counts["directory_probes"] == 0
    assert profile.operations.bytes_transferred > 0
    assert profile.operations.failure_count == 0
    phase_names = {phase.name for phase in profile.phases}
    assert "scheduler_acquisition" in phase_names
    assert "job_run_resolution" in phase_names
    assert "producer_submission_provenance" in phase_names
    assert "calculation_workflow_acquisition" in phase_names
    assert "vasp_scientific_evidence" in phase_names
    assert "vasp_trajectory_evidence" in phase_names
    assert "local_vasp_parsing" in phase_names
    assert "diagnostic_custodian_evidence" in phase_names
    assert "oom_resource_analysis" in phase_names

    assert inspection.run_resolution is not None
    assert inspection.run_resolution.resolution_status == RESOLVED
    assert inspection.scheduler is not None
    assert inspection.scheduler.state == "FAILED"
    assert inspection.bmd_compute is not None
    run = inspection.bmd_compute.inspection
    assert run.workflow_stages[0].theory == "hse06"
    assert run.workflow_stages[0].modifiers == ("soc",)
    assert inspection.bmd_compute.trajectories[0].incomplete_electronic_iteration_count == 4
    assert len(run.custodian_evidence) == 1
    assert len(run.custodian_evidence[0].corrections) == 5
    assert run.custodian_evidence[0].repeated_interventions[0].count == 5
    assert run.custodian_evidence[0].repeated_interventions[0].timeout_seconds == 21600
    assert run.custodian_policy.available is False
    assert run.execution_diagnostics.logs[0].path == f"{run_dir}/std_err.txt"
    assert "SIGTERM" in run.execution_diagnostics.logs[0].messages[0]
    assert len(run.execution_diagnostics.error_archives) == 5
    assert inspection.bmd_compute.termination.assessment is not None
    assert (
        inspection.bmd_compute.termination.assessment.classification
        == "custodian_triggered_process_termination"
    )
    assert inspection.bmd_compute.termination.assessment.status == "supported"
    assert inspection.oom is not None
    assert inspection.oom.assessment == NO_OOM_EVIDENCE
    contextual_query = build_bmdex_domain_query_for_job(inspection)
    assert contextual_query is not None
    assert contextual_query["calculation_family"] == "hybrid_functional"
    assert contextual_query["functional"] == "hse06"
    assert "incomplete_first_electronic_cycle" in contextual_query["observed_patterns"]
    remote_commands = [command[2] for command in profiled_remote.commands]
    assert len(profiled_remote.commands) == 5
    assert any(
        "bmd-agent-acquisition-v1" in command
        and f"{run_dir}/std_err.txt" in command
        for command in remote_commands
    )
    assert f"tail -c 128000 -- {run_dir}/std_err.txt" not in remote_commands
    assert not any(
        shlex.split(command)[0] in {"tar", "find", "ls"}
        for command in remote_commands
    )
    assert not any("error.*.tar.gz" in command or "POTCAR" in command for command in remote_commands)

    cli.print_job_inspection(inspection)
    output = capsys.readouterr().out
    assert "classification: custodian_triggered_process_termination" in output
    assert "status: supported" in output
    assert "submission has no persisted Custodian execution-policy provenance" in output
    assert "assessment: NO OOM EVIDENCE FOUND" in output


def test_remote_execution_diagnostics_use_bounded_exact_read_only_paths() -> None:
    directory = PurePosixPath(FLOW_ROOT)
    files = {
        f"{FLOW_ROOT}/std_err.txt": b"SIGTERM received by VASP\n",
        f"{FLOW_ROOT}/vasp.out": b"fatal: bounded diagnostic fixture\n",
        f"{FLOW_ROOT}/OUTCAR": b"forrtl: process killed\n",
        f"{FLOW_ROOT}/error.1.tar.gz": b"not read",
        f"{FLOW_ROOT}/error.2.tar.gz": b"not read",
    }
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    evidence = run_resource._observe_remote_execution_diagnostics(
        "powerslurm-bmdguest",
        directory,
        {},
        directory,
        (WorkflowStage(1, "static", "hse06", (), None),),
        allowed_roots=(PurePosixPath("/bmd-db/guest"),),
        runner=remote,
        timeout=20,
    )

    assert [item.label for item in evidence.logs] == ["std_err.txt", "vasp.out", "OUTCAR"]
    assert evidence.logs[0].messages == ("SIGTERM received by VASP",)
    assert [item.label for item in evidence.error_archives] == [
        "error.1.tar.gz",
        "error.2.tar.gz",
    ]
    commands = [command[2] for command in remote.commands]
    assert f"tail -c 128000 -- {FLOW_ROOT}/std_err.txt" in commands
    archive_commands = [
        command
        for command in commands
        if "bmd-agent-archive-probe-v1" in command
    ]
    assert len(archive_commands) == 1
    assert shlex.split(archive_commands[0])[4:] == [FLOW_ROOT, "64"]
    assert not any(command.startswith("cat --") for command in commands)
    assert not any(shlex.split(command)[0] in {"tar", "find", "ls"} for command in commands)
    assert not any("error.*.tar.gz" in command or "POTCAR" in command for command in commands)


def test_missing_remote_execution_diagnostics_are_nonfatal() -> None:
    directory = PurePosixPath(FLOW_ROOT)
    remote = RemoteFixture(files={}, directories={FLOW_ROOT})

    evidence = run_resource._observe_remote_execution_diagnostics(
        "powerslurm-bmdguest",
        directory,
        {},
        directory,
        (WorkflowStage(1, "static", "pbe", (), None),),
        allowed_roots=(PurePosixPath("/bmd-db/guest"),),
        runner=remote,
        timeout=20,
    )

    assert evidence.logs == ()
    assert evidence.error_archives == ()


def test_completed_remote_bmd_run_remains_free_of_custodian_failure_sections(
    capsys: pytest.CaptureFixture[str],
) -> None:
    remote = RemoteFixture(files=default_files(), directories=default_directories())

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=slurm_runner,
        max_vasprun_bytes=0,
    )

    assert diagnosis.inspection.scheduler is not None
    assert diagnosis.inspection.scheduler.state == "COMPLETED"
    assert diagnosis.inspection.custodian_evidence == ()
    assert diagnosis.inspection.execution_diagnostics.error_archives == ()

    cli.print_run_diagnosis(diagnosis)
    output = capsys.readouterr().out
    assert "Custodian intervention evidence" not in output
    assert "error archives" not in output


def test_direct_vasp_truncated_vasprun_keeps_oszicar_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def malformed_loader(*args: object, **kwargs: object) -> object:
        warnings.warn(
            "XML is malformed. Parsing has stopped but partial data is available.",
            UserWarning,
        )
        raise IndexError("list index out of range")

    monkeypatch.setattr(run_resource, "_load_vasprun", malformed_loader)
    remote = RemoteFixture(
        files=direct_vasp_files(oszicar=oszicar_incomplete_cycle(25)),
        directories={DIRECT_DIR},
    )

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(state="TIMEOUT"),
        scientific_parser=fake_direct_scientific_parser,
    )

    assert inspection.direct_vasp is not None
    trajectory = inspection.direct_vasp.trajectory
    assert trajectory.oszicar_present is True
    assert trajectory.completed_ionic_steps == 0
    assert trajectory.electronic_iterations_by_completed_ionic_step == ()
    assert trajectory.incomplete_electronic_iteration_count == 25
    assert trajectory.vasprun_present is True
    assert trajectory.vasprun_error == "file could not be parsed completely"
    assert "list index out of range" not in " ".join(trajectory.unavailable)
    stage = assessment_by_scope(inspection.direct_vasp.assessments, "stage")
    assert stage.label == INSUFFICIENT_EVIDENCE


def test_direct_vasp_scientific_parsing_captures_malformed_vasprun_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    remote = RemoteFixture(files=direct_vasp_files(), directories={DIRECT_DIR})

    with fake_pymatgen_modules(FakeMalformedVasprun), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inspection = inspect_slurm_job(
            cluster(),
            "20893681",
            remote_runner=remote,
            slurm_runner=job_slurm_runner(state="TIMEOUT"),
            scientific_parser=parse_vasp_output_files,
        )

    assert caught == []
    assert inspection.direct_vasp is not None
    scientific = inspection.direct_vasp.scientific
    assert "vasprun.xml could not be parsed completely" in scientific.unavailable
    assert "list index out of range" not in " ".join(scientific.unavailable)
    assert "XML is malformed" not in " ".join(scientific.unavailable)
    assert scientific.error is None

    cli.print_job_inspection(inspection)

    captured = capsys.readouterr()
    assert "vasprun.xml could not be parsed completely" in captured.out
    assert "file could not be parsed completely" in captured.out
    assert "list index out of range" not in captured.out
    assert "XML is malformed" not in captured.out


def test_bmd_run_scientific_parsing_captures_malformed_vasprun_warning() -> None:
    remote = RemoteFixture(files=single_stage_files(), directories={FLOW_ROOT})

    with fake_pymatgen_modules(FakeMalformedVasprun), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inspection = inspect_remote_run(
            cluster(),
            FLOW_ROOT,
            remote_runner=remote,
            slurm_runner=slurm_runner,
            scientific_parser=parse_vasp_output_files,
        )

    assert caught == []
    assert inspection.scientific.error is None
    assert "vasprun.xml could not be parsed completely" in inspection.scientific.unavailable
    assert "list index out of range" not in " ".join(inspection.scientific.unavailable)


def test_completed_direct_vasp_relaxation_reports_converged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTrajectoryVasprun:
        parameters = {"NELM": 60, "EDIFF": 1e-06, "NSW": 2, "EDIFFG": -0.01}
        converged_electronic = True
        converged_ionic = True
        ionic_steps = (
            {"forces": ((0.0, 0.0, 0.008),)},
            {"forces": ((0.0, 0.0, 0.004),)},
        )

    monkeypatch.setattr(run_resource, "_load_vasprun", lambda *args, **kwargs: FakeTrajectoryVasprun())
    remote = RemoteFixture(files=direct_vasp_files(), directories={DIRECT_DIR})

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(),
        scientific_parser=fake_direct_scientific_parser,
    )

    assert inspection.direct_vasp is not None
    assessments = inspection.direct_vasp.assessments
    assert assessment_by_scope(assessments, "electronic").label == CONVERGED
    assert assessment_by_scope(assessments, "ionic").label == CONVERGED
    assert assessment_by_scope(assessments, "stage").label == CONVERGED


def test_cli_prints_direct_vasp_job_summary_without_prediction(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trajectory = StageTrajectoryObservation(
        stage_index=1,
        stage_label="work_dir",
        stage_type="direct_vasp",
        theory="unknown",
        directory=DIRECT_DIR,
        oszicar_path=f"{DIRECT_DIR}/OSZICAR",
        oszicar_present=True,
        vasprun_path=f"{DIRECT_DIR}/vasprun.xml",
        vasprun_present=True,
        criteria={"NELM": 60, "NSW": 1, "EDIFFG": -0.01},
        completed_ionic_steps=1,
        electronic_iterations_by_completed_ionic_step=(8,),
        final_electronic_iteration_count=8,
        recent_electronic_iterations=(
            ElectronicIterationObservation(8, "DAV", -10.0, -1e-5, -1e-6),
        ),
        recent_ionic_steps=(IonicStepObservation(1, 8, -10.0, -9.99, -0.1, 0.005),),
        converged_electronic=True,
        converged_ionic=True,
    )
    assessments = assess_convergence_progress((trajectory,))
    direct = run_resource.DirectVaspInspection(
        directory=DIRECT_DIR,
        artifacts=(
            PathObservation("incar", f"{DIRECT_DIR}/INCAR", "file", True, ARTIFACT_OBSERVATION),
            PathObservation("poscar", f"{DIRECT_DIR}/POSCAR", "file", True, ARTIFACT_OBSERVATION),
            PathObservation("kpoints", f"{DIRECT_DIR}/KPOINTS", "file", True, ARTIFACT_OBSERVATION),
            PathObservation("oszicar", f"{DIRECT_DIR}/OSZICAR", "file", True, ARTIFACT_OBSERVATION),
        ),
        executed_inputs=(
            IncarObservation(
                "work_dir",
                f"{DIRECT_DIR}/INCAR",
                True,
                1,
                values={"ENCUT": 520, "NSW": 1},
            ),
        ),
        scientific=ScientificResult(
            source_paths=(f"{DIRECT_DIR}/CONTCAR",),
            final_formula="Example",
            electronic_convergence=True,
        ),
        trajectory=trajectory,
        assessments=assessments,
    )
    inspection = JobInspection(
        job_id="20893681",
        scheduler=SlurmAccountingRecord(
            job_id="20893681",
            name="direct-vasp",
            state="COMPLETED",
            elapsed="00:10:00",
            start="2026-08-30T00:00:00",
            end="2026-08-30T00:10:00",
            partition="leeburton-pool",
            exit_code="0:0",
            allocated_cpus=24,
            work_dir=DIRECT_DIR,
        ),
        scheduler_error=None,
        scheduler_work_dir=DIRECT_DIR,
        calculation_directory=DIRECT_DIR,
        calculation_type="direct VASP",
        calculation_reason=None,
        direct_vasp=direct,
    )

    cli.print_job_inspection(inspection)

    captured = capsys.readouterr()
    assert "BMD Job Inspection" not in captured.out
    assert "Job (scheduler_observation)" in captured.out
    assert "scheduler WorkDir: /bmd-db/guest/flows/direct-vasp" in captured.out
    assert "type: direct VASP" in captured.out
    assert "Producer provenance (producer_provenance):" in captured.out
    assert "unavailable" in captured.out
    assert "Executed VASP inputs (executed_input)" in captured.out
    assert "ENCUT=520" in captured.out
    assert "Trajectory evidence (trajectory_observation)" in captured.out
    assert "stage 1: Direct VASP calculation (work_dir)" in captured.out
    assert "UNKNOWN Direct Vasp" not in captured.out
    assert "Convergence-progress assessment (convergence_progress_assessment)" in captured.out
    assert "more walltime" not in captured.out.lower()


def test_inspect_run_uses_submission_stage_dirs_and_preserves_sources() -> None:
    remote = RemoteFixture(files=default_files(), directories=default_directories())

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=slurm_runner,
        scientific_parser=fake_scientific_parser,
        modifier_policies=(modifier_policy(),),
    )

    assert [stage.stage_type for stage in inspection.workflow_stages] == [
        "relax",
        "static",
        "dos",
        "band_structure",
    ]
    assert inspection.workflow_stages[1].options == {"custom_modifier": {"nested": "kept"}}
    assert inspection.workflow_stages[3].options == {"dispersion": {"method": "dftd3"}}
    assert inspection.initial_structure.status == "available"
    assert inspection.initial_structure.representation_hash is not None
    assert [item.label for item in inspection.stage_directories] == [
        "producer-alpha",
        "producer-beta",
        "producer-gamma",
        "producer-delta",
    ]
    assert all("stage_01" not in " ".join(command) for command in remote.commands)
    assert inspection.producer_git["git_commit"] == "abcdef0123456789"
    assert inspection.scheduler is not None
    assert inspection.scheduler.state == "COMPLETED"
    assert inspection.runtime.evidence_type == LOG_OBSERVATION
    assert inspection.runtime.packages["pymatgen"] == "2026.8.13"
    assert inspection.runtime.environment["PMG_VASP_PSP_DIR"] == "/bmd-db/potcars"
    assert inspection.runtime.stage_uuids == {
        "stage_01": "64210872-5626-40c7-a7eb-79f7e49272ba",
        "custom.second": "328290de-b493-4d39-a4c7-238eb9055720",
        "final stage": "48f52369-d3b9-40b5-9a9f-a87bae6d007f",
    }
    assert inspection.scientific.evidence_type == PYMATGEN_DERIVED
    assert inspection.scientific.final_formula == "Example2"
    assert all(item.evidence_type == ARTIFACT_OBSERVATION for item in inspection.final_artifacts)
    incar_inputs = {item.label: item for item in inspection.executed_inputs}
    assert incar_inputs["producer-alpha"].evidence_type == EXECUTED_INPUT
    assert incar_inputs["producer-alpha"].stage_index == 1
    assert incar_inputs["producer-alpha"].values["IVDW"] == 11
    assert incar_inputs["producer-delta"].stage_index == 4
    assert incar_inputs["producer-beta"].present is False
    assert "result_dir" not in incar_inputs
    assert all(item.stage_index is not None for item in inspection.executed_inputs)
    assert inspection.input_expectations[0].status == "supported"


@pytest.mark.parametrize("custom", [False, True])
def test_single_stage_runs_bind_result_dir_to_stage_one(custom: bool) -> None:
    remote = RemoteFixture(files=single_stage_files(custom=custom), directories={FLOW_ROOT})

    def parser(
        local_paths: dict[str, Path],
        display_paths: dict[str, str],
        workflow_spec: dict,
    ) -> ScientificResult:
        assert set(local_paths) == {"contcar", "vasprun"}
        return ScientificResult(
            source_paths=tuple(display_paths.values()),
            final_formula="Example",
        )

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
        scientific_parser=parser,
        modifier_policies=(modifier_policy(),),
    )

    assert inspection.workflow_stages[0].options == {"dispersion": {"method": "dftd3"}}
    assert inspection.executed_inputs == (
        IncarObservation(
            "result_dir",
            f"{FLOW_ROOT}/INCAR",
            True,
            1,
            values={"IVDW": 11},
        ),
    )
    assert inspection.input_expectations[0].status == "supported"
    assert inspection.input_expectations[0].observed_value == 11


def test_legacy_submission_without_provenance_remains_inspectable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    remote = RemoteFixture(files=legacy_static_files(), directories={FLOW_ROOT})

    def parser(
        local_paths: dict[str, Path],
        display_paths: dict[str, str],
        workflow_spec: dict,
    ) -> ScientificResult:
        assert set(local_paths) == {"contcar", "vasprun"}
        assert workflow_spec["stages"][0]["stage_type"] == "static"
        return ScientificResult(
            source_paths=tuple(display_paths.values()),
            final_formula="MgO",
            executed_parameters=(
                IncarObservation(
                    "vasprun_xml.parameters",
                    f"{FLOW_ROOT}/vasprun.xml",
                    True,
                    1,
                    source_type="vasprun_xml.parameters",
                    values={"ENCUT": 520},
                ),
            ),
        )

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
        scientific_parser=parser,
    )

    assert inspection.producer_git == {}
    assert inspection.workflow_stages == (
        WorkflowStage(
            1,
            "static",
            "pbe",
            (),
            None,
            {},
        ),
    )
    assert inspection.result_directory.present is True
    assert inspection.executed_inputs[0] == IncarObservation(
        "result_dir",
        f"{FLOW_ROOT}/INCAR",
        True,
        1,
        values={"ENCUT": 520},
    )
    assert inspection.executed_inputs[1].source_type == "vasprun_xml.parameters"
    assert inspection.executed_inputs[1].values["ENCUT"] == 520
    assert inspection.input_expectations == ()

    cli.print_run_inspection(inspection)
    captured = capsys.readouterr()
    assert "git commit:  unavailable" in captured.out
    assert "git state:   unavailable" in captured.out


def test_parse_oszicar_trajectory_preserves_electronic_and_ionic_evidence() -> None:
    trajectory = parse_oszicar_trajectory(OSZICAR_TWO_STEP)

    assert trajectory.ionic_steps_observed == 2
    assert trajectory.completed_ionic_steps == 2
    assert trajectory.electronic_iterations_by_ionic_step == (2, 2)
    assert trajectory.electronic_iterations_by_completed_ionic_step == (2, 2)
    assert [cycle.iterations for cycle in trajectory.electronic_cycles] == [2, 2]
    assert all(cycle.completed_ionic_step for cycle in trajectory.electronic_cycles)
    assert trajectory.incomplete_electronic_iteration_count is None
    assert trajectory.final_electronic_iteration_count == 2
    assert [step.step_index for step in trajectory.ionic_steps] == [1, 2]
    assert trajectory.recent_electronic_iterations[-1].algorithm == "RMM"
    assert trajectory.recent_electronic_iterations[-1].iteration == 2
    assert trajectory.recent_electronic_iterations[-1].energy == -12.1
    assert trajectory.recent_ionic_steps[-1].step_index == 2
    assert trajectory.recent_ionic_steps[-1].free_energy == -12.1


def test_parse_oszicar_distinguishes_incomplete_first_electronic_cycle() -> None:
    trajectory = parse_oszicar_trajectory(oszicar_incomplete_cycle(25))

    assert trajectory.ionic_steps_observed == 0
    assert trajectory.completed_ionic_steps == 0
    assert trajectory.electronic_iterations_by_ionic_step == (25,)
    assert trajectory.electronic_iterations_by_completed_ionic_step == ()
    assert trajectory.incomplete_electronic_iteration_count == 25
    assert trajectory.final_electronic_iteration_count == 25
    assert trajectory.recent_incomplete_electronic_iterations[-1].iteration == 25
    assert trajectory.recent_ionic_steps == ()


def test_parse_outcar_force_blocks_preserves_complete_atomic_force_evidence() -> None:
    blocks = parse_outcar_force_blocks(
        OUTCAR_TWO_FORCE_BLOCKS,
        source_path=f"{FLOW_ROOT}/OUTCAR",
        expected_site_count=2,
    )

    assert blocks == (
        OutcarForceBlockObservation(
            1,
            2,
            "complete",
            True,
            f"{FLOW_ROOT}/OUTCAR",
            5.0,
        ),
        OutcarForceBlockObservation(
            2,
            2,
            "complete",
            True,
            f"{FLOW_ROOT}/OUTCAR",
            0.5,
        ),
    )
    assert all(block.evidence_type == TRAJECTORY_OBSERVATION for block in blocks)


def test_parse_outcar_force_blocks_accepts_scientific_notation_and_signs() -> None:
    blocks = parse_outcar_force_blocks(
        OUTCAR_SCIENTIFIC_NOTATION_BLOCK,
        source_path=f"{FLOW_ROOT}/OUTCAR",
        expected_site_count=1,
    )

    assert len(blocks) == 1
    assert blocks[0].complete is True
    assert blocks[0].max_force_eV_per_A == pytest.approx(0.3)


def test_parse_outcar_force_blocks_reports_incomplete_final_block() -> None:
    blocks = parse_outcar_force_blocks(
        OUTCAR_INCOMPLETE_FINAL_BLOCK,
        source_path=f"{FLOW_ROOT}/OUTCAR",
        expected_site_count=2,
    )

    assert [(block.block_index, block.status, block.complete) for block in blocks] == [
        (1, "complete", True),
        (2, "incomplete", False),
    ]
    assert blocks[1].max_force_eV_per_A is None


def test_parse_outcar_force_blocks_reports_malformed_rows() -> None:
    blocks = parse_outcar_force_blocks(
        OUTCAR_MALFORMED_FORCE_BLOCK,
        source_path=f"{FLOW_ROOT}/OUTCAR",
        expected_site_count=1,
    )

    assert len(blocks) == 1
    assert blocks[0].status == "malformed"
    assert blocks[0].complete is False
    assert blocks[0].row_count == 0


def test_parse_outcar_force_blocks_validates_expected_site_count() -> None:
    blocks = parse_outcar_force_blocks(
        OUTCAR_TWO_FORCE_BLOCKS,
        source_path=f"{FLOW_ROOT}/OUTCAR",
        expected_site_count=3,
    )

    assert all(block.status == "row_count_mismatch" for block in blocks)
    assert all(block.complete is False for block in blocks)


def test_parse_outcar_force_extraction_accepts_readable_zero_block_output() -> None:
    blocks = parse_outcar_force_extraction(
        b"schema\tbmd-agent-outcar-force-v1\nexpected_site_count\t24\n",
        source_path=f"{FLOW_ROOT}/OUTCAR",
    )

    assert blocks == ()


def test_parse_outcar_force_extraction_rejects_malformed_output() -> None:
    with pytest.raises(RunInspectionError, match="malformed output"):
        parse_outcar_force_extraction(
            b"not-json-and-not-the-extractor-protocol\n",
            source_path=f"{FLOW_ROOT}/OUTCAR",
        )


def test_diagnose_run_uses_producer_stage_dirs_for_oszicar_evidence() -> None:
    files = default_files(include_vasprun=False)
    files[f"{FLOW_ROOT}/producer-alpha/INCAR"] = b"NELM = 3\nEDIFF = 1E-6\nNSW = 1\nEDIFFG = -0.02\n"
    files[f"{FLOW_ROOT}/producer-alpha/OSZICAR"] = OSZICAR_NELM_LIMIT
    files[f"{RESULT_DIR}/OSZICAR"] = OSZICAR_TWO_STEP
    remote = RemoteFixture(files=files, directories=default_directories())

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=slurm_runner,
    )

    assert diagnosis.termination.scheduler_state == "COMPLETED"
    assert diagnosis.termination.scheduler_timelimit == "72:00:00"
    assert diagnosis.termination.scheduler_reports_timeout is False
    assert [trajectory.stage_label for trajectory in diagnosis.trajectories] == [
        "producer-alpha",
        "producer-beta",
        "producer-gamma",
        "producer-delta",
    ]
    assert "result_dir" not in [trajectory.stage_label for trajectory in diagnosis.trajectories]

    first_stage = diagnosis.trajectories[0]
    assert first_stage.evidence_type == TRAJECTORY_OBSERVATION
    assert first_stage.stage_index == 1
    assert first_stage.oszicar_present is True
    assert first_stage.criteria["NELM"] == 3
    assert first_stage.criteria["EDIFF"] == 1e-06
    assert first_stage.criteria["NSW"] == 1
    assert first_stage.criteria["EDIFFG"] == -0.02
    assert first_stage.final_electronic_iteration_count == 3
    assert first_stage.completed_ionic_steps == 1
    assert first_stage.incomplete_electronic_iteration_count is None
    assert first_stage.vasprun_present is False

    final_stage = diagnosis.trajectories[-1]
    assert final_stage.stage_index == 4
    assert final_stage.ionic_steps_observed == 2
    assert final_stage.completed_ionic_steps == 2
    assert final_stage.electronic_iterations_by_ionic_step == (2, 2)


@pytest.mark.parametrize("custom", [False, True])
def test_diagnose_single_stage_uses_result_dir_binding(custom: bool) -> None:
    files = single_stage_files(custom=custom)
    files[f"{FLOW_ROOT}/INCAR"] = b"NELM = 60\nNSW = 2\nEDIFFG = -0.01\n"
    files[f"{FLOW_ROOT}/OSZICAR"] = OSZICAR_TWO_STEP
    files.pop(f"{FLOW_ROOT}/vasprun.xml")
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    assert len(diagnosis.trajectories) == 1
    trajectory = diagnosis.trajectories[0]
    assert trajectory.stage_label == "result_dir"
    assert trajectory.stage_index == 1
    assert trajectory.criteria["NSW"] == 2
    assert trajectory.ionic_steps_observed == 2
    assert trajectory.completed_ionic_steps == 2


def test_diagnose_vasprun_size_cap_skips_large_transfer() -> None:
    files = single_stage_files()
    files[f"{FLOW_ROOT}/OSZICAR"] = OSZICAR_TWO_STEP
    files[f"{FLOW_ROOT}/vasprun.xml"] = b"x" * 20
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
        max_vasprun_bytes=5,
    )

    trajectory = diagnosis.trajectories[0]
    assert trajectory.vasprun_present is True
    assert trajectory.vasprun_error is None
    assert trajectory.vasprun_skipped_reason is not None
    assert "exceeds limit 5" in trajectory.vasprun_skipped_reason
    assert all(
        "cat -- /bmd-db/guest/flows/validation-run/vasprun.xml" not in command
        for command in (" ".join(item) for item in remote.commands)
    )


def test_diagnose_vasprun_enriches_force_and_convergence_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTrajectoryVasprun:
        parameters = {"NELM": 60, "EDIFF": 1e-06, "NSW": 2, "EDIFFG": -0.01}
        converged_electronic = True
        converged_ionic = False
        ionic_steps = (
            {"forces": ((3.0, 4.0, 0.0), (0.0, 0.0, 1.0))},
            {"forces": ((0.0, 0.3, 0.4),)},
        )

    monkeypatch.setattr(run_resource, "_load_vasprun", lambda *args, **kwargs: FakeTrajectoryVasprun())

    files = single_stage_files()
    files[f"{FLOW_ROOT}/INCAR"] = b"NELM = 60\nEDIFF = 1E-6\nNSW = 2\nEDIFFG = -0.01\n"
    files[f"{FLOW_ROOT}/OSZICAR"] = OSZICAR_TWO_STEP
    files[f"{FLOW_ROOT}/vasprun.xml"] = b"<modeling/>"
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    trajectory = diagnosis.trajectories[0]
    assert trajectory.vasprun_present is True
    assert trajectory.vasprun_ionic_steps == 2
    assert trajectory.converged_electronic is True
    assert trajectory.converged_ionic is False
    assert trajectory.recent_ionic_steps[-1].max_force == 0.5
    assert trajectory.criteria["NELM"] == 60
    assert trajectory.criteria_discrepancies == ()


def test_diagnose_outcar_force_blocks_align_with_oszicar_completed_steps() -> None:
    files = single_stage_files()
    files[f"{FLOW_ROOT}/INCAR"] = b"NELM = 60\nEDIFF = 1E-6\nNSW = 2\nEDIFFG = -0.01\n"
    files[f"{FLOW_ROOT}/OSZICAR"] = OSZICAR_TWO_STEP
    files[f"{FLOW_ROOT}/OUTCAR"] = OUTCAR_TWO_FORCE_BLOCKS
    files.pop(f"{FLOW_ROOT}/vasprun.xml")
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    trajectory = diagnosis.trajectories[0]
    assert trajectory.outcar_present is True
    assert trajectory.outcar_expected_site_count == 2
    assert trajectory.outcar_complete_force_blocks == 2
    assert trajectory.outcar_force_alignment_status == "aligned"
    assert [block.max_force_eV_per_A for block in trajectory.outcar_force_blocks] == [5.0, 0.5]
    assert trajectory.recent_ionic_steps[-1].max_force == 0.5
    assert trajectory.recent_ionic_steps[-1].max_force_source == "OUTCAR"
    remote_commands = " ".join(" ".join(command) for command in remote.commands)
    assert "cat -- /bmd-db/guest/flows/validation-run/OUTCAR" not in remote_commands
    assert "awk -v expected=2" in remote_commands
    assert "find " not in remote_commands
    assert "POTCAR" not in remote_commands


def test_diagnose_outcar_incomplete_final_block_keeps_complete_step_forces() -> None:
    files = single_stage_files()
    files[f"{FLOW_ROOT}/INCAR"] = b"NELM = 60\nEDIFF = 1E-6\nNSW = 1\nEDIFFG = -0.01\n"
    files[f"{FLOW_ROOT}/OSZICAR"] = OSZICAR_NELM_LIMIT
    files[f"{FLOW_ROOT}/OUTCAR"] = OUTCAR_INCOMPLETE_FINAL_BLOCK
    files.pop(f"{FLOW_ROOT}/vasprun.xml")
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    trajectory = diagnosis.trajectories[0]
    assert trajectory.completed_ionic_steps == 1
    assert trajectory.outcar_complete_force_blocks == 1
    assert trajectory.outcar_force_alignment_status == "aligned"
    assert trajectory.recent_ionic_steps[-1].max_force == 0.2
    assert any("OUTCAR force block 2 incomplete" in item for item in trajectory.unavailable)


def test_diagnose_outcar_count_disagreement_does_not_align_uncertain_forces() -> None:
    files = single_stage_files()
    files[f"{FLOW_ROOT}/INCAR"] = b"NELM = 60\nEDIFF = 1E-6\nNSW = 3\nEDIFFG = -0.01\n"
    files[f"{FLOW_ROOT}/OSZICAR"] = OSZICAR_TWO_STEP
    files[f"{FLOW_ROOT}/OUTCAR"] = OUTCAR_INCOMPLETE_FINAL_BLOCK
    files.pop(f"{FLOW_ROOT}/vasprun.xml")
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    trajectory = diagnosis.trajectories[0]
    assert trajectory.completed_ionic_steps == 2
    assert trajectory.outcar_complete_force_blocks == 1
    assert trajectory.outcar_force_alignment_status == "discrepancy"
    assert all(step.max_force is None for step in trajectory.ionic_steps)
    assert "OUTCAR force-block alignment discrepancy" in " ".join(trajectory.unavailable)


def test_diagnose_readable_outcar_with_zero_force_blocks_is_not_transport_failure() -> None:
    files = single_stage_files()
    files[f"{FLOW_ROOT}/INCAR"] = b"NELM = 60\nEDIFF = 1E-6\nNSW = 2\nEDIFFG = -0.01\n"
    files[f"{FLOW_ROOT}/OSZICAR"] = OSZICAR_TWO_STEP
    files[f"{FLOW_ROOT}/OUTCAR"] = b"readable OUTCAR with no standard force table\n"
    files.pop(f"{FLOW_ROOT}/vasprun.xml")
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    trajectory = diagnosis.trajectories[0]
    assert trajectory.outcar_present is True
    assert trajectory.outcar_error is None
    assert trajectory.outcar_failure_kind is None
    assert trajectory.outcar_complete_force_blocks == 0
    assert trajectory.outcar_force_alignment_status == "unavailable"
    assert "OUTCAR contained no standard force blocks" in " ".join(trajectory.unavailable)


def test_diagnose_outcar_extractor_invocation_failure_preserves_kind(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingExtractorRemote(RemoteFixture):
        def __call__(
            self,
            command: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[bytes]:
            remote_command = command[2]
            if shlex.split(remote_command)[0:2] == ["awk", "-v"]:
                raise subprocess.CalledProcessError(
                    2,
                    command,
                    stderr=b"sh: 1: Syntax error: Unterminated quoted string\n",
                )
            return super().__call__(command, **kwargs)

    files = single_stage_files()
    files[f"{FLOW_ROOT}/INCAR"] = b"NELM = 60\nEDIFF = 1E-6\nNSW = 2\nEDIFFG = -0.01\n"
    files[f"{FLOW_ROOT}/OSZICAR"] = OSZICAR_TWO_STEP
    files.pop(f"{FLOW_ROOT}/vasprun.xml")
    remote = FailingExtractorRemote(files=files, directories={FLOW_ROOT})

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    trajectory = diagnosis.trajectories[0]
    assert trajectory.outcar_present is True
    assert trajectory.outcar_error == "extractor command invocation failed"
    assert trajectory.outcar_failure_kind == "command_invocation_failed"
    assert trajectory.outcar_failure_returncode == 2
    assert "Unterminated quoted string" in (trajectory.outcar_failure_detail or "")
    assert all(step.max_force is None for step in trajectory.ionic_steps)

    cli.print_run_diagnosis(diagnosis)

    captured = capsys.readouterr()
    assert "OUTCAR force trajectory unavailable: extractor command invocation failed" in captured.out
    assert "OUTCAR extractor diagnostic: command_invocation_failed, exit 2" in captured.out
    assert "Unterminated quoted string" not in captured.out


def test_diagnose_malformed_outcar_extractor_output_is_distinct_from_read_failure() -> None:
    class MalformedExtractorRemote(RemoteFixture):
        def __call__(
            self,
            command: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[bytes]:
            remote_command = command[2]
            if shlex.split(remote_command)[0:2] == ["awk", "-v"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=b"this is not extractor output\n",
                    stderr=b"",
                )
            return super().__call__(command, **kwargs)

    files = single_stage_files()
    files[f"{FLOW_ROOT}/OSZICAR"] = OSZICAR_TWO_STEP
    files.pop(f"{FLOW_ROOT}/vasprun.xml")
    remote = MalformedExtractorRemote(files=files, directories={FLOW_ROOT})

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    trajectory = diagnosis.trajectories[0]
    assert trajectory.outcar_present is True
    assert trajectory.outcar_error == "OUTCAR force extraction returned malformed output"
    assert trajectory.outcar_failure_kind == "malformed_extractor_output"
    assert trajectory.outcar_failure_returncode is None


def test_direct_vasp_outcar_force_extraction_uses_fixed_read_only_command() -> None:
    remote = RemoteFixture(files=direct_vasp_files(), directories={DIRECT_DIR})

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(),
        scientific_parser=fake_direct_scientific_parser,
    )

    assert inspection.direct_vasp is not None
    trajectory = inspection.direct_vasp.trajectory
    assert trajectory.outcar_force_alignment_status == "aligned"
    assert trajectory.recent_ionic_steps[-1].max_force_source == "OUTCAR"
    commands = " ".join(" ".join(command) for command in remote.commands)
    assert "cat -- /bmd-db/guest/flows/direct-vasp/OUTCAR" not in commands
    assert "awk -v expected=2" in commands
    assert "find " not in commands
    assert "ls " not in commands
    assert "POTCAR" not in commands


def test_trajectory_progress_evidence_describes_force_history_without_assessment_change() -> None:
    trajectory = force_progress_trajectory(
        (0.854786, 0.2, 0.014801, 0.08, 0.071891),
        criteria={"NELM": 200, "EDIFF": 1e-6, "NSW": 99, "EDIFFG": -0.01, "ISIF": 3},
        electronic_counts=(12, 14, 20, 18, 16),
    )

    progress = derive_trajectory_progress_evidence(trajectory)
    assessments = assess_convergence_progress((trajectory,))

    assert progress.evidence_type == TRAJECTORY_PROGRESS_EVIDENCE
    assert progress.atomic_force_status == "available"
    assert progress.force_source == "OUTCAR"
    assert progress.force_observation_count == 5
    assert progress.force_criterion_magnitude_eV_A == 0.01
    assert progress.initial_max_force_eV_A == 0.854786
    assert progress.current_max_force_eV_A == 0.071891
    assert progress.best_max_force_eV_A == 0.014801
    assert progress.best_force_step == 3
    assert progress.initial_force_over_abs_EDIFFG == pytest.approx(85.4786)
    assert progress.current_force_over_abs_EDIFFG == pytest.approx(7.1891)
    assert progress.best_force_over_abs_EDIFFG == pytest.approx(1.4801)
    assert progress.initial_to_current_force_ratio == pytest.approx(
        round(0.854786 / 0.071891, 6)
    )
    assert progress.initial_to_best_force_ratio == pytest.approx(
        round(0.854786 / 0.014801, 6)
    )
    assert progress.current_to_best_force_ratio == pytest.approx(
        round(0.071891 / 0.014801, 6)
    )
    assert progress.new_best_force_count == 3
    assert progress.min_electronic_iterations == 12
    assert progress.median_electronic_iterations == 16.0
    assert progress.max_electronic_iterations == 20
    assert "variable-cell convergence" in " ".join(progress.limitations)
    assert assessment_by_scope(assessments, "ionic").label == INSUFFICIENT_EVIDENCE
    assert assessment_by_scope(assessments, "stage").label == INSUFFICIENT_EVIDENCE


def test_trajectory_progress_evidence_uses_first_best_force_occurrence() -> None:
    trajectory = force_progress_trajectory((0.5, 0.2, 0.2, 0.3))

    progress = derive_trajectory_progress_evidence(trajectory)

    assert progress.best_max_force_eV_A == 0.2
    assert progress.best_force_step == 2
    assert progress.new_best_force_count == 2


@pytest.mark.parametrize("criteria", [{"EDIFFG": 0.01}, {"EDIFFG": 0}, {}])
def test_trajectory_progress_evidence_has_null_criterion_ratios_without_negative_ediffg(
    criteria: dict[str, object],
) -> None:
    trajectory = force_progress_trajectory((0.5, 0.25), criteria=criteria)

    progress = derive_trajectory_progress_evidence(trajectory)

    assert progress.force_criterion_magnitude_eV_A is None
    assert progress.initial_force_over_abs_EDIFFG is None
    assert progress.current_force_over_abs_EDIFFG is None
    assert progress.best_force_over_abs_EDIFFG is None


def test_trajectory_progress_evidence_handles_zero_force_denominators() -> None:
    trajectory = force_progress_trajectory((1.0, 0.0))

    progress = derive_trajectory_progress_evidence(trajectory)

    assert progress.current_max_force_eV_A == 0.0
    assert progress.best_max_force_eV_A == 0.0
    assert progress.current_force_over_abs_EDIFFG == 0.0
    assert progress.best_force_over_abs_EDIFFG == 0.0
    assert progress.initial_to_current_force_ratio is None
    assert progress.initial_to_best_force_ratio is None
    assert progress.current_to_best_force_ratio is None
    assert "denominator force is zero" in " ".join(progress.limitations)


def test_trajectory_progress_evidence_reports_even_length_electronic_median() -> None:
    trajectory = force_progress_trajectory(
        (0.5, 0.4, 0.3, 0.2),
        electronic_counts=(5, 9, 17, 21),
    )

    progress = derive_trajectory_progress_evidence(trajectory)

    assert progress.min_electronic_iterations == 5
    assert progress.median_electronic_iterations == 13.0
    assert progress.max_electronic_iterations == 21


def test_trajectory_progress_evidence_reports_missing_force_evidence() -> None:
    trajectory = force_progress_trajectory((None, None), electronic_counts=(7, 9))

    progress = derive_trajectory_progress_evidence(trajectory)

    assert progress.atomic_force_status == "unavailable"
    assert progress.initial_max_force_eV_A is None
    assert progress.best_force_step is None
    assert progress.new_best_force_count is None
    assert progress.electronic_iteration_status == "available"
    assert progress.min_electronic_iterations == 7
    assert "completed-step atomic maximum-force evidence unavailable" in " ".join(
        progress.limitations
    )


def test_trajectory_progress_evidence_rejects_discrepant_outcar_alignment() -> None:
    trajectory = force_progress_trajectory(
        (0.5, 0.25),
        alignment_status="discrepancy",
        alignment_reason="OSZICAR completed ionic steps 2 did not align with OUTCAR blocks",
    )

    progress = derive_trajectory_progress_evidence(trajectory)

    assert progress.atomic_force_status == "unavailable"
    assert progress.force_observation_count == 0
    assert progress.initial_max_force_eV_A is None
    assert "aligned OUTCAR force evidence unavailable" in " ".join(progress.limitations)


def test_serialize_job_trajectory_evidence_completed_direct_vasp_job() -> None:
    remote = RemoteFixture(files=direct_vasp_files(), directories={DIRECT_DIR})

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(),
        scientific_parser=fake_direct_scientific_parser,
    )

    payload = serialize_job_trajectory_evidence(inspection)
    commands_after_inspection = list(remote.commands)
    serialize_job_trajectory_evidence(inspection)
    assert remote.commands == commands_after_inspection

    assert payload["schema_version"] == 1
    assert payload["job"]["job_id"] == "20893681"
    assert payload["job"]["job_name"] == "direct-vasp"
    assert payload["job"]["node_list"] == "compute-0-269"
    assert payload["job"]["scheduler_state"] == "COMPLETED"
    assert payload["job"]["elapsed_raw"] == 21620
    assert payload["job"]["timelimit"] == "06:00:00"
    assert payload["job"]["allocated_cpus"] == 24
    assert payload["job"]["work_dir"] == DIRECT_DIR
    assert payload["oom_diagnostic_evidence"]["assessment"] == INSUFFICIENT_OOM_EVIDENCE
    assert payload["calculation"]["calculation_type"] == "direct VASP"
    assert payload["calculation"]["producer_provenance"]["status"] == "unavailable"

    stages = payload["stages"]
    assert len(stages) == 1
    stage = stages[0]
    assert stage["stage_index"] == 1
    assert stage["stage_label"] == "work_dir"
    assert stage["criteria"]["NELM"] == 60
    assert stage["criteria"]["EDIFF"] == 1e-06
    assert stage["criteria"]["EDIFFG"] == -0.01
    assert stage["criteria"]["ISIF"] is None
    assert stage["completed_ionic_steps"] == 2
    assert stage["outcar_force_block_count"] == 2
    assert stage["outcar_force_alignment_status"] == "aligned"
    assert stage["outcar"]["force_block_count"] == 2
    assert stage["outcar"]["complete_force_blocks"] == 2
    assert stage["outcar"]["force_alignment"]["status"] == "aligned"
    assert [block["max_force_eV_per_A"] for block in stage["outcar"]["force_blocks"]] == [5.0, 0.5]
    assert len(stage["ionic_steps"]) == 2
    assert stage["ionic_steps"][0]["step_index"] == 1
    assert stage["ionic_steps"][0]["free_energy"] == -11.0
    assert stage["ionic_steps"][0]["energy_zero"] == -10.95
    assert stage["ionic_steps"][0]["ionic_dE"] == -11.0
    assert stage["ionic_steps"][0]["electronic_iterations"] == 2
    assert stage["ionic_steps"][0]["max_force"] == 5.0
    assert stage["ionic_steps"][0]["max_force_source"] == "OUTCAR"
    assert stage["ionic_steps"][0]["force_over_abs_EDIFFG"] == 500.0
    progress = stage["trajectory_progress_evidence"]
    assert progress["evidence_type"] == TRAJECTORY_PROGRESS_EVIDENCE
    assert progress["atomic_force_status"] == "available"
    assert progress["initial_max_force_eV_A"] == 5.0
    assert progress["current_max_force_eV_A"] == 0.5
    assert progress["best_force_step"] == 2
    assert progress["ionic_dE_semantics"].startswith("VASP OSZICAR ionic-line d E")
    assert payload["convergence_progress_assessment"]
    assert "POTCAR" not in json.dumps(payload, sort_keys=True)


def test_serialize_job_trajectory_evidence_timeout_keeps_oszicar_without_vasprun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def malformed_loader(*args: object, **kwargs: object) -> object:
        warnings.warn(
            "XML is malformed. Parsing has stopped but partial data is available.",
            UserWarning,
        )
        raise IndexError("list index out of range")

    monkeypatch.setattr(run_resource, "_load_vasprun", malformed_loader)
    remote = RemoteFixture(
        files=direct_vasp_files(oszicar=oszicar_incomplete_cycle(25)),
        directories={DIRECT_DIR},
    )

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(state="TIMEOUT"),
        scientific_parser=fake_direct_scientific_parser,
    )

    payload = serialize_job_trajectory_evidence(inspection)
    stage = payload["stages"][0]

    assert payload["job"]["scheduler_state"] == "TIMEOUT"
    assert stage["completed_ionic_steps"] == 0
    assert stage["ionic_steps"] == []
    assert stage["electronic"]["incomplete_electronic_iteration_count"] == 25
    assert stage["electronic"]["electronic_iterations_by_completed_ionic_step"] == []
    assert stage["vasprun"]["present"] is True
    assert stage["vasprun"]["error"] == "file could not be parsed completely"
    assert stage["vasprun"]["ionic_steps"] is None
    assert stage["converged_electronic"] is None
    assert stage["converged_ionic"] is None
    assert stage["criteria"]["ISIF"] is None
    assert "list index out of range" not in json.dumps(payload)
    assert "XML is malformed" not in json.dumps(payload)
    assessment_labels = {
        assessment["scope"]: assessment["label"]
        for assessment in payload["convergence_progress_assessment"]
    }
    assert assessment_labels["stage"] == INSUFFICIENT_EVIDENCE


def test_serialize_job_trajectory_evidence_preserves_force_alignment_discrepancy() -> None:
    remote = RemoteFixture(
        files=direct_vasp_files(outcar=OUTCAR_INCOMPLETE_FINAL_BLOCK),
        directories={DIRECT_DIR},
    )

    inspection = inspect_slurm_job(
        cluster(),
        "20893681",
        remote_runner=remote,
        slurm_runner=job_slurm_runner(state="TIMEOUT"),
        scientific_parser=fake_direct_scientific_parser,
    )

    payload = serialize_job_trajectory_evidence(inspection)
    stage = payload["stages"][0]

    assert stage["outcar"]["force_block_count"] == 2
    assert stage["outcar"]["complete_force_blocks"] == 1
    assert stage["outcar_force_block_count"] == 2
    assert stage["outcar_force_alignment_status"] == "discrepancy"
    assert stage["outcar"]["force_alignment"]["status"] == "discrepancy"
    assert "did not align" in stage["outcar"]["force_alignment"]["reason"]
    assert stage["ionic_steps"][0]["max_force"] is None
    assert stage["ionic_steps"][0]["force_over_abs_EDIFFG"] is None
    assert "OUTCAR force-block alignment discrepancy" in " ".join(stage["unavailable"])


def test_serialize_job_trajectory_evidence_retains_full_ionic_sequence() -> None:
    ionic_steps = tuple(
        IonicStepObservation(
            index,
            8,
            -10.0 - index,
            -9.9 - index,
            -0.1,
            round(0.1 * index, 1),
            "OUTCAR",
        )
        for index in range(1, 8)
    )
    trajectory = trajectory_observation(
        stage_type="relax",
        criteria={"NELM": 60, "EDIFF": 1e-6, "NSW": 20, "EDIFFG": -0.01, "ISIF": 3},
        completed_ionic_steps=7,
        electronic_iterations=(8,) * 7,
        electronic_cycles=tuple(
            electronic_cycle(index, 8, dE=1e-7)
            for index in range(1, 8)
        ),
        ionic_steps=ionic_steps,
        converged_electronic=True,
        converged_ionic=False,
    )
    trajectory = replace(
        trajectory,
        outcar_path=f"{DIRECT_DIR}/OUTCAR",
        outcar_present=True,
        outcar_expected_site_count=24,
        outcar_force_blocks=tuple(
            OutcarForceBlockObservation(
                index,
                24,
                "complete",
                True,
                f"{DIRECT_DIR}/OUTCAR",
                round(0.1 * index, 1),
            )
            for index in range(1, 8)
        ),
        outcar_complete_force_blocks=7,
        outcar_force_alignment_status="aligned",
        outcar_force_alignment_reason=(
            "7 OUTCAR force block(s) aligned with OSZICAR completed ionic steps"
        ),
    )
    direct = run_resource.DirectVaspInspection(
        directory=DIRECT_DIR,
        artifacts=(),
        executed_inputs=(),
        scientific=ScientificResult(source_paths=()),
        trajectory=trajectory,
        assessments=assess_convergence_progress((trajectory,)),
    )
    inspection = JobInspection(
        job_id="20893681",
        scheduler=None,
        scheduler_error=None,
        scheduler_work_dir=DIRECT_DIR,
        calculation_directory=DIRECT_DIR,
        calculation_type="direct VASP",
        calculation_reason=None,
        direct_vasp=direct,
    )

    payload = serialize_job_trajectory_evidence(inspection)
    exported_steps = payload["stages"][0]["ionic_steps"]

    assert len(exported_steps) == 7
    assert [step["step_index"] for step in exported_steps] == list(range(1, 8))
    assert payload["stages"][0]["completed_ionic_steps"] == 7
    assert payload["stages"][0]["ionic_steps"][-1]["max_force"] == 0.7
    assert payload["stages"][0]["trajectory_progress_evidence"]["force_observation_count"] == 7


def test_cli_job_trajectory_json_uses_existing_inspection_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trajectory = trajectory_observation(
        completed_ionic_steps=1,
        electronic_iterations=(8,),
        ionic_steps=(IonicStepObservation(1, 8, -10.0, -9.9, -0.1, 0.02, "OUTCAR"),),
        criteria={"NELM": 60, "EDIFFG": -0.01},
    )
    direct = run_resource.DirectVaspInspection(
        directory=DIRECT_DIR,
        artifacts=(),
        executed_inputs=(),
        scientific=ScientificResult(source_paths=()),
        trajectory=trajectory,
        assessments=assess_convergence_progress((trajectory,)),
    )
    inspection = JobInspection(
        job_id="20893681",
        scheduler=SlurmAccountingRecord(
            job_id="20893681",
            name="direct-vasp",
            state="TIMEOUT",
            elapsed="06:00:20",
            start="2026-08-30T00:00:00",
            end="2026-08-30T06:00:20",
            partition="leeburton-pool",
            exit_code="0:0",
            node_list="compute-0-269",
            allocated_cpus=24,
            work_dir=DIRECT_DIR,
        ),
        scheduler_error=None,
        scheduler_work_dir=DIRECT_DIR,
        calculation_directory=DIRECT_DIR,
        calculation_type="direct VASP",
        calculation_reason=None,
        direct_vasp=direct,
    )
    registry = ResourceRegistry(repositories={}, clusters={"powerslurm": cluster()})
    calls: list[str] = []

    def fake_inspect_slurm_job(
        cluster: SlurmClusterResource,
        job_id: str,
        **kwargs: object,
    ) -> JobInspection:
        calls.append(job_id)
        return inspection

    def fail_text_summary(inspection: JobInspection) -> None:
        raise AssertionError("text summary should not be used for trajectory JSON")

    monkeypatch.setattr(cli, "load_resources", lambda: registry)
    monkeypatch.setattr(
        cli,
        "compute_policy_from_registry",
        lambda registry, **kwargs: compute_policy_not_configured("not configured in test"),
    )
    monkeypatch.setattr(cli, "inspect_slurm_job", fake_inspect_slurm_job)
    monkeypatch.setattr(cli, "print_job_inspection", fail_text_summary)

    exit_code = cli.main(["job", "20893681", "--trajectory-json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert calls == ["20893681"]
    assert "BMD Job Inspection" not in captured.out
    assert payload["schema_version"] == 1
    assert payload["job"]["scheduler_state"] == "TIMEOUT"
    assert payload["stages"][0]["ionic_steps"][0]["max_force"] == 0.02
    assert captured.err == ""


def test_cli_job_defaults_to_concise_summary_and_verbose_keeps_detailed_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = ResourceRegistry(repositories={}, clusters={"powerslurm": cluster()})
    inspection = JobInspection(
        job_id="20893681",
        scheduler=None,
        scheduler_error="not available",
        scheduler_work_dir=None,
        calculation_directory=None,
        calculation_type="unknown",
        calculation_reason="scheduler accounting was unavailable",
    )

    monkeypatch.setattr(cli, "load_resources", lambda: registry)
    monkeypatch.setattr(
        cli,
        "compute_policy_from_registry",
        lambda registry, **kwargs: compute_policy_not_configured("not configured in test"),
    )
    monkeypatch.setattr(cli, "inspect_slurm_job", lambda *args, **kwargs: inspection)

    exit_code = cli.main(["job", "20893681"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("BMD Agent\n=========\n\n")
    assert "Status: UNKNOWN" in captured.out
    assert "bmd-agent job 20893681 --verbose" in captured.out
    assert "Job (scheduler_observation):" not in captured.out
    assert "{" not in captured.out

    exit_code = cli.main(["job", "20893681", "--verbose"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("BMD Job Inspection\n==================\n\n")
    assert "Job (scheduler_observation):" in captured.out


def test_diagnose_malformed_vasprun_keeps_oszicar_evidence_and_quiet_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def malformed_loader(*args: object, **kwargs: object) -> object:
        warnings.warn(
            "XML is malformed. Parsing has stopped but partial data is available.",
            UserWarning,
        )
        raise IndexError("list index out of range")

    monkeypatch.setattr(run_resource, "_load_vasprun", malformed_loader)

    files = single_stage_files()
    files[f"{FLOW_ROOT}/INCAR"] = b"NELM = 200\nEDIFF = 1E-6\n"
    files[f"{FLOW_ROOT}/OSZICAR"] = oszicar_incomplete_cycle(25)
    files[f"{FLOW_ROOT}/vasprun.xml"] = b"<modeling><calculation>"
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    trajectory = diagnosis.trajectories[0]
    assert trajectory.oszicar_present is True
    assert trajectory.completed_ionic_steps == 0
    assert trajectory.electronic_iterations_by_completed_ionic_step == ()
    assert trajectory.incomplete_electronic_iteration_count == 25
    assert trajectory.recent_incomplete_electronic_iterations[-1].iteration == 25
    assert trajectory.vasprun_present is True
    assert trajectory.vasprun_error == "file could not be parsed completely"
    assert "list index out of range" not in " ".join(trajectory.unavailable)
    assert "XML is malformed" not in " ".join(trajectory.unavailable)

    cli.print_run_diagnosis(diagnosis)

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "completed ionic steps: 0" in captured.out
    assert "electronic iterations for completed ionic steps: none" in captured.out
    assert "incomplete electronic cycle: 25 iterations observed / NELM 200" in captured.out
    assert "vasprun trajectory enrichment unavailable: file could not be parsed completely" in captured.out
    assert "list index out of range" not in captured.out
    assert "XML is malformed" not in captured.out


def test_convergence_progress_assessment_reports_converged_static() -> None:
    trajectory = trajectory_observation(
        stage_type="static",
        criteria={"NELM": 60, "EDIFF": 1e-6, "NSW": 0},
        completed_ionic_steps=1,
        electronic_iterations=(8,),
        electronic_cycles=(electronic_cycle(1, 8, dE=1e-7),),
        converged_electronic=True,
    )

    assessments = assess_convergence_progress((trajectory,))

    assert assessment_by_scope(assessments, "electronic").label == CONVERGED
    stage = assessment_by_scope(assessments, "stage")
    assert stage.label == CONVERGED
    assert stage.evidence_type == "convergence_progress_assessment"
    assert all(assessment.scope != "ionic" for assessment in assessments)


def test_convergence_progress_assessment_reports_converged_relaxation() -> None:
    trajectory = trajectory_observation(
        stage_type="relax",
        criteria={"NELM": 200, "EDIFF": 1e-6, "NSW": 99, "EDIFFG": -0.01},
        completed_ionic_steps=65,
        electronic_iterations=(12,) * 65,
        electronic_cycles=tuple(
            electronic_cycle(index, 12, dE=1e-7)
            for index in range(1, 66)
        ),
        ionic_steps=(IonicStepObservation(65, 12, -100.0, -99.9, -0.01, 0.009465),),
        converged_electronic=True,
        converged_ionic=True,
    )

    assessments = assess_convergence_progress((trajectory,))

    assert assessment_by_scope(assessments, "electronic").label == CONVERGED
    assert assessment_by_scope(assessments, "ionic").label == CONVERGED
    assert assessment_by_scope(assessments, "stage").label == CONVERGED
    assert assessment_by_scope(assessments, "stage").features["completed_ionic_steps"] == 65


def test_convergence_progress_assessment_can_use_reached_configured_criteria() -> None:
    trajectory = trajectory_observation(
        stage_type="relax",
        criteria={"NELM": 60, "EDIFF": 1e-6, "NSW": 20, "EDIFFG": -0.01},
        completed_ionic_steps=3,
        electronic_iterations=(8, 8, 8),
        electronic_cycles=(
            electronic_cycle(1, 8, dE=1e-7),
            electronic_cycle(2, 8, dE=1e-7),
            electronic_cycle(3, 8, dE=1e-7),
        ),
        ionic_steps=(IonicStepObservation(3, 8, -10.1, -10.0, -0.01, 0.009),),
        converged_electronic=None,
        converged_ionic=None,
    )

    assessments = assess_convergence_progress((trajectory,))

    assert assessment_by_scope(assessments, "electronic").label == CONVERGED
    assert assessment_by_scope(assessments, "ionic").label == CONVERGED
    assert assessment_by_scope(assessments, "stage").label == CONVERGED


def test_outcar_atomic_forces_do_not_claim_variable_cell_convergence_alone() -> None:
    trajectory = trajectory_observation(
        stage_type="relax",
        criteria={"NELM": 60, "EDIFF": 1e-6, "NSW": 20, "EDIFFG": -0.01, "ISIF": 3},
        completed_ionic_steps=3,
        electronic_iterations=(8, 8, 8),
        electronic_cycles=(
            electronic_cycle(1, 8, dE=1e-7),
            electronic_cycle(2, 8, dE=1e-7),
            electronic_cycle(3, 8, dE=1e-7),
        ),
        ionic_steps=(
            IonicStepObservation(
                3,
                8,
                -10.1,
                -10.0,
                -0.01,
                0.009,
                "OUTCAR",
            ),
        ),
        converged_electronic=True,
        converged_ionic=None,
    )

    assessments = assess_convergence_progress((trajectory,))

    ionic = assessment_by_scope(assessments, "ionic")
    assert ionic.label == INSUFFICIENT_EVIDENCE
    assert "OUTCAR atomic maximum force reached EDIFFG" in " ".join(ionic.basis)
    assert "cell degrees of freedom" in " ".join(ionic.limitations)
    assert assessment_by_scope(assessments, "stage").label == INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize("iterations", [25, 17])
def test_convergence_progress_assessment_timeout_first_scf_is_insufficient(
    iterations: int,
) -> None:
    trajectory = trajectory_observation(
        stage_type="relax",
        criteria={"NELM": 200, "EDIFF": 1e-6, "NSW": 99, "EDIFFG": -0.01},
        completed_ionic_steps=0,
        electronic_iterations=(iterations,),
        electronic_cycles=(
            electronic_cycle(iterations, iterations, completed=False, dE=-1e-3),
        ),
        incomplete_electronic_iteration_count=iterations,
        recent_incomplete=(
            ElectronicIterationObservation(iterations, "DAV", -10.0, -1e-3, -1e-4, 0.1, 0.02),
        ),
        converged_electronic=None,
        converged_ionic=None,
    )

    assessments = assess_convergence_progress((trajectory,))

    assert assessment_by_scope(assessments, "electronic").label == INSUFFICIENT_EVIDENCE
    assert assessment_by_scope(assessments, "ionic").label == INSUFFICIENT_EVIDENCE
    stage = assessment_by_scope(assessments, "stage")
    assert stage.label == INSUFFICIENT_EVIDENCE
    assert "incomplete first SCF cycle" in " ".join(stage.limitations)
    assert all(assessment.label != NO_CLEAR_EVIDENCE_OF_PROGRESS for assessment in assessments)


def test_force_based_relaxation_stage_does_not_promote_electronic_convergence_alone() -> None:
    trajectory = trajectory_observation(
        stage_type="relax",
        criteria={"NELM": 200, "EDIFF": 1e-6, "NSW": 99, "EDIFFG": -0.01},
        completed_ionic_steps=50,
        electronic_iterations=(8,) * 50,
        electronic_cycles=tuple(
            electronic_cycle(index, 8, dE=1e-7)
            for index in range(1, 51)
        ),
        ionic_steps=(
            IonicStepObservation(50, 8, -100.0, -99.9, -0.1, None),
        ),
        converged_electronic=True,
        converged_ionic=None,
    )

    assessments = assess_convergence_progress((trajectory,))

    assert assessment_by_scope(assessments, "electronic").label == CONVERGED
    ionic = assessment_by_scope(assessments, "ionic")
    assert ionic.label == INSUFFICIENT_EVIDENCE
    assert "maximum force evidence unavailable" in ionic.limitations
    stage = assessment_by_scope(assessments, "stage")
    assert stage.label == INSUFFICIENT_EVIDENCE
    assert "converged_electronic=True" in " ".join(stage.basis)
    assert "ionic progress evidence is insufficient" in " ".join(stage.limitations)


def test_unreached_force_criterion_without_trend_rule_is_insufficient_progress_evidence() -> None:
    ionic_steps = tuple(
        IonicStepObservation(
            index,
            8,
            -100.0 - index,
            -99.9 - index,
            -0.1,
            force,
            "OUTCAR",
        )
        for index, force in enumerate(
            (
                0.028543,
                0.026002,
                0.022678,
                0.026931,
                0.071891,
            ),
            start=46,
        )
    )
    trajectory = trajectory_observation(
        stage_type="relax",
        criteria={"NELM": 200, "EDIFF": 1e-6, "NSW": 99, "EDIFFG": -0.01, "ISIF": 3},
        completed_ionic_steps=50,
        electronic_iterations=(8,) * 50,
        electronic_cycles=tuple(
            electronic_cycle(index, 8, dE=1e-7)
            for index in range(1, 51)
        ),
        ionic_steps=ionic_steps,
        converged_electronic=True,
        converged_ionic=None,
    )
    trajectory = replace(
        trajectory,
        outcar_path=f"{FLOW_ROOT}/OUTCAR",
        outcar_present=True,
        outcar_expected_site_count=24,
        outcar_force_blocks=tuple(
            OutcarForceBlockObservation(
                index,
                24,
                "complete",
                True,
                f"{FLOW_ROOT}/OUTCAR",
                force,
            )
            for index, force in enumerate(
                (
                    0.028543,
                    0.026002,
                    0.022678,
                    0.026931,
                    0.071891,
                ),
                start=46,
            )
        ),
        outcar_complete_force_blocks=50,
        outcar_force_alignment_status="aligned",
        outcar_force_alignment_reason=(
            "50 OUTCAR force block(s) aligned with OSZICAR completed ionic steps"
        ),
    )

    assessments = assess_convergence_progress((trajectory,))

    ionic = assessment_by_scope(assessments, "ionic")
    assert ionic.label == INSUFFICIENT_EVIDENCE
    assert "final OUTCAR atomic maximum force has not reached EDIFFG" in " ".join(
        ionic.counter_evidence
    )
    assert "trend-based ionic progress assessment is not implemented" in " ".join(
        ionic.limitations
    )
    stage = assessment_by_scope(assessments, "stage")
    assert stage.label == INSUFFICIENT_EVIDENCE
    assert "ionic progress evidence is insufficient" in " ".join(stage.limitations)
    assert all(assessment.label != NO_CLEAR_EVIDENCE_OF_PROGRESS for assessment in assessments)


def test_convergence_progress_assessment_missing_oszicar_is_insufficient() -> None:
    trajectory = trajectory_observation(
        stage_type="static",
        criteria={"NELM": 60, "EDIFF": 1e-6},
        completed_ionic_steps=None,
        oszicar_present=False,
        vasprun_present=False,
    )

    electronic = assessment_by_scope(
        assess_convergence_progress((trajectory,)),
        "electronic",
    )

    assert electronic.label == INSUFFICIENT_EVIDENCE
    assert "OSZICAR trajectory unavailable" in electronic.limitations


def test_convergence_progress_assessment_malformed_vasprun_keeps_oszicar_basis() -> None:
    trajectory = trajectory_observation(
        stage_type="relax",
        criteria={"NELM": 60, "EDIFF": 1e-6, "NSW": 20, "EDIFFG": -0.01},
        completed_ionic_steps=2,
        electronic_iterations=(8, 9),
        electronic_cycles=(
            electronic_cycle(1, 8, dE=1e-5),
            electronic_cycle(2, 9, dE=1e-5),
        ),
        ionic_steps=(IonicStepObservation(1, 8, -10.0), IonicStepObservation(2, 9, -10.2)),
        vasprun_error="file could not be parsed completely",
        converged_electronic=None,
        converged_ionic=None,
    )

    assessments = assess_convergence_progress((trajectory,))

    assert assessment_by_scope(assessments, "electronic").label == EVIDENCE_OF_PROGRESS
    ionic = assessment_by_scope(assessments, "ionic")
    assert ionic.label == INSUFFICIENT_EVIDENCE
    assert "maximum force evidence unavailable" in ionic.limitations


def test_convergence_progress_assessment_missing_criteria_is_insufficient() -> None:
    trajectory = trajectory_observation(
        stage_type="static",
        criteria={},
        completed_ionic_steps=1,
        electronic_iterations=(8,),
        electronic_cycles=(electronic_cycle(1, 8, dE=1e-7),),
        converged_electronic=None,
    )

    electronic = assessment_by_scope(
        assess_convergence_progress((trajectory,)),
        "electronic",
    )

    assert electronic.label == INSUFFICIENT_EVIDENCE
    assert "EDIFF criterion unavailable" in electronic.limitations


def test_convergence_progress_assessment_does_not_treat_unreached_force_as_no_progress() -> None:
    trajectory = trajectory_observation(
        stage_type="relax",
        criteria={"NELM": 60, "EDIFF": 1e-6, "NSW": 20, "EDIFFG": -0.01},
        completed_ionic_steps=2,
        electronic_iterations=(8, 8),
        electronic_cycles=(
            electronic_cycle(1, 8, dE=1e-7),
            electronic_cycle(2, 8, dE=1e-7),
        ),
        ionic_steps=(IonicStepObservation(2, 8, -10.1, -10.0, -0.1, 0.5),),
        converged_electronic=True,
        converged_ionic=False,
    )

    assessments = assess_convergence_progress((trajectory,))

    assert assessment_by_scope(assessments, "electronic").label == CONVERGED
    ionic = assessment_by_scope(assessments, "ionic")
    assert ionic.label == INSUFFICIENT_EVIDENCE
    assert "final maximum force has not reached EDIFFG" in " ".join(ionic.counter_evidence)
    assert "trend-based ionic progress assessment is not implemented" in " ".join(
        ionic.limitations
    )
    stage = assessment_by_scope(assessments, "stage")
    assert stage.label == INSUFFICIENT_EVIDENCE
    assert "ionic progress evidence is insufficient" in " ".join(stage.limitations)


def test_convergence_progress_assessment_preserves_contradictory_evidence() -> None:
    trajectory = trajectory_observation(
        stage_type="relax",
        criteria={"NELM": 60, "EDIFF": 1e-6, "NSW": 20, "EDIFFG": -0.01},
        completed_ionic_steps=2,
        electronic_iterations=(8, 8),
        electronic_cycles=(
            electronic_cycle(1, 8, dE=1e-7),
            electronic_cycle(2, 8, dE=1e-7),
        ),
        ionic_steps=(IonicStepObservation(2, 8, -10.1, -10.0, -0.1, 0.005),),
        converged_electronic=True,
        converged_ionic=False,
    )

    ionic = assessment_by_scope(
        assess_convergence_progress((trajectory,)),
        "ionic",
    )

    assert ionic.label == INSUFFICIENT_EVIDENCE
    assert ionic.sufficiency == "contradictory"
    assert "final maximum force reached EDIFFG" in " ".join(ionic.basis)
    assert "converged_ionic=False" in " ".join(ionic.counter_evidence)


def test_cli_prints_convergence_progress_assessment_without_prediction(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trajectory = trajectory_observation(
        stage_type="static",
        criteria={"NELM": 60, "EDIFF": 1e-6, "NSW": 0},
        completed_ionic_steps=1,
        electronic_iterations=(8,),
        electronic_cycles=(electronic_cycle(1, 8, dE=1e-7),),
        converged_electronic=True,
    )
    diagnosis = RunDiagnosis(
        inspection=cli_run_inspection(
            workflow_stages=(WorkflowStage(1, "static", "pbe", (), None),),
            executed_inputs=(),
        ),
        termination=TerminationObservation(),
        trajectories=(trajectory,),
        assessments=assess_convergence_progress((trajectory,)),
    )

    cli.print_run_diagnosis(diagnosis)

    captured = capsys.readouterr()
    assert "Convergence-progress assessment (convergence_progress_assessment)" in captured.out
    assert "based on observed trajectory evidence, not a prediction" in captured.out
    assert "stage 1 (result_dir) electronic: CONVERGED" in captured.out
    assert "will converge" not in captured.out.lower()
    assert "more walltime" not in captured.out.lower()


def test_diagnose_scheduler_timeout_is_evidence_not_prediction() -> None:
    files = default_files(include_vasprun=False)
    remote = RemoteFixture(files=files, directories=default_directories())

    def timeout_slurm_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "20893681|validation|TIMEOUT|72:00:00|2026-08-21T12:14:10|"
                "2026-08-24T12:14:10|leeburton-pool|0:0|72:00:00\n"
            ),
            stderr="",
        )

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=timeout_slurm_runner,
    )

    assert diagnosis.termination.scheduler_state == "TIMEOUT"
    assert diagnosis.termination.scheduler_reports_timeout is True


def test_diagnose_rejects_producer_paths_outside_allowed_roots() -> None:
    files = {
        f"{FLOW_ROOT}/submission.json": json.dumps(
            submission_payload(outside_path=True)
        ).encode("utf-8")
    }
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    with pytest.raises(RemotePathError, match="paths.stage_dirs.producer-beta"):
        diagnose_remote_run(
            cluster(),
            FLOW_ROOT,
            remote_runner=remote,
            slurm_runner=slurm_runner,
        )


def test_producer_supplied_paths_outside_allowed_roots_are_rejected() -> None:
    files = {
        f"{FLOW_ROOT}/submission.json": json.dumps(
            submission_payload(outside_path=True)
        ).encode("utf-8")
    }
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    with pytest.raises(RemotePathError, match="paths.stage_dirs.producer-beta"):
        inspect_remote_run(
            cluster(),
            FLOW_ROOT,
            remote_runner=remote,
            slurm_runner=slurm_runner,
            scientific_parser=fake_scientific_parser,
        )

    assert len(remote.commands) == 1


def test_missing_artifacts_are_reported_not_invented() -> None:
    remote = RemoteFixture(
        files=default_files(include_vasprun=False),
        directories=default_directories(),
    )

    def parser_reports_missing_vasprun(
        local_paths: dict[str, Path],
        display_paths: dict[str, str],
        workflow_spec: dict,
    ) -> ScientificResult:
        assert set(local_paths) == {"contcar", "kpoints"}
        return ScientificResult(
            source_paths=tuple(display_paths.values()),
            final_formula="Example2",
            unavailable=("vasprun.xml is unavailable",),
        )

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=slurm_runner,
        scientific_parser=parser_reports_missing_vasprun,
    )

    artifacts = {item.label: item for item in inspection.final_artifacts}
    assert artifacts["vasprun"].present is False
    assert inspection.scientific.final_formula == "Example2"
    assert "vasprun.xml is unavailable" in inspection.scientific.unavailable


def test_invalid_job_id_is_not_sent_to_scheduler() -> None:
    remote = RemoteFixture(
        files=default_files(invalid_job_id=True),
        directories=default_directories(),
    )
    slurm_calls = 0

    def unused_slurm_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal slurm_calls
        slurm_calls += 1
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=unused_slurm_runner,
        scientific_parser=fake_scientific_parser,
    )

    assert slurm_calls == 0
    assert inspection.job_id is None
    assert inspection.scheduler_error == "No valid SLURM job ID was found in inspected producer artifacts."


def test_run_inspector_source_has_no_validation_run_specific_logic() -> None:
    source = Path("src/bmd_agent/resources/run.py").read_text(encoding="utf-8").lower()

    assert "hse06" not in source
    assert "sns2" not in source
    assert "dftd3" not in source
    assert "stage_03" not in source


class FakeComposition:
    formula = "X2"
    reduced_formula = "X2"


class FakeLattice:
    a = 3.0
    b = 4.0
    c = 6.0
    alpha = 90.0
    beta = 91.0
    gamma = 120.0


class FakeStructure:
    composition = FakeComposition()
    lattice = FakeLattice()
    volume = 62.5
    density = 4.2

    def __len__(self) -> int:
        return 2


class FakeBandStructure:
    kpoints = [object(), object(), object(), object()]
    bands = {
        "up": [
            [-2.0, -1.9, -1.8, -1.7],
            [0.4, 0.5, 0.6, 0.7],
        ]
    }

    def get_band_gap(self) -> dict[str, float]:
        return {"energy": 1.25}


class FakeBandVasprun:
    calls: list[dict[str, object]] = []
    band_calls: list[dict[str, object]] = []

    def __init__(self, path: str, **kwargs: object) -> None:
        self.calls.append({"path": path, "kwargs": kwargs})
        self.final_structure = FakeStructure()
        self.final_energy = -4.0
        self.converged_electronic = True
        self.converged = True
        self.incar = {"IVDW": 11}
        self.parameters = {"ENCUT": 520}
        self.efermi = None if kwargs.get("parse_dos") is False else 0.3

    def get_band_structure(self, **kwargs: object) -> FakeBandStructure:
        self.band_calls.append(kwargs)
        if self.efermi is None:
            raise ValueError("e_fermi is None.")
        return FakeBandStructure()


class FakeFailingBandVasprun(FakeBandVasprun):
    def get_band_structure(self, **kwargs: object) -> FakeBandStructure:
        self.band_calls.append(kwargs)
        raise ValueError("e_fermi is None.")


class FakeMalformedVasprun(FakeBandVasprun):
    def __init__(self, path: str, **kwargs: object) -> None:
        warnings.warn(
            "XML is malformed. Parsing has stopped but partial data is available.",
            UserWarning,
        )
        raise IndexError("list index out of range")


class FakeUnexpectedWarningVasprun(FakeBandVasprun):
    def __init__(self, path: str, **kwargs: object) -> None:
        warnings.warn("unexpected parser diagnostic", RuntimeWarning)
        super().__init__(path, **kwargs)


@contextmanager
def fake_pymatgen_modules(vasprun_cls: type[FakeBandVasprun]):
    modules = {
        "pymatgen": types.ModuleType("pymatgen"),
        "pymatgen.core": types.ModuleType("pymatgen.core"),
        "pymatgen.io": types.ModuleType("pymatgen.io"),
        "pymatgen.io.vasp": types.ModuleType("pymatgen.io.vasp"),
        "pymatgen.io.vasp.outputs": types.ModuleType("pymatgen.io.vasp.outputs"),
    }
    modules["pymatgen.core"].Structure = types.SimpleNamespace(
        from_file=lambda path: FakeStructure(),
    )
    modules["pymatgen.io.vasp.outputs"].Vasprun = vasprun_cls

    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def fake_vasp_paths(tmp_path: Path) -> tuple[dict[str, Path], dict[str, str]]:
    local_paths = {
        "contcar": tmp_path / "CONTCAR",
        "vasprun": tmp_path / "vasprun.xml",
        "kpoints": tmp_path / "KPOINTS",
    }
    display_paths = {
        "contcar": "/remote/result/CONTCAR",
        "vasprun": "/remote/result/vasprun.xml",
        "kpoints": "/remote/result/KPOINTS",
    }
    return local_paths, display_paths


def arbitrary_band_workflow_spec() -> dict[str, object]:
    return {
        "stages": [
            {"stage_type": "prepare", "theory": "example"},
            {"stage_type": "screen", "theory": "example"},
            {"stage_type": "static", "theory": "example"},
            {"stage_type": "checkpoint", "theory": "example"},
            {"stage_type": "band_structure", "theory": "example"},
        ]
    }


def test_band_parsing_keeps_final_vasprun_fermi_reference(tmp_path: Path) -> None:
    FakeBandVasprun.calls = []
    FakeBandVasprun.band_calls = []
    local_paths, display_paths = fake_vasp_paths(tmp_path)

    with fake_pymatgen_modules(FakeBandVasprun):
        result = parse_vasp_output_files(
            local_paths,
            display_paths,
            arbitrary_band_workflow_spec(),
        )

    assert FakeBandVasprun.calls
    vasprun_kwargs = FakeBandVasprun.calls[-1]["kwargs"]
    assert "parse_dos" not in vasprun_kwargs
    assert "parse_eigenvalues" not in vasprun_kwargs
    assert FakeBandVasprun.band_calls[-1]["line_mode"] is True
    assert FakeBandVasprun.band_calls[-1]["kpoints_filename"].endswith("KPOINTS")
    assert result.error is None
    assert result.final_formula == "X2"
    assert result.final_energy_ev == -4.0
    assert result.energy_per_atom_ev == -2.0
    assert result.electronic_convergence is True
    assert result.band_gap_ev == 1.25
    assert result.band_kpoints == 4
    assert result.bands == 2
    assert result.structure is not None
    assert result.structure.formula == "X2"
    assert result.structure.reduced_formula == "X2"
    assert result.structure.site_count == 2
    assert result.structure.lattice_a == 3.0
    assert result.structure.lattice_b == 4.0
    assert result.structure.lattice_c == 6.0
    assert result.structure.alpha == 90.0
    assert result.structure.beta == 91.0
    assert result.structure.gamma == 120.0
    assert result.structure.volume == 62.5
    assert result.structure.density == 4.2
    assert result.structure.c_over_a == 2.0
    assert result.executed_parameters[0].source_type == "vasprun_xml.incar"
    assert result.executed_parameters[0].stage_index == 5
    assert result.executed_parameters[0].values["IVDW"] == 11
    assert result.executed_parameters[1].source_type == "vasprun_xml.parameters"
    assert result.executed_parameters[1].values["ENCUT"] == 520


def test_tolerant_vasprun_boundary_does_not_suppress_unexpected_warnings(
    tmp_path: Path,
) -> None:
    local_paths, display_paths = fake_vasp_paths(tmp_path)

    with (
        fake_pymatgen_modules(FakeUnexpectedWarningVasprun),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        result = parse_vasp_output_files(
            local_paths,
            display_paths,
            arbitrary_band_workflow_spec(),
        )

    assert result.final_formula == "X2"
    assert [str(item.message) for item in caught] == ["unexpected parser diagnostic"]
    assert result.error is None


def test_scalar_observations_survive_band_structure_failure(tmp_path: Path) -> None:
    FakeFailingBandVasprun.calls = []
    FakeFailingBandVasprun.band_calls = []
    local_paths, display_paths = fake_vasp_paths(tmp_path)

    with fake_pymatgen_modules(FakeFailingBandVasprun):
        result = parse_vasp_output_files(
            local_paths,
            display_paths,
            arbitrary_band_workflow_spec(),
        )

    assert result.error is None
    assert result.final_formula == "X2"
    assert result.final_energy_ev == -4.0
    assert result.energy_per_atom_ev == -2.0
    assert result.electronic_convergence is True
    assert result.band_gap_ev is None
    assert result.band_kpoints is None
    assert result.bands is None
    assert "band structure could not be derived: e_fermi is None." in result.unavailable


def modifier_policy() -> dict:
    return {
        "modifier": "dispersion",
        "option_key": "dispersion",
        "method_key": "method",
        "methods": [
            {"value": "dftd3", "label": "DFT-D3", "incar_effect": {"IVDW": 11}},
            {"value": "dftd3-bj", "label": "DFT-D3(BJ)", "incar_effect": {"IVDW": 12}},
        ],
    }


def test_parse_incar_observes_absent_and_selected_ivdw_values() -> None:
    no_dispersion, error = parse_incar_contents("ENCUT = 520\n")
    assert error is None
    assert "IVDW" not in no_dispersion

    d3, error = parse_incar_contents("IVDW = 11\n")
    assert error is None
    assert d3["IVDW"] == 11

    d3bj, error = parse_incar_contents("IVDW = 12\n")
    assert error is None
    assert d3bj["IVDW"] == 12


def test_requested_options_are_compared_to_executed_input_by_policy() -> None:
    stages = (
        WorkflowStage(
            1,
            "relax",
            "pbe",
            ("dispersion",),
            None,
            {"dispersion": {"method": "dftd3"}},
        ),
    )
    executed = (
        IncarObservation(
            "producer-alpha",
            f"{FLOW_ROOT}/producer-alpha/INCAR",
            True,
            1,
            values={"IVDW": 12},
        ),
    )

    observations = compare_requested_options_to_executed_inputs(
        stages,
        executed,
        (modifier_policy(),),
    )

    assert observations == (
        InputExpectationObservation(
            stage_label="producer-alpha",
            stage_index=1,
            option_path="dispersion.method",
            requested_value="dftd3",
            input_key="IVDW",
            expected_value=11,
            observed_value=12,
            status="discrepancy",
            source_values={"retained_incar:producer-alpha": 12},
            reason="executed value differs from producer-requested option effect",
        ),
    )


def test_requested_check_statuses_distinguish_unavailable_absent_supported_and_disagreement() -> None:
    stages = (
        WorkflowStage(
            1,
            "relax",
            "pbe",
            ("dispersion",),
            None,
            {"dispersion": {"method": "dftd3"}},
        ),
    )

    unavailable = compare_requested_options_to_executed_inputs(
        stages,
        (
            IncarObservation(
                "result_dir",
                f"{FLOW_ROOT}/INCAR",
                False,
                1,
            ),
        ),
        (modifier_policy(),),
    )[0]
    assert unavailable.status == "unavailable"
    assert unavailable.reason == "no readable executed-input evidence was bound to this stage"

    absent = compare_requested_options_to_executed_inputs(
        stages,
        (
            IncarObservation(
                "result_dir",
                f"{FLOW_ROOT}/INCAR",
                True,
                1,
                values={"ENCUT": 520},
            ),
        ),
        (modifier_policy(),),
    )[0]
    assert absent.status == "absent"
    assert absent.source_values == {"retained_incar:result_dir": None}

    supported = compare_requested_options_to_executed_inputs(
        stages,
        (
            IncarObservation(
                "result_dir",
                f"{FLOW_ROOT}/INCAR",
                False,
                1,
            ),
            IncarObservation(
                "vasprun_xml.incar",
                f"{FLOW_ROOT}/vasprun.xml",
                True,
                1,
                source_type="vasprun_xml.incar",
                values={"IVDW": 11},
            ),
        ),
        (modifier_policy(),),
    )[0]
    assert supported.status == "supported"
    assert supported.observed_value == 11
    assert supported.source_values == {"vasprun_xml.incar:vasprun_xml.incar": 11}

    disagreement = compare_requested_options_to_executed_inputs(
        stages,
        (
            IncarObservation(
                "result_dir",
                f"{FLOW_ROOT}/INCAR",
                True,
                1,
                values={"IVDW": 11},
            ),
            IncarObservation(
                "vasprun_xml.incar",
                f"{FLOW_ROOT}/vasprun.xml",
                True,
                1,
                source_type="vasprun_xml.incar",
                values={"IVDW": 12},
            ),
        ),
        (modifier_policy(),),
    )[0]
    assert disagreement.status == "discrepancy"
    assert disagreement.source_values == {
        "retained_incar:result_dir": 11,
        "vasprun_xml.incar:vasprun_xml.incar": 12,
    }
    assert disagreement.reason == "executed-input sources disagree about parameter value"


def test_vasp_reported_parameter_observations_are_generic() -> None:
    vasprun = types.SimpleNamespace(
        incar={"IVDW": 11},
        parameters={"ENCUT": 520},
    )

    observations = vasp_reported_parameter_observations(
        vasprun,
        source_path="/remote/result/vasprun.xml",
        stage_index=1,
    )

    assert [observation.source_type for observation in observations] == [
        "vasprun_xml.incar",
        "vasprun_xml.parameters",
    ]
    assert observations[0].values["IVDW"] == 11
    assert observations[1].values["ENCUT"] == 520


def comparison_inspection(
    flow_root: str,
    *,
    a: float | None,
    b: float | None,
    c: float | None,
    volume: float | None,
    density: float | None,
    energy_per_atom: float | None,
    band_gap: float | None,
    modifiers: tuple[str, ...] = (),
    options: dict | None = None,
) -> RunInspection:
    structure = None
    if a is not None:
        structure = StructureObservation(
            source_path=f"{flow_root}/result/CONTCAR",
            formula="X2",
            reduced_formula="X2",
            site_count=2,
            lattice_a=a,
            lattice_b=b,
            lattice_c=c,
            alpha=90.0,
            beta=90.0,
            gamma=120.0,
            volume=volume,
            density=density,
            c_over_a=c / a if a and c is not None else None,
        )

    return RunInspection(
        flow_root=flow_root,
        submission_path=f"{flow_root}/submission.json",
        workflow_stages=(
            WorkflowStage(1, "relax", "pbe", modifiers, None, options or {}),
        ),
        stage_directories=(),
        result_directory=PathObservation("result_dir", f"{flow_root}/result", "directory", True, ARTIFACT_OBSERVATION),
        log_paths=(),
        final_artifacts=(),
        producer_git={"git_commit": "abcdef012345", "state": "clean"},
        cluster_request={},
        resources_request={},
        environment_policy={},
        attempt_state=AttemptStateObservation(path=None, present=False),
        job_id="1",
        scheduler=SlurmAccountingRecord(
            job_id="1",
            name="comparison",
            state="COMPLETED",
            elapsed="00:01:00",
            start="2026-08-25T00:00:00",
            end="2026-08-25T00:01:00",
            partition="leeburton-pool",
            exit_code="0:0",
        ),
        scheduler_error=None,
        runtime=LogRuntimeObservation(LOG_OBSERVATION, (), None, {}, {}, {}),
        scientific=ScientificResult(
            source_paths=(),
            final_formula="X2",
            energy_per_atom_ev=energy_per_atom,
            band_gap_ev=band_gap,
            structure=structure,
        ),
        comparison=ComparisonObservation("unavailable", "producer_provenance"),
        initial_structure=InitialStructureObservation(
            status="available",
            representation_type="pasted_text:poscar",
            representation_hash="same-hash",
        ),
    )


def test_run_labels_are_derived_from_provenance_and_modifier_policy() -> None:
    control = comparison_inspection("/flow/control", a=3, b=3, c=6, volume=54, density=4, energy_per_atom=-1, band_gap=1)
    corrected = comparison_inspection(
        "/flow/corrected",
        a=3,
        b=3,
        c=6,
        volume=54,
        density=4,
        energy_per_atom=-1,
        band_gap=1,
        modifiers=("dispersion",),
        options={"dispersion": {"method": "dftd3-bj"}},
    )

    assert run_label_from_provenance(control, modifier_policies=(modifier_policy(),)) == "PBE"
    assert run_label_from_provenance(corrected, modifier_policies=(modifier_policy(),)) == "PBE + DFT-D3(BJ)"


def test_build_run_comparison_uses_first_run_as_baseline_and_handles_three_runs() -> None:
    baseline = comparison_inspection("/flow/baseline", a=3.0, b=4.0, c=6.0, volume=72.0, density=4.0, energy_per_atom=-4.0, band_gap=2.0)
    second = comparison_inspection("/flow/second", a=3.0, b=4.1, c=5.4, volume=66.42, density=4.2, energy_per_atom=-4.2, band_gap=1.8)
    third = comparison_inspection("/flow/third", a=2.9, b=4.0, c=5.1, volume=59.16, density=4.4, energy_per_atom=-4.5, band_gap=1.5)

    comparison = build_run_comparison(
        (baseline, second, third),
        modifier_policies=(modifier_policy(),),
    )

    assert comparison.initial_structure.status == "match"
    assert len({quantity.flow_root for quantity in comparison.quantities}) == 2
    c_quantity = next(
        quantity
        for quantity in comparison.quantities
        if quantity.flow_root == "/flow/second" and quantity.quantity == "lattice_c"
    )
    assert c_quantity.baseline_value == 6.0
    assert c_quantity.comparison_value == 5.4
    assert c_quantity.delta == -0.6
    assert c_quantity.percent_delta == -10.0


def test_comparison_reports_missing_values_and_energy_warning_for_different_configs() -> None:
    baseline = comparison_inspection("/flow/baseline", a=3.0, b=4.0, c=6.0, volume=72.0, density=4.0, energy_per_atom=-4.0, band_gap=2.0)
    comparison_run = comparison_inspection(
        "/flow/missing",
        a=None,
        b=None,
        c=None,
        volume=None,
        density=None,
        energy_per_atom=-4.2,
        band_gap=None,
        modifiers=("dispersion",),
        options={"dispersion": {"method": "dftd3"}},
    )

    comparison = build_run_comparison(
        (baseline, comparison_run),
        modifier_policies=(modifier_policy(),),
    )

    lattice_a = next(quantity for quantity in comparison.quantities if quantity.quantity == "lattice_a")
    band_gap = next(quantity for quantity in comparison.quantities if quantity.quantity == "band_gap_ev")
    assert lattice_a.status == "unavailable"
    assert lattice_a.reason == "comparison value unavailable"
    assert band_gap.status == "unavailable"
    assert comparison.energy_warning is not None
    assert "not a ranking of method quality" in comparison.energy_warning


def test_compare_remote_runs_requires_at_least_two_roots() -> None:
    with pytest.raises(RunInspectionError, match="at least two"):
        compare_remote_runs(cluster(), [FLOW_ROOT])


def cli_run_inspection(
    *,
    workflow_stages: tuple[WorkflowStage, ...],
    executed_inputs: tuple[IncarObservation, ...],
    input_expectations: tuple[InputExpectationObservation, ...] = (),
) -> RunInspection:
    return RunInspection(
        flow_root=FLOW_ROOT,
        submission_path=f"{FLOW_ROOT}/submission.json",
        workflow_stages=workflow_stages,
        stage_directories=(),
        result_directory=PathObservation("result_dir", RESULT_DIR, "directory", True, ARTIFACT_OBSERVATION),
        log_paths=(),
        final_artifacts=(),
        producer_git={},
        cluster_request={},
        resources_request={},
        environment_policy={},
        attempt_state=AttemptStateObservation(path=None, present=False),
        job_id=None,
        scheduler=None,
        scheduler_error=None,
        runtime=LogRuntimeObservation(LOG_OBSERVATION, (), None, {}, {}, {}),
        scientific=ScientificResult(source_paths=()),
        comparison=ComparisonObservation("unavailable", "producer_provenance"),
        executed_inputs=executed_inputs,
        input_expectations=input_expectations,
    )


def test_cli_executed_input_summary_reports_present_settings_without_optional_absences(
    capsys: pytest.CaptureFixture[str],
) -> None:
    magmom = [1.0, 0.0, 0.0] * 30
    inspection = cli_run_inspection(
        workflow_stages=(
            WorkflowStage(1, "static", "pbe", ("hubbard",), None),
            WorkflowStage(2, "static", "pbe", ("hubbard", "soc"), None),
        ),
        executed_inputs=(
            IncarObservation(
                "stage_01",
                f"{FLOW_ROOT}/stage_01/INCAR",
                True,
                1,
                values={
                    "ENCUT": 520,
                    "LDAU": True,
                    "LDAUTYPE": 2,
                    "LDAUL": [2, -1],
                    "LDAUU": [5.3, 0],
                    "LMAXMIX": 4,
                },
            ),
            IncarObservation(
                "stage_02",
                f"{FLOW_ROOT}/stage_02/INCAR",
                True,
                2,
                values={
                    "ENCUT": 520,
                    "LDAU": True,
                    "LDAUU": [5.3, 0],
                    "LSORBIT": True,
                    "LNONCOLLINEAR": True,
                    "SAXIS": [0, 0, 1],
                    "MAGMOM": magmom,
                },
            ),
            IncarObservation(
                "vasprun_xml.incar",
                f"{FLOW_ROOT}/stage_02/vasprun.xml",
                True,
                2,
                source_type="vasprun_xml.incar",
                values={
                    "ENCUT": 520,
                    "LSORBIT": True,
                    "LNONCOLLINEAR": True,
                    "SAXIS": [0, 0, 1],
                    "MAGMOM": magmom,
                },
            ),
        ),
    )

    cli.print_run_inspection(inspection)

    captured = capsys.readouterr()
    assert "stage 1: PBE Static Energy (stage_01)" in captured.out
    assert "stage 2: PBE Static Energy (stage_02)" in captured.out
    assert "LDAU=true" in captured.out
    assert "LDAUU=[5.3, 0]" in captured.out
    assert "LSORBIT=true" in captured.out
    assert "SAXIS=[0, 0, 1]" in captured.out
    assert "MAGMOM=present, 30 sites / 90 noncollinear components" in captured.out
    assert captured.out.count("LSORBIT=true") == 1
    assert "IVDW absent" not in captured.out


def test_cli_executed_input_summary_reports_source_disagreement(
    capsys: pytest.CaptureFixture[str],
) -> None:
    inspection = cli_run_inspection(
        workflow_stages=(WorkflowStage(1, "static", "pbe", (), None),),
        executed_inputs=(
            IncarObservation(
                "stage_01",
                f"{FLOW_ROOT}/stage_01/INCAR",
                True,
                1,
                values={"ENCUT": 520},
            ),
            IncarObservation(
                "vasprun_xml.incar",
                f"{FLOW_ROOT}/stage_01/vasprun.xml",
                True,
                1,
                source_type="vasprun_xml.incar",
                values={"ENCUT": 400},
            ),
        ),
    )

    cli.print_run_inspection(inspection)

    captured = capsys.readouterr()
    assert "discrepancy: ENCUT" in captured.out
    assert "retained_incar:stage_01=520" in captured.out
    assert "vasprun_xml.incar:vasprun_xml.incar=400" in captured.out


def test_cli_inspect_run_summary_is_evidence_oriented(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = ResourceRegistry(repositories={}, clusters={"powerslurm": cluster()})
    inspection = RunInspection(
        flow_root=FLOW_ROOT,
        submission_path=f"{FLOW_ROOT}/submission.json",
        workflow_stages=(
            WorkflowStage(1, "relax", "pbe", (), None),
            WorkflowStage(2, "static", "r2scan", (), None),
        ),
        stage_directories=(
            PathObservation("producer-alpha", f"{FLOW_ROOT}/producer-alpha", "directory", True, ARTIFACT_OBSERVATION),
        ),
        result_directory=PathObservation("result_dir", RESULT_DIR, "directory", True, ARTIFACT_OBSERVATION),
        log_paths=(
            PathObservation("log_out", f"{LOG_ROOT}/validation-run.out", "file", True, ARTIFACT_OBSERVATION),
        ),
        final_artifacts=(
            PathObservation("vasprun", f"{RESULT_DIR}/vasprun.xml", "file", False, ARTIFACT_OBSERVATION),
        ),
        producer_git={"git_commit": "abcdef012345", "state": "clean"},
        cluster_request={"partition": "leeburton-pool"},
        resources_request={"ntasks": 24},
        environment_policy={"PMG_VASP_PSP_DIR": "/bmd-db/potcars"},
        attempt_state=AttemptStateObservation(path=None, present=False),
        job_id="20893681",
        scheduler=SlurmAccountingRecord(
            job_id="20893681",
            name="validation",
            state="COMPLETED",
            elapsed="02:48:37",
            start="2026-08-21T12:14:10",
            end="2026-08-21T15:02:47",
            partition="leeburton-pool",
            exit_code="0:0",
        ),
        scheduler_error=None,
        runtime=LogRuntimeObservation(
            evidence_type=LOG_OBSERVATION,
            sources=(f"{LOG_ROOT}/validation-run.out",),
            python="3.12.13",
            packages={"pymatgen": "2026.8.13"},
            environment={},
            stage_uuids={},
        ),
        scientific=ScientificResult(
            source_paths=(f"{RESULT_DIR}/vasprun.xml",),
            unavailable=("vasprun.xml is unavailable",),
        ),
        comparison=ComparisonObservation(
            status="unavailable",
            evidence_type="producer_provenance",
            reason="No durable producer result.",
        ),
    )

    monkeypatch.setattr(cli, "load_resources", lambda: registry)
    monkeypatch.setattr(cli, "inspect_remote_run", lambda cluster, flow_root, **kwargs: inspection)

    exit_code = cli.main(["inspect-run", FLOW_ROOT])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Producer provenance (producer_provenance)" in captured.out
    assert "Scheduler (scheduler_observation)" in captured.out
    assert "Logs (log_observation)" in captured.out
    assert "Evidence paths (artifact_observation)" in captured.out
    assert "Independent parsing (pymatgen_derived)" in captured.out


def test_cli_diagnose_run_summary_is_descriptive_not_predictive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    inspection = cli_run_inspection(
        workflow_stages=(WorkflowStage(1, "relax", "pbe", (), None),),
        executed_inputs=(),
    )
    diagnosis = RunDiagnosis(
        inspection=inspection,
        termination=TerminationObservation(
            scheduler_state="TIMEOUT",
            scheduler_exit_code="0:0",
            scheduler_elapsed="72:00:00",
            scheduler_timelimit="72:00:00",
            scheduler_reports_timeout=True,
            unavailable=("VASP normal-completion marker is unavailable in diagnose-run v1",),
        ),
        trajectories=(
            StageTrajectoryObservation(
                stage_index=1,
                stage_label="result_dir",
                stage_type="relax",
                theory="pbe",
                directory=FLOW_ROOT,
                oszicar_path=f"{FLOW_ROOT}/OSZICAR",
                oszicar_present=True,
                vasprun_path=f"{FLOW_ROOT}/vasprun.xml",
                vasprun_present=True,
                criteria={"NELM": 60, "EDIFFG": -0.01},
                ionic_steps_observed=2,
                electronic_iterations_by_ionic_step=(4, 60),
                final_electronic_iteration_count=60,
                recent_electronic_iterations=(
                    run_resource.ElectronicIterationObservation(
                        iteration=60,
                        algorithm="RMM",
                        energy=-12.1,
                        dE=-0.001,
                        deps=-0.0001,
                        rms=0.02,
                        rms_c=0.01,
                    ),
                ),
                recent_ionic_steps=(
                    run_resource.IonicStepObservation(
                        step_index=2,
                        electronic_iterations=60,
                        free_energy=-12.1,
                        energy_zero=-12.05,
                        dE=-0.1,
                        max_force=0.5,
                    ),
                ),
                vasprun_ionic_steps=2,
                converged_electronic=False,
                converged_ionic=False,
            ),
        ),
    )

    cli.print_run_diagnosis(diagnosis)

    captured = capsys.readouterr()
    assert "Termination evidence (termination_observation)" in captured.out
    assert "scheduler reports timeout: True" in captured.out
    assert "Trajectory evidence (trajectory_observation)" in captured.out
    assert "EDIFFG: -0.01 eV/A force criterion" in captured.out
    assert "RMM N=60 E=-12.1 dE=-0.001 deps=-0.0001 rms=0.02 rms(c)=0.01" in captured.out
    assert "step 2 electronic_iterations=60 F=-12.1 E0=-12.05 dE=-0.1 max_force=0.5" in captured.out
    forbidden = ("likely to benefit", "more walltime alone", "stalled", "oscillating", "diverging")
    assert all(term not in captured.out.lower() for term in forbidden)


def test_cli_prints_compact_outcar_force_evidence_without_full_history(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ionic_steps = tuple(
        IonicStepObservation(
            index,
            8,
            -10.0 - index,
            -9.9 - index,
            -0.1,
            round(0.1 * index, 1),
            "OUTCAR",
        )
        for index in range(1, 8)
    )
    trajectory = trajectory_observation(
        stage_type="relax",
        criteria={"NELM": 60, "EDIFF": 1e-6, "NSW": 20, "EDIFFG": -0.01, "ISIF": 2},
        completed_ionic_steps=7,
        electronic_iterations=(8,) * 7,
        electronic_cycles=tuple(
            electronic_cycle(index, 8, dE=1e-7)
            for index in range(1, 8)
        ),
        ionic_steps=ionic_steps,
        converged_electronic=True,
        converged_ionic=False,
    )
    trajectory = replace(
        trajectory,
        outcar_path=f"{FLOW_ROOT}/OUTCAR",
        outcar_present=True,
        outcar_expected_site_count=24,
        outcar_force_blocks=tuple(
            OutcarForceBlockObservation(
                index,
                24,
                "complete",
                True,
                f"{FLOW_ROOT}/OUTCAR",
                round(0.1 * index, 1),
            )
            for index in range(1, 8)
        ),
        outcar_complete_force_blocks=7,
        outcar_force_alignment_status="aligned",
        outcar_force_alignment_reason=(
            "7 OUTCAR force block(s) aligned with OSZICAR completed ionic steps"
        ),
    )

    cli._print_trajectory_observations((trajectory,))

    captured = capsys.readouterr()
    assert "OUTCAR atomic forces: present" in captured.out
    assert "OUTCAR complete force blocks: 7" in captured.out
    assert "OUTCAR expected site count: 24" in captured.out
    assert "ISIF: 2" in captured.out
    assert "atomic-force trajectory summary (trajectory_progress_evidence):" in captured.out
    assert "criterion magnitude: 0.010000 eV/A" in captured.out
    assert "initial/current/best: 0.100000 / 0.700000 / 0.100000 eV/A" in captured.out
    assert "best observed at ionic step: 1" in captured.out
    assert "electronic iterations per completed ionic step: min 8, median 8, max 8" in captured.out
    assert "step 3" in captured.out
    assert "step 7" in captured.out
    assert "max_force=0.7 [OUTCAR]" in captured.out
    assert "step 1 electronic_iterations" not in captured.out
    assert "step 2 electronic_iterations" not in captured.out


def test_cli_executed_input_wording_distinguishes_unavailable_and_absent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    inspection = RunInspection(
        flow_root=FLOW_ROOT,
        submission_path=f"{FLOW_ROOT}/submission.json",
        workflow_stages=(
            WorkflowStage(1, "relax", "pbe", ("dispersion",), None, {"dispersion": {"method": "dftd3"}}),
        ),
        stage_directories=(),
        result_directory=PathObservation("result_dir", FLOW_ROOT, "directory", True, ARTIFACT_OBSERVATION),
        log_paths=(),
        final_artifacts=(),
        producer_git={},
        cluster_request={},
        resources_request={},
        environment_policy={},
        attempt_state=AttemptStateObservation(path=None, present=False),
        job_id=None,
        scheduler=None,
        scheduler_error=None,
        runtime=LogRuntimeObservation(LOG_OBSERVATION, (), None, {}, {}, {}),
        scientific=ScientificResult(source_paths=()),
        comparison=ComparisonObservation("unavailable", "producer_provenance"),
        executed_inputs=(
            IncarObservation("missing_result", f"{FLOW_ROOT}/INCAR", False, 1),
            IncarObservation("parsed_result", f"{FLOW_ROOT}/INCAR", True, 1, values={"ENCUT": 520}),
        ),
        input_expectations=(
            InputExpectationObservation(
                stage_label="missing_result",
                stage_index=1,
                option_path="dispersion.method",
                requested_value="dftd3",
                input_key="IVDW",
                expected_value=11,
                observed_value=None,
                status="unavailable",
                reason="no readable executed-input evidence was bound to this stage",
            ),
            InputExpectationObservation(
                stage_label="parsed_result",
                stage_index=1,
                option_path="dispersion.method",
                requested_value="dftd3",
                input_key="IVDW",
                expected_value=11,
                observed_value=None,
                status="absent",
                source_values={"retained_incar:parsed_result": None},
                reason="IVDW was absent from readable executed-input evidence",
            ),
        ),
    )

    cli.print_run_inspection(inspection)

    captured = capsys.readouterr()
    assert "IVDW unavailable, expected 11: unavailable" in captured.out
    assert "IVDW absent, expected 11: absent" in captured.out
    assert "observed absent" not in captured.out
    assert "present (retained_incar, /bmd-db/guest/flows/validation-run/INCAR); IVDW absent" not in captured.out


def test_cli_requested_executed_magmom_discrepancy_remains_visible(
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected_magmom = [1.0, 0.0, 0.0] * 30
    observed_magmom = [0.0, 1.0, 0.0] * 30
    inspection = cli_run_inspection(
        workflow_stages=(WorkflowStage(1, "static", "pbe", ("soc",), None),),
        executed_inputs=(
            IncarObservation(
                "stage_01",
                f"{FLOW_ROOT}/stage_01/INCAR",
                True,
                1,
                values={
                    "LSORBIT": True,
                    "LNONCOLLINEAR": True,
                    "MAGMOM": observed_magmom,
                },
            ),
        ),
        input_expectations=(
            InputExpectationObservation(
                stage_label="stage_01",
                stage_index=1,
                option_path="magnetism.magmom",
                requested_value="noncollinear",
                input_key="MAGMOM",
                expected_value=expected_magmom,
                observed_value=observed_magmom,
                status="discrepancy",
                source_values={"retained_incar:stage_01": observed_magmom},
                reason="executed value differs from producer-requested option effect",
            ),
        ),
    )

    cli.print_run_inspection(inspection)

    captured = capsys.readouterr()
    assert "MAGMOM=present, 30 sites / 90 noncollinear components" in captured.out
    assert "MAGMOM observed present, 90 sites" in captured.out
    assert "expected present, 90 sites" in captured.out
    assert "discrepancy" in captured.out
    assert "retained_incar:stage_01: present, 90 sites, fingerprint" in captured.out


def test_remote_custodian_acquisition_is_fixed_bounded_and_read_only() -> None:
    fixture = (
        Path(__file__).parent / "fixtures" / "custodian_frozen_repeated.json"
    ).read_bytes()
    commands: list[str] = []

    def runner(command, **_kwargs):
        remote_command = command[2]
        commands.append(remote_command)
        if remote_command.startswith("test -f "):
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        if remote_command.startswith("stat -c %s -- "):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=str(len(fixture)).encode("ascii"),
                stderr=b"",
            )
        if remote_command.startswith("cat -- "):
            return subprocess.CompletedProcess(command, 0, stdout=fixture, stderr=b"")
        raise AssertionError(f"unexpected remote command: {remote_command}")

    evidence = run_resource._observe_remote_custodian_evidence(
        "bmd-vm",
        {"stage_01": PurePosixPath("/bmd-db/guest/flows/run/stage_01")},
        PurePosixPath("/bmd-db/guest/flows/run/stage_01"),
        (WorkflowStage(1, "static", "hse06", ("soc",), None),),
        allowed_roots=(PurePosixPath("/bmd-db/guest/flows"),),
        runner=runner,
        timeout=20,
    )

    assert len(evidence) == 1
    assert len(evidence[0].corrections) == 5
    assert len(commands) == 3
    assert all("/bmd-db/guest/flows/run/stage_01/custodian.json" in item for item in commands)
    assert not any("find " in item or "POTCAR" in item for item in commands)


def test_remote_custodian_absence_does_not_become_an_intervention() -> None:
    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"")

    evidence = run_resource._observe_remote_custodian_evidence(
        "bmd-vm",
        {},
        PurePosixPath("/bmd-db/guest/flows/run"),
        (WorkflowStage(1, "static", "pbe", (), None),),
        allowed_roots=(PurePosixPath("/bmd-db/guest/flows"),),
        runner=runner,
        timeout=20,
    )

    assert evidence == ()


def test_runtime_log_parsing_preserves_bounded_sigterm_termination_evidence() -> None:
    path = "/bmd-db/guest/flows/run/std_err.txt"

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"routine line\nSIGTERM received by VASP\n",
            stderr=b"",
        )

    runtime = run_resource._parse_runtime_logs(
        "bmd-vm",
        (PathObservation("log_err", path, "file", True, LOG_OBSERVATION),),
        runner=runner,
        timeout=20,
    )

    assert runtime.termination_diagnostic_messages == (
        (path, "SIGTERM received by VASP"),
    )
