"""Regression tests: sacct ExitCode for an active job is not an observed exit.

SLURM records a job's (or step's) exit code when it ends; before then sacct
reports the initial value 0:0. Verbose evidence must keep that raw value but
must not present it as an exit that has occurred.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bmd_agent import cli
from bmd_agent.presentation import build_job_concise_summary, render_concise_summary
from bmd_agent.resources.lifecycle import analyze_calculation_directory
from bmd_agent.resources.run import serialize_job_trajectory_evidence
from bmd_agent.resources.slurm import SlurmStepAccountingRecord, describe_exit_code

from test_lifecycle import scheduler_record, write_inputs, write_submission
from test_presentation import bmd_job, scheduler, trajectory


ACTIVE_TEXT = "not yet applicable (job is RUNNING; SLURM currently reports 0:0)"


def running_job_with_steps():
    record = replace(
        scheduler(state="RUNNING", exit_code="0:0"),
        steps=(
            SlurmStepAccountingRecord(job_id_raw="22351669.batch", name="batch", state="RUNNING", exit_code="0:0"),
            SlurmStepAccountingRecord(job_id_raw="22351669.extern", name="extern", state="RUNNING", exit_code="0:0"),
            SlurmStepAccountingRecord(job_id_raw="22351669.0", name="vasp_std", state="FAILED", exit_code="1:0"),
        ),
    )
    return bmd_job(
        scheduler_record=record,
        trajectories=(trajectory(stage_type="static", incomplete_iterations=4),),
    )


@pytest.mark.parametrize(
    ("state", "exit_code", "expected"),
    [
        ("RUNNING", "0:0", ACTIVE_TEXT),
        ("running", "0:0", ACTIVE_TEXT),
        ("PENDING", "0:0", "not yet applicable (job is PENDING; SLURM currently reports 0:0)"),
        ("COMPLETING", "0:0", "not yet applicable (job is COMPLETING; SLURM currently reports 0:0)"),
        ("RUNNING", "", "not yet applicable (job is RUNNING; SLURM reports no exit code)"),
        ("COMPLETED", "0:0", "0:0"),
        ("FAILED", "1:0", "1:0"),
        ("CANCELLED by 1234", "0:15", "0:15"),
        ("COMPLETED", "", "unavailable"),
        (None, None, "unavailable"),
    ],
)
def test_exit_code_is_described_relative_to_scheduler_state(state, exit_code, expected) -> None:
    assert describe_exit_code(state, exit_code) == expected


def test_verbose_running_job_does_not_present_exit_as_observed(capsys) -> None:
    cli.print_job_inspection(running_job_with_steps())
    output = capsys.readouterr().out

    assert f"  exit: {ACTIVE_TEXT}" in output
    assert f"  scheduler exit: {ACTIVE_TEXT}" in output
    assert "22351669.batch: state=RUNNING, exit=not yet applicable (job is RUNNING; SLURM currently reports 0:0)" in output
    assert "22351669.extern: state=RUNNING, exit=not yet applicable" in output
    # A finished step keeps its observed exit code verbatim.
    assert "22351669.0: state=FAILED, exit=1:0" in output
    assert "exit: 0:0" not in output
    assert "scheduler exit: 0:0" not in output


def test_verbose_run_inspection_labels_active_exit(capsys) -> None:
    cli.print_run_inspection(running_job_with_steps().bmd_compute.inspection)
    output = capsys.readouterr().out

    assert f"  exit:      {ACTIVE_TEXT}" in output


def test_verbose_path_mode_labels_active_exit(tmp_path: Path, capsys) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(
        root,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="RUNNING", exit_code="0:0"),
    )
    cli.print_lifecycle_analysis(analysis)
    output = capsys.readouterr().out

    assert f"  exit: {ACTIVE_TEXT}" in output


def test_raw_exit_code_is_preserved_in_evidence() -> None:
    job = running_job_with_steps()

    assert job.scheduler.exit_code == "0:0"
    assert job.bmd_compute.termination.scheduler_exit_code == "0:0"
    summary = serialize_job_trajectory_evidence(job)["job"]
    assert summary["scheduler_state"] == "RUNNING"
    assert summary["exit_code"] == "0:0"


def test_completed_job_still_shows_observed_exit(capsys) -> None:
    cli.print_job_inspection(bmd_job())
    output = capsys.readouterr().out

    assert "  exit: 0:0" in output
    assert "  scheduler exit: 0:0" in output
    assert "not yet applicable" not in output


def test_concise_running_output_does_not_mention_exit_code() -> None:
    summary = build_job_concise_summary(running_job_with_steps())
    output = render_concise_summary(summary)

    assert summary.status == "RUNNING"
    assert "0:0" not in output
    assert "exit" not in output.lower()
