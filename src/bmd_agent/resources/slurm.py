from dataclasses import dataclass, field
import re
import shlex
import subprocess
from typing import Callable

from bmd_agent.config import (
    DEFAULT_SCHEDULER_ACCOUNTING_TIMEOUT_SECONDS,
    DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS,
)
from bmd_agent.resources.transport import run_ssh_command


Runner = Callable[..., subprocess.CompletedProcess[str]]

_PARTITION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_JOB_ID_RE = re.compile(r"^\d+(?:_\d+)?(?:\.(?:batch|extern|\d+))?$")
_SQUEUE_FORMAT = "%i|%u|%j|%t|%M|%R"
_SACCT_FIELDS = (
    "JobIDRaw",
    "JobName%30",
    "User%20",
    "Account%30",
    "State",
    "ExitCode",
    "Reason%40",
    "Elapsed",
    "ElapsedRaw",
    "Start",
    "End",
    "Partition%20",
    "Timelimit%20",
    "NodeList%80",
    "NNodes",
    "AllocCPUS",
    "NTasks",
    "ReqMem",
    "ReqTRES%120",
    "AllocTRES%120",
    "TotalCPU",
    "CPUTimeRAW",
    "MaxRSS",
    "MaxVMSize",
    "AveRSS",
    "StdOut%160",
    "StdErr%160",
    "WorkDir%160",
)
_SACCT_FORMAT = ",".join(_SACCT_FIELDS)


@dataclass
class SlurmJob:
    job_id: str
    user: str
    name: str
    state: str
    elapsed: str
    reason: str


@dataclass
class SlurmAccountingRecord:
    job_id: str
    name: str
    state: str
    elapsed: str
    start: str
    end: str
    partition: str
    exit_code: str
    reason: str | None = None
    timelimit: str | None = None
    user: str | None = None
    account: str | None = None
    elapsed_raw: int | None = None
    node_list: str | None = None
    node_count: int | None = None
    allocated_cpus: int | None = None
    task_count: int | None = None
    req_mem: str | None = None
    req_tres: str | None = None
    alloc_tres: str | None = None
    total_cpu: str | None = None
    cpu_time_raw: int | None = None
    max_rss: str | None = None
    max_vm_size: str | None = None
    ave_rss: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    work_dir: str | None = None
    steps: tuple["SlurmStepAccountingRecord", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SlurmStepAccountingRecord:
    job_id_raw: str
    name: str
    state: str
    exit_code: str
    reason: str | None = None
    elapsed: str | None = None
    elapsed_raw: int | None = None
    node_list: str | None = None
    node_count: int | None = None
    allocated_cpus: int | None = None
    task_count: int | None = None
    req_mem: str | None = None
    req_tres: str | None = None
    alloc_tres: str | None = None
    total_cpu: str | None = None
    cpu_time_raw: int | None = None
    max_rss: str | None = None
    max_vm_size: str | None = None
    ave_rss: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None


def get_queue(
    ssh_host: str,
    partition: str,
    *,
    runner: Runner = run_ssh_command,
    timeout: float = 20,
    ssh_connect_timeout: int = DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS,
) -> list[SlurmJob]:
    """Return jobs visible in a configured SLURM partition."""

    if ssh_connect_timeout <= 0:
        raise ValueError("SSH connection timeout must be positive")

    command = [
        "ssh",
        "-o",
        f"ConnectTimeout={ssh_connect_timeout}",
        ssh_host,
        build_squeue_command(partition),
    ]

    result = runner(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )

    return parse_squeue_output(result.stdout)


def get_job_accounting(
    ssh_host: str,
    job_id: str,
    *,
    runner: Runner = run_ssh_command,
    timeout: float = DEFAULT_SCHEDULER_ACCOUNTING_TIMEOUT_SECONDS,
    ssh_connect_timeout: int = DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS,
) -> SlurmAccountingRecord | None:
    """Return completed-job accounting visible for one SLURM job ID."""

    if ssh_connect_timeout <= 0:
        raise ValueError("SSH connection timeout must be positive")

    command = [
        "ssh",
        "-o",
        f"ConnectTimeout={ssh_connect_timeout}",
        ssh_host,
        build_sacct_command(job_id),
    ]

    result = runner(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )

    return parse_sacct_output(job_id, result.stdout)


def build_squeue_command(partition: str) -> str:
    """Build a shell-quoted read-only remote squeue command."""

    if not _PARTITION_RE.fullmatch(partition):
        raise ValueError("partition contains unsafe characters")

    return " ".join(
        [
            "squeue",
            "-p",
            shlex.quote(partition),
            "--noheader",
            shlex.quote(f"--format={_SQUEUE_FORMAT}"),
        ]
    )


def build_sacct_command(job_id: str) -> str:
    """Build a shell-quoted read-only remote sacct command."""

    normalized_job_id = normalize_job_id(job_id)

    return " ".join(
        [
            "sacct",
            "-P",
            "-n",
            "-j",
            shlex.quote(normalized_job_id),
            shlex.quote(f"--format={_SACCT_FORMAT}"),
        ]
    )


def parse_squeue_output(output: str) -> list[SlurmJob]:
    """Parse pipe-delimited squeue output."""

    jobs: list[SlurmJob] = []

    for line in output.splitlines():
        if not line.strip():
            continue

        parts = line.split("|", maxsplit=5)

        if len(parts) != 6:
            continue

        jobs.append(
            SlurmJob(
                job_id=parts[0].strip(),
                user=parts[1].strip(),
                name=parts[2].strip(),
                state=parts[3].strip(),
                elapsed=parts[4].strip(),
                reason=parts[5].strip(),
            )
        )

    return jobs


def parse_sacct_output(job_id: str, output: str) -> SlurmAccountingRecord | None:
    """Parse pipe-delimited sacct output for one requested job."""

    normalized_job_id = normalize_job_id(job_id)
    parent: SlurmAccountingRecord | None = None
    fallback: SlurmAccountingRecord | None = None
    steps: list[SlurmStepAccountingRecord] = []

    for line in output.splitlines():
        if not line.strip():
            continue

        parts = line.split("|")
        raw_job_id = parts[0].strip()
        if len(parts) < 8 or not (
            raw_job_id == normalized_job_id
            or raw_job_id.startswith(f"{normalized_job_id}.")
        ):
            continue

        record = _parse_sacct_record(parts)

        if raw_job_id == normalized_job_id:
            parent = record
            continue

        fallback = fallback or record
        if len(parts) >= len(_SACCT_FIELDS):
            steps.append(_parse_sacct_step(parts))

    selected = parent or fallback
    if selected is not None:
        selected.steps = tuple(steps)
    return selected


def _parse_sacct_record(parts: list[str]) -> SlurmAccountingRecord:
    if len(parts) >= len(_SACCT_FIELDS):
        return SlurmAccountingRecord(
            job_id=normalize_job_id(parts[0].strip()),
            name=_optional_part(parts, 1) or "",
            user=_optional_part(parts, 2),
            account=_optional_part(parts, 3),
            state=_optional_part(parts, 4) or "",
            exit_code=_optional_part(parts, 5) or "",
            reason=_optional_part(parts, 6),
            elapsed=_optional_part(parts, 7) or "",
            elapsed_raw=_optional_int(parts, 8),
            start=_optional_part(parts, 9) or "",
            end=_optional_part(parts, 10) or "",
            partition=_optional_part(parts, 11) or "",
            timelimit=_optional_part(parts, 12),
            node_list=_optional_part(parts, 13),
            node_count=_optional_int(parts, 14),
            allocated_cpus=_optional_int(parts, 15),
            task_count=_optional_int(parts, 16),
            req_mem=_optional_part(parts, 17),
            req_tres=_optional_part(parts, 18),
            alloc_tres=_optional_part(parts, 19),
            total_cpu=_optional_part(parts, 20),
            cpu_time_raw=_optional_int(parts, 21),
            max_rss=_optional_part(parts, 22),
            max_vm_size=_optional_part(parts, 23),
            ave_rss=_optional_part(parts, 24),
            stdout_path=_optional_part(parts, 25),
            stderr_path=_optional_part(parts, 26),
            work_dir=_optional_part(parts, 27),
        )

    return SlurmAccountingRecord(
        job_id=normalize_job_id(parts[0].strip()),
        name=parts[1].strip(),
        state=parts[2].strip(),
        elapsed=parts[3].strip(),
        start=parts[4].strip(),
        end=parts[5].strip(),
        partition=parts[6].strip(),
        exit_code=parts[7].strip(),
        timelimit=parts[8].strip() if len(parts) > 8 and parts[8].strip() else None,
    )


def _parse_sacct_step(parts: list[str]) -> SlurmStepAccountingRecord:
    return SlurmStepAccountingRecord(
        job_id_raw=parts[0].strip(),
        name=_optional_part(parts, 1) or "",
        state=_optional_part(parts, 4) or "",
        exit_code=_optional_part(parts, 5) or "",
        reason=_optional_part(parts, 6),
        elapsed=_optional_part(parts, 7),
        elapsed_raw=_optional_int(parts, 8),
        node_list=_optional_part(parts, 13),
        node_count=_optional_int(parts, 14),
        allocated_cpus=_optional_int(parts, 15),
        task_count=_optional_int(parts, 16),
        req_mem=_optional_part(parts, 17),
        req_tres=_optional_part(parts, 18),
        alloc_tres=_optional_part(parts, 19),
        total_cpu=_optional_part(parts, 20),
        cpu_time_raw=_optional_int(parts, 21),
        max_rss=_optional_part(parts, 22),
        max_vm_size=_optional_part(parts, 23),
        ave_rss=_optional_part(parts, 24),
        stdout_path=_optional_part(parts, 25),
        stderr_path=_optional_part(parts, 26),
    )


def _optional_part(parts: list[str], index: int) -> str | None:
    if index >= len(parts):
        return None
    value = parts[index].strip()
    return value or None


def _optional_int(parts: list[str], index: int) -> int | None:
    value = _optional_part(parts, index)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def normalize_job_id(job_id: str) -> str:
    """Validate a SLURM job ID and strip non-primary job-step suffixes."""

    candidate = str(job_id or "").strip()

    if not _JOB_ID_RE.fullmatch(candidate):
        raise ValueError("SLURM job ID contains unsafe characters")
    if int(re.split(r"[_.]", candidate, maxsplit=1)[0]) <= 0:
        raise ValueError("SLURM job ID must be a positive decimal integer")

    return re.sub(r"\.(?:batch|extern|\d+)$", "", candidate)


# SLURM states in which the job (or step) has not finished. For these states
# sacct's ExitCode field is not a termination observation: SLURM records the
# exit code when the job or step ends, and reports its initial value (0:0)
# before then.
ACTIVE_SLURM_STATES = frozenset(
    {
        "BOOT_FAIL_REQUEUE_FED",
        "CONFIGURING",
        "COMPLETING",
        "PENDING",
        "REQUEUED",
        "RESIZING",
        "RUNNING",
        "STAGE_OUT",
        "SUSPENDED",
    }
)


def describe_exit_code(state: str | None, exit_code: str | None) -> str:
    """Describe a sacct ExitCode field without treating it as an exit that occurred.

    For an active job or step the raw value is kept but labelled as the value
    SLURM currently reports, because no exit has happened yet. For an inactive
    record the raw value is returned unchanged, or ``unavailable`` if absent.
    """

    raw = str(exit_code or "").strip()
    tokens = str(state or "").strip().upper().split()
    current_state = tokens[0] if tokens else ""
    if current_state in ACTIVE_SLURM_STATES:
        reported = f"SLURM currently reports {raw}" if raw else "SLURM reports no exit code"
        return f"not yet applicable (job is {current_state}; {reported})"
    return raw or "unavailable"
