import subprocess

import pytest

from bmd_agent.resources.slurm import (
    DEFAULT_SCHEDULER_ACCOUNTING_TIMEOUT_SECONDS,
    DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS,
    build_sacct_command,
    build_squeue_command,
    get_job_accounting,
    get_queue,
    normalize_job_id,
    parse_sacct_output,
    parse_squeue_output,
)


SQUEUE_OUTPUT = """\
123|alice|relax_NaCl|R|00:12|None
124|bob|static_Si|PD|00:00|Priority
"""
SACCT_FORMAT = (
    "--format=JobIDRaw,JobName%30,User%20,Account%30,State,ExitCode,Reason%40,Elapsed,"
    "ElapsedRaw,Start,End,Partition%20,Timelimit%20,NodeList%80,NNodes,"
    "AllocCPUS,NTasks,ReqMem,ReqTRES%120,AllocTRES%120,TotalCPU,CPUTimeRAW,"
    "MaxRSS,MaxVMSize,AveRSS,StdOut%160,StdErr%160,WorkDir%160"
)


def test_parse_squeue_output() -> None:
    jobs = parse_squeue_output(SQUEUE_OUTPUT)

    assert len(jobs) == 2
    assert jobs[0].job_id == "123"
    assert jobs[0].user == "alice"
    assert jobs[0].state == "R"
    assert jobs[1].reason == "Priority"


def test_parse_squeue_output_ignores_malformed_lines() -> None:
    jobs = parse_squeue_output("bad line\n" + SQUEUE_OUTPUT)

    assert [job.job_id for job in jobs] == ["123", "124"]


def test_build_squeue_command_rejects_unsafe_partition() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        build_squeue_command("pool;scancel 1")


def test_get_queue_uses_mocked_ssh_transport() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 20
        return subprocess.CompletedProcess(command, 0, stdout=SQUEUE_OUTPUT, stderr="")

    jobs = get_queue("powerslurm-bmdguest", "leeburton-pool", runner=runner)

    assert [job.job_id for job in jobs] == ["123", "124"]
    assert calls == [
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS}",
            "powerslurm-bmdguest",
            "squeue -p leeburton-pool --noheader '--format=%i|%u|%j|%t|%M|%R'",
        ]
    ]


def test_job_id_validation_strips_safe_slurm_suffixes() -> None:
    assert normalize_job_id("21853598") == "21853598"
    assert normalize_job_id("20893681") == "20893681"
    assert normalize_job_id("20893681.batch") == "20893681"
    assert normalize_job_id("20893681.extern") == "20893681"
    assert normalize_job_id("20893681.0") == "20893681"

    with pytest.raises(ValueError, match="unsafe"):
        normalize_job_id("20893681;scancel 1")
    with pytest.raises(ValueError, match="unsafe"):
        normalize_job_id("20893681.0;id")
    with pytest.raises(ValueError, match="positive"):
        normalize_job_id("0")


def test_build_sacct_command_is_fixed_and_read_only() -> None:
    command = build_sacct_command("20893681")

    assert command == (
        "sacct -P -n -j 20893681 "
        f"{SACCT_FORMAT}"
    )
    assert "sbatch" not in command
    assert "scancel" not in command
    assert "scontrol" not in command


def test_parse_sacct_output_prefers_primary_job_row() -> None:
    output = (
        "20893681.batch|batch|COMPLETED|02:48:37|2026-08-21T12:14:10|"
        "2026-08-21T15:02:47|leeburton-pool|0:0|72:00:00\n"
        "20893681|run|COMPLETED|02:48:37|2026-08-21T12:14:10|"
        "2026-08-21T15:02:47|leeburton-pool|0:0|72:00:00\n"
    )

    record = parse_sacct_output("20893681.batch", output)

    assert record is not None
    assert record.job_id == "20893681"
    assert record.state == "COMPLETED"
    assert record.elapsed == "02:48:37"
    assert record.partition == "leeburton-pool"
    assert record.exit_code == "0:0"
    assert record.timelimit == "72:00:00"


def test_parse_sacct_output_preserves_workdir_and_resource_fields() -> None:
    output = (
        "20893681|vasp|guest|power-leeburton-users_v2|TIMEOUT|0:0|TimeLimit|06:00:20|21620|"
        "2026-08-30T00:00:00|2026-08-30T06:00:20|leeburton-pool|06:00:00|"
        "compute-0-269|1|24|24|128G|billing=24,cpu=24,mem=128G,node=1|"
        "billing=24,cpu=24,mem=128G,node=1|120:00:00|518880|"
        "120000M|140000M|110000M|/logs/job.out|/logs/job.err|"
        "/bmd-db/guest/flows/direct-vasp\n"
    )

    record = parse_sacct_output("20893681", output)

    assert record is not None
    assert record.job_id == "20893681"
    assert record.name == "vasp"
    assert record.user == "guest"
    assert record.account == "power-leeburton-users_v2"
    assert record.state == "TIMEOUT"
    assert record.exit_code == "0:0"
    assert record.reason == "TimeLimit"
    assert record.elapsed == "06:00:20"
    assert record.elapsed_raw == 21620
    assert record.timelimit == "06:00:00"
    assert record.node_list == "compute-0-269"
    assert record.node_count == 1
    assert record.allocated_cpus == 24
    assert record.task_count == 24
    assert record.req_mem == "128G"
    assert record.req_tres == "billing=24,cpu=24,mem=128G,node=1"
    assert record.alloc_tres == "billing=24,cpu=24,mem=128G,node=1"
    assert record.total_cpu == "120:00:00"
    assert record.cpu_time_raw == 518880
    assert record.max_rss == "120000M"
    assert record.max_vm_size == "140000M"
    assert record.ave_rss == "110000M"
    assert record.stdout_path == "/logs/job.out"
    assert record.stderr_path == "/logs/job.err"
    assert record.work_dir == "/bmd-db/guest/flows/direct-vasp"


def test_parse_sacct_output_allows_missing_optional_job_fields() -> None:
    output = (
        "20893681|vasp|||COMPLETED|0:0||00:10:00||2026-08-30T00:00:00|"
        "2026-08-30T00:10:00|leeburton-pool||||||||||||||||"
    )

    record = parse_sacct_output("20893681", output)

    assert record is not None
    assert record.state == "COMPLETED"
    assert record.work_dir is None
    assert record.allocated_cpus is None
    assert record.elapsed_raw is None


def test_parse_sacct_output_preserves_parent_and_job_step_memory_evidence() -> None:
    parent = (
        "20893681|vasp|guest|acct|FAILED|1:0|NonZeroExitCode|01:00:00|3600|"
        "start|end|pool|02:00:00|node-a|1|24|24|4Gc|cpu=24,mem=96G|"
        "cpu=24,mem=96G|20:00:00|86400||||/logs/out|/logs/err|/work"
    )
    batch = (
        "20893681.batch|batch|guest|acct|OUT_OF_MEMORY|0:125|OutOfMemory|01:00:00|"
        "3600|start|end|pool|02:00:00|node-a|1|1|1|96Gn|cpu=1,mem=96G|"
        "cpu=1,mem=96G|00:59:00|3540|94G|100G|90G|/logs/out|/logs/err|"
    )
    extern = (
        "20893681.extern|extern|guest|acct|COMPLETED|0:0||01:00:00|3600|start|"
        "end|pool|02:00:00|node-a|1|24|24||||00:01:00|1440|20M|30M|10M|||"
    )
    vasp_step = (
        "20893681.0|vasp_std|guest|acct|FAILED|1:0||01:00:00|3600|start|end|pool|"
        "02:00:00|node-a|1|24|24||||20:00:00|86400|28.80G|32G|27G|||"
    )

    record = parse_sacct_output(
        "20893681",
        "\n".join((batch, parent, extern, vasp_step)),
    )

    assert record is not None
    assert record.state == "FAILED"
    assert [step.job_id_raw for step in record.steps] == [
        "20893681.batch",
        "20893681.extern",
        "20893681.0",
    ]
    assert record.steps[0].state == "OUT_OF_MEMORY"
    assert record.steps[0].reason == "OutOfMemory"
    assert record.steps[0].max_rss == "94G"
    assert record.steps[1].max_rss == "20M"
    assert record.steps[2].max_rss == "28.80G"


def test_get_job_accounting_uses_mocked_ssh_transport() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is True
        assert kwargs["timeout"] == DEFAULT_SCHEDULER_ACCOUNTING_TIMEOUT_SECONDS
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "20893681|run|COMPLETED|02:48:37|2026-08-21T12:14:10|"
                "2026-08-21T15:02:47|leeburton-pool|0:0|72:00:00\n"
            ),
            stderr="",
        )

    record = get_job_accounting("powerslurm-bmdguest", "20893681", runner=runner)

    assert record is not None
    assert record.state == "COMPLETED"
    assert calls == [
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS}",
            "powerslurm-bmdguest",
            (
                "sacct -P -n -j 20893681 "
                f"{SACCT_FORMAT}"
            ),
        ]
    ]


def test_get_job_accounting_uses_explicit_finite_scheduler_timeout() -> None:
    observed: dict[str, object] = {}

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    get_job_accounting(
        "powerslurm-bmdguest",
        "21853598.0",
        runner=runner,
        timeout=45,
        ssh_connect_timeout=7,
    )

    assert observed["timeout"] == 45
    assert observed["command"][:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=7", "powerslurm-bmdguest"]
    assert "-j 21853598" in observed["command"][-1]
