from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import re
import shlex
import subprocess
import time
from typing import Any


PROFILE_SCHEMA_VERSION = 3
PROFILE_EVIDENCE_TYPE = "agent_performance_telemetry"

_COUNT_KEYS = (
    "subprocess_invocations",
    "ssh_invocations",
    "ssh_connections",
    "ssh_exec_channels",
    "ssh_control_operations",
    "remote_commands",
    "remote_file_reads",
    "bounded_remote_file_reads",
    "remote_extractor_operations",
    "existence_probes",
    "stat_probes",
    "directory_probes",
    "directory_listing_operations",
    "archive_probes",
    "archive_probe_batches",
    "metadata_manifest_operations",
    "batched_file_read_operations",
    "logical_files_described",
    "logical_files_read",
    "scheduler_operations",
    "producer_operations",
    "bmdex_operations",
)
_ELAPSED_KEYS = (
    "subprocess_wait",
    "ssh_wait",
    "remote_file_read_wait",
    "bounded_remote_file_read_wait",
    "remote_extractor_wait",
    "existence_probe_wait",
    "stat_probe_wait",
    "directory_probe_wait",
    "directory_listing_wait",
    "scheduler_wait",
    "producer_wait",
    "bmdex_wait",
    "archive_probe_wait",
    "metadata_manifest_wait",
    "batched_file_read_wait",
)
_ARCHIVE_PROBE_RE = re.compile(r"(?:^|/)error\.\d+\.tar\.gz(?:['\"\s]|$)")

Clock = Callable[[], float]
Runner = Callable[..., subprocess.CompletedProcess[Any]]


@dataclass(frozen=True)
class PhaseTiming:
    name: str
    calls: int
    elapsed_seconds: float
    failures: int = 0


@dataclass(frozen=True)
class OperationTelemetry:
    counts: Mapping[str, int]
    elapsed_seconds: Mapping[str, float]
    bytes_transferred: int
    ssh_bytes_transferred: int
    failure_count: int
    timeout_count: int
    byte_semantics: str = (
        "captured subprocess stdout and stderr byte lengths; measured from existing "
        "results without additional reads"
    )
    count_semantics: str = (
        "ssh_invocations counts local SSH client processes; ssh_connections counts "
        "transport connections established by those processes; ssh_exec_channels "
        "counts remote command channels; archive_probes counts remote metadata "
        "commands and archive_probe_batches identifies bounded multi-candidate probes; "
        "logical_files_described counts structured records returned by fixed batched "
        "operations and logical_files_read counts records reporting bounded contents"
    )


@dataclass(frozen=True)
class PerformanceProfile:
    total_elapsed_seconds: float
    phases: tuple[PhaseTiming, ...]
    operations: OperationTelemetry
    schema_version: int = PROFILE_SCHEMA_VERSION
    evidence_type: str = PROFILE_EVIDENCE_TYPE
    clock: str = "time.perf_counter"
    timing_semantics: str = (
        "total is wall-clock elapsed time; phase timings are inclusive and may overlap "
        "through nesting; operation wait timings are nested acquisition time"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_type": self.evidence_type,
            "clock": self.clock,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "timing_semantics": self.timing_semantics,
            "phases": [
                {
                    "name": phase.name,
                    "calls": phase.calls,
                    "elapsed_seconds": phase.elapsed_seconds,
                    "failures": phase.failures,
                }
                for phase in self.phases
            ],
            "operations": {
                "counts": dict(self.operations.counts),
                "elapsed_seconds": dict(self.operations.elapsed_seconds),
                "bytes_transferred": self.operations.bytes_transferred,
                "ssh_bytes_transferred": self.operations.ssh_bytes_transferred,
                "failure_count": self.operations.failure_count,
                "timeout_count": self.operations.timeout_count,
                "byte_semantics": self.operations.byte_semantics,
                "count_semantics": self.operations.count_semantics,
            },
        }


@dataclass
class _MutableTiming:
    calls: int = 0
    elapsed_seconds: float = 0.0
    failures: int = 0


class PerformanceProfiler:
    """Collect opt-in Agent execution telemetry without changing acquisition."""

    def __init__(self, *, clock: Clock = time.perf_counter) -> None:
        self._clock = clock
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._phases: dict[str, _MutableTiming] = {}
        self._counts = {key: 0 for key in _COUNT_KEYS}
        self._elapsed = {key: 0.0 for key in _ELAPSED_KEYS}
        self._bytes_transferred = 0
        self._ssh_bytes_transferred = 0
        self._failure_count = 0
        self._timeout_count = 0

    @contextmanager
    def activate(self):
        if self._started_at is not None:
            raise RuntimeError("performance profiler is already active or finished")
        self._started_at = self._clock()
        token = _ACTIVE_PROFILER.set(self)
        try:
            yield self
        finally:
            self._finished_at = self._clock()
            _ACTIVE_PROFILER.reset(token)

    @contextmanager
    def phase(self, name: str):
        started_at = self._clock()
        failed = False
        try:
            yield
        except Exception:
            failed = True
            raise
        finally:
            elapsed = max(0.0, self._clock() - started_at)
            timing = self._phases.setdefault(name, _MutableTiming())
            timing.calls += 1
            timing.elapsed_seconds += elapsed
            if failed:
                timing.failures += 1

    def wrap_runner(self, runner: Runner, *, role: str) -> Runner:
        if isinstance(runner, _ProfiledRunner) and runner.profiler is self:
            return runner
        return _ProfiledRunner(self, runner, role)

    def snapshot(self) -> PerformanceProfile:
        if self._started_at is None or self._finished_at is None:
            raise RuntimeError("performance profiler has not completed")
        phases = tuple(
            PhaseTiming(
                name=name,
                calls=timing.calls,
                elapsed_seconds=_seconds(timing.elapsed_seconds),
                failures=timing.failures,
            )
            for name, timing in self._phases.items()
        )
        return PerformanceProfile(
            total_elapsed_seconds=_seconds(self._finished_at - self._started_at),
            phases=phases,
            operations=OperationTelemetry(
                counts=dict(self._counts),
                elapsed_seconds={
                    key: _seconds(value)
                    for key, value in self._elapsed.items()
                },
                bytes_transferred=self._bytes_transferred,
                ssh_bytes_transferred=self._ssh_bytes_transferred,
                failure_count=self._failure_count,
                timeout_count=self._timeout_count,
            ),
        )

    def _record_subprocess(
        self,
        command: object,
        *,
        role: str,
        elapsed: float,
        result: object | None,
        failed: bool,
        timed_out: bool,
    ) -> None:
        self._counts["subprocess_invocations"] += 1
        self._elapsed["subprocess_wait"] += elapsed
        if role == "scheduler":
            self._counts["scheduler_operations"] += 1
            self._elapsed["scheduler_wait"] += elapsed
        elif role == "producer":
            self._counts["producer_operations"] += 1
            self._elapsed["producer_wait"] += elapsed
        elif role == "bmdex":
            self._counts["bmdex_operations"] += 1
            self._elapsed["bmdex_wait"] += elapsed

        command_parts = _command_parts(command)
        is_ssh = bool(command_parts) and command_parts[0] == "ssh"
        remote_command = command_parts[-1] if is_ssh else ""
        local_observation = _local_observation_text(command)
        if local_observation is not None:
            # Local transport: the same fixed operation, counted as an
            # observational command but never as an SSH invocation.
            self._counts["remote_commands"] += 1
            categories = _classify_remote_command(local_observation, self._counts)
            for category in categories:
                self._elapsed[f"{category}_wait"] += elapsed
            if _is_acquisition_command(local_observation):
                described, read = _acquisition_result_counts(result)
                self._counts["logical_files_described"] += described
                self._counts["logical_files_read"] += read
        if is_ssh:
            self._counts["ssh_invocations"] += 1
            self._elapsed["ssh_wait"] += elapsed
            control_operation = bool(
                getattr(command, "ssh_control_operation", False)
            ) or _is_ssh_control_operation(command_parts)
            exec_channel = bool(
                getattr(command, "ssh_exec_channel", not control_operation)
            )
            opens_connection = bool(
                getattr(command, "ssh_opens_connection", not control_operation)
            )
            if control_operation:
                self._counts["ssh_control_operations"] += 1
            if exec_channel:
                self._counts["ssh_exec_channels"] += 1
                self._counts["remote_commands"] += 1
                categories = _classify_remote_command(remote_command, self._counts)
                for category in categories:
                    self._elapsed[f"{category}_wait"] += elapsed
                if _is_acquisition_command(remote_command):
                    described, read = _acquisition_result_counts(result)
                    self._counts["logical_files_described"] += described
                    self._counts["logical_files_read"] += read
            if opens_connection and not _is_ssh_transport_failure(
                result,
                timed_out=timed_out,
            ):
                self._counts["ssh_connections"] += 1

        transferred = _result_bytes(result)
        self._bytes_transferred += transferred
        if is_ssh:
            self._ssh_bytes_transferred += transferred
        if failed:
            self._failure_count += 1
        if timed_out:
            self._timeout_count += 1


class _ProfiledRunner:
    def __init__(self, profiler: PerformanceProfiler, runner: Runner, role: str) -> None:
        self.profiler = profiler
        self.runner = runner
        self.role = role

    def __call__(self, command: object, **kwargs: object):
        started_at = self.profiler._clock()
        result = None
        failed = False
        timed_out = False
        try:
            result = self.runner(command, **kwargs)
            returncode = getattr(result, "returncode", 0)
            failed = (
                returncode != 0
                and not _is_expected_negative_probe(command, returncode)
            )
            return result
        except subprocess.TimeoutExpired as exc:
            result = exc
            failed = True
            timed_out = True
            raise
        except subprocess.CalledProcessError as exc:
            result = exc
            failed = True
            raise
        except Exception:
            failed = True
            raise
        finally:
            self.profiler._record_subprocess(
                command,
                role=self.role,
                elapsed=max(0.0, self.profiler._clock() - started_at),
                result=result,
                failed=failed,
                timed_out=timed_out,
            )


_ACTIVE_PROFILER: ContextVar[PerformanceProfiler | None] = ContextVar(
    "bmd_agent_active_profiler",
    default=None,
)


@contextmanager
def profile_phase(name: str):
    profiler = _ACTIVE_PROFILER.get()
    if profiler is None:
        yield
        return
    with profiler.phase(name):
        yield


def profiled_runner(runner: Runner, *, role: str) -> Runner:
    profiler = _ACTIVE_PROFILER.get()
    return runner if profiler is None else profiler.wrap_runner(runner, role=role)


def _local_observation_text(command: object) -> str | None:
    if getattr(command, "transport", None) != "local":
        return None
    text = getattr(command, "observation_text", None)
    return text if isinstance(text, str) else None


def _command_parts(command: object) -> tuple[str, ...]:
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        return ()
    return tuple(str(item) for item in command)


def _classify_remote_command(command: str, counts: dict[str, int]) -> tuple[str, ...]:
    categories: list[str] = []
    acquisition = _acquisition_command_counts(command)
    if acquisition is not None:
        _described, requested_reads, archive_count = acquisition
        counts["metadata_manifest_operations"] += 1
        categories.append("metadata_manifest")
        if requested_reads:
            counts["batched_file_read_operations"] += 1
            categories.append("batched_file_read")
        if archive_count:
            counts["archive_probes"] += 1
            counts["archive_probe_batches"] += 1
            categories.append("archive_probe")
    elif command.startswith("cat -- "):
        counts["remote_file_reads"] += 1
        categories.append("remote_file_read")
    elif command.startswith("tail -c "):
        counts["remote_file_reads"] += 1
        counts["bounded_remote_file_reads"] += 1
        categories.extend(("remote_file_read", "bounded_remote_file_read"))
    elif command.startswith("test -f "):
        counts["existence_probes"] += 1
        categories.append("existence_probe")
    elif command.startswith("test -d "):
        counts["existence_probes"] += 1
        counts["directory_probes"] += 1
        categories.extend(("existence_probe", "directory_probe"))
    elif command.startswith("stat -c "):
        counts["stat_probes"] += 1
        categories.append("stat_probe")
    elif command.startswith("awk "):
        counts["remote_extractor_operations"] += 1
        categories.append("remote_extractor")
    elif command.startswith("find ") or command.startswith("ls "):
        counts["directory_listing_operations"] += 1
        categories.append("directory_listing")

    archive_batch = (
        command.startswith("sh -c ")
        and "bmd-agent-archive-probe-v1" in command
    )
    archive_probe = command.startswith("test -f ") and bool(
        _ARCHIVE_PROBE_RE.search(command)
    )
    if archive_batch and acquisition is None:
        counts["archive_probes"] += 1
        counts["archive_probe_batches"] += 1
        categories.append("archive_probe")
    elif archive_probe:
        counts["archive_probes"] += 1
        categories.append("archive_probe")
    return tuple(categories)


def _is_acquisition_command(command: str) -> bool:
    return _acquisition_command_counts(command) is not None


def _acquisition_command_counts(command: str) -> tuple[int, int, int] | None:
    if not command.startswith("sh -s -- ") or "bmd-agent-acquisition-v1" not in command:
        return None
    try:
        parts = shlex.split(command)
        if parts[:4] != ["sh", "-s", "--", "bmd-agent-acquisition-v1"]:
            return None
        return int(parts[5]), int(parts[6]), int(parts[7])
    except (IndexError, ValueError):
        return None


def _acquisition_result_counts(result: object | None) -> tuple[int, int]:
    stdout = getattr(result, "stdout", b"")
    if isinstance(stdout, bytes):
        text = stdout.decode("ascii", "ignore")
    elif isinstance(stdout, str):
        text = stdout
    else:
        return 0, 0
    records = [
        line.split("\t")
        for line in text.splitlines()
        if line.startswith("item\t") and len(line.split("\t")) == 6
    ]
    return len(records), sum(fields[3] == "read" for fields in records)


def _is_ssh_control_operation(command: Sequence[str]) -> bool:
    return "-O" in command and "exit" in command


def _is_ssh_transport_failure(result: object | None, *, timed_out: bool) -> bool:
    if timed_out or result is None:
        return True
    return getattr(result, "returncode", None) == 255


def _is_expected_negative_probe(command: object, returncode: object) -> bool:
    parts = _command_parts(command)
    local_observation = _local_observation_text(command)
    if returncode != 1:
        return False
    if local_observation is not None:
        remote_command = local_observation
    elif parts and parts[0] == "ssh":
        remote_command = parts[-1]
    else:
        return False
    return remote_command.startswith("test -f ") or remote_command.startswith("test -d ")


def _result_bytes(result: object | None) -> int:
    if result is None:
        return 0
    return _value_bytes(getattr(result, "stdout", None)) + _value_bytes(
        getattr(result, "stderr", None)
    )


def _value_bytes(value: object) -> int:
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return 0


def _seconds(value: float) -> float:
    return round(max(0.0, value), 6)
