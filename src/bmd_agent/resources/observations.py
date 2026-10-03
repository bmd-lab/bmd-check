"""The closed set of fixed observational operations BMD Agent performs.

Each operation is a frozen value object whose variable inputs (paths, job IDs,
partitions, byte limits) are typed and validated on construction. An operation
renders exactly one fixed argument vector; there is no operation that carries
caller-supplied command text. Transports (``transport.py``) decide only how
that vector runs: as an argv on this machine, or shell-quoted over SSH.

Where an observation needs shell control flow (the error-archive probe and the
batched acquisition), the shell program is an Agent-owned constant in this
module and every variable input is a separate positional argument, never
interpolated into program text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PosixPath, PurePosixPath
import re
import shlex

_ARCHIVE_PROBE_MARKER = "bmd-agent-archive-probe-v1"
_MAX_ARCHIVE_PROBE_LIMIT = 64
_ACQUISITION_MARKER = "bmd-agent-acquisition-v1"
_MAX_ACQUISITION_REQUESTS = 128
_MAX_BATCH_FILE_BYTES = 2_000_000
_MAX_BATCH_TOTAL_BYTES = 8_000_000

_ACQUISITION_SCRIPT = """\
printf 'schema\\tbmd-agent-acquisition-v1\\n'
total_limit=$1
shift
used=0
while [ "$#" -ge 4 ]; do
    item_id=$1
    kind=$2
    read_limit=$3
    path=$4
    shift 4
    if [ "$kind" = "archives" ]; then
        archive_index=1
        archive_count=0
        while [ "$archive_index" -le "$read_limit" ]; do
            candidate="$path/error.$archive_index.tar.gz"
            if [ ! -f "$candidate" ]; then
                break
            fi
            archive_count=$archive_index
            archive_index=$((archive_index + 1))
        done
        printf 'item\\t%s\\tarchives\\tpresent\\t%s\\t\\n' "$item_id" "$archive_count"
        continue
    fi
    if [ "$kind" = "directory" ]; then
        if [ -d "$path" ]; then
            printf 'item\\t%s\\tdirectory\\tpresent\\t\\t\\n' "$item_id"
        else
            printf 'item\\t%s\\tdirectory\\tmissing\\t\\t\\n' "$item_id"
        fi
        continue
    fi
    if [ ! -f "$path" ]; then
        printf 'item\\t%s\\tfile\\tmissing\\t\\t\\n' "$item_id"
        continue
    fi
    if ! size=$(stat -c %s -- "$path" 2>/dev/null); then
        printf 'item\\t%s\\tfile\\tread_error\\t\\t\\n' "$item_id"
        continue
    fi
    if [ "$read_limit" -le 0 ]; then
        printf 'item\\t%s\\tfile\\tpresent\\t%s\\t\\n' "$item_id" "$size"
        continue
    fi
    next_used=$((used + size))
    if [ "$size" -gt "$read_limit" ] || [ "$next_used" -gt "$total_limit" ]; then
        printf 'item\\t%s\\tfile\\tdeferred\\t%s\\t\\n' "$item_id" "$size"
        continue
    fi
    if encoded=$(head -c "$size" -- "$path" 2>/dev/null | base64 | tr -d '\\n'); then
        printf 'item\\t%s\\tfile\\tread\\t%s\\t%s\\n' "$item_id" "$size" "$encoded"
        used=$next_used
    else
        printf 'item\\t%s\\tfile\\tread_error\\t%s\\t\\n' "$item_id" "$size"
    fi
done
"""

_OUTCAR_FORCE_EXTRACTOR_AWK_EMIT = (
    'complete=(emit_status=="complete"&&rows>0&&malformed==0);'
    'final_status=emit_status;value="";'
    'if(complete&&expected!=""&&rows!=expected+0){complete=0;final_status="row_count_mismatch"}'
    'if(malformed){complete=0;final_status="malformed"}'
    'if(complete){value=max_force}'
    'printf("block\\t%d\\t%d\\t%s\\t%d\\t%s\\n",block_no,rows,final_status,complete,value);'
    'state=0;rows=0;max_force=0;malformed=0'
)

_OUTCAR_FORCE_EXTRACTOR_AWK = (
    'BEGIN{print "schema\\tbmd-agent-outcar-force-v1";'
    'print "expected_site_count\\t" expected;'
    'num="^[-+]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([Ee][-+]?[0-9]+)?$"}'
    '/^[[:space:]]*POSITION[[:space:]]+TOTAL-FORCE[[:space:]]+\\(eV\\/Angst\\)[[:space:]]*$/{'
    'if(state){emit_status="incomplete";'
    + _OUTCAR_FORCE_EXTRACTOR_AWK_EMIT
    + '};block_no++;state=1;rows=0;max_force=0;malformed=0;next}'
    'state&&/^[[:space:]]*---[-]*[[:space:]]*$/{'
    'if(state==1){state=2}else{emit_status="complete";'
    + _OUTCAR_FORCE_EXTRACTOR_AWK_EMIT
    + '};next}'
    'state&&NF{'
    'if(state==1){state=2;malformed=1}'
    'if(NF<6||$4!~num||$5!~num||$6!~num){malformed=1;next}'
    'fx=$4+0;fy=$5+0;fz=$6+0;rows++;force=sqrt(fx*fx+fy*fy+fz*fz);'
    'if(force>max_force){max_force=force}next}'
    'END{if(state){emit_status="incomplete";'
    + _OUTCAR_FORCE_EXTRACTOR_AWK_EMIT
    + '}}'
)

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

_ARCHIVE_PROBE_PROGRAM = (
    'i=1; while [ "$i" -le "$2" ]; do '
    'candidate="$1/error.$i.tar.gz"; '
    'if [ -f "$candidate" ]; then printf "%s\\n" "$candidate"; else break; fi; '
    'i=$((i + 1)); done'
)

_PARTITION_RE = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_.-]*$")
_NORMALIZED_JOB_ID_RE = re.compile(r"^[1-9]\d*(?:_\d+)?$")


class ObservationOperationError(ValueError):
    """An observational operation was constructed from an invalid value."""


def _authorized_path(value: object) -> PurePosixPath:
    """Require an absolute, lexically normalized POSIX path (already authorized)."""

    if type(value) not in (str, PurePosixPath, PosixPath):
        raise ObservationOperationError("observation path must be a POSIX path")
    text = str(value)
    if "\x00" in text or "\\" in text or not text.startswith("/"):
        raise ObservationOperationError("observation path must be an absolute POSIX path")
    if text != "/" and (
        text.endswith("/")
        or "//" in text
        or any(part in (".", "..") for part in text.split("/"))
    ):
        raise ObservationOperationError("observation path must be lexically normalized")
    return PurePosixPath(text)


def _bounded_int(value: object, *, minimum: int, maximum: int, label: str) -> int:
    if type(value) is not int:
        raise ObservationOperationError(f"{label} must be an integer")
    if value < minimum or value > maximum:
        raise ObservationOperationError(f"{label} must be between {minimum} and {maximum}")
    return value


class ObservationOperation:
    """Base of the closed operation set. Subclasses are defined only here."""

    __slots__ = ()

    def argv(self) -> tuple[str, ...]:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def stdin(self) -> bytes | None:
        return None

    def remote_command(self) -> str:
        """The same argv, shell-quoted for a POSIX remote login shell."""

        return operation_remote_command(self)


@dataclass(frozen=True, slots=True)
class SqueuePartition(ObservationOperation):
    partition: str

    def __post_init__(self) -> None:
        if type(self.partition) is not str or not _PARTITION_RE.fullmatch(self.partition):
            raise ObservationOperationError("partition contains unsafe characters")

    def argv(self) -> tuple[str, ...]:
        return ("squeue", "-p", self.partition, "--noheader", f"--format={_SQUEUE_FORMAT}")


@dataclass(frozen=True, slots=True)
class SacctJob(ObservationOperation):
    job_id: str

    def __post_init__(self) -> None:
        if type(self.job_id) is not str or not _NORMALIZED_JOB_ID_RE.fullmatch(self.job_id):
            raise ObservationOperationError("SLURM job ID must be a normalized positive job ID")

    def argv(self) -> tuple[str, ...]:
        return ("sacct", "-P", "-n", "-j", self.job_id, f"--format={_SACCT_FORMAT}")


@dataclass(frozen=True, slots=True)
class ReadFile(ObservationOperation):
    path: PurePosixPath

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _authorized_path(self.path))

    def argv(self) -> tuple[str, ...]:
        return ("cat", "--", str(self.path))


@dataclass(frozen=True, slots=True)
class ReadFileTail(ObservationOperation):
    path: PurePosixPath
    limit: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _authorized_path(self.path))
        _bounded_int(self.limit, minimum=1, maximum=2**53, label="tail byte limit")

    def argv(self) -> tuple[str, ...]:
        return ("tail", "-c", str(self.limit), "--", str(self.path))


@dataclass(frozen=True, slots=True)
class FileSize(ObservationOperation):
    path: PurePosixPath

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _authorized_path(self.path))

    def argv(self) -> tuple[str, ...]:
        return ("stat", "-c", "%s", "--", str(self.path))


@dataclass(frozen=True, slots=True)
class PathTest(ObservationOperation):
    path: PurePosixPath
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _authorized_path(self.path))
        if type(self.kind) is not str or self.kind not in ("file", "directory"):
            raise ObservationOperationError("path test kind must be file or directory")

    def argv(self) -> tuple[str, ...]:
        flag = "-f" if self.kind == "file" else "-d"
        return ("test", flag, str(self.path))


@dataclass(frozen=True, slots=True)
class ProbeErrorArchives(ObservationOperation):
    directory: PurePosixPath
    limit: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", _authorized_path(self.directory))
        _bounded_int(self.limit, minimum=1, maximum=_MAX_ARCHIVE_PROBE_LIMIT, label="archive probe limit")

    def argv(self) -> tuple[str, ...]:
        # $0 is the marker; $1 and $2 are data, never program text.
        return ("sh", "-c", _ARCHIVE_PROBE_PROGRAM, _ARCHIVE_PROBE_MARKER, str(self.directory), str(self.limit))


@dataclass(frozen=True, slots=True)
class ExtractOutcarForces(ObservationOperation):
    path: PurePosixPath
    expected_site_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _authorized_path(self.path))
        if self.expected_site_count is not None:
            if type(self.expected_site_count) is not int:
                raise ObservationOperationError("expected site count must be an integer")
            if self.expected_site_count <= 0:
                raise ValueError("expected site count must be positive")

    def argv(self) -> tuple[str, ...]:
        expected = "" if self.expected_site_count is None else str(self.expected_site_count)
        return ("awk", "-v", "expected=" + expected, _OUTCAR_FORCE_EXTRACTOR_AWK, str(self.path))


_ACQUISITION_KINDS = ("file", "directory", "archives")


@dataclass(frozen=True, slots=True)
class AcquisitionItem:
    path: PurePosixPath
    kind: str
    read_limit: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _authorized_path(self.path))
        if type(self.kind) is not str or self.kind not in _ACQUISITION_KINDS:
            raise ObservationOperationError("acquisition kind must be file, directory, or archives")
        maximum = _MAX_ARCHIVE_PROBE_LIMIT if self.kind == "archives" else _MAX_BATCH_FILE_BYTES
        _bounded_int(self.read_limit, minimum=0, maximum=maximum, label="acquisition read limit")


@dataclass(frozen=True, slots=True)
class AcquireBatch(ObservationOperation):
    items: tuple[AcquisitionItem, ...]

    def __post_init__(self) -> None:
        if type(self.items) is not tuple:
            raise ObservationOperationError("acquisition batch items must be a tuple")
        items = self.items
        if not items or len(items) > _MAX_ACQUISITION_REQUESTS:
            raise ObservationOperationError("acquisition batch size is out of range")
        if not all(type(item) is AcquisitionItem for item in items):
            raise ObservationOperationError("acquisition batch items must be AcquisitionItem values")
        object.__setattr__(self, "items", items)

    def argv(self) -> tuple[str, ...]:
        read_count = sum(bool(item.read_limit) for item in self.items if item.kind == "file")
        archive_count = sum(item.kind == "archives" for item in self.items)
        arguments = [
            "sh",
            "-s",
            "--",
            _ACQUISITION_MARKER,
            str(_MAX_BATCH_TOTAL_BYTES),
            str(len(self.items)),
            str(read_count),
            str(archive_count),
        ]
        for index, item in enumerate(self.items, start=1):
            arguments.extend((str(index), item.kind, str(item.read_limit), str(item.path)))
        return tuple(arguments)

    @property
    def stdin(self) -> bytes:
        return _ACQUISITION_SCRIPT.encode("ascii")


# The closed set. Transports refuse any operation whose exact type is not here.
OBSERVATION_OPERATION_TYPES: tuple[type[ObservationOperation], ...] = (
    SqueuePartition,
    SacctJob,
    ReadFile,
    ReadFileTail,
    FileSize,
    PathTest,
    ProbeErrorArchives,
    ExtractOutcarForces,
    AcquireBatch,
)


def require_operation(operation: object) -> ObservationOperation:
    if type(operation) not in OBSERVATION_OPERATION_TYPES:
        raise ObservationOperationError("not a fixed BMD Agent observational operation")
    return operation  # type: ignore[return-value]


def snapshot_operation(operation: object) -> ObservationOperation:
    """Return a freshly validated exact-type copy of one trusted operation."""

    operation = require_operation(operation)
    operation_type = type(operation)
    if operation_type is SqueuePartition:
        return SqueuePartition(operation.partition)
    if operation_type is SacctJob:
        return SacctJob(operation.job_id)
    if operation_type is ReadFile:
        return ReadFile(operation.path)
    if operation_type is ReadFileTail:
        return ReadFileTail(operation.path, operation.limit)
    if operation_type is FileSize:
        return FileSize(operation.path)
    if operation_type is PathTest:
        return PathTest(operation.path, operation.kind)
    if operation_type is ProbeErrorArchives:
        return ProbeErrorArchives(operation.directory, operation.limit)
    if operation_type is ExtractOutcarForces:
        return ExtractOutcarForces(operation.path, operation.expected_site_count)
    if operation_type is AcquireBatch:
        items = tuple(
            AcquisitionItem(item.path, item.kind, item.read_limit)
            for item in operation.items
            if type(item) is AcquisitionItem
        )
        if len(items) != len(operation.items):
            raise ObservationOperationError(
                "acquisition batch items must be exact AcquisitionItem values"
            )
        return AcquireBatch(items)
    raise ObservationOperationError("not a fixed BMD Agent observational operation")


def operation_argv(operation: object) -> tuple[str, ...]:
    """Render an operation only after exact-type copying and validation."""

    snapshot = snapshot_operation(operation)
    return type(snapshot).argv(snapshot)


def operation_stdin(operation: object) -> bytes | None:
    """Return only fixed Agent-owned stdin for an exact trusted operation."""

    snapshot = snapshot_operation(operation)
    if type(snapshot) is AcquireBatch:
        return _ACQUISITION_SCRIPT.encode("ascii")
    return None


def operation_remote_command(operation: object) -> str:
    """Shell-quote a freshly validated operation argv for remote SSH."""

    return " ".join(shlex.quote(item) for item in operation_argv(operation))
