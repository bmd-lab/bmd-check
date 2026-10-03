import base64
import binascii
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import PurePosixPath
import subprocess
from typing import Callable, Iterable, Iterator, Sequence

from pymatgen.io.vasp import Poscar
from bmd_agent.resources.observations import (
    _ACQUISITION_MARKER,
    _MAX_ACQUISITION_REQUESTS,
    _MAX_ARCHIVE_PROBE_LIMIT,
    _MAX_BATCH_FILE_BYTES,
    AcquireBatch,
    AcquisitionItem,
    ExtractOutcarForces,
    FileSize,
    PathTest,
    ProbeErrorArchives,
    ReadFile,
    ReadFileTail,
)
from bmd_agent.resources.transport import observe, run_invocation


Runner = Callable[..., subprocess.CompletedProcess[bytes]]

class RemoteOutcarForceExtractionError(RuntimeError):
    """Raised when the fixed remote OUTCAR extractor fails before valid output."""

    def __init__(
        self,
        public_message: str,
        *,
        kind: str,
        returncode: int | None = None,
        stderr_summary: str | None = None,
    ) -> None:
        super().__init__(public_message)
        self.public_message = public_message
        self.kind = kind
        self.returncode = returncode
        self.stderr_summary = stderr_summary


class RemotePathError(ValueError):
    """Raised when a requested remote path is outside configured policy."""


@dataclass
class StructureInfo:
    source: str
    formula: str
    reduced_formula: str
    sites: int
    volume: float
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class RemoteAcquisitionRequest:
    path: PurePosixPath
    kind: str = "file"
    read_limit: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"file", "directory", "archives"}:
            raise ValueError(
                "remote acquisition kind must be file, directory, or archives"
            )
        maximum = (
            _MAX_ARCHIVE_PROBE_LIMIT
            if self.kind == "archives"
            else _MAX_BATCH_FILE_BYTES
        )
        if self.read_limit < 0 or self.read_limit > maximum:
            raise ValueError(
                f"remote acquisition read limit must be between 0 and {maximum}"
            )
        if self.kind == "directory" and self.read_limit:
            raise ValueError("remote directory acquisition cannot request contents")
        if self.kind == "archives" and self.read_limit <= 0:
            raise ValueError("remote archive acquisition limit must be positive")


@dataclass(frozen=True)
class _RemoteAcquisitionRecord:
    path: PurePosixPath
    kind: str
    present: bool
    size: int | None = None
    contents: bytes | None = None
    content_status: str | None = None


class RemoteAcquisitionCache:
    """Invocation-local metadata and bounded-content cache for exact remote paths."""

    def __init__(
        self,
        ssh_host: str,
        allowed_roots: Iterable[PurePosixPath | str],
    ) -> None:
        self.ssh_host = ssh_host
        self.allowed_roots = tuple(PurePosixPath(str(root)) for root in allowed_roots)
        self._records: dict[tuple[str, str], _RemoteAcquisitionRecord] = {}
        self._disabled = False

    def prime(
        self,
        requests: Iterable[RemoteAcquisitionRequest],
        *,
        runner: Runner,
        timeout: float,
    ) -> None:
        if self._disabled:
            return
        pending = self._pending_requests(requests)
        for start in range(0, len(pending), _MAX_ACQUISITION_REQUESTS):
            batch = pending[start : start + _MAX_ACQUISITION_REQUESTS]
            try:
                self._acquire_batch(batch, runner=runner, timeout=timeout)
            except (OSError, UnicodeError, ValueError, subprocess.SubprocessError):
                self._disabled = True
                return

    def record(self, path: PurePosixPath, kind: str) -> _RemoteAcquisitionRecord | None:
        return self._records.get((str(path), kind))

    def remember_file(self, path: PurePosixPath, contents: bytes) -> None:
        self._records[(str(path), "file")] = _RemoteAcquisitionRecord(
            path=path,
            kind="file",
            present=True,
            size=len(contents),
            contents=contents,
            content_status="read",
        )

    def _pending_requests(
        self,
        requests: Iterable[RemoteAcquisitionRequest],
    ) -> list[RemoteAcquisitionRequest]:
        merged: dict[tuple[str, str], RemoteAcquisitionRequest] = {}
        for request in requests:
            path = authorize_remote_path(
                request.path,
                allowed_roots=self.allowed_roots,
            )
            normalized = RemoteAcquisitionRequest(
                path,
                request.kind,
                request.read_limit,
            )
            key = (str(path), request.kind)
            previous = merged.get(key)
            if previous is None or normalized.read_limit > previous.read_limit:
                merged[key] = normalized

        pending: list[RemoteAcquisitionRequest] = []
        for key, request in merged.items():
            existing = self._records.get(key)
            if existing is None:
                pending.append(request)
            elif (
                request.kind == "file"
                and request.read_limit
                and existing.present
                and existing.contents is None
            ):
                pending.append(request)
        return pending

    def _acquire_batch(
        self,
        requests: Sequence[RemoteAcquisitionRequest],
        *,
        runner: Runner,
        timeout: float,
    ) -> None:
        if not requests:
            return
        operation = AcquireBatch(
            tuple(
                AcquisitionItem(request.path, request.kind, request.read_limit)
                for request in requests
            )
        )
        result = observe(
            self.ssh_host,
            operation,
            runner=runner,
            capture_output=True,
            check=True,
            timeout=timeout,
        )
        parsed = _parse_acquisition_output(result.stdout, requests)
        for record in parsed:
            self._records[(str(record.path), record.kind)] = record


_ACTIVE_REMOTE_ACQUISITION: ContextVar[RemoteAcquisitionCache | None] = ContextVar(
    "bmd_agent_remote_acquisition",
    default=None,
)


@contextmanager
def remote_acquisition_cache(
    ssh_host: str,
    allowed_roots: Iterable[PurePosixPath | str],
) -> Iterator[RemoteAcquisitionCache]:
    cache = RemoteAcquisitionCache(ssh_host, allowed_roots)
    token = _ACTIVE_REMOTE_ACQUISITION.set(cache)
    try:
        yield cache
    finally:
        _ACTIVE_REMOTE_ACQUISITION.reset(token)


def prime_remote_acquisition(
    ssh_host: str,
    requests: Iterable[RemoteAcquisitionRequest],
    *,
    runner: Runner,
    timeout: float,
) -> None:
    cache = _active_acquisition(ssh_host)
    if cache is not None:
        cache.prime(requests, runner=runner, timeout=timeout)


def _active_acquisition(ssh_host: str) -> RemoteAcquisitionCache | None:
    cache = _ACTIVE_REMOTE_ACQUISITION.get()
    return cache if cache is not None and cache.ssh_host == ssh_host else None


def _parse_acquisition_output(
    stdout: bytes,
    requests: Sequence[RemoteAcquisitionRequest],
) -> tuple[_RemoteAcquisitionRecord, ...]:
    text = stdout.decode("ascii", "strict")
    lines = text.splitlines()
    if not lines or lines[0] != f"schema\t{_ACQUISITION_MARKER}":
        raise ValueError("remote acquisition returned an unsupported schema")
    records_by_index: dict[int, _RemoteAcquisitionRecord] = {}
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 6 or fields[0] != "item":
            raise ValueError("remote acquisition returned a malformed record")
        try:
            index = int(fields[1])
        except ValueError as exc:
            raise ValueError("remote acquisition returned an invalid record index") from exc
        if index < 1 or index > len(requests) or index in records_by_index:
            raise ValueError("remote acquisition returned an unexpected record index")
        request = requests[index - 1]
        kind, status, size_text, encoded = fields[2:]
        if kind != request.kind or status not in {
            "present",
            "missing",
            "deferred",
            "read",
            "read_error",
        }:
            raise ValueError("remote acquisition returned invalid record metadata")
        try:
            size = int(size_text) if size_text else None
        except ValueError as exc:
            raise ValueError("remote acquisition returned an invalid file size") from exc
        if size is not None and size < 0:
            raise ValueError("remote acquisition returned an invalid file size")
        if kind == "directory" and (
            status not in {"present", "missing"} or size is not None or encoded
        ):
            raise ValueError("remote acquisition returned invalid directory metadata")
        if kind == "archives" and (
            status != "present"
            or size is None
            or size > request.read_limit
            or encoded
        ):
            raise ValueError("remote acquisition returned invalid archive metadata")
        if kind == "file":
            if status == "read" and not request.read_limit:
                raise ValueError("remote acquisition returned unexpected file contents")
            if status == "missing" and size is not None:
                raise ValueError("remote acquisition returned invalid missing-file metadata")
            if status in {"present", "deferred", "read"} and size is None:
                raise ValueError("remote acquisition returned missing file size")
        contents: bytes | None = None
        content_status = status if request.kind == "file" and request.read_limit else None
        if status == "read":
            try:
                contents = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                content_status = "read_error"
            if contents is not None and size != len(contents):
                contents = None
                content_status = "read_error"
        elif encoded:
            raise ValueError("remote acquisition returned unexpected file contents")
        records_by_index[index] = _RemoteAcquisitionRecord(
            path=request.path,
            kind=kind,
            present=status != "missing",
            size=size,
            contents=contents,
            content_status=content_status,
        )
    if len(records_by_index) != len(requests):
        raise ValueError("remote acquisition did not describe every requested path")
    return tuple(records_by_index[index] for index in range(1, len(requests) + 1))


def read_remote_structure(
    ssh_host: str,
    directory: str,
    allowed_roots: Iterable[PurePosixPath | str],
    filename: str = "POSCAR",
    *,
    runner: Runner = run_invocation,
    timeout: float = 20,
) -> StructureInfo:
    """Read a VASP structure remotely without modifying the source."""

    remote_path = build_remote_file_path(
        directory,
        filename,
        allowed_roots=allowed_roots,
    )
    contents = retrieve_remote_file(
        ssh_host,
        remote_path,
        runner=runner,
        timeout=timeout,
    )

    return parse_poscar(contents, source=str(remote_path))


def retrieve_remote_file(
    ssh_host: str,
    remote_path: PurePosixPath,
    *,
    runner: Runner = run_invocation,
    timeout: float = 20,
) -> bytes:
    """Retrieve one already-authorized remote file over SSH."""

    cache = _active_acquisition(ssh_host)
    if cache is not None:
        cached = cache.record(remote_path, "file")
        if cached is not None and cached.contents is not None:
            return cached.contents
        if cached is not None and not cached.present:
            raise subprocess.CalledProcessError(
                1,
                ["ssh", ssh_host, "cached remote file read"],
                stderr=b"missing",
            )

    result = observe(
        ssh_host,
        ReadFile(remote_path),
        runner=runner,
        capture_output=True,
        check=True,
        timeout=timeout,
    )

    if cache is not None:
        cache.remember_file(remote_path, result.stdout)
    return result.stdout


def retrieve_remote_file_tail(
    ssh_host: str,
    remote_path: PurePosixPath,
    *,
    limit: int,
    runner: Runner = run_invocation,
    timeout: float = 20,
) -> bytes:
    """Retrieve a bounded tail from one already-authorized remote file."""

    if limit <= 0:
        raise ValueError("remote tail byte limit must be positive")
    cache = _active_acquisition(ssh_host)
    if cache is not None:
        cached = cache.record(remote_path, "file")
        if cached is not None and cached.contents is not None:
            return cached.contents[-limit:]
        if cached is not None and not cached.present:
            raise subprocess.CalledProcessError(
                1,
                ["ssh", ssh_host, "cached remote tail read"],
                stderr=b"missing",
            )
    result = observe(
        ssh_host,
        ReadFileTail(remote_path, limit),
        runner=runner,
        capture_output=True,
        check=True,
        timeout=timeout,
    )
    return result.stdout


def remote_file_exists(
    ssh_host: str,
    remote_path: PurePosixPath,
    *,
    runner: Runner = run_invocation,
    timeout: float = 20,
) -> bool:
    """Return whether one already-authorized remote file exists."""

    cache = _active_acquisition(ssh_host)
    if cache is not None:
        cached = cache.record(remote_path, "file")
        if cached is not None:
            return cached.present

    return _remote_path_exists(
        ssh_host,
        remote_path,
        kind="file",
        runner=runner,
        timeout=timeout,
    )


def remote_file_size(
    ssh_host: str,
    remote_path: PurePosixPath,
    *,
    runner: Runner = run_invocation,
    timeout: float = 20,
) -> int:
    """Return the byte size of one already-authorized remote file."""

    cache = _active_acquisition(ssh_host)
    if cache is not None:
        cached = cache.record(remote_path, "file")
        if cached is not None and cached.present and cached.size is not None:
            return cached.size

    result = observe(
        ssh_host,
        FileSize(remote_path),
        runner=runner,
        capture_output=True,
        check=True,
        timeout=timeout,
    )

    return int(result.stdout.decode("utf-8", "replace").strip())


def probe_remote_error_archives(
    ssh_host: str,
    directory: PurePosixPath | str,
    *,
    allowed_roots: Iterable[PurePosixPath | str],
    limit: int,
    runner: Runner = run_invocation,
    timeout: float = 20,
) -> tuple[PurePosixPath, ...]:
    """Probe a bounded contiguous error.N.tar.gz sequence without reading archives."""

    if limit <= 0 or limit > _MAX_ARCHIVE_PROBE_LIMIT:
        raise ValueError(
            f"remote error archive probe limit must be between 1 and {_MAX_ARCHIVE_PROBE_LIMIT}"
        )
    authorized_directory = authorize_remote_path(
        directory,
        allowed_roots=allowed_roots,
    )
    cache = _active_acquisition(ssh_host)
    if cache is not None:
        request = RemoteAcquisitionRequest(
            authorized_directory,
            kind="archives",
            read_limit=limit,
        )
        cache.prime((request,), runner=runner, timeout=timeout)
        record = cache.record(authorized_directory, "archives")
        if record is not None and record.size is not None:
            return tuple(
                authorized_directory / f"error.{index}.tar.gz"
                for index in range(1, record.size + 1)
            )
    result = observe(
        ssh_host,
        ProbeErrorArchives(authorized_directory, limit),
        runner=runner,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        return ()

    lines = result.stdout.decode("utf-8", "strict").splitlines()
    if len(lines) > limit:
        raise ValueError("remote error archive probe returned too many paths")
    archives: list[PurePosixPath] = []
    for index, line in enumerate(lines, start=1):
        expected = authorized_directory / f"error.{index}.tar.gz"
        if line != str(expected):
            raise ValueError("remote error archive probe returned an unexpected path")
        archives.append(expected)
    return tuple(archives)


def extract_remote_outcar_force_blocks(
    ssh_host: str,
    remote_path: PurePosixPath,
    *,
    expected_site_count: int | None = None,
    runner: Runner = run_invocation,
    timeout: float = 20,
) -> str:
    """Return compact OUTCAR force-block evidence without transferring the OUTCAR."""

    operation = ExtractOutcarForces(remote_path, expected_site_count)
    try:
        result = observe(
            ssh_host,
            operation,
            runner=runner,
            capture_output=True,
            check=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        raise _outcar_extraction_error(exc) from exc

    return result.stdout.decode("utf-8", "replace")


def build_remote_outcar_force_command(
    remote_path: PurePosixPath,
    *,
    expected_site_count: int | None = None,
) -> str:
    """Build the fixed read-only remote OUTCAR force extractor command."""

    return ExtractOutcarForces(remote_path, expected_site_count).remote_command()


def _outcar_extraction_error(
    exc: subprocess.CalledProcessError,
) -> RemoteOutcarForceExtractionError:
    stderr_excerpt = _stderr_excerpt(exc.stderr)
    stderr_summary = _stderr_summary(stderr_excerpt)
    lower = (stderr_excerpt or "").lower()
    kind = "remote_command_failed"
    message = "extractor command failed"

    if (
        "cannot open" in lower
        or "permission denied" in lower
        or "no such file" in lower
    ):
        kind = "outcar_read_failed"
        message = "file could not be read"
    elif exc.returncode == 127 or "command not found" in lower or "awk: not found" in lower:
        kind = "extractor_runtime_unavailable"
        message = "remote extractor runtime unavailable"
    elif (
        "syntax error" in lower
        or "unexpected eof" in lower
        or "unterminated" in lower
        or "unexpected token" in lower
    ):
        kind = "command_invocation_failed"
        message = "extractor command invocation failed"

    return RemoteOutcarForceExtractionError(
        message,
        kind=kind,
        returncode=exc.returncode,
        stderr_summary=stderr_summary,
    )


def _stderr_summary(stderr: bytes | str | None) -> str | None:
    if stderr is None:
        return None
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:240]
    return None


def _stderr_excerpt(stderr: bytes | str | None) -> str | None:
    if stderr is None:
        return None
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
        if len(lines) >= 12:
            break
    excerpt = "\n".join(lines)
    return excerpt[:2000] if excerpt else None


def remote_directory_exists(
    ssh_host: str,
    remote_path: PurePosixPath,
    *,
    runner: Runner = run_invocation,
    timeout: float = 20,
) -> bool:
    """Return whether one already-authorized remote directory exists."""

    cache = _active_acquisition(ssh_host)
    if cache is not None:
        cached = cache.record(remote_path, "directory")
        if cached is not None:
            return cached.present

    return _remote_path_exists(
        ssh_host,
        remote_path,
        kind="directory",
        runner=runner,
        timeout=timeout,
    )


def _remote_path_exists(
    ssh_host: str,
    remote_path: PurePosixPath,
    *,
    kind: str,
    runner: Runner,
    timeout: float,
) -> bool:
    result = observe(
        ssh_host,
        PathTest(remote_path, kind),
        runner=runner,
        capture_output=True,
        check=False,
        timeout=timeout,
    )

    return result.returncode == 0


def parse_poscar(contents: bytes | str, *, source: str) -> StructureInfo:
    """Parse VASP POSCAR content using pymatgen."""

    text = contents.decode("utf-8") if isinstance(contents, bytes) else contents
    poscar = Poscar.from_str(text)
    structure = poscar.structure

    lattice = structure.lattice

    return StructureInfo(
        source=source,
        formula=structure.composition.formula,
        reduced_formula=structure.composition.reduced_formula,
        sites=len(structure),
        volume=structure.volume,
        a=lattice.a,
        b=lattice.b,
        c=lattice.c,
    )


def build_remote_file_path(
    directory: str | PurePosixPath,
    filename: str,
    *,
    allowed_roots: Iterable[PurePosixPath | str],
) -> PurePosixPath:
    """Build and authorize a POSIX remote file path."""

    remote_directory = normalize_remote_path(directory)
    remote_filename = PurePosixPath(filename)

    if (
        remote_filename.is_absolute()
        or not remote_filename.parts
        or remote_filename.parts == (".",)
        or ".." in remote_filename.parts
    ):
        raise RemotePathError("remote filename must be relative and must not contain '..'")

    return authorize_remote_path(
        remote_directory / remote_filename,
        allowed_roots=allowed_roots,
    )


def authorize_remote_path(
    path: str | PurePosixPath,
    *,
    allowed_roots: Iterable[PurePosixPath | str],
) -> PurePosixPath:
    """Authorize one absolute POSIX remote path against configured roots."""

    remote_path = normalize_remote_path(path)
    normalized_roots = tuple(normalize_remote_path(root) for root in allowed_roots)

    if not normalized_roots:
        raise RemotePathError("no allowed remote roots are configured")

    if not any(is_relative_to(remote_path, root) for root in normalized_roots):
        raise RemotePathError("remote path is outside configured allowed roots")

    return remote_path


def normalize_remote_path(path: str | PurePosixPath) -> PurePosixPath:
    """Lexically normalize an absolute POSIX remote path."""

    raw_path = PurePosixPath(str(path).replace("\\", "/"))

    if not raw_path.is_absolute():
        raise RemotePathError("remote path must be absolute")

    parts: list[str] = []

    for part in raw_path.parts:
        if part in ("", "/", "."):
            continue

        if part == "..":
            if not parts:
                raise RemotePathError("remote path escapes the filesystem root")

            parts.pop()
            continue

        parts.append(part)

    if not parts:
        return PurePosixPath("/")

    return PurePosixPath("/" + "/".join(parts))


def is_relative_to(path: PurePosixPath, root: PurePosixPath) -> bool:
    """Return whether path is inside root using POSIX path parts."""

    return path == root or path.parts[: len(root.parts)] == root.parts
