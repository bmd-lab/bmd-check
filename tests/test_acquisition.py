from __future__ import annotations

import base64
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess

import pytest

from bmd_agent.resources import observations as observation_operations
from bmd_agent.profiling import PerformanceProfiler, profiled_runner
from bmd_agent.resources.vasp import (
    RemoteAcquisitionRequest,
    RemotePathError,
    prime_remote_acquisition,
    probe_remote_error_archives,
    remote_acquisition_cache,
    remote_directory_exists,
    remote_file_exists,
    remote_file_size,
    retrieve_remote_file,
)


ROOT = PurePosixPath("/allowed/run")


class AcquisitionRunner:
    def __init__(
        self,
        *,
        files: dict[str, bytes],
        directories: set[str],
        corrupt_read_index: int | None = None,
    ) -> None:
        self.files = files
        self.directories = directories
        self.corrupt_read_index = corrupt_read_index
        self.commands: list[list[str]] = []

    def __call__(
        self,
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        self.commands.append(command)
        remote_command = command[-1]
        parts = shlex.split(remote_command)
        if parts[:4] == ["sh", "-s", "--", "bmd-agent-acquisition-v1"]:
            assert kwargs["capture_output"] is True
            assert kwargs["check"] is True
            assert kwargs["timeout"] == 20
            script = kwargs["input"]
            assert isinstance(script, bytes)
            assert b"find " not in script
            assert b"ls " not in script
            assert b"tar " not in script
            total_limit = int(parts[4])
            request_count = int(parts[5])
            request_parts = parts[8:]
            assert len(request_parts) == request_count * 4
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
                if index == self.corrupt_read_index:
                    encoded = "not-valid-base64!"
                lines.append(f"item\t{index}\tfile\tread\t{size}\t{encoded}")
                used += size
            stdout = ("\n".join(lines) + "\n").encode("ascii")
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

        if parts[:2] == ["cat", "--"]:
            contents = self.files.get(parts[2])
            if contents is None:
                raise subprocess.CalledProcessError(1, command, stderr=b"missing")
            return subprocess.CompletedProcess(command, 0, stdout=contents, stderr=b"")
        if parts[:2] == ["test", "-f"]:
            return subprocess.CompletedProcess(
                command,
                0 if parts[2] in self.files else 1,
                stdout=b"",
                stderr=b"",
            )
        if parts[:2] == ["test", "-d"]:
            return subprocess.CompletedProcess(
                command,
                0 if parts[2] in self.directories else 1,
                stdout=b"",
                stderr=b"",
            )
        if parts[:4] == ["stat", "-c", "%s", "--"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{len(self.files[parts[4]])}\n".encode("ascii"),
                stderr=b"",
            )
        raise AssertionError(f"unexpected command: {remote_command}")


def test_one_manifest_replaces_repeated_exists_stat_and_read_operations() -> None:
    path = ROOT / "OSZICAR"
    runner = AcquisitionRunner(files={str(path): b"one\ntwo\n"}, directories={str(ROOT)})

    with remote_acquisition_cache("host", (PurePosixPath("/allowed"),)):
        prime_remote_acquisition(
            "host",
            (
                RemoteAcquisitionRequest(ROOT, kind="directory"),
                RemoteAcquisitionRequest(path, read_limit=100),
            ),
            runner=runner,
            timeout=20,
        )
        assert remote_directory_exists("host", ROOT, runner=runner)
        assert remote_file_exists("host", path, runner=runner)
        assert remote_file_exists("host", path, runner=runner)
        assert remote_file_size("host", path, runner=runner) == 8
        assert retrieve_remote_file("host", path, runner=runner) == b"one\ntwo\n"
        assert retrieve_remote_file("host", path, runner=runner) == b"one\ntwo\n"

    assert len(runner.commands) == 1


def test_batched_reads_preserve_binary_boundaries_and_missing_files() -> None:
    first = ROOT / "submission.json"
    second = ROOT / "OSZICAR"
    missing = ROOT / "custodian.json"
    runner = AcquisitionRunner(
        files={str(first): b'{"value":"a\\tb"}\n', str(second): b"line\x00two\n"},
        directories={str(ROOT)},
    )

    with remote_acquisition_cache("host", (PurePosixPath("/allowed"),)):
        prime_remote_acquisition(
            "host",
            tuple(
                RemoteAcquisitionRequest(path, read_limit=100)
                for path in (first, second, missing)
            ),
            runner=runner,
            timeout=20,
        )
        assert retrieve_remote_file("host", first, runner=runner) == runner.files[str(first)]
        assert retrieve_remote_file("host", second, runner=runner) == runner.files[str(second)]
        assert remote_file_exists("host", missing, runner=runner) is False

    assert len(runner.commands) == 1


def test_malformed_one_file_does_not_corrupt_neighboring_batched_result() -> None:
    first = ROOT / "INCAR"
    second = ROOT / "OSZICAR"
    runner = AcquisitionRunner(
        files={str(first): b"ENCUT=520\n", str(second): b"DAV: 1\n"},
        directories={str(ROOT)},
        corrupt_read_index=1,
    )

    with remote_acquisition_cache("host", (PurePosixPath("/allowed"),)):
        prime_remote_acquisition(
            "host",
            (
                RemoteAcquisitionRequest(first, read_limit=100),
                RemoteAcquisitionRequest(second, read_limit=100),
            ),
            runner=runner,
            timeout=20,
        )
        assert retrieve_remote_file("host", second, runner=runner) == b"DAV: 1\n"
        assert retrieve_remote_file("host", first, runner=runner) == b"ENCUT=520\n"

    assert len(runner.commands) == 2
    assert shlex.split(runner.commands[-1][-1])[:2] == ["cat", "--"]


def test_per_file_limit_defers_to_existing_exact_read_path() -> None:
    path = ROOT / "OSZICAR"
    runner = AcquisitionRunner(files={str(path): b"123456"}, directories={str(ROOT)})

    with remote_acquisition_cache("host", (PurePosixPath("/allowed"),)):
        prime_remote_acquisition(
            "host",
            (RemoteAcquisitionRequest(path, read_limit=5),),
            runner=runner,
            timeout=20,
        )
        assert remote_file_size("host", path, runner=runner) == 6
        assert retrieve_remote_file("host", path, runner=runner) == b"123456"

    assert len(runner.commands) == 2


def test_manifest_rejects_unauthorized_path_before_remote_execution() -> None:
    runner = AcquisitionRunner(files={}, directories=set())

    with remote_acquisition_cache("host", (PurePosixPath("/allowed"),)):
        with pytest.raises(RemotePathError, match="outside configured allowed roots"):
            prime_remote_acquisition(
                "host",
                (RemoteAcquisitionRequest(PurePosixPath("/etc/passwd"), read_limit=100),),
                runner=runner,
                timeout=20,
            )

    assert runner.commands == []


def test_exact_authorized_path_is_positional_data_not_remote_shell_source() -> None:
    path = ROOT / "name; touch should-not-run"
    runner = AcquisitionRunner(files={str(path): b"safe"}, directories={str(ROOT)})

    with remote_acquisition_cache("host", (PurePosixPath("/allowed"),)):
        prime_remote_acquisition(
            "host",
            (RemoteAcquisitionRequest(path, read_limit=100),),
            runner=runner,
            timeout=20,
        )
        assert retrieve_remote_file("host", path, runner=runner) == b"safe"

    parts = shlex.split(runner.commands[0][-1])
    assert parts[-1] == str(path)
    assert "touch" not in parts[:-1]


def test_archive_manifest_is_bounded_and_never_reads_archive_contents() -> None:
    files = {
        str(ROOT / "error.1.tar.gz"): b"archive-one",
        str(ROOT / "error.2.tar.gz"): b"archive-two",
        str(ROOT / "error.4.tar.gz"): b"not reported after the first gap",
    }
    runner = AcquisitionRunner(files=files, directories={str(ROOT)})

    with remote_acquisition_cache("host", (PurePosixPath("/allowed"),)):
        paths = probe_remote_error_archives(
            "host",
            ROOT,
            allowed_roots=(PurePosixPath("/allowed"),),
            limit=64,
            runner=runner,
            timeout=20,
        )

    assert paths == (ROOT / "error.1.tar.gz", ROOT / "error.2.tar.gz")
    assert len(runner.commands) == 1
    parts = shlex.split(runner.commands[0][-1])
    assert int(parts[5]) == 1
    assert int(parts[6]) == 0
    assert int(parts[7]) == 1
    assert all("POTCAR" not in part for part in parts)


def test_cache_is_invocation_local() -> None:
    path = ROOT / "INCAR"
    runner = AcquisitionRunner(files={str(path): b"ENCUT=520\n"}, directories={str(ROOT)})

    with remote_acquisition_cache("host", (PurePosixPath("/allowed"),)):
        prime_remote_acquisition(
            "host",
            (RemoteAcquisitionRequest(path),),
            runner=runner,
            timeout=20,
        )
        assert remote_file_exists("host", path, runner=runner)
    assert remote_file_exists("host", path, runner=runner)

    assert len(runner.commands) == 2
    assert shlex.split(runner.commands[-1][-1])[:2] == ["test", "-f"]


def test_failed_batch_degrades_to_existing_exact_operations() -> None:
    path = ROOT / "INCAR"
    base_runner = AcquisitionRunner(
        files={str(path): b"ENCUT=520\n"},
        directories={str(ROOT)},
    )

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if "bmd-agent-acquisition-v1" in command[-1]:
            base_runner.commands.append(command)
            raise subprocess.CalledProcessError(1, command, stderr=b"helper unavailable")
        return base_runner(command, **kwargs)

    with remote_acquisition_cache("host", (PurePosixPath("/allowed"),)):
        prime_remote_acquisition(
            "host",
            (RemoteAcquisitionRequest(path, read_limit=100),),
            runner=runner,
            timeout=20,
        )
        assert remote_file_exists("host", path, runner=runner)
        assert retrieve_remote_file("host", path, runner=runner) == b"ENCUT=520\n"

    assert len(base_runner.commands) == 3
    assert shlex.split(base_runner.commands[1][-1])[:2] == ["test", "-f"]
    assert shlex.split(base_runner.commands[2][-1])[:2] == ["cat", "--"]


def test_profile_distinguishes_physical_batch_and_logical_file_operations() -> None:
    first = ROOT / "INCAR"
    second = ROOT / "OSZICAR"
    base_runner = AcquisitionRunner(
        files={str(first): b"ENCUT=520\n", str(second): b"DAV: 1\n"},
        directories={str(ROOT)},
    )
    profiler = PerformanceProfiler()

    with profiler.activate():
        runner = profiled_runner(base_runner, role="remote")
        with remote_acquisition_cache("host", (PurePosixPath("/allowed"),)):
            prime_remote_acquisition(
                "host",
                (
                    RemoteAcquisitionRequest(first, read_limit=100),
                    RemoteAcquisitionRequest(second, read_limit=100),
                ),
                runner=runner,
                timeout=20,
            )
    counts = profiler.snapshot().operations.counts

    assert counts["ssh_exec_channels"] == 1
    assert counts["remote_commands"] == 1
    assert counts["metadata_manifest_operations"] == 1
    assert counts["batched_file_read_operations"] == 1
    assert counts["logical_files_described"] == 2
    assert counts["logical_files_read"] == 2


def test_profiling_adds_no_acquisition_operations() -> None:
    path = ROOT / "INCAR"

    def acquire(runner) -> None:
        with remote_acquisition_cache("host", (PurePosixPath("/allowed"),)):
            prime_remote_acquisition(
                "host",
                (RemoteAcquisitionRequest(path, read_limit=100),),
                runner=runner,
                timeout=20,
            )
            assert retrieve_remote_file("host", path, runner=runner) == b"ENCUT=520\n"

    normal = AcquisitionRunner(files={str(path): b"ENCUT=520\n"}, directories={str(ROOT)})
    acquire(normal)

    measured = AcquisitionRunner(files={str(path): b"ENCUT=520\n"}, directories={str(ROOT)})
    profiler = PerformanceProfiler()
    with profiler.activate():
        acquire(profiled_runner(measured, role="remote"))

    assert [command[-1] for command in measured.commands] == [
        command[-1] for command in normal.commands
    ]


def test_fixed_acquisition_protocol_runs_with_system_sh_when_available(
    tmp_path: Path,
) -> None:
    if os.name == "nt" or shutil.which("sh") is None:
        pytest.skip("system POSIX shell is not available in this environment")
    root = PurePosixPath(tmp_path.as_posix())
    path = root / "OSZICAR"
    (tmp_path / "OSZICAR").write_bytes(b"DAV: 1\n")
    commands: list[list[str]] = []
    results: list[subprocess.CompletedProcess[bytes]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        result = subprocess.run(shlex.split(command[-1]), **kwargs)
        results.append(result)
        return result

    with remote_acquisition_cache("host", (root,)) as cache:
        prime_remote_acquisition(
            "host",
            (RemoteAcquisitionRequest(path, read_limit=100),),
            runner=runner,
            timeout=20,
        )
        assert retrieve_remote_file("host", path, runner=runner) == b"DAV: 1\n"
        assert cache._disabled is False

    assert len(commands) == 1
    assert results[0].stderr == b""
    lines = results[0].stdout.decode("ascii").splitlines()
    assert lines[0] == "schema\tbmd-agent-acquisition-v1"
    assert lines[1].startswith("item\t1\tfile\tread\t")
    assert not any(line.startswith("item\t8000000\t") for line in lines)


def test_fixed_acquisition_protocol_enforces_aggregate_limit_with_system_sh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt" or shutil.which("sh") is None:
        pytest.skip("system POSIX shell is not available in this environment")
    monkeypatch.setattr(observation_operations, "_MAX_BATCH_TOTAL_BYTES", 5)
    root = PurePosixPath(tmp_path.as_posix())
    first = root / "first"
    second = root / "second"
    (tmp_path / "first").write_bytes(b"aaaa")
    (tmp_path / "second").write_bytes(b"bbbb")
    commands: list[list[str]] = []
    results: list[subprocess.CompletedProcess[bytes]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        result = subprocess.run(shlex.split(command[-1]), **kwargs)
        results.append(result)
        return result

    with remote_acquisition_cache("host", (root,)) as cache:
        prime_remote_acquisition(
            "host",
            (
                RemoteAcquisitionRequest(first, read_limit=4),
                RemoteAcquisitionRequest(second, read_limit=4),
            ),
            runner=runner,
            timeout=20,
        )
        first_record = cache.record(first, "file")
        second_record = cache.record(second, "file")
        assert first_record is not None and first_record.contents == b"aaaa"
        assert second_record is not None and second_record.contents is None
        assert second_record.content_status == "deferred"
        assert cache._disabled is False

    assert len(commands) == 1
    assert results[0].stderr == b""
    lines = results[0].stdout.decode("ascii").splitlines()
    assert lines[1].startswith("item\t1\tfile\tread\t4\t")
    assert lines[2] == "item\t2\tfile\tdeferred\t4\t"


def test_fixed_acquisition_protocol_rejects_wrong_marker_with_system_sh() -> None:
    if os.name == "nt" or shutil.which("sh") is None:
        pytest.skip("system POSIX shell is not available in this environment")

    result = subprocess.run(
        ["sh", "-s", "--", "wrong-marker", "5", "0", "0", "0"],
        input=observation_operations._ACQUISITION_SCRIPT.encode("ascii"),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr == b""
