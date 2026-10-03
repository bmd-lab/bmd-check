from __future__ import annotations

from copy import deepcopy
import json
from pathlib import PurePosixPath
import shlex
import subprocess

import pytest

from bmd_agent.config import SlurmClusterResource
from bmd_agent.deployment import DeploymentContext, load_deployment_profile
from bmd_agent.resources.job_resolution import (
    AMBIGUOUS,
    INVALID,
    NOT_BMD_COMPUTE,
    RESOLVED,
    UNAVAILABLE,
    resolve_bmd_compute_job,
)


JOB_ID = "21906221"
RUN_DIR = "/bmd-db/guest/flows/vasp_run_hse_static-20260917-211618"
STATE_PATH = f"/bmd-db/guest/logs/job_{JOB_ID}.json"
ATTEMPT_PATH = "/bmd-db/guest/logs/submission_attempts/acceptance-21906221.json"


class RemoteFiles:
    def __init__(self, files: dict[str, bytes], directories: set[str]) -> None:
        self.files = files
        self.directories = directories
        self.commands: list[str] = []

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert command[:4] == ["ssh", "-o", "BatchMode=yes", "powerslurm-bmdguest"]
        assert kwargs["capture_output"] is True
        assert kwargs["timeout"] == 20
        remote_command = command[-1]
        self.commands.append(remote_command)
        parts = shlex.split(remote_command)

        if parts[:2] == ["test", "-f"]:
            assert kwargs["check"] is False
            return subprocess.CompletedProcess(command, 0 if parts[2] in self.files else 1)
        if parts[:2] == ["test", "-d"]:
            assert kwargs["check"] is False
            return subprocess.CompletedProcess(command, 0 if parts[2] in self.directories else 1)
        if parts[:4] == ["stat", "-c", "%s", "--"]:
            assert kwargs["check"] is True
            path = parts[4]
            if path not in self.files:
                raise subprocess.CalledProcessError(1, command, stderr=b"missing")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{len(self.files[path])}\n".encode(),
                stderr=b"",
            )
        if parts[:2] == ["cat", "--"]:
            assert kwargs["check"] is True
            path = parts[2]
            if path not in self.files:
                raise subprocess.CalledProcessError(1, command, stderr=b"missing")
            return subprocess.CompletedProcess(command, 0, stdout=self.files[path], stderr=b"")
        raise AssertionError(f"unexpected remote command: {remote_command}")


def cluster(*, roots: tuple[PurePosixPath, ...] | None = None) -> SlurmClusterResource:
    return SlurmClusterResource(
        key="powerslurm",
        name="PowerSLURM",
        ssh_host="powerslurm-bmdguest",
        partition="leeburton-pool",
        access="observational",
        allowed_remote_roots=roots
        or (PurePosixPath("/bmd-db/guest/flows"), PurePosixPath("/bmd-db/guest/logs")),
        deployment_profile="power",
    )


def deployment(resource: SlurmClusterResource) -> DeploymentContext:
    return DeploymentContext(profile=load_deployment_profile("power"), cluster=resource)


def submission() -> dict:
    return {
        "run_name": "vasp_run_hse_static-20260917-211618",
        "flow_spec": {
            "workflow_spec": {
                "stages": [
                    {
                        "stage_type": "static",
                        "theory": "hse06",
                        "modifiers": ["soc"],
                        "label": None,
                        "options": {},
                    }
                ]
            }
        },
        "paths": {
            "run_dir": RUN_DIR,
            "result_dir": RUN_DIR,
            "stage_dirs": {},
            "submission_attempt_state": ATTEMPT_PATH,
        },
        "submission": {
            "attempt_id": "acceptance-21906221",
            "attempt_state": ATTEMPT_PATH,
        },
    }


def producer_state(spec: dict | None = None) -> dict:
    spec = deepcopy(spec or submission())
    return {
        "job_id": JOB_ID,
        "run_name": spec["run_name"],
        "run_dir": RUN_DIR,
        "remote_script": f"{RUN_DIR}.sbatch.sh",
        "log_paths": {},
        "cluster": {"partition": "leeburton-pool"},
        "resources": {"nodes": 1, "ntasks": 24, "mem_gb": 192},
        "submitted_at": "2026-09-17T21:16:20+03:00",
        "status": "submitted",
        "submission_spec": spec,
        "remote_state_path": STATE_PATH,
    }


def attempt_state() -> dict:
    return {
        "attempt_id": "acceptance-21906221",
        "state": "SUBMITTED",
        "job_id": JOB_ID,
        "run_dir": RUN_DIR,
        "job_record": {
            "job_id": JOB_ID,
            "run_dir": RUN_DIR,
            "submission_spec": submission(),
        },
    }


def remote_payloads(
    *,
    state: dict | bytes | None = None,
    spec: dict | bytes | None = None,
    attempt: dict | bytes | None = None,
    include_directory: bool = True,
) -> RemoteFiles:
    files: dict[str, bytes] = {}
    for path, payload in (
        (STATE_PATH, producer_state() if state is None else state),
        (f"{RUN_DIR}/submission.json", submission() if spec is None else spec),
        (ATTEMPT_PATH, attempt_state() if attempt is None else attempt),
    ):
        files[path] = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return RemoteFiles(files, {RUN_DIR} if include_directory else set())


def resolve(remote: RemoteFiles, *, resource: SlurmClusterResource | None = None):
    resource = resource or cluster()
    return resolve_bmd_compute_job(
        resource,
        deployment(resource),
        JOB_ID,
        runner=remote,
    )


def test_exact_job_state_resolves_and_cross_checks_submission_attempt() -> None:
    remote = remote_payloads()

    result = resolve(remote)

    assert result.resolution_status == RESOLVED
    assert result.run_directory == RUN_DIR
    assert result.producer_state_path == STATE_PATH
    assert result.submission_attempt_id == "acceptance-21906221"
    assert remote.commands[0] == f"test -f {STATE_PATH}"
    assert not any("find " in command or "ls " in command for command in remote.commands)
    assert not any("POTCAR" in command for command in remote.commands)


def test_missing_producer_state_is_nonfatal_not_bmd_compute() -> None:
    remote = RemoteFiles({}, set())

    result = resolve(remote)

    assert result.resolution_status == NOT_BMD_COMPUTE
    assert result.run_directory is None
    assert remote.commands == [f"test -f {STATE_PATH}"]


@pytest.mark.parametrize(
    ("state", "expected_status", "reason"),
    (
        (b"{not-json", INVALID, "malformed JSON"),
        ({**producer_state(), "job_id": "21906222"}, AMBIGUOUS, "job IDs conflict"),
        ({**producer_state(), "run_dir": "relative/run"}, INVALID, "must be absolute"),
        ({**producer_state(), "run_dir": "/etc/private"}, INVALID, "not authorized"),
    ),
)
def test_invalid_or_conflicting_producer_state_is_not_guessed(
    state: dict | bytes,
    expected_status: str,
    reason: str,
) -> None:
    result = resolve(remote_payloads(state=state))

    assert result.resolution_status == expected_status
    assert reason in (result.reason or "")


def test_missing_resolved_directory_is_unavailable() -> None:
    result = resolve(remote_payloads(include_directory=False))

    assert result.resolution_status == UNAVAILABLE
    assert result.run_directory == RUN_DIR
    assert "not a readable directory" in (result.reason or "")


def test_missing_submission_is_unavailable_without_guessing_another_run() -> None:
    remote = remote_payloads()
    remote.files.pop(f"{RUN_DIR}/submission.json")

    result = resolve(remote)

    assert result.resolution_status == UNAVAILABLE
    assert "no readable submission.json" in (result.reason or "")
    assert not any("find " in command or "ls " in command for command in remote.commands)


def test_malformed_submission_is_invalid() -> None:
    result = resolve(remote_payloads(spec=b"{not-json"))

    assert result.resolution_status == INVALID
    assert "malformed JSON" in (result.reason or "")


def test_conflicting_submission_run_directory_is_ambiguous() -> None:
    spec = submission()
    spec["paths"]["run_dir"] = "/bmd-db/guest/flows/different-run"

    result = resolve(remote_payloads(spec=spec))

    assert result.resolution_status == AMBIGUOUS
    assert "run directories conflict" in (result.reason or "")


def test_job_id_in_submission_json_is_not_read() -> None:
    # submission.json is written before sbatch; BMD Compute never records a
    # job ID there, so Agent no longer looks for one.
    spec = submission()
    spec["job_id"] = "21906222"
    spec["submission"]["job_id"] = "21906223"

    result = resolve(remote_payloads(spec=spec))

    assert result.resolution_status == RESOLVED


def test_attempt_state_job_id_conflict_is_ambiguous() -> None:
    attempt = attempt_state()
    attempt["job_id"] = "21906222"

    result = resolve(remote_payloads(attempt=attempt))

    assert result.resolution_status == AMBIGUOUS
    assert "job IDs conflict" in (result.reason or "")


def test_attempt_id_conflict_is_ambiguous() -> None:
    attempt = attempt_state()
    attempt["attempt_id"] = "different-attempt"

    result = resolve(remote_payloads(attempt=attempt))

    assert result.resolution_status == AMBIGUOUS
    assert "attempt IDs conflict" in (result.reason or "")


def test_missing_historical_attempt_state_is_a_resolved_limitation() -> None:
    spec = submission()
    spec.pop("submission")
    spec["paths"].pop("submission_attempt_state")
    state = producer_state(spec)
    remote = remote_payloads(state=state, spec=spec)
    remote.files.pop(ATTEMPT_PATH)

    result = resolve(remote)

    assert result.resolution_status == RESOLVED
    assert result.submission_attempt_id is None
    assert "submission attempt-state path is unavailable" in result.limitations


def test_configured_profile_paths_do_not_expand_allowed_roots() -> None:
    resource = cluster(roots=(PurePosixPath("/bmd-db/guest/flows"),))
    remote = RemoteFiles({}, set())

    result = resolve_bmd_compute_job(
        resource,
        deployment(resource),
        JOB_ID,
        runner=remote,
    )

    assert result.resolution_status == INVALID
    assert "not authorized" in (result.reason or "")
    assert remote.commands == []


def test_absent_deployment_profile_is_resolution_unavailable_without_remote_reads() -> None:
    resource = cluster()
    remote = RemoteFiles({}, set())

    result = resolve_bmd_compute_job(resource, None, JOB_ID, runner=remote)

    assert result.resolution_status == UNAVAILABLE
    assert remote.commands == []


def test_producer_state_record_read_is_bounded() -> None:
    remote = remote_payloads(state=b"{" + b" " * (1024 * 1024))

    result = resolve(remote)

    assert result.resolution_status == INVALID
    assert "read limit" in (result.reason or "")
    assert f"cat -- {STATE_PATH}" not in remote.commands
