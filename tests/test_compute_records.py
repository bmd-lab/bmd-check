"""Agent handling of the BMD Compute run-record contracts.

bmd_compute.submission v1 and bmd_compute.job_record v1 are owned by BMD
Compute. The v1 documents used here are vendored snapshots of Compute's
canonical generated fixtures (see fixtures/compute_run_records/SOURCE.md).
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from bmd_agent.resources.compute_records import (
    CONTRACT_LEGACY,
    CONTRACT_V1,
    INVALID_RECORD,
    LEGACY_JOB_RECORD_LIMITATION,
    LEGACY_SUBMISSION_LIMITATION,
    UNSUPPORTED_LEGACY,
    UNSUPPORTED_VERSION,
    ComputeRecordError,
    parse_job_record,
    parse_submission_record,
)
from bmd_agent.resources.job_resolution import (
    AMBIGUOUS,
    INVALID,
    RESOLVED,
    UNSUPPORTED_RECORD,
    resolve_bmd_compute_job,
)
from bmd_agent.resources.lifecycle import analyze_calculation_directory
from bmd_agent.resources.run import UnsupportedComputeRecordError, inspect_remote_run

import test_job_resolution as jr
import test_run_inspection as ri


FIXTURES = Path(__file__).parent / "fixtures" / "compute_run_records" / "v1"
CASES = ("single_stage_pbe_static", "hse06_soc_after_pbe_relax", "pbe_double_relax")
HISTORICAL_JOB = Path(__file__).parent / "fixtures" / "job_21906221.json"


def v1(case: str) -> tuple[dict, dict]:
    return (
        json.loads((FIXTURES / case / "submission.json").read_text(encoding="utf-8")),
        json.loads((FIXTURES / case / "job_record.json").read_text(encoding="utf-8")),
    )


def legacy(document: dict) -> dict:
    stripped = deepcopy(document)
    for key in ("schema", "schema_version", "attempt_id"):
        stripped.pop(key, None)
    return stripped


def assert_kind(parser, payload, kind):
    with pytest.raises(ComputeRecordError) as exc_info:
        parser(payload)
    assert exc_info.value.kind == kind
    return exc_info.value


# --- known current version ---------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_vendored_compute_v1_records_are_read_as_v1(case):
    submission, job_record = v1(case)

    record = parse_submission_record(submission)
    job = parse_job_record(job_record)

    assert record.contract == CONTRACT_V1 and record.limitations == ()
    assert record.label == "bmd_compute.submission v1"
    assert job.contract == CONTRACT_V1 and job.limitations == ()
    assert job.attempt_id == record.attempt_id == submission["submission"]["attempt_id"]
    assert job.run_dir == record.run_dir
    stage_count = len(record.workflow_stages)
    assert [index for index, _label, _path in record.stage_dirs] == (
        [] if stage_count == 1 else list(range(1, stage_count + 1))
    )


def test_double_relax_identifiers_map_to_stage_indices():
    record = parse_submission_record(v1("pbe_double_relax")[0])
    assert [(index, label) for index, label, _path in record.stage_dirs] == [
        (1, "relax_01"),
        (2, "relax_02"),
    ]


@pytest.mark.parametrize("case", ["hse06_soc_after_pbe_relax", "pbe_double_relax"])
def test_stage_mapping_ignores_json_key_order(case):
    submission = v1(case)[0]
    reordered = deepcopy(submission)
    reordered["paths"]["stage_dirs"] = dict(reversed(list(submission["paths"]["stage_dirs"].items())))

    assert parse_submission_record(reordered).stage_dirs == parse_submission_record(submission).stage_dirs


def test_v1_additive_and_non_contractual_changes_are_accepted():
    submission, job_record = v1("hse06_soc_after_pbe_relax")
    submission["future_optional_field"] = {"additive": True}
    for key in ("status", "modules", "runner", "potcar", "preflight"):
        submission.pop(key)
    minimal_job = {
        key: job_record[key]
        for key in ("schema", "schema_version", "job_id", "run_name", "run_dir", "attempt_id")
    }
    minimal_job["remote_state_path"] = "not checked for v1"

    assert parse_submission_record(submission).contract == CONTRACT_V1
    assert parse_job_record(minimal_job).contract == CONTRACT_V1


def test_record_status_fields_are_never_required_or_read():
    submission, job_record = v1("single_stage_pbe_static")
    job_record.pop("status")
    submission["status"] = "COMPLETED"
    submission["submission"]["submitted"] = True

    assert parse_job_record(job_record).contract == CONTRACT_V1
    assert parse_job_record(legacy(job_record)).contract == CONTRACT_LEGACY
    assert "status" not in vars(parse_job_record(job_record))
    assert "status" not in vars(parse_submission_record(submission))


# --- legacy unversioned records ----------------------------------------------------------


def test_historical_fixture_is_read_as_legacy_with_limitation():
    state = json.loads(HISTORICAL_JOB.read_text(encoding="utf-8"))

    job = parse_job_record(state)
    record = parse_submission_record(state["submission_spec"])

    assert job.contract == CONTRACT_LEGACY
    assert job.limitations == (LEGACY_JOB_RECORD_LIMITATION,)
    assert job.attempt_id == "acceptance-21906221"
    assert record.contract == CONTRACT_LEGACY
    assert record.limitations == (LEGACY_SUBMISSION_LIMITATION,)


def test_legacy_shape_before_submission_attempts_and_provenance_is_readable():
    # 2026-08-11 to 2026-08-17: workflow_spec and stage_dirs exist, but no
    # submission block, attempt-state record, or provenance.
    submission = legacy(v1("hse06_soc_after_pbe_relax")[0])
    for key in ("submission", "provenance"):
        submission.pop(key)
    submission["paths"].pop("submission_attempt_state")

    record = parse_submission_record(submission)

    assert record.contract == CONTRACT_LEGACY
    assert record.attempt_id is None and record.attempt_state is None
    assert len(record.stage_dirs) == 2


def test_legacy_single_stage_without_stage_dirs_is_readable():
    submission = legacy(v1("single_stage_pbe_static")[0])
    submission["paths"].pop("stage_dirs")

    assert parse_submission_record(submission).stage_dirs == ()


def test_pre_workflow_spec_legacy_record_is_explicitly_unsupported():
    # Before 2026-08-11 flow_spec carried only workflow/calculation_spec.
    submission = legacy(v1("hse06_soc_after_pbe_relax")[0])
    submission["flow_spec"] = {"workflow": "hse_static", "calculation_spec": {"purpose": "static"}}

    error = assert_kind(parse_submission_record, submission, UNSUPPORTED_LEGACY)
    assert "flow_spec.workflow_spec" in str(error)
    assert "not inferred" in str(error)


def test_legacy_stage_directories_must_be_mappable_identifiers():
    submission = legacy(v1("hse06_soc_after_pbe_relax")[0])
    paths = submission["paths"]["stage_dirs"]
    submission["paths"]["stage_dirs"] = {"alpha": paths["stage_01"], "beta": paths["stage_02"]}

    assert_kind(parse_submission_record, submission, INVALID_RECORD)


# --- unsupported future version ------------------------------------------------------------


@pytest.mark.parametrize("version", [2, 99])
def test_future_versions_are_unsupported_not_invalid_and_not_legacy(version):
    submission, job_record = v1("hse06_soc_after_pbe_relax")
    submission["schema_version"] = version
    job_record["schema_version"] = version
    # A future version may legitimately drop v1 fields; that must not turn
    # into "malformed" or be re-read by the legacy reader.
    submission.pop("flow_spec")
    job_record.pop("run_dir")

    assert_kind(parse_submission_record, submission, UNSUPPORTED_VERSION)
    assert_kind(parse_job_record, job_record, UNSUPPORTED_VERSION)


# --- malformed known version -----------------------------------------------------------------


SUBMISSION_BREAKS = {
    "wrong schema id": lambda d: d.update(schema="bmd_compute.job_record"),
    "non-string schema": lambda d: d.update(schema=1),
    "boolean version": lambda d: d.update(schema_version=True),
    "string version": lambda d: d.update(schema_version="1"),
    "version without schema": lambda d: d.pop("schema"),
    "missing run_dir": lambda d: d["paths"].pop("run_dir"),
    "missing attempt_id": lambda d: d["submission"].pop("attempt_id"),
    "missing stage_dirs": lambda d: d["paths"].pop("stage_dirs"),
    "missing stage": lambda d: d["paths"]["stage_dirs"].pop("stage_02"),
    "duplicate stage": lambda d: d["paths"]["stage_dirs"].update(relax_01="/x"),
    "stage beyond workflow": lambda d: d["paths"]["stage_dirs"].update(stage_03="/x"),
    "unmappable identifier": lambda d: d["paths"]["stage_dirs"].update({"producer-alpha": "/x"}),
    "empty stages": lambda d: d["flow_spec"]["workflow_spec"].update(stages=[]),
    "bad resources": lambda d: d["resources"].update(mem_gb="128"),
}


@pytest.mark.parametrize("name", sorted(SUBMISSION_BREAKS))
def test_malformed_v1_submission_is_invalid(name):
    submission = v1("hse06_soc_after_pbe_relax")[0]
    SUBMISSION_BREAKS[name](submission)
    assert_kind(parse_submission_record, submission, INVALID_RECORD)


JOB_RECORD_BREAKS = {
    "missing attempt_id": lambda d: d.pop("attempt_id"),
    "missing run_name": lambda d: d.pop("run_name"),
    "empty job_id": lambda d: d.update(job_id=""),
    "bad submitted_at": lambda d: d.update(submitted_at=5),
    "wrong schema id": lambda d: d.update(schema="bmd_compute.submission"),
}


@pytest.mark.parametrize("name", sorted(JOB_RECORD_BREAKS))
def test_malformed_v1_job_record_is_invalid(name):
    job_record = v1("hse06_soc_after_pbe_relax")[1]
    JOB_RECORD_BREAKS[name](job_record)
    assert_kind(parse_job_record, job_record, INVALID_RECORD)


# --- job resolution end to end --------------------------------------------------------------


def v1_remote(case="hse06_soc_after_pbe_relax", *, submission=None, job_record=None, attempt=None):
    base_submission, base_job = v1(case)
    submission = base_submission if submission is None else submission
    job_record = base_job if job_record is None else job_record
    run_dir = base_submission["paths"]["run_dir"]
    attempt_path = base_submission["submission"]["attempt_state"]
    attempt = attempt if attempt is not None else {
        "version": 1,
        "attempt_id": base_submission["submission"]["attempt_id"],
        "state": "SUBMITTED",
        "job_id": base_job["job_id"],
        "run_dir": run_dir,
    }
    files = {
        f"/bmd-db/guest/logs/job_{base_job['job_id']}.json": job_record,
        f"{run_dir}/submission.json": submission,
        attempt_path: attempt,
    }
    encoded = {
        path: payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        for path, payload in files.items()
    }
    return jr.RemoteFiles(encoded, {run_dir}), base_job["job_id"]


def resolve_v1(remote, job_id):
    resource = jr.cluster()
    return resolve_bmd_compute_job(resource, jr.deployment(resource), job_id, runner=remote)


def test_v1_records_resolve_without_legacy_limitations():
    remote, job_id = v1_remote()
    result = resolve_v1(remote, job_id)

    assert result.resolution_status == RESOLVED
    assert result.job_record_contract == "bmd_compute.job_record v1"
    assert result.submission_record_contract == "bmd_compute.submission v1"
    assert result.submission_attempt_id == v1("hse06_soc_after_pbe_relax")[0]["submission"]["attempt_id"]
    assert not any("legacy" in item for item in result.limitations)
    assert any("not a scheduler lifecycle record" in item for item in result.limitations)


def test_legacy_records_resolve_with_legacy_limitations():
    submission, job_record = v1("hse06_soc_after_pbe_relax")
    remote, job_id = v1_remote(submission=legacy(submission), job_record=legacy(job_record))
    result = resolve_v1(remote, job_id)

    assert result.resolution_status == RESOLVED
    assert "legacy unversioned" in result.job_record_contract
    assert LEGACY_JOB_RECORD_LIMITATION in result.limitations
    assert LEGACY_SUBMISSION_LIMITATION in result.limitations


def test_legacy_job_record_without_status_resolves():
    state = jr.producer_state()
    state.pop("status")
    assert jr.resolve(jr.remote_payloads(state=state)).resolution_status == RESOLVED


@pytest.mark.parametrize("which", ["job_record", "submission"])
def test_future_version_resolves_to_unsupported_record(which):
    submission, job_record = v1("hse06_soc_after_pbe_relax")
    target = job_record if which == "job_record" else submission
    target["schema_version"] = 2
    remote, job_id = v1_remote(submission=submission, job_record=job_record)

    result = resolve_v1(remote, job_id)

    assert result.resolution_status == UNSUPPORTED_RECORD
    assert "schema_version 2 is not supported" in result.reason


def test_pre_workflow_spec_submission_resolves_to_unsupported_record():
    submission = legacy(v1("hse06_soc_after_pbe_relax")[0])
    submission["flow_spec"] = {"workflow": "hse_static"}
    remote, job_id = v1_remote(submission=submission)

    result = resolve_v1(remote, job_id)

    assert result.resolution_status == UNSUPPORTED_RECORD
    assert "unsupported legacy" in result.reason


def test_malformed_v1_job_record_resolves_to_invalid():
    job_record = v1("hse06_soc_after_pbe_relax")[1]
    job_record.pop("attempt_id")
    remote, job_id = v1_remote(job_record=job_record)

    assert resolve_v1(remote, job_id).resolution_status == INVALID


def test_v1_attempt_id_disagreement_is_ambiguous():
    job_record = v1("hse06_soc_after_pbe_relax")[1]
    job_record["attempt_id"] = "00000000-0000-4000-8000-00000000ffff"
    remote, job_id = v1_remote(job_record=job_record)

    assert resolve_v1(remote, job_id).resolution_status == AMBIGUOUS


def test_v1_job_record_status_does_not_affect_resolution():
    job_record = v1("hse06_soc_after_pbe_relax")[1]
    job_record["status"] = "COMPLETED"
    remote, job_id = v1_remote(job_record=job_record)
    completed = resolve_v1(remote, job_id)
    job_record.pop("status")
    remote, job_id = v1_remote(job_record=job_record)
    absent = resolve_v1(remote, job_id)

    assert completed.resolution_status == absent.resolution_status == RESOLVED
    assert completed.limitations == absent.limitations


# --- run inspection and local lifecycle -----------------------------------------------------


def test_remote_run_stage_bindings_follow_identifiers_not_key_order():
    payload = ri.submission_payload()
    payload["paths"]["stage_dirs"] = dict(reversed(list(payload["paths"]["stage_dirs"].items())))
    files = ri.default_files()
    files[f"{ri.FLOW_ROOT}/submission.json"] = json.dumps(payload).encode("utf-8")
    remote = ri.RemoteFixture(files=files, directories=ri.default_directories())

    inspection = inspect_remote_run(
        ri.cluster(),
        ri.FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=ri.slurm_runner,
        scientific_parser=ri.fake_scientific_parser,
        modifier_policies=(ri.modifier_policy(),),
    )

    assert [item.label for item in inspection.stage_directories] == [
        "stage_01",
        "stage_02",
        "stage_03",
        "stage_04",
    ]
    inputs = {item.label: item for item in inspection.executed_inputs}
    assert inputs["stage_01"].stage_index == 1
    assert inputs["stage_04"].stage_index == 4
    assert inspection.submission_record.contract == "bmd_compute.submission (legacy unversioned)"


def test_remote_run_with_future_submission_version_is_unsupported():
    payload = ri.submission_payload()
    payload["schema"] = "bmd_compute.submission"
    payload["schema_version"] = 2
    files = ri.default_files()
    files[f"{ri.FLOW_ROOT}/submission.json"] = json.dumps(payload).encode("utf-8")
    remote = ri.RemoteFixture(files=files, directories=ri.default_directories())

    with pytest.raises(UnsupportedComputeRecordError, match="schema_version 2"):
        inspect_remote_run(
            ri.cluster(),
            ri.FLOW_ROOT,
            remote_runner=remote,
            slurm_runner=ri.slurm_runner,
            scientific_parser=ri.fake_scientific_parser,
            modifier_policies=(ri.modifier_policy(),),
        )


def write_local_run(root: Path, submission: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "submission.json").write_text(json.dumps(submission), encoding="utf-8")


def local_v1_submission(root: Path, case: str) -> dict:
    submission = v1(case)[0]
    producer_root = submission["paths"]["run_dir"]
    submission["paths"] = {
        **submission["paths"],
        "run_dir": str(root),
        "result_dir": submission["paths"]["result_dir"].replace(producer_root, str(root)),
        "stage_dirs": {
            key: value.replace(producer_root, str(root))
            for key, value in submission["paths"]["stage_dirs"].items()
        },
    }
    submission["submission"]["attempt_state"] = str(root / "attempt.json")
    submission["paths"]["submission_attempt_state"] = str(root / "attempt.json")
    return submission


def test_local_v1_double_relax_binds_stages_by_identifier(tmp_path):
    submission = local_v1_submission(tmp_path, "pbe_double_relax")
    submission["paths"]["stage_dirs"] = dict(reversed(list(submission["paths"]["stage_dirs"].items())))
    write_local_run(tmp_path, submission)
    for label in ("relax_01", "relax_02"):
        (tmp_path / label).mkdir()

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.record_contract == "bmd_compute.submission v1"
    assert [(b.label, b.stage_index) for b in analysis.bmd_workflow.stage_bindings] == [
        ("relax_01", 1),
        ("relax_02", 2),
    ]
    assert not any("legacy" in item for item in analysis.limitations)


def test_local_legacy_record_carries_legacy_limitation(tmp_path):
    write_local_run(tmp_path, legacy(local_v1_submission(tmp_path, "single_stage_pbe_static")))

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.bmd_workflow is not None
    assert LEGACY_SUBMISSION_LIMITATION in analysis.limitations


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d.update(schema_version=2), "schema_version 2 is not supported"),
        (lambda d: (d.pop("schema"), d.pop("schema_version"), d.update(flow_spec={"workflow": "static"})), "unsupported legacy"),
        (lambda d: d["paths"].pop("run_dir"), "paths.run_dir"),
    ],
)
def test_local_unread_submission_is_reported_not_silently_ignored(tmp_path, mutate, expected):
    submission = local_v1_submission(tmp_path, "single_stage_pbe_static")
    mutate(submission)
    write_local_run(tmp_path, submission)

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.bmd_workflow is None
    assert any("submission.json was not used" in item and expected in item for item in analysis.limitations)
