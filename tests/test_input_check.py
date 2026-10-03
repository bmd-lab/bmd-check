import json
from pathlib import Path
import shlex
import subprocess
from typing import Any

import pytest

from bmd_agent import cli
from bmd_agent.config import GitRepositoryResource, ResourceRegistry, SlurmClusterResource
from bmd_agent.resources import input_check as input_check_module
from bmd_agent.resources.input_check import (
    DIFFERS_FROM_REFERENCE,
    DIFFERS_FROM_THIS_REFERENCE,
    INSUFFICIENT_INFORMATION,
    MATCH,
    MATCHES_THIS_REFERENCE,
    NO_REFERENCE_FOR_THIS_REQUEST,
    REFERENCE_CONTEXT_LIMITATION,
    REFERENCE_MISSING_FROM_SUPPLIED,
    SUPPLIED_EXTRA,
    InputCheckObservation,
    InputReferenceObservation,
    KpointsComparison,
    ProposedInputObservation,
    build_input_reference_request,
    check_remote_input_directory,
    compare_incar_settings,
)
from bmd_agent.resources.input_reference import (
    PRODUCER_MODULE,
    InputReferenceError,
    generate_input_reference,
    parse_input_reference_payload,
)
from bmd_agent.resources.vasp import RemotePathError


HOST = "powerslurm-bmdguest"
ROOT = "/bmd-db/guest/check-input"
REMOTE_DIR = f"{ROOT}/si-relax"

POSCAR = """\
Si
1.0
5.43 0.00 0.00
0.00 5.43 0.00
0.00 0.00 5.43
Si
2
Direct
0.00 0.00 0.00
0.25 0.25 0.25
"""

REFERENCE_SETTINGS = {
    "ADDGRID": True,
    "EDIFF": 1e-6,
    "EDIFFG": -0.01,
    "ENCUT": 580.0,
    "MAGMOM": [0.6, 0.6],
}

SUPPLIED_INCAR = """\
ADDGRID = .TRUE.
EDIFF = 0.000001
EDIFFG = -0.01
ENCUT = 580
MAGMOM = 2*0.6
"""

KPOINTS = """\
Automatic mesh
0
Gamma
7 7 7
0 0 0
"""


def cluster() -> SlurmClusterResource:
    return SlurmClusterResource(
        key="powerslurm",
        name="PowerSLURM",
        ssh_host=HOST,
        partition="leeburton-pool",
        access="observational",
        allowed_remote_roots=(ROOT,),
    )


def repository(tmp_path: Path, *, missing_python: bool = False) -> GitRepositoryResource:
    checkout = tmp_path / "bmd_compute"
    checkout.mkdir()
    python = tmp_path / "python"
    if not missing_python:
        python.write_text("", encoding="utf-8")
    return GitRepositoryResource(
        key="bmd_compute",
        name="BMD Compute",
        path=checkout,
        role="compute",
        access="read_only",
        protected=True,
        live=True,
        capability_python=python,
    )


def remote_files(
    *,
    incar: str = SUPPLIED_INCAR,
    poscar: str = POSCAR,
    kpoints: str = KPOINTS,
) -> dict[str, bytes]:
    return {
        f"{REMOTE_DIR}/INCAR": incar.encode("utf-8"),
        f"{REMOTE_DIR}/POSCAR": poscar.encode("utf-8"),
        f"{REMOTE_DIR}/KPOINTS": kpoints.encode("utf-8"),
    }


def remote_runner_for(
    files: dict[str, bytes],
    calls: list[list[str]] | None = None,
):
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if calls is not None:
            calls.append(command)
        assert command[0:4] == ["ssh", "-o", "BatchMode=yes", HOST]
        assert kwargs["capture_output"] is True
        assert kwargs["timeout"] == 20
        remote_command = command[-1]
        assert "POTCAR" not in remote_command
        assert "sacct" not in remote_command
        assert "sbatch" not in remote_command
        assert "scancel" not in remote_command

        parts = shlex.split(remote_command)
        if parts[0:2] == ["test", "-f"]:
            path = parts[2]
            return subprocess.CompletedProcess(
                command,
                0 if path in files else 1,
                stdout=b"",
                stderr=b"",
            )
        if parts[0:2] == ["cat", "--"]:
            path = parts[2]
            if path not in files:
                raise subprocess.CalledProcessError(1, command, stderr=b"missing\n")
            return subprocess.CompletedProcess(command, 0, stdout=files[path], stderr=b"")
        raise AssertionError(f"unexpected remote command: {remote_command}")

    return runner


def producer_payload(
    *,
    status: str = "ok",
    schema_version: int = 1,
    settings: dict[str, Any] | None = None,
    kpoints_as_dict: dict[str, Any] | None = None,
    dirty: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "scope": "BMD Compute generated pre-execution VASP input reference",
        "status": status,
        "reference_phase": "generated_pre_execution",
        "producer": {
            "repository": "bmd_compute",
            "source": {
                "repository": "bmd_compute",
                "commit": "0396e5eabcdef",
                "dirty": dirty,
                "provenance_available": True,
                "unavailable_reason": None,
            },
        },
        "contract": {"reference": "generated through BMD Compute"},
        "request": {"potcar_functional": "PBE_64"} if status == "ok" else {},
        "workflow": {
            "label": "PBE Geometry Optimisation",
            "stage_count": 1,
            "stages": [],
        },
    }
    if status == "ok":
        payload["reference"] = {
            "stages": [
                {
                    "index": 1,
                    "stage_type": "relax",
                    "theory": "pbe",
                    "incar": {
                        "settings": dict(settings or REFERENCE_SETTINGS),
                        "text": "",
                    },
                    "kpoints": {
                        "text": KPOINTS,
                        "as_dict": kpoints_as_dict
                        or {
                            "generation_style": "Gamma",
                            "kpoints": [[7, 7, 7]],
                            "usershift": [0, 0, 0],
                        },
                    },
                    "poscar": {
                        "text": POSCAR,
                        "sha256": "not-used",
                        "formula": "Si2",
                        "reduced_formula": "Si",
                        "num_sites": 2,
                    },
                }
            ]
        }
    else:
        payload["error"] = {
            "code": "unsupported_combination" if status == "unsupported" else "reference_generation_failed",
            "message": "producer reported a structured problem",
            "suggestion": None,
        }
        payload["reference"] = None
    return payload


def producer_runner_for(
    payload: dict[str, Any] | str,
    calls: list[tuple[list[str], dict[str, object]]] | None = None,
    *,
    returncode: int = 0,
    stderr: str = "",
):
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append((command, dict(kwargs)))
        stdout = payload if isinstance(payload, str) else json.dumps(payload)
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    return runner


def run_check(
    tmp_path: Path,
    *,
    files: dict[str, bytes] | None = None,
    payload: dict[str, Any] | str | None = None,
    returncode: int = 0,
    remote_calls: list[list[str]] | None = None,
    producer_calls: list[tuple[list[str], dict[str, object]]] | None = None,
    missing_python: bool = False,
) -> InputCheckObservation:
    return check_remote_input_directory(
        cluster(),
        repository(tmp_path, missing_python=missing_python),
        REMOTE_DIR,
        stage="relax",
        theory="pbe",
        remote_runner=remote_runner_for(files or remote_files(), remote_calls),
        producer_runner=producer_runner_for(
            producer_payload() if payload is None else payload,
            producer_calls,
            returncode=returncode,
        ),
    )


def test_input_reference_invokes_fixed_producer_with_stdin_and_check_false(
    tmp_path: Path,
) -> None:
    repo = repository(tmp_path)
    calls: list[tuple[list[str], dict[str, object]]] = []

    response = generate_input_reference(
        repo,
        build_input_reference_request(POSCAR, stage="relax", theory="pbe"),
        runner=producer_runner_for(producer_payload(), calls),
    )

    assert response.status == "ok"
    command, kwargs = calls[0]
    assert command == [str(repo.capability_python), "-B", "-m", PRODUCER_MODULE]
    assert kwargs["cwd"] == repo.path
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    request = json.loads(str(kwargs["input"]))
    assert "resources" not in request
    assert "potcar_functional" not in request
    assert request["workflow_spec"]["stages"][0] == {
        "stage_type": "relax",
        "theory": "pbe",
        "modifiers": [],
        "label": None,
        "options": {},
    }


def test_exact_generated_reference_match_matches_this_reference(tmp_path: Path) -> None:
    remote_calls: list[list[str]] = []
    producer_calls: list[tuple[list[str], dict[str, object]]] = []

    observation = run_check(
        tmp_path,
        remote_calls=remote_calls,
        producer_calls=producer_calls,
    )

    assert observation.overall_status == MATCHES_THIS_REFERENCE
    assert observation.limitations == (REFERENCE_CONTEXT_LIMITATION,)
    assert observation.matching_incar_settings == len(REFERENCE_SETTINGS)
    assert all(item.status == MATCH for item in observation.incar_comparisons)
    assert observation.kpoints_comparison is not None
    assert observation.kpoints_comparison.status == MATCH
    assert observation.reference.producer_commit == "0396e5eabcdef"
    assert observation.reference.reference_phase == "generated_pre_execution"
    assert observation.proposed.reduced_formula == "Si"
    assert not any("POTCAR" in call[2] for call in remote_calls)
    assert not any(any(token in call[2] for token in ("sacct", "sbatch", "scancel")) for call in remote_calls)
    assert len(producer_calls) == 1


def test_changed_incar_value_differs_from_this_reference(tmp_path: Path) -> None:
    files = remote_files(incar=SUPPLIED_INCAR.replace("EDIFFG = -0.01", "EDIFFG = -0.02"))

    observation = run_check(tmp_path, files=files)

    assert observation.overall_status == DIFFERS_FROM_THIS_REFERENCE
    [difference] = [
        item for item in observation.incar_comparisons
        if item.setting == "EDIFFG"
    ]
    assert difference.status == DIFFERS_FROM_REFERENCE
    assert difference.supplied_value == -0.02
    assert difference.reference_value == -0.01


def test_typed_equivalent_incar_representations_match() -> None:
    comparisons = compare_incar_settings(
        {"ADDGRID": True, "EDIFF": 0.000001, "MAGMOM": "2*0.6"},
        {"ADDGRID": True, "EDIFF": 1e-6, "MAGMOM": [0.6, 0.6]},
    )

    assert {item.setting: item.status for item in comparisons} == {
        "ADDGRID": MATCH,
        "EDIFF": MATCH,
        "MAGMOM": MATCH,
    }


def test_supplied_extra_setting_is_reported_as_a_difference(tmp_path: Path) -> None:
    files = remote_files(incar=SUPPLIED_INCAR + "LORBIT = 11\n")

    observation = run_check(tmp_path, files=files)

    assert observation.overall_status == DIFFERS_FROM_THIS_REFERENCE
    extras = [item for item in observation.incar_comparisons if item.status == SUPPLIED_EXTRA]
    assert [item.setting for item in extras] == ["LORBIT"]


def test_omitted_reference_setting_is_reported_as_a_difference(tmp_path: Path) -> None:
    files = remote_files(incar=SUPPLIED_INCAR.replace("MAGMOM = 2*0.6\n", ""))

    observation = run_check(tmp_path, files=files)

    assert observation.overall_status == DIFFERS_FROM_THIS_REFERENCE
    missing = [
        item for item in observation.incar_comparisons
        if item.status == REFERENCE_MISSING_FROM_SUPPLIED
    ]
    assert [item.setting for item in missing] == ["MAGMOM"]


def test_differing_structured_kpoints_differs_from_this_reference(tmp_path: Path) -> None:
    files = remote_files(kpoints=KPOINTS.replace("7 7 7", "6 6 6"))

    observation = run_check(tmp_path, files=files)

    assert observation.overall_status == DIFFERS_FROM_THIS_REFERENCE
    assert observation.kpoints_comparison is not None
    assert observation.kpoints_comparison.status == DIFFERS_FROM_REFERENCE
    assert observation.kpoints_comparison.supplied_summary == "Gamma 6x6x6"
    assert observation.kpoints_comparison.reference_summary == "Gamma 7x7x7"


def test_missing_required_input_is_insufficient_without_calling_producer(
    tmp_path: Path,
) -> None:
    files = remote_files()
    del files[f"{REMOTE_DIR}/KPOINTS"]
    producer_calls: list[tuple[list[str], dict[str, object]]] = []

    observation = run_check(tmp_path, files=files, producer_calls=producer_calls)

    assert observation.overall_status == INSUFFICIENT_INFORMATION
    assert "KPOINTS" in observation.limitations[0]
    assert producer_calls == []


def test_malformed_incar_is_insufficient_without_calling_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer_calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(
        input_check_module,
        "parse_incar_contents",
        lambda contents: ({}, "bad INCAR"),
    )

    observation = run_check(tmp_path, producer_calls=producer_calls)

    assert observation.overall_status == INSUFFICIENT_INFORMATION
    assert "Supplied INCAR could not be parsed" in observation.limitations[0]
    assert producer_calls == []


def test_malformed_poscar_is_insufficient_without_calling_producer(
    tmp_path: Path,
) -> None:
    producer_calls: list[tuple[list[str], dict[str, object]]] = []

    observation = run_check(
        tmp_path,
        files=remote_files(poscar="not a POSCAR\n"),
        producer_calls=producer_calls,
    )

    assert observation.overall_status == INSUFFICIENT_INFORMATION
    assert "Supplied POSCAR could not be parsed" in observation.limitations[0]
    assert producer_calls == []


def test_malformed_kpoints_is_insufficient_without_calling_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer_calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(
        input_check_module,
        "parse_kpoints_contents",
        lambda contents: input_check_module._ParsedKpoints(None, None, "bad KPOINTS"),
    )

    observation = run_check(tmp_path, producer_calls=producer_calls)

    assert observation.overall_status == INSUFFICIENT_INFORMATION
    assert "Supplied KPOINTS could not be parsed" in observation.limitations[0]
    assert producer_calls == []


def test_producer_refusal_is_no_reference_with_computes_reason(tmp_path: Path) -> None:
    observation = run_check(tmp_path, payload=producer_payload(status="unsupported"))

    assert observation.overall_status == NO_REFERENCE_FOR_THIS_REQUEST
    assert observation.reference.status == "unsupported"
    assert observation.reference.error_code == "unsupported_combination"
    assert observation.limitations[0] == (
        "BMD Compute did not generate a reference: producer reported a structured problem"
    )


def test_structured_producer_error_despite_nonzero_exit_is_insufficient(
    tmp_path: Path,
) -> None:
    observation = run_check(
        tmp_path,
        payload=producer_payload(status="error"),
        returncode=2,
    )

    assert observation.overall_status == INSUFFICIENT_INFORMATION
    assert observation.reference.status == "error"
    assert observation.reference.error_code == "reference_generation_failed"


def test_malformed_producer_json_is_insufficient(tmp_path: Path) -> None:
    observation = run_check(tmp_path, payload="{not json", returncode=0)

    assert observation.overall_status == INSUFFICIENT_INFORMATION
    assert observation.reference.status == "unavailable"
    assert observation.reference.error_code == "malformed_json"


def test_unsupported_producer_schema_is_insufficient(tmp_path: Path) -> None:
    observation = run_check(tmp_path, payload=producer_payload(schema_version=2))

    assert observation.overall_status == INSUFFICIENT_INFORMATION
    assert observation.reference.status == "unavailable"
    assert observation.reference.error_code == "unsupported_schema"


def test_missing_configured_compute_python_is_insufficient(tmp_path: Path) -> None:
    observation = run_check(tmp_path, missing_python=True)

    assert observation.overall_status == INSUFFICIENT_INFORMATION
    assert observation.reference.error_code == "missing_interpreter"


def test_producer_timeout_is_insufficient(tmp_path: Path) -> None:
    def timeout_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout=20)

    observation = check_remote_input_directory(
        cluster(),
        repository(tmp_path),
        REMOTE_DIR,
        stage="relax",
        theory="pbe",
        remote_runner=remote_runner_for(remote_files()),
        producer_runner=timeout_runner,
    )

    assert observation.overall_status == INSUFFICIENT_INFORMATION
    assert observation.reference.error_code == "timeout"


def test_outside_allowed_remote_root_is_rejected_before_reads(tmp_path: Path) -> None:
    remote_calls: list[list[str]] = []

    with pytest.raises(RemotePathError):
        check_remote_input_directory(
            cluster(),
            repository(tmp_path),
            "/outside/run",
            stage="relax",
            theory="pbe",
            remote_runner=remote_runner_for(remote_files(), remote_calls),
            producer_runner=producer_runner_for(producer_payload()),
        )

    assert remote_calls == []


def test_cli_check_input_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = ResourceRegistry(repositories={}, clusters={})
    observation = InputCheckObservation(
        overall_status=DIFFERS_FROM_THIS_REFERENCE,
        remote_directory=REMOTE_DIR,
        stage_type="relax",
        theory="pbe",
        modifiers=(),
        proposed=ProposedInputObservation(
            directory=REMOTE_DIR,
            files={},
            reduced_formula="Si",
            site_count=2,
        ),
        reference=InputReferenceObservation(
            status="ok",
            schema_version=1,
            reference_phase="generated_pre_execution",
            producer_repository="bmd_compute",
            producer_commit="0396e5eabcdef",
            producer_dirty=False,
            workflow_label="PBE Geometry Optimisation",
        ),
        incar_comparisons=(
            input_check_module.InputSettingComparison("ENCUT", 580, 580.0, MATCH),
            input_check_module.InputSettingComparison(
                "EDIFFG",
                -0.02,
                -0.01,
                DIFFERS_FROM_REFERENCE,
            ),
        ),
        kpoints_comparison=KpointsComparison(
            MATCH,
            supplied_summary="Gamma 7x7x7",
            reference_summary="Gamma 7x7x7",
        ),
    )

    monkeypatch.setattr(cli, "load_resources", lambda: registry)
    monkeypatch.setattr(cli, "powerslurm_cluster", lambda registry: cluster())
    monkeypatch.setattr(cli, "bmd_compute_repository", lambda registry: object())
    monkeypatch.setattr(cli, "check_remote_input_directory", lambda *args, **kwargs: observation)

    exit_code = cli.main(
        [
            "check-input",
            REMOTE_DIR,
            "--stage",
            "relax",
            "--theory",
            "pbe",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "BMD VASP Input Check" in captured.out
    assert "PBE Geometry Optimisation" in captured.out
    assert "commit:         0396e5eabcd" in captured.out
    assert "1 settings match" in captured.out
    assert "EDIFFG" in captured.out
    assert "DIFFERS FROM THIS REFERENCE" in captured.out
    assert "not by itself evidence" in captured.out


def test_cli_check_input_requires_stage_and_theory(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(["check-input", REMOTE_DIR, "--stage", "relax"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "missing required option(s): --theory" in captured.out


@pytest.mark.parametrize("option", ["--nodes", "--ntasks", "--mem-gb"])
def test_cli_check_input_rejects_resource_options_that_do_not_affect_the_reference(
    option: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(["check-input", REMOTE_DIR, "--stage", "relax", "--theory", "pbe", option, "24"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "do not affect the generated BMD Compute reference" in captured.out


def test_raw_traceback_from_failed_producer_is_not_exposed(tmp_path: Path) -> None:
    repo = repository(tmp_path)

    def failing_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="Traceback (most recent call last):\nsecret stack\n",
        )

    with pytest.raises(InputReferenceError) as exc_info:
        generate_input_reference(
            repo,
            build_input_reference_request(POSCAR, stage="relax", theory="pbe"),
            runner=failing_runner,
        )

    assert "Traceback" not in str(exc_info.value)


@pytest.mark.parametrize(
    "scope",
    (
        "BMD Compute generated pre-execution VASP input reference",
        "Any reworded human description",
        None,
    ),
)
def test_input_reference_scope_prose_is_not_a_compatibility_contract(scope: str | None) -> None:
    payload = producer_payload()
    if scope is None:
        del payload["scope"]
    else:
        payload["scope"] = scope

    response = parse_input_reference_payload(json.dumps(payload))

    assert response.status == "ok"
    assert response.reference_phase == "generated_pre_execution"


@pytest.mark.parametrize(
    ("mutate", "kind"),
    (
        (lambda payload: payload.update(schema_version=2), "unsupported_schema"),
        (lambda payload: payload.pop("schema_version"), "unsupported_schema"),
        (lambda payload: payload.update(reference_phase="post_execution"), "unexpected_reference_phase"),
        (lambda payload: payload["producer"].update(repository="bmdex"), "unexpected_producer"),
        (lambda payload: payload.update(scope=["not", "text"]), "malformed_payload"),
    ),
)
def test_input_reference_compatibility_uses_machine_readable_fields(mutate, kind: str) -> None:
    payload = producer_payload()
    mutate(payload)

    with pytest.raises(InputReferenceError) as exc_info:
        parse_input_reference_payload(json.dumps(payload))

    assert exc_info.value.kind == kind
