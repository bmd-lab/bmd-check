"""Generate Agent's BMD Compute v1 contract fixtures from real Compute code (read-only).

Run with BMD Compute's own Python environment, against a clean BMD Compute
checkout, without writing bytecode into that checkout:

    BMD_SUBMISSION_IDENTITY_SECRET=fixture-only PYTHONDONTWRITEBYTECODE=1 \
        python -B generate_from_compute.py <bmd_compute checkout> <output dir>

Every payload is produced by BMD Compute's own producer functions; Agent adds
no scientific content. Normalizations for stable diffs (all in fields that
describe *when/where* something ran, never in scientific content):

* runtime records: ``recorded_at`` and ``host`` are fixed; atomate2's
  ``CONFIG_FILE`` (a home-directory path) is fixed; ``SLURM_JOB_ID`` is supplied.
* the failed runtime record is produced by Compute's real
  ``enforce_runtime_environment`` with a substituted version lookup and
  atomate2 settings object, which is how Compute's own tests exercise it.
* the Desired Output and Custom workflow submissions use fixed timestamps and
  attempt IDs, and the fixture-only identity-token secret. Both are produced by
  Compute's ``main.build_submission_state_from_structure`` (the web
  application's Prepare path); only the Desired Output one receives Compute's
  automatic-treatment resolution.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import types
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
os.environ.setdefault("BMD_SUBMISSION_IDENTITY_SECRET", "fixture-only")
compute_root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
sys.path.insert(0, str(compute_root))

from pymatgen.core import Lattice, Structure  # noqa: E402

from backend.calculations.capabilities import build_capability_payload  # noqa: E402
from backend.calculations.default_treatments import resolve_default_treatments  # noqa: E402
from backend.calculations.registry import desired_output_workflow_spec  # noqa: E402
from backend.calculations.resources import default_execution_resources  # noqa: E402
from backend.runtime_environment import (  # noqa: E402
    RuntimeParityError,
    atomate2_settings_snapshot,
    enforce_runtime_environment,
    installed_versions,
    PARITY_CRITICAL_PACKAGES,
    RECORDED_SUPPORTING_PACKAGES,
)
import main as compute_app  # noqa: E402

FIXED_RECORDED_AT = "2026-10-01T09:00:00+00:00"
FIXED_HOST = "power-node-fixture"
FIXED_CONFIG_FILE = "/home/bmdguest/.atomate2.yaml"
NIO_POSCAR = """NiO
1.0
4.17 0.0 0.0
0.0 4.17 0.0
0.0 0.0 4.17
Ni O
4 4
direct
0.0 0.0 0.0
0.5 0.5 0.0
0.5 0.0 0.5
0.0 0.5 0.5
0.5 0.5 0.5
0.0 0.0 0.5
0.0 0.5 0.0
0.5 0.0 0.0
"""


def dump(name: str, payload) -> None:
    (out / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def desired_output_submission() -> dict:
    structure = Structure.from_str(NIO_POSCAR, fmt="poscar")
    desired = "energy_only"
    resolution = resolve_default_treatments(
        structure, desired_output_workflow_spec(desired), desired_output=desired
    )
    _, _, _, submission = compute_app.build_submission_state_from_structure(
        structure_obj=structure,
        structure_text=NIO_POSCAR,
        fmt="poscar",
        workflow_spec=resolution.resolved_workflow,
        execution_resources=default_execution_resources(),
        timestamp="20261001-090000",
        submission_attempt_id="00000000-0000-4000-8000-000000000201",
        default_treatment_resolution=resolution,
    )
    return submission


def custom_workflow_submission() -> dict:
    from backend.calculations.models import WorkflowSpec

    structure = Structure.from_str(NIO_POSCAR, fmt="poscar")
    workflow = WorkflowSpec.from_dict(
        {
            "stages": [
                {"stage_type": "static", "theory": "pbe", "modifiers": [], "label": None, "options": {}}
            ],
            "label": None,
            "recipe": None,
        }
    )
    _, _, _, submission = compute_app.build_submission_state_from_structure(
        structure_obj=structure,
        structure_text=NIO_POSCAR,
        fmt="poscar",
        workflow_spec=workflow,
        execution_resources=default_execution_resources(),
        timestamp="20261001-090100",
        submission_attempt_id="00000000-0000-4000-8000-000000000202",
    )
    return submission


def runtime_record(spec: dict, *, version_lookup=None, settings=None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runtime_environment.json"
        try:
            enforce_runtime_environment(
                spec,
                version_lookup=version_lookup,
                settings=settings,
                record_path=path,
                environ={"SLURM_JOB_ID": "10000201"},
            )
        except RuntimeParityError:
            pass
        record = json.loads(path.read_text(encoding="utf-8"))
    record["recorded_at"] = FIXED_RECORDED_AT
    record["host"] = FIXED_HOST
    return record


def settings_object(**overrides):
    snapshot = atomate2_settings_snapshot()
    snapshot["CONFIG_FILE"] = FIXED_CONFIG_FILE
    snapshot.update(overrides)
    return types.SimpleNamespace(**snapshot)


def main() -> None:
    out.mkdir(parents=True, exist_ok=True)
    # Build everything before writing so Git provenance describes the checkout.
    capabilities = build_capability_payload(repo_root=compute_root)
    submission = desired_output_submission()
    custom = custom_workflow_submission()
    passed = runtime_record(submission, settings=settings_object())
    real = installed_versions(PARITY_CRITICAL_PACKAGES + RECORDED_SUPPORTING_PACKAGES)

    def drifted(name: str) -> str:
        if name == "pymatgen-core":
            return "2026.6.1"
        return real[name]

    failed = runtime_record(
        submission,
        version_lookup=drifted,
        settings=settings_object(VASP_INCAR_UPDATES={"ENCUT": 600}),
    )
    dump("capabilities.json", capabilities)
    (out / "desired_output_nio").mkdir(exist_ok=True)
    (out / "desired_output_nio" / "submission.json").write_text(
        json.dumps(submission, indent=2) + "\n", encoding="utf-8"
    )
    (out / "custom_workflow_nio").mkdir(exist_ok=True)
    (out / "custom_workflow_nio" / "submission.json").write_text(
        json.dumps(custom, indent=2) + "\n", encoding="utf-8"
    )
    dump("runtime_environment_passed.json", passed)
    dump("runtime_environment_failed.json", failed)
    files = sorted(p for p in out.rglob("*.json"))
    (out / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}\n"
            for p in files
        ),
        encoding="utf-8",
    )
    print({"capabilities_source": capabilities["source"], "passed": passed["status"],
           "failed": failed["status"], "failed_problems": failed["problems"]})
    print({"automatic_treatments": [t["modifier"] for t in submission["provenance"]["execution"]["automatic_treatments"]["applied_treatments"]]})


if __name__ == "__main__":
    main()
