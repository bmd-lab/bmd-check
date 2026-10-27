"""Versioned electronic-character assessment over calculated VASP evidence.

C2 decides whether one completed VASP stage supports classifying the system as
``metallic`` or ``nonmetallic``, and otherwise returns ``uncertain``. It is an
assessment of calculated electronic evidence only: it never selects ``ISMEAR``,
``SIGMA`` or any other VASP setting. Consumers (BMD Compute) own any policy that
maps this classification to methodology.

Design rules:

* The verdict function :func:`assess_electronic_character` is pure. Evidence is
  read once, read-only and bounded, by :func:`read_vasprun_electronic_evidence`
  and normalized into :class:`ElectronicCharacterInput`.
* Every numerical input is validated (finite, correctly typed and shaped)
  before any arithmetic; invalid evidence is ``uncertain``.
* Sampling density is accepted only from a mesh that VASP itself generated
  (recorded ``Gamma``, ``Monkhorst-Pack`` or ``Auto`` divisions) and that the
  irreducible k-points and their symmetry weights are consistent with. Explicit
  k-point lists never establish a mesh.
* Classification is by electron counting and band dispersion relative to the
  Fermi level, per spin channel, with deliberate gaps between the "clearly
  metallic" and "clearly gapped" criteria (see :data:`THRESHOLDS`): evidence
  that falls between them is ``uncertain``. Fractional occupations alone never
  establish metallicity, and a small positive gap never establishes
  nonmetallicity.
* Convergence is not recomputed here: stage completion and convergence come
  from C1's :class:`~bmd_agent.resources.completion.StageCompletion`.
* Every numerical threshold is a named, versioned constant in
  :data:`THRESHOLDS`. They are provisional engineering tolerances chosen to fail
  towards ``uncertain``; they are not validated physical boundaries.
* The serialized response contains no raw eigenvalue arrays and no absolute
  paths, and every published band edge, gap and gap type is derived from one
  named evidence set.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import Any
import warnings

from bmd_agent.resources.assessment import (
    ProducerProvenance,
    _local_producer_provenance,
    _optional_text,
    _package_version,
    _stage_label,
    _stage_type,
    _submission_subject_id,
)
from bmd_agent.resources.completion import STAGE_CONVERGED, StageCompletion
from bmd_agent.resources.lifecycle import LifecycleAnalysis, _analyze_direct_vasp_directory


SCHEMA_ID = "bmd_check.electronic_character_assessment"
SCHEMA_VERSION = 1
ASSESSMENT_ID = "electronic_character"
METHOD_VERSION = "1.0.0-c2"

CLASSIFICATION_METALLIC = "metallic"
CLASSIFICATION_NONMETALLIC = "nonmetallic"
CLASSIFICATION_UNCERTAIN = "uncertain"

SCOPE_STAGE_DIRECTORY = "stage_directory"

STATUS_ASSESSED = "assessed"
STATUS_UNSUPPORTED = "unsupported"
STATUS_INSUFFICIENT_EVIDENCE = "insufficient_evidence"

CHANNEL_METALLIC = "metallic"
CHANNEL_GAPPED = "gapped"
CHANNEL_UNCERTAIN = "uncertain"

SPIN_NON_POLARIZED = "non_spin_polarized"
SPIN_COLLINEAR = "collinear_spin_polarized"
SPIN_NONCOLLINEAR = "noncollinear"

EVIDENCE_AVAILABLE = "available"
EVIDENCE_MISSING = "missing"
EVIDENCE_TOO_LARGE = "too_large"
EVIDENCE_UNREADABLE = "unreadable"
EVIDENCE_INCOMPLETE = "incomplete"
EVIDENCE_AMBIGUOUS = "ambiguous"

# Which k-points the published band edges were taken from.
EDGE_EVIDENCE_WEIGHTED = "weighted_mesh"
EDGE_EVIDENCE_WITH_ZERO_WEIGHT = "weighted_mesh_and_zero_weight_kpoints"


# Provisional, versioned tolerances for method 1.0.0-c2. Energies in eV,
# reciprocal spacings in 1/Angstrom with the 2*pi factor included (the VASP
# KSPACING convention). Each is chosen so that doubtful evidence becomes
# ``uncertain``; none is a validated physical boundary.
#
# The filling and Fermi-level tolerances are deliberately *not* single cut-offs.
# A channel is "clearly integer" only within integer_filling_tolerance and
# "clearly fractional" only beyond fractional_filling_min; a band "clearly
# crosses" the Fermi level only beyond fermi_crossing_margin_ev on both sides,
# whereas a gapped channel may let a band edge pass the Fermi level by at most
# fermi_edge_tolerance_ev. Evidence in the bands between these limits (near-
# integer filling, shallow crossings) satisfies neither the metallic nor the
# nonmetallic criteria and is ``uncertain``. The ordering is enforced below.
THRESHOLDS: Mapping[str, float] = {
    # vasprun.xml occupations are per state in [0, 1]; Methfessel-Paxton
    # smearing legitimately overshoots slightly.
    "occupation_range_tolerance": 0.1,
    # Bloechl-corrected tetrahedron weights are not bounded per k-point; this
    # wider range applies to tetrahedron methods only.
    "tetrahedron_occupation_range_tolerance": 0.25,
    # K-point weights at or below this are treated as zero-weight (path) points.
    "zero_weight_tolerance": 1e-10,
    # Normalized weights must sum to one within this tolerance.
    "weight_sum_tolerance": 1e-4,
    # Occupation-weighted electron count must reproduce NELECT within this.
    # This bounds numerical integration error only: across 51 recorded VASP runs
    # (pymatgen test suite) the largest deviation was 1.9e-4 electrons.
    "electron_count_tolerance": 0.005,
    # NELECT itself must be an integer within this. vasprun.xml records NELECT
    # with eight decimals (representation error <= 5e-9); a larger fractional
    # part is a deliberately fractional electron count (fractional charge,
    # virtual-crystal or similar), which C2 does not classify.
    "nelect_integer_tolerance": 1e-6,
    # Channel electron count within this of an integer: clearly integer-filled.
    "integer_filling_tolerance": 0.005,
    # Channel electron count at least this far from an integer: clearly
    # fractional. Between the two: uncertain.
    "fractional_filling_min": 0.02,
    # A band clearly crosses the Fermi level only if it has states at least this
    # far below and above it on the weighted mesh.
    "fermi_crossing_margin_ev": 0.05,
    # In a gapped channel the valence-band maximum may lie at most this far
    # above, and the conduction-band minimum at most this far below, the Fermi
    # level (VASP pins the Fermi level at a band edge in insulators).
    "fermi_edge_tolerance_ev": 0.01,
    # A negative gap (band overlap) is metallic only beyond this magnitude.
    "band_overlap_min_ev": 0.05,
    # Gaps below this are never treated as establishing nonmetallicity.
    "min_reliable_gap_ev": 0.1,
    # For Gaussian, Fermi-Dirac and Methfessel-Paxton smearing, the gap must be
    # at least this multiple of SIGMA.
    "smearing_gap_ratio": 4.0,
    # Highest band must be empty to this tolerance (enough bands computed), and
    # a gapped channel may hold at most this many electrons in states above
    # the gap (charge spilled across the gap by smearing).
    "edge_occupation_tolerance": 0.01,
    # Occupations far from the Fermi level must be within this of 0 or 1.
    "far_occupation_tolerance": 0.05,
    # "Far" from the Fermi level, as a multiple of SIGMA for smearing methods ...
    "far_window_sigma_multiple": 6.0,
    # ... but never less than this ...
    "far_window_min_ev": 0.05,
    # ... and for tetrahedron methods, whose per-k occupations can be
    # fractional across the local band dispersion, this fixed window.
    "far_window_tetrahedron_ev": 1.0,
    # Band states within this of a band extremum are degenerate with it.
    # vasprun.xml records eigenvalues with four decimals (1e-4 eV), so this is
    # ten times the representation resolution.
    "band_extremum_degeneracy_tolerance_ev": 0.001,
    # If the smallest direct (same-k) gap exceeds the fundamental gap by more
    # than the degeneracy tolerance but less than this, the direct/indirect
    # label is reported as undetermined rather than chosen.
    "gap_type_resolution_ev": 0.01,
    # Fractional k-point coordinates (modulo reciprocal-lattice vectors) closer
    # than this identify the same k-point.
    "kpoint_equivalence_tolerance": 1e-4,
    # Weighted k-point coordinates must lie on the generated mesh within this
    # (fractional reciprocal units, times the division).
    "mesh_coordinate_tolerance": 1e-4,
    # weight x (number of mesh points) must be an integer multiplicity within this.
    "mesh_multiplicity_tolerance": 1e-3,
    # Largest possible star of a k-point: the order of the largest
    # crystallographic point group (time reversal does not exceed it).
    "max_kpoint_multiplicity": 48,
    # Largest mesh division accepted.
    "max_mesh_division": 1000,
    # Largest reciprocal-mesh spacing accepted for a definitive classification
    # (equal to VASP's default KSPACING).
    "max_reciprocal_spacing_inv_angstrom": 0.5,
    # vasprun.xml files larger than this are not parsed.
    "max_vasprun_bytes": 1_000_000_000,
}

if not (
    THRESHOLDS["integer_filling_tolerance"] < THRESHOLDS["fractional_filling_min"] < 0.5
    and THRESHOLDS["fermi_edge_tolerance_ev"] < THRESHOLDS["fermi_crossing_margin_ev"]
    and 2 * THRESHOLDS["fermi_crossing_margin_ev"] <= THRESHOLDS["min_reliable_gap_ev"]
):  # pragma: no cover - guards future edits of the table above
    raise RuntimeError("electronic-character thresholds must leave an uncertain band between criteria")

_SMEARING_METHODS = {
    -1: "fermi_dirac",
    0: "gaussian",
}
_TETRAHEDRON_METHODS = {-4, -5, -14, -15}
_UNSUPPORTED_SMEARING = {
    -2: "fixed occupations supplied by the user",
    -3: "multiple smearing runs",
}
_UNSUPPORTED_ALGOS = frozenset(
    {"gw", "gw0", "g0w0", "evgw", "evgw0", "qpgw", "qpgw0", "scgw", "scgw0", "bse", "chi", "tdhf", "acfdt", "rpa"}
)
# KPOINTS generation modes for which VASP records the mesh it generated.
_GENERATED_MESH_STYLES = frozenset({"gamma", "monkhorst", "automatic"})
_PHONON_STAGE_TYPES = frozenset({"phonon", "phonons"})


@dataclass(frozen=True)
class VasprunElectronicEvidence:
    """Bounded electronic evidence read from one vasprun.xml.

    Arrays are kept only in memory for the pure assessment; they are never
    serialized.
    """

    status: str
    reason: str | None = None
    eigenvalues: tuple[Any, ...] = ()  # per channel: (nkpts, nbands) energies
    occupations: tuple[Any, ...] = ()  # per channel: (nkpts, nbands) in [0, 1]
    weights: Any = None  # (nkpts,)
    kpoints_frac: Any = None  # (nkpts, 3) reciprocal fractional coordinates
    efermi: float | None = None
    ispin: int | None = None
    lsorbit: bool = False
    lnoncollinear: bool = False
    ismear: int | None = None
    sigma: float | None = None
    icharg: int | None = None
    nelect: float | None = None
    nbands: int | None = None
    algo: str | None = None
    lhfcalc: bool = False
    kpoints_style: str | None = None
    declared_divisions: tuple[int, int, int] | None = None
    # Effective mesh shift (units of the mesh spacing) as recorded by VASP in
    # <generation><v name="shift">; None when not recorded.
    kpoints_shift: tuple[float, float, float] | None = None
    reciprocal_lengths: tuple[float, float, float] | None = None
    converged_electronic: bool | None = None


@dataclass(frozen=True)
class SelectedStage:
    index: int
    label: str
    stage_type: str
    theory: str | None = None


@dataclass(frozen=True)
class ElectronicCharacterInput:
    subject_id: str
    calculation_type: str
    scope: str
    stage: SelectedStage | None
    stage_completion: StageCompletion | None
    evidence: VasprunElectronicEvidence
    producer_provenance: ProducerProvenance
    limitations: tuple[str, ...] = ()
    assessor_version: str = "unknown"


@dataclass(frozen=True)
class ElectronicFinding:
    reason_code: str
    status: str
    message: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceReference:
    evidence_id: str
    evidence_type: str
    source: str
    stage_index: int | None = None


@dataclass(frozen=True)
class BandEdge:
    """One band edge and the k-point it was found at."""

    energy_ev: float  # the exact extremum energy
    kpoint: tuple[float, float, float] | None  # representative, reduced into [0, 1)
    kpoint_weight: str  # "weighted" or "zero_weight" (of the representative)
    # Number of inequivalent k-points whose state lies within the degeneracy
    # tolerance of this extremum (the representative is the lexicographically
    # first of them, or of the shared set for a direct gap).
    degenerate_kpoints: int = 1


@dataclass(frozen=True)
class SpinChannelSummary:
    channel: str
    state: str
    electron_count: float | None
    filling: str | None
    vbm: BandEdge | None = None
    cbm: BandEdge | None = None
    band_gap_ev: float | None = None
    gap_type: str | None = None
    # Smallest gap between the same two bands at one k-point (same evidence
    # set); ``None`` when an ambiguous k-point equivalence class leaves it
    # unestablished.
    min_direct_gap_ev: float | None = None
    # Lowest value the smallest direct gap could take under any reading of
    # ambiguous k-point equivalence; equals ``min_direct_gap_ev`` when that is
    # established. The direct/indirect label is decided against this bound.
    min_direct_gap_lower_bound_ev: float | None = None
    # True when the band extrema or the smallest direct gap involve k-points
    # whose equivalence modulo reciprocal-lattice vectors is ambiguous (a
    # non-transitive chain).
    kpoint_equivalence_ambiguous: bool = False
    reason_code: str | None = None


@dataclass(frozen=True)
class ElectronicEvidenceSummary:
    spin_treatment: str | None = None
    fermi_level_ev: float | None = None
    band_edge_evidence: str | None = None
    band_gap_ev: float | None = None
    gap_type: str | None = None
    vbm: BandEdge | None = None
    cbm: BandEdge | None = None
    vbm_channel: str | None = None
    cbm_channel: str | None = None
    half_metallic: bool | None = None
    channels: tuple[SpinChannelSummary, ...] = ()
    smearing_method: str | None = None
    ismear: int | None = None
    sigma_ev: float | None = None
    icharg: int | None = None
    nelect: float | None = None
    nbands: int | None = None
    kpoints_style: str | None = None
    mesh_divisions: tuple[int, int, int] | None = None
    mesh_status: str | None = None
    weighted_kpoints: int | None = None
    zero_weight_kpoints: int | None = None
    max_reciprocal_spacing_inv_angstrom: float | None = None
    electronic_convergence: str | None = None


@dataclass(frozen=True)
class ElectronicCharacterAssessment:
    subject_id: str
    calculation_type: str
    scope: str
    stage: SelectedStage | None
    assessment_status: str
    classification: str
    findings: tuple[ElectronicFinding, ...]
    evidence: ElectronicEvidenceSummary
    evidence_references: tuple[EvidenceReference, ...]
    producer_provenance: ProducerProvenance
    assessor_version: str
    limitations: tuple[str, ...] = ()
    thresholds: Mapping[str, float] = field(default_factory=lambda: dict(THRESHOLDS))
    schema_id: str = SCHEMA_ID
    schema_version: int = SCHEMA_VERSION
    assessment_id: str = ASSESSMENT_ID
    method_version: str = METHOD_VERSION


class StageSelectionError(ValueError):
    """The requested stage cannot be selected unambiguously."""


# ---------------------------------------------------------------------------
# Evidence acquisition (read-only, bounded)
# ---------------------------------------------------------------------------


def read_vasprun_electronic_evidence(
    path: Path,
    *,
    max_bytes: int = int(THRESHOLDS["max_vasprun_bytes"]),
) -> VasprunElectronicEvidence:
    """Read eigenvalues, occupations and settings from one vasprun.xml.

    This only reads the file. Malformed or truncated XML is reported as
    incomplete evidence, never partially used.
    """

    try:
        if not path.is_file():
            return VasprunElectronicEvidence(EVIDENCE_MISSING, "vasprun.xml is not present")
        size = path.stat().st_size
    except OSError:
        return VasprunElectronicEvidence(EVIDENCE_UNREADABLE, "vasprun.xml could not be inspected")
    if size <= 0:
        return VasprunElectronicEvidence(EVIDENCE_MISSING, "vasprun.xml is empty")
    if size > max_bytes:
        return VasprunElectronicEvidence(
            EVIDENCE_TOO_LARGE,
            f"vasprun.xml exceeds the {max_bytes}-byte parsing bound",
        )
    try:
        import numpy as np
        from pymatgen.io.vasp.outputs import Vasprun

        from bmd_agent.resources.run import _load_vasprun_tolerantly
    except Exception:
        return VasprunElectronicEvidence(EVIDENCE_UNREADABLE, "the VASP output parser is unavailable")

    # Parser warnings (for example pymatgen's unconverged-run warning, which
    # embeds the absolute path) are not surfaced: convergence and malformed
    # XML are reported through structured findings instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vasprun, malformed, error = _load_vasprun_tolerantly(Vasprun, path, parse_eigenvalues=True)
    if error is not None or vasprun is None:
        return VasprunElectronicEvidence(EVIDENCE_UNREADABLE, "vasprun.xml could not be parsed")
    if malformed:
        return VasprunElectronicEvidence(EVIDENCE_INCOMPLETE, "vasprun.xml is malformed or truncated")

    try:
        parameters = dict(getattr(vasprun, "parameters", {}) or {})
        incar = dict(getattr(vasprun, "incar", {}) or {})

        def setting(name: str) -> Any:
            return parameters.get(name, incar.get(name))

        channels = sorted(
            (getattr(vasprun, "eigenvalues", None) or {}).items(),
            key=lambda item: -int(item[0]),  # Spin.up (1) before Spin.down (-1)
        )
        eigenvalues = tuple(np.asarray(array)[:, :, 0].astype(float) for _, array in channels)
        occupations = tuple(np.asarray(array)[:, :, 1].astype(float) for _, array in channels)
        weights = np.asarray(getattr(vasprun, "actual_kpoints_weights", None) or (), dtype=float)
        kpoints_frac = np.asarray(getattr(vasprun, "actual_kpoints", None) or (), dtype=float)
        kpoints = getattr(vasprun, "kpoints", None)
        generation = _recorded_kpoint_generation(path)
        if generation is not None:
            style, declared, recorded_shift = generation
        else:
            # No VASP-generated mesh recorded (for example an explicit list).
            style = getattr(getattr(kpoints, "style", None), "name", None)
            declared, recorded_shift = None, None
        structure = getattr(vasprun, "final_structure", None)
        reciprocal = None
        if structure is not None:
            reciprocal = tuple(float(item) for item in structure.lattice.reciprocal_lattice.abc)
        try:
            converged = getattr(vasprun, "converged_electronic", None)
        except Exception:
            converged = None
        efermi = getattr(vasprun, "efermi", None)
        return VasprunElectronicEvidence(
            EVIDENCE_AVAILABLE,
            eigenvalues=eigenvalues,
            occupations=occupations,
            weights=weights,
            kpoints_frac=kpoints_frac,
            efermi=float(efermi) if efermi is not None else None,
            ispin=_int_or_none(setting("ISPIN")),
            lsorbit=bool(setting("LSORBIT") or False),
            lnoncollinear=bool(setting("LNONCOLLINEAR") or False),
            ismear=_int_or_none(setting("ISMEAR")),
            sigma=_float_or_none(setting("SIGMA")),
            icharg=_int_or_none(setting("ICHARG")),
            nelect=_float_or_none(setting("NELECT")),
            nbands=_int_or_none(setting("NBANDS")),
            algo=_optional_text(str(setting("ALGO"))) if setting("ALGO") is not None else None,
            lhfcalc=bool(setting("LHFCALC") or False),
            kpoints_style=style,
            declared_divisions=declared,
            kpoints_shift=recorded_shift,
            reciprocal_lengths=reciprocal,
            converged_electronic=None if converged is None else bool(converged),
        )
    except Exception:
        return VasprunElectronicEvidence(
            EVIDENCE_UNREADABLE,
            "electronic evidence could not be extracted from vasprun.xml",
        )


def electronic_character_input_from_lifecycle(
    analysis: LifecycleAnalysis,
    *,
    stage_index: int | None = None,
    reader=read_vasprun_electronic_evidence,
) -> ElectronicCharacterInput:
    """Select one stage of a local calculation and normalize its evidence.

    Raises :class:`StageSelectionError` when the stage is not identifiable.
    Ambiguous stage *evidence* (several bound directories holding a
    vasprun.xml) is reported as an assessment finding, not an exception.
    """

    workflow = analysis.bmd_workflow
    limitations: list[str] = []
    if workflow is not None:
        declared = workflow.workflow_stages
        if not declared:
            raise StageSelectionError("the workflow declares no stages")
        if stage_index is None:
            if len(declared) != 1:
                listing = ", ".join(
                    f"{index}: {_stage_label(index, item)}"
                    for index, item in enumerate(declared, start=1)
                )
                raise StageSelectionError(
                    "the workflow declares several stages; select one with --stage-index "
                    f"({listing})"
                )
            stage_index = 1
        if not 1 <= stage_index <= len(declared):
            raise StageSelectionError(
                f"stage index {stage_index} is not declared (stages 1 to {len(declared)})"
            )
        payload = declared[stage_index - 1]
        stage = SelectedStage(
            index=stage_index,
            label=_stage_label(stage_index, payload),
            stage_type=_stage_type(payload.get("stage_type")),
            theory=_optional_text(payload.get("theory")),
        )
        directories = tuple(
            dict.fromkeys(
                binding.path
                for binding in workflow.stage_bindings
                if binding.stage_index == stage_index
            )
        )
        subject_id = _submission_subject_id(workflow.submission, workflow.workflow_root.name)
        provenance = _local_producer_provenance(workflow.submission)
        scope = "bmd_compute_workflow"
        limitations.extend(workflow.record_limitations)
        if workflow.relocated:
            limitations.append("stage directories were resolved inside a relocated workflow snapshot")
    else:
        if analysis.calculation_kind == "none":
            raise StageSelectionError("no recognizable VASP calculation was found")
        if stage_index not in (None, 1):
            raise StageSelectionError("a direct VASP calculation has only stage 1")
        stage_index = 1
        stage = SelectedStage(index=1, label="Direct VASP calculation", stage_type="direct_vasp")
        directories = (analysis.directory,)
        subject_id = analysis.directory.name or "calculation"
        provenance = ProducerProvenance(status="unavailable")
        scope = "direct_vasp_calculation"

    candidates = tuple(directory for directory in directories if (directory / "vasprun.xml").is_file())
    if len(candidates) > 1:
        evidence = VasprunElectronicEvidence(
            EVIDENCE_AMBIGUOUS,
            "several directories bound to the selected stage contain a vasprun.xml",
        )
    elif not candidates:
        evidence = VasprunElectronicEvidence(
            EVIDENCE_MISSING,
            "no vasprun.xml was found for the selected stage",
        )
    else:
        evidence = reader(candidates[0] / "vasprun.xml")

    completion = next(
        (item for item in analysis.stage_completion if item.index == stage_index),
        None,
    )
    return ElectronicCharacterInput(
        subject_id=subject_id,
        calculation_type=analysis.calculation_kind,
        scope=scope,
        stage=stage,
        stage_completion=completion,
        evidence=evidence,
        producer_provenance=provenance,
        limitations=tuple(limitations),
        assessor_version=_package_version(),
    )


def assess_stage_directory(
    directory: Path | str,
    *,
    stage_index: int,
    stage_type: str,
    theory: str | None = None,
    label: str | None = None,
    subject_id: str | None = None,
) -> ElectronicCharacterAssessment:
    """Assess one completed stage from the files in a single directory.

    For a producer that holds one stage's outputs in a directory of its own
    (BMD Compute gives a later stage copies of the previous stage's
    ``vasprun.xml``, ``OUTCAR`` and ``OSZICAR``). The caller declares which
    stage the files belong to. Only ``directory`` is read: workflow records in
    parent directories are not discovered, nothing is written and no
    scheduler is contacted.

    Completion and convergence come from C1's direct-calculation analysis of
    that directory and the verdict from :func:`assess_electronic_character`,
    both unchanged. Missing, unreadable or incomplete evidence is an
    ``uncertain`` assessment, never an exception; only an invalid
    ``stage_index`` raises (:class:`ValueError`).
    """

    if isinstance(stage_index, bool) or not isinstance(stage_index, int) or stage_index < 1:
        raise ValueError("stage_index must be a positive integer")
    target = Path(directory).expanduser().resolve()
    analysis = _analyze_direct_vasp_directory(target)
    completion = next(iter(analysis.stage_completion), None)
    stage = SelectedStage(
        index=stage_index,
        label=label or _stage_label(stage_index, {"stage_type": stage_type, "theory": theory}),
        stage_type=_stage_type(stage_type),
        theory=_optional_text(theory),
    )
    if completion is not None:
        completion = replace(completion, index=stage_index, label=stage.label)
    return assess_electronic_character(
        ElectronicCharacterInput(
            subject_id=subject_id or target.name or "stage",
            calculation_type=analysis.calculation_kind,
            scope=SCOPE_STAGE_DIRECTORY,
            stage=stage,
            stage_completion=completion,
            evidence=read_vasprun_electronic_evidence(target / "vasprun.xml"),
            producer_provenance=ProducerProvenance(status="unavailable"),
            limitations=tuple(analysis.limitations),
            assessor_version=_package_version(),
        )
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Pure assessment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ValidEvidence:
    """Numerically validated arrays and scalars used by the classifier."""

    eigenvalues: tuple[Any, ...]
    occupations: tuple[Any, ...]
    weights: Any
    kpoints: Any
    efermi: float
    nelect: float | None
    sigma: float | None
    reciprocal_lengths: tuple[float, float, float] | None
    divisions: tuple[int, int, int] | None
    shift: tuple[float, float, float] | None


def assess_electronic_character(data: ElectronicCharacterInput) -> ElectronicCharacterAssessment:
    """The single C2 electronic-character verdict implementation."""

    import numpy as np

    stage_index = data.stage.index if data.stage is not None else None
    vasprun_ref = f"stage-{stage_index}-vasprun-electronic-structure"
    convergence_ref = f"stage-{stage_index}-convergence"
    references = (
        EvidenceReference(
            vasprun_ref,
            "artifact_derived_observation",
            "eigenvalues, occupations, k-point weights and settings from the selected stage's vasprun.xml",
            stage_index,
        ),
        EvidenceReference(
            convergence_ref,
            "trajectory_observation",
            "C1 stage completion and convergence evidence",
            stage_index,
        ),
    )
    findings: list[ElectronicFinding] = []
    limitations = list(data.limitations)
    limitations.append(
        "Thresholds are provisional method-1.0.0-c2 tolerances, not validated physical boundaries."
    )
    limitations.append(
        "Classification reflects the calculated electronic state of this stage only; "
        "it does not select ISMEAR, SIGMA or other VASP settings."
    )

    def done(status: str, classification: str, code: str, message: str, summary: ElectronicEvidenceSummary):
        findings.insert(0, ElectronicFinding(code, classification, message, (vasprun_ref, convergence_ref)))
        return ElectronicCharacterAssessment(
            subject_id=data.subject_id,
            calculation_type=data.calculation_type,
            scope=data.scope,
            stage=data.stage,
            assessment_status=status,
            classification=classification,
            findings=tuple(findings),
            evidence=summary,
            evidence_references=references,
            producer_provenance=data.producer_provenance,
            assessor_version=data.assessor_version,
            limitations=tuple(dict.fromkeys(limitations)),
        )

    def blocked(code: str, message: str, status: str = STATUS_INSUFFICIENT_EVIDENCE, summary=None):
        findings.append(ElectronicFinding(code, "unavailable", message, (vasprun_ref,)))
        return done(
            status,
            CLASSIFICATION_UNCERTAIN,
            "ELECTRONIC_CHARACTER_UNSUPPORTED" if status == STATUS_UNSUPPORTED else "ELECTRONIC_CHARACTER_UNDETERMINED",
            (
                "C2 does not support electronic-character assessment for this evidence."
                if status == STATUS_UNSUPPORTED
                else "The evidence does not support a definitive electronic-character classification."
            ),
            summary or ElectronicEvidenceSummary(),
        )

    if data.stage is None:
        return blocked("STAGE_UNAVAILABLE", "No calculation stage was selected.")
    if data.stage.stage_type in _PHONON_STAGE_TYPES:
        return blocked(
            "STAGE_TYPE_UNSUPPORTED",
            "Phonon stages are not supported by C2.",
            STATUS_UNSUPPORTED,
        )

    evidence = data.evidence
    if evidence.status != EVIDENCE_AVAILABLE:
        code = {
            EVIDENCE_MISSING: "VASPRUN_UNAVAILABLE",
            EVIDENCE_TOO_LARGE: "VASPRUN_EXCEEDS_PARSING_BOUND",
            EVIDENCE_INCOMPLETE: "VASPRUN_INCOMPLETE",
            EVIDENCE_AMBIGUOUS: "STAGE_EVIDENCE_AMBIGUOUS",
        }.get(evidence.status, "VASPRUN_UNREADABLE")
        return blocked(code, f"{evidence.reason or 'vasprun.xml evidence is unavailable'}.")

    # --- numerical validity: nothing below this point sees NaN, inf or bad types
    valid, invalid = _validate_numerical_evidence(evidence)
    if valid is None:
        if invalid == "fermi level":
            return blocked("FERMI_LEVEL_UNAVAILABLE", "vasprun.xml does not record a usable Fermi level.")
        if invalid == "eigenvalues and occupations":
            return blocked("EIGENVALUES_UNAVAILABLE", "Eigenvalues and occupations are not available.")
        return blocked(
            "INVALID_NUMERICAL_EVIDENCE",
            f"The {invalid} evidence is missing, non-numeric, non-finite or inconsistently shaped.",
        )

    # --- settings and treatment -------------------------------------------------
    noncollinear = evidence.lsorbit or evidence.lnoncollinear
    channel_count = len(valid.eigenvalues)
    if noncollinear:
        spin_treatment = SPIN_NONCOLLINEAR
        degeneracy = 1.0
        channel_names: tuple[str, ...] = ("spinor",)
    elif channel_count == 2:
        spin_treatment = SPIN_COLLINEAR
        degeneracy = 1.0
        channel_names = ("up", "down")
    else:
        spin_treatment = SPIN_NON_POLARIZED
        degeneracy = 2.0
        channel_names = ("total",)
    smearing_method = _smearing_method(evidence.ismear)
    weights = valid.weights
    weighted_mask = weights > THRESHOLDS["zero_weight_tolerance"]
    weighted_count = int(weighted_mask.sum())
    zero_count = int(len(weights) - weighted_count)
    mesh_status, mesh_message = _verify_generated_mesh(evidence, valid, weighted_mask)
    divisions = valid.divisions if mesh_status == "verified" else None
    spacing = None
    if divisions is not None and valid.reciprocal_lengths is not None:
        spacing = max(
            length / division for length, division in zip(valid.reciprocal_lengths, divisions)
        )

    completion = data.stage_completion
    base_summary = ElectronicEvidenceSummary(
        spin_treatment=spin_treatment,
        fermi_level_ev=_round(valid.efermi),
        smearing_method=smearing_method,
        ismear=evidence.ismear,
        sigma_ev=_round(valid.sigma),
        icharg=evidence.icharg,
        nelect=_round(valid.nelect),
        nbands=evidence.nbands,
        kpoints_style=(evidence.kpoints_style or "").lower() or None,
        mesh_divisions=valid.divisions,
        mesh_status=mesh_status,
        weighted_kpoints=weighted_count,
        zero_weight_kpoints=zero_count,
        max_reciprocal_spacing_inv_angstrom=_round(spacing),
        electronic_convergence=_convergence_text(completion, evidence),
    )

    # --- gates: unsupported evidence -------------------------------------------
    if evidence.algo and evidence.algo.strip().lower() in _UNSUPPORTED_ALGOS:
        return blocked(
            "ELECTRONIC_METHOD_UNSUPPORTED",
            f"ALGO = {evidence.algo} (many-body or response method) is not supported by C2.",
            STATUS_UNSUPPORTED,
            base_summary,
        )
    if evidence.ismear in _UNSUPPORTED_SMEARING:
        return blocked(
            "OCCUPATION_METHOD_UNSUPPORTED",
            f"ISMEAR = {evidence.ismear} ({_UNSUPPORTED_SMEARING[evidence.ismear]}) does not give "
            "self-consistent Fermi-level occupations.",
            STATUS_UNSUPPORTED,
            base_summary,
        )
    if smearing_method is None:
        return blocked(
            "OCCUPATION_METHOD_UNKNOWN",
            "The occupation (ISMEAR) method was not recorded or is not recognized.",
            STATUS_UNSUPPORTED,
            base_summary,
        )
    if (evidence.kpoints_style or "").lower().startswith("line"):
        return blocked(
            "BAND_PATH_SAMPLING_UNSUPPORTED",
            "Line-mode band-path sampling cannot establish electronic character.",
            STATUS_UNSUPPORTED,
            base_summary,
        )
    if evidence.icharg is not None and evidence.icharg >= 10 and mesh_status != "verified":
        return blocked(
            "NON_SELF_CONSISTENT_PATH_UNSUPPORTED",
            "A non-self-consistent run without a verified generated mesh (for example a band path) "
            "does not provide a Brillouin-zone Fermi level.",
            STATUS_UNSUPPORTED,
            base_summary,
        )

    # --- gates: completion and convergence (from C1) ----------------------------
    if completion is None or not completion.started:
        return blocked("STAGE_EXECUTION_UNAVAILABLE", "No execution evidence was acquired for the selected stage.", summary=base_summary)
    if completion.terminated_normally is False:
        return blocked(
            "STAGE_TERMINATION_NOT_ESTABLISHED",
            "VASP normal-termination evidence was not found for the selected stage.",
            summary=base_summary,
        )
    if completion.convergence.status != STAGE_CONVERGED:
        return blocked(
            "STAGE_CONVERGENCE_NOT_ESTABLISHED",
            "C1 did not establish convergence for the selected stage.",
            summary=base_summary,
        )
    if evidence.converged_electronic is False:
        return blocked(
            "ELECTRONIC_CONVERGENCE_NOT_REACHED",
            "vasprun.xml reports that the final electronic step did not converge.",
            summary=base_summary,
        )

    # --- gates: consistency of the evidence ------------------------------------
    if noncollinear and channel_count != 1:
        return blocked(
            "SPIN_EVIDENCE_INCONSISTENT",
            "Noncollinear evidence must have a single spinor channel.",
            summary=base_summary,
        )
    if not noncollinear and evidence.ispin is not None and evidence.ispin != channel_count:
        return blocked(
            "SPIN_EVIDENCE_INCONSISTENT",
            "ISPIN does not match the number of eigenvalue spin channels.",
            summary=base_summary,
        )
    if evidence.nbands is None:
        return blocked(
            "NBANDS_UNAVAILABLE",
            "NBANDS was not recorded, so the eigenvalue evidence cannot be checked for completeness.",
            summary=base_summary,
        )
    represented = {item.shape[1] for item in valid.eigenvalues}
    if represented != {int(evidence.nbands)}:
        return blocked(
            "NBANDS_INCONSISTENT",
            f"NBANDS = {int(evidence.nbands)} does not match the {sorted(represented)} bands present "
            "in the eigenvalue evidence.",
            summary=base_summary,
        )
    if weighted_count == 0 or abs(float(weights.sum()) - 1.0) > THRESHOLDS["weight_sum_tolerance"]:
        return blocked(
            "KPOINT_WEIGHTS_INVALID",
            "K-point weights are missing, zero or not normalized.",
            summary=base_summary,
        )
    tolerance = THRESHOLDS[
        "tetrahedron_occupation_range_tolerance"
        if smearing_method == "tetrahedron"
        else "occupation_range_tolerance"
    ]
    if any(
        float(np.min(item)) < -tolerance or float(np.max(item)) > 1.0 + tolerance
        for item in valid.occupations
    ):
        return blocked(
            "OCCUPATIONS_OUT_OF_RANGE",
            "Band occupations lie outside the expected per-state range.",
            summary=base_summary,
        )
    w = weights[weighted_mask]
    counts = [float((occ[weighted_mask] * w[:, None]).sum()) for occ in valid.occupations]
    total = degeneracy * sum(counts)
    if valid.nelect is not None and abs(total - valid.nelect) > THRESHOLDS["electron_count_tolerance"]:
        return blocked(
            "ELECTRON_COUNT_INCONSISTENT",
            f"Occupations account for {total:.4f} electrons but NELECT is {valid.nelect:g}.",
            summary=base_summary,
        )
    # The requested electron count is a separate fact from the integration
    # accuracy of the occupations: a deliberately fractional NELECT is never
    # rounded into an ordinary integer-filled state, and is not assumed metallic.
    requested = valid.nelect if valid.nelect is not None else total
    if abs(requested - round(requested)) > THRESHOLDS["nelect_integer_tolerance"] and (
        valid.nelect is not None or abs(requested - round(requested)) > THRESHOLDS["electron_count_tolerance"]
    ):
        return blocked(
            "FRACTIONAL_ELECTRON_COUNT",
            (
                f"NELECT = {valid.nelect:.8g} is not an integer; C2 does not classify calculations "
                "with a deliberately fractional electron count."
                if valid.nelect is not None
                else f"NELECT was not recorded and the occupations hold a non-integer {total:.4f} electrons."
            ),
            summary=base_summary,
        )
    if valid.nelect is None:
        limitations.append("NELECT was not recorded; the electron count was taken from the occupations.")
    if smearing_method != "tetrahedron" and valid.sigma is None:
        return blocked(
            "SMEARING_WIDTH_UNAVAILABLE",
            "SIGMA was not recorded for a smearing occupation method.",
            summary=base_summary,
        )
    if _occupations_contradict_energies(valid, weighted_mask, smearing_method):
        return blocked(
            "OCCUPATIONS_CONTRADICT_EIGENVALUES",
            "Occupations far from the Fermi level are inconsistent with the eigenvalues.",
            summary=base_summary,
        )

    # --- gates: sampling -------------------------------------------------------
    if weighted_count == 1:
        return blocked(
            "SINGLE_KPOINT_SAMPLING",
            "A single weighted k-point (for example Gamma-only) cannot establish electronic character of a solid.",
            summary=base_summary,
        )
    if mesh_status != "verified":
        return blocked(
            "KPOINT_MESH_NOT_ESTABLISHED" if mesh_status == "not_generated" else "KPOINT_MESH_INCONSISTENT",
            mesh_message,
            summary=base_summary,
        )
    if spacing is None:
        return blocked(
            "KPOINT_SAMPLING_UNDETERMINED",
            "The k-point mesh density could not be established (reciprocal lattice unavailable).",
            summary=base_summary,
        )
    if spacing > THRESHOLDS["max_reciprocal_spacing_inv_angstrom"] + 1e-9:
        return blocked(
            "SPARSE_KPOINT_SAMPLING",
            f"The k-point mesh spacing ({spacing:.3f} 1/Angstrom) is coarser than the "
            f"{THRESHOLDS['max_reciprocal_spacing_inv_angstrom']} 1/Angstrom required for a definitive classification.",
            summary=base_summary,
        )

    # --- per-channel classification --------------------------------------------
    edge_evidence = EDGE_EVIDENCE_WITH_ZERO_WEIGHT if zero_count else EDGE_EVIDENCE_WEIGHTED
    channel_results = [
        _classify_channel(name, eig, occ, weighted_mask, count, valid, smearing_method)
        for name, eig, occ, count in zip(channel_names, valid.eigenvalues, valid.occupations, counts)
    ]
    for result in channel_results:
        findings.append(
            ElectronicFinding(
                result.reason_code or "CHANNEL_ASSESSED",
                result.state,
                _channel_message(result),
                (vasprun_ref,),
            )
        )
    summary_fields = dict(base_summary.__dict__)
    summary_fields["channels"] = tuple(channel_results)
    summary_fields["band_edge_evidence"] = edge_evidence
    if zero_count:
        limitations.append(
            "Band edges and gaps include zero-weight k-points; occupations, electron counts "
            "and Fermi-level crossings use the weighted mesh only."
        )

    states = [result.state for result in channel_results]
    if CHANNEL_METALLIC in states:
        half = None
        if spin_treatment == SPIN_COLLINEAR:
            half = states.count(CHANNEL_GAPPED) == 1 and states.count(CHANNEL_METALLIC) == 1
        summary_fields["half_metallic"] = half
        return done(
            STATUS_ASSESSED,
            CLASSIFICATION_METALLIC,
            "HALF_METALLIC_STATE_ESTABLISHED" if half else "METALLIC_STATE_ESTABLISHED",
            (
                "One spin channel crosses the Fermi level while the other is gapped (half-metal)."
                if half
                else "Partially filled or overlapping bands cross the Fermi level."
            ),
            ElectronicEvidenceSummary(**summary_fields),
        )
    if all(state == CHANNEL_GAPPED for state in states):
        vbm, cbm, vbm_channel, cbm_channel, overall_type = _overall_band_edges(channel_results)
        gap = None if vbm is None else cbm.energy_ev - vbm.energy_ev
        summary_fields.update(vbm=vbm, cbm=cbm, vbm_channel=vbm_channel, cbm_channel=cbm_channel)
        if gap is None or gap < THRESHOLDS["min_reliable_gap_ev"]:
            findings.append(
                ElectronicFinding(
                    "SMALL_GAP",
                    "unavailable",
                    "The fundamental gap across spin channels is below the minimum reliable gap.",
                    (vasprun_ref,),
                )
            )
            return done(
                STATUS_INSUFFICIENT_EVIDENCE,
                CLASSIFICATION_UNCERTAIN,
                "ELECTRONIC_CHARACTER_UNDETERMINED",
                "The evidence does not support a definitive electronic-character classification.",
                ElectronicEvidenceSummary(**summary_fields),
            )
        summary_fields["band_gap_ev"] = _round(gap)
        summary_fields["gap_type"] = overall_type
        summary_fields["half_metallic"] = False if spin_treatment == SPIN_COLLINEAR else None
        limitations.append(
            "A Kohn-Sham band gap on a finite mesh is evidence of a gapped calculated state, "
            "not a converged or experimental band gap."
        )
        return done(
            STATUS_ASSESSED,
            CLASSIFICATION_NONMETALLIC,
            "NONMETALLIC_STATE_ESTABLISHED",
            "Every spin channel is integer-filled with the Fermi level inside a reliable gap.",
            ElectronicEvidenceSummary(**summary_fields),
        )
    return done(
        STATUS_INSUFFICIENT_EVIDENCE,
        CLASSIFICATION_UNCERTAIN,
        "ELECTRONIC_CHARACTER_UNDETERMINED",
        "The evidence does not support a definitive electronic-character classification.",
        ElectronicEvidenceSummary(**summary_fields),
    )


def _recorded_kpoint_generation(
    path: Path,
) -> tuple[str, tuple[int, int, int] | None, tuple[float, float, float] | None] | None:
    """Return (mode, divisions, effective shift) of the mesh VASP generated.

    Streams vasprun.xml only up to the first top-level ``<kpoints>`` block.
    Returns ``None`` when that block records no generated mesh (an explicit
    k-point list) or cannot be read.
    """

    import xml.etree.ElementTree as ElementTree

    from monty.io import zopen

    try:
        with zopen(str(path), "rb") as handle:
            return _first_generation(ElementTree.iterparse(handle, events=("end",)))
    except (ElementTree.ParseError, OSError, ValueError, EOFError):
        return None


def _first_generation(events: Any):
    """Scan parse events up to the first top-level <kpoints> block."""

    for _, element in events:
        if element.tag == "generation":
            param = (element.get("param") or "").strip().lower()
            values = {item.get("name"): (item.text or "") for item in element.findall("v")}
            if param.startswith("monk"):
                mode = "monkhorst"
            elif param.startswith("gamma"):
                mode = "gamma"
            elif param.startswith("auto"):
                mode = "automatic"
            elif param.startswith("list"):
                mode = "line_mode"
            else:
                mode = param or "unknown"
            divisions = _three(values.get("divisions"), int)
            shift = _three(values.get("shift"), float)
            return mode, divisions, shift  # type: ignore[return-value]
        if element.tag == "kpoints":
            return None
    return None


def _three(text: str | None, kind: Any) -> tuple[Any, Any, Any] | None:
    if text is None:
        return None
    parts = text.split()
    if len(parts) != 3:
        return None
    try:
        return tuple(kind(float(item)) if kind is int and float(item).is_integer() else kind(item) for item in parts)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _overall_band_edges(channels: list[SpinChannelSummary]):
    """Fundamental edges across spin channels, from the published channel edges.

    The gap is direct only when one channel holds both extrema at a shared
    k-point; extrema in different spin channels give an indirect (spin-flip)
    gap unless the gap-type resolution leaves the label undetermined.
    """

    tolerance = THRESHOLDS["band_extremum_degeneracy_tolerance_ev"]
    with_vbm = [item for item in channels if item.vbm is not None]
    cbm_energy = min(item.cbm.energy_ev for item in channels)  # type: ignore[union-attr]
    if not with_vbm:
        first = next(item for item in channels if item.cbm.energy_ev <= cbm_energy + tolerance)  # type: ignore[union-attr]
        return None, replace(first.cbm, energy_ev=cbm_energy), None, first.channel, None
    vbm_energy = max(item.vbm.energy_ev for item in with_vbm)  # type: ignore[union-attr]
    gap = cbm_energy - vbm_energy
    direct = [
        item for item in with_vbm
        if item.gap_type == "direct"
        and item.vbm.energy_ev >= vbm_energy - tolerance  # type: ignore[union-attr]
        and item.cbm.energy_ev <= cbm_energy + tolerance  # type: ignore[union-attr]
    ]
    if direct:
        chosen = direct[0]
        return (
            replace(chosen.vbm, energy_ev=vbm_energy),
            replace(chosen.cbm, energy_ev=cbm_energy),
            chosen.channel,
            chosen.channel,
            "direct",
        )
    vbm_source = next(item for item in with_vbm if item.vbm.energy_ev >= vbm_energy - tolerance)  # type: ignore[union-attr]
    cbm_source = next(item for item in channels if item.cbm.energy_ev <= cbm_energy + tolerance)  # type: ignore[union-attr]
    excess = min(
        item.min_direct_gap_lower_bound_ev for item in with_vbm if item.min_direct_gap_lower_bound_ev is not None
    ) - gap
    gap_type = "indirect" if excess >= THRESHOLDS["gap_type_resolution_ev"] else "undetermined"
    if any(item.gap_type == "undetermined" for item in with_vbm if item in (vbm_source, cbm_source)):
        gap_type = "undetermined"
    return (
        replace(vbm_source.vbm, energy_ev=vbm_energy),  # type: ignore[arg-type]
        replace(cbm_source.cbm, energy_ev=cbm_energy),  # type: ignore[arg-type]
        vbm_source.channel,
        cbm_source.channel,
        gap_type,
    )


def _three_vector(value: Any) -> tuple | None:
    """A length-3 sequence of scalars, from a tuple, list or one-dimensional
    NumPy array; ``None`` for any other structure (scalars, 0-d or nested arrays,
    strings, mappings)."""

    import numpy as np

    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.shape != (3,):
            return None
        return tuple(value.tolist())
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    if any(isinstance(item, (tuple, list, dict, str, bytes, np.ndarray)) for item in value):
        return None
    return tuple(value)


def _validate_numerical_evidence(
    evidence: VasprunElectronicEvidence,
) -> tuple[_ValidEvidence | None, str | None]:
    """Return validated float arrays, or the name of the first invalid category."""

    import numpy as np

    def array(value: Any, ndim: int) -> Any:
        if value is None:
            return None
        try:
            result = np.asarray(value)
            if result.dtype == object or not (
                np.issubdtype(result.dtype, np.floating) or np.issubdtype(result.dtype, np.integer)
            ) or np.issubdtype(result.dtype, np.bool_):
                return None
            result = result.astype(float)
        except (TypeError, ValueError):
            return None
        if result.ndim != ndim or not np.all(np.isfinite(result)):
            return None
        return result

    def scalar(value: Any, *, optional: bool, positive: bool = False, non_negative: bool = False):
        if value is None:
            return (True, None) if optional else (False, None)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            return False, None
        number = float(value)
        if not math.isfinite(number) or (positive and number <= 0) or (non_negative and number < 0):
            return False, None
        return True, number

    # Structural types first: nothing below may call len(), test truth, index,
    # iterate or call string methods on a value of unexpected type.
    if not isinstance(evidence.eigenvalues, (tuple, list)) or not isinstance(evidence.occupations, (tuple, list)):
        return None, "eigenvalue and occupation container"
    for name, value in (("k-point generation style", evidence.kpoints_style), ("ALGO", evidence.algo)):
        if value is not None and not isinstance(value, str):
            return None, name
    for name, value in (("LSORBIT", evidence.lsorbit), ("LNONCOLLINEAR", evidence.lnoncollinear), ("LHFCALC", evidence.lhfcalc)):
        if not isinstance(value, (bool, np.bool_)):
            return None, name
    if evidence.converged_electronic is not None and not isinstance(evidence.converged_electronic, (bool, np.bool_)):
        return None, "electronic-convergence flag"
    vectors: dict[str, tuple | None] = {}
    for name, field_name, value in (
        ("k-point mesh division", "divisions", evidence.declared_divisions),
        ("k-point mesh shift", "shift", evidence.kpoints_shift),
        ("reciprocal-lattice", "reciprocal", evidence.reciprocal_lengths),
    ):
        if value is None:
            vectors[field_name] = None
            continue
        items = _three_vector(value)
        if items is None:
            return None, name
        vectors[field_name] = items
    if len(evidence.eigenvalues) == 0 and len(evidence.occupations) == 0:
        return None, "eigenvalues and occupations"
    if len(evidence.eigenvalues) != len(evidence.occupations) or len(evidence.eigenvalues) > 2:
        return None, "eigenvalue and occupation container"
    eigenvalues = tuple(array(item, 2) for item in evidence.eigenvalues)
    if any(item is None for item in eigenvalues):
        return None, "eigenvalue"
    occupations = tuple(array(item, 2) for item in evidence.occupations)
    if any(item is None for item in occupations):
        return None, "occupation"
    weights = array(evidence.weights, 1)
    if weights is None or not len(weights) or np.any(weights < 0):
        return None, "k-point weight"
    kpoints = array(evidence.kpoints_frac, 2)
    if kpoints is None or kpoints.shape != (len(weights), 3):
        return None, "k-point coordinate"
    shapes = {item.shape for item in (*eigenvalues, *occupations)}
    if len(shapes) != 1 or next(iter(shapes))[0] != len(weights) or next(iter(shapes))[1] < 1:
        return None, "eigenvalue, occupation and k-point shape"
    if evidence.efermi is None:
        return None, "fermi level"
    ok, efermi = scalar(evidence.efermi, optional=False)
    if not ok:
        return None, "Fermi-level"
    ok, nelect = scalar(evidence.nelect, optional=True, positive=True)
    if not ok:
        return None, "NELECT"
    ok, sigma = scalar(evidence.sigma, optional=True, non_negative=True)
    if not ok:
        return None, "SIGMA"
    reciprocal = None
    if vectors["reciprocal"] is not None:
        values = array(vectors["reciprocal"], 1)
        if values is None or values.shape != (3,) or np.any(values <= 0):
            return None, "reciprocal-lattice"
        reciprocal = tuple(float(item) for item in values)
    divisions = None
    if vectors["divisions"] is not None:
        raw = vectors["divisions"]
        if (
            len(raw) != 3
            or any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in raw)
            or any(not 1 <= int(item) <= THRESHOLDS["max_mesh_division"] for item in raw)
        ):
            return None, "k-point mesh division"
        divisions = tuple(int(item) for item in raw)
    shift = None
    if vectors["shift"] is not None:
        values = array(vectors["shift"], 1)
        if values is None or values.shape != (3,):
            return None, "k-point mesh shift"
        shift = tuple(float(item) for item in values)
    for name, value in (("ISMEAR", evidence.ismear), ("NBANDS", evidence.nbands), ("ICHARG", evidence.icharg), ("ISPIN", evidence.ispin)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer))):
            return None, name
    if evidence.nbands is not None and int(evidence.nbands) < 1:
        return None, "NBANDS"
    return (
        _ValidEvidence(
            eigenvalues=eigenvalues,
            occupations=occupations,
            weights=weights,
            kpoints=kpoints,
            efermi=efermi,
            nelect=nelect,
            sigma=sigma,
            reciprocal_lengths=reciprocal,  # type: ignore[arg-type]
            divisions=divisions,  # type: ignore[arg-type]
            shift=shift,  # type: ignore[arg-type]
        ),
        None,
    )


def _verify_generated_mesh(
    evidence: VasprunElectronicEvidence,
    valid: _ValidEvidence,
    weighted_mask: Any,
) -> tuple[str, str]:
    """Check that the weighted k-points are the irreducible part of the
    VASP-generated mesh recorded in vasprun.xml.

    No mesh is inferred from coordinates. The checks use only the recorded
    divisions and shift: every weighted point must lie on that mesh, and its
    weight times the number of mesh points must be an integer multiplicity
    between 1 and the largest possible k-point star, summing to the mesh size.
    This verifies consistency with symmetry reduction without reconstructing
    the symmetry operations.
    """

    import numpy as np

    style = (evidence.kpoints_style or "").lower()
    if style not in _GENERATED_MESH_STYLES or valid.divisions is None:
        return (
            "not_generated",
            "The k-points are not a VASP-generated regular mesh (for example an explicit list), "
            "so the sampling density cannot be established.",
        )
    divisions = np.asarray(valid.divisions, dtype=float)
    if valid.shift is not None:
        offsets = np.asarray(valid.shift, dtype=float)
    else:
        # VASP convention when the effective shift was not recorded; a wrong
        # guess cannot pass the on-mesh check below.
        offsets = np.where((divisions % 2 == 0) & (style == "monkhorst"), 0.5, 0.0)
    points = valid.kpoints[weighted_mask]
    scaled = points * divisions - offsets
    if np.any(np.abs(scaled - np.round(scaled)) > THRESHOLDS["mesh_coordinate_tolerance"] * max(1.0, float(divisions.max()))):
        return "inconsistent", "Weighted k-points do not lie on the recorded k-point mesh."
    total = float(np.prod(divisions))
    multiplicity = valid.weights[weighted_mask] * total
    rounded = np.round(multiplicity)
    if (
        np.any(np.abs(multiplicity - rounded) > THRESHOLDS["mesh_multiplicity_tolerance"] * np.maximum(1.0, rounded))
        or np.any(rounded < 1)
        or np.any(rounded > THRESHOLDS["max_kpoint_multiplicity"])
        or int(rounded.sum()) != int(total)
    ):
        return (
            "inconsistent",
            "K-point weights are not integer symmetry multiplicities of the recorded mesh.",
        )
    return "verified", "The weighted k-points are consistent with the recorded generated mesh."


def _classify_channel(
    name: str,
    eig: Any,
    occ: Any,
    weighted_mask: Any,
    count: float,
    valid: _ValidEvidence,
    smearing_method: str,
) -> SpinChannelSummary:
    import numpy as np

    efermi = valid.efermi
    margin = THRESHOLDS["fermi_crossing_margin_ev"]
    edge_tolerance = THRESHOLDS["fermi_edge_tolerance_ev"]
    e_weighted = eig[weighted_mask]
    o_weighted = occ[weighted_mask]
    w = valid.weights[weighted_mask]
    nbands = eig.shape[1]
    nearest = int(round(count))
    deviation = abs(count - nearest)
    if deviation <= THRESHOLDS["integer_filling_tolerance"]:
        filling = "integer"
    elif deviation >= THRESHOLDS["fractional_filling_min"]:
        filling = "fractional"
    else:
        filling = "near_integer"

    def result(state: str, code: str, **extra: Any) -> SpinChannelSummary:
        return SpinChannelSummary(
            channel=name,
            state=state,
            electron_count=_round(count, 4),
            filling=filling,
            reason_code=code,
            **extra,
        )

    edge_occupation = THRESHOLDS["edge_occupation_tolerance"]
    if float(np.max(o_weighted[:, -1])) > edge_occupation:
        return result(CHANNEL_UNCERTAIN, "INSUFFICIENT_BANDS_ABOVE_FERMI")

    clear_crossing = bool(
        np.any((e_weighted.min(axis=0) < efermi - margin) & (e_weighted.max(axis=0) > efermi + margin))
    )
    if filling == "near_integer":
        return result(CHANNEL_UNCERTAIN, "FILLING_NEAR_INTEGER")
    if filling == "fractional":
        if clear_crossing:
            return result(CHANNEL_METALLIC, "PARTIALLY_FILLED_BAND_CROSSES_FERMI")
        return result(CHANNEL_UNCERTAIN, "FRACTIONAL_FILLING_WITHOUT_FERMI_CROSSING")

    if nearest >= nbands:
        return result(CHANNEL_UNCERTAIN, "INSUFFICIENT_BANDS_ABOVE_FERMI")
    # Band edges over every k-point, weighted and zero-weight: one evidence set
    # for the channel's edges, gap, direct gap and gap type.
    edges = _channel_band_edges(eig, valid, weighted_mask, nearest)
    vbm, cbm = edges.vbm, edges.cbm
    gap = None if vbm is None else cbm.energy_ev - vbm.energy_ev
    extra: dict[str, Any] = {
        "vbm": vbm,
        "cbm": cbm,
        "band_gap_ev": _round(gap),
        "gap_type": edges.gap_type,
        "min_direct_gap_ev": _round(edges.min_direct_gap),
        "min_direct_gap_lower_bound_ev": _round(edges.min_direct_gap_lower_bound),
        "kpoint_equivalence_ambiguous": edges.equivalence_ambiguous,
    }
    if gap is not None and gap <= -THRESHOLDS["band_overlap_min_ev"]:
        if clear_crossing:
            return result(CHANNEL_METALLIC, "BAND_OVERLAP_AT_FERMI", **extra)
        return result(CHANNEL_UNCERTAIN, "BAND_OVERLAP_WITHOUT_FERMI_CROSSING", **extra)
    if gap is not None and gap < THRESHOLDS["min_reliable_gap_ev"]:
        return result(CHANNEL_UNCERTAIN, "SMALL_GAP", **extra)
    vbm_excess = -math.inf if vbm is None else vbm.energy_ev - efermi
    cbm_deficit = efermi - cbm.energy_ev
    if vbm_excess > edge_tolerance or cbm_deficit > edge_tolerance:
        # A band edge passes the Fermi level: shallow (within the crossing
        # margin) or deep, it is not a gapped channel.
        code = (
            "SHALLOW_FERMI_CROSSING"
            if max(vbm_excess, cbm_deficit) <= margin
            else "FERMI_LEVEL_OUTSIDE_GAP"
        )
        return result(CHANNEL_UNCERTAIN, code, **extra)
    if clear_crossing:
        return result(CHANNEL_UNCERTAIN, "INTEGER_FILLING_WITH_FERMI_CROSSING", **extra)
    # Electrons smeared into states above the gap. A Fermi level pinned at a
    # band edge may leave edge occupations slightly non-integer on one side;
    # that alone is not evidence of charge spilling across the gap.
    spilled = float((np.clip(o_weighted[:, nearest:], 0.0, None) * w[:, None]).sum())
    if spilled > edge_occupation:
        return result(CHANNEL_UNCERTAIN, "SMEARING_SPILLS_ACROSS_GAP", **extra)
    if (
        smearing_method != "tetrahedron"
        and valid.sigma is not None
        and gap is not None
        and gap < THRESHOLDS["smearing_gap_ratio"] * valid.sigma
    ):
        return result(CHANNEL_UNCERTAIN, "GAP_COMPARABLE_TO_SMEARING", **extra)
    return result(CHANNEL_GAPPED, "CHANNEL_GAP_ESTABLISHED", **extra)


@dataclass(frozen=True, kw_only=True)
class _ChannelEdges:
    vbm: BandEdge | None
    cbm: BandEdge
    gap_type: str | None
    min_direct_gap: float | None
    min_direct_gap_lower_bound: float | None = None
    equivalence_ambiguous: bool = False

    def __post_init__(self) -> None:
        # Energies are real numbers or None, never flags; the flag is a bool.
        for name in ("min_direct_gap", "min_direct_gap_lower_bound"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise TypeError(f"{name} must be an energy in eV or None, not {value!r}")
        if not isinstance(self.equivalence_ambiguous, bool):
            raise TypeError("equivalence_ambiguous must be a bool")


def _periodic_distance(first: Any, second: Any) -> Any:
    """Largest per-axis fractional distance modulo reciprocal-lattice vectors."""

    import numpy as np

    difference = np.asarray(first) - np.asarray(second)
    return np.max(np.abs(difference - np.round(difference)), axis=-1)


def _reduced_key(point: Any) -> tuple[float, float, float]:
    """Reduced coordinate in [0, 1), rounded for publication."""

    values = []
    for item in point:
        value = round(float(item) % 1.0, 4)
        values.append(0.0 if value >= 1.0 else value + 0.0)
    return tuple(values)  # type: ignore[return-value]


def _kpoint_classes(valid: _ValidEvidence) -> tuple[dict[tuple[float, float, float], list[int]], set]:
    """Group k-points that coincide modulo reciprocal-lattice vectors.

    Two k-points are linked when their periodic fractional distance is at most
    ``kpoint_equivalence_tolerance``; classes are the connected components of
    that graph, which do not depend on k-point order. Because approximate
    equality is not transitive, a class whose members are not all pairwise
    within tolerance (a chain of near-equivalent points) is returned as
    ambiguous. Each class is keyed by the lexicographically smallest reduced
    coordinate of its members.
    """

    import numpy as np

    tolerance = THRESHOLDS["kpoint_equivalence_tolerance"]
    points = valid.kpoints
    parent = list(range(len(points)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index in range(len(points) - 1):
        close = np.nonzero(_periodic_distance(points[index + 1:], points[index]) <= tolerance)[0]
        for offset in close:
            left, right = find(index), find(index + 1 + int(offset))
            if left != right:
                parent[max(left, right)] = min(left, right)
    members: dict[int, list[int]] = {}
    for index in range(len(points)):
        members.setdefault(find(index), []).append(index)
    classes: dict[tuple[float, float, float], list[int]] = {}
    ambiguous: set = set()
    for group in members.values():
        key = min(_reduced_key(points[index]) for index in group)
        classes[key] = group
        if len(group) >= 2:
            block = points[group]
            spread = max(
                float(np.max(_periodic_distance(block, block[position]))) for position in range(len(group))
            )
            if spread > tolerance:
                ambiguous.add(key)
    return classes, ambiguous


def _channel_band_edges(eig: Any, valid: _ValidEvidence, weighted_mask: Any, nearest: int) -> _ChannelEdges:
    """Band edges, smallest direct gap and gap type for one integer-filled channel.

    Every extremum within the degeneracy tolerance is considered, over
    k-points identified modulo reciprocal-lattice vectors. Representatives are
    chosen independently of k-point order (lexicographically first reduced
    coordinate). The gap is direct when some k-point holds both a valence
    maximum and a conduction minimum within tolerance; indirect when the
    smallest direct gap exceeds the fundamental gap by at least the gap-type
    resolution; otherwise undetermined.
    """

    tolerance = THRESHOLDS["band_extremum_degeneracy_tolerance_ev"]
    class_map, ambiguous = _kpoint_classes(valid)
    classes = sorted(class_map.items())
    conduction = {key: min(float(eig[index, nearest]) for index in members) for key, members in classes}
    weighted = {key: any(bool(weighted_mask[index]) for index in members) for key, members in classes}
    cbm_energy = min(conduction.values())
    cbm_keys = [key for key, _ in classes if conduction[key] <= cbm_energy + tolerance]

    def edge(energy: float, keys: list, representative) -> BandEdge:
        return BandEdge(
            energy_ev=energy,
            kpoint=representative,
            kpoint_weight="weighted" if weighted[representative] else "zero_weight",
            degenerate_kpoints=len(keys),
        )

    if nearest == 0:
        # No occupied band: no VBM, gap type or direct gap (nor its bound).
        return _ChannelEdges(
            vbm=None,
            cbm=edge(cbm_energy, cbm_keys, cbm_keys[0]),
            gap_type=None,
            min_direct_gap=None,
            min_direct_gap_lower_bound=None,
            equivalence_ambiguous=bool(set(cbm_keys) & ambiguous),
        )
    valence = {key: max(float(eig[index, nearest - 1]) for index in members) for key, members in classes}
    vbm_energy = max(valence.values())
    vbm_keys = [key for key, _ in classes if valence[key] >= vbm_energy - tolerance]
    gap = cbm_energy - vbm_energy
    # Direct gap of each class, combining its members. For an ambiguous class
    # this combines states at k-points that are not all equivalent, so it is
    # only a lower bound: every consistent reading of the chain merges a subset
    # of the class, whose combined gap cannot be smaller.
    direct_gaps = {key: conduction[key] - valence[key] for key, _ in classes}
    # Direct gaps supported by sampled k-points: the combined gap of an
    # unambiguous class, or the same-k gap of a single member of an ambiguous
    # class. The true smallest direct gap lies between the two minima.
    supported = {
        key: (
            min(float(eig[index, nearest]) - float(eig[index, nearest - 1]) for index in members)
            if key in ambiguous
            else direct_gaps[key]
        )
        for key, members in classes
    }
    lower_bound = min(direct_gaps.values())
    direct_gap_ambiguous = lower_bound < min(supported.values())
    min_direct = None if direct_gap_ambiguous else lower_bound
    near_direct = [key for key, value in direct_gaps.items() if value <= gap + tolerance]
    edges_ambiguous = bool(ambiguous & (set(vbm_keys) | set(cbm_keys) | set(near_direct)))
    flagged = edges_ambiguous or direct_gap_ambiguous

    def result(gap_type: str, key=None) -> _ChannelEdges:
        return _ChannelEdges(
            vbm=edge(vbm_energy, vbm_keys, vbm_keys[0] if key is None else key),
            cbm=edge(cbm_energy, cbm_keys, cbm_keys[0] if key is None else key),
            gap_type=gap_type,
            min_direct_gap=min_direct,
            min_direct_gap_lower_bound=lower_bound,
            equivalence_ambiguous=flagged,
        )

    # A direct gap is established only by an unambiguous class: one whose
    # members are all equivalent and that holds both extrema, or whose direct
    # gap equals the fundamental gap within tolerance.
    shared = [key for key in vbm_keys if key in set(cbm_keys) and key not in ambiguous]
    if shared:
        return result("direct", shared[0])
    established = [key for key in near_direct if key not in ambiguous]
    if established:
        return result("direct", min(established))
    if ambiguous & set(near_direct):
        # An ambiguous chain might hold both extrema: the label is not chosen.
        return result("undetermined")
    # No class can be direct; the lower bound decides indirect against the
    # gap-type resolution under every reading of the equivalence.
    excess = lower_bound - gap
    return result("indirect" if excess >= THRESHOLDS["gap_type_resolution_ev"] else "undetermined")


def _occupations_contradict_energies(
    valid: _ValidEvidence,
    weighted_mask: Any,
    smearing_method: str,
) -> bool:
    import numpy as np

    if smearing_method == "tetrahedron":
        window = THRESHOLDS["far_window_tetrahedron_ev"]
    else:
        window = max(
            THRESHOLDS["far_window_sigma_multiple"] * float(valid.sigma or 0.0),
            THRESHOLDS["far_window_min_ev"],
        )
    tolerance = THRESHOLDS["far_occupation_tolerance"]
    for eig, occ in zip(valid.eigenvalues, valid.occupations):
        e = eig[weighted_mask]
        o = occ[weighted_mask]
        if np.any(o[e < valid.efermi - window] < 1.0 - tolerance):
            return True
        if np.any(o[e > valid.efermi + window] > tolerance):
            return True
    return False


def _smearing_method(ismear: int | None) -> str | None:
    if ismear is None or isinstance(ismear, bool):
        return None
    if ismear in _SMEARING_METHODS:
        return _SMEARING_METHODS[ismear]
    if ismear >= 1:
        return "methfessel_paxton"
    if ismear in _TETRAHEDRON_METHODS:
        return "tetrahedron"
    return None


def _convergence_text(completion: StageCompletion | None, evidence: VasprunElectronicEvidence) -> str:
    if completion is None or not completion.started:
        return "unavailable"
    if evidence.converged_electronic is False:
        return "not_converged"
    return completion.convergence.status


def _channel_message(result: SpinChannelSummary) -> str:
    label = {
        "total": "The (spin-degenerate) channel",
        "spinor": "The spinor channel",
        "up": "The spin-up channel",
        "down": "The spin-down channel",
    }.get(result.channel, f"Channel {result.channel}")
    text = {
        "PARTIALLY_FILLED_BAND_CROSSES_FERMI": "has fractional filling and a band clearly crossing the Fermi level.",
        "BAND_OVERLAP_AT_FERMI": "is integer-filled, but overlapping bands cross the Fermi level (semimetal).",
        "CHANNEL_GAP_ESTABLISHED": "is integer-filled with the Fermi level inside a reliable gap.",
        "INSUFFICIENT_BANDS_ABOVE_FERMI": "has no empty band above the Fermi level at every k-point.",
        "FILLING_NEAR_INTEGER": "has an electron count too close to an integer to be clearly fractional, but not clearly integer.",
        "FRACTIONAL_FILLING_WITHOUT_FERMI_CROSSING": "has fractional filling without a clear Fermi-level crossing.",
        "BAND_OVERLAP_WITHOUT_FERMI_CROSSING": "shows band overlap that does not clearly cross the Fermi level.",
        "SMALL_GAP": "has a gap below the minimum reliable gap.",
        "SHALLOW_FERMI_CROSSING": "has a band edge passing the Fermi level by less than the crossing margin.",
        "FERMI_LEVEL_OUTSIDE_GAP": "has a gap that does not contain the recorded Fermi level.",
        "INTEGER_FILLING_WITH_FERMI_CROSSING": "is integer-filled, yet a band crosses the Fermi level.",
        "SMEARING_SPILLS_ACROSS_GAP": "has smearing-induced occupation of states above the gap.",
        "GAP_COMPARABLE_TO_SMEARING": "has a gap comparable to the smearing width.",
    }.get(result.reason_code or "", "was assessed.")
    return f"{label} {text}"


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    rounded = round(number, digits)
    return 0.0 if rounded == 0 else rounded


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        number = float(value)
        return int(number) if math.isfinite(number) and number.is_integer() else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Serialization and rendering (one immutable result)
# ---------------------------------------------------------------------------


def build_lifecycle_electronic_character_assessment(
    analysis: LifecycleAnalysis,
    *,
    stage_index: int | None = None,
) -> ElectronicCharacterAssessment:
    return assess_electronic_character(
        electronic_character_input_from_lifecycle(analysis, stage_index=stage_index)
    )


def serialize_electronic_character_assessment(
    assessment: ElectronicCharacterAssessment,
) -> dict[str, Any]:
    """Bounded, deterministic JSON-safe form; no arrays or absolute paths."""

    evidence = assessment.evidence
    stage = assessment.stage
    return {
        "schema_id": assessment.schema_id,
        "schema_version": assessment.schema_version,
        "assessment_id": assessment.assessment_id,
        "method_version": assessment.method_version,
        "assessment_status": assessment.assessment_status,
        "classification": assessment.classification,
        "subject": {
            "id": assessment.subject_id,
            "calculation_type": assessment.calculation_type,
            "scope": assessment.scope,
            "stage": (
                None
                if stage is None
                else {
                    "index": stage.index,
                    "label": stage.label,
                    "stage_type": stage.stage_type,
                    "theory": stage.theory,
                }
            ),
        },
        "electronic_evidence": {
            "spin_treatment": evidence.spin_treatment,
            "fermi_level_ev": evidence.fermi_level_ev,
            "band_edge_evidence": evidence.band_edge_evidence,
            "band_gap_ev": evidence.band_gap_ev,
            "gap_type": evidence.gap_type,
            "vbm": _edge_payload(evidence.vbm, evidence.vbm_channel),
            "cbm": _edge_payload(evidence.cbm, evidence.cbm_channel),
            "half_metallic": evidence.half_metallic,
            "electronic_convergence": evidence.electronic_convergence,
            "spin_channels": [
                {
                    "channel": item.channel,
                    "state": item.state,
                    "reason_code": item.reason_code,
                    "electron_count": item.electron_count,
                    "filling": item.filling,
                    "vbm": _edge_payload(item.vbm),
                    "cbm": _edge_payload(item.cbm),
                    "band_gap_ev": item.band_gap_ev,
                    "gap_type": item.gap_type,
                    "min_direct_gap_ev": item.min_direct_gap_ev,
                    "min_direct_gap_lower_bound_ev": item.min_direct_gap_lower_bound_ev,
                    "kpoint_equivalence_ambiguous": item.kpoint_equivalence_ambiguous,
                }
                for item in evidence.channels
            ],
            "occupation_method": {
                "smearing_method": evidence.smearing_method,
                "ismear": evidence.ismear,
                "sigma_ev": evidence.sigma_ev,
            },
            "calculation": {
                "icharg": evidence.icharg,
                "nelect": evidence.nelect,
                "nbands": evidence.nbands,
            },
            "kpoints": {
                "generation_style": evidence.kpoints_style,
                "mesh_divisions": list(evidence.mesh_divisions) if evidence.mesh_divisions else None,
                "mesh_status": evidence.mesh_status,
                "weighted_count": evidence.weighted_kpoints,
                "zero_weight_count": evidence.zero_weight_kpoints,
                "max_reciprocal_spacing_inv_angstrom": evidence.max_reciprocal_spacing_inv_angstrom,
            },
        },
        "thresholds": {key: assessment.thresholds[key] for key in sorted(assessment.thresholds)},
        "findings": [
            {
                "reason_code": finding.reason_code,
                "status": finding.status,
                "message": finding.message,
                "evidence_refs": list(finding.evidence_refs),
            }
            for finding in assessment.findings
        ],
        "evidence_references": [
            {
                "id": reference.evidence_id,
                "evidence_type": reference.evidence_type,
                "source": reference.source,
                "stage_index": reference.stage_index,
            }
            for reference in assessment.evidence_references
        ],
        "producer_provenance": {
            "status": assessment.producer_provenance.status,
            "producer": assessment.producer_provenance.producer,
            "repository": assessment.producer_provenance.repository,
            "git_commit": assessment.producer_provenance.git_commit,
            "git_state": assessment.producer_provenance.git_state,
        },
        "assessor_provenance": {
            "name": "BMD Check",
            "package": "bmd-agent",
            "version": assessment.assessor_version,
        },
        "limitations": list(assessment.limitations),
    }


def render_electronic_character_assessment(assessment: ElectronicCharacterAssessment) -> str:
    """Human-readable view of the same immutable result used for JSON."""

    payload = serialize_electronic_character_assessment(assessment)
    evidence = payload["electronic_evidence"]
    stage = payload["subject"]["stage"]
    lines = [
        "Electronic Character Assessment",
        "===============================",
        f"Subject: {payload['subject']['id']}",
        f"Scope: {payload['subject']['scope']}",
        (
            f"Stage: {stage['index']} ({stage['label']}, {stage['stage_type']})"
            if stage
            else "Stage: unavailable"
        ),
        f"Method: {payload['assessment_id']} {payload['method_version']}",
        f"Assessment status: {payload['assessment_status']}",
        f"Classification: {payload['classification'].upper()}",
    ]
    if evidence["spin_treatment"]:
        lines.append(f"Spin treatment: {evidence['spin_treatment']}")
    if evidence["band_gap_ev"] is not None:
        lines.append(f"Band gap: {evidence['band_gap_ev']} eV ({evidence['gap_type']})")
        lines.append(f"Band-edge evidence: {evidence['band_edge_evidence']}")
    if evidence["half_metallic"]:
        lines.append("Half-metallic: yes")
    kpoints = evidence["kpoints"]
    if kpoints["weighted_count"] is not None:
        mesh = kpoints["mesh_divisions"]
        lines.append(
            "K-points: "
            f"{kpoints['weighted_count']} weighted, {kpoints['zero_weight_count']} zero-weight"
            + (f", mesh {'x'.join(str(item) for item in mesh)} ({kpoints['mesh_status']})" if mesh else f", mesh {kpoints['mesh_status']}")
            + (
                f", max spacing {kpoints['max_reciprocal_spacing_inv_angstrom']} 1/Angstrom"
                if kpoints["max_reciprocal_spacing_inv_angstrom"] is not None
                else ""
            )
        )
    for channel in evidence["spin_channels"]:
        details = [f"{channel['state']}", f"electrons {channel['electron_count']} ({channel['filling']})"]
        if channel["band_gap_ev"] is not None:
            details.append(
                f"gap {channel['band_gap_ev']} eV ({channel['gap_type']}; "
                f"VBM {channel['vbm']['energy_ev']} eV, CBM {channel['cbm']['energy_ev']} eV)"
            )
        lines.append(f"  channel {channel['channel']}: " + ", ".join(details))
    lines.extend(("", "Findings:"))
    lines.extend(
        f"  [{finding['status']}] {finding['reason_code']}: {finding['message']}"
        for finding in payload["findings"]
    )
    provenance = payload["producer_provenance"]
    if provenance["status"] == "available":
        details = provenance["git_commit"] or "commit unavailable"
        if provenance["git_state"]:
            details += f" ({provenance['git_state']})"
        lines.extend(("", f"Producer provenance: {details}"))
    if payload["limitations"]:
        lines.extend(("", "Limitations:"))
        lines.extend(f"  - {item}" for item in payload["limitations"])
    return "\n".join(lines) + "\n"


def _edge_payload(edge: BandEdge | None, channel: str | None = None) -> dict[str, Any] | None:
    if edge is None:
        return None
    payload: dict[str, Any] = {
        "energy_ev": _round(edge.energy_ev),
        "kpoint": list(edge.kpoint) if edge.kpoint is not None else None,
        "kpoint_weight": edge.kpoint_weight,
        "degenerate_kpoints": edge.degenerate_kpoints,
    }
    if channel is not None:
        payload["channel"] = channel
    return payload
