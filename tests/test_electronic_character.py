"""C2 electronic-character assessment.

Synthetic evidence exercises each decision path of the pure assessment;
recorded VASP outputs (``tests/fixtures/electronic_character``) exercise the
real parser and the end-to-end CLI.
"""

from __future__ import annotations

from dataclasses import replace
import gzip
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from bmd_agent import cli
from bmd_agent.resources import electronic_character as ec
from bmd_agent.resources.assessment import ProducerProvenance
from bmd_agent.resources.completion import (
    STAGE_CONVERGED,
    STAGE_CONVERGENCE_UNDETERMINED,
    StageCompletion,
    StageConvergence,
)
from bmd_agent.resources.electronic_character import (
    ElectronicCharacterInput,
    SelectedStage,
    StageSelectionError,
    VasprunElectronicEvidence,
    assess_electronic_character,
    electronic_character_input_from_lifecycle,
    read_vasprun_electronic_evidence,
    render_electronic_character_assessment,
    serialize_electronic_character_assessment,
)
from bmd_agent.resources.lifecycle import analyze_calculation_directory

import test_lifecycle as lifecycle_fixtures


FIXTURES = Path(__file__).parent / "fixtures" / "electronic_character"


# --- synthetic evidence --------------------------------------------------------------------


def mesh_kpoints(n: int = 4) -> np.ndarray:
    """Sixteen points of a Gamma-centred 4x4x4 mesh, each standing for a star
    of four (multiplicity 4): a symmetry-reduced list consistent with the mesh."""

    return np.array([[i / n, j / n, 0.0] for i in range(n) for j in range(n)], dtype=float)


def evidence(
    eigenvalues,
    occupations,
    *,
    efermi: float | None = 0.0,
    weights=None,
    kpoints=None,
    ismear: int = -5,
    sigma: float = 0.05,
    nelect: float | None = None,
    lsorbit: bool = False,
    divisions=(4, 4, 4),
    reciprocal=(2.0, 2.0, 2.0),
    converged_electronic: bool | None = True,
    **extra,
) -> VasprunElectronicEvidence:
    eigenvalues = tuple(np.asarray(item, dtype=float) for item in eigenvalues)
    occupations = tuple(np.asarray(item, dtype=float) for item in occupations)
    nk = eigenvalues[0].shape[0]
    if weights is None:
        weights = np.full(nk, 1.0 / nk)
    if kpoints is None:
        kpoints = mesh_kpoints()[:nk] if nk <= 16 else np.zeros((nk, 3))
    if nelect is None:
        degeneracy = 1.0 if (lsorbit or len(eigenvalues) == 2) else 2.0
        mask = np.asarray(weights) > 0
        nelect = degeneracy * sum(
            float((occ[mask] * np.asarray(weights)[mask][:, None]).sum()) for occ in occupations
        )
    return VasprunElectronicEvidence(
        "available",
        eigenvalues=eigenvalues,
        occupations=occupations,
        weights=np.asarray(weights, dtype=float),
        kpoints_frac=np.asarray(kpoints, dtype=float),
        efermi=efermi,
        ispin=None if lsorbit else len(eigenvalues),
        lsorbit=lsorbit,
        lnoncollinear=lsorbit,
        ismear=ismear,
        sigma=sigma,
        icharg=extra.pop("icharg", 2),
        nelect=nelect,
        nbands=eigenvalues[0].shape[1],
        algo=extra.pop("algo", "Normal"),
        kpoints_style=extra.pop("kpoints_style", "Gamma"),
        declared_divisions=divisions,
        reciprocal_lengths=reciprocal,
        converged_electronic=converged_electronic,
        **extra,
    )


def gapped_channel(nk: int = 16, *, vbm: float = -0.5, cbm: float = 1.0, nfilled: int = 2, nbands: int = 4):
    """Bands dispersing over k; ``nfilled`` occupied bands below a gap."""

    k = np.linspace(0.0, 1.0, nk)
    eig = np.zeros((nk, nbands))
    for band in range(nbands):
        if band < nfilled:
            top = vbm - 2.0 * (nfilled - 1 - band)
            eig[:, band] = top - 0.5 * k
        else:
            bottom = cbm + 2.0 * (band - nfilled)
            eig[:, band] = bottom + 0.5 * k
    occ = np.zeros((nk, nbands))
    occ[:, :nfilled] = 1.0
    return eig, occ


def metallic_channel(nk: int = 16, *, nbands: int = 4):
    """One band disperses from -1 to +1 eV through the Fermi level at 0."""

    eig, _ = gapped_channel(nk, nfilled=1, nbands=nbands)
    eig[:, 1] = np.linspace(-1.0, 1.0, nk)
    eig[:, 2:] = np.sort(eig[:, 2:], axis=1) + 1.5
    occ = np.zeros_like(eig)
    occ[:, 0] = 1.0
    occ[:, 1] = (eig[:, 1] < 0.0).astype(float)
    return eig, occ


def integer_metal_channel(nk: int = 16, *, nbands: int = 4):
    """Integer-filled metallic channel: two bands overlap across the Fermi level
    and together hold exactly one band of electrons (how a metal with an even
    number of electrons per channel appears in band-index terms)."""

    eig, _ = gapped_channel(nk, nbands=nbands)
    eig[:, 1] = np.concatenate([np.linspace(-0.6, -0.1, 10), np.linspace(0.1, 0.3, 6)])
    eig[:, 2] = np.concatenate([np.linspace(-0.3, -0.1, 6), np.linspace(0.1, 0.6, 10)])
    occ = (eig < 0.0).astype(float)
    return eig, occ


def gaussian_occupations(eig, efermi: float, sigma: float):
    from math import erfc

    return np.vectorize(lambda e: 0.5 * erfc((e - efermi) / sigma))(eig)


COMPLETE = StageCompletion(1, "stage 1", True, True, StageConvergence(STAGE_CONVERGED))


def assess(
    vasprun: VasprunElectronicEvidence,
    *,
    completion: StageCompletion | None = COMPLETE,
    stage_type: str = "static",
):
    return assess_electronic_character(
        ElectronicCharacterInput(
            subject_id="fixture",
            calculation_type="BMD Compute",
            scope="bmd_compute_workflow",
            stage=SelectedStage(1, "stage 1", stage_type, "pbe"),
            stage_completion=completion,
            evidence=vasprun,
            producer_provenance=ProducerProvenance("unavailable"),
            assessor_version="test",
        )
    )


def codes(result) -> list[str]:
    return [finding.reason_code for finding in result.findings]


# --- core classification -------------------------------------------------------------------


def test_gapped_non_spin_polarized_channel_is_nonmetallic() -> None:
    eig, occ = gapped_channel()
    result = assess(evidence((eig,), (occ,), efermi=0.0))

    assert result.classification == "nonmetallic"
    assert result.assessment_status == "assessed"
    assert result.evidence.band_gap_ev == 1.5
    assert result.evidence.spin_treatment == "non_spin_polarized"
    assert [item.channel for item in result.evidence.channels] == ["total"]
    assert result.evidence.half_metallic is None


def test_partially_filled_dispersive_band_is_metallic() -> None:
    eig, occ = metallic_channel()
    result = assess(evidence((eig,), (occ,), efermi=0.0))

    assert result.classification == "metallic"
    assert "PARTIALLY_FILLED_BAND_CROSSES_FERMI" in codes(result)
    assert result.evidence.band_gap_ev is None


def test_integer_filled_band_overlap_is_metallic_semimetal() -> None:
    eig, _ = gapped_channel(nk=16)
    # Valence band rises above, conduction band dips below the Fermi level at
    # different k-points; together they hold exactly one band of electrons.
    eig[:, 1] = np.concatenate([np.linspace(-0.6, -0.1, 10), np.linspace(0.1, 0.3, 6)])
    eig[:, 2] = np.concatenate([np.linspace(-0.3, -0.1, 6), np.linspace(0.1, 0.6, 10)])
    occ = (eig < 0.0).astype(float)
    result = assess(evidence((eig,), (occ,), efermi=0.0, ismear=0, sigma=0.01))

    assert result.evidence.channels[0].filling == "integer"
    assert result.classification == "metallic"
    assert "BAND_OVERLAP_AT_FERMI" in codes(result)


def test_half_metal_is_metallic_and_flagged() -> None:
    # Integer total and integer moment: the metallic channel is integer-filled
    # with bands overlapping across the Fermi level.
    up, up_occ = integer_metal_channel()
    down, down_occ = gapped_channel(nfilled=1)
    result = assess(evidence((up, down), (up_occ, down_occ), efermi=0.0))

    assert result.classification == "metallic"
    assert result.evidence.half_metallic is True
    assert "HALF_METALLIC_STATE_ESTABLISHED" in codes(result)
    assert [item.state for item in result.evidence.channels] == ["metallic", "gapped"]


def test_spin_polarized_metal_with_both_channels_metallic_is_not_half_metallic() -> None:
    # bcc-Fe-like evidence: exchange-split channels, both crossing the Fermi level.
    up, up_occ = metallic_channel()
    down, down_occ = metallic_channel()
    up[:, 1] -= 0.4
    up_occ[:, 1] = (up[:, 1] < 0.0).astype(float)
    down[:, 1] += 0.4
    down_occ[:, 1] = (down[:, 1] < 0.0).astype(float)
    result = assess(evidence((up, down), (up_occ, down_occ), efermi=0.0))
    assert result.evidence.nelect == 3.0  # fractional channels, integer total

    assert result.classification == "metallic"
    assert result.evidence.half_metallic is False
    channels = result.evidence.channels
    assert channels[0].electron_count != channels[1].electron_count  # net moment


def test_ferromagnetic_insulator_with_different_channel_fillings_is_nonmetallic() -> None:
    up, up_occ = gapped_channel(nfilled=3, nbands=5, vbm=-0.6, cbm=1.4)
    down, down_occ = gapped_channel(nfilled=1, nbands=5, vbm=-0.8, cbm=0.9)
    result = assess(evidence((up, down), (up_occ, down_occ), efermi=0.0))

    assert result.classification == "nonmetallic"
    assert result.evidence.half_metallic is False
    # Fundamental gap is across channels: min CBM (down 0.9) minus max VBM (up -0.6).
    assert result.evidence.band_gap_ev == 1.5
    assert [item.electron_count for item in result.evidence.channels] == [3.0, 1.0]


def test_spin_channels_are_reported_up_then_down() -> None:
    up, up_occ = gapped_channel()
    result = assess(evidence((up, up.copy()), (up_occ, up_occ.copy()), efermi=0.0))

    assert result.evidence.spin_treatment == "collinear_spin_polarized"
    assert [item.channel for item in result.evidence.channels] == ["up", "down"]


# --- SOC / noncollinear --------------------------------------------------------------------


def test_noncollinear_uses_one_spinor_channel_without_spin_up_down() -> None:
    eig, occ = gapped_channel(nfilled=4, nbands=6)
    result = assess(evidence((eig,), (occ,), efermi=0.0, lsorbit=True))

    assert result.classification == "nonmetallic"
    assert result.evidence.spin_treatment == "noncollinear"
    assert [item.channel for item in result.evidence.channels] == ["spinor"]
    assert result.evidence.half_metallic is None
    payload = serialize_electronic_character_assessment(result)
    assert {item["channel"] for item in payload["electronic_evidence"]["spin_channels"]} == {"spinor"}
    # One electron per spinor state: NELECT equals the channel count, not twice it.
    assert payload["electronic_evidence"]["spin_channels"][0]["electron_count"] == 4.0


def test_noncollinear_metal_is_metallic() -> None:
    eig, occ = integer_metal_channel()
    result = assess(evidence((eig,), (occ,), efermi=0.0, lsorbit=True))

    assert result.classification == "metallic"
    assert result.evidence.half_metallic is None


def test_noncollinear_evidence_with_two_spin_channels_is_inconsistent() -> None:
    eig, occ = gapped_channel()
    vasprun = evidence((eig, eig), (occ, occ), efermi=0.0)
    result = assess(replace(vasprun, lsorbit=True, lnoncollinear=True))

    assert result.classification == "uncertain"
    assert "SPIN_EVIDENCE_INCONSISTENT" in codes(result)


def test_soc_small_gap_is_uncertain() -> None:
    eig, occ = gapped_channel(vbm=-0.03, cbm=0.03, nfilled=4, nbands=6)
    result = assess(evidence((eig,), (occ,), efermi=0.0, lsorbit=True))

    assert result.classification == "uncertain"
    assert "SMALL_GAP" in codes(result)


# --- smearing ------------------------------------------------------------------------------


def test_smearing_tails_in_a_wide_gap_are_not_metallic() -> None:
    eig, _ = gapped_channel(vbm=-0.25, cbm=0.25)
    occ = 1.0 / (1.0 + np.exp(eig / 0.05))  # Fermi-Dirac, SIGMA = 0.05 eV
    assert np.any((occ > 1e-3) & (occ < 1 - 1e-3))  # genuinely fractional occupations
    result = assess(evidence((eig,), (occ,), efermi=0.0, ismear=-1, sigma=0.05))

    assert result.classification == "nonmetallic"


def test_fermi_level_pinned_at_valence_edge_with_mp_overshoot_is_nonmetallic() -> None:
    eig, occ = gapped_channel(vbm=0.0, cbm=5.0)
    occ[:, 1] = np.where(np.isclose(eig[:, 1], 0.0), 1.02, 1.0)  # Methfessel-Paxton overshoot
    result = assess(evidence((eig,), (occ,), efermi=0.0, ismear=1, sigma=0.2, nelect=4.0))

    assert result.classification == "nonmetallic"


def test_smearing_comparable_to_gap_is_uncertain_not_metallic() -> None:
    eig, _ = gapped_channel(vbm=-0.15, cbm=0.15)
    occ = gaussian_occupations(eig, efermi=0.0, sigma=0.1)
    result = assess(evidence((eig,), (occ,), efermi=0.0, ismear=0, sigma=0.1))

    assert result.classification == "uncertain"
    assert {"SMEARING_SPILLS_ACROSS_GAP", "GAP_COMPARABLE_TO_SMEARING"} & set(codes(result))


def test_gap_smaller_than_smearing_ratio_is_uncertain() -> None:
    eig, occ = gapped_channel(vbm=-0.2, cbm=0.2)
    result = assess(evidence((eig,), (occ,), efermi=0.0, ismear=0, sigma=0.2))

    assert result.classification == "uncertain"
    assert "GAP_COMPARABLE_TO_SMEARING" in codes(result)


@pytest.mark.parametrize("gap", (0.0, 0.05, 0.099))
def test_small_positive_gap_never_establishes_nonmetallicity(gap: float) -> None:
    eig, occ = gapped_channel(vbm=-gap / 2, cbm=gap / 2)
    result = assess(evidence((eig,), (occ,), efermi=0.0))

    assert result.classification == "uncertain"


def test_fractional_filling_without_dispersive_crossing_is_uncertain() -> None:
    eig, occ = gapped_channel()
    eig[:, 1] = 0.0  # flat band pinned at the Fermi level
    occ[:, 1] = 0.5
    result = assess(evidence((eig,), (occ,), efermi=0.0, ismear=0, sigma=0.05))

    assert result.classification == "uncertain"
    assert "FRACTIONAL_FILLING_WITHOUT_FERMI_CROSSING" in codes(result)


# --- uncertainty and gates -----------------------------------------------------------------


def test_single_kpoint_is_uncertain_even_when_gapped() -> None:
    eig, occ = gapped_channel(nk=1)
    result = assess(evidence((eig,), (occ,), efermi=0.0, divisions=(1, 1, 1), kpoints=[[0, 0, 0]]))

    assert result.classification == "uncertain"
    assert "SINGLE_KPOINT_SAMPLING" in codes(result)


def test_sparse_mesh_is_uncertain() -> None:
    full_2x2x2 = np.array([[i / 2, j / 2, k / 2] for i in range(2) for j in range(2) for k in range(2)])
    eig, occ = metallic_channel(nk=8)
    result = assess(evidence((eig,), (occ,), efermi=0.0, divisions=(2, 2, 2), kpoints=full_2x2x2))

    assert result.classification == "uncertain"
    assert "SPARSE_KPOINT_SAMPLING" in codes(result)
    assert result.evidence.mesh_status == "verified"
    assert result.evidence.max_reciprocal_spacing_inv_angstrom == 1.0


def test_explicit_kpoint_list_never_establishes_a_mesh() -> None:
    eig, occ = gapped_channel()
    vasprun = evidence((eig,), (occ,), efermi=0.0, divisions=None)
    result = assess(replace(vasprun, kpoints_style="Reciprocal"))

    assert result.classification == "uncertain"
    assert "KPOINT_MESH_NOT_ESTABLISHED" in codes(result)
    assert result.evidence.mesh_status == "not_generated"
    assert result.evidence.max_reciprocal_spacing_inv_angstrom is None


def test_zero_weight_points_inside_the_gap_make_the_result_uncertain() -> None:
    eig, occ = gapped_channel(nk=16)
    eig = np.vstack([eig, [[-2.5, 0.8, 1.2, 3.0]]])  # path point with valence state at 0.8 eV
    occ = np.vstack([occ, [[0.0, 0.0, 0.0, 0.0]]])
    weights = np.append(np.full(16, 1 / 16), 0.0)
    kpoints = np.vstack([mesh_kpoints(), [[0.25, 0.25, 0.25]]])
    result = assess(evidence((eig,), (occ,), efermi=0.0, weights=weights, kpoints=kpoints))

    assert result.classification == "uncertain"
    assert "FERMI_LEVEL_OUTSIDE_GAP" in codes(result)
    assert result.evidence.zero_weight_kpoints == 1
    channel = result.evidence.channels[0]
    assert channel.vbm is not None and channel.vbm.kpoint_weight == "zero_weight"
    assert channel.vbm.energy_ev == 0.8


def test_benign_zero_weight_points_are_only_a_cross_check() -> None:
    eig, occ = gapped_channel(nk=16)
    eig = np.vstack([eig, eig[:1]])
    occ = np.vstack([occ, np.zeros((1, 4))])
    weights = np.append(np.full(16, 1 / 16), 0.0)
    kpoints = np.vstack([mesh_kpoints(), [[0.0, 0.0, 0.0]]])
    result = assess(evidence((eig,), (occ,), efermi=0.0, weights=weights, kpoints=kpoints))

    assert result.classification == "nonmetallic"
    assert result.evidence.band_edge_evidence == "weighted_mesh_and_zero_weight_kpoints"
    assert any("zero-weight k-points" in item for item in result.limitations)


@pytest.mark.parametrize(
    "completion",
    (
        None,
        StageCompletion(1, "s", False, None, StageConvergence(STAGE_CONVERGENCE_UNDETERMINED)),
        StageCompletion(1, "s", True, False, StageConvergence(STAGE_CONVERGED)),
        StageCompletion(1, "s", True, True, StageConvergence(STAGE_CONVERGENCE_UNDETERMINED, undetermined_scopes=("electronic",))),
        StageCompletion(1, "s", True, True, StageConvergence("not_converged", not_converged_scopes=("electronic",))),
    ),
)
def test_unconverged_or_unfinished_stage_is_uncertain(completion) -> None:
    eig, occ = metallic_channel()
    result = assess(evidence((eig,), (occ,), efermi=0.0), completion=completion)

    assert result.classification == "uncertain"


def test_vasprun_reported_electronic_nonconvergence_is_uncertain() -> None:
    eig, occ = gapped_channel()
    result = assess(evidence((eig,), (occ,), efermi=0.0, converged_electronic=False))

    assert result.classification == "uncertain"
    assert "ELECTRONIC_CONVERGENCE_NOT_REACHED" in codes(result)


@pytest.mark.parametrize(
    ("status", "code"),
    (
        ("missing", "VASPRUN_UNAVAILABLE"),
        ("incomplete", "VASPRUN_INCOMPLETE"),
        ("unreadable", "VASPRUN_UNREADABLE"),
        ("too_large", "VASPRUN_EXCEEDS_PARSING_BOUND"),
        ("ambiguous", "STAGE_EVIDENCE_AMBIGUOUS"),
    ),
)
def test_unavailable_evidence_is_uncertain(status: str, code: str) -> None:
    result = assess(VasprunElectronicEvidence(status, "reason"))

    assert result.classification == "uncertain"
    assert result.assessment_status == "insufficient_evidence"
    assert code in codes(result)


def test_missing_fermi_level_is_uncertain() -> None:
    eig, occ = gapped_channel()
    result = assess(evidence((eig,), (occ,), efermi=None))

    assert "FERMI_LEVEL_UNAVAILABLE" in codes(result)
    assert result.classification == "uncertain"


def test_occupations_contradicting_eigenvalues_are_uncertain() -> None:
    eig, occ = gapped_channel()
    occ[0, 0] = 0.0  # deep valence state left empty
    occ[0, 2] = 1.0  # conduction state occupied instead (count preserved)
    result = assess(evidence((eig,), (occ,), efermi=0.0))

    assert result.classification == "uncertain"
    assert "OCCUPATIONS_CONTRADICT_EIGENVALUES" in codes(result)


def test_electron_count_inconsistent_with_nelect_is_uncertain() -> None:
    eig, occ = gapped_channel()
    result = assess(evidence((eig,), (occ,), efermi=0.0, nelect=6.0))

    assert "ELECTRON_COUNT_INCONSISTENT" in codes(result)


def test_insufficient_bands_above_fermi_is_uncertain() -> None:
    eig, occ = gapped_channel(nfilled=4, nbands=4)
    result = assess(evidence((eig,), (occ,), efermi=0.0))

    assert result.classification == "uncertain"
    assert "INSUFFICIENT_BANDS_ABOVE_FERMI" in codes(result)


def test_occupations_out_of_range_are_uncertain() -> None:
    eig, occ = gapped_channel()
    occ[0, 0] = 2.0
    result = assess(evidence((eig,), (occ,), efermi=0.0, nelect=None))

    assert "OCCUPATIONS_OUT_OF_RANGE" in codes(result)


def test_inconsistent_array_shapes_are_uncertain() -> None:
    eig, occ = gapped_channel()
    vasprun = evidence((eig,), (occ,), efermi=0.0)
    result = assess(replace(vasprun, weights=np.full(8, 1 / 8)))

    assert "INVALID_NUMERICAL_EVIDENCE" in codes(result)


@pytest.mark.parametrize(
    ("changes", "code"),
    (
        ({"ismear": -2}, "OCCUPATION_METHOD_UNSUPPORTED"),
        ({"ismear": -3}, "OCCUPATION_METHOD_UNSUPPORTED"),
        ({"ismear": None}, "OCCUPATION_METHOD_UNKNOWN"),
        ({"algo": "G0W0"}, "ELECTRONIC_METHOD_UNSUPPORTED"),
        ({"icharg": 11, "kpoints_style": "Reciprocal"}, "NON_SELF_CONSISTENT_PATH_UNSUPPORTED"),
        ({"kpoints_style": "Line_mode"}, "BAND_PATH_SAMPLING_UNSUPPORTED"),
    ),
)
def test_unsupported_electronic_evidence(changes, code: str) -> None:
    eig, occ = metallic_channel()
    result = assess(replace(evidence((eig,), (occ,), efermi=0.0), **changes))

    assert result.classification == "uncertain"
    assert result.assessment_status == "unsupported"
    assert code in codes(result)


def test_phonon_stage_is_unsupported() -> None:
    eig, occ = metallic_channel()
    result = assess(evidence((eig,), (occ,), efermi=0.0), stage_type="phonon")

    assert result.classification == "uncertain"
    assert result.assessment_status == "unsupported"


def test_non_self_consistent_run_on_a_generated_uniform_mesh_is_assessed() -> None:
    eig, occ = gapped_channel()
    result = assess(evidence((eig,), (occ,), efermi=0.0, icharg=11))

    assert result.classification == "nonmetallic"


# --- contract ------------------------------------------------------------------------------


def test_classification_is_deterministic_and_json_stable() -> None:
    eig, occ = metallic_channel()
    first = serialize_electronic_character_assessment(assess(evidence((eig,), (occ,), efermi=0.0)))
    second = serialize_electronic_character_assessment(assess(evidence((eig,), (occ,), efermi=0.0)))

    assert first == second
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(second, sort_keys=True, allow_nan=False)


def test_json_contract_is_versioned_bounded_and_array_free() -> None:
    eig, occ = gapped_channel(nk=16)
    payload = serialize_electronic_character_assessment(assess(evidence((eig,), (occ,), efermi=0.0)))

    assert payload["schema_id"] == "bmd_check.electronic_character_assessment"
    assert payload["schema_version"] == 1
    assert payload["assessment_id"] == "electronic_character"
    assert payload["method_version"] == "1.0.0-c2"
    assert payload["classification"] in {"metallic", "nonmetallic", "uncertain"}
    assert set(payload) == {
        "schema_id", "schema_version", "assessment_id", "method_version", "assessment_status",
        "classification", "subject", "electronic_evidence", "thresholds", "findings",
        "evidence_references", "producer_provenance", "assessor_provenance", "limitations",
    }
    assert payload["thresholds"] == dict(sorted(ec.THRESHOLDS.items()))

    def lists(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from lists(item)
        elif isinstance(value, list):
            yield value
            for item in value:
                yield from lists(item)

    # No list scales with the number of k-points or bands.
    assert max(len(item) for item in lists(payload)) <= 8
    json.dumps(payload, allow_nan=False)


def test_human_and_json_render_the_same_result() -> None:
    eig, occ = gapped_channel()
    result = assess(evidence((eig,), (occ,), efermi=0.0))
    payload = serialize_electronic_character_assessment(result)
    text = render_electronic_character_assessment(result)

    assert f"Classification: {payload['classification'].upper()}" in text
    assert f"Band gap: {payload['electronic_evidence']['band_gap_ev']} eV" in text
    for finding in payload["findings"]:
        assert finding["reason_code"] in text


def test_contract_never_selects_vasp_settings() -> None:
    eig, occ = metallic_channel()
    text = json.dumps(serialize_electronic_character_assessment(assess(evidence((eig,), (occ,), efermi=0.0))))

    assert "recommended" not in text.lower()
    assert "select" not in json.dumps(
        serialize_electronic_character_assessment(assess(evidence((eig,), (occ,), efermi=0.0)))["electronic_evidence"]
    )


# --- recorded VASP evidence ----------------------------------------------------------------


def write_direct_vasp(directory: Path, fixture: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lifecycle_fixtures.write_inputs(directory)
    (directory / "OUTCAR").write_text(lifecycle_fixtures.NORMAL_OUTCAR, encoding="utf-8")
    (directory / "OSZICAR").write_text(lifecycle_fixtures.OSZICAR_STATIC_CONVERGED, encoding="utf-8")
    (directory / "vasprun.xml").write_bytes(gzip.decompress((FIXTURES / fixture).read_bytes()))
    return directory


RECORDED = (
    ("Al_mp_smearing.vasprun.xml.gz", "metallic", "METALLIC_STATE_ESTABLISHED"),
    ("Si_extra_electron.vasprun.xml.gz", "metallic", "METALLIC_STATE_ESTABLISHED"),
    ("LiF_tetrahedron.vasprun.xml.gz", "nonmetallic", "NONMETALLIC_STATE_ESTABLISHED"),
    ("Si_static.vasprun.xml.gz", "nonmetallic", "NONMETALLIC_STATE_ESTABLISHED"),
    ("Xe_automatic_mesh.vasprun.xml.gz", "nonmetallic", "NONMETALLIC_STATE_ESTABLISHED"),
    ("Cu_gamma_only.vasprun.xml.gz", "uncertain", "SINGLE_KPOINT_SAMPLING"),
    ("Li_sparse_tetrahedron.vasprun.xml.gz", "uncertain", "SPARSE_KPOINT_SAMPLING"),
    ("Si_unconverged.vasprun.xml.gz", "uncertain", "ELECTRONIC_CONVERGENCE_NOT_REACHED"),
)


@pytest.mark.parametrize(("fixture", "classification", "code"), RECORDED)
def test_recorded_vasp_evidence(tmp_path: Path, fixture: str, classification: str, code: str) -> None:
    vasprun = read_vasprun_electronic_evidence(_decompressed(tmp_path, fixture))
    result = assess(vasprun)

    assert result.classification == classification
    assert code in codes(result)


def test_recorded_al_is_metallic_in_both_spin_channels(tmp_path: Path) -> None:
    result = assess(read_vasprun_electronic_evidence(_decompressed(tmp_path, "Al_mp_smearing.vasprun.xml.gz")))

    assert result.evidence.spin_treatment == "collinear_spin_polarized"
    assert [item.state for item in result.evidence.channels] == ["metallic", "metallic"]
    assert result.evidence.smearing_method == "methfessel_paxton"
    assert result.evidence.mesh_divisions == (13, 13, 13)
    assert result.evidence.mesh_status == "verified"


def test_recorded_lif_gap_with_fermi_level_at_valence_edge(tmp_path: Path) -> None:
    result = assess(read_vasprun_electronic_evidence(_decompressed(tmp_path, "LiF_tetrahedron.vasprun.xml.gz")))

    assert result.classification == "nonmetallic"
    assert result.evidence.band_gap_ev == pytest.approx(8.726, abs=1e-3)
    assert result.evidence.smearing_method == "tetrahedron"


def test_recorded_gamma_only_cu_is_not_given_a_spurious_gap(tmp_path: Path) -> None:
    # Occupation-threshold gap finders report several eV for this Gamma-only metal.
    result = assess(read_vasprun_electronic_evidence(_decompressed(tmp_path, "Cu_gamma_only.vasprun.xml.gz")))

    assert result.classification == "uncertain"
    assert result.evidence.band_gap_ev is None


def test_recorded_automatic_mesh_is_verified_from_vasp_divisions(tmp_path: Path) -> None:
    # KPOINTS "Auto" mode: VASP records the Gamma-centred divisions it generated.
    result = assess(read_vasprun_electronic_evidence(_decompressed(tmp_path, "Xe_automatic_mesh.vasprun.xml.gz")))

    assert result.evidence.kpoints_style == "automatic"
    assert result.evidence.mesh_divisions == (7, 7, 7)
    assert result.evidence.mesh_status == "verified"


def test_truncated_recorded_vasprun_is_incomplete(tmp_path: Path) -> None:
    path = _decompressed(tmp_path, "Si_static.vasprun.xml.gz")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])

    vasprun = read_vasprun_electronic_evidence(path)
    result = assess(vasprun)

    assert vasprun.status in {"incomplete", "unreadable"}
    assert result.classification == "uncertain"


def test_oversized_vasprun_is_not_parsed(tmp_path: Path) -> None:
    path = _decompressed(tmp_path, "Si_static.vasprun.xml.gz")

    assert read_vasprun_electronic_evidence(path, max_bytes=10).status == "too_large"


def _decompressed(tmp_path: Path, fixture: str) -> Path:
    path = tmp_path / "vasprun.xml"
    path.write_bytes(gzip.decompress((FIXTURES / fixture).read_bytes()))
    return path


# --- local workflow adapter and stage selection --------------------------------------------


def three_stage_workflow(root: Path, *, result_dir: Path | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stages = {label: str(root / label) for label in ("stage_01", "stage_02", "stage_03")}
    lifecycle_fixtures.write_three_stage_submission(
        root,
        producer_root=str(root),
        stage_paths=stages,
        result_dir=str(result_dir) if result_dir else None,
    )
    fixtures = {
        "stage_01": "Al_mp_smearing.vasprun.xml.gz",
        "stage_02": "LiF_tetrahedron.vasprun.xml.gz",
        "stage_03": "Si_static.vasprun.xml.gz",
    }
    for label, fixture in fixtures.items():
        directory = root / label
        lifecycle_fixtures.write_stage_complete(directory)
        (directory / "vasprun.xml").write_bytes(gzip.decompress((FIXTURES / fixture).read_bytes()))
    return root


def test_multistage_workflow_requires_explicit_stage(tmp_path: Path) -> None:
    root = three_stage_workflow(tmp_path / "flow")
    analysis = analyze_calculation_directory(root)

    with pytest.raises(StageSelectionError, match="--stage-index"):
        electronic_character_input_from_lifecycle(analysis)
    with pytest.raises(StageSelectionError, match="not declared"):
        electronic_character_input_from_lifecycle(analysis, stage_index=4)


def test_explicit_stage_selection_reads_that_stage_only(tmp_path: Path) -> None:
    root = three_stage_workflow(tmp_path / "flow")
    analysis = analyze_calculation_directory(root)
    seen: list[Path] = []

    def reader(path: Path):
        seen.append(path)
        return read_vasprun_electronic_evidence(path)

    data = electronic_character_input_from_lifecycle(analysis, stage_index=2, reader=reader)

    assert seen == [root / "stage_02" / "vasprun.xml"]
    assert data.stage is not None and data.stage.index == 2 and data.stage.stage_type == "static"
    assert data.stage_completion is not None and data.stage_completion.index == 2


def test_stage_with_several_bound_vasprun_files_is_ambiguous(tmp_path: Path) -> None:
    results = tmp_path / "flow" / "results"
    root = three_stage_workflow(tmp_path / "flow", result_dir=results)
    results.mkdir()
    (results / "vasprun.xml").write_bytes((root / "stage_03" / "vasprun.xml").read_bytes())
    analysis = analyze_calculation_directory(root)

    result = assess_electronic_character(electronic_character_input_from_lifecycle(analysis, stage_index=3))

    assert result.classification == "uncertain"
    assert "STAGE_EVIDENCE_AMBIGUOUS" in codes(result)


def test_direct_calculation_has_only_stage_one(tmp_path: Path) -> None:
    analysis = analyze_calculation_directory(write_direct_vasp(tmp_path / "calc", "Al_mp_smearing.vasprun.xml.gz"))

    with pytest.raises(StageSelectionError):
        electronic_character_input_from_lifecycle(analysis, stage_index=2)
    assert electronic_character_input_from_lifecycle(analysis, stage_index=1).stage.index == 1


# --- single stage directory (a producer's copy of one completed stage) ----------------------


def write_stage_copy(directory: Path, fixture: str, *, oszicar: bool = True) -> Path:
    """Exactly the files a later atomate2 stage holds from the previous one."""

    directory.mkdir(parents=True, exist_ok=True)
    lifecycle_fixtures.write_inputs(directory)
    (directory / "CONTCAR").write_bytes((directory / "POSCAR").read_bytes())
    for name in ("POSCAR", "KPOINTS", "POTCAR"):
        (directory / name).unlink(missing_ok=True)
    (directory / "OUTCAR").write_text(lifecycle_fixtures.NORMAL_OUTCAR, encoding="utf-8")
    if oszicar:
        (directory / "OSZICAR").write_text(lifecycle_fixtures.OSZICAR_STATIC_CONVERGED, encoding="utf-8")
    (directory / "vasprun.xml").write_bytes(gzip.decompress((FIXTURES / fixture).read_bytes()))
    return directory


def assess_copy(directory: Path, **kwargs):
    return ec.assess_stage_directory(directory, stage_index=2, stage_type="static", theory="hse06", **kwargs)


@pytest.mark.parametrize(("fixture", "classification", "code"), RECORDED)
def test_stage_directory_reuses_the_unchanged_c1_and_c2_verdict(
    tmp_path: Path, fixture: str, classification: str, code: str
) -> None:
    directory = write_stage_copy(tmp_path / "copy", fixture)
    result = assess_copy(directory)
    # The same files analysed as a direct calculation give the same verdict.
    reference = assess_electronic_character(
        electronic_character_input_from_lifecycle(analyze_calculation_directory(directory), stage_index=1)
    )

    assert result.classification == reference.classification == classification
    assert code in codes(result)
    assert codes(result) == codes(reference)
    assert result.evidence == reference.evidence
    assert result.method_version == ec.METHOD_VERSION
    assert result.thresholds == ec.THRESHOLDS


def test_stage_directory_records_the_declared_stage(tmp_path: Path) -> None:
    result = assess_copy(write_stage_copy(tmp_path / "copy", "Al_mp_smearing.vasprun.xml.gz"), subject_id="run-1")
    payload = serialize_electronic_character_assessment(result)

    assert result.scope == ec.SCOPE_STAGE_DIRECTORY == "stage_directory"
    assert (result.stage.index, result.stage.stage_type, result.stage.theory) == (2, "static", "hse06")
    assert result.subject_id == "run-1"
    assert result.evidence_references[0].stage_index == 2
    assert payload["classification"] == "metallic"
    assert str(tmp_path) not in json.dumps(payload)


def test_stage_directory_without_oszicar_cannot_establish_convergence(tmp_path: Path) -> None:
    # C1 needs the OSZICAR trajectory; the producer must supply it.
    result = assess_copy(write_stage_copy(tmp_path / "copy", "Al_mp_smearing.vasprun.xml.gz", oszicar=False))

    assert result.classification == "uncertain"
    assert "STAGE_CONVERGENCE_NOT_ESTABLISHED" in codes(result)


@pytest.mark.parametrize("damage", ["missing_directory", "empty_directory", "no_vasprun", "truncated", "empty_vasprun"])
def test_stage_directory_with_unusable_evidence_is_uncertain(tmp_path: Path, damage: str) -> None:
    directory = tmp_path / "copy"
    if damage != "missing_directory":
        directory.mkdir()
    if damage in ("no_vasprun", "truncated", "empty_vasprun"):
        write_stage_copy(directory, "Si_static.vasprun.xml.gz")
        vasprun = directory / "vasprun.xml"
        data = vasprun.read_bytes()
        if damage == "no_vasprun":
            vasprun.unlink()
        elif damage == "truncated":
            vasprun.write_bytes(data[: len(data) // 2])
        else:
            vasprun.write_bytes(b"")

    result = assess_copy(directory)

    assert result.classification == "uncertain"
    assert result.assessment_status in {"insufficient_evidence", "unsupported"}


def test_stage_directory_ignores_workflow_records_in_parent_directories(tmp_path: Path) -> None:
    # Inside a BMD Compute run, stage_03 holds a copy of stage_02; only that
    # directory's files may be assessed, never the declared workflow stage 3.
    root = three_stage_workflow(tmp_path / "flow")
    copy = write_stage_copy(root / "stage_03", "Al_mp_smearing.vasprun.xml.gz")

    result = assess_copy(copy)

    assert result.scope == "stage_directory"
    assert result.classification == "metallic"
    assert result.stage.index == 2


def test_stage_directory_is_read_only(tmp_path: Path) -> None:
    directory = write_stage_copy(tmp_path / "copy", "LiF_tetrahedron.vasprun.xml.gz")
    before = {path.name: (path.stat().st_mtime_ns, path.stat().st_size) for path in directory.iterdir()}

    assess_copy(directory)

    assert {path.name: (path.stat().st_mtime_ns, path.stat().st_size) for path in directory.iterdir()} == before


@pytest.mark.parametrize("stage_index", [0, -1, True, "2", 2.0])
def test_stage_directory_requires_a_valid_stage_index(tmp_path: Path, stage_index) -> None:
    with pytest.raises(ValueError):
        ec.assess_stage_directory(tmp_path, stage_index=stage_index, stage_type="static")


# --- CLI -----------------------------------------------------------------------------------


def run_cli(argv: list[str], capsys) -> tuple[int, str]:
    code = cli.main(argv, command_name="bmd-check")
    return code, capsys.readouterr().out


def test_cli_json_is_the_internal_serialization(tmp_path: Path, capsys) -> None:
    root = three_stage_workflow(tmp_path / "flow")
    code, output = run_cli(
        ["assess", "electronic-character", str(root), "--stage-index", "1", "--format", "json"],
        capsys,
    )
    expected = serialize_electronic_character_assessment(
        assess_electronic_character(
            electronic_character_input_from_lifecycle(analyze_calculation_directory(root), stage_index=1)
        )
    )

    assert code == 0
    assert json.loads(output) == expected
    assert expected["classification"] == "metallic"
    assert expected["subject"]["stage"]["index"] == 1


def test_cli_option_order_does_not_matter(tmp_path: Path, capsys) -> None:
    root = three_stage_workflow(tmp_path / "flow")
    _, first = run_cli(["assess", "electronic-character", str(root), "--format", "json", "--stage-index", "2"], capsys)
    _, second = run_cli(["assess", "electronic-character", str(root), "--stage-index", "2", "--format", "json"], capsys)

    assert first == second
    assert json.loads(first)["classification"] == "nonmetallic"


def test_cli_human_output_matches_json(tmp_path: Path, capsys) -> None:
    root = three_stage_workflow(tmp_path / "flow")
    _, human = run_cli(["assess", "electronic-character", str(root), "--stage-index", "3"], capsys)
    _, raw = run_cli(["assess", "electronic-character", str(root), "--stage-index", "3", "--format", "json"], capsys)
    payload = json.loads(raw)

    assert f"Classification: {payload['classification'].upper()}" in human
    assert "Electronic Character Assessment" in human


def test_cli_from_stage_directory_is_assessed_at_workflow_root(tmp_path: Path, capsys) -> None:
    root = three_stage_workflow(tmp_path / "flow")
    _, from_root = run_cli(["assess", "electronic-character", str(root), "--stage-index", "2", "--format", "json"], capsys)
    _, from_stage = run_cli(
        ["assess", "electronic-character", str(root / "stage_02"), "--stage-index", "2", "--format", "json"],
        capsys,
    )

    assert from_root == from_stage


def test_cli_rejects_ambiguous_stage_selection(tmp_path: Path, capsys) -> None:
    root = three_stage_workflow(tmp_path / "flow")
    code, output = run_cli(["assess", "electronic-character", str(root), "--format", "json"], capsys)

    assert code == 2
    assert "--stage-index" in output
    with pytest.raises(json.JSONDecodeError):
        json.loads(output)


@pytest.mark.parametrize(
    "extra",
    (
        ["--stage-index", "0"],
        ["--stage-index", "-1"],
        ["--stage-index", "two"],
        ["--stage-index", "1", "--stage-index", "2"],
        ["--format", "yaml"],
        ["--format"],
        ["--unknown", "1"],
    ),
)
def test_cli_rejects_malformed_options(tmp_path: Path, capsys, extra: list[str]) -> None:
    code, output = run_cli(["assess", "electronic-character", str(tmp_path), *extra], capsys)

    assert code == 2
    assert "Usage: bmd-check assess electronic-character" in output


def test_cli_direct_calculation_end_to_end(tmp_path: Path, capsys) -> None:
    directory = write_direct_vasp(tmp_path / "calc", "LiF_tetrahedron.vasprun.xml.gz")
    code, output = run_cli(["assess", "electronic-character", str(directory), "--format", "json"], capsys)
    payload = json.loads(output)

    assert code == 0
    assert payload["classification"] == "nonmetallic"
    assert payload["subject"]["scope"] == "direct_vasp_calculation"


def test_cli_output_contains_no_absolute_paths(tmp_path: Path, capsys) -> None:
    root = three_stage_workflow(tmp_path / "flow")
    _, output = run_cli(["assess", "electronic-character", str(root), "--stage-index", "2", "--format", "json"], capsys)

    assert str(tmp_path) not in output
    assert "POTCAR" not in output


def test_cli_is_read_only_and_runs_no_commands(tmp_path: Path, capsys, monkeypatch) -> None:
    root = three_stage_workflow(tmp_path / "flow")

    def snapshot() -> dict[str, tuple[int, int]]:
        return {
            str(path.relative_to(tmp_path)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(tmp_path.rglob("*"))
        }

    def refuse(*args, **kwargs):
        raise AssertionError("no subprocess may run during the assessment")

    before = snapshot()
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(os, "system", refuse)
    monkeypatch.setattr(cli, "load_resources", refuse)
    code, _ = run_cli(["assess", "electronic-character", str(root), "--stage-index", "2", "--format", "json"], capsys)

    assert code == 0
    assert snapshot() == before


def test_existing_completion_command_is_unchanged(tmp_path: Path, capsys, monkeypatch) -> None:
    from bmd_agent.config import ConfigurationError

    monkeypatch.setattr(cli, "load_resources", lambda: (_ for _ in ()).throw(ConfigurationError("none")))
    directory = write_direct_vasp(tmp_path / "calc", "LiF_tetrahedron.vasprun.xml.gz")
    code, output = run_cli(["assess", "completion", str(directory), "--format", "json"], capsys)

    assert code == 0
    payload = json.loads(output)
    assert payload["schema_id"] == "bmd_check.scientific_completion_assessment"
    code, output = run_cli(["assess", "completion", str(directory), "--stage-index", "1"], capsys)
    assert code == 2
    assert "Usage: bmd-check assess completion <RUN_ROOT> [--format json]" in output


# === Regression tests for the independent review of C2 =================================
#
# Each scenario below reproduced a false or invalid result in the first C2
# implementation (recorded in the review report) and must now be uncertain or
# internally consistent. Synthetic evidence is marked as such; recorded VASP
# evidence comes from tests/fixtures/electronic_character.


# --- 1. explicit k-point lists and mesh verification (synthetic) ---------------------------


def two_point_channel():
    eig, occ = gapped_channel(nk=2)
    return eig, occ


def test_two_point_explicit_list_is_not_inferred_as_a_dense_mesh() -> None:
    # Reproduction: Gamma and (1/8, 1/8, 1/8) were read as an 8x8x8 mesh -> nonmetallic.
    eig, occ = two_point_channel()
    vasprun = evidence(
        (eig,), (occ,), efermi=0.0, divisions=None,
        kpoints=[[0, 0, 0], [1 / 8, 1 / 8, 1 / 8]], weights=[0.5, 0.5],
        kpoints_style="Reciprocal",
    )
    result = assess(vasprun)

    assert result.classification == "uncertain"
    assert "KPOINT_MESH_NOT_ESTABLISHED" in codes(result)
    assert result.evidence.mesh_divisions is None


def test_declared_mesh_with_too_few_irreducible_points_is_inconsistent() -> None:
    # Two points cannot be the irreducible part of an 8x8x8 mesh: each would need a
    # star of 256, larger than any crystallographic point group (48).
    eig, occ = two_point_channel()
    result = assess(
        evidence((eig,), (occ,), efermi=0.0, divisions=(8, 8, 8),
                 kpoints=[[0, 0, 0], [1 / 8, 1 / 8, 1 / 8]], weights=[0.5, 0.5])
    )

    assert result.classification == "uncertain"
    assert "KPOINT_MESH_INCONSISTENT" in codes(result)
    assert result.evidence.mesh_status == "inconsistent"


def test_points_off_the_declared_mesh_are_inconsistent() -> None:
    eig, occ = gapped_channel()
    off_mesh = mesh_kpoints() + np.array([1 / 8, 0.0, 0.0])
    result = assess(evidence((eig,), (occ,), efermi=0.0, kpoints=off_mesh))

    assert "KPOINT_MESH_INCONSISTENT" in codes(result)


def test_weights_that_are_not_integer_multiplicities_are_inconsistent() -> None:
    eig, occ = gapped_channel()
    weights = np.full(16, 1 / 16)
    weights[0] += 0.01
    weights[1] -= 0.01  # still normalized, but 0.64 extra mesh points is not a star
    result = assess(evidence((eig,), (occ,), efermi=0.0, weights=weights))

    assert "KPOINT_MESH_INCONSISTENT" in codes(result)


def test_full_unreduced_regular_mesh_is_verified() -> None:
    full = np.array([[i / 4, j / 4, k / 4] for i in range(4) for j in range(4) for k in range(4)])
    eig, occ = gapped_channel(nk=64)
    result = assess(evidence((eig,), (occ,), efermi=0.0, kpoints=full))

    assert result.evidence.mesh_status == "verified"
    assert result.classification == "nonmetallic"


def test_symmetry_reduced_mesh_with_varying_multiplicities_is_verified() -> None:
    points = [[0, 0, 0], [1 / 4, 0, 0], [1 / 2, 0, 0], [1 / 4, 1 / 4, 0], [1 / 2, 1 / 4, 0], [1 / 4, 1 / 4, 1 / 4], [1 / 2, 1 / 2, 0]]
    multiplicities = np.array([1, 6, 3, 12, 24, 8, 10])
    assert multiplicities.sum() == 64
    eig, occ = gapped_channel(nk=7)
    result = assess(evidence((eig,), (occ,), efermi=0.0, kpoints=points, weights=multiplicities / 64))

    assert result.evidence.mesh_status == "verified"
    assert result.classification == "nonmetallic"


def test_shifted_monkhorst_mesh_uses_the_recorded_vasp_shift() -> None:
    shifted = np.array([[(2 * i + 1) / 8, (2 * j + 1) / 8, 1 / 8] for i in range(4) for j in range(4)])
    eig, occ = gapped_channel()
    recorded = evidence((eig,), (occ,), efermi=0.0, kpoints=shifted, kpoints_style="Monkhorst")
    verified = assess(replace(recorded, kpoints_shift=(0.5, 0.5, 0.5)))
    contradicted = assess(replace(recorded, kpoints_shift=(0.0, 0.0, 0.0)))
    convention = assess(recorded)  # shift not recorded: even Monkhorst-Pack convention

    assert verified.evidence.mesh_status == "verified"
    assert contradicted.evidence.mesh_status == "inconsistent"
    assert convention.evidence.mesh_status == "verified"


@pytest.mark.parametrize(
    ("fixture", "mesh"),
    (
        ("Al_mp_smearing.vasprun.xml.gz", (13, 13, 13)),
        ("LiF_tetrahedron.vasprun.xml.gz", (7, 7, 7)),
        ("Si_static.vasprun.xml.gz", (4, 4, 4)),  # shifted Monkhorst-Pack
        ("Li_sparse_tetrahedron.vasprun.xml.gz", (6, 6, 6)),  # "Monk" with recorded zero shift
        ("Xe_automatic_mesh.vasprun.xml.gz", (7, 7, 7)),  # "Auto"
    ),
)
def test_recorded_symmetry_reduced_meshes_are_verified(tmp_path: Path, fixture: str, mesh) -> None:
    vasprun = read_vasprun_electronic_evidence(_decompressed(tmp_path, fixture))
    result = assess(vasprun)

    assert vasprun.declared_divisions == mesh
    assert result.evidence.mesh_status == "verified"
    assert result.evidence.weighted_kpoints < int(np.prod(mesh))  # genuinely reduced


def test_recorded_explicit_list_without_generation_block_is_not_a_mesh(tmp_path: Path) -> None:
    path = _decompressed(tmp_path, "Si_static.vasprun.xml.gz")
    text = path.read_text(encoding="utf-8")
    start = text.index("<generation")
    end = text.index("</generation>") + len("</generation>")
    path.write_text(text[:start] + text[end:], encoding="utf-8")

    result = assess(read_vasprun_electronic_evidence(path))

    assert result.classification == "uncertain"
    assert "KPOINT_MESH_NOT_ESTABLISHED" in codes(result)


# --- 2. non-finite and invalid numerical evidence (synthetic + recorded) -------------------


def _with(base: VasprunElectronicEvidence, **changes):
    return replace(base, **changes)


def _array_with(value, index, replacement):
    copy = np.array(value, dtype=float)
    copy[index] = replacement
    return copy


NONFINITE_CASES = (
    ("eigenvalue NaN", lambda b: _with(b, eigenvalues=(_array_with(b.eigenvalues[0], (0, 0), np.nan),))),
    ("eigenvalue +inf", lambda b: _with(b, eigenvalues=(_array_with(b.eigenvalues[0], (0, 0), np.inf),))),
    ("eigenvalue -inf", lambda b: _with(b, eigenvalues=(_array_with(b.eigenvalues[0], (0, 0), -np.inf),))),
    ("occupation NaN", lambda b: _with(b, occupations=(_array_with(b.occupations[0], (0, 0), np.nan),))),
    ("occupation inf", lambda b: _with(b, occupations=(_array_with(b.occupations[0], (0, 0), np.inf),))),
    ("weight NaN", lambda b: _with(b, weights=_array_with(b.weights, 0, np.nan))),
    ("weight inf", lambda b: _with(b, weights=_array_with(b.weights, 0, np.inf))),
    ("weight negative", lambda b: _with(b, weights=_array_with(b.weights, 0, -0.1))),
    ("kpoint NaN", lambda b: _with(b, kpoints_frac=_array_with(b.kpoints_frac, (0, 0), np.nan))),
    ("kpoint inf", lambda b: _with(b, kpoints_frac=_array_with(b.kpoints_frac, (0, 0), np.inf))),
    ("Fermi NaN", lambda b: _with(b, efermi=float("nan"))),
    ("Fermi +inf", lambda b: _with(b, efermi=float("inf"))),
    ("Fermi -inf", lambda b: _with(b, efermi=float("-inf"))),
    ("Fermi string", lambda b: _with(b, efermi="0.0")),
    ("NELECT NaN", lambda b: _with(b, nelect=float("nan"))),
    ("NELECT inf", lambda b: _with(b, nelect=float("inf"))),
    ("NELECT negative", lambda b: _with(b, nelect=-4.0)),
    ("SIGMA NaN", lambda b: _with(b, ismear=0, sigma=float("nan"))),
    ("SIGMA inf", lambda b: _with(b, ismear=0, sigma=float("inf"))),
    ("SIGMA negative", lambda b: _with(b, ismear=0, sigma=-0.1)),
    ("reciprocal NaN", lambda b: _with(b, reciprocal_lengths=(float("nan"), 2.0, 2.0))),
    ("reciprocal inf", lambda b: _with(b, reciprocal_lengths=(float("inf"), 2.0, 2.0))),
    ("reciprocal zero", lambda b: _with(b, reciprocal_lengths=(0.0, 2.0, 2.0))),
    ("mesh division zero", lambda b: _with(b, declared_divisions=(0, 4, 4))),
    ("mesh division float", lambda b: _with(b, declared_divisions=(4.5, 4, 4))),
    ("mesh division bool", lambda b: _with(b, declared_divisions=(True, 4, 4))),
    ("mesh shift NaN", lambda b: _with(b, kpoints_shift=(float("nan"), 0.0, 0.0))),
    ("weights as strings", lambda b: _with(b, weights=np.array(["0.0625"] * 16))),
    ("eigenvalues as objects", lambda b: _with(b, eigenvalues=(np.array(b.eigenvalues[0], dtype=object),))),
    ("eigenvalues one-dimensional", lambda b: _with(b, eigenvalues=(np.zeros(16),))),
    ("ISMEAR as float", lambda b: _with(b, ismear=0.5)),
    ("NBANDS as string", lambda b: _with(b, nbands="4")),
)


@pytest.mark.parametrize(("label", "mutate"), NONFINITE_CASES, ids=[item[0] for item in NONFINITE_CASES])
def test_invalid_numerical_evidence_is_uncertain_and_json_safe(label: str, mutate) -> None:
    eig, occ = gapped_channel()
    base = evidence((eig,), (occ,), efermi=0.0)
    assert assess(base).classification == "nonmetallic"  # the unmutated evidence is definitive

    result = assess(mutate(base))
    payload = serialize_electronic_character_assessment(result)

    assert result.classification == "uncertain"
    assert {"INVALID_NUMERICAL_EVIDENCE", "OCCUPATION_METHOD_UNKNOWN"} & set(codes(result)), codes(result)
    json.dumps(payload, allow_nan=False)  # never invalid JSON
    render_electronic_character_assessment(result)


def test_missing_fermi_level_has_its_own_reason_code() -> None:
    eig, occ = gapped_channel()

    assert "FERMI_LEVEL_UNAVAILABLE" in codes(assess(evidence((eig,), (occ,), efermi=None)))


def test_smearing_method_without_sigma_is_uncertain() -> None:
    eig, occ = gapped_channel()
    result = assess(evidence((eig,), (occ,), efermi=0.0, ismear=0, sigma=None))

    assert "SMEARING_WIDTH_UNAVAILABLE" in codes(result)


@pytest.mark.parametrize("token", ("NaN", "Infinity", "-inf"))
def test_recorded_vasprun_with_non_finite_fermi_level_is_uncertain(tmp_path: Path, capsys, token: str) -> None:
    directory = write_direct_vasp(tmp_path / "calc", "LiF_tetrahedron.vasprun.xml.gz")
    path = directory / "vasprun.xml"
    text = path.read_text(encoding="utf-8")
    marker = '<i name="efermi">'
    start = text.index(marker) + len(marker)
    end = text.index("</i>", start)
    path.write_text(text[:start] + f" {token} " + text[end:], encoding="utf-8")

    code, output = run_cli(["assess", "electronic-character", str(directory), "--format", "json"], capsys)
    payload = json.loads(output)  # strict JSON parse: no NaN/Infinity tokens accepted below

    assert code == 0
    assert payload["classification"] == "uncertain"
    assert "NaN" not in output and "Infinity" not in output


# --- 3. threshold interactions (synthetic) ----------------------------------------------


def shallow_channel(*, edge: float, deficit: float = 0.0, nk: int = 16):
    """Gapped channel whose valence band rises to ``edge`` eV relative to EF=0
    at the last k-point, with ``deficit`` electrons removed from that state."""

    eig, occ = gapped_channel(nk)
    eig[:, 1] = np.linspace(-0.6, edge, nk)
    occ[:, 1] = 1.0
    occ[-1, 1] = 1.0 - deficit * nk
    return eig, occ


def test_codex_reproduction_shallow_tetrahedron_crossing_with_near_integer_filling() -> None:
    # Reproduction (review 1): band to EF + 0.04 eV, channel count 1.99, NELECT 3.98
    # -> nonmetallic. NELECT 3.98 is now itself a deliberately fractional count.
    eig, occ = shallow_channel(edge=0.04, deficit=0.01)
    result = assess(evidence((eig,), (occ,), efermi=0.0, ismear=-5, nelect=3.98))

    assert result.classification == "uncertain"
    assert "FRACTIONAL_ELECTRON_COUNT" in codes(result)
    # With an integer NELECT the same occupations miss 0.02 electrons.
    integer = assess(evidence((eig,), (occ,), efermi=0.0, ismear=-5, nelect=4.0))
    assert integer.classification == "uncertain"
    assert "ELECTRON_COUNT_INCONSISTENT" in codes(integer)


@pytest.mark.parametrize(
    ("edge", "expected", "code"),
    (
        (0.009, "nonmetallic", "CHANNEL_GAP_ESTABLISHED"),  # Fermi level pinned at the edge
        (0.011, "uncertain", "SHALLOW_FERMI_CROSSING"),
        (0.04, "uncertain", "SHALLOW_FERMI_CROSSING"),
        (0.05, "uncertain", "SHALLOW_FERMI_CROSSING"),
        (0.06, "uncertain", "FERMI_LEVEL_OUTSIDE_GAP"),
    ),
)
def test_valence_edge_relative_to_fermi_level_boundaries(edge: float, expected: str, code: str) -> None:
    eig, occ = shallow_channel(edge=edge)
    result = assess(evidence((eig,), (occ,), efermi=0.0, ismear=-5))

    assert result.classification == expected
    assert code in codes(result)


def moment_split_channels(transfer: float):
    """Collinear gapped channels with ``transfer`` electrons moved from the
    down to the up channel: integer total (NELECT 4), near-integer channels."""

    up, up_occ = shallow_channel(edge=-0.1)
    up_occ[0, 2] = transfer * 16  # conduction state at k-point 0 (1.0 eV)
    down, down_occ = shallow_channel(edge=-0.1, deficit=transfer)
    return (up, down), (up_occ, down_occ)


@pytest.mark.parametrize(
    ("transfer", "filling", "expected"),
    (
        (0.004, "integer", "nonmetallic"),
        (0.006, "near_integer", "uncertain"),
        (0.019, "near_integer", "uncertain"),
        (0.021, "fractional", "uncertain"),  # fractional, but no band crosses the Fermi level
    ),
)
def test_integer_filling_boundaries_for_gapped_channels(transfer: float, filling: str, expected: str) -> None:
    # Revised from review 1: channel near-integer filling is tested with an
    # integer recorded NELECT, so the requested count and the occupation
    # integration error are not confused.
    eigenvalues, occupations = moment_split_channels(transfer)
    result = assess(evidence(eigenvalues, occupations, efermi=0.0, ismear=-5, nelect=4.0))

    assert {item.filling for item in result.evidence.channels} == {filling}
    assert result.classification == expected


def fractional_channel(half_width: float, nk: int = 16):
    """A band from -half_width to +half_width eV around EF holding about half an electron."""

    eig, occ = gapped_channel(nk, nfilled=1)
    eig[:, 1] = np.linspace(-half_width, half_width, nk)
    eig[:, 2:] += 1.0
    occ[:, 1] = (eig[:, 1] < 0.0).astype(float)
    return eig, occ


@pytest.mark.parametrize(
    ("half_width", "expected", "code"),
    (
        (0.049, "uncertain", "FRACTIONAL_FILLING_WITHOUT_FERMI_CROSSING"),
        (0.051, "metallic", "PARTIALLY_FILLED_BAND_CROSSES_FERMI"),
    ),
)
def test_fermi_crossing_margin_boundary_for_metallic_evidence(half_width: float, expected: str, code: str) -> None:
    eig, occ = fractional_channel(half_width)
    result = assess(evidence((eig,), (occ,), efermi=0.0, ismear=-5))

    assert result.evidence.channels[0].filling == "fractional"
    assert result.classification == expected
    assert code in codes(result)


def test_fractional_filling_minimum_boundary() -> None:
    eig, occ = metallic_channel()
    weights = np.full(16, 1 / 16)
    count = float((occ * weights[:, None]).sum())
    assert abs(count - round(count)) >= ec.THRESHOLDS["fractional_filling_min"]
    assert assess(evidence((eig,), (occ,), efermi=0.0)).classification == "metallic"


@pytest.mark.parametrize(("gap", "expected"), ((0.0999, "uncertain"), (0.1, "nonmetallic"), (0.1001, "nonmetallic")))
def test_minimum_reliable_gap_boundary(gap: float, expected: str) -> None:
    eig, occ = gapped_channel(vbm=-gap / 2, cbm=gap / 2)
    assert assess(evidence((eig,), (occ,), efermi=0.0)).classification == expected


@pytest.mark.parametrize(("gap", "expected"), ((0.399, "uncertain"), (0.4, "nonmetallic")))
def test_smearing_gap_ratio_boundary(gap: float, expected: str) -> None:
    eig, occ = gapped_channel(vbm=-gap / 2, cbm=gap / 2)
    assert assess(evidence((eig,), (occ,), efermi=0.0, ismear=0, sigma=0.1)).classification == expected


@pytest.mark.parametrize(
    ("vbm_top", "cbm_bottom", "expected", "code"),
    (
        (0.0245, -0.0245, "uncertain", "SMALL_GAP"),  # overlap 0.049 eV < band_overlap_min_ev
        (0.04, -0.04, "uncertain", "BAND_OVERLAP_WITHOUT_FERMI_CROSSING"),  # no clear crossing
        (0.06, -0.02, "metallic", "BAND_OVERLAP_AT_FERMI"),
    ),
)
def test_band_overlap_and_crossing_boundaries(vbm_top: float, cbm_bottom: float, expected: str, code: str) -> None:
    eig, _ = gapped_channel(nk=16)
    eig[:, 1] = np.concatenate([np.linspace(-0.6, -0.1, 10), np.linspace(0.01, vbm_top, 6)])
    eig[:, 2] = np.concatenate([np.linspace(cbm_bottom, -0.01, 6), np.linspace(0.1, 0.6, 10)])
    occ = (eig < 0.0).astype(float)
    result = assess(evidence((eig,), (occ,), efermi=0.0, ismear=-5))

    assert result.evidence.channels[0].filling == "integer"
    assert result.classification == expected
    assert code in codes(result)


def test_thresholds_leave_an_uncertain_band_between_criteria() -> None:
    t = ec.THRESHOLDS
    assert t["integer_filling_tolerance"] < t["fractional_filling_min"]
    assert t["fermi_edge_tolerance_ev"] < t["fermi_crossing_margin_ev"]
    assert 2 * t["fermi_crossing_margin_ev"] <= t["min_reliable_gap_ev"]


# --- 4. band-gap metadata consistency with zero-weight k-points (synthetic) ---------------


def with_zero_weight_point(eig, occ, zero_row, coordinate):
    eig = np.vstack([eig, [zero_row]])
    occ = np.vstack([occ, np.zeros((1, eig.shape[1]))])
    weights = np.append(np.full(len(eig) - 1, 1 / (len(eig) - 1)), 0.0)
    kpoints = np.vstack([mesh_kpoints()[: len(eig) - 1], [coordinate]])
    return eig, occ, weights, kpoints


def assert_gap_metadata_consistent(result) -> None:
    evidence_summary = result.evidence
    for channel in evidence_summary.channels:
        if channel.vbm is not None and channel.cbm is not None and channel.band_gap_ev is not None:
            assert channel.band_gap_ev == pytest.approx(channel.cbm.energy_ev - channel.vbm.energy_ev, abs=1e-4)
            assert (channel.gap_type == "direct") == (channel.vbm.kpoint == channel.cbm.kpoint)
    if evidence_summary.band_gap_ev is not None:
        assert evidence_summary.band_gap_ev == pytest.approx(
            evidence_summary.cbm.energy_ev - evidence_summary.vbm.energy_ev, abs=1e-4
        )
        assert evidence_summary.vbm.energy_ev == max(c.vbm.energy_ev for c in evidence_summary.channels if c.vbm)
        assert evidence_summary.cbm.energy_ev == min(c.cbm.energy_ev for c in evidence_summary.channels if c.cbm)


def test_zero_weight_point_narrowing_the_gap_updates_every_published_edge() -> None:
    # Reproduction: the overall gap narrowed to 1.2 eV but channel edges, channel gap and
    # gap type still described the 1.5 eV weighted-mesh gap.
    eig, occ = gapped_channel(nk=16)
    eig, occ, weights, kpoints = with_zero_weight_point(eig, occ, [-2.5, -0.3, 0.9, 3.0], [0.375, 0.375, 0.375])
    result = assess(evidence((eig,), (occ,), efermi=0.0, weights=weights, kpoints=kpoints))
    channel = result.evidence.channels[0]

    assert result.classification == "nonmetallic"
    assert result.evidence.band_edge_evidence == "weighted_mesh_and_zero_weight_kpoints"
    assert result.evidence.band_gap_ev == channel.band_gap_ev == 1.2
    assert channel.vbm.kpoint_weight == channel.cbm.kpoint_weight == "zero_weight"
    assert channel.gap_type == result.evidence.gap_type == "direct"
    assert_gap_metadata_consistent(result)


def test_zero_weight_point_changing_gap_type_is_reported_consistently() -> None:
    # Weighted mesh: VBM and CBM at the same k-point (direct). A zero-weight point
    # lowers only the CBM, making the gap indirect.
    eig, occ = gapped_channel(nk=16)
    eig[:, 1] = -0.5 - 0.5 * np.linspace(0.0, 1.0, 16)
    eig[:, 2] = 1.0 + 0.5 * np.linspace(0.0, 1.0, 16)  # both edges at k-point 0
    weighted = assess(evidence((eig,), (occ,), efermi=0.0))
    assert weighted.evidence.gap_type == "direct"

    eig, occ, weights, kpoints = with_zero_weight_point(eig, occ, [-3.0, -1.5, 0.8, 3.0], [0.375, 0.375, 0.375])
    result = assess(evidence((eig,), (occ,), efermi=0.0, weights=weights, kpoints=kpoints))
    channel = result.evidence.channels[0]

    assert result.classification == "nonmetallic"
    assert channel.cbm.kpoint_weight == "zero_weight" and channel.vbm.kpoint_weight == "weighted"
    assert channel.gap_type == result.evidence.gap_type == "indirect"
    assert channel.band_gap_ev == result.evidence.band_gap_ev == 1.3
    assert_gap_metadata_consistent(result)


def test_zero_weight_point_placing_a_state_across_the_fermi_level_is_uncertain() -> None:
    eig, occ = gapped_channel(nk=16)
    eig, occ, weights, kpoints = with_zero_weight_point(eig, occ, [-2.5, 0.3, 1.2, 3.0], [0.375, 0.375, 0.375])
    result = assess(evidence((eig,), (occ,), efermi=0.0, weights=weights, kpoints=kpoints))

    assert result.classification == "uncertain"
    assert "FERMI_LEVEL_OUTSIDE_GAP" in codes(result)


def test_collinear_overall_edges_come_from_the_published_channel_edges() -> None:
    up, up_occ = gapped_channel(nfilled=3, nbands=5, vbm=-0.6, cbm=1.4)
    down, down_occ = gapped_channel(nfilled=1, nbands=5, vbm=-0.8, cbm=0.9)
    result = assess(evidence((up, down), (up_occ, down_occ), efermi=0.0))

    assert result.evidence.vbm_channel == "up" and result.evidence.cbm_channel == "down"
    assert result.evidence.gap_type == "indirect"  # edges in different spin channels
    assert_gap_metadata_consistent(result)
    payload = serialize_electronic_character_assessment(result)
    assert payload["electronic_evidence"]["vbm"]["channel"] == "up"
    assert payload["electronic_evidence"]["cbm"]["channel"] == "down"


@pytest.mark.parametrize(("fixture", "classification", "code"), RECORDED)
def test_recorded_gap_metadata_is_consistent(tmp_path: Path, fixture: str, classification: str, code: str) -> None:
    assert_gap_metadata_consistent(assess(read_vasprun_electronic_evidence(_decompressed(tmp_path, fixture))))


# --- parity and determinism ------------------------------------------------------------


def test_human_output_reports_the_same_edges_as_json() -> None:
    eig, occ = gapped_channel(nk=16)
    eig, occ, weights, kpoints = with_zero_weight_point(eig, occ, [-2.5, -0.3, 0.9, 3.0], [0.375, 0.375, 0.375])
    result = assess(evidence((eig,), (occ,), efermi=0.0, weights=weights, kpoints=kpoints))
    payload = serialize_electronic_character_assessment(result)
    text = render_electronic_character_assessment(result)
    channel = payload["electronic_evidence"]["spin_channels"][0]

    assert f"Band gap: {payload['electronic_evidence']['band_gap_ev']} eV ({payload['electronic_evidence']['gap_type']})" in text
    assert f"Band-edge evidence: {payload['electronic_evidence']['band_edge_evidence']}" in text
    assert f"VBM {channel['vbm']['energy_ev']} eV, CBM {channel['cbm']['energy_ev']} eV" in text


def test_serialization_is_deterministic_for_every_regression_scenario() -> None:
    shallow, shallow_occ = shallow_channel(edge=0.04, deficit=0.01)
    two, two_occ = gapped_channel(nk=2)
    narrowed, narrowed_occ = gapped_channel(nk=16)
    narrowed, narrowed_occ, weights, kpoints = with_zero_weight_point(
        narrowed, narrowed_occ, [-2.5, -0.3, 0.9, 3.0], [0.375, 0.375, 0.375]
    )
    base, base_occ = gapped_channel()
    scenarios = (
        evidence((shallow,), (shallow_occ,), efermi=0.0, ismear=-5, nelect=3.98),
        evidence((two,), (two_occ,), efermi=0.0, divisions=None, kpoints=[[0, 0, 0], [1 / 8, 1 / 8, 1 / 8]],
                 weights=[0.5, 0.5], kpoints_style="Reciprocal"),
        evidence((narrowed,), (narrowed_occ,), efermi=0.0, weights=weights, kpoints=kpoints),
        replace(evidence((base,), (base_occ,), efermi=0.0), efermi=float("nan")),
    )
    for vasprun in scenarios:
        first = json.dumps(serialize_electronic_character_assessment(assess(vasprun)), sort_keys=True, allow_nan=False)
        second = json.dumps(serialize_electronic_character_assessment(assess(vasprun)), sort_keys=True, allow_nan=False)
        assert first == second


# === Regression tests for the second independent review of C2 ==========================


# --- 1. malformed container types (synthetic) ------------------------------------------

MALFORMED_CASES = (
    ("declared_divisions=5", {"declared_divisions": 5}),
    ("declared_divisions string", {"declared_divisions": "444"}),
    ("declared_divisions length 2", {"declared_divisions": (4, 4)}),
    ("declared_divisions nested", {"declared_divisions": [[4, 4, 4]]}),
    ("declared_divisions mapping", {"declared_divisions": {"a": 4}}),
    ("eigenvalues as one ndarray", "eigen-ndarray"),
    ("occupations as one ndarray", "occ-ndarray"),
    ("eigenvalues None", {"eigenvalues": None}),
    ("eigenvalues string", {"eigenvalues": "abc"}),
    ("three spin channels", "three-channels"),
    ("kpoints_style=5", {"kpoints_style": 5}),
    ("kpoints_style bytes", {"kpoints_style": b"Gamma"}),
    ("kpoints_style list", {"kpoints_style": ["Gamma"]}),
    ("ALGO=5", {"algo": 5}),
    ("LSORBIT array", {"lsorbit": np.array([True, False])}),
    ("LSORBIT string", {"lsorbit": "yes"}),
    ("LNONCOLLINEAR None", {"lnoncollinear": None}),
    ("converged flag string", {"converged_electronic": "yes"}),
    ("reciprocal scalar", {"reciprocal_lengths": 2.0}),
    ("reciprocal string", {"reciprocal_lengths": "2 2 2"}),
    ("shift scalar", {"kpoints_shift": 0.5}),
    ("weights scalar", {"weights": 1.0}),
    ("kpoints flat", {"kpoints_frac": np.zeros(48)}),
)


@pytest.mark.parametrize(("label", "change"), MALFORMED_CASES, ids=[item[0] for item in MALFORMED_CASES])
def test_malformed_containers_are_uncertain_without_exceptions(label: str, change) -> None:
    eig, occ = gapped_channel()
    base = evidence((eig,), (occ,), efermi=0.0)
    if change == "eigen-ndarray":
        mutated = replace(base, eigenvalues=np.asarray(base.eigenvalues))
    elif change == "occ-ndarray":
        mutated = replace(base, occupations=np.asarray(base.occupations))
    elif change == "three-channels":
        mutated = replace(base, eigenvalues=(eig, eig, eig), occupations=(occ, occ, occ))
    else:
        mutated = replace(base, **change)

    result = assess(mutated)

    assert result.classification == "uncertain"
    assert "INVALID_NUMERICAL_EVIDENCE" in codes(result)
    json.dumps(serialize_electronic_character_assessment(result), allow_nan=False)
    render_electronic_character_assessment(result)


def test_codex_round_two_container_reproductions() -> None:
    # Reproductions: TypeError, ambiguous-truth ValueError and AttributeError.
    eig, occ = gapped_channel()
    base = evidence((eig,), (occ,), efermi=0.0)
    for mutated in (
        replace(base, declared_divisions=5),
        replace(base, eigenvalues=np.asarray(base.eigenvalues)),
        replace(base, kpoints_style=5),
    ):
        assert "INVALID_NUMERICAL_EVIDENCE" in codes(assess(mutated))


# --- 2. fractional NELECT (synthetic) --------------------------------------------------


def test_codex_round_two_fractional_nelect_reproduction() -> None:
    # Reproduction: channel count 1.996 with recorded NELECT 3.992 -> nonmetallic.
    eig, occ = shallow_channel(edge=-0.1, deficit=0.004)
    result = assess(evidence((eig,), (occ,), efermi=0.0, ismear=-5, nelect=3.992))

    assert result.classification == "uncertain"
    assert "FRACTIONAL_ELECTRON_COUNT" in codes(result)


@pytest.mark.parametrize(("offset", "expected"), ((0.0, "nonmetallic"), (9e-7, "nonmetallic"), (1.1e-6, "uncertain"), (1e-3, "uncertain")))
@pytest.mark.parametrize("treatment", ("non_spin_polarized", "collinear", "noncollinear"))
def test_nelect_integer_tolerance_boundary(treatment: str, offset: float, expected: str) -> None:
    eig, occ = gapped_channel(nfilled=2)
    if treatment == "non_spin_polarized":
        vasprun = evidence((eig,), (occ,), efermi=0.0, nelect=4.0 + offset)
    elif treatment == "collinear":
        vasprun = evidence((eig, eig.copy()), (occ, occ.copy()), efermi=0.0, nelect=4.0 + offset)
    else:
        vasprun = evidence((eig,), (occ,), efermi=0.0, lsorbit=True, nelect=2.0 + offset)
    result = assess(vasprun)

    assert result.classification == expected
    if expected == "uncertain":
        assert "FRACTIONAL_ELECTRON_COUNT" in codes(result)


def test_fractional_nelect_is_not_classified_metallic() -> None:
    eig, occ = metallic_channel()  # clearly fractional channel, clear Fermi crossing
    weights = np.full(16, 1 / 16)
    count = 2 * float((occ * weights[:, None]).sum())
    result = assess(evidence((eig,), (occ,), efermi=0.0, nelect=count + 0.25, ismear=-5))
    shifted = assess(evidence((eig,), (occ,), efermi=0.0, nelect=count, ismear=-5))

    assert shifted.classification == "metallic"  # integer NELECT (3) is metallic
    assert result.classification == "uncertain"  # NELECT mismatch or fractional
    fractional = assess(replace(evidence((eig,), (occ,), efermi=0.0, ismear=-5), nelect=3.2))
    assert fractional.classification == "uncertain"


@pytest.mark.parametrize(("deficit", "expected", "code"), ((0.0, "nonmetallic", None), (0.01, "uncertain", "FRACTIONAL_ELECTRON_COUNT")))
def test_missing_nelect_uses_the_occupation_count(deficit: float, expected: str, code) -> None:
    eig, occ = shallow_channel(edge=-0.1, deficit=deficit)
    vasprun = replace(evidence((eig,), (occ,), efermi=0.0, ismear=-5), nelect=None)
    result = assess(vasprun)

    assert result.classification == expected
    if code:
        assert code in codes(result)
    else:
        assert any("NELECT was not recorded" in item for item in result.limitations)


def test_occupation_integration_noise_within_tolerance_is_accepted() -> None:
    eig, occ = shallow_channel(edge=-0.1, deficit=0.0002)  # 0.0004 electrons in total
    result = assess(evidence((eig,), (occ,), efermi=0.0, ismear=-5, nelect=4.0))

    assert result.classification == "nonmetallic"


# --- 3. degenerate band extrema (synthetic) --------------------------------------------

GRID = mesh_kpoints()  # 16 Gamma-mesh points (i/4, j/4, 0)
A, B, C, D = 0, 5, 10, 15  # (0,0,0), (1/4,1/4,0), (1/2,1/2,0), (3/4,3/4,0)


def flat_bands(*, vbm_points: dict[int, float], cbm_points: dict[int, float], valence=-1.0, conduction=2.0):
    eig, occ = gapped_channel()
    eig[:, 0] = -3.0
    eig[:, 1] = valence
    eig[:, 2] = conduction
    eig[:, 3] = 4.0
    for index, energy in vbm_points.items():
        eig[index, 1] = energy
    for index, energy in cbm_points.items():
        eig[index, 2] = energy
    return eig, occ


def grid_key(index: int):
    return tuple(round(float(item), 4) for item in GRID[index])


def test_codex_round_two_degenerate_direct_gap_reproduction() -> None:
    # Reproduction: VBM degenerate at A and B, CBM at B and C, direct at B -> "indirect".
    eig, occ = flat_bands(vbm_points={A: -0.5, B: -0.5}, cbm_points={B: 1.0, C: 1.0})
    result = assess(evidence((eig,), (occ,), efermi=0.0))
    channel = result.evidence.channels[0]

    assert result.evidence.gap_type == channel.gap_type == "direct"
    assert channel.vbm.kpoint == channel.cbm.kpoint == grid_key(B)
    assert (channel.vbm.degenerate_kpoints, channel.cbm.degenerate_kpoints) == (2, 2)
    assert channel.min_direct_gap_ev == channel.band_gap_ev == 1.5


def test_indirect_gap_with_degenerate_extrema() -> None:
    eig, occ = flat_bands(vbm_points={A: -0.5, B: -0.5}, cbm_points={C: 1.0, D: 1.0})
    result = assess(evidence((eig,), (occ,), efermi=0.0))
    channel = result.evidence.channels[0]

    assert channel.gap_type == "indirect"
    assert channel.vbm.kpoint == grid_key(A) and channel.cbm.kpoint == grid_key(C)  # first of each set
    assert (channel.vbm.degenerate_kpoints, channel.cbm.degenerate_kpoints) == (2, 2)
    assert channel.min_direct_gap_ev == 2.0  # at C: 1.0 - (-1.0)


@pytest.mark.parametrize("equivalent", ([1.0, 0.0, 0.0], [-1.0, 1.0, 2.0], [1.0 - 1e-6, 0.0, 0.0]))
def test_extrema_at_periodically_equivalent_kpoints_are_direct(equivalent) -> None:
    eig, occ = flat_bands(vbm_points={A: -0.5}, cbm_points={})
    eig = np.vstack([eig, [[-3.0, -1.0, 1.0, 4.0]]])  # CBM only at a zero-weight copy of Gamma
    occ = np.vstack([occ, np.zeros((1, 4))])
    weights = np.append(np.full(16, 1 / 16), 0.0)
    kpoints = np.vstack([GRID, [equivalent]])
    result = assess(evidence((eig,), (occ,), efermi=0.0, weights=weights, kpoints=kpoints))
    channel = result.evidence.channels[0]

    assert channel.gap_type == "direct"
    assert channel.vbm.kpoint == channel.cbm.kpoint == (0.0, 0.0, 0.0)
    assert channel.cbm.kpoint_weight == "weighted"  # Gamma's class contains a weighted member


@pytest.mark.parametrize(
    ("delta", "count", "gap_type"),
    (
        (0.0009, 2, "direct"),  # B's valence state is degenerate with the VBM at A
        (0.0011, 1, "undetermined"),  # just outside: smallest direct gap 1.1 meV above
        (0.0099, 1, "undetermined"),
        (0.0101, 1, "indirect"),
    ),
)
def test_near_degenerate_extrema_on_either_side_of_the_tolerances(delta: float, count: int, gap_type: str) -> None:
    eig, occ = flat_bands(vbm_points={A: -0.5, B: -0.5 - delta}, cbm_points={B: 1.0})
    result = assess(evidence((eig,), (occ,), efermi=0.0))
    channel = result.evidence.channels[0]

    assert result.classification == "nonmetallic"
    assert channel.vbm.degenerate_kpoints == count
    assert channel.gap_type == result.evidence.gap_type == gap_type
    assert_gap_metadata_consistent(result)


def test_zero_weight_only_extremum_is_labelled_zero_weight() -> None:
    eig, occ = flat_bands(vbm_points={A: -0.5}, cbm_points={C: 1.2})
    eig = np.vstack([eig, [[-3.0, -1.0, 1.0, 4.0]]])
    occ = np.vstack([occ, np.zeros((1, 4))])
    weights = np.append(np.full(16, 1 / 16), 0.0)
    kpoints = np.vstack([GRID, [[0.375, 0.375, 0.375]]])
    result = assess(evidence((eig,), (occ,), efermi=0.0, weights=weights, kpoints=kpoints))
    channel = result.evidence.channels[0]

    assert channel.cbm.kpoint_weight == "zero_weight" and channel.cbm.kpoint == (0.375, 0.375, 0.375)
    assert channel.vbm.kpoint_weight == "weighted"
    assert channel.gap_type == "indirect"


def test_degenerate_edge_classification_is_invariant_to_kpoint_order() -> None:
    eig, occ = flat_bands(vbm_points={A: -0.5, B: -0.5, D: -0.5004}, cbm_points={B: 1.0, C: 1.0003})
    reference = serialize_electronic_character_assessment(assess(evidence((eig,), (occ,), efermi=0.0)))
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(16)
        permuted = assess(evidence((eig[order],), (occ[order],), efermi=0.0, kpoints=GRID[order]))
        assert serialize_electronic_character_assessment(permuted) == reference


def test_collinear_degenerate_channels_report_one_consistent_direct_gap() -> None:
    eig, occ = flat_bands(vbm_points={A: -0.5, B: -0.5}, cbm_points={B: 1.0, C: 1.0})
    result = assess(evidence((eig, eig.copy()), (occ, occ.copy()), efermi=0.0))

    assert result.evidence.gap_type == "direct"
    assert result.evidence.vbm_channel == result.evidence.cbm_channel == "up"
    assert result.evidence.vbm.kpoint == result.evidence.cbm.kpoint == grid_key(B)
    assert_gap_metadata_consistent(result)


def test_direct_gap_in_one_spin_channel_with_extrema_split_across_channels() -> None:
    # Up channel: VBM -0.5 at A, CBM 1.0 at A (direct). Down channel: CBM 0.9 at C.
    # The fundamental gap is between channels (1.4 eV) -> not direct.
    up, up_occ = flat_bands(vbm_points={A: -0.5}, cbm_points={A: 1.0})
    down, down_occ = flat_bands(vbm_points={}, cbm_points={C: 0.9})
    result = assess(evidence((up, down), (up_occ, down_occ), efermi=0.0))

    assert result.evidence.band_gap_ev == 1.4
    assert result.evidence.vbm_channel == "up" and result.evidence.cbm_channel == "down"
    assert result.evidence.gap_type in {"indirect", "undetermined"}
    assert result.evidence.gap_type == "indirect"  # smallest direct gap 1.5 eV exceeds 1.4 eV by 0.1 eV


# === Regression tests for the third independent review of C2 ===========================


# --- 1. zero-dimensional and malformed NumPy metadata; NBANDS (synthetic) --------------

ARRAY_CASES = (
    ("divisions 0-d array", {"declared_divisions": np.array(4)}),
    ("divisions 2-vector", {"declared_divisions": np.array([4, 4])}),
    ("divisions 2-d array", {"declared_divisions": np.array([[4, 4, 4]])}),
    ("divisions bool array", {"declared_divisions": np.array([True, True, True])}),
    ("divisions float array", {"declared_divisions": np.array([4.0, 4.0, 4.0])}),
    ("divisions with 0-d items", {"declared_divisions": (np.array(4), 4, 4)}),
    ("shift 0-d array", {"kpoints_shift": np.array(0.5)}),
    ("shift 2-d array", {"kpoints_shift": np.zeros((1, 3))}),
    ("reciprocal 0-d array", {"reciprocal_lengths": np.array(2.0)}),
    ("reciprocal 2-d array", {"reciprocal_lengths": np.full((3, 1), 2.0)}),
    ("reciprocal 4-vector", {"reciprocal_lengths": np.full(4, 2.0)}),
    ("Fermi 0-d array", {"efermi": np.array(0.0)}),
    ("NELECT 0-d array", {"nelect": np.array(4.0)}),
    ("SIGMA 0-d array", {"sigma": np.array(0.05)}),
    ("ISMEAR 0-d array", {"ismear": np.array(-5)}),
    ("NBANDS 0-d array", {"nbands": np.array(4)}),
    ("weights 0-d array", {"weights": np.array(1.0)}),
    ("weights 2-d array", {"weights": np.full((16, 1), 1 / 16)}),
    ("kpoints 1-d array", {"kpoints_frac": np.zeros(48)}),
    ("kpoints 3-d array", {"kpoints_frac": np.zeros((16, 3, 1))}),
    ("channel 0-d array", "channel-0d"),
    ("channel 3-d array", "channel-3d"),
)


@pytest.mark.parametrize(("label", "change"), ARRAY_CASES, ids=[item[0] for item in ARRAY_CASES])
def test_malformed_numpy_metadata_is_uncertain_and_json_safe(label: str, change) -> None:
    eig, occ = gapped_channel()
    base = evidence((eig,), (occ,), efermi=0.0)
    if change == "channel-0d":
        mutated = replace(base, eigenvalues=(np.array(1.0),))
    elif change == "channel-3d":
        mutated = replace(base, eigenvalues=(eig[:, :, None],))
    else:
        mutated = replace(base, **change)

    result = assess(mutated)

    assert result.classification == "uncertain"
    assert "INVALID_NUMERICAL_EVIDENCE" in codes(result)
    json.dumps(serialize_electronic_character_assessment(result), allow_nan=False)
    render_electronic_character_assessment(result)


def test_valid_one_dimensional_numpy_metadata_is_accepted() -> None:
    eig, occ = gapped_channel()
    base = evidence((eig,), (occ,), efermi=0.0)
    result = assess(
        replace(
            base,
            declared_divisions=np.array([4, 4, 4]),
            kpoints_shift=np.zeros(3),
            reciprocal_lengths=np.full(3, 2.0),
        )
    )

    assert result.classification == "nonmetallic"
    assert result.evidence.mesh_divisions == (4, 4, 4)


def test_codex_round_three_zero_dimensional_divisions_reproduction() -> None:
    # Reproduction: declared_divisions=np.array(4) -> TypeError (iteration over a 0-d array).
    eig, occ = gapped_channel()
    result = assess(replace(evidence((eig,), (occ,), efermi=0.0), declared_divisions=np.array(4)))

    assert "INVALID_NUMERICAL_EVIDENCE" in codes(result)


@pytest.mark.parametrize(
    ("nbands", "code"),
    (
        (-1, "INVALID_NUMERICAL_EVIDENCE"),
        (0, "INVALID_NUMERICAL_EVIDENCE"),
        (1, "NBANDS_INCONSISTENT"),
        (3, "NBANDS_INCONSISTENT"),
        (5, "NBANDS_INCONSISTENT"),
        (True, "INVALID_NUMERICAL_EVIDENCE"),
        (4.0, "INVALID_NUMERICAL_EVIDENCE"),
        ("4", "INVALID_NUMERICAL_EVIDENCE"),
        (None, "NBANDS_UNAVAILABLE"),
    ),
)
def test_nbands_must_match_four_band_evidence(nbands, code: str) -> None:
    eig, occ = gapped_channel(nbands=4)
    result = assess(replace(evidence((eig,), (occ,), efermi=0.0), nbands=nbands))

    assert result.classification == "uncertain"
    assert code in codes(result)
    json.dumps(serialize_electronic_character_assessment(result), allow_nan=False)


@pytest.mark.parametrize("treatment", ("non_spin_polarized", "collinear", "noncollinear"))
def test_matching_nbands_is_accepted_for_every_spin_treatment(treatment: str) -> None:
    eig, occ = gapped_channel(nbands=4)
    if treatment == "collinear":
        vasprun = evidence((eig, eig.copy()), (occ, occ.copy()), efermi=0.0)
    else:
        vasprun = evidence((eig,), (occ,), efermi=0.0, lsorbit=treatment == "noncollinear")
    result = assess(replace(vasprun, nbands=4))

    assert result.classification == "nonmetallic"


def test_recorded_fixtures_record_matching_nbands(tmp_path: Path) -> None:
    for fixture, _, _ in RECORDED:
        vasprun = read_vasprun_electronic_evidence(_decompressed(tmp_path, fixture))
        assert vasprun.nbands is not None
        assert {item.shape[1] for item in vasprun.eigenvalues} == {vasprun.nbands}


# --- 2. periodic k-point equivalence (synthetic) ---------------------------------------


def gamma_vbm_with_zero_weight_cbm(*extra_points, cbm_point: int = 0, cbm: float = 1.0):
    """VBM -0.5 eV at Gamma (weighted); CBM at one added zero-weight point."""

    eig, occ = flat_bands(vbm_points={A: -0.5}, cbm_points={})
    rows, points = [], []
    for position, point in enumerate(extra_points):
        rows.append([-3.0, -1.0, cbm if position == cbm_point else 2.0, 4.0])
        points.append(point)
    eig = np.vstack([eig, rows])
    occ = np.vstack([occ, np.zeros((len(rows), 4))])
    weights = np.append(np.full(16, 1 / 16), np.zeros(len(rows)))
    kpoints = np.vstack([GRID, points])
    return evidence((eig,), (occ,), efermi=0.0, weights=weights, kpoints=kpoints)


def test_codex_round_three_kpoint_equivalence_reproduction() -> None:
    # Reproduction: VBM at (0,0,0), CBM at zero-weight (0.00006,0,0) -> indirect.
    result = assess(gamma_vbm_with_zero_weight_cbm([0.00006, 0.0, 0.0]))
    channel = result.evidence.channels[0]

    assert result.evidence.band_gap_ev == 1.5
    assert channel.gap_type == result.evidence.gap_type == "direct"
    assert channel.vbm.kpoint == channel.cbm.kpoint == (0.0, 0.0, 0.0)
    assert channel.cbm.kpoint_weight == "weighted"  # Gamma's class includes a weighted member
    assert channel.kpoint_equivalence_ambiguous is False


@pytest.mark.parametrize(
    ("offset", "gap_type"),
    ((0.00009, "direct"), (0.0000999, "direct"), (0.000101, "indirect"), (0.00011, "indirect")),
)
def test_equivalence_tolerance_boundary(offset: float, gap_type: str) -> None:
    result = assess(gamma_vbm_with_zero_weight_cbm([offset, 0.0, 0.0]))
    channel = result.evidence.channels[0]

    assert channel.gap_type == gap_type
    if gap_type == "indirect":
        assert channel.cbm.kpoint_weight == "zero_weight"


@pytest.mark.parametrize(
    "point",
    ([0.99995, 0.0, 0.0], [-0.00005, 1.0, 0.0], [1.00004, -1.0, 2.0], [0.99996, 0.99996, 0.00004]),
)
def test_equivalence_wraps_across_reciprocal_lattice_boundaries(point) -> None:
    result = assess(gamma_vbm_with_zero_weight_cbm(point))
    channel = result.evidence.channels[0]

    assert channel.gap_type == "direct"
    assert channel.cbm.kpoint == (0.0, 0.0, 0.0)


def test_non_transitive_near_equivalence_chain_at_the_extrema_is_undetermined() -> None:
    # a ~ b and b ~ c within 1e-4, but a and c are 1.6e-4 apart.
    a, b, c = [0.5, 0.0, 0.0], [0.50008, 0.0, 0.0], [0.50016, 0.0, 0.0]
    eig, occ = flat_bands(vbm_points={8: -0.5}, cbm_points={})  # VBM at a = GRID[8]
    eig = np.vstack([eig, [[-3.0, -1.0, 2.0, 4.0], [-3.0, -1.0, 1.0, 4.0]]])  # CBM at c
    occ = np.vstack([occ, np.zeros((2, 4))])
    weights = np.append(np.full(16, 1 / 16), [0.0, 0.0])
    kpoints = np.vstack([GRID, [b, c]])
    assert np.allclose(GRID[8], a)
    result = assess(evidence((eig,), (occ,), efermi=0.0, weights=weights, kpoints=kpoints))
    channel = result.evidence.channels[0]

    assert result.classification == "nonmetallic"  # the gap value itself is unaffected
    assert channel.kpoint_equivalence_ambiguous is True
    assert channel.gap_type == result.evidence.gap_type == "undetermined"
    payload = serialize_electronic_character_assessment(result)
    assert payload["electronic_evidence"]["spin_channels"][0]["kpoint_equivalence_ambiguous"] is True


def test_ambiguous_chain_away_from_the_extrema_does_not_change_the_gap_type() -> None:
    chain = ([0.25, 0.5, 0.0], [0.25008, 0.5, 0.0], [0.25016, 0.5, 0.0])  # starts at GRID[6]
    vasprun = gamma_vbm_with_zero_weight_cbm([0.00006, 0.0, 0.0], *chain[1:])
    result = assess(vasprun)

    assert result.evidence.channels[0].gap_type == "direct"
    assert result.evidence.channels[0].kpoint_equivalence_ambiguous is False


def test_equivalence_classes_are_invariant_to_kpoint_order() -> None:
    # The chain starts at the grid point (0.5, 0, 0) already in the mesh.
    b, c = [0.50008, 0.0, 0.0], [0.50016, 0.0, 0.0]
    for cbm_offset in ([0.00006, 0.0, 0.0], [0.99995, 0.0, 0.0]):
        vasprun = gamma_vbm_with_zero_weight_cbm(cbm_offset, b, c)
        reference = serialize_electronic_character_assessment(assess(vasprun))
        for seed in range(6):
            order = np.random.default_rng(seed).permutation(len(vasprun.weights))
            permuted = replace(
                vasprun,
                eigenvalues=(vasprun.eigenvalues[0][order],),
                occupations=(vasprun.occupations[0][order],),
                weights=vasprun.weights[order],
                kpoints_frac=vasprun.kpoints_frac[order],
            )
            assert serialize_electronic_character_assessment(assess(permuted)) == reference


def test_zero_weight_only_class_keeps_its_zero_weight_label() -> None:
    result = assess(gamma_vbm_with_zero_weight_cbm([0.375, 0.375, 0.375]))
    channel = result.evidence.channels[0]

    assert channel.cbm.kpoint_weight == "zero_weight"
    assert channel.cbm.kpoint == (0.375, 0.375, 0.375)
    assert channel.gap_type == "indirect"


def test_periodic_distance_and_classes_directly() -> None:
    assert float(ec._periodic_distance([0.99995, 0, 0], [0, 0, 0])) == pytest.approx(5e-5)
    assert float(ec._periodic_distance([-0.5, 0, 0], [0.5, 0, 0])) == pytest.approx(0.0)
    eig, occ = gapped_channel()
    base = evidence((eig,), (occ,), efermi=0.0)
    valid, _ = ec._validate_numerical_evidence(base)
    classes, ambiguous = ec._kpoint_classes(valid)
    assert len(classes) == 16 and not ambiguous


# ---------------------------------------------------------------------------
# Regression tests for the fourth independent review of C2: an ambiguous
# k-point equivalence chain must not manufacture the smallest direct gap.
# ---------------------------------------------------------------------------

CHAIN = ([0.25008, 0.0, 0.0], [0.25016, 0.0, 0.0])  # with GRID[4] = (0.25, 0, 0)
FAR_CBM = [0.375, 0.375, 0.375]


def chain_evidence(*, b=(-0.7, 3.0), c=(-1.0, 1.2), cbm=1.0, spin_down=None):
    """VBM -0.5 eV at Gamma, CBM ``cbm`` at a far zero-weight point, and an
    ambiguous chain 0.25 ~ 0.25008 ~ 0.25016 (0.25 and 0.25016 are 1.6e-4 apart)
    whose members carry (valence, conduction) energies ``b`` and ``c``."""

    assert np.allclose(GRID[4], [0.25, 0.0, 0.0])

    def channel():
        eig, occ = flat_bands(vbm_points={A: -0.5}, cbm_points={})
        rows = [[-3.0, -1.0, cbm, 4.0], [-3.0, b[0], b[1], 4.0], [-3.0, c[0], c[1], 4.0]]
        return np.vstack([eig, rows]), np.vstack([occ, np.zeros((3, 4))])

    eig, occ = channel()
    eigenvalues, occupations = [eig], [occ]
    if spin_down is not None:
        eigenvalues.append(spin_down(eig.copy()))
        occupations.append(occ)
    weights = np.append(np.full(16, 1 / 16), np.zeros(3))
    kpoints = np.vstack([GRID, [FAR_CBM, *CHAIN]])
    return evidence(tuple(eigenvalues), tuple(occupations), efermi=0.0, weights=weights, kpoints=kpoints)


def sampled_direct_gaps(vasprun, channel=0):
    eig = vasprun.eigenvalues[channel]
    return eig[:, 2] - eig[:, 1]


def test_codex_round_four_manufactured_min_direct_gap_reproduction() -> None:
    # Previously: min_direct_gap_ev = 1.9 (valence of 0.25008 with conduction of
    # 0.25016) with kpoint_equivalence_ambiguous = False. No sampled k-point
    # has a same-k gap below 2.0 eV.
    vasprun = chain_evidence()
    assert float(sampled_direct_gaps(vasprun).min()) == pytest.approx(2.0)
    result = assess(vasprun)
    channel = result.evidence.channels[0]

    assert result.classification == "nonmetallic"
    assert result.evidence.band_gap_ev == channel.band_gap_ev == 1.5
    assert channel.vbm.kpoint == (0.0, 0.0, 0.0)
    assert channel.cbm.kpoint == (0.375, 0.375, 0.375)
    assert channel.min_direct_gap_ev is None  # not established
    assert channel.min_direct_gap_lower_bound_ev == 1.9  # a bound, not an observation
    assert channel.kpoint_equivalence_ambiguous is True
    # Even the lowest possible direct gap exceeds the fundamental gap by more
    # than the resolution, so the indirect label holds under every reading.
    assert channel.gap_type == result.evidence.gap_type == "indirect"
    spin = serialize_electronic_character_assessment(result)["electronic_evidence"]["spin_channels"][0]
    assert spin["min_direct_gap_ev"] is None
    assert spin["min_direct_gap_lower_bound_ev"] == 1.9
    assert spin["kpoint_equivalence_ambiguous"] is True
    assert render_electronic_character_assessment(result)


def test_ambiguous_min_direct_gap_within_resolution_is_undetermined() -> None:
    # The chain does not touch the VBM (-0.5) or CBM (1.0) sets, but its
    # combined direct gap 1.506 eV is within 10 meV of the 1.5 eV gap.
    result = assess(chain_evidence(b=(-0.502, 3.0), c=(-1.0, 1.004)))
    channel = result.evidence.channels[0]

    assert result.classification == "nonmetallic"
    assert channel.band_gap_ev == 1.5
    assert channel.vbm.degenerate_kpoints == channel.cbm.degenerate_kpoints == 1
    assert channel.min_direct_gap_ev is None
    assert channel.min_direct_gap_lower_bound_ev == pytest.approx(1.506)
    assert channel.kpoint_equivalence_ambiguous is True
    assert channel.gap_type == result.evidence.gap_type == "undetermined"


def test_ambiguous_chain_not_setting_the_min_direct_gap_publishes_observed_value() -> None:
    # The chain's combined gap (2.7 eV) is above the 2.0 eV same-k gap observed
    # at the CBM point, so ambiguity does not reach any published quantity.
    vasprun = chain_evidence(c=(-1.0, 2.0))
    result = assess(vasprun)
    channel = result.evidence.channels[0]

    assert channel.min_direct_gap_ev == channel.min_direct_gap_lower_bound_ev == 2.0
    assert channel.min_direct_gap_ev == pytest.approx(float(sampled_direct_gaps(vasprun).min()))
    assert channel.kpoint_equivalence_ambiguous is False
    assert channel.gap_type == "indirect"


def test_ambiguous_chain_without_mixing_publishes_observed_value() -> None:
    # Members of the chain with identical bands: combining them changes nothing.
    result = assess(chain_evidence(b=(-1.0, 1.4), c=(-1.0, 1.4)))
    channel = result.evidence.channels[0]

    assert channel.min_direct_gap_ev == channel.min_direct_gap_lower_bound_ev == 2.0
    assert channel.kpoint_equivalence_ambiguous is False


def test_ambiguous_chain_does_not_override_an_established_direct_gap() -> None:
    # Gamma (an unambiguous class) holds both extrema; the chain member 0.25008
    # ties the VBM, so the edges are flagged, but direct remains established.
    vasprun = chain_evidence(cbm=2.0, b=(-0.5, 3.0), c=(-1.0, 3.0))
    vasprun.eigenvalues[0][A, 2] = 1.0
    result = assess(vasprun)
    channel = result.evidence.channels[0]

    assert channel.vbm.kpoint == channel.cbm.kpoint == (0.0, 0.0, 0.0)
    assert channel.vbm.degenerate_kpoints == 2
    assert channel.band_gap_ev == 1.5
    assert channel.gap_type == result.evidence.gap_type == "direct"
    assert channel.min_direct_gap_ev == channel.min_direct_gap_lower_bound_ev == 1.5
    assert channel.kpoint_equivalence_ambiguous is True


def test_ambiguous_min_direct_gap_in_one_spin_channel() -> None:
    def shift_down(eig):
        eig[:, 2:] += 0.3  # spin-down conduction bands 0.3 eV higher
        return eig

    result = assess(chain_evidence(spin_down=shift_down))
    up, down = result.evidence.channels

    assert result.classification == "nonmetallic"
    assert result.evidence.band_gap_ev == 1.5
    assert up.min_direct_gap_ev is None and up.kpoint_equivalence_ambiguous is True
    assert down.min_direct_gap_ev is None and down.min_direct_gap_lower_bound_ev == pytest.approx(2.2)
    assert result.evidence.gap_type == "indirect"


def test_overall_gap_type_uses_the_lower_bound_across_channels() -> None:
    # Spin-down chain gap 1.504 eV is within resolution of the 1.5 eV gap set
    # by spin-up: the overall label must not be indirect.
    def close_chain(eig):
        eig[16:, :] = [[-3.0, -1.0, 2.0, 4.0], [-3.0, -0.502, 3.0, 4.0], [-3.0, -1.0, 1.002, 4.0]]
        eig[:, 1] = np.minimum(eig[:, 1], -0.502)
        return eig

    result = assess(chain_evidence(c=(-1.0, 2.0), spin_down=close_chain))
    up, down = result.evidence.channels

    assert up.kpoint_equivalence_ambiguous is False and up.gap_type == "indirect"
    assert down.kpoint_equivalence_ambiguous is True
    assert down.min_direct_gap_ev is None
    assert result.evidence.band_gap_ev == 1.5
    assert result.evidence.gap_type == "undetermined"


def test_min_direct_gap_ambiguity_is_invariant_to_kpoint_order() -> None:
    vasprun = chain_evidence()
    reference = serialize_electronic_character_assessment(assess(vasprun))
    for seed in range(6):
        order = np.random.default_rng(seed).permutation(len(vasprun.weights))
        permuted = replace(
            vasprun,
            eigenvalues=(vasprun.eigenvalues[0][order],),
            occupations=(vasprun.occupations[0][order],),
            weights=vasprun.weights[order],
            kpoints_frac=vasprun.kpoints_frac[order],
        )
        assert serialize_electronic_character_assessment(assess(permuted)) == reference


@pytest.mark.parametrize("seed", range(20))
def test_published_min_direct_gap_is_always_observed_or_withheld(seed: int) -> None:
    rng = np.random.default_rng(seed)
    b = (float(rng.uniform(-0.95, -0.55)), float(rng.uniform(1.1, 3.0)))
    c = (float(rng.uniform(-0.95, -0.55)), float(rng.uniform(1.1, 3.0)))
    vasprun = chain_evidence(b=b, c=c)
    channel = assess(vasprun).evidence.channels[0]
    observed = float(sampled_direct_gaps(vasprun).min())
    combined = max(-1.0, b[0], c[0])
    combined = min(2.0, b[1], c[1]) - combined  # the chain's combined gap

    assert channel.min_direct_gap_lower_bound_ev == pytest.approx(min(observed, combined), abs=1e-4)
    assert channel.min_direct_gap_lower_bound_ev <= observed + 1e-4
    if channel.min_direct_gap_ev is not None:
        assert channel.min_direct_gap_ev == pytest.approx(observed, abs=1e-4)
        assert channel.kpoint_equivalence_ambiguous is False
    else:
        assert combined < observed
        assert channel.kpoint_equivalence_ambiguous is True


# ---------------------------------------------------------------------------
# Regression tests for the fifth independent review of C2: zero-electron
# spin channels must not publish the ambiguity flag as an energy.
# ---------------------------------------------------------------------------


def zero_electron_down_channel(*, chain: bool):
    """Collinear ISPIN = 2: spin up holds two electrons per k-point (VBM -0.5 eV
    at Gamma); spin down is empty, with its lowest band at 1.5 eV, or at
    1.2 eV on the ambiguous chain member 0.25016 when ``chain`` is set."""

    up, up_occ = flat_bands(vbm_points={A: -0.5}, cbm_points={})
    down = np.tile([1.5, 2.5, 3.5, 4.5], (16, 1))
    down_occ = np.zeros((16, 4))
    kpoints, weights = GRID, np.full(16, 1 / 16)
    if chain:
        up = np.vstack([up, [[-3.0, -1.0, 2.0, 4.0]] * 2])
        up_occ = np.vstack([up_occ, [[1.0, 1.0, 0.0, 0.0]] * 2])
        down = np.vstack([down, [[1.6, 2.5, 3.5, 4.5], [1.2, 2.5, 3.5, 4.5]]])
        down_occ = np.vstack([down_occ, np.zeros((2, 4))])
        kpoints, weights = np.vstack([GRID, CHAIN]), np.append(weights, [0.0, 0.0])
    return evidence((up, down), (up_occ, down_occ), efermi=0.0, weights=weights, kpoints=kpoints)


def energy_fields(payload, path=""):
    """Every (path, value) whose key names an electron-volt quantity."""

    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.endswith("_ev") and not isinstance(value, (dict, list)):
                yield f"{path}.{key}", value
            yield from energy_fields(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for position, value in enumerate(payload):
            yield from energy_fields(value, f"{path}[{position}]")


@pytest.mark.parametrize(
    ("chain", "gap", "cbm", "ambiguous"),
    ((False, 2.0, (0.0, 0.0, 0.0), False), (True, 1.7, (0.25, 0.0, 0.0), True)),
)
def test_zero_electron_collinear_channel_publishes_no_valence_quantities(chain, gap, cbm, ambiguous) -> None:
    result = assess(zero_electron_down_channel(chain=chain))
    up, down = result.evidence.channels

    assert result.classification == "nonmetallic"
    assert result.evidence.band_gap_ev == gap
    assert down.electron_count == 0.0
    assert down.vbm is None and down.band_gap_ev is None and down.gap_type is None
    assert down.min_direct_gap_ev is None
    assert down.min_direct_gap_lower_bound_ev is None
    assert down.cbm.kpoint == cbm
    assert down.kpoint_equivalence_ambiguous is ambiguous
    # Spin up's flat conduction band puts the chain in its CBM set too.
    assert up.kpoint_equivalence_ambiguous is ambiguous
    assert up.min_direct_gap_ev == up.min_direct_gap_lower_bound_ev == 2.5
    spin = serialize_electronic_character_assessment(result)["electronic_evidence"]["spin_channels"][1]
    assert spin["min_direct_gap_ev"] is None
    assert spin["min_direct_gap_lower_bound_ev"] is None
    assert spin["kpoint_equivalence_ambiguous"] is ambiguous


def test_channel_edges_cannot_be_constructed_positionally() -> None:
    with pytest.raises(TypeError):
        ec._ChannelEdges(None, None, None, None, True)  # type: ignore[misc]


@pytest.mark.parametrize("field", ("min_direct_gap", "min_direct_gap_lower_bound"))
@pytest.mark.parametrize("value", (True, False))
def test_channel_edges_reject_a_boolean_energy(field: str, value: bool) -> None:
    fields = {"vbm": None, "cbm": None, "gap_type": None, "min_direct_gap": None, field: value}
    with pytest.raises(TypeError, match=field):
        ec._ChannelEdges(**fields)


def test_channel_edges_reject_a_non_boolean_flag() -> None:
    with pytest.raises(TypeError):
        ec._ChannelEdges(vbm=None, cbm=None, gap_type=None, min_direct_gap=None, equivalence_ambiguous=1.0)


@pytest.mark.parametrize(
    "build",
    (
        lambda: zero_electron_down_channel(chain=False),
        lambda: zero_electron_down_channel(chain=True),
        lambda: chain_evidence(),
        lambda: chain_evidence(b=(-0.502, 3.0), c=(-1.0, 1.004)),
        lambda: gamma_vbm_with_zero_weight_cbm([0.00006, 0.0, 0.0]),
        lambda: evidence(*map(lambda x: (x,), gapped_channel()), efermi=0.0),
    ),
)
def test_energy_fields_are_never_booleans(build) -> None:
    payload = serialize_electronic_character_assessment(assess(build()))
    fields = list(energy_fields(payload))

    assert any(path.endswith("min_direct_gap_lower_bound_ev") for path, _ in fields)
    for path, value in fields:
        assert value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)), path
    assert json.loads(json.dumps(payload)) == payload
