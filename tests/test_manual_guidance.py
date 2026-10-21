"""R3: evidence-driven VASP Wiki guidance.

Two kinds of evidence are used here and are kept apart:

* the VASP Wiki corpus is the REAL frozen corpus v1.0.0 shipped with the
  package (1,047 pinned pages); retrieval is tested against its actual text;
* the calculation directories are SYNTHETIC: invented INCAR/OSZICAR/OUTCAR
  contents written to temporary directories (the repository contains no real
  OSZICAR/OUTCAR sets). The real pymatgen ``vasprun.xml`` fixtures are used
  only to show that they produce no unsupported findings.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import shutil
import socket
from types import SimpleNamespace

import pytest

from bmd_agent import cli
from bmd_agent import vasp_manual_index as manual
from bmd_agent import vasp_wiki_corpus as corpus
from bmd_agent.presentation import build_lifecycle_concise_summary, render_concise_summary
from bmd_agent.resources import manual_guidance as guidance
from bmd_agent.resources.assessment import build_lifecycle_completion_assessment, serialize_completion_assessment
from bmd_agent.resources.lifecycle import analyze_calculation_directory


PACKAGED = Path(corpus.__file__).parent / "reference_corpus" / "vasp_wiki"
REAL_VASPRUN_FIXTURES = sorted((Path(__file__).parent / "fixtures" / "electronic_character").glob("*.vasprun.xml.gz"))

# ---------------------------------------------------------------------------
# SYNTHETIC calculation directories
# ---------------------------------------------------------------------------

POSCAR = "Si\n1.0\n5.43 0 0\n0 5.43 0\n0 0 5.43\nSi\n2\nDirect\n0 0 0\n0.25 0.25 0.25\n"
KPOINTS = "Automatic mesh\n0\nGamma\n4 4 4\n0 0 0\n"
NORMAL_OUTCAR = " General timing and accounting informations for this job:\n Voluntary context switches\n"
HEADER = "       N       E                     dE             d eps       ncg     rms          rms(c)\n"


def _electronic_steps(count: int, *, converged: bool = False, label: str = "DAV") -> str:
    lines = []
    for step in range(1, count + 1):
        change = 1e-7 if (converged and step == count) else 0.5 * 10.0 ** (-(step % 3))
        lines.append(f"{label}:{step:4d}   -1.000000000000E+01   {change: .5E}   {change: .5E}   10   1.000E+00   2.000E-01\n")
    return "".join(lines)


def _forces(value: float) -> str:
    return (
        " POSITION                                       TOTAL-FORCE (eV/Angst)\n"
        " -----------------------------------------------------------------------------------\n"
        f"      0.00000000      0.00000000      0.00000000      {value:.8f}      0.00000000      0.00000000\n"
        f"      0.50000000      0.50000000      0.50000000      0.00000000      0.00000000      {value / 2:.8f}\n"
        " -----------------------------------------------------------------------------------\n"
    )


SCF_NOT_CONVERGED = HEADER + _electronic_steps(8) + "   1 F= -.10000000E+02 E0= -.10000000E+02  d E =-.100000E+02\n"
RELAX_TWO_STEPS = (
    HEADER + _electronic_steps(3, converged=True) + "   1 F= -.11000000E+02 E0= -.10950000E+02  d E =-.110000E+02\n"
    + HEADER + _electronic_steps(3, converged=True, label="RMM") + "   2 F= -.12100000E+02 E0= -.12050000E+02  d E =-.110000E+01\n"
)
INCAR_SCF = "ENCUT = 520\nEDIFF = 1E-6\nNELM = 8\nALGO = Fast\n"
INCAR_HYBRID = "ENCUT = 400\nEDIFF = 1E-6\nNELM = 8\nLHFCALC = .TRUE.\nHFSCREEN = 0.2\nALGO = Damped\nTIME = 0.4\n"
INCAR_DFTU = "ENCUT = 520\nEDIFF = 1E-6\nNELM = 8\nISPIN = 2\nMAGMOM = 2*5\nLDAU = .TRUE.\nLDAUTYPE = 2\nLDAUL = 2\nLDAUU = 5\nLMAXMIX = 4\nAMIX = 0.2\n"
INCAR_RELAX = "ENCUT = 520\nEDIFF = 1E-6\nEDIFFG = -0.02\nIBRION = 2\nISIF = 2\nNSW = 2\nPOTIM = 0.5\n"
OOM_LOG = (
    "slurmstepd: error: Detected 1 oom-kill event(s) in StepId=123.0. Some of your processes may have "
    "been killed by the cgroup out-of-memory handler.\n"
)
ZBRENT_LOG = " ZBRENT: fatal error in bracketing\n     please rerun with smaller EDIFF, or copy CONTCAR\n"


def write_calculation(directory: Path, incar: str, *, oszicar: str | None = None, outcar: str | None = None, **extra: str) -> Path:
    directory.mkdir(parents=True)
    (directory / "POSCAR").write_text(POSCAR)
    (directory / "KPOINTS").write_text(KPOINTS)
    (directory / "INCAR").write_text(incar)
    if oszicar is not None:
        (directory / "OSZICAR").write_text(oszicar)
    if outcar is not None:
        (directory / "OUTCAR").write_text(outcar)
    for name, text in extra.items():
        (directory / name.replace("__", ".")).write_text(text)
    return directory


SCENARIOS = {
    "electronic": lambda d: write_calculation(d, INCAR_SCF, oszicar=SCF_NOT_CONVERGED, outcar=NORMAL_OUTCAR),
    "hybrid": lambda d: write_calculation(d, INCAR_HYBRID, oszicar=SCF_NOT_CONVERGED, outcar=NORMAL_OUTCAR),
    "dftu_magnetic": lambda d: write_calculation(d, INCAR_DFTU, oszicar=SCF_NOT_CONVERGED, outcar=NORMAL_OUTCAR),
    "relaxation": lambda d: write_calculation(d, INCAR_RELAX, oszicar=RELAX_TWO_STEPS, outcar=_forces(3.0) + _forces(0.5) + NORMAL_OUTCAR),
    "memory": lambda d: write_calculation(
        d, "ENCUT = 520\nEDIFF = 1E-6\nNCORE = 4\nKPAR = 2\n", oszicar=HEADER + _electronic_steps(2), outcar="partial\n", std_err__txt=OOM_LOG
    ),
    "zbrent": lambda d: write_calculation(
        d,
        "ENCUT = 520\nEDIFF = 1E-6\nEDIFFG = -0.02\nIBRION = 2\nNSW = 50\n",
        oszicar=HEADER + _electronic_steps(3, converged=True) + "   1 F= -.11000000E+02 E0= -.10950000E+02  d E =-.110000E+02\n",
        outcar=_forces(1.0),
        vasp__out=ZBRENT_LOG,
    ),
}


@pytest.fixture(scope="module")
def index() -> manual.ManualIndex:
    loaded = manual.load_manual_index()
    assert isinstance(loaded, manual.ManualIndex), loaded
    return loaded


def run_guidance(tmp_path: Path, name: str) -> tuple[object, guidance.ManualGuidance]:
    analysis = analyze_calculation_directory(SCENARIOS[name](tmp_path / name))
    return analysis, guidance.build_manual_guidance(analysis)


def titles(item: guidance.FindingGuidance) -> list[str]:
    return [reference.title for reference in item.references]


# ---------------------------------------------------------------------------
# The index covers the whole frozen corpus and keeps provenance
# ---------------------------------------------------------------------------


def test_index_covers_every_preserved_page_with_pinned_provenance(index: manual.ManualIndex) -> None:
    loaded = corpus.load_vasp_wiki_corpus()
    assert index.pages == len(loaded.pages) == 1047
    assert index.corpus_version == "1.0.0" and index.corpus_digest == loaded.corpus_digest
    pages = {page.id: page for page in loaded.pages}
    covered = {entry.passage.page_id for entry in index.entries}
    assert len(covered) >= 1000  # nearly every page yields at least one passage
    assert index.passages > 5000
    for entry in index.entries:
        page = pages[entry.passage.page_id]
        assert entry.passage.revision_id == page.revision_id
        assert entry.passage.permalink == page.permalink and f"oldid={page.revision_id}" in page.permalink
        assert entry.passage.title == page.title


def test_index_is_deterministic_and_bound_to_the_corpus_identity(index: manual.ManualIndex) -> None:
    rebuilt = manual.build_manual_index(corpus.load_vasp_wiki_corpus())
    assert rebuilt.index_identity == index.index_identity
    expected = hashlib.sha256(f"{manual.INDEX_ALGORITHM}\n{index.corpus_digest}".encode("ascii")).hexdigest()
    assert index.index_identity == f"sha256:{expected}"
    query = {"memory": 3.0, "NCORE": 1.0}
    assert rebuilt.search(query, required=("memory",)) == index.search(query, required=("memory",))
    assert [entry.passage for entry in rebuilt.entries] == [entry.passage for entry in index.entries]


def test_wikitext_markup_is_simplified_without_inventing_content() -> None:
    raw = (
        "Set {{TAG|NELM}} and {{TAG|ALGO|Fast}}; see [[ALGO|the ALGO page]] and [[Category:INCAR tag]]"
        " {{Video|x|y}} [https://example.org external] &Aring; '''bold''' <b>tag</b>"
    )
    text = manual.simplify_wikitext(raw)
    assert text == "Set NELM and ALGO = Fast; see the ALGO page and external Å bold tag"


def test_input_listings_are_not_treated_as_explanatory_prose() -> None:
    assert manual.passage_kind("ALGO = Fast\nNELM = 60\nEDIFF = 1E-6", None) == "listing"
    assert manual.passage_kind("NCORE, KPAR, LREAL, ALGO, NBANDS", None) == "listing"
    assert manual.passage_kind("Anything at all.", "Related tags and articles") == "listing"
    assert manual.passage_kind("If the loop does not converge within forty steps it will probably not converge.", None) == "prose"


def test_required_terms_keep_common_vocabulary_from_promoting_unrelated_pages(index: manual.ManualIndex) -> None:
    generic = index.search({"convergence": 1.0, "calculation": 1.0, "ZBRENT": 1.0}, required=("ZBRENT",))
    assert [item.passage.title for item in generic] == ["Structure optimization"]
    assert index.search({"convergence": 1.0}, required=("NOSUCHVASPTOKEN",)) == ()


# ---------------------------------------------------------------------------
# Findings come only from existing assessments
# ---------------------------------------------------------------------------


def test_successful_or_quiet_evidence_yields_no_findings_and_never_loads_the_corpus(tmp_path: Path) -> None:
    directory = write_calculation(
        tmp_path / "ok",
        "ENCUT = 520\nEDIFF = 1E-6\nNELM = 60\n",
        oszicar=HEADER + _electronic_steps(3, converged=True) + "   1 F= -.10800000E+02 E0= -.10800000E+02  d E =-.108000E+02\n",
        outcar=NORMAL_OUTCAR,
    )
    calls: list[int] = []
    result = guidance.build_manual_guidance(
        analyze_calculation_directory(directory), index_loader=lambda: calls.append(1)
    )
    assert result.status == guidance.GUIDANCE_NO_FINDINGS and not result.findings and not calls


@pytest.mark.parametrize(
    ("name", "kind", "observed"),
    [
        ("electronic", guidance.ELECTRONIC_NOT_CONVERGED, "did not reach EDIFF = 1e-06 after using all NELM = 8"),
        ("relaxation", guidance.IONIC_NOT_CONVERGED, "has not reached EDIFFG = -0.02 after all NSW = 2 ionic steps"),
        ("memory", guidance.MEMORY_EXHAUSTION, "establishes that the job ran out of memory"),
        ("zbrent", guidance.VASP_ERROR_MESSAGE, "ZBRENT: fatal error in bracketing"),
    ],
)
def test_findings_restate_the_existing_evidence(tmp_path: Path, name: str, kind: str, observed: str) -> None:
    findings = guidance.derive_findings(analyze_calculation_directory(SCENARIOS[name](tmp_path / name)))
    assert findings[0].kind == kind
    assert observed in findings[0].observed


def test_findings_never_change_c1_completion_or_the_analysis(tmp_path: Path) -> None:
    analysis = analyze_calculation_directory(SCENARIOS["relaxation"](tmp_path / "r"))
    before = serialize_completion_assessment(build_lifecycle_completion_assessment(analysis))
    snapshot = repr(analysis)
    guidance.build_manual_guidance(analysis)
    assert repr(analysis) == snapshot
    assert serialize_completion_assessment(build_lifecycle_completion_assessment(analysis)) == before


def test_c2_smearing_uncertainty_becomes_a_finding_without_changing_the_verdict() -> None:
    channel = SimpleNamespace(channel="total", reason_code="GAP_COMPARABLE_TO_SMEARING")
    assessment = SimpleNamespace(
        classification="uncertain",
        evidence=SimpleNamespace(channels=(channel,), ismear=0, sigma_ev=0.2),
        stage=SimpleNamespace(label="static"),
    )
    analysis = SimpleNamespace(diagnostics=None, stage_completion=(), incar_settings={})
    (finding,) = guidance.derive_findings(analysis, electronic_character=assessment)
    assert finding.kind == guidance.SMEARING_QUESTIONABLE
    assert "uncertain" in finding.observed and "comparable to the smearing width" in finding.observed
    assert assessment.classification == "uncertain"


@pytest.mark.parametrize("fixture", REAL_VASPRUN_FIXTURES, ids=lambda path: path.name.split(".")[0])
def test_real_vasprun_fixtures_produce_no_unsupported_findings(tmp_path: Path, fixture: Path) -> None:
    # REAL pymatgen VASP outputs (vasprun.xml only). Without OSZICAR/OUTCAR the
    # existing assessments establish no failure, so R3 must add nothing.
    (tmp_path / "vasprun.xml").write_bytes(gzip.decompress(fixture.read_bytes()))
    analysis = analyze_calculation_directory(tmp_path)
    assert guidance.derive_findings(analysis) == ()
    assert cli.lifecycle_manual_guidance(analysis).status == guidance.GUIDANCE_NO_FINDINGS


# ---------------------------------------------------------------------------
# Retrieval relevance against the real corpus
# ---------------------------------------------------------------------------


def test_electronic_non_convergence_cites_electronic_convergence_guidance(tmp_path: Path) -> None:
    _, result = run_guidance(tmp_path, "electronic")
    (item,) = result.findings
    assert titles(item)[0] == "Troubleshooting electronic convergence"
    assert "NELM" in titles(item) or "ALGO" in titles(item)


def test_hybrid_context_adds_hybrid_specific_guidance(tmp_path: Path) -> None:
    _, result = run_guidance(tmp_path, "hybrid")
    item = result.findings[0]
    assert "LHFCALC" in titles(item)
    lhfcalc = next(reference for reference in item.references if reference.title == "LHFCALC")
    assert "reduce the timestep TIME" in lhfcalc.excerpt


def test_dft_u_magnetic_context_cites_the_magnetic_troubleshooting_section(tmp_path: Path) -> None:
    _, result = run_guidance(tmp_path, "dftu_magnetic")
    item = result.findings[0]
    sections = [(reference.title, reference.section) for reference in item.references]
    assert ("Troubleshooting electronic convergence", "Magnetic calculations") in sections


def test_ionic_relaxation_failure_cites_relaxation_guidance(tmp_path: Path) -> None:
    _, result = run_guidance(tmp_path, "relaxation")
    item = result.findings[0]
    assert {"EDIFFG", "Structure optimization"} <= set(titles(item))
    # EDIFFG = -0.02 is a force criterion: the explanation for negative EDIFFG is cited.
    assert item.references[0].excerpt.startswith("When EDIFFG is negative, the relaxation is stopped when the norms of all the forces")


def test_memory_exhaustion_cites_the_memory_pages(tmp_path: Path) -> None:
    _, result = run_guidance(tmp_path, "memory")
    item = result.findings[0]
    assert titles(item)[0] == "Not enough memory"
    assert item.linked_settings == ("NCORE", "KPAR")


def test_vasp_error_message_cites_only_passages_that_mention_it(tmp_path: Path) -> None:
    _, result = run_guidance(tmp_path, "zbrent")
    item = result.findings[0]
    assert titles(item) == ["Structure optimization"]
    assert "ZBRENT" in item.references[0].excerpt


def test_unknown_error_messages_are_reported_without_references() -> None:
    finding = guidance.DiagnosticFinding(
        guidance.VASP_ERROR_MESSAGE,
        observed="vasp.out contains the message: QWXYZ failed",
        source="bounded log excerpt",
        signals=(guidance.EvidenceSignal("QWXYZ", None, "message"),),
    )
    (item,) = guidance.guidance_for_findings((finding,), manual.load_manual_index())
    assert item.references == () and item.next_step is None
    assert item.limitation == "No passage of the frozen VASP Wiki mentions this message."


# ---------------------------------------------------------------------------
# Method-specific passages are filtered by their content
# ---------------------------------------------------------------------------

# Test-side identifiers of methods, written independently of the module's
# vocabulary; an excerpt or passage cited for a finding whose stage does not
# use the method must contain none of them.
UNUSED_METHOD_IDENTIFIERS = {
    "hybrid_functional": ("LHFCALC", "HFSCREEN", "AEXX", "HSE06", "PBE0", "Hartree-Fock"),
    "dft_plus_u": ("LDAU", "LDAUU", "DFT+U", "LDA+U"),
    "gw_rpa_bse": ("GW", "EVGW0", "ACFDT", "RPA", "BSE", "quasiparticle"),
    "machine_learned_force_field": ("ML_LMLFF", "ML_MODE", "MLFF", "machine-learned", "machine learned"),
    "phonons_linear_response": ("DFPT", "LEPSILON", "phonon"),
    "molecular_dynamics": ("MDALGO", "thermostat"),
    "optics": ("LOPTICS", "CSHIFT"),
    "nmr": ("LCHIMAG", "NMR"),
    "nudged_elastic_band": ("IMAGES", "LCLIMB", "nudged elastic band"),
    "magnetism_spin_orbit": ("MAGMOM", "LNONCOLLINEAR", "LSORBIT", "AMIX_MAG", "SAXIS"),
    "meta_gga": ("METAGGA", "r2SCAN", "MBJ"),
}


def _contains_identifier(text: str, identifier: str) -> bool:
    import re

    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(identifier) + r"(?![A-Za-z0-9_])", text) is not None


def cited_passage(index: manual.ManualIndex, reference: guidance.ManualReference) -> manual.Passage:
    return next(
        entry.passage
        for entry in index.entries
        if entry.passage.page_id == reference.page_id and entry.passage.position == reference.passage_position
    )


def assert_no_unused_method_content(index: manual.ManualIndex, item: guidance.FindingGuidance) -> None:
    active = guidance._finding_domains(item.finding)
    for reference in item.references:
        passage = cited_passage(index, reference)
        for domain, identifiers in UNUSED_METHOD_IDENTIFIERS.items():
            if domain in active:
                continue
            for identifier in identifiers:
                assert not _contains_identifier(reference.excerpt, identifier), (domain, reference.title, reference.excerpt)
                assert not _contains_identifier(passage.text, identifier), (domain, reference.title, passage.text)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_cited_excerpts_and_passages_never_discuss_methods_the_stage_does_not_use(
    tmp_path: Path, name: str, index: manual.ManualIndex
) -> None:
    _, result = run_guidance(tmp_path, name)
    assert result.status == guidance.GUIDANCE_AVAILABLE
    for item in result.findings:
        assert_no_unused_method_content(index, item)
        active = guidance._finding_domains(item.finding)
        for reference in item.references:
            passage = cited_passage(index, reference)
            assert guidance.passage_method_domains(passage.text, title=passage.title, heading=passage.section) <= active


def test_out_of_memory_case_does_not_cite_the_gw_and_mlff_shared_memory_passage(
    tmp_path: Path, index: manual.ManualIndex, capsys
) -> None:
    gw_mlff = next(
        entry.passage
        for entry in index.entries
        if entry.passage.title == "Shared memory" and "memory demands of GW" in entry.passage.text
    )
    assert guidance.passage_method_domains(gw_mlff.text, title=gw_mlff.title, heading=gw_mlff.section) == {
        "gw_rpa_bse",
        "machine_learned_force_field",
    }
    hybrid_only = next(
        entry.passage
        for entry in index.entries
        if entry.passage.title == "Shared memory" and "hybrid-functional calculations" in entry.passage.text
    )
    assert guidance.passage_method_domains(hybrid_only.text, title=hybrid_only.title) == {"hybrid_functional"}

    _, result = run_guidance(tmp_path, "memory")
    (item,) = result.findings
    assert guidance._finding_domains(item.finding) == set()
    cited = [(reference.page_id, reference.passage_position) for reference in item.references]
    assert (gw_mlff.page_id, gw_mlff.position) not in cited
    assert (hybrid_only.page_id, hybrid_only.position) not in cited
    for reference in item.references:
        for word in ("GW", "EVGW0", "machine-learned", "hybrid-functional"):
            assert not _contains_identifier(reference.excerpt, word), reference.excerpt
    # General guidance on a page that also discusses GW, MLFF and hybrid
    # functionals is still usable: the general shared-memory passage remains
    # eligible for this conventional out-of-memory case.
    general = next(
        entry.passage for entry in index.entries if entry.passage.title == "Shared memory" and entry.passage.position == 0
    )
    assert "NCSHMEM" in general.text
    assert guidance.passage_method_domains(general.text, title=general.title, heading=general.section) == frozenset()

    # Both outputs are filtered: neither mentions the excluded methods.
    directory = SCENARIOS["memory"](tmp_path / "cli")
    for arguments in ([str(directory)], [str(directory), "--verbose"]):
        assert cli.main(arguments) == 0
        out = capsys.readouterr().out
        section = out.split("VASP manual guidance", 1)[1]
        assert "memory demands of GW" not in section and "machine-learned" not in section
        assert "hybrid-functional calculations" not in section


def test_general_guidance_with_a_passing_method_mention_is_preserved(tmp_path: Path, index: manual.ManualIndex) -> None:
    _, result = run_guidance(tmp_path, "electronic")
    (item,) = result.findings
    first = item.references[0]
    assert (first.title, first.section) == ("Troubleshooting electronic convergence", None)
    passage = cited_passage(index, first)
    assert "magnetic systems" in passage.text  # one passing mention in general guidance
    assert guidance.passage_method_domains(passage.text, title=passage.title, heading=passage.section) == frozenset()
    assert first.excerpt.startswith("There can be many reasons why convergence to the electronic ground state fails.")


def test_method_passages_become_eligible_only_when_the_stage_uses_the_method(index: manual.ManualIndex) -> None:
    gw_mlff = next(
        entry.passage
        for entry in index.entries
        if entry.passage.title == "Shared memory" and "memory demands of GW" in entry.passage.text
    )
    base = dict(kind=guidance.MEMORY_EXHAUSTION, observed="out of memory", source="memory (OOM) evidence")
    conventional = guidance.DiagnosticFinding(**base, stage_settings={"ENCUT": 520, "NCORE": 4})
    gw_stage = guidance.DiagnosticFinding(**base, stage_settings={"ALGO": "EVGW0", "ML_MODE": "train"})
    domains = guidance.passage_method_domains(gw_mlff.text, title=gw_mlff.title, heading=gw_mlff.section)
    assert not domains <= guidance._finding_domains(conventional)
    assert domains <= guidance._finding_domains(gw_stage)
    # A quoted log message that itself names a method makes that method's passages eligible.
    message = guidance.DiagnosticFinding(
        guidance.VASP_ERROR_MESSAGE, observed="x", source="log", message="ERROR: LHFCALC requires ALGO = Damped"
    )
    assert "hybrid_functional" in guidance._finding_domains(message)


def test_method_domains_are_read_from_content_not_page_titles() -> None:
    domains = guidance.passage_method_domains
    assert domains("Increase NCORE to reduce the memory per core.", title="Not enough memory") == frozenset()
    assert domains("Use ALGO = Damped with LHFCALC = .TRUE.", title="ALGO") == {"hybrid_functional"}
    assert domains("Set LDAUU and LDAUJ for each species.", title="Practical guide") == {"dft_plus_u"}
    assert domains("The ML_LMLFF tag switches on training.", title="Anything") == {"machine_learned_force_field"}
    # One passing description among several sentences is not enough ...
    general = "There are many reasons for slow convergence. Charged or magnetic systems are discussed below. Start simple."
    assert domains(general) == frozenset()
    # ... but two descriptions, or a single-sentence passage about the method, are.
    assert domains("Magnetic systems converge slowly. Magnetization mixing helps.") == {"magnetism_spin_orbit"}
    assert domains("Use shared-memory segments in hybrid-functional calculations.") == {"hybrid_functional"}
    # A section heading about a method marks its passages as method-specific.
    assert domains("Reduce the time step.", heading="Hybrid functionals") == {"hybrid_functional"}
    # Lower-case words that merely resemble identifiers are not identifiers.
    assert domains("Scan the lattice constant and plot the energy.") == frozenset()


# ---------------------------------------------------------------------------
# Multistage workflows: each finding uses its own stage's settings only
# ---------------------------------------------------------------------------

PBE_RELAX_INCAR = "ENCUT = 520\nEDIFF = 1E-6\nNELM = 8\nIBRION = 2\nISIF = 3\nNSW = 20\nEDIFFG = -0.02\n"
HSE_STATIC_INCAR = "ENCUT = 400\nEDIFF = 1E-8\nNELM = 60\nALGO = Damped\nLHFCALC = .TRUE.\nHFSCREEN = 0.2\nTIME = 0.4\n"
HSE_STAGE_VALUES = {"NELM": "60", "EDIFF": "1e-08", "ALGO": "Damped", "LHFCALC": ".TRUE.", "HFSCREEN": "0.2", "TIME": "0.4"}


def write_two_stage_workflow(
    root: Path,
    *,
    stage_1_incar: str | None = PBE_RELAX_INCAR,
    stage_2_oszicar: str | None = None,
    root_incar: str | None = HSE_STATIC_INCAR,
    **stage_1_extra: str,
) -> tuple[Path, Path, Path]:
    """SYNTHETIC BMD workflow: stage 1 is a PBE relaxation whose electronic
    loop fails with NELM = 8; stage 2 is an HSE06 static calculation."""

    import json

    stage_1 = write_calculation(
        root / "stage_01", stage_1_incar or "", oszicar=SCF_NOT_CONVERGED, outcar="partial\n", **stage_1_extra
    )
    if stage_1_incar is None:
        (stage_1 / "INCAR").unlink()
    stage_2 = write_calculation(
        root / "stage_02",
        HSE_STATIC_INCAR,
        oszicar=stage_2_oszicar if stage_2_oszicar is not None else HEADER + _electronic_steps(2),
        outcar="partial\n",
    )
    if root_incar is not None:
        (root / "INCAR").write_text(root_incar)
    (root / "attempt.json").write_text(json.dumps({"attempt_id": "a", "state": "SUBMITTED", "job_id": "4242"}))
    payload = {
        "flow_spec": {
            "workflow_spec": {
                "stages": [
                    {"stage_type": "relax", "theory": "pbe", "modifiers": [], "options": {}},
                    {"stage_type": "static", "theory": "hse06", "modifiers": [], "options": {}},
                ]
            }
        },
        "submission": {"attempt_id": "a", "attempt_state": str(root / "attempt.json")},
        "paths": {"stage_dirs": {"stage_01": str(stage_1), "stage_02": str(stage_2)}, "result_dir": str(stage_2)},
    }
    (root / "submission.json").write_text(json.dumps(payload))
    return root, stage_1, stage_2


def analyze_workflow(target: Path, *, state: str = "FAILED"):
    from bmd_agent.resources.slurm import SlurmAccountingRecord

    def lookup(job_id: str) -> SlurmAccountingRecord:
        return SlurmAccountingRecord(
            job_id=job_id, name="vasp", state=state, elapsed="00:10:00", start="2026-09-16T10:00:00", end="",
            partition="p", exit_code="1:0", timelimit="01:00:00", node_list="n", allocated_cpus=24, work_dir="/unused",
        )

    return analyze_calculation_directory(target, scheduler_lookup=lookup)


def assert_no_stage_2_settings(finding: guidance.DiagnosticFinding) -> None:
    values = {signal.name: signal.value for signal in finding.signals}
    for name in ("LHFCALC", "HFSCREEN", "ALGO", "TIME"):
        assert name not in values and name not in finding.stage_settings, name
    assert values.get("NELM") in (None, "8") and values.get("EDIFF") in (None, "1e-06")
    assert "60" not in finding.observed and "1e-08" not in finding.observed
    assert not any(signal.role == "context" for signal in finding.signals)


@pytest.mark.parametrize("analyzed", ["workflow root", "stage 2 directory"])
def test_stage_1_failure_uses_stage_1_settings_not_the_later_hse_stage(
    tmp_path: Path, analyzed: str, index: manual.ManualIndex
) -> None:
    root, _, stage_2 = write_two_stage_workflow(tmp_path / "flow")
    analysis = analyze_workflow(root if analyzed == "workflow root" else stage_2)
    # The analyzed directory's INCAR is the HSE one: it must not be used.
    assert analysis.incar_settings.get("LHFCALC") is True
    result = guidance.build_manual_guidance(analysis)
    assert result.status == guidance.GUIDANCE_AVAILABLE
    (item,) = [item for item in result.findings if item.finding.kind == guidance.ELECTRONIC_NOT_CONVERGED]
    finding = item.finding
    assert (finding.stage_index, finding.stage_label) == (1, "stage_01")
    assert finding.observed == (
        "Stage 1 (stage_01): the final completed electronic cycle did not reach EDIFF = 1e-06 "
        "after using all NELM = 8 electronic iterations."
    )
    assert [(signal.name, signal.value) for signal in finding.signals] == [("NELM", "8"), ("EDIFF", "1e-06")]
    assert finding.stage_settings["NELM"] == 8 and finding.stage_settings["IBRION"] == 2
    assert finding.incar_available and finding.settings_source == "retained INCAR and trajectory criteria of stage 1 (stage_01)"
    assert_no_stage_2_settings(finding)
    assert guidance._finding_domains(finding) == set()
    assert item.references
    for reference in item.references:
        assert reference.title not in ("LHFCALC", "HFSCREEN", "Hybrid functionals: formalism")
        for word in ("LHFCALC", "HFSCREEN", "HSE06", "hybrid", "Hartree-Fock"):
            assert not _contains_identifier(reference.excerpt, word), reference.excerpt
    assert_no_unused_method_content(index, item)
    assert all(name not in ("ALGO", "TIME", "LHFCALC") for name in item.linked_settings)


def test_each_stage_finding_gets_its_own_method_guidance(tmp_path: Path, index: manual.ManualIndex) -> None:
    hse_not_converged = HEADER + _electronic_steps(60) + "   1 F= -.10000000E+02 E0= -.10000000E+02  d E =-.100000E+02\n"
    root, _, _ = write_two_stage_workflow(tmp_path / "flow", stage_2_oszicar=hse_not_converged)
    result = guidance.build_manual_guidance(analyze_workflow(root))
    by_stage = {
        item.finding.stage_index: item for item in result.findings if item.finding.kind == guidance.ELECTRONIC_NOT_CONVERGED
    }
    assert set(by_stage) == {1, 2}
    pbe, hse = by_stage[1], by_stage[2]
    assert_no_stage_2_settings(pbe.finding)
    assert guidance._finding_domains(pbe.finding) == set()
    assert_no_unused_method_content(index, pbe)
    hse_values = {signal.name: signal.value for signal in hse.finding.signals}
    assert {name: hse_values.get(name) for name in HSE_STAGE_VALUES} == HSE_STAGE_VALUES
    assert "NELM = 60" in hse.finding.observed and "EDIFF = 1e-08" in hse.finding.observed
    assert guidance._finding_domains(hse.finding) == {"hybrid_functional"}
    assert "LHFCALC" in [reference.title for reference in hse.references]
    assert "LHFCALC" not in [reference.title for reference in pbe.references]


def test_missing_stage_incar_omits_settings_instead_of_borrowing_them(tmp_path: Path, index: manual.ManualIndex) -> None:
    # Stage 1 has no retained INCAR; stage 2 and the workflow root have HSE ones.
    root, _, _ = write_two_stage_workflow(tmp_path / "flow", stage_1_incar=None, vasp__out=ZBRENT_LOG)
    analysis = analyze_workflow(root)
    inputs = {item.stage_index: item for item in analysis.diagnostics.executed_inputs}
    assert inputs[1].present is False and inputs[2].values["LHFCALC"] is True
    result = guidance.build_manual_guidance(analysis)
    # Without stage 1's NELM and EDIFF no electronic-convergence finding is
    # established for it, and none is invented from stage 2's criteria.
    assert not [item for item in result.findings if item.finding.kind == guidance.ELECTRONIC_NOT_CONVERGED]
    (item,) = [item for item in result.findings if item.finding.kind == guidance.VASP_ERROR_MESSAGE]
    finding = item.finding
    assert (finding.stage_index, finding.stage_label) == (1, "stage_01")
    assert finding.stage_settings == {} and finding.incar_available is False
    assert finding.settings_source == (
        "the executed INCAR of stage 1 (stage_01) is not available, so its settings are omitted"
    )
    assert guidance._finding_domains(finding) == set()
    assert_no_unused_method_content(index, item)


def test_stage_settings_use_only_that_stages_evidence() -> None:
    from bmd_agent.resources.run import IncarObservation

    diagnostics = SimpleNamespace(
        executed_inputs=(
            IncarObservation("stage_01", "/w/stage_01/INCAR", False, 1),
            IncarObservation("stage_02", "/w/stage_02/INCAR", True, 2, values={"LHFCALC": True, "NELM": 60}),
            IncarObservation("stage_03", "/w/stage_03/INCAR", True, 3, values={}, error="unparsable line"),
        ),
        trajectories=(
            SimpleNamespace(stage_index=1, stage_label="stage_01", criteria={"NELM": 8, "EDIFF": 1e-6}),
            SimpleNamespace(stage_index=2, stage_label="stage_02", criteria={"NELM": 40}),
            SimpleNamespace(stage_index=3, stage_label="stage_03", criteria={}),
        ),
    )
    first = guidance._stage_settings(diagnostics, 1, "stage_01")
    assert first.values == {"NELM": 8, "EDIFF": 1e-6} and not first.incar_available
    assert first.source == (
        "trajectory criteria of stage 1 (stage_01); the executed INCAR of stage 1 (stage_01) is not available, "
        "so its settings are omitted"
    )
    second = guidance._stage_settings(diagnostics, 2, "stage_02")
    assert second.values == {"LHFCALC": True, "NELM": 40}  # the trajectory criteria are authoritative
    third = guidance._stage_settings(diagnostics, 3, "stage_03")
    assert third.values == {} and not third.incar_available
    # A stage without evidence, or no stage at all, has no settings.
    assert guidance._stage_settings(diagnostics, 4, "stage_04").values == {}
    assert guidance._stage_settings(diagnostics, None, None).values == {}


def test_logs_outside_every_stage_directory_of_a_multistage_workflow_carry_no_stage_settings(tmp_path: Path) -> None:
    root, _, _ = write_two_stage_workflow(tmp_path / "flow")
    (root / "vasp.out").write_text(ZBRENT_LOG)
    result = guidance.build_manual_guidance(analyze_workflow(root))
    (item,) = [item for item in result.findings if item.finding.kind == guidance.VASP_ERROR_MESSAGE]
    assert item.finding.stage_index is None and item.finding.stage_settings == {}
    assert item.finding.settings_source.startswith("no single stage could be attributed")
    assert guidance._finding_domains(item.finding) == set()


def test_verbose_output_states_which_stage_settings_were_considered(tmp_path: Path, capsys) -> None:
    root, _, _ = write_two_stage_workflow(tmp_path / "flow")
    from bmd_agent.cli import print_manual_guidance

    print_manual_guidance(guidance.build_manual_guidance(analyze_workflow(root)))
    out = capsys.readouterr().out
    assert "settings considered: retained INCAR and trajectory criteria of stage 1 (stage_01)" in out
    assert "LHFCALC" not in out and "HFSCREEN" not in out


# ---------------------------------------------------------------------------
# Grounding: excerpts are verbatim, next steps never invent settings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_excerpt_is_verbatim_from_the_cited_pinned_revision(tmp_path: Path, name: str) -> None:
    _, result = run_guidance(tmp_path, name)
    pages = {page.id: page for page in corpus.load_vasp_wiki_corpus().pages}
    assert result.status == guidance.GUIDANCE_AVAILABLE
    for item in result.findings:
        for reference in item.references:
            page = pages[reference.page_id]
            assert reference.revision_id == page.revision_id and reference.permalink == page.permalink
            source = " ".join(manual.simplify_wikitext(page.read_wikitext()).split())
            excerpt = reference.excerpt.removesuffix(" ...")
            for sentence in (part for part in excerpt.split(". ") if part):
                assert sentence.rstrip(".") in source, (reference.title, sentence)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_next_steps_name_only_settings_observed_here_and_named_in_the_cited_passage(tmp_path: Path, name: str) -> None:
    _, result = run_guidance(tmp_path, name)
    index = manual.load_manual_index()
    for item in result.findings:
        observed = {signal.name for signal in item.finding.signals}
        first = None
        if item.references:
            cited = item.references[0]
            first = next(
                entry.passage
                for entry in index.entries
                if entry.passage.page_id == cited.page_id and entry.passage.position == cited.passage_position
            )
        for name_ in item.linked_settings:
            assert name_ in observed
            assert first is not None and name_ in first.text
        if item.next_step:
            assert "set " not in item.next_step.lower() and "increase" not in item.next_step.lower()
            assert "not established" in item.next_step or "Compare" in item.next_step or "Read" in item.next_step


# ---------------------------------------------------------------------------
# Student CLI integration
# ---------------------------------------------------------------------------


def test_default_output_adds_one_concise_cited_diagnosis(tmp_path: Path, capsys) -> None:
    directory = SCENARIOS["electronic"](tmp_path / "scf")
    assert cli.main([str(directory)]) == 0
    out = capsys.readouterr().out
    section = out.split("VASP manual guidance\n--------------------\n", 1)[1].split("\nDetailed evidence:", 1)[0]
    assert section.count("Observed:") == 1 and section.count("VASP Wiki (") == 1 and section.count("Next step:") == 1
    assert 'VASP Wiki (Troubleshooting electronic convergence): "' in section
    assert "frozen VASP Wiki corpus 1.0.0" in section
    assert len(section.strip().splitlines()) <= 14
    assert "Assessment" in out  # existing sections remain


def test_verbose_output_lists_full_citations(tmp_path: Path, capsys) -> None:
    directory = SCENARIOS["hybrid"](tmp_path / "hse")
    assert cli.main([str(directory), "--verbose"]) == 0
    out = capsys.readouterr().out
    assert "VASP manual guidance (vasp_manual_guidance):" in out
    assert "corpus: VASP Wiki 1.0.0 (sha256:f7929354612f779f8ff3a921b1a3901426fcdaced1cf3d7604d87b78472fcfef)" in out
    assert "permanent URL: https://vasp.at/wiki/index.php?title=LHFCALC&oldid=" in out
    assert "observed (BMD Check, electronic convergence progress assessment)" in out


def test_successful_calculation_output_is_unchanged(tmp_path: Path, capsys) -> None:
    directory = write_calculation(
        tmp_path / "ok",
        "ENCUT = 520\nEDIFF = 1E-6\nNELM = 60\n",
        oszicar=HEADER + _electronic_steps(3, converged=True) + "   1 F= -.10800000E+02 E0= -.10800000E+02  d E =-.108000E+02\n",
        outcar=NORMAL_OUTCAR,
    )
    analysis = analyze_calculation_directory(directory)
    expected = render_concise_summary(
        build_lifecycle_concise_summary(analysis, detailed_evidence_command=f"bmd-agent {directory} --verbose")
    )
    assert cli.main([str(directory)]) == 0
    assert capsys.readouterr().out == expected
    assert "VASP manual guidance" not in expected


def test_unavailable_corpus_keeps_the_scientific_checks_and_says_so(tmp_path: Path, monkeypatch, capsys) -> None:
    damaged = tmp_path / "corpus"
    shutil.copytree(PACKAGED, damaged)
    page = sorted((damaged / "pages").iterdir())[0]
    page.write_bytes(page.read_bytes() + b"tampered")
    unavailable = manual.load_manual_index(damaged)
    assert isinstance(unavailable, manual.ManualIndexUnavailable) and unavailable.reason == corpus.UNAVAILABLE_INTEGRITY

    monkeypatch.setattr(manual, "load_manual_index", lambda: unavailable)
    directory = SCENARIOS["electronic"](tmp_path / "scf")
    assert cli.main([str(directory)]) == 0
    out = capsys.readouterr().out
    assert "VASP manual references are unavailable (integrity); the findings above do not depend on them." in " ".join(out.split())
    assert "What happened" in out and "Assessment" in out
    assert 'VASP Wiki (' not in out


def test_guidance_failure_never_breaks_the_report(tmp_path: Path, monkeypatch, capsys) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(guidance, "derive_findings", broken)
    directory = SCENARIOS["electronic"](tmp_path / "scf")
    assert cli.main([str(directory)]) == 0
    out = capsys.readouterr().out
    assert "Assessment" in out and "VASP manual guidance" not in out


def test_cli_guidance_is_offline_read_only_and_deterministic(tmp_path: Path, monkeypatch, capsys) -> None:
    directory = SCENARIOS["dftu_magnetic"](tmp_path / "u")
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    corpus_before = {path: path.read_bytes() for path in PACKAGED.rglob("*") if path.is_file()}

    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    manual.clear_cache()
    assert cli.main([str(directory)]) == 0
    first = capsys.readouterr().out
    manual.clear_cache()
    assert cli.main([str(directory)]) == 0
    assert capsys.readouterr().out == first
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    assert {path: path.read_bytes() for path in PACKAGED.rglob("*") if path.is_file()} == corpus_before


# ---------------------------------------------------------------------------
# Further finding types
# ---------------------------------------------------------------------------


def test_relaxation_whose_scf_steps_hit_nelm_reports_both_findings(tmp_path: Path) -> None:
    # SYNTHETIC: ionic step 1 used all NELM = 3 iterations without reaching EDIFF,
    # step 2 converged electronically, and the final force misses EDIFFG.
    oszicar = (
        HEADER + _electronic_steps(3) + "   1 F= -.11000000E+02 E0= -.10950000E+02  d E =-.110000E+02\n"
        + HEADER + _electronic_steps(2, converged=True, label="RMM") + "   2 F= -.12100000E+02 E0= -.12050000E+02  d E =-.110000E+01\n"
    )
    directory = write_calculation(
        tmp_path / "relax",
        "ENCUT = 520\nEDIFF = 1E-6\nEDIFFG = -0.02\nIBRION = 2\nISIF = 2\nNSW = 2\nNELM = 3\n",
        oszicar=oszicar,
        outcar=_forces(3.0) + _forces(0.5) + NORMAL_OUTCAR,
    )
    result = guidance.build_manual_guidance(analyze_calculation_directory(directory))
    kinds = [item.finding.kind for item in result.findings]
    assert kinds == [guidance.IONIC_NOT_CONVERGED, guidance.ELECTRONIC_ITERATION_LIMIT]
    limit = result.findings[1]
    assert "1 of 2 completed ionic steps used all NELM = 3 electronic iterations" in limit.finding.observed
    assert "NELM" in titles(limit)


def test_smearing_finding_cites_smearing_guidance_and_links_the_observed_settings(index: manual.ManualIndex) -> None:
    finding = guidance.DiagnosticFinding(
        guidance.SMEARING_QUESTIONABLE,
        observed="The electronic-character assessment is uncertain: the band gap is comparable to the smearing width.",
        source="C2 electronic-character assessment",
        signals=(guidance.EvidenceSignal("ISMEAR", "0", "setting"), guidance.EvidenceSignal("SIGMA", "0.2", "setting")),
    )
    (item,) = guidance.guidance_for_findings((finding,), index)
    assert all("smearing" in reference.title.lower() or "ISMEAR" in reference.title for reference in item.references)
    assert item.linked_settings and set(item.linked_settings) <= {"ISMEAR", "SIGMA"}
    assert "verdict itself is unchanged" in item.next_step


def test_stage_without_execution_evidence_is_reported_without_a_manual_citation(index: manual.ManualIndex) -> None:
    from bmd_agent.resources.completion import STAGE_CONVERGENCE_UNDETERMINED, StageCompletion, StageConvergence

    stages = (
        StageCompletion(1, "relax", True, True, StageConvergence("converged")),
        StageCompletion(2, "static", False, None, StageConvergence(STAGE_CONVERGENCE_UNDETERMINED)),
    )
    analysis = SimpleNamespace(diagnostics=None, stage_completion=stages, incar_settings={})
    (finding,) = guidance.derive_findings(analysis)
    assert finding.kind == guidance.STAGE_NOT_STARTED and "Stage 2 (static)" in finding.observed
    (item,) = guidance.guidance_for_findings((finding,), index)
    assert item.references == () and item.next_step is None and "workflow" in item.limitation


def test_missing_outputs_alone_produce_no_invented_guidance(tmp_path: Path, capsys) -> None:
    directory = write_calculation(tmp_path / "partial", INCAR_SCF, outcar="partial\n")
    assert cli.main([str(directory)]) == 0
    out = capsys.readouterr().out
    assert "VASP manual guidance" not in out


def test_quiet_analysis_does_not_import_the_corpus(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    directory = write_calculation(tmp_path / "pre", INCAR_SCF)  # inputs only: PRE_RUN
    code = (
        "import sys\n"
        "from bmd_agent.cli import main\n"
        f"main([{str(directory)!r}])\n"
        "assert 'bmd_agent.vasp_manual_index' not in sys.modules\n"
        "assert 'bmd_agent.vasp_wiki_corpus' not in sys.modules\n"
    )
    src = Path(__file__).resolve().parents[1] / "src"
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(src)}, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr


def test_index_build_and_search_are_fast_enough_for_the_cli(index: manual.ManualIndex) -> None:
    import time

    started = time.perf_counter()
    manual.build_manual_index(corpus.load_vasp_wiki_corpus())
    build = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(20):
        index.search({"NELM": 2.0, "electronic": 1.5, "convergence": 1.5}, required=("NELM", "electronic"))
    search = (time.perf_counter() - started) / 20
    assert build < 15.0 and search < 0.25, (build, search)


# ---------------------------------------------------------------------------
# Out-of-memory evidence is attributed to the stage that holds the marker
# ---------------------------------------------------------------------------


def memory_item(result: guidance.ManualGuidance) -> guidance.FindingGuidance:
    (item,) = [item for item in result.findings if item.finding.kind == guidance.MEMORY_EXHAUSTION]
    return item


def test_oom_marker_in_stage_1_is_attributed_to_stage_1_not_the_last_stage(
    tmp_path: Path, index: manual.ManualIndex
) -> None:
    root, stage_1, _ = write_two_stage_workflow(tmp_path / "flow", std_err__txt=OOM_LOG)
    analysis = analyze_workflow(root)
    assert analysis.diagnostics.trajectories[-1].stage_index == 2  # the last trajectory is not the failed stage
    item = memory_item(guidance.build_manual_guidance(analysis))
    finding = item.finding
    assert (finding.stage_index, finding.stage_label) == (1, "stage_01")
    assert finding.observed == (
        "The memory evidence establishes that the job ran out of memory (evidence in stage 1 (stage_01))."
    )
    assert finding.source == f"memory (OOM) evidence: std_err.txt ({stage_1 / 'std_err.txt'})"
    assert finding.stage_settings["IBRION"] == 2 and finding.incar_available
    assert_no_stage_2_settings(finding)
    assert guidance._finding_domains(finding) == set()
    assert_no_unused_method_content(index, item)


def test_oom_marker_in_stage_2_uses_the_hse_stage_settings(tmp_path: Path) -> None:
    root, _, stage_2 = write_two_stage_workflow(tmp_path / "flow")
    (stage_2 / "std_err.txt").write_text(OOM_LOG)
    finding = memory_item(guidance.build_manual_guidance(analyze_workflow(root))).finding
    assert (finding.stage_index, finding.stage_label) == (2, "stage_02")
    assert "evidence in stage 2 (stage_02)" in finding.observed
    assert finding.stage_settings["LHFCALC"] is True and finding.stage_settings["NELM"] == 60
    assert ("LHFCALC", ".TRUE.") in [(signal.name, signal.value) for signal in finding.signals]
    assert guidance._finding_domains(finding) == {"hybrid_functional"}
    assert finding.settings_source == "retained INCAR and trajectory criteria of stage 2 (stage_02)"


@pytest.mark.parametrize("stage_2_marker", [OOM_LOG, "forrtl: severe (41): insufficient virtual memory; cannot allocate memory\n"])
def test_oom_evidence_in_several_stages_keeps_the_finding_without_stage_settings(
    tmp_path: Path, stage_2_marker: str, index: manual.ManualIndex
) -> None:
    root, _, stage_2 = write_two_stage_workflow(tmp_path / "flow", std_err__txt=OOM_LOG)
    (stage_2 / "std_err.txt").write_text(stage_2_marker)
    item = memory_item(guidance.build_manual_guidance(analyze_workflow(root)))
    finding = item.finding
    assert finding.observed == (
        "The memory evidence establishes that the job ran out of memory; the evidence does not identify one stage."
    )
    assert finding.stage_index is None and finding.stage_settings == {} and finding.signals == ()
    assert finding.settings_source == (
        "the memory evidence is in several stages (1, 2), so no stage settings are considered"
    )
    assert "stage_01" in finding.source  # the provenance of the decisive marker is kept
    assert guidance._finding_domains(finding) == set()
    assert_no_unused_method_content(index, item)
    assert "not known" in item.next_step or item.next_step is None


def test_root_level_oom_log_in_a_multistage_workflow_identifies_no_stage(tmp_path: Path) -> None:
    root, _, _ = write_two_stage_workflow(tmp_path / "flow")
    (root / "std_err.txt").write_text(OOM_LOG)
    finding = memory_item(guidance.build_manual_guidance(analyze_workflow(root))).finding
    assert finding.stage_index is None and finding.stage_settings == {}
    assert finding.settings_source.startswith("the memory evidence is job-level")
    assert f"std_err.txt ({root / 'std_err.txt'})" in finding.source
    assert "does not identify one stage" in finding.observed


def test_scheduler_only_oom_in_a_multistage_workflow_identifies_no_stage(tmp_path: Path) -> None:
    root, _, _ = write_two_stage_workflow(tmp_path / "flow")
    analysis = analyze_workflow(root, state="OUT_OF_MEMORY")
    assert analysis.diagnostics.oom.assessment == "OOM ESTABLISHED"
    finding = memory_item(guidance.build_manual_guidance(analysis)).finding
    assert finding.stage_index is None and finding.stage_settings == {}
    assert finding.settings_source.startswith("the memory evidence is job-level")
    assert finding.source == "memory (OOM) evidence: SLURM job 4242"


@pytest.mark.parametrize("where", ["scheduler", "root log"])
def test_job_level_oom_with_a_single_executed_stage_uses_that_stage(tmp_path: Path, where: str) -> None:
    root, _, stage_2 = write_two_stage_workflow(tmp_path / "flow")
    for name in ("OSZICAR", "OUTCAR"):
        (stage_2 / name).unlink()  # stage 2 never ran
    if where == "root log":
        (root / "std_err.txt").write_text(OOM_LOG)
    analysis = analyze_workflow(root, state="OUT_OF_MEMORY" if where == "scheduler" else "FAILED")
    assert [trajectory.stage_index for trajectory in analysis.diagnostics.trajectories] == [1]
    finding = memory_item(guidance.build_manual_guidance(analysis)).finding
    assert (finding.stage_index, finding.stage_label) == (1, "stage_01")
    assert finding.observed == "The memory evidence establishes that the job ran out of memory."
    assert_no_stage_2_settings(finding)


def test_single_stage_oom_is_attributed_to_that_stage(tmp_path: Path) -> None:
    analysis, result = run_guidance(tmp_path, "memory")
    finding = memory_item(result).finding
    assert finding.stage_index == 1 and finding.stage_settings["NCORE"] == 4
    assert finding.observed == "The memory evidence establishes that the job ran out of memory."
    assert finding.source.startswith("memory (OOM) evidence: std_err.txt (")


# ---------------------------------------------------------------------------
# Stage identity: the stage index is authoritative within one workflow
# ---------------------------------------------------------------------------


def _record(index, label, directory=None, criteria=None):
    return SimpleNamespace(stage_index=index, stage_label=label, directory=directory, criteria=criteria or {})


def _diagnostics(*records, inputs=()):
    return SimpleNamespace(trajectories=tuple(records), executed_inputs=tuple(inputs))


def test_matching_indices_identify_the_stage_whatever_the_labels() -> None:
    from bmd_agent.resources.run import IncarObservation

    diagnostics = _diagnostics(
        _record(1, "result_dir", "/w/run", {"NELM": 60}),
        inputs=(IncarObservation("result_dir", "/w/run/INCAR", True, 1, values={"ISMEAR": 0, "SIGMA": 0.05}),),
    )
    stage = guidance._stage_settings(diagnostics, 1, "PBE Static Energy")
    assert stage.index == 1 and stage.label == "result_dir"
    assert stage.values == {"ISMEAR": 0, "SIGMA": 0.05, "NELM": 60} and stage.incar_available


def test_conflicting_indices_are_not_matched_by_label() -> None:
    diagnostics = _diagnostics(_record(2, "PBE Static Energy", "/w/s2", {"NELM": 60}))
    stage = guidance._stage_settings(diagnostics, 1, "PBE Static Energy")
    assert stage.values == {} and stage.index == 1 and not stage.incar_available
    assert stage.source == "no execution evidence of stage 1 was identified, so no stage settings are considered"


def test_labels_are_a_fallback_only_when_an_index_is_unavailable() -> None:
    diagnostics = _diagnostics(_record(None, "static", "/w/s", {"NELM": 40}), _record(2, "relax", "/w/r", {"NELM": 8}))
    assert guidance._stage_settings(diagnostics, None, "static").values == {"NELM": 40}
    assert guidance._stage_settings(diagnostics, 1, "static").values == {"NELM": 40}  # the record has no index
    assert guidance._stage_settings(diagnostics, None, "relax").values == {"NELM": 8}
    assert guidance._stage_settings(diagnostics, None, None).values == {}
    assert guidance._stage_settings(diagnostics, None, "missing").values == {}


def test_ambiguous_stage_bindings_match_nothing() -> None:
    from bmd_agent.resources.run import IncarObservation

    # Two directories bound to stage 2 (for example a stage directory and a
    # separate result_dir with the same index).
    diagnostics = _diagnostics(
        _record(2, "stage_02", "/w/s2", {"NELM": 60}),
        _record(2, "result_dir", "/w/result", {"NELM": 40}),
    )
    assert guidance._stage_settings(diagnostics, 2, "PBE Static Energy").values == {}
    # The INCARs of two bindings in one directory are ambiguous too.
    one = _diagnostics(
        _record(1, "stage_01", "/w/s1", {}),
        inputs=(
            IncarObservation("stage_01", "/w/s1/INCAR", True, 1, values={"LHFCALC": True}),
            IncarObservation("result_dir", "/w/s1/INCAR", True, 1, values={"LHFCALC": True}),
        ),
    )
    stage = guidance._stage_settings(one, 1, None)
    assert stage.values == {} and not stage.incar_available


def _c2_smearing(subject_id: str, scope: str, index, label: str) -> SimpleNamespace:
    return SimpleNamespace(
        subject_id=subject_id,
        scope=scope,
        classification="UNCERTAIN",
        evidence=SimpleNamespace(
            channels=(SimpleNamespace(channel="total", reason_code="GAP_COMPARABLE_TO_SMEARING"),),
            ismear=0,
            sigma_ev=0.2,
        ),
        stage=SimpleNamespace(index=index, label=label),
    )


def write_single_stage_result_dir_workflow(root: Path, incar: str) -> Path:
    import json

    result = write_calculation(root / "result", incar, oszicar=HEADER + _electronic_steps(2), outcar="partial\n")
    (root / "attempt.json").write_text(json.dumps({"attempt_id": "a", "state": "SUBMITTED", "job_id": "4242"}))
    payload = {
        "flow_spec": {"workflow_spec": {"stages": [{"stage_type": "static", "theory": "pbe", "modifiers": [], "options": {}}]}},
        "submission": {"attempt_id": "a", "attempt_state": str(root / "attempt.json")},
        "paths": {"result_dir": str(result)},
    }
    (root / "submission.json").write_text(json.dumps(payload))
    return root


def test_c2_stage_with_a_scientific_label_matches_the_lifecycle_result_dir_by_index(tmp_path: Path) -> None:
    root = write_single_stage_result_dir_workflow(tmp_path / "flow", "ISMEAR = 0\nSIGMA = 0.2\nISPIN = 2\nMAGMOM = 2*1\n")
    analysis = analyze_workflow(root)
    assert [(item.stage_index, item.stage_label) for item in analysis.diagnostics.trajectories] == [(1, "result_dir")]
    c2 = _c2_smearing("flow", "bmd_compute_workflow", 1, "PBE Static Energy")
    (finding,) = [item for item in guidance.derive_findings(analysis, electronic_character=c2) if item.kind == guidance.SMEARING_QUESTIONABLE]
    assert finding.stage_index == 1 and finding.stage_label == "PBE Static Energy"
    assert finding.stage_settings["ISPIN"] == 2 and finding.incar_available
    assert finding.settings_source == "retained INCAR of stage 1 (result_dir)"
    assert guidance._finding_domains(finding) == {"magnetism_spin_orbit"}


@pytest.mark.parametrize(
    ("subject_id", "scope"),
    [("another-run", "bmd_compute_workflow"), ("flow", "direct_vasp_calculation"), (None, None)],
)
def test_c2_records_of_another_calculation_are_never_matched_by_index(tmp_path: Path, subject_id, scope) -> None:
    root = write_single_stage_result_dir_workflow(tmp_path / "flow", "ISMEAR = 0\nSIGMA = 0.2\nISPIN = 2\n")
    c2 = _c2_smearing(subject_id, scope, 1, "PBE Static Energy")
    (finding,) = [
        item
        for item in guidance.derive_findings(analyze_workflow(root), electronic_character=c2)
        if item.kind == guidance.SMEARING_QUESTIONABLE
    ]
    assert finding.stage_index is None and finding.stage_settings == {}
    assert finding.settings_source == (
        "the C2 assessment does not identify this calculation, so no stage settings are considered"
    )
    # The C2 observation itself is retained.
    assert [(signal.name, signal.value) for signal in finding.signals] == [("ISMEAR", "0"), ("SIGMA", "0.2")]


def test_c2_of_a_direct_calculation_matches_its_only_stage(tmp_path: Path) -> None:
    directory = write_calculation(tmp_path / "si-static", "ISMEAR = 0\nSIGMA = 0.2\n", oszicar=HEADER + _electronic_steps(2), outcar="partial\n")
    c2 = _c2_smearing("si-static", "direct_vasp_calculation", 1, "Direct VASP calculation")
    (finding,) = [
        item
        for item in guidance.derive_findings(analyze_calculation_directory(directory), electronic_character=c2)
        if item.kind == guidance.SMEARING_QUESTIONABLE
    ]
    assert finding.stage_index == 1 and finding.stage_settings == {"ISMEAR": 0, "SIGMA": 0.2}
    assert finding.settings_source == "retained INCAR of stage 1 (work_dir)"
