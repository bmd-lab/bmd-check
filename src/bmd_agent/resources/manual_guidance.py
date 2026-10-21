"""Evidence-driven VASP Wiki guidance for BMD Check findings (R3).

This module connects findings that BMD Check's existing assessments have
already established to passages of the frozen VASP Wiki corpus. It never
changes those findings or their verdicts:

* *Findings* are derived only from existing typed evidence: convergence
  progress assessments and trajectories (electronic and ionic), the C1 stage
  completion records, memory (OOM) evidence, bounded VASP log messages and,
  for completed calculations, the C2 electronic-character assessment. No new
  scientific criterion is introduced; each finding restates what the existing
  assessment established.
* *References* are passages retrieved from the whole corpus with a query built
  from the finding's observed evidence (the INCAR settings and criteria that
  the finding involves, settings that change the calculation's method, and
  distinctive log-message tokens). A passage is eligible only if it mentions
  at least one anchor term of the finding.
* *Stage attribution.* Every finding is tied to the workflow stage it comes
  from, and is read with that stage's own executed settings only: the stage's
  retained INCAR, with the stage's trajectory criteria (NELM, EDIFF, NSW,
  EDIFFG, ISIF) taking precedence. Settings are never borrowed from another
  stage or from the INCAR of the analyzed directory; when a stage's INCAR is
  not available its settings are omitted. Within one workflow the stage index
  is the identity (labels are only a fallback). Out-of-memory evidence is
  attributed to the stage whose directory holds the decisive marker, never to
  the last stage by default; when no one stage is identified, no stage
  settings are attached.
* *Method filtering by content.* A passage whose own text (or its page title
  or section heading) is about a method that the finding's stage does not use
  (hybrid functionals, DFT+U, GW/RPA/BSE, machine-learned force fields,
  phonons and linear response, molecular dynamics, optics, NMR, NEB, dimer,
  magnetism and spin-orbit coupling, meta-GGA) is excluded, so it can be
  cited neither in the concise nor in the verbose output. A single passing
  mention of such a method in general guidance does not exclude a passage.
* *Next steps* never prescribe a parameter value. A setting is named in a next
  step only when it is both observed in this calculation's evidence and named
  in the cited passage; otherwise the step points to the cited passage.

Everything here is pure: it reads no files itself, writes nothing and uses no
network. The corpus index is loaded lazily, only when a finding exists.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import PurePath
import re
from typing import Any, Callable

from bmd_agent.resources.completion import StageCompletion
from bmd_agent.resources.oom import OOM_ESTABLISHED, OOM_POSSIBLE
from bmd_agent.resources.run import CONVERGED, StageTrajectoryObservation


GUIDANCE_AVAILABLE = "available"
GUIDANCE_UNAVAILABLE = "references_unavailable"
GUIDANCE_NO_FINDINGS = "no_findings"

VASP_ERROR_MESSAGE = "vasp_error_message"
MEMORY_EXHAUSTION = "memory_exhaustion"
ELECTRONIC_NOT_CONVERGED = "electronic_not_converged"
IONIC_NOT_CONVERGED = "ionic_not_converged"
ELECTRONIC_ITERATION_LIMIT = "electronic_iteration_limit"
SMEARING_QUESTIONABLE = "smearing_questionable"
STAGE_NOT_STARTED = "stage_not_started"

# Display priority: the most specific, most actionable finding first.
_PRIORITY = (
    VASP_ERROR_MESSAGE,
    MEMORY_EXHAUSTION,
    ELECTRONIC_NOT_CONVERGED,
    IONIC_NOT_CONVERGED,
    ELECTRONIC_ITERATION_LIMIT,
    SMEARING_QUESTIONABLE,
    STAGE_NOT_STARTED,
)

_C2_SMEARING_REASONS = frozenset({"GAP_COMPARABLE_TO_SMEARING", "SMEARING_SPILLS_ACROSS_GAP"})
_TRUE_VALUES = {"true", ".true.", "t", ".t.", "1", "yes"}

_GW_ALGOS = frozenset(
    {"GW0", "GW", "EVGW0", "EVGW", "QPGW0", "QPGW", "G0W0R", "GW0R", "GWR", "ACFDT", "ACFDTR", "RPA", "RPAR", "BSE", "TDHF", "CHI", "CRPA"}
)


@dataclass(frozen=True)
class EvidenceSignal:
    """One observed setting, criterion or message token that a finding involves."""

    name: str
    value: str | None
    role: str  # "criterion", "setting", "context" or "message"


@dataclass(frozen=True)
class DiagnosticFinding:
    """A BMD Check finding restated from an existing assessment."""

    kind: str
    observed: str
    source: str
    stage_label: str | None = None
    signals: tuple[EvidenceSignal, ...] = ()
    message: str | None = None
    stage_index: int | None = None
    # The executed settings of the finding's own stage (empty when they could
    # not be established) and where they come from.
    stage_settings: Mapping[str, Any] = field(default_factory=dict, hash=False)
    settings_source: str | None = None
    incar_available: bool = False


@dataclass(frozen=True)
class ManualReference:
    title: str
    page_id: str
    revision_id: int
    permalink: str
    section: str | None
    excerpt: str
    matched_terms: tuple[str, ...]
    score: float
    passage_position: int = 0


@dataclass(frozen=True)
class FindingGuidance:
    finding: DiagnosticFinding
    references: tuple[ManualReference, ...] = ()
    next_step: str | None = None
    linked_settings: tuple[str, ...] = ()
    limitation: str | None = None


@dataclass(frozen=True)
class ManualGuidance:
    status: str
    findings: tuple[FindingGuidance, ...] = ()
    corpus_version: str | None = None
    corpus_digest: str | None = None
    index_identity: str | None = None
    unavailable_reason: str | None = None
    limitations: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Findings from existing evidence
# ---------------------------------------------------------------------------


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE_VALUES


def _int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class _StageSettings:
    index: int | None
    label: str | None
    values: Mapping[str, Any]
    source: str | None
    incar_available: bool


def _unattributed(reason: str) -> _StageSettings:
    return _StageSettings(None, None, {}, f"{reason}, so no stage settings are considered", False)


_NO_STAGE = _unattributed("no single stage could be attributed")


def _valid_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _resolve_stage_record(
    records: Sequence[Any],
    index: Any,
    label: Any,
    index_of: Callable[[Any], Any],
    label_of: Callable[[Any], Any],
) -> Any | None:
    """The one record of the *same workflow* that is the requested stage.

    A valid stage index is the authoritative identity: a record with a valid
    index matches only on that index, whatever its label (lifecycle labels
    are directory roles such as ``result_dir``, C2 labels are scientific
    names such as ``PBE Static Energy``). A label is used only as a fallback,
    against records whose own index is unavailable, or when the request has
    no valid index. Several matching records (an ambiguous binding) match
    nothing.
    """

    if _valid_index(index):
        hits = [record for record in records if index_of(record) == index]
        if not hits and label:
            hits = [
                record for record in records if not _valid_index(index_of(record)) and label_of(record) == label
            ]
    elif label:
        hits = [record for record in records if label_of(record) == label]
    else:
        hits = []
    return hits[0] if len(hits) == 1 else None


def _same_directory(left: Any, right: Any) -> bool:
    try:
        return bool(left) and bool(right) and PurePath(str(left)) == PurePath(str(right))
    except (TypeError, ValueError):
        return False


def _stage_settings(
    diagnostics: Any,
    index: int | None,
    label: str | None,
    *,
    trajectory: Any | None = None,
) -> _StageSettings:
    """The executed settings of one stage, from that stage's evidence only.

    The stage's retained INCAR is the base; its trajectory criteria take
    precedence. Nothing is taken from any other stage, and nothing from the
    INCAR of the analyzed directory: if the stage's INCAR is not available,
    only its trajectory criteria are known. ``diagnostics`` holds the records
    of one workflow, so stage indices are comparable within it. When the
    caller already holds the stage's own ``trajectory``, its directory
    identifies the INCAR.
    """

    if diagnostics is None or (trajectory is None and not _valid_index(index) and not label):
        return _StageSettings(index, label, {}, None, False)
    trajectories = tuple(getattr(diagnostics, "trajectories", ()) or ())
    inputs = tuple(getattr(diagnostics, "executed_inputs", ()) or ())
    if trajectory is None:
        trajectory = _resolve_stage_record(
            trajectories, index, label, lambda item: item.stage_index, lambda item: item.stage_label
        )
    incar = None
    directory = getattr(trajectory, "directory", None) if trajectory is not None else None
    if directory:
        located = [item for item in inputs if _same_directory(PurePath(str(item.path)).parent, directory)]
        incar = located[0] if len(located) == 1 else None
    elif trajectory is not None:
        incar = _resolve_stage_record(
            inputs, trajectory.stage_index, trajectory.stage_label, lambda item: item.stage_index, lambda item: item.label
        )
    else:
        incar = _resolve_stage_record(inputs, index, label, lambda item: item.stage_index, lambda item: item.label)
    if trajectory is None and incar is None:
        return _StageSettings(
            index,
            label,
            {},
            f"no execution evidence of stage {index if _valid_index(index) else label} was identified, "
            "so no stage settings are considered",
            False,
        )
    values: dict[str, Any] = {}
    sources: list[str] = []
    incar_available = False
    if incar is not None and incar.present and not incar.error:
        values.update({str(key).upper(): value for key, value in (incar.values or {}).items()})
        sources.append("retained INCAR")
        incar_available = True
    if trajectory is not None:
        criteria = {
            str(key).upper(): value for key, value in (trajectory.criteria or {}).items() if value is not None
        }
        if criteria:
            values.update(criteria)  # the stage's own criteria are authoritative
            sources.append("trajectory criteria")
    record = trajectory if trajectory is not None else incar
    index = getattr(record, "stage_index", None)
    label = getattr(record, "stage_label", None) if trajectory is not None else getattr(record, "label", None)
    name = f"stage {index} ({label})" if index is not None and label else f"stage {label or index}"
    source = f"{' and '.join(sources)} of {name}" if sources else None
    if not incar_available:
        missing = f"the executed INCAR of {name} is not available, so its settings are omitted"
        source = f"{source}; {missing}" if source else missing
    return _StageSettings(index, label, values, source, incar_available)


def _stage_fields(stage: _StageSettings) -> dict[str, Any]:
    return {
        "stage_index": stage.index,
        "stage_settings": dict(stage.values),
        "settings_source": stage.source,
        "incar_available": stage.incar_available,
    }


def _stage_of_directory(
    directory: Any,
    trajectories: Sequence[StageTrajectoryObservation],
) -> StageTrajectoryObservation | None:
    """The stage bound to ``directory`` (the lifecycle stage-directory binding)."""

    located = [trajectory for trajectory in trajectories if _same_directory(trajectory.directory, directory)]
    return located[0] if len(located) == 1 else None


def _log_stage(
    diagnostics: Any,
    log: Any,
    trajectories: Sequence[StageTrajectoryObservation],
) -> _StageSettings:
    """The stage a log belongs to: the stage whose directory holds it, or the
    only stage with execution evidence. A log of a multistage workflow outside
    every stage directory is attributed to no stage (and so no settings)."""

    try:
        parent = PurePath(str(log.path)).parent
    except (TypeError, ValueError):
        parent = None
    stage = _stage_of_directory(parent, trajectories) if parent is not None else None
    if stage is not None:
        return _stage_settings(diagnostics, stage.stage_index, stage.stage_label, trajectory=stage)
    if len(trajectories) == 1:
        only = trajectories[0]
        return _stage_settings(diagnostics, only.stage_index, only.stage_label, trajectory=only)
    return _NO_STAGE


def _oom_stage(
    diagnostics: Any,
    oom: Any,
    trajectories: Sequence[StageTrajectoryObservation],
) -> tuple[_StageSettings, tuple[str, ...]]:
    """The stage that holds the decisive OOM evidence, and that evidence's sources.

    The decisive markers are the explicit ones when OOM is established, else
    the suggestive ones. A marker from a log or custodian record is located in
    a stage through the lifecycle stage-directory bindings; scheduler
    accounting and logs outside every stage directory (for example the
    workflow root) are job-level and identify no stage. The finding is
    attributed to a stage only when:

    * the decisive stage-local markers are all in one stage and no marker of
      any kind is in another stage; or
    * all markers are job-level and exactly one stage has execution evidence.

    Otherwise (markers in several stages, or job-level evidence in a
    multistage workflow) no stage settings are attached.
    """

    explicit = tuple(getattr(oom, "explicit_evidence", ()) or ())
    suggestive = tuple(getattr(oom, "suggestive_evidence", ()) or ())
    decisive = explicit if oom.assessment == OOM_ESTABLISHED and explicit else suggestive or explicit
    sources = tuple(dict.fromkeys(marker.source for marker in decisive))

    log_directories: dict[str, Any] = {}
    for log in getattr(diagnostics, "logs", ()) or ():
        log_directories[f"{log.label} ({log.path})"] = PurePath(str(log.path)).parent
    custodian = getattr(diagnostics, "custodian", None)
    if custodian is not None and getattr(custodian, "path", None):
        log_directories[f"custodian.json ({custodian.path})"] = PurePath(str(custodian.path)).parent

    def stage_of(marker: Any) -> StageTrajectoryObservation | None:
        if marker.source_type != "log_observation" or marker.source not in log_directories:
            return None
        return _stage_of_directory(log_directories[marker.source], trajectories)

    decisive_stages = {id(stage): stage for stage in map(stage_of, decisive) if stage is not None}
    every_stage = {id(stage): stage for stage in map(stage_of, explicit + suggestive) if stage is not None}
    if len(decisive_stages) == 1 and set(every_stage) <= set(decisive_stages):
        (stage,) = decisive_stages.values()
        return _stage_settings(diagnostics, stage.stage_index, stage.stage_label, trajectory=stage), sources
    if len(every_stage) > 1:
        numbers = ", ".join(
            str(stage.stage_index) for stage in sorted(every_stage.values(), key=lambda item: item.stage_index or 0)
        )
        return _unattributed(f"the memory evidence is in several stages ({numbers})"), sources
    if not every_stage and len(trajectories) == 1:
        only = trajectories[0]
        return _stage_settings(diagnostics, only.stage_index, only.stage_label, trajectory=only), sources
    return (
        _unattributed(
            "the memory evidence is job-level (scheduler accounting or logs outside every stage directory) "
            "and does not identify one stage"
        ),
        sources,
    )


def _c2_identifies_this_calculation(analysis: Any, assessment: Any) -> bool:
    """Whether a C2 assessment is of this analysis's calculation.

    C2 identifies its subject by scope and subject id (the submission's
    ``run_name`` or the workflow directory name; the directory name for a
    direct calculation); a stage index is meaningful only within that subject.
    An assessment that cannot be verified is not matched to any stage.
    """

    scope = getattr(assessment, "scope", None)
    subject = getattr(assessment, "subject_id", None)
    workflow = getattr(analysis, "bmd_workflow", None)
    if workflow is not None:
        submission = getattr(workflow, "submission", None)
        run_name = submission.get("run_name") if isinstance(submission, Mapping) else None
        root = getattr(workflow, "workflow_root", None)
        expected = run_name if isinstance(run_name, str) and run_name else getattr(root, "name", None)
        return scope == "bmd_compute_workflow" and bool(expected) and subject == expected
    directory = getattr(analysis, "directory", None)
    if directory is None:
        return False
    expected = getattr(directory, "name", None) or "calculation"
    return scope == "direct_vasp_calculation" and subject == expected


def _context_signals(settings: Mapping[str, Any]) -> tuple[EvidenceSignal, ...]:
    """Settings that change the method and therefore the relevant documentation."""

    signals: list[EvidenceSignal] = []
    if _is_true(settings.get("LHFCALC", False)):
        signals.append(EvidenceSignal("LHFCALC", _text(settings["LHFCALC"]), "context"))
        if "HFSCREEN" in settings:
            signals.append(EvidenceSignal("HFSCREEN", _text(settings["HFSCREEN"]), "context"))
    if _is_true(settings.get("LDAU", False)):
        signals.append(EvidenceSignal("LDAU", _text(settings["LDAU"]), "context"))
    if _int(settings.get("ISPIN")) == 2:
        signals.append(EvidenceSignal("ISPIN", "2", "context"))
    if "MAGMOM" in settings:
        signals.append(EvidenceSignal("MAGMOM", None, "context"))
    if _is_true(settings.get("LSORBIT", False)):
        signals.append(EvidenceSignal("LSORBIT", _text(settings["LSORBIT"]), "context"))
    if _is_true(settings.get("LNONCOLLINEAR", False)):
        signals.append(EvidenceSignal("LNONCOLLINEAR", _text(settings["LNONCOLLINEAR"]), "context"))
    if "METAGGA" in settings:
        signals.append(EvidenceSignal("METAGGA", _text(settings["METAGGA"]), "context"))
    return tuple(signals)


def _present(settings: Mapping[str, Any], names: Iterable[str], role: str = "setting") -> list[EvidenceSignal]:
    return [EvidenceSignal(name, _text(settings[name]), role) for name in names if name in settings]


def _stage_name(trajectory: StageTrajectoryObservation, multi: bool) -> str:
    if multi and trajectory.stage_index is not None:
        return f"Stage {trajectory.stage_index} ({trajectory.stage_label})"
    return "The calculation"


def _final_cycle_iterations(trajectory: StageTrajectoryObservation) -> int | None:
    completed = [cycle for cycle in trajectory.electronic_cycles if cycle.completed_ionic_step]
    if completed:
        return completed[-1].iterations
    counts = trajectory.electronic_iterations_by_completed_ionic_step
    return counts[-1] if counts else None


def derive_findings(
    analysis: Any,
    *,
    electronic_character: Any | None = None,
) -> tuple[DiagnosticFinding, ...]:
    """Restate the significant findings of the existing assessments.

    ``analysis`` is a ``LifecycleAnalysis``; ``electronic_character`` an
    optional ``ElectronicCharacterAssessment`` already computed by C2.
    """

    findings: list[DiagnosticFinding] = []
    diagnostics = getattr(analysis, "diagnostics", None)
    trajectories: Sequence[StageTrajectoryObservation] = getattr(diagnostics, "trajectories", ()) or ()
    assessments = getattr(diagnostics, "assessments", ()) or ()
    multi = len(trajectories) > 1

    # VASP error messages from bounded log excerpts.
    seen_messages: set[str] = set()
    for log in getattr(diagnostics, "logs", ()) or ():
        for message in log.messages:
            tokens = distinctive_message_tokens(message)
            if not tokens or message in seen_messages:
                continue
            seen_messages.add(message)
            stage = _log_stage(diagnostics, log, trajectories)
            findings.append(
                DiagnosticFinding(
                    VASP_ERROR_MESSAGE,
                    observed=f"{log.label} contains the message: {message}",
                    source=f"bounded log excerpt ({log.label})",
                    stage_label=stage.label,
                    signals=tuple(EvidenceSignal(token, None, "message") for token in tokens),
                    message=message,
                    **_stage_fields(stage),
                )
            )

    # Memory exhaustion established (or strongly suggested) by OOM evidence.
    oom = getattr(diagnostics, "oom", None)
    if oom is not None and oom.assessment in (OOM_ESTABLISHED, OOM_POSSIBLE):
        # The stage is the one whose directory holds the decisive marker;
        # it is never inferred from the order of the trajectories.
        stage, marker_sources = _oom_stage(diagnostics, oom, trajectories)
        settings = stage.values
        observed = (
            "The memory evidence establishes that the job ran out of memory"
            if oom.assessment == OOM_ESTABLISHED
            else "The memory evidence suggests memory pressure, but does not establish an out-of-memory stop"
        )
        if multi and stage.index is not None:
            observed += f" (evidence in stage {stage.index} ({stage.label}))."
        elif multi:
            observed += "; the evidence does not identify one stage."
        else:
            observed += "."
        findings.append(
            DiagnosticFinding(
                MEMORY_EXHAUSTION,
                observed=observed,
                source="memory (OOM) evidence" + (f": {'; '.join(marker_sources)}" if marker_sources else ""),
                stage_label=stage.label,
                signals=tuple(
                    _present(settings, ("NCORE", "NPAR", "KPAR", "LREAL", "NBANDS", "ALGO", "ISYM"))
                )
                + _context_signals(settings),
                **_stage_fields(stage),
            )
        )

    by_stage: dict[tuple[int | None, str], dict[str, Any]] = {}
    for assessment in assessments:
        by_stage.setdefault((assessment.stage_index, assessment.stage_label), {})[assessment.scope] = assessment

    for trajectory in trajectories:
        stage = _stage_settings(diagnostics, trajectory.stage_index, trajectory.stage_label, trajectory=trajectory)
        settings = stage.values
        stage_assessments = by_stage.get((trajectory.stage_index, trajectory.stage_label), {})
        name = _stage_name(trajectory, multi)
        nelm = _int(settings.get("NELM"))
        electronic = stage_assessments.get("electronic")
        electronic_failed = bool(
            electronic is not None
            and electronic.label != CONVERGED
            and electronic.counter_evidence
            and electronic.sufficiency != "contradictory"
        )
        if electronic_failed:
            iterations = _final_cycle_iterations(trajectory)
            ediff = _text(settings.get("EDIFF"))
            details = f"the final completed electronic cycle did not reach EDIFF = {ediff}" if ediff else (
                "electronic convergence was not reached"
            )
            if iterations is not None and nelm is not None and iterations >= nelm:
                details += f" after using all NELM = {nelm} electronic iterations"
            findings.append(
                DiagnosticFinding(
                    ELECTRONIC_NOT_CONVERGED,
                    observed=f"{name}: {details}.",
                    source="electronic convergence progress assessment",
                    stage_label=trajectory.stage_label,
                    signals=tuple(_present(settings, ("NELM", "EDIFF"), "criterion"))
                    + tuple(
                        _present(
                            settings,
                            ("ALGO", "IALGO", "AMIX", "BMIX", "AMIX_MAG", "BMIX_MAG", "MAXMIX", "LMAXMIX", "NBANDS", "TIME", "LSUBROT"),
                        )
                    )
                    + _context_signals(settings),
                    **_stage_fields(stage),
                )
            )
        elif nelm is not None:
            counts = [
                cycle.iterations for cycle in trajectory.electronic_cycles if cycle.completed_ionic_step
            ] or list(trajectory.electronic_iterations_by_completed_ionic_step)
            exhausted = sum(1 for count in counts if count >= nelm)
            if exhausted:
                findings.append(
                    DiagnosticFinding(
                        ELECTRONIC_ITERATION_LIMIT,
                        observed=(
                            f"{name}: {exhausted} of {len(counts)} completed ionic steps used all "
                            f"NELM = {nelm} electronic iterations."
                        ),
                        source="electronic trajectory observation",
                        stage_label=trajectory.stage_label,
                        signals=tuple(_present(settings, ("NELM", "EDIFF"), "criterion"))
                        + tuple(_present(settings, ("ALGO", "IALGO", "AMIX", "BMIX")))
                        + _context_signals(settings),
                        **_stage_fields(stage),
                    )
                )

        ionic = stage_assessments.get("ionic")
        if (
            ionic is not None
            and ionic.label != CONVERGED
            and ionic.counter_evidence
            and ionic.sufficiency != "contradictory"
        ):
            nsw = _int(settings.get("NSW"))
            steps = trajectory.completed_ionic_steps
            ediffg = _text(settings.get("EDIFFG"))
            details = f"the final maximum force has not reached EDIFFG = {ediffg}" if ediffg else (
                "ionic convergence was not reached"
            )
            exhausted_nsw = nsw is not None and steps is not None and steps >= nsw
            if exhausted_nsw:
                details += f" after all NSW = {nsw} ionic steps"
            findings.append(
                DiagnosticFinding(
                    IONIC_NOT_CONVERGED,
                    observed=f"{name}: {details}.",
                    source="ionic convergence progress assessment",
                    stage_label=trajectory.stage_label,
                    signals=tuple(_present(settings, ("EDIFFG",) + (("NSW",) if exhausted_nsw else ()), "criterion"))
                    + tuple(_present(settings, (() if exhausted_nsw else ("NSW",)) + ("IBRION", "POTIM", "ISIF")))
                    + _context_signals(settings),
                    **_stage_fields(stage),
                )
            )

    # Stages declared by the workflow that never started (C1).
    for stage in getattr(analysis, "stage_completion", ()) or ():
        if isinstance(stage, StageCompletion) and not stage.started and len(analysis.stage_completion) > 1:
            findings.append(
                DiagnosticFinding(
                    STAGE_NOT_STARTED,
                    observed=f"Stage {stage.index} ({stage.label}) has no execution evidence.",
                    source="C1 stage completion",
                    stage_label=stage.label,
                    stage_index=stage.index,
                )
            )

    # C2: smearing-related uncertainty in a completed calculation.
    if electronic_character is not None:
        evidence = getattr(electronic_character, "evidence", None)
        for channel in getattr(evidence, "channels", ()) or ():
            if channel.reason_code in _C2_SMEARING_REASONS:
                ismear = getattr(evidence, "ismear", None)
                sigma = getattr(evidence, "sigma_ev", None)
                selected = getattr(electronic_character, "stage", None)
                if _c2_identifies_this_calculation(analysis, electronic_character):
                    stage = _stage_settings(
                        diagnostics, getattr(selected, "index", None), getattr(selected, "label", None)
                    )
                else:
                    stage = _unattributed("the C2 assessment does not identify this calculation")
                text = {
                    "GAP_COMPARABLE_TO_SMEARING": "the band gap is comparable to the smearing width",
                    "SMEARING_SPILLS_ACROSS_GAP": "smearing places occupation in states above the gap",
                }[channel.reason_code]
                signals = []
                if ismear is not None:
                    signals.append(EvidenceSignal("ISMEAR", str(ismear), "setting"))
                if sigma is not None:
                    signals.append(EvidenceSignal("SIGMA", f"{sigma:g}", "setting"))
                findings.append(
                    DiagnosticFinding(
                        SMEARING_QUESTIONABLE,
                        observed=(
                            f"The electronic-character assessment is {electronic_character.classification}: "
                            f"in the {channel.channel} channel {text}."
                        ),
                        source="C2 electronic-character assessment",
                        stage_label=getattr(selected, "label", None),
                        signals=tuple(signals),
                        **_stage_fields(stage),
                    )
                )
                break

    order = {kind: position for position, kind in enumerate(_PRIORITY)}
    unique: dict[tuple[str, str], DiagnosticFinding] = {}
    for finding in findings:
        unique.setdefault((finding.kind, finding.observed), finding)
    return tuple(sorted(unique.values(), key=lambda item: order[item.kind]))


_GENERIC_MESSAGE_WORDS = frozenset(
    {
        "ERROR", "ERRORS", "WARNING", "WARNINGS", "FATAL", "VASP", "INFO", "NOTE", "ABORT", "ABORTING",
        "EXIT", "EXITED", "STATUS", "SIGNAL", "SIGSEGV", "SIGKILL", "SIGTERM", "CODE", "RANK", "MPI",
        "TRACEBACK", "STOP", "PROGRAM", "TERMINATED", "INTERNAL", "VERY", "BAD", "NEWS", "PLEASE",
    }
)
_MESSAGE_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")


def distinctive_message_tokens(message: str) -> tuple[str, ...]:
    """Identifier-like tokens of a log message (routine names such as ZBRENT,
    EDDDAV or ZHEGV); generic words such as ERROR or FATAL are ignored."""

    tokens = []
    for token in _MESSAGE_TOKEN_RE.findall(message):
        if token not in _GENERIC_MESSAGE_WORDS and token not in tokens:
            tokens.append(token)
    return tuple(tokens[:6])


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Method domains. A passage is *about* a method when its text, page title or
# section heading names one of the method's identifiers (INCAR tags or
# established acronyms, matched case-sensitively), or when its text describes
# the method in words at least twice (or its title or heading does once).
# Such passages are excluded for findings whose stage does not use the method.
# The markers describe scientific content, not page names.
@dataclass(frozen=True)
class _Domain:
    name: str
    identifiers: tuple[str, ...]  # case-sensitive whole tokens
    descriptions: tuple[str, ...]  # case-insensitive phrases (plural allowed)
    identifier_pattern: str | None = None  # an additional identifier family


_DOMAINS: tuple[_Domain, ...] = (
    _Domain(
        "hybrid_functional",
        ("LHFCALC", "HFSCREEN", "AEXX", "AGGAX", "AGGAC", "ALDAC", "PRECFOCK", "NKRED", "NKREDX", "NKREDY",
         "NKREDZ", "HFRCUT", "LMODELHF", "HSE", "HSE03", "HSE06", "PBE0", "B3LYP", "HFRCUT"),
        ("hybrid functional", "Hartree-Fock", "exact exchange", "Fock exchange"),
    ),
    _Domain(
        "dft_plus_u",
        ("LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ", "LDAUPRINT", "DFT+U", "LDA+U", "GGA+U"),
        ("Hubbard", "on-site Coulomb"),
    ),
    _Domain(
        "gw_rpa_bse",
        ("GW", "G0W0", "GW0", "EVGW0", "EVGW", "QPGW0", "QPGW", "scGW", "G0W0R", "GW0R", "ACFDT", "ACFDTR",
         "RPA", "RPAR", "BSE", "TDHF", "cRPA", "CRPA", "NBANDSGW", "OMEGAMAX", "NOMEGA", "ENCUTGW", "LSPECTRAL"),
        ("quasiparticle", "Bethe-Salpeter", "random-phase approximation", "many-body perturbation"),
    ),
    _Domain(
        "machine_learned_force_field",
        ("MLFF", "MLFFs"),
        ("machine-learned", "machine learning", "machine-learning", "force field"),
        identifier_pattern=r"(?<![A-Za-z0-9_])ML_[A-Z0-9_]+",
    ),
    _Domain(
        "phonons_linear_response",
        ("DFPT", "LEPSILON", "LRPA", "phonopy"),
        ("phonon", "linear response", "Born effective charge"),
    ),
    _Domain(
        "molecular_dynamics",
        ("MDALGO", "TEBEG", "TEEND", "SMASS", "ANDERSEN_PROB", "LANGEVIN_GAMMA", "NVT", "NpT", "NVE", "NPT"),
        ("molecular dynamics", "molecular-dynamics", "thermostat", "MD"),
    ),
    _Domain(
        "optics",
        ("LOPTICS", "CSHIFT"),
        ("dielectric function", "optical propert", "frequency-dependent dielectric"),
    ),
    _Domain(
        "nmr",
        ("NMR", "LCHIMAG", "LEFG", "QUAD_EFG", "DQ", "ICHIBARE", "LNMR_SYM_RED"),
        ("chemical shift", "electric field gradient", "magnetic susceptibility"),
    ),
    _Domain(
        "nudged_elastic_band",
        ("NEB", "CI-NEB", "IMAGES", "LCLIMB", "SPRING", "VTST"),
        ("nudged elastic band", "climbing image", "transition state"),
    ),
    _Domain(
        "dimer",
        ("DdR", "DRotMax", "DFNMin", "DFNMax"),
        ("dimer method", "improved dimer"),
    ),
    _Domain(
        "magnetism_spin_orbit",
        ("MAGMOM", "LNONCOLLINEAR", "LSORBIT", "AMIX_MAG", "BMIX_MAG", "SAXIS", "I_CONSTRAINED_M", "M_CONSTR",
         "LSPIRAL", "QSPIRAL", "SOC"),
        ("magnetic", "magnetism", "magnetization", "magnetisation", "spin-polarized", "spin polarized",
         "noncollinear", "non-collinear", "spin-orbit", "ferromagnetic", "antiferromagnetic"),
    ),
    _Domain(
        "meta_gga",
        ("METAGGA", "SCAN", "r2SCAN", "R2SCAN", "rSCAN", "MBJ", "mBJ", "CMBJ", "TPSS", "RTPSS", "M06L", "LTBOUNDLIBXC"),
        ("meta-GGA", "kinetic-energy density", "kinetic energy density"),
    ),
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def _sentences(text: str) -> list[str]:
    """Sentences of a passage. Sentences of fewer than three words are residue
    of removed templates (for example "This ." where a video template stood)
    and are skipped."""

    return [
        part.strip()
        for part in _SENTENCE_SPLIT_RE.split(" ".join(text.split()))
        if len(re.findall(r"[A-Za-z0-9]+", part)) >= 3
    ]


_BOUNDARY_LEFT = r"(?<![A-Za-z0-9_])"
_BOUNDARY_RIGHT = r"(?![A-Za-z0-9_])"


def _identifier_regex(domain: _Domain) -> re.Pattern[str]:
    names = sorted(set(domain.identifiers), key=len, reverse=True)
    alternatives = [_BOUNDARY_LEFT + re.escape(name) + r"(?![A-Za-z0-9_+])" for name in names]
    if domain.identifier_pattern:
        alternatives.append(domain.identifier_pattern)
    return re.compile("|".join(alternatives))


def _description_regex(domain: _Domain) -> re.Pattern[str]:
    # "MD" is an abbreviation, matched case-sensitively; the phrases are
    # matched case-insensitively and may be plural or continue a word stem.
    phrases = []
    for phrase in sorted(set(domain.descriptions), key=len, reverse=True):
        body = re.escape(phrase).replace(r"\ ", r"[\s-]+").replace(r"\-", r"[\s-]+")
        if phrase.isupper():
            phrases.append(_BOUNDARY_LEFT + body + _BOUNDARY_RIGHT)
        else:
            phrases.append(_BOUNDARY_LEFT + "(?i:" + body + r")[a-z]*")
    return re.compile("|".join(phrases))


_DOMAIN_PATTERNS: tuple[tuple[str, re.Pattern[str], re.Pattern[str]], ...] = tuple(
    (domain.name, _identifier_regex(domain), _description_regex(domain)) for domain in _DOMAINS
)


@lru_cache(maxsize=4096)
def passage_method_domains(text: str, *, title: str = "", heading: str | None = None) -> frozenset[str]:
    """The method domains a passage is *about* (see ``_DOMAINS``).

    A passage is about a method when the text, title or heading names one of
    the method's identifiers; when the text describes the method in words at
    least twice, or in more than half of its sentences (a short passage such
    as "Use shared-memory segments ... in hybrid-functional calculations."); or
    when the title or heading describes it. A single passing description in
    longer general guidance ("... for specific cases, e.g., charged systems or
    magnetic systems.") does not make a passage method-specific.
    """

    context = f"{title}\n{heading or ''}"
    sentences = _sentences(text)
    domains: set[str] = set()
    for name, identifiers, descriptions in _DOMAIN_PATTERNS:
        if identifiers.search(text) or identifiers.search(context) or descriptions.search(context):
            domains.add(name)
            continue
        mentions = len(descriptions.findall(text))
        if mentions >= 2:
            domains.add(name)
        elif mentions == 1:
            describing = sum(1 for sentence in sentences if descriptions.search(sentence))
            if describing * 2 > len(sentences):
                domains.add(name)
    return frozenset(domains)


def _active_domains(settings: Mapping[str, Any]) -> set[str]:
    """Method domains that the stage's executed settings establish."""

    active: set[str] = set()
    algo = str(settings.get("ALGO", "")).strip().upper()
    if _is_true(settings.get("LHFCALC", False)):
        active.add("hybrid_functional")
    if _is_true(settings.get("LDAU", False)):
        active.add("dft_plus_u")
    if algo in _GW_ALGOS:
        active.add("gw_rpa_bse")
    if _is_true(settings.get("ML_LMLFF", False)) or "ML_MODE" in settings:
        active.add("machine_learned_force_field")
    ibrion = _int(settings.get("IBRION"))
    if ibrion in (5, 6, 7, 8) or _is_true(settings.get("LEPSILON", False)):
        active.add("phonons_linear_response")
    if ibrion == 0 or "MDALGO" in settings:
        active.add("molecular_dynamics")
    if _is_true(settings.get("LOPTICS", False)):
        active.add("optics")
    if _is_true(settings.get("LCHIMAG", False)) or _is_true(settings.get("LEFG", False)):
        active.add("nmr")
    if "METAGGA" in settings:
        active.add("meta_gga")
    if "IMAGES" in settings:
        active.add("nudged_elastic_band")
    if ibrion == 44:
        active.add("dimer")
    if (
        _int(settings.get("ISPIN")) == 2
        or "MAGMOM" in settings
        or _is_true(settings.get("LNONCOLLINEAR", False))
        or _is_true(settings.get("LSORBIT", False))
    ):
        active.add("magnetism_spin_orbit")
    return active


def _finding_domains(finding: DiagnosticFinding) -> set[str]:
    """Methods in use for the finding: those its own stage's settings
    establish, plus any method that a quoted log message itself names."""

    active = _active_domains(finding.stage_settings)
    if finding.message:
        for name, identifiers, _ in _DOMAIN_PATTERNS:
            if identifiers.search(finding.message):
                active.add(name)
    return active


@dataclass(frozen=True)
class _QuerySpec:
    """One weighted sub-query. Every group in ``anchor_groups`` must be matched."""

    label: str
    terms: Mapping[str, float]
    anchor_groups: tuple[tuple[str, ...], ...]
    take: int
    title_phrases: Mapping[tuple[str, ...], float] = field(default_factory=dict)


# Two-word descriptions of each finding's problem; a page whose title contains
# one is about that problem (for example an electronic-convergence how-to).
_TITLE_PHRASES: Mapping[str, Mapping[tuple[str, ...], float]] = {
    "electronic_not_converged": {("electronic", "convergence"): 4.0, ("electronic", "minimization"): 2.0},
    "electronic_iteration_limit": {("electronic", "convergence"): 4.0, ("electronic", "minimization"): 2.0},
    "ionic_not_converged": {("structure", "optimization"): 4.0, ("ionic", "relaxation"): 2.0, ("ionic", "minimization"): 2.0},
    "memory_exhaustion": {("memory",): 2.0},
    "smearing_questionable": {("smearing",): 2.0},
}


# Concept vocabulary for each finding type (not page names): these words
# describe the kind of problem, and are weighted below the observed settings.
_CONCEPTS: Mapping[str, tuple[tuple[str, float], ...]] = {
    ELECTRONIC_NOT_CONVERGED: (
        ("electronic", 1.5), ("convergence", 1.5), ("minimization", 1.2), ("mixing", 1.0),
        ("self-consistency", 0.8), ("fails", 0.6),
    ),
    ELECTRONIC_ITERATION_LIMIT: (("electronic", 1.2), ("convergence", 1.2), ("steps", 0.8)),
    IONIC_NOT_CONVERGED: (
        ("relaxation", 1.8), ("ionic", 1.5), ("optimization", 1.5), ("forces", 1.2), ("convergence", 1.2),
    ),
    MEMORY_EXHAUSTION: (("memory", 3.0), ("requirements", 1.0), ("reduce", 0.8)),
    SMEARING_QUESTIONABLE: (("smearing", 2.0), ("gap", 1.0), ("insulators", 1.0), ("semiconductors", 1.0)),
}
_PRIMARY_ANCHORS: Mapping[str, tuple[str, ...]] = {
    ELECTRONIC_NOT_CONVERGED: ("NELM", "ALGO", "IALGO", "AMIX", "BMIX", "electronic"),
    ELECTRONIC_ITERATION_LIMIT: ("NELM",),
    IONIC_NOT_CONVERGED: ("EDIFFG", "NSW", "IBRION", "POTIM", "relaxation"),
    MEMORY_EXHAUSTION: ("memory",),
    SMEARING_QUESTIONABLE: ("ISMEAR", "SIGMA", "smearing"),
}
# Words that make a context passage about the finding's problem, not just
# about the method in general.
_PROBLEM_WORDS: Mapping[str, tuple[str, ...]] = {
    ELECTRONIC_NOT_CONVERGED: ("convergence", "converge", "converged", "NELM", "ALGO", "TIME", "mixing", "minimization"),
    ELECTRONIC_ITERATION_LIMIT: ("convergence", "converge", "NELM"),
    IONIC_NOT_CONVERGED: ("relaxation", "EDIFFG", "forces", "IBRION"),
    MEMORY_EXHAUSTION: ("memory",),
    SMEARING_QUESTIONABLE: ("ISMEAR", "SIGMA", "smearing"),
}
# Settings that may be named in a next step (when also in the cited passage).
_LINKABLE: Mapping[str, tuple[str, ...]] = {
    ELECTRONIC_NOT_CONVERGED: (
        "ALGO", "IALGO", "AMIX", "BMIX", "AMIX_MAG", "BMIX_MAG", "MAXMIX", "LMAXMIX", "TIME", "LSUBROT",
        "NBANDS", "ISMEAR", "SIGMA", "NELM", "EDIFF", "LHFCALC", "LDAU", "MAGMOM", "ISPIN", "METAGGA",
    ),
    ELECTRONIC_ITERATION_LIMIT: ("ALGO", "IALGO", "AMIX", "BMIX", "NELM", "EDIFF"),
    IONIC_NOT_CONVERGED: ("IBRION", "POTIM", "ISIF", "NSW", "EDIFFG"),
    MEMORY_EXHAUSTION: ("NCORE", "NPAR", "KPAR", "LREAL", "NBANDS", "ISYM", "ALGO"),
    SMEARING_QUESTIONABLE: ("ISMEAR", "SIGMA"),
}
_MAX_LINKED_SETTINGS = 3
_SIGNAL_WEIGHTS = {"message": 4.0, "criterion": 1.6, "setting": 1.3, "context": 1.2}


def _queries(finding: DiagnosticFinding) -> tuple[_QuerySpec, ...]:
    """Evidence-weighted sub-queries for one finding.

    The primary query combines the settings and criteria the finding involves
    with the finding type's concept words. When the calculation uses a method
    that changes the relevant documentation (hybrid functional, DFT+U, spin
    polarization, SOC, meta-GGA), a context query requires a passage that names
    that method's settings *and* the finding's problem.
    """

    kind = finding.kind
    if kind == VASP_ERROR_MESSAGE:
        tokens = tuple(signal.name for signal in finding.signals if signal.role == "message")
        return (_QuerySpec("message", {token: 4.0 for token in tokens}, (tokens,), 2),) if tokens else ()
    if kind not in _CONCEPTS:
        return ()
    terms: dict[str, float] = {}
    for signal in finding.signals:
        if signal.role != "context":
            terms[signal.name] = max(terms.get(signal.name, 0.0), _SIGNAL_WEIGHTS[signal.role])
    for concept, weight in _CONCEPTS[kind]:
        terms[concept] = max(terms.get(concept, 0.0), weight)
    if kind == IONIC_NOT_CONVERGED:
        # The sign of the observed EDIFFG selects its meaning (force versus
        # energy criterion), so it selects the matching explanation.
        ediffg = next((signal.value for signal in finding.signals if signal.name == "EDIFFG"), None)
        try:
            value = float(ediffg) if ediffg is not None else None
        except ValueError:
            value = None
        if value is not None and value < 0:
            terms["negative"] = 3.0
            terms["forces"] = max(terms.get("forces", 0.0), 1.5)
        elif value is not None and value > 0:
            terms["positive"] = 3.0
            terms["energy"] = 1.2
    # Practical how-to articles are preferred over reference pages at equal
    # relevance: "howto" matches the upstream "Howto" category only.
    terms["howto"] = 1.5
    context = tuple(signal.name for signal in finding.signals if signal.role == "context")
    specs = [_QuerySpec("primary", terms, (_PRIMARY_ANCHORS[kind],), 2 if context else 3, _TITLE_PHRASES.get(kind, {}))]
    if context:
        context_terms = dict(terms)
        for name in context:
            context_terms[name] = 2.5
        words = {"LHFCALC": ("hybrid",), "HFSCREEN": ("hybrid",), "LDAU": ("LDA+U", "DFT+U"), "ISPIN": ("magnetic",), "MAGMOM": ("magnetic",), "LSORBIT": ("spin-orbit",), "LNONCOLLINEAR": ("noncollinear",), "METAGGA": ("meta-gga",)}
        method_words = tuple(dict.fromkeys(word for name in context for word in words.get(name, ())))
        for word in method_words:
            context_terms[word] = max(context_terms.get(word, 0.0), 1.5)
        specs.append(
            _QuerySpec("context", context_terms, (context + method_words, _PROBLEM_WORDS[kind]), 1)
        )
    return tuple(specs)




def _best_excerpt(
    text: str,
    weights: Mapping[str, float],
    tokenize: Callable[[str], list[str]],
    normal: Callable[[str], str | None],
    *,
    limit: int = 360,
) -> str:
    """The window of consecutive sentences that covers the most query weight,
    verbatim from the (markup-simplified) passage text."""

    sentences = _sentences(text)
    if not sentences:
        return ""
    normalized: dict[str, float] = {}
    for term, weight in weights.items():
        key = normal(term)
        if key:
            normalized[key] = max(normalized.get(key, 0.0), weight)
    best_window = (0, 0)
    best_score = -1.0
    for start in range(len(sentences)):
        end = start
        length = len(sentences[start])
        while end + 1 < len(sentences) and length + 1 + len(sentences[end + 1]) <= limit:
            end += 1
            length += 1 + len(sentences[end])
        covered = {token for sentence in sentences[start : end + 1] for token in tokenize(sentence)}
        score = sum(normalized.get(token, 0.0) for token in covered)
        if score > best_score:
            best_window, best_score = (start, end), score
    excerpt = " ".join(sentences[best_window[0] : best_window[1] + 1])
    if len(excerpt) > limit:
        cut = excerpt.rfind(" ", 0, limit)
        excerpt = excerpt[: cut if cut > 0 else limit].rstrip(" ,;:") + " ..."
    return excerpt


def _linked_settings(finding: DiagnosticFinding, retrieved: Sequence[Any]) -> tuple[str, ...]:
    """Settings both observed in this calculation and named in the first cited passage."""

    observed = {signal.name for signal in finding.signals if signal.role != "message"}
    named: set[str] = set()
    for item in retrieved[:1]:
        named.update(item.passage.tags)
        named.update(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", item.passage.text))
    linked = [name for name in _LINKABLE.get(finding.kind, ()) if name in observed and name in named]
    return tuple(linked[:_MAX_LINKED_SETTINGS])


def _format_settings(finding: DiagnosticFinding, names: Sequence[str]) -> str:
    values = {signal.name: signal.value for signal in finding.signals}
    return ", ".join(f"{name} = {values[name]}" if values.get(name) else name for name in names)


def _next_step(finding: DiagnosticFinding, linked: Sequence[str], has_reference: bool) -> str | None:
    if not has_reference:
        return None
    settings = _format_settings(finding, linked) if linked else ""
    kind = finding.kind
    if kind == VASP_ERROR_MESSAGE:
        return "Read the cited passage, which discusses this message, alongside the log excerpt."
    if kind in (ELECTRONIC_NOT_CONVERGED, ELECTRONIC_ITERATION_LIMIT):
        if settings:
            return (
                f"Compare this run's {settings} with the cited guidance before changing the electronic "
                "minimization settings; BMD Check has not established which setting is responsible."
            )
        return "Read the cited guidance on electronic convergence; BMD Check has not established the cause."
    if kind == IONIC_NOT_CONVERGED:
        if settings:
            return (
                f"Compare this run's {settings} with the cited guidance; whether more ionic steps alone "
                "would reach the force criterion is not established."
            )
        return "Read the cited guidance on structure relaxation; the cause is not established."
    if kind == MEMORY_EXHAUSTION:
        if settings:
            return f"Compare this run's {settings} with the cited guidance on reducing memory per core."
        if not finding.incar_available:
            return (
                "The cited guidance describes settings that reduce memory per core; the executed INCAR of "
                "this stage is not available, so which of them were set is not known."
            )
        return (
            "The cited guidance describes settings that reduce memory per core; none of them is set in "
            "this stage's INCAR, so the defaults apply."
        )
    if kind == SMEARING_QUESTIONABLE:
        return (
            f"Compare {settings or 'the smearing settings'} with the cited guidance for this kind of system; "
            "the electronic-character verdict itself is unchanged."
        )
    return None


_EXAMPLE_CATEGORY_PENALTY = 0.5


def guidance_for_findings(
    findings: Sequence[DiagnosticFinding],
    index: Any,
    *,
    references_per_finding: int = 3,
) -> tuple[FindingGuidance, ...]:
    """Retrieve, excerpt and link references for each finding (deterministic).

    Each finding is searched with its own stage's settings: passages about a
    method that stage does not use are excluded (``passage_method_domains``).
    """

    from bmd_agent import vasp_manual_index as manual

    results: list[FindingGuidance] = []
    for finding in findings:
        active = frozenset(_finding_domains(finding))

        def unrelated_method(passage: Any, active: frozenset[str] = active) -> bool:
            return not passage_method_domains(passage.text, title=passage.title, heading=passage.section) <= active

        specs = _queries(finding)
        if not specs:
            limitation = (
                "This finding concerns the workflow rather than VASP settings; no manual passage is cited."
                if finding.kind == STAGE_NOT_STARTED
                else None
            )
            results.append(FindingGuidance(finding, limitation=limitation))
            continue
        chosen: list[tuple[Any, _QuerySpec]] = []
        pages: set[str] = set()
        for spec in specs:
            found = index.search(
                spec.terms,
                required_groups=spec.anchor_groups,
                limit=spec.take + references_per_finding,
                per_page=1,
                exclude=unrelated_method,
                penalized_categories=("examples",),
                category_penalty_factor=_EXAMPLE_CATEGORY_PENALTY,
                title_phrases=spec.title_phrases,
            )
            taken = 0
            for item in found:
                if item.passage.page_id in pages or taken >= spec.take:
                    continue
                chosen.append((item, spec))
                pages.add(item.passage.page_id)
                taken += 1
        chosen = chosen[:references_per_finding]
        references = tuple(
            ManualReference(
                title=item.passage.title,
                page_id=item.passage.page_id,
                revision_id=item.passage.revision_id,
                permalink=item.passage.permalink,
                section=item.passage.section,
                excerpt=_best_excerpt(item.passage.text, spec.terms, manual.tokenize, manual.normalize_term),
                matched_terms=item.matched_terms,
                score=item.score,
                passage_position=item.passage.position,
            )
            for item, spec in chosen
        )
        retrieved = [item for item, _ in chosen]
        linked = _linked_settings(finding, retrieved)
        limitation = None
        if not references:
            limitation = (
                "No passage of the frozen VASP Wiki mentions this message."
                if finding.kind == VASP_ERROR_MESSAGE
                else "No sufficiently specific VASP Wiki passage was found for this finding."
            )
        results.append(
            FindingGuidance(
                finding,
                references=references,
                next_step=_next_step(finding, linked, bool(references)),
                linked_settings=linked,
                limitation=limitation,
            )
        )
    return tuple(results)


def build_manual_guidance(
    analysis: Any,
    *,
    electronic_character: Any | None = None,
    index_loader: Callable[[], Any] | None = None,
) -> ManualGuidance:
    """Findings plus VASP Wiki references for one lifecycle analysis.

    The corpus is loaded only when there is at least one finding. An
    unavailable corpus never suppresses findings: they are returned without
    references, with the reason.
    """

    findings = derive_findings(analysis, electronic_character=electronic_character)
    if not findings:
        return ManualGuidance(GUIDANCE_NO_FINDINGS)
    if index_loader is None:
        from bmd_agent.vasp_manual_index import load_manual_index as index_loader  # lazy: corpus only when needed
    from bmd_agent.vasp_manual_index import ManualIndexUnavailable

    index = index_loader()
    if isinstance(index, ManualIndexUnavailable):
        return ManualGuidance(
            GUIDANCE_UNAVAILABLE,
            findings=tuple(FindingGuidance(finding) for finding in findings),
            unavailable_reason=f"{index.reason}: {index.detail}",
        )
    return ManualGuidance(
        GUIDANCE_AVAILABLE,
        findings=guidance_for_findings(findings, index),
        corpus_version=index.corpus_version,
        corpus_digest=index.corpus_digest,
        index_identity=index.index_identity,
        limitations=(
            "Passages are excerpts of the preserved VASP Wiki wikitext with markup simplified; "
            "content produced only by upstream template expansion is not available.",
        ),
    )
