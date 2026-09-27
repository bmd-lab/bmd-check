from __future__ import annotations

import copy
import json
from pathlib import Path

from bmd_agent import cli
from bmd_agent.presentation import (
    build_job_concise_summary,
    build_lifecycle_concise_summary,
    render_concise_summary,
)
from bmd_agent.resources.bmdex import (
    BmdexContextualAssessment,
    BmdexContextualReferenceRecord,
    BmdexDomainContextEnrichment,
    BmdexDomainContextEvidence,
)
from bmd_agent.resources.completion import (
    STAGE_CONVERGED,
    StageCompletion,
    StageConvergence,
)
from bmd_agent.resources.custodian import (
    ConfiguredCustodianComponent,
    CustodianPolicyEvidence,
    CustodianStagePolicy,
    TerminationEvidenceAssessment,
    parse_custodian_json,
)
from bmd_agent.resources.lifecycle import (
    BmdWorkflowDiscovery,
    LifecycleAnalysis,
    LifecycleState,
    LocalStageEvidence,
)
from bmd_agent.resources.oom import (
    INSUFFICIENT_OOM_EVIDENCE,
    NO_OOM_EVIDENCE,
    OOM_ESTABLISHED,
    MemoryObservation,
    OomDiagnosticEvidence,
)
from bmd_agent.resources.run import (
    AttemptStateObservation,
    ComparisonObservation,
    DirectVaspInspection,
    IonicStepObservation,
    JobInspection,
    LogRuntimeObservation,
    PathObservation,
    RunDiagnosis,
    RunInspection,
    ScientificResult,
    StageTrajectoryObservation,
    StructureObservation,
    TerminationObservation,
    WorkflowStage,
)
from bmd_agent.resources.slurm import SlurmAccountingRecord


def scheduler(
    *,
    state: str = "COMPLETED",
    exit_code: str = "0:0",
    reason: str | None = None,
) -> SlurmAccountingRecord:
    return SlurmAccountingRecord(
        job_id="21906221",
        name="fixture",
        state=state,
        elapsed="01:00:00",
        start="2026-01-01T00:00:00",
        end="2026-01-01T01:00:00",
        partition="compute",
        exit_code=exit_code,
        reason=reason,
    )


def trajectory(
    *,
    stage_index: int = 1,
    stage_type: str = "static",
    theory: str = "pbe",
    completed_ionic_steps: int = 0,
    incomplete_iterations: int | None = None,
    converged_electronic: bool | None = None,
    converged_ionic: bool | None = None,
    vasprun_error: str | None = None,
    ionic_steps: tuple[IonicStepObservation, ...] = (),
    criteria: dict | None = None,
) -> StageTrajectoryObservation:
    return StageTrajectoryObservation(
        stage_index=stage_index,
        stage_label=f"stage_{stage_index:02d}",
        stage_type=stage_type,
        theory=theory,
        directory=f"/flow/stage_{stage_index:02d}",
        oszicar_present=True,
        vasprun_present=vasprun_error is not None or converged_electronic is not None,
        vasprun_error=vasprun_error,
        completed_ionic_steps=completed_ionic_steps,
        incomplete_electronic_iteration_count=incomplete_iterations,
        converged_electronic=converged_electronic,
        converged_ionic=converged_ionic,
        ionic_steps=ionic_steps,
        criteria=criteria or {},
    )


def scientific(
    *,
    formula: str = "Si",
    sites: int = 2,
    converged: bool | None = True,
    energy: float | None = -12.579584,
    band_gap: float | None = 1.232,
) -> ScientificResult:
    return ScientificResult(
        source_paths=("/flow/vasprun.xml",),
        final_formula=formula,
        final_energy_ev=energy,
        electronic_convergence=converged,
        band_gap_ev=band_gap,
        structure=StructureObservation(
            source_path="/flow/CONTCAR",
            formula=formula,
            reduced_formula=formula,
            site_count=sites,
        ),
    )


def bmd_job(
    *,
    stages: tuple[WorkflowStage, ...] | None = None,
    trajectories: tuple[StageTrajectoryObservation, ...] | None = None,
    scheduler_record: SlurmAccountingRecord | None = None,
    result: ScientificResult | None = None,
    termination: TerminationEvidenceAssessment | None = None,
    custodian=(),
    policy: CustodianPolicyEvidence | None = None,
    oom: OomDiagnosticEvidence | None = None,
) -> JobInspection:
    stages = stages or (WorkflowStage(1, "static", "hse06", ("soc",), "stage_01"),)
    trajectories = trajectories or (
        trajectory(stage_type=stages[0].stage_type, theory=stages[0].theory, converged_electronic=True),
    )
    scheduler_record = scheduler_record or scheduler()
    result = result or scientific()
    run = RunInspection(
        flow_root="/flow",
        submission_path="/flow/submission.json",
        workflow_stages=stages,
        stage_directories=(),
        result_directory=PathObservation("result_dir", "/flow", "directory", True, "producer_provenance"),
        log_paths=(),
        final_artifacts=(),
        producer_git={"git_commit": "a" * 40, "state": "clean"},
        cluster_request={},
        resources_request={},
        environment_policy={},
        attempt_state=AttemptStateObservation(None, False),
        job_id="21906221",
        scheduler=scheduler_record,
        scheduler_error=None,
        runtime=LogRuntimeObservation("log_observation", (), None, {}, {}, {}),
        scientific=result,
        comparison=ComparisonObservation("unavailable", "agent_comparison"),
        custodian_policy=policy
        or CustodianPolicyEvidence(False, reason="submission has no persisted policy"),
        custodian_evidence=tuple(custodian),
    )
    diagnosis = RunDiagnosis(
        inspection=run,
        termination=TerminationObservation(
            scheduler_state=scheduler_record.state,
            scheduler_exit_code=scheduler_record.exit_code,
            assessment=termination,
        ),
        trajectories=trajectories,
    )
    return JobInspection(
        job_id="21906221",
        scheduler=scheduler_record,
        scheduler_error=None,
        scheduler_work_dir="/flow",
        calculation_directory="/flow",
        calculation_type="BMD Compute",
        calculation_reason=None,
        bmd_compute=diagnosis,
        oom=oom,
    )


def contextual_enrichment(
    statement: str,
    *,
    input_tags: dict | None = None,
) -> BmdexDomainContextEnrichment:
    record = BmdexContextualReferenceRecord(
        record={
            "id": "vasp.hybrid.test",
            "title": "Hybrid iteration context",
            "contextual_statement": statement,
            "applicability": {"calculation_family": "hybrid_functional"},
            "diagnostic_relevance": "Context only.",
            "limitations": ["Does not diagnose a specific run."],
            "sources": [],
            "record_provenance": {},
        },
        match={"matched_fields": ["calculation_family"]},
    )
    return BmdexDomainContextEnrichment(
        query={
            "calculation_family": "hybrid_functional",
            "input_tags": input_tags or {},
        },
        evidence=BmdexDomainContextEvidence(
            payload={
                "schema_version": 1,
                "evidence_type": "contextual_reference_evidence",
                "producer": {},
                "query": {},
            },
            records=(record,),
        ),
        assessment=BmdexContextualAssessment(
            basis=("The producer-owned record is applicable to the observed run.",),
            limitations=(
                "Contextual reference evidence does not establish a hang or a method incompatibility by itself.",
            ),
            source_record_ids=(record.record_id,),
        ),
    )


def normalized(text: str) -> str:
    return " ".join(text.split())


def rendered_job(job: JobInspection, **kwargs) -> str:
    return render_concise_summary(
        build_job_concise_summary(
            job,
            detailed_evidence_command="bmd-agent 21906221 --verbose",
            **kwargs,
        )
    )


def test_completed_static_calculation_is_especially_concise() -> None:
    output = rendered_job(bmd_job())

    assert "HSE06 Static Energy + SOC" in output
    assert "Status: COMPLETED" in output
    assert "Si\n2 atoms" in output
    assert "Electronic convergence: reached" in output
    assert "Final energy: -12.579584 eV" in output
    assert "Band gap: 1.232 eV" in output
    assert "SLURM recorded a successful exit." in output
    assert "Convergence criteria were met." in output
    assert "No execution problems were detected." not in output
    assert "What happened" not in output


def test_completed_multi_stage_workflow_preserves_compact_stage_statuses() -> None:
    stages = (
        WorkflowStage(1, "relax", "pbe", (), "stage_01"),
        WorkflowStage(2, "static", "hse06", (), "stage_02"),
        WorkflowStage(3, "dos", "pbe", (), "stage_03"),
    )
    trajectories = (
        trajectory(stage_index=1, stage_type="relax", converged_electronic=True, converged_ionic=True),
        trajectory(stage_index=2, stage_type="static", theory="hse06", converged_electronic=True),
        trajectory(stage_index=3, stage_type="dos", converged_electronic=True),
    )

    output = rendered_job(bmd_job(stages=stages, trajectories=trajectories))

    assert "PBE Geometry Optimisation -> HSE06 Static Energy -> PBE DOS" in output
    assert output.count("COMPLETED") == 4


def test_completed_si_hse_path_summary_uses_producer_workflow_identity() -> None:
    root = Path("/calculation/hse06_dos_completed")
    workflow_stages = (
        {"index": 1, "stage_type": "relax", "theory": "pbe", "modifiers": []},
        {"index": 2, "stage_type": "static", "theory": "hse06", "modifiers": []},
        {"index": 3, "stage_type": "dos", "theory": "hse06", "modifiers": []},
    )
    stage_evidence = tuple(
        LocalStageEvidence(
            label=f"stage_{index:02d}",
            path=root / f"stage_{index:02d}",
            stage_index=index,
            producer_path=f"/producer/stage_{index:02d}",
            has_required_inputs=True,
            has_meaningful_execution=True,
            normal_completion=True,
        )
        for index in range(1, 4)
    )
    workflow = BmdWorkflowDiscovery(
        workflow_root=root,
        submission_path=root / "submission.json",
        submission={},
        workflow_stages=workflow_stages,
        stage_bindings=(),
        stage_evidence=stage_evidence,
    )
    analysis = LifecycleAnalysis(
        state=LifecycleState.COMPLETED,
        directory=root,
        calculation_kind="BMD Compute",
        message="complete",
        bmd_workflow=workflow,
        scientific=scientific(),
        stage_completion=tuple(
            StageCompletion(
                index=index,
                label=f"stage {index}",
                started=True,
                terminated_normally=True,
                convergence=StageConvergence(STAGE_CONVERGED),
            )
            for index in range(1, 4)
        ),
    )

    output = render_concise_summary(build_lifecycle_concise_summary(analysis))

    assert "PBE Geometry Optimisation -> HSE06 Static Energy -> HSE06 DOS" in output
    assert "Status: COMPLETED" in output
    assert "Stage 1  COMPLETED" in output
    assert "Stage 2  COMPLETED" in output
    assert "Stage 3  COMPLETED" in output
    assert "Si\n2 atoms" in output
    assert "Electronic convergence: reached" in output
    assert "Final energy: -12.579584 eV" in output
    assert "Band gap: 1.232 eV" in output
    assert "VASP recorded normal termination in every workflow stage." in output
    assert "Convergence criteria were met in every workflow stage." in output
    assert "No execution problems were detected." not in output
    assert "SOC" not in output
    assert "What happened" not in output
    assert "Why this may have happened" not in output
    assert "Assessment" not in output
    assert "vasprun" not in output
    assert "provenance" not in output


def test_running_job_reports_progress_without_calling_it_failed() -> None:
    job = bmd_job(
        scheduler_record=scheduler(state="RUNNING", exit_code="0:0"),
        trajectories=(
            trajectory(
                stage_type="relax",
                completed_ionic_steps=14,
                incomplete_iterations=2,
                converged_electronic=True,
            ),
        ),
    )

    output = rendered_job(job)

    assert "Status: RUNNING" in output
    assert "14 ionic steps were completed." in output
    assert "2 electronic iterations have been observed in the current electronic cycle." in normalized(output)
    assert "before the calculation was interrupted" not in output
    assert "Electronic convergence was reached." in output
    assert "No execution error has been established." in output
    assert "Status: FAILED" not in output


def test_pending_job_is_minimal_and_includes_useful_queue_reason() -> None:
    job = bmd_job(scheduler_record=scheduler(state="PENDING", reason="Priority"))

    output = rendered_job(job)

    assert "Status: PENDING" in output
    assert "waiting in the SLURM queue" in output
    assert "Queue reason: Priority." in output
    assert "Progress" not in output


def test_historical_21906221_fixture_renders_supported_custodian_failure(capsys) -> None:
    job_fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "job_21906221.json").read_text(
            encoding="utf-8"
        )
    )
    stage_payload = job_fixture["submission_spec"]["flow_spec"]["workflow_spec"][
        "stages"
    ][0]
    stage = WorkflowStage(
        1,
        stage_payload["stage_type"],
        stage_payload["theory"],
        tuple(stage_payload["modifiers"]),
        stage_payload["label"],
        stage_payload["options"],
    )
    assert job_fixture["job_id"] == "21906221"
    assert job_fixture["resources"]["mem_gb"] == 192
    evidence = parse_custodian_json(
        (Path(__file__).parent / "fixtures" / "custodian_frozen_repeated.json").read_text(
            encoding="utf-8"
        ),
        source_path="/flow/custodian.json",
    )
    termination = TerminationEvidenceAssessment(
        classification="custodian_triggered_process_termination",
        status="supported",
        basis=("5 correction records", "SIGTERM evidence", "5 error archives"),
        limitations=(
            "the evidence does not prove the interrupted VASP operation would eventually converge",
        ),
    )
    oom = OomDiagnosticEvidence(
        assessment=NO_OOM_EVIDENCE,
        maximum_rss=MemoryObservation(
            "SLURM batch step maximum RSS",
            "45045764K",
            45_045_764 * 1024,
            "task_max",
        ),
        requested_memory=MemoryObservation(
            "SLURM requested memory",
            "192G",
            192 * 1024**3,
            "job_allocation",
        ),
    )
    job = bmd_job(
        stages=(stage,),
        scheduler_record=scheduler(state="FAILED", exit_code="1:0"),
        trajectories=(
            trajectory(
                incomplete_iterations=4,
                converged_electronic=False,
                vasprun_error="file could not be parsed completely",
            ),
        ),
        termination=termination,
        custodian=(evidence,),
        oom=oom,
    )
    enrichment = contextual_enrichment(
        "VASP hybrid-functional and HF-type calculations include nonlocal Hartree-Fock "
        "exchange, whose evaluation can be much more expensive than semilocal DFT. "
        "In starts where NELMDL applies, VASP may first perform non-self-consistent "
        "blocked-Davidson delay steps with a fixed Hamiltonian; for HF-type calculations "
        "with NELMDL >= 3, these delay steps use the corresponding semilocal local "
        "Hamiltonian. When the calculation leaves that delay and begins evaluating the "
        "hybrid/HF exact-exchange workload, the wall time per electronic step can "
        "increase dramatically."
    )

    output = rendered_job(job, contextual_enrichment=enrichment)
    plain = normalized(output)

    assert "HSE06 Static Energy + SOC" in output
    assert "Status: FAILED" in output
    assert "Automatic frozen-job protection intervened 5 times" in plain
    assert "after 6 hours passed without new VASP output" in plain
    assert (
        "Each intervention also applied the same automatic VASP setting correction."
        in plain
    )
    assert "INCAR.SYMPREC" not in output
    assert "1e-08" not in output
    assert "VASP hybrid-functional and HF-type calculations include" in plain
    assert "NELMDL" not in output
    assert "When the calculation leaves that delay" not in output
    assert (
        "A long period without new VASP output does not, by itself, establish that the "
        "calculation is frozen."
        in plain
    )
    assert "This is contextual scientific evidence" not in output
    assert "No evidence of an out-of-memory failure was found." in output
    assert "Peak observed use was about 43 GB of 192 GB allocated." in output
    assert (
        "4 electronic iterations were observed before the calculation was interrupted."
        in plain
    )
    assert "Electronic convergence was not reached." in output
    assert "immediate cause of termination" in plain
    assert "It is not known whether the calculation would eventually have converged." in plain
    assert "historical submission does not record the execution policy" not in output
    assert "vasprun.xml" not in output
    assert "file could not be parsed completely" not in output
    assert "bmd-agent 21906221 --verbose" in output
    for internal_name in (
        "termination_observation",
        "oom_diagnostic_evidence",
        "contextual_reference_evidence",
        "producer_provenance",
        "convergence_progress_assessment",
        "custodian_triggered_process_termination",
        "max_errors_per_job",
        "FrozenJobErrorHandler",
    ):
        assert internal_name not in output
    assert len(output.splitlines()) <= 45

    cli.print_job_inspection(job, contextual_enrichment=enrichment)
    verbose = capsys.readouterr().out
    assert "repeated correction: INCAR.SYMPREC -> 1e-08" in verbose
    assert "submission has no persisted policy" in verbose
    assert (
        "vasprun trajectory enrichment unavailable: file could not be parsed completely"
        in verbose
    )
    assert "NELMDL applies" in verbose


def test_concise_context_is_extractive_and_does_not_mutate_evidence() -> None:
    job = bmd_job(
        scheduler_record=scheduler(state="FAILED", exit_code="1:0"),
        termination=TerminationEvidenceAssessment("unknown", "insufficient_evidence"),
    )
    enrichment = contextual_enrichment(
        "EXTRA_PARAM detail is producer-specific. Producer-owned concise explanation."
    )
    original_job = copy.deepcopy(job)
    original_enrichment = copy.deepcopy(enrichment)

    output = rendered_job(job, contextual_enrichment=enrichment)

    assert "Producer-owned concise explanation." in output
    assert "EXTRA_PARAM" not in output
    assert "exact exchange" not in output.lower()
    assert job == original_job
    assert enrichment == original_enrichment


def test_supported_interruption_keeps_decisive_convergence_uncertainty() -> None:
    job = bmd_job(
        scheduler_record=scheduler(state="TIMEOUT", exit_code="0:0"),
        trajectories=(
            trajectory(
                incomplete_iterations=4,
                vasprun_error="file could not be parsed completely",
            ),
        ),
        result=ScientificResult(
            source_paths=("/flow/vasprun.xml",),
            error="file could not be parsed completely",
        ),
        termination=TerminationEvidenceAssessment(
            "slurm_timeout",
            "supported",
            basis=("scheduler reports TIMEOUT",),
        ),
    )

    output = rendered_job(job)
    plain = normalized(output)

    assert "4 electronic iterations were observed before the calculation was interrupted." in plain
    assert (
        "Agent could not determine whether electronic convergence was reached from the "
        "available results."
        in plain
    )
    assert "vasprun.xml" not in output
    assert "file could not be parsed completely" not in output


def test_established_oom_is_explained_without_changing_termination_logic() -> None:
    job = bmd_job(
        scheduler_record=scheduler(state="OUT_OF_MEMORY", exit_code="0:125"),
        termination=TerminationEvidenceAssessment(
            "slurm_out_of_memory",
            "supported",
            basis=("SLURM accounting reports OUT_OF_MEMORY",),
        ),
        oom=OomDiagnosticEvidence(assessment=OOM_ESTABLISHED),
    )

    output = rendered_job(job)

    assert "Status: FAILED" in output
    assert "SLURM reported that the job stopped because it ran out of memory." in output
    assert "Evidence indicates an out-of-memory failure." in output


def test_failed_job_with_insufficient_evidence_stays_uncertain() -> None:
    job = bmd_job(
        scheduler_record=scheduler(state="FAILED", exit_code="1:0"),
        termination=TerminationEvidenceAssessment(
            "unknown",
            "insufficient_evidence",
            limitations=("no decisive termination evidence was available",),
        ),
        oom=OomDiagnosticEvidence(assessment=INSUFFICIENT_OOM_EVIDENCE),
    )

    output = rendered_job(job)

    assert "SLURM recorded the job as FAILED." in output
    assert "Agent could not determine the cause from the available evidence." in output
    assert "Available evidence was insufficient to assess memory exhaustion." in output


def test_non_hybrid_failure_has_no_hybrid_context_without_matching_enrichment() -> None:
    job = bmd_job(
        stages=(WorkflowStage(1, "static", "pbe", (), "stage_01"),),
        scheduler_record=scheduler(state="FAILED", exit_code="1:0"),
        termination=TerminationEvidenceAssessment("unknown", "insufficient_evidence"),
    )

    output = rendered_job(job)

    assert "hybrid" not in output.lower()
    assert "Why this may have happened" not in output


def test_geometry_relaxation_progress_uses_existing_force_evidence() -> None:
    ionic_steps = (
        IonicStepObservation(1, max_force=0.2, max_force_source="OUTCAR"),
        IonicStepObservation(2, max_force=0.071891, max_force_source="OUTCAR"),
    )
    job = bmd_job(
        stages=(WorkflowStage(1, "relax", "pbe", (), "stage_01"),),
        scheduler_record=scheduler(state="FAILED", exit_code="1:0"),
        trajectories=(
            trajectory(
                stage_type="relax",
                completed_ionic_steps=2,
                converged_electronic=True,
                converged_ionic=False,
                ionic_steps=ionic_steps,
                criteria={"EDIFFG": -0.01, "NSW": 100},
            ),
        ),
    )

    output = rendered_job(job)
    plain = normalized(output)

    assert "2 ionic steps were completed." in output
    assert "Ionic convergence was not reached." in output
    assert "0.0719 eV/A" in plain
    assert "0.0100 eV/A" in plain


def test_current_policy_does_not_emit_historical_policy_gap() -> None:
    evidence = parse_custodian_json(
        (Path(__file__).parent / "fixtures" / "custodian_frozen_repeated.json").read_text(
            encoding="utf-8"
        ),
        source_path="/flow/custodian.json",
    )
    policy = CustodianPolicyEvidence(
        True,
        stages=(
            CustodianStagePolicy(
                stage_index=1,
                policy_id="policy",
                policy_version=1,
                stage_type="static",
                theory="hse06",
                handlers=(ConfiguredCustodianComponent("FrozenJobErrorHandler"),),
                explicit_handler_exclusions=(),
                vasp_error_exclusions=(),
                validators=(),
                validators_source=None,
                validators_explicit_override=None,
                walltime_authority=None,
                walltime_handler=None,
                custodian_version=None,
                implementation_source=None,
                rationale=None,
            ),
        ),
    )
    job = bmd_job(
        scheduler_record=scheduler(state="FAILED", exit_code="1:0"),
        termination=TerminationEvidenceAssessment(
            "custodian_triggered_process_termination",
            "supported",
        ),
        custodian=(evidence,),
        policy=policy,
    )

    output = rendered_job(job)

    assert "historical submission does not record" not in output


def test_manual_direct_vasp_and_partial_outputs_remain_neutral() -> None:
    direct = DirectVaspInspection(
        directory="/manual",
        artifacts=(),
        executed_inputs=(),
        scientific=ScientificResult(
            source_paths=("/manual/vasprun.xml",),
            error="file could not be parsed completely",
        ),
        trajectory=trajectory(
            stage_type="direct_vasp",
            theory="unknown",
            incomplete_iterations=3,
            vasprun_error="file could not be parsed completely",
        ),
        assessments=(),
    )
    job = JobInspection(
        job_id="42",
        scheduler=scheduler(state="FAILED", exit_code="1:0"),
        scheduler_error=None,
        scheduler_work_dir="/manual",
        calculation_directory="/manual",
        calculation_type="direct VASP",
        calculation_reason=None,
        direct_vasp=direct,
        oom=OomDiagnosticEvidence(assessment=INSUFFICIENT_OOM_EVIDENCE),
    )

    output = render_concise_summary(build_job_concise_summary(job))

    assert "Direct VASP calculation" in output
    assert "Status: FAILED" in output
    assert "3 observed iterations" in output
    assert "Agent could not determine the cause" in output
    assert (
        "Agent could not determine whether electronic convergence was reached from the "
        "available results."
        in normalized(output)
    )
    assert "vasprun.xml" not in output
    assert "file could not be parsed completely" not in output
    assert "list index out of range" not in output


def test_failed_multi_stage_summary_focuses_on_active_stage() -> None:
    stages = (
        WorkflowStage(1, "relax", "pbe", (), "stage_01"),
        WorkflowStage(2, "static", "hse06", (), "stage_02"),
        WorkflowStage(3, "dos", "pbe", (), "stage_03"),
    )
    trajectories = (
        trajectory(stage_index=1, stage_type="relax", converged_electronic=True, converged_ionic=True),
        trajectory(
            stage_index=2,
            stage_type="static",
            theory="hse06",
            incomplete_iterations=7,
            converged_electronic=False,
        ),
        StageTrajectoryObservation(
            stage_index=3,
            stage_label="stage_03",
            stage_type="dos",
            theory="pbe",
            directory="/flow/stage_03",
        ),
    )
    job = bmd_job(
        stages=stages,
        trajectories=trajectories,
        scheduler_record=scheduler(state="FAILED", exit_code="1:0"),
    )

    output = rendered_job(job)

    assert "Stage 1  COMPLETED" in output
    assert "Stage 2  FAILED" in output
    assert "Stage 3  NOT STARTED" in output
    assert "incomplete electronic cycle contains 7 observed iterations" in output
    assert "before the calculation was interrupted" not in output


def test_lifecycle_summary_is_structured_and_uses_caller_detail_hint() -> None:
    analysis = LifecycleAnalysis(
        state=LifecycleState.PRE_RUN,
        directory=Path("/calculation"),
        calculation_kind="direct VASP",
        message="inputs ready",
    )

    summary = build_lifecycle_concise_summary(
        analysis,
        detailed_evidence_command="bmd-agent /calculation --verbose",
    )
    output = render_concise_summary(summary)

    assert summary.status == "PRE_RUN"
    assert "required calculation inputs are present" in output
    assert "bmd-agent /calculation --verbose" in output


def test_relocated_bmd_snapshot_keeps_current_and_original_locations() -> None:
    workflow = BmdWorkflowDiscovery(
        workflow_root=Path("/current/copy"),
        submission_path=Path("/current/copy/submission.json"),
        submission={},
        workflow_stages=(
            {"stage_type": "static", "theory": "pbe", "modifiers": []},
        ),
        stage_bindings=(),
        producer_root="/bmd-db/guest/flows/original",
        relocated=True,
    )
    analysis = LifecycleAnalysis(
        state=LifecycleState.UNKNOWN,
        directory=Path("/current/copy"),
        calculation_kind="BMD Compute",
        message="relocated partial snapshot",
        bmd_workflow=workflow,
    )

    output = render_concise_summary(build_lifecycle_concise_summary(analysis))

    assert f"Current acquisition directory: {workflow.workflow_root}" in output
    assert "Original BMD Compute run directory: /bmd-db/guest/flows/original" in output


def test_presentation_module_has_no_acquisition_or_action_plane() -> None:
    source = Path("src/bmd_agent/presentation.py").read_text(encoding="utf-8")

    forbidden = (
        "subprocess",
        "inspect_slurm_job",
        "analyze_calculation_directory(",
        "open(",
        ".read_text(",
        "ssh",
        "sacct",
        "sbatch",
        "scancel",
        "scontrol",
        "POTCAR",
    )
    assert not [token for token in forbidden if token in source]
