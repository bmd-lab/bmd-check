from collections.abc import Callable, Iterable, Mapping
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from bmd_agent.config import (
    ConfigurationError,
    GitRepositoryResource,
    ResourceRegistry,
    SlurmClusterResource,
    load_resources,
)
from bmd_agent.deployment import DeploymentContext, resolve_deployment_context
from bmd_agent.profiling import (
    PerformanceProfile,
    PerformanceProfiler,
    profile_phase,
    profiled_runner,
)
from bmd_agent.presentation import (
    build_job_concise_summary,
    build_lifecycle_concise_summary,
    render_concise_summary,
)
from bmd_agent.resources.bmdex import (
    BmdexDomainContextEnrichment,
    bmdex_repository,
    enrich_job_with_bmdex_domain_context,
    enrich_lifecycle_with_bmdex_domain_context,
)
from bmd_agent.resources.compute import (
    COMPUTE_POLICY_NOT_CONFIGURED,
    POLICY_ALIGNMENT_DIFFERENT,
    POLICY_ALIGNMENT_SAME,
    POLICY_ALIGNMENT_SAME_UNVERIFIED,
    ComputeCapabilities,
    ComputeCapabilityError,
    ComputePolicyObservation,
    compute_policy_alignment,
    compute_policy_not_configured,
    inspect_compute_capabilities,
    observe_compute_policies,
)
from bmd_agent.resources.git import GitInspection, inspect_repository
from bmd_agent.resources.input_check import (
    InputCheckObservation,
    check_remote_input_directory,
)
from bmd_agent.resources.lifecycle import (
    LifecycleAnalysis,
    LifecycleState,
    analyze_calculation_directory,
)
from bmd_agent.resources.oom import MemoryObservation, OomDiagnosticEvidence
from bmd_agent.resources.remote import ReusableSshSession
from bmd_agent.resources.run import (
    PRODUCER_REQUESTED,
    JobInspection,
    RunDiagnosis,
    RunInspection,
    RunComparison,
    RunInspectionError,
    compare_remote_runs,
    derive_trajectory_progress_evidence,
    diagnose_remote_run,
    inspect_slurm_job,
    inspect_remote_run,
    serialize_job_trajectory_evidence,
)
from bmd_agent.resources.slurm import describe_exit_code, get_job_accounting, get_queue
from bmd_agent.resources.vasp import (
    RemotePathError,
    read_remote_structure,
    remote_acquisition_cache,
)


_EXECUTED_INPUT_DISPLAY_KEYS = (
    "ENCUT",
    "EDIFF",
    "EDIFFG",
    "NSW",
    "ISIF",
    "IBRION",
    "ISMEAR",
    "SIGMA",
    "ISPIN",
    "MAGMOM",
    "LDAU",
    "LDAUTYPE",
    "LDAUL",
    "LDAUU",
    "LDAUJ",
    "LMAXMIX",
    "LHFCALC",
    "HFSCREEN",
    "AEXX",
    "ALGO",
    "PRECFOCK",
    "LSORBIT",
    "LNONCOLLINEAR",
    "SAXIS",
    "ISYM",
    "GGA_COMPAT",
    "IVDW",
    "NCORE",
    "NBANDS",
    "LWAVE",
    "LCHARG",
)


def show_repository(inspection: GitInspection) -> None:
    """Display read-only information about a Git repository inspection."""

    print(inspection.name)
    print(f"  path:   {inspection.path}")

    if not inspection.exists:
        print("  exists: no")
        print()
        return

    print("  exists: yes")

    if inspection.error:
        print(f"  git:    inspection failed: {inspection.error}")
        print()
        return

    print(f"  branch: {inspection.branch}")
    print(f"  commit: {inspection.commit[:12] if inspection.commit else ''}")

    if inspection.modified:
        print("  state:  modified")
        print("  changes:")

        for line in inspection.status_lines:
            print(f"    {line}")
    else:
        print("  state:  clean")

    print()


def show_status(registry: ResourceRegistry | None = None) -> int:
    """Display the state of configured BMD repositories."""

    registry = registry or load_resources()

    print("BMD Agent")
    print("=========")
    print()

    for repository in registry.repositories.values():
        show_repository(inspect_repository(repository))

    for cluster in registry.clusters.values():
        show_deployment_context(
            resolve_deployment_context(registry, cluster_key=cluster.key)
        )

    return 0


def show_deployment_context(context: DeploymentContext) -> None:
    """Display configured acquisition context without presenting it as live evidence."""

    cluster = context.cluster
    if cluster is None:
        return

    print("Deployment context:")
    print(f"  acquisition resource: {cluster.name} ({cluster.key})")
    print(f"  SSH alias: {cluster.ssh_host}")
    if context.profile is None:
        print("  profile: unknown/generic")
    else:
        profile = context.profile
        print(f"  profile: {profile.deployment_id} ({profile.name})")
        print(f"  profile schema: {profile.schema} v{profile.schema_version}")
        print(f"  expected scheduler: {profile.expected.scheduler}")
    print()


def show_queue(registry: ResourceRegistry | None = None) -> int:
    """Display a summary of the configured BMD SLURM queue."""

    registry = registry or load_resources()
    cluster = powerslurm_cluster(registry)

    print("BMD PowerSLURM Queue")
    print("====================")
    print()

    try:
        jobs = get_queue(
            ssh_host=cluster.ssh_host,
            partition=cluster.partition,
            timeout=cluster.remote_command_timeout_seconds,
            ssh_connect_timeout=cluster.ssh_connect_timeout_seconds,
        )

    except subprocess.TimeoutExpired:
        print("PowerSLURM connection timed out.")
        return 1

    except subprocess.CalledProcessError as exc:
        print("Unable to inspect PowerSLURM.")

        if exc.stderr:
            print(exc.stderr.strip())

        return 1

    if not jobs:
        print(f"No jobs in {cluster.partition}.")
        return 0

    states: dict[str, int] = {}
    users: dict[str, int] = {}

    for job in jobs:
        states[job.state] = states.get(job.state, 0) + 1
        users[job.user] = users.get(job.user, 0) + 1

    print(f"Total jobs: {len(jobs)}")
    print()

    print("States:")

    for state, count in sorted(states.items()):
        print(f"  {state}: {count}")

    print()

    print("Users:")

    for user, count in sorted(users.items()):
        print(f"  {user}: {count}")

    return 0


def show_current_directory(
    directory: Path | None = None,
    registry: ResourceRegistry | None = None,
    *,
    verbose: bool = False,
    detailed_evidence_command: str | None = None,
) -> int:
    """Analyze the calculation associated with one local directory."""

    directory = directory or Path.cwd()
    scheduler_lookup = None
    if registry is None:
        try:
            registry = load_resources()
        except ConfigurationError:
            registry = None
    if registry is not None:
        try:
            cluster = powerslurm_cluster(registry)
        except ConfigurationError:
            cluster = None
        if cluster is not None:
            scheduler_lookup = lambda job_id: get_job_accounting(
                cluster.ssh_host,
                job_id,
                timeout=cluster.scheduler_accounting_timeout_seconds,
                ssh_connect_timeout=cluster.ssh_connect_timeout_seconds,
            )

    analysis = analyze_calculation_directory(
        directory,
        scheduler_lookup=scheduler_lookup,
    )
    contextual_enrichment = enrich_lifecycle_with_bmdex_domain_context(
        analysis,
        bmdex_repository(registry) if registry is not None else None,
    )
    if verbose:
        print_lifecycle_analysis(analysis, contextual_enrichment=contextual_enrichment)
    else:
        command = detailed_evidence_command or _path_detail_command(directory)
        print(
            render_concise_summary(
                build_lifecycle_concise_summary(
                    analysis,
                    contextual_enrichment=contextual_enrichment,
                    detailed_evidence_command=command,
                )
            ),
            end="",
        )
    return 0


def print_lifecycle_analysis(
    analysis: LifecycleAnalysis,
    *,
    contextual_enrichment: BmdexDomainContextEnrichment | None = None,
) -> None:
    """Print the detailed lifecycle evidence report."""

    print("BMD Agent")
    print("=========")
    print()
    print(f"Calculation directory: {analysis.directory}")
    print(f"Calculation state: {analysis.state.value}")
    print(f"Calculation type: {analysis.calculation_kind}")
    print(f"Summary: {analysis.message}")
    print()

    if analysis.calculation_kind == "none":
        return

    if analysis.bmd_workflow is not None:
        workflow = analysis.bmd_workflow
        print("BMD Compute provenance:")
        if workflow.relocated:
            print(f"  current acquisition directory: {workflow.workflow_root}")
            if workflow.producer_root:
                print(f"  original producer run directory: {workflow.producer_root}")
        else:
            print(f"  workflow root: {workflow.workflow_root}")
        if workflow.stage_evidence:
            print("  workflow stages:")
            completion = {item.index: item for item in analysis.stage_completion}
            for stage_evidence in workflow.stage_evidence:
                index = stage_evidence.stage_index if stage_evidence.stage_index is not None else "?"
                status = _local_stage_status(
                    stage_evidence,
                    completion.get(stage_evidence.stage_index),
                )
                print(f"    {index}. {stage_evidence.label} - {status}")
        if workflow.current_stage is not None:
            stage = workflow.current_stage
            print(f"  current stage: {stage.label} ({stage.path})")
            if workflow.relocated and stage.producer_path:
                print(f"  original stage path: {stage.producer_path}")
        if workflow.job_id:
            print(f"  job id: {workflow.job_id}")
        print()
        if analysis.diagnostics is not None:
            _print_custodian_policy_context(workflow.custodian_policy)
            print()

    if analysis.scheduler is not None:
        print("Scheduler observation:")
        _print_optional_value("state", analysis.scheduler.state)
        _print_optional_value(
            "exit",
            describe_exit_code(analysis.scheduler.state, analysis.scheduler.exit_code),
        )
        _print_optional_value("elapsed", analysis.scheduler.elapsed)
        _print_optional_value("node", analysis.scheduler.node_list)
        print()
    elif analysis.scheduler_error and analysis.bmd_workflow is not None:
        print("Scheduler observation:")
        print(f"  unavailable: {analysis.scheduler_error}")
        print()

    print("Input evidence:")
    for name in ("POSCAR", "INCAR", "KPOINTS"):
        observation = analysis.input_files.get(name)
        print(f"  {name}: {_local_file_status(observation)}")
    print()

    print("Execution evidence:")
    for name in ("OUTCAR", "OSZICAR", "vasprun.xml", "CONTCAR"):
        observation = analysis.output_files.get(name)
        print(f"  {name}: {_local_file_status(observation)}")
    if analysis.normal_completion is not None:
        print(f"  VASP normal completion marker: {_display_bool(analysis.normal_completion)}")
    print()

    if analysis.diagnostics is not None:
        print("Progress:")
        print("  execution has started")
        print(
            "  normal VASP completion: "
            f"{'observed' if analysis.normal_completion else 'not observed'}"
        )
        _print_trajectory_observations(analysis.diagnostics.trajectories)
        print()
        _print_lifecycle_diagnostic_evidence(analysis.diagnostics)
        print()
        _print_oom_evidence(analysis.diagnostics.oom)
        print()
        _print_convergence_progress_assessment_values(analysis.diagnostics.assessments)
        print()
        _print_lifecycle_suggested_checks(analysis.diagnostics)
        print()

    if analysis.structure is not None:
        print("Structure:")
        print(f"  formula: {analysis.structure.reduced_formula}")
        print(f"  sites: {analysis.structure.sites}")
        print()

    if analysis.incar_settings:
        print("Executed/input settings:")
        for key in _EXECUTED_INPUT_DISPLAY_KEYS:
            if key in analysis.incar_settings:
                print(f"  {key}: {_format_input_value(analysis.incar_settings[key])}")
        print()

    summarize_scientific = (
        analysis.scientific is not None
        and analysis.diagnostics is not None
        and analysis.state != LifecycleState.COMPLETED
        and bool(analysis.scientific.error or analysis.scientific.unavailable)
    )
    if analysis.scientific is not None and not summarize_scientific:
        print("Scientific observations:")
        _print_scientific_result(analysis.scientific)
        print()
    elif analysis.scientific is not None and summarize_scientific:
        print("Scientific observations:")
        print("  final-result parsing incomplete; see progress and diagnostic evidence above")
        if analysis.scientific.error:
            print(f"  unavailable: {analysis.scientific.error}")
        elif analysis.scientific.unavailable:
            print(f"  unavailable: {analysis.scientific.unavailable[0]}")
        print()

    if analysis.evidence_gaps:
        print("Evidence gaps:")
        for gap in analysis.evidence_gaps:
            print(f"  {gap}")
        print()

    if analysis.limitations:
        print("Limitations:")
        for limitation in analysis.limitations:
            print(f"  {limitation}")
        print()

    if contextual_enrichment is not None:
        _print_bmdex_contextual_enrichment(contextual_enrichment)


def _print_bmdex_contextual_enrichment(
    enrichment: BmdexDomainContextEnrichment,
) -> None:
    if enrichment.query is None:
        return

    if enrichment.evidence_gaps:
        print("Contextual reference evidence (contextual_reference_evidence):")
        for gap in enrichment.evidence_gaps:
            print(f"  unavailable: {gap.reason}")
        print()
        return

    evidence = enrichment.evidence
    if evidence is None:
        return
    if not evidence.records:
        print("Contextual reference evidence (contextual_reference_evidence):")
        print("  no matching BMDex contextual references")
        print()
        return

    producer = evidence.producer
    git = producer.get("git") if isinstance(producer, Mapping) else None
    print("Contextual reference evidence (contextual_reference_evidence):")
    print("  producer: BMDex")
    if isinstance(git, Mapping):
        print(f"  producer commit: {_display_commit(git.get('commit'))}")
        print(f"  producer state: {git.get('state') or 'unavailable'}")
    for record in evidence.records:
        provenance = record.record_provenance
        print(f"  record: {record.record_id}")
        print(f"    title: {record.title}")
        print(
            "    version: "
            f"{provenance.get('record_version', 'unknown')} "
            f"({provenance.get('machine_readable_schema', 'unknown')})"
        )
        print(f"    contextual statement: {record.contextual_statement}")
        print(f"    diagnostic relevance: {record.diagnostic_relevance}")
        matched = record.match.get("matched_fields", ())
        if isinstance(matched, list) and matched:
            print(f"    matched fields: {', '.join(str(item) for item in matched)}")
        print(f"    source provenance: {_compact_reference_sources(record.sources)}")
        print(f"    producer-supplied limitations retained: {len(record.limitations)}")
    print()

    assessment = enrichment.assessment
    if assessment is None:
        return
    print("Contextual assessment (assessment):")
    for basis in assessment.basis:
        print(f"  {basis}")
    for limitation in assessment.limitations:
        print(f"  limitation: {limitation}")
    print()


def _compact_reference_sources(sources: tuple[Mapping[str, Any], ...]) -> str:
    by_authority: dict[str, tuple[int, str | None]] = {}
    for source in sources:
        authority = str(source.get("authority") or source.get("source_type") or "unknown")
        count, first_url = by_authority.get(authority, (0, None))
        url = source.get("url")
        by_authority[authority] = (
            count + 1,
            first_url or (str(url) if isinstance(url, str) and url else None),
        )
    entries = []
    for authority, (count, url) in by_authority.items():
        label = f"{authority} ({count})"
        entries.append(f"{label}: {url}" if url else label)
    return "; ".join(entries)


def show_compute(registry: ResourceRegistry | None = None) -> int:
    """Display BMD Compute's advertised executable capabilities."""

    registry = registry or load_resources()
    repository = bmd_compute_repository(registry)

    try:
        capabilities = inspect_compute_capabilities(repository)

    except ComputeCapabilityError as exc:
        print(f"Unable to inspect BMD Compute capabilities: {exc}")
        return 1

    print_compute_capabilities(capabilities)
    return 0


def print_compute_capabilities(capabilities: ComputeCapabilities) -> None:
    """Print a concise user-facing BMD Compute capability summary."""

    source = capabilities.source

    print("BMD Compute Capabilities")
    print("========================")
    print()

    print("Source:")
    print(f"  repository: {source.get('repository', 'unknown')}")
    print(f"  commit:     {_display_commit(source.get('commit'))}")
    print(f"  state:      {_display_dirty_state(source.get('dirty'))}")
    print()

    print("Scope:")
    print(f"  {capabilities.scope}")
    print()

    print("Supported capabilities:")
    for capability in capabilities.capabilities:
        theory = _display_theory(capability["theory"])
        stage = _display_stage(capability["stage_type"])
        print(f"  {theory:<6} {stage}")


def show_structure(directory: str, registry: ResourceRegistry | None = None) -> int:
    """Display structural information from a remote VASP POSCAR."""

    registry = registry or load_resources()
    cluster = powerslurm_cluster(registry)

    print("BMD VASP Structure")
    print("==================")
    print()

    try:
        info = read_remote_structure(
            ssh_host=cluster.ssh_host,
            directory=directory,
            allowed_roots=cluster.allowed_remote_roots,
        )

    except RemotePathError as exc:
        print(f"Refusing remote read: {exc}")
        return 2

    except subprocess.CalledProcessError as exc:
        print("Unable to read structure from PowerSLURM.")

        if exc.stderr:
            print(exc.stderr.decode(errors="replace").strip())

        return 1

    except Exception as exc:
        print(f"Unable to parse structure: {exc}")
        return 1

    print(f"Source:          {info.source}")
    print(f"Formula:         {info.formula}")
    print(f"Reduced formula: {info.reduced_formula}")
    print(f"Sites:           {info.sites}")
    print(f"Volume:          {info.volume:.3f} A^3")
    print()

    print("Lattice:")
    print(f"  a: {info.a:.6f} A")
    print(f"  b: {info.b:.6f} A")
    print(f"  c: {info.c:.6f} A")

    return 0


def show_check_input(args: list[str], registry: ResourceRegistry | None = None) -> int:
    """Display a BMD Compute generated-reference comparison for proposed inputs."""

    parsed, error = _parse_check_input_args(args)
    if error:
        print(f"Usage: {_check_input_usage()}")
        print(f"Error: {error}")
        return 2

    assert parsed is not None
    registry = registry or load_resources()
    cluster = powerslurm_cluster(registry)
    repository = bmd_compute_repository(registry)

    try:
        observation = check_remote_input_directory(
            cluster,
            repository,
            parsed["directory"],
            stage=parsed["stage"],
            theory=parsed["theory"],
            nodes=parsed["nodes"],
            ntasks=parsed["ntasks"],
            mem_gb=parsed["mem_gb"],
        )

    except RemotePathError as exc:
        print(f"Refusing remote read: {exc}")
        return 2

    except ValueError as exc:
        print(f"Usage: {_check_input_usage()}")
        print(f"Error: {exc}")
        return 2

    print_input_check(observation)
    return 0


def print_input_check(observation: InputCheckObservation) -> None:
    """Print a concise user-facing BMD VASP input check summary."""

    print("BMD VASP Input Check")
    print("====================")
    print()

    print("Proposed calculation:")
    print(f"  {_display_theory(observation.theory)} {_display_stage(observation.stage_type)}")
    print()

    print("Input directory:")
    print(f"  {observation.remote_directory}")
    print()

    print("Supplied structure:")
    if observation.proposed.error:
        print(f"  unavailable: {observation.proposed.error}")
    else:
        _print_optional_value("formula", observation.proposed.reduced_formula)
        _print_optional_value("sites", observation.proposed.site_count)
    print()

    print("Reference resources:")
    _print_mapping_values(dict(observation.resources), ("nodes", "ntasks", "mem_gb"))
    print()

    print("BMD Compute reference:")
    print(f"  producer:       {observation.reference.producer_repository or 'unavailable'}")
    print(f"  commit:         {_display_commit(observation.reference.producer_commit)}")
    print(f"  state:          {_display_dirty_state(observation.reference.producer_dirty)}")
    print(f"  schema_version: {_display_value(observation.reference.schema_version)}")
    print(f"  phase:          {_display_value(observation.reference.reference_phase)}")
    if observation.reference.workflow_label:
        print(f"  workflow:       {observation.reference.workflow_label}")
    if observation.reference.error_code or observation.reference.error_message:
        print("  producer status:")
        if observation.reference.error_code:
            print(f"    code: {observation.reference.error_code}")
        if observation.reference.error_message:
            print(f"    message: {observation.reference.error_message}")
    print()

    print("INCAR:")
    if observation.incar_comparisons:
        print(f"  {observation.matching_incar_settings} settings match")
        differences = observation.nonmatching_incar_settings
        if differences:
            print()
            print("  differences:")
            for comparison in differences:
                print(f"    {comparison.setting}")
                print(f"      supplied:  {_format_input_value(comparison.supplied_value)}")
                print(f"      reference: {_format_input_value(comparison.reference_value)}")
                print(f"      status:    {comparison.status}")
    else:
        print("  unavailable")
    print()

    print("KPOINTS:")
    if observation.kpoints_comparison is None:
        print("  unavailable")
    else:
        comparison = observation.kpoints_comparison
        print(f"  supplied:  {_display_value(comparison.supplied_summary)}")
        print(f"  reference: {_display_value(comparison.reference_summary)}")
        print(f"  status:    {comparison.status}")
        if comparison.reason:
            print(f"  reason:    {comparison.reason}")
    print()

    print("Overall:")
    print(f"  {observation.overall_status}")
    if observation.limitations:
        for limitation in observation.limitations:
            print(f"  reason: {limitation}")
    print()

    print("Note:")
    print(
        "  This compares the supplied input with the current BMD Compute "
        "generated reference."
    )
    print(
        "  A difference is not by itself evidence that the supplied setting "
        "is scientifically invalid."
    )


def show_inspect_run(flow_root: str, registry: ResourceRegistry | None = None) -> int:
    """Display evidence gathered for a BMD Compute run."""

    registry = registry or load_resources()
    cluster = powerslurm_cluster(registry)
    compute_policy = compute_policy_from_registry(registry)

    print("BMD Compute Run Inspection")
    print("==========================")
    print()

    try:
        inspection = inspect_remote_run(
            cluster,
            flow_root,
            compute_policy=compute_policy,
        )

    except RemotePathError as exc:
        print(f"Refusing remote read: {exc}")
        return 2

    except subprocess.TimeoutExpired:
        print("Run inspection timed out.")
        return 1

    except subprocess.CalledProcessError as exc:
        print("Unable to inspect run through PowerSLURM.")
        if exc.stderr:
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr)
            )
            print(stderr.strip())
        return 1

    except RunInspectionError as exc:
        print(f"Unable to inspect run: {exc}")
        return 1

    print_run_inspection(inspection)
    return 0


def show_compare_runs(flow_roots: list[str], registry: ResourceRegistry | None = None) -> int:
    """Display baseline-relative evidence comparisons for completed runs."""

    if len(flow_roots) < 2:
        print("Usage: bmd-agent compare-runs <flow-a> <flow-b> [<flow-c> ...]")
        return 2

    registry = registry or load_resources()
    cluster = powerslurm_cluster(registry)
    compute_policy = compute_policy_from_registry(registry)

    print("BMD Compute Run Comparison")
    print("==========================")
    print()

    try:
        comparison = compare_remote_runs(
            cluster,
            flow_roots,
            compute_policy=compute_policy,
        )

    except RemotePathError as exc:
        print(f"Refusing remote read: {exc}")
        return 2

    except subprocess.TimeoutExpired:
        print("Run comparison timed out.")
        return 1

    except subprocess.CalledProcessError as exc:
        print("Unable to compare runs through PowerSLURM.")
        if exc.stderr:
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr)
            )
            print(stderr.strip())
        return 1

    except RunInspectionError as exc:
        print(f"Unable to compare runs: {exc}")
        return 1

    print_run_comparison(comparison, compute_policy=compute_policy)
    return 0


def show_diagnose_run(flow_root: str, registry: ResourceRegistry | None = None) -> int:
    """Display descriptive convergence and termination evidence for one run."""

    registry = registry or load_resources()
    cluster = powerslurm_cluster(registry)
    compute_policy = compute_policy_from_registry(registry)

    print("BMD Compute Run Diagnosis")
    print("=========================")
    print()

    try:
        diagnosis = diagnose_remote_run(
            cluster,
            flow_root,
            compute_policy=compute_policy,
        )

    except RemotePathError as exc:
        print(f"Refusing remote read: {exc}")
        return 2

    except subprocess.TimeoutExpired:
        print("Run diagnosis timed out.")
        return 1

    except subprocess.CalledProcessError as exc:
        print("Unable to diagnose run through PowerSLURM.")
        if exc.stderr:
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr)
            )
            print(stderr.strip())
        return 1

    except RunInspectionError as exc:
        print(f"Unable to diagnose run: {exc}")
        return 1

    print_run_diagnosis(diagnosis)
    return 0


def show_job(
    job_id: str,
    registry: ResourceRegistry | None = None,
    *,
    trajectory_json: bool = False,
    profile: bool = False,
    verbose: bool = False,
    detailed_evidence_command: str | None = None,
) -> int:
    """Display scheduler-bound evidence for one calculation job."""

    if not profile:
        return _show_job(
            job_id,
            registry,
            trajectory_json=trajectory_json,
            verbose=verbose,
            detailed_evidence_command=detailed_evidence_command,
        )

    profiler = PerformanceProfiler()
    with profiler.activate():
        exit_code = _show_job(
            job_id,
            registry,
            trajectory_json=trajectory_json,
            profiling=True,
            verbose=verbose,
            detailed_evidence_command=detailed_evidence_command,
        )
    print_performance_profile(profiler.snapshot())
    return exit_code


def _show_job(
    job_id: str,
    registry: ResourceRegistry | None,
    *,
    trajectory_json: bool,
    profiling: bool = False,
    verbose: bool = False,
    detailed_evidence_command: str | None = None,
) -> int:
    """Run the shared job inspection path with optional active telemetry."""

    with profile_phase("configuration_and_policy"):
        registry = registry or load_resources()
        cluster = powerslurm_cluster(registry)
        deployment = resolve_deployment_context(registry, cluster_key=cluster.key)
        if profiling:
            compute_policy = compute_policy_from_registry(
                registry,
                runner=profiled_runner(subprocess.run, role="producer"),
            )
        else:
            compute_policy = compute_policy_from_registry(registry)

    if not trajectory_json and verbose:
        print("BMD Job Inspection")
        print("==================")
        print()

    try:
        with ReusableSshSession(
            cluster.ssh_host,
            close_timeout=cluster.ssh_connect_timeout_seconds,
        ) as ssh_session:
            with remote_acquisition_cache(
                cluster.ssh_host,
                cluster.allowed_remote_roots,
            ):
                inspection = inspect_slurm_job(
                    cluster,
                    job_id,
                    compute_policy=compute_policy,
                    deployment=deployment,
                    remote_runner=ssh_session.runner("remote"),
                    slurm_runner=ssh_session.runner("scheduler"),
                )

    except ValueError as exc:
        print(f"Unable to inspect job: {exc}")
        return 2

    except (
        RemotePathError,
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        RunInspectionError,
    ) as exc:
        print(f"Unable to inspect job: {exc}")
        return 1

    if trajectory_json:
        with profile_phase("synthesis_rendering"):
            print(
                json.dumps(
                    serialize_job_trajectory_evidence(inspection),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
    else:
        with profile_phase("bmdex_acquisition"):
            if profiling:
                contextual_enrichment = enrich_job_with_bmdex_domain_context(
                    inspection,
                    bmdex_repository(registry),
                    runner=profiled_runner(subprocess.run, role="bmdex"),
                )
            else:
                contextual_enrichment = enrich_job_with_bmdex_domain_context(
                    inspection,
                    bmdex_repository(registry),
                )
        with profile_phase("synthesis_rendering"):
            if verbose:
                print_job_inspection(
                    inspection,
                    contextual_enrichment=contextual_enrichment,
                )
            else:
                command = detailed_evidence_command or f"bmd-agent {job_id} --verbose"
                print(
                    render_concise_summary(
                        build_job_concise_summary(
                            inspection,
                            contextual_enrichment=contextual_enrichment,
                            detailed_evidence_command=command,
                        )
                    ),
                    end="",
                )
    return 0


def print_performance_profile(profile: PerformanceProfile) -> None:
    """Render opt-in developer telemetry separately from scientific evidence."""

    print()
    print("Performance profile (developer telemetry)")
    print("=========================================")
    print(f"Total wall time: {profile.total_elapsed_seconds:.3f} s")
    print("Phase timings (inclusive; nested phases may overlap):")
    for phase in profile.phases:
        suffix = f", failures={phase.failures}" if phase.failures else ""
        print(
            f"  {phase.name}: {phase.elapsed_seconds:.3f} s "
            f"(calls={phase.calls}{suffix})"
        )

    counts = profile.operations.counts
    elapsed = profile.operations.elapsed_seconds
    print("Remote/subprocess operations:")
    for key in (
        "subprocess_invocations",
        "ssh_invocations",
        "ssh_connections",
        "ssh_exec_channels",
        "ssh_control_operations",
        "remote_commands",
        "remote_file_reads",
        "bounded_remote_file_reads",
        "remote_extractor_operations",
        "existence_probes",
        "stat_probes",
        "directory_probes",
        "directory_listing_operations",
        "archive_probes",
        "archive_probe_batches",
        "metadata_manifest_operations",
        "batched_file_read_operations",
        "logical_files_described",
        "logical_files_read",
        "scheduler_operations",
        "producer_operations",
        "bmdex_operations",
    ):
        print(f"  {key}: {counts[key]}")
    print("Nested operation wait time:")
    for key, value in elapsed.items():
        print(f"  {key}: {value:.3f} s")
    print(f"Captured subprocess output: {profile.operations.bytes_transferred} bytes")
    print(f"Captured SSH output: {profile.operations.ssh_bytes_transferred} bytes")
    print(f"Subprocess failures: {profile.operations.failure_count}")
    print(f"Subprocess timeouts: {profile.operations.timeout_count}")


def print_job_inspection(
    inspection: JobInspection,
    *,
    contextual_enrichment: BmdexDomainContextEnrichment | None = None,
) -> None:
    """Print the detailed job evidence report."""

    print("Job (scheduler_observation):")
    print(f"  ID: {_diagnosis_value(inspection.job_id)}")
    record = inspection.scheduler
    if record is None:
        print(f"  unavailable: {inspection.scheduler_error or 'no scheduler accounting record found'}")
    else:
        _print_job_record(record)
    print()

    print("Calculation:")
    print(f"  scheduler WorkDir: {_diagnosis_value(inspection.scheduler_work_dir)}")
    print(f"  directory: {_diagnosis_value(inspection.calculation_directory)}")
    print(f"  type: {inspection.calculation_type}")
    if inspection.calculation_reason:
        print(f"  reason: {inspection.calculation_reason}")
    print()

    resolution = inspection.run_resolution
    if resolution is not None:
        print(f"BMD Compute run resolution ({resolution.evidence_type}):")
        status = (
            "not found"
            if resolution.resolution_status == "not_bmd_compute"
            else resolution.resolution_status
        )
        print(f"  status: {status}")
        if resolution.producer_state_path:
            print(f"  producer state: {resolution.producer_state_path}")
        if resolution.run_directory:
            print(f"  run directory: {resolution.run_directory}")
        if resolution.submission_attempt_id:
            print(f"  submission attempt: {resolution.submission_attempt_id}")
        if resolution.reason:
            print(f"  reason: {resolution.reason}")
        for limitation in resolution.limitations:
            print(f"  limitation: {limitation}")
        print()

    if inspection.bmd_compute is not None:
        print_run_diagnosis(inspection.bmd_compute)
        if contextual_enrichment is not None:
            _print_bmdex_contextual_enrichment(contextual_enrichment)
        return

    _print_oom_evidence(inspection.oom)
    print()

    if inspection.direct_vasp is None:
        print("Producer provenance (producer_provenance):")
        print("  unavailable")
        print("  reason: no supported calculation evidence was identified")
        if contextual_enrichment is not None:
            _print_bmdex_contextual_enrichment(contextual_enrichment)
        return

    direct = inspection.direct_vasp
    print("Producer provenance (producer_provenance):")
    print("  unavailable")
    print(f"  reason: {direct.producer_reason}")
    print()

    if _has_relevant_custodian_evidence(direct.custodian_evidence):
        _print_custodian_intervention_evidence(direct.custodian_evidence)
        _print_termination_assessment(direct.termination_assessment)
        print()

    print("Executed VASP inputs (executed_input):")
    _print_executed_input_observations(direct.executed_inputs, ())
    print()

    print("Artifact observation (artifact_observation):")
    for artifact in direct.artifacts:
        print(f"  {artifact.label}: {_present_text(artifact)}")
    print()

    print("Independent parsing (pymatgen_derived):")
    _print_scientific_result(direct.scientific)
    print()

    _print_trajectory_observations((direct.trajectory,))
    _print_convergence_progress_assessment_values(direct.assessments)
    if contextual_enrichment is not None:
        print()
        _print_bmdex_contextual_enrichment(contextual_enrichment)


def _print_job_record(record: object) -> None:
    for label, attribute in (
        ("name", "name"),
        ("user", "user"),
        ("account", "account"),
        ("state", "state"),
        ("reason", "reason"),
        ("node", "node_list"),
        ("elapsed", "elapsed"),
        ("elapsed raw seconds", "elapsed_raw"),
        ("time limit", "timelimit"),
        ("partition", "partition"),
    ):
        print(f"  {label}: {_diagnosis_value(getattr(record, attribute, None))}")
        if attribute == "state":
            print(
                "  exit: "
                + describe_exit_code(
                    getattr(record, "state", None),
                    getattr(record, "exit_code", None),
                )
            )
    print("  resources:")
    for label, attribute in (
        ("nodes", "node_count"),
        ("allocated CPUs", "allocated_cpus"),
        ("tasks", "task_count"),
        ("requested memory", "req_mem"),
        ("ReqTRES", "req_tres"),
        ("AllocTRES", "alloc_tres"),
        ("total CPU", "total_cpu"),
        ("CPUTimeRAW", "cpu_time_raw"),
        ("MaxRSS", "max_rss"),
        ("MaxVMSize", "max_vm_size"),
        ("AveRSS", "ave_rss"),
    ):
        print(f"    {label}: {_diagnosis_value(getattr(record, attribute, None))}")
    steps = getattr(record, "steps", ())
    if steps:
        print("  job steps:")
        for step in steps:
            print(
                f"    {getattr(step, 'job_id_raw', 'step')}: "
                f"state={_diagnosis_value(getattr(step, 'state', None))}, "
                f"exit={describe_exit_code(getattr(step, 'state', None), getattr(step, 'exit_code', None))}, "
                f"MaxRSS={_diagnosis_value(getattr(step, 'max_rss', None))}"
            )


def print_run_inspection(inspection: RunInspection) -> None:
    """Print a concise evidence-oriented run inspection summary."""

    print("Producer provenance (producer_provenance):")
    print(f"  flow root:   {inspection.flow_root}")
    print(f"  submission:  {inspection.submission_path}")
    print(f"  git commit:  {_display_commit(inspection.producer_git.get('git_commit'))}")
    print(f"  git state:   {inspection.producer_git.get('state', 'unavailable')}")
    print()

    _print_custodian_policy_context(inspection.custodian_policy)
    print()

    if _has_relevant_custodian_evidence(inspection.custodian_evidence):
        _print_custodian_intervention_evidence(inspection.custodian_evidence)
        print()

    print(f"Requested workflow ({PRODUCER_REQUESTED}):")
    for stage in inspection.workflow_stages:
        modifiers = f" [{', '.join(stage.modifiers)}]" if stage.modifiers else ""
        print(
            f"  {stage.index}. {_display_theory(stage.theory):<6} "
            f"{_display_stage(stage.stage_type)}{modifiers}"
        )
        options = _format_options(stage.options)
        if options:
            print(f"     options: {options}")
    print()

    print("Requested execution (producer_provenance):")
    _print_mapping_values(
        inspection.cluster_request,
        ("partition", "account"),
    )
    _print_mapping_values(
        inspection.resources_request,
        ("nodes", "ntasks", "mem_gb", "walltime"),
    )
    _print_mapping_values(
        inspection.environment_policy,
        ("VASP_CMD", "JOBFLOW_CONFIG_FILE", "PMG_VASP_PSP_DIR"),
    )
    print()

    print("Scheduler (scheduler_observation):")
    if inspection.scheduler is None:
        print(f"  unavailable: {inspection.scheduler_error or 'no accounting record found'}")
    else:
        record = inspection.scheduler
        print(f"  job id:    {record.job_id}")
        print(f"  state:     {record.state}")
        print(f"  exit:      {describe_exit_code(record.state, record.exit_code)}")
        print(f"  elapsed:   {record.elapsed}")
        print(f"  start:     {record.start}")
        print(f"  end:       {record.end}")
        print(f"  partition: {record.partition}")
    print()

    _print_oom_evidence(inspection.oom)
    print()

    print("Logs (log_observation):")
    if inspection.runtime.sources:
        if inspection.runtime.python:
            print(f"  python: {inspection.runtime.python}")
        for package, version in sorted(inspection.runtime.packages.items()):
            print(f"  {package}: {version}")
        for key, value in sorted(inspection.runtime.environment.items()):
            print(f"  {key}: {value}")
        if inspection.runtime.stage_uuids:
            print("  stage UUIDs:")
            for label, uuid in sorted(inspection.runtime.stage_uuids.items()):
                print(f"    {label}: {uuid}")
    else:
        print("  unavailable: no runner log content was readable")
    print()

    print("Evidence paths (artifact_observation):")
    print(f"  result_dir: {_present_text(inspection.result_directory)}")
    if inspection.stage_directories:
        print("  stage directories:")
        for observation in inspection.stage_directories:
            print(f"    {observation.label}: {_present_text(observation)}")
    if inspection.log_paths:
        print("  logs:")
        for observation in inspection.log_paths:
            print(f"    {observation.label}: {_present_text(observation)}")
    print("  final artifacts:")
    for observation in inspection.final_artifacts:
        print(f"    {observation.label}: {_present_text(observation)}")
    print()

    print("Executed VASP inputs (executed_input):")
    _print_executed_input_summary(inspection)
    print()
    _print_compute_policy_context(
        inspection.compute_policy,
        run_labels=((None, inspection.producer_git),),
        has_checks=bool(inspection.input_expectations),
        trailing_blank=not inspection.input_expectations,
    )
    if inspection.input_expectations:
        print("  requested/executed checks (agent_comparison):")
        for expectation in inspection.input_expectations:
            print(f"    {expectation.stage_label}: {expectation.option_path}={expectation.requested_value} -> "
                  f"{expectation.input_key} {_expectation_observed_text(expectation)}, "
                  f"expected {_expectation_expected_text(expectation)}: "
                  f"{expectation.status}")
            for source, value in sorted(expectation.source_values.items()):
                print(f"      {source}: {_expectation_source_value(expectation, value)}")
            if expectation.reason:
                print(f"      reason: {expectation.reason}")
    print()

    print("Independent parsing (pymatgen_derived):")
    _print_scientific_result(inspection.scientific)
    print()

    print("Comparison with durable producer result:")
    print(f"  {inspection.comparison.status}")
    if inspection.comparison.reason:
        print(f"  reason: {inspection.comparison.reason}")
    if inspection.comparison.mismatches:
        print(f"  mismatches: {', '.join(inspection.comparison.mismatches)}")


def print_run_diagnosis(diagnosis: RunDiagnosis) -> None:
    """Print descriptive termination and trajectory evidence."""

    inspection = diagnosis.inspection
    print("Producer provenance (producer_provenance):")
    print(f"  flow root:   {inspection.flow_root}")
    print(f"  git commit:  {_display_commit(inspection.producer_git.get('git_commit'))}")
    print(f"  git state:   {inspection.producer_git.get('state', 'unavailable')}")
    print()

    print(f"Requested workflow ({PRODUCER_REQUESTED}):")
    for stage in inspection.workflow_stages:
        modifiers = f" [{', '.join(stage.modifiers)}]" if stage.modifiers else ""
        print(
            f"  {stage.index}. {_display_theory(stage.theory):<6} "
            f"{_display_stage(stage.stage_type)}{modifiers}"
        )
        options = _format_options(stage.options)
        if options:
            print(f"     options: {options}")
    print()

    if (
        _has_relevant_custodian_evidence(inspection.custodian_evidence)
        or inspection.scheduler is None
        or inspection.scheduler.state.upper() != "COMPLETED"
    ):
        _print_custodian_policy_context(inspection.custodian_policy)
        print()

    if _has_relevant_custodian_evidence(inspection.custodian_evidence):
        _print_custodian_intervention_evidence(inspection.custodian_evidence)
        print()

    termination = diagnosis.termination
    print(f"Termination evidence ({termination.evidence_type}):")
    _print_optional_value("scheduler state", termination.scheduler_state)
    if termination.scheduler_state is not None or termination.scheduler_exit_code is not None:
        print(
            "  scheduler exit: "
            + describe_exit_code(termination.scheduler_state, termination.scheduler_exit_code)
        )
    _print_optional_value("elapsed", termination.scheduler_elapsed)
    _print_optional_value("time limit", termination.scheduler_timelimit)
    _print_optional_value("scheduler reports timeout", termination.scheduler_reports_timeout)
    _print_optional_value("VASP normal completion", termination.vasp_completed_normally)
    for item in termination.unavailable:
        print(f"  unavailable: {item}")
    _print_termination_assessment(termination.assessment, indent="  ")
    print()

    _print_oom_evidence(inspection.oom)
    print()

    if not _print_trajectory_observations(diagnosis.trajectories):
        return

    _print_convergence_progress_assessments(diagnosis)


def print_run_comparison(
    comparison: RunComparison,
    *,
    compute_policy: ComputePolicyObservation | None = None,
) -> None:
    """Print a concise baseline-relative run comparison."""

    baseline = comparison.inspections[0]
    baseline_label = comparison.labels[baseline.flow_root]

    print("Runs:")
    for index, inspection in enumerate(comparison.inspections):
        role = "baseline" if index == 0 else "comparison"
        label = comparison.labels[inspection.flow_root]
        scheduler_state = inspection.scheduler.state if inspection.scheduler else "unavailable"
        print(f"  {index + 1}. {label} ({role})")
        print(f"     flow root: {inspection.flow_root}")
        print(f"     git: {_display_commit(inspection.producer_git.get('git_commit'))} "
              f"{inspection.producer_git.get('state', 'unavailable')}")
        print(f"     scheduler: {scheduler_state}")
    print()

    print("Initial structure (agent_comparison):")
    print(f"  {comparison.initial_structure.status}")
    if comparison.initial_structure.reason:
        print(f"  reason: {comparison.initial_structure.reason}")
    print()

    _print_compute_policy_context(
        compute_policy,
        run_labels=tuple(
            (comparison.labels[inspection.flow_root], inspection.producer_git)
            for inspection in comparison.inspections
        ),
        has_checks=any(inspection.input_expectations for inspection in comparison.inspections),
    )
    print("Executed-input checks:")
    any_checks = False
    for inspection in comparison.inspections:
        label = comparison.labels[inspection.flow_root]
        for expectation in inspection.input_expectations:
            any_checks = True
            print(f"  {label} {expectation.stage_label}: "
                  f"{expectation.input_key} {_expectation_observed_text(expectation)}, "
                  f"expected {_expectation_expected_text(expectation)}: "
                  f"{expectation.status}")
            for source, value in sorted(expectation.source_values.items()):
                print(f"    {source}: {_expectation_source_value(expectation, value)}")
            if expectation.reason:
                print(f"    reason: {expectation.reason}")
    if not any_checks:
        if compute_policy is not None and not compute_policy.available:
            print("  not performed: BMD Compute policy evidence is unavailable")
        elif compute_policy is not None:
            print("  none: no BMD Compute input-effect policy applies to the requested options")
        else:
            print("  unavailable: no producer option/input-effect checks were available")
    print()

    print("Final structure/result differences (agent_comparison):")
    print(f"  baseline: {baseline_label}")
    current_flow = None
    for quantity in comparison.quantities:
        if quantity.flow_root != current_flow:
            current_flow = quantity.flow_root
            print(f"  {quantity.run_label}:")
        if quantity.status != "available":
            print(f"    {quantity.label}: unavailable ({quantity.reason})")
            continue
        unit = f" {quantity.unit}" if quantity.unit else ""
        percent = (
            f", {quantity.percent_delta:+.3f}%"
            if quantity.percent_delta is not None
            else ""
        )
        print(
            f"    {quantity.label}: {quantity.baseline_value} -> "
            f"{quantity.comparison_value}{unit}, delta {quantity.delta:+.6f}{unit}{percent}"
        )
    if comparison.energy_warning:
        print()
        print(f"Warning: {comparison.energy_warning}")


def _trajectory_heading(trajectory: object) -> str:
    stage_index = getattr(trajectory, "stage_index", None)
    if stage_index is None:
        prefix = "unindexed stage"
    else:
        prefix = f"stage {stage_index}"
    theory = getattr(trajectory, "theory", None)
    stage_type = getattr(trajectory, "stage_type", None)
    if stage_type == "direct_vasp":
        prefix = f"{prefix}: Direct VASP calculation"
    elif theory and stage_type:
        prefix = f"{prefix}: {_display_theory(theory)} {_display_stage(stage_type)}"
    label = getattr(trajectory, "stage_label", None)
    return f"{prefix} ({label})" if label else prefix


def _print_trajectory_observations(trajectories: tuple[object, ...]) -> bool:
    print("Trajectory evidence (trajectory_observation):")
    if not trajectories:
        print("  unavailable: no producer-bound VASP stage directories were available")
        return False

    for trajectory in trajectories:
        print(f"  {_trajectory_heading(trajectory)}:")
        print(f"    directory: {trajectory.directory}")
        print(
            f"    OSZICAR: "
            f"{'present' if trajectory.oszicar_present else 'unavailable'} "
            f"({trajectory.oszicar_path})"
        )
        if trajectory.oszicar_error:
            print(f"      error: {trajectory.oszicar_error}")
        _print_vasprun_trajectory_source(trajectory)
        _print_trajectory_criteria(trajectory)
        _print_electronic_trajectory(trajectory)
        _print_ionic_trajectory(trajectory)
        _print_trajectory_progress_evidence(trajectory)
        print("    convergence flags:")
        print(f"      electronic: {_diagnosis_value(trajectory.converged_electronic)}")
        print(f"      ionic: {_diagnosis_value(trajectory.converged_ionic)}")
        for item in trajectory.unavailable:
            print(f"    unavailable: {item}")
    return True


def _print_convergence_progress_assessments(diagnosis: RunDiagnosis) -> None:
    print()
    _print_convergence_progress_assessment_values(getattr(diagnosis, "assessments", ()))


def _print_convergence_progress_assessment_values(assessments: tuple[object, ...]) -> None:
    evidence_type = (
        getattr(assessments[0], "evidence_type", "convergence_progress_assessment")
        if assessments
        else "convergence_progress_assessment"
    )
    print(f"Convergence-progress assessment ({evidence_type}):")
    print("  based on observed trajectory evidence, not a prediction")
    if not assessments:
        print("  unavailable: no assessment was produced")
        return
    for assessment in assessments:
        print(f"  {_assessment_heading(assessment)} {assessment.scope}: {assessment.label}")
        print(f"    sufficiency: {assessment.sufficiency}")
        if assessment.basis:
            print("    basis:")
            for item in assessment.basis:
                print(f"      - {item}")
        if assessment.counter_evidence:
            print("    counter-evidence:")
            for item in assessment.counter_evidence:
                print(f"      - {item}")
        if assessment.limitations:
            print("    limitations:")
            for item in assessment.limitations:
                print(f"      - {item}")


def _assessment_heading(assessment: object) -> str:
    stage_index = getattr(assessment, "stage_index", None)
    stage_label = getattr(assessment, "stage_label", None)
    if stage_index is None:
        prefix = "unindexed stage"
    else:
        prefix = f"stage {stage_index}"
    return f"{prefix} ({stage_label})" if stage_label else prefix


def _print_vasprun_trajectory_source(trajectory: object) -> None:
    path = getattr(trajectory, "vasprun_path", None)
    if getattr(trajectory, "vasprun_present", False):
        print(f"    vasprun.xml: present ({path})")
    else:
        print(f"    vasprun.xml: unavailable ({path})")
    if getattr(trajectory, "vasprun_skipped_reason", None):
        print(f"      skipped: {getattr(trajectory, 'vasprun_skipped_reason')}")
    if getattr(trajectory, "vasprun_error", None):
        print(
            "      vasprun trajectory enrichment unavailable: "
            f"{getattr(trajectory, 'vasprun_error')}"
        )


def _print_trajectory_criteria(trajectory: object) -> None:
    criteria = getattr(trajectory, "criteria", {})
    print("    criteria:")
    if not criteria:
        print("      unavailable")
    for key in ("NELM", "EDIFF", "NSW", "EDIFFG", "ISIF"):
        if key in criteria:
            print(f"      {key}: {_format_trajectory_criterion(key, criteria[key])}")
    discrepancies = getattr(trajectory, "criteria_discrepancies", ())
    if discrepancies:
        print(f"      discrepancies: {', '.join(discrepancies)}")
        source_values = getattr(trajectory, "criteria_source_values", {})
        for key in discrepancies:
            values = source_values.get(key, {})
            for source, value in sorted(values.items()):
                print(f"        {key} {source}: {_format_executed_value(key, value)}")


def _format_trajectory_criterion(key: str, value: object) -> str:
    if key != "EDIFFG":
        return _format_executed_value(key, value)
    numeric = _float_or_none(value)
    if numeric is None:
        return _format_executed_value(key, value)
    if numeric < 0:
        return f"{value} eV/A force criterion"
    if numeric > 0:
        return f"{value} eV energy-change criterion"
    return "0 (no generic convergence interpretation inferred)"


def _print_electronic_trajectory(trajectory: object) -> None:
    print("    electronic trajectory:")
    completed_ionic_steps = getattr(
        trajectory,
        "completed_ionic_steps",
        getattr(trajectory, "ionic_steps_observed", None),
    )
    print(f"      completed ionic steps: {_diagnosis_value(completed_ionic_steps)}")

    counts = getattr(
        trajectory,
        "electronic_iterations_by_completed_ionic_step",
        getattr(trajectory, "electronic_iterations_by_ionic_step", ()),
    )
    if counts:
        print(
            "      electronic iterations for completed ionic steps: "
            f"{_format_count_sequence(counts)}"
        )
    else:
        print("      electronic iterations for completed ionic steps: none")

    incomplete_count = getattr(trajectory, "incomplete_electronic_iteration_count", None)
    if incomplete_count is not None:
        print(
            "      incomplete electronic cycle: "
            f"{incomplete_count} iterations observed{_nelm_suffix(trajectory)}"
        )
        recent = getattr(trajectory, "recent_incomplete_electronic_iterations", ())
        recent_label = "recent incomplete-cycle iterations"
    else:
        final_count = getattr(trajectory, "final_electronic_iteration_count", None)
        if final_count is None:
            print("      final electronic cycle: unavailable")
        else:
            print(
                "      final electronic cycle: "
                f"{final_count} iterations observed{_nelm_suffix(trajectory)}"
            )
        recent = getattr(trajectory, "recent_electronic_iterations", ())
        recent_label = "recent final-cycle iterations"

    if recent:
        print(f"      {recent_label}:")
        for iteration in recent:
            print(f"        {_format_electronic_iteration(iteration)}")
    else:
        print(f"      {recent_label}: unavailable")


def _print_ionic_trajectory(trajectory: object) -> None:
    print("    ionic trajectory:")
    completed_ionic_steps = getattr(
        trajectory,
        "completed_ionic_steps",
        getattr(trajectory, "ionic_steps_observed", None),
    )
    print(f"      completed ionic steps: {_diagnosis_value(completed_ionic_steps)}")
    vasprun_steps = getattr(trajectory, "vasprun_ionic_steps", None)
    if vasprun_steps is not None:
        print(f"      vasprun ionic steps: {vasprun_steps}")
    _print_outcar_force_source(trajectory)
    recent = getattr(trajectory, "recent_ionic_steps", ())
    if recent:
        print("      recent completed ionic steps:")
        for step in recent:
            print(f"        {_format_ionic_step(step)}")
    else:
        print("      recent completed ionic steps: unavailable")


def _print_outcar_force_source(trajectory: object) -> None:
    path = getattr(trajectory, "outcar_path", None)
    if path is None:
        return
    if getattr(trajectory, "outcar_present", False):
        print(f"      OUTCAR atomic forces: present ({path})")
        complete_blocks = getattr(trajectory, "outcar_complete_force_blocks", None)
        if complete_blocks is not None:
            print(f"      OUTCAR complete force blocks: {complete_blocks}")
    else:
        print(f"      OUTCAR atomic forces: unavailable ({path})")
    expected = getattr(trajectory, "outcar_expected_site_count", None)
    if expected is not None:
        print(f"      OUTCAR expected site count: {expected}")
    if getattr(trajectory, "outcar_error", None):
        print(
            "      OUTCAR force trajectory unavailable: "
            f"{getattr(trajectory, 'outcar_error')}"
        )
        failure_kind = getattr(trajectory, "outcar_failure_kind", None)
        if failure_kind:
            diagnostic = failure_kind
            returncode = getattr(trajectory, "outcar_failure_returncode", None)
            if returncode is not None:
                diagnostic = f"{diagnostic}, exit {returncode}"
            print(f"      OUTCAR extractor diagnostic: {diagnostic}")
    alignment_status = getattr(trajectory, "outcar_force_alignment_status", None)
    alignment_reason = getattr(trajectory, "outcar_force_alignment_reason", None)
    if alignment_status and alignment_reason:
        print(f"      OUTCAR force alignment: {alignment_status} ({alignment_reason})")


def _print_trajectory_progress_evidence(trajectory: object) -> None:
    evidence = derive_trajectory_progress_evidence(trajectory)
    if evidence.atomic_force_status != "available":
        return

    print("    atomic-force trajectory summary (trajectory_progress_evidence):")
    if evidence.force_criterion_magnitude_eV_A is not None:
        print(
            "      criterion magnitude: "
            f"{_format_force(evidence.force_criterion_magnitude_eV_A)} eV/A"
        )
    print(
        "      initial/current/best: "
        f"{_format_force(evidence.initial_max_force_eV_A)} / "
        f"{_format_force(evidence.current_max_force_eV_A)} / "
        f"{_format_force(evidence.best_max_force_eV_A)} eV/A"
    )
    if evidence.best_force_step is not None:
        print(f"      best observed at ionic step: {evidence.best_force_step}")
    if evidence.initial_force_over_abs_EDIFFG is not None:
        print(
            "      relative to criterion: "
            f"initial {_format_ratio(evidence.initial_force_over_abs_EDIFFG)}, "
            f"current {_format_ratio(evidence.current_force_over_abs_EDIFFG)}, "
            f"best {_format_ratio(evidence.best_force_over_abs_EDIFFG)}"
        )
    if evidence.initial_to_best_force_ratio is not None:
        print(
            "      force ratios: "
            f"initial/current {_format_ratio(evidence.initial_to_current_force_ratio)}, "
            f"initial/best {_format_ratio(evidence.initial_to_best_force_ratio)}, "
            f"current/best {_format_ratio(evidence.current_to_best_force_ratio)}"
        )
    if evidence.new_best_force_count is not None:
        print(f"      new best observations: {evidence.new_best_force_count}")
    if evidence.min_electronic_iterations is not None:
        print(
            "      electronic iterations per completed ionic step: "
            f"min {evidence.min_electronic_iterations}, "
            f"median {_format_decimal(evidence.median_electronic_iterations)}, "
            f"max {evidence.max_electronic_iterations}"
        )
    for limitation in evidence.limitations:
        if "variable-cell convergence" in limitation:
            print(f"      note: {limitation}")


def _nelm_suffix(trajectory: object) -> str:
    criteria = getattr(trajectory, "criteria", {})
    if isinstance(criteria, Mapping) and "NELM" in criteria:
        return f" / NELM {criteria['NELM']}"
    return ""


def _format_electronic_iteration(iteration: object) -> str:
    parts = [
        f"N={_diagnosis_value(getattr(iteration, 'iteration', None))}",
    ]
    algorithm = getattr(iteration, "algorithm", None)
    if algorithm:
        parts.insert(0, str(algorithm))
    for label, attribute in (
        ("E", "energy"),
        ("dE", "dE"),
        ("deps", "deps"),
        ("rms", "rms"),
        ("rms(c)", "rms_c"),
    ):
        value = getattr(iteration, attribute, None)
        if value is not None:
            parts.append(f"{label}={value}")
    return " ".join(parts)


def _format_ionic_step(step: object) -> str:
    parts = [f"step {getattr(step, 'step_index')}"]
    for label, attribute in (
        ("electronic_iterations", "electronic_iterations"),
        ("F", "free_energy"),
        ("E0", "energy_zero"),
        ("dE", "dE"),
        ("max_force", "max_force"),
    ):
        value = getattr(step, attribute, None)
        if value is not None:
            if attribute == "max_force" and getattr(step, "max_force_source", None):
                parts.append(f"{label}={value} [{getattr(step, 'max_force_source')}]")
            else:
                parts.append(f"{label}={value}")
    return " ".join(parts)


def _format_count_sequence(values: tuple[int, ...]) -> str:
    if len(values) <= 8:
        return "[" + ", ".join(str(value) for value in values) + "]"
    recent = ", ".join(str(value) for value in values[-5:])
    return f"{len(values)} values, last five [{recent}]"


def _format_force(value: object) -> str:
    numeric = _float_or_none(value)
    return "unavailable" if numeric is None else f"{numeric:.6f}"


def _format_ratio(value: object) -> str:
    numeric = _float_or_none(value)
    return "unavailable" if numeric is None else f"{numeric:.2f}x"


def _format_decimal(value: object) -> str:
    numeric = _float_or_none(value)
    return "unavailable" if numeric is None else f"{numeric:g}"


def _diagnosis_value(value: object) -> str:
    return "unavailable" if value is None else str(value)


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_check_input_args(args: list[str]) -> tuple[dict[str, Any] | None, str | None]:
    if not args:
        return None, "missing remote directory"

    directory = args[0]
    remaining = args[1:]
    if len(remaining) % 2 != 0:
        return None, "options must be provided as --name value pairs"

    option_names = {"--stage", "--theory", "--nodes", "--ntasks", "--mem-gb"}
    values: dict[str, str] = {}
    for index in range(0, len(remaining), 2):
        name = remaining[index]
        value = remaining[index + 1]
        if name not in option_names:
            return None, f"unknown option {name}"
        if name in values:
            return None, f"duplicate option {name}"
        values[name] = value

    missing = [name for name in option_names if name not in values]
    if missing:
        return None, "missing required option(s): " + ", ".join(sorted(missing))

    try:
        nodes = _positive_cli_int(values["--nodes"], "--nodes")
        ntasks = _positive_cli_int(values["--ntasks"], "--ntasks")
        mem_gb = _positive_cli_int(values["--mem-gb"], "--mem-gb")
    except ValueError as exc:
        return None, str(exc)

    stage = values["--stage"].strip()
    theory = values["--theory"].strip()
    if not stage:
        return None, "--stage must not be empty"
    if not theory:
        return None, "--theory must not be empty"

    return {
        "directory": directory,
        "stage": stage,
        "theory": theory,
        "nodes": nodes,
        "ntasks": ntasks,
        "mem_gb": mem_gb,
    }, None


def _positive_cli_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _check_input_usage() -> str:
    return (
        "bmd-agent check-input <remote-directory> --stage <stage> "
        "--theory <theory> --nodes <n> --ntasks <n> --mem-gb <n>"
    )


def _display_value(value: object) -> str:
    return "unavailable" if value is None else str(value)


def _display_bool(value: bool) -> str:
    return "yes" if value else "no"


def _local_file_status(observation: object) -> str:
    if observation is None:
        return "not checked"
    if not getattr(observation, "present", False):
        return "missing"
    size = getattr(observation, "size", None)
    if size is None:
        return "present, size unavailable"
    if size == 0:
        return "present, empty"
    return f"present, {size} bytes"


def _local_stage_status(stage_evidence: object, completion: object = None) -> str:
    convergence = getattr(getattr(completion, "convergence", None), "status", None)
    convergence_text = {
        "converged": "convergence established",
        "not_converged": "convergence not reached",
        "undetermined": "convergence not established",
    }.get(convergence)
    if getattr(stage_evidence, "normal_completion", False):
        if convergence == "converged":
            return "completed (VASP normal termination; convergence established)"
        if convergence_text:
            return f"VASP normal termination; {convergence_text}"
        return "VASP normal termination"
    if getattr(stage_evidence, "has_meaningful_execution", False):
        return f"partial; {convergence_text}" if convergence_text else "partial"
    if getattr(stage_evidence, "has_required_inputs", False):
        return "inputs present"
    return "unavailable"


def _print_lifecycle_diagnostic_evidence(diagnostics: object) -> None:
    print("Diagnostic evidence:")
    custodian = getattr(diagnostics, "custodian", None)
    if _has_relevant_custodian_evidence((custodian,) if custodian is not None else ()):
        _print_custodian_intervention_evidence((custodian,), indent="  ")
        _print_termination_assessment(getattr(diagnostics, "termination", None), indent="  ")
    logs = getattr(diagnostics, "logs", ())
    if logs:
        print("  bounded log excerpts:")
        for log in logs:
            print(f"    {getattr(log, 'label', 'log')}: {getattr(log, 'path', '')}")
            if getattr(log, "error", None):
                print(f"      unavailable: {getattr(log, 'error')}")
            elif getattr(log, "messages", ()):
                for message in getattr(log, "messages", ()):
                    print(f"      {message}")
            else:
                print("      no fatal/error excerpt found in bounded read")
    archives = getattr(diagnostics, "error_archives", ())
    if archives:
        print("  error archives:")
        for archive in archives:
            print(f"    {getattr(archive, 'name', 'archive')}: present ({getattr(archive, 'path', '')})")
        print("    not unpacked by BMD Agent")
    if not logs and custodian is None and not archives:
        print("  unavailable: no local diagnostic logs, custodian.json, or error archives were found")


def _print_custodian_intervention_evidence(
    evidence_items: Iterable[object],
    *,
    indent: str = "",
) -> None:
    print(f"{indent}Custodian intervention evidence:")
    for evidence in evidence_items:
        print(f"{indent}  source: {getattr(evidence, 'source_path', 'unavailable')}")
        error = getattr(evidence, "error", None)
        if error:
            print(f"{indent}  unavailable: {error}")
            continue
        repeated = getattr(evidence, "repeated_interventions", ())
        repeated_positions = {
            position
            for item in repeated
            for position in getattr(item, "correction_positions", ())
        }
        for item in repeated:
            print(f"{indent}  {_short_class_name(getattr(item, 'handler', 'unknown'))}:")
            print(f"{indent}    interventions: {getattr(item, 'count', 0)}")
            timeout = getattr(item, "timeout_seconds", None)
            if timeout is not None:
                hours = float(timeout) / 3600
                print(f"{indent}    inactivity timeout: {timeout:g} s ({hours:g} h)")
            for reported_error in getattr(item, "errors", ()):
                print(f"{indent}    reported error: {reported_error}")
            for action in getattr(item, "action_summaries", ()):
                print(f"{indent}    repeated correction: {action}")
        for correction in getattr(evidence, "corrections", ()):
            position = (
                getattr(correction, "attempt_index", 0),
                getattr(correction, "correction_index", 0),
            )
            if position in repeated_positions:
                continue
            print(
                f"{indent}  correction {getattr(correction, 'sequence_index', '?')}: "
                f"{_short_class_name(getattr(correction, 'handler', 'unknown'))}"
            )
            timeout = getattr(correction, "timeout_seconds", None)
            if timeout is not None:
                print(f"{indent}    timeout: {timeout:g} s")
            for reported_error in getattr(correction, "errors", ()):
                print(f"{indent}    reported error: {reported_error}")
            for action in getattr(correction, "actions", ()):
                print(f"{indent}    correction: {getattr(action, 'summary', action)}")
        for flag in getattr(evidence, "terminal_flags", ()):
            if getattr(flag, "value", None) is True:
                print(
                    f"{indent}  terminal flag: {getattr(flag, 'name', 'unknown')}=true "
                    f"({getattr(flag, 'interpretation', 'raw Custodian flag')})"
                )
        if not getattr(evidence, "corrections", ()) and not error:
            print(f"{indent}  no correction interventions recorded")
        for limitation in getattr(evidence, "limitations", ()):
            print(f"{indent}  limitation: {limitation}")


def _print_custodian_policy_context(policy: object) -> None:
    print("Execution-policy context (producer_provenance):")
    if not getattr(policy, "available", False):
        print(f"  unavailable: {getattr(policy, 'reason', 'no persisted Custodian policy provenance')}")
        return
    for stage in getattr(policy, "stages", ()):
        print(
            f"  stage {getattr(stage, 'stage_index', '?')}: "
            f"{getattr(stage, 'stage_type', 'unknown')} / {getattr(stage, 'theory', 'unknown')}"
        )
        print(
            f"    policy: {getattr(stage, 'policy_id', 'unknown')} "
            f"v{getattr(stage, 'policy_version', 'unknown')}"
        )
        print(
            "    FrozenJobErrorHandler: "
            f"{getattr(stage, 'frozen_job_handler_status', 'unknown')}"
        )
        print(
            "    configured handlers: "
            f"{_component_names(getattr(stage, 'handlers', ()))}"
        )
        for component in getattr(stage, "handlers", ()):
            configuration = getattr(component, "configuration", {})
            if configuration:
                print(
                    f"      {_short_class_name(getattr(component, 'class_name', 'unknown'))}: "
                    f"{_format_options(configuration)}"
                )
        handler_exclusions = getattr(stage, "explicit_handler_exclusions", ())
        print(
            "    explicit handler exclusions: "
            f"{', '.join(_short_class_name(item) for item in handler_exclusions) if handler_exclusions else 'none recorded'}"
        )
        exclusions = getattr(stage, "vasp_error_exclusions", ())
        print(
            "    VaspErrorHandler exclusions: "
            f"{', '.join(exclusions) if exclusions else 'none recorded'}"
        )
        print(
            "    validators: "
            f"{_component_names(getattr(stage, 'validators', ()))}"
        )
        validators_source = getattr(stage, "validators_source", None)
        if validators_source:
            print(f"    validators source: {validators_source}")
        print(
            "    walltime authority: "
            f"{getattr(stage, 'walltime_authority', None) or 'unavailable'}"
        )
        walltime_handler = getattr(stage, "walltime_handler", None)
        print(
            "    internal walltime handler: "
            f"{'none' if walltime_handler is None else walltime_handler}"
        )
        print(
            "    Custodian version: "
            f"{getattr(stage, 'custodian_version', None) or 'unavailable'}"
        )
        implementation_source = getattr(stage, "implementation_source", None)
        if implementation_source:
            print(f"    implementation source: {implementation_source}")
        rationale = getattr(stage, "rationale", None)
        if rationale:
            print(f"    rationale: {rationale}")


def _print_termination_assessment(
    assessment: object | None,
    *,
    indent: str = "",
) -> None:
    if assessment is None:
        return
    print(f"{indent}termination assessment:")
    print(f"{indent}  classification: {getattr(assessment, 'classification', 'unknown')}")
    print(f"{indent}  status: {getattr(assessment, 'status', 'insufficient_evidence')}")
    for basis in getattr(assessment, "basis", ()):
        print(f"{indent}  basis: {basis}")
    for limitation in getattr(assessment, "limitations", ()):
        print(f"{indent}  limitation: {limitation}")


def _has_relevant_custodian_evidence(evidence_items: Iterable[object]) -> bool:
    return any(
        getattr(evidence, "error", None)
        or getattr(evidence, "corrections", ())
        or any(
            getattr(flag, "value", None) is True
            for flag in getattr(evidence, "terminal_flags", ())
        )
        for evidence in evidence_items
    )


def _component_names(components: Iterable[object]) -> str:
    names = [
        _short_class_name(getattr(component, "class_name", "unknown"))
        for component in components
    ]
    return ", ".join(names) if names else "none recorded"


def _short_class_name(value: object) -> str:
    return str(value).rsplit(".", 1)[-1]


def _print_lifecycle_suggested_checks(diagnostics: object) -> None:
    suggestions = getattr(diagnostics, "suggested_checks", ())
    print("Suggested checks:")
    if not suggestions:
        print("  No additional diagnostic checks were suggested from local evidence.")
        return
    for suggestion in suggestions:
        print(f"  {suggestion}")


def _print_oom_evidence(evidence: OomDiagnosticEvidence | None) -> None:
    print("Memory / OOM evidence (oom_diagnostic_evidence):")
    if evidence is None:
        print("  unavailable: OOM diagnostic evidence was not derived")
        return
    print(f"  assessment: {evidence.assessment}")
    _print_memory_observation("requested memory", evidence.requested_memory)
    _print_memory_observation("allocated memory", evidence.allocated_memory)
    _print_memory_observation("maximum RSS", evidence.maximum_rss)
    _print_memory_observation("maximum VM size", evidence.maximum_vm_size)
    _print_memory_observation("average RSS", evidence.average_rss)
    if evidence.memory_utilization_percent is not None:
        print(f"  scheduler-derived memory utilization: {evidence.memory_utilization_percent:.1f}%")
    print(
        "  scheduler OOM state: "
        f"{'observed' if evidence.scheduler_oom_state else 'not observed'}"
    )
    if evidence.explicit_evidence:
        print("  explicit evidence:")
        for marker in evidence.explicit_evidence:
            print(f"    {marker.source}: {marker.text}")
    elif evidence.suggestive_evidence:
        print("  suggestive evidence:")
        for marker in evidence.suggestive_evidence:
            print(f"    {marker.source}: {marker.text}")
    else:
        print("  positive OOM markers: none observed in inspected sources")
    for limitation in evidence.limitations:
        print(f"  limitation: {limitation}")
    for guidance in evidence.guidance:
        print(f"  guidance: {guidance}")


def _print_memory_observation(
    label: str,
    observation: MemoryObservation | None,
) -> None:
    if observation is None:
        print(f"  {label}: unavailable")
        return
    print(
        f"  {label}: {observation.raw_value} "
        f"[{observation.scope}; {observation.source}]"
    )


def _format_input_value(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def bmd_compute_repository(registry: ResourceRegistry) -> GitRepositoryResource:
    """Return the configured BMD Compute repository resource."""

    try:
        return registry.repositories["bmd_compute"]

    except KeyError:
        pass

    for repository in registry.repositories.values():
        if repository.role == "compute":
            return repository

    raise ConfigurationError(
        "Resource configuration must include a BMD Compute repository resource."
    )


def powerslurm_cluster(registry: ResourceRegistry) -> SlurmClusterResource:
    """Return the configured PowerSLURM resource."""

    try:
        return registry.clusters["powerslurm"]

    except KeyError as exc:
        raise ConfigurationError(
            "Resource configuration must include [clusters.powerslurm]."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    """BMD Agent command-line entry point."""

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return show_current_directory(
            detailed_evidence_command="bmd-agent --verbose",
        )

    if argv == ["--verbose"]:
        return show_current_directory(verbose=True)

    command = argv[0]

    try:
        if command == "status":
            return show_status()

        if command == "queue":
            return show_queue()

        if command == "job":
            options = _parse_job_options(argv[2:], allow_trajectory_json=True)
            if len(argv) < 2 or options is None:
                print(
                    "Usage: bmd-agent job <SLURM_JOB_ID> "
                    "[--trajectory-json | --verbose [--profile] | --profile]"
                )
                return 2
            trajectory_json, verbose, profile = options
            return show_job(
                argv[1],
                trajectory_json=trajectory_json,
                verbose=verbose,
                profile=profile,
                detailed_evidence_command=f"bmd-agent job {argv[1]} --verbose",
            )

        if command == "compute":
            return show_compute()

        if command == "structure":
            if len(argv) < 2:
                print("Usage: bmd-agent structure <remote-directory>")
                return 2

            return show_structure(argv[1])

        if command == "check-input":
            return show_check_input(argv[1:])

        if command == "inspect-run":
            if len(argv) < 2:
                print("Usage: bmd-agent inspect-run <remote-flow-root>")
                return 2

            return show_inspect_run(argv[1])

        if command == "compare-runs":
            return show_compare_runs(argv[1:])

        if command == "diagnose-run":
            if len(argv) < 2:
                print("Usage: bmd-agent diagnose-run <remote-flow-root>")
                return 2

            return show_diagnose_run(argv[1])

        if _is_positive_decimal_job_id(command):
            options = _parse_job_options(argv[1:], allow_trajectory_json=False)
            if options is None:
                print("Usage: bmd-agent <SLURM_JOB_ID> [--verbose] [--profile]")
                return 2
            _, verbose, profile = options
            return show_job(
                command,
                verbose=verbose,
                profile=profile,
                detailed_evidence_command=f"bmd-agent {command} --verbose",
            )

        if len(argv) in {1, 2} and (len(argv) == 1 or argv[1] == "--verbose"):
            target = _existing_target_path(command)
            if target is not None:
                verbose = len(argv) == 2
                return show_current_directory(
                    target,
                    verbose=verbose,
                    detailed_evidence_command=(
                        None
                        if verbose
                        else f"bmd-agent {_quote_cli_target(command)} --verbose"
                    ),
                )

            print(
                "Target was not recognized as a SLURM job ID or existing "
                "calculation path."
            )
            return 2

    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print("Target was not recognized as a SLURM job ID or existing calculation path.")
    print()
    print("Usage: bmd-agent [TARGET] [--verbose] [--profile]")
    print("  no target [--verbose]: analyze the current calculation directory")
    print("  positive decimal integer [--verbose] [--profile]: analyze that SLURM job")
    print("  existing filesystem path [--verbose]: analyze that calculation directory")
    print()
    print("Expert commands:")
    print("  status")
    print("  queue")
    print("  job <SLURM_JOB_ID> [--trajectory-json | --verbose [--profile] | --profile]")
    print("  compute")
    print("  structure <remote-directory>")
    print(f"  {_check_input_usage()}")
    print("  inspect-run <remote-flow-root>")
    print("  compare-runs <flow-a> <flow-b> [<flow-c> ...]")
    print("  diagnose-run <remote-flow-root>")
    return 2


def _is_positive_decimal_job_id(target: str) -> bool:
    return target.isascii() and target.isdecimal() and int(target) > 0


def _existing_target_path(target: str) -> Path | None:
    try:
        path = Path(target).expanduser()
        return path if path.exists() else None
    except (OSError, RuntimeError):
        return None


def _parse_job_options(
    options: list[str],
    *,
    allow_trajectory_json: bool,
) -> tuple[bool, bool, bool] | None:
    allowed = {"--verbose", "--profile"}
    if allow_trajectory_json:
        allowed.add("--trajectory-json")
    if len(options) != len(set(options)) or any(option not in allowed for option in options):
        return None
    trajectory_json = "--trajectory-json" in options
    if trajectory_json and len(options) != 1:
        return None
    return trajectory_json, "--verbose" in options, "--profile" in options


def _path_detail_command(target: Path | None) -> str:
    if target is None:
        return "bmd-agent --verbose"
    return f"bmd-agent {_quote_cli_target(str(target))} --verbose"


def _quote_cli_target(target: str) -> str:
    return f'"{target}"' if any(character.isspace() for character in target) else target


def compute_policy_from_registry(
    registry: ResourceRegistry,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> ComputePolicyObservation:
    """Return BMD Compute policy evidence, or why it is unavailable."""

    try:
        repository = bmd_compute_repository(registry)
    except ConfigurationError as exc:
        return compute_policy_not_configured(str(exc))
    return observe_compute_policies(repository, runner=runner)


_POLICY_ALIGNMENT_TEXT = {
    POLICY_ALIGNMENT_SAME: "same commit as the current BMD Compute checkout",
    POLICY_ALIGNMENT_SAME_UNVERIFIED: (
        "same commit, but a clean worktree was not verified for both checkouts; "
        "the policy used when this run was produced is not established"
    ),
    POLICY_ALIGNMENT_DIFFERENT: (
        "different commit; checks use the current checkout's policy, which is not "
        "established to be the policy used when this run was produced"
    ),
}


def _print_compute_policy_context(
    policy: ComputePolicyObservation | None,
    *,
    run_labels: tuple[tuple[str | None, Mapping[str, Any]], ...],
    has_checks: bool,
    trailing_blank: bool = True,
) -> None:
    """Print where input-effect expectations came from, or why they are absent."""

    if policy is None:
        return
    print(f"BMD Compute input-effect policy ({policy.evidence_type}):")
    if not policy.available:
        label = "not configured" if policy.status == COMPUTE_POLICY_NOT_CONFIGURED else "unavailable"
        print(f"  status: {label}")
        if policy.reason:
            print(f"  reason: {policy.reason}")
        print("  requested/executed checks were not performed; this is missing evidence, not a finding")
        print()
        return
    print(f"  status: available ({len(policy.policies)} polic{'y' if len(policy.policies) == 1 else 'ies'})")
    print(
        f"  source: current checkout {_display_commit(policy.commit)} "
        f"({_display_dirty_state(policy.dirty)})"
    )
    for label, producer_git in run_labels:
        prefix = f"{label} " if label else ""
        print(
            f"  {prefix}run produced by: {_display_commit(producer_git.get('git_commit'))} "
            f"({producer_git.get('state', 'unavailable')})"
        )
        alignment = compute_policy_alignment(policy, producer_git)
        print(
            f"  {prefix}policy alignment: "
            + _POLICY_ALIGNMENT_TEXT.get(
                alignment,
                "unknown; commit provenance is incomplete, so the policy used when "
                "this run was produced is not established",
            )
        )
    if not has_checks:
        print("  checks: no policy applies to the requested options")
    if trailing_blank:
        print()


def _display_commit(value: object) -> str:
    if isinstance(value, str) and value:
        return value[:12]

    return "unavailable"


def _display_dirty_state(value: object) -> str:
    if value is True:
        return "dirty"

    if value is False:
        return "clean"

    return "unavailable"


def _display_theory(value: object) -> str:
    return str(value).upper()


def _display_stage(value: object) -> str:
    text = str(value)
    labels = {
        "band_structure": "Band Structure",
        "dos": "DOS",
        "relax": "Geometry Optimisation",
        "static": "Static Energy",
    }

    if text in labels:
        return labels[text]

    return text.replace("_", " ").title()


def _print_mapping_values(mapping: object, keys: tuple[str, ...]) -> None:
    if not isinstance(mapping, dict):
        return
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            print(f"  {key}: {value}")


def _present_text(observation: object) -> str:
    path = getattr(observation, "path")
    state = "present" if getattr(observation, "present") else "absent"
    return f"{state} ({path})"


def _print_optional_value(label: str, value: object) -> None:
    if value is not None:
        print(f"  {label}: {value}")


def _print_scientific_result(scientific: object) -> None:
    if getattr(scientific, "error", None):
        print(f"  unavailable: {getattr(scientific, 'error')}")
        return
    _print_optional_value("final formula", getattr(scientific, "final_formula", None))
    structure = getattr(scientific, "structure", None)
    if structure:
        _print_structure_observation(structure)
    _print_optional_value("final energy eV", getattr(scientific, "final_energy_ev", None))
    _print_optional_value("energy/atom eV", getattr(scientific, "energy_per_atom_ev", None))
    _print_optional_value("electronic convergence", getattr(scientific, "electronic_convergence", None))
    _print_optional_value("band gap eV", getattr(scientific, "band_gap_ev", None))
    _print_optional_value("band k-points", getattr(scientific, "band_kpoints", None))
    _print_optional_value("bands", getattr(scientific, "bands", None))
    for item in getattr(scientific, "unavailable", ()):
        print(f"  unavailable: {item}")


def _print_structure_observation(observation: object) -> None:
    print("  final structure:")
    _print_optional_nested_value("formula", getattr(observation, "formula", None))
    _print_optional_nested_value("reduced formula", getattr(observation, "reduced_formula", None))
    _print_optional_nested_value("site count", getattr(observation, "site_count", None))
    _print_optional_nested_value("lattice a A", getattr(observation, "lattice_a", None))
    _print_optional_nested_value("lattice b A", getattr(observation, "lattice_b", None))
    _print_optional_nested_value("lattice c A", getattr(observation, "lattice_c", None))
    _print_optional_nested_value("alpha deg", getattr(observation, "alpha", None))
    _print_optional_nested_value("beta deg", getattr(observation, "beta", None))
    _print_optional_nested_value("gamma deg", getattr(observation, "gamma", None))
    _print_optional_nested_value("volume A^3", getattr(observation, "volume", None))
    _print_optional_nested_value("density g/cm^3", getattr(observation, "density", None))
    _print_optional_nested_value("c/a cell-axis ratio", getattr(observation, "c_over_a", None))
    for item in getattr(observation, "unavailable", ()):
        print(f"    unavailable: {item}")


def _print_optional_nested_value(label: str, value: object) -> None:
    if value is not None:
        print(f"    {label}: {value}")


def _format_options(options: Mapping[str, Any]) -> str:
    flattened = []
    for key, value in _flatten_options(options):
        flattened.append(f"{key}={value}")
    return ", ".join(flattened)


def _flatten_options(options: Mapping[str, Any], prefix: str = "") -> list[tuple[str, object]]:
    flattened: list[tuple[str, object]] = []
    for key, value in sorted(options.items()):
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.extend(_flatten_options(value, path))
        else:
            flattened.append((path, value))
    return flattened


def _print_executed_input_summary(inspection: RunInspection) -> None:
    _print_executed_input_observations(inspection.executed_inputs, inspection.workflow_stages)


def _print_executed_input_observations(
    executed_inputs: tuple[object, ...],
    workflow_stages: tuple[object, ...],
) -> None:
    groups = _executed_input_groups(executed_inputs)
    if not groups:
        print("  unavailable: no executed-input observations were gathered")
        return

    stages = {getattr(stage, "index"): stage for stage in workflow_stages}
    for stage_index, observations in groups:
        print(f"  {_executed_stage_heading(stage_index, observations, stages)}:")
        for observation in observations:
            print(f"    source {_executed_source_label(observation)}: {_executed_source_state(observation)}")

        settings, discrepancies = _reconciled_executed_settings(observations)
        if settings:
            print(f"    settings: {', '.join(settings)}")
        elif any(_readable_executed_source(observation) for observation in observations):
            print("    settings: no displayed INCAR settings present")
        else:
            print("    settings: unavailable")

        for key, source_values in discrepancies:
            values = "; ".join(
                f"{source}={_format_executed_value(key, value, fingerprint=True)}"
                for source, value in source_values
            )
            print(f"    discrepancy: {key}: {values}")


def _executed_input_groups(
    observations: tuple[object, ...],
) -> list[tuple[int | None, tuple[object, ...]]]:
    grouped: dict[int | None, list[object]] = {}
    ordered_keys: list[int | None] = []
    for observation in observations:
        key = getattr(observation, "stage_index", None)
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append(observation)

    return [
        (key, tuple(grouped[key]))
        for key in sorted(
            ordered_keys,
            key=lambda value: (value is None, value if isinstance(value, int) else 0),
        )
    ]


def _executed_stage_heading(
    stage_index: int | None,
    observations: tuple[object, ...],
    stages: Mapping[int, object],
) -> str:
    if stage_index is None:
        base = "unindexed"
    else:
        stage = stages.get(stage_index)
        if stage is None:
            base = f"stage {stage_index}"
        else:
            base = (
                f"stage {stage_index}: "
                f"{_display_theory(getattr(stage, 'theory'))} "
                f"{_display_stage(getattr(stage, 'stage_type'))}"
            )

    labels = _ordered_unique(
        str(getattr(observation, "label"))
        for observation in observations
        if getattr(observation, "source_type", None) == "retained_incar"
    )
    if labels:
        return f"{base} ({', '.join(labels)})"
    return base


def _executed_source_label(observation: object) -> str:
    source_type = getattr(observation, "source_type")
    label = getattr(observation, "label")
    return f"{source_type}:{label}"


def _executed_source_state(observation: object) -> str:
    path = getattr(observation, "path")
    if not getattr(observation, "present"):
        return f"unavailable ({path})"
    error = getattr(observation, "error")
    if error:
        return f"present but unparsed ({path}): {error}"
    return f"present ({path})"


def _reconciled_executed_settings(
    observations: tuple[object, ...],
) -> tuple[list[str], list[tuple[str, list[tuple[str, object]]]]]:
    readable = [
        observation
        for observation in observations
        if _readable_executed_source(observation)
    ]
    settings: list[str] = []
    discrepancies: list[tuple[str, list[tuple[str, object]]]] = []

    for key in _EXECUTED_INPUT_DISPLAY_KEYS:
        source_values = [
            (_executed_source_label(observation), getattr(observation, "values").get(key))
            for observation in readable
            if key in getattr(observation, "values")
        ]
        if not source_values:
            continue

        values = [value for _, value in source_values]
        if _all_values_match(values):
            context = _first_mapping_with_key(readable, key)
            settings.append(f"{key}={_format_executed_value(key, values[0], context)}")
        else:
            discrepancies.append((key, source_values))

    return settings, discrepancies


def _readable_executed_source(observation: object) -> bool:
    return (
        bool(getattr(observation, "present"))
        and getattr(observation, "error") is None
        and isinstance(getattr(observation, "values"), Mapping)
    )


def _first_mapping_with_key(observations: list[object], key: str) -> Mapping[str, object]:
    for observation in observations:
        values = getattr(observation, "values")
        if isinstance(values, Mapping) and key in values:
            return values
    return {}


def _all_values_match(values: list[object]) -> bool:
    if not values:
        return True
    first = values[0]
    return all(_display_values_match(first, value) for value in values[1:])


def _display_values_match(left: object, right: object) -> bool:
    return _format_canonical_value(left) == _format_canonical_value(right)


def _format_canonical_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(_format_canonical_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_format_canonical_value(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _format_canonical_value(item)) for key, item in value.items())
        )
    return value


def _expectation_observed_text(expectation: object) -> str:
    status = getattr(expectation, "status")
    value = getattr(expectation, "observed_value")
    if status == "unavailable":
        return "unavailable"
    if status == "absent":
        return "absent"
    formatted = _format_executed_value(
        getattr(expectation, 'input_key'),
        value,
        fingerprint=status == 'discrepancy',
    )
    return f"observed {formatted}"


def _expectation_expected_text(expectation: object) -> str:
    return _format_executed_value(
        getattr(expectation, "input_key"),
        getattr(expectation, "expected_value"),
        fingerprint=getattr(expectation, "status") == "discrepancy",
    )


def _expectation_source_value(expectation: object, value: object) -> str:
    return _format_executed_value(
        getattr(expectation, "input_key"),
        value,
        fingerprint=getattr(expectation, "status") == "discrepancy",
    )


def _format_executed_value(
    key: object,
    value: object,
    context: Mapping[str, object] | None = None,
    *,
    fingerprint: bool = False,
) -> str:
    if value is None:
        return "absent"
    if str(key).upper() == "MAGMOM":
        return _format_magmom(value, context or {}, fingerprint=fingerprint)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        return _format_sequence(value)
    if isinstance(value, Mapping):
        return "{" + ", ".join(
            f"{item_key}: {_format_executed_value(item_key, item_value)}"
            for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
        ) + "}"
    return str(value)


def _format_sequence(value: list[object] | tuple[object, ...]) -> str:
    if len(value) <= 8:
        return "[" + ", ".join(_format_executed_value("", item) for item in value) + "]"
    return f"present, {len(value)} values"


def _format_magmom(
    value: object,
    context: Mapping[str, object],
    *,
    fingerprint: bool = False,
) -> str:
    flattened = _flatten_numeric_values(value)
    if flattened is None:
        return "present"
    count = len(flattened)
    if count == 0:
        return "present, 0 values"
    if count <= 8:
        return "[" + ", ".join(_format_executed_value("", item) for item in flattened) + "]"

    noncollinear = (
        _truthy_vasp_value(context.get("LNONCOLLINEAR"))
        or _truthy_vasp_value(context.get("LSORBIT"))
    )
    if noncollinear and count % 3 == 0:
        summary = f"present, {count // 3} sites / {count} noncollinear components"
    else:
        summary = f"present, {count} sites"
    if fingerprint:
        summary = f"{summary}, fingerprint {_value_fingerprint(value)}"
    return summary


def _value_fingerprint(value: object) -> str:
    canonical = repr(_format_canonical_value(value)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:12]


def _flatten_numeric_values(value: object) -> list[object] | None:
    if isinstance(value, int | float):
        return [value]
    if not isinstance(value, list | tuple):
        return None
    flattened: list[object] = []
    for item in value:
        nested = _flatten_numeric_values(item)
        if nested is None:
            return None
        flattened.extend(nested)
    return flattened


def _truthy_vasp_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().strip(".").upper() in {"T", "TRUE"}
    return bool(value)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


if __name__ == "__main__":
    raise SystemExit(main())
