# bmd-check Architecture

## 1. Purpose

bmd-check is the observational diagnostic layer of the BMD ecosystem.

It does not replace the systems it observes.

Its role is to connect information from those systems so that
researchers can understand what is known, what has been calculated,
what has been validated, what has failed, and what should happen next.

## 2. Ecosystem boundaries

The core responsibilities are:

    bmd-compute   generate
    bmd-store     preserve
    bmd-help      explain
    BMD website   present and connect
    bmd-check     inspect, diagnose, explain, and advise

These boundaries are architectural invariants.

bmd-compute must function without bmd-check.

bmd-store must remain independently authoritative for curated supporting
scientific data, reference evidence, and non-core scientific tools.

bmd-help must remain independently usable as human-oriented
documentation.

The BMD website remains the public face of the group.

bmd-check may consume information from all of these systems without
requiring them to become components of the agent.

In this ecosystem, bmd-compute is the core VASP data-generation pipeline. It
owns the authoritative implementation of decisions required to construct,
validate, execute, and provenance BMD VASP calculations. bmd-store contains
BMD-curated supporting scientific data, reference evidence, and non-core
scientific tools outside that pipeline. bmd-check consumes exposed capabilities
and evidence to inspect, diagnose, explain, and advise without duplicating
their authority.

If a capability determines how BMD generates a VASP calculation, its
authoritative implementation belongs in bmd-compute. If it provides supporting
scientific data or tooling but is not part of the core VASP data-generation
pipeline, it belongs in bmd-store.

## 3. Scientific architecture

The following long-term scientific model is speculative architecture, not a
description of bmd-check's current functionality or authority:

                         Human researcher
                                |
                                v
                           bmd-check
                                |
                scientific reasoning / planning
                                |
          +---------------------+---------------------+
          |                     |                     |
          v                     v                     v
       Knowledge             Analysis              Generation
       bmd-store             pymatgen              bmd-compute
       bmd-help              ML models                 |
       literature            other tools               v
       Materials Project                          PowerSLURM
                                                       |
                                                       v
                                                      VASP
          |                     |                     |
          +---------------------+---------------------+
                                |
                                v
                             Evidence
                                |
                                v
                     scientific interpretation
                                |
                                v
                         human validation
                                |
                                v
                              bmd-store

AI/ML can progressively accelerate this loop while physics-based
validation remains central.

## 4. Observation and action planes

Observation and action are separate security domains.

### Observation plane

The observation plane includes operations such as:

- reading repository files;
- inspecting Git state and history;
- searching bmd-store;
- reading bmd-help;
- reading public BMD website information;
- inspecting PowerSLURM queue state;
- inspecting SLURM accounting;
- reading authorized calculation files;
- parsing VASP outputs;
- performing pymatgen analysis;
- querying Materials Project;
- running derived analyses in designated agent workspaces.

### Action plane

The action plane includes operations such as:

- modifying authoritative files;
- changing Git working trees;
- committing or pushing;
- merging changes;
- submitting calculations;
- cancelling calculations;
- changing running jobs;
- restarting services;
- deploying software;
- publishing information;
- changing BMD scientific standards.

bmd-check v0 has no action plane.

Future action capabilities must be introduced deliberately and must
require appropriate authorization.

## 5. Current infrastructure

bmd-check is hosted on the BMD university Linux VM.

Current project layout:

    ~/projects/
        bmd-compute/
        bmd-store/
        bmd-check/

The bmd-compute checkout is a live operational checkout and must be
treated as a protected resource.

bmd-compute currently runs from this checkout using Uvicorn.

The agent accesses PowerSLURM through the dedicated SSH identity:

    powerslurm-bmdguest

which connects as:

    bmdguest

The intended scheduler scope for normal bmd-check operation is:

    leeburton-pool

The agent should not expose unrelated university cluster activity merely
because the underlying scheduler permits it to inspect that activity.

## 6. Cluster data model

PowerSLURM represents current computational reality, not permanent BMD
knowledge.

Relevant resources include:

- queue state;
- accounting state;
- job metadata;
- group calculation directories;
- VASP inputs and outputs;
- logs and scheduler output.

The `bmdguest` account is intended to observe group resources without
modifying researchers' files.

bmd-check-owned scratch space may be used for temporary analysis.

Temporary agent artifacts do not become authoritative scientific
knowledge merely because the agent generated them.

Scientifically valuable datasets or conclusions that should persist
beyond individual projects should be deliberately curated into bmd-store.

## 7. People and identity

The BMD website represents the public-facing view of group people,
publications, teaching, and resources.

PowerSLURM users are not equivalent to the group membership roster.

Likewise, the following are distinct identities:

- public/person identity;
- PowerSLURM identity;
- filesystem identity;
- GitHub identity;
- publication authorship;
- project ownership.

The agent must not infer that these identities refer to the same person
solely from similar names.

Relationships between identities should eventually be represented
explicitly.

## 8. Scientific task model

Substantial scientific work should eventually be represented as
persistent tasks rather than only ephemeral conversations.

The conceptual objects are:

### Task

A scientific, technical, or operational question being investigated.

### Observation

Something directly observed from an authoritative or identified source.

### Artifact

A file, calculation output, table, plot, structured result, or other
object produced or inspected during a task.

### Failure

A failed calculation, parsing error, unavailable resource, rejected
candidate, or unsuccessful operation.

Failures are first-class scientific and operational information.

### Evidence

Information supporting or contradicting a claim.

Evidence must retain its origin and method.

### Claim

An interpretation supported to some degree by evidence.

### Recommendation

A proposed next step.

### Decision

A human-approved conclusion or choice.

### Approval

Authorization permitting a protected transition or action.

These concepts are inspired partly by reproducible scientific-agent
workflows in the literature, including systems that retain intermediate
artifacts, failed candidates, explicit screening criteria, and
validation gaps.

The initial implementation does not need to implement every object
immediately.

## 9. Evidence and validation

bmd-check must distinguish different validation states.

Candidate states include:

    Proposed
    Implemented
    Tested locally
    Tested remotely
    Validated on PowerSLURM
    Adopted standard
    Deprecated

These states are not interchangeable.

For example:

    code exists
        !=
    code was tested
        !=
    workflow ran successfully on PowerSLURM
        !=
    scientific methodology was validated
        !=
    methodology became a BMD standard

Scientific evidence should also preserve its type.

Examples include:

- experimental evidence;
- first-principles calculations;
- derived computational analysis;
- external database evidence;
- literature evidence;
- ML-derived predictions;
- human scientific assessment.

## 10. Tool boundaries

bmd-check consumes controlled observational interfaces and evidence from
specialist tools rather than reproducing their functionality.

Examples:

    structure manipulation       -> pymatgen
    electronic structure         -> VASP
    workflow generation          -> atomate2/jobflow
    scheduling                   -> SLURM
    version control              -> Git
    remote collaboration         -> GitHub
    materials reference data     -> Materials Project

Current bmd-check does not coordinate or execute calculations through these
systems. Any future orchestration or action capability is speculative and would
require a separately reviewed, controlled interface and explicit authorization.

## 11. Resource registry

Resources should be explicitly registered rather than discovered and
trusted automatically.

A resource definition should eventually describe concepts such as:

    identity
    type
    role
    location
    authority
    access policy
    capabilities
    protection status

For example, conceptually:

    bmd_compute:
        type: git_repository
        role: compute
        status: live
        access: read_only
        protection: high

    bmdex:
        type: git_repository
        role: supporting_scientific_data_and_evidence
        access: read_only

    powerslurm:
        type: slurm_cluster
        role: compute_observation
        ssh_identity: powerslurm-bmdguest
        scheduler_scope: leeburton-pool
        access: observational

## 12. Runtime direction

The initial runtime should remain simple.

Likely components include:

- Python;
- a small command-line interface;
- explicit resource adapters;
- structured configuration;
- tests;
- audit logging.

Persistent state may later use SQLite plus ordinary structured files.

An HTTP API or web interface may be added when justified.

The Burton Materials Discovery Lab website may eventually provide an interface to bmd-check,
but this is not an initial architectural requirement.

## 13. LLM boundary

The agent runtime and the language model are separate concepts.

The runtime owns:

- tools;
- permissions;
- resource definitions;
- persistent state;
- evidence;
- tasks;
- audit records.

The model provides capabilities such as:

- interpretation;
- planning;
- reasoning;
- synthesis;
- natural-language interaction.

A model must not be able to bypass runtime permissions.

The model should not receive unrestricted shell authority simply because
the underlying operating-system account possesses it.

## 14. Development trajectory

The following trajectory is speculative planning material, not a claim of
implemented capability. Current bmd-check remains the read-only observer
described above.

### v0 — Observer

- resource registry;
- read-only Git inspection;
- bmd-compute inspection;
- bmd-store inspection;
- PowerSLURM queue inspection;
- calculation-file inspection;
- pymatgen/VASP parsing;
- Materials Project access;
- evidence-backed troubleshooting.

### v0.5 — Persistent investigations

- task IDs;
- observations;
- artifacts;
- failures;
- provenance;
- recommendations;
- audit records.

### v1 — Scientific task planning

- structured objectives;
- explicit constraints;
- scientific assumptions;
- tool selection;
- validation-gap analysis;
- reproducible reports.

### v2 — Computational proposals

- formulate validation strategies;
- propose bmd-compute workflows;
- prepare proposed changes without executing protected actions.

### v3 — Controlled action plane

- explicit PI approval;
- selected repository actions;
- selected computational actions;
- full audit trail.

### Later — AI-accelerated discovery

Potential capabilities include:

- ML screening;
- generative materials models;
- adaptive candidate funnels;
- automated validation planning;
- specialized scientific agents;
- iterative physics-based discovery loops.

These later capabilities must grow from validated lower-level
infrastructure rather than being assumed at project inception.
