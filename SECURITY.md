# bmd-check Security Model

## 1. Core rule

bmd-check v0 has no authority to mutate authoritative BMD resources.

This is a system design requirement, not merely an instruction to the
language model.

Observation and action are separate security domains.

## 2. Governance

bmd-check is PI-governed.

Students and other group members may:

- use the agent;
- inspect results;
- conduct investigations;
- propose improvements;
- contribute code to the bmd-check project;
- use agent development as a learning exercise.

They may not use the agent to make authoritative changes without PI
approval.

Future action capabilities must preserve this boundary.

## 3. Protected resources

Protected resources include, but are not limited to:

- the live bmd-compute checkout;
- bmd-store;
- bmd-help;
- student and researcher project files;
- PowerSLURM jobs;
- BMD scientific datasets;
- deployment configuration;
- Git remotes;
- public-facing BMD resources.

## 4. bmd-compute

The local bmd-compute checkout is a live production resource.

Current location:

    ~/projects/bmd-compute

The running bmd-compute service uses this checkout.

The agent may inspect it but must not modify it in v0.

Examples of allowed operations:

- read source;
- read configuration;
- inspect Git status;
- inspect Git history;
- inspect commit identity;
- inspect tests;
- search files.

Examples of prohibited operations:

- edit files;
- git pull;
- git checkout;
- git reset;
- git clean;
- git commit;
- git push;
- modify the Python environment;
- restart Uvicorn;
- deploy changes.

A dirty working tree must be reported, not "fixed."

## 5. bmd-store

bmd-store is authoritative for curated supporting scientific data, reference
evidence, and non-core scientific tools. It is not the authority for
bmd-compute's core VASP data-generation implementation.

bmd-check v0 may read and search bmd-store.

It may not:

- modify bmd-store records;
- promote an observation into a standard;
- commit;
- push;
- merge;
- declare a standard adopted.

Scientific observations made by the agent do not automatically become
BMD knowledge.

## 6. PowerSLURM

bmd-check observes PowerSLURM in one of two deployment modes, selected by
`ssh_host` in the deployment-local `resources.toml`:

- **Remote mode** (`ssh_host` is an SSH host alias): bmd-check runs elsewhere and
  connects over SSH with the dedicated observational guest identity.

      SSH alias:        powerslurm-bmdguest
      Remote identity:  bmdguest

  The SSH invocation is built entirely by bmd-check with `BatchMode=yes`, so a
  missing key or unknown host fails immediately instead of prompting.

- **Local mode** (`ssh_host = "local"`): bmd-check runs on the cluster itself, for
  example from a shared installation on a login node, and SSH is not used.
  The same fixed observational operations execute directly **under the Unix
  identity of the user who invoked `bmd-check`/`bmd-agent`**, with that
  user's environment (including `PATH` for `sacct`/`squeue`). Local mode is
  POSIX-only; selecting it on another platform is a configuration error.

Operating-system permissions therefore differ by deployment mode. In remote
mode the boundary is the `bmdguest` identity's permissions; in local mode it is
each invoking user's own permissions, which may be broader or narrower than
`bmdguest`'s and may differ between users. In both modes those OS permissions
remain part of the security boundary (see section 9).

Normal scheduler inspection should be restricted to the BMD group
scope:

    leeburton-pool

The agent should not expose unrelated TAU cluster activity merely
because SLURM permits broad scheduler visibility.

Allowed cluster operations include:

- inspect queue state;
- inspect job metadata;
- inspect accounting information;
- read authorized calculation files;
- inspect VASP inputs and outputs;
- inspect scheduler logs;
- copy information into an explicitly designated agent workspace when
  necessary for analysis.

Prohibited operations include:

- sbatch;
- scancel;
- scontrol update or other state-changing scheduler operations;
- modifying researcher files;
- deleting researcher files;
- moving or renaming researcher files;
- changing permissions;
- altering running calculations.

## 7. bmd-check workspace

The agent may require writable storage for:

- temporary copies;
- parsed data;
- derived analyses;
- reports;
- indexes;
- task state;
- logs.

Writable agent storage must be explicitly designated.

Writable agent storage is not authoritative BMD scientific storage.

The agent must not treat an artifact as validated merely because it
exists in its workspace.

## 8. Command execution

The long-term public tool interface should prefer explicit operations
over unrestricted shell access.

Prefer interfaces such as:

    get_bmd_queue()
    get_job(job_id)
    inspect_repository(name)
    read_calculation(path)
    parse_vasp_output(path)

rather than:

    shell(command)

Internal development may temporarily use lower-level mechanisms, but the
security model must not depend on the language model voluntarily
avoiding dangerous commands.

Current producer integrations invoke fixed Python modules from explicitly
configured bmd-compute and bmd-store checkouts. They do not expose a user-selected
module or arbitrary shell command. These checkouts and their configured Python
environments are trusted code dependencies: imported module code executes with
the operating-system privileges of the bmd-check caller. bmd-check must not be
configured to execute untrusted third-party checkout code.

Observational scheduler and file reads are a closed set of typed operations
(squeue for one partition, sacct for one job, read/tail/stat/test of one
authorized path, a bounded error-archive probe, the OUTCAR force extractor,
and a batched metadata acquisition). Paths, job IDs, partitions and byte
limits are separately validated values. Each operation renders one fixed
argument vector:

- in local mode it runs as an argument vector with no shell;
- in remote mode it is shell-quoted as a whole for the remote login shell, and
  bmd-check builds the `ssh` command itself, allowing only its own options
  (`BatchMode=yes` first and unconditionally, `ConnectTimeout` from typed
  configuration, and a reusable session's own control-socket options).

Two operations (the archive probe and the batched acquisition) need shell
control flow. Their shell programs are fixed bmd-check constants; every variable
input is passed as a separate positional argument, never interpolated into
program text. No interface accepts caller-supplied command text, a
preconstructed command vector, or caller SSH options.

Python producer invocations use `-B` to avoid creating import bytecode caches in
trusted checkouts. This reduces incidental writes but does not sandbox producer
code or remove the need to trust it.

## 9. Path authorization and OS enforcement

`allowed_remote_roots` constrains observational reads using lexical POSIX
path normalization. It rejects traversal and paths outside configured roots,
but it is not a filesystem sandbox. In particular, it does not resolve
symlinks (`realpath`) and does not prove that a symlink beneath an authorized
root resolves beneath that root. If the observing identity can follow such a
symlink, the operating system may permit access outside the lexical root.

This limitation applies to both deployment modes, but its consequences depend
on the mode: in remote mode a symlink is followed with `bmdguest`'s
permissions; in local mode it is followed with the invoking user's
permissions. A local-mode user can therefore cause bmd-check to read, through a
symlink they can create beneath an allowed root, any file that user could
already read directly; bmd-check grants no access beyond the invoking user's own,
but allowed roots must not be relied on to confine what that user can read.
Choose allowed roots accordingly, and treat realpath containment as a separate
hardening item rather than a current guarantee.

Likewise, configuration values such as `access = "read_only"` and
`access = "observational"` are enforced bmd-check policy declarations. They do not
technically remove write permissions from the configured OS or SSH identity.

Safe deployment therefore combines:

1. bmd-check's fixed-purpose, action-free implementation;
2. correctly configured allowed roots; and
3. least-privileged OS/SSH credentials, filesystem permissions, and ACLs for
   the identity that actually performs the reads (`bmdguest` in remote mode,
   the invoking user in local mode).

Do not describe bmd-check as a filesystem security sandbox. Server-side
identity and permission controls remain part of the security boundary.

## 10. Least privilege

Where practical, security should be enforced by:

- Unix users and groups;
- filesystem permissions;
- ACLs;
- restricted SSH identities;
- Git/GitHub permissions;
- explicit tool allowlists;
- application authorization.

Prompt instructions are not a security boundary.

The persistent production agent should eventually run under a dedicated
VM service identity rather than inheriting the PI's personal
credentials and filesystem authority.

## 11. Credentials

The agent must not unnecessarily inherit:

- personal GitHub credentials;
- personal SSH credentials;
- cluster write credentials;
- deployment credentials.

Credentials should be scoped to the minimum capabilities required.

Read credentials and write credentials should be separate where
possible.

## 12. Future action plane

Action capabilities may be introduced later.

Examples include:

- preparing changes in an isolated worktree;
- committing;
- pushing;
- opening pull requests;
- submitting validation calculations;
- cancelling jobs;
- deploying services;
- publishing documentation.

Introducing such a capability requires:

1. an explicit tool implementation;
2. a defined authorization policy;
3. identification of the requesting user;
4. an explicit approval state;
5. PI authorization where required;
6. an audit record;
7. clear reporting of the resulting action.

The ability of the underlying operating-system account to perform an
operation does not imply that bmd-check is authorized to perform it.

## 13. Auditability

Material agent operations should eventually record:

- requester;
- task;
- timestamp;
- resource inspected;
- operation performed;
- inputs;
- artifacts produced;
- evidence used;
- failures;
- conclusions;
- recommendations;
- approvals;
- resulting Git commits or computational jobs where applicable.

The objective is to make it possible to reconstruct:

- what the agent inspected;
- what it concluded;
- why it reached that conclusion;
- what it changed, if anything;
- what tests it ran;
- what scientific evidence supported its claims.

## 14. Scientific safety

The agent must distinguish observation from inference.

It should preserve important methodological distinctions such as:

    ML prediction != first-principles result
    first-principles result != experiment
    calculation completed != calculation validated
    workflow implemented != workflow validated
    database absence != proven novelty
    negative formation energy != convex-hull stability

When evidence is incomplete, the agent should state the validation gap
rather than silently promote the claim.

## 15. v0 guarantee

The central security property of v0 is:

    no user-facing calculation, scheduler, or repository mutation actions exist

The first implementation should prove that useful scientific
observation, troubleshooting, and advice are possible before any action
capabilities are introduced.

bmd-check does not intentionally modify calculations, submit or cancel jobs,
restart calculations, alter scientific inputs, delete calculation files, or
read POTCAR contents. This statement describes bmd-check's implemented interfaces;
it is not a claim that the configured OS identity lacks write permissions or
that trusted producer module code is sandboxed.
