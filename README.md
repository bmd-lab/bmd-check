# bmd-check

bmd-check is a small, deterministic observation and diagnostic layer for the
Burton Materials Discovery Lab at Tel Aviv University. It helps researchers
understand VASP calculations without replacing the software and knowledge
sources that produced them.

bmd-check observes and diagnoses calculations. It can inspect:

- VASP inputs and results;
- SLURM state, accounting, resource use, and out-of-memory evidence;
- Custodian interventions;
- bmd-compute workflow and Git provenance;
- compact convergence trajectories; and
- applicable contextual knowledge supplied by bmd-store.

The default output is a concise student-facing diagnosis. Detailed evidence
remains available for advanced users and reproducibility.

## Install

bmd-check requires Python 3.12 or newer. From a fresh clone:

```bash
git clone https://github.com/bmd-lab/bmd-check.git
cd bmd-check
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

For development, install the declared test dependency and run the suite:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

No bmd-compute checkout, bmd-store checkout, SSH connection, PowerSLURM access, or
VASP installation is required to run the unit tests.

## Configure resources

The tracked [example configuration](config/resources.example.toml) documents
the supported resource fields. Copy it to a deployment-local location and edit
the copy. The example is never used automatically for real execution.

The normal Linux location is:

```text
~/.config/bmd-agent/resources.toml
```

On Windows, the normal location is:

```text
%APPDATA%\bmd-agent\resources.toml
```

Set `BMD_AGENT_RESOURCES` to use a different file:

```bash
export BMD_AGENT_RESOURCES=/secure/local/path/resources.toml
```

Deployment configuration identifies resources; it does not define scientific
workflow policy. The main fields are:

- `repositories`: trusted bmd-compute and bmd-store checkout paths;
- `capability_python`: the Python executable for each checkout's own
  environment;
- `ssh_host`: a configured SSH alias for the observational cluster identity;
- `allowed_remote_roots`: absolute POSIX roots authorized for remote reads;
- `deployment_profile`: an optional shipped infrastructure reference profile;
- timeout values for SSH, fixed remote commands, and scheduler accounting; and
- `access`, `protected`, and `live` declarations describing bmd-check policy.

Keep `resources.toml` local. Do not commit credentials, private keys, or
deployment-local configuration. SSH credentials belong in the user's SSH
configuration or agent, not in this repository or `resources.toml`.

## Use

The normal interface is deliberately simple:

```bash
bmd-check
bmd-check JOB_ID
bmd-check PATH
```

- No target analyzes the current working directory.
- A bare positive decimal integer inspects that SLURM job and resolves its BMD
  bmd-compute run when producer evidence permits.
- Any other existing filesystem path analyzes that calculation or workflow.

`bmd-check` is the canonical command. The `bmd-agent` command remains a fully
supported compatibility alias for the same program. The Python distribution
name remains `bmd-agent`, and the import package remains `bmd_agent`.

Examples:

```bash
cd /path/to/calculation
bmd-check

bmd-check 21853598
bmd-check ./copied-calculation
```

The reported status is an evidence claim about the whole calculation:

- `COMPLETED`: every declared workflow stage has positive convergence evidence
  (electronic, plus ionic for relaxations) and VASP normal termination or a
  successful scheduler exit. SLURM `COMPLETED 0:0` alone is never enough.
- `INCOMPLETE`: execution ended, but a required stage is missing, did not
  terminate normally, or did not meet its convergence criteria.
- `FAILED`: the scheduler reports an unsuccessful state or a nonzero exit.
- `UNKNOWN`: the available evidence cannot establish completion. Missing
  evidence, or the absence of reported errors, is not treated as convergence.
- `PENDING`, `RUNNING`, and `PRE_RUN` describe queue and execution state.

The same workflow-level status is reported whether bmd-check is run from the
workflow root or from any stage directory.

Use `--verbose` for detailed evidence, provenance, parser limitations, and
scheduler accounting. On job inspection, use `--profile` for
developer-oriented acquisition and performance telemetry. Additional
expert/debug subcommands remain available for compatibility and testing, but
are not required for normal use.

bmd-check remains independently callable on a cluster with these same commands.
Its Python interfaces may also be called by bmd-compute in the future, but this
repository does not implement that integration.

## bmd-compute run records

For bmd-compute runs, bmd-check reads two bmd-compute-owned records: the Prepare-time
submission specification `submission.json` (`bmd_compute.submission`) and the
run-resolution record `job_<JOB_ID>.json` (`bmd_compute.job_record`).
bmd-compute defines both contracts. bmd-check reads version 1 strictly, reads older
records without a `schema` as legacy unversioned records and says so, reports
an unsupported future version as unsupported rather than guessing, and reports
malformed records as invalid. Records written before 2026-08-11 lack
`flow_spec.workflow_spec` and are reported as unsupported legacy records;
stages are not inferred from older fields. Status fields in these records are
never used for execution state: SLURM accounting and VASP artifacts decide
that.

## Observation boundary

bmd-check does not intentionally modify calculations, submit or cancel jobs,
restart calculations, alter scientific inputs, delete calculation files, or
extract Custodian error archives. It does not read POTCAR contents.

In particular, bmd-check does not provide commands to edit `INCAR`, `KPOINTS`, or
`POSCAR`, and it does not expose arbitrary user-controlled remote shell
execution. Remote scheduler and file operations are fixed-purpose observation
interfaces.

Safe deployment relies on all three of the following:

1. bmd-check's read-only, action-free implementation;
2. correctly configured `allowed_remote_roots`; and
3. appropriately restricted OS/SSH credentials and filesystem permissions.

SSH observation runs with `BatchMode=yes`, so missing keys or unknown hosts
fail immediately instead of prompting. When bmd-check runs on the POSIX cluster it
observes, `ssh_host = "local"` selects the local transport: the same typed
observational operations, path authorization and size limits apply, but they
run as argument vectors under the invoking user's identity without SSH. That
user's filesystem permissions are then the operating-system boundary,
including for symlinks beneath allowed roots. Local mode is not available on
Windows.

Remote path authorization is lexical. It rejects paths outside configured
roots after POSIX normalization, but it is not a filesystem sandbox and does
not resolve every remote symlink before access. A symlink beneath an allowed
root may point outside that root if the SSH identity can follow it. Likewise,
`access = "read_only"` is a bmd-check policy declaration; it does not remove write
permission from the operating-system account. Deploy bmd-check with a
least-privileged observational identity and suitable server-side permissions.

bmd-check invokes only fixed producer modules from explicitly configured
bmd-compute and bmd-store checkouts, using each checkout's configured Python
executable.
Those checkouts are trusted code dependencies: their module code executes with
the bmd-check caller's OS privileges. Do not configure arbitrary third-party
checkouts as producers.

See [SECURITY.md](SECURITY.md) for the complete security model.

## Deployment profiles

[`src/bmd_agent/deployment_profiles/power.toml`](src/bmd_agent/deployment_profiles/power.toml)
is a versioned record of expected and observed POWER infrastructure facts. It
supports reproducibility and compatibility checks; its paths do not grant
access.

Deployment-local `resources.toml` remains authoritative for the SSH alias,
allowed roots, operational timeouts, and access policy. The current VM-to-POWER
SSH route is a supported deployment, not a promise about the final production
architecture.

## bmd-compute v1 records

bmd-check reads bmd-compute's versioned records; it does not recompute them.

- `runtime_environment.json` (`bmd_compute.runtime_environment` v1), when a
  run's `submission.json` declares one: detailed output reports bmd-compute's own
  verdict (`Runtime parity: PASSED` or `FAILED` with bmd-compute's reasons), the
  prepared and runtime versions of the parity-critical packages, supporting
  packages and the effective atomate2 settings. Runs prepared before this
  record existed show `Runtime parity: not recorded` and keep the runner-log
  evidence. A declared record that cannot be read is reported as unavailable,
  never as a pass.
- `provenance.execution.automatic_treatments`: for BMD-managed Desired Output
  workflows, which automatic treatments bmd-compute applied (including the frozen
  automatic DFT+U record). bmd-check never infers automatic treatments from stage
  modifiers; bmd-compute does not separately record Custom workflows.
- The capability payload's `stage_modifier_support` is shown by
  `bmd-check compute` exactly as bmd-compute declares it.

Unsupported major versions of these records (and of the submission
`provenance` block) are reported as unsupported rather than read under v1
assumptions.

## Scientific and ecosystem boundaries

bmd-check preserves the responsibilities of independently version-controlled
BMD projects:

- **bmd-compute** owns what its VASP workflows execute.
- **bmd-store** owns curated supporting scientific data and contextual evidence.
- **bmd-help** owns human-oriented tutorials and explanations.
- **bmd-check** connects evidence so researchers can understand and diagnose a
  calculation.

Implementation is not scientific validation, a completed job is not methodology
adoption, and bmd-check observations do not automatically become BMD standards.
Human scientific review and governance remain separate.

The architecture is described in [ARCHITECTURE.md](ARCHITECTURE.md). Graduate
students can contribute using the lightweight process in
[CONTRIBUTING.md](CONTRIBUTING.md).

## VASP Wiki reference corpus

The package reserves an offline, versioned corpus of preserved VASP Wiki pages
in `src/bmd_agent/reference_corpus/vasp_wiki/`. bmd-check never fetches the
VASP Wiki at runtime, so a change to the live wiki cannot change an installed
release. Corpus updates are explicit maintainer operations with
[`tools/vasp_reference/fetch_corpus.py`](tools/vasp_reference/README.md),
reviewed through normal pull requests.

The corpus is currently unpopulated, and bmd-check does not use it in its
output yet. Preserved pages are third-party material; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

bmd-check's repository-owned source and documentation are released under the
[MIT License](LICENSE). This license does not grant rights to VASP, POTCAR/PAW
datasets, preserved third-party reference material, or third-party
dependencies; those remain subject to their own licenses and access terms (see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
