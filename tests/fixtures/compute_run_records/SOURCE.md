# Vendored BMD Compute run-record snapshots

These files are verbatim copies of BMD Compute's canonical generated run-record
fixtures. BMD Compute owns them and their contracts
(`bmd_compute.submission` v1, `bmd_compute.job_record` v1). Agent keeps copies
only so its tests do not depend on a BMD Compute checkout.

## Identity

The snapshot is identified by its content. `v1/SHA256SUMS` lists the SHA-256
of every vendored file and is a byte-for-byte copy of BMD Compute's
`tests/fixtures/run_records/v1/SHA256SUMS`. Agent's tests fail if a vendored
file does not match it or if the listed and vendored files differ.

To check the snapshot against any BMD Compute revision, compare the two
`SHA256SUMS` files (or run `sha256sum -c SHA256SUMS` in each `v1/` directory).
Identical sums mean identical fixtures; no Git history is needed.

## Generation provenance

- Source repository and path: bmd_compute, `tests/fixtures/run_records/v1/`
- Generated on branch `claude/compute-run-record-contracts` by the code at
  commit `b0f9462`, which each record also names in
  `provenance.bmd_compute.source.git_commit`.

This is provenance, not identity. The generating commit may not be reachable
from BMD Compute's `main` (for example after a squash merge). That does not
make the snapshot stale, and it is not a reason to regenerate or re-vendor.

## Refreshing

Re-vendor only when BMD Compute's `SHA256SUMS` changes: copy the `v1/`
directory, including `SHA256SUMS`, and update the generation provenance above.
