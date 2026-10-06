# VASP Wiki reference corpus

This directory is the frozen, offline VASP Wiki corpus that BMD Check ships.
Runtime BMD Check never fetches the VASP Wiki: a change to the live wiki cannot
change an installed release. The corpus changes only through an explicit
maintainer update reviewed in a normal pull request.

## Contents

`NOTICE` and `manifest.json` state whether this corpus currently contains any
VASP Wiki material. An unpopulated corpus (`corpus_kind: "unpopulated"`,
version `0.0.0`) contains none. Pages enter the corpus only by acquisition from
the official VASP Wiki with the maintainer tool. They are never reconstructed
from memory or copied from secondary sources.

The intended first populated corpus is these ten VASP Wiki titles:
`NELM`, `EDIFF`, `ALGO`, `EDIFFG`, `NSW`, `IBRION`, `ISIF`,
`Not enough memory`, `Difficult_to_converge_systems`, `Memory`.

## Third-party material

When populated, `pages/*.wiki` are verbatim copies of VASP Wiki wikitext at
pinned upstream revisions, and the license text file is a verbatim copy of that
license. They are third-party material: not authored by BMD and not covered by
the BMD Agent MIT License. `NOTICE` (generated from `manifest.json`) records
attribution, the captured site rights, the maintainer-reviewed license fields
and every page's revision identity.

The other files here (`README.md`, `manifest.json`, `NOTICE`, `RELEASES.json`,
`SHA256SUMS`) are BMD-authored metadata. BMD-authored explanatory notes about
VASP topics never live in this directory.

## Files

| File | Role |
| --- | --- |
| `manifest.json` | Strict schema-v1 record of the corpus, upstream site, license review and each page's identity. Canonical bytes only (sorted keys, two-space indent, ASCII). |
| `RELEASES.json` | Append-only ledger binding each corpus version to its identity digest; starts at the unpopulated `0.0.0` anchor and chains each release to the previous digest. |
| `NOTICE` | Attribution and license notice rendered deterministically from the manifest. |
| `SHA256SUMS` | SHA-256 of every other file here; the file set must match it exactly. |
| `pages/<slug>.r<revision>.wiki` | Verbatim UTF-8 wikitext of one pinned revision (populated corpus only). |
| license text file | Verbatim license text named by the manifest (populated corpus only). |

## Archival policy

Page files hold the exact UTF-8 bytes of the pinned revision: line endings are
not normalized, nothing is sanitized, converted, template-expanded, rendered or
link-rewritten, and no header is added. Attribution lives in sidecar files.

Preserved text may contain any Unicode scalar value except: C0 controls other
than TAB and LF (so CR, ESC and NUL are refused), DEL and C1 controls, U+061C,
U+200B-U+200F, U+2028, U+2029, U+202A-U+202E, U+2066-U+2069, U+FEFF,
noncharacters, and tag characters U+E0000-U+E007F. Content outside this policy
is never rewritten: acquisition and loading refuse it for human review.
Markup-like text (HTML tags, templates, links) is preserved as inert text;
BMD Check never interprets it.

## Identity

`corpus_digest` is a SHA-256 over the content-defining fields of the manifest:
page and revision identity, content hashes, URLs, captured site rights, the
reviewed license fields and review outcomes. Provenance that does not change
content (retrieval and assembly times, updater version, reviewer names and
dates, the MediaWiki generator string) is excluded. `RELEASES.json` binds each
version to exactly one digest, so changed content needs a new version and a new
ledger entry. Rewriting an earlier ledger entry shows up in review as a change
to recorded history, and the test suite compares the ledger against `origin/main`
when that ref is available.

## Loading

`bmd_agent.vasp_wiki_corpus.load_vasp_wiki_corpus()` verifies everything above
and fails closed. A populated corpus also needs an approved license review,
every page reviewed as free of "unless otherwise noted" exceptions, and a
license text whose SHA-256 is pinned in the loader's accepted policy. While the
corpus is unpopulated the loader reports it as unavailable. BMD Check does not
consume the corpus yet.

## Updating

Use the maintainer-only tool `tools/vasp_reference/fetch_corpus.py` (see its
README). It writes to a staging directory, never over this directory.
