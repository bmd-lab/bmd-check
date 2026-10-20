# VASP Wiki corpus maintainer tool

`fetch_corpus.py` is the only way pages enter the frozen VASP Wiki corpus in
`src/bmd_agent/reference_corpus/vasp_wiki/`. It is a maintainer tool: it is
not part of the installed package and bmd-check never imports or runs it.
It needs a workstation that can reach the official VASP Wiki over HTTPS.

The tool reads the committed corpus and writes **only to a new staging
directory** outside `src/`. Promoting staging into the committed corpus is a
manual copy reviewed in a normal pull request.

## Full acquisition (`fetch --all`)

`fetch --all` acquires the complete public VASP Wiki documentation instead of
a hand-picked list. It discovers pages systematically and records how every
discovered title was handled, so "complete" has a measurable meaning.

### Inclusion rules `vasp-wiki-main-namespace-1`

1. **Universe.** Every page that the public MediaWiki API lists in the main
   namespace (0) with `list=allpages&apfilterredir=nonredirects`, following
   every continuation until the API reports the list complete.
2. **Accounting.** Each discovered title has exactly one outcome, recorded in
   `manifest.json` under `acquisition`:
   - *preserved*: its latest revision at retrieval time is stored verbatim,
     pinned by revision ID, with its upstream categories;
   - *excluded*: deliberately not preserved for a closed reason:
     `archival_character_policy` (text outside the archival character policy,
     never rewritten), `license_exception` (a page-specific licensing exception
     found in review), `content_model` (not wikitext), `revision_unavailable`
     (hidden or suppressed revision) or `page_size_limit`;
   - *failed*: could not be acquired reliably (`missing_at_retrieval`,
     `integrity_mismatch`, `malformed_response`, `request_failed`).
3. **Completeness.** The acquisition is `complete` exactly when no title
   failed: preserved + excluded = discovered. Any failure labels it
   `incomplete`; `release` refuses an incomplete corpus unless
   `--allow-incomplete` is given, and the loader rejects a corpus whose label
   disagrees with its counts.
4. **Redirects** in the main namespace are recorded as aliases
   (`title -> target`) and never preserved as pages, so a renamed page is
   stored once under its canonical title.
5. **Other namespaces** (Talk, User, Project, File, MediaWiki, Template, Help,
   Category and any site-specific ones) are counted, not preserved: they hold
   discussion, user pages, media descriptions, rendering templates and
   generated category listings rather than documentation text. Media files
   (images) are never included.
6. **Identifiers** derive from the title (`vasp.wiki.<slug>`). Titles whose
   slugs collide (for example differing only in case) all receive their
   upstream page ID (`<slug>_p<pageid>`); a title with no ASCII letters or
   digits becomes `p<pageid>`. The rule depends only on the preserved page
   set, never on acquisition order.

### Etiquette and robustness

Requests go to the official API only, one at a time with `--delay` seconds
(default 1) between them; list queries use at most 500 entries and content
queries at most 50 titles, the limits for ordinary clients. Transport failures
are retried (`--retries`, default 3, with exponential back-off). A content
batch that the API cannot serve whole (continuation, size limit, warning) is
halved until single titles remain, so one problematic page never hides its
neighbours.

Redirects are resolved with `redirects=1`, for which MediaWiki follows chains
hop by hop: a batch may report more hops than titles (a double redirect gives
two), or omit the `redirects` key when nothing resolves. Each hop must be well
formed and reachable from a requested title; the first hop is recorded.

### Interrupted acquisitions

There is no resume mode. Nothing is staged until the whole acquisition has
finished, so an interrupted run (network failure, Ctrl+C, closed terminal)
never leaves a partial corpus: restart the same command from the beginning,
with a staging directory that does not exist yet or is empty. A complete
`fetch --all` makes a few hundred requests and takes a few minutes.

### Staging directory safety

Keep the staging directory somewhere only you can write to. The tool refuses
a staging path that is, or contains, a symbolic link, and re-checks right
before writing that the staging directory and its `pages/` directory are real
directories; every file is created exclusively, so an existing name or link is
never written through. It cannot defend against another account that can
rename entries in the staging directory's parent while the tool runs.

### Full workflow

```bash
export PYTHONPATH=src
TOOL="python -B tools/vasp_reference/fetch_corpus.py"
$TOOL fetch --all --api-url https://www.vasp.at/wiki/api.php --staging STAGING
$TOOL scan-licensing --staging STAGING --write-unflagged unflagged.txt
#   Pages printed as REVIEW contain licensing, copyright, permission,
#   quotation or file markers; check each against the live site.
$TOOL review --staging STAGING ... --page-reviewed-file reviewed.txt \
    [--page-exception ID "description"]
$TOOL release --staging STAGING --corpus-version X.Y.Z [--exclude-excepted]
$TOOL report STAGING --output report.md
$TOOL verify-upstream STAGING      # re-fetch every pinned revision, compare bytes
```

`release --exclude-excepted` moves license-excepted pages into the recorded
exclusions instead of shipping them. `verify-upstream` proves the corpus can be
reproduced from the pinned upstream revisions.

## Workflow (explicit titles)

Run from a checkout:

```bash
export PYTHONPATH=src
TOOL="python -B tools/vasp_reference/fetch_corpus.py"

# 1. Acquire. Pin revisions with Title@REVISION, or omit to take the latest.
$TOOL fetch --api-url https://<official VASP Wiki host>/<path>/api.php \
    --staging /tmp/vasp-corpus-staging --intended

# 2. Compare with the committed corpus; ALERT lines need attention.
$TOOL diff --staging /tmp/vasp-corpus-staging

# 3. Review against the live site, then record the review.
$TOOL review --staging /tmp/vasp-corpus-staging \
    --reviewer "<name>" --reviewed-on YYYY-MM-DD \
    --license-identifier GFDL-1.2-only|GFDL-1.2-or-later \
    --site-notice "<footer notice exactly as displayed>" \
    --invariant-sections none_declared --cover-texts none_declared \
    --approve-license \
    --page-reviewed vasp.wiki.nelm --page-reviewed vasp.wiki.ediff ...

#    A page with an "unless otherwise noted" exception is recorded with
#    --page-exception ID "description"; it cannot be released. Notes exist only
#    for exceptions: a no-exceptions or pending review never carries one.

# 4. Version, append the release ledger and verify with bmd-check's loader.
$TOOL release --staging /tmp/vasp-corpus-staging --corpus-version 1.0.0

# 5. Copy the staging files over the committed corpus in a branch and open a PR.
```

`verify [DIR]` runs the loader on any corpus directory.

## What `fetch` does

- Uses the MediaWiki API only: `meta=siteinfo` (general + rightsinfo, and
  namespaces for `--all`), `list=allpages` (`--all`), `prop=categories`, and
  `prop=info|revisions` with `rvslots=main`. A pinned `Title@REVISION`
  first resolves the title, then fetches `revids=REVISION` and refuses a
  revision that belongs to another page.
- HTTPS only, official VASP hosts only (`vasp.at`, `www.vasp.at`), HTTP
  redirects refused, timeout and response-size caps.
- Refuses API errors, warnings, missing, redirect or non-main-namespace
  pages, hidden revisions, non-`wikitext` content, duplicate JSON keys and
  malformed fields. Continuations are followed only for list and category
  queries; a content response that asks to continue is never interpreted
  (`--all` splits the batch instead). With explicit titles any refusal
  aborts the run; with `--all` it is recorded against that one page as an
  exclusion or failure (see the inclusion rules).
- Encodes the returned text as strict UTF-8 and requires its size and
  MediaWiki SHA-1 to equal the upstream revision's, which proves the bytes are
  the stored revision. Content is never modified: text that violates the
  archival character policy is refused for human review (explicit titles) or
  recorded as an `archival_character_policy` exclusion (`--all`).
- Captures site rights from the API and refuses if they do not name GNU Free
  Documentation License 1.2. If they differ from the committed corpus's, it
  refuses unless `--accept-site-rights-change` is given (the license review
  then starts over).
- Obtains the license text from `https://www.gnu.org/licenses/old-licenses/fdl-1.2.txt`
  (or reuses the committed copy) and records its SHA-256.
- Marks the license and every page `pending`. A pending corpus never loads.

## Before the first real corpus can load

1. Acquire the pages from the official site (`fetch --all` for the full corpus).
2. Verify the API endpoint and permalink forms, the exact site rights, the
   license version qualifier, per-page "unless otherwise noted" exceptions, and
   that no Invariant Sections or Cover Texts are declared.
3. Verify the fetched license text against the FSF original and pin its
   SHA-256 as `AUTHORITATIVE_POLICY.license_text_sha256` in
   `src/bmd_agent/vasp_wiki_corpus.py` in the same pull request. Until then
   the loader refuses every populated corpus.
