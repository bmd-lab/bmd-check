# VASP Wiki corpus maintainer tool

`fetch_corpus.py` is the only way pages enter the frozen VASP Wiki corpus in
`src/bmd_agent/reference_corpus/vasp_wiki/`. It is a maintainer tool: it is
not part of the installed package and BMD Check never imports or runs it.
It needs a workstation that can reach the official VASP Wiki over HTTPS.

The tool reads the committed corpus and writes **only to a new staging
directory** outside `src/`. Promoting staging into the committed corpus is a
manual copy reviewed in a normal pull request.

## Workflow

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

# 4. Version, append the release ledger and verify with BMD Check's loader.
$TOOL release --staging /tmp/vasp-corpus-staging --corpus-version 1.0.0

# 5. Copy the staging files over the committed corpus in a branch and open a PR.
```

`verify [DIR]` runs the loader on any corpus directory.

## What `fetch` does

- Uses the MediaWiki API only: `meta=siteinfo` (general + rightsinfo) and
  `prop=info|revisions` with `rvslots=main`. A pinned `Title@REVISION`
  first resolves the title, then fetches `revids=REVISION` and refuses a
  revision that belongs to another page.
- HTTPS only, official VASP hosts only (`vasp.at`, `www.vasp.at`), HTTP
  redirects refused, timeout and response-size caps.
- Refuses API errors, warnings, continuations, missing, redirect or
  non-main-namespace pages, hidden revisions, non-`wikitext` content,
  duplicate JSON keys and malformed fields.
- Encodes the returned text as strict UTF-8 and requires its size and
  MediaWiki SHA-1 to equal the upstream revision's, which proves the bytes are
  the stored revision. Content is never modified: text that violates the
  archival character policy aborts acquisition for human review.
- Captures site rights from the API and refuses if they do not name GNU Free
  Documentation License 1.2. If they differ from the committed corpus's, it
  refuses unless `--accept-site-rights-change` is given (the license review
  then starts over).
- Obtains the license text from `https://www.gnu.org/licenses/old-licenses/fdl-1.2.txt`
  (or reuses the committed copy) and records its SHA-256.
- Marks the license and every page `pending`. A pending corpus never loads.

## Before the first real corpus can load

1. Acquire the ten intended pages from the official site.
2. Verify the API endpoint and permalink forms, the exact site rights, the
   license version qualifier, per-page "unless otherwise noted" exceptions, and
   that no Invariant Sections or Cover Texts are declared.
3. Verify the fetched license text against the FSF original and pin its
   SHA-256 as `AUTHORITATIVE_POLICY.license_text_sha256` in
   `src/bmd_agent/vasp_wiki_corpus.py` in the same pull request. Until then
   the loader refuses every populated corpus.
