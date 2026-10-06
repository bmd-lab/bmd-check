"""Maintainer-only acquisition tool for the frozen VASP Wiki reference corpus.

This script is never installed, imported or executed by BMD Check. It runs on a
maintainer workstation that can reach the official VASP Wiki, and it only ever
writes to a staging directory. Promoting a reviewed staging corpus into
``src/bmd_agent/reference_corpus/vasp_wiki`` is a manual copy that goes through
a normal pull request.

Workflow (see README.md beside this file):

    fetch    acquire pinned revisions + site rights into a new staging directory
    diff     compare a staging corpus with the committed corpus
    review   record the maintainer's license and per-page review in staging
    release  assign a corpus version, append the release ledger, verify staging
    verify   run BMD Check's loader on a corpus directory

Downloaded content is untrusted bytes. The tool never rewrites it: content that
violates the archival policy aborts acquisition for human review.

Run from a checkout with ``PYTHONPATH=src python -B tools/vasp_reference/fetch_corpus.py``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
import urllib.error
from urllib.parse import quote, urlencode, urlsplit
import urllib.request

from bmd_agent import vasp_wiki_corpus as corpus


TOOL_NAME = "tools/vasp_reference/fetch_corpus.py"
TOOL_VERSION = "1"

INTENDED_TITLES = (
    "NELM",
    "EDIFF",
    "ALGO",
    "EDIFFG",
    "NSW",
    "IBRION",
    "ISIF",
    "Not enough memory",
    "Difficult_to_converge_systems",
    "Memory",
)

# Where the verbatim license text is obtained. The digest of what is fetched is
# printed for review; the loader accepts it only once that digest is pinned in
# bmd_agent.vasp_wiki_corpus.AUTHORITATIVE_POLICY in a reviewed change.
LICENSE_TEXT_URL = "https://www.gnu.org/licenses/old-licenses/fdl-1.2.txt"
LICENSE_TEXT_HOSTS = frozenset({"www.gnu.org"})

MAX_API_RESPONSE_BYTES = 8_388_608
MAX_LICENSE_TEXT_BYTES = corpus.MAX_AUXILIARY_FILE_BYTES
DEFAULT_TIMEOUT_SECONDS = 30
USER_AGENT = f"bmd-agent-vasp-corpus-maintainer/{TOOL_VERSION} (explicit maintainer acquisition)"

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_CORPUS = REPO_ROOT / "src" / "bmd_agent" / "reference_corpus" / "vasp_wiki"

_RVPROP = "content|contentmodel|ids|sha1|size|timestamp"
_TIMESTAMP_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_PAGE_ARGUMENT_RE = re.compile(r"^(?P<title>.+?)(?:@(?P<revision>[1-9][0-9]{0,11}))?$", re.DOTALL)

# (url, expected content kind "json" or "text") -> response body bytes
Fetcher = Callable[[str, str], bytes]


class FetchError(RuntimeError):
    """Acquisition or staging was refused; nothing authoritative was written."""


# ---------------------------------------------------------------------------
# HTTPS transport (real network; replaced by a fake fetcher in tests)
# ---------------------------------------------------------------------------


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802 - urllib API
        raise FetchError(f"HTTP redirect refused ({code}); request the final URL explicitly")


def validate_request_url(url: str, allowed_hosts: frozenset[str]) -> str:
    """Require an HTTPS URL on an allowed host with no credentials or port."""

    if type(url) is not str or not url.isascii() or any(ch.isspace() for ch in url):
        raise FetchError("request URL must be ASCII without whitespace")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise FetchError("request URL is malformed") from exc
    if parts.scheme != "https":
        raise FetchError("request URL must use HTTPS")
    if parts.username is not None or parts.password is not None or port is not None:
        raise FetchError("request URL must not carry credentials or a port")
    host = parts.hostname or ""
    if parts.netloc != host or host not in allowed_hosts:
        raise FetchError("request URL host is not allowed")
    return url


def https_fetch(url: str, expected: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bytes:
    """Fetch one URL over HTTPS with redirects refused and a response size cap."""

    allowed = LICENSE_TEXT_HOSTS if expected == "text" else corpus.AUTHORITATIVE_POLICY.allowed_hosts
    validate_request_url(url, allowed)
    limit = MAX_LICENSE_TEXT_BYTES if expected == "text" else MAX_API_RESPONSE_BYTES
    opener = urllib.request.build_opener(_RefuseRedirects())
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise FetchError(f"unexpected HTTP status {response.status}")
            if response.geturl() != url:
                raise FetchError("response URL differs from the request URL")
            content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            wanted = "application/json" if expected == "json" else "text/plain"
            if content_type != wanted:
                raise FetchError(f"unexpected content type {content_type!r}")
            data = response.read(limit + 1)
    except FetchError:
        raise
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise FetchError(f"request failed: {type(exc).__name__}") from exc
    if len(data) > limit:
        raise FetchError("response exceeds the size limit")
    return data


def _guarded_fetch(fetcher: Fetcher, url: str, expected: str) -> bytes:
    allowed = LICENSE_TEXT_HOSTS if expected == "text" else corpus.AUTHORITATIVE_POLICY.allowed_hosts
    validate_request_url(url, allowed)
    data = fetcher(url, expected)
    if not isinstance(data, bytes):
        raise FetchError("fetcher returned a non-bytes response")
    limit = MAX_LICENSE_TEXT_BYTES if expected == "text" else MAX_API_RESPONSE_BYTES
    if len(data) > limit:
        raise FetchError("response exceeds the size limit")
    return data


# ---------------------------------------------------------------------------
# MediaWiki API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageRequest:
    title: str
    revision_id: int | None = None


def parse_page_argument(text: str) -> PageRequest:
    """``Title`` or ``Title@REVISION`` (pin an exact upstream revision)."""

    match = _PAGE_ARGUMENT_RE.fullmatch(text)
    if match is None:
        raise FetchError("page argument is malformed")
    title = match.group("title")
    if title != title.strip() or not title or len(title) > 255:
        raise FetchError("page title is malformed")
    if corpus.find_disallowed_character(title) is not None or "\n" in title or "\t" in title:
        raise FetchError("page title contains disallowed characters")
    revision = match.group("revision")
    return PageRequest(title, int(revision) if revision else None)


def api_request_url(endpoint: str, params: Mapping[str, str]) -> str:
    return endpoint + "?" + urlencode(sorted(params.items()))


def _api_json(data: bytes) -> Mapping[str, Any]:
    try:
        text = data.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise FetchError("API response is not valid UTF-8") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except FetchError:
        raise
    except (ValueError, RecursionError) as exc:
        raise FetchError("API response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise FetchError("API response is not a JSON object")
    if "error" in payload:
        code = payload["error"].get("code") if isinstance(payload["error"], dict) else None
        safe = isinstance(code, str) and re.fullmatch(r"[a-z0-9_-]{1,64}", code) is not None
        raise FetchError(f"API returned an error: {code if safe else 'unrecognized error code'}")
    if "warnings" in payload:
        raise FetchError("API returned warnings; refusing to interpret a partial response")
    if "continue" in payload:
        raise FetchError("API response is incomplete (continuation requested)")
    query = payload.get("query")
    if not isinstance(query, dict):
        raise FetchError("API response has no query object")
    return query


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FetchError("API response has a duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise FetchError("API response contains a non-finite number")


def fetch_site(endpoint: str, fetcher: Fetcher) -> dict[str, str]:
    """Capture site identity and rights exactly as the MediaWiki API reports them."""

    query = _api_json(
        _guarded_fetch(
            fetcher,
            api_request_url(
                endpoint,
                {
                    "action": "query",
                    "format": "json",
                    "formatversion": "2",
                    "meta": "siteinfo",
                    "siprop": "general|rightsinfo",
                },
            ),
            "json",
        )
    )
    general = query.get("general")
    rights = query.get("rightsinfo")
    if not isinstance(general, dict) or not isinstance(rights, dict):
        raise FetchError("siteinfo response lacks general or rightsinfo")
    values: dict[str, str] = {}
    for key, source in (
        ("site_name", "sitename"),
        ("generator", "generator"),
        ("server", "server"),
        ("script_path", "script"),
        ("article_path", "articlepath"),
    ):
        value = general.get(source)
        if type(value) is not str or not value:
            raise FetchError(f"siteinfo general.{source} is missing")
        values[key] = value
    if values["server"].startswith("//"):
        values["server"] = "https:" + values["server"]
    rights_url = rights.get("url")
    rights_text = rights.get("text")
    if type(rights_url) is not str or type(rights_text) is not str or not rights_text:
        raise FetchError("siteinfo rightsinfo is missing; the site license cannot be captured")
    values["rights_url"] = rights_url
    values["rights_text"] = rights_text
    values["api_endpoint"] = endpoint
    return values


def fetch_page(
    endpoint: str,
    request: PageRequest,
    site: Mapping[str, str],
    fetcher: Fetcher,
    *,
    retrieved_at: str,
) -> tuple[dict[str, Any], bytes]:
    """Acquire one revision and return its manifest record and verbatim bytes."""

    slug = corpus.title_slug(request.title)
    if not slug:
        raise FetchError("page title does not yield a usable corpus identifier")

    if request.revision_id is None:
        query = _api_json(
            _guarded_fetch(
                fetcher,
                api_request_url(
                    endpoint,
                    {
                        "action": "query",
                        "format": "json",
                        "formatversion": "2",
                        "prop": "info|revisions",
                        "inprop": "url",
                        "redirects": "1",
                        "titles": request.title,
                        "rvprop": _RVPROP,
                        "rvslots": "main",
                    },
                ),
                "json",
            )
        )
        resolved_title, redirected_from = _resolved_title(request.title, query)
        page = _single_page(query)
        if page["title"] != resolved_title:
            raise FetchError("API returned a different page than the one requested")
    else:
        resolve = _api_json(
            _guarded_fetch(
                fetcher,
                api_request_url(
                    endpoint,
                    {
                        "action": "query",
                        "format": "json",
                        "formatversion": "2",
                        "prop": "info",
                        "inprop": "url",
                        "redirects": "1",
                        "titles": request.title,
                    },
                ),
                "json",
            )
        )
        resolved_title, redirected_from = _resolved_title(request.title, resolve)
        resolved_page = _single_page(resolve, expect_revisions=False)
        if resolved_page["title"] != resolved_title:
            raise FetchError("API resolved the title to a different page")
        query = _api_json(
            _guarded_fetch(
                fetcher,
                api_request_url(
                    endpoint,
                    {
                        "action": "query",
                        "format": "json",
                        "formatversion": "2",
                        "prop": "info|revisions",
                        "inprop": "url",
                        "revids": str(request.revision_id),
                        "rvprop": _RVPROP,
                        "rvslots": "main",
                    },
                ),
                "json",
            )
        )
        page = _single_page(query)
        if page["pageid"] != resolved_page["pageid"] or page["title"] != resolved_title:
            raise FetchError("pinned revision belongs to a different page")

    revision = page["revisions"][0]
    if request.revision_id is not None and revision["revid"] != request.revision_id:
        raise FetchError("API returned a different revision than the pinned one")
    content = revision["slots"]["main"]["content"]
    try:
        data = content.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise FetchError("page content is not representable as UTF-8") from exc
    if len(data) > corpus.MAX_PAGE_BYTES:
        raise FetchError("page content exceeds the page size limit")
    if len(data) != revision["size"]:
        raise FetchError("page content size does not match the upstream revision size")
    if hashlib.sha1(data, usedforsecurity=False).hexdigest() != revision["sha1"]:
        raise FetchError("page content does not match the upstream MediaWiki SHA-1")
    if corpus.find_disallowed_character(content) is not None:
        raise FetchError("page content violates the archival character policy; human review required")

    script_url = site["server"] + site["script_path"]
    title_parameter = quote(page["title"].replace(" ", "_"), safe="")
    revision_id = revision["revid"]
    record = {
        "id": corpus.page_id_for_slug(slug),
        "requested_title": request.title,
        "title": page["title"],
        "redirected_from": redirected_from,
        "page_id": page["pageid"],
        "namespace": page["ns"],
        "revision_id": revision_id,
        "parent_revision_id": revision["parentid"] or None,
        "revision_timestamp": revision["timestamp"],
        "retrieved_at": retrieved_at,
        "canonical_url": page["canonicalurl"],
        "permalink": f"{script_url}?title={title_parameter}&oldid={revision_id}",
        "history_url": f"{script_url}?title={title_parameter}&action=history",
        "content_model": "wikitext",
        "content_file": corpus.content_file_for(slug, revision_id),
        "content_bytes": len(data),
        "content_sha256": corpus.sha256_hex(data),
        "upstream_sha1": revision["sha1"],
        "license_review": {
            "status": corpus.PAGE_REVIEW_PENDING,
            "reviewer": None,
            "reviewed_on": None,
            "notes": None,
        },
    }
    return record, data


def _resolved_title(requested: str, query: Mapping[str, Any]) -> tuple[str, str | None]:
    title = requested
    for item in _from_to_list(query, "normalized"):
        if item["from"] == title:
            title = item["to"]
    redirected_from: str | None = None
    for item in _from_to_list(query, "redirects"):
        if item["from"] == title:
            if redirected_from is None:
                redirected_from = title
            title = item["to"]
    return title, redirected_from


def _from_to_list(query: Mapping[str, Any], key: str) -> list[Mapping[str, str]]:
    items = query.get(key, [])
    if not isinstance(items, list) or len(items) > 8:
        raise FetchError(f"API {key} list is malformed")
    for item in items:
        if not isinstance(item, dict) or type(item.get("from")) is not str or type(item.get("to")) is not str:
            raise FetchError(f"API {key} entry is malformed")
    return items


def _single_page(query: Mapping[str, Any], *, expect_revisions: bool = True) -> Mapping[str, Any]:
    pages = query.get("pages")
    if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
        raise FetchError("API response must describe exactly one page")
    page = pages[0]
    for flag in ("missing", "invalid", "special"):
        if page.get(flag):
            raise FetchError(f"requested page is {flag}")
    if page.get("redirect"):
        raise FetchError("API returned a redirect page instead of its target")
    if type(page.get("pageid")) is not int or page["pageid"] <= 0:
        raise FetchError("API page ID is malformed")
    if page.get("ns") != 0 or type(page.get("ns")) is not int:
        raise FetchError("only main-namespace pages may be preserved")
    title = page.get("title")
    if type(title) is not str or not title or corpus.find_disallowed_character(title) is not None:
        raise FetchError("API page title is malformed")
    if page.get("contentmodel") != "wikitext":
        raise FetchError("page content model is not wikitext")
    if type(page.get("canonicalurl")) is not str:
        raise FetchError("API page lacks a canonical URL")
    if not expect_revisions:
        return page
    revisions = page.get("revisions")
    if not isinstance(revisions, list) or len(revisions) != 1 or not isinstance(revisions[0], dict):
        raise FetchError("API response must contain exactly one revision")
    revision = revisions[0]
    for flag in ("texthidden", "sha1hidden", "suppressed"):
        if revision.get(flag):
            raise FetchError(f"revision is {flag}")
    if type(revision.get("revid")) is not int or revision["revid"] <= 0:
        raise FetchError("revision ID is malformed")
    if type(revision.get("parentid")) is not int or revision["parentid"] < 0:
        raise FetchError("parent revision ID is malformed")
    if revision["parentid"] >= revision["revid"]:
        raise FetchError("parent revision does not precede the revision")
    if type(revision.get("timestamp")) is not str or not _TIMESTAMP_RE.fullmatch(revision["timestamp"]):
        raise FetchError("revision timestamp is malformed")
    if type(revision.get("sha1")) is not str or not _SHA1_RE.fullmatch(revision["sha1"]):
        raise FetchError("revision SHA-1 is malformed")
    if type(revision.get("size")) is not int or revision["size"] < 0:
        raise FetchError("revision size is malformed")
    slots = revision.get("slots")
    main = slots.get("main") if isinstance(slots, dict) else None
    if not isinstance(main, dict):
        raise FetchError("revision has no main slot")
    if main.get("missing") or main.get("badcontentformat"):
        raise FetchError("revision main slot content is unavailable")
    if main.get("contentmodel") != "wikitext" or main.get("contentformat") != "text/x-wiki":
        raise FetchError("revision content is not wikitext")
    if type(main.get("content")) is not str:
        raise FetchError("revision content is missing")
    return page


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


def read_corpus_directory(directory: Path) -> dict[str, Any]:
    """Read a corpus directory's manifest and ledger without the full loader."""

    manifest_bytes = (directory / corpus.MANIFEST_FILE).read_bytes()
    manifest = corpus.parse_strict_json(manifest_bytes, label=corpus.MANIFEST_FILE)
    if corpus.canonical_manifest_bytes(manifest) != manifest_bytes:
        raise FetchError(f"{directory / corpus.MANIFEST_FILE} is not canonical")
    _validate_staging(manifest, require_reviewed=False)
    releases_bytes = (directory / corpus.RELEASES_FILE).read_bytes()
    releases = corpus.parse_strict_json(releases_bytes, label=corpus.RELEASES_FILE)
    entries = corpus.validate_releases(releases)
    return {"manifest": manifest, "releases": releases, "entries": entries}


def _require_safe_staging(staging: Path, committed: Path, *, must_be_new: bool) -> Path:
    staging = staging.resolve()
    protected = (committed.resolve(), (REPO_ROOT / "src").resolve())
    for root in protected:
        if staging == root or root in staging.parents:
            raise FetchError("staging directory must be outside the committed corpus and src/")
    if committed.resolve() in staging.parents or staging in committed.resolve().parents:
        raise FetchError("staging directory must not contain or be inside the committed corpus")
    if must_be_new and staging.exists() and any(staging.iterdir()):
        raise FetchError("staging directory already exists and is not empty")
    if not must_be_new and not (staging / corpus.MANIFEST_FILE).is_file():
        raise FetchError("staging directory has no manifest")
    if staging.exists() and any(path.is_symlink() for path in staging.rglob("*")):
        raise FetchError("staging directory must not contain symbolic links")
    return staging


def _write_corpus(directory: Path, files: Mapping[str, bytes], *, exclusive: bool) -> None:
    """Write every corpus file plus SHA256SUMS; remove stale files in staging."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / corpus.PAGES_DIRECTORY).mkdir(exist_ok=True)
    if not exclusive:
        wanted = set(files) | {corpus.SHA256SUMS_FILE}
        for path in list(directory.rglob("*")):
            relative = path.relative_to(directory).as_posix()
            if path.is_file() and relative not in wanted:
                path.unlink()
    outputs = dict(files)
    outputs[corpus.SHA256SUMS_FILE] = corpus.render_sha256sums(files)
    for relative, data in sorted(outputs.items()):
        target = directory / relative
        if not exclusive and (target.exists() or target.is_symlink()):
            target.unlink()
        # Exclusive creation never follows or overwrites an existing path.
        with open(target, "xb") as handle:
            handle.write(data)


def _corpus_files(directory: Path, manifest: Mapping[str, Any], releases: Mapping[str, Any]) -> dict[str, bytes]:
    files = {
        corpus.MANIFEST_FILE: corpus.canonical_manifest_bytes(manifest),
        corpus.RELEASES_FILE: corpus.canonical_releases_bytes(releases),
        corpus.NOTICE_FILE: corpus.render_notice(manifest),
        corpus.README_FILE: (directory / corpus.README_FILE).read_bytes(),
    }
    license_block = manifest.get("license")
    if license_block is not None:
        files[license_block["license_text_file"]] = (directory / license_block["license_text_file"]).read_bytes()
    for page in manifest["pages"]:
        files[page["content_file"]] = (directory / page["content_file"]).read_bytes()
    return files


def _proposed_version(entries: Sequence[Mapping[str, Any]]) -> str:
    major, minor, _ = (int(part) for part in entries[-1]["corpus_version"].split("."))
    return f"{major}.{minor + 1}.0"


def _validate_staging(manifest: Mapping[str, Any], *, require_reviewed: bool) -> None:
    try:
        corpus.validate_manifest(manifest, corpus.AUTHORITATIVE_POLICY, require_reviewed=require_reviewed)
    except corpus.CorpusError as exc:
        raise FetchError(f"staged manifest is invalid: {exc.detail}") from exc


def command_fetch(args: argparse.Namespace, fetcher: Fetcher) -> int:
    committed = Path(args.committed)
    staging = _require_safe_staging(Path(args.staging), committed, must_be_new=True)
    endpoint = validate_request_url(args.api_url, corpus.AUTHORITATIVE_POLICY.allowed_hosts)
    current = read_corpus_directory(committed)
    committed_manifest = current["manifest"]

    requests = [parse_page_argument(item) for item in (args.page or [])]
    if args.intended:
        requests.extend(PageRequest(title) for title in INTENDED_TITLES)
    if not requests:
        raise FetchError("no pages requested; use --page or --intended")
    slugs = [corpus.title_slug(item.title) for item in requests]
    if len(set(slugs)) != len(slugs):
        raise FetchError("requested titles collide on the same corpus identifier")

    retrieved_at = args.retrieved_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not _TIMESTAMP_RE.fullmatch(retrieved_at):
        raise FetchError("--retrieved-at must be a UTC timestamp")

    site = fetch_site(endpoint, fetcher)
    previous_upstream = committed_manifest.get("upstream")
    if previous_upstream is not None and not args.accept_site_rights_change:
        if (previous_upstream["rights_url"], previous_upstream["rights_text"]) != (
            site["rights_url"],
            site["rights_text"],
        ):
            raise FetchError(
                "site rights differ from the committed corpus; review the license change and "
                "rerun with --accept-site-rights-change"
            )
    if corpus.AUTHORITATIVE_POLICY.required_rights_text_fragment not in site["rights_text"]:
        raise FetchError("site rights text does not name the accepted license; refusing to acquire")

    pages: list[tuple[dict[str, Any], bytes]] = [
        fetch_page(endpoint, request, site, fetcher, retrieved_at=retrieved_at) for request in requests
    ]
    page_ids = [record["page_id"] for record, _ in pages]
    if len(set(page_ids)) != len(page_ids):
        raise FetchError("two requested titles resolve to the same upstream page")

    license_text, license_source = _license_text(args, committed, committed_manifest, fetcher)
    license_sha = corpus.sha256_hex(license_text)
    pinned = corpus.AUTHORITATIVE_POLICY.license_text_sha256
    if pinned is not None and license_sha != pinned:
        raise FetchError("license text differs from the pinned accepted license text")

    manifest = {
        "schema": corpus.MANIFEST_SCHEMA,
        "schema_version": corpus.SCHEMA_VERSION,
        "corpus_kind": corpus.KIND_AUTHORITATIVE,
        "corpus_version": _proposed_version(current["entries"]),
        "corpus_digest": "",
        "assembly": {
            "assembled_at": retrieved_at,
            "updater": {"name": TOOL_NAME, "version": TOOL_VERSION},
        },
        "upstream": dict(site),
        "license": {
            "identifier": None,
            "site_notice_as_displayed": None,
            "invariant_sections": None,
            "cover_texts": None,
            "license_text_file": corpus.AUTHORITATIVE_POLICY.license_text_file,
            "license_text_source_url": license_source,
            "license_text_sha256": license_sha,
            "review": {"status": corpus.LICENSE_REVIEW_PENDING, "reviewer": None, "reviewed_on": None},
        },
        "pages": sorted((record for record, _ in pages), key=lambda record: record["id"]),
    }
    manifest["corpus_digest"] = corpus.compute_corpus_digest(manifest)
    _validate_staging(manifest, require_reviewed=False)

    files = {
        corpus.MANIFEST_FILE: corpus.canonical_manifest_bytes(manifest),
        corpus.RELEASES_FILE: corpus.canonical_releases_bytes(current["releases"]),
        corpus.NOTICE_FILE: corpus.render_notice(manifest),
        corpus.README_FILE: (committed / corpus.README_FILE).read_bytes(),
        corpus.AUTHORITATIVE_POLICY.license_text_file: license_text,
    }
    for record, data in pages:
        files[record["content_file"]] = data
    _write_corpus(staging, files, exclusive=True)

    print(f"Staged {len(pages)} page(s) in {staging}")
    print(f"Captured site rights: {site['rights_text']} <{site['rights_url'] or 'no URL'}>")
    print(f"License text SHA-256: {license_sha} (source {license_source})")
    if pinned is None:
        print("NOTE: the accepted license text digest is not pinned in the loader policy yet.")
    print("Every page and the license are pending review. Next: diff, review, release.")
    return 0


def _license_text(
    args: argparse.Namespace,
    committed: Path,
    committed_manifest: Mapping[str, Any],
    fetcher: Fetcher,
) -> tuple[bytes, str]:
    license_block = committed_manifest.get("license")
    if license_block is not None and not args.refetch_license_text:
        data = (committed / license_block["license_text_file"]).read_bytes()
        if corpus.sha256_hex(data) != license_block["license_text_sha256"]:
            raise FetchError("committed license text does not match its manifest")
        return data, license_block["license_text_source_url"]
    data = _guarded_fetch(fetcher, LICENSE_TEXT_URL, "text")
    try:
        data.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise FetchError("license text is not valid UTF-8") from exc
    return data, LICENSE_TEXT_URL


def command_review(args: argparse.Namespace) -> int:
    staging = _require_safe_staging(Path(args.staging), Path(args.committed), must_be_new=False)
    state = read_corpus_directory(staging)
    manifest = state["manifest"]
    if manifest["corpus_kind"] != corpus.KIND_AUTHORITATIVE:
        raise FetchError("only an authoritative staging corpus can be reviewed")
    reviewer = args.reviewer
    reviewed_on = args.reviewed_on
    license_block = manifest["license"]
    for key, value in (
        ("identifier", args.license_identifier),
        ("site_notice_as_displayed", args.site_notice),
        ("invariant_sections", args.invariant_sections),
        ("cover_texts", args.cover_texts),
    ):
        if value is not None:
            license_block[key] = value
    if args.approve_license:
        license_block["review"] = {
            "status": corpus.LICENSE_REVIEW_APPROVED,
            "reviewer": reviewer,
            "reviewed_on": reviewed_on,
        }
    pages = {page["id"]: page for page in manifest["pages"]}
    for page_id in args.page_reviewed or []:
        if page_id not in pages:
            raise FetchError(f"unknown page id {page_id}")
        pages[page_id]["license_review"] = {
            "status": corpus.PAGE_REVIEW_NO_EXCEPTIONS,
            "reviewer": reviewer,
            "reviewed_on": reviewed_on,
            "notes": None,
        }
    for page_id, note in args.page_exception or []:
        if page_id not in pages:
            raise FetchError(f"unknown page id {page_id}")
        pages[page_id]["license_review"] = {
            "status": corpus.PAGE_REVIEW_EXCEPTION_FOUND,
            "reviewer": reviewer,
            "reviewed_on": reviewed_on,
            "notes": note,
        }
    manifest["corpus_digest"] = corpus.compute_corpus_digest(manifest)
    _validate_staging(manifest, require_reviewed=False)
    _write_corpus(staging, _corpus_files(staging, manifest, state["releases"]), exclusive=False)
    print(f"Recorded review in {staging}")
    return 0


def command_release(args: argparse.Namespace) -> int:
    committed = Path(args.committed)
    staging = _require_safe_staging(Path(args.staging), committed, must_be_new=False)
    state = read_corpus_directory(staging)
    current = read_corpus_directory(committed)
    manifest = state["manifest"]
    if state["releases"] != current["releases"]:
        raise FetchError("staging release ledger differs from the committed ledger; refetch")
    _validate_staging(manifest, require_reviewed=True)
    if any(page["license_review"]["status"] != corpus.PAGE_REVIEW_NO_EXCEPTIONS for page in manifest["pages"]):
        raise FetchError("every page must be reviewed as free of license exceptions")

    version = args.corpus_version
    latest = current["entries"][-1]
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version or ""):
        raise FetchError("--corpus-version must be MAJOR.MINOR.PATCH")
    if tuple(int(p) for p in version.split(".")) <= tuple(int(p) for p in latest["corpus_version"].split(".")):
        raise FetchError("--corpus-version must be greater than the committed corpus version")

    manifest["corpus_version"] = version
    manifest["corpus_digest"] = corpus.compute_corpus_digest(manifest)
    if manifest["corpus_digest"] == latest["corpus_digest"]:
        raise FetchError("staged corpus is identical to the committed release; nothing to release")
    releases = json.loads(json.dumps(current["releases"]))
    releases["releases"].append(
        {
            "corpus_version": version,
            "corpus_kind": manifest["corpus_kind"],
            "corpus_digest": manifest["corpus_digest"],
            "previous_corpus_digest": latest["corpus_digest"],
        }
    )
    try:
        corpus.validate_releases(releases)
    except corpus.CorpusError as exc:
        raise FetchError(f"release ledger would be invalid: {exc.detail}") from exc
    _write_corpus(staging, _corpus_files(staging, manifest, releases), exclusive=False)

    policy = corpus.AUTHORITATIVE_POLICY
    license_sha = manifest["license"]["license_text_sha256"]
    if policy.license_text_sha256 is None:
        policy = replace(policy, license_text_sha256=license_sha)
        print(
            "NOTE: the loader does not pin a license text digest yet. After verifying the "
            f"license text, pin {license_sha} as AUTHORITATIVE_POLICY.license_text_sha256 "
            "in src/bmd_agent/vasp_wiki_corpus.py in the same reviewed change."
        )
    result = corpus.load_vasp_wiki_corpus(staging, policy=policy)
    if isinstance(result, corpus.CorpusUnavailable):
        print(f"Staged release does not load: {result.reason}: {result.detail}")
        return 1
    print(f"Released corpus {version} ({manifest['corpus_digest']}) in staging {staging}")
    print(f"Copy the staging directory over {committed} in a reviewed pull request.")
    return 0


def diff_corpora(committed: Mapping[str, Any], staged: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Compare two manifests; return (report lines, alerts)."""

    lines: list[str] = []
    alerts: list[str] = []
    old_pages = {page["id"]: page for page in committed["pages"]}
    new_pages = {page["id"]: page for page in staged["pages"]}
    for page_id in sorted(set(old_pages) | set(new_pages)):
        old = old_pages.get(page_id)
        new = new_pages.get(page_id)
        if old is None:
            lines.append(f"added      {page_id}: {new['title']} r{new['revision_id']}")
            continue
        if new is None:
            lines.append(f"removed    {page_id}: {old['title']} r{old['revision_id']}")
            continue
        if old["page_id"] != new["page_id"]:
            alerts.append(f"{page_id}: now resolves to a different upstream page ID")
        if old["title"] != new["title"]:
            lines.append(f"renamed    {page_id}: {old['title']} -> {new['title']}")
        if old["revision_id"] != new["revision_id"]:
            lines.append(f"revision   {page_id}: r{old['revision_id']} -> r{new['revision_id']}")
        elif old["content_sha256"] != new["content_sha256"] or old["upstream_sha1"] != new["upstream_sha1"]:
            alerts.append(f"{page_id}: content changed while the revision ID stayed r{new['revision_id']}")
        else:
            lines.append(f"unchanged  {page_id}: r{new['revision_id']}")
    old_upstream = committed.get("upstream") or {}
    new_upstream = staged.get("upstream") or {}
    for key in ("rights_text", "rights_url"):
        if old_upstream and old_upstream.get(key) != new_upstream.get(key):
            alerts.append(f"site {key} changed")
    old_license = committed.get("license") or {}
    new_license = staged.get("license") or {}
    if old_license and old_license.get("license_text_sha256") != new_license.get("license_text_sha256"):
        alerts.append("license text changed")
    if committed["corpus_digest"] != staged["corpus_digest"]:
        lines.append(f"identity   {committed['corpus_digest']} -> {staged['corpus_digest']}")
        if committed["corpus_version"] == staged["corpus_version"]:
            alerts.append("corpus identity changed without a corpus version change")
    else:
        lines.append("identity   unchanged")
    return lines, alerts


def command_diff(args: argparse.Namespace) -> int:
    committed = read_corpus_directory(Path(args.committed))["manifest"]
    staged = read_corpus_directory(Path(args.staging))["manifest"]
    lines, alerts = diff_corpora(committed, staged)
    for line in lines:
        print(line)
    for alert in alerts:
        print(f"ALERT      {alert}")
    return 1 if alerts else 0


def command_verify(args: argparse.Namespace) -> int:
    policy = corpus.AUTHORITATIVE_POLICY
    if args.license_text_sha256:
        policy = replace(policy, license_text_sha256=args.license_text_sha256)
    result = corpus.load_vasp_wiki_corpus(Path(args.directory), policy=policy)
    if isinstance(result, corpus.CorpusUnavailable):
        print(f"unavailable: {result.reason}: {result.detail}")
        return 0 if result.reason == corpus.UNAVAILABLE_UNPOPULATED else 1
    print(f"verified corpus {result.corpus_version} ({result.corpus_digest}), {len(result.pages)} page(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--committed", default=str(COMMITTED_CORPUS), help="committed corpus directory")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="acquire pages into a new staging directory")
    fetch.add_argument("--api-url", required=True, help="official VASP Wiki api.php URL (HTTPS)")
    fetch.add_argument("--staging", required=True)
    fetch.add_argument("--page", action="append", help="Title or Title@REVISION; repeatable")
    fetch.add_argument("--intended", action="store_true", help="add the ten intended corpus titles")
    fetch.add_argument("--retrieved-at", help="override the UTC retrieval timestamp")
    fetch.add_argument("--accept-site-rights-change", action="store_true")
    fetch.add_argument("--refetch-license-text", action="store_true")

    review = commands.add_parser("review", help="record license and page review in staging")
    review.add_argument("--staging", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--reviewed-on", required=True, help="YYYY-MM-DD")
    review.add_argument("--license-identifier", choices=sorted(corpus.AUTHORITATIVE_POLICY.allowed_license_identifiers))
    review.add_argument("--site-notice", help="license notice exactly as the site displays it")
    review.add_argument("--invariant-sections", choices=[corpus.NONE_DECLARED])
    review.add_argument("--cover-texts", choices=[corpus.NONE_DECLARED])
    review.add_argument("--approve-license", action="store_true")
    review.add_argument("--page-reviewed", action="append", metavar="ID")
    review.add_argument("--page-exception", action="append", nargs=2, metavar=("ID", "NOTE"))

    release = commands.add_parser("release", help="version and verify a reviewed staging corpus")
    release.add_argument("--staging", required=True)
    release.add_argument("--corpus-version", required=True)

    diff = commands.add_parser("diff", help="compare staging with the committed corpus")
    diff.add_argument("--staging", required=True)

    verify = commands.add_parser("verify", help="run the BMD Check loader on a corpus directory")
    verify.add_argument("directory", nargs="?", default=str(COMMITTED_CORPUS))
    verify.add_argument("--license-text-sha256", help="digest to accept before it is pinned")
    return parser


def main(argv: Sequence[str] | None = None, *, fetcher: Fetcher | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "fetch":
            return command_fetch(args, fetcher or https_fetch)
        if args.command == "review":
            return command_review(args)
        if args.command == "release":
            return command_release(args)
        if args.command == "diff":
            return command_diff(args)
        return command_verify(args)
    except (FetchError, corpus.CorpusError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
