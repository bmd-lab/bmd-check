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


class PageRefused(FetchError):
    """One page could not be preserved.

    ``outcome`` is ``"excluded"`` (a deliberate, policy-defined exclusion such
    as content outside the archival character policy) or ``"failed"`` (the page
    could not be acquired reliably, which makes a discovery incomplete).
    ``reason`` is from the loader's closed exclusion or failure reasons.
    """

    def __init__(self, outcome: str, reason: str, message: str) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.reason = reason


def _excluded(reason: str, message: str) -> PageRefused:
    assert reason in corpus.EXCLUSION_REASONS
    return PageRefused("excluded", reason, message)


def _failed(reason: str, message: str) -> PageRefused:
    assert reason in corpus.FAILURE_REASONS
    return PageRefused("failed", reason, message)


# ---------------------------------------------------------------------------
# HTTPS transport (real network; replaced by a fake fetcher in tests)
# ---------------------------------------------------------------------------


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802 - urllib API
        # Name the target only when it is itself an acceptable request URL, so
        # the maintainer can rerun with it explicitly; never follow it.
        try:
            validate_request_url(newurl, corpus.AUTHORITATIVE_POLICY.allowed_hosts | LICENSE_TEXT_HOSTS)
            target = urlsplit(newurl)
            hint = f" to {target.scheme}://{target.netloc}{target.path}"
        except (FetchError, ValueError, TypeError):
            hint = ""
        raise FetchError(f"HTTP redirect refused ({code}){hint}; request the final URL explicitly")


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
    query, continuation = _api_payload(data)
    if continuation is not None:
        raise FetchError("API response is incomplete (continuation requested)")
    return query


def _api_payload(data: bytes) -> tuple[Mapping[str, Any], Mapping[str, str] | None]:
    """Parse one API response; return ``(query, continue)``.

    Errors and warnings are always refused. A continuation is returned (not
    interpreted) so that list and category queries can paginate explicitly.
    """

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
    continuation = payload.get("continue")
    if continuation is not None:
        if (
            not isinstance(continuation, dict)
            or not continuation
            or len(continuation) > 8
            or any(type(key) is not str or type(value) is not str for key, value in continuation.items())
            or any(not re.fullmatch(r"[a-z]{1,32}", key) for key in continuation)
        ):
            raise FetchError("API continuation is malformed")
    query = payload.get("query")
    if query is None and continuation is None and payload.get("batchcomplete"):
        query = {}
    if not isinstance(query, dict):
        raise FetchError("API response has no query object")
    return query, continuation


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
    categories = fetch_categories(endpoint, [page["title"]], fetcher)[page["title"]]
    return page_record(
        page,
        site,
        requested_title=request.title,
        redirected_from=redirected_from,
        retrieved_at=retrieved_at,
        categories=categories,
    )


def page_record(
    page: Mapping[str, Any],
    site: Mapping[str, str],
    *,
    requested_title: str,
    redirected_from: str | None,
    retrieved_at: str,
    categories: Sequence[str],
    slug: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Verify one validated API page entry and build its manifest record."""

    if slug is None:
        slug = corpus.title_slug(requested_title)
        if not slug:
            raise FetchError("page title does not yield a usable corpus identifier")
    revision = page["revisions"][0]
    content = revision["slots"]["main"]["content"]
    try:
        data = content.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise _excluded(
            "archival_character_policy", "page content is not representable as UTF-8"
        ) from exc
    if len(data) > corpus.MAX_PAGE_BYTES:
        raise _excluded("page_size_limit", "page content exceeds the page size limit")
    if len(data) != revision["size"]:
        raise _failed("integrity_mismatch", "page content size does not match the upstream revision size")
    if hashlib.sha1(data, usedforsecurity=False).hexdigest() != revision["sha1"]:
        raise _failed("integrity_mismatch", "page content does not match the upstream MediaWiki SHA-1")
    if corpus.find_disallowed_character(content) is not None:
        raise _excluded(
            "archival_character_policy",
            "page content violates the archival character policy; human review required",
        )

    script_url = site["server"] + site["script_path"]
    title_parameter = quote(page["title"].replace(" ", "_"), safe="")
    revision_id = revision["revid"]
    record = {
        "id": corpus.page_id_for_slug(slug),
        "requested_title": requested_title,
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
        "categories": sorted(set(categories)),
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
    return validate_page_entry(pages[0], expect_revisions=expect_revisions)


def validate_page_entry(page: Any, *, expect_revisions: bool = True) -> Mapping[str, Any]:
    """Validate one ``formatversion=2`` page entry; refusals carry an outcome."""

    if not isinstance(page, dict):
        raise FetchError("API page entry is malformed")
    for flag in ("missing", "invalid", "special"):
        if page.get(flag):
            raise _failed("missing_at_retrieval", f"requested page is {flag}")
    if page.get("redirect"):
        raise _failed("missing_at_retrieval", "API returned a redirect page instead of its target")
    if type(page.get("pageid")) is not int or page["pageid"] <= 0:
        raise FetchError("API page ID is malformed")
    if page.get("ns") != 0 or type(page.get("ns")) is not int:
        raise FetchError("only main-namespace pages may be preserved")
    title = page.get("title")
    if type(title) is not str or not title or corpus.find_disallowed_character(title) is not None:
        raise FetchError("API page title is malformed")
    if page.get("contentmodel") != "wikitext":
        raise _excluded("content_model", "page content model is not wikitext")
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
            raise _excluded("revision_unavailable", f"revision is {flag}")
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
        raise _excluded("revision_unavailable", "revision main slot content is unavailable")
    if main.get("contentmodel") != "wikitext" or main.get("contentformat") != "text/x-wiki":
        raise _excluded("content_model", "revision content is not wikitext")
    if type(main.get("content")) is not str:
        raise FetchError("revision content is missing")
    return page


# ---------------------------------------------------------------------------
# Systematic discovery and batched acquisition (``fetch --all``)
# ---------------------------------------------------------------------------

# Non-bot MediaWiki clients may request up to 500 list entries and up to 50
# titles per query (and 50 pages of revision content).
LIST_LIMIT = 500
TITLE_BATCH = 50
MAX_CONTINUATIONS = 10_000
DEFAULT_REQUEST_DELAY_SECONDS = 1.0
DEFAULT_RETRIES = 3


class PoliteFetcher:
    """Wrap a fetcher with a fixed delay between requests and bounded retries.

    Only transport failures are retried (network errors and HTTP statuses);
    a response that arrives but is malformed is never retried silently.
    """

    def __init__(
        self,
        fetcher: Fetcher,
        *,
        delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
        retries: int = DEFAULT_RETRIES,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        import time

        self._fetcher = fetcher
        self._delay = max(0.0, float(delay))
        self._retries = max(0, int(retries))
        self._sleep = sleep or time.sleep
        self._first = True
        self.requests = 0

    def __call__(self, url: str, expected: str) -> bytes:
        attempt = 0
        while True:
            if not self._first and self._delay:
                self._sleep(self._delay * (2**attempt))
            self._first = False
            self.requests += 1
            try:
                return self._fetcher(url, expected)
            except FetchError as exc:
                transient = str(exc).startswith(("request failed", "unexpected HTTP status"))
                if not transient or attempt >= self._retries:
                    raise
                attempt += 1


def _query(endpoint: str, params: Mapping[str, str], fetcher: Fetcher) -> tuple[Mapping[str, Any], Mapping[str, str] | None]:
    base = {"action": "query", "format": "json", "formatversion": "2"}
    return _api_payload(_guarded_fetch(fetcher, api_request_url(endpoint, {**base, **params}), "json"))


def _paginate(endpoint: str, params: Mapping[str, str], fetcher: Fetcher) -> list[Mapping[str, Any]]:
    """Follow ``continue`` until the API reports the result set complete."""

    queries: list[Mapping[str, Any]] = []
    continuation: Mapping[str, str] | None = {}
    seen: set[tuple[tuple[str, str], ...]] = set()
    while continuation is not None:
        key = tuple(sorted(continuation.items()))
        if key in seen:
            raise FetchError("API continuation does not advance")
        seen.add(key)
        if len(seen) > MAX_CONTINUATIONS:
            raise FetchError("API continuation exceeds the request bound")
        query, continuation = _query(endpoint, {**params, **continuation}, fetcher)
        queries.append(query)
    return queries


def fetch_namespaces(endpoint: str, fetcher: Fetcher) -> list[tuple[int, str, bool]]:
    """``(id, canonical name, content flag)`` for every real namespace."""

    query, continuation = _query(endpoint, {"meta": "siteinfo", "siprop": "namespaces"}, fetcher)
    if continuation is not None:
        raise FetchError("siteinfo namespaces response is incomplete")
    namespaces = query.get("namespaces")
    if not isinstance(namespaces, dict) or "0" not in namespaces:
        raise FetchError("siteinfo namespaces response is malformed")
    result: list[tuple[int, str, bool]] = []
    for key, item in namespaces.items():
        if not isinstance(item, dict) or type(item.get("id")) is not int or str(item["id"]) != key:
            raise FetchError("siteinfo namespace entry is malformed")
        if item["id"] < 0:
            continue  # Special and Media are virtual namespaces without pages
        name = item.get("canonical", item.get("name", ""))
        if type(name) is not str:
            raise FetchError("siteinfo namespace name is malformed")
        result.append((item["id"], name or "(Main)", bool(item.get("content"))))
    return sorted(result)


def list_namespace_titles(
    endpoint: str, namespace: int, fetcher: Fetcher, *, redirects: bool
) -> list[tuple[str, int]]:
    """Every ``(title, page ID)`` in a namespace, redirects or non-redirects only."""

    queries = _paginate(
        endpoint,
        {
            "list": "allpages",
            "apnamespace": str(namespace),
            "apfilterredir": "redirects" if redirects else "nonredirects",
            "aplimit": str(LIST_LIMIT),
        },
        fetcher,
    )
    found: dict[str, int] = {}
    for query in queries:
        items = query.get("allpages")
        if not isinstance(items, list):
            raise FetchError("allpages response is malformed")
        for item in items:
            if (
                not isinstance(item, dict)
                or type(item.get("title")) is not str
                or type(item.get("pageid")) is not int
                or item["pageid"] <= 0
                or item.get("ns") != namespace
            ):
                raise FetchError("allpages entry is malformed")
            title = item["title"]
            if not title or corpus.find_disallowed_character(title) is not None or len(title) > 255:
                raise FetchError("allpages title is malformed")
            if title in found and found[title] != item["pageid"]:
                raise FetchError("allpages lists one title with two page IDs")
            found[title] = item["pageid"]
    page_ids = list(found.values())
    if len(set(page_ids)) != len(page_ids):
        raise FetchError("allpages lists one page ID under two titles")
    return sorted(found.items())


def _batches(items: Sequence[str], size: int = TITLE_BATCH) -> list[list[str]]:
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


# Keys MediaWiki's ApiPageSet emits for each resolved redirect.
_REDIRECT_ENTRY_KEYS = frozenset({"from", "to", "tofragment", "tointerwiki"})
# A redirect chain is resolved hop by hop; bound the entries one batch may add.
MAX_REDIRECT_ENTRIES_PER_TITLE = 8


def _redirect_title(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 255
        or value != value.strip()
        or corpus.find_disallowed_character(value) is not None
        or "\n" in value
        or "\t" in value
    ):
        raise FetchError(f"API redirect {label} is malformed")
    return value


def resolve_redirects(endpoint: str, titles: Sequence[str], fetcher: Fetcher) -> dict[str, str | None]:
    """Map each redirect title to its immediate target (``None`` if unresolved).

    With ``redirects=1`` MediaWiki resolves redirects recursively and reports
    every hop, so a batch can legitimately return more entries than titles
    (a double redirect ``A -> B -> C`` yields two entries for the one title
    ``A``), and returns no ``redirects`` key at all when nothing resolves.
    Every entry must be a well-formed hop reachable from a requested title;
    only the first hop of each requested title is recorded.
    """

    targets: dict[str, str | None] = {title: None for title in titles}
    for batch in _batches(sorted(titles)):
        query, continuation = _query(
            endpoint, {"prop": "info", "redirects": "1", "titles": "|".join(batch)}, fetcher
        )
        if continuation is not None:
            raise FetchError("redirect resolution response is incomplete")
        items = query.get("redirects", [])
        if not isinstance(items, list):
            raise FetchError("API redirects list is not a list")
        if len(items) > len(batch) * MAX_REDIRECT_ENTRIES_PER_TITLE:
            raise FetchError(
                f"API redirects list has {len(items)} entries for {len(batch)} titles, beyond the chain bound"
            )
        hops: dict[str, str] = {}
        for item in items:
            if not isinstance(item, dict) or set(item) - _REDIRECT_ENTRY_KEYS or not {"from", "to"} <= set(item):
                raise FetchError("API redirects entry is malformed")
            source = _redirect_title(item["from"], "source")
            target = _redirect_title(item["to"], "target")
            if "tofragment" in item and type(item["tofragment"]) is not str:
                raise FetchError("API redirect fragment is malformed")
            if "tointerwiki" in item:
                prefix = _redirect_title(item["tointerwiki"], "interwiki prefix")
                target = f"{prefix}:{target}"
            if source in hops:
                raise FetchError("API redirects list resolves one title twice")
            hops[source] = target
        # Every hop must start at a requested title or continue an earlier hop.
        reachable = set(batch) | set(hops.values())
        if set(hops) - reachable:
            raise FetchError("API redirects list contains a hop unrelated to the requested titles")
        for title in batch:
            if title in hops:
                targets[title] = hops[title]
    return targets


def fetch_categories(endpoint: str, titles: Sequence[str], fetcher: Fetcher) -> dict[str, list[str]]:
    """Every category of each page, following ``clcontinue`` pagination."""

    result: dict[str, set[str]] = {title: set() for title in titles}
    for batch in _batches(list(titles)):
        for query in _paginate(
            endpoint, {"prop": "categories", "cllimit": "max", "titles": "|".join(batch)}, fetcher
        ):
            pages = query.get("pages", [])
            if not isinstance(pages, list):
                raise FetchError("categories response is malformed")
            for page in pages:
                if not isinstance(page, dict) or type(page.get("title")) is not str:
                    raise FetchError("categories page entry is malformed")
                if page["title"] not in result:
                    raise FetchError("categories response names a page that was not requested")
                items = page.get("categories", [])
                if not isinstance(items, list):
                    raise FetchError("categories list is malformed")
                for item in items:
                    title = item.get("title") if isinstance(item, dict) else None
                    if (
                        type(title) is not str
                        or not title
                        or len(title) > 255
                        or corpus.find_disallowed_character(title) is not None
                        or "\n" in title
                        or "\t" in title
                        or title != title.strip()
                    ):
                        raise FetchError("category title is malformed")
                    result[page["title"]].add(title)
    return {title: sorted(values) for title, values in result.items()}


def _content_params(titles: Sequence[str]) -> dict[str, str]:
    return {
        "prop": "info|revisions",
        "inprop": "url",
        "titles": "|".join(titles),
        "rvprop": _RVPROP,
        "rvslots": "main",
    }


def fetch_content_batch(
    endpoint: str, titles: Sequence[str], fetcher: Fetcher
) -> dict[str, Mapping[str, Any] | PageRefused]:
    """Latest revision of each title, splitting the batch when it cannot be served whole.

    A batch that is refused for any reason (continuation because the response
    is too large, warnings, transport failure) is halved until single titles
    remain, so one bad page never hides its neighbours. The outcome for each
    title is either a validated page entry or a ``PageRefused``.
    """

    outcomes: dict[str, Mapping[str, Any] | PageRefused] = {}
    pending: list[list[str]] = [list(titles)]
    while pending:
        batch = pending.pop(0)
        try:
            query, continuation = _query(endpoint, _content_params(batch), fetcher)
            if continuation is not None:
                raise FetchError("content response is incomplete (continuation requested)")
            if query.get("normalized") or query.get("redirects"):
                raise FetchError("discovered titles must not need normalization or redirects")
            pages = query.get("pages")
            if not isinstance(pages, list) or len(pages) != len(batch):
                raise FetchError("content response does not describe every requested title")
            by_title: dict[str, Any] = {}
            for page in pages:
                title = page.get("title") if isinstance(page, dict) else None
                if type(title) is not str or title not in batch or title in by_title:
                    raise FetchError("content response names an unexpected page")
                by_title[title] = page
        except FetchError as exc:
            if len(batch) > 1:
                middle = len(batch) // 2
                pending[:0] = [batch[:middle], batch[middle:]]
                continue
            reason = "request_failed" if str(exc).startswith(("request failed", "unexpected HTTP status")) else (
                "malformed_response"
            )
            outcomes[batch[0]] = exc if isinstance(exc, PageRefused) else _failed(reason, str(exc))
            continue
        for title in batch:
            try:
                outcomes[title] = validate_page_entry(by_title[title])
            except PageRefused as exc:
                outcomes[title] = exc
            except FetchError as exc:
                outcomes[title] = _failed("malformed_response", str(exc))
    return outcomes


def discover_and_fetch(
    endpoint: str,
    site: Mapping[str, str],
    fetcher: Fetcher,
    *,
    retrieved_at: str,
    progress: Callable[[str], None] = lambda message: None,
) -> tuple[list[tuple[dict[str, Any], bytes]], dict[str, Any]]:
    """Discover every main-namespace page and acquire its latest revision.

    Returns the preserved ``(record, bytes)`` pairs and the acquisition record.
    Inclusion rule ``vasp-wiki-main-namespace-1``: every non-redirect page in
    the main namespace (0) that the public API lists. Redirects are recorded as
    aliases, never preserved as pages; other namespaces are counted only.
    """

    namespaces = fetch_namespaces(endpoint, fetcher)
    progress(f"namespaces: {', '.join(f'{ns}={name}' for ns, name, _ in namespaces)}")
    other_namespaces = []
    for namespace, name, _content in namespaces:
        if namespace == 0:
            continue
        count = len(list_namespace_titles(endpoint, namespace, fetcher, redirects=False))
        other_namespaces.append({"namespace": namespace, "name": name, "pages": count})
    discovered = list_namespace_titles(endpoint, 0, fetcher, redirects=False)
    progress(f"discovered {len(discovered)} main-namespace pages")
    redirect_titles = [title for title, _ in list_namespace_titles(endpoint, 0, fetcher, redirects=True)]
    targets = resolve_redirects(endpoint, redirect_titles, fetcher)
    progress(f"recorded {len(redirect_titles)} redirects")

    titles = [title for title, _ in discovered]
    expected_ids = dict(discovered)
    entries: dict[str, Mapping[str, Any] | PageRefused] = {}
    for number, batch in enumerate(_batches(titles), start=1):
        entries.update(fetch_content_batch(endpoint, batch, fetcher))
        progress(f"content batch {number}/{(len(titles) + TITLE_BATCH - 1) // TITLE_BATCH}")
    good_titles = [title for title in titles if not isinstance(entries[title], PageRefused)]
    categories = fetch_categories(endpoint, good_titles, fetcher)
    progress("categories recorded")

    slugs = dict(
        zip(good_titles, corpus.assign_page_slugs([(title, entries[title]["pageid"]) for title in good_titles]))
    )
    preserved: list[tuple[dict[str, Any], bytes]] = []
    excluded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for title in titles:
        entry = entries[title]
        try:
            if isinstance(entry, PageRefused):
                raise entry
            if entry["pageid"] != expected_ids[title]:
                raise _failed("missing_at_retrieval", "page ID changed between discovery and retrieval")
            preserved.append(
                page_record(
                    entry,
                    site,
                    requested_title=title,
                    redirected_from=None,
                    retrieved_at=retrieved_at,
                    categories=categories[title],
                    slug=slugs[title],
                )
            )
        except PageRefused as exc:
            page = entry if isinstance(entry, Mapping) else {}
            revisions = page.get("revisions") if isinstance(page.get("revisions"), list) else []
            revision_id = revisions[0].get("revid") if revisions and isinstance(revisions[0], dict) else None
            if exc.outcome == "excluded":
                excluded.append(
                    {
                        "title": title,
                        "page_id": expected_ids[title],
                        "revision_id": revision_id if type(revision_id) is int and revision_id > 0 else None,
                        "reason": exc.reason,
                        "detail": str(exc),
                    }
                )
            else:
                failed.append({"title": title, "reason": exc.reason, "detail": str(exc)})
    if slugs and len(good_titles) != len(preserved):
        # Pages that failed verification after slug assignment would leave
        # gaps in disambiguated groups; recompute over the preserved set.
        final = corpus.assign_page_slugs([(record["requested_title"], record["page_id"]) for record, _ in preserved])
        preserved = [_reslug(record, data, slug) for (record, data), slug in zip(preserved, final)]

    acquisition = {
        "method": corpus.ACQUISITION_ALL_PAGES,
        "inclusion_rules": corpus.INCLUSION_RULES,
        "namespace": 0,
        "discovered_at": retrieved_at,
        "completeness": corpus.COMPLETENESS_INCOMPLETE if failed else corpus.COMPLETENESS_COMPLETE,
        "discovered_titles": len(titles),
        "redirects": [{"title": title, "target": targets[title]} for title in sorted(redirect_titles)],
        "excluded": sorted(excluded, key=lambda item: item["title"]),
        "failed": sorted(failed, key=lambda item: item["title"]),
        "other_namespaces": other_namespaces,
    }
    return preserved, acquisition


def _reslug(record: dict[str, Any], data: bytes, slug: str) -> tuple[dict[str, Any], bytes]:
    record = dict(record)
    record["id"] = corpus.page_id_for_slug(slug)
    record["content_file"] = corpus.content_file_for(slug, record["revision_id"])
    return record, data


def explicit_acquisition(pages: Sequence[tuple[dict[str, Any], bytes]], retrieved_at: str) -> dict[str, Any]:
    return {
        "method": corpus.ACQUISITION_EXPLICIT_TITLES,
        "inclusion_rules": corpus.INCLUSION_RULES,
        "namespace": 0,
        "discovered_at": retrieved_at,
        "completeness": corpus.COMPLETENESS_NOT_ASSESSED,
        "discovered_titles": len(pages),
        "redirects": [],
        "excluded": [],
        "failed": [],
        "other_namespaces": [],
    }


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
    # Checked before resolving: a staging path that is itself a symbolic link
    # would otherwise be followed to wherever it points.
    if staging.is_symlink() or (staging / corpus.PAGES_DIRECTORY).is_symlink():
        raise FetchError("staging directory must not be or contain symbolic links")
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

    # Re-check at write time, not only when the command started: the staging
    # directory and pages/ must be real directories (never links) right before
    # any file is written into them. Files are created with exclusive creation
    # ("xb"), which refuses an existing name including a symbolic link.
    for path in (directory, directory / corpus.PAGES_DIRECTORY):
        if path.is_symlink():
            raise FetchError("staging directory must not be or contain symbolic links")
        path.mkdir(parents=path == directory, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise FetchError("staging directory must not be or contain symbolic links")
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


def exclude_pages(manifest: dict[str, Any], excepted: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Move license-excepted pages into the acquisition exclusions.

    Remaining pages are re-identified with the deterministic rule (a removed
    page can dissolve a collision group). Returns ``{new content_file: old}``.
    """

    excepted_ids = {page["id"] for page in excepted}
    acquisition = manifest["acquisition"]
    for page in excepted:
        acquisition["excluded"].append(
            {
                "title": page["title"],
                "page_id": page["page_id"],
                "revision_id": page["revision_id"],
                "reason": "license_exception",
                "detail": page["license_review"]["notes"],
            }
        )
    acquisition["excluded"].sort(key=lambda item: item["title"])
    remaining = [page for page in manifest["pages"] if page["id"] not in excepted_ids]
    slugs = corpus.assign_page_slugs([(page["requested_title"], page["page_id"]) for page in remaining])
    renamed: dict[str, str] = {}
    for page, slug in zip(remaining, slugs):
        old = page["content_file"]
        page["id"] = corpus.page_id_for_slug(slug)
        page["content_file"] = corpus.content_file_for(slug, page["revision_id"])
        renamed[page["content_file"]] = old
    manifest["pages"] = sorted(remaining, key=lambda page: page["id"])
    return renamed


def _corpus_files(
    directory: Path,
    manifest: Mapping[str, Any],
    releases: Mapping[str, Any],
    *,
    renamed: Mapping[str, str] | None = None,
) -> dict[str, bytes]:
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
        source = (renamed or {}).get(page["content_file"], page["content_file"])
        files[page["content_file"]] = (directory / source).read_bytes()
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
    if args.all and requests:
        raise FetchError("--all discovers every page; it cannot be combined with --page or --intended")
    if not requests and not args.all:
        raise FetchError("no pages requested; use --all, --page or --intended")
    slugs = [corpus.title_slug(item.title) for item in requests]
    if len(set(slugs)) != len(slugs):
        raise FetchError("requested titles collide on the same corpus identifier")
    if args.all:
        fetcher = PoliteFetcher(fetcher, delay=args.delay, retries=args.retries)

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

    if args.all:
        pages, acquisition = discover_and_fetch(
            endpoint,
            site,
            fetcher,
            retrieved_at=retrieved_at,
            progress=lambda message: print(message, file=sys.stderr),
        )
        if not pages:
            raise FetchError("discovery preserved no pages; refusing to stage an empty corpus")
    else:
        pages = [fetch_page(endpoint, request, site, fetcher, retrieved_at=retrieved_at) for request in requests]
        acquisition = explicit_acquisition(pages, retrieved_at)
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
        "schema_version": corpus.CURRENT_MANIFEST_SCHEMA_VERSION,
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
        "acquisition": acquisition,
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
    print(
        f"Acquisition: {acquisition['method']}, completeness {acquisition['completeness']}; "
        f"{acquisition['discovered_titles']} discovered, {len(acquisition['excluded'])} excluded, "
        f"{len(acquisition['failed'])} failed, {len(acquisition['redirects'])} redirects recorded"
    )
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


def read_id_list(path: Path) -> list[str]:
    """Page IDs, one per line; blank lines and ``#`` comments are ignored."""

    ids: list[str] = []
    for line in path.read_text("utf-8").splitlines():
        item = line.split("#", 1)[0].strip()
        if item:
            ids.append(item)
    if len(set(ids)) != len(ids):
        raise FetchError(f"{path} lists a page ID twice")
    return ids


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
    reviewed_list = list(args.page_reviewed or [])
    for path in args.page_reviewed_file or []:
        reviewed_list.extend(read_id_list(Path(path)))
    args.page_reviewed = reviewed_list
    reviewed_ids = set(reviewed_list)
    excepted_ids = {page_id for page_id, _ in args.page_exception or []}
    if reviewed_ids & excepted_ids:
        raise FetchError("a page cannot be both free of exceptions and have an exception")
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
        if not note.strip() or note != note.strip():
            raise FetchError("a license exception note must be non-empty text without surrounding whitespace")
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
    renamed: dict[str, str] = {}
    if any(page["license_review"]["status"] == corpus.PAGE_REVIEW_PENDING for page in manifest["pages"]):
        raise FetchError("every page must be reviewed before release")
    excepted = [
        page for page in manifest["pages"] if page["license_review"]["status"] == corpus.PAGE_REVIEW_EXCEPTION_FOUND
    ]
    if excepted:
        if not args.exclude_excepted:
            raise FetchError(
                "every page must be reviewed as free of license exceptions "
                "(or rerun with --exclude-excepted to exclude excepted pages explicitly)"
            )
        if manifest.get("schema_version") != corpus.CURRENT_MANIFEST_SCHEMA_VERSION:
            raise FetchError("excluding pages needs a schema v2 staging corpus with an acquisition record")
        renamed = exclude_pages(manifest, excepted)
    acquisition = manifest.get("acquisition")
    if (
        acquisition is not None
        and acquisition["completeness"] == corpus.COMPLETENESS_INCOMPLETE
        and not args.allow_incomplete
    ):
        raise FetchError(
            f"acquisition is incomplete ({len(acquisition['failed'])} failed page(s)); refetch, "
            "or rerun with --allow-incomplete to release a corpus labelled incomplete"
        )
    _validate_staging(manifest, require_reviewed=True)

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
    _write_corpus(staging, _corpus_files(staging, manifest, releases, renamed=renamed), exclusive=False)

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


# ---------------------------------------------------------------------------
# Acquisition report, licensing scan and upstream reproduction
# ---------------------------------------------------------------------------

# Text that may signal a page-specific "unless otherwise noted" exception. A
# hit is not an exception: it marks the page for the maintainer's review.
LICENSE_MARKERS = (
    ("copyright", re.compile(r"copyright|\u00a9|\(c\)\s*[0-9]{4}", re.IGNORECASE)),
    ("license", re.compile(r"licen[cs]e|GFDL|GNU Free Documentation|creative ?commons|\bCC[- ]BY", re.IGNORECASE)),
    ("permission", re.compile(r"(used|reproduced|reprinted) (with|by) (kind )?permission|all rights reserved", re.IGNORECASE)),
    ("quotation", re.compile(r"<blockquote|\{\{\s*quote", re.IGNORECASE)),
    ("file", re.compile(r"\[\[\s*(file|image|media)\s*:", re.IGNORECASE)),
)


def scan_licensing(directory: Path) -> list[tuple[str, str, list[str]]]:
    """``(page id, title, markers)`` for every preserved page, sorted by id."""

    state = read_corpus_directory(directory)
    rows: list[tuple[str, str, list[str]]] = []
    for page in state["manifest"]["pages"]:
        text = (directory / page["content_file"]).read_bytes().decode("utf-8", "strict")
        markers = [name for name, pattern in LICENSE_MARKERS if pattern.search(text)]
        rows.append((page["id"], page["title"], markers))
    return rows


def command_scan_licensing(args: argparse.Namespace) -> int:
    rows = scan_licensing(Path(args.staging))
    flagged = [row for row in rows if row[2]]
    for page_id, title, markers in flagged:
        print(f"REVIEW  {page_id}\t{title}\t{','.join(markers)}")
    if args.write_unflagged:
        lines = ["# Page IDs whose wikitext has no licensing marker (generated by scan-licensing).", ""]
        lines.extend(page_id for page_id, _, markers in rows if not markers)
        Path(args.write_unflagged).write_text("\n".join(lines) + "\n", "utf-8")
    print(f"{len(rows)} page(s) scanned; {len(flagged)} flagged for manual licensing review", file=sys.stderr)
    return 0


def verify_upstream(directory: Path, endpoint: str, fetcher: Fetcher) -> tuple[int, list[str]]:
    """Re-fetch every pinned revision and compare it with the preserved bytes."""

    state = read_corpus_directory(directory)
    pages = {page["revision_id"]: page for page in state["manifest"]["pages"]}
    problems: list[str] = []
    for batch in _batches([str(revision) for revision in sorted(pages)]):
        query, continuation = _query(
            endpoint,
            {"prop": "revisions", "revids": "|".join(batch), "rvprop": _RVPROP, "rvslots": "main"},
            fetcher,
        )
        if continuation is not None:
            raise FetchError("revision response is incomplete (continuation requested)")
        found: dict[int, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
        for entry in query.get("pages", []):
            for revision in entry.get("revisions", []) if isinstance(entry, dict) else []:
                if isinstance(revision, dict) and type(revision.get("revid")) is int:
                    found[revision["revid"]] = (entry, revision)
        for revision_id in (int(item) for item in batch):
            page = pages[revision_id]
            if revision_id not in found:
                problems.append(f"{page['id']}: revision r{revision_id} is not available upstream")
                continue
            entry, revision = found[revision_id]
            content = ((revision.get("slots") or {}).get("main") or {}).get("content")
            if entry.get("pageid") != page["page_id"] or entry.get("title") != page["title"]:
                problems.append(f"{page['id']}: revision r{revision_id} now belongs to another page or title")
            if type(content) is not str:
                problems.append(f"{page['id']}: revision r{revision_id} content is unavailable")
                continue
            data = content.encode("utf-8", "surrogatepass")
            local = (directory / page["content_file"]).read_bytes()
            if data != local or corpus.sha256_hex(data) != page["content_sha256"]:
                problems.append(f"{page['id']}: upstream r{revision_id} bytes differ from the preserved file")
    return len(pages), problems


def command_verify_upstream(args: argparse.Namespace, fetcher: Fetcher) -> int:
    directory = Path(args.directory)
    manifest = read_corpus_directory(directory)["manifest"]
    if manifest["corpus_kind"] == corpus.KIND_UNPOPULATED:
        raise FetchError("an unpopulated corpus has no pinned revisions")
    endpoint = validate_request_url(manifest["upstream"]["api_endpoint"], corpus.AUTHORITATIVE_POLICY.allowed_hosts)
    polite = PoliteFetcher(fetcher, delay=args.delay, retries=args.retries)
    count, problems = verify_upstream(directory, endpoint, polite)
    for problem in problems:
        print(f"MISMATCH   {problem}")
    print(f"{count - len(problems) if problems else count} of {count} pinned revision(s) reproduced byte for byte")
    return 1 if problems else 0


def render_report(directory: Path) -> str:
    """Deterministic Markdown acquisition report rendered from a corpus directory."""

    state = read_corpus_directory(directory)
    manifest = state["manifest"]
    if manifest["corpus_kind"] == corpus.KIND_UNPOPULATED:
        return "# VASP Wiki corpus acquisition report\n\nThe corpus is unpopulated.\n"
    acquisition = manifest.get("acquisition") or {}
    pages = manifest["pages"]
    total = sum(page["content_bytes"] for page in pages)
    sums = (directory / corpus.SHA256SUMS_FILE).read_bytes()
    on_disk = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    lines = [
        "# VASP Wiki corpus acquisition report",
        "",
        "Generated by `tools/vasp_reference/fetch_corpus.py report` from the corpus manifest.",
        "",
        "## Release",
        "",
        f"- Corpus version: `{manifest['corpus_version']}`",
        f"- Corpus identity: `{manifest['corpus_digest']}`",
        f"- `manifest.json` SHA-256: `{corpus.sha256_hex((directory / corpus.MANIFEST_FILE).read_bytes())}`",
        f"- `SHA256SUMS` SHA-256: `{corpus.sha256_hex(sums)}`",
        f"- Upstream: {manifest['upstream']['site_name']} <{manifest['upstream']['api_endpoint']}>"
        f" ({manifest['upstream']['generator']})",
        f"- Site rights (MediaWiki API): {manifest['upstream']['rights_text']}",
        f"- License review: {manifest['license']['review']['status']}",
        "",
        "## Coverage",
        "",
        f"- Method: `{acquisition.get('method', 'explicit_titles (schema v1)')}`",
        f"- Inclusion rules: `{acquisition.get('inclusion_rules', 'n/a')}`",
        f"- Discovered at: {acquisition.get('discovered_at', 'n/a')}",
        f"- Completeness: **{acquisition.get('completeness', 'not_assessed')}**",
        f"- Main-namespace titles discovered: {acquisition.get('discovered_titles', len(pages))}",
        f"- Pages preserved: {len(pages)}",
        f"- Pages excluded: {len(acquisition.get('excluded', []))}",
        f"- Pages failed: {len(acquisition.get('failed', []))}",
        f"- Redirects recorded as aliases (not preserved): {len(acquisition.get('redirects', []))}",
        f"- Preserved wikitext: {total:,} bytes in {len(pages)} files; corpus directory {on_disk:,} bytes",
        "",
    ]
    reviews: dict[str, int] = {}
    for page in pages:
        status = page["license_review"]["status"]
        reviews[status] = reviews.get(status, 0) + 1
    lines.extend(["## Page license review", ""])
    lines.extend(f"- `{status}`: {count}" for status, count in sorted(reviews.items()))
    lines.append("")
    for heading, key in (("Excluded pages", "excluded"), ("Failed pages", "failed")):
        items = acquisition.get(key, [])
        lines.extend([f"## {heading}", ""])
        if not items:
            lines.append("None.")
        else:
            lines.extend(["| Title | Reason | Detail |", "| --- | --- | --- |"])
            lines.extend(f"| {item['title']} | `{item['reason']}` | {item['detail']} |" for item in items)
        lines.append("")
    others = acquisition.get("other_namespaces", [])
    lines.extend(["## Other namespaces (counted, not preserved)", ""])
    if others:
        lines.extend(["| Namespace | Name | Non-redirect pages |", "| --- | --- | --- |"])
        lines.extend(f"| {item['namespace']} | {item['name']} | {item['pages']} |" for item in others)
    else:
        lines.append("Not recorded.")
    lines.append("")
    counts: dict[str, list[int]] = {}
    for page in pages:
        for category in page.get("categories", []):
            entry = counts.setdefault(category, [0, 0])
            entry[0] += 1
            entry[1] += page["content_bytes"]
    uncategorized = sum(1 for page in pages if not page.get("categories"))
    lines.extend(["## Coverage by upstream category", ""])
    lines.extend(["| Category | Pages | Wikitext bytes |", "| --- | ---: | ---: |"])
    lines.extend(f"| {name} | {pages_} | {size:,} |" for name, (pages_, size) in sorted(counts.items()))
    lines.extend(["", f"Pages without any category: {uncategorized}", ""])
    redirects = acquisition.get("redirects", [])
    lines.extend(["## Redirects", ""])
    if redirects:
        preserved = {page["title"] for page in pages}
        dangling = [item for item in redirects if item["target"] not in preserved]
        lines.append(
            f"{len(redirects)} redirect titles map to targets; {len(dangling)} point outside the preserved pages."
        )
        lines.extend(["", "| Redirect | Target |", "| --- | --- |"])
        lines.extend(f"| {item['title']} | {item['target'] or '(unresolved)'} |" for item in redirects)
    else:
        lines.append("None recorded.")
    return "\n".join(lines) + "\n"


def command_report(args: argparse.Namespace) -> int:
    text = render_report(Path(args.directory))
    if args.output:
        Path(args.output).write_text(text, "utf-8")
    else:
        sys.stdout.write(text)
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
    fetch.add_argument(
        "--all",
        action="store_true",
        help="discover and acquire every main-namespace page (inclusion rules vasp-wiki-main-namespace-1)",
    )
    fetch.add_argument("--delay", type=float, default=DEFAULT_REQUEST_DELAY_SECONDS, help="seconds between requests")
    fetch.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="retries for transport failures")
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
    review.add_argument("--page-reviewed-file", action="append", metavar="FILE", help="page IDs, one per line")
    review.add_argument("--page-exception", action="append", nargs=2, metavar=("ID", "NOTE"))

    release = commands.add_parser("release", help="version and verify a reviewed staging corpus")
    release.add_argument("--staging", required=True)
    release.add_argument("--corpus-version", required=True)
    release.add_argument("--exclude-excepted", action="store_true", help="exclude license-excepted pages explicitly")
    release.add_argument("--allow-incomplete", action="store_true", help="release a corpus labelled incomplete")

    scan = commands.add_parser("scan-licensing", help="flag pages whose text may carry a licensing exception")
    scan.add_argument("--staging", required=True)
    scan.add_argument("--write-unflagged", metavar="FILE", help="write the IDs of unflagged pages to FILE")

    report = commands.add_parser("report", help="render the acquisition report for a corpus directory")
    report.add_argument("directory")
    report.add_argument("--output", metavar="FILE")

    upstream = commands.add_parser("verify-upstream", help="re-fetch every pinned revision and compare bytes")
    upstream.add_argument("directory")
    upstream.add_argument("--delay", type=float, default=DEFAULT_REQUEST_DELAY_SECONDS)
    upstream.add_argument("--retries", type=int, default=DEFAULT_RETRIES)

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
        if args.command == "scan-licensing":
            return command_scan_licensing(args)
        if args.command == "report":
            return command_report(args)
        if args.command == "verify-upstream":
            return command_verify_upstream(args, fetcher or https_fetch)
        return command_verify(args)
    except (FetchError, corpus.CorpusError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
