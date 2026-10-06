"""Offline loader and validator for the frozen VASP Wiki reference corpus.

The corpus is third-party material (verbatim MediaWiki wikitext of pinned VASP
Wiki revisions) shipped as package data under
``bmd_agent/reference_corpus/vasp_wiki``. This module is the only BMD Agent code
that reads it, and it only reads it:

* no network, subprocess, dynamic import, ``eval``/``exec``, YAML or pickle;
* package data is reached through ``importlib.resources``;
* wikitext is returned as an inert ``str`` and is never parsed, rendered,
  template-expanded or executed;
* nothing here prints, so no new terminal-output path exists.

``load_vasp_wiki_corpus()`` never raises. It returns either a frozen
``VaspWikiCorpus`` or a ``CorpusUnavailable`` explaining why the corpus cannot
be used. The corpus fails closed: any integrity, schema, identity or licensing
inconsistency makes it unavailable instead of partially loaded.

The pure helpers that define the corpus format (canonical manifest bytes,
identity digest, NOTICE and SHA256SUMS rendering, release-ledger rules) are
also used by the maintainer-only tool ``tools/vasp_reference/fetch_corpus.py``
so that both sides share one definition. This module never imports that tool.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from importlib import resources
from importlib.resources.abc import Traversable
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit


MANIFEST_SCHEMA = "bmd_agent.vasp_wiki_corpus"
RELEASES_SCHEMA = "bmd_agent.vasp_wiki_corpus_releases"
SCHEMA_VERSION = 1

KIND_UNPOPULATED = "unpopulated"
KIND_AUTHORITATIVE = "authoritative"
KIND_SYNTHETIC_FIXTURE = "synthetic_fixture"

MANIFEST_FILE = "manifest.json"
RELEASES_FILE = "RELEASES.json"
NOTICE_FILE = "NOTICE"
README_FILE = "README.md"
SHA256SUMS_FILE = "SHA256SUMS"
PAGES_DIRECTORY = "pages"

UNPOPULATED_CORPUS_VERSION = "0.0.0"

MAX_PAGES = 64
MAX_PAGE_BYTES = 262_144
MAX_TOTAL_PAGE_BYTES = 2_097_152
MAX_MANIFEST_BYTES = 1_048_576
MAX_AUXILIARY_FILE_BYTES = 131_072
MAX_URL_LENGTH = 2_048

LICENSE_REVIEW_PENDING = "pending"
LICENSE_REVIEW_APPROVED = "approved"
PAGE_REVIEW_PENDING = "pending"
PAGE_REVIEW_NO_EXCEPTIONS = "reviewed_no_exceptions"
PAGE_REVIEW_EXCEPTION_FOUND = "exception_found"
NONE_DECLARED = "none_declared"

# Closed set of reasons a corpus is unavailable.
UNAVAILABLE_MISSING = "missing"
UNAVAILABLE_UNPOPULATED = "unpopulated"
UNAVAILABLE_KIND_NOT_PERMITTED = "kind_not_permitted"
UNAVAILABLE_INVALID = "invalid"
UNAVAILABLE_INTEGRITY = "integrity"
UNAVAILABLE_LICENSE = "license"
UNAVAILABLE_INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class CorpusPolicy:
    """What a caller is prepared to accept as a corpus.

    ``license_text_sha256`` pins the exact bytes of the shipped license text.
    ``None`` means the digest has not been verified and pinned yet, and a
    populated corpus is then refused.
    """

    corpus_kind: str
    allowed_hosts: frozenset[str]
    allowed_license_identifiers: frozenset[str]
    required_rights_text_fragment: str
    license_text_file: str
    license_text_sha256: str | None


# The only policy used when no policy is passed. It accepts GFDL 1.2 VASP Wiki
# material served from the official VASP hosts only.
#
# license_text_sha256 is deliberately unpinned: the verbatim GFDL 1.2 text has
# not been obtained from its authoritative source and verified yet. It must be
# pinned, in a reviewed PR, before any populated corpus can load.
AUTHORITATIVE_POLICY = CorpusPolicy(
    corpus_kind=KIND_AUTHORITATIVE,
    allowed_hosts=frozenset({"vasp.at", "www.vasp.at"}),
    allowed_license_identifiers=frozenset({"GFDL-1.2-only", "GFDL-1.2-or-later"}),
    required_rights_text_fragment="GNU Free Documentation License 1.2",
    license_text_file="COPYING.GFDL-1.2.txt",
    license_text_sha256=None,
)


class CorpusError(ValueError):
    """A corpus failed validation. ``reason`` is one of the unavailable reasons."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class CorpusIntegrityError(CorpusError):
    """Preserved page bytes no longer match their recorded identity."""

    def __init__(self, detail: str) -> None:
        super().__init__(UNAVAILABLE_INTEGRITY, detail)


@dataclass(frozen=True)
class CorpusUnavailable:
    """The corpus cannot be used; ``reason`` is from a closed set."""

    reason: str
    detail: str


@dataclass(frozen=True)
class UpstreamSite:
    api_endpoint: str
    site_name: str
    generator: str
    server: str
    script_path: str
    article_path: str
    rights_url: str
    rights_text: str


@dataclass(frozen=True)
class CorpusLicense:
    identifier: str
    site_notice_as_displayed: str
    invariant_sections: str
    cover_texts: str
    license_text_file: str
    license_text_source_url: str
    license_text_sha256: str
    reviewer: str
    reviewed_on: str


@dataclass(frozen=True)
class PageLicenseReview:
    status: str
    reviewer: str
    reviewed_on: str
    notes: str | None


@dataclass(frozen=True)
class PreservedPage:
    """One preserved revision. The wikitext is read lazily and re-verified."""

    id: str
    requested_title: str
    title: str
    redirected_from: str | None
    page_id: int
    namespace: int
    revision_id: int
    parent_revision_id: int | None
    revision_timestamp: str
    retrieved_at: str
    canonical_url: str
    permalink: str
    history_url: str
    content_model: str
    content_file: str
    content_bytes: int
    content_sha256: str
    upstream_sha1: str
    license_review: PageLicenseReview
    _source: Traversable = field(repr=False, compare=False)

    def read_wikitext(self) -> str:
        """Return the verbatim wikitext as inert text.

        The bytes are re-read and re-verified on every call; a mismatch raises
        ``CorpusIntegrityError``. The result is never interpreted as markup.
        """

        data = _read_bounded(self._source, MAX_PAGE_BYTES, self.content_file)
        return _verified_page_text(
            data,
            content_file=self.content_file,
            content_bytes=self.content_bytes,
            content_sha256=self.content_sha256,
            upstream_sha1=self.upstream_sha1,
        )


@dataclass(frozen=True)
class VaspWikiCorpus:
    corpus_kind: str
    corpus_version: str
    corpus_digest: str
    upstream: UpstreamSite
    license: CorpusLicense
    pages: tuple[PreservedPage, ...]

    def page(self, page_id: str) -> PreservedPage | None:
        return next((page for page in self.pages if page.id == page_id), None)


# ---------------------------------------------------------------------------
# Character policy
# ---------------------------------------------------------------------------

# Closed policy for preserved page text. Allowed: TAB and LF plus every other
# Unicode scalar value except the ones below. Disallowed content is never
# rewritten; acquisition and loading refuse it for human review.
_DISALLOWED_TEXT_RE = re.compile(
    "["
    "\x00-\x08\x0b-\x1f"  # C0 controls except TAB and LF (includes CR, ESC, NUL)
    "\x7f-\x9f"  # DEL and C1 controls (includes CSI)
    "؜"  # Arabic letter mark
    "​-‏"  # zero-width space/non-joiner/joiner, LRM, RLM
    "  "  # line and paragraph separators
    "‪-‮"  # bidi embeddings and overrides
    "⁦-⁩"  # bidi isolates
    "﻿"  # byte-order mark / zero-width no-break space
    "﷐-﷯￾￿"  # BMP noncharacters
    "\U000e0000-\U000e007f"  # tag characters (invisible text smuggling)
    "]"
)
_SINGLE_LINE_DISALLOWED = re.compile("[\t\n]")


def find_disallowed_character(text: str) -> tuple[int, int] | None:
    """Return ``(offset, code point)`` of the first disallowed character, if any."""

    match = _DISALLOWED_TEXT_RE.search(text)
    if match is not None:
        return match.start(), ord(match.group())
    for offset, character in enumerate(text):
        code_point = ord(character)
        if code_point > 0xFFFF and (code_point & 0xFFFE) == 0xFFFE:
            return offset, code_point
    return None


# ---------------------------------------------------------------------------
# Format helpers shared with the maintainer tool
# ---------------------------------------------------------------------------

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})$")
_TIMESTAMP_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_PAGE_ID_RE = re.compile(r"^vasp\.wiki\.([a-z0-9]+(?:_[a-z0-9]+)*)$")
_CONTENT_FILE_RE = re.compile(r"^pages/([a-z0-9]+(?:_[a-z0-9]+)*)\.r([1-9][0-9]{0,11})\.wiki$")
_AUX_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SCRIPT_PATH_RE = re.compile(r"^/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+\.php$")
_SUMS_LINE_RE = re.compile(r"^([0-9a-f]{64})  (\S+)$")
_MAX_SLUG_LENGTH = 80


def title_slug(title: str) -> str:
    """Return the filesystem-safe slug for a requested title (may be empty)."""

    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:_MAX_SLUG_LENGTH].strip("_")


def page_id_for_slug(slug: str) -> str:
    return f"vasp.wiki.{slug}"


def content_file_for(slug: str, revision_id: int) -> str:
    return f"{PAGES_DIRECTORY}/{slug}.r{revision_id}.wiki"


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """The single byte serialization a manifest may have on disk."""

    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return (text + "\n").encode("ascii")


def canonical_releases_bytes(releases: Mapping[str, Any]) -> bytes:
    return canonical_manifest_bytes(releases)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unpopulated_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "corpus_kind": KIND_UNPOPULATED,
        "corpus_version": UNPOPULATED_CORPUS_VERSION,
        "corpus_digest": "",
        "assembly": None,
        "upstream": None,
        "license": None,
        "pages": [],
    }
    manifest["corpus_digest"] = compute_corpus_digest(manifest)
    return manifest


def unpopulated_release_anchor() -> dict[str, Any]:
    """The mandatory first entry of every release ledger."""

    return {
        "corpus_version": UNPOPULATED_CORPUS_VERSION,
        "corpus_kind": KIND_UNPOPULATED,
        "corpus_digest": unpopulated_manifest()["corpus_digest"],
        "previous_corpus_digest": None,
    }


def compute_corpus_digest(manifest: Mapping[str, Any]) -> str:
    """Identity of the reviewed corpus content.

    The digest covers everything that defines what is preserved and under
    which license: page identity, revision identity, content hashes, URLs,
    captured site rights, reviewed license fields and review outcomes. It
    deliberately excludes provenance that does not change content (retrieval
    and assembly timestamps, updater version, reviewer names and dates, the
    MediaWiki generator string) and the version label itself, which the
    release ledger binds to the digest.
    """

    upstream = manifest.get("upstream")
    license_block = manifest.get("license")
    view = {
        "schema": manifest.get("schema"),
        "schema_version": manifest.get("schema_version"),
        "corpus_kind": manifest.get("corpus_kind"),
        "upstream": None
        if upstream is None
        else {
            key: upstream.get(key)
            for key in (
                "api_endpoint",
                "site_name",
                "server",
                "script_path",
                "article_path",
                "rights_url",
                "rights_text",
            )
        },
        "license": None
        if license_block is None
        else {
            **{
                key: license_block.get(key)
                for key in (
                    "identifier",
                    "site_notice_as_displayed",
                    "invariant_sections",
                    "cover_texts",
                    "license_text_file",
                    "license_text_source_url",
                    "license_text_sha256",
                )
            },
            "review_status": (license_block.get("review") or {}).get("status"),
        },
        "pages": [
            {
                **{
                    key: page.get(key)
                    for key in (
                        "id",
                        "requested_title",
                        "title",
                        "redirected_from",
                        "page_id",
                        "namespace",
                        "revision_id",
                        "parent_revision_id",
                        "revision_timestamp",
                        "canonical_url",
                        "permalink",
                        "history_url",
                        "content_model",
                        "content_file",
                        "content_bytes",
                        "content_sha256",
                        "upstream_sha1",
                    )
                },
                "license_review_status": (page.get("license_review") or {}).get("status"),
            }
            for page in manifest.get("pages") or ()
        ],
    }
    encoded = json.dumps(
        view,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return "sha256:" + sha256_hex(encoded)


def render_sha256sums(files: Mapping[str, bytes]) -> bytes:
    """SHA256SUMS over every corpus file except itself, sorted by path."""

    lines = [f"{sha256_hex(files[path])}  {path}\n" for path in sorted(files)]
    return "".join(lines).encode("ascii")


def render_notice(manifest: Mapping[str, Any]) -> bytes:
    """Deterministic attribution notice generated from the manifest."""

    lines = [
        "VASP Wiki reference corpus - third-party material notice",
        "=========================================================",
        "",
        f"Corpus kind: {manifest['corpus_kind']}",
        f"Corpus version: {manifest['corpus_version']}",
        f"Corpus identity: {manifest['corpus_digest']}",
        "",
    ]
    if manifest["corpus_kind"] == KIND_UNPOPULATED:
        lines.extend(
            [
                "This corpus is unpopulated. It contains no VASP Wiki material and",
                "no third-party page content.",
            ]
        )
        return ("\n".join(lines) + "\n").encode("utf-8")

    upstream = manifest["upstream"]
    license_block = manifest["license"]
    if manifest["corpus_kind"] == KIND_SYNTHETIC_FIXTURE:
        lines.extend(
            [
                "SYNTHETIC TEST FIXTURE. This corpus contains no VASP Wiki material;",
                "its pages, site details and license text are invented placeholders.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The files under pages/ are verbatim copies of MediaWiki wikitext at",
                "the pinned upstream revisions listed below. They are third-party",
                "material: they are not authored by BMD and are not covered by the",
                "BMD Agent MIT License. They are distributed unmodified under the",
                "license recorded here; its full text is in the license text file.",
                "",
            ]
        )
    lines.extend(
        [
            f"Site name (MediaWiki API): {upstream['site_name']}",
            f"Site server: {upstream['server']}",
            f"Site rights text (MediaWiki API): {upstream['rights_text']}",
            f"Site rights URL (MediaWiki API): {upstream['rights_url'] or '(none reported)'}",
            "License notice displayed by the site (maintainer-recorded): "
            f"{_display_optional(license_block['site_notice_as_displayed'])}",
            f"License identifier (maintainer-reviewed): {_display_optional(license_block['identifier'])}",
            f"Invariant Sections: {_display_optional(license_block['invariant_sections'])}",
            f"Front-Cover and Back-Cover Texts: {_display_optional(license_block['cover_texts'])}",
            f"License text file: {license_block['license_text_file']}",
            f"License text source: {license_block['license_text_source_url']}",
            f"License text SHA-256: {license_block['license_text_sha256']}",
            f"License review: {license_block['review']['status']}",
            "",
            "Pages:",
        ]
    )
    for page in manifest["pages"]:
        lines.extend(
            [
                f"- {page['title']} (requested as {page['requested_title']})",
                f"  redirected from: {page['redirected_from'] or '(not a redirect)'}",
                f"  page ID {page['page_id']}, revision {page['revision_id']} ({page['revision_timestamp']})",
                f"  retrieved: {page['retrieved_at']}",
                f"  canonical URL: {page['canonical_url']}",
                f"  this revision: {page['permalink']}",
                f"  history and authors: {page['history_url']}",
                f"  file: {page['content_file']}",
                f"  SHA-256: {page['content_sha256']}",
                f"  MediaWiki SHA-1: {page['upstream_sha1']}",
                f"  license review: {page['license_review']['status']}",
            ]
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _display_optional(value: Any) -> str:
    return "(not yet reviewed)" if value is None else str(value)


def parse_strict_json(data: bytes, *, label: str) -> Any:
    """Parse canonical ASCII JSON, rejecting duplicate keys and non-finite numbers."""

    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} is not ASCII JSON") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except CorpusError:
        raise
    except (ValueError, RecursionError) as exc:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} is not valid JSON") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CorpusError(UNAVAILABLE_INVALID, "JSON object has a duplicate key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise CorpusError(UNAVAILABLE_INVALID, "JSON contains a non-finite number")


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------

_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "corpus_kind",
        "corpus_version",
        "corpus_digest",
        "assembly",
        "upstream",
        "license",
        "pages",
    }
)
_ASSEMBLY_KEYS = frozenset({"assembled_at", "updater"})
_UPDATER_KEYS = frozenset({"name", "version"})
_UPSTREAM_KEYS = frozenset(
    {
        "api_endpoint",
        "site_name",
        "generator",
        "server",
        "script_path",
        "article_path",
        "rights_url",
        "rights_text",
    }
)
_LICENSE_KEYS = frozenset(
    {
        "identifier",
        "site_notice_as_displayed",
        "invariant_sections",
        "cover_texts",
        "license_text_file",
        "license_text_source_url",
        "license_text_sha256",
        "review",
    }
)
_LICENSE_REVIEW_KEYS = frozenset({"status", "reviewer", "reviewed_on"})
_PAGE_KEYS = frozenset(
    {
        "id",
        "requested_title",
        "title",
        "redirected_from",
        "page_id",
        "namespace",
        "revision_id",
        "parent_revision_id",
        "revision_timestamp",
        "retrieved_at",
        "canonical_url",
        "permalink",
        "history_url",
        "content_model",
        "content_file",
        "content_bytes",
        "content_sha256",
        "upstream_sha1",
        "license_review",
    }
)
_PAGE_REVIEW_KEYS = frozenset({"status", "reviewer", "reviewed_on", "notes"})


def validate_manifest(
    manifest: Any,
    policy: CorpusPolicy,
    *,
    require_reviewed: bool,
) -> None:
    """Validate one parsed manifest against schema v1 and ``policy``.

    With ``require_reviewed`` false (maintainer staging), pending reviews are
    allowed; everything else is still enforced.
    """

    record = _object(manifest, _MANIFEST_KEYS, "manifest")
    if record["schema"] != MANIFEST_SCHEMA:
        raise CorpusError(UNAVAILABLE_INVALID, "manifest schema is not supported")
    if _int(record["schema_version"], "manifest schema_version") != SCHEMA_VERSION:
        raise CorpusError(UNAVAILABLE_INVALID, "manifest schema_version is not supported")
    kind = record["corpus_kind"]
    if kind not in (KIND_UNPOPULATED, KIND_AUTHORITATIVE, KIND_SYNTHETIC_FIXTURE):
        raise CorpusError(UNAVAILABLE_INVALID, "manifest corpus_kind is not recognized")
    version = _string(record["corpus_version"], "corpus_version", max_length=32)
    if not _VERSION_RE.fullmatch(version):
        raise CorpusError(UNAVAILABLE_INVALID, "corpus_version is not MAJOR.MINOR.PATCH")
    if not _DIGEST_RE.fullmatch(_string(record["corpus_digest"], "corpus_digest", max_length=80)):
        raise CorpusError(UNAVAILABLE_INVALID, "corpus_digest is malformed")
    pages = record["pages"]
    if not isinstance(pages, list):
        raise CorpusError(UNAVAILABLE_INVALID, "manifest pages must be a list")

    if kind == KIND_UNPOPULATED:
        if version != UNPOPULATED_CORPUS_VERSION:
            raise CorpusError(UNAVAILABLE_INVALID, "an unpopulated corpus must be version 0.0.0")
        if record["assembly"] is not None or record["upstream"] is not None or record["license"] is not None:
            raise CorpusError(UNAVAILABLE_INVALID, "an unpopulated corpus records no upstream or license")
        if pages:
            raise CorpusError(UNAVAILABLE_INVALID, "an unpopulated corpus contains no pages")
        return

    if kind != policy.corpus_kind:
        raise CorpusError(UNAVAILABLE_KIND_NOT_PERMITTED, "corpus kind is not permitted by this policy")
    if version == UNPOPULATED_CORPUS_VERSION:
        raise CorpusError(UNAVAILABLE_INVALID, "a populated corpus cannot be version 0.0.0")
    _validate_assembly(record["assembly"])
    upstream = _validate_upstream(record["upstream"], policy)
    _validate_license(record["license"], policy, require_reviewed=require_reviewed)
    if not pages:
        raise CorpusError(UNAVAILABLE_INVALID, "a populated corpus must contain pages")
    if len(pages) > MAX_PAGES:
        raise CorpusError(UNAVAILABLE_INVALID, "corpus contains too many pages")

    seen: dict[str, set[Any]] = {
        "id": set(),
        "requested_title": set(),
        "title": set(),
        "page_id": set(),
        "revision_id": set(),
        "content_file": set(),
    }
    ids: list[str] = []
    for index, page in enumerate(pages):
        _validate_page(page, index, upstream, policy, require_reviewed=require_reviewed)
        for key, values in seen.items():
            if page[key] in values:
                raise CorpusError(UNAVAILABLE_INVALID, f"pages contain a duplicate {key}")
            values.add(page[key])
        ids.append(page["id"])
    if ids != sorted(ids):
        raise CorpusError(UNAVAILABLE_INVALID, "pages must be sorted by id")
    total = sum(page["content_bytes"] for page in pages)
    if total > MAX_TOTAL_PAGE_BYTES:
        raise CorpusError(UNAVAILABLE_INVALID, "corpus pages exceed the total size limit")


def _validate_assembly(value: Any) -> None:
    assembly = _object(value, _ASSEMBLY_KEYS, "assembly")
    _timestamp(assembly["assembled_at"], "assembly assembled_at")
    updater = _object(assembly["updater"], _UPDATER_KEYS, "assembly updater")
    _text(updater["name"], "updater name", max_length=200)
    _text(updater["version"], "updater version", max_length=40)


def _validate_upstream(value: Any, policy: CorpusPolicy) -> Mapping[str, Any]:
    upstream = _object(value, _UPSTREAM_KEYS, "upstream")
    endpoint = _url(upstream["api_endpoint"], "api_endpoint", allowed_hosts=policy.allowed_hosts)
    endpoint_parts = urlsplit(endpoint)
    if endpoint_parts.query or endpoint_parts.fragment or not endpoint_parts.path.endswith(".php"):
        raise CorpusError(UNAVAILABLE_INVALID, "api_endpoint must be a bare .php endpoint")
    _text(upstream["site_name"], "site_name", max_length=200)
    generator = _text(upstream["generator"], "generator", max_length=200)
    if not generator.startswith("MediaWiki "):
        raise CorpusError(UNAVAILABLE_INVALID, "generator must identify MediaWiki")
    server = _url(upstream["server"], "server", allowed_hosts=policy.allowed_hosts)
    server_parts = urlsplit(server)
    if server_parts.path or server_parts.query or server_parts.fragment:
        raise CorpusError(UNAVAILABLE_INVALID, "server must be scheme and host only")
    script_path = _text(upstream["script_path"], "script_path", max_length=200)
    if not _SCRIPT_PATH_RE.fullmatch(script_path):
        raise CorpusError(UNAVAILABLE_INVALID, "script_path is malformed")
    article_path = _text(upstream["article_path"], "article_path", max_length=200)
    if not article_path.startswith("/") or "$1" not in article_path:
        raise CorpusError(UNAVAILABLE_INVALID, "article_path is malformed")
    rights_url = upstream["rights_url"]
    if rights_url != "":
        _url(rights_url, "rights_url", allowed_hosts=None, allow_http=True)
    rights_text = _text(upstream["rights_text"], "rights_text", max_length=500)
    if policy.required_rights_text_fragment not in rights_text:
        raise CorpusError(UNAVAILABLE_LICENSE, "captured site rights do not match the accepted license")
    return upstream


def _validate_license(value: Any, policy: CorpusPolicy, *, require_reviewed: bool) -> None:
    license_block = _object(value, _LICENSE_KEYS, "license")
    identifier = license_block["identifier"]
    if identifier is not None and identifier not in policy.allowed_license_identifiers:
        raise CorpusError(UNAVAILABLE_LICENSE, "license identifier is not accepted by this policy")
    notice = license_block["site_notice_as_displayed"]
    if notice is not None:
        _text(notice, "site_notice_as_displayed", max_length=500)
    for key in ("invariant_sections", "cover_texts"):
        if license_block[key] not in (None, NONE_DECLARED):
            raise CorpusError(UNAVAILABLE_LICENSE, f"license {key} must be {NONE_DECLARED}")
    if license_block["license_text_file"] != policy.license_text_file:
        raise CorpusError(UNAVAILABLE_LICENSE, "license text file is not the one this policy accepts")
    _url(license_block["license_text_source_url"], "license_text_source_url", allowed_hosts=None)
    license_sha = _string(license_block["license_text_sha256"], "license_text_sha256", max_length=64)
    if not _SHA256_RE.fullmatch(license_sha):
        raise CorpusError(UNAVAILABLE_INVALID, "license_text_sha256 is malformed")
    review = _object(license_block["review"], _LICENSE_REVIEW_KEYS, "license review")
    status = review["status"]
    if status not in (LICENSE_REVIEW_PENDING, LICENSE_REVIEW_APPROVED):
        raise CorpusError(UNAVAILABLE_INVALID, "license review status is not recognized")
    _optional_reviewer(review["reviewer"], review["reviewed_on"], "license review")
    if status == LICENSE_REVIEW_APPROVED:
        if review["reviewer"] is None or review["reviewed_on"] is None:
            raise CorpusError(UNAVAILABLE_LICENSE, "approved license review must name reviewer and date")
        if None in (identifier, notice, license_block["invariant_sections"], license_block["cover_texts"]):
            raise CorpusError(UNAVAILABLE_LICENSE, "approved license review must record every license field")
    elif require_reviewed:
        raise CorpusError(UNAVAILABLE_LICENSE, "license review is not approved")


def _validate_page(
    value: Any,
    index: int,
    upstream: Mapping[str, Any],
    policy: CorpusPolicy,
    *,
    require_reviewed: bool,
) -> None:
    label = f"page {index}"
    page = _object(value, _PAGE_KEYS, label)
    page_id = _string(page["id"], f"{label} id", max_length=100)
    id_match = _PAGE_ID_RE.fullmatch(page_id)
    if id_match is None:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} id is malformed")
    slug = id_match.group(1)
    requested = _text(page["requested_title"], f"{label} requested_title", max_length=255)
    if title_slug(requested) != slug:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} id does not derive from requested_title")
    _text(page["title"], f"{label} title", max_length=255)
    if page["redirected_from"] is not None:
        _text(page["redirected_from"], f"{label} redirected_from", max_length=255)
    _positive_int(page["page_id"], f"{label} page_id")
    if _int(page["namespace"], f"{label} namespace") != 0:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} must be in the main namespace")
    revision_id = _positive_int(page["revision_id"], f"{label} revision_id")
    parent = page["parent_revision_id"]
    if parent is not None and (_positive_int(parent, f"{label} parent_revision_id") >= revision_id):
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} parent revision must precede its revision")
    _timestamp(page["revision_timestamp"], f"{label} revision_timestamp")
    _timestamp(page["retrieved_at"], f"{label} retrieved_at")
    canonical = _url(page["canonical_url"], f"{label} canonical_url", allowed_hosts=policy.allowed_hosts)
    if "oldid" in parse_qs(urlsplit(canonical).query):
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} canonical_url must not pin a revision")
    script_url = upstream["server"] + upstream["script_path"]
    permalink = _url(page["permalink"], f"{label} permalink", allowed_hosts=policy.allowed_hosts)
    if not permalink.startswith(script_url + "?") or _query_values(permalink, "oldid") != [str(revision_id)]:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} permalink does not pin its revision")
    history = _url(page["history_url"], f"{label} history_url", allowed_hosts=policy.allowed_hosts)
    if not history.startswith(script_url + "?") or _query_values(history, "action") != ["history"]:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} history_url is malformed")
    if page["content_model"] != "wikitext":
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} content_model must be wikitext")
    if page["content_file"] != content_file_for(slug, revision_id):
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} content_file does not match its id and revision")
    size = _int(page["content_bytes"], f"{label} content_bytes")
    if size < 0 or size > MAX_PAGE_BYTES:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} content_bytes is outside the page size limit")
    if not _SHA256_RE.fullmatch(_string(page["content_sha256"], f"{label} content_sha256", max_length=64)):
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} content_sha256 is malformed")
    if not _SHA1_RE.fullmatch(_string(page["upstream_sha1"], f"{label} upstream_sha1", max_length=40)):
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} upstream_sha1 is malformed")
    review = _object(page["license_review"], _PAGE_REVIEW_KEYS, f"{label} license_review")
    status = review["status"]
    if status not in (PAGE_REVIEW_PENDING, PAGE_REVIEW_NO_EXCEPTIONS, PAGE_REVIEW_EXCEPTION_FOUND):
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} license review status is not recognized")
    _optional_reviewer(review["reviewer"], review["reviewed_on"], f"{label} license review")
    if review["notes"] is not None:
        _text(review["notes"], f"{label} license review notes", max_length=500)
    if status != PAGE_REVIEW_PENDING and (review["reviewer"] is None or review["reviewed_on"] is None):
        raise CorpusError(UNAVAILABLE_LICENSE, f"{label} license review must name reviewer and date")
    if status == PAGE_REVIEW_EXCEPTION_FOUND and review["notes"] is None:
        raise CorpusError(UNAVAILABLE_LICENSE, f"{label} license exception must be described")
    if require_reviewed and status != PAGE_REVIEW_NO_EXCEPTIONS:
        raise CorpusError(
            UNAVAILABLE_LICENSE,
            f"{label} is not reviewed as free of license exceptions",
        )


def _optional_reviewer(reviewer: Any, reviewed_on: Any, label: str) -> None:
    if reviewer is not None:
        _text(reviewer, f"{label} reviewer", max_length=100)
    if reviewed_on is not None:
        _date(reviewed_on, f"{label} reviewed_on")


def _object(value: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} must be an object")
    present = set(value)
    if present - keys:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} has unknown fields")
    if keys - present:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} is missing required fields")
    return value


def _string(value: Any, label: str, *, max_length: int) -> str:
    if type(value) is not str or not value or len(value) > max_length:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} must be a non-empty string")
    return value


def _text(value: Any, label: str, *, max_length: int) -> str:
    text = _string(value, label, max_length=max_length)
    if find_disallowed_character(text) is not None or _SINGLE_LINE_DISALLOWED.search(text):
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} contains disallowed characters")
    if text != text.strip():
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} has surrounding whitespace")
    return text


def _int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} must be an integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    number = _int(value, label)
    if number <= 0 or number >= 10**12:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} must be a positive revision-style integer")
    return number


def _timestamp(value: Any, label: str) -> str:
    text = _string(value, label, max_length=20)
    if not _TIMESTAMP_RE.fullmatch(text):
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} must be a UTC timestamp")
    return text


def _date(value: Any, label: str) -> str:
    text = _string(value, label, max_length=10)
    if not _DATE_RE.fullmatch(text):
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} must be a date")
    return text


def _url(
    value: Any,
    label: str,
    *,
    allowed_hosts: frozenset[str] | None,
    allow_http: bool = False,
) -> str:
    url = _text(value, label, max_length=MAX_URL_LENGTH)
    if any(character.isspace() for character in url) or not url.isascii():
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} must be an ASCII URL without spaces")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} is not a valid URL") from exc
    schemes = ("https", "http") if allow_http else ("https",)
    if parts.scheme not in schemes:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} must use HTTPS")
    if parts.username is not None or parts.password is not None or port is not None:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} must not carry credentials or a port")
    host = parts.hostname or ""
    if not host or parts.netloc != host:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} host is malformed")
    if allowed_hosts is not None and host not in allowed_hosts:
        raise CorpusError(UNAVAILABLE_INVALID, f"{label} host is not an allowed corpus host")
    return url


def _query_values(url: str, key: str) -> list[str]:
    return parse_qs(urlsplit(url).query, keep_blank_values=True).get(key, [])


# ---------------------------------------------------------------------------
# Release ledger
# ---------------------------------------------------------------------------

_RELEASES_KEYS = frozenset({"schema", "schema_version", "releases"})
_RELEASE_KEYS = frozenset({"corpus_version", "corpus_kind", "corpus_digest", "previous_corpus_digest"})


def validate_releases(releases: Any) -> tuple[Mapping[str, Any], ...]:
    """Validate the append-only release ledger and return its entries."""

    record = _object(releases, _RELEASES_KEYS, "release ledger")
    if record["schema"] != RELEASES_SCHEMA:
        raise CorpusError(UNAVAILABLE_INVALID, "release ledger schema is not supported")
    if _int(record["schema_version"], "release ledger schema_version") != SCHEMA_VERSION:
        raise CorpusError(UNAVAILABLE_INVALID, "release ledger schema_version is not supported")
    entries = record["releases"]
    if not isinstance(entries, list) or not entries:
        raise CorpusError(UNAVAILABLE_INVALID, "release ledger must list releases")
    if entries[0] != unpopulated_release_anchor():
        raise CorpusError(UNAVAILABLE_INVALID, "release ledger must start at the unpopulated anchor")
    previous_version: tuple[int, int, int] | None = None
    previous_digest: str | None = None
    digests: set[str] = set()
    for entry in entries:
        item = _object(entry, _RELEASE_KEYS, "release")
        version = _string(item["corpus_version"], "release corpus_version", max_length=32)
        match = _VERSION_RE.fullmatch(version)
        if match is None:
            raise CorpusError(UNAVAILABLE_INVALID, "release corpus_version is malformed")
        if item["corpus_kind"] not in (KIND_UNPOPULATED, KIND_AUTHORITATIVE, KIND_SYNTHETIC_FIXTURE):
            raise CorpusError(UNAVAILABLE_INVALID, "release corpus_kind is not recognized")
        digest = _string(item["corpus_digest"], "release corpus_digest", max_length=80)
        if not _DIGEST_RE.fullmatch(digest):
            raise CorpusError(UNAVAILABLE_INVALID, "release corpus_digest is malformed")
        numeric = tuple(int(part) for part in match.groups())
        if previous_version is not None and numeric <= previous_version:
            raise CorpusError(UNAVAILABLE_INVALID, "release versions must strictly increase")
        if item["previous_corpus_digest"] != previous_digest:
            raise CorpusError(UNAVAILABLE_INVALID, "release ledger chain is broken")
        if digest in digests:
            raise CorpusError(UNAVAILABLE_INVALID, "release ledger reuses a corpus identity")
        digests.add(digest)
        previous_version = numeric  # type: ignore[assignment]
        previous_digest = digest
    return tuple(entries)


def releases_extend(base: Sequence[Mapping[str, Any]], current: Sequence[Mapping[str, Any]]) -> bool:
    """Whether ``current`` keeps every release in ``base`` unchanged, in order."""

    return len(current) >= len(base) and list(current[: len(base)]) == list(base)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def packaged_corpus_root() -> Traversable:
    return resources.files("bmd_agent.reference_corpus").joinpath("vasp_wiki")


def load_vasp_wiki_corpus(
    root: Traversable | Path | None = None,
    *,
    policy: CorpusPolicy = AUTHORITATIVE_POLICY,
) -> VaspWikiCorpus | CorpusUnavailable:
    """Load and fully verify a frozen corpus. Never raises."""

    try:
        corpus_root = packaged_corpus_root() if root is None else root
        return _load(corpus_root, policy)
    except CorpusError as exc:
        return CorpusUnavailable(exc.reason, exc.detail)
    except Exception as exc:  # fail closed without disturbing callers
        return CorpusUnavailable(UNAVAILABLE_INTERNAL_ERROR, f"corpus loading failed: {type(exc).__name__}")


def _load(root: Traversable, policy: CorpusPolicy) -> VaspWikiCorpus | CorpusUnavailable:
    try:
        root_present = root.is_dir()
    except OSError:
        root_present = False
    if not root_present or not root.joinpath(MANIFEST_FILE).is_file():
        return CorpusUnavailable(UNAVAILABLE_MISSING, "corpus directory or manifest is missing")

    listed = _verify_sha256sums(root)
    manifest_bytes = listed[MANIFEST_FILE]
    manifest = parse_strict_json(manifest_bytes, label=MANIFEST_FILE)
    if canonical_manifest_bytes(manifest) != manifest_bytes:
        raise CorpusError(UNAVAILABLE_INVALID, "manifest is not in canonical form")
    validate_manifest(manifest, policy, require_reviewed=True)
    if compute_corpus_digest(manifest) != manifest["corpus_digest"]:
        raise CorpusError(UNAVAILABLE_INTEGRITY, "corpus_digest does not match the corpus content")

    releases_bytes = listed.get(RELEASES_FILE)
    if releases_bytes is None:
        raise CorpusError(UNAVAILABLE_INTEGRITY, "release ledger is missing")
    releases = parse_strict_json(releases_bytes, label=RELEASES_FILE)
    if canonical_releases_bytes(releases) != releases_bytes:
        raise CorpusError(UNAVAILABLE_INVALID, "release ledger is not in canonical form")
    latest = validate_releases(releases)[-1]
    if (
        latest["corpus_version"] != manifest["corpus_version"]
        or latest["corpus_kind"] != manifest["corpus_kind"]
        or latest["corpus_digest"] != manifest["corpus_digest"]
    ):
        raise CorpusError(UNAVAILABLE_INTEGRITY, "manifest is not the latest recorded release")

    if listed.get(NOTICE_FILE) != render_notice(manifest):
        raise CorpusError(UNAVAILABLE_INTEGRITY, "NOTICE does not match the manifest")
    readme = listed.get(README_FILE)
    if readme is None:
        raise CorpusError(UNAVAILABLE_INTEGRITY, "corpus README is missing")
    _verified_text_file(readme, README_FILE)

    expected_files = {MANIFEST_FILE, RELEASES_FILE, NOTICE_FILE, README_FILE}
    if manifest["corpus_kind"] == KIND_UNPOPULATED:
        if set(listed) != expected_files:
            raise CorpusError(UNAVAILABLE_INTEGRITY, "unpopulated corpus contains unexpected files")
        return CorpusUnavailable(UNAVAILABLE_UNPOPULATED, "the corpus contains no preserved pages yet")

    license_block = manifest["license"]
    expected_files.add(license_block["license_text_file"])
    expected_files.update(page["content_file"] for page in manifest["pages"])
    if set(listed) != expected_files:
        raise CorpusError(UNAVAILABLE_INTEGRITY, "corpus files do not match the manifest")
    license_text = listed[license_block["license_text_file"]]
    if sha256_hex(license_text) != license_block["license_text_sha256"]:
        raise CorpusError(UNAVAILABLE_INTEGRITY, "license text does not match the manifest")
    if policy.license_text_sha256 is None:
        raise CorpusError(UNAVAILABLE_LICENSE, "the accepted license text digest has not been pinned")
    if license_block["license_text_sha256"] != policy.license_text_sha256:
        raise CorpusError(UNAVAILABLE_LICENSE, "license text is not the pinned accepted license text")

    pages: list[PreservedPage] = []
    for page in manifest["pages"]:
        source = _resolve_page_file(root, page["content_file"])
        _verified_page_text(
            listed[page["content_file"]],
            content_file=page["content_file"],
            content_bytes=page["content_bytes"],
            content_sha256=page["content_sha256"],
            upstream_sha1=page["upstream_sha1"],
        )
        review = page["license_review"]
        pages.append(
            PreservedPage(
                id=page["id"],
                requested_title=page["requested_title"],
                title=page["title"],
                redirected_from=page["redirected_from"],
                page_id=page["page_id"],
                namespace=page["namespace"],
                revision_id=page["revision_id"],
                parent_revision_id=page["parent_revision_id"],
                revision_timestamp=page["revision_timestamp"],
                retrieved_at=page["retrieved_at"],
                canonical_url=page["canonical_url"],
                permalink=page["permalink"],
                history_url=page["history_url"],
                content_model=page["content_model"],
                content_file=page["content_file"],
                content_bytes=page["content_bytes"],
                content_sha256=page["content_sha256"],
                upstream_sha1=page["upstream_sha1"],
                license_review=PageLicenseReview(
                    status=review["status"],
                    reviewer=review["reviewer"],
                    reviewed_on=review["reviewed_on"],
                    notes=review["notes"],
                ),
                _source=source,
            )
        )

    upstream = manifest["upstream"]
    review = license_block["review"]
    return VaspWikiCorpus(
        corpus_kind=manifest["corpus_kind"],
        corpus_version=manifest["corpus_version"],
        corpus_digest=manifest["corpus_digest"],
        upstream=UpstreamSite(**{key: upstream[key] for key in _UPSTREAM_KEYS}),
        license=CorpusLicense(
            identifier=license_block["identifier"],
            site_notice_as_displayed=license_block["site_notice_as_displayed"],
            invariant_sections=license_block["invariant_sections"],
            cover_texts=license_block["cover_texts"],
            license_text_file=license_block["license_text_file"],
            license_text_source_url=license_block["license_text_source_url"],
            license_text_sha256=license_block["license_text_sha256"],
            reviewer=review["reviewer"],
            reviewed_on=review["reviewed_on"],
        ),
        pages=tuple(pages),
    )


def _verify_sha256sums(root: Traversable) -> dict[str, bytes]:
    """Read every corpus file through SHA256SUMS; the file set must match exactly."""

    sums_node = root.joinpath(SHA256SUMS_FILE)
    if not sums_node.is_file():
        raise CorpusError(UNAVAILABLE_INTEGRITY, "SHA256SUMS is missing")
    sums = _read_bounded(sums_node, MAX_AUXILIARY_FILE_BYTES, SHA256SUMS_FILE)
    try:
        text = sums.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CorpusError(UNAVAILABLE_INVALID, "SHA256SUMS is not ASCII") from exc
    if not text.endswith("\n"):
        raise CorpusError(UNAVAILABLE_INVALID, "SHA256SUMS must end with a newline")
    expected: dict[str, str] = {}
    for line in text[:-1].split("\n"):
        match = _SUMS_LINE_RE.fullmatch(line)
        if match is None:
            raise CorpusError(UNAVAILABLE_INVALID, "SHA256SUMS has a malformed line")
        digest, path = match.groups()
        _validate_relative_path(path)
        if path in expected:
            raise CorpusError(UNAVAILABLE_INVALID, "SHA256SUMS lists a file twice")
        expected[path] = digest
    if list(expected) != sorted(expected):
        raise CorpusError(UNAVAILABLE_INVALID, "SHA256SUMS must be sorted by path")

    present = _enumerate_files(root)
    if present != set(expected):
        raise CorpusError(UNAVAILABLE_INTEGRITY, "corpus files do not match SHA256SUMS")

    contents: dict[str, bytes] = {}
    for path, digest in expected.items():
        limit = MAX_PAGE_BYTES if path.startswith(PAGES_DIRECTORY + "/") else (
            MAX_MANIFEST_BYTES if path == MANIFEST_FILE else MAX_AUXILIARY_FILE_BYTES
        )
        node = _node_for(root, path)
        data = _read_bounded(node, limit, path)
        if sha256_hex(data) != digest:
            raise CorpusError(UNAVAILABLE_INTEGRITY, f"{path} does not match SHA256SUMS")
        contents[path] = data
    if MANIFEST_FILE not in contents:
        raise CorpusError(UNAVAILABLE_INTEGRITY, "manifest is not covered by SHA256SUMS")
    return contents


def _validate_relative_path(path: str) -> None:
    parts = path.split("/")
    if len(parts) == 1 and _AUX_FILE_RE.fullmatch(path) and path != SHA256SUMS_FILE:
        return
    if len(parts) == 2 and _CONTENT_FILE_RE.fullmatch(path):
        return
    raise CorpusError(UNAVAILABLE_INVALID, "SHA256SUMS lists a path outside the corpus layout")


def _enumerate_files(root: Traversable) -> set[str]:
    files: set[str] = set()
    for entry in root.iterdir():
        _reject_symlink(entry)
        if entry.is_dir():
            if entry.name != PAGES_DIRECTORY:
                raise CorpusError(UNAVAILABLE_INTEGRITY, "corpus contains an unexpected directory")
            for page in entry.iterdir():
                _reject_symlink(page)
                if not page.is_file():
                    raise CorpusError(UNAVAILABLE_INTEGRITY, "pages contains a non-file entry")
                files.add(f"{PAGES_DIRECTORY}/{page.name}")
        elif entry.is_file():
            if entry.name != SHA256SUMS_FILE:
                files.add(entry.name)
        else:
            raise CorpusError(UNAVAILABLE_INTEGRITY, "corpus contains an unsupported entry")
    return files


def _reject_symlink(node: Traversable) -> None:
    if isinstance(node, Path) and node.is_symlink():
        raise CorpusError(UNAVAILABLE_INTEGRITY, "corpus must not contain symbolic links")


def _node_for(root: Traversable, path: str) -> Traversable:
    if path.startswith(PAGES_DIRECTORY + "/"):
        return _resolve_page_file(root, path)
    return root.joinpath(path)


def _resolve_page_file(root: Traversable, content_file: str) -> Traversable:
    match = _CONTENT_FILE_RE.fullmatch(content_file)
    if match is None:
        raise CorpusError(UNAVAILABLE_INVALID, "page path is outside the corpus layout")
    node = root.joinpath(PAGES_DIRECTORY).joinpath(content_file.split("/", 1)[1])
    _reject_symlink(node)
    return node


def _read_bounded(node: Traversable, limit: int, label: str) -> bytes:
    _reject_symlink(node)
    try:
        with node.open("rb") as handle:
            data = handle.read(limit + 1)
    except OSError as exc:
        raise CorpusError(UNAVAILABLE_INTEGRITY, f"{label} could not be read") from exc
    if len(data) > limit:
        raise CorpusError(UNAVAILABLE_INTEGRITY, f"{label} exceeds its size limit")
    return data


def _verified_text_file(data: bytes, label: str) -> str:
    try:
        text = data.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise CorpusError(UNAVAILABLE_INTEGRITY, f"{label} is not valid UTF-8") from exc
    if find_disallowed_character(text) is not None or "\t" in text:
        raise CorpusError(UNAVAILABLE_INTEGRITY, f"{label} contains disallowed characters")
    return text


def _verified_page_text(
    data: bytes,
    *,
    content_file: str,
    content_bytes: int,
    content_sha256: str,
    upstream_sha1: str,
) -> str:
    if len(data) != content_bytes:
        raise CorpusIntegrityError(f"{content_file} size does not match the manifest")
    if sha256_hex(data) != content_sha256:
        raise CorpusIntegrityError(f"{content_file} SHA-256 does not match the manifest")
    if hashlib.sha1(data, usedforsecurity=False).hexdigest() != upstream_sha1:
        raise CorpusIntegrityError(f"{content_file} does not match the upstream MediaWiki SHA-1")
    try:
        text = data.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise CorpusIntegrityError(f"{content_file} is not valid UTF-8") from exc
    if find_disallowed_character(text) is not None:
        raise CorpusIntegrityError(f"{content_file} contains characters outside the archival policy")
    return text
