"""Test support for the frozen VASP Wiki corpus.

Everything here is SYNTHETIC. No VASP Wiki content is reproduced: page titles
are invented ("Synthetic ..."), page text says it is a synthetic fixture, and
the license text is a placeholder. Synthetic corpora are built under their own
corpus kind and host (``synthetic.invalid``) so BMD Check's default loader
refuses them. Authoritative-kind corpora are only ever built in temporary
directories to exercise the loader's authoritative code path.

Regenerate the committed synthetic fixture with:

    PYTHONPATH=src python -B tests/vasp_wiki_support.py tests/fixtures/vasp_wiki_synthetic
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit

from bmd_agent import vasp_wiki_corpus as corpus


SYNTHETIC_HOST = "synthetic.invalid"
SYNTHETIC_LICENSE_TEXT = (
    b"SYNTHETIC TEST FIXTURE LICENSE PLACEHOLDER\n"
    b"This is not a real license text and grants nothing.\n"
)
SYNTHETIC_POLICY = corpus.CorpusPolicy(
    corpus_kind=corpus.KIND_SYNTHETIC_FIXTURE,
    allowed_hosts=frozenset({SYNTHETIC_HOST}),
    allowed_license_identifiers=frozenset({"LicenseRef-BMD-Synthetic-Test-Fixture"}),
    required_rights_text_fragment="SYNTHETIC",
    license_text_file="COPYING.SYNTHETIC-FIXTURE.txt",
    license_text_sha256=hashlib.sha256(SYNTHETIC_LICENSE_TEXT).hexdigest(),
)

# A license text used only to exercise the authoritative code path in
# temporary directories; the matching policy pins its digest.
TEST_AUTHORITATIVE_LICENSE_TEXT = b"SYNTHETIC STAND-IN FOR A LICENSE TEXT (tests only)\n"
TEST_AUTHORITATIVE_POLICY = replace(
    corpus.AUTHORITATIVE_POLICY,
    license_text_sha256=hashlib.sha256(TEST_AUTHORITATIVE_LICENSE_TEXT).hexdigest(),
)

FIXED_TIMESTAMP = "2026-01-02T03:04:05Z"
FIXED_DATE = "2026-01-02"

SYNTHETIC_README = b"""# SYNTHETIC test fixture corpus

This directory is a synthetic test fixture for BMD Agent's frozen VASP Wiki
corpus loader. It contains no VASP Wiki content. Titles, page text, site
rights and the license text are invented placeholders served from the
reserved host synthetic.invalid. BMD Check's default loader refuses this
corpus kind.

Regenerate with:
PYTHONPATH=src python -B tests/vasp_wiki_support.py tests/fixtures/vasp_wiki_synthetic
"""


@dataclass(frozen=True)
class SyntheticPage:
    requested_title: str
    title: str
    page_id: int
    revision_id: int
    parent_revision_id: int | None
    content: bytes
    redirected_from: str | None = None


SYNTHETIC_PAGES = (
    SyntheticPage(
        requested_title="Synthetic Alpha",
        title="Synthetic Alpha",
        page_id=11,
        revision_id=101,
        parent_revision_id=100,
        content=(
            "== Synthetic heading ==\n"
            "SYNTHETIC TEST FIXTURE: this is not VASP Wiki content.\n"
            "<script>alert('inert text, never executed')</script>\n"
            "{{SyntheticTemplate|value=1}} [[Synthetic link]] [https://synthetic.invalid ext]\n"
            "Unicode is preserved: café, αβγ, −, Å.\n"
            "\tIndented with a tab.\n"
        ).encode("utf-8"),
    ),
    SyntheticPage(
        requested_title="Synthetic_Old_Name",
        title="Synthetic Beta",
        page_id=12,
        revision_id=202,
        parent_revision_id=None,
        content=b"SYNTHETIC TEST FIXTURE page without a trailing newline",
        redirected_from="Synthetic Old Name",
    ),
)


def page_record(page: SyntheticPage, *, host: str, review_status: str = corpus.PAGE_REVIEW_NO_EXCEPTIONS) -> dict[str, Any]:
    slug = corpus.title_slug(page.requested_title)
    title_parameter = quote(page.title.replace(" ", "_"), safe="")
    script_url = f"https://{host}/wiki/index.php"
    reviewed = review_status != corpus.PAGE_REVIEW_PENDING
    return {
        "id": corpus.page_id_for_slug(slug),
        "requested_title": page.requested_title,
        "title": page.title,
        "redirected_from": page.redirected_from,
        "page_id": page.page_id,
        "namespace": 0,
        "revision_id": page.revision_id,
        "parent_revision_id": page.parent_revision_id,
        "revision_timestamp": FIXED_TIMESTAMP,
        "retrieved_at": FIXED_TIMESTAMP,
        "canonical_url": f"https://{host}/wiki/{title_parameter}",
        "permalink": f"{script_url}?title={title_parameter}&oldid={page.revision_id}",
        "history_url": f"{script_url}?title={title_parameter}&action=history",
        "content_model": "wikitext",
        "content_file": corpus.content_file_for(slug, page.revision_id),
        "content_bytes": len(page.content),
        "content_sha256": hashlib.sha256(page.content).hexdigest(),
        "upstream_sha1": hashlib.sha1(page.content).hexdigest(),
        "license_review": {
            "status": review_status,
            "reviewer": "Synthetic Reviewer" if reviewed else None,
            "reviewed_on": FIXED_DATE if reviewed else None,
            "notes": "synthetic exception" if review_status == corpus.PAGE_REVIEW_EXCEPTION_FOUND else None,
        },
    }


def build_corpus(
    directory: Path,
    *,
    policy: corpus.CorpusPolicy = SYNTHETIC_POLICY,
    pages: Sequence[SyntheticPage] = SYNTHETIC_PAGES,
    license_text: bytes = SYNTHETIC_LICENSE_TEXT,
    corpus_version: str = "1.0.0",
    readme: bytes = SYNTHETIC_README,
    identifier: str | None = None,
    rights_text: str | None = None,
) -> dict[str, Any]:
    """Write a complete, loadable corpus for ``policy`` and return its manifest."""

    host = sorted(policy.allowed_hosts)[0]
    manifest: dict[str, Any] = {
        "schema": corpus.MANIFEST_SCHEMA,
        "schema_version": corpus.SCHEMA_VERSION,
        "corpus_kind": policy.corpus_kind,
        "corpus_version": corpus_version,
        "corpus_digest": "",
        "assembly": {
            "assembled_at": FIXED_TIMESTAMP,
            "updater": {"name": "tests/vasp_wiki_support.py", "version": "1"},
        },
        "upstream": {
            "api_endpoint": f"https://{host}/wiki/api.php",
            "site_name": "Synthetic Test Wiki",
            "generator": "MediaWiki 1.0.0-synthetic",
            "server": f"https://{host}",
            "script_path": "/wiki/index.php",
            "article_path": "/wiki/$1",
            "rights_url": "",
            "rights_text": rights_text or policy.required_rights_text_fragment + " (synthetic fixture rights)",
        },
        "license": {
            "identifier": identifier or sorted(policy.allowed_license_identifiers)[0],
            "site_notice_as_displayed": "SYNTHETIC notice placeholder.",
            "invariant_sections": corpus.NONE_DECLARED,
            "cover_texts": corpus.NONE_DECLARED,
            "license_text_file": policy.license_text_file,
            "license_text_source_url": f"https://{host}/synthetic-license.txt",
            "license_text_sha256": hashlib.sha256(license_text).hexdigest(),
            "review": {
                "status": corpus.LICENSE_REVIEW_APPROVED,
                "reviewer": "Synthetic Reviewer",
                "reviewed_on": FIXED_DATE,
            },
        },
        "pages": sorted((page_record(page, host=host) for page in pages), key=lambda item: item["id"]),
    }
    files = {page_record(page, host=host)["content_file"]: page.content for page in pages}
    files[policy.license_text_file] = license_text
    files[corpus.README_FILE] = readme
    write_manifest_and_release(directory, manifest, files)
    return manifest


def write_manifest_and_release(directory: Path, manifest: dict[str, Any], files: Mapping[str, bytes]) -> None:
    """Write ``files`` plus manifest, a fresh ledger, NOTICE and SHA256SUMS."""

    manifest["corpus_digest"] = corpus.compute_corpus_digest(manifest)
    anchor = corpus.unpopulated_release_anchor()
    releases = {
        "schema": corpus.RELEASES_SCHEMA,
        "schema_version": corpus.SCHEMA_VERSION,
        "releases": [
            anchor,
            {
                "corpus_version": manifest["corpus_version"],
                "corpus_kind": manifest["corpus_kind"],
                "corpus_digest": manifest["corpus_digest"],
                "previous_corpus_digest": anchor["corpus_digest"],
            },
        ],
    }
    all_files = dict(files)
    all_files[corpus.MANIFEST_FILE] = corpus.canonical_manifest_bytes(manifest)
    all_files[corpus.RELEASES_FILE] = corpus.canonical_releases_bytes(releases)
    all_files[corpus.NOTICE_FILE] = corpus.render_notice(manifest)
    if directory.exists():
        shutil.rmtree(directory)
    (directory / corpus.PAGES_DIRECTORY).mkdir(parents=True)
    for relative, data in all_files.items():
        (directory / relative).write_bytes(data)
    rewrite_sums(directory)


def rewrite_sums(directory: Path) -> None:
    files = {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != corpus.SHA256SUMS_FILE
    }
    (directory / corpus.SHA256SUMS_FILE).write_bytes(corpus.render_sha256sums(files))


def read_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / corpus.MANIFEST_FILE).read_text("ascii"))


def write_manifest(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    recompute_digest: bool = False,
    update_notice: bool = True,
    raw: bytes | None = None,
) -> None:
    """Rewrite the manifest (optionally raw bytes) and keep SHA256SUMS consistent."""

    manifest = json.loads(json.dumps(manifest))
    if recompute_digest:
        manifest["corpus_digest"] = corpus.compute_corpus_digest(manifest)
    data = raw if raw is not None else corpus.canonical_manifest_bytes(manifest)
    (directory / corpus.MANIFEST_FILE).write_bytes(data)
    if update_notice and raw is None:
        try:
            (directory / corpus.NOTICE_FILE).write_bytes(corpus.render_notice(manifest))
        except (KeyError, TypeError):
            pass
    rewrite_sums(directory)


def write_unpopulated_corpus(directory: Path, readme: bytes | None = None) -> Path:
    """An unpopulated corpus (the release-ledger anchor only), as the committed
    corpus was before its first population. Maintainer-tool tests start from
    it so that they exercise first acquisitions independently of the packaged
    corpus release."""

    manifest = corpus.unpopulated_manifest()
    releases = {
        "schema": corpus.RELEASES_SCHEMA,
        "schema_version": corpus.SCHEMA_VERSION,
        "releases": [corpus.unpopulated_release_anchor()],
    }
    if readme is None:
        readme = (Path(__file__).resolve().parents[1] / "src" / "bmd_agent" / "reference_corpus" / "vasp_wiki" / "README.md").read_bytes()
    files = {
        corpus.MANIFEST_FILE: corpus.canonical_manifest_bytes(manifest),
        corpus.RELEASES_FILE: corpus.canonical_releases_bytes(releases),
        corpus.NOTICE_FILE: corpus.render_notice(manifest),
        corpus.README_FILE: readme,
    }
    directory.mkdir(parents=True)
    for name, data in files.items():
        (directory / name).write_bytes(data)
    (directory / corpus.SHA256SUMS_FILE).write_bytes(corpus.render_sha256sums(files))
    return directory


def build_synthetic_fixture(directory: Path) -> None:
    build_corpus(directory)


# ---------------------------------------------------------------------------
# Fake MediaWiki API for the maintainer tool
# ---------------------------------------------------------------------------


@dataclass
class FakeRevision:
    revid: int
    parentid: int
    timestamp: str
    content: str


class FakeWiki:
    """A small, deterministic stand-in for the MediaWiki action API.

    Response shapes follow the documented ``format=json&formatversion=2`` API.
    ``mutate`` lets a test corrupt a response for a given request.
    """

    def __init__(self, *, server: str = "https://vasp.at") -> None:
        self.server = server
        self.general = {
            "sitename": "Synthetic Test Wiki",
            "generator": "MediaWiki 1.41.0",
            "server": server,
            "script": "/wiki/index.php",
            "articlepath": "/wiki/$1",
        }
        self.rights = {"url": "https://synthetic.invalid/rights", "text": "GNU Free Documentation License 1.2"}
        self.pages: dict[str, tuple[int, list[FakeRevision]]] = {}
        self.redirects: dict[str, str] = {}
        self.license_text = TEST_AUTHORITATIVE_LICENSE_TEXT
        self.requests: list[str] = []
        self.mutate: Callable[[dict[str, Any], dict[str, str]], Any] | None = None
        # Discovery support (``fetch --all``).
        self.categories: dict[str, list[str]] = {}
        self.other_namespace_pages: dict[int, list[str]] = {}
        self.namespaces: dict[int, tuple[str, bool]] = {
            0: ("", True),
            1: ("Talk", False),
            2: ("User", False),
            6: ("File", False),
            10: ("Template", False),
            14: ("Category", False),
        }
        self.list_limit = 500
        self.category_page_size = 10_000
        self.content_batch_limit = 50  # larger batches get a continuation
        self.redirect_page_ids: dict[str, int] = {}

    def add_page(self, title: str, page_id: int, revisions: Sequence[FakeRevision]) -> None:
        self.pages[title] = (page_id, list(revisions))

    def __call__(self, url: str, expected: str) -> bytes:
        self.requests.append(url)
        parts = urlsplit(url)
        if parts.hostname == "www.gnu.org":
            assert expected == "text"
            return self.license_text
        params = dict(parse_qsl(parts.query, keep_blank_values=True))
        if params.get("meta") == "siteinfo" and params.get("siprop") == "namespaces":
            payload: Any = {"batchcomplete": True, "query": {"namespaces": self._namespaces()}}
        elif params.get("meta") == "siteinfo":
            payload = {"batchcomplete": True, "query": {"general": dict(self.general), "rightsinfo": dict(self.rights)}}
        elif params.get("list") == "allpages":
            payload = self._allpages(params)
        elif params.get("prop") == "categories":
            payload = self._categories(params)
        elif "revids" in params and "inprop" not in params:
            payload = self._by_revisions([int(item) for item in params["revids"].split("|")])
        elif "revids" in params:
            payload = self._by_revision(int(params["revids"]), params)
        elif "|" in params.get("titles", ""):
            payload = self._by_titles(params["titles"].split("|"), params)
        else:
            payload = self._by_title(params["titles"], params)
        if self.mutate is not None:
            payload = self.mutate(payload, params)
        if isinstance(payload, bytes):
            return payload
        try:
            return json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError:
            # Lone surrogates can only travel as JSON escapes, as a server would send them.
            return json.dumps(payload, ensure_ascii=True).encode("ascii")

    def _namespaces(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "-2": {"id": -2, "case": "first-letter", "name": "Media", "canonical": "Media"},
            "-1": {"id": -1, "case": "first-letter", "name": "Special", "canonical": "Special"},
        }
        for number, (name, content) in self.namespaces.items():
            item: dict[str, Any] = {"id": number, "case": "first-letter", "name": name}
            if name:
                item["canonical"] = name
            if content:
                item["content"] = True
            result[str(number)] = item
        return result

    def _allpages(self, params: Mapping[str, str]) -> dict[str, Any]:
        namespace = int(params["apnamespace"])
        redirects = params.get("apfilterredir") == "redirects"
        if namespace == 0:
            if redirects:
                entries = sorted((title, self._redirect_id(title)) for title in self.redirects)
            else:
                entries = sorted((title, page_id) for title, (page_id, _) in self.pages.items())
        else:
            entries = [] if redirects else [
                (title, 100_000 + namespace * 1_000 + index)
                for index, title in enumerate(sorted(self.other_namespace_pages.get(namespace, [])))
            ]
        start = params.get("apcontinue")
        if start is not None:
            entries = [entry for entry in entries if entry[0].replace(" ", "_") >= start]
        limit = min(int(params["aplimit"]), self.list_limit)
        chunk, rest = entries[:limit], entries[limit:]
        payload: dict[str, Any] = {
            "query": {"allpages": [{"pageid": page_id, "ns": namespace, "title": title} for title, page_id in chunk]}
        }
        if rest:
            payload["continue"] = {"apcontinue": rest[0][0].replace(" ", "_"), "continue": "-||"}
        else:
            payload["batchcomplete"] = True
        return payload

    def _redirect_id(self, title: str) -> int:
        if title not in self.redirect_page_ids:
            self.redirect_page_ids[title] = 90_000 + sorted(self.redirects).index(title)
        return self.redirect_page_ids[title]

    def _categories(self, params: Mapping[str, str]) -> dict[str, Any]:
        titles = params["titles"].split("|")
        pairs = [(title, category) for title in titles for category in sorted(self.categories.get(title, []))]
        offset = int(params.get("clcontinue", "0"))
        chunk = pairs[offset : offset + self.category_page_size]
        pages = []
        for title in titles:
            page_id = self.pages[title][0] if title in self.pages else None
            entry: dict[str, Any] = {"ns": 0, "title": title}
            if page_id is None:
                entry["missing"] = True
            else:
                entry["pageid"] = page_id
            mine = [{"ns": 14, "title": category} for owner, category in chunk if owner == title]
            if mine:
                entry["categories"] = mine
            pages.append(entry)
        payload: dict[str, Any] = {"query": {"pages": pages}}
        if offset + self.category_page_size < len(pairs):
            payload["continue"] = {"clcontinue": str(offset + self.category_page_size), "continue": "||"}
        else:
            payload["batchcomplete"] = True
        return payload

    def _by_titles(self, requested: Sequence[str], params: Mapping[str, str]) -> dict[str, Any]:
        query: dict[str, Any] = {"pages": []}
        with_revisions = "revisions" in params.get("prop", "")
        if with_revisions and len(requested) > self.content_batch_limit:
            first = self._by_titles(requested[: self.content_batch_limit], params)
            first["continue"] = {"rvcontinue": "1|2", "continue": "||"}
            first.pop("batchcomplete", None)
            return first
        seen_hops: set[str] = set()
        for title in requested:
            # Like ApiPageSet, follow redirect chains hop by hop and report
            # every hop once; omit the key entirely when nothing resolves.
            visited = {title}
            while params.get("redirects") == "1" and title in self.redirects:
                if title not in seen_hops:
                    query.setdefault("redirects", []).append({"from": title, "to": self.redirects[title]})
                    seen_hops.add(title)
                title = self.redirects[title]
                if title in visited:
                    break
                visited.add(title)
            if title not in self.pages:
                query["pages"].append({"ns": 0, "title": title, "missing": True})
                continue
            page_id, revisions = self.pages[title]
            query["pages"].append(self._page(title, page_id, revisions[-1] if with_revisions else None))
        return {"batchcomplete": True, "query": query}

    def _by_revisions(self, revids: Sequence[int]) -> dict[str, Any]:
        pages: dict[int, dict[str, Any]] = {}
        for title, (page_id, revisions) in self.pages.items():
            for revision in revisions:
                if revision.revid in revids:
                    entry = pages.setdefault(page_id, self._page(title, page_id, None))
                    entry.setdefault("revisions", []).extend(self._page(title, page_id, revision)["revisions"])
        return {"batchcomplete": True, "query": {"pages": list(pages.values())}}

    @staticmethod
    def normalize(title: str) -> str:
        title = title.replace("_", " ")
        return title[:1].upper() + title[1:]

    def _by_title(self, requested: str, params: Mapping[str, str]) -> dict[str, Any]:
        query: dict[str, Any] = {}
        title = self.normalize(requested)
        if title != requested:
            query["normalized"] = [{"fromencoded": False, "from": requested, "to": title}]
        if params.get("redirects") == "1" and title in self.redirects:
            query["redirects"] = [{"from": title, "to": self.redirects[title]}]
            title = self.redirects[title]
        if title not in self.pages:
            query["pages"] = [{"ns": 0, "title": title, "missing": True}]
            return {"batchcomplete": True, "query": query}
        page_id, revisions = self.pages[title]
        with_revisions = "revisions" in params.get("prop", "")
        query["pages"] = [self._page(title, page_id, revisions[-1] if with_revisions else None)]
        return {"batchcomplete": True, "query": query}

    def _by_revision(self, revid: int, params: Mapping[str, str]) -> dict[str, Any]:
        for title, (page_id, revisions) in self.pages.items():
            for revision in revisions:
                if revision.revid == revid:
                    return {"batchcomplete": True, "query": {"pages": [self._page(title, page_id, revision)]}}
        return {"batchcomplete": True, "query": {"badrevids": {str(revid): {"revid": revid, "missing": True}}}}

    def _page(self, title: str, page_id: int, revision: FakeRevision | None) -> dict[str, Any]:
        path = quote(title.replace(" ", "_"), safe="")
        page: dict[str, Any] = {
            "pageid": page_id,
            "ns": 0,
            "title": title,
            "contentmodel": "wikitext",
            "pagelanguage": "en",
            "fullurl": f"{self.server}/wiki/{path}",
            "canonicalurl": f"{self.server}/wiki/{path}",
        }
        if revision is not None:
            data = revision.content.encode("utf-8", "surrogatepass")
            page["revisions"] = [
                {
                    "revid": revision.revid,
                    "parentid": revision.parentid,
                    "timestamp": revision.timestamp,
                    "size": len(data),
                    "sha1": hashlib.sha1(data).hexdigest(),
                    "slots": {
                        "main": {
                            "contentmodel": "wikitext",
                            "contentformat": "text/x-wiki",
                            "content": revision.content,
                        }
                    },
                }
            ]
        return page


if __name__ == "__main__":
    build_synthetic_fixture(Path(sys.argv[1]))
