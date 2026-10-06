"""Maintainer-only VASP Wiki acquisition tool, exercised against a fake MediaWiki API.

No network is used. ``FakeWiki`` serves synthetic pages with invented titles;
nothing here reproduces VASP Wiki content.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType

import pytest

from bmd_agent import vasp_wiki_corpus as corpus
from bmd_agent.vasp_wiki_corpus import CorpusUnavailable, VaspWikiCorpus
from vasp_wiki_support import (
    TEST_AUTHORITATIVE_LICENSE_TEXT,
    TEST_AUTHORITATIVE_POLICY,
    FakeRevision,
    FakeWiki,
    read_manifest,
)

REPO = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO / "tools" / "vasp_reference" / "fetch_corpus.py"
PACKAGED = REPO / "src" / "bmd_agent" / "reference_corpus" / "vasp_wiki"
API = "https://vasp.at/wiki/api.php"
WHEN = "2026-03-04T05:06:07Z"

ALPHA_V1 = "SYNTHETIC alpha v1\n== Heading ==\n<script>inert()</script> {{Tpl}} café\n"
ALPHA_V2 = "SYNTHETIC alpha v2\n"
BETA = "SYNTHETIC beta without trailing newline"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bmd_vasp_fetch_corpus_under_test", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = load_tool()


@pytest.fixture
def wiki() -> FakeWiki:
    fake = FakeWiki()
    fake.add_page(
        "Synthetic Alpha",
        501,
        [FakeRevision(1001, 1000, "2025-01-01T00:00:00Z", ALPHA_V1), FakeRevision(1002, 1001, "2025-02-01T00:00:00Z", ALPHA_V2)],
    )
    fake.add_page("Synthetic Beta", 502, [FakeRevision(2001, 0, "2025-03-01T00:00:00Z", BETA)])
    fake.redirects["Synthetic Old Name"] = "Synthetic Beta"
    return fake


@pytest.fixture
def committed(tmp_path: Path) -> Path:
    directory = tmp_path / "committed"
    shutil.copytree(PACKAGED, directory)
    return directory


def tree(directory: Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def run(committed: Path, *args: str, fetcher=None) -> int:
    return tool.main(["--committed", str(committed), *args], fetcher=fetcher)


def fetch(committed: Path, staging: Path, wiki: FakeWiki, *pages: str, extra: tuple[str, ...] = ()) -> int:
    arguments = ["fetch", "--api-url", API, "--staging", str(staging), "--retrieved-at", WHEN, *extra]
    for page in pages:
        arguments.extend(["--page", page])
    return run(committed, *arguments, fetcher=wiki)


def approve(committed: Path, staging: Path, *page_ids: str) -> int:
    arguments = [
        "review", "--staging", str(staging), "--reviewer", "Maintainer", "--reviewed-on", "2026-03-05",
        "--license-identifier", "GFDL-1.2-only",
        "--site-notice", "Content is available under the synthetic test notice.",
        "--invariant-sections", "none_declared", "--cover-texts", "none_declared", "--approve-license",
    ]
    for page_id in page_ids:
        arguments.extend(["--page-reviewed", page_id])
    return run(committed, *arguments)


@pytest.fixture(autouse=True)
def pin_test_license(monkeypatch):
    """The fake wiki serves a synthetic license text; pin it like a maintainer would."""

    monkeypatch.setattr(corpus, "AUTHORITATIVE_POLICY", TEST_AUTHORITATIVE_POLICY)
    monkeypatch.setattr(tool.corpus, "AUTHORITATIVE_POLICY", TEST_AUTHORITATIVE_POLICY)


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def test_intended_titles_are_the_ten_approved_topics() -> None:
    assert tool.INTENDED_TITLES == (
        "NELM", "EDIFF", "ALGO", "EDIFFG", "NSW", "IBRION", "ISIF",
        "Not enough memory", "Difficult_to_converge_systems", "Memory",
    )
    ids = [corpus.page_id_for_slug(corpus.title_slug(title)) for title in tool.INTENDED_TITLES]
    assert len(set(ids)) == 10
    assert "vasp.wiki.not_enough_memory" in ids and "vasp.wiki.difficult_to_converge_systems" in ids


def test_intended_flag_requests_exactly_the_intended_titles(tmp_path: Path, committed: Path, wiki: FakeWiki, monkeypatch) -> None:
    monkeypatch.setattr(tool, "INTENDED_TITLES", ("Synthetic Alpha", "Synthetic Beta"))
    assert run(committed, "fetch", "--api-url", API, "--staging", str(tmp_path / "s"), "--intended",
               "--retrieved-at", WHEN, fetcher=wiki) == 0
    assert {p["requested_title"] for p in read_manifest(tmp_path / "s")["pages"]} == {"Synthetic Alpha", "Synthetic Beta"}


def test_fetch_stages_verbatim_pages_pending_review_without_touching_committed(
    tmp_path: Path, committed: Path, wiki: FakeWiki
) -> None:
    before = tree(committed)
    staging = tmp_path / "staging"
    assert fetch(committed, staging, wiki, "Synthetic Alpha", "Synthetic Old Name") == 0
    assert tree(committed) == before

    manifest = read_manifest(staging)
    assert manifest["corpus_kind"] == corpus.KIND_AUTHORITATIVE
    assert manifest["license"]["review"]["status"] == "pending"
    assert manifest["license"]["identifier"] is None
    assert manifest["license"]["license_text_source_url"] == tool.LICENSE_TEXT_URL
    assert manifest["upstream"]["rights_text"] == "GNU Free Documentation License 1.2"
    assert manifest["upstream"]["generator"] == "MediaWiki 1.41.0"
    alpha, beta = manifest["pages"]
    assert alpha["id"] == "vasp.wiki.synthetic_alpha" and alpha["revision_id"] == 1002
    assert alpha["parent_revision_id"] == 1001
    assert alpha["permalink"] == "https://vasp.at/wiki/index.php?title=Synthetic_Alpha&oldid=1002"
    assert alpha["history_url"] == "https://vasp.at/wiki/index.php?title=Synthetic_Alpha&action=history"
    assert alpha["canonical_url"] == "https://vasp.at/wiki/Synthetic_Alpha"
    assert all(page["license_review"]["status"] == "pending" for page in manifest["pages"])
    assert beta["id"] == "vasp.wiki.synthetic_old_name" and beta["title"] == "Synthetic Beta"
    assert beta["redirected_from"] == "Synthetic Old Name" and beta["parent_revision_id"] is None

    assert (staging / alpha["content_file"]).read_bytes() == ALPHA_V2.encode("utf-8")
    assert (staging / beta["content_file"]).read_bytes() == BETA.encode("utf-8")
    assert (staging / "COPYING.GFDL-1.2.txt").read_bytes() == TEST_AUTHORITATIVE_LICENSE_TEXT
    # Pending review never loads, even with the license digest pinned.
    result = corpus.load_vasp_wiki_corpus(staging, policy=TEST_AUTHORITATIVE_POLICY)
    assert result == CorpusUnavailable(corpus.UNAVAILABLE_LICENSE, "license review is not approved")


def test_fetch_is_deterministic(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    assert fetch(committed, tmp_path / "a", wiki, "Synthetic Alpha", "Synthetic Beta") == 0
    assert fetch(committed, tmp_path / "b", wiki, "Synthetic Beta", "Synthetic Alpha") == 0
    assert tree(tmp_path / "a") == tree(tmp_path / "b")


def test_requests_use_https_api_and_fixed_parameters(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    assert fetch(committed, tmp_path / "s", wiki, "Synthetic Alpha") == 0
    assert all(url.startswith(("https://vasp.at/wiki/api.php?", "https://www.gnu.org/")) for url in wiki.requests)
    page_request = wiki.requests[1]
    assert "rvslots=main" in page_request and "formatversion=2" in page_request and "redirects=1" in page_request


def test_revision_pinning_fetches_the_exact_older_revision(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    assert fetch(committed, tmp_path / "s", wiki, "Synthetic Alpha@1001") == 0
    page = read_manifest(tmp_path / "s")["pages"][0]
    assert page["revision_id"] == 1001 and page["content_file"] == "pages/synthetic_alpha.r1001.wiki"
    assert (tmp_path / "s" / page["content_file"]).read_bytes() == ALPHA_V1.encode("utf-8")
    assert any("revids=1001" in url for url in wiki.requests)


def test_pinned_revision_from_another_page_is_refused(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    assert fetch(committed, tmp_path / "s", wiki, "Synthetic Alpha@2001") == 2
    assert not (tmp_path / "s").exists()


def test_revision_substitution_is_refused(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    def substitute(payload, params):
        if "revids" in params:
            payload["query"]["pages"][0]["revisions"][0]["revid"] = 1002
        return payload

    wiki.mutate = substitute
    assert fetch(committed, tmp_path / "s", wiki, "Synthetic Alpha@1001") == 2
    assert not (tmp_path / "s").exists()


def test_normalized_titles_are_followed_and_recorded(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    assert fetch(committed, tmp_path / "s", wiki, "Synthetic_Alpha") == 0
    page = read_manifest(tmp_path / "s")["pages"][0]
    assert page["requested_title"] == "Synthetic_Alpha" and page["title"] == "Synthetic Alpha"
    assert page["redirected_from"] is None


def _mutating(mutation):
    def mutate(payload, params):
        if params.get("meta") == "siteinfo":
            return payload
        return mutation(payload)

    return mutate


def _revision(payload):
    return payload["query"]["pages"][0]["revisions"][0]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: b"not json",
        lambda p: b'{"query": {"pages": []}, "query": {}}',
        lambda p: b'{"query": {"x": NaN}}',
        lambda p: b"\xff\xfe",
        lambda p: b"[1, 2]",
        lambda p: {"error": {"code": "readapidenied"}},
        lambda p: {**p, "warnings": {"main": {"warnings": "unrecognized parameter"}}},
        lambda p: {**p, "continue": {"rvcontinue": "x"}},
        lambda p: {"batchcomplete": True},
        lambda p: {"query": {"pages": [{"ns": 0, "title": "Synthetic Alpha", "missing": True}]}},
        lambda p: {"query": {"pages": p["query"]["pages"] * 2}},
        lambda p: (p["query"]["pages"][0].update(redirect=True), p)[1],
        lambda p: (p["query"]["pages"][0].update(ns=1), p)[1],
        lambda p: (p["query"]["pages"][0].update(pageid=True), p)[1],
        lambda p: (p["query"]["pages"][0].update(contentmodel="javascript"), p)[1],
        lambda p: (p["query"]["pages"][0].update(title="Synthetic Other"), p)[1],
        lambda p: (p["query"]["pages"][0].update(title="Synthetic ‮ Alpha"), p)[1],
        lambda p: (p["query"]["pages"][0].pop("canonicalurl"), p)[1],
        lambda p: (p["query"]["pages"][0].update(revisions=[]), p)[1],
        lambda p: (_revision(p).update(texthidden=True), p)[1],
        lambda p: (_revision(p).update(sha1hidden=True), p)[1],
        lambda p: (_revision(p).update(revid=0), p)[1],
        lambda p: (_revision(p).update(parentid=99999), p)[1],
        lambda p: (_revision(p).update(timestamp="2025-02-01 00:00:00"), p)[1],
        lambda p: (_revision(p).update(sha1="ABC"), p)[1],
        lambda p: (_revision(p).update(sha1="0" * 40), p)[1],
        lambda p: (_revision(p).update(size=1), p)[1],
        lambda p: (_revision(p).update(slots={}), p)[1],
        lambda p: (_revision(p)["slots"]["main"].update(contentmodel="css"), p)[1],
        lambda p: (_revision(p)["slots"]["main"].update(contentformat="text/html"), p)[1],
        lambda p: (_revision(p)["slots"]["main"].update(content=None), p)[1],
        lambda p: (_revision(p)["slots"]["main"].update(missing=True), p)[1],
        lambda p: {**p, "query": {**p["query"], "redirects": "bad"}},
    ],
)
def test_malformed_or_unexpected_api_responses_are_refused(tmp_path: Path, committed: Path, wiki: FakeWiki, mutation) -> None:
    wiki.mutate = _mutating(mutation)
    assert fetch(committed, tmp_path / "s", wiki, "Synthetic Alpha") == 2
    assert not (tmp_path / "s").exists()


@pytest.mark.parametrize(
    "content",
    ["SYNTHETIC \x00", "SYNTHETIC \x1b[2J", "SYNTHETIC \r\n", "SYNTHETIC ‮", "SYNTHETIC ⁦",
     "SYNTHETIC ﻿", "SYNTHETIC \U000e0041", "SYNTHETIC \ud800"],
)
def test_content_outside_the_archival_policy_is_refused_not_rewritten(
    tmp_path: Path, committed: Path, wiki: FakeWiki, content: str
) -> None:
    wiki.add_page("Synthetic Gamma", 503, [FakeRevision(3001, 0, "2025-04-01T00:00:00Z", content)])
    assert fetch(committed, tmp_path / "s", wiki, "Synthetic Gamma") == 2
    assert not (tmp_path / "s").exists()


def test_oversized_page_is_refused(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    big = "SYNTHETIC " + "x" * corpus.MAX_PAGE_BYTES
    wiki.add_page("Synthetic Big", 504, [FakeRevision(4001, 0, "2025-04-01T00:00:00Z", big)])
    assert fetch(committed, tmp_path / "s", wiki, "Synthetic Big") == 2


def test_oversized_response_is_refused(tmp_path: Path, committed: Path, wiki: FakeWiki, monkeypatch) -> None:
    monkeypatch.setattr(tool, "MAX_API_RESPONSE_BYTES", 200)
    assert fetch(committed, tmp_path / "s", wiki, "Synthetic Alpha") == 2


def test_site_rights_must_name_the_accepted_license(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    wiki.rights = {"url": "", "text": "All rights reserved"}
    assert fetch(committed, tmp_path / "s", wiki, "Synthetic Alpha") == 2
    wiki.rights = {}
    assert fetch(committed, tmp_path / "t", wiki, "Synthetic Alpha") == 2


def test_license_text_mismatch_with_pinned_digest_is_refused(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    wiki.license_text = b"SYNTHETIC a different license text\n"
    assert fetch(committed, tmp_path / "s", wiki, "Synthetic Alpha") == 2


def test_duplicate_or_colliding_requests_are_refused(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    assert fetch(committed, tmp_path / "a", wiki, "Synthetic Alpha", "Synthetic_Alpha") == 2
    assert fetch(committed, tmp_path / "b", wiki, "Synthetic Beta", "Synthetic Old Name") == 2
    assert fetch(committed, tmp_path / "c", wiki) == 2


@pytest.mark.parametrize("title", ["...", "éè", "///"])
def test_titles_without_a_safe_identifier_are_refused(tmp_path: Path, committed: Path, wiki: FakeWiki, title: str) -> None:
    assert fetch(committed, tmp_path / "s", wiki, title) == 2


def test_hostile_titles_cannot_escape_the_pages_directory(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    hostile = "../../../Synthetic Escape/..\\x"
    wiki.add_page(hostile, 505, [FakeRevision(5001, 0, "2025-04-01T00:00:00Z", "SYNTHETIC escape attempt")])
    wiki.normalize = staticmethod(lambda title: title)  # type: ignore[method-assign]
    assert fetch(committed, tmp_path / "s", wiki, hostile) == 0
    page = read_manifest(tmp_path / "s")["pages"][0]
    assert page["content_file"] == "pages/synthetic_escape_x.r5001.wiki"
    assert {p.parent for p in (tmp_path / "s" / "pages").iterdir()} == {tmp_path / "s" / "pages"}
    assert "%2F" in page["permalink"] and "../" not in page["permalink"]


@pytest.mark.parametrize("argument", ["Bad\ttitle", "Bad\x1btitle", " padded", "Title@0", "Title@-1"])
def test_malformed_page_arguments_are_refused(argument: str) -> None:
    if argument.endswith(("@0", "@-1")):
        assert tool.parse_page_argument(argument).revision_id is None
        return
    with pytest.raises(tool.FetchError):
        tool.parse_page_argument(argument)


# ---------------------------------------------------------------------------
# HTTPS transport
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    ["http://vasp.at/wiki/api.php", "https://evil.example/api.php", "https://vasp.at.evil.example/api.php",
     "https://user@vasp.at/api.php", "https://vasp.at:444/api.php", "https://vasp.at/api php",
     "https://VASP.AT/api.php", "ftp://vasp.at/api.php", "https://vasp.at\\@evil.example/"],
)
def test_request_urls_are_restricted(url: str) -> None:
    with pytest.raises(tool.FetchError):
        tool.validate_request_url(url, frozenset({"vasp.at", "www.vasp.at"}))


def test_disallowed_api_url_is_refused_before_any_request(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    assert run(committed, "fetch", "--api-url", "https://evil.example/api.php", "--staging", str(tmp_path / "s"),
               "--page", "Synthetic Alpha", fetcher=wiki) == 2
    assert wiki.requests == []


def test_license_text_host_is_restricted() -> None:
    with pytest.raises(tool.FetchError):
        tool._guarded_fetch(lambda url, kind: b"x", "https://vasp.at/fdl.txt", "text")


def test_http_redirects_are_refused() -> None:
    handler = tool._RefuseRedirects()
    with pytest.raises(tool.FetchError):
        handler.redirect_request(None, None, 302, "Found", {}, "https://evil.example/")


def test_https_fetch_refuses_disallowed_urls_without_connecting(monkeypatch) -> None:
    def no_network(*args, **kwargs):
        raise AssertionError("network must not be reached")

    monkeypatch.setattr(tool.urllib.request, "build_opener", no_network)
    with pytest.raises(tool.FetchError):
        tool.https_fetch("http://vasp.at/wiki/api.php", "json")


# ---------------------------------------------------------------------------
# Staging safety
# ---------------------------------------------------------------------------


def test_existing_staging_directory_is_not_overwritten(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    staging = tmp_path / "s"
    staging.mkdir()
    (staging / "keep.txt").write_bytes(b"keep")
    assert fetch(committed, staging, wiki, "Synthetic Alpha") == 2
    assert tree(staging) == {"keep.txt": b"keep"}


def test_staging_inside_the_committed_corpus_or_src_is_refused(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    before = tree(committed)
    assert fetch(committed, committed / "staging", wiki, "Synthetic Alpha") == 2
    assert fetch(committed, committed, wiki, "Synthetic Alpha") == 2
    assert fetch(committed, REPO / "src" / "bmd_agent" / "staging-test", wiki, "Synthetic Alpha") == 2
    assert tree(committed) == before
    assert not (REPO / "src" / "bmd_agent" / "staging-test").exists()
    assert wiki.requests == []


def test_review_and_release_refuse_the_committed_corpus(committed: Path) -> None:
    before = tree(committed)
    assert approve(committed, committed) == 2
    assert run(committed, "release", "--staging", str(committed), "--corpus-version", "1.0.0") == 2
    assert tree(committed) == before


# ---------------------------------------------------------------------------
# Review, release, diff and verify
# ---------------------------------------------------------------------------


def _released(tmp_path: Path, committed: Path, wiki: FakeWiki, version: str = "1.0.0") -> Path:
    staging = tmp_path / "staging"
    assert fetch(committed, staging, wiki, "Synthetic Alpha", "Synthetic Old Name") == 0
    assert approve(committed, staging, "vasp.wiki.synthetic_alpha", "vasp.wiki.synthetic_old_name") == 0
    assert run(committed, "release", "--staging", str(staging), "--corpus-version", version) == 0
    return staging


def test_full_maintainer_flow_produces_a_loadable_authoritative_corpus(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    staging = _released(tmp_path, committed, wiki)
    result = corpus.load_vasp_wiki_corpus(staging, policy=TEST_AUTHORITATIVE_POLICY)
    assert isinstance(result, VaspWikiCorpus)
    assert result.corpus_version == "1.0.0"
    assert result.license.site_notice_as_displayed == "Content is available under the synthetic test notice."
    assert result.page("vasp.wiki.synthetic_alpha").read_wikitext() == ALPHA_V2
    entries = json.loads((staging / "RELEASES.json").read_text("ascii"))["releases"]
    assert entries[0] == corpus.unpopulated_release_anchor()
    assert entries[1]["previous_corpus_digest"] == entries[0]["corpus_digest"]
    assert entries[1]["corpus_digest"] == result.corpus_digest
    notice = (staging / "NOTICE").read_text("utf-8")
    assert "this revision: https://vasp.at/wiki/index.php?title=Synthetic_Alpha&oldid=1002" in notice
    assert "not covered by the" in notice and "MIT License" in notice


def test_release_refuses_unreviewed_or_excepted_pages(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    staging = tmp_path / "staging"
    assert fetch(committed, staging, wiki, "Synthetic Alpha", "Synthetic Beta") == 0
    assert approve(committed, staging, "vasp.wiki.synthetic_alpha") == 0
    assert run(committed, "release", "--staging", str(staging), "--corpus-version", "1.0.0") == 2
    assert run(committed, "review", "--staging", str(staging), "--reviewer", "Maintainer", "--reviewed-on",
               "2026-03-05", "--page-exception", "vasp.wiki.synthetic_beta", "image licensed separately") == 0
    assert read_manifest(staging)["pages"][1]["license_review"]["status"] == "exception_found"
    assert run(committed, "release", "--staging", str(staging), "--corpus-version", "1.0.0") == 2


def test_license_cannot_be_approved_with_unrecorded_fields(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    staging = tmp_path / "staging"
    assert fetch(committed, staging, wiki, "Synthetic Alpha") == 0
    assert run(committed, "review", "--staging", str(staging), "--reviewer", "M", "--reviewed-on", "2026-03-05",
               "--approve-license") == 2


@pytest.mark.parametrize("version", ["0.0.0", "1.0", "01.0.0", "abc"])
def test_release_requires_a_new_greater_version(tmp_path: Path, committed: Path, wiki: FakeWiki, version: str) -> None:
    staging = tmp_path / "staging"
    assert fetch(committed, staging, wiki, "Synthetic Alpha") == 0
    assert approve(committed, staging, "vasp.wiki.synthetic_alpha") == 0
    assert run(committed, "release", "--staging", str(staging), "--corpus-version", version) == 2


def _promote(staging: Path, committed: Path) -> None:
    """What a maintainer does by hand in a reviewed PR."""

    shutil.rmtree(committed)
    shutil.copytree(staging, committed)


def test_updates_against_a_populated_corpus(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    _promote(_released(tmp_path, committed, wiki), committed)

    # Unchanged revisions: identity unchanged, nothing to release.
    again = tmp_path / "again"
    assert fetch(committed, again, wiki, "Synthetic Alpha", "Synthetic Old Name") == 0
    assert run(committed, "diff", "--staging", str(again)) == 0

    # A new upstream revision is visible in the diff and needs a new version.
    wiki.pages["Synthetic Alpha"][1].append(FakeRevision(1003, 1002, "2025-05-01T00:00:00Z", "SYNTHETIC alpha v3\n"))
    newer = tmp_path / "newer"
    assert fetch(committed, newer, wiki, "Synthetic Alpha", "Synthetic Old Name") == 0
    lines, alerts = tool.diff_corpora(read_manifest(committed), read_manifest(newer))
    assert "revision   vasp.wiki.synthetic_alpha: r1002 -> r1003" in lines and not alerts
    assert approve(committed, newer, "vasp.wiki.synthetic_alpha", "vasp.wiki.synthetic_old_name") == 0
    assert run(committed, "release", "--staging", str(newer), "--corpus-version", "1.0.0") == 2
    assert run(committed, "release", "--staging", str(newer), "--corpus-version", "1.1.0") == 0
    committed_entries = json.loads((committed / "RELEASES.json").read_text("ascii"))["releases"]
    newer_entries = json.loads((newer / "RELEASES.json").read_text("ascii"))["releases"]
    assert corpus.releases_extend(committed_entries, newer_entries)


def test_site_rights_change_requires_explicit_acceptance(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    _promote(_released(tmp_path, committed, wiki), committed)
    wiki.rights = {"url": "https://synthetic.invalid/rights-v2", "text": "GNU Free Documentation License 1.2 (amended)"}
    assert fetch(committed, tmp_path / "a", wiki, "Synthetic Alpha") == 2
    assert fetch(committed, tmp_path / "b", wiki, "Synthetic Alpha", extra=("--accept-site-rights-change",)) == 0
    manifest = read_manifest(tmp_path / "b")
    assert manifest["license"]["review"]["status"] == "pending"
    _, alerts = tool.diff_corpora(read_manifest(committed), manifest)
    assert "site rights_text changed" in alerts and "site rights_url changed" in alerts


def test_diff_alerts_on_content_change_under_the_same_revision(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    _promote(_released(tmp_path, committed, wiki), committed)
    staged = read_manifest(committed)
    staged["pages"][0]["content_sha256"] = "f" * 64
    staged["corpus_digest"] = corpus.compute_corpus_digest(staged)
    _, alerts = tool.diff_corpora(read_manifest(committed), staged)
    assert any("content changed while the revision ID stayed" in alert for alert in alerts)
    assert "corpus identity changed without a corpus version change" in alerts


def test_release_refuses_a_staging_ledger_that_diverged(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    staging = tmp_path / "staging"
    assert fetch(committed, staging, wiki, "Synthetic Alpha") == 0
    assert approve(committed, staging, "vasp.wiki.synthetic_alpha") == 0
    releases = json.loads((staging / "RELEASES.json").read_text("ascii"))
    releases["releases"].append({**releases["releases"][0], "corpus_version": "0.5.0", "previous_corpus_digest": releases["releases"][0]["corpus_digest"]})
    (staging / "RELEASES.json").write_bytes(corpus.canonical_releases_bytes(releases))
    assert run(committed, "release", "--staging", str(staging), "--corpus-version", "1.0.0") == 2


def test_verify_reports_the_packaged_unpopulated_corpus(capsys) -> None:
    assert tool.main(["verify", str(PACKAGED)]) == 0
    assert "unavailable: unpopulated" in capsys.readouterr().out


def test_tool_reports_refusals_on_stderr_without_page_content(tmp_path: Path, committed: Path, wiki: FakeWiki, capsys) -> None:
    wiki.add_page("Synthetic Gamma", 503, [FakeRevision(3001, 0, "2025-04-01T00:00:00Z", "SYNTHETIC-SECRET \x1b[2J")])
    assert fetch(committed, tmp_path / "s", wiki, "Synthetic Gamma") == 2
    captured = capsys.readouterr()
    assert "refused:" in captured.err
    assert "SYNTHETIC-SECRET" not in captured.err + captured.out and "\x1b" not in captured.err + captured.out


def test_verify_with_an_unpinned_policy_refuses_populated_corpora(tmp_path: Path, committed: Path, wiki: FakeWiki, monkeypatch) -> None:
    staging = _released(tmp_path, committed, wiki)
    unpinned = corpus.CorpusPolicy(**{**TEST_AUTHORITATIVE_POLICY.__dict__, "license_text_sha256": None})
    monkeypatch.setattr(tool.corpus, "AUTHORITATIVE_POLICY", unpinned)
    assert tool.main(["verify", str(staging)]) == 1
    digest = hashlib.sha256(TEST_AUTHORITATIVE_LICENSE_TEXT).hexdigest()
    assert tool.main(["verify", str(staging), "--license-text-sha256", digest]) == 0


def test_staging_symlinks_cannot_redirect_writes_into_the_committed_corpus(
    tmp_path: Path, committed: Path, wiki: FakeWiki
) -> None:
    staging = tmp_path / "staging"
    assert fetch(committed, staging, wiki, "Synthetic Alpha") == 0
    before = tree(committed)
    (staging / "NOTICE").unlink()
    (staging / "NOTICE").symlink_to(committed / "NOTICE")
    assert approve(committed, staging, "vasp.wiki.synthetic_alpha") == 2
    assert run(committed, "release", "--staging", str(staging), "--corpus-version", "1.0.0") == 2
    assert tree(committed) == before


def test_untrusted_api_error_codes_are_not_echoed(tmp_path: Path, committed: Path, wiki: FakeWiki, capsys) -> None:
    wiki.mutate = _mutating(lambda p: {"error": {"code": "\x1b]0;owned\x07"}})
    assert fetch(committed, tmp_path / "s", wiki, "Synthetic Alpha") == 2
    captured = capsys.readouterr()
    assert "\x1b" not in captured.err and "unrecognized error code" in captured.err


def test_diff_refuses_a_hand_edited_manifest_with_control_characters(
    tmp_path: Path, committed: Path, wiki: FakeWiki, capsys
) -> None:
    staging = tmp_path / "staging"
    assert fetch(committed, staging, wiki, "Synthetic Alpha") == 0
    manifest = read_manifest(staging)
    manifest["pages"][0]["title"] = "Synthetic \x1b[2J Alpha"
    (staging / "manifest.json").write_bytes(corpus.canonical_manifest_bytes(manifest))
    assert run(committed, "diff", "--staging", str(staging)) == 2
    assert "\x1b" not in capsys.readouterr().out


@pytest.mark.parametrize("note", ["", "   ", " padded", "trailing "])
def test_review_refuses_empty_or_padded_exception_notes(
    tmp_path: Path, committed: Path, wiki: FakeWiki, note: str
) -> None:
    staging = tmp_path / "staging"
    assert fetch(committed, staging, wiki, "Synthetic Alpha") == 0
    before = tree(staging)
    assert run(committed, "review", "--staging", str(staging), "--reviewer", "M", "--reviewed-on", "2026-03-05",
               "--page-exception", "vasp.wiki.synthetic_alpha", note) == 2
    assert tree(staging) == before


def test_review_refuses_a_page_marked_both_clean_and_excepted(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    staging = tmp_path / "staging"
    assert fetch(committed, staging, wiki, "Synthetic Alpha") == 0
    assert run(committed, "review", "--staging", str(staging), "--reviewer", "M", "--reviewed-on", "2026-03-05",
               "--page-reviewed", "vasp.wiki.synthetic_alpha",
               "--page-exception", "vasp.wiki.synthetic_alpha", "figure licensed separately") == 2


def test_review_states_written_by_the_tool_follow_the_notes_rule(tmp_path: Path, committed: Path, wiki: FakeWiki) -> None:
    staging = tmp_path / "staging"
    assert fetch(committed, staging, wiki, "Synthetic Alpha") == 0
    assert read_manifest(staging)["pages"][0]["license_review"]["notes"] is None
    assert run(committed, "review", "--staging", str(staging), "--reviewer", "M", "--reviewed-on", "2026-03-05",
               "--page-exception", "vasp.wiki.synthetic_alpha", "figure licensed separately") == 0
    assert read_manifest(staging)["pages"][0]["license_review"]["notes"] == "figure licensed separately"
    assert run(committed, "review", "--staging", str(staging), "--reviewer", "M", "--reviewed-on", "2026-03-06",
               "--page-reviewed", "vasp.wiki.synthetic_alpha") == 0
    review = read_manifest(staging)["pages"][0]["license_review"]
    assert review == {"status": "reviewed_no_exceptions", "reviewer": "M", "reviewed_on": "2026-03-06", "notes": None}
