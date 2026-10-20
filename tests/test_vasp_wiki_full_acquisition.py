"""Systematic full-corpus acquisition (``fetch --all``) against a fake MediaWiki API.

No network is used. ``FakeWiki`` serves synthetic pages with invented titles;
nothing here reproduces VASP Wiki content.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from bmd_agent import vasp_wiki_corpus as corpus
from bmd_agent.vasp_wiki_corpus import CorpusUnavailable, VaspWikiCorpus
from vasp_wiki_support import (
    TEST_AUTHORITATIVE_POLICY,
    FakeRevision,
    FakeWiki,
    read_manifest,
    rewrite_sums,
    write_unpopulated_corpus,
)
from test_vasp_wiki_fetch_tool import API, WHEN, approve, run, tool


@pytest.fixture(autouse=True)
def pin_test_license(monkeypatch):
    monkeypatch.setattr(corpus, "AUTHORITATIVE_POLICY", TEST_AUTHORITATIVE_POLICY)
    monkeypatch.setattr(tool.corpus, "AUTHORITATIVE_POLICY", TEST_AUTHORITATIVE_POLICY)


@pytest.fixture
def committed(tmp_path: Path) -> Path:
    return write_unpopulated_corpus(tmp_path / "committed")


def synthetic_wiki(count: int = 7) -> FakeWiki:
    """A wiki with ``count`` ordinary pages, a case-colliding pair, a redirect,
    a dangling redirect, a page with a non-ASCII-only title, categories that
    paginate, and pages in other namespaces."""

    wiki = FakeWiki()
    for index in range(count):
        title = f"Synthetic Page {index:02d}"
        wiki.add_page(title, 700 + index, [FakeRevision(7000 + index, 0, "2025-05-01T00:00:00Z", f"SYNTHETIC {index}\n")])
        wiki.categories[title] = [f"Category:Synthetic group {index % 3}"]
    wiki.add_page("Synthetic Case", 801, [FakeRevision(8001, 0, "2025-05-02T00:00:00Z", "SYNTHETIC upper\n")])
    wiki.add_page("Synthetic case", 802, [FakeRevision(8002, 0, "2025-05-02T00:00:00Z", "SYNTHETIC lower\n")])
    wiki.add_page("Συνθετικό", 803, [FakeRevision(8003, 0, "2025-05-02T00:00:00Z", "SYNTHETIC greek title\n")])
    wiki.categories["Synthetic Case"] = ["Category:Synthetic group 0", "Category:Synthetic hidden"]
    wiki.redirects["Synthetic Alias"] = "Synthetic Page 00"
    wiki.redirects["Synthetic Dangling"] = "Synthetic Deleted"
    wiki.other_namespace_pages = {10: ["Template:Synthetic"], 14: ["Category:Synthetic group 0"], 2: ["User:Synthetic"]}
    wiki.list_limit = 3  # force allpages pagination
    wiki.category_page_size = 2  # force clcontinue pagination
    return wiki


def fetch_all(committed: Path, staging: Path, wiki: FakeWiki, *extra: str) -> int:
    return run(
        committed,
        "fetch", "--api-url", API, "--staging", str(staging), "--retrieved-at", WHEN, "--all", "--delay", "0",
        *extra,
        fetcher=wiki,
    )


def page_ids(staging: Path) -> list[str]:
    return [page["id"] for page in read_manifest(staging)["pages"]]


def review_all(committed: Path, staging: Path) -> None:
    assert approve(committed, staging, *page_ids(staging)) == 0


# ---------------------------------------------------------------------------
# Discovery and coverage
# ---------------------------------------------------------------------------


def test_fetch_all_discovers_every_main_namespace_page(tmp_path: Path, committed: Path) -> None:
    wiki = synthetic_wiki()
    assert fetch_all(committed, tmp_path / "s", wiki) == 0
    manifest = read_manifest(tmp_path / "s")
    assert manifest["schema_version"] == 2
    titles = {page["title"] for page in manifest["pages"]}
    assert titles == set(wiki.pages)
    acquisition = manifest["acquisition"]
    assert acquisition["method"] == "mediawiki_allpages"
    assert acquisition["inclusion_rules"] == "vasp-wiki-main-namespace-1"
    assert acquisition["completeness"] == "complete"
    assert acquisition["discovered_titles"] == len(wiki.pages) == 10
    assert acquisition["excluded"] == [] and acquisition["failed"] == []
    assert acquisition["redirects"] == [
        {"title": "Synthetic Alias", "target": "Synthetic Page 00"},
        {"title": "Synthetic Dangling", "target": "Synthetic Deleted"},
    ]
    assert acquisition["other_namespaces"] == [
        {"namespace": 1, "name": "Talk", "pages": 0},
        {"namespace": 2, "name": "User", "pages": 1},
        {"namespace": 6, "name": "File", "pages": 0},
        {"namespace": 10, "name": "Template", "pages": 1},
        {"namespace": 14, "name": "Category", "pages": 1},
    ]
    # Pagination was followed for allpages and for categories.
    assert any("apcontinue=" in url for url in wiki.requests)
    assert any("clcontinue=" in url for url in wiki.requests)


def test_pages_record_identity_revision_categories_and_namespace(tmp_path: Path, committed: Path) -> None:
    assert fetch_all(committed, tmp_path / "s", synthetic_wiki()) == 0
    pages = {page["title"]: page for page in read_manifest(tmp_path / "s")["pages"]}
    page = pages["Synthetic Case"]
    assert page["page_id"] == 801 and page["revision_id"] == 8001 and page["namespace"] == 0
    assert page["categories"] == ["Category:Synthetic group 0", "Category:Synthetic hidden"]
    assert page["permalink"].endswith("title=Synthetic_Case&oldid=8001")
    assert page["retrieved_at"] == WHEN
    assert page["redirected_from"] is None and page["requested_title"] == page["title"]
    assert pages["Συνθετικό"]["categories"] == []


def test_colliding_and_unsluggable_titles_get_deterministic_identifiers(tmp_path: Path, committed: Path) -> None:
    assert fetch_all(committed, tmp_path / "s", synthetic_wiki()) == 0
    ids = {page["title"]: page["id"] for page in read_manifest(tmp_path / "s")["pages"]}
    assert ids["Synthetic Case"] == "vasp.wiki.synthetic_case_p801"
    assert ids["Synthetic case"] == "vasp.wiki.synthetic_case_p802"
    assert ids["Συνθετικό"] == "vasp.wiki.p803"
    assert ids["Synthetic Page 03"] == "vasp.wiki.synthetic_page_03"


def test_fetch_all_is_deterministic_and_order_independent(tmp_path: Path, committed: Path) -> None:
    first, second = synthetic_wiki(), synthetic_wiki()
    second.pages = dict(reversed(list(second.pages.items())))
    second.list_limit = 500
    assert fetch_all(committed, tmp_path / "a", first) == 0
    assert fetch_all(committed, tmp_path / "b", second) == 0
    files = lambda d: {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}
    assert files(tmp_path / "a") == files(tmp_path / "b")


def test_large_batches_are_split_when_the_api_asks_to_continue(tmp_path: Path, committed: Path) -> None:
    wiki = synthetic_wiki(count=12)
    wiki.content_batch_limit = 4
    assert fetch_all(committed, tmp_path / "s", wiki) == 0
    manifest = read_manifest(tmp_path / "s")
    assert manifest["acquisition"]["completeness"] == "complete"
    assert len(manifest["pages"]) == 15


def test_fetch_all_cannot_be_combined_with_explicit_titles(tmp_path: Path, committed: Path) -> None:
    assert fetch_all(committed, tmp_path / "s", synthetic_wiki(), "--page", "Synthetic Case") == 2
    assert not (tmp_path / "s").exists()


def test_requests_are_throttled_and_transport_failures_retried(monkeypatch) -> None:
    calls: list[str] = []
    sleeps: list[float] = []

    def flaky(url: str, expected: str) -> bytes:
        calls.append(url)
        if len(calls) == 2:
            raise tool.FetchError("request failed: URLError")
        return b"{}"

    polite = tool.PoliteFetcher(flaky, delay=0.5, retries=2, sleep=sleeps.append)
    assert polite("a", "json") == b"{}"
    assert polite("b", "json") == b"{}"
    assert calls == ["a", "b", "b"] and sleeps == [0.5, 1.0]

    def broken(url: str, expected: str) -> bytes:
        raise tool.FetchError("response exceeds the size limit")

    with pytest.raises(tool.FetchError):
        tool.PoliteFetcher(broken, delay=0, retries=5, sleep=sleeps.append)("c", "json")


# ---------------------------------------------------------------------------
# Exclusions, failures and completeness
# ---------------------------------------------------------------------------


def test_policy_violations_are_excluded_explicitly_not_rewritten(tmp_path: Path, committed: Path) -> None:
    wiki = synthetic_wiki()
    wiki.add_page("Synthetic CRLF", 901, [FakeRevision(9001, 0, "2025-06-01T00:00:00Z", "SYNTHETIC\r\n")])
    assert fetch_all(committed, tmp_path / "s", wiki) == 0
    manifest = read_manifest(tmp_path / "s")
    acquisition = manifest["acquisition"]
    assert acquisition["completeness"] == "complete"
    assert acquisition["excluded"] == [
        {
            "title": "Synthetic CRLF",
            "page_id": 901,
            "revision_id": 9001,
            "reason": "archival_character_policy",
            "detail": "page content violates the archival character policy; human review required",
        }
    ]
    assert "Synthetic CRLF" not in {page["title"] for page in manifest["pages"]}
    assert not any("9001" in path.name for path in (tmp_path / "s" / "pages").iterdir())
    notice = (tmp_path / "s" / "NOTICE").read_text("utf-8")
    assert "excluded: Synthetic CRLF (archival_character_policy)" in notice


def test_failed_pages_make_the_corpus_incomplete(tmp_path: Path, committed: Path) -> None:
    wiki = synthetic_wiki()

    def corrupt(payload, params):
        for page in payload.get("query", {}).get("pages", []):
            if page.get("title") == "Synthetic Page 01" and "revisions" in page:
                page["revisions"][0]["sha1"] = "0" * 40
        return payload

    wiki.mutate = corrupt
    assert fetch_all(committed, tmp_path / "s", wiki) == 0
    manifest = read_manifest(tmp_path / "s")
    acquisition = manifest["acquisition"]
    assert acquisition["completeness"] == "incomplete"
    assert acquisition["failed"] == [
        {
            "title": "Synthetic Page 01",
            "reason": "integrity_mismatch",
            "detail": "page content does not match the upstream MediaWiki SHA-1",
        }
    ]
    review_all(committed, tmp_path / "s")
    # An incomplete acquisition is never released as complete.
    assert run(committed, "release", "--staging", str(tmp_path / "s"), "--corpus-version", "1.0.0") == 2
    assert run(
        committed, "release", "--staging", str(tmp_path / "s"), "--corpus-version", "1.0.0", "--allow-incomplete"
    ) == 0
    loaded = corpus.load_vasp_wiki_corpus(tmp_path / "s", policy=TEST_AUTHORITATIVE_POLICY)
    assert isinstance(loaded, VaspWikiCorpus)
    assert loaded.acquisition.completeness == "incomplete"
    assert [item.title for item in loaded.acquisition.failed] == ["Synthetic Page 01"]


def test_transport_failure_isolates_the_single_page(tmp_path: Path, committed: Path) -> None:
    wiki = synthetic_wiki()
    original = wiki.__call__

    def failing(url: str, expected: str) -> bytes:
        if "rvprop" in url and "Synthetic+Page+04" in url:
            raise tool.FetchError("request failed: URLError")
        return original(url, expected)

    assert fetch_all(committed, tmp_path / "s", failing, "--retries", "0") == 0
    acquisition = read_manifest(tmp_path / "s")["acquisition"]
    assert [(item["title"], item["reason"]) for item in acquisition["failed"]] == [("Synthetic Page 04", "request_failed")]
    assert acquisition["discovered_titles"] == len(read_manifest(tmp_path / "s")["pages"]) + 1


def test_page_deleted_after_discovery_is_a_failure(tmp_path: Path, committed: Path) -> None:
    wiki = synthetic_wiki()
    original = wiki._by_titles

    def deleting(requested, params):
        payload = original(requested, params)
        for page in payload["query"]["pages"]:
            if page["title"] == "Synthetic Page 02" and "revisions" in page:
                page.clear()
                page.update({"ns": 0, "title": "Synthetic Page 02", "missing": True})
        return payload

    wiki._by_titles = deleting
    assert fetch_all(committed, tmp_path / "s", wiki) == 0
    failed = read_manifest(tmp_path / "s")["acquisition"]["failed"]
    assert [(item["title"], item["reason"]) for item in failed] == [("Synthetic Page 02", "missing_at_retrieval")]


# ---------------------------------------------------------------------------
# Review, licensing exclusions, release, report and reproduction
# ---------------------------------------------------------------------------


def test_full_flow_with_reviewed_list_file_and_license_exclusion(tmp_path: Path, committed: Path) -> None:
    staging = tmp_path / "s"
    assert fetch_all(committed, staging, synthetic_wiki()) == 0
    ids = page_ids(staging)
    excepted = "vasp.wiki.synthetic_case_p802"
    listing = tmp_path / "reviewed.txt"
    listing.write_text("# reviewed\n" + "\n".join(i for i in ids if i != excepted) + "\n", "utf-8")
    assert run(
        committed, "review", "--staging", str(staging), "--reviewer", "Maintainer", "--reviewed-on", "2026-03-05",
        "--license-identifier", "GFDL-1.2-only", "--site-notice", "Synthetic notice.",
        "--invariant-sections", "none_declared", "--cover-texts", "none_declared", "--approve-license",
        "--page-reviewed-file", str(listing),
        "--page-exception", excepted, "synthetic image credited to a third party",
    ) == 0
    assert run(committed, "release", "--staging", str(staging), "--corpus-version", "1.0.0") == 2
    assert run(committed, "release", "--staging", str(staging), "--corpus-version", "1.0.0", "--exclude-excepted") == 0

    loaded = corpus.load_vasp_wiki_corpus(staging, policy=TEST_AUTHORITATIVE_POLICY)
    assert isinstance(loaded, VaspWikiCorpus)
    assert loaded.acquisition.completeness == "complete"
    titles = {page.title for page in loaded.pages}
    assert "Synthetic case" not in titles and "Synthetic Case" in titles
    # The collision group dissolved, so the survivor takes the plain identifier.
    assert loaded.page("vasp.wiki.synthetic_case").title == "Synthetic Case"
    assert loaded.page("vasp.wiki.synthetic_case").read_wikitext() == "SYNTHETIC upper\n"
    assert [(item.title, item.reason) for item in loaded.acquisition.excluded] == [("Synthetic case", "license_exception")]
    assert loaded.acquisition.excluded[0].detail == "synthetic image credited to a third party"
    assert not any("8002" in path.name for path in (staging / "pages").iterdir())
    assert loaded.page("vasp.wiki.synthetic_case").categories == ("Category:Synthetic group 0", "Category:Synthetic hidden")


def test_report_lists_coverage_exclusions_redirects_and_checksums(tmp_path: Path, committed: Path) -> None:
    staging = tmp_path / "s"
    wiki = synthetic_wiki()
    wiki.add_page("Synthetic CRLF", 901, [FakeRevision(9001, 0, "2025-06-01T00:00:00Z", "SYNTHETIC\r\n")])
    assert fetch_all(committed, staging, wiki) == 0
    review_all(committed, staging)
    assert run(committed, "release", "--staging", str(staging), "--corpus-version", "1.0.0") == 0
    report_file = tmp_path / "report.md"
    assert run(committed, "report", str(staging), "--output", str(report_file)) == 0
    report = report_file.read_text("utf-8")
    assert "Completeness: **complete**" in report
    assert "Main-namespace titles discovered: 11" in report
    assert "Pages preserved: 10" in report and "Pages excluded: 1" in report
    assert "| Synthetic CRLF | `archival_character_policy` |" in report
    assert "| Category:Synthetic group 0 | 4 |" in report
    assert "| Synthetic Dangling | Synthetic Deleted |" in report
    assert "1 point outside the preserved pages" in report
    assert corpus.sha256_hex((staging / "SHA256SUMS").read_bytes()) in report
    assert run(committed, "report", str(staging)) == 0  # stdout form


def test_verify_upstream_reproduces_every_pinned_revision(tmp_path: Path, committed: Path, capsys) -> None:
    staging = tmp_path / "s"
    wiki = synthetic_wiki()
    assert fetch_all(committed, staging, wiki) == 0
    capsys.readouterr()
    assert run(committed, "verify-upstream", str(staging), "--delay", "0", fetcher=wiki) == 0
    assert "10 of 10 pinned revision(s) reproduced byte for byte" in capsys.readouterr().out

    # A later edit upstream does not disturb reproduction of the pinned revision.
    page_id, revisions = wiki.pages["Synthetic Page 00"]
    revisions.append(FakeRevision(9999, 7000, "2025-07-01T00:00:00Z", "SYNTHETIC edited later\n"))
    assert run(committed, "verify-upstream", str(staging), "--delay", "0", fetcher=wiki) == 0

    # Divergent upstream bytes for a pinned revision are reported.
    revisions[0] = FakeRevision(7000, 0, "2025-05-01T00:00:00Z", "SYNTHETIC tampered\n")
    capsys.readouterr()
    assert run(committed, "verify-upstream", str(staging), "--delay", "0", fetcher=wiki) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_scan_licensing_flags_markers_and_writes_unflagged_ids(tmp_path: Path, committed: Path, capsys) -> None:
    wiki = synthetic_wiki()
    wiki.add_page("Synthetic Figure", 905, [FakeRevision(9005, 0, "2025-06-01T00:00:00Z", "[[File:x.png]] (c) 2020 someone\n")])
    staging = tmp_path / "s"
    assert fetch_all(committed, staging, wiki) == 0
    capsys.readouterr()
    unflagged = tmp_path / "unflagged.txt"
    assert run(committed, "scan-licensing", "--staging", str(staging), "--write-unflagged", str(unflagged)) == 0
    out = capsys.readouterr().out
    assert "REVIEW  vasp.wiki.synthetic_figure\tSynthetic Figure\tcopyright,file" in out
    listed = tool.read_id_list(unflagged)
    assert "vasp.wiki.synthetic_figure" not in listed and len(listed) == 10


# ---------------------------------------------------------------------------
# Loader checks on full-format corpora
# ---------------------------------------------------------------------------


@pytest.fixture
def released(tmp_path: Path, committed: Path) -> Path:
    staging = tmp_path / "released"
    wiki = synthetic_wiki()
    wiki.add_page("Synthetic CRLF", 901, [FakeRevision(9001, 0, "2025-06-01T00:00:00Z", "SYNTHETIC\r\n")])
    assert fetch_all(committed, staging, wiki) == 0
    review_all(committed, staging)
    assert run(committed, "release", "--staging", str(staging), "--corpus-version", "1.0.0") == 0
    return staging


def reseal(directory: Path, mutate) -> None:
    manifest = read_manifest(directory)
    mutate(manifest)
    manifest["corpus_digest"] = corpus.compute_corpus_digest(manifest)
    (directory / "manifest.json").write_bytes(corpus.canonical_manifest_bytes(manifest))
    (directory / "NOTICE").write_bytes(corpus.render_notice(manifest))
    releases = json.loads((directory / "RELEASES.json").read_text("ascii"))
    releases["releases"][-1]["corpus_digest"] = manifest["corpus_digest"]
    (directory / "RELEASES.json").write_bytes(corpus.canonical_releases_bytes(releases))
    rewrite_sums(directory)


def unavailable(directory: Path) -> CorpusUnavailable:
    result = corpus.load_vasp_wiki_corpus(directory, policy=TEST_AUTHORITATIVE_POLICY)
    assert isinstance(result, CorpusUnavailable), result
    return result


def test_released_full_corpus_loads_deterministically(released: Path) -> None:
    first = corpus.load_vasp_wiki_corpus(released, policy=TEST_AUTHORITATIVE_POLICY)
    second = corpus.load_vasp_wiki_corpus(released, policy=TEST_AUTHORITATIVE_POLICY)
    assert isinstance(first, VaspWikiCorpus) and first == second
    assert [page.id for page in first.pages] == sorted(page.id for page in first.pages)
    assert first.acquisition.discovered_titles == 11
    assert [r.title for r in first.acquisition.redirects] == ["Synthetic Alias", "Synthetic Dangling"]
    for page in first.pages:
        assert page.read_wikitext().startswith("SYNTHETIC")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m["acquisition"].update(completeness="complete", failed=[{"title": "X", "reason": "request_failed", "detail": "d"}],
                                          discovered_titles=m["acquisition"]["discovered_titles"] + 1),
        lambda m: m["acquisition"].update(completeness="incomplete"),
        lambda m: m["acquisition"].update(discovered_titles=m["acquisition"]["discovered_titles"] + 1),
        lambda m: m["acquisition"]["excluded"][0].update(reason="not_a_reason"),
        lambda m: m["acquisition"]["excluded"].append(dict(m["acquisition"]["excluded"][0])),
        lambda m: m["acquisition"]["redirects"].reverse(),
        lambda m: m["acquisition"]["redirects"].append({"title": m["pages"][0]["title"], "target": None}),
        lambda m: m["acquisition"].update(method="explicit_titles"),
        lambda m: m["acquisition"].update(inclusion_rules="anything"),
        lambda m: m["acquisition"].update(namespace=14),
        lambda m: m["acquisition"].pop("other_namespaces"),
        lambda m: m["acquisition"].update(extra=1),
        lambda m: m["pages"][0].update(categories=["b", "a"]),
        lambda m: m["pages"][0].update(categories=["a", "a"]),
        lambda m: m["pages"][0].pop("categories"),
        lambda m: m.pop("acquisition"),
        lambda m: m.update(schema_version=3),
    ],
)
def test_inconsistent_acquisition_records_are_refused(released: Path, mutate) -> None:
    reseal(released, mutate)
    assert unavailable(released).reason == "invalid"


def test_identifier_rule_is_enforced(released: Path) -> None:
    def rename(manifest):
        page = next(p for p in manifest["pages"] if p["id"] == "vasp.wiki.synthetic_case_p801")
        page["id"] = "vasp.wiki.synthetic_case"
        page["content_file"] = corpus.content_file_for("synthetic_case", page["revision_id"])
        manifest["pages"].sort(key=lambda p: p["id"])

    old = released / "pages" / "synthetic_case_p801.r8001.wiki"
    old.rename(released / "pages" / "synthetic_case.r8001.wiki")
    reseal(released, rename)
    assert "identifier rule" in unavailable(released).detail


def test_acquisition_and_categories_are_part_of_the_identity(released: Path) -> None:
    manifest = read_manifest(released)
    digest = corpus.compute_corpus_digest(manifest)
    changed = json.loads(json.dumps(manifest))
    changed["pages"][0]["categories"] = ["Category:Other"]
    assert corpus.compute_corpus_digest(changed) != digest
    changed = json.loads(json.dumps(manifest))
    changed["acquisition"]["redirects"] = []
    assert corpus.compute_corpus_digest(changed) != digest
    changed = json.loads(json.dumps(manifest))
    changed["acquisition"]["discovered_at"] = "2030-01-01T00:00:00Z"
    assert corpus.compute_corpus_digest(changed) == digest


def test_corrupt_missing_or_extra_page_files_are_detected(released: Path) -> None:
    pages = sorted((released / "pages").iterdir())
    pages[0].write_bytes(pages[0].read_bytes() + b"x")
    assert unavailable(released).reason == "integrity"
    rewrite_sums(released)
    assert unavailable(released).reason == "integrity"
    pages[0].unlink()
    rewrite_sums(released)
    assert unavailable(released).reason == "integrity"


def test_unpopulated_anchor_cannot_be_rewritten_as_schema_v2() -> None:
    manifest = corpus.unpopulated_manifest()
    manifest["schema_version"] = 2
    manifest["acquisition"] = None
    with pytest.raises(corpus.CorpusError):
        corpus.validate_manifest(manifest, corpus.AUTHORITATIVE_POLICY, require_reviewed=True)
    assert corpus.unpopulated_manifest()["schema_version"] == 1


# ---------------------------------------------------------------------------
# Redirect responses as MediaWiki actually shapes them
# ---------------------------------------------------------------------------


def test_double_redirects_report_every_hop_and_record_the_first(tmp_path: Path, committed: Path) -> None:
    # Regression: with redirects=1 MediaWiki resolves chains hop by hop, so a
    # batch returns more redirect entries than requested titles. The old
    # parser refused this as "API redirects list is malformed".
    wiki = synthetic_wiki()
    wiki.redirects["Synthetic Chain Start"] = "Synthetic Chain Middle"
    wiki.redirects["Synthetic Chain Middle"] = "Synthetic Page 01"
    wiki.redirects["Synthetic Loop A"] = "Synthetic Loop B"
    wiki.redirects["Synthetic Loop B"] = "Synthetic Loop A"
    assert fetch_all(committed, tmp_path / "s", wiki) == 0
    redirects = {item["title"]: item["target"] for item in read_manifest(tmp_path / "s")["acquisition"]["redirects"]}
    assert redirects["Synthetic Chain Start"] == "Synthetic Chain Middle"
    assert redirects["Synthetic Chain Middle"] == "Synthetic Page 01"
    assert redirects["Synthetic Loop A"] == "Synthetic Loop B"
    assert redirects["Synthetic Loop B"] == "Synthetic Loop A"


def _redirect_response(entries):
    payload = {"batchcomplete": True, "query": {"pages": [{"ns": 0, "title": "Synthetic Page 00", "pageid": 700}]}}
    if entries is not None:
        payload["query"]["redirects"] = entries
    return json.dumps(payload).encode("utf-8")


def _resolve(entries, titles=("Synthetic A",)):
    return tool.resolve_redirects(API, list(titles), lambda url, expected: _redirect_response(entries))


def test_redirect_response_shapes_that_mediawiki_legitimately_returns() -> None:
    # Key omitted: nothing resolved (e.g. the redirect was deleted meanwhile).
    assert _resolve(None) == {"Synthetic A": None}
    assert _resolve([]) == {"Synthetic A": None}
    # A chain of hops for one requested title.
    chain = [{"from": "Synthetic A", "to": "Synthetic B"}, {"from": "Synthetic B", "to": "Synthetic C"}]
    assert _resolve(chain) == {"Synthetic A": "Synthetic B"}
    # Section redirects and interwiki redirects.
    assert _resolve([{"from": "Synthetic A", "to": "Synthetic B", "tofragment": "Section"}]) == {"Synthetic A": "Synthetic B"}
    assert _resolve([{"from": "Synthetic A", "to": "Page", "tointerwiki": "wikipedia"}]) == {"Synthetic A": "wikipedia:Page"}


@pytest.mark.parametrize(
    "entries",
    [
        {"from": "Synthetic A", "to": "Synthetic B"},  # not a list
        [["Synthetic A", "Synthetic B"]],
        [{"from": "Synthetic A"}],
        [{"from": "Synthetic A", "to": ""}],
        [{"from": "Synthetic A", "to": "Synthetic‮B"}],
        [{"from": "Synthetic A", "to": "Synthetic B", "unexpected": 1}],
        [{"from": "Synthetic A", "to": "Synthetic B", "tofragment": 3}],
        [{"from": "Synthetic A", "to": "Synthetic B"}, {"from": "Synthetic A", "to": "Synthetic C"}],
        [{"from": "Synthetic Unrelated", "to": "Synthetic B"}],
        [{"from": f"Synthetic {i}", "to": f"Synthetic {i + 1}"} for i in range(9)],
    ],
)
def test_malformed_redirect_responses_are_still_refused(entries) -> None:
    with pytest.raises(tool.FetchError):
        _resolve(entries)


# ---------------------------------------------------------------------------
# Interrupted acquisition: nothing partial is staged; restart from scratch
# ---------------------------------------------------------------------------


def test_interrupted_acquisition_stages_nothing_and_a_restart_is_identical(tmp_path: Path, committed: Path) -> None:
    reference = tmp_path / "reference"
    assert fetch_all(committed, reference, synthetic_wiki()) == 0

    wiki = synthetic_wiki()
    calls = {"n": 0}

    def interrupted(url: str, expected: str) -> bytes:
        calls["n"] += 1
        if calls["n"] == 12:
            raise KeyboardInterrupt  # the maintainer presses Ctrl+C mid-run
        return wiki(url, expected)

    staging = tmp_path / "s"
    with pytest.raises(KeyboardInterrupt):
        fetch_all(committed, staging, interrupted)
    assert not staging.exists()  # nothing is written before acquisition completes

    assert fetch_all(committed, staging, synthetic_wiki()) == 0  # restart from the beginning
    files = lambda d: {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}
    assert files(staging) == files(reference)


def test_the_resume_cache_option_no_longer_exists(tmp_path: Path, committed: Path) -> None:
    with pytest.raises(SystemExit) as refused:
        fetch_all(committed, tmp_path / "s", synthetic_wiki(), "--resume-cache", str(tmp_path / "cache"))
    assert refused.value.code == 2
    assert not (tmp_path / "cache").exists() and not (tmp_path / "s").exists()
    assert not hasattr(tool, "CachingFetcher")


# ---------------------------------------------------------------------------
# Staging writes never follow a directory link (planted or swapped in)
# ---------------------------------------------------------------------------


def _outside_directory(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_bytes(b"PRECIOUS\n")
    return outside, {"precious.txt": b"PRECIOUS\n"}


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def test_a_staging_path_that_is_a_link_is_refused(tmp_path: Path, committed: Path) -> None:
    # An empty directory elsewhere, reached through a link: previously the link
    # was resolved and the corpus written into the link target.
    outside = tmp_path / "outside-empty"
    outside.mkdir()
    staging = tmp_path / "staging-link"
    staging.symlink_to(outside, target_is_directory=True)
    assert fetch_all(committed, staging, synthetic_wiki()) == 2
    assert list(outside.iterdir()) == []


def test_a_staging_directory_swapped_for_a_link_during_fetch_is_refused(tmp_path: Path, committed: Path) -> None:
    outside, before = _outside_directory(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()  # an existing empty staging directory is accepted at the start
    wiki = synthetic_wiki()
    swapped = {"done": False}

    def swapping(url: str, expected: str) -> bytes:
        if not swapped["done"] and "allpages" in url:
            staging.rmdir()
            staging.symlink_to(outside, target_is_directory=True)
            swapped["done"] = True
        return wiki(url, expected)

    assert fetch_all(committed, staging, swapping) == 2
    assert swapped["done"] and _snapshot(outside) == before


@pytest.mark.parametrize("command", ["review", "release"])
def test_pages_directory_swapped_for_a_link_before_rewrite_is_refused(
    tmp_path: Path, committed: Path, command: str
) -> None:
    staging = tmp_path / "s"
    assert fetch_all(committed, staging, synthetic_wiki()) == 0
    if command == "release":
        review_all(committed, staging)
    outside, before = _outside_directory(tmp_path)
    pages = staging / "pages"
    for page in pages.iterdir():
        (outside / page.name).write_bytes(page.read_bytes())
    before = _snapshot(outside)
    shutil.rmtree(pages)
    pages.symlink_to(outside, target_is_directory=True)
    if command == "review":
        assert approve(committed, staging, *page_ids(staging)) == 2
    else:
        assert run(committed, "release", "--staging", str(staging), "--corpus-version", "1.0.0") == 2
    assert _snapshot(outside) == before
