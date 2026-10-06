"""Frozen VASP Wiki corpus: schema, identity, integrity and fail-closed loading.

All page content used here is synthetic (see ``vasp_wiki_support``).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess

import pytest

from bmd_agent import vasp_wiki_corpus as corpus
from bmd_agent.vasp_wiki_corpus import CorpusUnavailable, VaspWikiCorpus
import vasp_wiki_support as support
from vasp_wiki_support import (
    SYNTHETIC_PAGES,
    SYNTHETIC_POLICY,
    TEST_AUTHORITATIVE_LICENSE_TEXT,
    TEST_AUTHORITATIVE_POLICY,
    SyntheticPage,
    build_corpus,
    read_manifest,
    rewrite_sums,
    write_manifest,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "vasp_wiki_synthetic"
PACKAGED = REPO / "src" / "bmd_agent" / "reference_corpus" / "vasp_wiki"


@pytest.fixture
def synthetic(tmp_path: Path) -> Path:
    directory = tmp_path / "corpus"
    shutil.copytree(FIXTURE, directory)
    return directory


@pytest.fixture
def authoritative(tmp_path: Path) -> Path:
    directory = tmp_path / "authoritative"
    build_corpus(
        directory,
        policy=TEST_AUTHORITATIVE_POLICY,
        license_text=TEST_AUTHORITATIVE_LICENSE_TEXT,
        identifier="GFDL-1.2-only",
    )
    return directory


def unavailable(directory: Path, policy: corpus.CorpusPolicy = SYNTHETIC_POLICY) -> str:
    result = corpus.load_vasp_wiki_corpus(directory, policy=policy)
    assert isinstance(result, CorpusUnavailable), result
    return result.reason


def loaded(directory: Path, policy: corpus.CorpusPolicy = SYNTHETIC_POLICY) -> VaspWikiCorpus:
    result = corpus.load_vasp_wiki_corpus(directory, policy=policy)
    assert isinstance(result, VaspWikiCorpus), result
    return result


def reseal(directory: Path, *, version: str | None = None) -> None:
    """Re-derive digest, latest ledger entry, NOTICE and SHA256SUMS (an attacker's full rewrite)."""

    manifest = read_manifest(directory)
    if version is not None:
        manifest["corpus_version"] = version
    manifest["corpus_digest"] = corpus.compute_corpus_digest(manifest)
    (directory / corpus.MANIFEST_FILE).write_bytes(corpus.canonical_manifest_bytes(manifest))
    (directory / corpus.NOTICE_FILE).write_bytes(corpus.render_notice(manifest))
    releases = json.loads((directory / corpus.RELEASES_FILE).read_text("ascii"))
    releases["releases"][-1]["corpus_digest"] = manifest["corpus_digest"]
    releases["releases"][-1]["corpus_version"] = manifest["corpus_version"]
    (directory / corpus.RELEASES_FILE).write_bytes(corpus.canonical_releases_bytes(releases))
    rewrite_sums(directory)


def alpha_path(directory: Path) -> Path:
    return directory / "pages" / "synthetic_alpha.r101.wiki"


# ---------------------------------------------------------------------------
# Packaged corpus
# ---------------------------------------------------------------------------


def test_packaged_corpus_is_verified_and_unpopulated() -> None:
    result = corpus.load_vasp_wiki_corpus()
    assert result == CorpusUnavailable(corpus.UNAVAILABLE_UNPOPULATED, "the corpus contains no preserved pages yet")


def test_packaged_corpus_files_are_the_canonical_unpopulated_renderings() -> None:
    manifest = corpus.unpopulated_manifest()
    assert (PACKAGED / "manifest.json").read_bytes() == corpus.canonical_manifest_bytes(manifest)
    assert (PACKAGED / "NOTICE").read_bytes() == corpus.render_notice(manifest)
    assert json.loads((PACKAGED / "RELEASES.json").read_text("ascii"))["releases"] == [
        corpus.unpopulated_release_anchor()
    ]
    names = {path.relative_to(PACKAGED).as_posix() for path in PACKAGED.rglob("*") if path.is_file()}
    assert names == {"README.md", "NOTICE", "manifest.json", "RELEASES.json", "SHA256SUMS"}


def test_packaged_corpus_contains_no_vasp_material_or_license_text() -> None:
    assert not (PACKAGED / "pages").exists() or not any((PACKAGED / "pages").iterdir())
    assert not list(PACKAGED.glob("COPYING*"))
    assert read_manifest(PACKAGED)["corpus_kind"] == corpus.KIND_UNPOPULATED


def test_packaged_corpus_is_never_a_synthetic_fixture() -> None:
    assert read_manifest(PACKAGED)["corpus_kind"] in (corpus.KIND_UNPOPULATED, corpus.KIND_AUTHORITATIVE)


def test_a_populated_packaged_corpus_must_load_with_a_pinned_license_digest() -> None:
    """Invariant for the population PR: authoritative content ships only with a pinned license."""

    if read_manifest(PACKAGED)["corpus_kind"] == corpus.KIND_UNPOPULATED:
        assert isinstance(corpus.load_vasp_wiki_corpus(), CorpusUnavailable)
        return
    assert corpus.AUTHORITATIVE_POLICY.license_text_sha256 is not None
    assert isinstance(corpus.load_vasp_wiki_corpus(), VaspWikiCorpus)


def test_default_policy_accepts_only_official_hosts_and_gfdl_1_2() -> None:
    policy = corpus.AUTHORITATIVE_POLICY
    assert policy.corpus_kind == corpus.KIND_AUTHORITATIVE
    assert policy.allowed_hosts == frozenset({"vasp.at", "www.vasp.at"})
    assert policy.allowed_license_identifiers == frozenset({"GFDL-1.2-only", "GFDL-1.2-or-later"})
    assert policy.license_text_file == "COPYING.GFDL-1.2.txt"


# ---------------------------------------------------------------------------
# Synthetic fixture and determinism
# ---------------------------------------------------------------------------


def test_default_loader_refuses_the_synthetic_fixture() -> None:
    assert unavailable(FIXTURE, corpus.AUTHORITATIVE_POLICY) == corpus.UNAVAILABLE_KIND_NOT_PERMITTED


def test_synthetic_fixture_loads_under_its_own_policy_with_verbatim_pages() -> None:
    result = loaded(FIXTURE)
    assert result.corpus_kind == corpus.KIND_SYNTHETIC_FIXTURE
    assert [page.id for page in result.pages] == ["vasp.wiki.synthetic_alpha", "vasp.wiki.synthetic_old_name"]
    for page, source in zip(result.pages, SYNTHETIC_PAGES):
        assert page.read_wikitext().encode("utf-8") == source.content
        assert page.read_wikitext().encode("utf-8") == (FIXTURE / page.content_file).read_bytes()
    beta = result.page("vasp.wiki.synthetic_old_name")
    assert beta is not None and beta.redirected_from == "Synthetic Old Name" and beta.title == "Synthetic Beta"
    assert result.page("vasp.wiki.missing") is None


def test_synthetic_fixture_provenance_is_explicit() -> None:
    assert b"SYNTHETIC" in (FIXTURE / "README.md").read_bytes()
    assert b"SYNTHETIC TEST FIXTURE" in (FIXTURE / "NOTICE").read_bytes()
    manifest = read_manifest(FIXTURE)
    assert manifest["corpus_kind"] == corpus.KIND_SYNTHETIC_FIXTURE
    assert manifest["upstream"]["server"] == "https://synthetic.invalid"
    for page in manifest["pages"]:
        assert page["title"].startswith("Synthetic")
        assert b"SYNTHETIC" in (FIXTURE / page["content_file"]).read_bytes()


def test_regenerating_the_synthetic_fixture_is_byte_identical(tmp_path: Path) -> None:
    support.build_synthetic_fixture(tmp_path / "again")
    expected = {p.relative_to(FIXTURE).as_posix(): p.read_bytes() for p in FIXTURE.rglob("*") if p.is_file()}
    actual = {
        p.relative_to(tmp_path / "again").as_posix(): p.read_bytes()
        for p in (tmp_path / "again").rglob("*")
        if p.is_file()
    }
    assert actual == expected


def test_loaded_records_are_frozen() -> None:
    result = loaded(FIXTURE)
    with pytest.raises(FrozenInstanceError):
        result.corpus_version = "9.9.9"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.pages[0].revision_id = 1  # type: ignore[misc]
    assert isinstance(result.pages, tuple)


def test_markup_like_content_is_returned_as_inert_text() -> None:
    text = loaded(FIXTURE).pages[0].read_wikitext()
    assert "<script>alert('inert text, never executed')</script>" in text
    assert "{{SyntheticTemplate|value=1}}" in text


def test_authoritative_corpus_loads_only_with_its_pinned_license(authoritative: Path) -> None:
    result = loaded(authoritative, TEST_AUTHORITATIVE_POLICY)
    assert result.license.identifier == "GFDL-1.2-only"
    assert result.license.license_text_sha256 == hashlib.sha256(TEST_AUTHORITATIVE_LICENSE_TEXT).hexdigest()
    assert all(page.permalink.endswith(f"oldid={page.revision_id}") for page in result.pages)
    # The shipped policy has no pinned license digest yet, so it fails closed.
    assert unavailable(authoritative, corpus.AUTHORITATIVE_POLICY) == corpus.UNAVAILABLE_LICENSE
    other = replace(TEST_AUTHORITATIVE_POLICY, license_text_sha256="0" * 64)
    assert unavailable(authoritative, other) == corpus.UNAVAILABLE_LICENSE


# ---------------------------------------------------------------------------
# Identity and the release ledger
# ---------------------------------------------------------------------------


def test_identity_ignores_provenance_but_binds_content() -> None:
    manifest = read_manifest(FIXTURE)
    digest = corpus.compute_corpus_digest(manifest)
    assert digest == manifest["corpus_digest"]

    provenance = json.loads(json.dumps(manifest))
    provenance["assembly"]["assembled_at"] = "2030-01-01T00:00:00Z"
    provenance["upstream"]["generator"] = "MediaWiki 9.9"
    provenance["pages"][0]["retrieved_at"] = "2030-01-01T00:00:00Z"
    provenance["pages"][0]["license_review"]["reviewer"] = "Someone Else"
    provenance["license"]["review"]["reviewed_on"] = "2030-01-01"
    provenance["corpus_version"] = "7.0.0"
    assert corpus.compute_corpus_digest(provenance) == digest

    for mutate in (
        lambda m: m["pages"][0].__setitem__("content_sha256", "f" * 64),
        lambda m: m["pages"][0].__setitem__("revision_id", 102),
        lambda m: m["pages"][0].__setitem__("upstream_sha1", "f" * 40),
        lambda m: m["upstream"].__setitem__("rights_text", "SYNTHETIC changed"),
        lambda m: m["license"].__setitem__("license_text_sha256", "f" * 64),
        lambda m: m["pages"][0]["license_review"].__setitem__("status", "pending"),
    ):
        changed = json.loads(json.dumps(manifest))
        mutate(changed)
        assert corpus.compute_corpus_digest(changed) != digest


def test_changed_content_cannot_masquerade_as_the_reviewed_release(synthetic: Path) -> None:
    original_entries = json.loads((synthetic / corpus.RELEASES_FILE).read_text("ascii"))["releases"]
    tampered = b"SYNTHETIC TEST FIXTURE: tampered content\n"
    alpha_path(synthetic).write_bytes(tampered)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INTEGRITY  # SHA256SUMS

    rewrite_sums(synthetic)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INTEGRITY  # manifest size / SHA-256

    manifest = read_manifest(synthetic)
    page = manifest["pages"][0]
    page["content_bytes"] = len(tampered)
    page["content_sha256"] = hashlib.sha256(tampered).hexdigest()
    write_manifest(synthetic, manifest)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INTEGRITY  # corpus_digest no longer matches

    page["upstream_sha1"] = hashlib.sha1(tampered).hexdigest()
    write_manifest(synthetic, manifest, recompute_digest=True)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INTEGRITY  # not the recorded release

    # Only rewriting recorded release history makes it load, and that is
    # detectable: the new ledger no longer extends the reviewed one.
    reseal(synthetic)
    assert isinstance(corpus.load_vasp_wiki_corpus(synthetic, policy=SYNTHETIC_POLICY), VaspWikiCorpus)
    rewritten = json.loads((synthetic / corpus.RELEASES_FILE).read_text("ascii"))["releases"]
    assert not corpus.releases_extend(original_entries, rewritten)


def test_a_new_version_needs_a_new_ledger_entry(synthetic: Path) -> None:
    manifest = read_manifest(synthetic)
    manifest["corpus_version"] = "1.0.1"
    write_manifest(synthetic, manifest)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INTEGRITY


def test_properly_appended_release_loads_and_extends_history(synthetic: Path) -> None:
    before = json.loads((synthetic / corpus.RELEASES_FILE).read_text("ascii"))["releases"]
    manifest = read_manifest(synthetic)
    manifest["pages"][0]["revision_timestamp"] = "2026-02-02T00:00:00Z"
    manifest["corpus_version"] = "1.1.0"
    manifest["corpus_digest"] = corpus.compute_corpus_digest(manifest)
    releases = {"schema": corpus.RELEASES_SCHEMA, "schema_version": 1, "releases": before + [
        {
            "corpus_version": "1.1.0",
            "corpus_kind": corpus.KIND_SYNTHETIC_FIXTURE,
            "corpus_digest": manifest["corpus_digest"],
            "previous_corpus_digest": before[-1]["corpus_digest"],
        }
    ]}
    (synthetic / corpus.RELEASES_FILE).write_bytes(corpus.canonical_releases_bytes(releases))
    write_manifest(synthetic, manifest)
    assert loaded(synthetic).corpus_version == "1.1.0"
    assert corpus.releases_extend(before, releases["releases"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["releases"].pop(0),
        lambda r: r["releases"][0].__setitem__("corpus_digest", "sha256:" + "0" * 64),
        lambda r: r["releases"][1].__setitem__("previous_corpus_digest", None),
        lambda r: r["releases"][1].__setitem__("corpus_version", "0.0.0"),
        lambda r: r["releases"].append(dict(r["releases"][1])),
        lambda r: r["releases"][1].__setitem__("extra", 1),
        lambda r: r.__setitem__("schema_version", 2),
        lambda r: r.__setitem__("releases", []),
    ],
)
def test_release_ledger_rules_are_enforced(synthetic: Path, mutate) -> None:
    releases = json.loads((synthetic / corpus.RELEASES_FILE).read_text("ascii"))
    mutate(releases)
    (synthetic / corpus.RELEASES_FILE).write_bytes(corpus.canonical_releases_bytes(releases))
    rewrite_sums(synthetic)
    assert unavailable(synthetic) in (corpus.UNAVAILABLE_INVALID, corpus.UNAVAILABLE_INTEGRITY)


def test_release_ledger_is_append_only_relative_to_origin_main() -> None:
    relative = "src/bmd_agent/reference_corpus/vasp_wiki/RELEASES.json"
    try:
        base = subprocess.run(
            ["git", "show", f"origin/main:{relative}"],
            cwd=REPO,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is unavailable")
    if base.returncode != 0:
        pytest.skip("origin/main has no recorded corpus ledger to compare with")
    base_entries = corpus.validate_releases(json.loads(base.stdout))
    current_entries = corpus.validate_releases(json.loads((REPO / relative).read_text("ascii")))
    assert corpus.releases_extend(base_entries, current_entries)


# ---------------------------------------------------------------------------
# SHA256SUMS and file set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda text: text.replace("  ", " ", 1),
        lambda text: "\n".join(reversed(text.rstrip("\n").split("\n"))) + "\n",
        lambda text: text + text.split("\n")[0] + "\n",
        lambda text: text + "0" * 64 + "  ../escape\n",
        lambda text: text + "0" * 64 + "  /etc/passwd\n",
        lambda text: text + "0" * 64 + "  pages/../../escape.r1.wiki\n",
        lambda text: text + "0" * 64 + "  SHA256SUMS\n",
        lambda text: text.rstrip("\n"),
        lambda text: text + "0" * 64 + "  café\n",
    ],
)
def test_malformed_sha256sums_is_rejected(synthetic: Path, mutate) -> None:
    sums = synthetic / corpus.SHA256SUMS_FILE
    sums.write_bytes(mutate(sums.read_text("ascii")).encode("utf-8"))
    assert unavailable(synthetic) in (corpus.UNAVAILABLE_INVALID, corpus.UNAVAILABLE_INTEGRITY)


@pytest.mark.parametrize("extra", ["stray.txt", "pages/stray.r1.wiki", ".hidden"])
def test_unlisted_files_are_rejected(synthetic: Path, extra: str) -> None:
    (synthetic / extra).write_bytes(b"x")
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INTEGRITY
    rewrite_sums(synthetic)
    assert unavailable(synthetic) in (corpus.UNAVAILABLE_INVALID, corpus.UNAVAILABLE_INTEGRITY)


def test_unexpected_directory_is_rejected(synthetic: Path) -> None:
    (synthetic / "nested").mkdir()
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INTEGRITY


def test_missing_listed_file_is_rejected(synthetic: Path) -> None:
    alpha_path(synthetic).unlink()
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INTEGRITY


def test_symlinked_page_is_rejected(synthetic: Path, tmp_path: Path) -> None:
    target = tmp_path / "outside.wiki"
    target.write_bytes(alpha_path(synthetic).read_bytes())
    alpha_path(synthetic).unlink()
    alpha_path(synthetic).symlink_to(target)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INTEGRITY


def test_notice_cannot_drift_from_the_manifest(synthetic: Path) -> None:
    notice = synthetic / corpus.NOTICE_FILE
    notice.write_bytes(notice.read_bytes().replace(b"Synthetic Test Wiki", b"Forged Attribution"))
    rewrite_sums(synthetic)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INTEGRITY


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


def _mutated(synthetic: Path, mutate) -> str:
    manifest = read_manifest(synthetic)
    mutate(manifest)
    write_manifest(synthetic, manifest)
    return unavailable(synthetic)


@pytest.mark.parametrize(
    "path",
    [(), ("assembly",), ("assembly", "updater"), ("upstream",), ("license",), ("license", "review"),
     ("pages", 0), ("pages", 0, "license_review")],
)
def test_unknown_fields_are_rejected_at_every_level(synthetic: Path, path) -> None:
    def mutate(manifest):
        node = manifest
        for key in path:
            node = node[key]
        node["unexpected"] = None

    assert _mutated(synthetic, mutate) == corpus.UNAVAILABLE_INVALID


@pytest.mark.parametrize(
    "path,key",
    [((), "pages"), ((), "corpus_digest"), (("upstream",), "rights_text"), (("license",), "review"),
     (("pages", 0), "upstream_sha1"), (("pages", 0), "permalink"), (("pages", 0, "license_review"), "notes")],
)
def test_missing_fields_are_rejected(synthetic: Path, path, key) -> None:
    def mutate(manifest):
        node = manifest
        for item in path:
            node = node[item]
        del node[key]

    assert _mutated(synthetic, mutate) in (corpus.UNAVAILABLE_INVALID, corpus.UNAVAILABLE_LICENSE)


@pytest.mark.parametrize(
    "raw",
    [
        lambda data: data.replace(b'  "', b'   "', 1),
        lambda data: data.rstrip(b"\n"),
        lambda data: b'{"schema": "a", "schema": "b"}\n',
        lambda data: data.replace(b'"namespace": 0', b'"namespace": NaN', 1),
        lambda data: data.replace(b"Synthetic Alpha", "Synthétic".encode("utf-8"), 1),
        lambda data: b"\x00\xff garbage",
        lambda data: b"[]\n",
        lambda data: b"",
    ],
)
def test_non_canonical_or_malformed_manifest_bytes_are_rejected(synthetic: Path, raw) -> None:
    data = (synthetic / corpus.MANIFEST_FILE).read_bytes()
    write_manifest(synthetic, {}, raw=raw(data))
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INVALID


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.__setitem__("schema", "other"),
        lambda m: m.__setitem__("schema_version", True),
        lambda m: m.__setitem__("corpus_version", "1.0"),
        lambda m: m.__setitem__("corpus_version", "0.0.0"),
        lambda m: m.__setitem__("corpus_digest", "sha256:xyz"),
        lambda m: m.__setitem__("corpus_kind", "something_else"),
        lambda m: m.__setitem__("pages", []),
        lambda m: m.__setitem__("pages", {}),
        lambda m: m["assembly"].__setitem__("assembled_at", "yesterday"),
        lambda m: m["upstream"].__setitem__("generator", "Not a wiki"),
        lambda m: m["upstream"].__setitem__("script_path", "/../index.php"),
        lambda m: m["upstream"].__setitem__("server", "https://synthetic.invalid/path"),
        lambda m: m["upstream"].__setitem__("api_endpoint", "https://synthetic.invalid/api.php?x=1"),
        lambda m: m["upstream"].__setitem__("article_path", "wiki"),
        lambda m: m["pages"][0].__setitem__("id", "vasp.wiki.Bad"),
        lambda m: m["pages"][0].__setitem__("id", "vasp.wiki.other"),
        lambda m: m["pages"][0].__setitem__("requested_title", "Synthetic Gamma"),
        lambda m: m["pages"][0].__setitem__("title", "Bad\ttitle"),
        lambda m: m["pages"][0].__setitem__("title", "Bidi ‮ title"),
        lambda m: m["pages"][0].__setitem__("title", "Escape \x1b[2J title"),
        lambda m: m["pages"][0].__setitem__("title", " padded"),
        lambda m: m["pages"][0].__setitem__("page_id", 0),
        lambda m: m["pages"][0].__setitem__("page_id", True),
        lambda m: m["pages"][0].__setitem__("namespace", 1),
        lambda m: m["pages"][0].__setitem__("revision_id", 0),
        lambda m: m["pages"][0].__setitem__("revision_id", -101),
        lambda m: m["pages"][0].__setitem__("revision_id", "101"),
        lambda m: m["pages"][0].__setitem__("parent_revision_id", 101),
        lambda m: m["pages"][0].__setitem__("parent_revision_id", 0),
        lambda m: m["pages"][0].__setitem__("revision_timestamp", "2026-01-02 03:04:05"),
        lambda m: m["pages"][0].__setitem__("content_model", "html"),
        lambda m: m["pages"][0].__setitem__("content_file", "pages/../../escape.r101.wiki"),
        lambda m: m["pages"][0].__setitem__("content_file", "/etc/passwd"),
        lambda m: m["pages"][0].__setitem__("content_file", "pages/synthetic_alpha.r102.wiki"),
        lambda m: m["pages"][0].__setitem__("content_file", "pages/synthetic_old_name.r202.wiki"),
        lambda m: m["pages"][0].__setitem__("content_bytes", -1),
        lambda m: m["pages"][0].__setitem__("content_bytes", corpus.MAX_PAGE_BYTES + 1),
        lambda m: m["pages"][0].__setitem__("content_sha256", "F" * 64),
        lambda m: m["pages"][0].__setitem__("upstream_sha1", "abc"),
        lambda m: m["pages"][0]["license_review"].__setitem__("status", "maybe"),
        lambda m: m["pages"][0]["license_review"].__setitem__("reviewed_on", "02/01/2026"),
    ],
)
def test_schema_violations_are_rejected(synthetic: Path, mutate) -> None:
    assert _mutated(synthetic, mutate) in (corpus.UNAVAILABLE_INVALID, corpus.UNAVAILABLE_KIND_NOT_PERMITTED)


@pytest.mark.parametrize(
    "url",
    [
        "http://synthetic.invalid/wiki/X",
        "https://evil.example/wiki/X",
        "https://user:pw@synthetic.invalid/wiki/X",
        "https://synthetic.invalid:8443/wiki/X",
        "https://SYNTHETIC.invalid/wiki/X",
        "https://synthetic.invalid/wiki/X?oldid=101",
        "https://synthetic.invalid/wiki/X Y",
        "javascript:alert(1)",
        "https://synthetic.invalid\\@evil.example/",
        "https:///wiki/X",
    ],
)
def test_bad_canonical_urls_are_rejected(synthetic: Path, url: str) -> None:
    assert _mutated(synthetic, lambda m: m["pages"][0].__setitem__("canonical_url", url)) == corpus.UNAVAILABLE_INVALID


@pytest.mark.parametrize(
    "key,url",
    [
        ("permalink", "https://synthetic.invalid/wiki/index.php?title=Synthetic_Alpha&oldid=100"),
        ("permalink", "https://synthetic.invalid/wiki/index.php?title=Synthetic_Alpha&oldid=101&oldid=1"),
        ("permalink", "https://evil.example/wiki/index.php?title=Synthetic_Alpha&oldid=101"),
        ("permalink", "https://synthetic.invalid/other.php?title=Synthetic_Alpha&oldid=101"),
        ("history_url", "https://synthetic.invalid/wiki/index.php?title=Synthetic_Alpha"),
        ("history_url", "https://evil.example/wiki/index.php?title=Synthetic_Alpha&action=history"),
    ],
)
def test_revision_and_history_urls_must_match_the_pinned_revision(synthetic: Path, key: str, url: str) -> None:
    assert _mutated(synthetic, lambda m: m["pages"][0].__setitem__(key, url)) == corpus.UNAVAILABLE_INVALID


@pytest.mark.parametrize("key", ["id", "requested_title", "title", "page_id", "revision_id"])
def test_duplicate_page_identity_is_rejected(synthetic: Path, key: str) -> None:
    def mutate(manifest):
        manifest["pages"][1][key] = manifest["pages"][0][key]

    assert _mutated(synthetic, mutate) == corpus.UNAVAILABLE_INVALID


def test_pages_must_be_sorted(synthetic: Path) -> None:
    assert _mutated(synthetic, lambda m: m["pages"].reverse()) == corpus.UNAVAILABLE_INVALID


# ---------------------------------------------------------------------------
# Licensing fails closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m["license"]["review"].update(status="pending"),
        lambda m: m["license"]["review"].update(reviewer=None),
        lambda m: m["license"].__setitem__("identifier", "MIT"),
        lambda m: m["license"].__setitem__("identifier", None),
        lambda m: m["license"].__setitem__("site_notice_as_displayed", None),
        lambda m: m["license"].__setitem__("invariant_sections", "Section A"),
        lambda m: m["license"].__setitem__("cover_texts", None),
        lambda m: m["license"].__setitem__("license_text_file", "COPYING.other"),
        lambda m: m["upstream"].__setitem__("rights_text", "All rights reserved"),
        lambda m: m["pages"][0]["license_review"].update(status="pending", reviewer=None, reviewed_on=None),
        lambda m: m["pages"][0]["license_review"].update(status="exception_found", notes="image is CC-BY-NC"),
        lambda m: m["pages"][0]["license_review"].update(reviewer=None),
    ],
)
def test_incomplete_or_inconsistent_licensing_fails_closed(synthetic: Path, mutate) -> None:
    assert _mutated(synthetic, mutate) == corpus.UNAVAILABLE_LICENSE


def test_unreviewed_page_cannot_load_even_when_fully_resealed(synthetic: Path) -> None:
    manifest = read_manifest(synthetic)
    manifest["pages"][0]["license_review"] = {"status": "pending", "reviewer": None, "reviewed_on": None, "notes": None}
    write_manifest(synthetic, manifest)
    reseal(synthetic)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_LICENSE


def test_replaced_license_text_is_rejected_even_when_resealed(synthetic: Path) -> None:
    license_file = synthetic / SYNTHETIC_POLICY.license_text_file
    license_file.write_bytes(b"SYNTHETIC forged license\n")
    manifest = read_manifest(synthetic)
    manifest["license"]["license_text_sha256"] = hashlib.sha256(b"SYNTHETIC forged license\n").hexdigest()
    write_manifest(synthetic, manifest)
    reseal(synthetic)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_LICENSE


def test_missing_license_text_is_rejected(synthetic: Path) -> None:
    (synthetic / SYNTHETIC_POLICY.license_text_file).unlink()
    rewrite_sums(synthetic)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INTEGRITY


# ---------------------------------------------------------------------------
# Archival character policy and size limits
# ---------------------------------------------------------------------------


def _with_page_content(directory: Path, content: bytes) -> Path:
    page = SyntheticPage("Synthetic Alpha", "Synthetic Alpha", 11, 101, None, content)
    build_corpus(directory, pages=(page,))
    return directory


@pytest.mark.parametrize(
    "character",
    ["\x00", "\x1b", "\r", "\x07", "\x7f", "\x85", "\x9b", "؜", "​", "‍", "‎",
     " ", "‪", "‮", "⁦", "⁩", "﻿", "￾", "﷐", "\U0001fffe",
     "\U000e0001", "\U000e0041"],
)
def test_disallowed_characters_in_pages_are_refused(tmp_path: Path, character: str) -> None:
    content = f"SYNTHETIC before{character}after\n".encode("utf-8")
    assert unavailable(_with_page_content(tmp_path / "c", content)) == corpus.UNAVAILABLE_INTEGRITY


def test_crlf_line_endings_are_refused_not_normalized(tmp_path: Path) -> None:
    assert unavailable(_with_page_content(tmp_path / "c", b"SYNTHETIC\r\nline\r\n")) == corpus.UNAVAILABLE_INTEGRITY


@pytest.mark.parametrize("data", [b"SYNTHETIC \xff\xfe", b"SYNTHETIC \xed\xa0\x80", b"SYNTHETIC \xc3"])
def test_invalid_utf8_pages_are_refused(tmp_path: Path, data: bytes) -> None:
    assert unavailable(_with_page_content(tmp_path / "c", data)) == corpus.UNAVAILABLE_INTEGRITY


def test_ordinary_unicode_and_empty_pages_are_accepted(tmp_path: Path) -> None:
    content = "SYNTHETIC éåΩ 中文 \U0001f600  nbsp\n".encode("utf-8")
    assert loaded(_with_page_content(tmp_path / "a", content)).pages[0].read_wikitext().encode() == content
    assert loaded(_with_page_content(tmp_path / "b", b"")).pages[0].read_wikitext() == ""


def test_oversized_page_is_refused(tmp_path: Path, monkeypatch) -> None:
    directory = _with_page_content(tmp_path / "c", b"SYNTHETIC " + b"x" * 200)
    monkeypatch.setattr(corpus, "MAX_PAGE_BYTES", 100)
    assert unavailable(directory) in (corpus.UNAVAILABLE_INVALID, corpus.UNAVAILABLE_INTEGRITY)


def test_total_page_size_limit_is_enforced(synthetic: Path, monkeypatch) -> None:
    monkeypatch.setattr(corpus, "MAX_TOTAL_PAGE_BYTES", 50)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INVALID


def test_oversized_manifest_is_refused(synthetic: Path, monkeypatch) -> None:
    monkeypatch.setattr(corpus, "MAX_MANIFEST_BYTES", 64)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INTEGRITY


# ---------------------------------------------------------------------------
# Graceful degradation and lazy re-verification
# ---------------------------------------------------------------------------


def test_missing_corpus_is_reported_not_raised(tmp_path: Path) -> None:
    assert unavailable(tmp_path / "absent") == corpus.UNAVAILABLE_MISSING
    (tmp_path / "empty").mkdir()
    assert unavailable(tmp_path / "empty") == corpus.UNAVAILABLE_MISSING


def test_missing_sha256sums_is_integrity_failure(synthetic: Path) -> None:
    (synthetic / corpus.SHA256SUMS_FILE).unlink()
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INTEGRITY


def test_loader_never_raises_on_random_corruption(synthetic: Path) -> None:
    generator = random.Random(1234)
    files = sorted(path for path in synthetic.rglob("*") if path.is_file())
    for _ in range(60):
        target = generator.choice(files)
        original = target.read_bytes()
        corrupted = bytearray(original or b"\x00")
        corrupted[generator.randrange(len(corrupted))] = generator.randrange(256)
        target.write_bytes(bytes(corrupted))
        if generator.random() < 0.5:
            rewrite_sums(synthetic)
        result = corpus.load_vasp_wiki_corpus(synthetic, policy=SYNTHETIC_POLICY)
        assert isinstance(result, (CorpusUnavailable, VaspWikiCorpus))
        target.write_bytes(original)
        rewrite_sums(synthetic)
    assert isinstance(corpus.load_vasp_wiki_corpus(synthetic, policy=SYNTHETIC_POLICY), VaspWikiCorpus)


def test_unexpected_internal_failure_becomes_unavailable(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(corpus, "_load", explode)
    assert corpus.load_vasp_wiki_corpus() == CorpusUnavailable(
        corpus.UNAVAILABLE_INTERNAL_ERROR, "corpus loading failed: RuntimeError"
    )


def test_page_tampered_after_loading_is_detected_on_read(synthetic: Path) -> None:
    result = loaded(synthetic)
    alpha_path(synthetic).write_bytes(b"SYNTHETIC replaced after load\n")
    with pytest.raises(corpus.CorpusIntegrityError):
        result.pages[0].read_wikitext()


def test_unavailable_details_never_echo_page_content(tmp_path: Path) -> None:
    secret = "SYNTHETIC-UNIQUE-MARKER"
    directory = _with_page_content(tmp_path / "c", f"{secret}\x1b[31m".encode())
    result = corpus.load_vasp_wiki_corpus(directory, policy=SYNTHETIC_POLICY)
    assert isinstance(result, CorpusUnavailable)
    assert secret not in result.detail and "\x1b" not in result.detail


def test_upstream_sha1_mismatch_alone_is_refused(synthetic: Path) -> None:
    manifest = read_manifest(synthetic)
    manifest["pages"][0]["upstream_sha1"] = "0" * 40
    write_manifest(synthetic, manifest)
    reseal(synthetic)
    result = corpus.load_vasp_wiki_corpus(synthetic, policy=SYNTHETIC_POLICY)
    assert result == CorpusUnavailable(
        corpus.UNAVAILABLE_INTEGRITY,
        "pages/synthetic_alpha.r101.wiki does not match the upstream MediaWiki SHA-1",
    )


def test_symlinked_pages_directory_is_rejected(synthetic: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside-pages"
    shutil.move(str(synthetic / "pages"), outside)
    (synthetic / "pages").symlink_to(outside, target_is_directory=True)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INTEGRITY


def test_corpus_bytes_are_protected_from_line_ending_conversion() -> None:
    rules = (REPO / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "src/bmd_agent/reference_corpus/vasp_wiki/** -text" in rules
    assert "tests/fixtures/vasp_wiki_synthetic/** -text" in rules


# ---------------------------------------------------------------------------
# Page license review notes are semantically closed
# ---------------------------------------------------------------------------


def _set_review(directory: Path, status: str, notes, *, reviewed: bool = True) -> None:
    manifest = read_manifest(directory)
    manifest["pages"][0]["license_review"] = {
        "status": status,
        "reviewer": "Synthetic Reviewer" if reviewed else None,
        "reviewed_on": "2026-01-02" if reviewed else None,
        "notes": notes,
    }
    write_manifest(directory, manifest)
    reseal(directory)


@pytest.mark.parametrize(
    "status,notes,reviewed,expected",
    [
        ("pending", None, False, corpus.UNAVAILABLE_LICENSE),
        ("pending", "awaiting check of figure licences", False, corpus.UNAVAILABLE_INVALID),
        ("reviewed_no_exceptions", None, True, None),
        ("reviewed_no_exceptions", "Figure 2 is licensed separately; not GFDL", True, corpus.UNAVAILABLE_INVALID),
        ("reviewed_no_exceptions", "", True, corpus.UNAVAILABLE_INVALID),
        ("exception_found", None, True, corpus.UNAVAILABLE_INVALID),
        ("exception_found", "", True, corpus.UNAVAILABLE_INVALID),
        ("exception_found", "   ", True, corpus.UNAVAILABLE_INVALID),
        ("exception_found", " padded note", True, corpus.UNAVAILABLE_INVALID),
        ("exception_found", "Figure 2 is CC-BY-NC, not GFDL", True, corpus.UNAVAILABLE_LICENSE),
    ],
)
def test_review_notes_are_allowed_only_for_recorded_exceptions(
    synthetic: Path, status: str, notes, reviewed: bool, expected: str | None
) -> None:
    _set_review(synthetic, status, notes, reviewed=reviewed)
    if expected is None:
        assert loaded(synthetic).pages[0].license_review.notes is None
    else:
        assert unavailable(synthetic) == expected


def test_meaningful_exception_note_is_structurally_valid_but_never_loads(synthetic: Path) -> None:
    _set_review(synthetic, "exception_found", "Figure 2 is CC-BY-NC, not GFDL")
    manifest = read_manifest(synthetic)
    corpus.validate_manifest(manifest, SYNTHETIC_POLICY, require_reviewed=False)
    with pytest.raises(corpus.CorpusError) as excinfo:
        corpus.validate_manifest(manifest, SYNTHETIC_POLICY, require_reviewed=True)
    assert excinfo.value.reason == corpus.UNAVAILABLE_LICENSE


def test_codex_contradictory_note_attack_is_refused(synthetic: Path) -> None:
    """Add a caveat to a no-exceptions review, keep digest and ledger, recompute everything else."""

    original = read_manifest(synthetic)
    manifest = json.loads(json.dumps(original))
    assert manifest["pages"][0]["license_review"]["status"] == "reviewed_no_exceptions"
    manifest["pages"][0]["license_review"]["notes"] = "Figure 2 is licensed separately; not GFDL"
    (synthetic / corpus.MANIFEST_FILE).write_bytes(corpus.canonical_manifest_bytes(manifest))
    (synthetic / corpus.NOTICE_FILE).write_bytes(corpus.render_notice(manifest))
    rewrite_sums(synthetic)
    # The notes are not part of the identity, so digest and ledger still match...
    assert corpus.compute_corpus_digest(manifest) == original["corpus_digest"] == manifest["corpus_digest"]
    # ...but the review state is contradictory and the corpus is refused.
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INVALID
    # A full reseal by the attacker does not help either.
    reseal(synthetic)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INVALID


@pytest.mark.parametrize(
    "notes",
    ["Resolved: actually GFDL", "none", "N/A", "No exception after all", "Figure removed upstream"],
)
def test_editing_exception_notes_cannot_make_a_page_authoritative(synthetic: Path, notes: str) -> None:
    _set_review(synthetic, "exception_found", "Figure 2 is CC-BY-NC, not GFDL")
    assert unavailable(synthetic) == corpus.UNAVAILABLE_LICENSE
    _set_review(synthetic, "exception_found", notes)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_LICENSE
    # Flipping the status while keeping any caveat is structurally invalid.
    _set_review(synthetic, "reviewed_no_exceptions", notes)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INVALID
    # Dropping the note while keeping the exception is structurally invalid.
    _set_review(synthetic, "exception_found", None)
    assert unavailable(synthetic) == corpus.UNAVAILABLE_INVALID
