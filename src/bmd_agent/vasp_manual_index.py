"""Offline passage index over the frozen VASP Wiki corpus (derived data).

The index is built in memory from the verified corpus returned by
:func:`bmd_agent.vasp_wiki_corpus.load_vasp_wiki_corpus`. It is derived data:
never written to disk, never treated as a second copy of the manual, and
rebuilt deterministically from the pinned page revisions. Every passage keeps
the page identity, pinned revision, permalink and section it came from, so a
retrieved passage can always be cited and checked against the preserved
wikitext.

Passages are paragraphs of the preserved wikitext with its markup simplified
for matching and display (``{{TAG|NELM}}`` becomes ``NELM``, link markup is
reduced to its label, unknown templates are dropped). This is not a rendering
of the wiki: content that only exists after template expansion upstream is not
available, and is never reconstructed.

Ranking is BM25 over passage text with additive field evidence (page title,
section heading, INCAR tags named in the passage, page categories). Scores and
ties are resolved deterministically (score, then page identifier, then passage
position).

This module and ``bmd_agent.vasp_wiki_corpus`` are the only code that touches
the corpus. It performs no network access, no subprocess calls and no writes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import html
import math
import re
from typing import Any

from bmd_agent import vasp_wiki_corpus as corpus_module


INDEX_ALGORITHM = "bmd-check-vasp-manual-bm25-v1"

# BM25 parameters and field weights (tuned once on the frozen corpus; part of
# the algorithm identity, so changing them changes INDEX_ALGORITHM).
_K1 = 1.2
_B = 0.75
_TITLE_WEIGHT = 2.5
_HEADING_WEIGHT = 1.2
_TAG_WEIGHT = 1.8
_CATEGORY_WEIGHT = 0.6

_MAX_PASSAGE_CHARS = 1_600
_MIN_PASSAGE_CHARS = 40

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+]*|[0-9]+")
_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")
_TAG_TEMPLATE_RE = re.compile(r"\{\{\s*(?:TAG|TAGBL|TAGDEF|tag)\s*\|\s*([A-Za-z0-9_]+)\s*(?:\|([^{}|]*))?[^{}]*\}\}")
_FILE_TEMPLATE_RE = re.compile(r"\{\{\s*(?:FILE|file)\s*\|\s*([^{}|]+?)\s*(?:\|[^{}]*)?\}\}")
_NOTE_TEMPLATE_RE = re.compile(r"\{\{\s*NB\s*\|\s*[A-Za-z]+\s*\|([^{}]*)\}\}")
_ANY_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
_CATEGORY_LINK_RE = re.compile(r"\[\[\s*Category\s*:[^\]]*\]\]", re.IGNORECASE)
_FILE_LINK_RE = re.compile(r"\[\[\s*(?:File|Image|Media)\s*:[^\]]*\]\]", re.IGNORECASE)
_INTERNAL_LINK_RE = re.compile(r"\[\[\s*:?([^\]|]+?)\s*(?:\|([^\]]*))?\]\]")
_EXTERNAL_LINK_RE = re.compile(r"\[(?:https?|ftp)://[^\s\]]+(?:\s+([^\]]*))?\]")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
_REF_RE = re.compile(r"<ref[^>]*?(?:/>|>.*?</ref>)", re.IGNORECASE | re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_EMPHASIS_RE = re.compile(r"'{2,}")
_SPACE_RE = re.compile(r"[ \t]+")

# Common English and wiki words carry no diagnostic signal.
_STOPWORDS = frozenset(
    """
    a about above after again against all also an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having here how however i if in into is it its itself just may more most much must no nor not now of
    off on once only or other our out over own same should so some such than that the their them then
    there these they this those through to too under until up us use used using very was we were what when
    where which while who whom why will with would you your see e g i e eg ie one two set sets value values
    default tag tags file files vasp
    """.split()
)


_SUFFIXES = ("ations", "ation", "ences", "ence", "ances", "ance", "ments", "ment", "ings", "ing", "ies", "ed", "es", "s", "e")


def _stem(word: str) -> str:
    """Light suffix stripping so that converge/converged/convergence match."""

    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


@lru_cache(maxsize=65_536)
def normalize_term(token: str) -> str | None:
    """Index form of one token: identifier-like tokens (INCAR tags, file names,
    numbers) are only lower-cased; ordinary words are lower-cased and stemmed."""

    lowered = token.lower()
    if lowered in _STOPWORDS:
        return None
    if token.isupper() or "_" in token or any(character.isdigit() for character in token):
        return lowered
    return _stem(lowered)


def tokenize(text: str) -> list[str]:
    """Content tokens; INCAR tag names survive as single unstemmed tokens."""

    tokens = []
    for match in _TOKEN_RE.findall(text):
        term = normalize_term(match)
        if term is not None:
            tokens.append(term)
    return tokens


_LISTING_HEADINGS = frozenset(
    {
        "related tags and articles",
        "related tags and sections",
        "related articles",
        "used incar tags",
        "incar",
        "kpoints",
        "poscar",
        "input",
        "download",
        "references",
    }
)
_ASSIGNMENT_LINE_RE = re.compile(r"^\s*[A-Z][A-Z0-9_]*\s*=")
_SENTENCE_RE = re.compile(r"(?:\b[a-z][a-z'-]+\b[\s,()]+){6,}[^.]*[.:!?]")


def passage_kind(text: str, section: str | None) -> str:
    """``prose`` (explanatory sentences), ``listing`` (input examples, tag
    lists, references) or ``other``."""

    if section and section.strip().lower() in _LISTING_HEADINGS:
        return "listing"
    lines = [line for line in text.split("\n") if line.strip()]
    if lines and sum(1 for line in lines if _ASSIGNMENT_LINE_RE.match(line)) / len(lines) >= 0.5:
        return "listing"
    words = _TOKEN_RE.findall(text)
    if words and sum(1 for word in words if word.isupper() and len(word) > 1) / len(words) > 0.5:
        return "listing"
    if _SENTENCE_RE.search(text):
        return "prose"
    return "other"


@dataclass(frozen=True)
class Passage:
    """One citable paragraph of a preserved page."""

    page_id: str
    title: str
    revision_id: int
    permalink: str
    section: str | None
    position: int
    text: str
    tags: tuple[str, ...]
    kind: str = "prose"


@dataclass(frozen=True)
class RetrievedPassage:
    passage: Passage
    score: float
    matched_terms: tuple[str, ...]


@dataclass
class _Entry:
    passage: Passage
    body: dict[str, int]
    length: int
    title: frozenset[str]
    heading: frozenset[str]
    tags: frozenset[str]
    categories: frozenset[str]


@dataclass
class ManualIndex:
    """In-memory passage index over one verified corpus release."""

    corpus_version: str
    corpus_digest: str
    index_identity: str
    pages: int
    entries: list[_Entry] = field(repr=False)
    document_frequency: dict[str, int] = field(repr=False)
    average_length: float = 0.0
    titles: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def passages(self) -> int:
        return len(self.entries)

    def idf(self, term: str) -> float:
        frequency = self.document_frequency.get(term, 0)
        count = len(self.entries)
        return math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))

    def search(
        self,
        terms: Mapping[str, float],
        *,
        required: Iterable[str] = (),
        required_groups: Sequence[Iterable[str]] = (),
        limit: int = 5,
        per_page: int = 1,
        min_score: float = 0.0,
        kinds: Iterable[str] = ("prose",),
        penalties: Iterable[str] = (),
        penalty_factor: float = 1.0,
        penalized_categories: Iterable[str] = (),
        category_penalty_factor: float = 1.0,
        title_phrases: Mapping[tuple[str, ...], float] = {},
        exclude: Callable[[Passage], bool] | None = None,
    ) -> tuple[RetrievedPassage, ...]:
        """Rank passages for weighted query ``terms``.

        ``required`` (and each of ``required_groups``) lists terms of which at least one must appear in a
        passage's text, heading or tags for it to be eligible: a passage that
        only shares common vocabulary with the query is never returned. At most
        ``per_page`` passages are returned from any one page. Passages that
        are about a ``penalties`` term (in the page title, section heading or
        INCAR tags, or at least twice in the text; methods the calculation does
        not use) have their score multiplied by ``penalty_factor``. Each of
        ``title_phrases`` whose terms all occur in a page title adds its weight
        times the terms' IDF: a page *about* the problem outranks a page that
        only mentions its settings. A passage for which ``exclude`` returns
        true is never returned (it is removed before ranking, so it cannot
        displace an eligible passage).
        """

        weights: dict[str, float] = {}
        for term, weight in terms.items():
            normal = normalize_term(term)
            if normal is not None and weight > 0:
                weights[normal] = max(weights.get(normal, 0.0), float(weight))
        groups = [
            {normal for normal in (normalize_term(term) for term in group) if normal is not None}
            for group in (tuple(required), *(tuple(group) for group in required_groups))
        ]
        groups = [group for group in groups if group]
        phrases = []
        for phrase, phrase_weight in title_phrases.items():
            normal_phrase = tuple(normal for normal in (normalize_term(term) for term in phrase) if normal is not None)
            if normal_phrase:
                phrases.append((frozenset(normal_phrase), float(phrase_weight)))
        category_terms = {
            normal for normal in (normalize_term(term) for term in penalized_categories) if normal is not None
        }
        allowed = frozenset(kinds)
        penalized = {normal for normal in (normalize_term(term) for term in penalties) if normal is not None}
        scored: list[tuple[float, str, int, _Entry, tuple[str, ...]]] = []
        for entry in self.entries:
            if entry.passage.kind not in allowed:
                continue
            if any(
                not any(anchor in entry.body or anchor in entry.heading or anchor in entry.tags for anchor in group)
                for group in groups
            ):
                continue
            if exclude is not None and exclude(entry.passage):
                continue
            score = 0.0
            matched: list[str] = []
            norm = _K1 * (1.0 - _B + _B * entry.length / (self.average_length or 1.0))
            for term, weight in weights.items():
                frequency = entry.body.get(term, 0)
                field_score = 0.0
                if frequency:
                    field_score += frequency * (_K1 + 1.0) / (frequency + norm)
                if term in entry.title:
                    field_score += _TITLE_WEIGHT
                if term in entry.heading:
                    field_score += _HEADING_WEIGHT
                if term in entry.tags:
                    field_score += _TAG_WEIGHT
                if term in entry.categories:
                    field_score += _CATEGORY_WEIGHT
                if field_score:
                    score += weight * self.idf(term) * field_score
                    matched.append(term)
            for phrase, phrase_weight in phrases:
                if phrase <= entry.title:
                    score += phrase_weight * sum(self.idf(term) for term in phrase)
                    matched.append("title:" + "+".join(sorted(phrase)))
            if penalized and any(
                entry.body.get(term, 0) >= 2 or term in entry.title or term in entry.heading or term in entry.tags
                for term in penalized
            ):
                score *= penalty_factor
            if category_terms and category_terms & entry.categories:
                score *= category_penalty_factor
            if score > min_score:
                scored.append((score, entry.passage.page_id, entry.passage.position, entry, tuple(sorted(matched))))
        scored.sort(key=lambda item: (-round(item[0], 9), item[1], item[2]))
        results: list[RetrievedPassage] = []
        per_page_counts: dict[str, int] = {}
        for score, page_id, _, entry, matched in scored:
            if per_page_counts.get(page_id, 0) >= per_page:
                continue
            per_page_counts[page_id] = per_page_counts.get(page_id, 0) + 1
            results.append(RetrievedPassage(entry.passage, round(score, 6), matched))
            if len(results) >= limit:
                break
        return tuple(results)


def simplify_wikitext(text: str) -> str:
    """Reduce wikitext markup to readable text without inventing content."""

    text = _COMMENT_RE.sub("", text)
    text = _REF_RE.sub("", text)
    text = _CATEGORY_LINK_RE.sub("", text)
    text = _FILE_LINK_RE.sub("", text)
    text = _TAG_TEMPLATE_RE.sub(lambda m: f"{m.group(1)} = {m.group(2).strip()}" if m.group(2) and m.group(2).strip() else m.group(1), text)
    text = _FILE_TEMPLATE_RE.sub(lambda m: m.group(1), text)
    for _ in range(3):
        text = _NOTE_TEMPLATE_RE.sub(lambda m: m.group(1), text)
    for _ in range(3):
        reduced = _ANY_TEMPLATE_RE.sub("", text)
        if reduced == text:
            break
        text = reduced
    text = _INTERNAL_LINK_RE.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)
    text = _EXTERNAL_LINK_RE.sub(lambda m: (m.group(1) or "").strip(), text)
    text = _HTML_TAG_RE.sub("", text)
    text = _EMPHASIS_RE.sub("", text)
    text = html.unescape(text)
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        line = re.sub(r"^[*#:;]+\s*", "", line)
        if line == "----" or line.startswith("{|") or line.startswith("|}"):
            continue
        lines.append(_SPACE_RE.sub(" ", line))
    return "\n".join(lines).strip()


def _tags_in(text: str) -> tuple[str, ...]:
    return tuple(sorted({match.group(1).upper() for match in _TAG_TEMPLATE_RE.finditer(text)}))


def extract_passages(page: Any) -> list[Passage]:
    """Paragraph passages of one preserved page, in document order."""

    wikitext = page.read_wikitext()
    passages: list[Passage] = []
    section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        raw = "\n".join(buffer).strip()
        buffer.clear()
        if not raw:
            return
        text = simplify_wikitext(raw)
        if len(text) < _MIN_PASSAGE_CHARS:
            return
        if len(text) > _MAX_PASSAGE_CHARS:
            cut = text.rfind(". ", 0, _MAX_PASSAGE_CHARS)
            text = text[: cut + 1 if cut > _MIN_PASSAGE_CHARS else _MAX_PASSAGE_CHARS].rstrip() + " ..."
        passages.append(
            Passage(
                page_id=page.id,
                title=page.title,
                revision_id=page.revision_id,
                permalink=page.permalink,
                section=section,
                position=len(passages),
                text=text,
                tags=_tags_in(raw),
                kind=passage_kind(text, section),
            )
        )

    for line in wikitext.split("\n"):
        heading = _HEADING_RE.match(line.strip())
        if heading:
            flush()
            section = simplify_wikitext(heading.group(2)) or None
            continue
        if not line.strip():
            flush()
            continue
        buffer.append(line)
    flush()
    return passages


def build_manual_index(corpus: Any) -> ManualIndex:
    """Index every passage of every preserved page (deterministic)."""

    entries: list[_Entry] = []
    frequency: dict[str, int] = {}
    titles: dict[str, str] = {}
    for page in sorted(corpus.pages, key=lambda item: item.id):
        titles[page.id] = page.title
        title_tokens = frozenset(tokenize(page.title))
        category_tokens = frozenset(
            token for category in page.categories for token in tokenize(category.split(":", 1)[-1])
        )
        for passage in extract_passages(page):
            body_tokens = tokenize(passage.text)
            counts: dict[str, int] = {}
            for token in body_tokens:
                counts[token] = counts.get(token, 0) + 1
            entry = _Entry(
                passage=passage,
                body=counts,
                length=len(body_tokens),
                title=title_tokens,
                heading=frozenset(tokenize(passage.section or "")),
                tags=frozenset(tag.lower() for tag in passage.tags),
                categories=category_tokens,
            )
            entries.append(entry)
            for term in set(counts) | entry.title | entry.heading | entry.tags:
                frequency[term] = frequency.get(term, 0) + 1
    average = sum(entry.length for entry in entries) / len(entries) if entries else 0.0
    identity = hashlib.sha256(f"{INDEX_ALGORITHM}\n{corpus.corpus_digest}".encode("ascii")).hexdigest()
    return ManualIndex(
        corpus_version=corpus.corpus_version,
        corpus_digest=corpus.corpus_digest,
        index_identity=f"sha256:{identity}",
        pages=len(corpus.pages),
        entries=entries,
        document_frequency=frequency,
        average_length=average,
        titles=titles,
    )


@dataclass(frozen=True)
class ManualIndexUnavailable:
    reason: str
    detail: str


_CACHE: dict[str, ManualIndex | ManualIndexUnavailable] = {}


def load_manual_index(
    root: Any | None = None,
    *,
    policy: Any | None = None,
) -> ManualIndex | ManualIndexUnavailable:
    """Load the verified corpus and build its index once per process.

    Never raises: an unavailable or damaged corpus yields
    ``ManualIndexUnavailable`` with the loader's closed reason.
    """

    key = "packaged" if root is None and policy is None else f"{root}|{id(policy)}"
    if key in _CACHE:
        return _CACHE[key]
    try:
        loaded = (
            corpus_module.load_vasp_wiki_corpus(root)
            if policy is None
            else corpus_module.load_vasp_wiki_corpus(root, policy=policy)
        )
        if isinstance(loaded, corpus_module.CorpusUnavailable):
            result: ManualIndex | ManualIndexUnavailable = ManualIndexUnavailable(loaded.reason, loaded.detail)
        else:
            result = build_manual_index(loaded)
    except corpus_module.CorpusIntegrityError as exc:
        result = ManualIndexUnavailable(corpus_module.UNAVAILABLE_INTEGRITY, exc.detail)
    except Exception as exc:  # fail closed: references are optional, checks are not
        result = ManualIndexUnavailable(corpus_module.UNAVAILABLE_INTERNAL_ERROR, type(exc).__name__)
    if root is None and policy is None:
        _CACHE[key] = result
    return result


def clear_cache() -> None:
    _CACHE.clear()


def passage_excerpt(passage: Passage, *, limit: int = 240) -> str:
    """A short single-line excerpt of a passage (markup already simplified)."""

    text = " ".join(passage.text.split())
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > 0 else limit].rstrip(" ,;:") + " ..."


__all__: Sequence[str] = (
    "INDEX_ALGORITHM",
    "ManualIndex",
    "ManualIndexUnavailable",
    "Passage",
    "RetrievedPassage",
    "build_manual_index",
    "clear_cache",
    "extract_passages",
    "load_manual_index",
    "passage_excerpt",
    "simplify_wikitext",
    "tokenize",
)
