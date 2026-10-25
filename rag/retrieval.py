"""Finds the pieces of text that best answer a question.

How it works, step by step:

1. **Keyword search (BM25)** -- good at exact words and names ("HOD",
   "Punit", "16241203"). Student shorthand is widened first using
   ``config.QUERY_SYNONYMS`` (hod -> head of department, dean, ...).
2. **Meaning search (vectors/FAISS)** -- good at paraphrases ("who leads
   the department" ~ "Professor & Head").
3. **Fusion** -- both candidate lists are merged with Reciprocal Rank
   Fusion: a chunk ranked high by either method rises to the top.
4. **Reranking** (optional) -- a cross-encoder reads question + chunk
   together and scores how well the chunk actually answers it.
5. **Recency boost** -- each chunk has a date (see ingestion/dates.py).
   Newer chunks get a small bonus, so when two chunks are about equally
   relevant the current one wins. Old chunks are still returned, so the
   answer can say "previously ...".
6. **De-duplication** -- the same timetable header repeated across ten
   PDFs only takes up one slot, and one page/PDF fills at most
   ``MAX_CHUNKS_PER_SOURCE`` slots.
7. **Current-state pass** -- for questions about a role or "now" ("Who is
   the HOD?", "current coordinator"), the newest chunks that mention the
   role are always included, so a recent change is never missed.

Every part is a separate class, so any of them can be replaced (e.g. a
different reranker) without touching the others.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from datetime import date

import numpy as np

import config

log = logging.getLogger("rag.retrieval")

_TOKEN = re.compile(r"[a-z0-9]+", re.I)
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at", "to",
    "for", "and", "or", "who", "what", "when", "where", "which", "how", "does",
    "do", "did", "me", "tell", "please", "about", "this", "that", "with", "by",
    "it", "be", "as", "from", "kya", "hai", "kaun", "ka", "ki", "ke", "batao",
}


def tokenize(text: str) -> list[str]:
    return [t for t in (m.group(0).lower() for m in _TOKEN.finditer(text)) if t not in STOPWORDS]


# Semester/year numbers are written many ways: 5th, 5, fifth, V, Semester-V.
_ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]
_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
_SUFFIX = ["st", "nd", "rd", "th", "th", "th", "th", "th"]
NUMBER_FORMS: dict[str, list[str]] = {}
for _n in range(1, 9):
    _forms = [str(_n), f"{_n}{_SUFFIX[_n - 1]}", _ROMAN[_n - 1], _WORDS[_n - 1]]
    for _f in _forms:
        NUMBER_FORMS[_f] = [x for x in _forms if x != _f]


def expand_query(query: str) -> list[str]:
    """Question tokens plus synonym tokens (deduplicated, order kept)."""
    tokens = tokenize(query)
    expanded = list(tokens)
    has_semester_word = any(t in ("sem", "semester", "year") for t in tokens)
    for tok in tokens:
        for phrase in config.QUERY_SYNONYMS.get(tok, []):
            expanded.extend(tokenize(phrase))
        # only widen numbers when they're about a semester/year ("5th sem"),
        # so "I" or "V" in ordinary questions aren't treated as numbers
        if has_semester_word and tok in NUMBER_FORMS:
            expanded.extend(NUMBER_FORMS[tok])
    seen, result = set(), []
    for tok in expanded:
        if tok not in seen:
            seen.add(tok)
            result.append(tok)
    return result


class BM25:
    """Classic keyword ranking. Small enough to rebuild in memory at startup."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [Counter(tokenize(d)) for d in documents]
        self.doc_len = np.array([sum(c.values()) for c in self.doc_tokens], dtype="float32")
        self.avg_len = float(self.doc_len.mean()) if len(documents) else 0.0
        df: Counter = Counter()
        for counts in self.doc_tokens:
            df.update(counts.keys())
        n = len(documents)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, counts in enumerate(self.doc_tokens):
            for t in counts:
                self.postings[t].append(i)

    def search(self, query_tokens: list[str], top_n: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for tok in query_tokens:
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for i in self.postings[tok]:
                tf = self.doc_tokens[i][tok]
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1))
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


class Reranker:
    """Cross-encoder that scores (question, chunk) pairs from 0 to 1."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            from rag.embeddings import resolve_device

            device = resolve_device()
            log.info("Loading reranker %s on %s", self.model_name, device)
            self._model = CrossEncoder(self.model_name, device=device, max_length=512)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        raw = np.asarray(self.model.predict([(query, t) for t in texts]), dtype="float64")
        if raw.size and (raw.min() < 0 or raw.max() > 1):  # logits -> 0..1
            raw = 1 / (1 + np.exp(-raw))
        return raw.tolist()


def recency_bonus(chunk_date: str | None, today: date | None = None) -> float:
    """0..RECENCY_WEIGHT: full bonus for this year, fading to 0 over the horizon."""
    if not chunk_date or config.RECENCY_WEIGHT <= 0:
        return 0.0
    try:
        d = date.fromisoformat(chunk_date)
    except ValueError:
        return 0.0
    today = today or date.today()
    age_years = max(0.0, (today - d).days / 365.25)
    return config.RECENCY_WEIGHT * max(0.0, 1 - age_years / config.RECENCY_HORIZON_YEARS)


# A person's name: title + up to 4 capitalised words, all on ONE line.
_NAME = r"(?:Dr|Prof|Mr|Ms|Mrs|Shri|Smt)\.?[ \t]+[A-Z][\w.]*(?:[ \t]+[A-Z][\w.]*){0,3}"


def _role_holder_pattern(role_words: set[str]) -> re.Pattern:
    """Matches a person's name close to a role word, e.g.
    "(Dr. Punit K Johari) Dean, School of IT" or "HoD: Dr. Sanjiv Sharma"."""
    roles = "|".join(sorted(re.escape(r) for r in role_words))
    # PDFs often put the name and the title on separate lines, so up to 40
    # characters of anything (line breaks included) may sit between them.
    return re.compile(
        rf"{_NAME}[\s\S]{{0,40}}?\b(?:{roles})\b|\b(?:{roles})\b[\s\S]{{0,40}}?{_NAME}",
        re.I,
    )


_NAME_RE = re.compile(_NAME)
_TITLE_WORDS = {"professor", "&", "and", "associate", "assistant", "time", "table", "time-table", "prof", "deputy", "vice"}


def role_holders(text: str, role_words: set[str]) -> list[tuple[str, str]]:
    """(name, title) pairs, pairing each title with the NEAREST name.

    College documents usually put the name just before the title
    ("(Dr. Punit K Johari)\nDean, School of IT"), sometimes after it
    ("HoD: Dr. Sanjiv Sharma"). Lists of names are common, so the closest
    name wins -- never just the first one in range.
    """
    roles = "|".join(sorted(re.escape(r) for r in role_words))
    pairs = []
    for m in re.finditer(rf"\b(?:{roles})\b", text, re.I):
        before = text[max(0, m.start() - 45): m.start()]
        names_before = list(_NAME_RE.finditer(before))
        name = None
        if names_before:
            last = names_before[-1]
            gap = before[last.end():]
            # A name on an earlier line that also has times/numbers is a
            # table row ("04:00-05:00 PM Prof. X"), not a signature.
            if "\n" in gap:
                name_line = before[: last.end()].rsplit("\n", 1)[-1]
                if not re.search(r"\d", name_line):
                    name = last.group(0)
            else:
                name = last.group(0)
        if not name:
            after = _NAME_RE.search(text, m.end(), m.end() + 45)
            name = after.group(0) if after else None
        if not name:
            continue
        name = re.sub(r"\s+", " ", name).strip()
        # drop title words the greedy name pattern may have swallowed
        name = re.split(rf"\s+(?:{roles}|Professor|Associate|Assistant|Time|Table|Coordinator)\b", name, flags=re.I)[0]
        # Title = title-ish words just before the role on the same line
        # ("Professor &", "Time Table") + the role + the rest of its line,
        # stopping before any other name.
        line_start = text.rfind("\n", 0, m.start()) + 1
        prefix_words = re.split(r"[\s,()]+", text[line_start: m.start()])
        prefix = []
        for w in reversed([w for w in prefix_words if w]):
            if w.lower().strip(".") in _TITLE_WORDS:
                prefix.insert(0, w)
            else:
                break
        line_end = text.find("\n", m.end())
        rest = text[m.end(): (m.end() + 35) if line_end == -1 else min(line_end, m.end() + 35)]
        other = _NAME_RE.search(rest)
        if other:
            rest = rest[: other.start()]
        rest = re.split(r"\s\d{1,2}[:.]\d{2}", rest)[0]  # drop "11:00-12:00" etc.
        title = re.sub(r"\s+", " ", " ".join(prefix + [m.group(0) + rest])).strip(" ,;:()-")
        pairs.append((name.strip(" ,;:()"), title))
    return pairs


def role_timeline(query: str, results: list[dict]) -> list[tuple[str, str, int]]:
    """For role questions: (date, "Dr. X ... Dean, School of IT", source n)
    for every named role holder in the results, newest first. Handed to the
    LLM so it doesn't have to work out who came when by itself."""
    words = {m.group(0).lower() for m in _TOKEN.finditer(query)}
    roles = words & config.ROLE_TERMS
    if not roles:
        return []
    role_words = set(roles)
    for r in roles:
        role_words.update(t for p in config.QUERY_SYNONYMS.get(r, []) for t in tokenize(p))
    role_words &= config.ROLE_TERMS
    seen, rows = set(), []
    for n, chunk in enumerate(results, 1):
        for name, title in role_holders(chunk["text"], role_words):
            snippet = f"{name}, {title}"
            key = (chunk.get("date"), snippet.lower())
            if key not in seen:
                seen.add(key)
                rows.append((chunk.get("date") or "", snippet, n))
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows[:12]


def _dedupe_key(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())[:300]


class HybridRetriever:
    def __init__(self, chunks: list[dict], faiss_index=None, embedder=None, reranker: Reranker | None = None):
        self.chunks = chunks
        self.index = faiss_index
        self.embedder = embedder
        self.reranker = reranker
        self.bm25 = BM25([f"{c.get('title', '')}\n{c['text']}" for c in chunks])

    def _vector_search(self, query: str, top_n: int) -> list[tuple[int, float]]:
        if self.index is None or self.embedder is None:
            return []
        vec = self.embedder.embed_queries([query])
        scores, ids = self.index.search(vec, top_n)
        return [(int(i), float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def _hybrid_relevance(self, candidates, keyword_hits, vector_hits) -> dict[int, float]:
        """0..1 relevance from the methods' own scores (not their ranks), so
        two equally good chunks get equal relevance and recency decides."""
        kw_max = max((s for _, s in keyword_hits), default=0.0) or 1.0
        kw = {i: s / kw_max for i, s in keyword_hits}
        # Cosine scores of one model sit in a narrow band (e.g. 0.75-0.85);
        # stretch that band to 0..1 so it carries as much weight as BM25.
        vec_scores = [s for _, s in vector_hits]
        lo, hi = (min(vec_scores), max(vec_scores)) if vec_scores else (0.0, 0.0)
        vec = {i: (s - lo) / (hi - lo) if hi > lo else 1.0 for i, s in vector_hits}
        methods = [m for m in (kw, vec) if m]
        if not methods:
            return {i: 0.0 for i in candidates}
        return {i: sum(m.get(i, 0.0) for m in methods) / len(methods) for i in candidates}

    def _newest_role_mentions(self, query: str, keyword_hits_wide) -> list[int]:
        """For "who is the HOD (now)?"-type questions: the newest chunks that
        mention the role, among those matching the question at all."""
        words = {m.group(0).lower() for m in _TOKEN.finditer(query)}
        roles = words & config.ROLE_TERMS
        if not roles and not (words & config.PRESENT_TERMS):
            return []
        role_words = set(roles)
        for r in roles:
            for phrase in config.QUERY_SYNONYMS.get(r, []):
                role_words.update(tokenize(phrase))
        role_words &= config.ROLE_TERMS
        if not role_words:
            return []
        holder = _role_holder_pattern(role_words)
        mentions = [
            i for i, _ in keyword_hits_wide
            if self.chunks[i].get("date")
            and role_words & self.bm25.doc_tokens[i].keys()
            and holder.search(self.chunks[i]["text"])
        ]
        mentions.sort(key=lambda i: self.chunks[i]["date"], reverse=True)
        picked, seen, sources = [], set(), set()
        for i in mentions:  # newest first, one chunk per document
            key = _dedupe_key(self.chunks[i]["text"])
            source = self.chunks[i]["source_url"]
            if key in seen or source in sources:
                continue
            seen.add(key)
            sources.add(source)
            picked.append(i)
            if len(picked) >= config.NEWEST_ROLE_CHUNKS:
                break
        return picked

    def search(self, query: str, top_k: int = config.TOP_K, today: date | None = None) -> list[dict]:
        n = config.CANDIDATES_PER_RETRIEVER
        keyword_wide = self.bm25.search(expand_query(query), max(n, 300))
        keyword_hits = keyword_wide[:n]
        vector_hits = self._vector_search(query, n)
        newest_roles = self._newest_role_mentions(query, keyword_wide)

        # Reciprocal Rank Fusion, then make sure the newest role mentions
        # are candidates too.
        fused: dict[int, float] = defaultdict(float)
        for hits in (keyword_hits, vector_hits):
            for rank, (idx, _score) in enumerate(hits):
                fused[idx] += 1.0 / (60 + rank)
        ranked = sorted(fused, key=fused.get, reverse=True)
        ranked = newest_roles + [i for i in ranked if i not in newest_roles]

        # Drop near-identical chunks before spending reranker time on them.
        candidates, seen = [], set()
        for idx in ranked:
            key = _dedupe_key(self.chunks[idx]["text"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append(idx)
            if len(candidates) >= config.RERANK_CANDIDATES + len(newest_roles):
                break
        if not candidates:
            return []

        hybrid = self._hybrid_relevance(candidates, keyword_wide, vector_hits)
        relevance = dict(hybrid)
        if self.reranker is not None:
            texts = [f"{self.chunks[i].get('title', '')}\n{self.chunks[i]['text']}" for i in candidates]
            try:
                rerank = self.reranker.score(query, texts)
                w = config.RERANKER_BLEND
                relevance = {i: w * r + (1 - w) * hybrid[i] for i, r in zip(candidates, rerank)}
            except Exception as exc:  # model missing, offline, out of memory...
                log.warning("Reranker unavailable (%s); continuing without it.", exc)
                self.reranker = None  # don't retry on every question

        scored = []
        for idx in candidates:
            chunk = dict(self.chunks[idx])
            bonus = recency_bonus(chunk.get("date"), today)
            chunk["relevance"] = round(float(relevance[idx]), 4)
            chunk["recency_bonus"] = round(bonus, 4)
            chunk["score"] = float(relevance[idx]) + bonus
            chunk["newest_role_mention"] = idx in newest_roles
            scored.append((idx, chunk))
        scored.sort(key=lambda pair: pair[1]["score"], reverse=True)

        # Newest role mentions are always kept; the rest by score, with at
        # most MAX_CHUNKS_PER_SOURCE from any one page/PDF.
        results = [c for i, c in scored if i in newest_roles][:top_k]
        per_source: dict[str, int] = defaultdict(int)
        for c in results:
            per_source[c["source_url"]] += 1
        for i, c in scored:
            if len(results) >= top_k:
                break
            if i in newest_roles or per_source[c["source_url"]] >= config.MAX_CHUNKS_PER_SOURCE:
                continue
            per_source[c["source_url"]] += 1
            results.append(c)
        results.sort(key=lambda c: c["score"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Loading the saved index from disk
# ---------------------------------------------------------------------------

class IndexMismatchError(RuntimeError):
    """The index was built with a different embedding model."""


def load_retriever() -> HybridRetriever:
    import faiss

    from rag.embed_index import read_index_info
    from rag.embeddings import get_embedder

    embedder = get_embedder()
    info = read_index_info()
    built_with = info.get("embedding_model")
    if built_with and built_with != embedder.model_name:
        raise IndexMismatchError(
            f"The index was built with '{built_with}' but EMBEDDING_MODEL is "
            f"'{embedder.model_name}'. Rebuild it: python run_pipeline.py --skip-crawl"
        )
    if not built_with:
        log.warning(
            "Index has no index_info.json (built by an older version). If answers look "
            "wrong, rebuild it: python run_pipeline.py --skip-crawl"
        )

    index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    chunks = []
    with open(config.INDEX_METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    reranker = None
    if config.RERANKER_MODEL and config.RERANKER_MODEL.lower() != "none":
        reranker = Reranker(config.RERANKER_MODEL)
    return HybridRetriever(chunks, faiss_index=index, embedder=embedder, reranker=reranker)
