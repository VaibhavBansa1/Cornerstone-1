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
7. **Role pass** -- for "who is the HOD/dean/coordinator" questions, the
   newest chunks that name someone in that role are always included, so a
   recent change is never missed (name/title matching: rag/roles.py).
8. **Batch pass** -- when query understanding worked out a batch/semester
   (rag/query_understanding.py), documents filed under it ("V SEM · 2024-28
   BATCH") get a bonus and are always included; other batches' get a penalty.
9. **Topic page** -- questions about labs, contacts, programmes or the
   department in general always include that website page
   (``config.TOPIC_PAGES``).

BM25, the reranker and the retriever are separate classes, so any of them
can be replaced (e.g. a different reranker) without touching the others.
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
from rag import roles

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


def expand_query(query: str, extra_terms: list[str] | None = None) -> list[str]:
    """Question tokens + synonym tokens + ``extra_terms`` (e.g. the batch
    worked out by query understanding), deduplicated, order kept."""
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
    expanded.extend(t.lower() for t in (extra_terms or []))
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


def _dedupe_key(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())[:300]


class HybridRetriever:
    def __init__(self, chunks: list[dict], faiss_index=None, embedder=None, reranker: Reranker | None = None):
        self.chunks = chunks
        self.index = faiss_index
        self.embedder = embedder
        self.reranker = reranker
        # Title words (file name + where it's linked, e.g. "Syllabus · V SEM ·
        # 2024-28 BATCH") say what a document IS, so they count extra --
        # otherwise long meeting minutes that mention "syllabus" 20 times win.
        w = max(1, config.TITLE_WEIGHT)
        self.bm25 = BM25([f"{(c.get('title', '') + ' ') * w}\n{c['text']}" for c in chunks])

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
        words = roles.role_words(query)
        if not words:
            return []
        holder = roles.holder_pattern(words)
        mentions = [
            i for i, _ in keyword_hits_wide
            if self.chunks[i].get("date")
            and words & self.bm25.doc_tokens[i].keys()
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

    def _batch_matches(self, understanding, keyword_wide) -> list[int]:
        """Best chunks from documents filed under the understood batch and
        semester ("V SEM · 2024-28 BATCH"), so they are always considered --
        course lists first, then by keyword score; one per document."""
        if understanding is None or understanding.empty:
            return []
        picked, sources = [], set()
        ordered = sorted(
            (i for i, _ in keyword_wide
             if understanding.label_matches(self.chunks[i].get("link_label", "")) > 0),
            key=lambda i: self.chunks[i].get("kind") != "course_list",  # stable: keeps score order
        )
        for i in ordered:
            src = self.chunks[i]["source_url"]
            if src in sources:
                continue
            sources.add(src)
            picked.append(i)
            if len(picked) >= config.BATCH_MATCH_CHUNKS:
                break
        return picked

    def _topic_page(self, understanding) -> dict | None:
        """The whole website page for the question's topic ("labs" -> the
        Laboratories page) as ONE source, see config.TOPIC_PAGES. These
        pages are short, and a list (all labs) is only useful complete."""
        page = config.TOPIC_PAGES.get(getattr(understanding, "topic", None) or "")
        if not page:
            return None
        parts = [c for c in self.chunks if c.get("title") == f"{page} - MITS Gwalior" and c.get("chunk_index", 0) >= 0]
        if not parts:
            return None
        parts.sort(key=lambda c: c.get("chunk_index", 0))
        text = parts[0]["text"]
        for c in parts[1:]:  # chunks overlap: skip the repeated start
            nxt = c["text"]
            cut = next((k for k in range(min(len(text), len(nxt), 400), 20, -1) if text.endswith(nxt[:k])), 0)
            text += "\n" + nxt[cut:]
        merged = dict(parts[0], text=text, relevance=1.0, recency_bonus=0.0, batch_bonus=0.0,
                      newest_role_mention=False, topic_page=True)
        merged["score"] = 1.0
        return merged

    def search(self, query: str, top_k: int = config.TOP_K, today: date | None = None,
               understanding=None) -> list[dict]:
        n = config.CANDIDATES_PER_RETRIEVER
        extra = understanding.search_terms() if understanding is not None else []
        if understanding is not None:  # "3rd year" -> "5th sem" for every search step
            query = understanding.rewrite(query)
        keyword_wide = self.bm25.search(expand_query(query, extra), max(n, 300))
        keyword_hits = keyword_wide[:n]
        vector_hits = self._vector_search(query, n)
        newest_roles = self._newest_role_mentions(query, keyword_wide)
        batch_hits = [i for i in self._batch_matches(understanding, keyword_wide) if i not in newest_roles]
        pinned = newest_roles + batch_hits

        # Reciprocal Rank Fusion, then make sure the pinned chunks (newest
        # role mentions, documents for the understood batch) are candidates.
        fused: dict[int, float] = defaultdict(float)
        for hits in (keyword_hits, vector_hits):
            for rank, (idx, _score) in enumerate(hits):
                fused[idx] += 1.0 / (60 + rank)
        ranked = sorted(fused, key=fused.get, reverse=True)
        ranked = pinned + [i for i in ranked if i not in pinned]

        # Drop near-identical chunks before spending reranker time on them.
        candidates, seen = [], set()
        for idx in ranked:
            key = _dedupe_key(self.chunks[idx]["text"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append(idx)
            if len(candidates) >= config.RERANK_CANDIDATES + len(pinned):
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
            # Right batch/semester (+), a different batch's document (-).
            match = understanding.label_matches(chunk.get("link_label", "")) if understanding is not None else 0
            batch_bonus = config.BATCH_MATCH_BONUS * match
            chunk["relevance"] = round(float(relevance[idx]), 4)
            chunk["recency_bonus"] = round(bonus, 4)
            chunk["batch_bonus"] = round(batch_bonus, 4)
            chunk["score"] = float(relevance[idx]) + bonus + batch_bonus
            chunk["newest_role_mention"] = idx in newest_roles
            scored.append((idx, chunk))
        scored.sort(key=lambda pair: pair[1]["score"], reverse=True)

        # Pinned chunks are always kept; the rest by score, with at most
        # MAX_CHUNKS_PER_SOURCE from any one page/PDF.
        results = [c for i, c in scored if i in pinned][:top_k]
        per_source: dict[str, int] = defaultdict(int)
        for c in results:
            per_source[c["source_url"]] += 1
        for i, c in scored:
            if len(results) >= top_k:
                break
            if i in pinned or per_source[c["source_url"]] >= config.MAX_CHUNKS_PER_SOURCE:
                continue
            per_source[c["source_url"]] += 1
            results.append(c)
        results.sort(key=lambda c: c["score"], reverse=True)

        page = self._topic_page(understanding)
        if page is not None:  # the whole topic page replaces its pieces
            results = [page] + [c for c in results if c["source_url"] != page["source_url"]][:top_k - 1]
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
