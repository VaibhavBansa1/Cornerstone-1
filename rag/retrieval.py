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
   PDFs only takes up one slot.

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


def expand_query(query: str) -> list[str]:
    """Question tokens plus synonym tokens (deduplicated, order kept)."""
    tokens = tokenize(query)
    expanded = list(tokens)
    for tok in tokens:
        for phrase in config.QUERY_SYNONYMS.get(tok, []):
            expanded.extend(tokenize(phrase))
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

    def search(self, query: str, top_k: int = config.TOP_K, today: date | None = None) -> list[dict]:
        n = config.CANDIDATES_PER_RETRIEVER
        keyword_hits = self.bm25.search(expand_query(query), n)
        vector_hits = self._vector_search(query, n)

        # Reciprocal Rank Fusion
        fused: dict[int, float] = defaultdict(float)
        for hits in (keyword_hits, vector_hits):
            for rank, (idx, _score) in enumerate(hits):
                fused[idx] += 1.0 / (60 + rank)
        ranked = sorted(fused, key=fused.get, reverse=True)

        # Drop near-identical chunks before spending reranker time on them.
        candidates, seen = [], set()
        for idx in ranked:
            key = _dedupe_key(self.chunks[idx]["text"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append(idx)
            if len(candidates) >= config.RERANK_CANDIDATES:
                break
        if not candidates:
            return []

        relevance = None
        if self.reranker is not None:
            texts = [f"{self.chunks[i].get('title', '')}\n{self.chunks[i]['text']}" for i in candidates]
            try:
                relevance = self.reranker.score(query, texts)
            except Exception as exc:  # model missing, offline, out of memory...
                log.warning("Reranker unavailable (%s); continuing without it.", exc)
                self.reranker = None  # don't retry on every question
        if relevance is None:
            # No reranker: average the methods' own scores (BM25 scaled to
            # 0..1, cosine similarity is already 0..1). Unlike rank-based
            # scores, two equally good chunks get equal relevance here, so
            # the recency bonus decides between them.
            kw_max = max((s for _, s in keyword_hits), default=0.0) or 1.0
            kw = {i: s / kw_max for i, s in keyword_hits}
            vec = {i: max(0.0, s) for i, s in vector_hits}
            methods = [m for m in (kw, vec) if m]
            relevance = [sum(m.get(i, 0.0) for m in methods) / len(methods) for i in candidates]

        results = []
        for idx, rel in zip(candidates, relevance):
            chunk = dict(self.chunks[idx])
            bonus = recency_bonus(chunk.get("date"), today)
            chunk["relevance"] = round(float(rel), 4)
            chunk["recency_bonus"] = round(bonus, 4)
            chunk["score"] = float(rel) + bonus
            results.append(chunk)
        results.sort(key=lambda c: c["score"], reverse=True)
        return results[:top_k]


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
