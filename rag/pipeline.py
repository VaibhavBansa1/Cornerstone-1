"""Question -> answer, gluing the three swappable parts together:

    retrieval  (rag/retrieval.py)  finds dated pieces of text
    prompt     (this file)         lays them out with dates + rules
    LLM        (rag/llm/)          writes the answer, with provider fallback

If no LLM is reachable, the retrieved passages are returned instead, so the
chatbot still works fully offline.

    python -m rag.pipeline "Who is the HOD of IT?"
"""
from __future__ import annotations

import logging
import threading
from datetime import date

import config
from ingestion.dates import label as date_label
from rag import llm
from rag.retrieval import HybridRetriever, load_retriever

log = logging.getLogger("rag.pipeline")

_retriever: HybridRetriever | None = None
_lock = threading.Lock()


def _get_retriever() -> HybridRetriever:
    global _retriever
    with _lock:
        if _retriever is None:
            _retriever = load_retriever()
        return _retriever


def _load_index():
    """(faiss_index, chunks) -- used by the admin page to list documents."""
    r = _get_retriever()
    return r.index, r.chunks


def reload_index():
    """Drop the cached index so the next query reads the rebuilt one."""
    global _retriever
    with _lock:
        _retriever = None


def index_exists() -> bool:
    return config.FAISS_INDEX_PATH.exists() and config.INDEX_METADATA_PATH.exists()


def retrieve(query: str, top_k: int = config.TOP_K) -> list[dict]:
    return _get_retriever().search(query, top_k=top_k)


SYSTEM_PROMPT = """You are the help desk assistant for the Information Technology department (School of IT) at MITS Gwalior.
Today's date is {today}.

Answer ONLY from the numbered sources below. Each source shows a date: the newest date mentioned in that text, or "undated".

Rules:
1. Facts change over time: who holds a post (HOD / Head / Dean / coordinator), timetables, syllabus, notices. When sources disagree, trust the most recent dated source and say "as of <month year>". Mention people from older sources only as former holders ("previously ..."), never as current.
2. The same post can have different titles in different years (e.g. "HoD", "Professor & Head", "Dean, School of IT"). Use the exact title the newest source uses, and mention if the title changed.
3. If the sources do not contain the answer, say so plainly and suggest contacting the department office. Never guess names, dates or numbers.
4. Cite sources with their [n] markers.
5. Reply in the same language the student used (English, Hindi or Hinglish). Keep it short and clear."""


def build_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or r["source_url"]
        parts.append(
            f"[{i}] {title} (type: {r.get('source_type')}, dated: {date_label(r.get('date'))})\n{r['text']}"
        )
    return "\n\n".join(parts)


def order_for_prompt(results: list[dict]) -> list[dict]:
    """Newest first, so the model sees current information before old."""
    return sorted(results, key=lambda r: r.get("date") or "", reverse=True)


def generate_answer(query: str, results: list[dict], today: date | None = None) -> llm.Reply:
    context = build_context(results)
    system = SYSTEM_PROMPT.format(today=(today or date.today()).strftime("%d %B %Y"))
    user_message = f"Sources:\n{context}\n\nQuestion: {query}"
    reply = llm.generate(system, user_message)
    if not reply.generated:
        why = "; ".join(reply.errors) or "no LLM provider is configured or reachable"
        reply.text = f"[No generated answer ({why}) -- showing the most relevant passages]\n\n{context}"
    return reply


def answer(query: str, top_k: int = config.TOP_K) -> dict:
    results = order_for_prompt(retrieve(query, top_k=top_k))
    reply = generate_answer(query, results)
    return {
        "query": query,
        "answer": reply.text,
        "sources": results,
        "provider": reply.provider,
        "generated": reply.generated,
        "provider_errors": reply.errors,
    }


def main():
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print('Usage: python -m rag.pipeline "your question"')
        raise SystemExit(1)
    result = answer(" ".join(sys.argv[1:]))
    print("\n=== ANSWER ===  (via", result["provider"] + ")\n")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for i, r in enumerate(result["sources"], 1):
        print(
            f"[{i}] {date_label(r.get('date')):>9}  rel={r.get('relevance', 0):.3f} "
            f"+new={r.get('recency_bonus', 0):.3f}  ({r['source_type']}) {r['source_url']}"
        )
    if result["provider_errors"]:
        print("\nProvider notes:", *result["provider_errors"], sep="\n  - ")


if __name__ == "__main__":
    main()
