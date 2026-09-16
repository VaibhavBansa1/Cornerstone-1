"""Retrieval-augmented query pipeline over the MITS IT department corpus.

Retrieval always works locally (FAISS + sentence-transformers, no API key
needed). Answer *generation* uses the Anthropic API when ANTHROPIC_API_KEY
is set; otherwise the pipeline falls back to returning the raw retrieved
chunks so it still works fully offline.
"""
from __future__ import annotations

import json
import logging
import os

import config
from rag.embed_index import embed_texts

log = logging.getLogger("rag.pipeline")

_index = None
_chunks: list[dict] | None = None


def _load_index():
    global _index, _chunks
    if _index is None:
        import faiss

        _index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        _chunks = []
        with open(config.INDEX_METADATA_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    _chunks.append(json.loads(line))
    return _index, _chunks


def retrieve(query: str, top_k: int = config.TOP_K) -> list[dict]:
    index, chunks = _load_index()
    query_vec = embed_texts([query])
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


def _build_context_block(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] (source: {r['source_url']}, type: {r['source_type']})\n{r['text']}"
        )
    return "\n\n".join(parts)


def generate_answer(query: str, results: list[dict]) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    context = _build_context_block(results)

    if not api_key:
        return (
            "[No ANTHROPIC_API_KEY set -- returning retrieved context only]\n\n"
            f"{context}"
        )

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    system = (
        "You answer questions about the MITS Gwalior Information Technology "
        "department using ONLY the provided context. If the context does not "
        "contain the answer, say so plainly. Cite sources by their [n] marker."
    )
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def answer(query: str, top_k: int = config.TOP_K) -> dict:
    results = retrieve(query, top_k=top_k)
    text = generate_answer(query, results)
    return {"query": query, "answer": text, "sources": results}


def main():
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print('Usage: python -m rag.pipeline "your question"')
        raise SystemExit(1)
    query = " ".join(sys.argv[1:])
    result = answer(query)
    print("\n=== ANSWER ===\n")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for r in result["sources"]:
        print(f"- ({r['source_type']}) {r['source_url']}  score={r['score']:.3f}")


if __name__ == "__main__":
    main()
