"""Retrieval-augmented query pipeline over the MITS IT department corpus.

Retrieval always works locally (FAISS + sentence-transformers, no API key
needed). Answer *generation* is pluggable across Anthropic, OpenAI (or any
OpenAI-compatible local server), Gemini, and a local Ollama model -- see
`_resolve_provider` -- and falls back to returning the raw retrieved chunks
if none is configured, so the pipeline still works fully offline.
"""
from __future__ import annotations

import json
import logging
import os

import requests

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


def reload_index():
    """Drop the cached index so the next query reads the rebuilt one."""
    global _index, _chunks
    _index = None
    _chunks = None


def index_exists() -> bool:
    return config.FAISS_INDEX_PATH.exists() and config.INDEX_METADATA_PATH.exists()


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


SYSTEM_PROMPT = (
    "You answer questions about the MITS Gwalior Information Technology "
    "department using ONLY the provided context. If the context does not "
    "contain the answer, say so plainly. Cite sources by their [n] marker. "
    "Reply in the same language the student used (English, Hindi, or "
    "Hinglish)."
)


def _ollama_reachable() -> bool:
    try:
        requests.get(config.OLLAMA_BASE_URL, timeout=2)
        return True
    except requests.RequestException:
        return False


def _resolve_provider() -> str:
    provider = config.GENERATION_PROVIDER
    if provider != "auto":
        return provider
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if _ollama_reachable():
        return "ollama"
    return "none"


def _generate_with_anthropic(system: str, user_message: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def _generate_with_openai(system: str, user_message: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=config.OPENAI_BASE_URL)
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def _generate_with_gemini(system: str, user_message: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(config.GEMINI_MODEL, system_instruction=system)
    response = model.generate_content(user_message)
    return response.text


def _generate_with_ollama(system: str, user_message: str) -> str:
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


_PROVIDERS = {
    "anthropic": _generate_with_anthropic,
    "openai": _generate_with_openai,
    "gemini": _generate_with_gemini,
    "ollama": _generate_with_ollama,
}


def generate_answer(query: str, results: list[dict], provider: str | None = None) -> str:
    context = _build_context_block(results)
    provider = provider or _resolve_provider()

    if provider == "none":
        return (
            "[No generation provider configured -- set ANTHROPIC_API_KEY, "
            "OPENAI_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY, or run a local "
            "Ollama server -- returning retrieved context only]\n\n"
            f"{context}"
        )

    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    try:
        return _PROVIDERS[provider](SYSTEM_PROMPT, user_message)
    except Exception as exc:  # missing SDK, bad key, model not pulled, etc.
        log.warning("Generation via '%s' failed: %s", provider, exc)
        return (
            f"[Generation via '{provider}' failed ({exc}) -- returning "
            f"retrieved context only]\n\n{context}"
        )


def answer(query: str, top_k: int = config.TOP_K) -> dict:
    results = retrieve(query, top_k=top_k)
    provider = _resolve_provider()
    text = generate_answer(query, results, provider)
    return {"query": query, "answer": text, "sources": results, "provider": provider}


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
