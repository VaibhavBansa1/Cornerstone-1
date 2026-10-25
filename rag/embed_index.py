"""Embed chunks (config.CHUNKS_PATH) and build the FAISS similarity index.

Next to the index it writes ``index_info.json`` recording which embedding
model built it. Questions must be embedded with the same model, so the
retriever refuses to run on a mismatched index and tells you to rebuild.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np

import config
from ingestion.dates import label as date_label
from rag.embeddings import get_embedder

log = logging.getLogger("embed_index")


def passage_text(chunk: dict) -> str:
    """What actually gets embedded: a small header + the chunk text.

    The header gives the vector context the chunk itself may lack ("this
    piece of a timetable PDF is from the IT department, July 2026").
    """
    header = " | ".join(
        part for part in (chunk.get("title", "").strip(), date_label(chunk.get("date"))) if part
    )
    return f"{header}\n{chunk['text']}" if header else chunk["text"]


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed document text (kept for older callers)."""
    return get_embedder().embed_passages(texts)


def load_chunks() -> list[dict]:
    chunks = []
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def read_index_info() -> dict:
    try:
        return json.loads(config.INDEX_INFO_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def build_index(chunks: list[dict] | None = None):
    import faiss

    if chunks is None:
        chunks = load_chunks()
    if not chunks:
        raise ValueError(f"No chunks found in {config.CHUNKS_PATH}; run ingestion first.")

    embedder = get_embedder()
    started = time.perf_counter()
    embeddings = embedder.embed_passages([passage_text(c) for c in chunks])

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    with open(config.INDEX_METADATA_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    config.INDEX_INFO_PATH.write_text(
        json.dumps(
            {
                "embedding_model": embedder.model_name,
                "dimension": int(dim),
                "chunks": len(chunks),
                "built_on_device": embedder.device,
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seconds": round(time.perf_counter() - started, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log.info(
        "Built FAISS index with %d vectors (dim=%d, model=%s, device=%s) in %.1fs -> %s",
        len(chunks), dim, embedder.model_name, embedder.device,
        time.perf_counter() - started, config.FAISS_INDEX_PATH,
    )
    return index, chunks


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_index()


if __name__ == "__main__":
    main()
