"""Embed chunks (config.CHUNKS_PATH) and build a FAISS similarity index."""
from __future__ import annotations

import json
import logging

import numpy as np

import config

log = logging.getLogger("embed_index")

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    )
    return embeddings.astype("float32")


def load_chunks() -> list[dict]:
    chunks = []
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_index(chunks: list[dict] | None = None):
    import faiss

    if chunks is None:
        chunks = load_chunks()
    if not chunks:
        raise ValueError(f"No chunks found in {config.CHUNKS_PATH}; run ingestion first.")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    with open(config.INDEX_METADATA_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    log.info("Built FAISS index with %d vectors (dim=%d) -> %s", len(chunks), dim, config.FAISS_INDEX_PATH)
    return index, chunks


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_index()


if __name__ == "__main__":
    main()
