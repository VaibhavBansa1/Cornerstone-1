"""Turns text into vectors (lists of numbers) for similarity search.

Swap the model with the EMBEDDING_MODEL env var and the device with
EMBEDDING_DEVICE (auto / cuda / cpu). "auto" picks the NVIDIA GPU when
PyTorch can see one and silently falls back to the CPU otherwise, so the
same code runs on a GPU desktop and a laptop without one.

Some models expect a short prefix that says whether the text is a question
or a document (e5: "query: " / "passage: "). That's handled here so the
rest of the code never has to know which model is in use.
"""
from __future__ import annotations

import logging

import numpy as np

import config

log = logging.getLogger("rag.embeddings")

# (query prefix, passage prefix) by a substring of the model name
_PREFIXES = [
    ("e5", ("query: ", "passage: ")),
    ("bge-m3", ("", "")),
    ("bge", ("Represent this sentence for searching relevant passages: ", "")),
]


def resolve_device(requested: str | None = None) -> str:
    requested = (requested or config.EMBEDDING_DEVICE or "auto").lower()
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:  # torch missing or broken CUDA install
        has_cuda = False
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda"):
        if has_cuda:
            return requested
        log.warning("EMBEDDING_DEVICE=%s but no CUDA GPU is visible to PyTorch; using CPU.", requested)
        return "cpu"
    return "cuda" if has_cuda else "cpu"


def prefixes_for(model_name: str) -> tuple[str, str]:
    name = model_name.lower()
    for key, prefixes in _PREFIXES:
        if key in name:
            return prefixes
    return "", ""


class Embedder:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.device = resolve_device(device)
        self.query_prefix, self.passage_prefix = prefixes_for(self.model_name)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading embedding model %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so inner product == cosine similarity
        )
        return np.asarray(vectors, dtype="float32")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])


_default: Embedder | None = None


def get_embedder() -> Embedder:
    global _default
    if _default is None or _default.model_name != config.EMBEDDING_MODEL_NAME:
        _default = Embedder()
    return _default
