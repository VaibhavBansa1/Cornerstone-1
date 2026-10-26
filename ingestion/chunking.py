"""Step 2b of indexing: cut a document's text into dated chunks.

    chunk_text(text)        -> pieces of ~900 characters, cut at paragraph or
                               sentence ends, neighbours overlapping a little
    make_chunks(...)        -> chunk records, each stamped with a date
    with_course_list(...)   -> + one extra chunk listing every course + code

A chunk record (one line of data/processed/chunks.jsonl):

    {
      "id": "<sha1>", "source_url": "...", "source_type": "page" | "pdf" | "upload" | "image" | ...,
      "local_path": "raw/pages/....html", "title": "...", "link_label": "V SEM · 2024-28 BATCH",
      "chunk_index": 0,           # -1 for the course-list chunk ("kind": "course_list")
      "date": "2026-07-27",       # newest full date in this chunk, else doc_date
      "doc_date": "2026-07-27",   # best date for the whole document (see make_chunks)
      "text": "..."
    }
"""
from __future__ import annotations

import hashlib
import re

import config
from ingestion.course_list import course_list_text
from ingestion.dates import date_from_batch_label, latest_date

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Last resort for text with no line or sentence breaks at all."""
    pieces, start = [], 0
    while start < len(text):
        end = start + size
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return pieces


def _units(text: str, size: int, overlap: int) -> list[tuple[str, bool]]:
    """Break text into paragraphs/sentences no longer than ``size``.

    Returns (piece, came_from_hard_split) pairs.
    """
    units: list[tuple[str, bool]] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append((para, False))
            continue
        for sentence in _SENTENCE_END.split(para):
            if len(sentence) <= size:
                units.append((sentence, False))
            else:
                units.extend((piece, True) for piece in _hard_split(sentence, size, overlap))
    return units


def _tail(text: str, overlap: int) -> str:
    """The last ``overlap`` characters, starting at a word boundary."""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if 0 <= space < len(tail) - 1 else tail


def chunk_text(text: str, size: int = config.CHUNK_SIZE_CHARS, overlap: int = config.CHUNK_OVERLAP_CHARS):
    """Split text into ~``size``-character chunks on paragraph/sentence
    boundaries, repeating ~``overlap`` characters between neighbours so a
    fact that sits on a boundary is still found whole in one of them."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    current = ""
    for piece, hard in _units(text, size, overlap):
        if hard:  # hard-split pieces already overlap each other
            if current:
                chunks.append(current)
            chunks.append(piece)
            current = ""
            continue
        candidate = f"{current}\n{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            carry = _tail(current, overlap)
            current = f"{carry}\n{piece}" if len(carry) + 1 + len(piece) <= size else piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def make_chunk_id(source_url: str, chunk_index: int) -> str:
    return hashlib.sha1(f"{source_url}::{chunk_index}".encode("utf-8")).hexdigest()


def make_chunks(
    *,
    source_url: str,
    source_type: str,
    local_path: str,
    title: str,
    text: str,
    fallback_date: str | None = None,
    link_label: str = "",
) -> list[dict]:
    """Chunk one document and stamp every chunk with a date.

    ``doc_date``: newest date anywhere in the document (or the fallback).
    ``date``:     newest date inside that chunk, else ``doc_date``.
    """
    pieces = chunk_text(text)
    # Best clue first: a real date in the text ("w.e.f. 27/07/2026") ->
    # the semester the link is for ("V SEM · 2024-28 BATCH") -> the file's
    # own date -> a bare year somewhere in the text (weakest: often a book's
    # publication year).
    full = f"{title}\n{text}"
    doc_date = (
        latest_date(full, allow_bare_years=False)
        or date_from_batch_label(link_label or title)
        or fallback_date
        or latest_date(full)
    )
    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append(
            {
                "id": make_chunk_id(source_url, i),
                "source_url": source_url,
                "source_type": source_type,
                "local_path": local_path,
                "title": title,
                "link_label": link_label,
                "chunk_index": i,
                "date": latest_date(piece, allow_bare_years=False) or doc_date,
                "doc_date": doc_date,
                "text": piece,
            }
        )
    return chunks


def with_course_list(chunks: list[dict], text: str) -> list[dict]:
    """Add one extra chunk listing every course (name + code) found in the
    document, so "which subjects..." questions get the full list."""
    if not chunks:
        return chunks
    first = chunks[0]
    name = first.get("link_label") or first.get("title") or first["source_url"]
    listing = course_list_text(text, name)
    if not listing:
        return chunks
    extra = dict(first)
    extra.update({
        "id": make_chunk_id(first["source_url"], -1),
        "chunk_index": -1,
        "kind": "course_list",
        "date": first.get("doc_date") or first.get("date"),
        "text": listing,
    })
    return chunks + [extra]
