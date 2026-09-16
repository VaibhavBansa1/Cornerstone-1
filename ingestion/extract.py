"""Turn crawled raw pages/assets into text chunks ready for embedding.

Reads ``config.MANIFEST_PATH`` (written by scraper.crawler) and produces
``config.CHUNKS_PATH``, a JSONL file where each line is:

    {
      "id": "<sha1>",
      "source_url": "...",
      "source_type": "page" | "pdf" | "image" | "audio" | "video",
      "local_path": "raw/pages/....html",
      "chunk_index": 0,
      "text": "..."
    }

Images/audio/video have no extractable text body, so a short descriptive
chunk (title of the referring page + filename + any alt text) is emitted
instead, so they can still be found by the retriever and cited with their
local file path.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from bs4 import BeautifulSoup

import config

log = logging.getLogger("ingest")

TEXT_STRIP_TAGS = ["script", "style", "nav", "footer", "header", "noscript"]


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(TEXT_STRIP_TAGS):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf not installed; skipping PDF text extraction for %s", path)
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # malformed PDFs happen in the wild
        log.warning("Failed to extract text from %s: %s", path, exc)
        return ""


def chunk_text(text: str, size: int = config.CHUNK_SIZE_CHARS, overlap: int = config.CHUNK_OVERLAP_CHARS):
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def make_chunk_id(source_url: str, chunk_index: int) -> str:
    return hashlib.sha1(f"{source_url}::{chunk_index}".encode("utf-8")).hexdigest()


def build_chunks() -> list[dict]:
    manifest = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    chunks: list[dict] = []

    # Map asset URL -> manifest entry so we can pull alt text / referring
    # page titles from the pages that mention them.
    page_title_by_url = {p["url"]: p.get("title", "") for p in manifest["pages"]}

    for page in manifest["pages"]:
        html_path = config.DATA_DIR / page["local_path"]
        if not html_path.exists():
            continue
        text = html_to_text(html_path.read_text(encoding="utf-8"))
        for i, piece in enumerate(chunk_text(text)):
            chunks.append(
                {
                    "id": make_chunk_id(page["url"], i),
                    "source_url": page["url"],
                    "source_type": "page",
                    "local_path": page["local_path"],
                    "title": page.get("title", ""),
                    "chunk_index": i,
                    "text": piece,
                }
            )

    media_kind_by_ext = {}
    for ext in config.DOCUMENT_EXTENSIONS:
        media_kind_by_ext[ext] = "document"
    for ext in config.IMAGE_EXTENSIONS:
        media_kind_by_ext[ext] = "image"
    for ext in config.AUDIO_EXTENSIONS:
        media_kind_by_ext[ext] = "audio"
    for ext in config.VIDEO_EXTENSIONS:
        media_kind_by_ext[ext] = "video"

    for asset in manifest["assets"]:
        ext = asset["extension"]
        kind = media_kind_by_ext.get(ext, "document")
        local_path = config.DATA_DIR / asset["local_path"]
        referring_title = page_title_by_url.get(asset["referenced_from"], "")
        display_name = asset.get("original_filename") or local_path.name

        if kind == "document" and ext == ".pdf" and local_path.exists():
            text = pdf_to_text(local_path)
            pieces = chunk_text(text) if text.strip() else []
            if not pieces:
                pieces = [f"PDF document '{display_name}' referenced from page '{referring_title}'."]
            for i, piece in enumerate(pieces):
                chunks.append(
                    {
                        "id": make_chunk_id(asset["url"], i),
                        "source_url": asset["url"],
                        "source_type": "pdf",
                        "local_path": asset["local_path"],
                        "title": referring_title,
                        "chunk_index": i,
                        "text": piece,
                    }
                )
        else:
            description = (
                f"{kind.capitalize()} file '{display_name}' "
                f"referenced from page '{referring_title}' ({asset['referenced_from']})."
            )
            chunks.append(
                {
                    "id": make_chunk_id(asset["url"], 0),
                    "source_url": asset["url"],
                    "source_type": kind,
                    "local_path": asset["local_path"],
                    "title": referring_title,
                    "chunk_index": 0,
                    "text": description,
                }
            )

    return chunks


def write_chunks(chunks: list[dict]):
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info("Wrote %d chunks to %s", len(chunks), config.CHUNKS_PATH)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    chunks = build_chunks()
    write_chunks(chunks)


if __name__ == "__main__":
    main()
