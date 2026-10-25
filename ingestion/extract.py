"""Turn crawled raw pages/assets into text chunks ready for embedding.

Reads ``config.MANIFEST_PATH`` (written by scraper.crawler) and produces
``config.CHUNKS_PATH``, a JSONL file where each line is:

    {
      "id": "<sha1>",
      "source_url": "...",
      "source_type": "page" | "pdf" | "image" | "audio" | "video",
      "local_path": "raw/pages/....html",
      "title": "...",
      "chunk_index": 0,
      "date": "2026-07-27",      # newest date mentioned in this chunk
      "doc_date": "2026-07-27",  # newest date in the whole document
      "text": "..."
    }

Dates come from ingestion/dates.py and are what let retrieval prefer
current information over old documents.

Images/audio/video have no extractable text body, so a short descriptive
chunk (title of the referring page + filename + any alt text) is emitted
instead, so they can still be found by the retriever and cited with their
local file path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
import zipfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from bs4 import BeautifulSoup

import config
from ingestion.dates import latest_date

log = logging.getLogger("ingest")

TEXT_STRIP_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]

# Tags that start a new line of text. Everything else (span, sup, a, b ...)
# is "inline" and stays glued to its neighbours, so "7<sup>th</sup> March"
# becomes "7th March" instead of "7\nth\nMarch".
BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "li", "ul", "ol", "tr", "table",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "dd", "dt", "dl",
    "figure", "figcaption", "hr", "tbody", "thead",
}
CELL_TAGS = {"td", "th"}


def normalize_text(text: str) -> str:
    """Fix PDF ligatures (ﬁ -> fi), odd spaces, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ").replace("�", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(TEXT_STRIP_TAGS):
        tag.decompose()
    # Only the page's main content, when the site layout provides it.
    root = soup.find(id=config.MAIN_CONTENT_CONTAINER_ID) or soup
    for br in root.find_all("br"):
        br.replace_with("\n")
    for tag in root.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    for tag in root.find_all(CELL_TAGS):
        tag.insert_after(" | ")
    text = root.get_text(separator="")
    text = re.sub(r"(\s*\|\s*)+\n", "\n", text)  # drop trailing cell separators
    return normalize_text(text)


PDF_CACHE_VERSION = "1"  # bump to force re-extraction of every PDF


def _pdf_cache_path(path: Path) -> Path:
    stat = path.stat()
    key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{PDF_CACHE_VERSION}"
    return config.PROCESSED_DIR / "pdf_text_cache" / (hashlib.sha1(key.encode()).hexdigest() + ".txt")


def pdf_to_text(path: Path) -> str:
    """PDF -> text. Reading PDFs is slow (scanned ones take ~20 s and give
    no text), so the result is cached; a rebuild only re-reads PDFs that
    are new or changed."""
    try:
        cache = _pdf_cache_path(path)
        if cache.exists():
            return cache.read_text(encoding="utf-8")
    except OSError:
        cache = None
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf not installed; skipping PDF text extraction for %s", path)
        return ""
    try:
        reader = PdfReader(str(path))
        text = normalize_text("\n".join(page.extract_text() or "" for page in reader.pages))
    except Exception as exc:  # malformed PDFs happen in the wild
        log.warning("Failed to extract text from %s: %s", path, exc)
        text = ""
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
        except OSError:
            pass
    return text


def docx_to_text(path: Path) -> str:
    """Pull paragraph text out of a .docx without needing python-docx."""
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8")
    except (KeyError, zipfile.BadZipFile) as exc:
        log.warning("Failed to read %s as .docx: %s", path, exc)
        return ""
    paragraphs = re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)
    lines = ["".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p)) for p in paragraphs]
    return "\n".join(line for line in lines if line.strip())


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


def upload_to_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return pdf_to_text(path)
    if ext == ".docx":
        return docx_to_text(path)
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def _last_modified_date(entry: dict) -> str | None:
    """The server's Last-Modified header, if it looks like a real date.

    Dynamic pages often report "right now" as Last-Modified, which says
    nothing about the content, so it is ignored when it equals fetch time.
    """
    raw = entry.get("last_modified")
    if not raw:
        return None
    try:
        modified = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    fetched = entry.get("fetched_at")
    if fetched and abs(modified.timestamp() - float(fetched)) < 24 * 3600:
        return None
    return modified.date().isoformat()


def make_chunks(
    *,
    source_url: str,
    source_type: str,
    local_path: str,
    title: str,
    text: str,
    fallback_date: str | None = None,
) -> list[dict]:
    """Chunk one document and stamp every chunk with a date.

    ``doc_date``: newest date anywhere in the document (or the fallback).
    ``date``:     newest date inside that chunk, else ``doc_date``.
    """
    pieces = chunk_text(text)
    doc_date = latest_date(f"{title}\n{text}") or fallback_date
    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append(
            {
                "id": make_chunk_id(source_url, i),
                "source_url": source_url,
                "source_type": source_type,
                "local_path": local_path,
                "title": title,
                "chunk_index": i,
                "date": latest_date(piece) or doc_date,
                "doc_date": doc_date,
                "text": piece,
            }
        )
    return chunks


def build_upload_chunks() -> list[dict]:
    """Chunks for documents staff uploaded through the admin page."""
    chunks: list[dict] = []
    if not config.UPLOADS_DIR.exists():
        return chunks
    for path in sorted(config.UPLOADS_DIR.iterdir()):
        if path.suffix.lower() not in config.UPLOAD_EXTENSIONS:
            continue
        uploaded = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()
        chunks.extend(
            make_chunks(
                source_url=f"upload://{path.name}",
                source_type="upload",
                local_path=f"uploads/{path.name}",
                title=path.stem,
                text=upload_to_text(path),
                # an upload with no dates inside is treated as "as of upload day"
                fallback_date=uploaded,
            )
        )
    return chunks


MEDIA_KIND_BY_EXT = {
    **{ext: "document" for ext in config.DOCUMENT_EXTENSIONS},
    **{ext: "image" for ext in config.IMAGE_EXTENSIONS},
    **{ext: "audio" for ext in config.AUDIO_EXTENSIONS},
    **{ext: "video" for ext in config.VIDEO_EXTENSIONS},
}


def build_chunks() -> list[dict]:
    chunks: list[dict] = []
    if not config.MANIFEST_PATH.exists():
        log.warning("No crawl manifest at %s; indexing uploaded documents only.", config.MANIFEST_PATH)
        return build_upload_chunks()
    manifest = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))

    page_title_by_url = {p["url"]: p.get("title", "") for p in manifest["pages"]}

    for page in manifest["pages"]:
        html_path = config.DATA_DIR / page["local_path"]
        if not html_path.exists():
            continue
        chunks.extend(
            make_chunks(
                source_url=page["url"],
                source_type="page",
                local_path=page["local_path"],
                title=page.get("title", ""),
                text=html_to_text(html_path.read_text(encoding="utf-8")),
                fallback_date=_last_modified_date(page),
            )
        )

    for asset in manifest["assets"]:
        ext = asset["extension"]
        kind = MEDIA_KIND_BY_EXT.get(ext, "document")
        local_path = config.DATA_DIR / asset["local_path"]
        referring_title = page_title_by_url.get(asset["referenced_from"], "")
        display_name = asset.get("original_filename") or local_path.name
        fallback = _last_modified_date(asset)

        if kind == "document" and ext == ".pdf" and local_path.exists():
            text = pdf_to_text(local_path)
            if not text.strip():
                text = f"PDF document '{display_name}' referenced from page '{referring_title}'."
            # The file name often carries the date ("TT_July_2026.pdf").
            doc_title = f"{referring_title} - {display_name}" if referring_title else display_name
            chunks.extend(
                make_chunks(
                    source_url=asset["url"],
                    source_type="pdf",
                    local_path=asset["local_path"],
                    title=doc_title,
                    text=text,
                    fallback_date=fallback,
                )
            )
        else:
            description = (
                f"{kind.capitalize()} file '{display_name}' "
                f"referenced from page '{referring_title}' ({asset['referenced_from']})."
            )
            chunks.extend(
                make_chunks(
                    source_url=asset["url"],
                    source_type=kind,
                    local_path=asset["local_path"],
                    title=referring_title,
                    text=description,
                    fallback_date=fallback,
                )
            )

    chunks.extend(build_upload_chunks())
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
