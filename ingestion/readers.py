"""Step 2a of indexing: turn a file into plain text.

    html_to_text(html)      web page -> its main content as text
    pdf_extract(path)       PDF -> text (+ OCR for scanned pages, cached)
    docx_to_text(path)      Word file -> text
    upload_to_text(path)    any file type the admin page accepts

Nothing here knows about chunks, dates or the index -- just "file in,
text out", so a new file type is one new function here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

import config
from ingestion import ocr

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


PDF_CACHE_VERSION = "3"  # 3: partly scanned pages OCR'd too. Bump to force re-extraction of every PDF
MIN_PAGE_TEXT_CHARS = 30  # a page with less text than this is treated as scanned


def _pdf_cache_path(path: Path) -> Path:
    """Cache key = the file's content, not its location, so a copied data/
    folder (e.g. to the demo laptop) reuses the cache instead of redoing OCR."""
    digest = hashlib.sha1(path.read_bytes()).hexdigest()
    return config.PROCESSED_DIR / "pdf_text_cache" / f"{digest}-v{PDF_CACHE_VERSION}.json"


def _pdf_date(reader) -> str | None:
    """When the PDF file itself says it was made/last changed."""
    try:
        meta = reader.metadata
        for d in (meta.modification_date, meta.creation_date) if meta else ():
            if d and 1990 <= d.year <= datetime.now().year + 1:
                return d.date().isoformat()
    except Exception:
        pass
    return None


def _letters(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def partly_scanned_pages(page_texts: list[str], image_fractions: list[float], scanned: set[int]) -> list[int]:
    """Pages with some typed text that are mostly a picture: a typed header
    over a scanned body. Pages with plenty of typed text are left alone (many
    scans already carry a full hidden text layer)."""
    return [i for i, t in enumerate(page_texts)
            if i not in scanned and i < len(image_fractions)
            and image_fractions[i] >= config.OCR_MIXED_IMAGE_FRACTION
            and len(t.strip()) < config.OCR_MIXED_MAX_TEXT_CHARS]


def pdf_extract(path: Path) -> dict:
    """PDF -> {"text", "pdf_date", "pages", "scanned_pages", "mixed_pages",
    "ocr_pages", "ocr_confidence", "low_confidence_pages"}.

    Text pages are read directly; pages with no text (scans) and partly
    scanned pages are OCR'd when an OCR engine is installed. Results are
    cached by file content, so a rebuild only re-reads new or changed PDFs.
    """
    cache = None
    try:
        cache = _pdf_cache_path(path)
        if cache.exists():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            # Re-do a cached result only if it was made on a machine without
            # OCR and OCR is available now. Pages OCR found no words on
            # (blank pages, photos) must NOT trigger a redo every time.
            ocr_done = (cached.get("ocr_checked") or cached.get("ocr_attempted")
                        or cached.get("ocr_pages", 0) > 0)
            if ocr_done or not ocr.available():
                return cached
    except (OSError, ValueError):
        pass

    result = {"text": "", "pdf_date": None, "pages": 0, "scanned_pages": 0, "mixed_pages": 0, "ocr_pages": 0}
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf not installed; skipping PDF text extraction for %s", path)
        return result
    try:
        reader = PdfReader(str(path))
        page_texts = []
        for page in reader.pages:
            try:
                page_texts.append(page.extract_text() or "")
            except Exception:
                page_texts.append("")
        result["pdf_date"] = _pdf_date(reader)
    except Exception as exc:  # malformed PDFs happen in the wild
        log.warning("Failed to read %s: %s", path, exc)
        page_texts = []

    scanned = [i for i, t in enumerate(page_texts) if len(t.strip()) < MIN_PAGE_TEXT_CHARS]
    result["pages"] = len(page_texts)
    result["scanned_pages"] = len(scanned)
    can_ocr = bool(page_texts) and ocr.available()
    mixed = partly_scanned_pages(page_texts, ocr.image_fractions(path), set(scanned)) if can_ocr else []
    result["mixed_pages"] = len(mixed)
    result["ocr_checked"] = can_ocr
    if can_ocr and (scanned or mixed):
        result["ocr_attempted"] = True
        log.info("OCR: %s (%d scanned + %d partly scanned of %d pages)", path.name, len(scanned), len(mixed),
                 len(page_texts))
        confidence = {}
        for i, (text, conf) in ocr.ocr_pdf_pages_scored(path, scanned + mixed).items():
            if not text.strip() or _letters(text) <= _letters(page_texts[i]):
                continue  # nothing read, or the typed text was already as good
            page_texts[i] = text
            result["ocr_pages"] += 1
            if conf is not None:
                confidence[i] = conf
        if confidence:
            result["ocr_confidence"] = round(sum(confidence.values()) / len(confidence), 3)
            result["low_confidence_pages"] = sorted(i + 1 for i, c in confidence.items()
                                                    if c < config.OCR_LOW_CONFIDENCE)
    result["text"] = normalize_text("\n".join(page_texts))

    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return result


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


def upload_to_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return pdf_extract(path)["text"]
    if ext == ".docx":
        return docx_to_text(path)
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""
