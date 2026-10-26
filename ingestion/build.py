"""Step 2 of indexing: everything that was crawled or uploaded -> chunks.

Reads data/manifest.json (written by the crawler) and data/uploads/, and
writes data/processed/chunks.jsonl. For every page and file it:

  1. reads the text                      (ingestion/readers.py, OCR for scans)
  2. names files linked as "CLICK HERE"  (ingestion/link_labels.py)
  3. cuts the text into dated chunks     (ingestion/chunking.py)
  4. adds a course-list chunk            (ingestion/course_list.py)

Images/audio/video have no text, so they get one short descriptive chunk
(file name + the page that links it) so they can still be found and cited.

    python -m ingestion.build    # rebuild chunks.jsonl from the saved crawl
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import config
from ingestion.chunking import make_chunks, with_course_list
from ingestion.link_labels import labels_from_page
from ingestion.readers import html_to_text, pdf_extract, upload_to_text

log = logging.getLogger("ingest")

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


def build_upload_chunks() -> list[dict]:
    """Chunks for documents staff uploaded through the admin page."""
    chunks: list[dict] = []
    if not config.UPLOADS_DIR.exists():
        return chunks
    for path in sorted(config.UPLOADS_DIR.iterdir()):
        if path.suffix.lower() not in config.UPLOAD_EXTENSIONS:
            continue
        uploaded = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()
        text = upload_to_text(path)
        chunks.extend(with_course_list(
            make_chunks(
                source_url=f"upload://{path.name}",
                source_type="upload",
                local_path=f"uploads/{path.name}",
                title=path.stem,
                text=text,
                # an upload with no dates inside is treated as "as of upload day"
                fallback_date=uploaded,
            ), text))
    return chunks


MEDIA_KIND_BY_EXT = {
    **{ext: "document" for ext in config.DOCUMENT_EXTENSIONS},
    **{ext: "image" for ext in config.IMAGE_EXTENSIONS},
    **{ext: "audio" for ext in config.AUDIO_EXTENSIONS},
    **{ext: "video" for ext in config.VIDEO_EXTENSIONS},
}


def _link_labels(manifest: dict) -> dict[str, str]:
    """{asset url: label} from how each file is linked on the saved pages."""
    from scraper.crawler import normalize_url

    labels: dict[str, str] = {}
    for page in manifest["pages"]:
        html_path = config.DATA_DIR / page["local_path"]
        if not html_path.exists():
            continue
        try:
            found = labels_from_page(html_path.read_text(encoding="utf-8"), page["url"], normalize_url)
        except Exception as exc:
            log.warning("Could not read link labels from %s: %s", page["url"], exc)
            continue
        for url, label in found.items():
            labels.setdefault(url, label)
    return labels


def build_chunks() -> list[dict]:
    chunks: list[dict] = []
    if not config.MANIFEST_PATH.exists():
        log.warning("No crawl manifest at %s; indexing uploaded documents only.", config.MANIFEST_PATH)
        return build_upload_chunks()
    manifest = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))

    page_title_by_url = {p["url"]: p.get("title", "") for p in manifest["pages"]}
    link_labels = _link_labels(manifest)

    for page in manifest["pages"]:
        html_path = config.DATA_DIR / page["local_path"]
        if not html_path.exists():
            continue
        page_text = html_to_text(html_path.read_text(encoding="utf-8"))
        chunks.extend(with_course_list(
            make_chunks(
                source_url=page["url"],
                source_type="page",
                local_path=page["local_path"],
                title=page.get("title", ""),
                text=page_text,
                fallback_date=_last_modified_date(page),
            ), page_text))

    for asset in manifest["assets"]:
        ext = asset["extension"]
        kind = MEDIA_KIND_BY_EXT.get(ext, "document")
        local_path = config.DATA_DIR / asset["local_path"]
        referring_title = page_title_by_url.get(asset["referenced_from"], "")
        display_name = asset.get("original_filename") or local_path.name
        fallback = _last_modified_date(asset)

        label = link_labels.get(asset["url"], "")
        if kind == "document" and ext == ".pdf" and local_path.exists():
            info = pdf_extract(local_path)
            text = info["text"]
            if not text.strip():
                text = f"PDF document '{display_name}' referenced from page '{referring_title}'."
            # Title = page + where it's linked + file name, e.g.
            # "Syllabus - MITS Gwalior - V SEM · 2024-28 BATCH - 5th 11_merged.pdf"
            doc_title = " - ".join(p for p in (referring_title, label, display_name) if p)
            chunks.extend(with_course_list(
                make_chunks(
                    source_url=asset["url"],
                    source_type="pdf",
                    local_path=asset["local_path"],
                    title=doc_title,
                    text=text,
                    # no date in the text -> when the PDF was made -> server date
                    fallback_date=info.get("pdf_date") or fallback,
                    link_label=label,
                ), text))
        else:
            description = (
                f"{kind.capitalize()} file '{display_name}' "
                f"referenced from page '{referring_title}' ({asset['referenced_from']})."
            )
            if label:
                description = f"{description} Linked as: {label}."
            chunks.extend(
                make_chunks(
                    source_url=asset["url"],
                    source_type=kind,
                    local_path=asset["local_path"],
                    title=referring_title,
                    text=description,
                    fallback_date=fallback,
                    link_label=label,
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
