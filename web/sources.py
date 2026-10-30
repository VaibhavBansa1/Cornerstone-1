"""What the chat shows about a source: its name, its links, citations.

    source_label(chunk)        "Syllabus · V SEM · 2024-28 BATCH", "Faculty Profiles", "1st sem TT.pdf"
    public_source(n, chunk)    the JSON one source card / Sources-panel entry needs
    saved_copy_link(chunk)     /files/... link to our downloaded copy (or None)
    cited_numbers(answer)      which [n] markers the AI's answer used
    render_saved_page(...)     a readable version of a saved web page

Kept apart from app.py so the routes file stays a short list of URLs.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote, unquote

import config
from ingestion.dates import label as date_label
from rag import pipeline


# Link texts that say nothing about the file.
_GENERIC_LINK = re.compile(
    r"^\s*(view( more)?( details)?|more details|details|read more|click here|here|download|link|open"
    r"|download file|<download>|file|report|activity report|brochure|pdf)\s*$", re.I)


def source_label(chunk: dict) -> str:
    """Short, human name for a source card. A file linked as "CLICK HERE"
    in a table gets its table position ("Syllabus · V SEM · 2024-28 BATCH");
    otherwise the file name for documents and the page title for pages."""
    label = (chunk.get("link_label") or "").strip()
    if _GENERIC_LINK.match(label):  # "View More Details", "Report": the file name says more
        page = (chunk.get("title") or "").split(" - ")[0].strip()
        name = unquote(_name_from_title_or_url(chunk))
        return f"{page} · {name}" if page and chunk.get("source_type") != "page" else name
    if label and chunk.get("source_type") != "page":
        page = (chunk.get("title") or "").split(" - ")[0].strip()
        if page and page.lower() not in label.lower():
            return f"{page} · {label}"
        return label
    return unquote(_name_from_title_or_url(chunk))


def _name_from_title_or_url(chunk: dict) -> str:
    title = (chunk.get("title") or "").strip()
    if chunk.get("source_type") == "upload":
        return title or chunk["source_url"].removeprefix("upload://")
    if chunk.get("source_type") in ("pdf", "image", "audio", "video", "document"):
        # PDFs are titled "<page title> - <file name>"; Drive links end in
        # "/view", so the URL is no good as a name.
        url_name = chunk["source_url"].rstrip("/").rsplit("/", 1)[-1]
        if " - " in title:
            name = title.rsplit(" - ", 1)[-1].strip()
            # "uc__1a2b.pdf" / "images__Dept__x.pdf__9f3e.pdf" are our own
            # saved-file names; the link's last part is the real file name.
            looks_internal = name.lower().startswith("uc__") or "__" in name
            if name and not looks_internal:
                return name
            if looks_internal and url_name.lower() not in ("view", "edit", "preview"):
                return url_name
        name = url_name
        if name.lower() in ("view", "edit", "preview") or name.startswith("uc?"):
            return title or "Google Drive file"
        return name
    return title.replace(" - MITS Gwalior", "") or chunk["source_url"].rstrip("/").rsplit("/", 1)[-1]


def saved_copy_link(chunk: dict) -> str | None:
    """URL of our own downloaded copy of this source (works even if the
    college site or Google Drive is down; the only link for uploads)."""
    local = (chunk.get("local_path") or "").replace("\\", "/")
    if not local or not (config.DATA_DIR / local).exists():
        return None
    return "/files/" + quote(local)


def public_source(n: int, chunk: dict) -> dict:
    url = chunk["source_url"]
    return {
        "n": n,
        "title": source_label(chunk),
        "doc_title": chunk.get("title") or "",
        "source_url": url,
        "link": None if url.startswith("upload://") else url,
        "saved_copy": saved_copy_link(chunk),
        "source_type": chunk.get("source_type"),
        "score": round(chunk.get("score", 0.0), 3),
        "date": date_label(chunk.get("date")),
        "snippet": chunk["text"][:400],
        "text": chunk["text"],
    }


_CITATION = re.compile(r"\[(\s*\d+(?:\s*[,\u2013-]\s*\d+)*\s*)\]")


def cited_numbers(answer: str) -> set[int]:
    """Source numbers cited in an answer: [2], [1, 3], [4-6] and [1][2]."""
    found: set[int] = set()
    for group in _CITATION.findall(answer or ""):
        for part in group.split(","):
            part = part.strip()
            bounds = re.split(r"\s*[\u2013-]\s*", part)
            if len(bounds) == 2 and all(b.isdigit() for b in bounds):
                lo, hi = int(bounds[0]), int(bounds[1])
                if 0 < hi - lo < 20:
                    found.update(range(lo, hi + 1))
                    continue
            if part.isdigit():
                found.add(int(part))
    return found


SAVED_PAGE_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{base}<title>{title} (saved copy)</title>
<style>
body {{ font-family: "Segoe UI", Roboto, Arial, sans-serif; margin: 0; background: #f4f6f9; color: #1c1f24; line-height: 1.55; }}
.bar {{ background: #1f3b73; color: #fff; padding: 10px 16px; font-size: 14px; }}
.bar a {{ color: #fff; font-weight: 600; }}
main {{ max-width: 860px; margin: 16px auto; background: #fff; border: 1px solid #dde2ea; border-radius: 12px; padding: 20px 24px; }}
img {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; display: block; overflow-x: auto; }}
td, th {{ border: 1px solid #dde2ea; padding: 6px 8px; vertical-align: top; }}
h1, h2, h3 {{ color: #1f3b73; }}
</style></head><body>
<div class="bar">Saved copy of <b>{title}</b> from the help desk index{live}</div>
<main>{content}</main></body></html>"""


def render_saved_page(path: Path, original: str | None) -> str:
    """Our saved copy of a web page, showing only its main content.

    The college site's own styles and menus don't load outside the site,
    so the raw page looks broken; this shows the readable part instead."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    main = soup.find(id=config.MAIN_CONTENT_CONTAINER_ID) or soup.body or soup
    for tag in main(["script", "style", "noscript", "iframe", "form", "nav", "aside"]):
        tag.decompose()
    esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")  # noqa: E731
    return SAVED_PAGE_TEMPLATE.format(
        base=f'<base href="{esc(original)}" target="_blank">' if original else "",
        title=esc(title),
        live=f' &middot; <a href="{esc(original)}" target="_blank" rel="noopener">Open the live page</a>' if original else "",
        content=str(main),
    )


def original_url_for(local_path: str) -> str | None:
    if not pipeline.index_exists():
        return None
    try:
        chunks = pipeline.indexed_chunks()
    except Exception:
        return None
    for c in chunks:
        if (c.get("local_path") or "").replace("\\", "/") == local_path:
            return c["source_url"]
    return None
