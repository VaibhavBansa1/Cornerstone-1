"""Give linked files a meaningful name from where they are linked.

College pages link most files as "CLICK HERE" inside a table, e.g. the
syllabus page:

                 2026-30 BATCH   2025-29 BATCH   2024-28 BATCH
    V SEM        ************    ************    CLICK HERE  -> 5th 11_merged.pdf

The file name ("5th 11_merged.pdf") says nothing about the batch, so
without this the chatbot can't tell which syllabus belongs to whom. Here we
re-read the saved page and label the file from its table row + column
header: "V SEM · 2024-28 BATCH". Links with real text ("Time Table July
2026") keep that text.
"""
from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

import config

GENERIC_LINK_TEXT = {
    "", "click here", "click", "here", "view", "download", "open", "link", "pdf",
    "read more", "more", "details", "view pdf", "download pdf", "see", "show",
}


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip(" :-|*")
    text = re.sub(r"(\d{4})\s*-\s*(\d{2,4})", r"\1-\2", text)  # "2024 -28" -> "2024-28"
    return "" if set(text) <= {"*", " "} else text


def _drive_view_url(url: str) -> str | None:
    m = re.search(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:.*&)?id=)([\w-]{10,})", url)
    return f"https://drive.google.com/file/d/{m.group(1)}/view" if m else None


def _cell_text(cell) -> str:
    return _clean(cell.get_text(" ", strip=True)) if cell else ""


def _table_label(a) -> str:
    cell = a.find_parent(["td", "th"])
    row = cell.find_parent("tr") if cell else None
    table = row.find_parent("table") if row else None
    if not table:
        return ""
    cells = row.find_all(["td", "th"], recursive=False)
    try:
        col = cells.index(cell)
    except ValueError:
        return ""
    parts = []
    row_head = _cell_text(cells[0]) if col > 0 else ""
    if row_head and row_head.lower() not in GENERIC_LINK_TEXT:
        parts.append(row_head)
    header_row = table.find("tr")
    if header_row is not None and header_row is not row:
        heads = header_row.find_all(["td", "th"], recursive=False)
        if col < len(heads):
            head = _cell_text(heads[col])
            if head and head.lower() not in GENERIC_LINK_TEXT:
                parts.append(head)
    return " · ".join(parts)


def _nearby_label(a) -> str:
    """Text around a generic link outside a table, e.g. "Notice for exams (click here)"."""
    parent = a.find_parent(["li", "p", "div"])
    if not parent:
        return ""
    text = _clean(parent.get_text(" ", strip=True).replace(a.get_text(" ", strip=True), ""))
    return text[:90] if 3 <= len(text) <= 160 else ""


def label_for_anchor(a) -> str:
    text = _clean(a.get_text(" ", strip=True))
    if text.lower() not in GENERIC_LINK_TEXT and not re.fullmatch(r"[\W\d_]*", text):
        return text[:120]
    return _table_label(a) or _nearby_label(a)


def labels_from_page(html: str, page_url: str, normalize) -> dict[str, str]:
    """{asset url (as the crawler stores it): label} for every link on a page."""
    soup = BeautifulSoup(html, "html.parser")
    scope = soup.find(id=config.MAIN_CONTENT_CONTAINER_ID) or soup
    labels: dict[str, str] = {}
    for a in scope.find_all("a", href=True):
        absolute = normalize(urljoin(page_url, a["href"]))
        key = _drive_view_url(absolute) or absolute
        label = label_for_anchor(a)
        if label and key not in labels:
            labels[key] = label
    return labels
