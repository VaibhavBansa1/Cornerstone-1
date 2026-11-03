"""What each document IS, read from the document's own text.

Website links often say nothing ("View More Details") and file names are
inconsistent ("5th sem TT.pdf", "jan 25 3rd.pdf", "jul24 5.pdf"). But the
document itself starts with a header written for people:

    Time-Table, B.Tech, IT Semester-V (Session: July-Dec 2026)

So when the index loads, the header of every document is read once and
turned into a label:

    type      timetable / exam_timetable / syllabus / None
    programme btech / mtech / None
    semesters {5}            (empty if none or unclear)
    session   "July-Dec 2026" (None if not stated)

Search then matches labels with what the question is about (see
rag/retrieval.py), so a new file name or a new way of asking needs no new
word list. Old scans whose OCR text is garbled get an empty label and are
simply not boosted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}
_R = r"(VIII|VII|VI|IV|V|III|II|I)"
_ORD = r"([1-8])\s*(?:st|nd|rd|th)"

# "Semester-V", "Semester: 5", "Sem V"
_SEM_AFTER = re.compile(rf"\bsem(?:ester)?\.?\s*[-:(]?\s*(?:{_R}|([1-8]))\b", re.I)
# "V Semester", "5th Sem", "III SEM", "B. TECH I Semester"
_SEM_BEFORE = re.compile(rf"\b(?:{_R}|{_ORD})\s*[-]?\s*sem(?:ester)?s?\b", re.I)
# "July-Dec 2026", "JULY-DEC. 2026", "Jan-June 2026", "july dec2024", "January - June, 2025"
_SESSION = re.compile(
    r"\b(jul(?:y)?|jan(?:uary)?)\.?\s*[-–to ]*\s*(dec(?:ember)?|jun(?:e)?)\.?[\s,.-]*((?:20)\d{2})", re.I)
_TIMETABLE = re.compile(r"\btime\s*-?\s*table", re.I)
_TT_NAME = re.compile(r"(?<![a-z])tt(?![a-z])|time\s*-?\s*table", re.I)   # "5th sem TT.pdf" (file names only)
_EXAM = re.compile(r"\b(mid\s*-?\s*sem|minor|major|exam(?:ination)?s?|evaluation|practical|back)\b", re.I)
_DAY = re.compile(r"\b(MON|TUE|WED|THU|FRI|SAT)\b")
_PERIOD = re.compile(r"\b\d{1,2}[:.]\d{2}\s*-\s*\d{1,2}[:.]\d{2}\b")
_SYLLABUS = re.compile(r"\b(syllabus|scheme of (?:examination|evaluation|studies)|course structure)\b", re.I)
_MTECH = re.compile(r"\bM\.?\s*Tech\b", re.I)
_BTECH = re.compile(r"\bB\.?\s*Tech\b", re.I)

HEADER_CHARS = 900


@dataclass
class DocLabel:
    type: str | None = None
    programme: str | None = None
    semesters: set[int] = field(default_factory=set)
    session: str | None = None

    def describe(self) -> str:
        parts = [self.type or "?", self.programme or "?"]
        parts.append("sem " + ",".join(map(str, sorted(self.semesters))) if self.semesters else "sem ?")
        parts.append(self.session or "session ?")
        return " · ".join(parts)


def _sem_numbers(text: str) -> list[int]:
    found = []
    for m in _SEM_AFTER.finditer(text):
        found.append(ROMAN[m.group(1).lower()] if m.group(1) else int(m.group(2)))
    for m in _SEM_BEFORE.finditer(text):
        found.append(ROMAN[m.group(1).lower()] if m.group(1) else int(m.group(2)))
    return found


def session_of(text: str) -> str | None:
    m = _SESSION.search(text)
    if not m:
        return None
    year = int(m.group(3))
    return f"July-Dec {year}" if m.group(1).lower().startswith("jul") else f"Jan-June {year}"


def _header(chunks: list[dict]) -> str:
    """Start of the document's real text (generated course lists skipped)."""
    parts = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
    text = " ".join(c["text"] for c in parts
                    if c.get("kind") != "course_list" and not c["text"].startswith("Course list ("))
    return re.sub(r"\s+", " ", text)


def label_document(chunks: list[dict]) -> DocLabel:
    first = chunks[0]
    name = first.get("title", "").rsplit(" - ", 1)[-1]          # file name or page title
    where = f"{first.get('link_label', '')} {first.get('title', '')}"
    text = _header(chunks)
    head = text[:HEADER_CHARS]
    label = DocLabel()

    # Weekday rows and period times: a class timetable grid.
    grid = len(set(_DAY.findall(text[:4000]))) >= 4 and len(_PERIOD.findall(text[:4000])) >= 3
    is_tt = bool(_TIMETABLE.search(head) or _TT_NAME.search(name))
    if is_tt and (_EXAM.search(head[:400]) or _EXAM.search(name)):
        label.type = "exam_timetable"
    elif is_tt and re.search(r"remedial", head[:400] + name, re.I):
        label.type = "remedial_timetable"
    elif is_tt or grid:
        label.type = "timetable"
    elif _SYLLABUS.search(head) or _SYLLABUS.search(where):
        label.type = "syllabus"

    if _MTECH.search(head[:400]) or _MTECH.search(name):
        label.programme = "mtech"
    elif _BTECH.search(head[:400]) or _BTECH.search(where):
        label.programme = "btech"

    if first.get("source_type") in ("page", "image"):
        label.type = None    # web pages list many documents; images are just references
    # Syllabus files cover many semesters, so where they are filed says more
    # ("V SEM · 2024-28 BATCH"); a timetable's own header is the best guide.
    order = ((first.get("link_label", ""), name, head) if label.type == "syllabus"
             else (head, name, first.get("link_label", "")))
    sems = next((found for found in map(_sem_numbers, order) if found), [])
    if sems:
        # A header names its own semester; a list of many (exam schedules
        # for "II, IV and VI sem") is not about one semester.
        distinct = set(sems)
        label.semesters = distinct if len(distinct) <= 2 else set()
        # Exam schedules usually cover several years ("II AND III YEAR").
        if label.type == "exam_timetable" and (len(distinct) > 1 or re.search(r"\byear\b", head[:400] + name, re.I)):
            label.semesters = set()
    label.session = session_of(head) or session_of(name)
    return label


def label_documents(by_source: dict[str, list[dict]]) -> dict[str, DocLabel]:
    return {url: label_document(chunks) for url, chunks in by_source.items() if chunks}
