"""Pull the list of courses (name + code) out of a syllabus or timetable.

Why: "what subjects are in 5th sem?" needs the WHOLE list, but a search
only returns a few small pieces of a 19-page syllabus -- often the pages
about projects, not subjects. So every document with courses also gets one
extra piece: its course list.

Works by layout, not by code format, because codes change: older batches
use 7-digit codes (3160512), newer ones (after MITS became a Deemed
University) 8-digit codes (16243101), and future ones may differ again.
Patterns found in MITS documents:

    MOBILE COMPUTING & WIRELESS NETWORK      <- name line
    16243101                                 <- code line (maybe "LAB 16243107")

    PROBABILITY AND RANDOM PROCESS 3160511   <- name and code on one line

    16251201 Data Structures 16251202 ...    <- timetable "Course Code / Course Name" table
"""
from __future__ import annotations

import re

CODE = r"\d{6,9}"
_CODE_LINE = re.compile(rf"^(?:lab|code|course\s*code)?\s*[:.-]?\s*({CODE})\b(.{{0,25}})$", re.I)
_NAME_THEN_CODE = re.compile(rf"^(.{{4,80}}?)[\s:,-]+({CODE})$")
_CODE_THEN_NAME = re.compile(rf"\b({CODE})\s+([A-Za-z][A-Za-z&/().,'\- ]{{3,70}}?)(?=\s+{CODE}\b|\s*$|\s{{2,}})")

BOILERPLATE = (
    "department", "institute", "naac", "university", "accredited", "government", "ministry",
    "gwalior", "objective", "total credit", "list of", "unit-", "unit ", "course outcome",
    "outcomes", "credits", "semester", "session", "time table", "timetable", "room no",
    "w.e.f", "page", "madhav", "declared", "category", "experiments", "programs", "co-po",
    "articulation", "matrix", "slightly", "moderately", "course code", "course name",
)


def _title_like(text: str) -> bool:
    t = text.strip(" :-,.()")
    if not 4 <= len(t) <= 80:
        return False
    letters = sum(ch.isalpha() for ch in t)
    if letters < 0.6 * len(t.replace(" ", "")):
        return False
    if not re.search(r"[A-Za-z]{4,}", t):
        return False
    low = t.lower()
    return not any(b in low for b in BOILERPLATE)


def _nice(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" :-,.()")
    if name.isupper():
        small = {"and", "of", "in", "for", "the", "to", "with", "&"}
        words = name.lower().split()
        name = " ".join(w if (i and w in small) else (w.upper() if len(w) <= 2 and w.isalpha() and w not in small and i else w.capitalize())
                        for i, w in enumerate(words))
    return name


def extract_courses(text: str) -> dict[str, list[str]]:
    """{code: [names]} in the order codes appear."""
    found: dict[str, list[str]] = {}

    def add(code: str, name: str):
        name = _nice(name)
        if not _title_like(name):
            return
        names = found.setdefault(code, [])
        if name.lower() not in (n.lower() for n in names):
            names.append(name)

    lines = [l.strip() for l in text.splitlines()]
    for i, line in enumerate(lines):
        if not line:
            continue
        m = _CODE_LINE.match(line)
        if m and not re.search(r"[A-Za-z]{4,}", m.group(2)):
            # name is on one of the 2 lines above, skipping letterhead lines
            for back in (1, 2):
                if i - back >= 0 and _title_like(lines[i - back]):
                    add(m.group(1), lines[i - back])
                    break
            continue
        m = _NAME_THEN_CODE.match(line)
        if m and _title_like(m.group(1)) and not re.search(CODE, m.group(1)):
            add(m.group(2), m.group(1))
            continue
        for code, name in _CODE_THEN_NAME.findall(line):
            if len(re.findall(r"[A-Za-z]{4,}", name)) >= 1 and not re.fullmatch(r"(?:[A-Z0-9+/]{1,4}\s*)+", name.strip()):
                add(code, name)
    return {c: n for c, n in found.items() if n}


def course_list_text(text: str, doc_name: str) -> str | None:
    """A readable course list for ``doc_name``, or None if fewer than 2 courses."""
    courses = extract_courses(text)
    if len(courses) < 2:
        return None
    lines = [f"Course list (all subjects/courses) in {doc_name}. Course codes exactly as printed in this document:"]
    for code, names in sorted(courses.items()):
        lines.append(f"- {code}: {' / '.join(names)}")
    return "\n".join(lines)
