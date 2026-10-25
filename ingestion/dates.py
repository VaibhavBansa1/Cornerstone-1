"""Find dates inside text, so every chunk knows roughly how old it is.

College documents rarely carry a clean "published on" date, but they are
full of clues: "w.e.f. 27/07/2026", "Session: July-Dec 2026", "10th March,
2025", "2021-22". We collect all of them and use the *newest* one as the
date of that piece of text. That is what lets the chatbot tell the current
HOD apart from one mentioned in a 2019 brochure.

Day-first order is assumed for numeric dates (Indian convention: 05/01/2026
is 5 January 2026).
"""
from __future__ import annotations

import re
from datetime import date, timedelta

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
_YEAR = r"((?:19|20)\d{2})"

# 27/07/2026, 05-01-2026, 25.05.2023
_NUMERIC = re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-]" + _YEAR + r"\b")
# 10th March, 2025 / 10 th March 2025 / 1st Dec. 2020
_DAY_MONTH_YEAR = re.compile(
    r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s+" + _MONTH + r",?\s+" + _YEAR + r"\b", re.I
)
# March 10, 2025
_MONTH_DAY_YEAR = re.compile(r"\b" + _MONTH + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+" + _YEAR + r"\b", re.I)
# July-Dec 2026 / Jan - June, 2026 (a session); we keep the starting month
_MONTH_RANGE_YEAR = re.compile(
    r"\b" + _MONTH + r"(?:\s*[-–/]\s*|\s+to\s+|\s+)" + _MONTH + r",?\s+" + _YEAR + r"\b", re.I
)
# March 2025 / Mar. 2025
_MONTH_YEAR = re.compile(r"\b" + _MONTH + r",?\s+" + _YEAR + r"\b", re.I)
# 2021-22 / 2021-2022 (academic year, starts in July)
_ACADEMIC_YEAR = re.compile(r"\b" + _YEAR + r"\s*[-–/]\s*(\d{2}|(?:19|20)\d{2})\b")
# a bare year on its own -- weakest clue, only used if nothing better exists
_BARE_YEAR = re.compile(r"(?<![\d/.\-])" + _YEAR + r"(?![\d/.\-]\d)")

MIN_YEAR = 1990


def _month(token: str) -> int:
    t = token.lower().rstrip(".")
    return MONTHS.get(t) or MONTHS[t[:3]]


def _safe_date(year: int, month: int, day: int = 1) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        try:
            return date(year, month, 1)
        except ValueError:
            return None


def _latest_allowed(today: date) -> date:
    # Timetables are often published a little before their session starts,
    # so allow ~2 months ahead. Anything later ("valid till April 2027",
    # "for three years") is a deadline, not when the text was written.
    return today + timedelta(days=62)


def find_dates(text: str, today: date | None = None) -> tuple[list[date], list[date]]:
    """Return (specific_dates, bare_year_dates) found in ``text``."""
    today = today or date.today()
    latest = _latest_allowed(today)
    specific: list[date] = []

    def add(d: date | None):
        if d and MIN_YEAR <= d.year and d <= latest:
            specific.append(d)

    # Strongest patterns run first; each blanks out what it matched so a
    # weaker pattern can't re-read part of it ("July-Dec 2026" must not
    # also count as "Dec 2026").
    def matches(pattern):
        nonlocal text
        found = []

        def blank(m):
            found.append(m)
            return " " * (m.end() - m.start())

        text = pattern.sub(blank, text)  # one pass, fast even on huge PDFs
        return found

    for m in matches(_NUMERIC):
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if month > 12 and day <= 12:  # clearly month-first (e.g. 07/27/2026)
            day, month = month, day
        if 1 <= month <= 12:
            add(_safe_date(year, month, day))
    for m in matches(_DAY_MONTH_YEAR):
        add(_safe_date(int(m.group(3)), _month(m.group(2)), int(m.group(1))))
    for m in matches(_MONTH_DAY_YEAR):
        add(_safe_date(int(m.group(3)), _month(m.group(1)), int(m.group(2))))
    for m in matches(_MONTH_RANGE_YEAR):
        start, end, year = _month(m.group(1)), _month(m.group(2)), int(m.group(3))
        # "Nov-Feb 2026": the session started in the previous year
        add(_safe_date(year - 1 if start > end else year, start))
    for m in matches(_MONTH_YEAR):
        add(_safe_date(int(m.group(2)), _month(m.group(1))))
    for m in matches(_ACADEMIC_YEAR):
        start = int(m.group(1))
        end_raw = m.group(2)
        end = int(end_raw) if len(end_raw) == 4 else (start // 100) * 100 + int(end_raw)
        if end == start + 1:
            add(_safe_date(start, 7))

    bare: list[date] = []
    for m in _BARE_YEAR.finditer(text):
        year = int(m.group(1))
        if MIN_YEAR <= year <= latest.year:
            bare.append(date(year, 1, 1))
    return specific, bare


def latest_date(text: str, today: date | None = None) -> str | None:
    """Newest date mentioned in ``text`` as 'YYYY-MM-DD', or None."""
    specific, bare = find_dates(text, today)
    pool = specific or bare
    return max(pool).isoformat() if pool else None


def label(iso: str | None) -> str:
    """'2026-07-27' -> 'Jul 2026'; None -> 'undated'."""
    if not iso:
        return "undated"
    try:
        d = date.fromisoformat(iso)
    except ValueError:
        return "undated"
    return d.strftime("%b %Y")
