"""Step 0 of answering: work out WHAT the student is asking about.

Before searching, a question like "fifth semester syllabus 2026" is turned
into facts the search can use:

    semester 5  ·  batch 2024-28 (admitted 2024)  ·  July-Dec 2026 session  ·  topic: syllabus

Why: documents are filed by batch ("V SEM · 2024-28 BATCH"), but students
ask by semester or year. And every batch has its own syllabus and course
codes (codes changed when MITS became a Deemed University), so the batch
must be right. The rules are written out here (no AI call), so they are
fast, free and predictable; ``understand()`` is the one function to change
if you later want an LLM to do this step instead.

Calendar facts used: sessions run July-June; odd semesters (1,3,5,7) are
July-Dec, even ones Jan-June; a batch "2024-28" was admitted in July 2024.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}
ROMAN_BY_NUM = {v: k.upper() for k, v in ROMAN.items()}
WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
         # Hinglish
         "pehla": 1, "pehle": 1, "dusra": 2, "dusre": 2, "teesra": 3, "teesre": 3, "chautha": 4, "chauthe": 4,
         "panchva": 5, "panchve": 5, "paanchva": 5, "paanchve": 5, "chhata": 6, "chhate": 6,
         "satva": 7, "satve": 7, "aathva": 8, "aathve": 8}
ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th", 8: "8th"}

_NUM = r"(?:[1-8](?:st|nd|rd|th)?|" + "|".join(sorted(WORDS, key=len, reverse=True)) + r"|viii|vii|vi|iv|v|iii|ii|i)"
_SEM_WORD = r"(?:sem(?:ester)?s?|sem\.)"
# "5th sem", "sem 5", "semester-V", "V sem", "fifth semester"
_SEM_BEFORE = re.compile(rf"\b({_NUM})\s*-?\s*{_SEM_WORD}\b", re.I)
_SEM_AFTER = re.compile(rf"\b{_SEM_WORD}\s*[-:]?\s*({_NUM})\b", re.I)
# "3rd year", "final year"
_YEAR_OF_STUDY = re.compile(r"\b([1-4](?:st|nd|rd|th)?|first|second|third|fourth|final)\s*year\b", re.I)
# "2024-28", "2024 - 2028", "2024-28 batch"
_BATCH_RANGE = re.compile(r"\b((?:19|20)\d{2})\s*-\s*(\d{2}|(?:19|20)\d{2})\b")
# "2025 batch", "batch 2025", "2025 students", "admitted in 2025", "2025 admission"
_BATCH_HINT = re.compile(
    r"\b((?:19|20)\d{2})\s*(?:batch|students?|walo|wale|admitted|admission|entry)\b"
    r"|\b(?:batch|admitted(?:\s+in)?|admission(?:\s+in)?|joined(?:\s+in)?)\s*((?:19|20)\d{2})\b",
    re.I,
)
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")

TOPICS = {
    "syllabus": ("syllabus", "syllabi", "subjects", "subject", "courses", "course", "curriculum", "scheme", "papers"),
    "timetable": ("timetable", "time table", "time-table", "tt", "schedule", "classes", "lecture"),
    "exam": ("exam", "exams", "evaluation", "major", "minor", "mid sem", "midsem", "test", "paper"),
    "result": ("result", "results", "marks", "grade", "sgpa", "cgpa"),
    "faculty": ("faculty", "teacher", "teachers", "professor", "hod", "dean", "coordinator", "head"),
    "labs": ("lab", "labs", "laboratory", "laboratories"),
    "contacts": ("contact", "contacts", "phone", "mobile", "email", "e-mail"),
    "programmes": ("programmes", "programs offered", "intake", "degrees"),
    "overview": ("everything", "overview", "introduction", "about the department", "about the school",
                 "about it dept", "about it department", "about the it", "about dept"),
}


def _num(token: str) -> int | None:
    t = token.lower().rstrip(".")
    if t in WORDS:
        return WORDS[t]
    if t in ROMAN:
        return ROMAN[t]
    m = re.match(r"([1-8])", t)
    return int(m.group(1)) if m else None


def session_start_year(today: date) -> int:
    return today.year if today.month >= 7 else today.year - 1


def semester_of(admitted: int, today: date) -> int:
    """Semester a batch is in right now (may be <1 or >8 for other batches)."""
    return 2 * (session_start_year(today) - admitted) + (1 if today.month >= 7 else 2)


def batch_in(semester: int, today: date) -> int:
    """Admission year of the batch that is in (or next takes) ``semester``."""
    return session_start_year(today) - (semester - 1) // 2


def session_label(session_start: int, odd: bool) -> str:
    return f"July-Dec {session_start}" if odd else f"Jan-June {session_start + 1}"


@dataclass
class Understanding:
    semester: int | None = None
    batch: int | None = None           # admission year: 2024 = the 2024-28 batch
    session: str | None = None         # "July-Dec 2026"
    topic: str | None = None           # syllabus / timetable / exam / result / faculty
    notes: list[str] = field(default_factory=list)  # how each fact was worked out
    year_phrase: str | None = None     # "3rd year", replaced by "5th sem" for keyword search

    @property
    def batch_label(self) -> str | None:
        return f"{self.batch}-{(self.batch + 4) % 100:02d}" if self.batch else None

    @property
    def empty(self) -> bool:
        return not (self.semester or self.batch)

    def describe(self) -> str:
        """'5th sem · 2024-28 batch · July-Dec 2026 session · syllabus'"""
        parts = []
        if self.semester:
            parts.append(f"{ORDINAL[self.semester]} sem")
        if self.batch:
            parts.append(f"{self.batch_label} batch")
        if self.session:
            parts.append(f"{self.session} session")
        if self.topic:
            parts.append(self.topic)
        return " · ".join(parts)

    def search_terms(self) -> list[str]:
        """Extra keywords so search finds documents filed under this batch."""
        terms = []
        if self.batch:
            terms += [str(self.batch), f"{(self.batch + 4) % 100:02d}", "batch"]
        if self.semester:
            terms += [str(self.semester), ORDINAL[self.semester], ROMAN_BY_NUM[self.semester].lower(), "sem"]
        return terms

    def rewrite(self, question: str) -> str:
        """The question with "3rd year" replaced by "5th sem", so keyword
        search doesn't match the 3rd *semester* timetable."""
        if self.year_phrase and self.semester:
            return question.replace(self.year_phrase, f"{ORDINAL[self.semester]} sem")
        return question

    def label_matches(self, label: str) -> int:
        """+1: a document filed under this batch (and semester, if known);
        -1: filed under a DIFFERENT batch or semester; 0: no batch info."""
        if not label or self.empty:
            return 0
        m = re.search(r"\b(VIII|VII|VI|IV|V|III|II|I)\s*SEM\b.*?\b((?:19|20)\d{2})\s*-\s*\d{2,4}", label, re.I)
        if not m:
            return 0
        sem, batch = ROMAN[m.group(1).lower()], int(m.group(2))
        if self.batch and batch != self.batch:
            return -1
        if self.semester and sem != self.semester:
            return -1
        return 1


def understand(question: str, today: date | None = None) -> Understanding:
    today = today or date.today()
    q = question.lower()
    u = Understanding()
    start = session_start_year(today)

    for word, names in TOPICS.items():
        if any(re.search(rf"\b{re.escape(n)}\b", q) for n in names):
            u.topic = word
            break

    m = _SEM_BEFORE.search(question) or _SEM_AFTER.search(question)
    if m:
        u.semester = _num(m.group(1))
    else:
        y = _YEAR_OF_STUDY.search(question)
        if y:
            yr = 4 if y.group(1).lower() == "final" else (_num(y.group(1)) or WORDS.get(y.group(1).lower()))
            if yr:
                # "3rd year" = sem 5 in July-Dec, sem 6 in Jan-June
                u.semester = 2 * yr - (1 if today.month >= 7 else 0)
                u.year_phrase = y.group(0)
                u.notes.append(f"year {yr} of study = semester {u.semester} right now")

    years = [int(x) for x in _YEAR.findall(question)]
    r = _BATCH_RANGE.search(question)
    h = _BATCH_HINT.search(question)
    if r:
        u.batch = int(r.group(1))
        u.notes.append(f'"{r.group(0)}" is a batch')
    elif h:
        u.batch = int(h.group(1) or h.group(2))
        u.notes.append(f"{u.batch} is the admission year")
    elif years and u.semester:
        y = years[0]
        # "2026 5th sem": the 2026 batch can't be in 5th sem until 2028, so
        # 2026 is the session year; "2024 5th sem" would be the 2024 batch.
        if y + (u.semester - 1) // 2 > start + 1 or y == start and u.semester > 2:
            u.batch = y - (u.semester - 1) // 2
            u.notes.append(f"{y} is the session year, so the {ORDINAL[u.semester]} sem batch is {u.batch_label}")
        else:
            u.batch = y
            u.notes.append(f"{y} read as the admission year")

    if u.semester and not u.batch:
        u.batch = batch_in(u.semester, today)
        u.notes.append(f"no year given, so the batch in {ORDINAL[u.semester]} sem now")
    elif u.batch and not u.semester and u.topic in ("syllabus", "timetable", "exam", "result", None):
        current = semester_of(u.batch, today)
        if 1 <= current <= 8:
            u.semester = current
            u.notes.append(f"the {u.batch_label} batch is in {ORDINAL[current]} sem now")

    if u.semester:
        odd = u.semester % 2 == 1
        if u.batch:
            sess_start = u.batch + (u.semester - 1) // 2
        else:
            sess_start = start
        u.session = session_label(sess_start, odd)
    return u
