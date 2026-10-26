"""Who holds (or held) a post: HOD, Dean, coordinator, ...

College documents rarely say "the HOD is X" in a sentence. They put a name
next to a title, often on separate lines and in lists of other names:

    (Dr. Punit K Johari)          HoD: Dr. Sanjiv Sharma
    Dean, School of IT

    role_words(question)     {"hod", "head", "dean"} for "Who is the HOD?" (synonyms included)
    holder_pattern(words)    regex: a person's name close to one of those words
    role_holders(text, ...)  [(name, title)], each title paired with its NEAREST name
    role_timeline(q, ...)    every named holder in the search results, newest first --
                             handed to the AI so it doesn't have to work out who came when
"""
from __future__ import annotations

import re

import config


def role_words(question: str) -> set[str]:
    """Role words a question is about, widened with synonyms (HOD -> head, dean)."""
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    roles = words & config.ROLE_TERMS
    widened = set(roles)
    for r in roles:
        for phrase in config.QUERY_SYNONYMS.get(r, []):
            widened.update(re.findall(r"[a-z0-9]+", phrase.lower()))
    return widened & config.ROLE_TERMS


_NAME = r"(?:Dr|Prof|Mr|Ms|Mrs|Shri|Smt)\.?[ \t]+[A-Z][\w.]*(?:[ \t]+[A-Z][\w.]*){0,3}"


def holder_pattern(words: set[str]) -> re.Pattern:
    """Matches a person's name close to a role word, e.g.
    "(Dr. Punit K Johari) Dean, School of IT" or "HoD: Dr. Sanjiv Sharma"."""
    roles = "|".join(sorted(re.escape(r) for r in words))
    # PDFs often put the name and the title on separate lines, so up to 40
    # characters of anything (line breaks included) may sit between them.
    return re.compile(
        rf"{_NAME}[\s\S]{{0,40}}?\b(?:{roles})\b|\b(?:{roles})\b[\s\S]{{0,40}}?{_NAME}",
        re.I,
    )


_NAME_RE = re.compile(_NAME)
_TITLE_WORDS = {"professor", "&", "and", "associate", "assistant", "time", "table", "time-table", "prof", "deputy", "vice"}


def role_holders(text: str, words: set[str]) -> list[tuple[str, str]]:
    """(name, title) pairs, pairing each title with the NEAREST name.

    College documents usually put the name just before the title
    ("(Dr. Punit K Johari)\nDean, School of IT"), sometimes after it
    ("HoD: Dr. Sanjiv Sharma"). Lists of names are common, so the closest
    name wins -- never just the first one in range.
    """
    roles = "|".join(sorted(re.escape(r) for r in words))
    pairs = []
    for m in re.finditer(rf"\b(?:{roles})\b", text, re.I):
        before = text[max(0, m.start() - 45): m.start()]
        names_before = list(_NAME_RE.finditer(before))
        name = None
        if names_before:
            last = names_before[-1]
            gap = before[last.end():]
            # A name on an earlier line that also has times/numbers is a
            # table row ("04:00-05:00 PM Prof. X"), not a signature.
            if "\n" in gap:
                name_line = before[: last.end()].rsplit("\n", 1)[-1]
                if not re.search(r"\d", name_line):
                    name = last.group(0)
            else:
                name = last.group(0)
        if not name:
            after = _NAME_RE.search(text, m.end(), m.end() + 45)
            name = after.group(0) if after else None
        if not name:
            continue
        name = re.sub(r"\s+", " ", name).strip()
        # drop title words the greedy name pattern may have swallowed
        name = re.split(rf"\s+(?:{roles}|Professor|Associate|Assistant|Time|Table|Coordinator)\b", name, flags=re.I)[0]
        # Title = title-ish words just before the role on the same line
        # ("Professor &", "Time Table") + the role + the rest of its line,
        # stopping before any other name.
        line_start = text.rfind("\n", 0, m.start()) + 1
        prefix_words = re.split(r"[\s,()]+", text[line_start: m.start()])
        prefix = []
        for w in reversed([w for w in prefix_words if w]):
            if w.lower().strip(".") in _TITLE_WORDS:
                prefix.insert(0, w)
            else:
                break
        line_end = text.find("\n", m.end())
        rest = text[m.end(): (m.end() + 35) if line_end == -1 else min(line_end, m.end() + 35)]
        other = _NAME_RE.search(rest)
        if other:
            rest = rest[: other.start()]
        rest = re.split(r"\s\d{1,2}[:.]\d{2}", rest)[0]  # drop "11:00-12:00" etc.
        title = re.sub(r"\s+", " ", " ".join(prefix + [m.group(0) + rest])).strip(" ,;:()-")
        pairs.append((name.strip(" ,;:()"), title))
    return pairs


def role_timeline(query: str, results: list[dict]) -> list[tuple[str, str, int]]:
    """For role questions: (date, "Dr. X ... Dean, School of IT", source n)
    for every named role holder in the results, newest first. Handed to the
    LLM so it doesn't have to work out who came when by itself."""
    words = role_words(query)
    if not words:
        return []
    seen, rows = set(), []
    for n, chunk in enumerate(results, 1):
        for name, title in role_holders(chunk["text"], words):
            snippet = f"{name}, {title}"
            key = (chunk.get("date"), snippet.lower())
            if key not in seen:
                seen.add(key)
                rows.append((chunk.get("date") or "", snippet, n))
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows[:12]
