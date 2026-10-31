"""SQLite usage log for the admin page's stats.

Privacy: the text of questions and answers is NOT stored -- only when a
question was asked, which sources were cited, which AI answered and how
long it took. (The table keeps its old question/answer columns, filled
with empty strings, so older databases keep working.)

Stands in for the "Chat Log" entity from the design: question, answer,
which sources were cited, and how long the answer took. Used by the admin
page for usage stats.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at REAL NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT NOT NULL,
    provider TEXT NOT NULL,
    response_ms INTEGER NOT NULL
)
"""


# "Was this helpful?" votes: the vote, the cited sources and the AI that
# answered -- never the question. ``comment`` is only filled when the student
# chooses to type one and press Send.
_FEEDBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    vote INTEGER NOT NULL,
    sources TEXT NOT NULL,
    provider TEXT NOT NULL,
    comment TEXT NOT NULL
)
"""


def _connect() -> sqlite3.Connection:
    config.CHAT_LOG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.CHAT_LOG_DB_PATH)
    conn.execute(_SCHEMA)
    conn.execute(_FEEDBACK_SCHEMA)
    return conn


def log_feedback(helpful: bool, sources: list[dict], provider: str, comment: str = ""):
    slim = [{"source_url": x.get("source_url"), "title": x.get("title")} for x in sources]
    with _connect() as conn:
        conn.execute(
            "INSERT INTO feedback (at, vote, sources, provider, comment) VALUES (?, ?, ?, ?, ?)",
            (time.time(), 1 if helpful else -1, json.dumps(slim), provider, comment),
        )


def log_question(sources: list[dict], provider: str, response_ms: int):
    """Record one answered question for the stats -- without its text."""
    slim = [{"source_url": x.get("source_url"), "title": x.get("title")} for x in sources]
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_log (asked_at, question, answer, sources, provider, response_ms) "
            "VALUES (?, '', '', ?, ?, ?)",
            (time.time(), json.dumps(slim), provider, response_ms),
        )


def stats(top_n: int = 5, label_for=None, link_for=None) -> dict:
    """Usage numbers for the admin page.

    Sources are counted by their link, and ``label_for(url)`` (supplied by
    the web app from the current index) gives the display name, so entries
    logged before a naming fix still show a proper name.
    """
    with _connect() as conn:
        rows = conn.execute("SELECT sources, response_ms FROM chat_log ORDER BY id DESC").fetchall()
    cited = Counter()
    fallback_name: dict[str, str] = {}
    for sources, _ms in rows:
        for s in json.loads(sources):
            url = s.get("source_url") or s.get("title") or "?"
            cited[url] += 1
            fallback_name.setdefault(url, s.get("title") or url)
    top = []
    for url, count in cited.most_common(top_n):
        name = (label_for(url) if label_for else None) or fallback_name[url]
        top.append({"name": name, "count": count, "link": link_for(url) if link_for else None})
    with _connect() as conn:
        votes = conn.execute("SELECT vote, sources, comment, at FROM feedback ORDER BY id DESC").fetchall()
    unhelpful = Counter()
    for vote, sources, _c, _at in votes:
        if vote < 0:
            for s in json.loads(sources):
                url = s.get("source_url") or s.get("title") or "?"
                unhelpful[url] += 1
                fallback_name.setdefault(url, s.get("title") or url)
    return {
        "total_questions": len(rows),
        "avg_response_ms": round(sum(r[1] for r in rows) / len(rows)) if rows else 0,
        "top_sources": top,
        "feedback": {
            "up": sum(v > 0 for v, *_ in votes),
            "down": sum(v < 0 for v, *_ in votes),
            "comments": [{"text": c, "at": at} for v, _s, c, at in votes if c][:5],
            "unhelpful_sources": [
                {"name": (label_for(u) if label_for else None) or fallback_name[u], "count": n,
                 "link": link_for(u) if link_for else None}
                for u, n in unhelpful.most_common(top_n)
            ],
        },
    }
