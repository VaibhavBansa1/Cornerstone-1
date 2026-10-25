"""SQLite log of every question asked through the web UI.

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


def _connect() -> sqlite3.Connection:
    config.CHAT_LOG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.CHAT_LOG_DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def log_question(question: str, answer: str, sources: list[dict], provider: str, response_ms: int):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_log (asked_at, question, answer, sources, provider, response_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), question, answer, json.dumps(sources), provider, response_ms),
        )


def stats(top_n: int = 5, label_for=None, link_for=None) -> dict:
    """Usage numbers for the admin page.

    Sources are counted by their link, and ``label_for(url)`` (supplied by
    the web app from the current index) gives the display name, so entries
    logged before a naming fix still show a proper name.
    """
    with _connect() as conn:
        rows = conn.execute("SELECT question, sources, response_ms, asked_at FROM chat_log ORDER BY id DESC").fetchall()
    cited = Counter()
    fallback_name: dict[str, str] = {}
    for _q, sources, _ms, _t in rows:
        for s in json.loads(sources):
            url = s.get("source_url") or s.get("title") or "?"
            cited[url] += 1
            fallback_name.setdefault(url, s.get("title") or url)
    top = []
    for url, count in cited.most_common(top_n):
        name = (label_for(url) if label_for else None) or fallback_name[url]
        top.append({"name": name, "count": count, "link": link_for(url) if link_for else None})
    return {
        "total_questions": len(rows),
        "avg_response_ms": round(sum(r[2] for r in rows) / len(rows)) if rows else 0,
        "top_sources": top,
        "recent_questions": [{"question": r[0], "asked_at": r[3]} for r in rows[:10]],
    }
