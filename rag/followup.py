"""Follow-up questions: "and who was before him?" -> a complete question.

The chat page sends the last few questions and answers of the current
session (kept only in the browser; nothing is stored on the server). Before
searching, a short AI call rewrites the new question so it makes sense on
its own; the search then uses that complete question.
"""
from __future__ import annotations

import logging

from rag import llm

log = logging.getLogger("rag.followup")

MAX_TURNS = 3
MAX_QUESTION_CHARS = 300
MAX_ANSWER_CHARS = 500

REWRITE_PROMPT = """You rewrite a student's latest chat message so it can be understood on its own, without the earlier conversation.
Replace words like "he", "him", "that", "it", "uske", "woh" and short follow-ups like "and 3rd sem?" with what they refer to in the conversation.
Keep the student's language and style (English, Hinglish or Hindi). Do not answer the question. Do not add facts that are not in the conversation.
If the latest message corrects or points to a source ("it's in the time table", "di hai time table mai"), combine it with the student's previous question into one question.
Always write a QUESTION the student is asking (ending with "?"), never a statement.
If the latest message is already a complete question on its own, return it unchanged.
Reply with the rewritten question only, on one line."""


def clean_history(history) -> list[dict]:
    """Keep only well-formed, short, recent turns: [{"q": ..., "a": ...}]."""
    turns = []
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        q = str(turn.get("q") or "").strip()[:MAX_QUESTION_CHARS]
        a = str(turn.get("a") or "").strip()[:MAX_ANSWER_CHARS]
        if q:
            turns.append({"q": q, "a": a})
    return turns[-MAX_TURNS:]


def standalone(question: str, history) -> str:
    """The question rewritten to stand on its own (unchanged if no history)."""
    turns = clean_history(history)
    if not turns:
        return question
    convo = "\n".join(f"Student: {t['q']}\nAssistant: {t['a']}" for t in turns)
    user = f"Conversation so far:\n{convo}\n\nLatest message: {question}\n\nRewritten question:"
    reply = llm.generate(REWRITE_PROMPT, user)
    text = (reply.text or "").strip().splitlines()[0].strip().strip('"').strip() if reply.generated and reply.text else ""
    if not text or len(text) > 2 * MAX_QUESTION_CHARS:
        # No AI available: at least search with the previous question's words too.
        log.info("Follow-up rewrite unavailable; searching with the previous question added")
        return f"{turns[-1]['q']} {question}"
    return text
