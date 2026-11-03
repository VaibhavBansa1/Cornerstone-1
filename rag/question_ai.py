"""Reading the question with AI, for wordings the rules don't know.

The rules in rag/query_understanding.py know "5th sem", "time table",
"3rd year"... but students also write "paanchve sem ka routine", "V sem
class schedule", "final yr TT". When the rules find no topic, or no
semester for a semester-type question, one short call to a small online
model (config.QUESTION_AI_PROVIDERS, never a local GPU model) reads the
question into the same facts, as JSON. If no provider answers, the rules'
reading is used unchanged, so nothing breaks offline.
"""
from __future__ import annotations

import json
import logging
import re

import config
from rag import llm
from rag.query_understanding import TOPICS

log = logging.getLogger("rag.question_ai")

PROMPT = """You read a student's question to the MITS Gwalior help desk (School of Information Technology) and reply with ONE line of JSON and nothing else:
{"topic": ..., "semester": ..., "year_of_study": ..., "batch": ..., "programme": ..., "english": ...}
topic: one of TOPICS or null.
  timetable = class time table / schedule / routine / TT / class or college (lecture) hours; NOT bus, library, hostel, mess or office timings (those are null)
  exam = exam dates, exam schedule or exam time table
  syllabus = subjects, courses, scheme, syllabus
  result = results, marks, grades
  faculty = one person or post (HOD, dean, coordinator, a named teacher)
  faculty_list = all teachers / faculty members
  contacts, labs, programmes, overview = as named
semester: 1-8 only if the student names a semester (5th sem, sem V, paanchve sem, fifth semester), else null.
year_of_study: 1-4 only if the student says a year of study (3rd year, final year = 4, teesra saal), else null.
batch: the admission year only if given (2024 batch, 2024-28 -> 2024), else null.
programme: "mtech" if the student says M.Tech, else "btech".
english: the question as a short English search query (max 12 words).
Never guess numbers that are not in the question."""


def read(question: str) -> dict | None:
    if not config.QUESTION_AI_PROVIDERS:
        return None
    system = PROMPT.replace("TOPICS", ", ".join(TOPICS))
    reply = llm.generate(system, f"Question: {question[:300]}", only=config.QUESTION_AI_PROVIDERS)
    if not reply.generated:
        log.info("Question AI unavailable: %s", "; ".join(reply.errors) or "no provider configured")
        return None
    m = re.search(r"\{.*\}", reply.text or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        log.info("Question AI gave no JSON: %.120s", reply.text)
        return None
    data["question"] = question
    return data
