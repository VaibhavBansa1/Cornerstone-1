"""Question -> answer, gluing the three swappable parts together:

    retrieval  (rag/retrieval.py)  finds dated pieces of text
    prompt     (this file)         lays them out with dates + rules
    LLM        (rag/llm/)          writes the answer, with provider fallback

If no LLM is reachable, the retrieved passages are returned instead, so the
chatbot still works fully offline.

    python -m rag.pipeline "Who is the HOD of IT?"
"""
from __future__ import annotations

import logging
import threading
from datetime import date

import config
from ingestion.dates import label as date_label
from rag import llm
from rag.query_understanding import Understanding, semester_of, session_start_year, understand
from rag.retrieval import HybridRetriever, load_retriever
from rag.roles import role_timeline

log = logging.getLogger("rag.pipeline")

_retriever: HybridRetriever | None = None
_lock = threading.Lock()


def _get_retriever() -> HybridRetriever:
    global _retriever
    with _lock:
        if _retriever is None:
            _retriever = load_retriever()
        return _retriever


def indexed_chunks() -> list[dict]:
    """Every chunk in the loaded index (the admin page lists documents from these)."""
    return _get_retriever().chunks


def reload_index():
    """Drop the cached index so the next query reads the rebuilt one."""
    global _retriever
    with _lock:
        _retriever = None


def index_exists() -> bool:
    return config.FAISS_INDEX_PATH.exists() and config.INDEX_METADATA_PATH.exists()


def retrieve(query: str, top_k: int = config.TOP_K, understanding: Understanding | None = None) -> list[dict]:
    return _get_retriever().search(query, top_k=top_k, understanding=understanding)


SYSTEM_PROMPT = """You are the help desk assistant for the Information Technology department (School of IT) at MITS Gwalior.
Today's date is {today}.

Answer ONLY from the numbered sources below. Each source shows a date: the newest date mentioned in that text, or "undated".

Rules:
1. Facts change over time: who holds a post (HOD / Head / Dean / coordinator), timetables, syllabus, notices. When sources disagree, trust the most recent dated source and say "as of <month year>". Mention people from older sources only as former holders ("previously ..."), never as current. If a source says someone held a post "from X to Y", that person no longer holds it after Y.
2. The same post can have different titles in different years (e.g. "HoD", "Professor & Head", "Dean, School of IT"). Use the exact title the newest source uses, and mention if the title changed.
3. If the sources do not contain the answer, say so plainly and suggest contacting the department office. Never guess names, dates or numbers.
4. Cite sources with their [n] markers.
5. If a "Timeline" is given at the top, use it to decide who is current: the top line is the newest. Lines with different titles can be different posts; say so instead of mixing them up.
6. Batches: "2024-28 batch" = students admitted in 2024. {batch_guide} A question like "5th sem syllabus" without a year means the batch that is in 5th sem now; "2025 students" means the 2025-29 batch. If "Question understood as" is given at the top, answer for that batch and semester; use other batches' sources only if nothing else exists, and then say which batch they are for.
7. Each batch has its own syllabus and course codes. Course codes changed when MITS became a Deemed University (older 7-digit codes like 3160512, newer 8-digit codes like 16243101), so the same subject can have different codes in different batches. Quote course codes exactly as written in the source for the batch asked about; never copy a code from another batch's document. If a "Course list" source is given, use it for "which subjects" questions and list every course in it.
8. Reply in the same language and script the student used: English question -> English answer; Hinglish (Hindi in English letters, e.g. "kaun hai") -> Hinglish answer; Hindi -> Hindi. Keep it short and clear.
9. You only have the IT department's own website (the department is now called the School of Information Technology, SoIT; "department", "dept" and "school" mean the same). For anything else (hostel, mess, canteen, fees, transport, directions, admissions and cutoffs, scholarships, other departments, general knowledge) say you don't have that information and suggest the main MITS website (mitsgwalior.in) or the department office. A passing mention is not an answer: a student magazine article, a project title or a line in someone's profile does not tell you what the college has today.
10. Never list individual students' names, roll numbers, marks or attendance from lists (debarred lists, marks, attendance sheets). Say which notice contains it and that the student should check their own roll number there.
11. "Lab" can mean a physical laboratory of the department or a lab course (practical subject) in a semester. If the question doesn't make clear which, list the department's laboratories and add one line saying that each semester's lab courses are in its syllabus.
12. If the question is very broad ("tell me everything"), give a short overview from the sources and suggest two or three specific questions the student could ask."""


def batch_guide(today: date | None = None) -> str:
    """'In the July-Dec 2026 session: 2026-30 batch = 1st sem, ...'"""
    today = today or date.today()
    start = session_start_year(today)
    session = f"July-Dec {start}" if today.month >= 7 else f"Jan-June {start + 1}"
    parts = []
    for admitted in range(start, start - 4, -1):
        sem = semester_of(admitted, today)
        if 1 <= sem <= 8:
            parts.append(f"{admitted}-{(admitted + 4) % 100:02d} batch = semester {sem}")
    return f"In the current {session} session: " + ", ".join(parts) + "."


def build_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or r["source_url"]
        parts.append(
            f"[{i}] {title} (type: {r.get('source_type')}, dated: {date_label(r.get('date'))})\n{r['text']}"
        )
    return "\n\n".join(parts)


def order_for_prompt(results: list[dict]) -> list[dict]:
    """Newest first, so the model sees current information before old."""
    return sorted(results, key=lambda r: r.get("date") or "", reverse=True)


def build_timeline(query: str, results: list[dict]) -> str:
    rows = role_timeline(query, results)
    if not rows:
        return ""
    lines = [f"- {date_label(d)}: {snippet} [{n}]" for d, snippet, n in rows]
    return "Timeline of people named with this role in the sources (newest first):\n" + "\n".join(lines)


def build_prompt(query: str, results: list[dict], today: date | None = None,
                 understanding: Understanding | None = None) -> tuple[str, str]:
    """(system prompt, user message) exactly as the LLM will see them."""
    context = build_context(results)
    timeline = build_timeline(query, results)
    system = SYSTEM_PROMPT.format(
        today=(today or date.today()).strftime("%d %B %Y"), batch_guide=batch_guide(today)
    )
    head = []
    if understanding is not None and not understanding.empty:
        why = f" ({'; '.join(understanding.notes)})" if understanding.notes else ""
        head.append(f"Question understood as: {understanding.describe()}{why}")
    if timeline:  # first: small models weigh what they read first
        head.append(timeline)
    head_text = "\n\n".join(head) + "\n\n" if head else ""
    return system, f"{head_text}Sources:\n{context}\n\nQuestion: {query}"


def generate_answer(query: str, results: list[dict], today: date | None = None,
                    understanding: Understanding | None = None, on_attempt=None) -> llm.Reply:
    system, user_message = build_prompt(query, results, today, understanding)
    context = build_context(results)
    reply = llm.generate(system, user_message, on_attempt=on_attempt)
    if not reply.generated:
        why = "; ".join(reply.errors) or "no LLM provider is configured or reachable"
        reply.text = f"[No generated answer ({why}) -- showing the most relevant passages]\n\n{context}"
    return reply


def answer(query: str, top_k: int = config.TOP_K, on_step=None) -> dict:
    """on_step(kind, data), if given, is called as each step starts, so the
    chat page can show progress: "understood", "searching", "found" (the
    results), "writing" (provider name; called again if it falls back)."""
    def step(kind, data=None):
        if on_step:
            on_step(kind, data)

    # Step 0: what is this about? (semester, batch, session, topic)
    understanding = understand(query)
    step("understood", understanding.describe())
    step("searching", len(_get_retriever().chunks))
    results = order_for_prompt(retrieve(query, top_k=top_k, understanding=understanding))
    step("found", results)
    reply = generate_answer(query, results, understanding=understanding,
                            on_attempt=lambda name, error: step("writing", {"provider": name, "after_error": error}))
    return {
        "query": query,
        "answer": reply.text,
        "sources": results,
        "provider": reply.provider,
        "generated": reply.generated,
        "provider_errors": reply.errors,
        "understood_as": understanding.describe(),
        "understanding_notes": understanding.notes,
    }


def main():
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = sys.argv[1:]
    show_prompt = "--show-prompt" in args
    args = [a for a in args if a != "--show-prompt"]
    if not args:
        print('Usage: python -m rag.pipeline [--show-prompt] "your question"')
        raise SystemExit(1)
    question = " ".join(args)
    if show_prompt:  # print exactly what the LLM would receive, don't call it
        u = understand(question)
        system, user_message = build_prompt(question, order_for_prompt(retrieve(question, understanding=u)),
                                            understanding=u)
        print("=== SYSTEM ===\n" + system + "\n\n=== USER ===\n" + user_message)
        return
    result = answer(question)
    print("\n=== ANSWER ===  (via", result["provider"] + ")\n")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for i, r in enumerate(result["sources"], 1):
        print(
            f"[{i}] {date_label(r.get('date')):>9}  rel={r.get('relevance', 0):.3f} "
            f"+new={r.get('recency_bonus', 0):.3f}  ({r['source_type']}) {r['source_url']}"
        )
    if result["provider_errors"]:
        print("\nProvider notes:", *result["provider_errors"], sep="\n  - ")


if __name__ == "__main__":
    main()
