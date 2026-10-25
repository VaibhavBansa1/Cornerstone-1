"""Retrieval tests with keyword search only (no model downloads needed)."""
from datetime import date

import config
from rag.retrieval import BM25, HybridRetriever, expand_query, recency_bonus

TODAY = date(2026, 10, 25)


def _chunk(text, d, title="IT Department"):
    return {"source_url": f"https://x/{abs(hash(text))}", "source_type": "pdf", "title": title,
            "text": text, "date": d}


def test_expand_query_adds_synonyms():
    tokens = expand_query("Who is the HOD of IT?")
    assert "hod" in tokens and "head" in tokens and "dean" in tokens
    assert "who" not in tokens and "the" not in tokens


def test_bm25_prefers_documents_with_query_words():
    bm = BM25(["the syllabus for semester five", "hostel mess menu", "semester five timetable"])
    hits = bm.search(["semester", "timetable"], top_n=3)
    assert hits[0][0] == 2


def test_recency_bonus_fades_with_age(monkeypatch):
    monkeypatch.setattr(config, "RECENCY_WEIGHT", 0.2)
    monkeypatch.setattr(config, "RECENCY_HORIZON_YEARS", 5)
    assert recency_bonus("2026-07-01", TODAY) > recency_bonus("2023-07-01", TODAY) > 0
    assert recency_bonus("2015-01-01", TODAY) == 0
    assert recency_bonus(None, TODAY) == 0
    assert recency_bonus("2027-01-01", TODAY) == 0.2  # next session: full bonus


HEADS = [
    _chunk("Dr. Old Person, Professor & Head, Department of IT", "2019-01-01"),
    _chunk("Dr. Middle Person, HoD, Dept. of IT. Session: Jan-June 2026", "2026-01-05"),
    _chunk("Dr. New Person, Dean, School of IT. Session: July-Dec 2026", "2026-07-27"),
    _chunk("Hostel fees are payable in July", "2026-07-01"),
]


def test_all_role_holders_are_retrieved_and_unrelated_text_is_not():
    """Retrieval must bring every HOD/Head/Dean mention -- synonyms included
    -- so the LLM can see the whole timeline."""
    results = HybridRetriever(HEADS).search("who is the HOD of IT", top_k=3, today=TODAY)
    names = {r["text"].split(",")[0] for r in results}
    assert names == {"Dr. Old Person", "Dr. Middle Person", "Dr. New Person"}


def test_prompt_lists_newest_source_first_with_dates():
    from rag.pipeline import build_context, order_for_prompt

    results = HybridRetriever(HEADS).search("who is the HOD of IT", top_k=3, today=TODAY)
    context = build_context(order_for_prompt(results))
    assert context.index("Dr. New Person") < context.index("Dr. Middle Person") < context.index("Dr. Old Person")
    assert "dated: Jul 2026" in context and "dated: Jan 2019" in context


def test_recency_breaks_ties_between_equally_relevant_chunks(monkeypatch):
    monkeypatch.setattr(config, "RECENCY_WEIGHT", 0.15)
    chunks = [
        _chunk("Time table for semester five, room A", "2024-07-01"),
        _chunk("Time table for semester five, room B", "2026-07-01"),
    ]
    results = HybridRetriever(chunks).search("semester five time table", top_k=2, today=TODAY)
    assert results[0]["relevance"] == results[1]["relevance"]
    assert results[0]["date"] == "2026-07-01"


def test_duplicate_chunks_take_one_slot():
    same = "Department of Information Technology Time Table Coordinator Dean School of IT"
    chunks = [_chunk(same, "2026-07-27") for _ in range(5)] + [_chunk("Time Table lab rooms", "2026-07-27")]
    results = HybridRetriever(chunks).search("time table coordinator", top_k=5, today=TODAY)
    assert sum(r["text"] == same for r in results) == 1
