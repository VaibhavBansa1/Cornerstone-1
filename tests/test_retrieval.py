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
        _chunk("Time table for semester five, room one", "2024-07-01"),
        _chunk("Time table for semester five, room two", "2026-07-01"),
    ]
    results = HybridRetriever(chunks).search("semester five time table", top_k=2, today=TODAY)
    assert results[0]["relevance"] == results[1]["relevance"]
    assert results[0]["date"] == "2026-07-01"


def test_duplicate_chunks_take_one_slot():
    same = "Department of Information Technology Time Table Coordinator Dean School of IT"
    chunks = [_chunk(same, "2026-07-27") for _ in range(5)] + [_chunk("Time Table lab rooms", "2026-07-27")]
    results = HybridRetriever(chunks).search("time table coordinator", top_k=5, today=TODAY)
    assert sum(r["text"] == same for r in results) == 1


def test_broken_reranker_does_not_break_search():
    class BrokenReranker:
        def score(self, query, texts):
            raise OSError("model files missing")

    retriever = HybridRetriever(HEADS, reranker=BrokenReranker())
    results = retriever.search("who is the HOD of IT", top_k=3, today=TODAY)
    assert len(results) == 3
    assert retriever.reranker is None


def test_semester_numbers_match_roman_numerals():
    from rag.retrieval import expand_query

    assert {"v", "5", "fifth"} <= set(expand_query("timetable for 5th semester"))
    assert "v" not in expand_query("what is the vision")  # no semester word: untouched
    chunks = [
        _chunk("Time-Table, B.Tech, IT Semester-V (Session: July-Dec 2026)", "2026-07-01"),
        _chunk("Time-Table, B.Tech, IT Semester-III (Session: July-Dec 2026)", "2026-07-01"),
    ]
    results = HybridRetriever(chunks).search("B.Tech IT 5th sem timetable", top_k=1, today=TODAY)
    assert "Semester-V " in results[0]["text"]


def test_role_timeline_lists_named_holders_newest_first():
    from rag.retrieval import role_timeline

    results = [
        {"text": "Editorial Board\nDr. Sanjiv Sharma, HoD IT", "date": "2024-07-01"},
        {"text": "w.e.f. 27/07/2026\n(Prof. Vikas Sejwar)\nTime Table Coordinator\n(Dr. Punit K Johari)\nDean, School of IT",
         "date": "2026-07-27"},
    ]
    rows = role_timeline("Who is the HOD?", results)
    assert rows[0] == ("2026-07-27", "Dr. Punit K Johari, Dean, School of IT", 2)
    assert rows[1][1].startswith("Dr. Sanjiv Sharma")
    coord = role_timeline("time table coordinator kaun hai", results)
    assert coord[0][1] == "Prof. Vikas Sejwar, Time Table Coordinator"
    assert role_timeline("what is the vision", results) == []


def test_role_holder_pairs_the_nearest_name_not_the_first_in_a_list():
    from rag.retrieval import role_holders

    text = "Ms. Surbhi Gupta\nMs. Nidhi Saxena\n(Dr. Punit Kumar Johari)\nDean, SoIT"
    assert role_holders(text, {"dean"}) == [("Dr. Punit Kumar Johari", "Dean, SoIT")]
    assert role_holders("HoD: Dr. Sanjiv Sharma", {"hod"}) == [("Dr. Sanjiv Sharma", "HoD")]


def test_timetable_row_speaker_is_not_paired_with_a_signature_title():
    from rag.retrieval import role_holders

    # Real layout from an induction schedule PDF: last table row, then the
    # signature title whose name wasn't extracted.
    text = "IT Policies & act. 04:00–05:00 PM Prof. Surbhi Gupta\nDean, SoIT"
    assert role_holders(text, {"dean"}) == []
    text2 = "Chancellor/Dean/Head 11:00–12:00 Dr. Punit Kumar Johari\nBreak 12:00–13:00"
    assert role_holders(text2, {"dean"})[0] == ("Dr. Punit Kumar Johari", "Dean/Head")


def test_batch_and_semester_mapping():
    from rag.retrieval import batch_for_semester, semester_for_batch

    oct26, mar27 = date(2026, 10, 26), date(2027, 3, 1)
    assert batch_for_semester(5, oct26) == 2024
    assert batch_for_semester(3, oct26) == 2025
    assert semester_for_batch(2024, oct26) == 5
    assert semester_for_batch(2024, mar27) == 6
    assert semester_for_batch(2026, oct26) == 1


def test_batch_guide_in_prompt():
    from rag.pipeline import batch_guide

    g = batch_guide(date(2026, 10, 26))
    assert "July-Dec 2026" in g and "2024-28 batch = semester 5" in g and "2025-29 batch = semester 3" in g


def test_document_title_beats_long_text_that_repeats_the_words():
    minutes = _chunk("BoS meeting. " + "The syllabus of each sem was discussed. " * 12, "2026-07-01",
                     title="BoS Proceedings 2022.pdf")
    syllabus = _chunk("Unit-I Mobile computing. Unit-II GPRS.", "2026-07-01",
                      title="Syllabus - V SEM · 2024-28 BATCH - 5th 11_merged.pdf")
    results = HybridRetriever([minutes, syllabus]).search("it V sem syllabus", top_k=2, today=TODAY)
    assert results[0]["title"].startswith("Syllabus - V SEM")


def test_understood_batch_wins_over_other_batches_and_its_course_list_is_included():
    from rag.query_understanding import understand

    def doc(label, text, kind=None):
        c = _chunk(text, "2026-07-01", title=f"Syllabus - {label} - file.pdf")
        c["link_label"] = label
        if kind:
            c["kind"] = kind
        return c

    chunks = [
        doc("V SEM · 2023-27 BATCH", "Syllabus. Compiler Design. Cloud Computing. Syllabus syllabus."),
        doc("V SEM · 2024-28 BATCH", "Unit-I GSM. Unit-II GPRS architecture."),
        doc("V SEM · 2024-28 BATCH", "Course list (all subjects/courses): 16243101 Mobile Computing", kind="course_list"),
        doc("IV SEM · 2024-28 BATCH", "Syllabus of IoT and Cloud."),
    ]
    q = "2026 5th semester syllabus"
    results = HybridRetriever(chunks).search(q, top_k=3, today=TODAY, understanding=understand(q, TODAY))
    labels = [r["link_label"] for r in results]
    assert labels[0] == "V SEM · 2024-28 BATCH"
    assert any(r.get("kind") == "course_list" for r in results)
    assert results[0]["batch_bonus"] > 0
    other = [r for r in results if r["link_label"] == "V SEM · 2023-27 BATCH"]
    assert all(r["batch_bonus"] < 0 for r in other)
