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
    from rag.roles import role_timeline

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
    from rag.roles import role_holders

    text = "Ms. Surbhi Gupta\nMs. Nidhi Saxena\n(Dr. Punit Kumar Johari)\nDean, SoIT"
    assert role_holders(text, {"dean"}) == [("Dr. Punit Kumar Johari", "Dean, SoIT")]
    assert role_holders("HoD: Dr. Sanjiv Sharma", {"hod"}) == [("Dr. Sanjiv Sharma", "HoD")]


def test_timetable_row_speaker_is_not_paired_with_a_signature_title():
    from rag.roles import role_holders

    # Real layout from an induction schedule PDF: last table row, then the
    # signature title whose name wasn't extracted.
    text = "IT Policies & act. 04:00–05:00 PM Prof. Surbhi Gupta\nDean, SoIT"
    assert role_holders(text, {"dean"}) == []
    text2 = "Chancellor/Dean/Head 11:00–12:00 Dr. Punit Kumar Johari\nBreak 12:00–13:00"
    assert role_holders(text2, {"dean"})[0] == ("Dr. Punit Kumar Johari", "Dean/Head")


def test_batch_and_semester_mapping():
    from rag.query_understanding import batch_in as batch_for_semester, semester_of as semester_for_batch

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


def test_topic_question_gets_the_whole_topic_page_as_one_source():
    from rag.query_understanding import understand

    page = [{"source_url": "https://x/labs", "source_type": "page", "title": "Laboratories - MITS Gwalior",
             "chunk_index": i, "date": None, "text": t}
            for i, t in enumerate(["Laboratories\nData Science Lab is for Python.\nProgramming Lab",
                                   "Programming Lab is for C.\nResearch Lab",
                                   "Research Lab is for projects."])]
    other = [_chunk("Lab timetable: DS LAB on Monday", "2026-07-01")]
    q = "what are all the labs?"
    results = HybridRetriever(page + other).search(q, top_k=3, today=TODAY, understanding=understand(q, TODAY))
    assert results[0]["title"] == "Laboratories - MITS Gwalior"
    text = results[0]["text"]
    assert all(name in text for name in ("Data Science Lab", "Programming Lab is", "Research Lab is"))
    assert text.count("Programming Lab\n") == 1          # overlap between pieces not repeated
    assert sum(r["source_url"] == "https://x/labs" for r in results) == 1


def test_year_of_study_is_searched_as_its_semester():
    from rag.query_understanding import understand

    u = understand("3rd year ka time table", date(2026, 10, 30))
    assert u.semester == 5 and u.rewrite("3rd year ka time table") == "5th sem ka time table"
    assert understand("5th sem syllabus", date(2026, 10, 30)).rewrite("5th sem syllabus") == "5th sem syllabus"


def test_odd_citation_styles_become_numbers():
    from rag.pipeline import clean_citations

    assert clean_citations("Head is X【4†L1-L3】 and Y [2†L9].") == "Head is X[4] and Y [2]."
    assert clean_citations("plain [1] stays") == "plain [1] stays"


def test_question_about_one_document_gets_the_whole_document(monkeypatch):
    monkeypatch.setattr(config, "WHOLE_DOC_TOP_MATCHES", 3)
    monkeypatch.setattr(config, "WHOLE_DOC_MAX_CHARS", 12000)
    rows = [f"G{n:02d} Student {n} Project allotment email spam classification" for n in range(1, 5)]
    doc = [{"source_url": "https://x/allot.pdf", "source_type": "pdf", "title": "Project Allotment List",
            "chunk_index": i, "date": "2026-07-01", "text": t} for i, t in enumerate(rows)]
    other = [_chunk("Hostel fees are payable in July", "2026-07-01")]
    results = HybridRetriever(doc + other).search("project allotment list email spam", top_k=3, today=TODAY)
    assert results[0].get("whole_document") and all(r in results[0]["text"] for r in rows)
    assert sum(r["source_url"] == "https://x/allot.pdf" for r in results) == 1

    monkeypatch.setattr(config, "WHOLE_DOC_MAX_CHARS", 50)  # too big -> normal pieces
    results = HybridRetriever(doc + other).search("project allotment list email spam", top_k=3, today=TODAY)
    assert not any(r.get("whole_document") for r in results)


def test_whole_documents_are_marked_in_the_prompt():
    from rag.pipeline import build_context

    ctx = build_context([{"source_url": "u", "source_type": "pdf", "title": "List", "date": None,
                          "text": "rows", "whole_document": True}])
    assert "whole document" in ctx.splitlines()[0]


def test_remedial_timetables_only_answer_questions_that_name_them():
    from rag.retrieval import _downrank

    remedial = {"title": "Time Table - Remedial Classes TT Jan june 2025.pdf"}
    assert _downrank(remedial, "college timings") > 0
    assert _downrank(remedial, "remedial class timings") == 0
    assert _downrank({"title": "Time Table - 3rd sem TT.pdf"}, "college timings") == 0


def test_this_sessions_timetables_win_when_no_semester_is_given():
    from datetime import date

    from rag.query_understanding import understand
    from rag.retrieval import _session_bonus

    today = date(2026, 11, 3)
    u = understand("What are the college timings?", today=today)
    assert _session_bonus({"date": "2026-07-27"}, u, today) > 0
    assert _session_bonus({"date": "2025-02-01"}, u, today) == 0
    assert _session_bonus({"date": "2026-07-27"}, understand("5th sem time table", today=today), today) == 0


def test_all_faculty_questions_get_a_name_and_designation_list():
    from rag.retrieval import faculty_roster

    page = [{"title": "Faculty Profiles - MITS Gwalior", "source_url": "u", "source_type": "page", "chunk_index": i,
             "date": "2025-10-24", "text": t} for i, t in enumerate([
        "DR. PUNIT KUMAR JOHARI\nDesignation: Associate Professor & Dean\nQualification: Ph.D.",
        "VIKAS SEJWAR\nDesignation: Assistant Professor\nE-Mail: x",
        "SEJWAR\nDesignation: Assistant Professor\nQualification: M. Tech",   # chunk starting mid-name
        "Dr. R.K. SINGH\nDesignation: Assistant Professor\nPublished in Vol 5: pp 1-2\nDesignation: none",
        "Prof. Aradhana Saxena\nDesignation: Assistant Professor"])]
    roster = faculty_roster(page)
    text = roster["text"]
    assert "Dr. Punit Kumar Johari: Associate Professor & Dean" in text
    assert "Vikas Sejwar" in text and "\n- Sejwar" not in text
    assert "Dr. R.K. Singh" in text and "Prof. Aradhana Saxena" in text and "(4 people" in text
    assert roster["topic_page"]


def test_faculty_list_questions_are_not_hod_questions():
    from rag.query_understanding import understand

    assert understand("What are the names of all the faculty members?").topic == "faculty_list"
    assert understand("Who is the HOD of IT?").topic == "faculty"
