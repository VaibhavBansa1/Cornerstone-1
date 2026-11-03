from rag.doc_labels import label_document, session_of
from rag.query_understanding import apply_ai, understand
from datetime import date


def _doc(title, text, link_label="View More Details", source_type="pdf"):
    return [{"title": title, "link_label": link_label, "source_type": source_type, "chunk_index": 0,
             "source_url": "u", "text": text},
            {"title": title, "link_label": link_label, "source_type": source_type, "chunk_index": -1,
             "kind": "course_list", "source_url": "u", "text": "Course list (all subjects/courses) in x: - 3rd Sem"}]


TT5 = ("Department of Information Technology Tentative (Prof. Vikas Sejwar) Time Table Coordinator "
       "Time-Table, B.Tech, IT Semester-V (Session: July-Dec 2026) Room No.: 115 Section A "
       "DAY 09:00-10:00 10:00 - 11:00 11:00 - 12:00 MON 16243103 SS1 TUE x WED y THU z FRI w")


def test_a_timetable_is_labelled_from_its_own_header_not_its_file_name():
    lab = label_document(_doc("Time Table - MITS Gwalior - View More Details - jul24 5.pdf", TT5))
    assert (lab.type, lab.programme, lab.semesters, lab.session) == ("timetable", "btech", {5}, "July-Dec 2026")


def test_mtech_and_exam_timetables_are_told_apart():
    mtech = label_document(_doc("x - M.Tech III TimeTable.pdf",
                                "Master Time Table for IT M.Tech Semester-III (Session: July-Dec. 2026) MON TUE"))
    assert mtech.programme == "mtech" and mtech.semesters == {3}
    exam = label_document(_doc("x - minor I time table.pdf", "MINOR I EXAM TIME TABLE B.Tech II Sem, IV Sem, VI Sem"))
    assert exam.type == "exam_timetable" and exam.semesters == set()   # many semesters: none in particular


def test_garbled_scans_and_web_pages_get_no_type():
    scan = label_document(_doc("Fee - images__fee.pdf", "qTq{ *frF{d ttrA0HAv tt{$TtTuTE 0r TEcttftolocy"))
    assert scan.type is None
    page = label_document(_doc("Time Table - MITS Gwalior", "Master Time Table B. TECH I Semester", source_type="page"))
    assert page.type is None


def test_sessions_are_read_in_many_spellings():
    assert session_of("JULY-DEC. 2026 Semester") == "July-Dec 2026"
    assert session_of("Session: Jan-June 2026") == "Jan-June 2026"
    assert session_of("Regular Time Table july dec2024") == "July-Dec 2024"


def test_question_and_document_labels_meet():
    today = date(2026, 11, 3)
    lab = label_document(_doc("Time Table - x - 5th sem TT.pdf", TT5))
    assert understand("What is the timetable for B.Tech IT 5th semester?", today=today).doc_matches(lab) == 1
    assert understand("7th sem timetable", today=today).doc_matches(lab) == -1
    assert understand("M.Tech 5th sem timetable", today=today).doc_matches(lab) == -1
    assert understand("who is the hod", today=today).doc_matches(lab) == 0
    # Same semester, other session (last year's 5th sem timetable): neutral, recency decides.
    assert understand("5th sem timetable 2025-29 batch", today=today).doc_matches(lab) == 0
    # A syllabus is per batch: the semester alone is no match.
    syl = label_document(_doc("Syllabus - IT 5th Sem Complete.pdf", "Scheme of Examination B.Tech V Semester"))
    assert syl.type == "syllabus" and understand("5th sem syllabus", today=today).doc_matches(syl) == 0


def test_exam_timetables_for_several_years_name_no_semester():
    lab = label_document(_doc("x - II MID SEM TIME TABLE II AND III YEAR.pdf",
                              "II MID SEM TIME TABLE B.Tech IV Sem II AND III YEAR"))
    assert lab.type == "exam_timetable" and lab.semesters == set()


def test_timetable_questions_get_the_labelled_timetables():
    from rag.retrieval import HybridRetriever

    today = date(2026, 11, 3)
    def tt(url, sem, session, d):
        return {"title": f"Time Table - MITS Gwalior - View More Details - {url}", "link_label": "View More Details",
                "source_type": "pdf", "chunk_index": 0, "source_url": url, "date": d,
                "text": f"Time-Table, B.Tech, IT Semester-{sem} (Session: {session}) DAY 09:00-10:00 MON TUE"}
    chunks = [tt("a.pdf", "V", "July-Dec 2026", "2026-07-01"), tt("b.pdf", "V", "July-Dec 2025", "2025-07-01"),
              tt("c.pdf", "VII", "July-Dec 2026", "2026-07-01"), tt("d.pdf", "IV", "Jan-June 2026", "2026-01-05"),
              {"title": "Notice", "source_type": "pdf", "chunk_index": 0, "source_url": "n.pdf", "date": "2026-10-01",
               "text": "Notice about the library timings 09:00 to 05:00"}]
    r = HybridRetriever(chunks)
    found = r.search("paanchve sem ka routine", today=today,
                     understanding=apply_ai(understand("paanchve sem ka routine", today=today),
                                            {"topic": "timetable", "semester": 5}, today))
    assert found[0]["source_url"] == "a.pdf" and found[0].get("whole_document")
    hours = r.search("college timings", today=today, understanding=understand("What are the college timings?", today=today))
    assert {c["source_url"] for c in hours} >= {"a.pdf", "c.pdf"}       # this session's class timetables
