from datetime import date

import pytest

from rag.query_understanding import understand

OCT_2026 = date(2026, 10, 26)


@pytest.mark.parametrize("question, semester, batch, session", [
    ("fifth semester student syllabus 2026", 5, 2024, "July-Dec 2026"),   # 2026 = session year
    ("2026 5th semester syllabus", 5, 2024, "July-Dec 2026"),
    ("it V sem syllabus", 5, 2024, "July-Dec 2026"),                     # no year: batch in 5th sem now
    ("semester-V subjects", 5, 2024, "July-Dec 2026"),
    ("panchve sem ka syllabus", 5, 2024, "July-Dec 2026"),               # Hinglish
    ("3rd year ka time table", 5, 2024, "July-Dec 2026"),                # 3rd year = sem 5 in Jul-Dec
    ("3rd sem for 2025 students syllabus", 3, 2025, "July-Dec 2026"),
    ("2025 batch timetable", 3, 2025, "July-Dec 2026"),                  # batch only: its current sem
    ("syllabus of 2024-28 batch", 5, 2024, "July-Dec 2026"),
    ("sem 4 syllabus 2024 batch", 4, 2024, "Jan-June 2026"),
])
def test_semester_batch_and_session_are_worked_out(question, semester, batch, session):
    u = understand(question, OCT_2026)
    assert (u.semester, u.batch, u.session) == (semester, batch, session)


def test_questions_without_semester_or_batch_are_left_alone():
    assert understand("who is the HOD of IT?", OCT_2026).empty
    assert understand("what is the vision of IT department", OCT_2026).describe() == ""


def test_description_and_document_label_matching():
    u = understand("2026 5th semester syllabus", OCT_2026)
    assert u.describe() == "5th sem · 2024-28 batch · July-Dec 2026 session · syllabus"
    assert u.label_matches("V SEM · 2024-28 BATCH") == 1
    assert u.label_matches("V SEM · 2023-27 BATCH") == -1   # last year's batch
    assert u.label_matches("IV SEM · 2024-28 BATCH") == -1  # same batch, other semester
    assert u.label_matches("Time Table July 2026") == 0     # not filed by batch: neutral


def test_in_the_even_semester_session_years_shift():
    u = understand("3rd year syllabus", date(2027, 2, 1))
    assert (u.semester, u.batch, u.session) == (6, 2024, "Jan-June 2027")


def test_college_hours_questions_are_timetable_questions_for_the_current_session():
    from datetime import date

    from rag.query_understanding import understand

    from rag.query_understanding import apply_ai, needs_ai

    today = date(2026, 11, 3)
    u = understand("What are the class timings?", today=today)
    assert u.topic == "timetable" and u.session == "July-Dec 2026" and not u.semester
    # "baje" and "library timings" are left to the AI reading
    u = understand("College kitne baje se kitne baje tak hai?", today=today)
    assert u.topic is None and needs_ai(u)
    u = apply_ai(u, {"topic": "timetable"}, today)
    assert u.topic == "timetable" and u.session == "July-Dec 2026" and not u.semester
    assert understand("What are the library timings?", today=today).topic is None


def test_semester_is_read_from_timetable_file_names():
    from datetime import date

    from rag.query_understanding import understand

    u = understand("What is the timetable for B.Tech IT 5th semester?", today=date(2026, 11, 3))
    assert u.title_matches("Time Table - MITS Gwalior - View More Details - 5th sem TT.pdf") == 1
    assert u.title_matches("Time Table - MITS Gwalior - View More Details - 2nd semMaster TimeTable Jan 2026.pdf") == -1
    assert u.title_matches("Time Table - View More Details - 6th Revised Master TimeTable Jan 2026.pdf") == -1
    assert u.title_matches("Time Table - Remedial Classes TT Jan june 2025 (1).pdf") == 0
    assert u.title_matches("Time Table - II MID SEM TIME TABLE II AND III YEAR (1).pdf") == 0
    assert understand("college timings").title_matches("x - 2nd sem TT.pdf") == 0


def test_ai_reading_fills_what_the_rules_miss():
    from rag.query_understanding import apply_ai, needs_ai

    today = date(2026, 11, 3)
    u = understand("paanchve sem ka routine dikhao", today=today)
    assert u.semester == 5 and u.topic is None and needs_ai(u)
    u = apply_ai(u, {"topic": "timetable", "semester": 5, "english": "5th semester timetable"}, today)
    assert u.topic == "timetable" and u.batch == 2024 and u.session == "July-Dec 2026"
    assert "timetable" in u.search_terms()

    u = understand("final yr ka TT bhejo", today=today)
    u = apply_ai(u, {"topic": "timetable", "year_of_study": 4}, today)
    assert u.semester == 7 and u.session == "July-Dec 2026"


def test_rules_win_and_bad_ai_replies_are_ignored():
    from rag.query_understanding import apply_ai, needs_ai

    today = date(2026, 11, 3)
    u = understand("5th sem time table", today=today)
    assert not needs_ai(u)
    u = apply_ai(u, {"topic": "exam", "semester": 3}, today)       # rules already knew both
    assert (u.topic, u.semester) == ("timetable", 5)
    u = apply_ai(understand("hello", today=today), {"topic": "weather", "semester": "nine"}, today)
    assert u.topic is None and u.semester is None
    assert apply_ai(understand("hello", today=today), None, today).topic is None
