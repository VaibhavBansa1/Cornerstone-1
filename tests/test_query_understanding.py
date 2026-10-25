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
