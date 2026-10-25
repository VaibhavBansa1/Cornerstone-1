from ingestion.course_list import course_list_text, extract_courses

NEW_FORMAT = """DEPARTMENT OF INFORMATION TECHNOLOGY
Mobile Computing & Wireless Network
16243101
Total Credits
THEORY OF COMPUTATION
16243102
Network Security Lab
LAB 16243107
LIST OF PROGRAMS"""

OLD_FORMAT = """DEPARTMENT OF INFORMATION TECHNOLOGY
PROBABILITY AND RANDOM PROCESS
3160511
Course Objective
DATA SCIENCE USING PYTHON 3160512"""

TIMETABLE = """MON 16243103 SS1 16243110 TS 16243104 NB
16243109 A1 PR LAB TJ+RKS
16251201 Data Structures 16251202 Computer Networks"""


def test_new_8_digit_codes():
    assert extract_courses(NEW_FORMAT) == {
        "16243101": ["Mobile Computing & Wireless Network"],
        "16243102": ["Theory of Computation"],
        "16243107": ["Network Security Lab"],
    }


def test_old_7_digit_codes_and_name_code_on_one_line():
    assert extract_courses(OLD_FORMAT) == {
        "3160511": ["Probability and Random Process"],
        "3160512": ["Data Science Using Python"],
    }


def test_timetable_course_table_but_not_the_grid():
    courses = extract_courses(TIMETABLE)
    assert courses == {"16251201": ["Data Structures"], "16251202": ["Computer Networks"]}


def test_course_list_text_needs_two_courses():
    assert course_list_text("ONE COURSE\n16243101", "x") is None
    listing = course_list_text(NEW_FORMAT, "V SEM · 2024-28 BATCH")
    assert listing.startswith("Course list (all subjects/courses) in V SEM · 2024-28 BATCH")
    assert "- 16243102: Theory of Computation" in listing
