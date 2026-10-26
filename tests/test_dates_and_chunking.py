from datetime import date

from ingestion.dates import label, latest_date
from ingestion.chunking import chunk_text, make_chunks
from ingestion.readers import html_to_text

TODAY = date(2026, 10, 25)


def test_latest_date_handles_college_formats():
    cases = {
        "w.e.f. 27/07/2026 Dean, School of IT (Session: July-Dec 2026)": "2026-07-27",
        "Session: July-Dec 2026": "2026-07-01",  # start of session, not December
        "Session Nov-Feb 2027": "2026-11-01",
        "HOD from 7 th March 2018 to 1 st December 2020 (forenoon)": "2020-12-01",
        "W.e.f. 05/01/2026": "2026-01-05",  # day-first (Indian convention)
        "Mode of Teaching Blended 2021-22": "2021-07-01",
        "March 10, 2025": "2025-03-10",
        "founded in 1957, renovated 2019": "2019-01-01",  # bare years as a last resort
    }
    for text, expected in cases.items():
        assert latest_date(text, TODAY) == expected, text


def test_latest_date_ignores_numbers_that_are_not_dates():
    assert latest_date("Phone No: 9407204907", TODAY) is None
    assert latest_date("Course code 16241203", TODAY) is None
    assert latest_date("no dates at all", TODAY) is None
    assert latest_date("exam in 2099", TODAY) is None  # implausibly far future


def test_label():
    assert label("2026-07-27") == "Jul 2026"
    assert label(None) == "undated"


def test_html_inline_tags_do_not_break_words():
    html = "<div><p>Head of the Department from 7<sup>th</sup> March 2018</p><p>Next</p></div>"
    text = html_to_text(html)
    assert "7th March 2018" in text
    assert text.splitlines() == ["Head of the Department from 7th March 2018", "Next"]


def test_pdf_ligatures_are_normalised():
    from ingestion.readers import normalize_text

    assert normalize_text("Certiﬁcate  of   recognition") == "Certificate of recognition"


def test_chunks_break_on_paragraphs_and_overlap():
    paras = [f"Paragraph {i} " + "word " * 40 for i in range(12)]
    chunks = chunk_text("\n".join(paras), size=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    # every chunk starts at a word, never mid-word
    assert all(not c[0].isspace() for c in chunks)
    # neighbouring chunks share some text
    assert chunks[0].split()[-1] in chunks[1]


def test_make_chunks_stamps_chunk_and_document_dates():
    text = "Timetable (Session: July-Dec 2026)\n" + "filler line\n" * 200 + "Old note from 2019. Revised 10/03/2020."
    chunks = make_chunks(
        source_url="u", source_type="pdf", local_path="p", title="Time Table", text=text
    )
    assert chunks[0]["doc_date"] == "2026-07-01"
    assert chunks[0]["date"] == "2026-07-01"
    assert chunks[-1]["date"] == "2020-03-10"  # its own, older full date
    undated_middle = [c for c in chunks[1:-1] if "2019" not in c["text"] and "2026" not in c["text"]]
    assert all(c["date"] == "2026-07-01" for c in undated_middle)  # falls back to doc date


def test_semester_start_from_batch_label():
    from ingestion.dates import date_from_batch_label

    assert date_from_batch_label("V SEM · 2024-28 BATCH") == "2026-07-01"
    assert date_from_batch_label("IV SEM · 2024-28 BATCH") == "2026-01-01"
    assert date_from_batch_label("I SEM · 2026-30 BATCH") == "2026-07-01"
    assert date_from_batch_label("Time Table July 2026") is None


def test_book_year_does_not_beat_better_date_clues():
    text = "Recommended Books: Pearson Education, 2000.\nUnit-I Introduction"
    chunks = make_chunks(source_url="u", source_type="pdf", local_path="p", title="Syllabus - 5th.pdf",
                         text=text, fallback_date="2026-07-23", link_label="V SEM · 2024-28 BATCH")
    assert chunks[0]["doc_date"] == "2026-07-01"   # from the batch label, not "2000"
    plain = make_chunks(source_url="u", source_type="pdf", local_path="p", title="x.pdf",
                        text=text, fallback_date="2026-07-23")
    assert plain[0]["doc_date"] == "2026-07-23"    # the PDF's own date, not "2000"
