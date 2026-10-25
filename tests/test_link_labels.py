from ingestion.link_labels import labels_from_page
from scraper.crawler import normalize_url

PAGE = "https://web.mitsgwalior.in/syllabus-it"
HTML = """<html><body><div id="sp-main-body">
<table>
 <tr><th>SEMESTER</th><th>2026 -30 BATCH</th><th>2025 -29 BATCH</th><th>2024 -28 BATCH</th></tr>
 <tr><td>I SEM</td><td><a href="https://drive.google.com/file/d/AAAAAAAAAAAA/view?usp=sharing">CLICK HERE</a></td><td>***</td><td>***</td></tr>
 <tr><td>V SEM</td><td>************</td><td>************</td>
     <td><a href="https://drive.google.com/file/d/BBBBBBBBBBBB/view?usp=sharing">CLICK HERE</a></td></tr>
</table>
<p>Exam notice for July 2026 <a href="/images/notice.pdf">click here</a></p>
<a href="/images/tt.pdf">Time Table July-Dec 2026</a>
</div></body></html>"""


def test_click_here_links_are_named_from_table_row_and_column():
    labels = labels_from_page(HTML, PAGE, normalize_url)
    assert labels["https://drive.google.com/file/d/BBBBBBBBBBBB/view"] == "V SEM · 2024-28 BATCH"
    assert labels["https://drive.google.com/file/d/AAAAAAAAAAAA/view"] == "I SEM · 2026-30 BATCH"


def test_other_links_use_their_text_or_surrounding_text():
    labels = labels_from_page(HTML, PAGE, normalize_url)
    assert labels["https://web.mitsgwalior.in/images/tt.pdf"] == "Time Table July-Dec 2026"
    assert labels["https://web.mitsgwalior.in/images/notice.pdf"] == "Exam notice for July 2026"
