"""Crawler tests using an in-memory fake site (no live network calls).

Verifies:
  * crawling stays within ALLOWED_DOMAIN + ALLOWED_PATH_PREFIX
  * a link cycle (A -> B -> A) does not cause infinite looping / re-fetching
  * each URL is fetched at most once
  * assets are downloaded even though they live outside the path prefix,
    as long as they are same-domain
  * off-domain links/assets are never fetched
"""
import config
from scraper.crawler import (
    ScopedCrawler,
    extract_google_drive_file_id,
    is_in_crawl_scope,
    normalize_url,
)


class FakeResponse:
    def __init__(self, status_code=200, text="", content=b"", headers=None):
        self.status_code = status_code
        self.text = text
        self.content = content
        self.headers = headers or {"Content-Type": "text/html"}

    def iter_content(self, chunk_size):
        yield self.content


class FakeSession:
    """Records every URL requested so tests can assert at-most-once fetching."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.headers = {}
        self.requested: list[str] = []

    def get(self, url, timeout=None, stream=False):
        self.requested.append(url)
        if url not in self.routes:
            return FakeResponse(status_code=404)
        return self.routes[url]

    def request(self, *a, **kw):  # pragma: no cover - unused
        raise NotImplementedError


DOMAIN = "web.mitsgwalior.in"
BASE = f"https://{DOMAIN}/academics/departments/information-technology"


def build_routes():
    page_a_html = f"""
    <html><head><title>IT Dept Home</title></head>
    <body>
      <a href="{BASE}/faculty">Faculty</a>
      <a href="{BASE}/">Self link (root, should normalize + not loop)</a>
      <a href="https://web.mitsgwalior.in/other-department">Other dept (out of scope)</a>
      <a href="https://external-site.example.com/page">External site (off domain)</a>
      <a href="https://{DOMAIN}/uploads/brochure.pdf">Brochure PDF</a>
      <img src="https://{DOMAIN}/uploads/logo.png">
    </body></html>
    """
    page_faculty_html = f"""
    <html><head><title>Faculty</title></head>
    <body>
      <a href="{BASE}">Back to home (creates a cycle)</a>
      <a href="https://{DOMAIN}/uploads/brochure.pdf">Same brochure again (dedup asset)</a>
      <img src="https://{DOMAIN}/uploads/logo.png">
      <video src="https://{DOMAIN}/uploads/intro.mp4"></video>
    </body></html>
    """
    other_dept_html = "<html><head><title>Other Dept</title></head><body>Should never be fetched.</body></html>"

    routes = {
        f"{BASE}": FakeResponse(text=page_a_html),
        f"{BASE}/faculty": FakeResponse(text=page_faculty_html),
        f"https://web.mitsgwalior.in/other-department": FakeResponse(text=other_dept_html),
        f"https://{DOMAIN}/uploads/brochure.pdf": FakeResponse(
            content=b"%PDF-fake", headers={"Content-Type": "application/pdf"}
        ),
        f"https://{DOMAIN}/uploads/logo.png": FakeResponse(
            content=b"\x89PNG-fake", headers={"Content-Type": "image/png"}
        ),
        f"https://{DOMAIN}/uploads/intro.mp4": FakeResponse(
            content=b"MP4-fake", headers={"Content-Type": "video/mp4"}
        ),
    }
    return routes


def test_scope_helpers():
    assert is_in_crawl_scope(f"{BASE}/faculty") is True
    assert is_in_crawl_scope("https://web.mitsgwalior.in/other-department") is False
    assert is_in_crawl_scope("https://external-site.example.com/academics/departments/information-technology") is False


def test_normalize_url_dedupes_trailing_slash_and_fragments():
    assert normalize_url(f"{BASE}/") == normalize_url(f"{BASE}")
    assert normalize_url(f"{BASE}/faculty#section") == normalize_url(f"{BASE}/faculty")


def test_crawl_stays_in_scope_and_visits_each_url_once(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_PAGES_DIR", tmp_path / "raw" / "pages")
    monkeypatch.setattr(config, "RAW_ASSETS_DIR", tmp_path / "raw" / "assets")
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "RESPECT_ROBOTS_TXT", False)
    monkeypatch.setattr(config, "REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(config, "MAX_PAGES", 50)

    session = FakeSession(build_routes())
    crawler = ScopedCrawler(session=session)
    manifest = crawler.crawl(start_url=BASE)

    page_urls = {p["url"] for p in manifest.pages}
    assert page_urls == {normalize_url(BASE), normalize_url(f"{BASE}/faculty")}

    assert "https://web.mitsgwalior.in/other-department" not in page_urls
    assert not any("external-site.example.com" in u for u in page_urls)

    asset_urls = {a["url"] for a in manifest.assets}
    assert asset_urls == {
        f"https://{DOMAIN}/uploads/brochure.pdf",
        f"https://{DOMAIN}/uploads/logo.png",
        f"https://{DOMAIN}/uploads/intro.mp4",
    }

    from collections import Counter

    counts = Counter(session.requested)
    for url, n in counts.items():
        assert n == 1, f"{url} was fetched {n} times, expected exactly once"

    assert len(crawler.visited_pages) == 2


def test_crawl_follows_department_nav_outside_path_prefix_and_ignores_global_chrome(tmp_path, monkeypatch):
    """Regression test for the real site's structure: most department pages
    (syllabus, faculty, ...) live at root-level URLs outside ALLOWED_PATH_PREFIX,
    linked only from the department's own sidebar <nav aria-label="...">.
    Meanwhile a site-wide mega-menu link (outside #sp-main-body) must NOT be
    treated as in-scope just because it's same-domain.
    """
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_PAGES_DIR", tmp_path / "raw" / "pages")
    monkeypatch.setattr(config, "RAW_ASSETS_DIR", tmp_path / "raw" / "assets")
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "RESPECT_ROBOTS_TXT", False)
    monkeypatch.setattr(config, "REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(config, "MAX_PAGES", 50)

    home_html = f"""
    <html><head><title>IT Dept Home</title></head>
    <body>
      <nav aria-label="Site mega menu">
        <a href="https://{DOMAIN}/images/mainmenu/hostel/SoP.pdf">Hostel SOP (global, out of scope)</a>
        <a href="https://{DOMAIN}/some-other-department-page">Some other department (global, out of scope)</a>
      </nav>
      <section id="sp-main-body">
        <aside>
          <nav aria-label="Department of Information Technology">
            <a href="/syllabus-it">Syllabus</a>
            <a href="/faculty-profiles-it-2">Faculty Profiles</a>
          </nav>
        </aside>
      </section>
    </body></html>
    """
    syllabus_html = """
    <html><head><title>Syllabus</title></head>
    <body>
      <section id="sp-main-body">
        <a href="/images/Departments/information_technology/syllabus.pdf">Syllabus PDF</a>
      </section>
    </body></html>
    """
    faculty_html = """
    <html><head><title>Faculty Profiles</title></head>
    <body><section id="sp-main-body"><p>Faculty list here.</p></section></body></html>
    """

    routes = {
        BASE: FakeResponse(text=home_html),
        f"https://{DOMAIN}/syllabus-it": FakeResponse(text=syllabus_html),
        f"https://{DOMAIN}/faculty-profiles-it-2": FakeResponse(text=faculty_html),
        f"https://{DOMAIN}/images/Departments/information_technology/syllabus.pdf": FakeResponse(
            content=b"%PDF-fake", headers={"Content-Type": "application/pdf"}
        ),
    }

    session = FakeSession(routes)
    crawler = ScopedCrawler(session=session)
    manifest = crawler.crawl(start_url=BASE)

    page_urls = {p["url"] for p in manifest.pages}
    assert page_urls == {
        normalize_url(BASE),
        normalize_url(f"https://{DOMAIN}/syllabus-it"),
        normalize_url(f"https://{DOMAIN}/faculty-profiles-it-2"),
    }

    # The global mega-menu link/asset (outside #sp-main-body) must never be
    # fetched, even though it's same-domain.
    assert f"https://{DOMAIN}/images/mainmenu/hostel/SoP.pdf" not in session.requested
    assert f"https://{DOMAIN}/some-other-department-page" not in session.requested

    # The syllabus PDF, found inside #sp-main-body on an in-scope page, must
    # be downloaded.
    asset_urls = {a["url"] for a in manifest.assets}
    assert asset_urls == {f"https://{DOMAIN}/images/Departments/information_technology/syllabus.pdf"}


def test_extract_google_drive_file_id():
    assert (
        extract_google_drive_file_id("https://drive.google.com/file/d/1KR51f6myf6RJIIFQIqGBB8cG_12LyLHK/view?usp=sharing")
        == "1KR51f6myf6RJIIFQIqGBB8cG_12LyLHK"
    )
    assert (
        extract_google_drive_file_id("https://drive.google.com/open?id=1KR51f6myf6RJIIFQIqGBB8cG_12LyLHK")
        == "1KR51f6myf6RJIIFQIqGBB8cG_12LyLHK"
    )
    assert extract_google_drive_file_id("https://docs.google.com/spreadsheets/d/abc/edit") is None
    assert extract_google_drive_file_id(f"https://{DOMAIN}/downloads-it") is None


def test_crawl_downloads_public_google_drive_file_referenced_by_page(tmp_path, monkeypatch):
    """Regression test: the real site publishes timetables as "View More
    Details" links to individually-shared Google Drive files rather than
    hosting the PDFs itself. These must be resolved to a direct-download URL
    and saved as assets -- without ever being queued as a crawlable page.
    """
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_PAGES_DIR", tmp_path / "raw" / "pages")
    monkeypatch.setattr(config, "RAW_ASSETS_DIR", tmp_path / "raw" / "assets")
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "RESPECT_ROBOTS_TXT", False)
    monkeypatch.setattr(config, "REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(config, "MAX_PAGES", 50)

    drive_id = "1KR51f6myf6RJIIFQIqGBB8cG_12LyLHK"
    home_html = f"""
    <html><head><title>IT Dept Home</title></head>
    <body>
      <section id="sp-main-body">
        <p>B.Tech I Semester:
          <a href="https://drive.google.com/file/d/{drive_id}/view?usp=sharing">View More Details</a>
        </p>
      </section>
    </body></html>
    """
    routes = {
        BASE: FakeResponse(text=home_html),
        f"https://drive.google.com/uc?export=download&id={drive_id}": FakeResponse(
            content=b"%PDF-faketimetable",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": 'attachment; filename="B.Tech I Sem TimeTable.pdf"',
            },
        ),
    }

    session = FakeSession(routes)
    crawler = ScopedCrawler(session=session)
    manifest = crawler.crawl(start_url=BASE)

    # Drive was never queued/fetched as a page.
    assert {p["url"] for p in manifest.pages} == {normalize_url(BASE)}

    assert len(manifest.assets) == 1
    asset = manifest.assets[0]
    assert asset["extension"] == ".pdf"
    assert asset["original_filename"] == "B.Tech I Sem TimeTable.pdf"
    saved = tmp_path / asset["local_path"]
    assert saved.read_bytes() == b"%PDF-faketimetable"


def test_failures_are_recorded_in_manifest_not_just_logged(tmp_path, monkeypatch):
    """Dead links, non-200 assets, and unreadable Google Drive files must
    show up in manifest["failed"] so they can be reviewed after the fact,
    not just printed to the console during the crawl.
    """
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_PAGES_DIR", tmp_path / "raw" / "pages")
    monkeypatch.setattr(config, "RAW_ASSETS_DIR", tmp_path / "raw" / "assets")
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "RESPECT_ROBOTS_TXT", False)
    monkeypatch.setattr(config, "REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(config, "MAX_PAGES", 50)

    drive_id = "1e9QO29kOgby-gqQi8ofmw9VIVlvTWeIa"
    home_html = f"""
    <html><head><title>IT Dept Home</title></head>
    <body>
      <section id="sp-main-body">
        <a href="https://{DOMAIN}/images/gallery/missing.jpeg">Missing image</a>
        <a href="https://drive.google.com/file/d/{drive_id}/view?usp=sharing">Big Drive file</a>
      </section>
    </body></html>
    """
    routes = {
        BASE: FakeResponse(text=home_html),
        f"https://{DOMAIN}/images/gallery/missing.jpeg": FakeResponse(status_code=404),
        f"https://drive.google.com/uc?export=download&id={drive_id}": FakeResponse(
            status_code=200,
            content=b"<html>Google Drive can't scan this file for viruses.</html>",
            headers={"Content-Type": "text/html; charset=utf-8"},
        ),
    }

    session = FakeSession(routes)
    crawler = ScopedCrawler(session=session)
    manifest = crawler.crawl(start_url=BASE)

    assert len(manifest.assets) == 0
    failed_by_kind = {f["kind"]: f for f in manifest.failed}
    assert failed_by_kind["asset"]["reason"] == "status 404"
    assert "interstitial" in failed_by_kind["drive"]["reason"]


def test_college_pages_are_fetched_with_their_files_but_their_links_are_not_followed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_PAGES_DIR", tmp_path / "raw" / "pages")
    monkeypatch.setattr(config, "RAW_ASSETS_DIR", tmp_path / "raw" / "assets")
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(config, "RESPECT_ROBOTS_TXT", False)
    rules = f"https://{DOMAIN}/academics/ordinances"
    sop = f"https://{DOMAIN}/images/sop/Exam.pdf"
    monkeypatch.setattr(config, "COLLEGE_PAGES", [rules, sop])
    routes = {
        BASE: FakeResponse(text="<html><body><div id='sp-main-body'>IT home</div></body></html>"),
        rules: FakeResponse(text=f"""<html><head><title>Ordinances</title></head><body><div id='sp-main-body'>
            <a href="https://{DOMAIN}/images/ordinance.pdf">Ordinance</a>
            <a href="https://{DOMAIN}/academics/some-other-page">Another page</a></div></body></html>"""),
        f"https://{DOMAIN}/images/ordinance.pdf": FakeResponse(content=b"%PDF-1", headers={"Content-Type": "application/pdf"}),
        sop: FakeResponse(content=b"%PDF-1", headers={"Content-Type": "application/pdf"}),
    }
    session = FakeSession(routes)
    manifest = ScopedCrawler(session=session).crawl(BASE)
    assert {p["url"] for p in manifest.pages} == {BASE, rules}
    assert {a["url"] for a in manifest.assets} == {f"https://{DOMAIN}/images/ordinance.pdf", sop}
    assert f"https://{DOMAIN}/academics/some-other-page" not in session.requested
