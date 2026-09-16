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
from scraper.crawler import ScopedCrawler, is_in_crawl_scope, normalize_url


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
