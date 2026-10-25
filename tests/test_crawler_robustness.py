"""What happens when the network hiccups or the crawl is stopped halfway?"""
import json
from collections import Counter

import pytest
import requests

import config
from scraper.crawler import ScopedCrawler
from tests.test_crawler import BASE, DOMAIN, FakeResponse, FakeSession

PDF = f"https://{DOMAIN}/uploads/notice.pdf"
FACULTY = f"{BASE}/faculty"
HOME = f"""<html><head><title>Home</title></head><body>
  <a href="{FACULTY}">Faculty</a>
  <a href="{PDF}">Notice</a>
</body></html>"""
FACULTY_HTML = "<html><head><title>Faculty</title></head><body>Dr. X, HoD</body></html>"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_PAGES_DIR", tmp_path / "raw" / "pages")
    monkeypatch.setattr(config, "RAW_ASSETS_DIR", tmp_path / "raw" / "assets")
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "RESPECT_ROBOTS_TXT", False)
    monkeypatch.setattr(config, "REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(config, "RETRY_DELAY_SECONDS", 0)
    return tmp_path


class FlakySession(FakeSession):
    """Fails the first N requests to some URLs, then behaves."""

    def __init__(self, routes, failures: dict):
        super().__init__(routes)
        self.failures = dict(failures)

    def get(self, url, timeout=None, stream=False):
        if self.failures.get(url, 0) > 0:
            self.failures[url] -= 1
            self.requested.append(url)
            raise requests.ConnectTimeout("simulated timeout")
        return super().get(url, timeout, stream)


def routes():
    return {
        BASE: FakeResponse(text=HOME),
        FACULTY: FakeResponse(text=FACULTY_HTML),
        PDF: FakeResponse(content=b"%PDF-new%%EOF", headers={"Content-Type": "application/pdf"}),
    }


def test_one_timeout_is_retried_and_the_page_is_saved(data_dir):
    session = FlakySession(routes(), {FACULTY: 1, PDF: 1})
    manifest = ScopedCrawler(session=session).crawl(start_url=BASE)
    assert {p["url"] for p in manifest.pages} == {BASE, FACULTY}
    assert [a["url"] for a in manifest.assets] == [PDF]
    assert manifest.failed == []
    assert manifest.status == "complete"
    assert Counter(session.requested)[FACULTY] == 2  # failed once, retried once


def test_permanent_failure_gives_up_after_two_attempts(data_dir):
    session = FlakySession(routes(), {FACULTY: 99})
    manifest = ScopedCrawler(session=session).crawl(start_url=BASE)
    assert Counter(session.requested)[FACULTY] == 2  # no endless retry loop
    assert manifest.failed[0]["url"] == FACULTY
    assert "after 2 attempts" in manifest.failed[0]["reason"]


def test_server_error_503_is_retried(data_dir):
    class Once503(FakeSession):
        seen = False

        def get(self, url, timeout=None, stream=False):
            if url == FACULTY and not self.seen:
                self.seen = True
                self.requested.append(url)
                return FakeResponse(status_code=503)
            return super().get(url, timeout, stream)

    manifest = ScopedCrawler(session=Once503(routes())).crawl(start_url=BASE)
    assert FACULTY in {p["url"] for p in manifest.pages}


def test_stopping_mid_download_keeps_the_old_file_intact(data_dir):
    first = ScopedCrawler(session=FakeSession(routes())).crawl(start_url=BASE)
    saved = data_dir / first.assets[0]["local_path"]
    assert saved.read_bytes() == b"%PDF-new%%EOF"

    class StopsMidDownload(FakeResponse):
        def iter_content(self, chunk_size):
            yield b"%PDF-half"
            raise KeyboardInterrupt  # user pressed Ctrl+C here

    r = routes()
    r[PDF] = StopsMidDownload(headers={"Content-Type": "application/pdf"})
    with pytest.raises(KeyboardInterrupt):
        ScopedCrawler(session=FakeSession(r)).crawl(start_url=BASE)
    assert saved.read_bytes() == b"%PDF-new%%EOF"  # not half-written


def test_stopped_crawl_keeps_the_complete_manifest_and_saves_a_partial_one(data_dir):
    ScopedCrawler(session=FakeSession(routes())).crawl(start_url=BASE)
    complete = config.MANIFEST_PATH.read_text(encoding="utf-8")

    class Interrupting(FakeSession):
        def get(self, url, timeout=None, stream=False):
            if url == FACULTY:
                raise KeyboardInterrupt
            return super().get(url, timeout, stream)

    with pytest.raises(KeyboardInterrupt):
        ScopedCrawler(session=Interrupting(routes())).crawl(start_url=BASE)

    assert config.MANIFEST_PATH.read_text(encoding="utf-8") == complete
    partial = json.loads((data_dir / "manifest.partial.json").read_text(encoding="utf-8"))
    assert partial["status"] == "interrupted"
    assert [p["url"] for p in partial["pages"]] == [BASE]

    # The next run starts fresh: the interrupted page is fetched normally.
    manifest = ScopedCrawler(session=FakeSession(routes())).crawl(start_url=BASE)
    assert FACULTY in {p["url"] for p in manifest.pages}
    assert not (data_dir / "manifest.partial.json").exists()
