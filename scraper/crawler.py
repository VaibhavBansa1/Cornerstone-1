"""Scope-limited, single-visit crawler for the MITS IT department pages.

Design goals (per project requirements):
  * Only ever touch ``config.ALLOWED_DOMAIN``.
  * Only follow links for further crawling that are genuinely part of the
    department: either under ``config.ALLOWED_PATH_PREFIX``, or found
    inside the department's own sidebar nav (``config.DEPARTMENT_NAV_ARIA_LABEL``)
    -- this site publishes most department sub-pages as root-level slugs
    (e.g. "/syllabus-it") that share no URL prefix with the seed page.
  * Never treat site-wide chrome (top mega-menu, footer "quick links"
    widgets) as in-scope just because it's on the same domain -- link/asset
    discovery is restricted to ``config.MAIN_CONTENT_CONTAINER_ID``.
  * Visit every URL at most once -- no re-queueing, no re-fetching, so a
    cyclic link graph can never turn into an infinite loop.
  * Persist every page (raw HTML) and every asset (PDF/image/audio/video)
    referenced from within scope to disk, plus a manifest describing what
    was collected and where it came from.
  * Some department content (e.g. timetables) is published as individual
    public Google Drive files rather than hosted on the site itself. These
    are resolved to a direct-download URL and fetched as *assets* only --
    Drive is never crawled as if it were another page on the site.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.robotparser
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("crawler")


# Server answers that usually mean "try again in a moment".
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def _atomic_write_text(path: Path, text: str):
    """Write to a temp file, then swap it in. If the crawl is stopped
    halfway, the old copy of the file stays intact instead of being left
    half-written."""
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_save_stream(resp, path: Path):
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    os.replace(tmp, path)


def _freshness(resp) -> dict:
    """When the server says the file last changed, and when we fetched it.

    Used by ingestion to date documents that contain no dates themselves.
    """
    return {
        "last_modified": resp.headers.get("Last-Modified", ""),
        "fetched_at": time.time(),
    }


def normalize_url(url: str) -> str:
    """Strip fragments and trailing slashes so equivalent URLs dedupe."""
    url, _frag = urldefrag(url)
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return parsed._replace(path=path).geturl()


def is_allowed_domain(url: str) -> bool:
    return urlparse(url).netloc.lower() == config.ALLOWED_DOMAIN.lower()


def is_in_crawl_scope(url: str) -> bool:
    """Whether a page URL may be *crawled further* (followed for links)."""
    parsed = urlparse(url)
    if parsed.netloc.lower() != config.ALLOWED_DOMAIN.lower():
        return False
    return parsed.path.startswith(config.ALLOWED_PATH_PREFIX)


def asset_extension(url: str) -> str | None:
    path = urlparse(url).path.lower()
    for ext in config.ASSET_EXTENSIONS:
        if path.endswith(ext):
            return ext
    return None


def slugify_for_filename(url: str) -> str:
    parsed = urlparse(url)
    slug = (parsed.path.strip("/") or "index").replace("/", "__")
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}__{digest}"


_GOOGLE_DRIVE_HOSTS = {"drive.google.com"}
_GOOGLE_DRIVE_ID_PATTERNS = [
    re.compile(r"/file/d/([\w-]{10,})"),
    re.compile(r"[?&]id=([\w-]{10,})"),
]


def extract_google_drive_file_id(url: str) -> str | None:
    """Pull the file id out of a "view"/"open" Google Drive share link."""
    if urlparse(url).netloc.lower() not in _GOOGLE_DRIVE_HOSTS:
        return None
    for pattern in _GOOGLE_DRIVE_ID_PATTERNS:
        m = pattern.search(url)
        if m:
            return m.group(1)
    return None


def google_drive_direct_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def filename_from_content_disposition(header: str) -> str | None:
    if not header:
        return None
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", header)
    return m.group(1) if m else None


@dataclass
class Manifest:
    pages: list = field(default_factory=list)
    assets: list = field(default_factory=list)
    skipped_out_of_scope: list = field(default_factory=list)
    # Every link/asset that was in scope but couldn't actually be fetched:
    # dead links, timeouts, non-200 responses, unreadable Google Drive
    # files. Persisted here (not just logged) so a re-crawl gives you a
    # structured list to review instead of console output.
    failed: list = field(default_factory=list)
    # "complete" when the crawl reached the end, "interrupted" otherwise.
    status: str = "running"
    finished_at: float | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "status": self.status,
                "finished_at": self.finished_at,
                "pages": self.pages,
                "assets": self.assets,
                "skipped_out_of_scope": self.skipped_out_of_scope,
                "failed": self.failed,
            },
            indent=2,
            ensure_ascii=False,
        )


class ScopedCrawler:
    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": config.USER_AGENT})
        self.visited_pages: set[str] = set()
        self.downloaded_assets: set[str] = set()
        self.in_scope_urls: set[str] = set()
        self.manifest = Manifest()
        self._robots = None
        if config.RESPECT_ROBOTS_TXT:
            self._robots = urllib.robotparser.RobotFileParser()
            robots_url = f"https://{config.ALLOWED_DOMAIN}/robots.txt"
            try:
                resp = self.session.get(robots_url, timeout=config.REQUEST_TIMEOUT_SECONDS)
                if resp.status_code == 200:
                    self._robots.parse(resp.text.splitlines())
                else:
                    self._robots = None
            except requests.RequestException:
                self._robots = None

    def _get(self, url: str, stream: bool = False):
        """GET with one retry for temporary problems (timeouts, dropped
        connections, 429/5xx). Returns (response, None) or (None, reason)."""
        last_reason = ""
        for attempt in range(1 + config.FETCH_RETRIES):
            if attempt:
                log.info("Retrying %s (%s)", url, last_reason)
                time.sleep(config.RETRY_DELAY_SECONDS)
            try:
                resp = self.session.get(url, timeout=config.REQUEST_TIMEOUT_SECONDS, stream=stream)
            except requests.RequestException as exc:
                last_reason = str(exc)
                continue
            if resp.status_code in TRANSIENT_STATUS:
                last_reason = f"status {resp.status_code}"
                continue
            return resp, None
        return None, f"{last_reason} (after {1 + config.FETCH_RETRIES} attempts)"

    def _robots_allows(self, url: str) -> bool:
        if not self._robots:
            return True
        return self._robots.can_fetch(config.USER_AGENT, url)

    def crawl(self, start_url: str = config.START_URL) -> Manifest:
        config.RAW_PAGES_DIR.mkdir(parents=True, exist_ok=True)
        config.RAW_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

        start_norm = normalize_url(start_url)
        queue: deque[str] = deque([start_norm])
        queued: set[str] = {start_norm}
        self.in_scope_urls.add(start_norm)

        try:
            self._crawl_loop(queue, queued)
        except BaseException as exc:  # Ctrl+C, crash, closed terminal...
            # Keep the last COMPLETE manifest.json untouched and save what
            # this run got to a separate file, clearly marked.
            self.manifest.status = "interrupted"
            self.manifest.finished_at = time.time()
            partial = config.MANIFEST_PATH.with_name("manifest.partial.json")
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(partial, self.manifest.to_json())
            log.warning(
                "Crawl interrupted (%s). The previous complete manifest was kept; "
                "this run's partial results are in %s. Run the crawl again to finish.",
                exc.__class__.__name__, partial,
            )
            raise

        self.manifest.status = "complete"
        self.manifest.finished_at = time.time()
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(config.MANIFEST_PATH, self.manifest.to_json())
        config.MANIFEST_PATH.with_name("manifest.partial.json").unlink(missing_ok=True)
        log.info(
            "Crawl finished: %d pages, %d assets, %d skipped-out-of-scope, %d failed.",
            len(self.manifest.pages),
            len(self.manifest.assets),
            len(self.manifest.skipped_out_of_scope),
            len(self.manifest.failed),
        )
        return self.manifest

    def _crawl_loop(self, queue: deque, queued: set):
        while queue:
            if len(self.visited_pages) >= config.MAX_PAGES:
                log.warning("Reached MAX_PAGES=%s, stopping crawl.", config.MAX_PAGES)
                break

            url = queue.popleft()
            if url in self.visited_pages:
                continue
            # Mark visited *before* fetching: guarantees at-most-once visitation
            # even on redirect loops or fetch errors.
            self.visited_pages.add(url)

            # Scope was already decided when this URL was discovered (see
            # _extract_and_queue_links): either it matched ALLOWED_PATH_PREFIX,
            # or it was found inside the department's own sidebar nav. This
            # re-check just guards against anything slipping into the queue
            # some other way.
            if url not in self.in_scope_urls:
                self.manifest.skipped_out_of_scope.append(url)
                continue
            if not self._robots_allows(url):
                log.info("robots.txt disallows %s, skipping.", url)
                continue

            log.info("Fetching page: %s", url)
            resp, error = self._get(url)
            if resp is None:
                log.warning("Failed to fetch %s: %s", url, error)
                self.manifest.failed.append({"url": url, "kind": "page", "reason": error})
                continue
            time.sleep(config.REQUEST_DELAY_SECONDS)

            content_type = resp.headers.get("Content-Type", "")
            if resp.status_code != 200 or "text/html" not in content_type:
                self.manifest.failed.append(
                    {
                        "url": url,
                        "kind": "page",
                        "reason": f"status {resp.status_code}, content-type {content_type!r}",
                    }
                )
                continue

            html = resp.text
            local_name = slugify_for_filename(url) + ".html"
            local_path = config.RAW_PAGES_DIR / local_name
            _atomic_write_text(local_path, html)

            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else ""
            self.manifest.pages.append(
                {
                    "url": url,
                    "title": title,
                    "local_path": str(local_path.relative_to(config.DATA_DIR)),
                    **_freshness(resp),
                }
            )

            main_scope = soup.find(id=config.MAIN_CONTENT_CONTAINER_ID) or soup
            if main_scope is soup:
                log.warning(
                    "Container #%s not found on %s; falling back to the whole "
                    "page for link/asset discovery (site-wide chrome may leak in).",
                    config.MAIN_CONTENT_CONTAINER_ID,
                    url,
                )

            self._extract_and_queue_links(main_scope, url, queue, queued)
            self._extract_and_download_assets(main_scope, url)

    def _extract_and_queue_links(self, main_scope, page_url, queue, queued):
        dept_nav = main_scope.find("nav", attrs={"aria-label": config.DEPARTMENT_NAV_ARIA_LABEL})
        dept_nav_urls: set[str] = set()
        if dept_nav:
            for a in dept_nav.find_all("a", href=True):
                dept_nav_urls.add(normalize_url(urljoin(page_url, a["href"])))

        for a in main_scope.find_all("a", href=True):
            absolute = normalize_url(urljoin(page_url, a["href"]))
            if asset_extension(absolute):
                continue  # handled by _extract_and_download_assets
            if not is_allowed_domain(absolute):
                continue
            if not (is_in_crawl_scope(absolute) or absolute in dept_nav_urls):
                continue
            self.in_scope_urls.add(absolute)
            if absolute in self.visited_pages or absolute in queued:
                continue
            queued.add(absolute)
            queue.append(absolute)

    def _extract_and_download_assets(self, main_scope, page_url):
        candidates: list[str] = []
        for a in main_scope.find_all("a", href=True):
            candidates.append(a["href"])
        for tag, attr in (("img", "src"), ("source", "src"), ("video", "src"), ("audio", "src")):
            for el in main_scope.find_all(tag):
                if el.get(attr):
                    candidates.append(el[attr])
                if el.get("data-src"):
                    candidates.append(el["data-src"])

        for href in candidates:
            absolute = normalize_url(urljoin(page_url, href))

            drive_id = extract_google_drive_file_id(absolute)
            if drive_id:
                if drive_id in self.downloaded_assets:
                    continue
                self.downloaded_assets.add(drive_id)
                self._download_google_drive_asset(drive_id, page_url)
                continue

            ext = asset_extension(absolute)
            if not ext:
                continue
            if not is_allowed_domain(absolute):
                log.info("Skipping off-domain asset: %s", absolute)
                continue
            if absolute in self.downloaded_assets:
                continue
            self.downloaded_assets.add(absolute)
            self._download_asset(absolute, ext, page_url)

    def _download_google_drive_asset(self, file_id: str, referenced_from: str):
        drive_view_url = f"https://drive.google.com/file/d/{file_id}/view"
        download_url = google_drive_direct_download_url(file_id)
        resp, error = self._get(download_url, stream=True)
        if resp is None:
            log.warning("Failed to fetch Google Drive file %s: %s", file_id, error)
            self.manifest.failed.append(
                {"url": drive_view_url, "kind": "drive", "reason": error, "referenced_from": referenced_from}
            )
            return
        time.sleep(config.REQUEST_DELAY_SECONDS)
        if resp.status_code != 200:
            log.warning("Google Drive file %s returned status %s", file_id, resp.status_code)
            self.manifest.failed.append(
                {
                    "url": drive_view_url,
                    "kind": "drive",
                    "reason": f"status {resp.status_code}",
                    "referenced_from": referenced_from,
                }
            )
            return

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            # Large files (~25MB+) serve an HTML "can't scan for viruses"
            # interstitial requiring a confirm token instead of the file
            # itself. Not handled -- department timetables/notices are small.
            log.warning(
                "Google Drive file %s returned an HTML page instead of the "
                "file (likely the large-file virus-scan interstitial); skipping.",
                file_id,
            )
            self.manifest.failed.append(
                {
                    "url": drive_view_url,
                    "kind": "drive",
                    "reason": "virus-scan interstitial (file likely >25MB)",
                    "referenced_from": referenced_from,
                }
            )
            return

        filename = filename_from_content_disposition(resp.headers.get("Content-Disposition", ""))
        ext = Path(filename).suffix.lower() if filename else ""
        if ext not in config.ASSET_EXTENSIONS:
            log.info(
                "Skipping Google Drive file %s: unrecognized/missing extension %r (filename=%r)",
                file_id, ext, filename,
            )
            self.manifest.failed.append(
                {
                    "url": drive_view_url,
                    "kind": "drive",
                    "reason": f"unrecognized/missing extension {ext!r} (filename={filename!r})",
                    "referenced_from": referenced_from,
                }
            )
            return

        local_name = slugify_for_filename(download_url) + ext
        local_path = config.RAW_ASSETS_DIR / local_name
        try:
            _atomic_save_stream(resp, local_path)
        except requests.RequestException as exc:  # connection dropped mid-download
            self.manifest.failed.append(
                {"url": drive_view_url, "kind": "drive", "reason": f"download cut off: {exc}",
                 "referenced_from": referenced_from}
            )
            return

        log.info("Downloaded Google Drive asset: %s (%s) -> %s", file_id, filename, local_path.name)
        self.manifest.assets.append(
            {
                "url": f"https://drive.google.com/file/d/{file_id}/view",
                "extension": ext,
                "local_path": str(local_path.relative_to(config.DATA_DIR)),
                "referenced_from": referenced_from,
                "content_type": content_type,
                "size_bytes": local_path.stat().st_size,
                "original_filename": filename,
                **_freshness(resp),
            }
        )

    def _download_asset(self, url: str, ext: str, referenced_from: str):
        resp, error = self._get(url, stream=True)
        if resp is None:
            log.warning("Failed to fetch asset %s: %s", url, error)
            self.manifest.failed.append(
                {"url": url, "kind": "asset", "reason": error, "referenced_from": referenced_from}
            )
            return
        time.sleep(config.REQUEST_DELAY_SECONDS)
        if resp.status_code != 200:
            log.warning("Asset %s returned status %s", url, resp.status_code)
            self.manifest.failed.append(
                {
                    "url": url,
                    "kind": "asset",
                    "reason": f"status {resp.status_code}",
                    "referenced_from": referenced_from,
                }
            )
            return

        local_name = slugify_for_filename(url) + ext
        local_path = config.RAW_ASSETS_DIR / local_name
        try:
            _atomic_save_stream(resp, local_path)
        except requests.RequestException as exc:  # connection dropped mid-download
            self.manifest.failed.append(
                {"url": url, "kind": "asset", "reason": f"download cut off: {exc}",
                 "referenced_from": referenced_from}
            )
            return

        log.info("Downloaded asset: %s -> %s", url, local_path.name)
        self.manifest.assets.append(
            {
                "url": url,
                "extension": ext,
                "local_path": str(local_path.relative_to(config.DATA_DIR)),
                "referenced_from": referenced_from,
                "content_type": resp.headers.get("Content-Type", ""),
                "size_bytes": local_path.stat().st_size,
                **_freshness(resp),
            }
        )


def main():
    crawler = ScopedCrawler()
    crawler.crawl()


if __name__ == "__main__":
    main()
