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

    def to_json(self) -> str:
        return json.dumps(
            {
                "pages": self.pages,
                "assets": self.assets,
                "skipped_out_of_scope": self.skipped_out_of_scope,
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
            try:
                resp = self.session.get(url, timeout=config.REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException as exc:
                log.warning("Failed to fetch %s: %s", url, exc)
                continue
            time.sleep(config.REQUEST_DELAY_SECONDS)

            content_type = resp.headers.get("Content-Type", "")
            if resp.status_code != 200 or "text/html" not in content_type:
                continue

            html = resp.text
            local_name = slugify_for_filename(url) + ".html"
            local_path = config.RAW_PAGES_DIR / local_name
            local_path.write_text(html, encoding="utf-8")

            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else ""
            self.manifest.pages.append(
                {
                    "url": url,
                    "title": title,
                    "local_path": str(local_path.relative_to(config.DATA_DIR)),
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

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.MANIFEST_PATH.write_text(self.manifest.to_json(), encoding="utf-8")
        log.info(
            "Crawl finished: %d pages, %d assets, %d skipped-out-of-scope.",
            len(self.manifest.pages),
            len(self.manifest.assets),
            len(self.manifest.skipped_out_of_scope),
        )
        return self.manifest

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
        download_url = google_drive_direct_download_url(file_id)
        try:
            resp = self.session.get(download_url, timeout=config.REQUEST_TIMEOUT_SECONDS, stream=True)
        except requests.RequestException as exc:
            log.warning("Failed to fetch Google Drive file %s: %s", file_id, exc)
            return
        time.sleep(config.REQUEST_DELAY_SECONDS)
        if resp.status_code != 200:
            log.warning("Google Drive file %s returned status %s", file_id, resp.status_code)
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
            return

        filename = filename_from_content_disposition(resp.headers.get("Content-Disposition", ""))
        ext = Path(filename).suffix.lower() if filename else ""
        if ext not in config.ASSET_EXTENSIONS:
            log.info(
                "Skipping Google Drive file %s: unrecognized/missing extension %r (filename=%r)",
                file_id, ext, filename,
            )
            return

        local_name = slugify_for_filename(download_url) + ext
        local_path = config.RAW_ASSETS_DIR / local_name
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

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
            }
        )

    def _download_asset(self, url: str, ext: str, referenced_from: str):
        try:
            resp = self.session.get(url, timeout=config.REQUEST_TIMEOUT_SECONDS, stream=True)
        except requests.RequestException as exc:
            log.warning("Failed to fetch asset %s: %s", url, exc)
            return
        time.sleep(config.REQUEST_DELAY_SECONDS)
        if resp.status_code != 200:
            log.warning("Asset %s returned status %s", url, resp.status_code)
            return

        local_name = slugify_for_filename(url) + ext
        local_path = config.RAW_ASSETS_DIR / local_name
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        log.info("Downloaded asset: %s -> %s", url, local_path.name)
        self.manifest.assets.append(
            {
                "url": url,
                "extension": ext,
                "local_path": str(local_path.relative_to(config.DATA_DIR)),
                "referenced_from": referenced_from,
                "content_type": resp.headers.get("Content-Type", ""),
                "size_bytes": local_path.stat().st_size,
            }
        )


def main():
    crawler = ScopedCrawler()
    crawler.crawl()


if __name__ == "__main__":
    main()
