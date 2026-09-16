"""Scope-limited, single-visit crawler for the MITS IT department pages.

Design goals (per project requirements):
  * Only ever touch ``config.ALLOWED_DOMAIN``.
  * Only follow links for further crawling that stay under
    ``config.ALLOWED_PATH_PREFIX`` (the IT department subtree).
  * Visit every URL at most once -- no re-queueing, no re-fetching, so a
    cyclic link graph can never turn into an infinite loop.
  * Persist every page (raw HTML) and every asset (PDF/image/audio/video)
    referenced from within scope to disk, plus a manifest describing what
    was collected and where it came from.
"""
from __future__ import annotations

import hashlib
import json
import logging
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

        queue: deque[str] = deque([normalize_url(start_url)])
        queued: set[str] = {normalize_url(start_url)}

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

            if not is_in_crawl_scope(url):
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

            self._extract_and_queue_links(soup, url, queue, queued)
            self._extract_and_download_assets(soup, url)

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.MANIFEST_PATH.write_text(self.manifest.to_json(), encoding="utf-8")
        log.info(
            "Crawl finished: %d pages, %d assets, %d skipped-out-of-scope.",
            len(self.manifest.pages),
            len(self.manifest.assets),
            len(self.manifest.skipped_out_of_scope),
        )
        return self.manifest

    def _extract_and_queue_links(self, soup, page_url, queue, queued):
        for a in soup.find_all("a", href=True):
            absolute = normalize_url(urljoin(page_url, a["href"]))
            if asset_extension(absolute):
                continue  # handled by _extract_and_download_assets
            if not is_in_crawl_scope(absolute):
                continue
            if absolute in self.visited_pages or absolute in queued:
                continue
            queued.add(absolute)
            queue.append(absolute)

    def _extract_and_download_assets(self, soup, page_url):
        candidates: list[str] = []
        for a in soup.find_all("a", href=True):
            candidates.append(a["href"])
        for tag, attr in (("img", "src"), ("source", "src"), ("video", "src"), ("audio", "src")):
            for el in soup.find_all(tag):
                if el.get(attr):
                    candidates.append(el[attr])
                if el.get("data-src"):
                    candidates.append(el["data-src"])

        for href in candidates:
            absolute = normalize_url(urljoin(page_url, href))
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
