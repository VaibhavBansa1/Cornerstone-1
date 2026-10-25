"""Audit the last crawl: did we really get everything?

    python scripts/check_crawl.py           # offline checks only
    python scripts/check_crawl.py --live    # also re-read the live department menu

What it checks:
1. Was the last crawl finished, or interrupted?
2. Every page/asset in the manifest really exists on disk.
3. Saved files that are NOT in the manifest (left over from an
   interrupted or older crawl).
4. Re-reads every saved page and lists every in-scope link/asset the
   crawler should have followed, then reports any that are in neither the
   "saved" nor the "failed" list. An empty list = nothing was silently lost.
5. What failed and why (dead links, timeouts, big Drive files).
6. (--live) Compares today's department sidebar menu with what we saved.
"""
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup  # noqa: E402

import config  # noqa: E402
from scraper.crawler import (  # noqa: E402
    asset_extension,
    extract_google_drive_file_id,
    is_allowed_domain,
    is_in_crawl_scope,
    normalize_url,
)


def main():
    live = "--live" in sys.argv
    manifest = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    pages = {p["url"]: p for p in manifest["pages"]}
    assets = {a["url"]: a for a in manifest["assets"]}
    failed = {f["url"]: f for f in manifest.get("failed", [])}

    print("1. Crawl status:", manifest.get("status", "unknown (manifest from before status was recorded)"))
    print(f"   {len(pages)} pages, {len(assets)} assets, {len(failed)} failed")

    missing = [u for u, e in {**pages, **assets}.items() if not (config.DATA_DIR / e["local_path"]).exists()]
    print(f"\n2. Manifest entries whose file is missing on disk: {len(missing)}")
    for u in missing[:10]:
        print("   -", u)

    listed = {Path(e["local_path"]).name for e in [*pages.values(), *assets.values()]}
    on_disk = [p for d in (config.RAW_PAGES_DIR, config.RAW_ASSETS_DIR) if d.exists() for p in d.iterdir()]
    orphans = [p.name for p in on_disk if p.name not in listed]
    print(f"\n3. Files on disk not in the manifest (leftovers): {len(orphans)}")
    for name in orphans[:10]:
        print("   -", name)

    # 4. Re-discover links from the saved pages using the crawler's own rules.
    should_pages, should_assets, drive_ids = set(), set(), set()
    for url, entry in pages.items():
        html = (config.DATA_DIR / entry["local_path"]).read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")
        scope = soup.find(id=config.MAIN_CONTENT_CONTAINER_ID) or soup
        nav = scope.find("nav", attrs={"aria-label": config.DEPARTMENT_NAV_ARIA_LABEL})
        nav_urls = {normalize_url(urljoin(url, a["href"])) for a in nav.find_all("a", href=True)} if nav else set()
        hrefs = [a["href"] for a in scope.find_all("a", href=True)]
        for tag in ("img", "source", "video", "audio"):
            hrefs += [el[k] for el in scope.find_all(tag) for k in ("src", "data-src") if el.get(k)]
        for href in hrefs:
            absolute = normalize_url(urljoin(url, href))
            drive = extract_google_drive_file_id(absolute)
            if drive:
                drive_ids.add(drive)
            elif asset_extension(absolute):
                if is_allowed_domain(absolute):
                    should_assets.add(absolute)
            elif is_allowed_domain(absolute) and (is_in_crawl_scope(absolute) or absolute in nav_urls):
                should_pages.add(absolute)

    got_drive = {extract_google_drive_file_id(u) for u in [*assets, *failed]} - {None}
    lost_pages = sorted(u for u in should_pages if u not in pages and u not in failed)
    lost_assets = sorted(u for u in should_assets if u not in assets and u not in failed)
    lost_drive = sorted(d for d in drive_ids if d not in got_drive)
    print("\n4. Links the crawler should have followed but are in NEITHER saved NOR failed:")
    print(f"   pages: {len(lost_pages)} of {len(should_pages)} | site files: {len(lost_assets)} of "
          f"{len(should_assets)} | Drive files: {len(lost_drive)} of {len(drive_ids)}")
    for u in (lost_pages + lost_assets)[:15]:
        print("   -", u)
    for d in lost_drive[:15]:
        print("   - drive:", d)

    print("\n5. Failed items by reason:")
    reasons = Counter()
    for f in failed.values():
        r = f["reason"]
        key = ("timeout/connection" if "timed out" in r.lower() or "connection" in r.lower()
               else "large Drive file (virus-scan page)" if "html" in r.lower() and f.get("kind") == "drive"
               else r[:60])
        reasons[(f.get("kind"), key)] += 1
    for (kind, key), n in reasons.most_common():
        print(f"   {n:>4}  {kind}: {key}")

    if live:
        import requests

        print("\n6. Live check of the department menu:")
        resp = requests.get(config.START_URL, timeout=20, headers={"User-Agent": config.USER_AGENT})
        soup = BeautifulSoup(resp.text, "html.parser")
        nav = soup.find("nav", attrs={"aria-label": config.DEPARTMENT_NAV_ARIA_LABEL})
        live_links = {normalize_url(urljoin(config.START_URL, a["href"])) for a in nav.find_all("a", href=True)} if nav else set()
        live_links = {u for u in live_links if is_allowed_domain(u) and not asset_extension(u)}
        not_saved = sorted(u for u in live_links if u not in pages)
        print(f"   menu links today: {len(live_links)} | not in our crawl: {len(not_saved)}")
        for u in not_saved:
            print("   -", u, "(failed: " + failed[u]["reason"][:50] + ")" if u in failed else "")


if __name__ == "__main__":
    main()
