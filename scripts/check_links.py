"""Check that every source the chatbot can cite actually opens.

    python scripts/check_links.py            # saved copies + original links
    python scripts/check_links.py --offline  # saved copies only

For each document in the index:
  * "saved copy": our downloaded file exists (what the Saved copy button opens)
  * "original":   the original link (college site / Google Drive) answers
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402

import requests  # noqa: E402

import config  # noqa: E402


def main():
    offline = "--offline" in sys.argv
    docs = {}
    with open(config.INDEX_METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            docs.setdefault(c["source_url"], c)

    missing_copy, broken_link, checked = [], [], 0
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT
    for url, c in docs.items():
        local = (c.get("local_path") or "").replace("\\", "/")
        if not local or not (config.DATA_DIR / local).is_file():
            missing_copy.append(url)
        if offline or url.startswith("upload://"):
            continue
        try:
            resp = session.get(url, timeout=20, stream=True, allow_redirects=True)
            ok = resp.status_code < 400
            resp.close()
        except requests.RequestException as exc:
            ok, resp = False, None
            reason = exc.__class__.__name__
        if not ok:
            broken_link.append((url, resp.status_code if resp is not None else reason))
        checked += 1
        time.sleep(0.2)

    print(f"Documents the bot can cite: {len(docs)}")
    print(f"Saved copy opens: {len(docs) - len(missing_copy)} / {len(docs)}")
    for u in missing_copy[:20]:
        print("   missing saved copy:", u)
    if not offline:
        print(f"Original link opens: {checked - len(broken_link)} / {checked}")
        for u, why in broken_link[:20]:
            print(f"   broken ({why}):", u)


if __name__ == "__main__":
    main()
