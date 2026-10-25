"""Web front end: student chat screen + staff admin page.

    python -m web.app          # then open http://localhost:5000

Endpoints:
    POST   /api/query              {"question": "..."} -> answer + sources
    GET    /api/status             index / generation backend readiness
    GET    /api/documents          what is currently indexed
    GET    /api/stats              usage stats from the chat log
    POST   /api/uploads            upload a PDF/DOCX/TXT (admin password)
    DELETE /api/uploads/<name>     remove an uploaded document (admin password)
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote, unquote

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

import config
from ingestion.dates import label as date_label
from ingestion.extract import build_chunks, write_chunks
from rag import llm, pipeline
from rag.embed_index import build_index, read_index_info
from rag.embeddings import resolve_device
from rag.retrieval import IndexMismatchError
from web import chatlog

log = logging.getLogger("web")

STATIC_DIR = Path(__file__).parent / "static"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "mits-admin")

app = Flask(__name__, static_folder=None)

_rebuild_lock = threading.Lock()
_rebuild_state = {"running": False, "last_error": None, "finished_at": None}


def _source_label(chunk: dict) -> str:
    """Short, human name for a source card: the file name for documents,
    the page title for web pages ("Time%20Table.pdf" -> "Time Table.pdf")."""
    return unquote(_raw_source_label(chunk))


def _raw_source_label(chunk: dict) -> str:
    title = (chunk.get("title") or "").strip()
    if chunk.get("source_type") == "upload":
        return title or chunk["source_url"].removeprefix("upload://")
    if chunk.get("source_type") in ("pdf", "image", "audio", "video", "document"):
        # PDFs are titled "<page title> - <file name>"; Drive links end in
        # "/view", so the URL is no good as a name.
        url_name = chunk["source_url"].rstrip("/").rsplit("/", 1)[-1]
        if " - " in title:
            name = title.rsplit(" - ", 1)[-1].strip()
            # "uc__1a2b.pdf" / "images__Dept__x.pdf__9f3e.pdf" are our own
            # saved-file names; the link's last part is the real file name.
            looks_internal = name.lower().startswith("uc__") or "__" in name
            if name and not looks_internal:
                return name
            if looks_internal and url_name.lower() not in ("view", "edit", "preview"):
                return url_name
        name = url_name
        if name.lower() in ("view", "edit", "preview") or name.startswith("uc?"):
            return title or "Google Drive file"
        return name
    return title.replace(" - MITS Gwalior", "") or chunk["source_url"].rstrip("/").rsplit("/", 1)[-1]


def _saved_copy_link(chunk: dict) -> str | None:
    """URL of our own downloaded copy of this source (works even if the
    college site or Google Drive is down; the only link for uploads)."""
    local = (chunk.get("local_path") or "").replace("\\", "/")
    if not local or not (config.DATA_DIR / local).exists():
        return None
    return "/files/" + quote(local)


def _public_source(n: int, chunk: dict) -> dict:
    url = chunk["source_url"]
    return {
        "n": n,
        "title": _source_label(chunk),
        "doc_title": chunk.get("title") or "",
        "source_url": url,
        "link": None if url.startswith("upload://") else url,
        "saved_copy": _saved_copy_link(chunk),
        "source_type": chunk.get("source_type"),
        "score": round(chunk.get("score", 0.0), 3),
        "date": date_label(chunk.get("date")),
        "snippet": chunk["text"][:400],
        "text": chunk["text"],
    }


_CITATION = re.compile(r"\[(\s*\d+(?:\s*[,\u2013-]\s*\d+)*\s*)\]")


def cited_numbers(answer: str) -> set[int]:
    """Source numbers cited in an answer: [2], [1, 3], [4-6] and [1][2]."""
    found: set[int] = set()
    for group in _CITATION.findall(answer or ""):
        for part in group.split(","):
            part = part.strip()
            bounds = re.split(r"\s*[\u2013-]\s*", part)
            if len(bounds) == 2 and all(b.isdigit() for b in bounds):
                lo, hi = int(bounds[0]), int(bounds[1])
                if 0 < hi - lo < 20:
                    found.update(range(lo, hi + 1))
                    continue
            if part.isdigit():
                found.add(int(part))
    return found


def _check_admin():
    supplied = request.headers.get("X-Admin-Password") or request.form.get("password")
    return supplied == ADMIN_PASSWORD


@app.get("/")
def chat_page():
    return send_from_directory(STATIC_DIR, "index.html", max_age=0)


@app.get("/admin")
def admin_page():
    return send_from_directory(STATIC_DIR, "admin.html", max_age=0)


@app.get("/static/<path:name>")
def static_files(name):
    # max_age=0: browsers re-check the file each time, so an updated page
    # never runs with an old cached script.
    return send_from_directory(STATIC_DIR, name, max_age=0)


# Only these folders inside data/ may be served -- never the index, the
# chat log or anything else.
SERVABLE_DIRS = ("raw/", "uploads/")


@app.get("/files/<path:subpath>")
def saved_file(subpath):
    """Open our saved copy of a crawled page / PDF / uploaded document."""
    subpath = subpath.replace("\\", "/")
    if not subpath.startswith(SERVABLE_DIRS) or ".." in subpath.split("/"):
        return jsonify({"error": "Not found."}), 404
    path = config.DATA_DIR / subpath
    if not path.is_file():
        return jsonify({"error": "Not found."}), 404
    if path.suffix.lower() == ".html":
        return _clean_saved_page(path, _original_url_for(subpath)), 200, {"Content-Type": "text/html; charset=utf-8"}
    return send_from_directory(config.DATA_DIR, subpath)


SAVED_PAGE_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{base}<title>{title} (saved copy)</title>
<style>
body {{ font-family: "Segoe UI", Roboto, Arial, sans-serif; margin: 0; background: #f4f6f9; color: #1c1f24; line-height: 1.55; }}
.bar {{ background: #1f3b73; color: #fff; padding: 10px 16px; font-size: 14px; }}
.bar a {{ color: #fff; font-weight: 600; }}
main {{ max-width: 860px; margin: 16px auto; background: #fff; border: 1px solid #dde2ea; border-radius: 12px; padding: 20px 24px; }}
img {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; display: block; overflow-x: auto; }}
td, th {{ border: 1px solid #dde2ea; padding: 6px 8px; vertical-align: top; }}
h1, h2, h3 {{ color: #1f3b73; }}
</style></head><body>
<div class="bar">Saved copy of <b>{title}</b> from the help desk index{live}</div>
<main>{content}</main></body></html>"""


def _clean_saved_page(path: Path, original: str | None) -> str:
    """Our saved copy of a web page, showing only its main content.

    The college site's own styles and menus don't load outside the site,
    so the raw page looks broken; this shows the readable part instead."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    main = soup.find(id=config.MAIN_CONTENT_CONTAINER_ID) or soup.body or soup
    for tag in main(["script", "style", "noscript", "iframe", "form", "nav", "aside"]):
        tag.decompose()
    esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")  # noqa: E731
    return SAVED_PAGE_TEMPLATE.format(
        base=f'<base href="{esc(original)}" target="_blank">' if original else "",
        title=esc(title),
        live=f' &middot; <a href="{esc(original)}" target="_blank" rel="noopener">Open the live page</a>' if original else "",
        content=str(main),
    )


def _original_url_for(local_path: str) -> str | None:
    if not pipeline.index_exists():
        return None
    try:
        _index, chunks = pipeline._load_index()
    except Exception:
        return None
    for c in chunks:
        if (c.get("local_path") or "").replace("\\", "/") == local_path:
            return c["source_url"]
    return None


@app.get("/api/status")
def status():
    info = read_index_info()
    return jsonify(
        {
            "index_ready": pipeline.index_exists(),
            # e.g. "openrouter:google/gemma-4-31b-it:free -> ollama:llama3.1"
            "provider": llm.describe_chain(),
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "index_built_with": info.get("embedding_model"),
            "embedding_device": resolve_device(),
            "reranker": config.RERANKER_MODEL,
            "rebuilding": _rebuild_state["running"],
            "last_rebuild_error": _rebuild_state["last_error"],
        }
    )


@app.post("/api/query")
def query():
    question = ((request.get_json(silent=True) or {}).get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please type a question."}), 400
    if not pipeline.index_exists():
        return jsonify({"error": "The document index has not been built yet. Run `python run_pipeline.py` first."}), 503

    started = time.perf_counter()
    try:
        result = pipeline.answer(question)
    except IndexMismatchError as exc:
        return jsonify({"error": str(exc)}), 503
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    sources = [_public_source(i, c) for i, c in enumerate(result["sources"], 1)]
    # When no LLM answered, the pipeline returns the retrieved passages
    # instead; the UI then shows those passages.
    generated = result.get("generated", result["provider"] != "none")
    numbers = cited_numbers(result["answer"]) if generated else set()
    cited = [s for s in sources if s["n"] in numbers] or sources[:1]

    chatlog.log_question(question, result["answer"], cited, result["provider"], elapsed_ms)
    return jsonify(
        {
            "answer": result["answer"] if generated else None,
            "generated": generated,
            "provider": result["provider"],
            "sources": sources,
            "cited": [s["n"] for s in cited],
            "response_ms": elapsed_ms,
        }
    )


@app.get("/api/documents")
def documents():
    if not pipeline.index_exists():
        return jsonify({"documents": []})
    _index, chunks = pipeline._load_index()
    docs: OrderedDict[str, dict] = OrderedDict()
    for c in chunks:
        doc = docs.setdefault(
            c["source_url"],
            {
                "title": _source_label(c),
                "source_url": c["source_url"],
                "source_type": c.get("source_type"),
                "chunks": 0,
                "date": date_label(c.get("doc_date") or c.get("date")),
                "saved_copy": _saved_copy_link(c),
                "deletable": c.get("source_type") == "upload",
            },
        )
        doc["chunks"] += 1
    return jsonify({"documents": list(docs.values())})


@app.get("/api/stats")
def stats():
    by_url: dict[str, dict] = {}
    if pipeline.index_exists():
        try:
            _index, chunks = pipeline._load_index()
            for c in chunks:
                by_url.setdefault(c["source_url"], c)
        except Exception:
            pass

    def label_for(url):
        chunk = by_url.get(url)
        return _source_label(chunk) if chunk else unquote(url.rstrip("/").rsplit("/", 1)[-1]) or None

    def link_for(url):
        chunk = by_url.get(url)
        if url.startswith("upload://"):
            return _saved_copy_link(chunk) if chunk else None
        return url if url.startswith("http") else None

    data = chatlog.stats(label_for=label_for, link_for=link_for)
    data["provider"] = llm.describe_chain()
    return jsonify(data)


def _rebuild_index():
    """Re-chunk everything (crawl + uploads) and rebuild the FAISS index."""
    with _rebuild_lock:
        _rebuild_state["running"] = True
        _rebuild_state["last_error"] = None
        try:
            chunks = build_chunks()
            write_chunks(chunks)
            if chunks:
                build_index(chunks)
            else:
                for p in (config.FAISS_INDEX_PATH, config.INDEX_METADATA_PATH):
                    p.unlink(missing_ok=True)
            pipeline.reload_index()
        except Exception as exc:
            log.exception("Index rebuild failed")
            _rebuild_state["last_error"] = str(exc)
        finally:
            _rebuild_state["running"] = False
            _rebuild_state["finished_at"] = time.time()


def _start_rebuild():
    threading.Thread(target=_rebuild_index, daemon=True).start()


@app.post("/api/uploads")
def upload():
    if not _check_admin():
        return jsonify({"error": "Wrong admin password."}), 403
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file selected."}), 400
    name = secure_filename(file.filename)
    if Path(name).suffix.lower() not in config.UPLOAD_EXTENSIONS:
        return jsonify({"error": "Only PDF, DOCX and TXT files are supported."}), 400
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file.save(config.UPLOADS_DIR / name)
    _start_rebuild()
    return jsonify({"saved": name, "rebuilding": True})


@app.delete("/api/uploads/<name>")
def delete_upload(name):
    if not _check_admin():
        return jsonify({"error": "Wrong admin password."}), 403
    path = config.UPLOADS_DIR / secure_filename(name)
    if not path.exists():
        return jsonify({"error": "No such uploaded document."}), 404
    path.unlink()
    _start_rebuild()
    return jsonify({"deleted": name, "rebuilding": True})


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    port = int(os.environ.get("PORT", "5000"))
    log.info("Chat: http://localhost:%d   Admin: http://localhost:%d/admin", port, port)
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
