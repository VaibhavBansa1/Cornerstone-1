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
from urllib.parse import quote

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
    if chunk.get("source_type") == "upload":
        return chunk.get("title") or chunk["source_url"].removeprefix("upload://")
    title = (chunk.get("title") or "").strip()
    name = chunk["source_url"].rstrip("/").rsplit("/", 1)[-1]
    if chunk.get("source_type") in ("pdf", "image", "audio", "video", "document"):
        return name
    return title or name


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
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/admin")
def admin_page():
    return send_from_directory(STATIC_DIR, "admin.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC_DIR, name)


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
        # Point relative links/images at the live site so the saved page
        # still shows pictures and its links work.
        original = _original_url_for(subpath)
        html = path.read_text(encoding="utf-8", errors="ignore")
        if original:
            banner = (
                '<div style="background:#fff4e0;padding:6px 12px;font:13px sans-serif;'
                'border-bottom:1px solid #e5c98f">Saved copy from the help desk index. '
                f'<a href="{original}" target="_blank" rel="noopener">Open the live page</a></div>'
            )
            html = re.sub(r"(<head[^>]*>)", rf'\1<base href="{original}">', html, count=1, flags=re.I)
            html = re.sub(r"(<body[^>]*>)", rf"\1{banner}", html, count=1, flags=re.I)
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}
    return send_from_directory(config.DATA_DIR, subpath)


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
                "deletable": c.get("source_type") == "upload",
            },
        )
        doc["chunks"] += 1
    return jsonify({"documents": list(docs.values())})


@app.get("/api/stats")
def stats():
    return jsonify(chatlog.stats())


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
