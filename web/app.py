"""Web front end: student chat screen + staff admin page.

    python -m web.app          # then open http://localhost:5000

Endpoints:
    POST   /api/query              {"question": "..."} -> answer + sources
    POST   /api/query/stream       same, with progress lines first (chat page)
    GET    /api/status             index / generation backend readiness
    GET    /api/documents          what is currently indexed
    GET    /api/stats              usage stats from the chat log
    POST   /api/uploads            upload a PDF/DOCX/TXT (admin password)
    DELETE /api/uploads/<name>     remove an uploaded document (admin password)
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import unquote

from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

import config
from ingestion.dates import label as date_label
from ingestion.build import build_chunks, write_chunks
from rag import llm, pipeline
from rag.embed_index import build_index, read_index_info
from rag.embeddings import resolve_device
from rag.retrieval import IndexMismatchError
from web import chatlog
from web.sources import (
    cited_numbers,
    original_url_for,
    public_source,
    render_saved_page,
    saved_copy_link,
    source_label,
)

log = logging.getLogger("web")

STATIC_DIR = Path(__file__).parent / "static"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "mits-admin")

app = Flask(__name__, static_folder=None)

_rebuild_lock = threading.Lock()
_rebuild_state = {"running": False, "last_error": None, "finished_at": None}


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
        return render_saved_page(path, original_url_for(subpath)), 200, {"Content-Type": "text/html; charset=utf-8"}
    return send_from_directory(config.DATA_DIR, subpath)




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


def _check_query():
    """(question, error response or None)"""
    question = ((request.get_json(silent=True) or {}).get("question") or "").strip()
    if not question:
        return question, (jsonify({"error": "Please type a question."}), 400)
    if not pipeline.index_exists():
        return question, (jsonify({"error": "The document index has not been built yet. "
                                            "Run `python run_pipeline.py` first."}), 503)
    return question, None


def _answer_payload(result: dict, elapsed_ms: int) -> dict:
    """The JSON the chat page renders; also logs the (anonymous) usage."""
    sources = [public_source(i, c) for i, c in enumerate(result["sources"], 1)]
    # When no LLM answered, the pipeline returns the retrieved passages
    # instead; the UI then shows those passages.
    generated = result.get("generated", result["provider"] != "none")
    numbers = cited_numbers(result["answer"]) if generated else set()
    # An answer that cites nothing (e.g. "I don't have that") shows no
    # source as "used"; passages-only mode highlights the best passage.
    cited = [s for s in sources if s["n"] in numbers] if generated else sources[:1]
    chatlog.log_question(cited, result["provider"], elapsed_ms)  # question text is never stored
    return {
        "answer": result["answer"] if generated else None,
        "generated": generated,
        "provider": result["provider"],
        "sources": sources,
        "cited": [s["n"] for s in cited],
        "response_ms": elapsed_ms,
        # e.g. "5th sem · 2024-28 batch · July-Dec 2026 session · syllabus"
        "understood_as": result.get("understood_as") or "",
    }


@app.post("/api/query")
def query():
    question, error = _check_query()
    if error:
        return error
    started = time.perf_counter()
    try:
        result = pipeline.answer(question)
    except IndexMismatchError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify(_answer_payload(result, int((time.perf_counter() - started) * 1000)))


def _step_event(kind: str, data) -> dict:
    """A progress line for the chat page, in plain words."""
    if kind == "found":
        return {"type": "step", "step": "found", "sources": [source_label(c) for c in data]}
    if kind == "writing":
        return {"type": "step", "step": "writing", "provider": data["provider"], "fallback": bool(data["after_error"])}
    return {"type": "step", "step": kind, "value": data}


@app.post("/api/query/stream")
def query_stream():
    """Same answer as /api/query, but sends progress first: one JSON object
    per line ({"type": "step", ...} lines, then {"type": "answer", ...})."""
    question, error = _check_query()
    if error:
        return error
    events: queue.Queue = queue.Queue()

    def work():
        started = time.perf_counter()
        try:
            result = pipeline.answer(question, on_step=lambda k, d: events.put(_step_event(k, d)))
            events.put({"type": "answer", **_answer_payload(result, int((time.perf_counter() - started) * 1000))})
        except Exception as exc:
            log.exception("Answering failed")
            events.put({"type": "error", "error": str(exc) if isinstance(exc, IndexMismatchError)
                        else "Something went wrong while answering."})

    threading.Thread(target=work, daemon=True).start()

    def lines():
        while True:
            event = events.get()
            yield json.dumps(event, ensure_ascii=False) + "\n"
            if event["type"] in ("answer", "error"):
                return

    return Response(lines(), mimetype="application/x-ndjson", headers={"Cache-Control": "no-cache",
                                                                       "X-Accel-Buffering": "no"})


@app.get("/api/documents")
def documents():
    if not pipeline.index_exists():
        return jsonify({"documents": []})
    chunks = pipeline.indexed_chunks()
    docs: OrderedDict[str, dict] = OrderedDict()
    for c in chunks:
        doc = docs.setdefault(
            c["source_url"],
            {
                "title": source_label(c),
                "source_url": c["source_url"],
                "source_type": c.get("source_type"),
                "chunks": 0,
                "date": date_label(c.get("doc_date") or c.get("date")),
                "saved_copy": saved_copy_link(c),
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
            chunks = pipeline.indexed_chunks()
            for c in chunks:
                by_url.setdefault(c["source_url"], c)
        except Exception:
            pass

    def label_for(url):
        chunk = by_url.get(url)
        return source_label(chunk) if chunk else unquote(url.rstrip("/").rsplit("/", 1)[-1]) or None

    def link_for(url):
        chunk = by_url.get(url)
        if url.startswith("upload://"):
            return saved_copy_link(chunk) if chunk else None
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


def _warm_up():
    """Load the index and search models now, so the first student doesn't
    wait ~20 s on "Searching..." while they load."""
    try:
        if pipeline.index_exists():
            pipeline.retrieve("warm up")
            log.info("Search models loaded.")
    except Exception as exc:  # a problem here shows up again on the first question
        log.warning("Warm-up failed: %s", exc)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    threading.Thread(target=_warm_up, daemon=True).start()
    port = int(os.environ.get("PORT", "5000"))
    log.info("Chat: http://localhost:%d   Admin: http://localhost:%d/admin", port, port)
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
