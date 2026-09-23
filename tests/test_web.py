"""Web API tests. The embedding model / LLM are stubbed so no network is needed."""
import io

import pytest

import config
from rag import pipeline
from web import app as webapp


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHAT_LOG_DB_PATH", tmp_path / "chat.db")
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(pipeline, "index_exists", lambda: True)
    monkeypatch.setattr(webapp, "_start_rebuild", lambda: None)
    return webapp.app.test_client()


def _fake_answer(provider, text):
    def answer(question, top_k=config.TOP_K):
        return {
            "query": question,
            "answer": text,
            "provider": provider,
            "sources": [
                {"source_url": "https://web.mitsgwalior.in/syllabus-it", "source_type": "page",
                 "title": "Syllabus IT", "text": "5th sem syllabus ...", "score": 0.8},
                {"source_url": "https://web.mitsgwalior.in/uploads/tt.pdf", "source_type": "pdf",
                 "title": "Time Table", "text": "Monday ...", "score": 0.6},
            ],
        }
    return answer


def test_query_returns_generated_answer_with_cited_sources(client, monkeypatch):
    monkeypatch.setattr(pipeline, "answer", _fake_answer("gemini", "See the timetable [2]."))
    res = client.post("/api/query", json={"question": "When is the class?"})
    data = res.get_json()
    assert res.status_code == 200
    assert data["generated"] is True
    assert data["cited"] == [2]
    assert data["sources"][1]["title"] == "tt.pdf"

    stats = client.get("/api/stats").get_json()
    assert stats["total_questions"] == 1
    assert stats["recent_questions"][0]["question"] == "When is the class?"
    assert stats["top_sources"][0]["name"] == "tt.pdf"


def test_query_without_llm_falls_back_to_passages(client, monkeypatch):
    monkeypatch.setattr(pipeline, "answer", _fake_answer("none", "[No generation provider configured ...]"))
    data = client.post("/api/query", json={"question": "syllabus?"}).get_json()
    assert data["generated"] is False
    assert data["answer"] is None
    assert data["cited"] == [1]


def test_query_rejects_empty_question_and_missing_index(client, monkeypatch):
    assert client.post("/api/query", json={"question": "  "}).status_code == 400
    monkeypatch.setattr(pipeline, "index_exists", lambda: False)
    assert client.post("/api/query", json={"question": "hi"}).status_code == 503


def test_upload_requires_admin_password_and_allowed_type(client):
    def post(name, password):
        return client.post(
            "/api/uploads",
            data={"file": (io.BytesIO(b"hello"), name), "password": password},
            content_type="multipart/form-data",
        )

    assert post("notes.txt", "wrong").status_code == 403
    assert post("virus.exe", webapp.ADMIN_PASSWORD).status_code == 400
    res = post("notes.txt", webapp.ADMIN_PASSWORD)
    assert res.status_code == 200
    assert (config.UPLOADS_DIR / "notes.txt").read_bytes() == b"hello"

    headers = {"X-Admin-Password": webapp.ADMIN_PASSWORD}
    assert client.delete("/api/uploads/notes.txt", headers=headers).status_code == 200
    assert not (config.UPLOADS_DIR / "notes.txt").exists()
    assert client.delete("/api/uploads/notes.txt", headers=headers).status_code == 404
