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


def test_cited_numbers_understands_grouped_citations():
    assert webapp.cited_numbers("See [2].") == {2}
    assert webapp.cited_numbers("From [4, 5, 6] and [1][3]") == {1, 3, 4, 5, 6}
    assert webapp.cited_numbers("Range [2-4]") == {2, 3, 4}


def test_saved_copy_is_served_but_only_from_data_folders(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    (tmp_path / "raw" / "assets").mkdir(parents=True)
    (tmp_path / "raw" / "assets" / "tt.pdf").write_bytes(b"%PDF-1.4 test")
    (tmp_path / "chat_log.db").write_bytes(b"secret")
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "faiss.index").write_bytes(b"secret")

    ok = client.get("/files/raw/assets/tt.pdf")
    assert ok.status_code == 200 and ok.data == b"%PDF-1.4 test"
    assert client.get("/files/chat_log.db").status_code == 404
    assert client.get("/files/index/faiss.index").status_code == 404
    assert client.get("/files/raw/../chat_log.db").status_code == 404
    assert client.get("/files/raw/assets/missing.pdf").status_code == 404


def test_answer_sources_carry_open_links(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    (tmp_path / "raw" / "pages").mkdir(parents=True)
    (tmp_path / "raw" / "pages" / "syllabus.html").write_text("<html></html>")

    def answer(question, top_k=config.TOP_K):
        return {"query": question, "answer": "It is in the syllabus [1, 2].", "provider": "groq:x",
                "generated": True,
                "sources": [
                    {"source_url": "https://web.mitsgwalior.in/syllabus-it", "source_type": "page",
                     "local_path": "raw\\pages\\syllabus.html", "title": "Syllabus", "text": "t", "date": "2026-07-01"},
                    {"source_url": "upload://rules.pdf", "source_type": "upload", "local_path": "uploads/rules.pdf",
                     "title": "rules", "text": "t"},
                ]}

    monkeypatch.setattr(pipeline, "answer", answer)
    data = client.post("/api/query", json={"question": "syllabus?"}).get_json()
    first, second = data["sources"]
    assert data["cited"] == [1, 2]
    assert first["link"] == "https://web.mitsgwalior.in/syllabus-it"
    assert first["saved_copy"] == "/files/raw/pages/syllabus.html"
    assert first["date"] == "Jul 2026"
    assert second["link"] is None and second["saved_copy"] is None  # file missing -> no broken link


def test_drive_pdf_sources_get_their_file_name_not_view():
    chunk = {"source_url": "https://drive.google.com/file/d/abc/view", "source_type": "pdf",
             "title": "Time Table - MITS Gwalior - 1st sem TT.pdf"}
    assert webapp._source_label(chunk) == "1st sem TT.pdf"
    chunk2 = {"source_url": "https://drive.google.com/file/d/abc/view", "source_type": "pdf", "title": ""}
    assert webapp._source_label(chunk2) == "Google Drive file"


def test_saved_web_page_shows_only_main_content(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    (tmp_path / "raw" / "pages").mkdir(parents=True)
    (tmp_path / "raw" / "pages" / "faculty.html").write_text(
        "<html><head><title>Faculty</title></head><body><nav>Huge site menu</nav>"
        "<div id='sp-main-body'><h2>Dr. X, HoD</h2><script>evil()</script></div></body></html>"
    )
    res = client.get("/files/raw/pages/faculty.html")
    html = res.get_data(as_text=True)
    assert res.status_code == 200
    assert "Dr. X, HoD" in html and "Saved copy of <b>Faculty</b>" in html
    assert "Huge site menu" not in html and "evil()" not in html


def test_site_pdf_uses_real_file_name_not_internal_slug():
    chunk = {"source_url": "https://web.mitsgwalior.in/images/BoS/8.%20BoS%20MEETING%202021.pdf",
             "source_type": "pdf",
             "title": "Board of Studies (BoS) - MITS Gwalior - images__BoS__8.%20BoS%20MEETING%202021.pdf__c427dbbf.pdf"}
    assert webapp._source_label(chunk) == "8. BoS MEETING 2021.pdf"
