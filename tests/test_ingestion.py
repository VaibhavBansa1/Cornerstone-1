import json

import config
from ingestion.extract import build_chunks, chunk_text, html_to_text


def test_html_to_text_strips_script_and_nav():
    html = """
    <html><body>
      <nav>Menu</nav>
      <script>var x = 1;</script>
      <main><h1>IT Department</h1><p>We offer B.Tech in IT.</p></main>
      <footer>Copyright</footer>
    </body></html>
    """
    text = html_to_text(html)
    assert "Menu" not in text
    assert "var x = 1" not in text
    assert "Copyright" not in text
    assert "IT Department" in text
    assert "B.Tech in IT" in text


def test_chunk_text_respects_size_and_overlap():
    text = "A" * 2500
    chunks = chunk_text(text, size=1000, overlap=150)
    assert len(chunks) == 3
    assert all(len(c) <= 1000 for c in chunks)
    # overlap: end of chunk 0 should reappear at the start of chunk 1
    assert chunks[0][-150:] == chunks[1][:150]


def test_chunk_text_short_text_single_chunk():
    assert chunk_text("short text") == ["short text"]
    assert chunk_text("   ") == []


def test_build_chunks_covers_pages_pdfs_and_media(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    pages_dir = tmp_path / "raw" / "pages"
    assets_dir = tmp_path / "raw" / "assets"
    pages_dir.mkdir(parents=True)
    assets_dir.mkdir(parents=True)

    page_html = "<html><head><title>Home</title></head><body><p>Welcome to IT dept.</p></body></html>"
    (pages_dir / "home.html").write_text(page_html, encoding="utf-8")
    (assets_dir / "logo.png").write_bytes(b"\x89PNG-fake")

    manifest = {
        "pages": [
            {"url": "https://web.mitsgwalior.in/academics/departments/information-technology", "title": "Home", "local_path": "raw/pages/home.html"}
        ],
        "assets": [
            {
                "url": "https://web.mitsgwalior.in/uploads/logo.png",
                "extension": ".png",
                "local_path": "raw/assets/logo.png",
                "referenced_from": "https://web.mitsgwalior.in/academics/departments/information-technology",
                "content_type": "image/png",
                "size_bytes": 9,
            }
        ],
        "skipped_out_of_scope": [],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(config, "MANIFEST_PATH", manifest_path)

    chunks = build_chunks()
    types = {c["source_type"] for c in chunks}
    assert "page" in types
    assert "image" in types

    image_chunk = next(c for c in chunks if c["source_type"] == "image")
    assert "logo.png" in image_chunk["text"]
    assert "Home" in image_chunk["text"]
