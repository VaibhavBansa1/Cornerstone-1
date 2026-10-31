from ingestion.ocr import boxes_to_text


def box(x, y, w=40, h=10):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def test_ocr_boxes_become_reading_order_lines_and_noise_is_dropped():
    results = [
        (box(200, 52), "Network", 0.9),
        (box(10, 50), "Mobile Computing &", 0.95),   # same line, further left
        (box(10, 80), "16243101", 0.99),              # next line
        (box(10, 5), "mera Hlulfd @7TF", 0.2),        # Hindi read by English model: dropped
        (box(300, 80), "~~", 0.9),                    # no letters: dropped
    ]
    assert boxes_to_text(results, min_conf=0.35) == "Mobile Computing & Network\n16243101"


def test_pdf_cache_is_reused_even_when_ocr_found_no_words(tmp_path, monkeypatch):
    """A scanned page with no readable words must not be re-OCR'd on every rebuild."""
    import json

    import config
    from ingestion import ocr, readers

    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    pdf = tmp_path / "photo.pdf"
    pdf.write_bytes(b"%PDF-fake")
    cache = readers._pdf_cache_path(pdf)
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"text": "", "pages": 1, "scanned_pages": 1, "ocr_pages": 0,
                                 "ocr_attempted": True, "pdf_date": None}))
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_pdf_pages", lambda *a: (_ for _ in ()).throw(AssertionError("re-OCR'd")))
    assert readers.pdf_extract(pdf)["ocr_attempted"] is True


def test_partly_scanned_pages_are_mostly_picture_with_little_typed_text(monkeypatch):
    import config
    from ingestion.readers import partly_scanned_pages

    monkeypatch.setattr(config, "OCR_MIXED_IMAGE_FRACTION", 0.5)
    monkeypatch.setattr(config, "OCR_MIXED_MAX_TEXT_CHARS", 1000)
    texts = ["MITS GWALIOR header only", "x" * 2000, "short text", "", "typed letter"]
    pictures = [0.9, 0.9, 0.1, 1.0, 0.6]
    # 0: header over a scan -> yes; 1: full hidden text layer -> no; 2: little picture -> no;
    # 3: already counted as scanned -> no; 4: yes
    assert partly_scanned_pages(texts, pictures, scanned={3}) == [0, 4]
    assert partly_scanned_pages(texts, [], scanned=set()) == []  # PDF couldn't be inspected


def test_ocr_confidence_counts_only_kept_words():
    from ingestion.ocr import mean_confidence

    assert mean_confidence([(box(0, 0), "abcd", 0.9), (box(0, 0), "ab", 0.6), (box(0, 0), "zz", 0.1)],
                           min_conf=0.35) == 0.8
    assert mean_confidence([(box(0, 0), "~~", 0.9)], min_conf=0.35) is None


def test_pdf_cache_made_without_ocr_is_redone_once_ocr_is_installed(tmp_path, monkeypatch):
    import json

    import config
    from ingestion import ocr, readers

    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    pdf = tmp_path / "letter.pdf"
    pdf.write_bytes(b"%PDF-fake")
    cache = readers._pdf_cache_path(pdf)
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"text": "old", "pages": 1, "scanned_pages": 0, "ocr_pages": 0,
                                 "ocr_checked": False, "pdf_date": None}))
    monkeypatch.setattr(ocr, "available", lambda: False)
    assert readers.pdf_extract(pdf)["text"] == "old"          # still no OCR: keep the cache
    monkeypatch.setattr(ocr, "available", lambda: True)
    assert readers.pdf_extract(pdf)["text"] != "old"          # OCR now installed: read again
