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
    from ingestion import extract, ocr

    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    pdf = tmp_path / "photo.pdf"
    pdf.write_bytes(b"%PDF-fake")
    cache = extract._pdf_cache_path(pdf)
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"text": "", "pages": 1, "scanned_pages": 1, "ocr_pages": 0,
                                 "ocr_attempted": True, "pdf_date": None}))
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_pdf_pages", lambda *a: (_ for _ in ()).throw(AssertionError("re-OCR'd")))
    assert extract.pdf_extract(pdf)["ocr_attempted"] is True
