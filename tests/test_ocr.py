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
