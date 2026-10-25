"""Read text from scanned PDF pages (OCR).

Many college PDFs are scans: images of paper with no text inside, so the
chatbot can't read them. For each page with no text, we render it as an
image and run OCR on it.

Engines (pick with OCR_ENGINE=auto|easyocr|rapidocr|none):
* easyocr  -- uses PyTorch, fast on an NVIDIA GPU (~2 s/page)
* rapidocr -- CPU only, no PyTorch needed (~10 s/page)
"auto" = easyocr if a GPU is visible, else rapidocr, else easyocr on CPU,
else OCR is skipped with a warning. Install with:  uv sync --extra ocr

OCR'd text is cached (see ingestion.extract.pdf_to_text), so each page is
only ever OCR'd once -- copy the data/ folder to another machine and it
won't redo the work.
"""
from __future__ import annotations

import logging

import config

log = logging.getLogger("ocr")

_engine = None          # (name, callable(image_ndarray) -> list[(box, text, conf)])
_engine_checked = False


def _load_easyocr(gpu: bool):
    import easyocr

    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def run(img):
        return [(box, text, float(conf)) for box, text, conf in reader.readtext(img)]

    return run


def _load_rapidocr():
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()

    def run(img):
        result, _ = engine(img)
        return [(box, text, float(conf)) for box, text, conf in (result or [])]

    return run


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def get_engine():
    """(name, run) for the best available OCR engine, or None."""
    global _engine, _engine_checked
    if _engine_checked:
        return _engine
    _engine_checked = True
    wanted = (config.OCR_ENGINE or "auto").lower()
    if wanted == "none":
        return None
    if wanted == "auto":
        order = (["easyocr-gpu"] if _has_cuda() else []) + ["rapidocr", "easyocr-cpu"]
    else:
        order = {"easyocr": ["easyocr-gpu" if _has_cuda() else "easyocr-cpu"], "rapidocr": ["rapidocr"]}.get(wanted, [])
    for name in order:
        try:
            run = _load_easyocr(name == "easyocr-gpu") if name.startswith("easyocr") else _load_rapidocr()
            _engine = (name, run)
            log.info("OCR engine: %s", name)
            return _engine
        except Exception as exc:  # not installed / broken install
            log.debug("OCR engine %s unavailable: %s", name, exc)
    log.warning(
        "No OCR engine installed: scanned PDF pages will be skipped. "
        "Install one with: uv sync --extra ocr"
    )
    return None


def available() -> bool:
    return get_engine() is not None


def boxes_to_text(results, min_conf: float | None = None) -> str:
    """Turn OCR boxes into reading-order lines.

    Boxes whose vertical centres are close are one line; lines go top to
    bottom, words left to right. Low-confidence boxes (usually Hindi text
    read by an English model, logos, stamps) are dropped.
    """
    min_conf = config.OCR_MIN_CONFIDENCE if min_conf is None else min_conf
    items = []
    for box, text, conf in results:
        text = (text or "").strip()
        if conf < min_conf or sum(ch.isalnum() for ch in text) < 1:
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append(((min(ys) + max(ys)) / 2, max(ys) - min(ys), min(xs), text))
    items.sort(key=lambda it: (it[0], it[2]))
    lines: list[list] = []
    for cy, h, x, text in items:
        if lines and abs(cy - lines[-1][0]) <= max(h, lines[-1][1]) * 0.5:
            lines[-1][2].append((x, text))
        else:
            lines.append([cy, h, [(x, text)]])
    return "\n".join(" ".join(t for _, t in sorted(words)) for _, _, words in lines)


def ocr_pdf_pages(path, page_numbers: list[int]) -> dict[int, str]:
    """{page_number: text} for the given (0-based) pages of a PDF."""
    engine = get_engine()
    if engine is None or not page_numbers:
        return {}
    import pypdfium2 as pdfium

    _name, run = engine
    out: dict[int, str] = {}
    pdf = pdfium.PdfDocument(str(path))
    try:
        for i in page_numbers:
            try:
                img = pdf[i].render(scale=config.OCR_RENDER_SCALE).to_numpy()
                out[i] = boxes_to_text(run(img))
            except Exception as exc:
                log.warning("OCR failed on %s page %d: %s", path, i + 1, exc)
    finally:
        pdf.close()
    return out
