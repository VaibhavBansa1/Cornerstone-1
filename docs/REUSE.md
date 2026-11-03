# Reusing the pipeline for another website or your own files

The project is three separate stages. Each one only reads the files the stage before it wrote, so any stage can be swapped.

```
scraper/      crawl a website        -> data/raw/ + data/manifest.json
ingestion/    read, date, chunk      -> data/processed/chunks.jsonl
rag/          index, search, answer  -> data/index/  (+ web/ for the chat page)
```

## Three ways to reuse it

**1. Your own files only (no website).**
Put PDFs, DOCX or TXT files in `data/uploads/` (or upload them on the admin page), then:
```
uv run run_pipeline.py --skip-crawl
```
With no crawl manifest, only the uploads are indexed. Scanned PDFs are OCR'd.

**2. Another website.**
In `config.py`, change `ALLOWED_DOMAIN`, `ALLOWED_PATH_PREFIX` (the part of the site to follow), `COLLEGE_PAGES` (extra single pages, or empty), `DEPARTMENT_NAV_ARIA_LABEL` and `USER_AGENT`. Then `uv run run_pipeline.py`. The crawler stays on that domain and respects `robots.txt`.

**3. Your own scraper.**
Write `data/manifest.json` in the same shape and save the files under `data/`:
```json
{"pages":  [{"url": "...", "title": "...", "local_path": "raw/pages/x.html", "fetched_at": "...", "last_modified": "..."}],
 "assets": [{"url": "...", "local_path": "raw/assets/y.pdf", "extension": ".pdf",
             "referenced_from": "<page url>", "original_filename": "y.pdf"}]}
```
Then `uv run run_pipeline.py --skip-crawl`.

## What works anywhere, and what is MITS-specific

| Works for any site or documents | MITS-specific: change for a new place |
|---|---|
| Reading HTML / PDF / DOCX, OCR for scans (`ingestion/readers.py`, `ocr.py`) | The answer rules in `SYSTEM_PROMPT` (`rag/pipeline.py`): HOD = Dean, School of IT, etc. |
| Finding a date for every chunk, chunking (`ingestion/dates.py`, `chunking.py`) | Semester / batch / session calendar (`rag/query_understanding.py`) |
| Hybrid search: BM25 + embeddings + re-ranker, newest first (`rag/retrieval.py`) | Document labels: B.Tech/M.Tech, "Semester-V", sessions (`rag/doc_labels.py`) |
| Whole-document mode, follow-up questions, AI question reading | `TOPIC_PAGES`, `FACULTY_PAGE_TITLE`, `QUERY_SYNONYMS` in `config.py` |
| AI providers with automatic fallback (`rag/llm/`) | Course-code patterns (`ingestion/course_list.py`) |
| Chat + admin pages, 👍/👎 feedback, no question text stored (`web/`) | Test questions (`scripts/eval_questions.json`) |

For a new place: keep the left column as it is, rewrite the right column, and write new test questions. Run `uv run scripts/eval.py --sources` to check that search finds the right documents before checking the answers.
