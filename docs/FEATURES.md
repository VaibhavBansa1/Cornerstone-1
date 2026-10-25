# How the MITS Help Desk works

A student asks a question → we find the right pieces of the college website → an AI writes the answer from only those pieces, with sources.

```
crawl the site ─► clean + date the text ─► cut into chunks ─► turn into vectors
                                                                   │
student question ─► find best chunks (keywords + meaning + dates) ─┘
                 ─► AI writes answer from them ─► answer + sources in chat
```

Each box is a separate part. You can change one (a new AI model, a new embedding model) without touching the others.

---

## 1. Crawler — collects the website
**Why:** we only want the IT department, nothing else, and every PDF it links to.
**What:** visits each department page once, saves it, downloads linked PDFs/images (also Google Drive files).
**How:** starts at the department page, follows only department links, remembers what it visited so it never loops. Temporary errors (timeout, server busy) are retried once. Files are written to a temp file first and then swapped in, so stopping the crawl never leaves a half-written file. If you stop it, the last complete `manifest.json` is kept and this run's progress goes to `manifest.partial.json`. The next run simply starts fresh.
**Where:** `scraper/crawler.py`. Check a crawl with `python scripts/check_crawl.py --live`.

## 2. Clean text + dates
**Why:** the old bot mixed up the current HOD with old ones, because it had no idea how old each piece of text was.
**What:** every piece of text gets a date: the newest date written in it ("w.e.f. 27/07/2026", "Session July–Dec 2026", "10th March 2025", "2021-22").
**How:** pattern matching for the date formats colleges use (day-first, like India). Deadlines far in the future ("valid till 2027") are ignored. Web pages are cleaned so "7<sup>th</sup>" stays "7th", and PDF letter-joins (ﬁ) are fixed.
**Where:** `ingestion/dates.py`, `ingestion/extract.py`.

## 3. Chunks
**Why:** the AI can only read a few pieces at once, so documents are cut up.
**What:** pieces of ~900 characters, cut at paragraph or sentence ends (never mid-word), with a little overlap so a fact on the edge isn't lost.
**Where:** `chunk_text()` in `ingestion/extract.py`.

## 4. Embeddings (meaning search) — GPU or CPU
**Why:** finds text with the same meaning even when the words differ ("who leads the department" ≈ "Professor & Head").
**What:** `multilingual-e5-small` turns each chunk into numbers; it understands English, Hindi and Hinglish.
**How:** uses the NVIDIA GPU if there is one, otherwise the CPU, automatically. The index remembers which model built it and refuses to run with a different one (instead of giving wrong answers).
**Where:** `rag/embeddings.py`, `rag/embed_index.py`. Set with `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`.

## 5. Search — finding the right chunks
**Why:** meaning search alone misses exact names and short words like "HOD".
**What / How**, in order:
1. **Keyword search** (BM25) with synonyms: HOD → head of department, head, dean.
2. **Meaning search** (vectors).
3. **Merge** both lists.
4. **Reranker**: a small model re-reads question + chunk and scores the fit (blended with steps 1–2 so it can't overrule a strong match).
5. **Newer wins ties**: a small bonus for recent text.
6. **Role questions** ("who is the HOD / dean / coordinator", "abhi", "current"): the newest documents that name a person in that role are always included.
7. **No repeats**: duplicate text is dropped; max 2 chunks from one document.

**Where:** `rag/retrieval.py`. See what each step did: `python scripts/debug_retrieval.py "your question"`.

## 6. Date-aware answers (stops "temporal hallucination")
**Why:** an AI happily states an old fact as current if nothing tells it otherwise.
**What:** the AI is told today's date, sees every source with its date, newest first, and must follow these rules: newest source = current; older ones only as "previously"; say "as of <month year>"; if the answer isn't in the sources, say so instead of guessing.
**Where:** `SYSTEM_PROMPT` in `rag/pipeline.py`.

## 7. Any AI model, with automatic backup
**Why:** free APIs have small limits; a demo must not die because one says "too many requests".
**What:** OpenRouter, Groq, GitHub Models, Gemini, Anthropic, OpenAI, LM Studio, Ollama. If one fails or hits its limit, the next one answers. If all fail, the chat shows the matching passages instead.
**How:** they all follow one small interface; most speak the same "OpenAI format", so they share one class. No SDKs to install.
**Where:** `rag/llm/`. Choose with `LLM_PROVIDERS=openrouter,groq,ollama` + the API key variables (see README).

## 8. Fast rebuilds
**Why:** every admin upload rebuilds the index; reading ~260 PDFs took ~10 minutes.
**What:** the text of each PDF is saved after the first read; only new or changed PDFs are read again. Rebuild now takes under a minute.
**Where:** `pdf_to_text()` in `ingestion/extract.py` (cache in `data/processed/pdf_text_cache/`).

## 9. Nothing optional can crash it
Reranker missing → search continues without it. AI down → next AI, or passages. Crawl stopped → old data kept. Each of these has a test in `tests/`.

---

## Everyday commands
| Do this | Command |
|---|---|
| Install (laptop / GPU PC) | `uv sync` / `uv sync --extra cuda` |
| Crawl again + rebuild | `uv run run_pipeline.py` |
| Rebuild only (after changing settings) | `uv run run_pipeline.py --skip-crawl` |
| Ask from the terminal | `uv run -m rag.pipeline "Who is the HOD?"` |
| Start the website | `uv run -m web.app` → http://localhost:5000 |
| Check answer quality on test questions | `uv run scripts/ask_batch.py` → `data/eval_answers.md` |
| Run tests | `uv run pytest tests -q` |

## Why not LlamaIndex?
LlamaIndex is a toolbox that contains these same parts (chunking, hybrid search, rerankers, model connectors). Switching would not fix anything by itself. The HOD bug was a *data* problem (no dates) plus a *prompt* problem, and that is fixed here. Our own code is a few hundred lines you can explain line by line in a viva. If you ever want it, it can replace `rag/retrieval.py` behind the same interface without touching the rest.

## Still to solve
1. **Scanned PDFs**: 93 of 263 PDFs are images with no text, so the bot can't read them. Needs OCR (e.g. Tesseract) during ingestion.
2. **Big Google Drive files** (>25 MB) are skipped; 1 right now.
3. **Re-crawl on a schedule** so new notices appear without anyone running a command.
4. **Follow-up questions**: each question is answered alone ("and who was before him?" won't work yet).
5. **A fixed test set**: grow `scripts/ask_batch.py` into ~30 questions with expected answers to measure accuracy after every change.
6. **Admin password**: default is `mits-admin`; set `ADMIN_PASSWORD` before any demo on a shared network.
