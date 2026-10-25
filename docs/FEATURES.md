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
1. **Keyword search** (BM25) with synonyms: HOD → head of department, head, dean; "5th sem" → 5, fifth, V (documents write "Semester-V").
2. **Meaning search** (vectors).
3. **Merge** both lists.
4. **Reranker**: a small model re-reads question + chunk and scores the fit (blended with steps 1–2 so it can't overrule a strong match).
5. **Newer wins ties**: a small bonus for recent text.
6. **Role questions** ("who is the HOD / dean / coordinator", "abhi", "current"): the newest documents that name a person in that role are always included, one per document.
7. **No repeats**: duplicate text is dropped; max 2 chunks from one document.

**Where:** `rag/retrieval.py`. See what each step did: `python scripts/debug_retrieval.py "your question"`.

## 6. Date-aware answers (stops "temporal hallucination")
**Why:** an AI happily states an old fact as current if nothing tells it otherwise.
**What:** the AI is told today's date, sees every source with its date, newest first, and must follow these rules: newest source = current; older ones only as "previously"; a post held "from X to Y" is over after Y; say "as of <month year>"; if the answer isn't in the sources, say so instead of guessing; reply in the student's language (English / Hinglish / Hindi).
For role questions, code first builds a **timeline** of every named person + title found in the sources, newest first ("Sep 2026: Dr. Punit Kumar Johari, Dean, SOIT [1]"), and puts it at the top of the prompt, so the AI doesn't have to work out the order itself. Names are paired with the *nearest* title, and names inside timetable rows are ignored.
**Where:** `SYSTEM_PROMPT` and `build_prompt()` in `rag/pipeline.py`, `role_holders()` in `rag/retrieval.py`. See the exact AI input: `python -m rag.pipeline --show-prompt "your question"`.

## 7. Any AI model, with automatic backup
**Why:** free APIs have small limits; a demo must not die because one says "too many requests".
**What:** OpenRouter, Groq, GitHub Models, Gemini, Anthropic, OpenAI, LM Studio, Ollama. If one fails or hits its limit, the next one answers. If all fail, the chat shows the matching passages instead.
**How:** they all follow one small interface; most speak the same "OpenAI format", so they share one class. No SDKs to install.
**Where:** `rag/llm/`. Choose with `LLM_PROVIDERS=openrouter,groq,ollama` + the API key variables (see README).

## 8. Fast rebuilds
**Why:** every admin upload rebuilds the index; reading ~260 PDFs took ~10 minutes.
**What:** the text of each PDF is saved after the first read; only new or changed PDFs are read again. Rebuild now takes under a minute.
**Where:** `pdf_to_text()` in `ingestion/extract.py` (cache in `data/processed/pdf_text_cache/`).

## 9. Chat page with clickable sources
**Why:** students (and examiners) should see *where* every answer comes from.
**What:** a chatbot page. Numbers like [1] in an answer are clickable; source cards sit under each answer; a Sources panel (beside the chat on a laptop, sliding up on a phone) shows each source's name, type, date and the exact passage used, with **Open original** (college site / Google Drive) and **Saved copy** (our downloaded file, which works even if the site is down and is the only link for admin uploads).
**Where:** `web/static/` (page), `/files/...` in `web/app.py` (saved copies; only `data/raw` and `data/uploads` can be served). Check every link: `uv run scripts/check_links.py`. Result on 26 Oct 2026: 377/377 saved copies and 377/377 original links open.

## 10. Nothing optional can crash it
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
| Check the crawl got everything | `uv run scripts/check_crawl.py --live` |
| See why a chunk was / wasn't found | `uv run scripts/debug_retrieval.py "question" "expected text"` |
| See exactly what the AI receives | `uv run -m rag.pipeline --show-prompt "question"` |
| Run tests | `uv run pytest tests -q` |

## Why not LlamaIndex?
LlamaIndex is a toolbox that contains these same parts (chunking, hybrid search, rerankers, model connectors). Switching would not fix anything by itself. The HOD bug was a *data* problem (no dates) plus a *prompt* problem, and that is fixed here. Our own code is a few hundred lines you can explain line by line in a viva. If you ever want it, it can replace `rag/retrieval.py` behind the same interface without touching the rest.

## Crawl check (25 Oct 2026, real data)
`check_crawl.py --live` result: 29 pages, 348 files saved; re-reading every saved page finds **0 links that were silently skipped**; today's department menu has 28 links and **all 28 are crawled**; only 2 failures (one >25 MB Drive file, one dead link). An earlier crawl had been stopped after 2 minutes; it overwrote 61 files but the complete manifest survived and all files were intact. Since then, stopping can no longer leave half-written files (tested in `tests/test_crawler_robustness.py`).

## Still to solve
0. **The local Ollama model (12B) is the weakest link.** With the right sources and timeline in front of it, it still sometimes calls a former HOD current. Use a larger model for the demo (`LLM_PROVIDERS=groq,openrouter,ollama` with a key), or a stronger local model.
1. **Scanned PDFs**: 93 of 263 PDFs are images with no text, so the bot can't read them. Needs OCR (e.g. Tesseract) during ingestion.
2. **Big Google Drive files** (>25 MB) are skipped; 1 right now.
3. **Re-crawl on a schedule** so new notices appear without anyone running a command.
4. **Follow-up questions**: each question is answered alone ("and who was before him?" won't work yet).
5. **A fixed test set**: grow `scripts/ask_batch.py` into ~30 questions with expected answers to measure accuracy after every change.
6. **Admin password**: default is `mits-admin`; set `ADMIN_PASSWORD` before any demo on a shared network.
