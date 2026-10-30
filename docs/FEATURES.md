# How the MITS Help Desk works

A student asks a question → we find the right pieces of the college website → an AI writes the answer from only those pieces, with sources.

This page explains each feature in plain words: **why** it exists, **what** it does, and **where** the code is. For the diagrams, the full decision table and the folder map, see **[architecture/architecture.md](architecture/architecture.md)**. The tags **[D1]**–**[D14]** below match that table.

The features are in the order a document travels through the system: collected → read → named and dated → chunked → indexed, then question → understood → searched → answered → shown.

---

## Indexing (offline)

### 1. Crawler: collects the website [D1 D2]
**Why:** we only want the IT department, nothing else, plus every PDF it links to.
**What:** visits each department page once, saves it, and downloads linked PDFs and images (including Google Drive files).
**How:** starts at the department page and follows only department links (URL prefix + the department's sidebar menu). It remembers what it visited, so it never loops. Temporary errors (timeout, server busy) are retried once. Files are written to a temp file and then swapped in, so stopping the crawl never leaves a half-written file. A stopped crawl keeps the last complete `manifest.json` and saves its progress to `manifest.partial.json`.
**Where:** `scraper/crawler.py`. Check a crawl: `uv run scripts/check_crawl.py --live`.

### 2. Reading text, including scanned PDFs [D3]
**Why:** 1,737 of 4,896 PDF pages were scans (pictures of paper), e.g. 14 of 19 pages of the 2026 V SEM syllabus, so the bot couldn't read them.
**What:** web pages, PDFs and Word files become plain text. PDF pages with no text are read with OCR: EasyOCR on an NVIDIA GPU (~2 s/page), RapidOCR on a laptop CPU (~10 s/page). Unsure words (Hindi letterheads, stamps) are dropped, and photo-only PDFs are marked "no text" and not retried. Web pages are cleaned so "7<sup>th</sup>" stays "7th", and PDF letter-joins (ﬁ) are fixed.
**How it stays fast:** each PDF's text is saved by the file's *content*, so a rebuild, or a copied `data/` folder on another machine, never redoes OCR. A full rebuild takes about a minute instead of an hour.
**Where:** `ingestion/readers.py`, `ingestion/ocr.py` (cache in `data/processed/pdf_text_cache/`).

### 3. Naming "CLICK HERE" files [D4]
**Why:** the syllabus page links every file as "CLICK HERE" inside a table, so a file name like `5th 11_merged.pdf` says nothing about which batch it is for.
**What:** the file is named from its table row + column: **"V SEM · 2024-28 BATCH"**. Links with real text ("Time Table July 2026") keep that text.
**Where:** `ingestion/link_labels.py`.

### 4. A date on every piece of text [D5]
**Why:** the original bot mixed up the current HOD with old ones, because it had no idea how old each piece of text was.
**What:** every piece gets a date, using the best clue available: a full date in the text ("w.e.f. 27/07/2026", "Session July–Dec 2026") › the semester the file is linked as (V SEM of the 2024 batch starts July 2026) › the PDF's own date › the server's date › a bare year (last, because it's often a book's publication year).
**How:** pattern matching for the date formats colleges use (day-first, Indian style). Deadlines far in the future ("valid till 2027") are ignored.
**Where:** `ingestion/dates.py`, `make_chunks()` in `ingestion/chunking.py`.

### 5. Chunks and course lists [D6]
**Why:** the AI can only read a few pieces at a time, so documents are cut up. But "what subjects are in 5th sem?" needs the *whole* list, which a few pieces of a 19-page syllabus don't give.
**What:** pieces of ~900 characters, cut at paragraph or sentence ends (never mid-word), with a little overlap so a fact on the edge isn't lost. Every syllabus or timetable also gets **one extra piece listing all its courses with codes**. Courses are found by layout (course name, then its code), not by code format, because codes changed when MITS became a Deemed University (old `3160512`, new `16243101`) and may change again.
**Where:** `ingestion/chunking.py`, `ingestion/course_list.py`. `ingestion/build.py` runs steps 2–5 for every crawled file and upload.

### 6. Embeddings (meaning search), GPU or CPU [D7]
**Why:** finds text with the same meaning even when the words differ ("who leads the department" ≈ "Professor & Head").
**What:** `multilingual-e5-small` turns each chunk into numbers and understands English, Hindi and Hinglish. It uses the NVIDIA GPU if there is one, otherwise the CPU. The index records which model built it and refuses to run with a different one, instead of giving wrong answers.
**Where:** `rag/embeddings.py`, `rag/embed_index.py`. Set with `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`.

---

## Answering (live)

### 7. Understanding the question [D8]
**Why:** documents are filed by batch, but students ask by semester or year ("5th sem syllabus 2026", "3rd year ka time table", "2025 students").
**What:** before searching, a small rule-based step works out semester, batch, session and topic, e.g. *"fifth semester syllabus 2026" → 5th sem · 2024-28 batch · July–Dec 2026 session · syllabus*. It understands Roman numerals, "fifth", Hinglish ("panchve sem"), "3rd year" and "2025 batch", and can tell when "2026" means the session year and when "2025" means the admission year. The chat shows the result as **"Understood as …"**.
**Not handled yet:** lateral entry, "this sem / next sem / my batch", typos, sections.
**Where:** `rag/query_understanding.py`. It's one function, `understand()`, to replace if you ever want an AI to do this step.

### 8. Search: finding the right pieces [D9 D10]
**Why:** meaning search alone misses exact names and short words like "HOD", and once ranked *last year's* syllabus first.
**What, in order:**
1. **Keyword search** (BM25) with synonyms (HOD → head of department, head, dean) and number forms ("5th sem" → 5, fifth, V). Words in a document's title count 3×.
2. **Meaning search** (vectors).
3. **Merge** both lists. Documents for the understood batch, and for role questions the newest documents naming someone in that role, are always included.
4. **Reranker**: a small model re-reads question + chunk and scores the fit, blended 50/50 with steps 1–2 so it can't overrule a strong match.
5. **Bonuses**: newer text up to +0.15, the right batch +0.30, a different batch −0.30.
6. **No repeats**: duplicate text is dropped; at most 2 pieces from one document; 6 pieces go to the AI.
7. **Topic pages**: a question about labs, contacts, programmes or the department in general gets that website page *whole*, as one source (`config.TOPIC_PAGES`), because a list like "all the labs" is only useful complete. "Lab" also matches "laboratories", "dept" matches "school", and "3rd year" is searched as "5th sem".

**Where:** `rag/retrieval.py`, `rag/roles.py`. See what each step did: `uv run scripts/debug_retrieval.py "your question"`.

### 9. Date-aware answers, which stop "temporal hallucination" [D11]
**Why:** an AI happily states an old fact as current if nothing tells it otherwise.
**What:** the AI is told today's date and how the question was understood. It sees every source with its date, newest first, and must follow these rules:
- the newest source is current; older ones only as "previously"
- a post held "from X to Y" is over after Y
- course codes are batch-specific
- if the answer isn't in the sources, say so
- reply in the student's language (English / Hinglish / Hindi)
- only IT-website facts: for the mess, canteen, fees, directions, other departments… say "I don't have that" and point to mitsgwalior.in (a 2019 magazine joke about mess food is not proof there is a mess)
- never read out students' names or roll numbers from debarred / marks / attendance lists; point to the notice instead
- "lab" can mean a room or a practical subject: list the labs and say lab courses are in each syllabus

For role questions, the code first builds a **timeline** of every named person + title in the sources, newest first ("Sep 2026: Dr. Punit Kumar Johari, Dean, SOIT [1]"), and puts it at the top. Names are paired with the *nearest* title, and names inside timetable rows are ignored.
**Where:** `SYSTEM_PROMPT` and `build_prompt()` in `rag/pipeline.py`, `rag/roles.py`. See the exact AI input: `uv run -m rag.pipeline --show-prompt "your question"`.

### 10. Any AI model, with automatic backup [D12]
**Why:** free APIs have small limits; a demo must not die because one says "too many requests".
**What:** OpenRouter, Groq, GitHub Models, Gemini, Anthropic, OpenAI, LM Studio and Ollama. If one fails or hits its limit, the next one answers (a rate-limited one is skipped for 60 s). If all fail, the chat shows the matching passages instead.
**How:** they all follow one small interface and are called over plain HTTP, so there are no SDKs to install.
**Where:** `rag/llm/`. Choose with `LLM_PROVIDERS=groq,openrouter,ollama` + the API key (see README).

### 11. Chat page with clickable sources [D13]
**Why:** students (and examiners) should see *where* every answer comes from.
**What:** a chatbot page showing "Understood as …", the answer with clickable numbers like [1], and source cards. The Sources panel (beside the chat on a laptop, sliding up on a phone) shows each source's name, type, date and the exact passage used, with:
- **Open original** (college site / Google Drive)
- **Saved copy** (our downloaded file; works even if the site is down, and is the only link for admin uploads)

Checked on 26 Oct 2026: 377/377 saved copies and 377/377 original links open.
While it works, the chat shows **live progress** instead of a spinner: "Reading your question → Understood as 5th sem · 2024-28 batch", "Searching 9,348 pieces of the IT website", "Found 6 matching sources → Reading: V SEM · 2024-28 BATCH …", "Writing the answer with groq · 4 s" (and "Previous AI was busy · now asking openrouter" when it falls back). The server sends these as they happen (`/api/query/stream`).
**Where:** `web/static/` (pages), `web/sources.py` (names, links, saved pages), `web/app.py` (URLs). Check every link: `uv run scripts/check_links.py`.

### 12. Admin page and privacy [D13 D14]
**What:** upload a PDF/DOCX/TXT and it goes through the same steps 2–6 automatically, in the background. You can also see what's indexed and which sources are cited most.
**Privacy:** question and answer text is **never stored**, only counts and cited sources. Saved copies are served only from `data/raw/` and `data/uploads/`, never the index or the log.
**Where:** `web/app.py`, `web/chatlog.py`, `web/static/admin.*`.

### 13. 55 test questions, graded automatically
**Why:** "it seems to work" isn't a measurement. Students ask weird things.
**What:** `scripts/eval_questions.json` has 55 questions in 9 groups: labs (rooms vs practicals), School vs department, schemes, things not on the IT site (mess, canteen, fees, directions, other departments, IPL), "where is …", mandatory tasks, people and contacts, syllabus / timetable / notices, and vague or misspelt questions. Each says what a correct answer must contain (checked against the real data) or that the bot must say it doesn't know. `scripts/eval.py` asks them all and writes `data/eval_report.md` with every answer and a score per group. The grading is keyword matching, so read the answers too.
**Result (30 Oct 2026, local Ollama model `huihui-gemma`, ~9 s per answer):**

| Group (questions) | Before | After |
|---|---|---|
| Labs: rooms vs practicals (6) | 4 | 6 |
| School / department name (6) | 6 | 6 |
| Schemes (6) | 4 | 5 |
| Not on the IT website: mess, canteen, fees, directions… (11) | 7 | 11 |
| Where is … (5) | 5 | 5 |
| Mandatory tasks (6) | 5 | 6 |
| People and contacts (5) | 4 | 5 |
| Syllabus, timetable, notices (6) | 5 | 6 |
| Vague / typos (4) | 3 | 4 |
| **Total (55)** | **43** | **54** |

The one left ("government scholarships?") passes on some runs and not others: the small local model sometimes opens with "Yes" when the source only says "check the website for scholarship updates". A larger model (Groq / OpenRouter) should do better; re-run `scripts/eval.py` with it before the demo.
**Where:** `scripts/eval.py`, `scripts/eval_questions.json`. Run: `uv run scripts/eval.py` (about 9 minutes on the GPU PC).

### 14. Nothing optional can crash it
Reranker missing → search continues without it. AI down → next AI, or passages. OCR not installed → scanned pages skipped with a warning. Crawl stopped → old data kept. Each case has a test in `tests/`.

---

## Everyday commands

| Do this | Command |
|---|---|
| Install on a laptop (CPU) | `uv sync --extra cpu --extra ocr` |
| Install on the GPU PC | `uv sync --extra cuda --extra ocr` |
| Then, in every new terminal (keeps that PyTorch build) | PowerShell `$env:UV_NO_SYNC = "1"` · bash `export UV_NO_SYNC=1` |
| Crawl again + rebuild | `uv run run_pipeline.py` |
| Rebuild only (after changing settings) | `uv run run_pipeline.py --skip-crawl` |
| Ask from the terminal | `uv run -m rag.pipeline "Who is the HOD?"` |
| Start the website | `uv run -m web.app` → http://localhost:5000 |
| Grade answers on the 55 test questions | `uv run scripts/eval.py` → `data/eval_report.md` |
| Answer your own list of questions | `uv run scripts/ask_batch.py questions.txt` → `data/eval_answers.md` |
| Check the crawl got everything | `uv run scripts/check_crawl.py --live` |
| See why a piece was / wasn't found | `uv run scripts/debug_retrieval.py "question" "expected text"` |
| See exactly what the AI receives | `uv run -m rag.pipeline --show-prompt "question"` |
| Run tests | `uv run pytest tests -q` |

## Why not LlamaIndex?
LlamaIndex is a toolbox that contains these same parts (chunking, hybrid search, rerankers, model connectors). Switching would not fix anything by itself. The HOD bug was a *data* problem (no dates) plus a *prompt* problem, and that is fixed here. Our own code is small enough to explain line by line in a viva. If you ever want it, it can replace `rag/retrieval.py` behind the same interface without touching the rest.

## Crawl check (25 Oct 2026, real data)
`check_crawl.py --live` found:
- 29 pages and 348 files saved
- re-reading every saved page: **0 links silently skipped**
- the department menu has 28 links today, and **all 28 are crawled**
- only 2 failures: one Drive file over 25 MB, and one dead link

An earlier crawl had been stopped after 2 minutes; the complete manifest survived and all files were intact. Since then, stopping can no longer leave half-written files (tested in `tests/test_crawler_robustness.py`).

## Still to solve
1. **The local Ollama model (12B) is the weakest link.** With the right sources in front of it, it still sometimes calls a former HOD current. For the demo, use a larger model: `LLM_PROVIDERS=groq,openrouter,ollama` with a key.
2. **Re-crawling costs time.** Re-checking only the ~30 department pages and downloading just new links (a "link diff") would make daily updates nearly free. The OCR cache already makes unchanged PDFs free.
3. **Big Google Drive files** (over 25 MB) are skipped; 1 right now.
4. **Follow-up questions:** each question is answered alone ("and who was before him?" won't work yet).
5. **"This semester / my batch":** the bot doesn't know who is asking, so "this semester" gets a general answer. Asking "5th sem" or "2024 batch" works.
6. **Admin password:** the default is `mits-admin`; set `ADMIN_PASSWORD` before any demo on a shared network.
