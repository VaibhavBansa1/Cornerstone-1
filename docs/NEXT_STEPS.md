# Next steps (state on 3 Nov 2026)

The project works and is finished for this semester. This page is the hand-over for whoever continues it (a person, or an AI assistant: read `AGENTS.md` first).

**Where things stand:** 123 tests pass. `uv run scripts/eval.py --sources` finds the right timetable for 10 out of 10 wordings ("5th sem TT", "paanchve sem ka routine", "V semester class schedule", "2nd year ka timetable", "M.Tech 3rd semester"...). The data was last built before 3 Nov; nothing has been rebuilt since.

## 1. Timetable grid: read it cell by cell (most important)

**Problem.** A timetable PDF stores loose words at positions on the page; the grid lines are only drawing. Plain text extraction drops the empty cells, so a row reads `MON 16243103 SS1 16243110 TS ...`. The AI then lines the first subject up with the first column (09:00), and the whole row shifts. Example: the 5th sem timetable (July–Dec 2026). Monday's first class is at **11:00**, but the bot said 09:00.

**Checked:** `pypdf` keeps the columns with `page.extract_text(extraction_mode="layout")`. In that output, Monday's first subject sits under 11:00–12:00 and Tuesday's under 10:00–11:00. (Try it: `data/try_layout.py` on the owner's PC.)

**Plan:**
1. In `ingestion/readers.py`: for a PDF page with a timetable grid (a `DAY` row with period times like `09:00-10:00`, and weekday rows), take each word's x-position (pypdf's `visitor_text`, or the layout text) and put it under the nearest period column. Lab blocks span two periods.
2. Write the grid as plain lines the AI can't misread, followed by the course list:
   `MON 11:00-12:00: 16243103 SS1` · `MON 12:00-01:00: 16243110 TS` · ...
3. Tests: a small sample grid in `tests/` (empty cells, a two-period lab, a two-line cell).
4. Rebuild (`uv run run_pipeline.py --skip-crawl`). **Note:** the PDF cache is version 3 (`PDF_CACHE_VERSION`, partly-scanned OCR), and the current data was built with an older version. So the next rebuild re-reads and re-OCRs every PDF: about 2 h on a GPU, much longer on CPU. Ask the owner before using his GPU.
5. After the rebuild, any other copy (e.g. the friend's laptop) needs the new `data\index` + `data\processed`.

## 2. College hours from real classes

The timetable header runs 09:00–06:00, but the 09:00 column is a buffer (quizzes, tests). Classes are about 10:00 to 5:00/6:00. Once the grid is read cell by cell (step 1), give "college timings" the first and last periods that actually have classes in this session's timetables, not the header. Then remove the "09:00-10:00 ... 05:00-06:00" example from rule 6 of the prompt (`rag/pipeline.py`).

## 3. Re-check answers with a fresh Groq budget

On 3 Nov the day's free Groq budget was used up by testing. The last answers that day came from backup models, and most cited no sources. Run `uv run scripts/eval.py` (all questions) early in a day, and keep long runs to one per day (Groq free: 200,000 tokens/day, one answer ≈ 4,000).

## 4. Smaller things
- **Agent loop (optional):** if the sources don't fit the question (e.g. the wrong semester's timetable), let the AI search once more with a better query.
- **Admin password:** still the default `mits-admin`; set `ADMIN_PASSWORD` before any shared demo.
- **Public link:** ngrok gives a new URL on every restart; a claimed static ngrok domain would keep one link.
- Old items still open: link-diff re-crawls, Drive files over 25 MB, "this semester / my batch" (the bot doesn't know who is asking).

## Reusing the pipeline elsewhere
See [REUSE.md](REUSE.md): the scraper, ingestion and RAG stages are separate, and only part of the code is MITS-specific.
