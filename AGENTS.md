# Notes for AI coding assistants

Read this first, then the docs it points to. Humans: see [HOW_TO_RUN.md](HOW_TO_RUN.md).

## What this is
MITS Help Desk: a chatbot for MITS Gwalior students. A crawler collects the IT department's pages and PDFs plus selected college-wide pages; the text is dated, chunked and indexed; questions are answered by hybrid search (BM25 + FAISS embeddings + re-ranker) and an LLM (Groq first, with backups), always citing sources.

## Where to read
| File | What it explains |
|---|---|
| `HOW_TO_RUN.md` | Running it on Windows with `run_demo.ps1`, updating, what is and isn't in git |
| `README.md` | Install, API keys, everyday commands, settings |
| `docs/architecture/architecture.md` | How it works, diagrams, every design decision (D1-D22) and why |
| `docs/FEATURES.md` | Each feature: why / what / where in the code; test results |
| `config.py` | Every setting (each can be overridden by an environment variable of the same name) |

## Code map
- `scraper/` crawl the website -> `data/raw/`
- `ingestion/` read HTML/PDF (OCR for scans), date and chunk -> `data/processed/chunks.jsonl`
- `rag/` embeddings + index (`embed_index.py`), search (`retrieval.py`), question understanding, follow-ups (`followup.py`), prompt + answer (`pipeline.py`), LLM providers (`llm/`)
- `web/` Flask app (`app.py`), chat and admin pages (`static/`), anonymous log + feedback (`chatlog.py`)
- `scripts/eval.py` + `eval_questions.json` graded test questions; `tests/` pytest
- `run_pipeline.py` crawl + build everything; `run_demo.ps1` one-command Windows setup and start

## Commands
- Start (Windows, any machine): `powershell -ExecutionPolicy Bypass -File .\run_demo.ps1`
- Tests: `.venv\Scripts\python.exe -m pytest tests -q` (or `uv run --no-sync pytest tests -q`)
- Ask from the terminal: `uv run --no-sync -m rag.pipeline "question"` (`--show-prompt` to see what the LLM gets)
- Rebuild the index without crawling: `uv run --no-sync run_pipeline.py --skip-crawl`
- Use `uv run --no-sync` (or set `UV_NO_SYNC=1`): plain `uv run` swaps the installed PyTorch build.

## Rules that must not be broken
- Never store question or answer text anywhere (logs, database, files). Feedback stores only the vote, cited sources, model and an optional comment.
- Never commit `.env` (API keys), `data/` contents, `models/`, `.venv/`.
- The bot answers only from the crawled sources and must say when they don't cover something; don't add facts to prompts or code.
- Don't add students' personal documents or name lists to the data.
- Keep docs short and plain; update `docs/FEATURES.md` and the decision table when behaviour changes.
- Run the tests after every change.
