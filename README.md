# MITS IT Department Scraper + RAG Pipeline

Scrapes **only** the IT department section of the MITS Gwalior website
(`https://web.mitsgwalior.in/academics/departments/information-technology`),
stores every page and every referenced file (PDF, images, audio, video),
then embeds the extracted content and serves it through a retrieval-augmented
generation (RAG) query pipeline.

## Scope guarantees

- **Domain-locked**: the crawler only ever contacts `web.mitsgwalior.in`
  (see `config.ALLOWED_DOMAIN`). No other website is ever fetched.
- **Path-locked crawling**: only pages under
  `/academics/departments/information-technology` are followed for further
  links (`config.ALLOWED_PATH_PREFIX`). Links elsewhere on the same domain,
  or to other domains entirely, are recorded as skipped and never queued.
- **Assets can live elsewhere on the same domain** (e.g. `/uploads/...`):
  PDFs/images/audio/video referenced *from* an in-scope page are downloaded
  as long as they stay on `web.mitsgwalior.in`, but they never cause further
  crawling.
- **No looping**: every URL is marked visited *before* it is fetched and is
  never re-queued, so cyclic links (A → B → A) or duplicate references
  cannot cause repeated visits or infinite loops. A `MAX_PAGES` safety cap
  is also enforced. This is covered by `tests/test_crawler.py`, which
  simulates a link cycle and asserts every URL is requested exactly once.
- `robots.txt` is respected by default (`config.RESPECT_ROBOTS_TXT`).

## Pipeline stages

1. **Crawl** (`scraper/crawler.py`) — BFS crawl within scope; saves raw HTML
   to `data/raw/pages/`, binary assets to `data/raw/assets/`, and an overall
   `data/manifest.json` describing what was collected and where each item
   came from.
2. **Ingest** (`ingestion/extract.py`) — converts HTML pages to clean text
   (stripping nav/script/style/footer), extracts text from PDFs, and emits a
   short descriptive record for images/audio/video (filename + referring
   page + alt text) so they remain searchable and citable even though they
   have no text body. Everything is chunked (~1000 chars, 150 overlap) into
   `data/processed/chunks.jsonl`.
3. **Embed & index** (`rag/embed_index.py`) — embeds every chunk locally
   with `sentence-transformers/all-MiniLM-L6-v2` (no API key required) and
   builds a FAISS cosine-similarity index in `data/index/`.
4. **Query** (`rag/pipeline.py`) — embeds the question, retrieves the top-k
   chunks, and hands them to a pluggable generation backend that answers
   using only that retrieved context, citing sources.

## Web interface

`web/app.py` is a small Flask server with two pages:

- **Student chat** (`/`) -- WhatsApp-style chat window. Every answer shows
  the source page/document it came from; if no LLM is configured it shows
  the best-matching passages instead.
- **Admin dashboard** (`/admin`) -- list of everything indexed, drag-and-drop
  upload of extra PDF/DOCX/TXT documents (e.g. an exam schedule that isn't
  on the website), removing uploads, and usage stats (questions asked,
  average response time, most-referenced sources, recent questions).
  Uploads and removals need the admin password (`ADMIN_PASSWORD` env var,
  default `mits-admin`) and trigger an index rebuild in the background.

Every question is logged to `data/chat_log.db` (SQLite).

```bash
python run_pipeline.py   # once, to build the index
python -m web.app        # http://localhost:5000  (admin: /admin)
```

## Usage

### With uv (recommended)

[uv](https://docs.astral.sh/uv/) manages the virtualenv and dependencies
from `pyproject.toml`/`uv.lock` for you -- no manual `venv`/`pip` steps.

```bash
uv sync                        # core deps only
uv sync --extra anthropic      # + Anthropic SDK
uv sync --extra openai         # + OpenAI SDK (also covers OpenAI-compatible local servers)
uv sync --extra gemini         # + Gemini SDK
uv sync --all-extras           # all three generation backends
uv sync --group dev            # add pytest for running the test suite

# One-shot: crawl -> ingest -> embed & index
uv run run_pipeline.py

# Ask a question
uv run -m rag.pipeline "What programs does the IT department offer?"

# Start the web UI (http://localhost:5000)
uv run -m web.app

# Run tests
uv run pytest tests/ -q
```

`uv run` transparently uses the `.venv` uv manages, so there's nothing to
activate manually. Add a new dependency with `uv add <package>` (or
`uv add --optional openai <package>` for an extras group) -- it updates
both `pyproject.toml` and `uv.lock`.

### With pip

```bash
pip install -r requirements.txt   # or just the SDK for the backend you use, see below

# One-shot: crawl -> ingest -> embed & index
python run_pipeline.py

# Ask a question
python -m rag.pipeline "What programs does the IT department offer?"
```

### Choosing a generation backend

Set exactly one of these env vars (or run a local Ollama server) and the
pipeline auto-detects it, in this priority order: Anthropic → OpenAI →
Gemini → Ollama. No env var and no local Ollama running means
retrieval-only mode: it prints the raw matched chunks instead of a
generated answer.

| Backend | Setup |
|---|---|
| Anthropic (Claude) | `export ANTHROPIC_API_KEY=sk-ant-...` |
| OpenAI | `export OPENAI_API_KEY=sk-...` (override model with `OPENAI_MODEL`) |
| Local model via an OpenAI-compatible server (LM Studio, llama.cpp, vLLM) | `export OPENAI_API_KEY=anything OPENAI_BASE_URL=http://localhost:1234/v1 OPENAI_MODEL=<your-local-model-name>` |
| Gemini | `export GEMINI_API_KEY=...` (or `GOOGLE_API_KEY`; override model with `GEMINI_MODEL`) |
| Local model via Ollama | `ollama pull llama3.1 && ollama serve` (defaults to `http://localhost:11434`, model `llama3.1` — override with `OLLAMA_BASE_URL` / `OLLAMA_MODEL`) |
| $env:key = "value"     | for PowerShell |

To force a specific backend instead of auto-detecting, set
`RAG_GENERATION_PROVIDER` to `anthropic`, `openai`, `gemini`, `ollama`, or
`none`. You only need to `pip install` the SDK for the backend you
actually use (`anthropic`, `openai`, or `google-generativeai` — Ollama
needs none beyond `requests`).

Each stage can also be run independently:

```bash
python -m scraper.crawler        # crawl only
python -m ingestion.extract      # (re)build chunks.jsonl from an existing manifest
python -m rag.embed_index        # (re)build the FAISS index from chunks.jsonl
```

## Configuration

All scope/behavior knobs live in `config.py`: allowed domain/path prefix,
crawl politeness delay, page cap, chunk size/overlap, embedding model name,
and the Claude model used for generation.

## Tests

```bash
python -m pytest tests/ -q
```

`tests/test_crawler.py` uses an in-memory fake site (no live network calls)
to verify scope enforcement, single-visit guarantees, and asset
deduplication. `tests/test_ingestion.py` covers HTML text extraction,
chunking and uploaded documents. `tests/test_web.py` covers the web API
with the model stubbed out.

## Data layout

```
data/
  manifest.json              # what was crawled/downloaded and from where
  raw/
    pages/                   # saved HTML, one file per crawled page
    assets/                  # downloaded PDFs, images, audio, video
  processed/
    chunks.jsonl             # extracted + chunked text ready for embedding
  uploads/                   # documents uploaded from the admin page
  chat_log.db                # every question asked via the web UI
  index/
    faiss.index              # vector index
    chunks_meta.jsonl        # chunk metadata aligned with the index
```

`data/` is gitignored (see `.gitignore`) since scraped content is
regenerated by `run_pipeline.py` and can include large binary files;
only `data/.gitkeep` is tracked so the directory exists on checkout.

## Note on this environment

This code was written and unit-tested (crawler scoping/dedup logic and
text chunking) in a sandboxed session whose network egress policy blocks
both `web.mitsgwalior.in` and the Hugging Face model hub. It has **not**
been run end-to-end against the live site here. Run `python run_pipeline.py`
from an environment with normal internet access to perform the actual
crawl and build the index.
