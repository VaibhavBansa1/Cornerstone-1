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
2. **Ingest** (`ingestion/extract.py`, `ingestion/dates.py`) -- converts
   HTML pages to clean text (inline tags no longer split words), extracts
   PDF text (ligatures fixed), and emits a short record for images/audio/
   video. Text is split on paragraph/sentence boundaries (~900 chars, 150
   overlap) into `data/processed/chunks.jsonl`. **Every chunk gets a date**:
   the newest date mentioned in it ("w.e.f. 27/07/2026", "Session: July-Dec
   2026", "10th March 2025", "2021-22"), falling back to the document's
   newest date or the server's Last-Modified header.
3. **Embed & index** (`rag/embeddings.py`, `rag/embed_index.py`) -- embeds
   each chunk (title + date header + text) with
   `intfloat/multilingual-e5-small` (English/Hindi/Hinglish) on the GPU if
   PyTorch can see one, else the CPU, into a FAISS index. `index_info.json`
   records which model built it; a mismatch is reported instead of giving
   silently wrong results.
4. **Retrieve** (`rag/retrieval.py`) -- hybrid search: BM25 keyword search
   (with synonyms such as HOD -> head / dean) + vector search, merged with
   Reciprocal Rank Fusion, de-duplicated, re-scored by a cross-encoder
   reranker, plus a small **recency bonus** so the newest of several
   similar chunks wins.
5. **Answer** (`rag/pipeline.py`, `rag/llm/`) -- the chunks are shown to
   the LLM newest-first with their dates and today's date, with rules to
   treat the most recent source as current and older ones as "previously".
   The LLM comes from a **fallback chain** of providers.

### How the parts connect (each can be swapped without touching the others)

```
scraper/crawler.py ─► ingestion/extract.py ─► rag/embed_index.py
                         (+ dates.py)              (embeddings.py)
                                                        │
web/app.py ─► rag/pipeline.py ─► rag/retrieval.py ◄─────┘
                    │
                    └─► rag/llm/  registry.py ─► openrouter / groq / github /
                                                 gemini / anthropic / openai /
                                                 lm studio / ollama
```

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
uv sync                        # CPU (laptop)
uv sync --extra cuda           # NVIDIA GPU build of PyTorch
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
pip install requests beautifulsoup4 lxml pypdf sentence-transformers faiss-cpu numpy flask

# One-shot: crawl -> ingest -> embed & index
python run_pipeline.py

# Ask a question
python -m rag.pipeline "What programs does the IT department offer?"
```

### Choosing LLM providers (with automatic fallback)

Set one or more API keys. With `LLM_PROVIDERS=auto` (the default) every
provider whose key is set is tried in this order, then a local Ollama if
it is running: OpenRouter -> Groq -> GitHub Models -> Gemini -> Anthropic
-> OpenAI -> local OpenAI-compatible server -> Ollama. If one fails or is
rate limited, the next one answers (a rate-limited provider is skipped for
60 s). If none answers, the chat shows the matching passages instead.

Pin your own order, e.g. for a demo: `LLM_PROVIDERS=openrouter,groq,ollama`.
No SDKs are needed -- all providers are called over plain HTTP.

| Provider | Key env var | Model env var (default) |
|---|---|---|
| OpenRouter (free `:free` models) | `OPENROUTER_API_KEY` | `OPENROUTER_MODEL` (`google/gemma-4-31b-it:free`) |
| Groq (free tier) | `GROQ_API_KEY` | `GROQ_MODEL` (`llama-3.3-70b-versatile`) |
| GitHub Models (free with a GitHub token) | `GITHUB_MODELS_TOKEN` | `GITHUB_MODELS_MODEL` (`openai/gpt-4.1-mini`) |
| Gemini | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `GEMINI_MODEL` |
| Anthropic | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL` |
| OpenAI | `OPENAI_API_KEY` | `OPENAI_MODEL` |
| LM Studio / llama.cpp / vLLM | set `OPENAI_COMPAT_BASE_URL=http://localhost:1234/v1` | `OPENAI_COMPAT_MODEL` |
| Ollama (local, no key) | -- (`OLLAMA_BASE_URL`) | `OLLAMA_MODEL` (falls back to the first pulled model) |

Free model names change over time; check the provider's model list and
override the `*_MODEL` variable if a default disappears.

PowerShell: `$env:OPENROUTER_API_KEY = "sk-or-..."`  
bash: `export OPENROUTER_API_KEY=sk-or-...`

Adding a provider = one class in `rag/llm/` (or just a new entry in
`config.PROVIDER_SETTINGS` if it speaks the OpenAI format).

### GPU or CPU

`EMBEDDING_DEVICE=auto` (default) uses an NVIDIA GPU when PyTorch sees
one, otherwise the CPU. On a GPU machine install the CUDA build of
PyTorch with `uv sync --extra cuda`; a laptop just uses `uv sync`. An index
built on the GPU works on the CPU and vice versa, as long as the
embedding model is the same. On a slow laptop, `RERANKER_MODEL=none`
makes answers faster at some cost in accuracy.

### Rebuilding after changes

Changed the embedding model or chunking? Rebuild from the already
downloaded data without crawling again:

```bash
uv run run_pipeline.py --skip-crawl
```

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
