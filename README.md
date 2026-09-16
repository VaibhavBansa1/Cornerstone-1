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

## Usage

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
deduplication. `tests/test_ingestion.py` covers HTML text extraction and
chunking.

## Data layout

```
data/
  manifest.json              # what was crawled/downloaded and from where
  raw/
    pages/                   # saved HTML, one file per crawled page
    assets/                  # downloaded PDFs, images, audio, video
  processed/
    chunks.jsonl             # extracted + chunked text ready for embedding
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
