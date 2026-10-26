# MITS IT Help Desk

A chatbot that answers questions about the **IT department of MITS Gwalior**, using only the department's own website and documents, and shows the source of every answer.

It crawls the IT department pages and every PDF they link to, reads them (scanned pages included), gives every piece of text a date, and answers questions with a hybrid search and an AI model of your choice.

- **How it works, with diagrams and every design decision:** [docs/architecture/architecture.md](docs/architecture/architecture.md)
- **Each feature in plain words (why / what / where):** [docs/FEATURES.md](docs/FEATURES.md)

![Architecture](docs/architecture/architecture.png)

## Quick start

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --extra cpu  --extra ocr    # laptop (CPU)
uv sync --extra cuda --extra ocr    # PC with an NVIDIA GPU
                                    # add --group dev to either line to get pytest

uv run run_pipeline.py              # crawl + build the index (once)
uv run -m web.app                   # http://localhost:5000   (admin: /admin)
```

Without uv: `pip install -r requirements.txt` (generated from `uv.lock`), then use `python` instead of `uv run`.

## Choose the AI model

Set one or more API keys. Every provider with a key is tried in order; if one fails or is rate-limited, the next one answers. If none answers, the chat shows the matching passages instead.

| Provider | Key | Model (default) |
|---|---|---|
| OpenRouter (free `:free` models) | `OPENROUTER_API_KEY` | `OPENROUTER_MODEL` (`google/gemma-4-31b-it:free`) |
| Groq (free tier) | `GROQ_API_KEY` | `GROQ_MODEL` (`llama-3.3-70b-versatile`) |
| GitHub Models | `GITHUB_MODELS_TOKEN` | `GITHUB_MODELS_MODEL` (`openai/gpt-4.1-mini`) |
| Gemini | `GEMINI_API_KEY` | `GEMINI_MODEL` |
| Anthropic | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL` |
| OpenAI | `OPENAI_API_KEY` | `OPENAI_MODEL` |
| LM Studio / llama.cpp / vLLM | `OPENAI_COMPAT_BASE_URL` | `OPENAI_COMPAT_MODEL` |
| Ollama (local, no key) | `OLLAMA_BASE_URL` | `OLLAMA_MODEL` |

Pick your own order: `LLM_PROVIDERS=groq,openrouter,ollama`.
PowerShell: `$env:GROQ_API_KEY = "gsk_..."` · bash: `export GROQ_API_KEY=gsk_...`

Free model names change over time; override the `*_MODEL` variable if a default disappears.

## Everyday commands

| Do this | Command |
|---|---|
| Crawl again + rebuild | `uv run run_pipeline.py` |
| Rebuild without crawling (after changing settings) | `uv run run_pipeline.py --skip-crawl` |
| Crawl only / chunks only / index only | `uv run -m scraper.crawler` · `uv run -m ingestion.build` · `uv run -m rag.embed_index` |
| Ask from the terminal | `uv run -m rag.pipeline "Who is the HOD?"` |
| See exactly what the AI receives | `uv run -m rag.pipeline --show-prompt "question"` |
| Run tests | `uv run pytest tests -q` |

More checking tools (crawl, links, retrieval debugging, batch answers) are listed in [docs/FEATURES.md](docs/FEATURES.md#everyday-commands).

## Settings

Every setting is in `config.py` and can be overridden with an environment variable of the same name, e.g. `EMBEDDING_DEVICE=cpu`, `RERANKER_MODEL=none` (faster on a slow laptop), `OCR_ENGINE=rapidocr`, `ADMIN_PASSWORD=...` (default `mits-admin`; change it before any shared demo).

## Scope and privacy

- The crawler only contacts `web.mitsgwalior.in`, follows only IT department pages, visits each URL once, and respects `robots.txt`.
- Question and answer text is **never stored**: the log keeps only counts and which sources were cited.
- `data/` (crawl, index, log) is not in git; `run_pipeline.py` rebuilds it.
