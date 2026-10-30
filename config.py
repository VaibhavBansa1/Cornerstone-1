"""All settings for the MITS IT-department scraper and RAG pipeline.

Every value that you might want to change between machines (which LLM to
use, CPU vs GPU, API keys) can be overridden with an environment variable,
so the code itself never has to be edited to switch setups.
"""
import os
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


# --- Crawl scope -------------------------------------------------------
# Only this domain is ever contacted.
ALLOWED_DOMAIN = "web.mitsgwalior.in"

# Only pages whose path starts with this prefix are crawled for links.
# Assets (pdf/image/audio/video) referenced from within that scope are
# downloaded even if they live under a different path (e.g. /uploads/...),
# but never from a different domain.
ALLOWED_PATH_PREFIX = "/academics/departments/information-technology"

START_URL = f"https://{ALLOWED_DOMAIN}{ALLOWED_PATH_PREFIX}"

# This site (Joomla) does NOT nest all department content under
# ALLOWED_PATH_PREFIX -- most sub-pages (syllabus, faculty, time table,
# downloads, ...) are published as root-level slugs like "/syllabus-it",
# linked only from the department's own sidebar menu. That sidebar is
# rendered as a <nav aria-label="..."> on every department page, so links
# found inside it are treated as in-scope even when their URL doesn't share
# ALLOWED_PATH_PREFIX.
DEPARTMENT_NAV_ARIA_LABEL = "Department of Information Technology"

# Link/asset discovery is restricted to this container id so site-wide
# chrome (top mega-menu, footer widgets) is never treated as in-scope.
MAIN_CONTENT_CONTAINER_ID = "sp-main-body"

RESPECT_ROBOTS_TXT = True
MAX_PAGES = 500
REQUEST_DELAY_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 20
# A timeout / dropped connection / 429 / 5xx is retried this many times
# before the URL is recorded as failed.
FETCH_RETRIES = 1
RETRY_DELAY_SECONDS = 3
USER_AGENT = "MITS-IT-Dept-RAG-Bot/1.0 (+educational research crawler)"

DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
ASSET_EXTENSIONS = (
    DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
)

# --- Storage layout ------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
RAW_PAGES_DIR = DATA_DIR / "raw" / "pages"
RAW_ASSETS_DIR = DATA_DIR / "raw" / "assets"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
MANIFEST_PATH = DATA_DIR / "manifest.json"
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt"}
CHAT_LOG_DB_PATH = DATA_DIR / "chat_log.db"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
INDEX_METADATA_PATH = INDEX_DIR / "chunks_meta.jsonl"
# Records which embedding model built the index, so a model change is
# detected instead of silently returning garbage matches.
INDEX_INFO_PATH = INDEX_DIR / "index_info.json"

# --- Chunking --------------------------------------------------------------
# Text is split on paragraph/sentence boundaries into pieces of roughly this
# size; neighbouring pieces share a little text so nothing is cut in half.
CHUNK_SIZE_CHARS = 900
CHUNK_OVERLAP_CHARS = 150

# --- OCR (scanned PDFs) -----------------------------------------------------
# auto | easyocr | rapidocr | none   (install engines with: uv sync --extra ocr)
OCR_ENGINE = _env("OCR_ENGINE", "auto")
OCR_RENDER_SCALE = 2.0      # page image size for OCR (2 = ~150 dpi)
OCR_MIN_CONFIDENCE = 0.35   # drop OCR words the engine isn't sure about

# --- Embeddings ------------------------------------------------------------
# Changing the model means the index must be rebuilt:
#   python run_pipeline.py --skip-crawl
# Multilingual e5-small handles English, Hindi and Hinglish and is small
# enough to run on a laptop CPU.
EMBEDDING_MODEL_NAME = _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
# "auto" uses an NVIDIA GPU when PyTorch can see one, otherwise the CPU.
# Force with EMBEDDING_DEVICE=cpu or EMBEDDING_DEVICE=cuda.
EMBEDDING_DEVICE = _env("EMBEDDING_DEVICE", "auto")
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32)

# --- Retrieval ---------------------------------------------------------------
TOP_K = 6                      # pieces of text handed to the LLM
TITLE_WEIGHT = 3               # keyword search: a title word counts this many times
# Query understanding: documents filed under the batch/semester the question
# is about ("V SEM · 2024-28 BATCH") get this bonus; other batches' get it
# taken away. And up to this many matching documents are always included.
BATCH_MATCH_BONUS = 0.3
BATCH_MATCH_CHUNKS = 2
CANDIDATES_PER_RETRIEVER = 30  # how many each search method proposes
# Re-scores the candidates with a model that reads question + text together.
# The default is small (~90 MB) and fast on a laptop CPU. More accurate but
# ~1 GB and slower on CPU: RERANKER_MODEL=BAAI/bge-reranker-base
# Switch it off entirely with RERANKER_MODEL=none.
RERANKER_MODEL = _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = _env_int("RERANK_CANDIDATES", 30)
# Final relevance = this share from the reranker + the rest from the
# keyword/meaning scores, so one model can't wipe out a strong match.
RERANKER_BLEND = _env_float("RERANKER_BLEND", 0.5)
# At most this many chunks from the same page/PDF in one answer.
MAX_CHUNKS_PER_SOURCE = 2
# Role questions ("who is the HOD?") always get the newest chunks that name
# someone in that role, so the answer can't miss a recent change.
ROLE_TERMS = {
    "hod", "head", "dean", "coordinator", "director", "principal", "chairman",
    "chairperson", "incharge", "convener", "registrar", "chancellor", "president",
}
NEWEST_ROLE_CHUNKS = 3
# How strongly newer text is preferred when relevance is similar (0 = off).
RECENCY_WEIGHT = _env_float("RECENCY_WEIGHT", 0.15)
# Text older than this many years gets no recency bonus at all.
RECENCY_HORIZON_YEARS = 6
# Words students use that mean the same thing as words in the documents.
# Used to widen the keyword search; add more as you notice gaps.
QUERY_SYNONYMS = {
    "hod": ["head of department", "head", "dean"],
    "head": ["hod", "dean"],
    "dean": ["head", "hod"],
    "tt": ["time table", "timetable"],
    "timetable": ["time table", "time-table"],
    "syllabus": ["scheme", "curriculum"],
    "sem": ["semester"],
    "prof": ["professor"],
    "lab": ["laboratory", "laboratories"],
    "labs": ["lab", "laboratory", "laboratories"],
    "dept": ["department"],
    "department": ["school"],
    "school": ["department"],
    "expert": ["industry person"],
    "experts": ["industry person"],
    "mandatory": ["compulsory", "mandatory workshop", "audit"],
    "compulsory": ["mandatory"],
}

# Questions about these topics always search the matching page of the
# department website too (topic found by rag/query_understanding.py),
# e.g. "what labs are there?" -> the "Laboratories" page.
TOPIC_PAGES = {
    "labs": "Laboratories",
    "contacts": "Contacts",
    "overview": "School of Information Technology",
    "programmes": "Programmes Offered",
}

# --- Answer generation (LLM) ---------------------------------------------------
# Comma-separated list of providers to try, in order. If one fails or hits a
# rate limit, the next one answers. "auto" = every provider that has its API
# key set (in the order below), then a local Ollama if it is running.
#   LLM_PROVIDERS=openrouter,groq,ollama
# Providers: openrouter, groq, github, openai, openai_compat, gemini,
# anthropic, ollama, none
LLM_PROVIDERS = _env("LLM_PROVIDERS", _env("RAG_GENERATION_PROVIDER", "auto"))
AUTO_PROVIDER_ORDER = [
    "openrouter", "groq", "github", "gemini", "anthropic", "openai", "openai_compat", "ollama",
]
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 1024)
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.2)
LLM_TIMEOUT_SECONDS = _env_int("LLM_TIMEOUT_SECONDS", 90)
# After a provider says "rate limited", skip it for this many seconds.
LLM_RATE_LIMIT_COOLDOWN_SECONDS = _env_int("LLM_RATE_LIMIT_COOLDOWN_SECONDS", 60)

# Settings for each provider: base URL, env var holding the key, model.
# Every model can be overridden with <NAME>_MODEL, e.g. OPENROUTER_MODEL=...
PROVIDER_SETTINGS = {
    "openrouter": {
        "kind": "openai",
        "base_url": _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "api_key_env": "OPENROUTER_API_KEY",
        "model": _env("OPENROUTER_MODEL", "google/gemma-4-31b-it:free"),
    },
    "groq": {
        "kind": "openai",
        "base_url": _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        "api_key_env": "GROQ_API_KEY",
        "model": _env("GROQ_MODEL", "llama-3.3-70b-versatile"),
    },
    "github": {  # GitHub Models -- free with a GitHub personal access token
        "kind": "openai",
        "base_url": _env("GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference"),
        "api_key_env": "GITHUB_MODELS_TOKEN",
        "model": _env("GITHUB_MODELS_MODEL", "openai/gpt-4.1-mini"),
    },
    "openai": {
        "kind": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": _env("OPENAI_MODEL", "gpt-4.1-mini"),
    },
    "openai_compat": {  # any local OpenAI-style server: LM Studio, llama.cpp, vLLM
        "kind": "openai",
        "base_url": _env("OPENAI_COMPAT_BASE_URL", "http://localhost:1234/v1"),
        "api_key_env": "OPENAI_COMPAT_API_KEY",
        "model": _env("OPENAI_COMPAT_MODEL", "local-model"),
        "key_optional": True,
        "auto_requires_env": "OPENAI_COMPAT_BASE_URL",
    },
    "gemini": {
        "kind": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key_env": "GEMINI_API_KEY",
        "alt_api_key_env": "GOOGLE_API_KEY",
        "model": _env("GEMINI_MODEL", "gemini-3.6-flash"),
    },
    "anthropic": {
        "kind": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": _env("ANTHROPIC_MODEL", "claude-sonnet-5"),
    },
    "ollama": {
        "kind": "ollama",
        "base_url": _env("OLLAMA_BASE_URL", "http://localhost:11434"),
        # If this model isn't pulled, the first installed model is used.
        "model": _env("OLLAMA_MODEL", "llama3.1"),
        # Loading a big local model the first time can take minutes.
        "timeout": _env_int("OLLAMA_TIMEOUT_SECONDS", 300),
        "keep_alive": _env("OLLAMA_KEEP_ALIVE", "30m"),
        # Room for the question + sources + answer (Ollama's default is small).
        "num_ctx": _env_int("OLLAMA_NUM_CTX", 8192),
        # Thinking models (Qwen3, DeepSeek-R1, Gemma-4 ...) are much faster
        # with thinking off, and the sources already contain the answer.
        "think": _env("OLLAMA_THINK", "false").lower() in ("1", "true", "yes"),
    },
}

# Back-compat names used by older code/docs.
OLLAMA_BASE_URL = PROVIDER_SETTINGS["ollama"]["base_url"]
OLLAMA_MODEL = PROVIDER_SETTINGS["ollama"]["model"]
