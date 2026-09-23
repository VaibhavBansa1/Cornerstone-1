"""Scope configuration for the MITS IT-department scraper and RAG pipeline."""
import os
from pathlib import Path

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
# ALLOWED_PATH_PREFIX. Set this to the aria-label used by the target
# department's nav (view the page source's <aside id="sp-left"> to find it).
DEPARTMENT_NAV_ARIA_LABEL = "Department of Information Technology"

# Link/asset discovery is restricted to this container id so site-wide
# chrome (top mega-menu, footer "quick links" widgets linking to unrelated
# hostel/NAAC/brochure PDFs etc.) is never treated as in-scope just because
# it happens to be on the same domain. Falls back to the whole page if the
# id isn't found (e.g. site structure changed, or a different site).
MAIN_CONTENT_CONTAINER_ID = "sp-main-body"

# Respect robots.txt rules for ALLOWED_DOMAIN.
RESPECT_ROBOTS_TXT = True

# Safety valve: hard cap on number of *pages* (not assets) visited, so a
# scope misconfiguration can never turn into an unbounded crawl.
MAX_PAGES = 500

# Seconds to wait between requests (politeness delay).
REQUEST_DELAY_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 20
USER_AGENT = "MITS-IT-Dept-RAG-Bot/1.0 (+educational research crawler)"

# File extensions treated as downloadable assets rather than crawlable pages.
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
# Documents uploaded by staff through the admin page (web/app.py). These are
# indexed alongside the crawled site on every rebuild.
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt"}
# Every question/answer asked through the web UI is logged here.
CHAT_LOG_DB_PATH = DATA_DIR / "chat_log.db"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
INDEX_METADATA_PATH = INDEX_DIR / "chunks_meta.jsonl"

# --- Embedding / RAG ------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150
TOP_K = 5

# --- Answer generation backend --------------------------------------------
# Which LLM answers questions using the retrieved context. Leave as "auto" to
# pick the first available option in this order: anthropic -> openai ->
# gemini -> ollama (local) -> none (retrieval-only, no generation).
# Force one explicitly via the RAG_GENERATION_PROVIDER env var, e.g.:
#   RAG_GENERATION_PROVIDER=ollama
GENERATION_PROVIDER = os.environ.get("RAG_GENERATION_PROVIDER", "auto")

# Anthropic (used when ANTHROPIC_API_KEY is set)
GENERATION_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

# OpenAI, or any OpenAI-compatible server (LM Studio, llama.cpp, vLLM) when
# OPENAI_BASE_URL points at it instead of https://api.openai.com/v1.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "huihui-ai.deepseek-r1-distill-qwen-14b-abliterated-v2")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")  # None => official OpenAI API

# Gemini (used when GEMINI_API_KEY or GOOGLE_API_KEY is set)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# Local LLM via Ollama's native API (https://github.com/ollama/ollama) -- no
# API key needed, just `ollama pull <model>` and `ollama serve` running.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
