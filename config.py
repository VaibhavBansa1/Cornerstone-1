"""Scope configuration for the MITS IT-department scraper and RAG pipeline."""
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
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
INDEX_METADATA_PATH = INDEX_DIR / "chunks_meta.jsonl"

# --- Embedding / RAG ------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150
TOP_K = 5

# Anthropic model used for answer generation when ANTHROPIC_API_KEY is set.
GENERATION_MODEL = "claude-sonnet-5"
