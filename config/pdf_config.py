"""
Configuration settings for Mark's Local PDF Creation & Artifact Pipeline.
All settings can be configured via environment variables with safe defaults.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MARK_DATA_DIR", BASE_DIR / "data"))
ARTIFACTS_DIR = Path(os.environ.get("MARK_ARTIFACTS_DIR", DATA_DIR / "artifacts"))
FONTS_DIR = Path(os.environ.get("MARK_FONTS_DIR", BASE_DIR / "fonts"))

# Limits
MAX_BLOCKS = int(os.environ.get("MARK_PDF_MAX_BLOCKS", "300"))
MAX_BLOCK_TEXT_LENGTH = int(os.environ.get("MARK_PDF_MAX_BLOCK_TEXT_LENGTH", "10000"))
MAX_TABLE_ROWS = int(os.environ.get("MARK_PDF_MAX_TABLE_ROWS", "100"))
MAX_TABLE_COLS = int(os.environ.get("MARK_PDF_MAX_TABLE_COLS", "15"))
MAX_NESTING_DEPTH = int(os.environ.get("MARK_PDF_MAX_NESTING_DEPTH", "5"))
MAX_IMAGES = int(os.environ.get("MARK_PDF_MAX_IMAGES", "10"))
MAX_IMAGE_BYTES = int(os.environ.get("MARK_PDF_MAX_IMAGE_BYTES", str(10 * 1024 * 1024))) # 10 MB
MAX_PAGES = int(os.environ.get("MARK_PDF_MAX_PAGES", "50"))
MAX_PDF_SIZE_BYTES = int(os.environ.get("MARK_PDF_MAX_SIZE_BYTES", str(25 * 1024 * 1024))) # 25 MB

# Timeouts (seconds)
RENDER_TIMEOUT_SECONDS = float(os.environ.get("MARK_PDF_RENDER_TIMEOUT", "30.0"))
LLM_CALL_TIMEOUT_SECONDS = float(os.environ.get("MARK_PDF_LLM_TIMEOUT", "120.0"))
LLM_OVERALL_TIMEOUT_SECONDS = float(os.environ.get("MARK_PDF_LLM_TOTAL_TIMEOUT", "600.0"))

# Concurrency
CONCURRENT_PDF_JOBS_LIMIT = int(os.environ.get("MARK_PDF_CONCURRENCY_LIMIT", "1"))

# Network & Origin Security
ALLOWED_HOSTS = [
    "127.0.0.1:8765",
    "localhost:8765",
    "127.0.0.1",
    "localhost",
]
ALLOWED_ORIGINS = [
    "http://127.0.0.1:8765",
    "http://localhost:8765",
]

# LLM Staged Generation
OLLAMA_NUM_CTX = int(os.environ.get("MARK_OLLAMA_NUM_CTX", "4096"))
DISABLE_THINKING = os.environ.get("MARK_DISABLE_THINKING", "1").lower() in ("1", "true", "yes")
