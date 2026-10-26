"""Central configuration. Every tunable value lives here.

Values come from environment variables (optionally loaded from a .env file
in the project root). Nothing secret is ever written in the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Placeholder text from .env.example. Treated the same as "no key".
_PLACEHOLDER_KEYS = {"", "your-api-key-here", "your_api_key_here"}


@dataclass(frozen=True)
class Settings:
    docs_dir: Path
    index_dir: Path
    model_cache_dir: Path
    embedding_model: str
    gemini_model: str
    gemini_api_key: str | None
    chunk_size: int       # characters per chunk
    chunk_overlap: int    # characters shared by neighbouring chunks
    top_k: int            # how many chunks to retrieve per question
    llm_timeout_s: int    # seconds before an LLM call is abandoned

    def require_api_key(self) -> str:
        key = (self.gemini_api_key or "").strip()
        if key in _PLACEHOLDER_KEYS:
            raise ConfigError(
                "GEMINI_API_KEY is not set.\n"
                "  1. Create a free key at https://aistudio.google.com/apikey\n"
                "  2. Copy .env.example to .env and put the key after GEMINI_API_KEY=\n"
                "(Retrieval-only commands, 'ingest' and 'search', work without a key.)"
            )
        return key


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}.") from None
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, got {value}.")
    return value


def _path_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings() -> Settings:
    chunk_size = _int_env("CHUNK_SIZE", 800)
    chunk_overlap = _int_env("CHUNK_OVERLAP", 150, minimum=0)
    if chunk_overlap >= chunk_size:
        raise ConfigError(
            f"CHUNK_OVERLAP ({chunk_overlap}) must be smaller than CHUNK_SIZE ({chunk_size})."
        )

    return Settings(
        docs_dir=_path_env("DOCS_DIR", PROJECT_ROOT / "data" / "docs"),
        index_dir=_path_env("INDEX_DIR", PROJECT_ROOT / "storage" / "index"),
        model_cache_dir=_path_env("MODEL_CACHE_DIR", PROJECT_ROOT / "storage" / "models"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=_int_env("TOP_K", 4),
        llm_timeout_s=_int_env("LLM_TIMEOUT_S", 60),
    )
