"""Process-wide configuration, read once from the project .env file.

Every other module imports its settings from here instead of calling
``os.getenv`` itself, so defaults live in exactly one place.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

load_dotenv(PROJECT_DIR / ".env")


def sync_url(url: str) -> str:
    """Return the synchronous (PyMySQL) form of an async (asyncmy) MySQL URL.

    URLs that are not asyncmy URLs are returned unchanged.
    """
    return url.replace("+asyncmy", "+pymysql", 1)


DATABASE_URL = os.getenv(
    "DATABASE_URL", "mysql+asyncmy://chatuser:chatpass@localhost:3306/groupchat"
)
SYNC_DATABASE_URL = sync_url(DATABASE_URL)

JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "43200"))

LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8001/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.2-3b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.30"))

USER_AGENT = "EE542-OIS-Assistant/0.1 (USC student course project)"

if not JWT_SECRET or JWT_SECRET == "please_change_me_to_a_long_random_value":
    raise RuntimeError("Set a unique JWT_SECRET in the project .env file.")


log = logging.getLogger("settings")


def load_thresholds(models_dir: Path) -> dict[str, float]:
    """Per-model not-covered thresholds written by eval_retrieval.py --calibrate."""
    path = models_dir / "thresholds.json"
    if not path.exists():
        return {}
    return {k: float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def threshold_for(model: str, thresholds: dict[str, float]) -> float:
    """OpenAI models use RAG_MIN_SCORE; histogram models use their calibrated entry."""
    from embeddings import is_histogram_model

    if not is_histogram_model(model):
        return RAG_MIN_SCORE
    if model in thresholds:
        return thresholds[model]
    log.warning("no calibrated threshold for %s; using RAG_MIN_SCORE=%.2f (run eval_retrieval.py --calibrate)", model, RAG_MIN_SCORE)
    return RAG_MIN_SCORE
