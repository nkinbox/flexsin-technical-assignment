"""Central configuration.

Every tunable lives here so the pipeline's behaviour can be adjusted without
hunting through modules. Values come from the environment with calibrated
defaults; see execution.md sections 4 and 5 for the reasoning behind them.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# --- Vertex AI ---------------------------------------------------------------
GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Extraction, not composition: we want the model to reproduce what the sources
# say, not to write creatively around them.
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.1)
LLM_MAX_OUTPUT_TOKENS = _int("LLM_MAX_OUTPUT_TOKENS", 2048)

# --- Chunking ----------------------------------------------------------------
CHUNK_SIZE = _int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
MIN_CHUNK_CHARS = _int("MIN_CHUNK_CHARS", 50)

# --- Retrieval ---------------------------------------------------------------
TOP_K = _int("TOP_K", 5)

# The relevance gate (execution.md section 5, layer 1).
#
# Chunks with cosine distance above this are discarded. If nothing survives,
# the API refuses WITHOUT invoking the model — a model that is never called
# cannot hallucinate.
#
# Calibrated against measured distances, not guessed. Across a sample of
# relevant and unrelated questions over real documents:
#
#   relevant   0.233 - 0.666   (worst case 0.666)
#   unrelated  0.950 - 1.080   (best case 0.950)
#
# 0.80 sits near the centre of that gap: ~0.13 of headroom before a genuine
# question is wrongly refused, ~0.15 before an unrelated one slips through.
# Raise it if legitimate questions get refused; lower it to be stricter.
# `tests/test_integration.py` asserts both sides of this margin hold.
RELEVANCE_THRESHOLD = _float("RELEVANCE_THRESHOLD", 0.80)

# --- Memory ------------------------------------------------------------------
MAX_HISTORY_TURNS = _int("MAX_HISTORY_TURNS", 6)

# --- Storage -----------------------------------------------------------------
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

# --- Limits ------------------------------------------------------------------
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 20)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# --- UI ----------------------------------------------------------------------
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# --- Shared copy -------------------------------------------------------------
# Both refusal paths (the gate, and the model reporting found=false) return
# this exact string, so callers observe one consistent behaviour.
REFUSAL_MESSAGE = (
    "I couldn't find an answer to that in the uploaded documents. "
    "Try rephrasing, or upload a document that covers this topic."
)


def ensure_data_dir() -> None:
    """Create the Chroma directory if absent (first run, or a fresh volume)."""
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
