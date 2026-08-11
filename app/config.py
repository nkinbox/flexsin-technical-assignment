"""Central configuration.

Every tunable lives here so the pipeline's behaviour can be adjusted without
hunting through modules. Values come from the environment with defaults chosen
for recall; see execution.md sections 4 and 5 for the reasoning.
"""

import os
import re
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


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- Vertex AI ---------------------------------------------------------------
GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Model used to read images and scanned PDFs at ingest. Same family; kept
# separate so vision work can be pointed at a stronger model without changing
# the model that answers questions.
VISION_MODEL = os.getenv("VISION_MODEL", LLM_MODEL)

# Extraction, not composition: reproduce what the sources say rather than
# writing creatively around them.
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.1)
LLM_MAX_OUTPUT_TOKENS = _int("LLM_MAX_OUTPUT_TOKENS", 4096)

# --- Embeddings --------------------------------------------------------------
# "vertex" -> Vertex AI embeddings (far better retrieval recall)
# "local"  -> in-process ONNX all-MiniLM-L6-v2 (no cost, no network, weaker)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "vertex").strip().lower()

VERTEX_EMBED_MODEL = os.getenv("VERTEX_EMBED_MODEL", "gemini-embedding-001")

# 768 balances quality against index size and latency; the model supports
# larger outputs if retrieval quality ever needs more.
EMBED_DIM = _int("EMBED_DIM", 768)

# Vertex caps how many texts one embed call accepts.
EMBED_BATCH_SIZE = _int("EMBED_BATCH_SIZE", 32)

# --- Chunking ----------------------------------------------------------------
# Deliberately large. Small chunks fragment a single idea across several
# records, so a simple question retrieves a piece of its answer rather than the
# whole thing. Larger chunks keep a fact together with the context that
# qualifies it, which is what makes straightforward questions answerable.
CHUNK_SIZE = _int("CHUNK_SIZE", 2000)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 300)
MIN_CHUNK_CHARS = _int("MIN_CHUNK_CHARS", 40)

# --- Retrieval ---------------------------------------------------------------
# Chunks passed to the model. Higher than a typical default: with hybrid
# retrieval the extra candidates cost little and materially reduce the chance
# that the answer simply was not in the context.
TOP_K = _int("TOP_K", 8)

# Candidates each retriever contributes before fusion.
CANDIDATE_POOL = _int("CANDIDATE_POOL", 25)

# Reciprocal Rank Fusion constant. 60 is the standard value from the original
# RRF paper; it damps the influence of a single retriever's top rank.
RRF_K = _int("RRF_K", 60)

HYBRID_SEARCH = _bool("HYBRID_SEARCH", True)

# The relevance gate (execution.md section 5, layer 1).
#
# Cosine distance above this is treated as unrelated. Deliberately lenient:
# the cost of wrongly refusing a question the documents DO answer is worse
# than passing a marginal chunk to the model, which still has `found: false`
# and citation validation behind it.
#
# Run `python scripts/calibrate_threshold.py` against your own corpus to tune.
RELEVANCE_THRESHOLD = _float("RELEVANCE_THRESHOLD", 1.0)

# A chunk also survives the gate on lexical evidence alone. This is what makes
# exact identifiers, product codes and rare names retrievable when the
# embedding is ambivalent about them.
#
# Admission needs BOTH a small absolute floor and a share of the best lexical
# score for this query. Neither works alone: BM25 scores scale with corpus size
# and IDF, so a fixed threshold that suits a large corpus rejects everything in
# a small one, while a pure ratio would admit the top match no matter how weak.
BM25_MIN_SCORE = _float("BM25_MIN_SCORE", 0.3)
BM25_MIN_RATIO = _float("BM25_MIN_RATIO", 0.3)

# --- Memory ------------------------------------------------------------------
MAX_HISTORY_TURNS = _int("MAX_HISTORY_TURNS", 6)

# --- Storage -----------------------------------------------------------------
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")


def _collection_name() -> str:
    """Name the collection after the embedding configuration that filled it.

    Vectors from different models are not comparable, and dimensions differ
    (768 for Vertex, 384 for MiniLM). Encoding the configuration in the name
    means switching providers starts a clean collection instead of silently
    querying one model's vectors with another model's embedding.
    """
    override = os.getenv("COLLECTION_NAME")
    if override:
        return override

    if EMBEDDING_PROVIDER == "vertex":
        tag = f"vertex_{VERTEX_EMBED_MODEL}_{EMBED_DIM}"
    else:
        tag = "local_minilm_384"

    return "docs_" + re.sub(r"[^a-zA-Z0-9]+", "_", tag).strip("_").lower()


COLLECTION_NAME = _collection_name()

# --- Limits ------------------------------------------------------------------
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 20)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

TEXT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS

IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}

# A PDF page yielding fewer characters than this is treated as having no usable
# text layer -- typically a scan. Such documents are re-read with the vision
# model rather than indexed as near-empty.
PDF_MIN_CHARS_PER_PAGE = _int("PDF_MIN_CHARS_PER_PAGE", 50)

# --- UI ----------------------------------------------------------------------
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# --- Shared copy -------------------------------------------------------------
REFUSAL_MESSAGE = (
    "I couldn't find an answer to that in the uploaded documents. "
    "Try rephrasing, or upload a document that covers this topic."
)


def ensure_data_dir() -> None:
    """Create the Chroma directory if absent (first run, or a fresh volume)."""
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
