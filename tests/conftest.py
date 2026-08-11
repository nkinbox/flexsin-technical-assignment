"""Test configuration.

Runs before any application module is imported, so these settings are in place
when `app.config` reads the environment.

The suite must work with no credentials, no network, and no cost:
  * embeddings are forced to the in-process ONNX backend
  * the model client is mocked in every test that would otherwise call it
"""

import os

os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("GCP_PROJECT", "test-project")
os.environ.setdefault("CHROMA_PATH", "./data/test_chroma")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_retrieval_cache():
    """Clear the cached BM25 index between tests.

    The lexical index is keyed on the store's version counter, and separate
    tests build separate stores whose counters both start at zero -- without
    this, one test's corpus can be served to the next.
    """
    from app.retrieval import reset_cache

    reset_cache()
    yield
    reset_cache()
