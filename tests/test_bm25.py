"""BM25 lexical search tests.

BM25 exists to catch what dense embeddings blur: exact identifiers, rare
names, version strings. These tests pin that behaviour.
"""

from app.bm25 import BM25Index, tokenize


# --- Tokenisation -------------------------------------------------------------


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_drops_stopwords():
    # "the", "is", "a" carry no discriminating signal.
    assert tokenize("the cat is a animal") == ["cat", "animal"]


def test_tokenize_preserves_identifiers():
    """Identifiers are exactly what BM25 is here to match."""
    assert "sku-4421" in tokenize("Order SKU-4421 shipped")
    assert "v2.5" in tokenize("Upgrade to v2.5 today")
    assert "invoice_9987" in tokenize("See invoice_9987 attached")


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("the is a") == []


# --- Ranking ------------------------------------------------------------------


def test_exact_term_ranks_first():
    docs = [
        "The remote work policy allows three days per week.",
        "Invoice SKU-4421 was shipped on the third of March.",
        "Health coverage includes dental and vision.",
    ]
    index = BM25Index(docs)
    results = index.search("SKU-4421", top_k=3)

    assert results
    assert results[0][0] == 1


def test_documents_without_query_terms_are_omitted():
    """Non-matching documents are absent, not returned with a zero score."""
    docs = ["alpha beta gamma", "delta epsilon zeta"]
    index = BM25Index(docs)
    results = index.search("alpha", top_k=10)

    assert [index for index, _ in results] == [0]


def test_unknown_term_returns_nothing():
    index = BM25Index(["alpha beta", "gamma delta"])
    assert index.search("nonexistentterm", top_k=5) == []


def test_rare_term_outranks_common_term():
    """IDF should favour the discriminating term."""
    docs = [
        "policy policy policy policy",
        "policy quetzalcoatl",
    ]
    index = BM25Index(docs)
    results = index.search("quetzalcoatl policy", top_k=2)

    assert results[0][0] == 1


def test_length_normalisation_prefers_focused_document():
    """A short document matching once beats a long one matching once."""
    docs = [
        "widget",
        "widget " + "filler " * 200,
    ]
    index = BM25Index(docs)
    results = index.search("widget", top_k=2)

    assert results[0][0] == 0


def test_multiple_query_terms_accumulate():
    docs = [
        "remote work policy",
        "remote",
        "unrelated content here",
    ]
    index = BM25Index(docs)
    results = index.search("remote work policy", top_k=3)

    # Matching all three terms should outrank matching one.
    assert results[0][0] == 0
    assert results[0][1] > results[1][1]


def test_top_k_limits_results():
    index = BM25Index([f"shared term document {i}" for i in range(10)])
    assert len(index.search("shared", top_k=3)) == 3


def test_empty_corpus():
    assert BM25Index([]).search("anything", top_k=5) == []


def test_scores_are_positive():
    """A term in most of the corpus must not score negative.

    Without IDF smoothing this is the classic BM25 failure -- a common term
    subtracts from a document that genuinely contained it.
    """
    docs = ["common term"] * 9 + ["common term rare"]
    index = BM25Index(docs)

    for _, score in index.search("common", top_k=10):
        assert score > 0
