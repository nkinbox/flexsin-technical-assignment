"""Okapi BM25 lexical search.

Implemented directly rather than pulled in as a dependency: it is ~80 lines,
has no install risk, and keeps the ranking behaviour inspectable and testable.

BM25 exists here to cover what dense embeddings are weakest at. An embedding
compresses meaning, which is exactly wrong for tokens that carry no meaning to
compress -- an invoice number, a product SKU, a rare surname, a version string.
Those are precisely the terms a user quotes verbatim when asking a simple
factual question, so lexical matching is not a fallback but a complement.

Scoring is standard Okapi BM25:

    score(D, Q) = sum over terms t in Q of
                  IDF(t) * f(t,D) * (k1 + 1)
                  ------------------------------------------
                  f(t,D) + k1 * (1 - b + b * |D| / avgdl)
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Split on anything that is not alphanumeric, keeping digits attached to
# letters so identifiers like "SKU-4421" and "v2.5" survive as useful tokens.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.\-_][a-z0-9]+)*")

# Very common words carry almost no discriminating signal and inflate scores on
# long chunks. IDF already suppresses them; removing them up front also keeps
# the postings lists smaller.
_STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from as is are was were
    be been being do does did doing have has had having i you he she it we they
    this that these those there here what which who whom how why when where
    will would shall should can could may might must not no nor so than then
    too very s t don now
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split into terms, and drop stopwords."""
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in _STOPWORDS
    ]


class BM25Index:
    """In-memory BM25 index over a fixed corpus.

    Rebuilt whenever the document set changes. At POC scale this costs
    milliseconds; a larger deployment would move lexical search into the
    datastore rather than holding a second copy in the process.
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        """Build the index.

        Args:
            documents: Chunk texts, positionally aligned with their ids.
            k1: Term-frequency saturation. Higher means repeated terms keep
                adding score for longer.
            b: Length normalisation, 0 to 1. At 0.75 a long chunk is penalised
                for diluting its matches without being ruled out.
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = len(documents)

        self._doc_tokens: list[Counter] = []
        self._doc_lengths: list[int] = []
        document_frequency: Counter = Counter()

        for text in documents:
            tokens = tokenize(text)
            counts = Counter(tokens)
            self._doc_tokens.append(counts)
            self._doc_lengths.append(len(tokens))
            # Document frequency counts documents containing the term, not
            # total occurrences -- hence the set.
            document_frequency.update(set(counts))

        self._avgdl = (
            sum(self._doc_lengths) / self.corpus_size if self.corpus_size else 0.0
        )
        self._idf = {
            term: self._compute_idf(freq) for term, freq in document_frequency.items()
        }

    def _compute_idf(self, doc_freq: int) -> float:
        """Smoothed inverse document frequency.

        The +0.5 terms are Robertson-Sparck-Jones smoothing; the outer +1 keeps
        the result positive for a term appearing in more than half the corpus,
        which would otherwise score negative and subtract from a document that
        genuinely matched.
        """
        return math.log(1 + (self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Rank documents against a query.

        Returns:
            Up to `top_k` (document_index, score) pairs, highest score first.
            Documents matching no query term are omitted rather than returned
            with a score of zero.
        """
        if not self.corpus_size:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: dict[int, float] = {}

        for term in set(query_terms):
            idf = self._idf.get(term)
            if idf is None:
                continue  # term appears nowhere in the corpus

            for index, counts in enumerate(self._doc_tokens):
                frequency = counts.get(term)
                if not frequency:
                    continue

                length_norm = 1 - self.b + self.b * (
                    self._doc_lengths[index] / self._avgdl if self._avgdl else 0
                )
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * length_norm
                scores[index] = scores.get(index, 0.0) + idf * (
                    numerator / denominator
                )

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_k]
