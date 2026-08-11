"""Measure retrieval distances against your own corpus.

The relevance gate compares cosine distance to `RELEVANCE_THRESHOLD`, and the
right value depends on the embedding model and the documents. Switching from
local MiniLM to Vertex embeddings moves the whole distribution, so a threshold
carried over from one is meaningless for the other.

This script measures the two populations that matter -- questions the documents
answer, and questions they do not -- and recommends a threshold sitting between
them.

    # Index some documents through the app first, then:
    python scripts/calibrate_threshold.py \
        --relevant "What was Q3 revenue?" "Who approves remote work?" \
        --unrelated "How do I bake bread?" "Capital of Peru?"

With no arguments it uses generic probes, which is enough to see the shape of
the distribution but far less useful than questions about your actual corpus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (  # noqa: E402
    EMBEDDING_PROVIDER,
    RELEVANCE_THRESHOLD,
    VERTEX_EMBED_MODEL,
)
from app.store import get_store  # noqa: E402

GENERIC_UNRELATED = [
    "What is the migratory pattern of Arctic terns?",
    "How do I bake sourdough bread?",
    "What is the boiling point of mercury?",
    "Who won the 1966 World Cup final?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--relevant",
        nargs="*",
        default=[],
        help="Questions your documents DO answer.",
    )
    parser.add_argument(
        "--unrelated",
        nargs="*",
        default=GENERIC_UNRELATED,
        help="Questions your documents do NOT answer.",
    )
    args = parser.parse_args()

    store = get_store()
    if store.count() == 0:
        print("No documents indexed. Upload some through the app first.")
        return 1

    model = VERTEX_EMBED_MODEL if EMBEDDING_PROVIDER == "vertex" else "all-MiniLM-L6-v2"
    print(f"Embedding provider : {EMBEDDING_PROVIDER} ({model})")
    print(f"Indexed chunks     : {store.count()}")
    print(f"Current threshold  : {RELEVANCE_THRESHOLD}")
    print("-" * 72)

    if not args.relevant:
        print(
            "No --relevant questions given. Distances below cover only the\n"
            "unrelated side, which cannot tell you where the boundary is.\n"
        )

    relevant_scores = _measure("RELEVANT ", args.relevant, store)
    unrelated_scores = _measure("UNRELATED", args.unrelated, store)

    print("-" * 72)

    if not relevant_scores or not unrelated_scores:
        print("Not enough data on both sides to recommend a threshold.")
        return 0

    worst_relevant = max(relevant_scores)
    best_unrelated = min(unrelated_scores)

    print(f"Worst relevant distance : {worst_relevant:.3f}")
    print(f"Best unrelated distance : {best_unrelated:.3f}")

    if worst_relevant >= best_unrelated:
        print(
            "\nThe two populations OVERLAP -- no single threshold separates them.\n"
            "Options, in order of usefulness:\n"
            "  1. Rely on hybrid retrieval: BM25 admits chunks the vector score\n"
            "     is unsure about, so a permissive threshold costs less than it\n"
            "     would with dense search alone.\n"
            "  2. Set the threshold above the worst relevant distance "
            f"({worst_relevant:.2f})\n"
            "     to avoid refusing answerable questions, and lean on the\n"
            "     model's `found` flag and citation validation behind it.\n"
            "  3. Switch to EMBEDDING_PROVIDER=vertex if you are on local\n"
            "     embeddings -- better separation is mostly a model problem."
        )
        return 0

    recommended = round((worst_relevant + best_unrelated) / 2, 2)
    print(f"\nRecommended RELEVANCE_THRESHOLD={recommended}")
    print(
        f"  ({recommended - worst_relevant:.2f} of headroom before a genuine "
        f"question is refused,\n   {best_unrelated - recommended:.2f} before an "
        "unrelated one slips through)"
    )
    return 0


def _measure(label: str, questions: list[str], store) -> list[float]:
    """Print and collect the top distance for each question."""
    scores: list[float] = []

    for question in questions:
        results = store.query(question, top_k=1)
        if not results or results[0].distance is None:
            print(f"{label}      n/a  {question}")
            continue
        distance = results[0].distance
        scores.append(distance)
        print(f"{label}  {distance:>8.3f}  {question}")

    return scores


if __name__ == "__main__":
    sys.exit(main())
