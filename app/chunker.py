"""Stage 2 — Chunking.

Splits extracted pages into overlapping, boundary-aware chunks carrying the
metadata needed for citation.

Strategy (see execution.md §4 for the full rationale):

  Recursive boundary-aware splitting. Separators are tried strongest-first --
  paragraph, then line, then sentence, then word -- and the strongest boundary
  that fits the size budget is used. A fixed-width split severs sentences
  mid-clause and produces chunks that embed poorly; keeping chunks aligned to
  natural boundaries keeps each one semantically self-contained.

  Overlap exists so that a fact spanning a boundary is fully present in at
  least one chunk rather than half-present in two.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS
from app.extract import Page

# Strongest semantic boundary first. The empty string is the final fallback:
# it permits a hard character split for pathological input (e.g. a single
# unbroken 5,000-character token) that contains none of the real separators.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """A retrievable unit of text plus the provenance that makes it citable."""

    text: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    char_start: int
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_metadata(self) -> dict:
        """Flatten to Chroma-compatible metadata (scalar values only)."""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
        }


def chunk_pages(
    pages: list[Page],
    doc_id: str,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Chunk every page of a document.

    Pages are chunked independently so a chunk never straddles a page boundary.
    That keeps `page_number` unambiguous — a citation pointing at "page 4" is
    then verifiable, which a chunk spanning pages 4 and 5 would not be.

    Args:
        pages: Output of `extract.extract`.
        doc_id: Stable id for this document.
        filename: Original filename, surfaced in citations.
        chunk_size: Target maximum characters per chunk.
        overlap: Characters shared between consecutive chunks.
        min_chars: Chunks shorter than this are discarded as noise.

    Returns:
        Chunks in document order, `chunk_index` numbered across the document.
    """
    chunks: list[Chunk] = []
    index = 0

    for page in pages:
        normalised = _normalise(page.text)

        for text, char_start in _split_text(normalised, chunk_size, overlap):
            # Fragments below the floor (page numbers, orphan headers, footer
            # artifacts) carry no retrievable meaning and would waste a top-k
            # slot that a real chunk could occupy.
            if len(text.strip()) < min_chars:
                continue

            chunks.append(
                Chunk(
                    text=text.strip(),
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page.page_number,
                    chunk_index=index,
                    char_start=char_start,
                )
            )
            index += 1

    return chunks


def _normalise(text: str) -> str:
    """Collapse whitespace noise that would otherwise distort chunk budgets.

    PDF extraction commonly yields runs of blank lines and trailing spaces.
    Left alone, these consume the character budget without carrying meaning.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int]]:
    """Split text into overlapping windows aligned to natural boundaries.

    Returns (chunk_text, char_start) pairs. `char_start` is the offset into the
    page, retained so a citation can be located precisely within the source.
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [(text, 0)]

    results: list[tuple[str, int]] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Final chunk — no boundary search needed.
        if end == len(text):
            results.append((text[start:end], start))
            break

        split_at = _find_boundary(text, start, end)
        results.append((text[start:split_at], start))

        # Step back by `overlap` so the next chunk repeats the tail of this one.
        next_start = split_at - overlap

        # Guard against non-advancing loops: if a boundary lands at or before
        # the current position, force forward progress. Without this, text
        # whose only boundary is near the start of the window loops forever.
        if next_start <= start:
            next_start = split_at

        start = next_start

    return results


def _find_boundary(text: str, start: int, end: int) -> int:
    """Find the best split point in text[start:end], strongest separator first.

    Returns the absolute index to split at. Falls back to a hard split at `end`
    when the window contains no separator at all.
    """
    window = text[start:end]

    for separator in SEPARATORS:
        if not separator:
            break

        position = window.rfind(separator)

        # Require the boundary to be past the halfway mark. A separator right
        # at the start of the window would produce a tiny chunk and defeat the
        # size budget; better to fall through to a weaker separator that splits
        # closer to the target size.
        if position > len(window) // 2:
            return start + position + len(separator)

    # No usable separator — hard split at the size limit.
    return end
