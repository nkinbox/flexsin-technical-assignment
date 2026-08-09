"""Stage 1 — Document ingestion.

Turns an uploaded file into a uniform list of pages. Every downstream stage
sees the same shape regardless of source format, so the chunker never needs to
know whether it came from a PDF or a text file.

Supported: PDF, DOCX, TXT/MD. Images are out of scope (see execution.md §7).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import docx
from pypdf import PdfReader

from app.config import SUPPORTED_EXTENSIONS


@dataclass
class Page:
    """One unit of source text with the provenance needed for citation.

    `page_number` is 1-indexed. Formats without real pagination (DOCX, TXT)
    report page 1 — the field always carries *something* citable.
    """

    text: str
    page_number: int


class UnsupportedFileError(ValueError):
    """Raised for a file type this build does not handle."""


def extract(filename: str, data: bytes) -> list[Page]:
    """Extract text from an uploaded file.

    Args:
        filename: Original filename — its extension selects the parser.
        data: Raw file bytes.

    Returns:
        Pages with non-empty text, in document order.

    Raises:
        UnsupportedFileError: Extension is not supported.
    """
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileError(
            f"Cannot process '{filename}'. Supported types: {supported}. "
            "Image input is not supported in this build."
        )

    if suffix == ".pdf":
        pages = _extract_pdf(data)
    elif suffix == ".docx":
        pages = _extract_docx(data)
    else:
        pages = _extract_text(data)

    # Drop pages that yielded nothing — scanned/image-only PDF pages land here.
    return [p for p in pages if p.text.strip()]


def _extract_pdf(data: bytes) -> list[Page]:
    """Per-page text extraction, preserving real page numbers for citation.

    Only the text layer is read. A scanned PDF with no text layer produces no
    pages, which surfaces to the user as an explicit error rather than as a
    silently empty document.
    """
    reader = PdfReader(io.BytesIO(data))
    pages: list[Page] = []

    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            # One malformed page shouldn't lose the rest of the document.
            text = ""
        pages.append(Page(text=text, page_number=index))

    return pages


def _extract_docx(data: bytes) -> list[Page]:
    """Extract paragraphs and table cells.

    Table content is included deliberately: a naive paragraph-only read silently
    drops tables, which in business documents often hold exactly the facts
    someone will ask about (pricing, dates, specifications).

    DOCX has no fixed pagination, so the whole document is one page.
    """
    document = docx.Document(io.BytesIO(data))
    parts: list[str] = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            parts.append(paragraph.text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                # Pipe-joined so the row reads as one related record rather than
                # dissolving into unrelated fragments.
                parts.append(" | ".join(cells))

    return [Page(text="\n\n".join(parts), page_number=1)]


def _extract_text(data: bytes) -> list[Page]:
    """Decode plain text, tolerating non-UTF-8 files rather than failing."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")

    return [Page(text=text, page_number=1)]
