"""Extraction tests — format handling and the unsupported-type boundary."""

import io

import docx
import pytest

from app.extract import UnsupportedFileError, extract


def test_txt_extraction():
    pages = extract("notes.txt", b"Hello world.\n\nSecond paragraph.")

    assert len(pages) == 1
    assert "Hello world." in pages[0].text
    assert pages[0].page_number == 1


def test_non_utf8_text_does_not_crash():
    """A latin-1 file should degrade gracefully rather than fail the upload."""
    pages = extract("legacy.txt", "Café résumé".encode("latin-1"))

    assert len(pages) == 1
    assert pages[0].text


def test_docx_extracts_paragraphs():
    document = docx.Document()
    document.add_paragraph("The Enterprise plan costs $99/month.")
    document.add_paragraph("Support is included.")

    buffer = io.BytesIO()
    document.save(buffer)

    pages = extract("plans.docx", buffer.getvalue())

    assert len(pages) == 1
    assert "$99/month" in pages[0].text
    assert "Support is included." in pages[0].text


def test_docx_includes_table_content():
    """Tables often hold exactly the facts users ask about."""
    document = docx.Document()
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Plan"
    table.cell(0, 1).text = "Price"
    table.cell(1, 0).text = "Enterprise"
    table.cell(1, 1).text = "$99"

    buffer = io.BytesIO()
    document.save(buffer)

    pages = extract("pricing.docx", buffer.getvalue())

    assert "Enterprise" in pages[0].text
    assert "$99" in pages[0].text


def test_unsupported_extension_is_rejected():
    with pytest.raises(UnsupportedFileError) as exc:
        extract("photo.png", b"\x89PNG\r\n")

    # The error should name what IS supported, not just what failed.
    assert ".pdf" in str(exc.value)


def test_image_extensions_are_rejected_explicitly():
    """Documented scope boundary: no image input in this build."""
    for name in ("scan.jpg", "diagram.jpeg", "chart.png"):
        with pytest.raises(UnsupportedFileError):
            extract(name, b"binary")


def test_empty_pages_are_dropped():
    assert extract("blank.txt", b"   \n\n   ") == []
