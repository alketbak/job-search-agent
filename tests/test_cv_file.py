"""Tests for extraction/cv_file.py - upload validation.

Covers empty/malformed CVs plus file type/size validation.
"""

from __future__ import annotations

import io

import pytest

from extraction.cv_file import InvalidCVFile, validate_and_extract_text


def test_valid_plain_text() -> None:
    text = validate_and_extract_text("cv.txt", b"Senior AI Engineer with 5 years Python experience.")
    assert "AI Engineer" in text


def test_empty_file_rejected() -> None:
    with pytest.raises(InvalidCVFile, match="empty"):
        validate_and_extract_text("cv.txt", b"")


def test_whitespace_only_text_rejected() -> None:
    with pytest.raises(InvalidCVFile, match="No extractable text"):
        validate_and_extract_text("cv.txt", b"   \n\t  ")


def test_unsupported_extension_rejected() -> None:
    with pytest.raises(InvalidCVFile, match="Unsupported file type"):
        validate_and_extract_text("cv.docx", b"some bytes")


def test_no_extension_rejected() -> None:
    with pytest.raises(InvalidCVFile, match="Unsupported file type"):
        validate_and_extract_text("cv", b"some bytes")


def test_oversized_file_rejected() -> None:
    from config import settings

    oversized = b"x" * (settings.cv_max_upload_mb * 1024 * 1024 + 1)
    with pytest.raises(InvalidCVFile, match="exceeds"):
        validate_and_extract_text("cv.txt", oversized)


def test_non_utf8_text_rejected() -> None:
    with pytest.raises(InvalidCVFile, match="not valid UTF-8"):
        validate_and_extract_text("cv.txt", b"\xff\xfe\x00\x01invalid")


def test_malformed_pdf_rejected() -> None:
    with pytest.raises(InvalidCVFile, match="Could not parse PDF"):
        validate_and_extract_text("cv.pdf", b"not actually a pdf file")


def test_valid_pdf_extracts_text() -> None:
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    # A blank page has no text - validate_and_extract_text should reject it
    # as having no extractable text, same as an empty CV.
    with pytest.raises(InvalidCVFile, match="No extractable text"):
        validate_and_extract_text("cv.pdf", buf.getvalue())
