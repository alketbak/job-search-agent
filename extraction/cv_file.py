"""CV upload validation and text extraction.

Accepts plain text and PDF. Validates file type and size before doing any
parsing work.
"""

from __future__ import annotations

import io
import logging

from config import settings

log = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


class InvalidCVFile(Exception):
    """Raised for a rejected upload - wrong type, too large, or unreadable.
    Callers map this to an HTTP 400/413, not a 500."""


def _extension(filename: str) -> str:
    idx = filename.rfind(".")
    return filename[idx:].lower() if idx != -1 else ""


def validate_and_extract_text(filename: str, content: bytes) -> str:
    """Validate an uploaded CV file and return its plain text.

    Raises `InvalidCVFile` for anything that fails validation. Never raises
    for a merely-hard-to-parse-but-valid file - a PDF with no extractable
    text raises `InvalidCVFile` too, since an empty CV can't be extracted
    from.
    """
    max_bytes = settings.cv_max_upload_mb * 1024 * 1024
    if len(content) == 0:
        raise InvalidCVFile("Uploaded file is empty.")
    if len(content) > max_bytes:
        raise InvalidCVFile(
            f"File is {len(content) / 1024 / 1024:.1f}MB, exceeds the "
            f"{settings.cv_max_upload_mb}MB limit."
        )

    ext = _extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidCVFile(
            f"Unsupported file type '{ext or '(none)'}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}."
        )

    if ext == ".txt":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as e:
            raise InvalidCVFile(f"File is not valid UTF-8 text: {e}") from e
    else:
        text = _extract_pdf_text(content)

    if not text.strip():
        raise InvalidCVFile("No extractable text found in the uploaded file.")
    return text


def _extract_pdf_text(content: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as e:
        raise InvalidCVFile(f"Could not parse PDF: {e}") from e
    return "\n".join(pages)
