"""
utils/file_parser.py
--------------------
Document parsing helpers for file preview.
Supports: PDF, DOCX, XLSX, TXT, CSV, and image formats.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import NamedTuple

PREVIEW_MAX_CHARS = 600
IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "webp"})
ACCEPTED_EXTENSIONS = ["pdf", "docx", "xlsx", "txt", "csv", "png", "jpg", "jpeg", "webp"]

# Optional library imports — gracefully absent until installed
try:
    import pypdf as _pypdf
    _HAVE_PYPDF = True
except ImportError:
    _HAVE_PYPDF = False

try:
    import docx as _docx_module
    _HAVE_DOCX = True
except ImportError:
    _HAVE_DOCX = False

try:
    import openpyxl as _openpyxl
    _HAVE_OPENPYXL = True
except ImportError:
    _HAVE_OPENPYXL = False

try:
    from PIL import Image as _PilImage
    _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False


class FileInfo(NamedTuple):
    """Parsed metadata and preview content for an uploaded file."""
    name: str
    ext: str
    size_bytes: int
    is_image: bool
    text_excerpt: str   # empty for image files
    image_bytes: bytes  # empty for text-based files


def fmt_size(size_bytes: int) -> str:
    """Human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 ** 2:.1f} MB"


def extract_text(file_bytes: bytes, ext: str) -> str:
    """Return a short text excerpt for preview purposes."""
    ext = ext.lower()

    if ext == "pdf":
        if not _HAVE_PYPDF:
            return "[Install pypdf to preview PDF content]"
        try:
            reader = _pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = [page.extract_text() or "" for page in reader.pages[:3]]
            return " ".join(pages_text)[:PREVIEW_MAX_CHARS]
        except Exception as exc:  # noqa: BLE001
            return f"[Could not read PDF: {exc}]"

    if ext == "docx":
        if not _HAVE_DOCX:
            return "[Install python-docx to preview DOCX content]"
        try:
            doc = _docx_module.Document(io.BytesIO(file_bytes))
            paras = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paras)[:PREVIEW_MAX_CHARS]
        except Exception as exc:  # noqa: BLE001
            return f"[Could not read DOCX: {exc}]"

    if ext == "xlsx":
        if not _HAVE_OPENPYXL:
            return "[Install openpyxl to preview XLSX content]"
        try:
            wb = _openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
            ws = wb.active
            rows = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i >= 10:
                    break
                rows.append("  |  ".join(str(c) if c is not None else "" for c in row))
            return "\n".join(rows)[:PREVIEW_MAX_CHARS]
        except Exception as exc:  # noqa: BLE001
            return f"[Could not read XLSX: {exc}]"

    if ext in ("txt", "csv"):
        try:
            return file_bytes.decode("utf-8", errors="replace")[:PREVIEW_MAX_CHARS]
        except Exception:  # noqa: BLE001
            return "[Could not decode file]"

    return ""


def parse_uploaded_file(uploaded_file) -> FileInfo:  # type: ignore[no-untyped-def]
    """Parse a Streamlit UploadedFile into a FileInfo."""
    name: str = uploaded_file.name
    ext = Path(name).suffix.lstrip(".").lower()
    raw: bytes = uploaded_file.getvalue()
    is_image = ext in IMAGE_EXTENSIONS

    if is_image:
        return FileInfo(
            name=name, ext=ext, size_bytes=len(raw),
            is_image=True, text_excerpt="", image_bytes=raw,
        )
    return FileInfo(
        name=name, ext=ext, size_bytes=len(raw),
        is_image=False, text_excerpt=extract_text(raw, ext), image_bytes=b"",
    )


def is_preview_available(fi: FileInfo) -> bool:
    """Return True if the file can be meaningfully previewed in the modal."""
    if fi.is_image:
        return bool(fi.image_bytes)
    if fi.text_excerpt and fi.text_excerpt.strip():
        # Disallow error/missing-dependency stub messages as valid previews
        if fi.text_excerpt.startswith("[Could not") or fi.text_excerpt.startswith("[Install"):
            return False
        return True
    return False


def open_image(image_bytes: bytes):  # type: ignore[return]
    """Return a PIL Image if available, else raw bytes for st.image."""
    if _HAVE_PIL:
        return _PilImage.open(io.BytesIO(image_bytes))
    return image_bytes

