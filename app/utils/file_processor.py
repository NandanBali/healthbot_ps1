"""
Phase 3.1 — File preprocessing for ECG uploads.

Supports JPEG, PNG, and PDF (first page only).
Returns a base64-encoded PNG string ready for the vision API.
"""

import base64
import io
from typing import Union

from PIL import Image


def convert_file_to_image(uploaded_file) -> str:
    """
    Convert an uploaded file (file-like object with .name and .read()) to a
    base64-encoded PNG string.

    Accepts: JPEG, PNG, PDF (first page converted via pdf2image).
    Returns: base64 string (no data-URI prefix).
    Raises: ValueError for unsupported file types.
    """
    filename = getattr(uploaded_file, "name", "")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in ("jpg", "jpeg", "png"):
        raw = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    elif ext == "pdf":
        from pdf2image import convert_from_bytes

        raw = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
        pages = convert_from_bytes(raw, first_page=1, last_page=1)
        if not pages:
            raise ValueError("PDF produced no pages.")
        img = pages[0].convert("RGB")
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Please upload a JPEG, PNG, or PDF."
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
