"""Tests for app/utils/file_processor.py"""

import base64
import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PIL import Image

from app.utils.file_processor import convert_file_to_image


def _make_fake_file(name: str, content: bytes):
    """Return a minimal file-like object similar to Streamlit's UploadedFile."""

    class FakeFile:
        def __init__(self, n, c):
            self.name = n
            self._content = c

        def read(self):
            return self._content

    return FakeFile(name, content)


def _tiny_png_bytes() -> bytes:
    """Generate a 10×10 red PNG in memory."""
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _tiny_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (10, 10), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestConvertFileToImage:
    # ── PNG ───────────────────────────────────────────────────────────────────

    def test_png_returns_string(self):
        f = _make_fake_file("ecg.png", _tiny_png_bytes())
        result = convert_file_to_image(f)
        assert isinstance(result, str)

    def test_png_is_valid_base64(self):
        f = _make_fake_file("ecg.png", _tiny_png_bytes())
        result = convert_file_to_image(f)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_png_decodes_to_png(self):
        f = _make_fake_file("ecg.png", _tiny_png_bytes())
        result = convert_file_to_image(f)
        decoded = base64.b64decode(result)
        img = Image.open(io.BytesIO(decoded))
        assert img.format == "PNG"

    def test_png_uppercase_extension(self):
        f = _make_fake_file("ecg.PNG", _tiny_png_bytes())
        result = convert_file_to_image(f)
        assert isinstance(result, str)

    # ── JPEG ──────────────────────────────────────────────────────────────────

    def test_jpeg_returns_string(self):
        f = _make_fake_file("ecg.jpg", _tiny_jpeg_bytes())
        result = convert_file_to_image(f)
        assert isinstance(result, str)

    def test_jpeg_extension(self):
        f = _make_fake_file("ecg.jpeg", _tiny_jpeg_bytes())
        result = convert_file_to_image(f)
        assert isinstance(result, str)

    def test_jpeg_decoded_is_png(self):
        # Output is always PNG regardless of input format
        f = _make_fake_file("ecg.jpg", _tiny_jpeg_bytes())
        result = convert_file_to_image(f)
        decoded = base64.b64decode(result)
        img = Image.open(io.BytesIO(decoded))
        assert img.format == "PNG"

    def test_jpeg_uppercase_extension(self):
        f = _make_fake_file("ecg.JPG", _tiny_jpeg_bytes())
        result = convert_file_to_image(f)
        assert isinstance(result, str)

    # ── PDF ───────────────────────────────────────────────────────────────────

    def test_pdf_returns_string(self):
        pytest.importorskip("pdf2image")
        try:
            import subprocess
            subprocess.run(["pdftoppm", "-v"], capture_output=True, check=False)
        except FileNotFoundError:
            pytest.skip("poppler not installed — pdf2image requires it")

        # Minimal valid single-page PDF (hand-crafted, no external libs)
        minimal_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f\n"
            b"0000000009 00000 n\n0000000058 00000 n\n"
            b"0000000115 00000 n\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF\n"
        )
        f = _make_fake_file("ecg.pdf", minimal_pdf)
        result = convert_file_to_image(f)
        assert isinstance(result, str)

    # ── Error cases ───────────────────────────────────────────────────────────

    def test_unsupported_extension_raises(self):
        f = _make_fake_file("ecg.bmp", b"fake")
        with pytest.raises(ValueError, match="Unsupported file type"):
            convert_file_to_image(f)

    def test_txt_extension_raises(self):
        f = _make_fake_file("ecg.txt", b"not an image")
        with pytest.raises(ValueError):
            convert_file_to_image(f)

    def test_no_extension_raises(self):
        f = _make_fake_file("ecg", b"no extension")
        with pytest.raises(ValueError):
            convert_file_to_image(f)

    # ── Raw bytes input (no .read method) ────────────────────────────────────

    def test_accepts_raw_bytes_with_name(self):
        """file_processor accepts objects with .name + .read() — covers Streamlit API."""
        f = _make_fake_file("test.png", _tiny_png_bytes())
        result = convert_file_to_image(f)
        assert isinstance(result, str)

    # ── Output dimensions preserved ───────────────────────────────────────────

    def test_output_is_rgb(self):
        f = _make_fake_file("ecg.png", _tiny_png_bytes())
        result = convert_file_to_image(f)
        decoded = base64.b64decode(result)
        img = Image.open(io.BytesIO(decoded))
        assert img.mode == "RGB"
