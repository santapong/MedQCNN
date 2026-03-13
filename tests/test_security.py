"""Tests for security module."""

from __future__ import annotations

from medqcnn.api.security import sanitize_filename_search


class TestInputSanitization:
    """Tests for input sanitization functions."""

    def test_sanitize_none(self):
        assert sanitize_filename_search(None) is None

    def test_sanitize_clean_input(self):
        assert sanitize_filename_search("image_001.png") == "image_001.png"

    def test_sanitize_strips_special_chars(self):
        result = sanitize_filename_search("file'; DROP TABLE--")
        assert ";" not in result
        assert "'" not in result
        assert "DROP" in result  # Alphanumeric preserved

    def test_sanitize_truncates_long_input(self):
        long_input = "a" * 500
        result = sanitize_filename_search(long_input)
        assert len(result) <= 128

    def test_sanitize_preserves_dots_hyphens(self):
        assert sanitize_filename_search("scan-01.dicom") == "scan-01.dicom"

    def test_sanitize_empty_after_strip(self):
        # All special chars should result in None
        assert sanitize_filename_search("!!!@@@") is None
