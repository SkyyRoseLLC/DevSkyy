from __future__ import annotations

from pathlib import Path

import pytest

from scripts.format_markup import format_markup


def test_format_markup_normalizes_line_endings_without_rewriting_content(tmp_path: Path) -> None:
    path = tmp_path / "asset.svg"
    path.write_bytes(
        b'<svg xmlns="http://www.w3.org/2000/svg">\r\n  <text>Skyy Rose</text>\r\n</svg>\r\n\r\n'
    )

    assert format_markup(path) is True
    assert (
        path.read_bytes()
        == b'<svg xmlns="http://www.w3.org/2000/svg">\n  <text>Skyy Rose</text>\n</svg>\n'
    )
    assert format_markup(path) is False


def test_format_markup_rejects_invalid_xml_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "broken.xml"
    original = b"<root><child></root>\r\n"
    path.write_bytes(original)

    with pytest.raises(ValueError, match="invalid XML"):
        format_markup(path)

    assert path.read_bytes() == original


def test_format_markup_rejects_external_entities(tmp_path: Path) -> None:
    path = tmp_path / "external.svg"
    original = (
        b'<!DOCTYPE svg [<!ENTITY external SYSTEM "file:///etc/passwd">]><svg>&external;</svg>\n'
    )
    path.write_bytes(original)

    with pytest.raises(ValueError, match="external entities are not allowed"):
        format_markup(path)

    assert path.read_bytes() == original
