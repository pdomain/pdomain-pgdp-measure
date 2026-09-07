from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdomain_pgdp_measure.f2 import F2LoadError, load_f2, parse_f2_pages


def test_parse_local_block_preserves_text() -> None:
    result = parse_f2_pages({"p001.png": "before /*  <i>verse</i>  */ after"})

    assert result.pages[0].source_text == "before /*  <i>verse</i>  */ after"
    assert result.pages[0].special_blocks == ("  <i>verse</i>  ",)
    assert result.pages[0].has_special_format is True


def test_parse_continued_block_crosses_natural_page_order() -> None:
    result = parse_f2_pages(
        {
            "p10.png": "continued #/ tail",
            "p2.png": "head /# first",
            "p3.png": "middle",
        }
    )

    assert [page.name for page in result.pages] == ["p2.png", "p3.png", "p10.png"]
    assert [page.has_special_format for page in result.pages] == [True, True, True]
    assert [page.special_blocks for page in result.pages] == [
        (" first",),
        ("middle",),
        ("continued ",),
    ]


def test_continued_block_keeps_only_each_pages_body() -> None:
    result = parse_f2_pages(
        {
            "p10.png": "third field   value #/ outside <i>text</i>",
            "p2.png": "before /# <i>first</i>\nfirst field   value",
            "p3.png": "<b>middle</b>\nsecond field   value",
        }
    )

    assert [page.special_blocks for page in result.pages] == [
        (" <i>first</i>\nfirst field   value",),
        ("<b>middle</b>\nsecond field   value",),
        ("third field   value ",),
    ]


@pytest.mark.parametrize(
    ("source_text", "expected_code"),
    [
        ("/* outer /* inner */", "nested_format_control"),
        ("/* local /# continued */", "overlapping_format_control"),
        ("text */ more", "unmatched_format_closer"),
        ("/* unfinished", "unmatched_format_opener"),
        ("/# unfinished", "unmatched_format_opener"),
    ],
)
def test_invalid_controls_report_diagnostics_and_exclude_bodies(
    source_text: str, expected_code: str
) -> None:
    result = parse_f2_pages({"p1.png": source_text})

    assert result.pages[0].special_blocks == ()
    assert [diagnostic.code for diagnostic in result.diagnostics] == [expected_code]
    assert result.diagnostics[0].page_name == "p1.png"


@pytest.mark.parametrize("payload", [[], {"p1.png": 1}])
def test_load_f2_rejects_non_string_page_mapping(payload: object, tmp_path: Path) -> None:
    source = tmp_path / "F2.json"
    _ = source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(F2LoadError):
        _ = load_f2(source)


def test_load_f2_preserves_decoded_utf8_text(tmp_path: Path) -> None:
    source = tmp_path / "F2.json"
    _ = source.write_text('{"p1.png": "caf\\u00e9 /*  text  */"}', encoding="utf-8")

    result = load_f2(source)

    assert result.pages[0].source_text == "caf\u00e9 /*  text  */"
