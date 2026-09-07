from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import pairwise

import pytest

from pdomain_pgdp_measure.alignment_source import tokenize_f2_pages, tokenize_source_page
from pdomain_pgdp_measure.f2 import parse_f2_pages


@pytest.mark.parametrize(
    ("source", "expected_lines", "expected_visible"),
    [
        ("plain text", ("plain text",), "plain text"),
        (
            "caf\u00e9\r\nnext\rfinal\n",
            ("caf\u00e9", "next", "final", ""),
            "caf\u00e9\nnext\nfinal\n",
        ),
        ("/* special */", (" special ",), " special "),
        ("[** proof note]word", ("word",), "word"),
    ],
)
def test_tokenize_source_preserves_visible_text_and_line_separators(
    source: str, expected_lines: tuple[str, ...], expected_visible: str
) -> None:
    tokenized = tokenize_source_page("001.png", source)

    assert tuple(line.visible_text for line in tokenized.lines) == expected_lines
    assert tokenized.visible_text == expected_visible
    assert tokenized.source_utf8 == source.encode("utf-8")


def test_tokenize_source_preserves_utf8_ranges_and_visible_case() -> None:
    source = "\u00c9ire\r\n<i>Mixed Case</i> [** proof note]\r"
    tokenized = tokenize_source_page("001.png", source)

    assert tuple(line.visible_text for line in tokenized.lines) == ("\u00c9ire", "Mixed Case ", "")
    assert tokenized.operations[0].byte_start == 0
    assert tokenized.operations[-1].byte_end == len(tokenized.source_utf8)
    assert all(left.byte_end == right.byte_start for left, right in pairwise(tokenized.operations))
    assert all(
        operation.output
        == tokenized.source_utf8[operation.byte_start : operation.byte_end].decode("utf-8")
        for operation in tokenized.operations
        if operation.kind == "copy"
    )
    assert "".join(operation.output for operation in tokenized.operations) == tokenized.visible_text


def test_tokenize_source_preserves_ligatures_punctuation_and_hyphenation() -> None:
    source = "\ufb02ower's well-\nknown co\u00f6peration"
    tokenized = tokenize_source_page("001.png", source)

    assert tokenized.visible_text == "\ufb02ower's well-\nknown co\u00f6peration"
    assert tuple(line.visible_text for line in tokenized.lines) == (
        "\ufb02ower's well-",
        "known co\u00f6peration",
    )


def test_tokenize_source_records_lines_with_half_open_utf8_spans() -> None:
    source = "\u00e9\r\nA\rB\n"
    tokenized = tokenize_source_page("001.png", source)

    assert [
        (
            line.ordinal,
            line.byte_start,
            line.byte_end,
            line.content_byte_start,
            line.content_byte_end,
            line.separator_byte_start,
            line.separator_byte_end,
            line.original_text,
        )
        for line in tokenized.lines
    ] == [
        (0, 0, 4, 0, 2, 2, 4, "\u00e9"),
        (1, 4, 6, 4, 5, 5, 6, "A"),
        (2, 6, 8, 6, 7, 7, 8, "B"),
        (3, 8, 8, 8, 8, 8, 8, ""),
    ]


def test_tokenize_source_deletes_recognized_tags_and_records_style_runs() -> None:
    source = "<i class='roman'>italics</i> <B>bold</b> <sc>CAPS</SC> <f x=1>font</f> <g>Greek</g>"
    tokenized = tokenize_source_page("001.png", source)

    assert tokenized.visible_text == "italics bold CAPS font Greek"
    assert [
        (run.kind, run.normalized_start, run.normalized_end) for run in tokenized.style_runs
    ] == [
        ("i", 0, 7),
        ("b", 8, 12),
        ("sc", 13, 17),
        ("f", 18, 22),
        ("g", 23, 28),
    ]
    assert [(run.byte_start, run.byte_end) for run in tokenized.style_runs] == [
        (17, 24),
        (32, 36),
        (45, 49),
        (62, 66),
        (74, 79),
    ]


def test_tokenize_source_records_standalone_tb_as_zero_width_style_run() -> None:
    tokenized = tokenize_source_page("001.png", "before <tb> after")

    assert tokenized.visible_text == "before  after"
    assert [
        (run.kind, run.normalized_start, run.normalized_end) for run in tokenized.style_runs
    ] == [("tb", 7, 7)]
    assert tokenized.lines[0].style_fitting_eligible is True


def test_tokenize_source_deletes_known_self_closing_tags() -> None:
    tokenized = tokenize_source_page("001.png", "<i/>visible<tb/> <u/>")

    assert tokenized.visible_text == "visible <u/>"
    assert [
        (run.kind, run.normalized_start, run.normalized_end) for run in tokenized.style_runs
    ] == [("tb", 7, 7)]
    assert [diagnostic.code for diagnostic in tokenized.diagnostics] == ["unknown_tag"]


def test_tokenize_source_keeps_unknown_tag_visible_and_marks_line_ineligible() -> None:
    tokenized = tokenize_source_page("001.png", "before <u>under</u> after")

    assert tokenized.visible_text == "before <u>under</u> after"
    assert [diagnostic.code for diagnostic in tokenized.diagnostics] == [
        "unknown_tag",
        "unknown_tag",
    ]
    assert tokenized.lines[0].matching_eligible is True
    assert tokenized.lines[0].style_fitting_eligible is False


@pytest.mark.parametrize(
    "source",
    [
        "[** note with /* unclosed control]",
        "<i data='/*'>visible</i>",
    ],
)
def test_tokenize_source_marks_raw_control_errors_ineligible_for_matching(source: str) -> None:
    tokenized = tokenize_source_page("001.png", source)

    assert "malformed_control" in [diagnostic.code for diagnostic in tokenized.diagnostics]
    assert tokenized.lines[0].matching_eligible is False
    assert tokenized.lines[0].style_fitting_eligible is False


def test_tokenize_source_marks_every_line_in_an_invalid_control_block_ineligible() -> None:
    source = "start /* outer\nmiddle /# nested\nend */ tail\noutside"

    tokenized = tokenize_source_page("001.png", source)

    assert [line.matching_eligible for line in tokenized.lines] == [False, False, False, True]
    assert [line.style_fitting_eligible for line in tokenized.lines] == [False, False, False, True]


def test_tokenize_source_marks_unclosed_control_through_page_end_ineligible() -> None:
    tokenized = tokenize_source_page("001.png", "start /# continued\nstill active\n")

    assert [line.matching_eligible for line in tokenized.lines[:2]] == [False, False]


def test_tokenize_source_splits_unknown_tags_around_valid_raw_controls() -> None:
    tokenized = tokenize_source_page("001.png", '<u data="/*">body*/</u>')

    assert tokenized.visible_text == '<u data="">body</u>'
    assert [operation.kind for operation in tokenized.operations] == [
        "copy",
        "delete_control",
        "copy",
        "copy",
        "delete_control",
        "copy",
    ]
    assert tokenized.lines[0].matching_eligible is True
    assert tokenized.lines[0].style_fitting_eligible is False


def test_tokenize_source_splits_proof_notes_around_valid_raw_controls() -> None:
    tokenized = tokenize_source_page("001.png", "[** note /* hidden */ text]")

    assert tokenized.visible_text == ""
    assert [operation.kind for operation in tokenized.operations] == [
        "delete_note",
        "delete_control",
        "delete_note",
        "delete_control",
        "delete_note",
    ]


def test_tokenize_source_splits_malformed_markup_around_valid_raw_controls() -> None:
    tokenized = tokenize_source_page("001.png", "<i broken /* body */")

    assert tokenized.visible_text == "<i broken  body "
    assert [operation.kind for operation in tokenized.operations] == [
        "copy",
        "delete_control",
        "copy",
        "delete_control",
    ]
    assert tokenized.lines[0].matching_eligible is True
    assert tokenized.lines[0].style_fitting_eligible is False


def test_tokenize_source_handles_a_large_control_rich_page_losslessly() -> None:
    source = "\n".join(f'<u data="/*">line {ordinal}*/</u>' for ordinal in range(256))

    tokenized = tokenize_source_page("001.png", source)

    assert len(tokenized.lines) == 256
    assert all(line.matching_eligible for line in tokenized.lines)
    assert all(not line.style_fitting_eligible for line in tokenized.lines)
    assert len(tokenized.operations) == 7 * 256 - 1
    assert "".join(operation.output for operation in tokenized.operations) == tokenized.visible_text


def test_tokenize_source_handles_tag_dense_single_line_input() -> None:
    source = "".join(f"<u data='{ordinal}'>" for ordinal in range(512))

    tokenized = tokenize_source_page("001.png", source)

    assert tokenized.visible_text == source
    assert len(tokenized.lines) == 1
    assert len(tokenized.operations) == 512
    assert len(tokenized.diagnostics) == 512


@pytest.mark.parametrize(
    "source",
    [
        "before <i broken",
        "before [** proof note",
        "before /* block",
        "before */ block",
    ],
)
def test_tokenize_source_preserves_malformed_markup(source: str) -> None:
    tokenized = tokenize_source_page("001.png", source)

    assert tokenized.visible_text == source
    assert tokenized.lines[0].style_fitting_eligible is False
    assert tokenized.diagnostics


def test_tokenize_source_preserves_all_controls_from_malformed_nested_block() -> None:
    source = "before /* outer /# inner */ after"

    tokenized = tokenize_source_page("001.png", source)

    assert tokenized.visible_text == source
    assert {operation.kind for operation in tokenized.operations} == {"copy"}
    assert tokenized.lines[0].style_fitting_eligible is False


def test_normalization_operations_are_a_complete_byte_partition() -> None:
    source = "a\u00e9\r\n<i>b</i>[** note]/* c */"
    tokenized = tokenize_source_page("001.png", source)

    assert [operation.byte_start for operation in tokenized.operations] == [
        0,
        3,
        5,
        8,
        9,
        13,
        22,
        24,
        27,
    ]
    assert [operation.byte_end for operation in tokenized.operations] == [
        3,
        5,
        8,
        9,
        13,
        22,
        24,
        27,
        29,
    ]
    assert [operation.kind for operation in tokenized.operations] == [
        "copy",
        "normalize_newline",
        "delete_tag",
        "copy",
        "delete_tag",
        "delete_note",
        "delete_control",
        "copy",
        "delete_control",
    ]


def test_source_records_are_immutable() -> None:
    tokenized = tokenize_source_page("001.png", "text")

    with pytest.raises(FrozenInstanceError):
        type(tokenized.lines[0]).__setattr__(tokenized.lines[0], "visible_text", "changed")


def test_continued_span_id_uses_opening_page_and_utf8_byte() -> None:
    result = tokenize_f2_pages({"p1.png": "\u00e9 /#alpha", "p2.png": "beta#/ tail"})

    first = result.pages[0].formatting_spans[0]
    second = result.pages[1].formatting_spans[0]
    assert first.opening_id == second.opening_id == "continued:p1.png:3"
    assert first.block_id == second.block_id == "continued:p1.png:3:p2.png:6"
    assert first.byte_start == len("\u00e9 /#".encode("utf-8"))
    assert second.byte_end == len(b"beta")
    assert first.closed is True
    assert second.closed is True


def test_tokenize_f2_pages_deletes_local_controls_after_a_utf8_prefix() -> None:
    result = tokenize_f2_pages({"p1.png": "\u00e9 /*local*/ tail"})

    assert result.pages[0].visible_text == "\u00e9 local tail"


def test_tokenize_f2_pages_deletes_continued_controls_after_a_utf8_prefix() -> None:
    result = tokenize_f2_pages({"p1.png": "\u00e9 /#first", "p2.png": "last#/ tail"})

    assert [page.visible_text for page in result.pages] == ["\u00e9 first", "last tail"]


def test_tokenize_f2_pages_orders_spans_naturally_and_keeps_local_identity() -> None:
    result = tokenize_f2_pages(
        {
            "p10.png": "z /*last*/",
            "p2.png": "\u00e9 /*two*/",
            "p1.png": "/*one*/",
        }
    )

    assert [page.page_name for page in result.pages] == ["p1.png", "p2.png", "p10.png"]
    assert [page.formatting_spans[0].opening_id for page in result.pages] == [
        "local:p1.png:0",
        "local:p2.png:3",
        "local:p10.png:2",
    ]
    assert [page.formatting_spans[0].block_id for page in result.pages] == [
        "local:p1.png:0:p1.png:7",
        "local:p2.png:3:p2.png:10",
        "local:p10.png:2:p10.png:10",
    ]
    assert all(page.lines[0].formatting_span_ids for page in result.pages)
    assert all(not page.lines[0].continued_block_ids for page in result.pages)


def test_continued_spans_retain_identity_for_intermediate_pages_without_joining_text() -> None:
    result = tokenize_f2_pages(
        {
            "p3.png": "three",
            "p1.png": "before /#one",
            "p2.png": "two",
            "p4.png": "four#/ after",
        }
    )

    spans = [page.formatting_spans[0] for page in result.pages]
    assert [(span.page_name, span.byte_start, span.byte_end) for span in spans] == [
        ("p1.png", len(b"before /#"), len(b"before /#one")),
        ("p2.png", 0, len(b"two")),
        ("p3.png", 0, len(b"three")),
        ("p4.png", 0, len(b"four")),
    ]
    assert {span.opening_id for span in spans} == {"continued:p1.png:7"}
    assert {span.block_id for span in spans} == {"continued:p1.png:7:p4.png:6"}
    assert [page.visible_text for page in result.pages] == [
        "before one",
        "two",
        "three",
        "four after",
    ]
    assert all(
        page.lines[0].continued_block_ids == ("continued:p1.png:7",) for page in result.pages
    )


@pytest.mark.parametrize(
    "source_pages",
    [
        {"p1.png": "bad */\nnext"},
        {"p1.png": "/* outer\n/# nested\nend */"},
        {"p1.png": "/* outer\nend #/"},
        {"p1.png": "/# start", "p2.png": "still open"},
    ],
)
def test_invalid_control_segments_keep_evidence_and_exclude_affected_lines(
    source_pages: dict[str, str],
) -> None:
    result = tokenize_f2_pages(source_pages)

    assert result.diagnostics
    assert all(diagnostic.page_name in source_pages for diagnostic in result.diagnostics)
    assert all(diagnostic.byte_end >= diagnostic.byte_start for diagnostic in result.diagnostics)
    assert any(page.formatting_spans for page in result.pages)
    assert any(not line.matching_eligible for page in result.pages for line in page.lines)


def test_tokenize_f2_pages_preserves_valid_f2_feature_parser_output() -> None:
    source_pages = {
        "p2.png": "first /#<i>styled</i>",
        "p10.png": "last #/ plain",
    }
    tokenized = tokenize_f2_pages(source_pages)

    assert [page.visible_text for page in tokenized.pages] == ["first styled", "last  plain"]
    assert [page.source_utf8.decode("utf-8") for page in tokenized.pages] == [
        source_pages["p2.png"],
        source_pages["p10.png"],
    ]
    parsed = parse_f2_pages(source_pages)
    assert [page.feature_text for page in parsed.pages] == ["first <i>styled</i>", "last  plain"]
    assert [page.special_blocks for page in parsed.pages] == [("<i>styled</i>",), ("last ",)]
