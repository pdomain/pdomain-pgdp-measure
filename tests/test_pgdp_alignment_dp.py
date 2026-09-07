from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from pdomain_pgdp_measure import alignment_dp
from pdomain_pgdp_measure.alignment_dp import (
    AlignmentResult,
    assess_alignment,
    derive_source_feature_exclusions,
    to_feature_page,
)
from pdomain_pgdp_measure.alignment_dp import (
    align_sequences as strict_align_sequences,
)
from pdomain_pgdp_measure.alignment_image import LineCandidate
from pdomain_pgdp_measure.alignment_source import (
    SourceLine,
    TokenizedSourcePage,
    tokenize_f2_pages,
)
from pdomain_pgdp_measure.f2 import parse_f2_pages


def _line(
    ordinal: int,
    text: str,
    *,
    matching_eligible: bool = True,
) -> SourceLine:
    return SourceLine(
        page_name="001",
        ordinal=ordinal,
        byte_start=ordinal * 10,
        byte_end=ordinal * 10 + len(text),
        content_byte_start=ordinal * 10,
        content_byte_end=ordinal * 10 + len(text),
        separator_byte_start=ordinal * 10 + len(text),
        separator_byte_end=ordinal * 10 + len(text),
        original_text=text,
        visible_text=text,
        operation_ordinals=(),
        formatting_span_ids=(),
        continued_block_ids=(),
        matching_eligible=matching_eligible,
        style_fitting_eligible=True,
    )


def _candidate(
    ordinal: int, *, x0: int = 0, width: int = 10, y0: int | None = None
) -> LineCandidate:
    top = ordinal * 10 if y0 is None else y0
    return LineCandidate(
        band_ordinal=ordinal,
        box=(x0, top, x0 + width, top + 10),
        component_ordinal=0,
        component_count=1,
        foreground_pixels=width * 5,
        width=width,
        height=10,
        fill_ratio=0.5,
        horizontal_ink_profile=tuple(1.0 / width for _ in range(width)),
    )


def _synthetic_page(source_lines: tuple[SourceLine, ...]) -> TokenizedSourcePage:
    return tokenize_f2_pages({"001": "\n".join(line.original_text for line in source_lines)}).pages[
        0
    ]


def align_sequences(
    source_lines: tuple[SourceLine, ...],
    candidates: tuple[LineCandidate, ...],
    *,
    foreground_bounds: tuple[int, int, int, int],
    inherited_exclusions: tuple[str, ...] = (),
    probable_multi_column: bool = False,
    source_page: TokenizedSourcePage | None = None,
) -> AlignmentResult:
    page = _synthetic_page(source_lines) if source_page is None else source_page
    return strict_align_sequences(
        page.lines,
        candidates,
        foreground_bounds=foreground_bounds,
        inherited_exclusions=inherited_exclusions,
        probable_multi_column=probable_multi_column,
        source_page=page,
    )


def test_alignment_requires_an_immutable_tokenized_source_page() -> None:
    keyword_defaults = strict_align_sequences.__kwdefaults__

    assert keyword_defaults is not None
    assert "source_page" not in keyword_defaults


def test_exact_lines_match_with_zero_cost() -> None:
    result = align_sequences(
        tuple(_line(index, "abcdefghij") for index in range(4)),
        tuple(_candidate(index) for index in range(4)),
        foreground_bounds=(0, 0, 10, 60),
    )

    assert [operation.kind for operation in result.operations] == ["match"] * 4
    assert result.total_cost == 0.0
    assert result.normalized_cost == 0.0
    assert result.matched_ratio == 1.0


def test_equal_cost_transitions_prefer_match_then_skip_image_then_skip_text() -> None:
    source = (
        _line(0, "         a"),
        _line(1, "         a"),
        _line(2, "b"),
        _line(3, "         a"),
    )
    candidates = (
        _candidate(0, x0=10, width=10),
        _candidate(1, x0=0, width=1),
        _candidate(2, x0=10, width=10),
        _candidate(3, x0=0, width=1),
    )
    result = align_sequences(source, candidates, foreground_bounds=(0, 0, 10, 50))

    assert result.second_best is not None
    assert result.best.total_cost == result.second_best.total_cost
    assert [operation.kind for operation in result.operations] == [
        "match",
        "skip_image",
        "match",
        "match",
        "skip_text",
    ]
    assert [operation.kind for operation in result.second_best.operations] == [
        "match",
        "skip_text",
        "match",
        "match",
        "skip_image",
    ]


def test_dp_reconstructs_known_equal_cost_complete_paths() -> None:
    source = (
        _line(0, "         a"),
        _line(1, "         a"),
        _line(2, "b"),
        _line(3, "         a"),
    )
    candidates = (
        _candidate(0, x0=10, width=10),
        _candidate(1, x0=0, width=1),
        _candidate(2, x0=10, width=10),
        _candidate(3, x0=0, width=1),
    )

    result = align_sequences(source, candidates, foreground_bounds=(0, 0, 10, 50))

    assert result.best.total_cost == 2.0
    assert result.second_best is not None
    assert result.second_best.total_cost == result.best.total_cost
    assert [
        (operation.kind, operation.source_ordinal, operation.candidate_ordinal)
        for operation in result.best.operations
    ] == [
        ("match", 0, 0),
        ("skip_image", None, 1),
        ("match", 1, 2),
        ("match", 2, 3),
        ("skip_text", 3, None),
    ]
    assert [
        (operation.kind, operation.source_ordinal, operation.candidate_ordinal)
        for operation in result.second_best.operations
    ] == [
        ("match", 0, 0),
        ("skip_text", 1, None),
        ("match", 2, 1),
        ("match", 3, 2),
        ("skip_image", None, 3),
    ]


def test_equal_cost_paths_with_unequal_depths_preserve_tuple_order() -> None:
    page = tokenize_f2_pages(
        {"001": "a" * 1000 + "\n" + "a" * 1000 + "\n\n\n\n         b\nc"}
    ).pages[0]
    candidates = (
        _candidate(0, width=1000),
        _candidate(1, x0=4222, width=1000),
        _candidate(2, width=1000),
        _candidate(3, width=1000),
    )

    result = align_sequences(
        page.lines,
        candidates,
        foreground_bounds=(0, 0, 1000, 40),
        source_page=page,
    )

    assert result.second_best is not None
    assert result.best.total_cost == result.second_best.total_cost
    assert [operation.kind for operation in result.best.operations] == [
        "match",
        "match",
        "match",
        "match",
    ]
    assert [operation.kind for operation in result.second_best.operations] == [
        "match",
        "skip_image",
        "match",
        "skip_text",
        "match",
    ]


def test_equal_cost_paths_support_sparse_source_ordinals() -> None:
    original = tokenize_f2_pages({"001": "         a\n         a\nb\n         a"}).pages[0]
    sparse_ordinals = (7, 19, 43, 97)
    page = replace(
        original,
        lines=tuple(
            replace(line, ordinal=ordinal)
            for line, ordinal in zip(original.lines, sparse_ordinals, strict=True)
        ),
    )
    candidates = (
        _candidate(0, x0=10, width=10),
        _candidate(1, x0=0, width=1),
        _candidate(2, x0=10, width=10),
        _candidate(3, x0=0, width=1),
    )

    result = align_sequences(
        page.lines,
        candidates,
        foreground_bounds=(0, 0, 10, 50),
        source_page=page,
    )

    assert result.second_best is not None
    assert result.best.total_cost == result.second_best.total_cost == 2.0
    assert [
        (operation.kind, operation.source_ordinal, operation.candidate_ordinal)
        for operation in result.best.operations
    ] == [
        ("match", 7, 0),
        ("skip_image", None, 1),
        ("match", 19, 2),
        ("match", 43, 3),
        ("skip_text", 97, None),
    ]
    assert [
        (operation.kind, operation.source_ordinal, operation.candidate_ordinal)
        for operation in result.second_best.operations
    ] == [
        ("match", 7, 0),
        ("skip_text", 19, None),
        ("match", 43, 1),
        ("match", 97, 2),
        ("skip_image", None, 3),
    ]


def test_equal_cost_complete_paths_prefer_match_before_skip_image() -> None:
    result = align_sequences(
        tuple(_line(index, "abcdefghij") for index in range(4)),
        tuple(_candidate(index) for index in range(5)),
        foreground_bounds=(0, 0, 10, 60),
    )

    assert [operation.kind for operation in result.operations] == [
        "match",
        "match",
        "match",
        "match",
        "skip_image",
    ]
    assert result.second_best is not None
    assert [operation.kind for operation in result.second_best.operations] == [
        "match",
        "match",
        "match",
        "skip_image",
        "match",
    ]


def test_missing_source_line_records_skip_text() -> None:
    result = align_sequences(
        tuple(_line(index, "abcdefghij") for index in range(4)),
        tuple(_candidate(index) for index in range(3)),
        foreground_bounds=(0, 0, 10, 50),
    )

    assert [operation.kind for operation in result.operations] == [
        "match",
        "match",
        "match",
        "skip_text",
    ]
    assert result.operations[-1].source_ordinal == 3
    assert result.operations[-1].candidate_ordinal is None


def test_extra_image_row_records_skip_image() -> None:
    result = align_sequences(
        tuple(_line(index, "abcdefghij") for index in range(4)),
        tuple(_candidate(index) for index in range(5)),
        foreground_bounds=(0, 0, 10, 60),
    )

    assert [operation.kind for operation in result.operations] == [
        "match",
        "match",
        "match",
        "match",
        "skip_image",
    ]
    assert result.operations[-1].source_ordinal is None
    assert result.operations[-1].candidate_ordinal == 4


def test_blank_source_lines_drive_gap_cost_and_clamp_after_three() -> None:
    source = (
        _line(0, "abcdefghij"),
        _line(1, ""),
        _line(2, ""),
        _line(3, ""),
        _line(4, ""),
        _line(5, "abcdefghij"),
        _line(6, "abcdefghij"),
        _line(7, "abcdefghij"),
    )
    result = align_sequences(
        source,
        tuple(_candidate(index, y0=index * 40) for index in range(4)),
        foreground_bounds=(0, 0, 10, 150),
    )

    assert result.operations[1].match_cost is not None
    assert result.operations[1].match_cost.gaps == 0.0
    assert result.eligible_source_lines[1].blank_lines_before == 4


def test_proof_note_only_normalized_blank_counts_as_source_gap() -> None:
    source = (
        _line(0, "abcdefghij"),
        _line(1, ""),
        _line(2, "abcdefghij"),
        _line(3, "abcdefghij"),
        _line(4, "abcdefghij"),
    )
    result = align_sequences(
        source,
        (
            _candidate(0, y0=0),
            _candidate(1, y0=30),
            _candidate(2, y0=40),
            _candidate(3, y0=50),
        ),
        foreground_bounds=(0, 0, 10, 60),
    )

    assert result.eligible_source_lines[1].blank_lines_before == 1
    assert result.operations[1].match_cost is not None
    assert result.operations[1].match_cost.gaps == 0.6666666666666667


def test_style_run_changes_the_weak_style_cost() -> None:
    page = tokenize_f2_pages(
        {"001": "<i>abcdefghij</i>\nabcdefghij\nabcdefghij\nabcdefghij"}
    ).pages[0]
    result = align_sequences(
        page.lines,
        tuple(_candidate(index) for index in range(4)),
        foreground_bounds=(0, 0, 10, 50),
        source_page=page,
    )

    assert result.operations[0].match_cost is not None
    assert result.operations[0].match_cost.style == 0.10000000000000009


def test_indent_cost_measures_candidate_offset_from_foreground_origin() -> None:
    result = align_sequences(
        tuple(_line(index, "abcdefghij") for index in range(4)),
        tuple(_candidate(index, x0=5) for index in range(4)),
        foreground_bounds=(5, 0, 15, 50),
    )

    assert result.operations[0].match_cost is not None
    assert result.operations[0].match_cost.indent == 0.0


def test_replays_equal_cost_inputs_deterministically() -> None:
    source = tuple(_line(index, "abcdefghij") for index in range(4))
    candidates = tuple(_candidate(index) for index in range(4))
    results = tuple(
        align_sequences(source, candidates, foreground_bounds=(0, 0, 10, 60)) for _ in range(100)
    )

    assert all(result == results[0] for result in results)


def test_assessment_thresholds_are_inclusive_for_accepted_pages() -> None:
    source = tuple(_line(index, "abcdefghij") for index in range(14))
    candidates = tuple(_candidate(index) for index in range(14))
    result = align_sequences(source, candidates, foreground_bounds=(0, 0, 10, 200))

    assert result.state == "accepted"
    assert result.accepted is True
    assert result.confidence is None


@pytest.mark.parametrize(
    ("line_count", "expected_state", "expected_exclusion"),
    [(80, "accepted", None), (81, "excluded", "line_count_out_of_range")],
)
def test_line_count_boundaries_are_inclusive(
    line_count: int, expected_state: str, expected_exclusion: str | None
) -> None:
    result = align_sequences(
        tuple(_line(index, "abcdefghij") for index in range(line_count)),
        tuple(_candidate(index) for index in range(line_count)),
        foreground_bounds=(0, 0, 10, line_count * 10),
    )

    assert result.state == expected_state
    assert (expected_exclusion in result.exclusions) is (expected_exclusion is not None)


@pytest.mark.parametrize(
    ("source_count", "candidate_count", "expected_exclusion"),
    [
        (3, 3, "line_count_out_of_range"),
        (81, 81, "line_count_out_of_range"),
        (4, 7, "line_count_difference_exceeds_maximum"),
    ],
)
def test_pre_dp_count_exclusions_skip_cost_matrix_and_alignment(
    source_count: int, candidate_count: int, expected_exclusion: str
) -> None:
    source = tuple(_line(index, "abcdefghij") for index in range(source_count))
    candidates = tuple(_candidate(index) for index in range(candidate_count))

    with (
        patch.object(
            alignment_dp,
            "_match_costs",
            side_effect=AssertionError("pre-DP exclusion calculated match costs"),
        ) as match_costs,
        patch.object(
            alignment_dp,
            "_align",
            side_effect=AssertionError("pre-DP exclusion invoked the DP"),
        ) as align,
    ):
        result = align_sequences(
            source,
            candidates,
            foreground_bounds=(0, 0, 10, max(source_count, candidate_count) * 10),
        )

    assert match_costs.call_count == 0
    assert align.call_count == 0
    assert result.state == "excluded"
    assert result.accepted is False
    assert result.exclusions == (expected_exclusion,)
    assert [operation.kind for operation in result.operations] == [
        *["skip_image"] * candidate_count,
        *["skip_text"] * source_count,
    ]
    assert [
        operation.candidate_ordinal for operation in result.operations[:candidate_count]
    ] == list(range(candidate_count))
    assert [operation.source_ordinal for operation in result.operations[candidate_count:]] == list(
        range(source_count)
    )


def test_pre_dp_exclusion_retains_derived_source_evidence() -> None:
    page = tokenize_f2_pages({"001": "[Illustration: a plate]\n" + "abcdefghij\n" * 80}).pages[0]

    result = align_sequences(
        page.lines,
        tuple(_candidate(index) for index in range(81)),
        foreground_bounds=(0, 0, 10, 820),
        source_page=page,
    )

    assert result.state == "excluded"
    assert result.exclusions == ("illustration_marker", "line_count_out_of_range")
    assert result.exclusion_evidence == (
        alignment_dp.SourceFeatureExclusion("illustration_marker", "pgdp-rank/v1", (0,)),
    )
    assert result.operations == result.best.operations
    assert result.operations[0] == alignment_dp.AlignmentOperation(
        kind="skip_image",
        source_ordinal=None,
        candidate_ordinal=0,
        cost=1.0,
    )
    assert result.operations[-1] == alignment_dp.AlignmentOperation(
        kind="skip_text",
        source_ordinal=80,
        candidate_ordinal=None,
        cost=1.0,
    )


def test_80_by_80_alignment_retains_two_reconstructed_paths() -> None:
    source = tuple(_line(index, "abcdefghij") for index in range(80))
    candidates = tuple(_candidate(index) for index in range(80))

    result = align_sequences(source, candidates, foreground_bounds=(0, 0, 10, 810))

    assert result.state == "accepted"
    assert result.total_cost == 0.0
    assert len(result.best.operations) == 80
    assert result.second_best is not None
    assert result.second_best.total_cost == 2.0
    assert len(result.second_best.operations) == 81


def test_dp_reconstructs_operations_only_for_final_paths() -> None:
    source = tuple(_line(index, "abcdefghij") for index in range(4))
    candidates = tuple(_candidate(index) for index in range(4))

    with patch.object(
        alignment_dp,
        "operations_from_node",
        wraps=alignment_dp.operations_from_node,
    ) as operations_from_node:
        result = align_sequences(source, candidates, foreground_bounds=(0, 0, 10, 50))

    assert result.second_best is not None
    assert operations_from_node.call_count == 2


def test_persistent_gutter_excludes_a_page() -> None:
    result = align_sequences(
        tuple(_line(index, "abcdefghij") for index in range(4)),
        tuple(_candidate(index) for index in range(4)),
        foreground_bounds=(0, 0, 10, 50),
        probable_multi_column=True,
    )

    assert result.state == "excluded"
    assert "persistent_gutter" in result.exclusions


def test_table_evidence_excludes_a_page() -> None:
    page = tokenize_f2_pages({"001": "/*left   right\nsecond   field\nthird   value*/"}).pages[0]
    result = align_sequences(
        page.lines,
        tuple(_candidate(index) for index in range(3)),
        foreground_bounds=(0, 0, 10, 50),
        source_page=page,
    )

    assert result.state == "excluded"
    assert "table_like" in result.exclusions
    evidence = next(item for item in result.exclusion_evidence if item.code == "table_like")
    assert evidence.predicate_method == "pgdp-rank/v1"
    assert evidence.source_ordinals == (0, 1, 2)


def test_illustration_evidence_excludes_a_page() -> None:
    page = tokenize_f2_pages({"001": "first\n[Illustration: a plate]\nsecond\nthird"}).pages[0]
    result = align_sequences(
        page.lines,
        tuple(_candidate(index) for index in range(4)),
        foreground_bounds=(0, 0, 10, 50),
        source_page=page,
    )

    assert result.state == "excluded"
    evidence = next(
        item for item in result.exclusion_evidence if item.code == "illustration_marker"
    )
    assert evidence.source_ordinals == (1,)


def test_source_feature_exclusions_have_verified_source_ordinals() -> None:
    page = tokenize_f2_pages(
        {"001": "/*left   right\nsecond   field\nthird   value*/\n[Decoration]"}
    ).pages[0]

    evidence = derive_source_feature_exclusions(page)

    assert [(item.code, item.source_ordinals) for item in evidence] == [
        ("table_like", (0, 1, 2)),
        ("illustration_marker", (3,)),
    ]


def test_assessment_accepts_actual_dp_cost_boundary() -> None:
    source = tuple(_line(index, "abcdefghij") for index in range(4))
    candidates = (
        _candidate(0, x0=2, width=2, y0=0),
        _candidate(1, x0=2, width=2, y0=10),
        _candidate(2, x0=1, width=2, y0=27),
        _candidate(3, x0=0, width=2, y0=47),
    )

    result = align_sequences(source, candidates, foreground_bounds=(0, 0, 2, 60))

    assert result.total_cost == 0.88
    assert max(len(result.eligible_source_lines), len(result.candidates)) == 4
    assert result.normalized_cost == 0.22
    assert result.state == "accepted"
    assert result.accepted is True


def test_assessment_accepts_actual_dp_margin_boundary() -> None:
    source = tuple(_line(index, "abcdefghij") for index in range(4))
    candidates = (
        _candidate(0, width=7, y0=0),
        _candidate(1, width=7, y0=10),
        _candidate(2, width=7, y0=20),
        _candidate(3, x0=2, width=7, y0=30),
        _candidate(4, x0=4, width=6, y0=40),
    )

    result = align_sequences(source, candidates, foreground_bounds=(0, 0, 7, 50))

    assert result.second_best is not None
    assert max(len(result.eligible_source_lines), len(result.candidates)) == 5
    # Accepted paths with a real ambiguity have a total cost of at least one
    # skip.  Binary64 subtraction at that magnitude cannot equal float(0.15),
    # so this is the nearest passing result without rounding production data.
    assert result.uniqueness_margin == pytest.approx(0.15)
    assert result.uniqueness_margin >= 0.15
    assert result.state == "accepted"


def test_assessment_proposes_low_match_ratio() -> None:
    source = tuple(_line(index, "abcdefghij") for index in range(4))
    candidates = tuple(_candidate(index) for index in range(4))
    initial = align_sequences(source, candidates, foreground_bounds=(0, 0, 10, 50))

    result = assess_alignment(replace(initial, matched_ratio=0.89))

    assert result.state == "proposed"
    assert "matched_ratio_below_minimum" in result.exclusions


def test_assessment_proposes_excessive_count_difference() -> None:
    source = tuple(_line(index, "abcdefghij") for index in range(4))
    candidates = tuple(_candidate(index) for index in range(4))
    initial = align_sequences(source, candidates, foreground_bounds=(0, 0, 10, 50))

    result = assess_alignment(
        replace(initial, candidates=initial.candidates + initial.candidates[:3])
    )

    assert result.state == "proposed"
    assert "line_count_difference_exceeds_maximum" in result.exclusions


def test_excludes_too_few_lines_and_inherited_page_exclusion() -> None:
    result = align_sequences(
        tuple(_line(index, "abcdefghij") for index in range(3)),
        tuple(_candidate(index) for index in range(3)),
        foreground_bounds=(0, 0, 10, 50),
        inherited_exclusions=("border_dominated",),
    )

    assert result.state == "excluded"
    assert result.accepted is False
    assert result.exclusions == ("border_dominated", "line_count_out_of_range")


def test_excludes_a_source_page_with_malformed_control_line() -> None:
    page = tokenize_f2_pages({"001": "first\n/* broken\nthird\nfourth"}).pages[0]
    result = align_sequences(
        page.lines,
        tuple(_candidate(index) for index in range(4)),
        foreground_bounds=(0, 0, 10, 50),
        source_page=page,
    )

    assert result.state == "excluded"
    assert "malformed_control" in result.exclusions


def test_feature_adapter_uses_tokenized_spans_without_reparsing_live_source() -> None:
    source_pages = {"p2.png": "first /#<i>styled</i>", "p10.png": "last #/ plain"}
    tokenized = tokenize_f2_pages(source_pages)
    expected = parse_f2_pages(source_pages).pages

    actual = tuple(to_feature_page(page) for page in tokenized.pages)

    assert actual == expected
