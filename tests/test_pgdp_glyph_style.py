"""Tests for resolving a cut glyph's face from PGDP's own F2 markup."""

from __future__ import annotations

from pdomain_pgdp_measure.alignment_models import (
    AlignmentOperation,
    PageAlignment,
    ProjectAlignment,
    WireSourceLine,
    WireStyleRun,
)
from pdomain_pgdp_measure.alignment_source import tokenize_source_page
from pdomain_pgdp_measure.glyph_style import (
    ROMAN,
    read_alignment_style,
    styles_for_word,
    word_character_offsets,
)

_PAGE = "p001.png"


def _book(f2_text: str) -> ProjectAlignment:
    """One page aligned from F2 text, built the way the aligner builds it."""

    page = tokenize_source_page(_PAGE, f2_text)
    lines = tuple(
        WireSourceLine(
            ordinal=line.ordinal,
            byte_start=line.byte_start,
            byte_end=line.byte_end,
            content_byte_start=line.content_byte_start,
            content_byte_end=line.content_byte_end,
            separator_byte_start=line.separator_byte_start,
            separator_byte_end=line.separator_byte_end,
            original_text=line.original_text,
            separator_text=page.source_utf8[
                line.separator_byte_start : line.separator_byte_end
            ].decode("utf-8"),
            visible_text=line.visible_text,
            operation_ordinals=line.operation_ordinals,
            formatting_span_ids=line.formatting_span_ids,
            continued_block_ids=line.continued_block_ids,
            matching_eligible=line.matching_eligible,
            style_fitting_eligible=line.style_fitting_eligible,
        )
        for line in page.lines
    )
    runs = tuple(
        WireStyleRun(
            kind=run.kind,
            normalized_start=run.normalized_start,
            normalized_end=run.normalized_end,
            byte_start=run.byte_start,
            byte_end=run.byte_end,
        )
        for run in page.style_runs
    )
    operations = tuple(
        AlignmentOperation(
            kind="skip_text", source_ordinal=line.ordinal, candidate_ordinal=None, cost=0.0
        )
        for line in lines
        if line.matching_eligible and line.visible_text.strip()
    )
    aligned = PageAlignment(
        page_name=_PAGE,
        f2_sha256="a" * 64,
        scan_sha256=None,
        source_frame=None,
        source_lines=lines,
        formatting_spans=(),
        candidates=(),
        operations=operations,
        total_cost=0.0,
        second_best_cost=None,
        normalized_cost=0.0,
        matched_count=0,
        matched_ratio=0.0,
        uniqueness_margin=0.0,
        mean_width_residual=None,
        mean_indentation_residual=None,
        state="excluded",
        accepted=False,
        exclusions=("no_candidates",),
        style_runs=runs,
    )
    return ProjectAlignment(project_id="projectIDone", pages=(aligned,))


def test_word_offsets_survive_collapsed_whitespace() -> None:
    assert word_character_offsets("  one   two three ") == (2, 8, 12)
    assert word_character_offsets("") == ()
    assert word_character_offsets("   ") == ()


def test_an_unmarked_line_reads_roman() -> None:
    style = read_alignment_style(_book("plain words here\n"))
    assert (
        styles_for_word(
            style,
            page_name=_PAGE,
            source_ordinal=0,
            visible_text="plain words here",
            word_ordinal=1,
            positions=[0, 1, 2, 3, 4],
        )
        == (ROMAN,) * 5
    )


def test_an_italic_word_reads_italic() -> None:
    style = read_alignment_style(_book("plain <i>words</i> here\n"))
    assert (
        styles_for_word(
            style,
            page_name=_PAGE,
            source_ordinal=0,
            visible_text="plain words here",
            word_ordinal=1,
            positions=[0, 1, 2, 3, 4],
        )
        == ("italic",) * 5
    )


def test_small_caps_are_named_rather_than_folded_into_roman() -> None:
    style = read_alignment_style(_book("<sc>Lowther Street</sc>, November\n"))
    visible = "Lowther Street, November"
    assert (
        styles_for_word(
            style,
            page_name=_PAGE,
            source_ordinal=0,
            visible_text=visible,
            word_ordinal=0,
            positions=[0, 1, 2],
        )
        == ("small_caps",) * 3
    )
    assert styles_for_word(
        style,
        page_name=_PAGE,
        source_ordinal=0,
        visible_text=visible,
        word_ordinal=2,
        positions=[0],
    ) == (ROMAN,)


def test_a_style_ending_mid_word_only_covers_its_own_characters() -> None:
    style = read_alignment_style(_book("<i>ab</i>cd\n"))
    assert styles_for_word(
        style,
        page_name=_PAGE,
        source_ordinal=0,
        visible_text="abcd",
        word_ordinal=0,
        positions=[0, 1, 2, 3],
    ) == ("italic", "italic", ROMAN, ROMAN)


def test_nested_styles_are_reported_together() -> None:
    style = read_alignment_style(_book("<sc><i>both</i></sc>\n"))
    assert styles_for_word(
        style,
        page_name=_PAGE,
        source_ordinal=0,
        visible_text="both",
        word_ordinal=0,
        positions=[0],
    ) == ("italic+small_caps",)


def test_a_proofer_note_mentioning_a_tag_is_not_a_tag() -> None:
    """The `<sc>` inside a proofer's query is text, and the note never reaches the visible line."""

    style = read_alignment_style(_book("Lowther Street[**mark place in <sc>?], November\n"))
    visible = "Lowther Street, November"
    assert style.line_agrees(_PAGE, 0, visible)
    assert (
        styles_for_word(
            style,
            page_name=_PAGE,
            source_ordinal=0,
            visible_text=visible,
            word_ordinal=0,
            positions=[0, 1, 2],
        )
        == (ROMAN,) * 3
    )


def test_a_line_that_does_not_match_the_alignment_yields_no_style() -> None:
    style = read_alignment_style(_book("plain <i>words</i> here\n"))
    assert styles_for_word(
        style,
        page_name=_PAGE,
        source_ordinal=0,
        visible_text="something else entirely",
        word_ordinal=0,
        positions=[0],
    ) == (None,)


def test_no_style_source_yields_none_rather_than_roman() -> None:
    assert styles_for_word(
        None,
        page_name=_PAGE,
        source_ordinal=0,
        visible_text="a b",
        word_ordinal=0,
        positions=[0],
    ) == (None,)


def test_a_word_ordinal_past_the_line_yields_no_style() -> None:
    style = read_alignment_style(_book("two words\n"))
    assert styles_for_word(
        style,
        page_name=_PAGE,
        source_ordinal=0,
        visible_text="two words",
        word_ordinal=9,
        positions=[0],
    ) == (None,)


def test_the_book_reports_how_many_lines_carry_style() -> None:
    style = read_alignment_style(_book("plain <i>words</i> here\nno markup here\n"))
    assert style.styled_line_count == 1
    assert style.project_id == "projectIDone"


def test_a_report_with_no_style_runs_still_reads_roman() -> None:
    style = read_alignment_style(_book("plain words here\n"))
    assert style.styled_line_count == 0
    assert style.style_at(_PAGE, 0, character_offset=0) == ROMAN
