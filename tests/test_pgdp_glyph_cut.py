"""Tests for cutting a word box into per-glyph boxes at its blank columns.

Every mask here is built inline. Nothing reads the corpus or opens an image.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdomain_pgdp_measure.glyph_cut import (
    GlyphCut,
    WordCut,
    crop_to_ink,
    cut_word_glyphs,
    ink_column_runs,
    label_letters,
)


def _mask(rows: list[str]) -> np.ndarray[tuple[int, int], np.dtype[np.bool]]:
    """Build a page ink mask from a picture, where `#` is ink and `.` is paper."""

    return np.array([[character == "#" for character in row] for row in rows], dtype=np.bool_)


_THREE_SEPARATE_LETTERS = [
    "..........",
    ".#..#..#..",
    ".#..#..#..",
    ".#..#..#..",
    "..........",
]

_TWO_TOUCHING_LETTERS = [
    "..........",
    ".####.....",
    ".####.....",
    "..........",
]


def test_a_separable_word_yields_one_glyph_per_letter() -> None:
    cut = cut_word_glyphs(_mask(_THREE_SEPARATE_LETTERS), box=(0, 0, 10, 5), text="ash")
    assert cut.separable
    assert [glyph.character for glyph in cut.glyphs] == ["a", "s", "h"]
    assert [glyph.ordinal for glyph in cut.glyphs] == [0, 1, 2]


def test_each_glyph_box_is_tight_to_its_own_ink() -> None:
    cut = cut_word_glyphs(_mask(_THREE_SEPARATE_LETTERS), box=(0, 0, 10, 5), text="ash")
    assert [glyph.box for glyph in cut.glyphs] == [(1, 1, 2, 4), (4, 1, 5, 4), (7, 1, 8, 4)]


def test_glyph_boxes_are_translated_into_the_page_frame() -> None:
    page = np.zeros((20, 30), dtype=np.bool_)
    page[5:9, 10:20] = _mask(_THREE_SEPARATE_LETTERS)[:4, :]
    cut = cut_word_glyphs(page, box=(10, 5, 20, 9), text="ash")
    assert [glyph.box for glyph in cut.glyphs] == [(11, 6, 12, 9), (14, 6, 15, 9), (17, 6, 18, 9)]


def test_a_touching_word_is_refused_and_records_both_counts() -> None:
    cut = cut_word_glyphs(_mask(_TWO_TOUCHING_LETTERS), box=(0, 0, 10, 4), text="in")
    assert cut.glyphs == ()
    assert cut.reason == "run_count_disagrees"
    assert (cut.run_count, cut.position_count) == (1, 2)


def test_a_word_that_splits_into_too_many_runs_is_refused() -> None:
    cut = cut_word_glyphs(_mask(_THREE_SEPARATE_LETTERS), box=(0, 0, 10, 5), text="in")
    assert cut.reason == "run_count_disagrees"
    assert (cut.run_count, cut.position_count) == (3, 2)
    assert not cut.separable


def test_a_word_that_splits_into_too_few_runs_is_refused() -> None:
    cut = cut_word_glyphs(_mask(_THREE_SEPARATE_LETTERS), box=(0, 0, 10, 5), text="ashy")
    assert cut.reason == "run_count_disagrees"
    assert (cut.run_count, cut.position_count) == (3, 4)


def test_a_word_with_no_labelled_letter_is_refused() -> None:
    cut = cut_word_glyphs(_mask(_THREE_SEPARATE_LETTERS), box=(0, 0, 10, 5), text="---")
    assert cut.reason == "no_labelled_letters"
    assert cut.position_count == 3


def test_a_box_with_no_ink_is_refused() -> None:
    cut = cut_word_glyphs(np.zeros((5, 10), dtype=np.bool_), box=(0, 0, 10, 5), text="ash")
    assert cut.reason == "no_ink"
    assert cut.run_count == 0


def test_punctuation_is_dropped_from_the_label_rather_than_counted() -> None:
    assert label_letters("don't,") == "dont"
    assert label_letters("14.") == "14"
    assert label_letters("—") == ""  # EM DASH


def test_ink_column_runs_are_half_open_and_left_to_right() -> None:
    assert ink_column_runs(_mask(_THREE_SEPARATE_LETTERS)) == ((1, 2), (4, 5), (7, 8))


def test_ink_column_runs_of_an_empty_crop_are_empty() -> None:
    assert ink_column_runs(np.zeros((0, 0), dtype=np.bool_)) == ()


def test_crop_to_ink_tightens_a_generous_box() -> None:
    assert crop_to_ink(_mask(_THREE_SEPARATE_LETTERS), box=(0, 0, 10, 5)) == (1, 1, 8, 4)


def test_crop_to_ink_returns_none_for_a_blank_box() -> None:
    assert crop_to_ink(np.zeros((5, 10), dtype=np.bool_), box=(0, 0, 10, 5)) is None


def test_crop_to_ink_returns_none_for_an_empty_box() -> None:
    assert crop_to_ink(_mask(_THREE_SEPARATE_LETTERS), box=(4, 0, 4, 5)) is None


def test_a_cut_may_not_carry_glyphs_and_a_reason_together() -> None:
    with pytest.raises(ValueError, match="exactly when it names no reason"):
        _ = WordCut(
            run_count=1,
            position_count=1,
            glyphs=(GlyphCut(character="a", ordinal=0, box=(0, 0, 1, 1)),),
            reason="no_ink",
        )


def test_a_glyph_carries_exactly_one_character() -> None:
    with pytest.raises(ValueError, match="exactly one character"):
        _ = GlyphCut(character="ab", ordinal=0, box=(0, 0, 1, 1))


def test_a_glyph_box_must_have_positive_extent() -> None:
    with pytest.raises(ValueError, match="positive width and height"):
        _ = GlyphCut(character="a", ordinal=0, box=(0, 0, 0, 1))
