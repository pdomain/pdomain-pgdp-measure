"""Tests for recording a cut glyph against its line's measured bands.

Every mask here is built inline. Nothing reads the corpus or opens an image.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdomain_pgdp_measure.glyph_quality import (
    GlyphQuality,
    LineBand,
    ascender_is_flat,
    character_width_reference,
    is_narrow,
    is_overtall,
    is_wide,
    line_x_height_reference,
    measure_glyph_quality,
    word_ascender_flatness,
)

_LINE = LineBand(box=(0, 10, 100, 30), baseline_row_px=24, x_height_top_row_px=16)


def _mask_with_ink(
    box: tuple[int, int, int, int],
) -> np.ndarray[tuple[int, int], np.dtype[np.bool]]:
    page = np.zeros((40, 100), dtype=np.bool_)
    x_start, y_start, x_end, y_end = box
    page[y_start:y_end, x_start:x_end] = True
    return page


def test_a_glyph_sitting_in_the_x_height_band_raises_no_band_flag() -> None:
    quality = measure_glyph_quality(
        _mask_with_ink((20, 16, 26, 25)), box=(20, 16, 26, 25), line=_LINE
    )
    assert quality.flags == ()
    assert (quality.top_offset_px, quality.bottom_offset_px) == (0, 0)


def test_an_ascender_is_flagged_as_ascending() -> None:
    quality = measure_glyph_quality(
        _mask_with_ink((20, 11, 26, 25)), box=(20, 11, 26, 25), line=_LINE
    )
    assert "ascends" in quality.flags
    assert quality.top_offset_px == -5


def test_a_descender_is_flagged_as_descending() -> None:
    quality = measure_glyph_quality(
        _mask_with_ink((20, 16, 26, 29)), box=(20, 16, 26, 29), line=_LINE
    )
    assert "descends" in quality.flags
    assert quality.bottom_offset_px == 4


def test_a_glyph_that_never_reaches_the_baseline_is_flagged() -> None:
    quality = measure_glyph_quality(
        _mask_with_ink((20, 16, 26, 20)), box=(20, 16, 26, 20), line=_LINE
    )
    assert "short_of_baseline" in quality.flags


def test_a_glyph_one_pixel_outside_a_band_is_within_tolerance() -> None:
    quality = measure_glyph_quality(
        _mask_with_ink((20, 15, 26, 26)), box=(20, 15, 26, 26), line=_LINE
    )
    assert quality.flags == ()


def test_a_glyph_reaching_the_line_box_top_is_flagged() -> None:
    quality = measure_glyph_quality(
        _mask_with_ink((20, 10, 26, 25)), box=(20, 10, 26, 25), line=_LINE
    )
    assert "touches_line_top" in quality.flags


def test_a_glyph_reaching_the_line_box_bottom_is_flagged() -> None:
    quality = measure_glyph_quality(
        _mask_with_ink((20, 16, 26, 30)), box=(20, 16, 26, 30), line=_LINE
    )
    assert "touches_line_bottom" in quality.flags


def test_a_full_box_of_ink_reads_a_density_of_one() -> None:
    quality = measure_glyph_quality(
        _mask_with_ink((20, 16, 26, 25)), box=(20, 16, 26, 25), line=_LINE
    )
    assert quality.ink_density == 1.0
    assert quality.row_extent_px == 9


def test_a_nearly_empty_box_is_flagged_as_sparse() -> None:
    page = np.zeros((40, 100), dtype=np.bool_)
    page[16, 20] = True
    quality = measure_glyph_quality(page, box=(20, 16, 30, 25), line=_LINE)
    assert "sparse_ink" in quality.flags
    assert quality.ink_density < 0.15


def test_a_glyph_with_no_line_records_density_alone() -> None:
    quality = measure_glyph_quality(_mask_with_ink((20, 2, 26, 8)), box=(20, 2, 26, 8), line=None)
    assert quality.top_offset_px is None
    assert quality.bottom_offset_px is None
    assert quality.flags == ()


def test_flags_come_back_sorted_and_deduplicated() -> None:
    quality = GlyphQuality(
        ink_density=0.5,
        row_extent_px=4,
        top_offset_px=0,
        bottom_offset_px=0,
        flags=("descends", "ascends", "ascends"),
    )
    assert quality.flags == ("ascends", "descends")


def test_the_two_band_offsets_are_recorded_together_or_not_at_all() -> None:
    with pytest.raises(ValueError, match="together or not at all"):
        _ = GlyphQuality(ink_density=0.5, row_extent_px=4, top_offset_px=0)


def test_a_line_band_may_not_place_its_x_height_top_below_its_baseline() -> None:
    with pytest.raises(ValueError, match="at or above its baseline"):
        _ = LineBand(box=(0, 10, 100, 30), baseline_row_px=16, x_height_top_row_px=24)


def test_an_empty_glyph_box_is_refused() -> None:
    with pytest.raises(ValueError, match="positive width and height"):
        _ = measure_glyph_quality(
            np.zeros((40, 100), dtype=np.bool_), box=(20, 16, 20, 25), line=None
        )


def test_ordinary_lowercase_runs_its_ascenders_half_again_the_x_height() -> None:
    assert word_ascender_flatness([("t", 26), ("h", 26), ("e", 17)]) == pytest.approx(26 / 17)
    assert not ascender_is_flat(word_ascender_flatness([("t", 26), ("h", 26), ("e", 17)]))


def test_a_word_set_in_small_capitals_reads_flat() -> None:
    flatness = word_ascender_flatness(
        [("l", 19), ("o", 19), ("w", 18), ("t", 19), ("h", 19), ("e", 19), ("r", 19)]
    )
    assert flatness == pytest.approx(1.0)
    assert ascender_is_flat(flatness)


def test_a_word_bound_to_the_wrong_ink_reads_flat() -> None:
    """`the` cut from ink that actually reads `man`: no ascender where the label promises one."""

    assert ascender_is_flat(word_ascender_flatness([("t", 17), ("h", 18), ("e", 17)]))


def test_a_word_with_no_ascender_cannot_be_judged() -> None:
    assert word_ascender_flatness([("o", 17), ("a", 17)]) is None
    assert not ascender_is_flat(None)


def test_a_word_with_no_x_height_letter_cannot_be_judged() -> None:
    assert word_ascender_flatness([("t", 26), ("h", 26)]) is None


def test_a_zero_x_height_is_refused_rather_than_divided_by() -> None:
    assert word_ascender_flatness([("h", 26), ("o", 0)]) is None


def test_a_line_needs_four_x_height_letters_to_set_a_reference() -> None:
    assert line_x_height_reference([("a", 17), ("o", 17), ("e", 17)]) is None
    assert line_x_height_reference([("a", 17), ("o", 17), ("e", 17), ("n", 17)]) == 17


def test_ascenders_do_not_count_toward_the_reference() -> None:
    assert line_x_height_reference([("h", 26), ("d", 26), ("l", 26), ("t", 26)]) is None


def test_a_glyph_matching_its_line_is_not_overtall() -> None:
    assert not is_overtall("o", 17, 17.0)
    assert not is_overtall("o", 22, 17.0)


def test_a_box_holding_two_letters_reads_overtall() -> None:
    assert is_overtall("o", 26, 17.0)


def test_only_x_height_letters_are_judged_this_way() -> None:
    assert not is_overtall("h", 26, 17.0)
    assert not is_overtall("p", 26, 17.0)


def test_no_reference_means_no_judgement() -> None:
    assert not is_overtall("o", 40, None)


def test_a_character_needs_thirty_samples_to_set_a_width_reference() -> None:
    assert character_width_reference([19] * 29) is None
    assert character_width_reference([19] * 30) == 19


def test_half_a_letter_reads_narrow() -> None:
    assert is_narrow(9, 19.0)
    assert is_narrow(8, 19.0)


def test_a_whole_letter_does_not() -> None:
    assert not is_narrow(19, 19.0)
    assert not is_narrow(12, 19.0)


def test_no_reference_means_no_narrow_judgement() -> None:
    assert not is_narrow(2, None)


def test_a_box_holding_several_letters_reads_wide() -> None:
    assert is_wide(30, 12.0)
    assert is_wide(18, 12.0)


def test_a_whole_letter_is_neither_narrow_nor_wide() -> None:
    assert not is_wide(19, 19.0)
    assert not is_narrow(19, 19.0)


def test_a_letter_with_its_comma_is_too_close_to_call_on_width() -> None:
    """Measured at 1.06 to 1.15 of median, inside the ordinary spread; no width test sees it."""

    assert not is_wide(15, 13.0)
    assert not is_wide(17, 16.0)


def test_no_reference_means_no_wide_judgement() -> None:
    assert not is_wide(99, None)
