"""Tests for scoring a glyph against the shape its character usually takes."""

from __future__ import annotations

import numpy as np

from pdomain_pgdp_measure.glyph_shape import (
    SHAPE_COLUMNS,
    SHAPE_ROWS,
    build_reference,
    build_references,
    normalise_glyph,
)


def _crop(rows: list[str]) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    """Build a grayscale crop from a picture, where `#` is ink and `.` is paper."""

    return np.array(
        [[0 if character == "#" else 255 for character in row] for row in rows], dtype=np.uint8
    )


_STEM = _crop(["..##..", "..##..", "..##..", "..##..", "..##..", "..##.."])
_BAR = _crop(["......", "######", "######", "......", "......", "......"])


def test_a_glyph_resamples_onto_the_comparison_grid() -> None:
    grid = normalise_glyph(_STEM)
    assert grid.shape == (SHAPE_ROWS, SHAPE_COLUMNS)
    assert grid.dtype == np.uint8


def test_ink_reads_high_and_paper_low() -> None:
    grid = normalise_glyph(_crop(["####", "####"]))
    assert grid.min() > 200
    assert normalise_glyph(_crop(["....", "...."])).max() < 55


def test_an_empty_crop_yields_an_empty_grid() -> None:
    grid = normalise_glyph(np.zeros((0, 0), dtype=np.uint8))
    assert grid.shape == (SHAPE_ROWS, SHAPE_COLUMNS)
    assert not grid.any()


def test_a_character_needs_sixty_samples_for_a_reference() -> None:
    assert build_reference([normalise_glyph(_STEM)] * 59) is None
    assert build_reference([normalise_glyph(_STEM)] * 60) is not None


def test_a_glyph_matching_its_character_is_not_an_outlier() -> None:
    reference = build_reference([normalise_glyph(_STEM)] * 80)
    assert reference is not None
    assert not reference.is_outlier(normalise_glyph(_STEM))


def _inked(value: int) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    """The same stem at a different impression, which is how a real book's samples vary."""

    crop = _STEM.copy()
    crop[crop == 0] = value
    return crop


def test_a_different_shape_is_an_outlier() -> None:
    """The reference needs real scatter, so the sample varies its impression the way ink does."""

    grids = [normalise_glyph(_inked(0))] * 40 + [normalise_glyph(_inked(70))] * 40
    reference = build_reference(grids)
    assert reference is not None
    assert reference.scatter_p95 > 0.0
    assert reference.is_outlier(normalise_glyph(_BAR))
    assert not reference.is_outlier(normalise_glyph(_STEM))


def test_a_reference_is_built_per_character_and_style() -> None:
    references = build_references(
        {
            ("o", "roman"): [normalise_glyph(_STEM)] * 70,
            ("o", "italic"): [normalise_glyph(_BAR)] * 70,
            ("q", "roman"): [normalise_glyph(_BAR)] * 10,
        }
    )
    assert set(references) == {("o", "roman"), ("o", "italic")}
    assert references[("o", "roman")].sample_count == 70


def test_a_reference_with_no_scatter_calls_nothing_an_outlier() -> None:
    """Identical samples give a zero spread, which is no evidence rather than infinite evidence."""

    reference = build_reference([normalise_glyph(_STEM)] * 80)
    assert reference is not None
    assert reference.scatter_p95 == 0.0
    assert not reference.is_outlier(normalise_glyph(_BAR))


def test_descriptive_flags_do_not_disqualify_a_glyph_from_its_own_reference() -> None:
    """Every `h` ascends, so treating `ascends` as a defect leaves `h` with no reference at all."""

    from pdomain_pgdp_measure.glyph_quality import DEFECT_FLAGS

    assert "ascends" not in DEFECT_FLAGS
    assert "descends" not in DEFECT_FLAGS
    assert "touches_line_top" not in DEFECT_FLAGS
    assert "unlike_character" in DEFECT_FLAGS
    assert "narrow" in DEFECT_FLAGS
