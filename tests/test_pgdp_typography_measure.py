"""Tests for font-free per-line typographic measurement.

Every fixture here is a synthetic block bitmap built from exact, known geometry -- vertical bars
standing in for glyph strokes, placed at exact rows for x-height bodies, ascenders, and
descenders. No font is opened or rendered, so these run on any machine with no fonts fetched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pdomain_pgdp_measure.image_measurement import _otsu_threshold as _grayscale_otsu_threshold
from pdomain_pgdp_measure.typography_measure import (
    LineTypographyExclusion,
    LineTypographyMeasurement,
    LineWordMeasurement,
    _otsu_threshold,
    derive_word_gap_threshold,
    measure_baseline_pitches,
    measure_line_typography,
    measure_line_words,
    word_gap_lengths,
    x_height_gap_threshold,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pdomain_pgdp_measure.typography_measure import Bounds

# Every synthetic line spans this many columns, chosen so `max(200, width // 8)` divides it
# into exactly eight 200 px segments with no trailing remainder, keeping the segment-merge
# edge case out of the accuracy fixtures below (it gets its own focused test instead).
_TOTAL_WIDTH = 1600
_PADDING_ROWS = 30
# Larger than `1600/2 * tan(1deg) ~= 14`, the widest per-bar vertical shift any fixture here
# produces, so a shifted bar's ink never runs off the top or bottom of the canvas.


@dataclass(frozen=True, slots=True)
class _SyntheticLine:
    """A synthetic block bitmap plus the exact geometry used to build it."""

    mask: np.ndarray[tuple[int, int], np.dtype[np.bool]]
    box: Bounds
    baseline_row_px: int
    x_height_top_row_px: int
    stroke_width_px: int


def _build_synthetic_line(
    *, x_height_px: int, stroke_width_px: int, skew_degrees: float, gap_px: int
) -> _SyntheticLine:
    """Build one synthetic line of vertical-bar glyph stubs at exact known rows.

    Most bars span only the x-height body; every fifth bar also carries an ascender, and
    every seventh carries a descender, so the projection each segment sees is dominated by
    plain x-height ink, matching how a real line of type is dominated by lowercase letters.
    Skew shifts each bar vertically by its own column position times the true slope, which
    reproduces a skewed scan without rotating or resampling any pixel.
    """

    ascender_extent = x_height_px
    descender_extent = max(2, x_height_px // 2)
    x_height_top = _PADDING_ROWS + ascender_extent
    baseline = x_height_top + x_height_px - 1
    descender_bottom = baseline + descender_extent
    canvas_height = descender_bottom + 1 + _PADDING_ROWS
    canvas = np.zeros((canvas_height, _TOTAL_WIDTH), dtype=np.bool)

    true_slope = math.tan(math.radians(skew_degrees))
    stride = stroke_width_px + gap_px
    bar_index = 0
    position = 0
    while position + stroke_width_px <= _TOTAL_WIDTH:
        center_x = position + stroke_width_px / 2
        offset = round(true_slope * (center_x - _TOTAL_WIDTH / 2))
        if bar_index % 5 == 0:
            row_top, row_bottom = _PADDING_ROWS + offset, baseline + offset
        elif bar_index % 7 == 0:
            row_top, row_bottom = x_height_top + offset, descender_bottom + offset
        else:
            row_top, row_bottom = x_height_top + offset, baseline + offset
        canvas[row_top : row_bottom + 1, position : position + stroke_width_px] = True
        bar_index += 1
        position += stride

    rows_with_ink = np.flatnonzero(canvas.any(axis=1))
    columns_with_ink = np.flatnonzero(canvas.any(axis=0))
    box: Bounds = (
        int(np.min(columns_with_ink)),
        int(np.min(rows_with_ink)),
        int(np.max(columns_with_ink)) + 1,
        int(np.max(rows_with_ink)) + 1,
    )
    return _SyntheticLine(
        mask=canvas,
        box=box,
        baseline_row_px=baseline,
        x_height_top_row_px=x_height_top,
        stroke_width_px=stroke_width_px,
    )


def _fixture_cases() -> tuple[tuple[int, int, float, int], ...]:
    """Build the Gate 3 fixture sweep: `(x_height_px, stroke_width_px, skew_degrees, gap_px)`.

    A full cross of x-height and stroke at zero skew, then a dedicated skew sweep and a
    dedicated gap sweep held at mid-range x-height and stroke. Together they cover every edge
    the plan names: x-height 8 to 40, stroke 1 to 5, skew 0 and +/-0.5 and +/-1.0 degrees, and
    gaps 2 to 20, with no fixture repeated.
    """

    x_heights = (8, 14, 20, 28, 40)
    strokes = (1, 2, 3, 4, 5)
    skews_degrees = (0.5, -0.5, 1.0, -1.0)
    gaps = (2, 6, 10, 14, 20)

    cases: list[tuple[int, int, float, int]] = [
        (x_height, stroke, 0.0, 8) for x_height in x_heights for stroke in strokes
    ]
    cases.extend((20, 2, skew, 8) for skew in skews_degrees)
    cases.extend((20, 2, 0.0, gap) for gap in gaps)
    return tuple(cases)


_FIXTURE_CASES = _fixture_cases()


def test_fixture_sweep_covers_the_gate_3_parameter_space() -> None:
    assert len(_FIXTURE_CASES) >= 24
    assert len(set(_FIXTURE_CASES)) == len(_FIXTURE_CASES)
    x_heights = {case[0] for case in _FIXTURE_CASES}
    strokes = {case[1] for case in _FIXTURE_CASES}
    skews = {case[2] for case in _FIXTURE_CASES}
    gaps = {case[3] for case in _FIXTURE_CASES}
    assert min(x_heights) == 8
    assert max(x_heights) == 40
    assert min(strokes) == 1
    assert max(strokes) == 5
    assert skews >= {0.0, 0.5, -0.5, 1.0, -1.0}
    assert min(gaps) == 2
    assert max(gaps) == 20


@pytest.mark.parametrize(
    ("x_height_px", "stroke_width_px", "skew_degrees", "gap_px"),
    _FIXTURE_CASES,
    ids=[
        f"xh{x_height}-st{stroke}-sk{skew}-gap{gap}"
        for x_height, stroke, skew, gap in _FIXTURE_CASES
    ],
)
def test_recovers_known_geometry_within_one_pixel(
    x_height_px: int, stroke_width_px: int, skew_degrees: float, gap_px: int
) -> None:
    line = _build_synthetic_line(
        x_height_px=x_height_px,
        stroke_width_px=stroke_width_px,
        skew_degrees=skew_degrees,
        gap_px=gap_px,
    )

    measured = measure_line_typography(line.mask, line.box)

    assert isinstance(measured, LineTypographyMeasurement)
    assert abs(measured.baseline_row_px - line.baseline_row_px) <= 1
    assert abs(measured.x_height_top_row_px - line.x_height_top_row_px) <= 1
    assert abs(measured.stroke_width_px - line.stroke_width_px) <= 1
    assert measured.x_height_px == measured.baseline_row_px - measured.x_height_top_row_px
    assert measured.ascender_extent_px == measured.baseline_row_px - line.box[1]
    assert measured.descender_extent_px == line.box[3] - measured.baseline_row_px


def test_excludes_a_box_too_narrow_for_three_segments() -> None:
    line = _build_synthetic_line(x_height_px=20, stroke_width_px=2, skew_degrees=0.0, gap_px=8)
    narrow_box: Bounds = (line.box[0], line.box[1], line.box[0] + 150, line.box[3])

    measured = measure_line_typography(line.mask, narrow_box)

    assert isinstance(measured, LineTypographyExclusion)
    assert measured.reason == "insufficient_segments"
    assert measured.segment_count < 3
    assert measured.skew_slope is None


def test_excludes_a_line_past_the_skew_backstop() -> None:
    line = _build_synthetic_line(x_height_px=20, stroke_width_px=2, skew_degrees=5.0, gap_px=8)

    measured = measure_line_typography(line.mask, line.box)

    assert isinstance(measured, LineTypographyExclusion)
    assert measured.reason == "skew_exceeds_maximum"
    assert measured.skew_slope is not None
    assert abs(measured.skew_slope) > 0.02


def test_measurement_rejects_an_inconsistent_x_height() -> None:
    with pytest.raises(ValueError, match="x_height_px"):
        _ = LineTypographyMeasurement(
            baseline_row_px=100,
            x_height_top_row_px=80,
            x_height_px=19,
            ascender_extent_px=100,
            descender_extent_px=10,
            stroke_width_px=2.0,
            skew_slope=0.0,
            segment_count=8,
        )


def test_exclusion_requires_a_slope_only_for_the_skew_reason() -> None:
    with pytest.raises(ValueError, match="skew_exceeds_maximum"):
        _ = LineTypographyExclusion(reason="skew_exceeds_maximum", segment_count=3, skew_slope=None)
    with pytest.raises(ValueError, match="insufficient_segments"):
        _ = LineTypographyExclusion(
            reason="insufficient_segments", segment_count=1, skew_slope=0.01
        )


# --- Task 3: word segmentation and reconciliation -------------------------------------------


def _build_word_line(
    word_gaps_px: Sequence[int], *, word_height_px: int, word_width_px: int = 30
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.bool]],
    Bounds,
    tuple[float, ...],
    tuple[Bounds, ...],
]:
    """Build a synthetic line of solid rectangular word blocks with exact known gaps.

    Each "word" is a single filled rectangle rather than a glyph stub, so its row
    projection and column extent are exact by construction and every expected value below
    is computed geometry, not a measurement. Returns the full-page mask, the line's box,
    its horizontal ink profile exactly as `alignment_image.LineCandidate` would store it
    (box-width-normalized to sum to one), and the expected box of each word.
    """

    left_margin_px = 10
    word_count = len(word_gaps_px) + 1
    total_width = 2 * left_margin_px + word_count * word_width_px + sum(word_gaps_px)
    canvas_height = word_height_px + 20
    canvas = np.zeros((canvas_height, total_width), dtype=np.bool)
    row_top = 10
    row_bottom = row_top + word_height_px

    boxes: list[Bounds] = []
    x = left_margin_px
    for index in range(word_count):
        canvas[row_top:row_bottom, x : x + word_width_px] = True
        boxes.append((x, row_top, x + word_width_px, row_bottom))
        x += word_width_px
        if index < len(word_gaps_px):
            x += word_gaps_px[index]

    rows_with_ink = np.flatnonzero(canvas.any(axis=1))
    columns_with_ink = np.flatnonzero(canvas.any(axis=0))
    box: Bounds = (
        int(np.min(columns_with_ink)),
        int(np.min(rows_with_ink)),
        int(np.max(columns_with_ink)) + 1,
        int(np.max(rows_with_ink)) + 1,
    )
    line_mask = canvas[box[1] : box[3], box[0] : box[2]]
    line_width = box[2] - box[0]
    column_counts = [int(line_mask[:, column].sum()) for column in range(line_width)]
    total = sum(column_counts)
    profile = tuple(count / total for count in column_counts)
    return canvas, box, profile, tuple(boxes)


@pytest.mark.parametrize("gap_threshold_px", [2, 3, 5, 8, 13, 20])
def test_word_runs_split_at_the_caller_supplied_gap_threshold(gap_threshold_px: int) -> None:
    """A gap one px short of the supplied threshold must bridge into a single run and a gap at
    it must split into two, for any threshold the caller hands down.
    """

    bridged_mask, bridged_box, bridged_profile, _ = _build_word_line(
        [gap_threshold_px - 1], word_height_px=20
    )
    bridged = measure_line_words(
        bridged_mask, bridged_box, bridged_profile, "oneword", gap_threshold_px
    )
    assert bridged.word_run_count == 1

    split_mask, split_box, split_profile, _ = _build_word_line(
        [gap_threshold_px], word_height_px=20
    )
    split = measure_line_words(split_mask, split_box, split_profile, "two words", gap_threshold_px)
    assert split.word_run_count == 2


def test_the_gap_threshold_no_longer_tracks_the_line_x_height() -> None:
    """One line, two thresholds, two answers. The v1 rule would have derived 5 px from this
    line's own 20 px x-height and split the 10 px gap; the caller's 12 px threshold bridges it.
    """

    mask, box, profile, _ = _build_word_line([10], word_height_px=20)

    assert x_height_gap_threshold(20) == 5
    assert measure_line_words(mask, box, profile, "two words", 5).word_run_count == 2
    assert measure_line_words(mask, box, profile, "oneword", 12).word_run_count == 1


def test_measure_line_words_rejects_a_threshold_below_the_shipped_minimum() -> None:
    mask, box, profile, _ = _build_word_line([10], word_height_px=20)

    with pytest.raises(ValueError, match="at least the shipped minimum"):
        _ = measure_line_words(mask, box, profile, "two words", 1)


def test_word_gap_lengths_counts_only_gaps_bounded_by_ink() -> None:
    profile = (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0)

    assert word_gap_lengths(profile) == (3,)


def _bimodal_gap_counts() -> dict[int, int]:
    """Letter gaps around 3 px and word gaps around 13 px, the shape the corpus shows."""

    letters = {2: 8000, 3: 30000, 4: 12000, 5: 2000}
    valley = {6: 300, 7: 200, 8: 200, 9: 400}
    words = {10: 1500, 11: 3000, 12: 6000, 13: 6000, 14: 3000, 15: 1200}
    return {**letters, **valley, **words}


def test_otsu_gap_threshold_lands_in_the_valley_between_the_two_populations() -> None:
    derived = derive_word_gap_threshold(_bimodal_gap_counts())

    assert derived.rule == "otsu-gap-histogram"
    assert derived.threshold_px is not None
    assert 5 <= derived.threshold_px <= 10
    assert derived.valley_ratio is not None
    assert derived.valley_ratio < 0.5
    assert derived.sample_count == 73800


@pytest.mark.parametrize(
    ("name", "gap_counts"),
    [
        ("uniform", dict.fromkeys(range(1, 65), 1000)),
        (
            "one_hump",
            {length: max(1, 20000 - 400 * abs(length - 8) ** 2) for length in range(1, 40)},
        ),
        (
            "geometric_decay",
            {length: max(1, round(50000 * 0.85**length)) for length in range(1, 65)},
        ),
    ],
)
def test_gap_threshold_falls_back_when_the_gaps_are_one_population(
    name: str, gap_counts: dict[int, int]
) -> None:
    """Otsu will split a single hump down its middle, so a shallow valley must be refused.

    Each shape here has one mode. Measured, the five corpus books score 0.104 to 0.152 on the
    valley ratio and these score at or above 1.0, so the 0.5 floor sits between them.
    """

    derived = derive_word_gap_threshold(gap_counts)

    assert derived.rule == "x-height-ratio", name
    assert derived.threshold_px is None
    assert derived.valley_ratio is not None
    assert derived.valley_ratio > 0.5


def test_gap_threshold_falls_back_when_the_book_carries_too_few_gaps() -> None:
    derived = derive_word_gap_threshold({3: 20, 13: 20})

    assert derived.rule == "x-height-ratio"
    assert derived.threshold_px is None
    assert derived.sample_count == 40


def test_gap_threshold_falls_back_when_every_gap_is_the_same_length() -> None:
    derived = derive_word_gap_threshold({4: 5000})

    assert derived.rule == "x-height-ratio"
    assert derived.threshold_px is None
    assert derived.valley_ratio is None


def test_gaps_wider_than_the_histogram_maximum_do_not_move_the_split() -> None:
    """Line-end and paragraph whitespace is clamped into one tail bin rather than left to drag
    the inter-word class mean, so adding a few very wide gaps must not move the threshold.
    """

    counts = _bimodal_gap_counts()
    widened = {**counts, 300: 40, 900: 40, 4000: 20}

    assert (
        derive_word_gap_threshold(widened).threshold_px
        == derive_word_gap_threshold(counts).threshold_px
    )


def test_the_gap_otsu_search_matches_the_shipped_grayscale_one() -> None:
    """Word segmentation runs its own copy of Otsu so the Gate 2 grayscale path is untouched.
    The two must agree exactly, or the corpus thresholds measured with one would not hold.
    """

    histograms = [
        tuple(_bimodal_gap_counts().get(level, 0) for level in range(65)),
        tuple(1000 for _ in range(65)),
        tuple(max(1, round(50000 * 0.85**level)) for level in range(65)),
        (0, 0, 0, 9000, 1000) + (0,) * 60,
        (0,) * 5 + (5000,) + (0,) * 59,
    ]
    for histogram in histograms:
        assert _otsu_threshold(histogram) == _grayscale_otsu_threshold(histogram)


def test_the_x_height_fallback_keeps_the_v1_rule() -> None:
    assert x_height_gap_threshold(0) == 2
    assert x_height_gap_threshold(4) == 2
    assert x_height_gap_threshold(11) == 3
    assert x_height_gap_threshold(18) == 5
    assert x_height_gap_threshold(80) == 20


def test_measure_line_words_binds_runs_to_words_when_counts_agree() -> None:
    gaps = (10, 14)
    x_height_px = 20
    mask, box, profile, expected_boxes = _build_word_line(gaps, word_height_px=x_height_px)

    result = measure_line_words(mask, box, profile, "alpha beta gamma", 5)

    assert isinstance(result, LineWordMeasurement)
    assert result.word_run_count == 3
    assert result.source_word_count == 3
    assert result.words_reconciled is True
    assert len(result.words) == 3
    for word, expected_box in zip(result.words, expected_boxes, strict=True):
        assert word.box == expected_box
    assert result.words[0].following_gap_px == gaps[0]
    assert result.words[1].following_gap_px == gaps[1]
    assert result.words[2].following_gap_px is None


def test_measure_line_words_records_disagreement_without_word_rows() -> None:
    gaps = (10, 14)
    x_height_px = 20
    mask, box, profile, _ = _build_word_line(gaps, word_height_px=x_height_px)

    result = measure_line_words(mask, box, profile, "alpha beta", 5)

    assert isinstance(result, LineWordMeasurement)
    assert result.word_run_count == 3
    assert result.source_word_count == 2
    assert result.words_reconciled is False
    assert result.words == ()


def test_source_word_count_collapses_whitespace_runs() -> None:
    mask, box, profile, _ = _build_word_line((10,), word_height_px=20)

    result = measure_line_words(mask, box, profile, "alpha   beta", 5)

    assert isinstance(result, LineWordMeasurement)
    assert result.source_word_count == 2


# --- The pitch addition: consecutive matched baselines only ----------------------------------


def _measurement(*, baseline_row_px: int) -> LineTypographyMeasurement:
    return LineTypographyMeasurement(
        baseline_row_px=baseline_row_px,
        x_height_top_row_px=baseline_row_px - 20,
        x_height_px=20,
        ascender_extent_px=baseline_row_px,
        descender_extent_px=10,
        stroke_width_px=2.0,
        skew_slope=0.0,
        segment_count=8,
    )


def test_measure_baseline_pitches_reports_adjacent_matched_lines() -> None:
    lines = (
        (0, _measurement(baseline_row_px=100)),
        (1, _measurement(baseline_row_px=140)),
        (2, _measurement(baseline_row_px=180)),
    )

    pitches = measure_baseline_pitches(lines)

    assert [pitch.pitch_px for pitch in pitches] == [40, 40]
    assert [(pitch.upper_ordinal, pitch.lower_ordinal) for pitch in pitches] == [(0, 1), (1, 2)]


def test_measure_baseline_pitches_skips_an_excluded_middle_line() -> None:
    """A page whose middle line is excluded must not report a doubled pitch across the gap.

    Without the ordinal-adjacency requirement, the naive next-available-pair reading would
    report a single pitch of 80 px here -- double the true 40 px single-line spacing.
    """

    lines = (
        (0, _measurement(baseline_row_px=100)),
        (1, LineTypographyExclusion(reason="insufficient_segments", segment_count=1)),
        (2, _measurement(baseline_row_px=180)),
    )

    pitches = measure_baseline_pitches(lines)

    assert pitches == ()
