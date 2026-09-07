"""Measure per-line vertical extents, stroke width, and skew, font-free.

Given the page ink mask `alignment_image.build_candidate_mask` builds and one matched
candidate's source-frame box, `measure_line_typography` estimates where the baseline and
x-height top sit, how thick strokes run, and how much the line skews -- all without opening a
font, rendering text, or leaving the `source` coordinate frame.

The estimator is local by design. A whole-line vertical projection smears under skew: at a
half-degree tilt a 1500 px line smears its baseline by `1500 * tan(0.5deg) ~= 13.1` px, which
swallows a 20 px x-height whole. Splitting the box into `max(200, width // 8)` px column
segments and projecting each one separately cuts that smear by the width ratio, to about
1.75 px for a 200 px segment at the same tilt. Taking the median baseline across segments then
recovers the line's typical geometry despite the smear each segment still carries. This is a
local estimator, not a rectification: it never builds a rotated frame or a resampled image.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

Bounds = tuple[int, int, int, int]
LineExclusionReason = Literal["insufficient_segments", "skew_exceeds_maximum"]
WordGapRule = Literal["otsu-gap-histogram", "x-height-ratio"]

TYPOGRAPHY_MEASURE_METHODS: dict[str, float | int | str] = {
    "algorithm": "segment-vertical-projection/v1",
    "segment_minimum_width_px": 200,
    "segment_width_divisor": 8,
    "minimum_usable_segments": 3,
    "maximum_skew_slope": 0.02,
}

_SEGMENT_MINIMUM_WIDTH_PX = 200
_SEGMENT_WIDTH_DIVISOR = 8
_MINIMUM_USABLE_SEGMENTS = 3
_MAXIMUM_SKEW_SLOPE = 0.02

WORD_SEGMENTATION_METHODS: dict[str, float | int | str] = {
    "algorithm": "ink-profile-word-runs/v2",
    "gap_threshold_minimum_px": 2,
    "gap_histogram_maximum_px": 64,
    "gap_histogram_minimum_samples": 64,
    "gap_valley_ratio_maximum": 0.5,
    "gap_threshold_x_height_ratio": 0.25,
}

_GAP_THRESHOLD_MINIMUM_PX = 2
_GAP_THRESHOLD_X_HEIGHT_RATIO = 0.25
_GAP_HISTOGRAM_MAXIMUM_PX = 64
"""Gaps wider than this are line-end and paragraph whitespace, not letter or word spacing.

Clamping them into one tail bin keeps a handful of very wide gaps from dragging the
inter-word class mean upward and pulling the split with it.
"""
_GAP_HISTOGRAM_MINIMUM_SAMPLES = 64
"""One sample per usable histogram bin. Below that the distribution cannot be read at all."""
_GAP_VALLEY_RATIO_MAXIMUM = 0.5
"""How shallow the valley at the split may be before the histogram is not two populations.

The ratio is the count at the chosen threshold over the smaller of the two class peaks, so a
real valley scores near zero and a single hump scores near one. Otsu's own separability was
tried first and rejected: it scores 0.65 to 0.75 on a uniform, a single Gaussian, and a
geometric decay, against 0.83 to 0.85 on the real books, so it does not separate the two cases.
Valley depth does. Measured, the five books score 0.104 to 0.152 and the same synthetic
unimodal shapes score 1.00 to 1.18, so this floor sits between them with margin on both sides.
"""


@dataclass(frozen=True, slots=True)
class LineTypographyMeasurement:
    """One matched line's font-free vertical extents, stroke width, and skew.

    Every field is a pixel count or a dimensionless slope in the `source` frame. Rows are
    box-relative half-open coordinates translated back into the page's absolute frame:
    `baseline_row_px` and `x_height_top_row_px` are absolute row indices, not offsets.
    """

    baseline_row_px: int
    x_height_top_row_px: int
    x_height_px: int
    ascender_extent_px: int
    descender_extent_px: int
    stroke_width_px: float
    skew_slope: float
    segment_count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("baseline_row_px", self.baseline_row_px),
            ("x_height_top_row_px", self.x_height_top_row_px),
            ("ascender_extent_px", self.ascender_extent_px),
            ("descender_extent_px", self.descender_extent_px),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if self.x_height_px != self.baseline_row_px - self.x_height_top_row_px:
            raise ValueError("x_height_px must equal baseline_row_px minus x_height_top_row_px.")
        if self.x_height_px < 0:
            raise ValueError("x_height_px must be nonnegative.")
        if not math.isfinite(self.stroke_width_px) or self.stroke_width_px < 0:
            raise ValueError("stroke_width_px must be a nonnegative finite number.")
        if not math.isfinite(self.skew_slope):
            raise ValueError("skew_slope must be finite.")
        if self.segment_count < _MINIMUM_USABLE_SEGMENTS:
            raise ValueError("A measurement requires the minimum usable segment count.")


@dataclass(frozen=True, slots=True)
class LineTypographyExclusion:
    """Why one matched line's box could not be measured."""

    reason: LineExclusionReason
    segment_count: int
    skew_slope: float | None = None

    def __post_init__(self) -> None:
        if self.segment_count < 0:
            raise ValueError("segment_count must be nonnegative.")
        if self.reason == "skew_exceeds_maximum" and self.skew_slope is None:
            raise ValueError("A skew_exceeds_maximum exclusion requires the measured slope.")
        if self.reason == "insufficient_segments" and self.skew_slope is not None:
            raise ValueError("An insufficient_segments exclusion has no measured slope.")
        if self.skew_slope is not None and not math.isfinite(self.skew_slope):
            raise ValueError("skew_slope must be finite.")


@dataclass(frozen=True, slots=True)
class WordMeasurement:
    """One reconciled word's ink box and the gap that follows it, in the `source` frame.

    `following_gap_px` is the width, in px, of the zero-ink column run immediately after
    this word's run and before the next; the last word on a line has none.
    """

    box: Bounds
    following_gap_px: int | None = None

    def __post_init__(self) -> None:
        x_start, y_start, x_end, y_end = self.box
        if x_end <= x_start or y_end <= y_start:
            raise ValueError("A word box must have positive width and height.")
        if self.following_gap_px is not None and self.following_gap_px < 0:
            raise ValueError("following_gap_px must be nonnegative.")


@dataclass(frozen=True, slots=True)
class LineWordMeasurement:
    """Word-run segmentation and reconciliation for one matched line.

    `words` carries a per-word box only when `words_reconciled` is true. PGDP text is
    fallible and so is gap detection, so a disagreement between `word_run_count` and
    `source_word_count` is recorded as evidence rather than resolved by a guess: binding
    a mismatched run count to a word in order would silently misattribute boxes to the
    wrong words, so no per-word row is emitted for a line that disagrees.
    """

    word_run_count: int
    source_word_count: int
    words_reconciled: bool
    words: tuple[WordMeasurement, ...] = ()

    def __post_init__(self) -> None:
        if self.word_run_count < 0:
            raise ValueError("word_run_count must be nonnegative.")
        if self.source_word_count < 0:
            raise ValueError("source_word_count must be nonnegative.")
        if self.words_reconciled != (self.word_run_count == self.source_word_count):
            raise ValueError("words_reconciled must match whether the two counts agree.")
        object.__setattr__(self, "words", tuple(self.words))
        if not self.words_reconciled and self.words:
            raise ValueError("An unreconciled line must not emit word rows.")
        if self.words_reconciled and len(self.words) != self.word_run_count:
            raise ValueError("A reconciled line must emit exactly one row per word run.")


@dataclass(frozen=True, slots=True)
class WordGapThreshold:
    """One book's word-gap threshold, the rule that produced it, and its evidence.

    `threshold_px` is `None` exactly when `rule` is `x-height-ratio`: Otsu found no genuine
    two-population split in the book's own gap lengths, so there is no book-wide value and the
    caller must fall back to `x_height_gap_threshold` per line. `valley_ratio` is `None` only
    when the split leaves one side empty, which is no split at all.
    """

    rule: WordGapRule
    threshold_px: int | None
    valley_ratio: float | None
    sample_count: int

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError("sample_count must be nonnegative.")
        if self.valley_ratio is not None and (
            not math.isfinite(self.valley_ratio) or self.valley_ratio < 0.0
        ):
            raise ValueError("valley_ratio must be a nonnegative finite number or null.")
        if self.rule == "otsu-gap-histogram":
            if self.threshold_px is None:
                raise ValueError("An Otsu-derived threshold must carry a pixel value.")
            if self.threshold_px < _GAP_THRESHOLD_MINIMUM_PX:
                raise ValueError("threshold_px must be at least the shipped minimum.")
        elif self.threshold_px is not None:
            raise ValueError("An x-height fallback carries no book-wide pixel value.")


@dataclass(frozen=True, slots=True)
class BaselinePitch:
    """The vertical distance between two consecutive matched lines' baselines.

    Emitted only for a pair of matched lines whose caller-assigned ordinals differ by
    exactly one and which both produced a baseline. Task 2's segment-floor and
    skew-backstop exclusions break the naive reading of "consecutive": measured against
    the five whole-book alignment reports, 13.9 percent of matched lines get no baseline,
    so the next available measured pair can sit two or more ordinals apart. Reporting that
    gap as a single-line pitch would silently overstate the leading by roughly a factor of
    two, so such a pair is skipped by `measure_baseline_pitches` rather than divided down.
    """

    upper_ordinal: int
    lower_ordinal: int
    pitch_px: int

    def __post_init__(self) -> None:
        if self.lower_ordinal != self.upper_ordinal + 1:
            raise ValueError("A baseline pitch requires ordinals exactly one apart.")
        if self.pitch_px < 0:
            raise ValueError("pitch_px must be nonnegative.")


@dataclass(frozen=True, slots=True)
class _SegmentEstimate:
    """One column segment's local baseline and x-height-top row, before pooling."""

    center_x: float
    baseline_row: int
    x_height_top_row: int


def measure_line_typography(
    mask: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    box: Bounds,
) -> LineTypographyMeasurement | LineTypographyExclusion:
    """Measure one matched candidate's box against the page ink mask that produced it.

    `mask` is the full-page boolean ink mask `alignment_image.build_candidate_mask` returns;
    `box` is the candidate's half-open source-frame box. The caller owns rebuilding that mask
    and verifying it reproduces the candidate's recorded evidence before calling this.
    """

    x_start, y_start, x_end, y_end = box
    width = x_end - x_start
    line_mask = mask[y_start:y_end, x_start:x_end]
    segment_width = max(_SEGMENT_MINIMUM_WIDTH_PX, width // _SEGMENT_WIDTH_DIVISOR)
    segments = tuple(
        _measure_segment(line_mask, segment_start, segment_end)
        for segment_start, segment_end in _segment_bounds(width, segment_width)
    )
    usable = tuple(segment for segment in segments if segment is not None)
    if len(usable) < _MINIMUM_USABLE_SEGMENTS:
        return LineTypographyExclusion(reason="insufficient_segments", segment_count=len(usable))

    skew_slope = _fit_skew_slope(usable)
    if abs(skew_slope) > _MAXIMUM_SKEW_SLOPE:
        return LineTypographyExclusion(
            reason="skew_exceeds_maximum", segment_count=len(usable), skew_slope=skew_slope
        )

    baseline_row_local = _round_half_up(median(segment.baseline_row for segment in usable))
    x_height_top_row_local = _round_half_up(median(segment.x_height_top_row for segment in usable))
    baseline_row_px = y_start + baseline_row_local
    x_height_top_row_px = y_start + x_height_top_row_local
    return LineTypographyMeasurement(
        baseline_row_px=baseline_row_px,
        x_height_top_row_px=x_height_top_row_px,
        x_height_px=baseline_row_px - x_height_top_row_px,
        ascender_extent_px=baseline_row_px - y_start,
        descender_extent_px=y_end - baseline_row_px,
        stroke_width_px=_median_stroke_width(line_mask, x_height_top_row_local, baseline_row_local),
        skew_slope=skew_slope,
        segment_count=len(usable),
    )


def measure_line_words(
    mask: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    box: Bounds,
    horizontal_ink_profile: Sequence[float],
    visible_text: str,
    gap_threshold_px: int,
) -> LineWordMeasurement:
    """Segment word runs from a matched line's stored ink profile and reconcile them
    against its aligned transcription.

    `horizontal_ink_profile` is the box-width-normalized column profile
    `alignment_image.LineCandidate` already stores, so finding runs needs no image; `mask`
    is `alignment_image.build_candidate_mask`'s page ink mask, consulted only to draw each
    reconciled word's own row projection. `gap_threshold_px` is supplied by the caller
    rather than derived here, because the value that separates letter gaps from word gaps
    is a property of the book's own gap distribution and cannot be read off one line: see
    `derive_word_gap_threshold`.

    `source_word_count` is `len(visible_text.split())`. Splitting with no argument
    collapses runs of whitespace and drops leading and trailing whitespace, so repeated
    spaces in the transcription never inflate the count with phantom empty words.
    """

    if gap_threshold_px < _GAP_THRESHOLD_MINIMUM_PX:
        raise ValueError("gap_threshold_px must be at least the shipped minimum.")

    x_start, _, x_end, _ = box
    width = x_end - x_start
    if len(horizontal_ink_profile) != width:
        raise ValueError("horizontal_ink_profile length must match the line box width.")

    runs = find_word_runs(horizontal_ink_profile, gap_threshold_px)
    source_word_count = len(visible_text.split())
    words_reconciled = len(runs) == source_word_count

    if not words_reconciled:
        return LineWordMeasurement(
            word_run_count=len(runs),
            source_word_count=source_word_count,
            words_reconciled=False,
        )

    following_gaps = tuple(
        runs[index + 1][0] - runs[index][1] if index + 1 < len(runs) else None
        for index in range(len(runs))
    )
    words = tuple(
        _measure_word_box(mask, box=box, run=run, following_gap_px=following_gap_px)
        for run, following_gap_px in zip(runs, following_gaps, strict=True)
    )
    return LineWordMeasurement(
        word_run_count=len(runs),
        source_word_count=source_word_count,
        words_reconciled=True,
        words=words,
    )


def word_gap_lengths(profile: Sequence[float]) -> tuple[int, ...]:
    """Every zero-ink column run strictly inside one line's ink, longest to shortest order aside.

    Leading and trailing blank columns are not gaps between anything, so only runs bounded by
    ink on both sides are counted.
    """

    ink_columns = [index for index, value in enumerate(profile) if value > 0.0]
    return tuple(
        later - earlier - 1
        for earlier, later in itertools.pairwise(ink_columns)
        if later - earlier - 1 > 0
    )


def derive_word_gap_threshold(gap_counts: Mapping[int, int]) -> WordGapThreshold:
    """Split one book's pooled gap lengths into letter gaps and word gaps by Otsu.

    Intra-word letter gaps and inter-word gaps are two populations with a valley between
    them, and Otsu finds that valley from the counts alone without ever being shown a
    reconciliation rate. Measured over 17,205 matched lines in five books, the shipped
    `0.25 * x_height` rule sat below the swept optimum in every one of them, by 1 px in the
    best case and 6 px in the worst, so it split inside words at ordinary letter gaps.

    Otsu assumes two populations and will split a single hump down the middle if handed one,
    so the split is checked before it is used: the count at the threshold must fall to at most
    `_GAP_VALLEY_RATIO_MAXIMUM` of the smaller class peak. A book that fails that check, or
    that carries too few gaps to read a distribution at all, gets the `x-height-ratio` rule
    and the caller falls back to `x_height_gap_threshold` per line.
    """

    histogram = [0] * (_GAP_HISTOGRAM_MAXIMUM_PX + 1)
    for length, count in gap_counts.items():
        if length <= 0:
            raise ValueError("A gap length must be positive.")
        if count < 0:
            raise ValueError("A gap count must be nonnegative.")
        histogram[min(length, _GAP_HISTOGRAM_MAXIMUM_PX)] += count

    sample_count = sum(histogram)
    threshold = _otsu_threshold(tuple(histogram))
    valley_ratio = _valley_ratio(tuple(histogram), threshold)
    bimodal = (
        sample_count >= _GAP_HISTOGRAM_MINIMUM_SAMPLES
        and valley_ratio is not None
        and valley_ratio <= _GAP_VALLEY_RATIO_MAXIMUM
    )
    if not bimodal:
        return WordGapThreshold(
            rule="x-height-ratio",
            threshold_px=None,
            valley_ratio=valley_ratio,
            sample_count=sample_count,
        )
    return WordGapThreshold(
        rule="otsu-gap-histogram",
        threshold_px=max(_GAP_THRESHOLD_MINIMUM_PX, threshold),
        valley_ratio=valley_ratio,
        sample_count=sample_count,
    )


def x_height_gap_threshold(x_height_px: int) -> int:
    """The v1 gap rule, kept as the fallback when a book's gaps are not two populations."""

    return max(
        _GAP_THRESHOLD_MINIMUM_PX, _round_half_up(_GAP_THRESHOLD_X_HEIGHT_RATIO * x_height_px)
    )


def measure_baseline_pitches(
    lines: Sequence[tuple[int, LineTypographyMeasurement | LineTypographyExclusion]],
) -> tuple[BaselinePitch, ...]:
    """Baseline pitches between consecutive matched lines on one page.

    `lines` carries every matched line's caller-assigned ordinal alongside its
    `measure_line_typography` result, in ordinal order. A pitch is emitted only for a pair
    of list-adjacent entries whose ordinals differ by exactly one and which are both a
    `LineTypographyMeasurement`; a pair whose ordinals are further apart, or where either
    side carries a `LineTypographyExclusion`, is skipped rather than measured, so a
    baseline gap wider than one line never enters the data as an understated pitch.
    """

    pitches: list[BaselinePitch] = []
    for (upper_ordinal, upper), (lower_ordinal, lower) in itertools.pairwise(lines):
        if lower_ordinal != upper_ordinal + 1:
            continue
        if not isinstance(upper, LineTypographyMeasurement):
            continue
        if not isinstance(lower, LineTypographyMeasurement):
            continue
        pitches.append(
            BaselinePitch(
                upper_ordinal=upper_ordinal,
                lower_ordinal=lower_ordinal,
                pitch_px=lower.baseline_row_px - upper.baseline_row_px,
            )
        )
    return tuple(pitches)


def _otsu_threshold(histogram: tuple[int, ...]) -> int:
    """Otsu's threshold over an integer histogram.

    This is `image_measurement._otsu_threshold`'s search, unchanged and compared against it by
    test: the same cross-multiplied integer comparison, so no float rounding can move the
    split. It is duplicated rather than imported because the grayscale path it lives on is
    verified byte-exactly by Gate 2 and must not be reshaped to serve a second caller.
    """

    total = sum(histogram)
    weighted_total = sum(level * count for level, count in enumerate(histogram))
    background_count = 0
    background_total = 0
    best_threshold = 0
    best_numerator = 0
    best_denominator = 1
    for threshold, count in enumerate(histogram):
        background_count += count
        background_total += threshold * count
        foreground_count = total - background_count
        if background_count == 0 or foreground_count == 0:
            continue
        foreground_total = weighted_total - background_total
        difference = background_total * foreground_count - foreground_total * background_count
        numerator = difference * difference
        denominator = background_count * foreground_count
        if numerator * best_denominator > best_numerator * denominator:
            best_threshold = threshold
            best_numerator = numerator
            best_denominator = denominator
    return best_threshold


def _valley_ratio(histogram: tuple[int, ...], threshold: int) -> float | None:
    """How deep the split sits below the smaller of the two class peaks, or `None` if empty.

    A histogram of two populations dips to near zero between them, so this runs near zero; a
    single hump split down the middle keeps its peak count at the split, so this runs near one.
    """

    lower_peak = max(histogram[: threshold + 1], default=0)
    upper_peak = max(histogram[threshold + 1 :], default=0)
    if lower_peak == 0 or upper_peak == 0:
        return None
    return histogram[threshold] / min(lower_peak, upper_peak)


def find_word_runs(profile: Sequence[float], gap_threshold_px: int) -> tuple[tuple[int, int], ...]:
    """Column ranges, tight to ink, of word runs bridging small intra-word gaps.

    Consecutive ink columns separated by fewer than `gap_threshold_px` zero-ink columns
    stay in the same run, so a word's own letter-spacing never fragments it into several
    runs; a zero-ink run at or above the threshold starts a new run. Each returned range is
    a half-open `(start, end)` pair of box-local column indices.
    """

    ink_columns = [index for index, value in enumerate(profile) if value > 0.0]
    if not ink_columns:
        return ()
    runs: list[tuple[int, int]] = []
    run_start = ink_columns[0]
    previous = ink_columns[0]
    for column in ink_columns[1:]:
        if column - previous - 1 >= gap_threshold_px:
            runs.append((run_start, previous + 1))
            run_start = column
        previous = column
    runs.append((run_start, previous + 1))
    return tuple(runs)


def _measure_word_box(
    mask: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    *,
    box: Bounds,
    run: tuple[int, int],
    following_gap_px: int | None,
) -> WordMeasurement:
    """One word run's absolute source-frame box, from the run's own row projection.

    The column range is already tight to ink by construction of `find_word_runs`; the
    row range is not known from the profile at all, so it is measured fresh from `mask`,
    the same page ink mask `alignment_image.build_candidate_mask` builds.
    """

    x_start, y_start, _, y_end = box
    run_start, run_end = run
    word_x_start = x_start + run_start
    word_x_end = x_start + run_end
    column_mask = mask[y_start:y_end, word_x_start:word_x_end]
    ink_rows = np.flatnonzero(column_mask.any(axis=1))
    row_top = int(np.min(ink_rows))
    row_bottom = int(np.max(ink_rows))
    word_box: Bounds = (word_x_start, y_start + row_top, word_x_end, y_start + row_bottom + 1)
    return WordMeasurement(box=word_box, following_gap_px=following_gap_px)


def _segment_bounds(width: int, segment_width: int) -> tuple[tuple[int, int], ...]:
    """Split `[0, width)` into `segment_width`-wide column ranges.

    A trailing remainder narrower than half a segment is folded into the previous segment
    rather than kept as its own, so a short leftover column strip never stands alone as a
    segment too thin to carry a reliable projection.
    """

    bounds: list[tuple[int, int]] = []
    start = 0
    while start < width:
        end = min(start + segment_width, width)
        bounds.append((start, end))
        start = end
    if len(bounds) >= 2:
        last_start, last_end = bounds[-1]
        if (last_end - last_start) * 2 < segment_width:
            previous_start, _ = bounds[-2]
            bounds[-2] = (previous_start, last_end)
            _ = bounds.pop()
    return tuple(bounds)


def _measure_segment(
    line_mask: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    segment_start: int,
    segment_end: int,
) -> _SegmentEstimate | None:
    """Estimate one segment's local baseline and x-height-top row, or `None` if unusable.

    The baseline is the sharpest downward drop in the lower half of the segment's vertical
    ink projection; the x-height top is the sharpest upward rise in the upper half. "Half" is
    taken over the rows where *this segment* actually carries ink, not over the matched box's
    full row range: under skew, one segment's ink can sit rows away from another's, and a box
    tall enough to hold an extreme ascender or descender at one edge would otherwise push a
    distant segment's own content out of a box-wide half-window entirely. A segment with no
    ink, or no clear drop or rise -- typically one that fell in a between-word gap -- is
    unusable and excluded from the line's pooled estimate rather than forced to report a value.
    """

    profile = line_mask[:, segment_start:segment_end].sum(axis=1)
    ink_rows = np.flatnonzero(profile)
    if ink_rows.size == 0:
        return None
    support_top = int(np.min(ink_rows))
    support_bottom = int(np.max(ink_rows))
    midpoint = (support_top + support_bottom) // 2

    if midpoint >= support_bottom:
        return None
    drops = profile[midpoint:support_bottom] - profile[midpoint + 1 : support_bottom + 1]
    drop_index = int(np.argmax(drops))
    if drops[drop_index] <= 0:
        return None
    baseline_row = midpoint + drop_index

    if support_top >= midpoint:
        return None
    rises = profile[support_top + 1 : midpoint + 1] - profile[support_top:midpoint]
    rise_index = int(np.argmax(rises))
    if rises[rise_index] <= 0:
        return None
    x_height_top_row = support_top + rise_index + 1

    return _SegmentEstimate(
        center_x=(segment_start + segment_end) / 2,
        baseline_row=baseline_row,
        x_height_top_row=x_height_top_row,
    )


def _fit_skew_slope(segments: tuple[_SegmentEstimate, ...]) -> float:
    """Least-squares slope of segment baseline row against segment center column."""

    mean_x = sum(segment.center_x for segment in segments) / len(segments)
    mean_y = sum(segment.baseline_row for segment in segments) / len(segments)
    numerator = sum(
        (segment.center_x - mean_x) * (segment.baseline_row - mean_y) for segment in segments
    )
    denominator = sum((segment.center_x - mean_x) ** 2 for segment in segments)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _median_stroke_width(
    line_mask: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    x_height_top_row_local: int,
    baseline_row_local: int,
) -> float:
    """Median horizontal ink run length across rows between the x-height top and baseline."""

    row_start = min(x_height_top_row_local, baseline_row_local)
    row_end = max(x_height_top_row_local, baseline_row_local)
    run_lengths = np.concatenate(
        [_run_lengths(line_mask[row, :]) for row in range(row_start, row_end + 1)]
    )
    if run_lengths.size == 0:
        return 0.0
    return float(np.median(run_lengths))


def _run_lengths(
    row: np.ndarray[tuple[int, ...], np.dtype[np.bool]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.signedinteger]]:
    """Lengths of every contiguous `True` run in one boolean row, left to right."""

    padded = np.concatenate((np.zeros(1, dtype=np.bool), row, np.zeros(1, dtype=np.bool)))
    later = padded[1:]
    earlier = padded[:-1]
    starts = np.flatnonzero(later & ~earlier)
    ends = np.flatnonzero(earlier & ~later)
    return ends - starts


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)
