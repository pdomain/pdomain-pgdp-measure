"""Pool typographic observables per page, then per book and per page class.

Pooling runs in two stages: a robust median over one page's lines first, then a median over
those page medians. The staging is forced by a shipped invariant worth keeping.
`Estimate.__post_init__` requires `sample_count == len(evidence_pages)` with no duplicate
pages, and a book-level x-height pooled directly over roughly 21,000 lines would have lines
as samples and pages as evidence. Two stages make `sample_count` a page count by
construction, and `extensions.line_sample_count` records the lines underneath. It is also the
better estimator: one dense page can no longer dominate a book.

Styles are grouped by the `page_class` M15a already ships. That is a cheaper stand-in for the
design's change-point detection over page index, not the same thing. It separates body pages
from chapter openings, and it does **not** separate title pages, prefaces, notes, indexes, or
a mid-book change of body face; those land in whatever class the template fit gave them.
Pages classed `unknown` are counted on the book and pooled into no style, because a style
built from pages that could not be classified would mix templates silently.

Every pooled value is a pixel count in the `source` frame, uncalibrated, with a median
absolute deviation and a 16th/84th percentile pair in `extensions`. Skew is the exception and
comes back as a bare median and dispersion rather than an `Estimate`, because a slope is
dimensionless and every `Estimate` in `pgdp-typography/v1` is in px.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Final

from pdomain_pgdp_measure.ordering import natural_page_key
from pdomain_pgdp_measure.profile_models import Estimate
from pdomain_pgdp_measure.typography_models import (
    BODY_PAGE_CLASSES,
    CHAPTER_OPENING_PAGE_CLASSES,
    TYPOGRAPHY_ESTIMATE_NAMES,
    StyleName,
    StyleTypography,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pdomain_pgdp_measure.profile_models import TruthClass
    from pdomain_pgdp_measure.typography_models import LineTypography, PageTypography


PAGE_POOLING_METHOD: Final = "typography-page-median/v1"
BOOK_POOLING_METHOD: Final = "typography-median-mad/v1"

TYPOGRAPHY_POOLING_METHODS: dict[str, float | int | str] = {
    "page_algorithm": PAGE_POOLING_METHOD,
    "book_algorithm": BOOK_POOLING_METHOD,
    "grouping": "page_class",
    "dispersion": "median_absolute_deviation",
}

STYLE_PAGE_CLASSES: Final[dict[StyleName, tuple[str, ...]]] = {
    "body": BODY_PAGE_CLASSES,
    "chapter_opening": CHAPTER_OPENING_PAGE_CLASSES,
}
STYLE_ORDER: Final[tuple[StyleName, ...]] = ("body", "chapter_opening")

#: Metrics pooled over every matched line, measured or not.
#:
#: Both come straight off the candidate box, so the segment floor that excludes short lines
#: does not apply to them. Pooling them over measured lines only would bias the width upward
#: by dropping every line under 600 px, which is exactly the paragraph-final and heading lines
#: that make a page's width distribution what it is.
OBSERVED_LINE_METRICS: Final = ("line_ink_width_px", "line_ink_height_px")

#: Metrics pooled over measured lines only, because the estimator produced them.
DERIVED_LINE_METRICS: Final = (
    "x_height_px",
    "ascender_extent_px",
    "descender_extent_px",
    "stroke_width_px",
)

_NO_MEASURED_LINES: Final = "no_measured_lines"
_NO_MATCHED_LINES: Final = "no_matched_lines"
_NO_INDENT_REFERENCE: Final = "no_indent_reference"
_NO_BASELINE_PITCH: Final = "no_baseline_pitch"
_NO_RECONCILED_WORD_GAPS: Final = "no_reconciled_word_gaps"
_UNPOOLED_PAGE_CLASS: Final = "unpooled_page_class"


@dataclass(frozen=True, slots=True)
class SkewSummary:
    """A pooled skew distribution, kept out of the estimate list on purpose."""

    median: float | None = None
    median_absolute_deviation: float | None = None


@dataclass(frozen=True, slots=True)
class PagePooling:
    """One page's first-stage medians and its skew distribution."""

    estimates: tuple[Estimate, ...]
    skew: SkewSummary


@dataclass(frozen=True, slots=True)
class BookPooling:
    """One book's second-stage medians, its per-style pooling, and its skew distribution."""

    estimates: tuple[Estimate, ...]
    styles: tuple[StyleTypography, ...]
    skew: SkewSummary
    pooled_page_count: int
    """Pages belonging to some style. Book estimates pool every page, not only these."""

    unknown_class_page_count: int


def pool_page(
    page_name: str,
    lines: Sequence[LineTypography],
    baseline_pitches: Sequence[int] = (),
) -> PagePooling:
    """Median every observable over one page's matched lines.

    `lines` is every matched line on the page, including the ones the estimator excluded, so
    the observed box metrics keep their full population and the exclusion counts stay
    honest. `baseline_pitches` carries only the pitches
    `typography_measure.measure_baseline_pitches` emitted, which are the ordinal-adjacent
    pairs where both lines were measured.

    Each returned estimate has `sample_count == 1` and this page as its only evidence, which
    is what makes the second stage's page-count semantics work.
    """

    measured = tuple(line for line in lines if line.baseline_row_px is not None)
    excluded_count = len(lines) - len(measured)
    estimates: list[Estimate] = []

    for name in OBSERVED_LINE_METRICS:
        values = tuple(float(getattr(line, name)) for line in lines)
        estimates.append(
            _page_estimate(
                page_name,
                name=name,
                values=values,
                truth_class="observed",
                line_sample_count=len(values),
                excluded_line_count=0,
                missing_code=_NO_MATCHED_LINES,
            )
        )

    for name in DERIVED_LINE_METRICS:
        values = tuple(
            float(value) for line in measured if (value := getattr(line, name)) is not None
        )
        estimates.append(
            _page_estimate(
                page_name,
                name=name,
                values=values,
                truth_class="derived",
                line_sample_count=len(values),
                excluded_line_count=excluded_count,
                missing_code=_NO_MEASURED_LINES,
            )
        )

    indents = tuple(float(line.line_indent_px) for line in lines if line.line_indent_px is not None)
    estimates.append(
        _page_estimate(
            page_name,
            name="line_indent_px",
            values=indents,
            truth_class="derived",
            line_sample_count=len(indents),
            excluded_line_count=len(lines) - len(indents),
            missing_code=_NO_INDENT_REFERENCE,
        )
    )

    pitches = tuple(float(pitch) for pitch in baseline_pitches)
    estimates.append(
        _page_estimate(
            page_name,
            name="baseline_pitch_px",
            values=pitches,
            truth_class="derived",
            line_sample_count=len(pitches),
            excluded_line_count=excluded_count,
            missing_code=_NO_BASELINE_PITCH,
        )
    )

    gap_lines = tuple(line for line in measured if line.words)
    gaps = tuple(
        float(word.following_gap_px)
        for line in gap_lines
        for word in line.words
        if word.following_gap_px is not None
    )
    estimates.append(
        _page_estimate(
            page_name,
            name="word_gap_px",
            values=gaps,
            truth_class="observed",
            line_sample_count=len(gap_lines),
            excluded_line_count=len(lines) - len(gap_lines),
            missing_code=_NO_RECONCILED_WORD_GAPS,
        )
    )

    slopes = tuple(line.skew_slope for line in measured if line.skew_slope is not None)
    return PagePooling(estimates=tuple(estimates), skew=_summarize_skew(slopes))


def pool_book(pages: Sequence[PageTypography]) -> BookPooling:
    """Pool one book's page medians, whole and per style.

    Book estimates pool every measured page, including pages the template fit could not
    class. Styles are what the classification gates: a page classed `unknown` is excluded
    from every style as `"{page_name}:unpooled_page_class"`, and a style is emitted only when
    it has at least one page, so a book with no chapter openings reports one style rather
    than an empty second one.

    Restricting the book estimates to style-eligible pages as well was measured on
    2026-09-03 and rejected. Of the four whole-book runs with accepted pages, one
    (projectID603d7d5e04ca0) classes all 52 of them `unknown` and another
    (projectID657550412c8dc) classes 8 of 14 that way, so a style-gated book estimate
    reported nothing at all for a book carrying 902 measured lines. A book-level median is a
    statement about the book, not about one template, and withholding it because the
    template fit was inconclusive throws away the measurement this slice exists to make.
    """

    ordered = tuple(sorted(pages, key=lambda page: natural_page_key(page.page_name)))
    styled = tuple(page for page in ordered if _style_for(page.page_class) is not None)
    unknown = tuple(page for page in ordered if _style_for(page.page_class) is None)
    styles = tuple(
        style
        for style in (_pool_style(style_name, ordered) for style_name in STYLE_ORDER)
        if style is not None
    )
    return BookPooling(
        estimates=_pool_pages(ordered, unpooled=()),
        styles=styles,
        skew=_pool_skew(ordered),
        pooled_page_count=len(styled),
        unknown_class_page_count=len(unknown),
    )


def _pool_style(style_name: StyleName, pages: Sequence[PageTypography]) -> StyleTypography | None:
    page_classes = STYLE_PAGE_CLASSES[style_name]
    members = tuple(page for page in pages if page.page_class in page_classes)
    if not members:
        return None
    others = tuple(page for page in pages if page.page_class not in page_classes)
    skew = _pool_skew(members)
    return StyleTypography(
        style=style_name,
        page_classes=page_classes,
        page_count=len(members),
        line_count=sum(page.matched_line_count for page in members),
        estimates=_pool_pages(members, unpooled=others),
        skew_slope_median=skew.median,
        skew_slope_mad=skew.median_absolute_deviation,
    )


def _pool_pages(
    pages: Sequence[PageTypography], *, unpooled: Sequence[PageTypography]
) -> tuple[Estimate, ...]:
    """Second-stage medians over page medians, one estimate per metric name.

    Every canonical metric name is emitted whether or not a page produced a value for it. A
    book whose pages all failed then reports nulls carrying one exclusion per page, which
    says "measured and got nothing" where an absent key would say nothing at all.
    """

    names: set[str] = set(TYPOGRAPHY_ESTIMATE_NAMES)
    names.update(estimate.name for page in pages for estimate in page.estimates)
    return tuple(_book_estimate(name, pages=pages, unpooled=unpooled) for name in sorted(names))


def _book_estimate(
    name: str, *, pages: Sequence[PageTypography], unpooled: Sequence[PageTypography]
) -> Estimate:
    contributing: list[tuple[str, float]] = []
    exclusions: list[tuple[str, str]] = []
    line_sample_count = 0
    excluded_line_count = 0
    for page in pages:
        estimate = _named_estimate(page, name)
        if estimate is None or estimate.value is None:
            code = _missing_code(estimate)
            exclusions.append((page.page_name, code))
            continue
        contributing.append((page.page_name, float(estimate.value)))
        line_sample_count += _extension_count(estimate, "line_sample_count")
        excluded_line_count += _extension_count(estimate, "excluded_line_count")
    exclusions.extend((page.page_name, _UNPOOLED_PAGE_CLASS) for page in unpooled)
    exclusions.sort(key=lambda item: (natural_page_key(item[0]), item[1]))

    values = tuple(value for _, value in contributing)
    extensions: dict[str, float | int] = {
        "line_sample_count": line_sample_count,
        "excluded_line_count": excluded_line_count,
    }
    extensions.update(_dispersion(values))
    return Estimate(
        name=name,
        value=float(median(values)) if values else None,
        unit="px",
        truth_class="pooled",
        method=BOOK_POOLING_METHOD,
        sample_count=len(contributing),
        evidence_pages=tuple(page_name for page_name, _ in contributing),
        exclusions=tuple(f"{page_name}:{code}" for page_name, code in exclusions),
        extensions=extensions,
    )


def _page_estimate(
    page_name: str,
    *,
    name: str,
    values: Sequence[float],
    truth_class: TruthClass,
    line_sample_count: int,
    excluded_line_count: int,
    missing_code: str,
) -> Estimate:
    extensions: dict[str, float | int | str] = {
        "line_sample_count": line_sample_count,
        "excluded_line_count": excluded_line_count,
    }
    extensions.update(_dispersion(values))
    if not values:
        extensions["missing_reason"] = missing_code
    return Estimate(
        name=name,
        value=float(median(values)) if values else None,
        unit="px",
        truth_class=truth_class,
        method=PAGE_POOLING_METHOD,
        sample_count=1 if values else 0,
        evidence_pages=(page_name,) if values else (),
        exclusions=() if values else (f"{page_name}:{missing_code}",),
        extensions=extensions,
    )


def _dispersion(values: Sequence[float]) -> dict[str, float]:
    """Median absolute deviation plus a 16th/84th percentile pair.

    The percentiles are nearest-rank on the sorted values rather than interpolated, so a
    pooled report never invents a value that no page or line produced.
    """

    if not values:
        return {}
    center = float(median(values))
    ordered = sorted(values)
    return {
        "median_absolute_deviation": float(median([abs(value - center) for value in ordered])),
        "percentile_16": _nearest_rank(ordered, 0.16),
        "percentile_84": _nearest_rank(ordered, 0.84),
    }


def _nearest_rank(ordered: Sequence[float], fraction: float) -> float:
    index = int(fraction * (len(ordered) - 1) + 0.5)
    return float(ordered[max(0, min(len(ordered) - 1, index))])


def _summarize_skew(slopes: Sequence[float]) -> SkewSummary:
    if not slopes:
        return SkewSummary()
    center = float(median(slopes))
    return SkewSummary(
        median=center,
        median_absolute_deviation=float(median([abs(slope - center) for slope in slopes])),
    )


def _pool_skew(pages: Iterable[PageTypography]) -> SkewSummary:
    return _summarize_skew(
        tuple(page.skew_slope_median for page in pages if page.skew_slope_median is not None)
    )


def _named_estimate(page: PageTypography, name: str) -> Estimate | None:
    for estimate in page.estimates:
        if estimate.name == name:
            return estimate
    return None


def _missing_code(estimate: Estimate | None) -> str:
    if estimate is None:
        return _NO_MEASURED_LINES
    reason = estimate.extensions.get("missing_reason")
    return reason if isinstance(reason, str) else _NO_MEASURED_LINES


def _extension_count(estimate: Estimate, name: str) -> int:
    value = estimate.extensions.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _style_for(page_class: str) -> StyleName | None:
    for style_name, page_classes in STYLE_PAGE_CLASSES.items():
        if page_class in page_classes:
            return style_name
    return None
