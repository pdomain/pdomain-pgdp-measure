"""Tests for two-stage typography pooling."""

from __future__ import annotations

import pytest

from pdomain_pgdp_measure.profile_models import CoordinateFrame, Estimate
from pdomain_pgdp_measure.typography_models import (
    BookTypography,
    LineTypography,
    PageTypography,
    WordTypography,
)
from pdomain_pgdp_measure.typography_pooling import (
    BOOK_POOLING_METHOD,
    PAGE_POOLING_METHOD,
    pool_book,
    pool_page,
)


def make_measured_line(
    ordinal: int,
    *,
    x_height: int = 20,
    width: int = 800,
    stroke: float = 2.0,
    skew: float = 0.001,
    indent: int | None = 0,
    gaps: tuple[int, ...] = (6,),
) -> LineTypography:
    top = 300 + ordinal * 40
    baseline = top + 30
    words = tuple(
        WordTypography(
            ordinal=index,
            box=(100 + index * 90, top + 6, 100 + index * 90 + 80, baseline),
            following_gap_px=gap,
        )
        for index, gap in enumerate([*gaps, None])
    )
    return LineTypography(
        candidate_ordinal=ordinal,
        source_ordinal=ordinal,
        box=(100, top, 100 + width, top + 40),
        line_ink_width_px=width,
        line_ink_height_px=40,
        line_indent_px=indent,
        baseline_row_px=baseline,
        x_height_top_row_px=baseline - x_height,
        x_height_px=x_height,
        ascender_extent_px=30,
        descender_extent_px=10,
        stroke_width_px=stroke,
        skew_slope=skew,
        segment_count=8,
        word_run_count=len(words),
        source_word_count=len(words),
        words_reconciled=True,
        words=words,
    )


def make_excluded_line(ordinal: int, *, width: int = 400) -> LineTypography:
    top = 300 + ordinal * 40
    return LineTypography(
        candidate_ordinal=ordinal,
        source_ordinal=ordinal,
        box=(100, top, 100 + width, top + 40),
        line_ink_width_px=width,
        line_ink_height_px=40,
        segment_count=2,
        exclusions=("insufficient_segments",),
    )


def _estimate(estimates: tuple[Estimate, ...], name: str) -> Estimate:
    for estimate in estimates:
        if estimate.name == name:
            return estimate
    raise AssertionError(f"No estimate named {name!r}.")


def _value(estimates: tuple[Estimate, ...], name: str) -> float | None:
    return _estimate(estimates, name).value


def make_page(
    page_name: str = "p2.png",
    *,
    page_class: str = "normal_recto",
    x_heights: tuple[int, ...] = (19, 20, 21),
    excluded: int = 1,
) -> PageTypography:
    lines = tuple(
        make_measured_line(index, x_height=x_height) for index, x_height in enumerate(x_heights)
    )
    lines += tuple(make_excluded_line(len(lines) + index) for index in range(excluded))
    pitches = tuple(40 for _ in range(max(0, len(x_heights) - 1)))
    pooling = pool_page(page_name, lines, pitches)
    return PageTypography(
        page_name=page_name,
        page_class=page_class,
        scan_sha256="a" * 64,
        source_frame=CoordinateFrame(width=1200, height=1600),
        matched_line_count=len(lines),
        measured_line_count=len(x_heights),
        excluded_line_count=excluded,
        reconciled_line_count=len(x_heights),
        baseline_pitch_count=len(pitches),
        estimates=pooling.estimates,
        skew_slope_median=pooling.skew.median,
        skew_slope_mad=pooling.skew.median_absolute_deviation,
    )


# --- Stage one: pages --------------------------------------------------------------------


def test_page_estimates_cite_exactly_one_page() -> None:
    pooling = pool_page("p2.png", (make_measured_line(0), make_measured_line(1)), (40,))
    for estimate in pooling.estimates:
        assert estimate.method == PAGE_POOLING_METHOD
        assert estimate.unit == "px"
        assert estimate.frame == "source"
        assert estimate.confidence is None
        if estimate.value is not None:
            assert estimate.sample_count == 1
            assert estimate.evidence_pages == ("p2.png",)


def test_page_medians_are_the_median_of_their_lines() -> None:
    lines = (
        make_measured_line(0, x_height=18, stroke=1.0),
        make_measured_line(1, x_height=20, stroke=2.0),
        make_measured_line(2, x_height=25, stroke=3.0),
    )
    pooling = pool_page("p2.png", lines, (40, 40))
    assert _value(pooling.estimates, "x_height_px") == 20.0
    assert _value(pooling.estimates, "stroke_width_px") == 2.0
    assert _value(pooling.estimates, "baseline_pitch_px") == 40.0
    assert _value(pooling.estimates, "word_gap_px") == 6.0


def test_box_metrics_keep_the_short_lines_the_estimator_excluded() -> None:
    """A width pooled over measured lines only would drop every line under 600 px."""

    lines = (
        make_measured_line(0, width=800),
        make_measured_line(1, width=800),
        make_excluded_line(2, width=200),
        make_excluded_line(3, width=200),
        make_excluded_line(4, width=200),
    )
    pooling = pool_page("p2.png", lines, ())
    assert _value(pooling.estimates, "line_ink_width_px") == 200.0
    width = _estimate(pooling.estimates, "line_ink_width_px")
    assert width.extensions["line_sample_count"] == 5
    x_height = _estimate(pooling.estimates, "x_height_px")
    assert x_height.extensions["line_sample_count"] == 2
    assert x_height.extensions["excluded_line_count"] == 3


def test_page_with_no_measured_lines_reports_a_null_with_an_exclusion() -> None:
    pooling = pool_page("p2.png", (make_excluded_line(0),), ())
    x_height = _estimate(pooling.estimates, "x_height_px")
    assert x_height.value is None
    assert x_height.sample_count == 0
    assert x_height.exclusions == ("p2.png:no_measured_lines",)


def test_page_dispersion_records_mad_and_a_percentile_pair() -> None:
    lines = tuple(
        make_measured_line(index, x_height=x_height)
        for index, x_height in enumerate((10, 18, 20, 22, 30))
    )
    extensions = _estimate(pool_page("p", lines, ()).estimates, "x_height_px").extensions
    assert extensions["median_absolute_deviation"] == 2.0
    assert extensions["percentile_16"] == 18.0
    assert extensions["percentile_84"] == 22.0


def test_percentiles_are_nearest_rank_so_they_name_a_real_observation() -> None:
    lines = tuple(
        make_measured_line(index, x_height=x_height)
        for index, x_height in enumerate((11, 13, 17, 19))
    )
    extensions = _estimate(pool_page("p", lines, ()).estimates, "x_height_px").extensions
    assert extensions["percentile_16"] in {11.0, 13.0, 17.0, 19.0}
    assert extensions["percentile_84"] in {11.0, 13.0, 17.0, 19.0}


def test_page_skew_is_summarized_outside_the_estimate_list() -> None:
    lines = (
        make_measured_line(0, skew=0.001),
        make_measured_line(1, skew=0.003),
        make_measured_line(2, skew=0.005),
    )
    pooling = pool_page("p2.png", lines, ())
    assert pooling.skew.median == pytest.approx(0.003)
    assert pooling.skew.median_absolute_deviation == pytest.approx(0.002)
    assert all(estimate.name != "skew_slope" for estimate in pooling.estimates)


def test_lines_without_an_indent_reference_are_excluded_from_the_indent_median() -> None:
    lines = (make_measured_line(0, indent=None), make_measured_line(1, indent=4))
    indent = _estimate(pool_page("p2.png", lines, ()).estimates, "line_indent_px")
    assert indent.value == 4.0
    assert indent.extensions["excluded_line_count"] == 1


# --- Stage two: books and styles ---------------------------------------------------------


def test_book_sample_count_is_a_page_count_and_lines_go_to_extensions() -> None:
    pages = (make_page("p2.png"), make_page("p3.png"), make_page("p4.png"))
    pooled = pool_book(pages)
    x_height = _estimate(pooled.estimates, "x_height_px")
    assert x_height.method == BOOK_POOLING_METHOD
    assert x_height.truth_class == "pooled"
    assert x_height.sample_count == 3
    assert x_height.evidence_pages == ("p2.png", "p3.png", "p4.png")
    assert x_height.extensions["line_sample_count"] == 9


def test_book_pools_page_medians_not_raw_lines() -> None:
    """One dense page cannot outvote the rest, which is why the staging exists."""

    dense = make_page("p2.png", x_heights=(40,) * 30)
    sparse = (
        make_page("p3.png", x_heights=(20,)),
        make_page("p4.png", x_heights=(20,)),
    )
    pooled = pool_book((dense, *sparse))
    assert _value(pooled.estimates, "x_height_px") == 20.0


def test_book_pages_are_pooled_in_natural_page_order() -> None:
    pages = (make_page("p10.png"), make_page("p2.png"))
    pooled = pool_book(pages)
    assert _estimate(pooled.estimates, "x_height_px").evidence_pages == (
        "p2.png",
        "p10.png",
    )


def test_unknown_class_pages_join_the_book_but_no_style() -> None:
    """Measured on 2026-09-03: one whole book classes all 52 accepted pages `unknown`.

    Gating the book estimate on the classification reported nothing for a book carrying 902
    measured lines, so the book pools every measured page and only styles are class-gated.
    """

    pages = (
        make_page("p2.png"),
        make_page("p3.png", page_class="unknown", x_heights=(40, 40, 40)),
    )
    pooled = pool_book(pages)
    assert pooled.unknown_class_page_count == 1
    assert pooled.pooled_page_count == 1
    assert _value(pooled.estimates, "x_height_px") == 30.0
    assert _estimate(pooled.estimates, "x_height_px").evidence_pages == ("p2.png", "p3.png")
    body = pooled.styles[0]
    assert _value(body.estimates, "x_height_px") == 20.0
    assert "p3.png:unpooled_page_class" in _estimate(body.estimates, "x_height_px").exclusions


def test_a_book_with_no_classified_page_still_reports_its_median() -> None:
    pages = (
        make_page("p2.png", page_class="unknown"),
        make_page("p3.png", page_class="unknown"),
    )
    pooled = pool_book(pages)
    assert pooled.styles == ()
    assert pooled.unknown_class_page_count == 2
    assert _value(pooled.estimates, "x_height_px") == 20.0


def test_recto_and_verso_pool_into_one_body_style() -> None:
    pages = (
        make_page("p2.png", page_class="normal_recto"),
        make_page("p3.png", page_class="normal_verso"),
        make_page("p4.png", page_class="chapter_opening", x_heights=(30, 30, 30)),
    )
    pooled = pool_book(pages)
    assert tuple(style.style for style in pooled.styles) == ("body", "chapter_opening")
    body, chapter = pooled.styles
    assert body.page_classes == ("normal_recto", "normal_verso")
    assert body.page_count == 2
    assert chapter.page_count == 1
    assert _value(body.estimates, "x_height_px") == 20.0
    assert _value(chapter.estimates, "x_height_px") == 30.0


def test_a_style_with_no_pages_is_not_emitted() -> None:
    pooled = pool_book((make_page("p2.png"), make_page("p3.png")))
    assert tuple(style.style for style in pooled.styles) == ("body",)


def test_style_estimates_exclude_the_pages_of_other_styles() -> None:
    pages = (
        make_page("p2.png", page_class="normal_recto"),
        make_page("p3.png", page_class="chapter_opening"),
    )
    body = pool_book(pages).styles[0]
    assert "p3.png:unpooled_page_class" in _estimate(body.estimates, "x_height_px").exclusions


def test_book_estimate_is_null_when_no_page_measured_the_metric() -> None:
    pages = (
        make_page("p2.png", x_heights=(), excluded=2),
        make_page("p3.png", x_heights=(), excluded=2),
    )
    pooled = pool_book(pages)
    x_height = _estimate(pooled.estimates, "x_height_px")
    assert x_height.value is None
    assert x_height.sample_count == 0
    assert x_height.exclusions == (
        "p2.png:no_measured_lines",
        "p3.png:no_measured_lines",
    )
    assert _value(pooled.estimates, "line_ink_width_px") == 400.0


def test_book_skew_pools_page_skew_medians() -> None:
    pages = (
        make_page("p2.png"),
        make_page("p3.png"),
    )
    pooled = pool_book(pages)
    assert pooled.skew.median == pytest.approx(0.001)
    assert pooled.skew.median_absolute_deviation == pytest.approx(0.0)


def test_pooled_estimates_survive_the_report_contract() -> None:
    """Pooling output has to be constructible as a BookTypography, not just as Estimates."""

    pages = (make_page("p2.png"), make_page("p3.png"))
    pooled = pool_book(pages)
    book = BookTypography(
        project_id="projectID0001",
        pages=pages,
        styles=pooled.styles,
        estimates=pooled.estimates,
        accepted_page_count=2,
        measured_page_count=2,
        matched_line_count=sum(page.matched_line_count for page in pages),
        measured_line_count=sum(page.measured_line_count for page in pages),
        reconciled_line_count=sum(page.reconciled_line_count for page in pages),
        unknown_class_page_count=pooled.unknown_class_page_count,
        skew_slope_median=pooled.skew.median,
        skew_slope_mad=pooled.skew.median_absolute_deviation,
    )
    assert book.estimates == pooled.estimates
