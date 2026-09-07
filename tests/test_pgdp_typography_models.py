"""Contract tests for ``pgdp-typography/v1``.

The latent-discipline test is the one that matters most: it is the mechanism that turns the
design's "these values are latent" list into something a build can fail on.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pdomain_pgdp_measure.profile_models import CoordinateFrame, Estimate, ProfileDiagnostic
from pdomain_pgdp_measure.typography_models import (
    LATENT_KEY_DENYLIST,
    TYPOGRAPHY_ALGORITHM_VERSION,
    TYPOGRAPHY_SCHEMA_VERSION,
    BookTypography,
    LineTypography,
    PageTypography,
    StyleTypography,
    TypographyReport,
    WordTypography,
    find_latent_discipline_violations,
)

SCHEMA_ARTIFACT = Path("schemas/pgdp-typography-v1.schema.json")


def _sha256(character: str = "a") -> str:
    return character * 64


def make_line(*, measured: bool = True, reconciled: bool = True) -> LineTypography:
    if not measured:
        return LineTypography(
            candidate_ordinal=1,
            source_ordinal=1,
            box=(100, 400, 500, 440),
            line_ink_width_px=400,
            line_ink_height_px=40,
            segment_count=2,
            exclusions=("insufficient_segments",),
        )
    words = (
        WordTypography(ordinal=0, box=(100, 306, 180, 330), following_gap_px=6),
        WordTypography(ordinal=1, box=(186, 306, 260, 332), following_gap_px=None),
    )
    return LineTypography(
        candidate_ordinal=0,
        source_ordinal=0,
        box=(100, 300, 900, 340),
        line_ink_width_px=800,
        line_ink_height_px=40,
        line_indent_px=4,
        baseline_row_px=330,
        x_height_top_row_px=310,
        x_height_px=20,
        ascender_extent_px=30,
        descender_extent_px=10,
        stroke_width_px=2.0,
        skew_slope=0.001,
        segment_count=8,
        word_run_count=2,
        source_word_count=2 if reconciled else 3,
        words_reconciled=reconciled,
        words=words if reconciled else (),
    )


def _page_estimate(name: str, value: float, page_name: str) -> Estimate:
    return Estimate(
        name=name,
        value=value,
        unit="px",
        truth_class="derived",
        method="typography-page-median/v1",
        sample_count=1,
        evidence_pages=(page_name,),
        extensions={"line_sample_count": 2},
    )


def _pooled_estimate(name: str, value: float, page_names: tuple[str, ...]) -> Estimate:
    return Estimate(
        name=name,
        value=value,
        unit="px",
        truth_class="pooled",
        method="typography-median-mad/v1",
        sample_count=len(page_names),
        evidence_pages=page_names,
        extensions={"median_absolute_deviation": 0.5, "line_sample_count": 4},
    )


def make_page(page_name: str = "p2.png", *, evidence: bool = True) -> PageTypography:
    lines = (make_line(), make_line(measured=False)) if evidence else ()
    return PageTypography(
        page_name=page_name,
        page_class="normal_recto",
        scan_sha256=_sha256(),
        source_frame=CoordinateFrame(width=1000, height=1500),
        matched_line_count=2,
        measured_line_count=1,
        excluded_line_count=1,
        reconciled_line_count=1,
        baseline_pitch_count=0,
        evidence_sampled=evidence,
        lines=lines,
        estimates=(_page_estimate("x_height_px", 20.0, page_name),),
        skew_slope_median=0.001,
        skew_slope_mad=0.0,
        diagnostics=(ProfileDiagnostic(code="short_line", message="Short lines were excluded."),),
    )


def make_report() -> TypographyReport:
    pages = (make_page("p2.png"), make_page("p3.png", evidence=False))
    page_names = ("p2.png", "p3.png")
    book = BookTypography(
        project_id="projectID0001",
        pages=pages,
        styles=(
            StyleTypography(
                style="body",
                page_classes=("normal_recto", "normal_verso"),
                page_count=2,
                line_count=2,
                estimates=(_pooled_estimate("x_height_px", 20.0, page_names),),
                skew_slope_median=0.001,
                skew_slope_mad=0.0,
            ),
        ),
        estimates=(_pooled_estimate("x_height_px", 20.0, page_names),),
        accepted_page_count=2,
        measured_page_count=2,
        excluded_page_count=0,
        unknown_class_page_count=0,
        matched_line_count=4,
        measured_line_count=2,
        reconciled_line_count=2,
        skew_slope_median=0.001,
        skew_slope_mad=0.0,
    )
    return TypographyReport(
        tool_version="0.0.0",
        alignment_label="alignment.json",
        alignment_sha256=_sha256("b"),
        profile_label="profile.json",
        profile_sha256=_sha256("c"),
        methods={"algorithm": "segment-vertical-projection/v1"},
        thresholds={"maximum_skew_slope": 0.02},
        evidence_pages_per_book=12,
        books=(book,),
    )


def test_report_round_trips_through_json() -> None:
    report = make_report()
    assert TypographyReport.from_json(report.to_json()) == report


def test_report_json_is_deterministic_and_sorted() -> None:
    payload = make_report().to_json()
    assert payload == make_report().to_json()
    assert list(json.loads(payload)) == sorted(json.loads(payload))


def test_report_pins_both_version_literals() -> None:
    report = make_report()
    assert report.schema_version == TYPOGRAPHY_SCHEMA_VERSION
    assert report.algorithm_version == TYPOGRAPHY_ALGORITHM_VERSION
    with pytest.raises(ValueError, match="Unsupported algorithm_version"):
        _ = replace(report, algorithm_version="pgdp-typography/v2")
    with pytest.raises(ValueError, match="Unsupported schema_version"):
        _ = replace(report, schema_version=2)


def test_reader_rejects_an_unknown_algorithm_version() -> None:
    payload = json.loads(make_report().to_json())
    payload["algorithm_version"] = "pgdp-typography/v2"
    with pytest.raises(ValueError, match="Invalid PGDP typography report"):
        _ = TypographyReport.from_dict(payload)


def test_reader_preserves_unknown_optional_fields() -> None:
    payload = json.loads(make_report().to_json())
    payload["future_field"] = {"kept": True}
    payload["books"][0]["pages"][0]["future_page_field"] = 7
    restored = TypographyReport.from_dict(payload)
    assert restored.extensions["future_field"] == {"kept": True}
    assert restored.books[0].pages[0].extensions["future_page_field"] == 7
    assert json.loads(restored.to_json())["future_field"] == {"kept": True}


# --- Latent discipline -------------------------------------------------------------------


def test_real_report_payload_names_nothing_latent() -> None:
    assert find_latent_discipline_violations(json.loads(make_report().to_json())) == ()


@pytest.mark.parametrize(
    "latent_key",
    [
        "font_family",
        "nominal_point_size",
        "advance_width",
        "left_side_bearing",
        "native_word_space_px",
        "tracking_px",
        "pair_kerning_px",
        "leading_px",
        "physical_margin_mm",
        "x_height_pt",
    ],
)
def test_denylist_catches_every_latent_claim(latent_key: str) -> None:
    payload = json.loads(make_report().to_json())
    payload["books"][0]["pages"][0][latent_key] = 12
    violations = find_latent_discipline_violations(payload)
    assert violations
    assert all(latent_key in violation for violation in violations)


def test_denylist_covers_the_designs_latent_list() -> None:
    """Each latent value the design names is matched by at least one denylist substring."""

    for latent_key in (
        "font_identity",
        "point_size",
        "advance",
        "side_bearing",
        "word_space",
        "tracking",
        "kerning",
        "leading",
        "margin_mm",
    ):
        assert any(term in latent_key for term in LATENT_KEY_DENYLIST), latent_key


def test_every_estimate_is_uncalibrated_source_frame_pixels() -> None:
    payload = json.loads(make_report().to_json())
    estimate_count = 0
    for book in payload["books"]:
        groups = [book["estimates"], *(style["estimates"] for style in book["styles"])]
        groups.extend(page["estimates"] for page in book["pages"])
        for group in groups:
            for estimate in group:
                estimate_count += 1
                assert estimate["unit"] == "px"
                assert estimate["frame"] == "source"
                assert estimate["confidence"] is None
                assert estimate["confidence_kind"] == "uncalibrated"
    assert estimate_count > 0


@pytest.mark.parametrize(
    ("key", "value"),
    [("unit", "pt"), ("frame", "rectified"), ("confidence_kind", "calibrated")],
)
def test_estimate_shape_check_catches_a_drifted_estimate(key: str, value: str) -> None:
    payload = json.loads(make_report().to_json())
    payload["books"][0]["estimates"][0][key] = value
    violations = find_latent_discipline_violations(payload)
    assert any(key in violation for violation in violations)


def test_estimate_shape_check_catches_a_calibrated_confidence() -> None:
    payload = json.loads(make_report().to_json())
    payload["books"][0]["pages"][0]["estimates"][0]["confidence"] = 0.9
    assert any(
        "confidence must be null" in violation
        for violation in find_latent_discipline_violations(payload)
    )


def test_records_reject_a_non_pixel_estimate() -> None:
    bad = Estimate(
        name="x_height_px",
        value=20.0,
        unit="pt",
        truth_class="derived",
        method="typography-page-median/v1",
        sample_count=1,
        evidence_pages=("p2.png",),
    )
    with pytest.raises(ValueError, match="measured in px"):
        _ = replace(make_page(), estimates=(bad,))


# --- Line records ------------------------------------------------------------------------


def test_line_measurement_fields_are_all_or_nothing() -> None:
    with pytest.raises(ValueError, match="present or absent together"):
        _ = replace(make_line(), skew_slope=None)


def test_unmeasured_line_must_record_an_exclusion() -> None:
    with pytest.raises(ValueError, match="at least one exclusion"):
        _ = replace(make_line(measured=False), exclusions=())


def test_line_derived_extents_must_agree_with_the_box() -> None:
    with pytest.raises(ValueError, match="ascender_extent_px"):
        _ = replace(make_line(), ascender_extent_px=31)
    with pytest.raises(ValueError, match="descender_extent_px"):
        _ = replace(make_line(), descender_extent_px=11)
    with pytest.raises(ValueError, match="x_height_px"):
        _ = replace(make_line(), x_height_px=21)


def test_line_baseline_must_lie_inside_its_box() -> None:
    with pytest.raises(ValueError, match="inside the line box rows"):
        _ = replace(
            make_line(),
            baseline_row_px=340,
            x_height_px=30,
            ascender_extent_px=40,
            descender_extent_px=0,
        )


def test_unreconciled_line_keeps_its_measurements_and_emits_no_word_rows() -> None:
    line = make_line(reconciled=False)
    assert line.x_height_px == 20
    assert line.words == ()
    assert line.word_run_count == 2
    assert line.source_word_count == 3
    assert line.words_reconciled is False


def test_unreconciled_line_rejects_word_rows() -> None:
    with pytest.raises(ValueError, match="must not emit word rows"):
        _ = replace(make_line(reconciled=False), words=make_line().words)


def test_word_counts_require_a_measured_x_height() -> None:
    with pytest.raises(ValueError, match="exactly when the line's x-height was measured"):
        _ = replace(
            make_line(measured=False),
            word_run_count=2,
            source_word_count=2,
            words_reconciled=True,
        )


def test_word_count_fields_are_all_or_nothing() -> None:
    with pytest.raises(ValueError, match="present or absent together"):
        _ = replace(make_line(), words_reconciled=None, words=())


def test_word_boxes_must_stay_inside_their_line() -> None:
    escaping = (
        WordTypography(ordinal=0, box=(100, 306, 180, 330), following_gap_px=6),
        WordTypography(ordinal=1, box=(186, 306, 1200, 332)),
    )
    with pytest.raises(ValueError, match="stay inside their line box"):
        _ = replace(make_line(), words=escaping)


def test_word_ordinals_must_be_consecutive() -> None:
    shuffled = (
        WordTypography(ordinal=1, box=(100, 306, 180, 330), following_gap_px=6),
        WordTypography(ordinal=0, box=(186, 306, 260, 332)),
    )
    with pytest.raises(ValueError, match="consecutive from zero"):
        _ = replace(make_line(), words=shuffled)


# --- Page, book, and report records -------------------------------------------------------


def test_page_line_counts_must_account_for_every_matched_line() -> None:
    with pytest.raises(ValueError, match="account for every matched line"):
        _ = replace(make_page(), excluded_line_count=0)


def test_only_an_evidence_page_emits_line_rows() -> None:
    with pytest.raises(ValueError, match="Only an evidence-sampled page"):
        _ = replace(make_page(), evidence_sampled=False)


def test_evidence_page_emits_one_row_per_matched_line() -> None:
    with pytest.raises(ValueError, match="one row per matched line"):
        _ = replace(make_page(), lines=(make_line(),))


def test_page_estimates_may_only_cite_their_own_page() -> None:
    stray = _page_estimate("x_height_px", 20.0, "p9.png")
    with pytest.raises(ValueError, match="cite pages this record contains"):
        _ = replace(make_page(), estimates=(stray,))


def test_book_estimates_may_only_cite_pages_the_book_contains() -> None:
    report = make_report()
    stray = _pooled_estimate("x_height_px", 20.0, ("p9.png",))
    with pytest.raises(ValueError, match="cite pages this record contains"):
        _ = replace(report.books[0], estimates=(stray,))


def test_book_pages_must_use_natural_page_order() -> None:
    report = make_report()
    reversed_pages = tuple(reversed(report.books[0].pages))
    with pytest.raises(ValueError, match="natural page order"):
        _ = replace(report.books[0], pages=reversed_pages)


def test_style_page_classes_must_match_the_shipped_classes() -> None:
    with pytest.raises(ValueError, match="shipped page classes"):
        _ = StyleTypography(
            style="body", page_classes=("normal_recto",), page_count=1, line_count=1
        )


def test_skew_is_reported_as_a_bare_median_not_an_estimate() -> None:
    payload = json.loads(make_report().to_json())
    book = payload["books"][0]
    assert book["skew_slope_median"] == 0.001
    assert all(estimate["name"] != "skew_slope" for estimate in book["estimates"])
    with pytest.raises(ValueError, match="present or absent together"):
        _ = replace(make_page(), skew_slope_mad=None)


def test_report_bounds_the_evidence_sample_per_book() -> None:
    report = make_report()
    with pytest.raises(ValueError, match="more evidence pages than the report's bound"):
        _ = replace(report, evidence_pages_per_book=0)


def test_report_requires_both_provenance_hashes() -> None:
    report = make_report()
    with pytest.raises(ValueError, match="alignment_sha256"):
        _ = replace(report, alignment_sha256="short")
    with pytest.raises(ValueError, match="profile_sha256"):
        _ = replace(report, profile_sha256="short")


def test_report_rejects_a_label_with_path_components() -> None:
    with pytest.raises(ValueError, match="without path components"):
        _ = replace(make_report(), alignment_label="a/alignment.json")


def test_schema_is_generated_from_typography_report_input() -> None:
    assert SCHEMA_ARTIFACT.read_text(encoding="utf-8") == TypographyReport.json_schema()


def test_schema_pins_the_algorithm_version() -> None:
    schema = json.loads(TypographyReport.json_schema())
    assert schema["properties"]["algorithm_version"]["const"] == TYPOGRAPHY_ALGORITHM_VERSION
