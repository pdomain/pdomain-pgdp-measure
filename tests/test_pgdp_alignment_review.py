"""Tests for review-only PGDP alignment overlays and ledger summaries."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256 as hashlib_sha256
from pathlib import Path

import pytest
from PIL import Image

import pdomain_pgdp_measure as pgdp
from pdomain_pgdp_measure import alignment_review
from pdomain_pgdp_measure.alignment_models import (
    AlignmentOperation,
    AlignmentReport,
    PageAlignment,
    ProjectAlignment,
    WireLineCandidate,
    WireSourceLine,
)
from pdomain_pgdp_measure.alignment_review import (
    MINIMUM_ACCEPTED_PAGES_PER_BOOK,
    BookYield,
    ReviewCategory,
    ReviewCounts,
    ReviewedLine,
    ReviewGateFailure,
    ReviewLedger,
    ReviewPage,
    classify_page,
    render_alignment_overlay,
    summarize_book_yield,
    summarize_review,
    validate_review_gate,
)
from pdomain_pgdp_measure.profile_models import CoordinateFrame


def _sha256() -> str:
    return "a" * 64


def test_review_helpers_are_public() -> None:
    assert pgdp.render_alignment_overlay is render_alignment_overlay
    assert pgdp.summarize_review is summarize_review
    assert pgdp.validate_review_gate is validate_review_gate
    assert pgdp.summarize_book_yield is summarize_book_yield
    assert pgdp.BookYield is BookYield
    assert pgdp.MINIMUM_ACCEPTED_PAGES_PER_BOOK == MINIMUM_ACCEPTED_PAGES_PER_BOOK


def _page(
    *,
    accepted: bool = True,
    exclusions: tuple[str, ...] = (),
    normalized_cost: float = 0.1,
    mean_width_residual: float | None = 0.1,
    mean_indentation_residual: float | None = 0.2,
    uniqueness_margin: float = 0.3,
) -> PageAlignment:
    source_lines = (
        WireSourceLine(
            ordinal=0,
            byte_start=0,
            byte_end=2,
            content_byte_start=0,
            content_byte_end=1,
            separator_byte_start=1,
            separator_byte_end=2,
            original_text="a",
            separator_text="\n",
            visible_text="a",
            matching_eligible=True,
            style_fitting_eligible=True,
        ),
        WireSourceLine(
            ordinal=1,
            byte_start=2,
            byte_end=3,
            content_byte_start=2,
            content_byte_end=3,
            separator_byte_start=3,
            separator_byte_end=3,
            original_text="b",
            separator_text="",
            visible_text="b",
            matching_eligible=True,
            style_fitting_eligible=True,
        ),
    )

    candidates = (
        WireLineCandidate(
            ordinal=0,
            box=(20, 10, 60, 20),
            width=40,
            height=10,
            fill_ratio=0.2,
            component_count=1,
            foreground_pixels=80,
        ),
        WireLineCandidate(
            ordinal=1,
            box=(20, 30, 60, 40),
            width=40,
            height=10,
            fill_ratio=0.2,
            component_count=1,
            foreground_pixels=80,
        ),
    )
    state = "accepted" if accepted else "proposed"
    return PageAlignment(
        page_name="p1.png",
        f2_sha256=_sha256(),
        scan_sha256=_sha256(),
        source_frame=CoordinateFrame(width=80, height=60),
        source_lines=source_lines,
        formatting_spans=(),
        candidates=candidates,
        operations=(
            AlignmentOperation(kind="match", source_ordinal=0, candidate_ordinal=0, cost=0.1),
            AlignmentOperation(
                kind="skip_image", source_ordinal=None, candidate_ordinal=1, cost=1.0
            ),
            AlignmentOperation(
                kind="skip_text", source_ordinal=1, candidate_ordinal=None, cost=1.0
            ),
        ),
        total_cost=2.1,
        second_best_cost=None,
        normalized_cost=normalized_cost,
        matched_count=1,
        matched_ratio=0.5,
        uniqueness_margin=uniqueness_margin,
        mean_width_residual=mean_width_residual,
        mean_indentation_residual=mean_indentation_residual,
        state=state,
        accepted=accepted,
        exclusions=exclusions,
    )


def _report(*pages: PageAlignment) -> AlignmentReport:
    return AlignmentReport(
        tool_version="0.1.0",
        profile_label="profile.json",
        profile_sha256=_sha256(),
        methods={},
        thresholds={},
        projects=(
            ProjectAlignment(
                project_id="book", pages=tuple(sorted(pages, key=lambda page: page.page_name))
            ),
        ),
    )


def _two_match_page() -> PageAlignment:
    return replace(
        _page(),
        operations=(
            AlignmentOperation(kind="match", source_ordinal=0, candidate_ordinal=0, cost=0.1),
            AlignmentOperation(kind="match", source_ordinal=1, candidate_ordinal=1, cost=0.1),
        ),
        total_cost=0.2,
        second_best_cost=0.3,
        matched_count=2,
        matched_ratio=1.0,
    )


def _with_scan_hash(page: PageAlignment, scan_path: Path) -> PageAlignment:
    return replace(page, scan_sha256=hashlib_sha256(scan_path.read_bytes()).hexdigest())


def test_overlay_is_source_sized_and_report_is_unchanged(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan.png"
    Image.new("L", (80, 60), color=255).save(scan_path)
    page = _with_scan_hash(_page(), scan_path)
    before = page.to_dict()

    overlay = render_alignment_overlay(scan_path, page, tmp_path / "overlay.png")

    assert overlay.mode == "RGB"
    assert overlay.size == (80, 60)
    assert overlay.getpixel((20, 10)) == (0, 192, 0)
    assert overlay.getpixel((20, 30)) == (224, 0, 0)
    assert page.to_dict() == before
    assert (tmp_path / "overlay.png").is_file()
    assert any(overlay.getpixel((x, y)) != (255, 255, 255) for x in range(20) for y in range(60))


def test_overlay_marks_proposed_matches_amber(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan.png"
    Image.new("RGB", (80, 60), color="white").save(scan_path)

    overlay = render_alignment_overlay(
        scan_path,
        _with_scan_hash(_page(accepted=False), scan_path),
        tmp_path / "overlay.png",
    )

    assert overlay.getpixel((20, 10)) == (224, 160, 0)


def test_overlay_rejects_same_sized_scan_hash_mismatch_without_writes(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan.png"
    output_path = tmp_path / "overlay.png"
    Image.new("RGB", (80, 60), color="white").save(scan_path)
    scan_bytes = scan_path.read_bytes()
    _ = output_path.write_bytes(b"previous overlay")

    with pytest.raises(ValueError, match="scan_sha256"):
        _ = render_alignment_overlay(scan_path, _page(), output_path)

    assert scan_path.read_bytes() == scan_bytes
    assert output_path.read_bytes() == b"previous overlay"


def test_overlay_decodes_the_immutable_bytes_whose_hash_was_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan_path = tmp_path / "scan.png"
    output_path = tmp_path / "overlay.png"
    Image.new("RGB", (80, 60), color="white").save(scan_path)
    page = _with_scan_hash(_page(), scan_path)
    expected_hash = page.scan_sha256
    assert expected_hash is not None

    class MutatingDigest:
        def hexdigest(self) -> str:
            Image.new("RGB", (80, 60), color="black").save(scan_path)
            return expected_hash

    def mutate_after_verified_hash(_scan_file: object, _algorithm: str) -> MutatingDigest:
        return MutatingDigest()

    monkeypatch.setattr(alignment_review, "file_digest", mutate_after_verified_hash)

    overlay = render_alignment_overlay(scan_path, page, output_path)

    assert overlay.getpixel((79, 59)) == (255, 255, 255)
    with Image.open(output_path) as saved_overlay:
        assert saved_overlay.getpixel((79, 59)) == (255, 255, 255)


def test_overlay_rejects_scan_as_output_without_changing_source(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan.png"
    output_path = tmp_path / "overlay.png"
    Image.new("RGB", (80, 60), color="white").save(scan_path)
    output_path.symlink_to(scan_path)
    source_bytes = scan_path.read_bytes()

    with pytest.raises(ValueError, match="scan path"):
        _ = render_alignment_overlay(scan_path, _page(), output_path)

    assert scan_path.read_bytes() == source_bytes


def test_overlay_rejects_output_inside_supplied_corpus_root(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    scan_path = tmp_path / "scan.png"
    Image.new("RGB", (80, 60), color="white").save(scan_path)

    with pytest.raises(ValueError, match="outside the corpus root"):
        _ = render_alignment_overlay(
            scan_path,
            _page(),
            corpus_root / "overlay.png",
            corpus_root=corpus_root,
        )


def test_overlay_keeps_existing_output_when_atomic_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan_path = tmp_path / "scan.png"
    output_path = tmp_path / "overlay.png"
    Image.new("RGB", (80, 60), color="white").save(scan_path)
    _ = output_path.write_bytes(b"previous overlay")

    def interrupted_replace(source: Path, destination: str | Path) -> Path:
        assert source != destination
        raise OSError("replace interrupted")

    monkeypatch.setattr(Path, "replace", interrupted_replace)

    with pytest.raises(OSError, match="replace interrupted"):
        _ = render_alignment_overlay(scan_path, _with_scan_hash(_page(), scan_path), output_path)

    assert output_path.read_bytes() == b"previous overlay"
    assert tuple(tmp_path.glob(".overlay.png.*.png")) == ()


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("probable_multi_column", ReviewCategory.DECLARED_COMPLEX),
        ("table_like", ReviewCategory.DECLARED_COMPLEX),
        ("illustration_marker", ReviewCategory.DECLARED_COMPLEX),
        ("border_dominated", ReviewCategory.DECLARED_COMPLEX),
        ("high_foreground_ratio", ReviewCategory.DECLARED_COMPLEX),
        ("f2_missing", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("f2_page_missing", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("page_missing", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("image_missing", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("image_unreadable", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("image_decode_failed", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("profile_page_missing", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("foreground_bounds_unavailable", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("insufficient_ink_bands", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("blank_page", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("fragmented_band", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("line_count_out_of_range", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("malformed_control", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("scan_hash_mismatch", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("inherited_profile_diagnostic", ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        ("source_changed", ReviewCategory.SOURCE_CHANGED),
        ("matched_ratio_below_minimum", ReviewCategory.ELIGIBLE),
        ("line_count_difference_exceeds_maximum", ReviewCategory.ELIGIBLE),
        ("normalized_cost_exceeds_maximum", ReviewCategory.ELIGIBLE),
        ("uniqueness_margin_below_minimum", ReviewCategory.ELIGIBLE),
        ("persistent_gutter", ReviewCategory.DECLARED_COMPLEX),
    ],
)
def test_classify_page_maps_each_known_exclusion_code(code: str, expected: ReviewCategory) -> None:
    assert classify_page(_page(accepted=False, exclusions=(code,))).category is expected


@pytest.mark.parametrize(
    ("codes", "expected"),
    [
        (("source_changed", "blank_page"), ReviewCategory.SOURCE_CHANGED),
        (("source_changed", "probable_multi_column"), ReviewCategory.SOURCE_CHANGED),
        (("source_changed", "normalized_cost_exceeds_maximum"), ReviewCategory.SOURCE_CHANGED),
        (("blank_page", "probable_multi_column"), ReviewCategory.UNAVAILABLE_OR_MALFORMED),
        (
            ("blank_page", "normalized_cost_exceeds_maximum"),
            ReviewCategory.UNAVAILABLE_OR_MALFORMED,
        ),
        (
            ("probable_multi_column", "normalized_cost_exceeds_maximum"),
            ReviewCategory.DECLARED_COMPLEX,
        ),
    ],
)
def test_classify_page_applies_priority_to_multiple_exclusions(
    codes: tuple[str, ...], expected: ReviewCategory
) -> None:
    assert classify_page(_page(accepted=False, exclusions=codes)).category is expected


def test_classify_page_rejects_unknown_m15b_code() -> None:
    with pytest.raises(ValueError, match="Unknown M15b exclusion code"):
        _ = classify_page(_page(accepted=False, exclusions=("unrecognized",)))


def test_review_counts_rejects_nonexhaustive_identity() -> None:
    with pytest.raises(ValueError, match="selected_total"):
        _ = ReviewCounts(
            selected_total=4,
            declared_complex=1,
            unavailable_or_malformed=1,
            source_changed=1,
            eligible=0,
        )


def test_direct_review_ledger_rejects_mismatched_record_counts() -> None:
    with pytest.raises(ValueError):
        _ = ReviewLedger(
            pages=(_page(),),
            references=(),
            classifications=(),
            reviewed_lines=(),
        )


def test_direct_review_ledger_rejects_page_reference_mismatch() -> None:
    page = _page()

    with pytest.raises(ValueError):
        _ = ReviewLedger(
            pages=(page,),
            references=(ReviewPage("book", "wrong.png"),),
            classifications=(classify_page(page),),
            reviewed_lines=(),
        )


def test_direct_review_ledger_rejects_classification_mismatch() -> None:
    page = _page()

    with pytest.raises(ValueError):
        _ = ReviewLedger(
            pages=(page,),
            references=(ReviewPage("book", "p1.png"),),
            classifications=(classify_page(_page(accepted=False, exclusions=("table_like",))),),
            reviewed_lines=(),
        )


def test_direct_review_ledger_rejects_review_operation_mismatch() -> None:
    page = _page()

    with pytest.raises(ValueError):
        _ = ReviewLedger(
            pages=(page,),
            references=(ReviewPage("book", "p1.png"),),
            classifications=(classify_page(page),),
            reviewed_lines=(ReviewedLine("book", "p1.png", 0, 0, "match", 0, 1, "correct"),),
        )


def test_review_summary_counts_disjoint_categories_and_preserves_null_residuals() -> None:
    pages = (
        replace(
            _page(),
            page_name="source-changed.png",
            accepted=False,
            state="excluded",
            exclusions=("source_changed",),
        ),
        replace(
            _page(),
            page_name="missing.png",
            accepted=False,
            state="excluded",
            exclusions=("blank_page",),
        ),
        replace(
            _page(),
            page_name="table.png",
            accepted=False,
            state="excluded",
            exclusions=("table_like",),
        ),
        replace(_two_match_page(), page_name="accepted.png"),
        replace(
            _page(),
            page_name="proposed.png",
            accepted=False,
            state="proposed",
            exclusions=("normalized_cost_exceeds_maximum",),
            normalized_cost=0.4,
            mean_width_residual=None,
            mean_indentation_residual=None,
            uniqueness_margin=0.0,
        ),
    )
    reviewed_lines = (
        ReviewedLine("book", "accepted.png", 0, 0, "match", 0, 0, "correct"),
        ReviewedLine("book", "accepted.png", 0, 0, "match", 1, 1, "incorrect"),
        ReviewedLine("book", "proposed.png", 1, None, "skip_text", 1, None, "correct"),
    )

    summary = summarize_review(_report(*pages), reviewed_lines)

    assert summary.counts.selected_total == 5
    assert summary.counts.source_changed == 1
    assert summary.counts.unavailable_or_malformed == 1
    assert summary.counts.declared_complex == 1
    assert summary.counts.eligible == 2
    assert summary.accepted_line_precision == 0.5
    assert summary.eligible_page_coverage == 0.5
    assert summary.declared_complex_accepted == 0
    assert summary.normalized_costs == (0.1, 0.1, 0.4, 0.1, 0.1)
    assert summary.width_residuals == (0.1, 0.1, None, 0.1, 0.1)
    assert summary.indentation_residuals == (0.2, 0.2, None, 0.2, 0.2)
    assert summary.uniqueness_margins == (0.3, 0.3, 0.0, 0.3, 0.3)


def test_review_summary_handles_no_reviewed_matches_without_inventing_precision() -> None:
    summary = summarize_review(_report(_page()), ())

    assert summary.accepted_line_precision is None
    assert summary.confidence is None
    assert summary.confidence_kind == "uncalibrated"


def test_review_summary_uses_explicit_report_page_selection() -> None:
    report = _report(_page(), replace(_page(), page_name="p2.png"))

    summary = summarize_review(
        report,
        (),
        selected_pages=(ReviewPage("book", "p2.png"),),
    )

    assert summary.counts.selected_total == 1
    assert summary.normalized_costs == (0.1,)


@pytest.mark.parametrize(
    "reviewed_lines",
    [
        (ReviewedLine("book", "p1.png", 0, 0, "match", 0, 1, "correct"),),
        (ReviewedLine("book", "absent.png", 0, 0, "match", 0, 0, "correct"),),
        (
            ReviewedLine("book", "p1.png", 0, 0, "match", 0, 0, "correct"),
            ReviewedLine("book", "p1.png", 0, 0, "match", 0, 0, "correct"),
        ),
    ],
)
def test_review_summary_rejects_unidentified_or_duplicate_operations(
    reviewed_lines: tuple[ReviewedLine, ...],
) -> None:
    with pytest.raises(ValueError):
        _ = summarize_review(_report(_page()), reviewed_lines)


def test_review_summary_rejects_match_on_a_proposed_page() -> None:
    page = replace(
        _page(),
        accepted=False,
        state="proposed",
        exclusions=("normalized_cost_exceeds_maximum",),
    )
    reviewed_line = ReviewedLine("book", "p1.png", 0, 0, "match", 0, 0, "correct")

    with pytest.raises(ValueError, match="accepted page"):
        _ = summarize_review(_report(page), (reviewed_line,))


@pytest.mark.parametrize(
    ("precision", "coverage", "complex_accepted", "passed", "failures"),
    [
        (0.98, 0.70, 0, True, ()),
        (0.979, 0.70, 0, False, (ReviewGateFailure.ACCEPTED_LINE_PRECISION,)),
        (0.98, 0.70, 1, False, (ReviewGateFailure.DECLARED_COMPLEX_ACCEPTED,)),
        (0.98, 0.0, 0, True, ()),
        (0.98, None, 0, True, ()),
        (
            None,
            None,
            1,
            False,
            (
                ReviewGateFailure.ACCEPTED_LINE_PRECISION,
                ReviewGateFailure.DECLARED_COMPLEX_ACCEPTED,
            ),
        ),
    ],
)
def test_review_gate_enforces_exact_thresholds(
    precision: float | None,
    coverage: float | None,
    complex_accepted: int,
    passed: bool,
    failures: tuple[ReviewGateFailure, ...],
) -> None:
    summary = summarize_review(_report(_page()), ())
    counts = (
        summary.counts
        if complex_accepted == 0
        else ReviewCounts(
            selected_total=1,
            declared_complex=1,
            unavailable_or_malformed=0,
            source_changed=0,
            eligible=0,
        )
    )
    threshold_summary = replace(
        summary,
        counts=counts,
        accepted_line_precision=precision,
        eligible_page_coverage=coverage,
        declared_complex_accepted=complex_accepted,
    )

    result = validate_review_gate(threshold_summary)

    assert result.passed is passed
    assert result.failures == failures


def _accepted_pages(count: int, *, prefix: str = "p") -> tuple[PageAlignment, ...]:
    return tuple(replace(_page(), page_name=f"{prefix}{index:04d}.png") for index in range(count))


def _books(*projects: ProjectAlignment) -> AlignmentReport:
    return AlignmentReport(
        tool_version="0.1.0",
        profile_label="profile.json",
        profile_sha256=_sha256(),
        methods={},
        thresholds={},
        projects=projects,
    )


@pytest.mark.parametrize(
    ("accepted_count", "admitted"),
    [
        (MINIMUM_ACCEPTED_PAGES_PER_BOOK, True),
        (MINIMUM_ACCEPTED_PAGES_PER_BOOK + 1, True),
        (MINIMUM_ACCEPTED_PAGES_PER_BOOK - 1, False),
        (0, False),
    ],
)
def test_book_yield_admits_only_at_the_accepted_page_minimum(
    accepted_count: int, admitted: bool
) -> None:
    pages = _accepted_pages(accepted_count)
    filler = replace(
        _page(accepted=False, exclusions=("uniqueness_margin_below_minimum",)),
        page_name="zzz.png",
    )
    report = _books(ProjectAlignment(project_id="book", pages=(*pages, filler)))

    (yielded,) = summarize_book_yield(report)

    assert yielded.project_id == "book"
    assert yielded.accepted_pages == accepted_count
    assert yielded.eligible_pages == accepted_count + 1
    assert yielded.admitted is admitted


def test_book_yield_reports_one_entry_per_book_in_report_order() -> None:
    report = _books(
        ProjectAlignment(project_id="book2", pages=_accepted_pages(1)),
        ProjectAlignment(project_id="book10", pages=_accepted_pages(2)),
    )

    yields = summarize_book_yield(report)

    assert [item.project_id for item in yields] == ["book2", "book10"]
    assert [item.accepted_pages for item in yields] == [1, 2]


def test_book_yield_keeps_non_eligible_pages_out_of_coverage() -> None:
    complex_page = replace(
        _page(accepted=False, exclusions=("table_like",)),
        page_name="zzz.png",
        state="excluded",
    )
    report = _books(ProjectAlignment(project_id="book", pages=(*_accepted_pages(3), complex_page)))

    (yielded,) = summarize_book_yield(report)

    assert yielded.eligible_pages == 3
    assert yielded.accepted_pages == 3
    assert yielded.eligible_page_coverage == 1.0


def test_book_yield_reports_a_book_with_no_page_at_all() -> None:
    report = _books(
        ProjectAlignment(project_id="book1", pages=_accepted_pages(1)),
        ProjectAlignment(project_id="book2", pages=()),
    )

    yields = summarize_book_yield(report)

    assert [item.project_id for item in yields] == ["book1", "book2"]
    assert yields[1].eligible_pages == 0
    assert yields[1].eligible_page_coverage is None
    assert yields[1].admitted is False


def test_book_yield_over_a_selection_covers_only_the_selected_books() -> None:
    report = _books(
        ProjectAlignment(project_id="book1", pages=_accepted_pages(1)),
        ProjectAlignment(project_id="book2", pages=_accepted_pages(1)),
    )

    yields = summarize_book_yield(report, selected_pages=(ReviewPage("book2", "p0000.png"),))

    assert [item.project_id for item in yields] == ["book2"]


def test_book_yield_reports_a_book_with_no_eligible_page() -> None:
    complex_page = replace(
        _page(accepted=False, exclusions=("table_like",)),
        state="excluded",
    )
    report = _books(ProjectAlignment(project_id="book", pages=(complex_page,)))

    (yielded,) = summarize_book_yield(report)

    assert yielded.eligible_pages == 0
    assert yielded.eligible_page_coverage is None
    assert yielded.admitted is False


@pytest.mark.parametrize(
    ("eligible", "accepted", "coverage", "admitted"),
    [
        (10, 11, 1.0, False),
        (10, 5, 0.4, False),
        (10, 5, 0.5, True),
        (0, 0, 0.0, False),
    ],
)
def test_book_yield_rejects_inconsistent_fields(
    eligible: int, accepted: int, coverage: float, admitted: bool
) -> None:
    with pytest.raises(ValueError):
        _ = BookYield(
            project_id="book",
            eligible_pages=eligible,
            accepted_pages=accepted,
            eligible_page_coverage=coverage,
            admitted=admitted,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_review_summary_rejects_invalid_precision(value: float) -> None:
    summary = summarize_review(_report(_page()), ())

    with pytest.raises(ValueError, match="accepted_line_precision"):
        _ = replace(summary, accepted_line_precision=value)


@pytest.mark.parametrize("value", [float("nan"), float("-inf"), -0.01, 1.01])
def test_review_summary_rejects_invalid_coverage(value: float) -> None:
    summary = summarize_review(_report(_page()), ())

    with pytest.raises(ValueError, match="eligible_page_coverage"):
        _ = replace(summary, eligible_page_coverage=value)


def test_review_summary_rejects_invalid_complex_accepted_count() -> None:
    summary = summarize_review(_report(_page()), ())

    with pytest.raises(ValueError, match="declared_complex_accepted"):
        _ = replace(summary, declared_complex_accepted=-1)


def test_review_summary_rejects_invalid_normalized_costs() -> None:
    summary = summarize_review(_report(_page()), ())

    with pytest.raises(ValueError, match="normalized_costs"):
        _ = replace(summary, normalized_costs=(float("nan"),))


def test_review_summary_rejects_invalid_width_residuals() -> None:
    summary = summarize_review(_report(_page()), ())

    with pytest.raises(ValueError, match="width_residuals"):
        _ = replace(summary, width_residuals=(float("inf"),))


def test_review_summary_rejects_invalid_indentation_residuals() -> None:
    summary = summarize_review(_report(_page()), ())

    with pytest.raises(ValueError, match="indentation_residuals"):
        _ = replace(summary, indentation_residuals=(-0.1,))


def test_review_summary_rejects_distribution_length_mismatch() -> None:
    summary = summarize_review(_report(_page()), ())

    with pytest.raises(ValueError, match="selected_total"):
        _ = replace(summary, uniqueness_margins=())
