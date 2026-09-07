"""Review-only overlays and summaries for PGDP source-line alignment."""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from hashlib import file_digest
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pdomain_pgdp_measure.alignment_models import AlignmentReport, PageAlignment
    from pdomain_pgdp_measure.typography_models import LineTypography


class ReviewCategory(StrEnum):
    """One disjoint page category for a bounded alignment review."""

    DECLARED_COMPLEX = "declared_complex"
    UNAVAILABLE_OR_MALFORMED = "unavailable_or_malformed"
    SOURCE_CHANGED = "source_changed"
    ELIGIBLE = "eligible"


class ReviewGateFailure(StrEnum):
    """One unmet bounded-review acceptance threshold."""

    ACCEPTED_LINE_PRECISION = "accepted_line_precision"
    DECLARED_COMPLEX_ACCEPTED = "declared_complex_accepted"


_DECLARED_COMPLEX_CODES = frozenset(
    {
        "probable_multi_column",
        "persistent_gutter",
        "table_like",
        "illustration_marker",
        "border_dominated",
        "high_foreground_ratio",
    }
)
_UNAVAILABLE_OR_MALFORMED_CODES = frozenset(
    {
        "f2_missing",
        "f2_page_missing",
        "page_missing",
        "image_missing",
        "image_unreadable",
        "image_decode_failed",
        "profile_page_missing",
        "foreground_bounds_unavailable",
        "insufficient_ink_bands",
        "blank_page",
        "fragmented_band",
        "line_count_out_of_range",
        "malformed_control",
        "scan_hash_mismatch",
        "inherited_profile_diagnostic",
    }
)
_SOURCE_CHANGED_CODE = "source_changed"
_ALIGNMENT_QUALITY_CODES = frozenset(
    {
        "matched_ratio_below_minimum",
        "line_count_difference_exceeds_maximum",
        "normalized_cost_exceeds_maximum",
        "uniqueness_margin_below_minimum",
    }
)
_KNOWN_EXCLUSION_CODES = (
    _DECLARED_COMPLEX_CODES
    | _UNAVAILABLE_OR_MALFORMED_CODES
    | _ALIGNMENT_QUALITY_CODES
    | {_SOURCE_CHANGED_CODE}
)

# Seed value, not a calibrated one: nothing fits typography from an aligned page yet, so no
# downstream consumer can state how many pages a per-book style fit needs. See
# docs/specs/2026-08-31-pgdp-whole-book-yield-gate-design.md.
MINIMUM_ACCEPTED_PAGES_PER_BOOK: Final = 30

_ACCEPTED_MATCH_COLOR = (0, 192, 0)
_REJECTED_MATCH_COLOR = (224, 160, 0)
_SKIPPED_CANDIDATE_COLOR = (224, 0, 0)
_BASELINE_COLOR = (0, 96, 255)
_X_HEIGHT_COLOR = (224, 0, 192)
_LINE_BOX_COLOR = (128, 128, 128)
_TYPOGRAPHY_OVERLAY_MARGIN_PX: Final = 8

ObservedOperation = Literal["match", "skip_image", "skip_text"]
ReviewerDecision = Literal["correct", "incorrect"]


@dataclass(frozen=True, slots=True)
class PageClassification:
    """The category assigned to one selected page."""

    category: ReviewCategory
    exclusions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReviewPage:
    """One report-identified page selected for visual review."""

    project_id: str
    page_name: str

    def __post_init__(self) -> None:
        _validate_identity(self.project_id, name="project_id")
        _validate_identity(self.page_name, name="page_name")


@dataclass(frozen=True, slots=True)
class ReviewedLine:
    """One human-reviewed expected mapping and observed operation."""

    project_id: str
    page_name: str
    intended_source_ordinal: int | None
    intended_candidate_ordinal: int | None
    observed_operation: ObservedOperation
    observed_source_ordinal: int | None
    observed_candidate_ordinal: int | None
    reviewer_decision: ReviewerDecision

    def __post_init__(self) -> None:
        _validate_identity(self.project_id, name="project_id")
        _validate_identity(self.page_name, name="page_name")
        _validate_ordinal(self.intended_source_ordinal, name="intended_source_ordinal")
        _validate_ordinal(self.intended_candidate_ordinal, name="intended_candidate_ordinal")
        _validate_ordinal(self.observed_source_ordinal, name="observed_source_ordinal")
        _validate_ordinal(self.observed_candidate_ordinal, name="observed_candidate_ordinal")
        if self.observed_operation == "match":
            if self.observed_source_ordinal is None or self.observed_candidate_ordinal is None:
                raise ValueError("Match review rows require source and candidate ordinals.")
        elif self.observed_operation == "skip_image":
            if self.observed_source_ordinal is not None or self.observed_candidate_ordinal is None:
                raise ValueError("skip_image review rows require only a candidate ordinal.")
        elif self.observed_operation == "skip_text":
            if self.observed_source_ordinal is None or self.observed_candidate_ordinal is not None:
                raise ValueError("skip_text review rows require only a source ordinal.")
        else:
            raise ValueError("Observed review operation is unsupported.")
        if self.reviewer_decision not in {"correct", "incorrect"}:
            raise ValueError("Reviewer decision is unsupported.")


@dataclass(frozen=True, slots=True)
class ReviewCounts:
    """The four exhaustive counts for a selected page set."""

    selected_total: int
    declared_complex: int
    unavailable_or_malformed: int
    source_changed: int
    eligible: int

    def __post_init__(self) -> None:
        values: tuple[object, ...] = (
            self.selected_total,
            self.declared_complex,
            self.unavailable_or_malformed,
            self.source_changed,
            self.eligible,
        )
        for value in values:
            _validate_nonnegative_integer(value, name="Review counts")
        if self.selected_total != (
            self.declared_complex
            + self.unavailable_or_malformed
            + self.source_changed
            + self.eligible
        ):
            raise ValueError("selected_total must equal the four disjoint review counts.")


@dataclass(frozen=True, slots=True)
class ReviewLedger:
    """Typed page classifications and human-reviewed line decisions."""

    pages: tuple[PageAlignment, ...]
    references: tuple[ReviewPage, ...]
    classifications: tuple[PageClassification, ...]
    reviewed_lines: tuple[ReviewedLine, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pages", tuple(self.pages))
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "classifications", tuple(self.classifications))
        object.__setattr__(self, "reviewed_lines", tuple(self.reviewed_lines))
        if not len(self.pages) == len(self.references) == len(self.classifications):
            raise ValueError(
                "Ledger pages, references, and classifications must have equal counts."
            )
        if len(set(self.references)) != len(self.references):
            raise ValueError("Ledger references must not repeat a project/page identity.")

        available: dict[ReviewPage, PageAlignment] = {}
        for page, reference, classification in zip(
            self.pages,
            self.references,
            self.classifications,
            strict=True,
        ):
            if page.page_name != reference.page_name:
                raise ValueError("Ledger page names must match their review references.")
            if classification != classify_page(page):
                raise ValueError("Ledger classifications must match their report pages.")
            available[reference] = page
        _validate_reviewed_lines(
            self.reviewed_lines,
            available,
            frozenset(self.references),
        )

    @classmethod
    def from_report(
        cls,
        report: AlignmentReport,
        reviewed_lines: Sequence[ReviewedLine] = (),
        *,
        selected_pages: Sequence[ReviewPage] | None = None,
    ) -> ReviewLedger:
        """Build the complete ledger from report-backed selected pages."""

        available = _report_pages(report)
        references = _selected_references(available, selected_pages)
        pages = tuple(available[reference] for reference in references)
        rows = tuple(reviewed_lines)
        _validate_reviewed_lines(rows, available, frozenset(references))
        return cls(
            pages=pages,
            references=references,
            classifications=tuple(classify_page(page) for page in pages),
            reviewed_lines=rows,
        )

    @property
    def counts(self) -> ReviewCounts:
        """Return exhaustive category counts for selected pages."""

        return ReviewCounts(
            selected_total=len(self.classifications),
            declared_complex=sum(
                item.category is ReviewCategory.DECLARED_COMPLEX for item in self.classifications
            ),
            unavailable_or_malformed=sum(
                item.category is ReviewCategory.UNAVAILABLE_OR_MALFORMED
                for item in self.classifications
            ),
            source_changed=sum(
                item.category is ReviewCategory.SOURCE_CHANGED for item in self.classifications
            ),
            eligible=sum(item.category is ReviewCategory.ELIGIBLE for item in self.classifications),
        )

    @property
    def accepted_line_precision(self) -> float | None:
        """Return correct reviewed matches divided by all reviewed matches."""

        reviewed_matches = tuple(
            item for item in self.reviewed_lines if item.observed_operation == "match"
        )
        if not reviewed_matches:
            return None
        return sum(item.reviewer_decision == "correct" for item in reviewed_matches) / len(
            reviewed_matches
        )


@dataclass(frozen=True, slots=True)
class ReviewSummary:
    """Review metrics that preserve page-level null residual evidence."""

    counts: ReviewCounts
    accepted_line_precision: float | None
    eligible_page_coverage: float | None
    declared_complex_accepted: int
    normalized_costs: tuple[float, ...]
    width_residuals: tuple[float | None, ...]
    indentation_residuals: tuple[float | None, ...]
    uniqueness_margins: tuple[float, ...]
    confidence: None = None
    confidence_kind: Literal["uncalibrated"] = "uncalibrated"

    def __post_init__(self) -> None:
        _validate_optional_proportion(
            self.accepted_line_precision,
            name="accepted_line_precision",
        )
        _validate_optional_proportion(
            self.eligible_page_coverage,
            name="eligible_page_coverage",
        )
        _validate_nonnegative_integer(
            self.declared_complex_accepted,
            name="declared_complex_accepted",
        )
        if self.declared_complex_accepted > self.counts.declared_complex:
            raise ValueError("declared_complex_accepted cannot exceed declared_complex pages.")

        object.__setattr__(self, "normalized_costs", tuple(self.normalized_costs))
        object.__setattr__(self, "width_residuals", tuple(self.width_residuals))
        object.__setattr__(self, "indentation_residuals", tuple(self.indentation_residuals))
        object.__setattr__(self, "uniqueness_margins", tuple(self.uniqueness_margins))
        distributions = (
            self.normalized_costs,
            self.width_residuals,
            self.indentation_residuals,
            self.uniqueness_margins,
        )
        if any(len(values) != self.counts.selected_total for values in distributions):
            raise ValueError("Review metric distributions must each match selected_total.")
        _validate_nonnegative_distribution(self.normalized_costs, name="normalized_costs")
        _validate_optional_nonnegative_distribution(self.width_residuals, name="width_residuals")
        _validate_optional_nonnegative_distribution(
            self.indentation_residuals,
            name="indentation_residuals",
        )
        _validate_nonnegative_distribution(self.uniqueness_margins, name="uniqueness_margins")
        if self.confidence is not None or self.confidence_kind != "uncalibrated":
            raise ValueError("Review confidence must be null and uncalibrated.")


def classify_page(page: PageAlignment) -> PageClassification:
    """Map one page to its one review category using fixed priority."""

    unknown_codes = set(page.exclusions) - _KNOWN_EXCLUSION_CODES
    if unknown_codes:
        codes = ", ".join(sorted(unknown_codes))
        raise ValueError(f"Unknown M15b exclusion code: {codes}.")
    exclusions = tuple(page.exclusions)
    if _SOURCE_CHANGED_CODE in exclusions:
        category = ReviewCategory.SOURCE_CHANGED
    elif set(exclusions) & _UNAVAILABLE_OR_MALFORMED_CODES:
        category = ReviewCategory.UNAVAILABLE_OR_MALFORMED
    elif set(exclusions) & _DECLARED_COMPLEX_CODES:
        category = ReviewCategory.DECLARED_COMPLEX
    else:
        category = ReviewCategory.ELIGIBLE
    return PageClassification(category=category, exclusions=exclusions)


def summarize_review(
    report: AlignmentReport,
    reviewed_lines: Sequence[ReviewedLine] = (),
    *,
    selected_pages: Sequence[ReviewPage] | None = None,
) -> ReviewSummary:
    """Summarize a selected review set without collapsing null residuals."""

    ledger = ReviewLedger.from_report(
        report,
        reviewed_lines,
        selected_pages=selected_pages,
    )
    reviewed_pages = ledger.pages
    counts = ledger.counts
    eligible_pages = tuple(
        page
        for page, classification in zip(reviewed_pages, ledger.classifications, strict=True)
        if classification.category is ReviewCategory.ELIGIBLE
    )
    eligible_page_coverage = (
        None
        if not eligible_pages
        else sum(page.accepted for page in eligible_pages) / len(eligible_pages)
    )
    declared_complex_accepted = sum(
        page.accepted
        for page, classification in zip(reviewed_pages, ledger.classifications, strict=True)
        if classification.category is ReviewCategory.DECLARED_COMPLEX
    )
    return ReviewSummary(
        counts=counts,
        accepted_line_precision=ledger.accepted_line_precision,
        eligible_page_coverage=eligible_page_coverage,
        declared_complex_accepted=declared_complex_accepted,
        normalized_costs=tuple(page.normalized_cost for page in reviewed_pages),
        width_residuals=tuple(page.mean_width_residual for page in reviewed_pages),
        indentation_residuals=tuple(page.mean_indentation_residual for page in reviewed_pages),
        uniqueness_margins=tuple(page.uniqueness_margin for page in reviewed_pages),
    )


@dataclass(frozen=True, slots=True)
class ReviewGateResult:
    """Deterministic bounded-review acceptance result."""

    passed: bool
    failures: tuple[ReviewGateFailure, ...]


def validate_review_gate(summary: ReviewSummary) -> ReviewGateResult:
    """Evaluate the fixed M15b review thresholds without changing evidence."""

    failures: list[ReviewGateFailure] = []
    if summary.accepted_line_precision is None or summary.accepted_line_precision < 0.98:
        failures.append(ReviewGateFailure.ACCEPTED_LINE_PRECISION)
    if summary.declared_complex_accepted != 0:
        failures.append(ReviewGateFailure.DECLARED_COMPLEX_ACCEPTED)
    return ReviewGateResult(passed=not failures, failures=tuple(failures))


@dataclass(frozen=True, slots=True)
class BookYield:
    """One book's accepted-page yield and its admission to synthesis."""

    project_id: str
    eligible_pages: int
    accepted_pages: int
    eligible_page_coverage: float | None
    admitted: bool

    def __post_init__(self) -> None:
        _validate_identity(self.project_id, name="project_id")
        _validate_nonnegative_integer(self.eligible_pages, name="eligible_pages")
        _validate_nonnegative_integer(self.accepted_pages, name="accepted_pages")
        if self.accepted_pages > self.eligible_pages:
            raise ValueError("accepted_pages cannot exceed eligible_pages.")
        _validate_optional_proportion(
            self.eligible_page_coverage,
            name="eligible_page_coverage",
        )
        expected_coverage = (
            None if not self.eligible_pages else self.accepted_pages / self.eligible_pages
        )
        if self.eligible_page_coverage != expected_coverage:
            raise ValueError("eligible_page_coverage must match accepted over eligible pages.")
        if self.admitted != (self.accepted_pages >= MINIMUM_ACCEPTED_PAGES_PER_BOOK):
            raise ValueError("admitted must match the accepted-page minimum.")


def summarize_book_yield(
    report: AlignmentReport,
    *,
    selected_pages: Sequence[ReviewPage] | None = None,
) -> tuple[BookYield, ...]:
    """Measure each book's accepted-page yield and whether it is admitted to synthesis."""

    ledger = ReviewLedger.from_report(report, selected_pages=selected_pages)
    eligible: Counter[str] = Counter()
    accepted: Counter[str] = Counter()
    # Report project order is already natural-sorted and validated, so preserving first-seen
    # order keeps replay byte-identical without imposing a second, different sort key.
    project_ids: dict[str, None] = {}
    if selected_pages is None:
        # An explicit selection defines its own book set; a whole report does not, so a project
        # holding no pages still has to report a zero yield rather than vanish.
        for project in report.projects:
            project_ids.setdefault(project.project_id)
    for page, reference, classification in zip(
        ledger.pages,
        ledger.references,
        ledger.classifications,
        strict=True,
    ):
        project_ids.setdefault(reference.project_id)
        if classification.category is not ReviewCategory.ELIGIBLE:
            continue
        eligible[reference.project_id] += 1
        if page.accepted:
            accepted[reference.project_id] += 1
    return tuple(
        BookYield(
            project_id=project_id,
            eligible_pages=eligible[project_id],
            accepted_pages=accepted[project_id],
            eligible_page_coverage=(
                None if not eligible[project_id] else accepted[project_id] / eligible[project_id]
            ),
            admitted=accepted[project_id] >= MINIMUM_ACCEPTED_PAGES_PER_BOOK,
        )
        for project_id in project_ids
    )


def render_alignment_overlay(
    scan_path: str | Path,
    page_alignment: PageAlignment,
    output_path: str | Path,
    *,
    corpus_root: str | Path | None = None,
) -> Image.Image:
    """Render one immutable page alignment as a source-sized RGB PNG overlay."""

    scan = Path(scan_path).resolve(strict=True)
    if not scan.is_file():
        raise ValueError(f"Overlay scan path must name a regular file: {scan!s}.")
    output = _validated_output_path(output_path, corpus_root=corpus_root)
    if output == scan or (output.exists() and output.samefile(scan)):
        raise ValueError("Overlay output must not resolve to the scan path.")
    with BytesIO(scan.read_bytes()) as scan_snapshot:
        scan_digest = file_digest(scan_snapshot, "sha256").hexdigest()
        if page_alignment.scan_sha256 is None or scan_digest != page_alignment.scan_sha256:
            raise ValueError("Overlay scan bytes must match page_alignment.scan_sha256.")
        _ = scan_snapshot.seek(0)
        with Image.open(scan_snapshot) as source:
            overlay = source.convert("RGB")
    if page_alignment.source_frame is not None and overlay.size != (
        page_alignment.source_frame.width,
        page_alignment.source_frame.height,
    ):
        overlay.close()
        raise ValueError("Overlay scan dimensions must match the alignment source frame.")

    matched_sources = {
        operation.candidate_ordinal: operation.source_ordinal
        for operation in page_alignment.operations
        if operation.kind == "match"
    }
    draw = ImageDraw.Draw(overlay)
    for candidate in page_alignment.candidates:
        source_ordinal = matched_sources.get(candidate.ordinal)
        color = _candidate_color(source_ordinal, accepted=page_alignment.accepted)
        x_start, y_start, x_end, y_end = candidate.box
        draw.rectangle((x_start, y_start, x_end - 1, y_end - 1), outline=color, width=1)
        if source_ordinal is not None:
            draw.text((0, y_start), str(source_ordinal), fill=color)
    _write_overlay_atomic(overlay, output)
    return overlay


def render_line_typography_overlay(
    scan_path: str | Path,
    page_alignment: PageAlignment,
    line: LineTypography,
    output_path: str | Path,
    *,
    corpus_root: str | Path | None = None,
    margin_px: int = _TYPOGRAPHY_OVERLAY_MARGIN_PX,
) -> Image.Image:
    """Render one measured line as a cropped RGB PNG with its baseline and x-height drawn.

    This is the instrument Gate 4 is judged on: a reviewer looks at the crop and says whether
    the blue baseline sits on the visible baseline and the magenta x-height rule sits on the
    lowercase tops. It exists for review only and no production report path calls it.

    The crop is the matched line's box grown by `margin_px` on every side and clipped to the
    scan, so a reviewer sees enough of the surrounding white to judge the rules against, and
    the two rules are drawn across the full crop width rather than the box width so neither
    can be mistaken for part of the box outline.
    """

    if line.baseline_row_px is None or line.x_height_top_row_px is None:
        raise ValueError("A typography overlay requires a measured line.")
    if margin_px < 0:
        raise ValueError("margin_px must be nonnegative.")
    scan = Path(scan_path).resolve(strict=True)
    if not scan.is_file():
        raise ValueError(f"Overlay scan path must name a regular file: {scan!s}.")
    output = _validated_output_path(output_path, corpus_root=corpus_root)
    if output == scan or (output.exists() and output.samefile(scan)):
        raise ValueError("Overlay output must not resolve to the scan path.")
    with BytesIO(scan.read_bytes()) as scan_snapshot:
        scan_digest = file_digest(scan_snapshot, "sha256").hexdigest()
        if page_alignment.scan_sha256 is None or scan_digest != page_alignment.scan_sha256:
            raise ValueError("Overlay scan bytes must match page_alignment.scan_sha256.")
        _ = scan_snapshot.seek(0)
        with Image.open(scan_snapshot) as source:
            page = source.convert("RGB")
    if page_alignment.source_frame is not None and page.size != (
        page_alignment.source_frame.width,
        page_alignment.source_frame.height,
    ):
        page.close()
        raise ValueError("Overlay scan dimensions must match the alignment source frame.")

    x_start, y_start, x_end, y_end = line.box
    crop_left = max(0, x_start - margin_px)
    crop_top = max(0, y_start - margin_px)
    crop_right = min(page.width, x_end + margin_px)
    crop_bottom = min(page.height, y_end + margin_px)
    overlay = page.crop((crop_left, crop_top, crop_right, crop_bottom))
    page.close()

    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        (
            x_start - crop_left,
            y_start - crop_top,
            x_end - 1 - crop_left,
            y_end - 1 - crop_top,
        ),
        outline=_LINE_BOX_COLOR,
        width=1,
    )
    for row, color in (
        (line.baseline_row_px, _BASELINE_COLOR),
        (line.x_height_top_row_px, _X_HEIGHT_COLOR),
    ):
        local_row = row - crop_top
        if 0 <= local_row < overlay.height:
            draw.line((0, local_row, overlay.width - 1, local_row), fill=color, width=1)
    _write_overlay_atomic(overlay, output)
    return overlay


def _validate_ordinal(value: object, *, name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer or null.")


def _validate_identity(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string.")


def _validate_finite(value: object, *, name: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_nonnegative_integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


def _validate_optional_proportion(value: float | None, *, name: str) -> None:
    if value is None:
        return
    numeric = _validate_finite(value, name=name)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1] or null.")


def _validate_nonnegative_distribution(values: tuple[float, ...], *, name: str) -> None:
    for value in values:
        if _validate_finite(value, name=name) < 0.0:
            raise ValueError(f"{name} must contain only nonnegative values.")


def _validate_optional_nonnegative_distribution(
    values: tuple[float | None, ...], *, name: str
) -> None:
    for value in values:
        if value is not None and _validate_finite(value, name=name) < 0.0:
            raise ValueError(f"{name} must contain only nonnegative values or null.")


def _report_pages(report: AlignmentReport) -> dict[ReviewPage, PageAlignment]:
    pages: dict[ReviewPage, PageAlignment] = {}
    for project in report.projects:
        for page in project.pages:
            reference = ReviewPage(project.project_id, page.page_name)
            if reference in pages:
                raise ValueError("Alignment report contains duplicate project/page identities.")
            pages[reference] = page
    return pages


def _selected_references(
    available: dict[ReviewPage, PageAlignment], selected_pages: Sequence[ReviewPage] | None
) -> tuple[ReviewPage, ...]:
    if selected_pages is None:
        return tuple(available)
    references = tuple(selected_pages)
    if len(set(references)) != len(references):
        raise ValueError("Selected review pages must not repeat a project/page identity.")
    missing = tuple(reference for reference in references if reference not in available)
    if missing:
        raise ValueError("Selected review pages must exist in the alignment report.")
    return references


def _validate_reviewed_lines(
    reviewed_lines: tuple[ReviewedLine, ...],
    available: dict[ReviewPage, PageAlignment],
    selected: frozenset[ReviewPage],
) -> None:
    observed_operations: set[tuple[ReviewPage, str, int | None, int | None]] = set()
    for reviewed_line in reviewed_lines:
        reference = ReviewPage(reviewed_line.project_id, reviewed_line.page_name)
        if reference not in selected:
            raise ValueError("Reviewed lines must identify a selected report page.")
        page = available[reference]
        identity = (
            reference,
            reviewed_line.observed_operation,
            reviewed_line.observed_source_ordinal,
            reviewed_line.observed_candidate_ordinal,
        )
        if identity in observed_operations:
            raise ValueError("Reviewed lines must not repeat an observed report operation.")
        observed_operations.add(identity)
        actual_operations = {
            (operation.kind, operation.source_ordinal, operation.candidate_ordinal)
            for operation in page.operations
        }
        if identity[1:] not in actual_operations:
            raise ValueError("Reviewed line must identify an actual report operation.")
        if reviewed_line.observed_operation == "match" and not page.accepted:
            raise ValueError("Reviewed match operations must come from an accepted page.")
        if reviewed_line.reviewer_decision == "correct" and (
            reviewed_line.intended_source_ordinal != reviewed_line.observed_source_ordinal
            or reviewed_line.intended_candidate_ordinal != reviewed_line.observed_candidate_ordinal
        ):
            raise ValueError("Correct review rows must match their intended source-line mapping.")


def _candidate_color(source_ordinal: int | None, *, accepted: bool) -> tuple[int, int, int]:
    if source_ordinal is None:
        return _SKIPPED_CANDIDATE_COLOR
    return _ACCEPTED_MATCH_COLOR if accepted else _REJECTED_MATCH_COLOR


def _validated_output_path(output_path: str | Path, *, corpus_root: str | Path | None) -> Path:
    output = Path(output_path).resolve()
    if corpus_root is not None:
        root = Path(corpus_root).resolve()
        if output == root or output.is_relative_to(root):
            raise ValueError("Overlay output must be outside the corpus root.")
    if not output.parent.is_dir():
        raise ValueError(f"Overlay output directory does not exist: {output.parent!s}.")
    return output


def _write_overlay_atomic(overlay: Image.Image, output: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".png",
    )
    temporary_path = Path(temporary_name)
    try:
        os.close(descriptor)
        overlay.save(temporary_path, format="PNG")
        temporary_descriptor = os.open(temporary_path, os.O_RDONLY)
        try:
            _ = os.fsync(temporary_descriptor)
        finally:
            os.close(temporary_descriptor)
        _ = temporary_path.replace(output)
    except Exception as write_error:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            raise ExceptionGroup(
                "Writing the overlay and removing its temporary file both failed.",
                [write_error, cleanup_error],
            ) from None
        raise
    directory_descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        _ = os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
