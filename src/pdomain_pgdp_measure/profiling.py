"""Build page observations and pooled source-frame geometry profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median
from typing import TYPE_CHECKING, Literal

from pdomain_pgdp_measure.image_measurement import (
    SnapshotSpoolError,
    measure_image_snapshot,
    open_image_snapshot,
)
from pdomain_pgdp_measure.ordering import natural_page_key
from pdomain_pgdp_measure.page_templates import (
    PAGE_TEMPLATE_METHOD,
    PageClassification,
    classify_pages,
    fit_book_templates,
)
from pdomain_pgdp_measure.profile_models import (
    Estimate,
    InkBand,
    PageMeasurement,
    PageTemplateRecord,
    ProfileDiagnostic,
    ProjectProfile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pdomain_pgdp_measure.image_measurement import ImageMeasurement
    from pdomain_pgdp_measure.profile_input import (
        ProfileInput,
        ProfileInputPage,
        ProfileInputProject,
    )


MetricName = Literal[
    "margin_left_px",
    "margin_top_px",
    "margin_right_px",
    "margin_bottom_px",
    "median_band_height_px",
    "median_band_top_pitch_px",
    "first_band_top_px",
]

_METRIC_NAMES: tuple[MetricName, ...] = (
    "margin_left_px",
    "margin_top_px",
    "margin_right_px",
    "margin_bottom_px",
    "median_band_height_px",
    "median_band_top_pitch_px",
    "first_band_top_px",
)
_MARGIN_INDEXES: dict[MetricName, int] = {
    "margin_left_px": 0,
    "margin_top_px": 1,
    "margin_right_px": 2,
    "margin_bottom_px": 3,
}
_THRESHOLD_METHOD = "otsu-256/v1"
_PAGE_DERIVATION_METHOD = "row-ink-bands/v1"
_POOLING_METHOD = "median-mad/v1"


@dataclass(frozen=True, slots=True)
class _MetricObservation:
    page_name: str
    value: float


def profile_selection(selection: ProfileInput) -> tuple[ProjectProfile, ...]:
    """Profile the selected projects while preserving their report order."""

    return tuple(profile_project(project) for project in selection.projects)


def profile_methods() -> dict[str, str]:
    """Return the versioned methods used by profile version 1."""

    return {
        "foreground_threshold": _THRESHOLD_METHOD,
        "ink_bands": _PAGE_DERIVATION_METHOD,
        "pooling": _POOLING_METHOD,
        "page_templates": PAGE_TEMPLATE_METHOD,
    }


def profile_project(project: ProfileInputProject) -> ProjectProfile:
    """Measure one selected project one image at a time."""

    measured = tuple(profile_page(project.project_id, page) for page in project.pages)
    # Templates and pooled estimates describe a book, so both are computed over
    # every measured page. Fitting them on the emitted subset would describe the
    # ranking instead, which is what whole-book mode exists to avoid.
    templates = fit_book_templates(measured)
    classified = {item.page_name: item for item in classify_pages(measured, templates)}
    emitted = tuple(
        _with_page_class(measurement, classified.get(measurement.page_name))
        for measurement, page in zip(measured, project.pages, strict=True)
        if page.emit
    )
    return ProjectProfile(
        project_id=project.project_id,
        title=project.title,
        author=project.author,
        genre=project.genre,
        pages=emitted,
        pooled_estimates=tuple(pool_estimate(name, measured) for name in _METRIC_NAMES),
        page_templates=tuple(
            PageTemplateRecord(
                page_class=template.page_class,
                first_band_top_px=template.first_band_top_px,
                first_band_spread_px=template.first_band_spread_px,
                text_left_px=template.text_left_px,
                text_right_px=template.text_right_px,
                band_count=template.band_count,
                page_count=template.page_count,
                page_share=template.page_share,
            )
            for template in templates.templates
        ),
    )


def _with_page_class(
    measurement: PageMeasurement, classification: PageClassification | None
) -> PageMeasurement:
    if classification is None:
        return measurement
    return replace(
        measurement,
        page_class=classification.page_class,
        template_residual_px=classification.template_residual_px,
        furniture_band_ordinals=classification.furniture_band_ordinals,
        page_class_confidence=classification.confidence,
    )


def profile_page(project_id: str, page: ProfileInputPage) -> PageMeasurement:
    """Measure one selected page without discarding an unavailable record."""

    if page.image_path is None:
        return _unavailable_page(project_id, page, diagnostic_code="image_missing")
    try:
        with open_image_snapshot(page.image_path) as snapshot:
            try:
                measured = measure_image_snapshot(snapshot)
            except (OSError, ValueError):
                return _unavailable_page(
                    project_id,
                    page,
                    diagnostic_code="image_decode_failed",
                    sha256=snapshot.sha256,
                )
    except SnapshotSpoolError:
        raise
    except FileNotFoundError:
        return _unavailable_page(project_id, page, diagnostic_code="image_missing")
    except (OSError, ValueError):
        return _unavailable_page(project_id, page, diagnostic_code="image_unreadable")
    return _measured_page(project_id, page, measured)


def pool_estimate(name: MetricName, pages: Iterable[PageMeasurement]) -> Estimate:
    """Pool one geometry metric with a median and median absolute deviation."""

    if name not in _METRIC_NAMES:
        raise ValueError(f"Unsupported pooled metric: {name!r}.")
    observations: list[_MetricObservation] = []
    exclusions: list[tuple[str, str]] = []
    for page in pages:
        value, exclusion = _metric_value_or_exclusion(page, name)
        if value is None:
            exclusions.append((page.page_name, exclusion))
        else:
            observations.append(_MetricObservation(page_name=page.page_name, value=value))
    observations.sort(key=lambda item: natural_page_key(item.page_name))
    exclusions.sort(key=lambda item: natural_page_key(item[0]))
    values = tuple(item.value for item in observations)
    center = float(median(values)) if values else None
    extensions: dict[str, float] = {}
    if center is not None:
        extensions["median_absolute_deviation"] = float(
            median(tuple(abs(value - center) for value in values))
        )
    return Estimate(
        name=name,
        value=center,
        unit="px",
        truth_class="pooled",
        method=_POOLING_METHOD,
        sample_count=len(observations),
        evidence_pages=tuple(item.page_name for item in observations),
        exclusions=tuple(f"{page_name}:{code}" for page_name, code in exclusions),
        extensions=extensions,
    )


def _measured_page(
    project_id: str, page: ProfileInputPage, measured: ImageMeasurement
) -> PageMeasurement:
    diagnostics = list(_measurement_diagnostics(project_id, page.name, measured))
    if measured.foreground_pixels > 0 and not measured.ink_bands:
        diagnostics.append(_diagnostic("no_ink_bands", project_id, page.name))
    derived_estimates = tuple(
        estimate
        for estimate in (
            _page_estimate(
                page.name,
                "median_band_height_px",
                measured.median_ink_band_height,
            ),
            _page_estimate(
                page.name,
                "median_band_top_pitch_px",
                measured.median_ink_band_top_pitch,
            ),
        )
        if estimate is not None
    )
    return PageMeasurement(
        page_name=page.name,
        source_path=_source_path(project_id, page),
        sha256=measured.sha256,
        source_frame=measured.source_frame,
        image_mode=measured.image_mode,
        grayscale_threshold=measured.grayscale_threshold,
        foreground_pixels=measured.foreground_pixels,
        foreground_bounds=measured.foreground_bounds,
        margins=measured.margins,
        ink_bands=tuple(InkBand(band.y_start, band.y_end) for band in measured.ink_bands),
        derived_estimates=derived_estimates,
        diagnostics=tuple(diagnostics),
        image_extensions=_image_extensions(measured),
    )


def _measurement_diagnostics(
    project_id: str, page_name: str, measured: ImageMeasurement
) -> tuple[ProfileDiagnostic, ...]:
    code_mapping = {"blank": "blank_page", "one_band": "one_ink_band"}
    return tuple(
        _diagnostic(code_mapping.get(code, code), project_id, page_name)
        for code in measured.diagnostics
    )


def _page_estimate(page_name: str, name: MetricName, value: float | None) -> Estimate | None:
    if value is None:
        return None
    return Estimate(
        name=name,
        value=value,
        unit="px",
        truth_class="derived",
        method=_PAGE_DERIVATION_METHOD,
        sample_count=1,
        evidence_pages=(page_name,),
    )


def _image_extensions(measured: ImageMeasurement) -> dict[str, int]:
    if measured.exif_orientation is None:
        return {}
    return {"exif_orientation": measured.exif_orientation}


def _unavailable_page(
    project_id: str,
    page: ProfileInputPage,
    *,
    diagnostic_code: str,
    sha256: str | None = None,
) -> PageMeasurement:
    return PageMeasurement(
        page_name=page.name,
        source_path=_source_path(project_id, page),
        sha256=sha256,
        source_frame=None,
        image_mode=None,
        grayscale_threshold=None,
        foreground_pixels=None,
        foreground_bounds=None,
        margins=None,
        ink_bands=None,
        diagnostics=(_diagnostic(diagnostic_code, project_id, page.name),),
    )


def _source_path(project_id: str, page: ProfileInputPage) -> str:
    if page.source_path is not None:
        return page.source_path
    return f"{project_id}/{page.name}"


def _diagnostic(code: str, project_id: str, page_name: str) -> ProfileDiagnostic:
    return ProfileDiagnostic(
        code=code,
        message=_diagnostic_message(code),
        project_id=project_id,
        page_name=page_name,
    )


def _diagnostic_message(code: str) -> str:
    messages = {
        "blank_page": "The scan has no foreground pixels.",
        "one_ink_band": "The scan has one retained ink band.",
        "border_dominated": "Foreground occupies opposing page-edge strips.",
        "high_foreground_ratio": "Foreground occupies more than sixty percent of the scan.",
        "no_ink_bands": "The scan has no retained ink bands.",
        "image_missing": "The selected scan is unavailable.",
        "image_unreadable": "The selected scan could not be read.",
        "image_decode_failed": "The selected scan could not be decoded.",
        "foreground_bounds_unavailable": "Foreground bounds are unavailable.",
    }
    return messages.get(code, "The page observation is unavailable.")


def _metric_value_or_exclusion(page: PageMeasurement, name: MetricName) -> tuple[float | None, str]:
    diagnostic_code = _pooling_diagnostic(page, name)
    if diagnostic_code is not None:
        return None, diagnostic_code
    margin_index = _MARGIN_INDEXES.get(name)
    if margin_index is not None:
        if page.margins is None:
            return None, "foreground_bounds_unavailable"
        margin = page.margins[margin_index]
        if margin is None:
            return None, "foreground_bounds_unavailable"
        return float(margin), ""
    if name == "first_band_top_px":
        # Where a book's text block starts. Its deviation across a book says
        # whether the book holds a stable type page at all.
        if not page.ink_bands:
            return None, "no_ink_bands"
        return float(page.ink_bands[0].y_start), ""
    for estimate in page.derived_estimates:
        if estimate.name == name and estimate.value is not None:
            return estimate.value, ""
    if name == "median_band_top_pitch_px" and _has_diagnostic(page, "one_ink_band"):
        return None, "one_ink_band"
    if _has_diagnostic(page, "no_ink_bands"):
        return None, "no_ink_bands"
    return None, "metric_unavailable"


def _pooling_diagnostic(page: PageMeasurement, name: MetricName) -> str | None:
    if page.foreground_pixels is None:
        return _first_diagnostic_code(page, default="measurement_unavailable")
    if page.foreground_pixels == 0:
        return "blank_page"
    if _has_diagnostic(page, "border_dominated"):
        return "border_dominated"
    if _has_diagnostic(page, "high_foreground_ratio"):
        return "high_foreground_ratio"
    if name in _MARGIN_INDEXES and page.foreground_bounds is None:
        return "foreground_bounds_unavailable"
    return None


def _first_diagnostic_code(page: PageMeasurement, *, default: str) -> str:
    if page.diagnostics:
        return page.diagnostics[0].code
    return default


def _has_diagnostic(page: PageMeasurement, code: str) -> bool:
    return any(diagnostic.code == code for diagnostic in page.diagnostics)
