from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from pdomain_pgdp_measure.profile_models import (
    ALGORITHM_VERSION,
    SCHEMA_VERSION,
    CoordinateFrame,
    Estimate,
    InkBand,
    PageMeasurement,
    ProfileDiagnostic,
    ProfileReport,
    ProfileReportWire,
    ProjectProfile,
    profile_schema_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ARTIFACT = REPO_ROOT / "schemas" / "pgdp-profile-v2.schema.json"
RETIRED_SCHEMA_ARTIFACT = REPO_ROOT / "schemas" / "pgdp-profile-v1.schema.json"


def _diagnostic(code: str, page_name: str) -> ProfileDiagnostic:
    return ProfileDiagnostic(code=code, message=code, project_id="projectID1", page_name=page_name)


def _page(
    *, foreground_bounds: tuple[int, int, int, int] | None = (10, 20, 90, 180)
) -> PageMeasurement:
    return PageMeasurement(
        page_name="001.png",
        source_path="projectID1/001.png",
        sha256="a" * 64,
        source_frame=CoordinateFrame(width=100, height=200),
        image_mode="L",
        grayscale_threshold=127,
        foreground_pixels=1000,
        foreground_bounds=foreground_bounds,
        margins=(10, 20, 10, 20),
        ink_bands=(InkBand(y_start=20, y_end=40), InkBand(y_start=55, y_end=78)),
        derived_estimates=(
            Estimate(
                name="median_ink_band_height",
                value=21.5,
                unit="px",
                truth_class="derived",
                method="pgdp-profile/v1/row-projection",
                sample_count=1,
                evidence_pages=("001.png",),
            ),
        ),
    )


def _report() -> ProfileReport:
    page = _page()
    project = ProjectProfile(
        project_id="projectID1",
        title="A book",
        author="An author",
        genre="Poetry",
        pages=(page,),
        pooled_estimates=(
            Estimate(
                name="median_margin_left",
                value=10.0,
                unit="px",
                truth_class="pooled",
                method="pgdp-profile/v1/median",
                sample_count=1,
                evidence_pages=("001.png",),
            ),
        ),
    )
    return ProfileReport(
        source_ranking={"algorithm_version": "pgdp-rank/v1", "sha256": "b" * 64},
        methods={"ink_geometry": "pgdp-profile/v1/row-projection"},
        projects=(project,),
    )


def test_profile_report_serializes_deterministically_with_normalized_tuples() -> None:
    report = _report()

    assert report.schema_version == SCHEMA_VERSION
    assert report.algorithm_version == ALGORITHM_VERSION
    assert report.projects == tuple(report.projects)
    assert report.to_json() == report.to_json()
    assert report.to_dict()["projects"][0]["pages"][0]["observations"]["ink_bands"] == [
        {"y_start": 20, "y_end": 40},
        {"y_start": 55, "y_end": 78},
    ]


def test_absent_bounds_do_not_become_zero() -> None:
    page = PageMeasurement(
        page_name="missing.png",
        source_path="projectID1/missing.png",
        sha256=None,
        source_frame=None,
        image_mode=None,
        grayscale_threshold=None,
        foreground_pixels=None,
        foreground_bounds=None,
        margins=None,
        ink_bands=None,
        diagnostics=(_diagnostic("image_missing", "missing.png"),),
    )

    assert page.to_dict()["observations"]["foreground_bounds"] is None
    assert page.to_dict()["sha256"] is None
    assert page.to_dict()["source_frame"] is None
    assert page.to_dict()["image"] is None


@pytest.mark.parametrize(
    "source_path",
    [
        "",
        "/projectID1/001.png",
        "projectID1/./001.png",
        "projectID1/../001.png",
        "projectID1//001.png",
        "projectID1/001.png/",
        "projectID1\\001.png",
        "projectID1/\x00.png",
    ],
)
def test_page_measurement_rejects_noncanonical_source_paths(source_path: str) -> None:
    with pytest.raises(ValueError, match="Source path"):
        _ = replace(_page(), source_path=source_path)


@pytest.mark.parametrize(
    "source_path",
    [
        "",
        "/projectID1/001.png",
        "projectID1/./001.png",
        "projectID1/../001.png",
        "projectID1//001.png",
        "projectID1/001.png/",
        "projectID1\\001.png",
        "projectID1/\x00.png",
    ],
)
def test_profile_wire_rejects_noncanonical_source_paths(source_path: str) -> None:
    payload = _report().to_dict()
    payload["projects"][0]["pages"][0]["source_path"] = source_path

    with pytest.raises(ValueError, match="source_path"):
        _ = ProfileReportWire.model_validate(payload)


def test_profile_schema_describes_the_source_path_format() -> None:
    schema = json.loads(profile_schema_json())
    source_path = schema["$defs"]["PageMeasurementWire"]["properties"]["source_path"]

    assert source_path["format"] == "pgdp-corpus-relative-posix-path"


def test_unavailable_foreground_measurements_remain_null() -> None:
    page = PageMeasurement(
        page_name="002.png",
        source_path="projectID1/002.png",
        sha256=None,
        source_frame=None,
        image_mode=None,
        grayscale_threshold=None,
        foreground_pixels=None,
        foreground_bounds=None,
        margins=None,
        ink_bands=None,
        diagnostics=(_diagnostic("image_missing", "002.png"),),
    )

    observations = page.to_dict()["observations"]

    assert observations["foreground_pixels"] is None
    assert observations["foreground_bounds"] is None
    assert observations["margins"] is None
    assert observations["ink_bands"] is None


def test_undecodable_but_readable_page_keeps_its_original_byte_sha256() -> None:
    page = PageMeasurement(
        page_name="corrupt.png",
        source_path="projectID1/corrupt.png",
        sha256="a" * 64,
        source_frame=None,
        image_mode=None,
        grayscale_threshold=None,
        foreground_pixels=None,
        foreground_bounds=None,
        margins=None,
        ink_bands=None,
        diagnostics=(_diagnostic("image_decode_failed", "corrupt.png"),),
    )

    payload = page.to_dict()

    assert payload["sha256"] == "a" * 64
    assert payload["source_frame"] is None
    assert payload["image"] is None


def test_unavailable_page_requires_an_unavailable_image_diagnostic() -> None:
    with pytest.raises(ValueError, match="image_missing or image_unreadable"):
        _ = PageMeasurement(
            page_name="unavailable.png",
            source_path="projectID1/unavailable.png",
            sha256=None,
            source_frame=None,
            image_mode=None,
            grayscale_threshold=None,
            foreground_pixels=None,
            foreground_bounds=None,
            margins=None,
            ink_bands=None,
            diagnostics=(_diagnostic("blank_page", "unavailable.png"),),
        )


def test_readable_undecodable_page_requires_a_decode_failure_diagnostic() -> None:
    with pytest.raises(ValueError, match="image_decode_failed"):
        _ = PageMeasurement(
            page_name="corrupt.png",
            source_path="projectID1/corrupt.png",
            sha256="a" * 64,
            source_frame=None,
            image_mode=None,
            grayscale_threshold=None,
            foreground_pixels=None,
            foreground_bounds=None,
            margins=None,
            ink_bands=None,
            diagnostics=(_diagnostic("image_missing", "corrupt.png"),),
        )


@pytest.mark.parametrize(
    ("sha256", "source_frame", "image_mode"),
    [
        (None, CoordinateFrame(width=100, height=200), None),
        (None, None, "L"),
        ("a" * 64, CoordinateFrame(width=100, height=200), None),
        ("a" * 64, None, "L"),
    ],
)
def test_unavailable_foreground_measurements_reject_partial_image_metadata(
    sha256: str | None,
    source_frame: CoordinateFrame | None,
    image_mode: str | None,
) -> None:
    with pytest.raises(ValueError, match="unavailable"):
        _ = PageMeasurement(
            page_name="partial.png",
            source_path="projectID1/partial.png",
            sha256=sha256,
            source_frame=source_frame,
            image_mode=image_mode,
            grayscale_threshold=None,
            foreground_pixels=None,
            foreground_bounds=None,
            margins=None,
            ink_bands=None,
        )


def test_measured_foreground_requires_image_metadata() -> None:
    with pytest.raises(ValueError, match="metadata"):
        _ = PageMeasurement(
            page_name="missing-metadata.png",
            source_path="projectID1/missing-metadata.png",
            sha256=None,
            source_frame=None,
            image_mode=None,
            grayscale_threshold=127,
            foreground_pixels=0,
            foreground_bounds=None,
            margins=None,
            ink_bands=(),
            diagnostics=(_diagnostic("image_missing", "blank.png"),),
        )


def test_unavailable_foreground_measurements_reject_derived_estimates() -> None:
    estimate = Estimate(
        name="median_band_height_px",
        value=10.0,
        unit="px",
        truth_class="derived",
        method="row-ink-bands/v1",
        sample_count=1,
        evidence_pages=("unavailable.png",),
    )

    with pytest.raises(ValueError, match="derived estimates"):
        _ = PageMeasurement(
            page_name="unavailable.png",
            source_path="projectID1/unavailable.png",
            sha256=None,
            source_frame=None,
            image_mode=None,
            grayscale_threshold=None,
            foreground_pixels=None,
            foreground_bounds=None,
            margins=None,
            ink_bands=None,
            derived_estimates=(estimate,),
        )


def test_verified_blank_page_keeps_zero_distinct_from_unavailable_measurements() -> None:
    page = PageMeasurement(
        page_name="003.png",
        source_path="projectID1/003.png",
        sha256="c" * 64,
        source_frame=CoordinateFrame(width=100, height=200),
        image_mode="L",
        grayscale_threshold=127,
        foreground_pixels=0,
        foreground_bounds=None,
        margins=None,
        ink_bands=(),
        diagnostics=(_diagnostic("blank_page", "003.png"),),
    )

    assert page.to_dict()["observations"]["foreground_pixels"] == 0


def test_verified_blank_page_requires_a_blank_diagnostic() -> None:
    with pytest.raises(ValueError, match="blank_page"):
        _ = PageMeasurement(
            page_name="blank.png",
            source_path="projectID1/blank.png",
            sha256="c" * 64,
            source_frame=CoordinateFrame(width=100, height=200),
            image_mode="L",
            grayscale_threshold=127,
            foreground_pixels=0,
            foreground_bounds=None,
            margins=None,
            ink_bands=(),
        )


def test_verified_blank_page_rejects_derived_estimates() -> None:
    with pytest.raises(ValueError, match="derived_estimates"):
        _ = PageMeasurement(
            page_name="blank.png",
            source_path="projectID1/blank.png",
            sha256="c" * 64,
            source_frame=CoordinateFrame(width=100, height=200),
            image_mode="L",
            grayscale_threshold=127,
            foreground_pixels=0,
            foreground_bounds=None,
            margins=None,
            ink_bands=(),
            derived_estimates=(_page().derived_estimates[0],),
            diagnostics=(_diagnostic("blank_page", "blank.png"),),
        )


def test_verified_blank_page_requires_an_empty_ink_band_tuple() -> None:
    with pytest.raises(ValueError, match="empty ink bands"):
        _ = replace(
            _page(),
            foreground_pixels=0,
            foreground_bounds=None,
            margins=None,
            ink_bands=None,
        )


@pytest.mark.parametrize("threshold", [-1, 256])
def test_page_measurement_requires_an_eight_bit_grayscale_threshold(threshold: int) -> None:
    with pytest.raises(ValueError, match="grayscale_threshold"):
        _ = replace(_page(), grayscale_threshold=threshold)


@pytest.mark.parametrize("sha256", ["a" * 63, "g" * 64, "a" * 64 + "\n"])
def test_page_measurement_requires_a_64_character_hex_sha256(sha256: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        _ = replace(_page(), sha256=sha256)


def test_ink_band_requires_at_least_two_rows() -> None:
    with pytest.raises(ValueError, match="two rows"):
        _ = InkBand(y_start=20, y_end=21)


def test_profile_wire_rejects_verified_blank_page_without_an_empty_ink_band_tuple() -> None:
    payload = _report().to_dict()
    observations = payload["projects"][0]["pages"][0]["observations"]
    observations["foreground_pixels"] = 0
    observations["foreground_bounds"] = None
    observations["margins"] = None
    observations["ink_bands"] = None

    with pytest.raises(ValueError, match="empty ink bands"):
        _ = ProfileReportWire.model_validate(payload)


def test_profile_wire_requires_a_blank_page_diagnostic() -> None:
    payload = _report().to_dict()
    page = payload["projects"][0]["pages"][0]
    page["observations"]["foreground_pixels"] = 0
    page["observations"]["foreground_bounds"] = None
    page["observations"]["margins"] = None
    page["observations"]["ink_bands"] = []
    page["derived_estimates"] = []

    with pytest.raises(ValueError, match="blank_page"):
        _ = ProfileReportWire.model_validate(payload)


def test_profile_wire_rejects_blank_page_derived_estimates() -> None:
    payload = _report().to_dict()
    page = payload["projects"][0]["pages"][0]
    page["observations"]["foreground_pixels"] = 0
    page["observations"]["foreground_bounds"] = None
    page["observations"]["margins"] = None
    page["observations"]["ink_bands"] = []
    page["diagnostics"] = [
        _diagnostic("blank_page", "001.png").to_dict(),
    ]

    with pytest.raises(ValueError, match="derived estimates"):
        _ = ProfileReportWire.model_validate(payload)


@pytest.mark.parametrize("threshold", [256])
def test_profile_wire_requires_an_eight_bit_grayscale_threshold(threshold: int) -> None:
    payload = _report().to_dict()
    payload["projects"][0]["pages"][0]["observations"]["grayscale_threshold"] = threshold

    with pytest.raises(ValueError, match="grayscale_threshold"):
        _ = ProfileReportWire.model_validate(payload)


def test_profile_wire_requires_a_grayscale_threshold_field_when_it_is_null() -> None:
    payload = _report().to_dict()
    page = payload["projects"][0]["pages"][0]
    page["sha256"] = None
    page["source_frame"] = None
    page["image"] = None
    page["observations"] = {
        "foreground_pixels": None,
        "foreground_bounds": None,
        "margins": None,
        "ink_bands": None,
    }
    page["derived_estimates"] = []

    with pytest.raises(ValueError, match="grayscale_threshold"):
        _ = ProfileReportWire.model_validate(payload)


@pytest.mark.parametrize("sha256", ["a" * 63, "g" * 64, "a" * 64 + "\n"])
def test_profile_wire_requires_a_64_character_hex_sha256(sha256: str) -> None:
    payload = _report().to_dict()
    payload["projects"][0]["pages"][0]["sha256"] = sha256

    with pytest.raises(ValueError, match="sha256"):
        _ = ProfileReportWire.model_validate(payload)


def test_profile_wire_requires_ink_bands_of_at_least_two_rows() -> None:
    payload = _report().to_dict()
    payload["projects"][0]["pages"][0]["observations"]["ink_bands"][0]["y_end"] = 21

    with pytest.raises(ValueError, match="two rows"):
        _ = ProfileReportWire.model_validate(payload)


def test_page_measurement_normalizes_geometry_tuples_before_input_mutation() -> None:
    bounds = [10, 20, 90, 180]
    margins = [10, 20, 10, 20]
    page = PageMeasurement(
        page_name="004.png",
        source_path="projectID1/004.png",
        sha256="d" * 64,
        source_frame=CoordinateFrame(width=100, height=200),
        image_mode="L",
        grayscale_threshold=127,
        foreground_pixels=1000,
        foreground_bounds=bounds,
        margins=margins,
    )
    bounds[0] = 0
    margins[0] = 0

    assert page.foreground_bounds == (10, 20, 90, 180)
    assert page.margins == (10, 20, 10, 20)


@pytest.mark.parametrize(
    ("foreground_pixels", "foreground_bounds", "margins", "ink_bands"),
    [
        (None, (10, 20, 90, 180), None, None),
        (None, None, (10, 20, 10, 20), None),
        (None, None, None, (InkBand(y_start=20, y_end=40),)),
    ],
)
def test_unavailable_foreground_measurements_reject_partial_geometry(
    foreground_pixels: int | None,
    foreground_bounds: tuple[int, int, int, int] | None,
    margins: tuple[int, int, int, int] | None,
    ink_bands: tuple[InkBand, ...] | None,
) -> None:
    with pytest.raises(ValueError, match="unavailable"):
        _ = PageMeasurement(
            page_name="005.png",
            source_path="projectID1/005.png",
            sha256=None,
            source_frame=None,
            image_mode=None,
            grayscale_threshold=None,
            foreground_pixels=foreground_pixels,
            foreground_bounds=foreground_bounds,
            margins=margins,
            ink_bands=ink_bands,
        )


@pytest.mark.parametrize(
    ("foreground_bounds", "margins", "message"),
    [
        ((10, 20, 10, 180), (10, 20, 90, 20), "nonempty"),
        ((10, 20, 90, 20), (10, 20, 10, 180), "nonempty"),
        ((10, 20, 90, 180), (10, 20, None, 20), "numeric"),
        ((10, 20, 90, 180), (9, 20, 10, 20), "match"),
    ],
)
def test_positive_foreground_requires_nonempty_bounds_and_matching_margins(
    foreground_bounds: tuple[int, int, int, int],
    margins: tuple[int | None, int | None, int | None, int | None],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = PageMeasurement(
            page_name="positive.png",
            source_path="projectID1/positive.png",
            sha256="f" * 64,
            source_frame=CoordinateFrame(width=100, height=200),
            image_mode="L",
            grayscale_threshold=127,
            foreground_pixels=1,
            foreground_bounds=foreground_bounds,
            margins=margins,
            ink_bands=(),
        )


@pytest.mark.parametrize(
    ("foreground_pixels", "ink_bands", "message"),
    [
        (12_801, (), "area"),
        (1, (InkBand(y_start=20, y_end=201),), "source frame"),
    ],
)
def test_positive_foreground_respects_bounds_area_and_source_frame(
    foreground_pixels: int,
    ink_bands: tuple[InkBand, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = PageMeasurement(
            page_name="bounded.png",
            source_path="projectID1/bounded.png",
            sha256="a" * 64,
            source_frame=CoordinateFrame(width=100, height=200),
            image_mode="L",
            grayscale_threshold=127,
            foreground_pixels=foreground_pixels,
            foreground_bounds=(10, 20, 90, 180),
            margins=(10, 20, 10, 20),
            ink_bands=ink_bands,
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: CoordinateFrame(width=-1, height=2), "nonnegative"),
        (lambda: InkBand(y_start=3, y_end=2), "half-open"),
        (
            lambda: Estimate(
                name="x",
                value=math.nan,
                unit="px",
                truth_class="observed",
                method="method",
                sample_count=0,
            ),
            "finite",
        ),
        (
            lambda: Estimate(
                name="x",
                value=1.0,
                unit="px",
                truth_class="observed",
                method="method",
                sample_count=2,
                evidence_pages=("001",),
            ),
            "sample_count",
        ),
        (
            lambda: Estimate(
                name="x",
                value=1.0,
                unit="px",
                truth_class="observed",
                method="method",
                sample_count=2,
                evidence_pages=("001", "001"),
            ),
            "duplicate",
        ),
    ],
)
def test_profile_records_reject_invalid_values(factory: object, message: str) -> None:
    if not callable(factory):
        raise AssertionError("test factory must be callable")

    with pytest.raises(ValueError, match=message):
        _ = factory()


def test_profile_records_reject_boolean_estimate_values() -> None:
    with pytest.raises(TypeError, match="number"):
        _ = Estimate(
            name="x",
            value=True,
            unit="px",
            truth_class="observed",
            method="method",
            sample_count=1,
            evidence_pages=("001",),
        )


def test_profile_report_round_trips_through_typed_wire_json() -> None:
    report = _report()

    reloaded = ProfileReport.from_json(report.to_json())

    assert reloaded.to_dict() == report.to_dict()


def test_profile_reader_preserves_unknown_v1_fields_in_extensions() -> None:
    payload = _report().to_dict()
    payload["future_report_field"] = {"switch": [1, True, None]}
    projects = payload["projects"]
    assert isinstance(projects, list)
    projects[0]["future_project_field"] = "kept"

    reloaded = ProfileReport.from_dict(payload)
    written = reloaded.to_dict()

    assert written["future_report_field"] == {"switch": [1, True, None]}
    assert written["projects"][0]["future_project_field"] == "kept"


def test_profile_reader_preserves_unknown_v1_fields_at_nested_boundaries() -> None:
    payload = _report().to_dict()
    projects = payload["projects"]
    assert isinstance(projects, list)
    pages = projects[0]["pages"]
    assert isinstance(pages, list)
    page = pages[0]
    page["future_page_field"] = {"valid": True}
    page["source_frame"]["future_frame_field"] = 1
    page["image"]["future_image_field"] = "kept"
    observations = page["observations"]
    observations["future_observation_field"] = ["kept"]
    observations["margins"]["future_margin_field"] = None
    bands = observations["ink_bands"]
    assert isinstance(bands, list)
    bands[0]["future_band_field"] = False
    estimates = page["derived_estimates"]
    assert isinstance(estimates, list)
    estimates[0]["future_estimate_field"] = {"kept": 1}

    written = ProfileReport.from_dict(payload).to_dict()
    written_page = written["projects"][0]["pages"][0]

    assert written_page["future_page_field"] == {"valid": True}
    assert written_page["source_frame"]["future_frame_field"] == 1
    assert written_page["image"]["future_image_field"] == "kept"
    assert written_page["observations"]["future_observation_field"] == ["kept"]
    assert written_page["observations"]["margins"]["future_margin_field"] is None
    assert written_page["observations"]["ink_bands"][0]["future_band_field"] is False
    assert written_page["derived_estimates"][0]["future_estimate_field"] == {"kept": 1}


def test_profile_records_reject_nonfinite_extension_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        _ = ProfileReport(
            source_ranking={"ranking_score": math.inf},
            methods={},
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
    ],
)
def test_profile_reader_rejects_coerced_schema_versions(key: str, value: object) -> None:
    payload = _report().to_dict()
    payload[key] = value

    with pytest.raises(ValueError, match="schema_version"):
        _ = ProfileReport.from_dict(payload)


def test_profile_extensions_are_deeply_immutable() -> None:
    nested = {"value": [1, {"two": 2}]}
    report = ProfileReport(source_ranking={"details": nested}, methods={})
    nested["value"][1]["two"] = 3

    assert report.to_dict()["source_ranking"] == {"details": {"value": [1, {"two": 2}]}}


def test_numeric_estimates_require_page_evidence() -> None:
    with pytest.raises(ValueError, match="numeric"):
        _ = Estimate(
            name="margin",
            value=10.0,
            unit="px",
            truth_class="observed",
            method="method",
            sample_count=0,
        )


def test_page_measurement_rejects_non_derived_estimates() -> None:
    estimate = Estimate(
        name="ink",
        value=10.0,
        unit="px",
        truth_class="observed",
        method="method",
        sample_count=1,
        evidence_pages=("006.png",),
    )

    with pytest.raises(ValueError, match="derived_estimates"):
        _ = PageMeasurement(
            page_name="006.png",
            source_path="projectID1/006.png",
            sha256="f" * 64,
            source_frame=CoordinateFrame(width=100, height=200),
            image_mode="L",
            grayscale_threshold=127,
            foreground_pixels=0,
            foreground_bounds=None,
            margins=None,
            ink_bands=(),
            derived_estimates=(estimate,),
        )


def test_project_profile_rejects_non_pooled_estimates() -> None:
    estimate = Estimate(
        name="margin",
        value=10.0,
        unit="px",
        truth_class="derived",
        method="method",
        sample_count=1,
        evidence_pages=("001.png",),
    )

    with pytest.raises(ValueError, match="pooled_estimates"):
        _ = ProjectProfile(
            project_id="projectID1",
            title=None,
            author=None,
            genre=None,
            pooled_estimates=(estimate,),
        )


@pytest.mark.parametrize("value", ["1.0", True])
def test_profile_wire_rejects_non_numeric_estimate_values(value: object) -> None:
    payload = _report().to_dict()
    payload["projects"][0]["pages"][0]["derived_estimates"][0]["value"] = value

    with pytest.raises(ValueError, match="value"):
        _ = ProfileReport.from_dict(payload)


@pytest.mark.parametrize("field", ["confidence", "confidence_kind"])
def test_profile_wire_requires_v1_confidence_fields(field: str) -> None:
    payload = _report().to_dict()
    estimate = payload["projects"][0]["pages"][0]["derived_estimates"][0]
    _ = estimate.pop(field)

    with pytest.raises(ValueError, match=field):
        _ = ProfileReport.from_dict(payload)


@pytest.mark.parametrize("field", ["schema_version", "algorithm_version"])
def test_profile_wire_rejects_unversioned_reports(field: str) -> None:
    payload = _report().to_dict()
    _ = payload.pop(field)

    with pytest.raises(ValueError, match=field):
        _ = ProfileReport.from_dict(payload)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("algorithm_version", "pgdp-profile/v3", "algorithm_version"),
    ],
)
def test_profile_reader_rejects_unknown_major_versions(
    key: str, value: object, message: str
) -> None:
    payload = _report().to_dict()
    payload[key] = value

    with pytest.raises(ValueError, match=message):
        _ = ProfileReport.from_dict(payload)


def test_profile_schema_artifact_matches_fresh_generation() -> None:
    assert SCHEMA_ARTIFACT.read_text(encoding="utf-8") == profile_schema_json()


def test_version_one_schema_is_retained_unchanged() -> None:
    """Version 1 survives as its schema file and git history, not as a readable payload.

    Rejecting an unknown major version is the intended behaviour, so a stored v1
    report does not load. What is retained is the contract it was written to.
    """

    assert RETIRED_SCHEMA_ARTIFACT.is_file()
    assert '"pgdp-profile/v1"' in RETIRED_SCHEMA_ARTIFACT.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="algorithm_version"):
        _ = ProfileReport.from_dict(
            {
                "schema_version": 1,
                "algorithm_version": "pgdp-profile/v1",
                "source_ranking": {"algorithm_version": "pgdp-rank/v1"},
                "methods": {},
                "projects": [],
                "diagnostics": [],
            }
        )


def test_profile_schema_declares_the_top_level_contract() -> None:
    payload = json.loads(SCHEMA_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["properties"].keys() >= {
        "schema_version",
        "algorithm_version",
        "source_ranking",
        "methods",
        "projects",
        "diagnostics",
    }


def test_profile_schema_constrains_sha256_to_exactly_64_characters() -> None:
    payload = json.loads(SCHEMA_ARTIFACT.read_text(encoding="utf-8"))
    sha256 = payload["$defs"]["PageMeasurementWire"]["properties"]["sha256"]
    string_schema = next(item for item in sha256["anyOf"] if item.get("type") == "string")

    assert string_schema["minLength"] == 64
    assert string_schema["maxLength"] == 64
    assert {item["type"] for item in sha256["anyOf"]} == {"null", "string"}
    assert "sha256" in payload["$defs"]["PageMeasurementWire"]["required"]


def test_unavailable_page_round_trips_through_the_profile_wire_contract() -> None:
    unavailable_page = PageMeasurement(
        page_name="unavailable.png",
        source_path="projectID1/unavailable.png",
        sha256=None,
        source_frame=None,
        image_mode=None,
        grayscale_threshold=None,
        foreground_pixels=None,
        foreground_bounds=None,
        margins=None,
        ink_bands=None,
        diagnostics=(_diagnostic("image_missing", "unavailable.png"),),
    )
    report = ProfileReport(
        source_ranking={"algorithm_version": "pgdp-rank/v1"},
        methods={},
        projects=(
            ProjectProfile(
                project_id="projectID1",
                title=None,
                author=None,
                genre=None,
                pages=(unavailable_page,),
            ),
        ),
    )

    parsed = ProfileReport.from_dict(report.to_dict())

    assert parsed.projects[0].pages[0].sha256 is None
    assert parsed.projects[0].pages[0].source_frame is None
    assert parsed.projects[0].pages[0].image_mode is None


def test_profile_wire_rejects_unavailable_page_with_measured_observations() -> None:
    payload = _report().to_dict()
    page = payload["projects"][0]["pages"][0]
    page["sha256"] = None
    page["source_frame"] = None
    page["image"] = None
    page["observations"]["foreground_pixels"] = None
    page["observations"]["foreground_bounds"] = None
    page["observations"]["margins"] = None
    page["observations"]["ink_bands"] = None

    with pytest.raises(ValueError, match="unavailable"):
        _ = ProfileReportWire.model_validate(payload)


def test_profile_wire_rejects_unavailable_page_with_derived_estimates() -> None:
    payload = _report().to_dict()
    page = payload["projects"][0]["pages"][0]
    page["sha256"] = None
    page["source_frame"] = None
    page["image"] = None
    page["observations"] = {
        "grayscale_threshold": None,
        "foreground_pixels": None,
        "foreground_bounds": None,
        "margins": None,
        "ink_bands": None,
    }

    with pytest.raises(ValueError, match="derived estimates"):
        _ = ProfileReportWire.model_validate(payload)


def test_profile_wire_requires_an_unavailable_image_diagnostic() -> None:
    payload = _report().to_dict()
    page = payload["projects"][0]["pages"][0]
    page["sha256"] = None
    page["source_frame"] = None
    page["image"] = None
    page["observations"] = {
        "grayscale_threshold": None,
        "foreground_pixels": None,
        "foreground_bounds": None,
        "margins": None,
        "ink_bands": None,
    }
    page["derived_estimates"] = []

    with pytest.raises(ValueError, match="image_missing or image_unreadable"):
        _ = ProfileReportWire.model_validate(payload)


def test_profile_wire_accepts_undecodable_page_with_an_original_byte_sha256() -> None:
    payload = _report().to_dict()
    page = payload["projects"][0]["pages"][0]
    page["source_frame"] = None
    page["image"] = None
    page["observations"] = {
        "grayscale_threshold": None,
        "foreground_pixels": None,
        "foreground_bounds": None,
        "margins": None,
        "ink_bands": None,
    }
    page["derived_estimates"] = []
    page["diagnostics"] = [
        _diagnostic("image_decode_failed", "001.png").to_dict(),
    ]

    parsed = ProfileReportWire.model_validate(payload).to_domain()

    assert parsed.projects[0].pages[0].sha256 == "a" * 64


def test_profile_schema_declares_mutually_exclusive_page_availability_states() -> None:
    payload = json.loads(SCHEMA_ARTIFACT.read_text(encoding="utf-8"))
    page_schema = payload["$defs"]["PageMeasurementWire"]

    assert len(page_schema["oneOf"]) == 4
    assert page_schema["oneOf"][0]["properties"]["derived_estimates"]["maxItems"] == 0
    assert page_schema["oneOf"][0]["properties"]["diagnostics"]["contains"] == {
        "properties": {
            "code": {"enum": ["image_missing", "image_unreadable"]},
        },
        "required": ["code"],
    }
    assert page_schema["oneOf"][1]["properties"]["sha256"]["type"] == "string"
    assert page_schema["oneOf"][1]["properties"]["source_frame"]["type"] == "null"
    assert page_schema["oneOf"][1]["properties"]["diagnostics"]["contains"] == {
        "properties": {"code": {"const": "image_decode_failed"}},
        "required": ["code"],
    }
    assert page_schema["oneOf"][2]["properties"]["derived_estimates"]["maxItems"] == 0
    assert page_schema["oneOf"][2]["properties"]["diagnostics"]["contains"] == {
        "properties": {"code": {"const": "blank_page"}},
        "required": ["code"],
    }
