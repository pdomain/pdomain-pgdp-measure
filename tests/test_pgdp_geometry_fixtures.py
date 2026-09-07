"""Reviewed source-frame geometry fixtures for PGDP measurements."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdomain_pgdp_measure.image_measurement import measure_image
from pdomain_pgdp_measure.profile_input import ProfileInputPage
from pdomain_pgdp_measure.profiling import pool_estimate, profile_page

_FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "pgdp_geometry"
_EXPECTED_PATH = _FIXTURE_DIRECTORY / "expected.json"


def test_reviewed_geometry_fixtures_meet_bounds_and_band_gates() -> None:
    expected = json.loads(_EXPECTED_PATH.read_text(encoding="utf-8"))
    bounds_errors: list[int] = []
    band_ious: list[float] = []
    for fixture in expected["fixtures"]:
        result = measure_image(_FIXTURE_DIRECTORY / fixture["file"])
        assert result.foreground_bounds == _as_bounds(fixture["foreground_bounds"])
        assert tuple((band.y_start, band.y_end) for band in result.ink_bands) == tuple(
            tuple(band) for band in fixture["ink_bands"]
        )
        assert result.diagnostics == tuple(fixture["diagnostics"])
        assert result.exif_orientation == fixture["exif_orientation"]
        if fixture["require_zero_bounds_error"]:
            bounds_errors.extend(
                _bounds_errors(result.foreground_bounds, fixture["foreground_bounds"])
            )
        band_ious.extend(
            _band_iou(actual, expected_band)
            for actual, expected_band in zip(
                tuple((band.y_start, band.y_end) for band in result.ink_bands),
                fixture["ink_bands"],
                strict=True,
            )
        )

    assert bounds_errors
    assert max(bounds_errors) == 0
    assert band_ious
    assert sum(band_ious) / len(band_ious) >= 0.95


@pytest.mark.parametrize(
    ("fixture_name", "metric", "expected_exclusion"),
    [
        ("border", "margin_left_px", "border_dominated"),
        ("blank", "median_band_height_px", "blank_page"),
    ],
)
def test_reviewed_geometry_fixture_exclusions(
    fixture_name: str, metric: str, expected_exclusion: str
) -> None:
    expected = json.loads(_EXPECTED_PATH.read_text(encoding="utf-8"))
    fixture = next(item for item in expected["fixtures"] if item["name"] == fixture_name)
    page = profile_page(
        "fixture-project",
        ProfileInputPage(
            name=fixture["file"],
            image_path=_FIXTURE_DIRECTORY / fixture["file"],
            source_path=f"fixture-project/{fixture['file']}",
        ),
    )

    estimate = pool_estimate(metric, (page,))

    assert estimate.exclusions == (f"{fixture['file']}:{expected_exclusion}",)


def _as_bounds(value: list[int] | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    return tuple(value)


def _bounds_errors(
    actual: tuple[int, int, int, int] | None, expected: list[int] | None
) -> tuple[int, ...]:
    assert actual is not None
    assert expected is not None
    return tuple(
        abs(observed - reviewed) for observed, reviewed in zip(actual, expected, strict=True)
    )


def _band_iou(actual: tuple[int, int], expected: list[int]) -> float:
    overlap = max(0, min(actual[1], expected[1]) - max(actual[0], expected[0]))
    union = max(actual[1], expected[1]) - min(actual[0], expected[0])
    return overlap / union
