"""Tests for PGDP page profiling and robust geometry pooling."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image, ImageDraw

from pdomain_pgdp_measure import image_measurement, profiling
from pdomain_pgdp_measure.profile_input import (
    ProfileInput,
    ProfileInputPage,
    ProfileInputProject,
)
from pdomain_pgdp_measure.profile_models import (
    CoordinateFrame,
    Estimate,
    InkBand,
    PageMeasurement,
    ProfileDiagnostic,
)
from pdomain_pgdp_measure.profiling import pool_estimate, profile_methods, profile_selection

if TYPE_CHECKING:
    from pdomain_pgdp_measure.image_measurement import ImageSnapshot


def test_profile_selection_keeps_page_order_and_unavailable_page_records(tmp_path: Path) -> None:
    measured_path = tmp_path / "p10.png"
    measured = Image.new("L", (40, 30), color=255)
    draw = ImageDraw.Draw(measured)
    draw.rectangle((5, 4, 20, 6), fill=0)
    draw.rectangle((6, 15, 22, 18), fill=0)
    measured.save(measured_path)

    blank_path = tmp_path / "p2.png"
    Image.new("L", (40, 30), color=255).save(blank_path)

    corrupt_path = tmp_path / "p3.png"
    corrupt_path.write_bytes(b"not an image")
    selection = ProfileInput(
        projects=(
            ProfileInputProject(
                project_id="projectID1",
                title="A book",
                author=None,
                genre=None,
                pages=(
                    ProfileInputPage(
                        name="p10.png",
                        image_path=measured_path,
                        source_path="projectID1/p10.png",
                    ),
                    ProfileInputPage(
                        name="p2.png",
                        image_path=blank_path,
                        source_path="projectID1/p2.png",
                    ),
                    ProfileInputPage(
                        name="p3.png",
                        image_path=corrupt_path,
                        source_path="projectID1/p3.png",
                    ),
                    ProfileInputPage(
                        name="p4.png",
                        image_path=tmp_path / "missing.png",
                        source_path="projectID1/p4.png",
                    ),
                ),
            ),
        )
    )

    result = profile_selection(selection)

    project = result[0]
    assert tuple(page.page_name for page in project.pages) == (
        "p10.png",
        "p2.png",
        "p3.png",
        "p4.png",
    )
    assert tuple(estimate.name for estimate in project.pages[0].derived_estimates) == (
        "median_band_height_px",
        "median_band_top_pitch_px",
    )
    assert project.pages[1].foreground_pixels == 0
    assert tuple(diagnostic.code for diagnostic in project.pages[1].diagnostics) == ("blank_page",)
    assert project.pages[2].foreground_pixels is None
    assert tuple(diagnostic.code for diagnostic in project.pages[2].diagnostics) == (
        "image_decode_failed",
    )
    assert project.pages[2].sha256 == sha256(corrupt_path.read_bytes()).hexdigest()
    assert project.pages[2].source_frame is None
    assert project.pages[2].image_mode is None
    assert project.pages[3].foreground_pixels is None
    assert tuple(diagnostic.code for diagnostic in project.pages[3].diagnostics) == (
        "image_missing",
    )
    assert project.pages[3].sha256 is None
    assert project.pages[3].source_path == "projectID1/p4.png"


def test_profile_selection_marks_unreadable_image_paths(tmp_path: Path) -> None:
    selection = ProfileInput(
        projects=(
            ProfileInputProject(
                project_id="projectID1",
                title=None,
                author=None,
                genre=None,
                pages=(
                    ProfileInputPage(
                        name="image-directory",
                        image_path=tmp_path,
                        source_path="projectID1/image-directory",
                    ),
                ),
            ),
        )
    )

    result = profile_selection(selection)

    page = result[0].pages[0]
    assert page.sha256 is None
    assert tuple(diagnostic.code for diagnostic in page.diagnostics) == ("image_unreadable",)


def test_profile_page_propagates_a_temporary_snapshot_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("L", (8, 6), color=255).save(image_path)

    class WriteFailingSnapshot:
        def __enter__(self) -> WriteFailingSnapshot:
            return self

        def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
            return None

        def close(self) -> None:
            return None

        def seek(self, _offset: int, _whence: int = 0) -> int:
            return 0

        def write(self, _data: bytes) -> int:
            raise OSError("No space left on device")

    def temporary_file_full(*_args: object, **_kwargs: object) -> WriteFailingSnapshot:
        return WriteFailingSnapshot()

    monkeypatch.setattr(image_measurement, "TemporaryFile", temporary_file_full)

    with pytest.raises(image_measurement.SnapshotSpoolError, match="temporary scan snapshot"):
        _ = profiling.profile_page(
            "projectID1",
            ProfileInputPage(
                name="source.png",
                image_path=image_path,
                source_path="projectID1/source.png",
            ),
        )


def test_profile_page_keeps_a_source_read_failure_as_an_unreadable_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("L", (8, 6), color=255).save(image_path)

    class ReadFailingSource:
        def __enter__(self) -> ReadFailingSource:
            return self

        def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            raise OSError("Input/output error")

    def open_source(path: Path, *_args: object, **_kwargs: object) -> ReadFailingSource:
        assert path == image_path
        return ReadFailingSource()

    monkeypatch.setattr(Path, "open", open_source)

    page = profiling.profile_page(
        "projectID1",
        ProfileInputPage(
            name="source.png",
            image_path=image_path,
            source_path="projectID1/source.png",
        ),
    )

    assert page.sha256 is None
    assert tuple(diagnostic.code for diagnostic in page.diagnostics) == ("image_unreadable",)


def test_profile_page_hashes_and_measures_one_captured_image_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "changing.png"
    Image.new("L", (8, 6), color=255).save(image_path)
    captured_bytes = image_path.read_bytes()

    replacement = Image.new("L", (8, 6), color=0)
    replacement_path = tmp_path / "replacement.png"
    replacement.save(replacement_path)
    replacement_bytes = replacement_path.read_bytes()

    original_open_snapshot = profiling.open_image_snapshot

    @contextmanager
    def open_snapshot_then_replace(path: str | Path) -> Iterator[ImageSnapshot]:
        with original_open_snapshot(path) as snapshot:
            Path(path).write_bytes(replacement_bytes)
            yield snapshot

    monkeypatch.setattr(profiling, "open_image_snapshot", open_snapshot_then_replace)

    page = profiling.profile_page(
        "projectID1",
        ProfileInputPage(
            name="changing.png",
            image_path=image_path,
            source_path="projectID1/changing.png",
        ),
    )

    assert page.sha256 == sha256(captured_bytes).hexdigest()
    assert page.foreground_pixels == 0
    assert page.sha256 != sha256(replacement_bytes).hexdigest()


def test_pool_uses_median_mad_natural_evidence_and_metric_exclusions() -> None:
    pages = (
        _page(
            "p10.png",
            margins=(10, 11, 12, 13),
            band_height=11.0,
            band_pitch=20.0,
        ),
        _page(
            "p2.png",
            margins=(4, 5, 6, 7),
            band_height=7.0,
            band_pitch=None,
            diagnostic_codes=("one_ink_band",),
        ),
        _page(
            "p3.png",
            margins=(2, 3, 4, 5),
            band_height=5.0,
            band_pitch=12.0,
            diagnostic_codes=("border_dominated",),
        ),
        _unavailable_page("p4.png", "image_missing"),
    )

    height = pool_estimate("median_band_height_px", pages)
    pitch = pool_estimate("median_band_top_pitch_px", pages)
    left_margin = pool_estimate("margin_left_px", pages)

    assert (height.value, height.sample_count) == (9.0, 2)
    assert height.evidence_pages == ("p2.png", "p10.png")
    assert height.exclusions == ("p3.png:border_dominated", "p4.png:image_missing")
    assert height.extensions == {"median_absolute_deviation": 2.0}
    assert (pitch.value, pitch.sample_count) == (20.0, 1)
    assert pitch.exclusions == (
        "p2.png:one_ink_band",
        "p3.png:border_dominated",
        "p4.png:image_missing",
    )
    assert (left_margin.value, left_margin.sample_count) == (7.0, 2)
    assert left_margin.evidence_pages == ("p2.png", "p10.png")
    assert left_margin.exclusions == ("p3.png:border_dominated", "p4.png:image_missing")
    assert left_margin.truth_class == "pooled"
    assert left_margin.method == "median-mad/v1"
    assert left_margin.confidence is None
    assert left_margin.confidence_kind == "uncalibrated"


def test_pool_excludes_missing_bounds_with_a_stable_metric_diagnostic() -> None:
    pages = (
        _page(
            "p2.png",
            margins=(4, 5, 6, 7),
            band_height=7.0,
            band_pitch=20.0,
        ),
        _unavailable_page("p10.png", "foreground_bounds_unavailable"),
    )

    result = pool_estimate("margin_right_px", pages)

    assert result.value == 6.0
    assert result.evidence_pages == ("p2.png",)
    assert result.exclusions == ("p10.png:foreground_bounds_unavailable",)


def test_high_foreground_pages_keep_raw_values_but_contribute_to_no_pooled_metric() -> None:
    pages = (
        _page(
            "p2.png",
            margins=(4, 5, 6, 7),
            band_height=7.0,
            band_pitch=20.0,
        ),
        _page(
            "p10.png",
            margins=(0, 0, 0, 0),
            band_height=100.0,
            band_pitch=100.0,
            diagnostic_codes=("high_foreground_ratio",),
        ),
    )

    margin = pool_estimate("margin_top_px", pages)
    height = pool_estimate("median_band_height_px", pages)
    pitch = pool_estimate("median_band_top_pitch_px", pages)

    assert margin.value == 5.0
    assert height.value == 7.0
    assert pitch.value == 20.0
    assert margin.exclusions == ("p10.png:high_foreground_ratio",)
    assert height.exclusions == ("p10.png:high_foreground_ratio",)
    assert pitch.exclusions == ("p10.png:high_foreground_ratio",)


def test_profile_method_versions_are_stable() -> None:
    assert profile_methods() == {
        "foreground_threshold": "otsu-256/v1",
        "ink_bands": "row-ink-bands/v1",
        "pooling": "median-mad/v1",
        "page_templates": "first-band-templates/v3",
    }


def test_page_template_constants_are_stable() -> None:
    from pdomain_pgdp_measure.page_templates import page_template_methods

    assert page_template_methods() == {
        "algorithm": "first-band-templates/v3",
        "head_mad_maximum_px": 25,
        "head_window_minimum_px": 8,
        "head_window_mad_factor": 3,
        "chapter_sink_minimum_px": 150,
        "side_minimum_share": 0.2,
    }


def _page(
    page_name: str,
    *,
    margins: tuple[int, int, int, int],
    band_height: float | None,
    band_pitch: float | None,
    diagnostic_codes: tuple[str, ...] = (),
) -> PageMeasurement:
    derived_estimates = tuple(
        estimate
        for estimate in (
            _derived_estimate(page_name, "median_band_height_px", band_height),
            _derived_estimate(page_name, "median_band_top_pitch_px", band_pitch),
        )
        if estimate is not None
    )
    return PageMeasurement(
        page_name=page_name,
        source_path=f"projectID1/{page_name}",
        sha256="a" * 64,
        source_frame=CoordinateFrame(width=100, height=200),
        image_mode="L",
        grayscale_threshold=127,
        foreground_pixels=100,
        foreground_bounds=(margins[0], margins[1], 100 - margins[2], 200 - margins[3]),
        margins=margins,
        ink_bands=(InkBand(y_start=10, y_end=20),),
        derived_estimates=derived_estimates,
        diagnostics=tuple(_diagnostic(code, page_name) for code in diagnostic_codes),
    )


def _derived_estimate(page_name: str, name: str, value: float | None) -> Estimate | None:
    if value is None:
        return None
    return Estimate(
        name=name,
        value=value,
        unit="px",
        truth_class="derived",
        method="row-ink-bands/v1",
        sample_count=1,
        evidence_pages=(page_name,),
    )


def _unavailable_page(page_name: str, diagnostic_code: str) -> PageMeasurement:
    diagnostic_codes = (
        (diagnostic_code,)
        if diagnostic_code in {"image_missing", "image_unreadable"}
        else (diagnostic_code, "image_missing")
    )
    return PageMeasurement(
        page_name=page_name,
        source_path=f"projectID1/{page_name}",
        sha256=None,
        source_frame=None,
        image_mode=None,
        grayscale_threshold=None,
        foreground_pixels=None,
        foreground_bounds=None,
        margins=None,
        ink_bands=None,
        diagnostics=tuple(_diagnostic(code, page_name) for code in diagnostic_codes),
    )


def _diagnostic(code: str, page_name: str) -> ProfileDiagnostic:
    return ProfileDiagnostic(code=code, message=code, project_id="projectID1", page_name=page_name)


def _page_with_bands(page_name: str, band_tops: tuple[int, ...]) -> PageMeasurement:
    """Build a measured page whose ink bands start at the given tops."""

    width, height = 100, 600
    bands = tuple(InkBand(y_start=top, y_end=top + 8) for top in band_tops)
    bounds = (10, band_tops[0], 90, band_tops[-1] + 8) if band_tops else None
    margins = (
        (bounds[0], bounds[1], width - bounds[2], height - bounds[3])
        if bounds is not None
        else None
    )
    return PageMeasurement(
        page_name=page_name,
        source_path=f"projectID1/{page_name}",
        sha256="a" * 64,
        source_frame=CoordinateFrame(width=width, height=height),
        image_mode="L",
        grayscale_threshold=127,
        foreground_pixels=100 if band_tops else 0,
        foreground_bounds=bounds,
        margins=margins,
        ink_bands=bands,
        derived_estimates=(),
        diagnostics=() if band_tops else (_diagnostic("blank_page", page_name),),
    )


def test_pool_first_band_top_reports_median_and_deviation() -> None:
    pages = (
        _page_with_bands("p1.png", (71, 130, 190)),
        _page_with_bands("p2.png", (71, 130, 190)),
        _page_with_bands("p3.png", (72, 131, 191)),
        _page_with_bands("p4.png", (367, 420)),
    )

    estimate = pool_estimate("first_band_top_px", pages)

    assert estimate.value == 71.5
    assert estimate.sample_count == 4
    assert estimate.unit == "px"
    assert estimate.truth_class == "pooled"
    assert estimate.extensions["median_absolute_deviation"] == 0.5


def test_pool_first_band_top_excludes_pages_without_bands() -> None:
    pages = (
        _page_with_bands("p1.png", (71,)),
        _page_with_bands("p2.png", ()),
        _page_with_bands("p3.png", (73,)),
    )

    estimate = pool_estimate("first_band_top_px", pages)

    assert estimate.value == 72.0
    assert estimate.sample_count == 2
    assert estimate.evidence_pages == ("p1.png", "p3.png")
    assert estimate.exclusions == ("p2.png:blank_page",)


def test_pool_first_band_top_handles_a_single_page() -> None:
    estimate = pool_estimate("first_band_top_px", (_page_with_bands("p1.png", (71,)),))

    assert estimate.value == 71.0
    assert estimate.sample_count == 1
    assert estimate.extensions["median_absolute_deviation"] == 0.0


def test_scattered_book_reports_a_large_first_band_deviation() -> None:
    pages = tuple(
        _page_with_bands(f"p{index}.png", (top,))
        for index, top in enumerate((60, 80, 100, 120, 140))
    )

    estimate = pool_estimate("first_band_top_px", pages)

    assert estimate.value == 100.0
    assert estimate.extensions["median_absolute_deviation"] == 20.0
