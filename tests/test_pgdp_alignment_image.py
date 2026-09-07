"""Tests for deterministic PGDP source-frame line candidate extraction."""

from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, override

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pdomain_pgdp_measure import alignment_image
from pdomain_pgdp_measure.image_measurement import (
    ImageMeasurement,
    ImageSnapshot,
    SnapshotSpoolError,
    measure_image_snapshot,
    open_image_snapshot,
)
from pdomain_pgdp_measure.profile_models import CoordinateFrame, InkBand, ProfileDiagnostic

if TYPE_CHECKING:
    from pdomain_pgdp_measure.alignment_image import (
        Bounds,
        CandidateExtraction,
        _Cluster,
    )


_SOURCE_FRAME = CoordinateFrame(width=100, height=80)
_FOREGROUND_BOUNDS = (10, 10, 90, 70)
_ROW_BANDS = (
    InkBand(y_start=20, y_end=25),
    InkBand(y_start=35, y_end=40),
    InkBand(y_start=50, y_end=55),
)
_ALIGNMENT_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "pgdp_alignment"


class _CandidateReplayReadFailingSnapshot(BytesIO):
    @override
    def read(self, _size: int | None = -1) -> bytes:
        raise OSError("Input/output error")


class _CandidateReplaySeekFailingSnapshot(BytesIO):
    @override
    def seek(self, _offset: int, _whence: int = 0) -> int:
        raise OSError("Input/output error")


def _extract(
    image_path: Path, *, diagnostics: tuple[ProfileDiagnostic, ...] = ()
) -> CandidateExtraction:
    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    with open_image_snapshot(image_path) as snapshot:
        return extract_line_candidates(
            snapshot,
            source_frame=_SOURCE_FRAME,
            foreground_bounds=_FOREGROUND_BOUNDS,
            ink_bands=_ROW_BANDS,
            diagnostics=diagnostics,
        )


def _measured_source_frame(_snapshot: ImageSnapshot) -> ImageMeasurement:
    return ImageMeasurement(
        sha256="a" * 64,
        source_frame=_SOURCE_FRAME,
        image_mode="L",
        exif_orientation=None,
        grayscale_threshold=0,
        foreground_pixels=1,
        foreground_bounds=(20, 20, 21, 21),
        margins=(20, 20, 79, 59),
        ink_bands=(),
        median_ink_band_height=None,
        median_ink_band_top_pitch=None,
        diagnostics=(),
    )


def _page_with_rows(*, border: bool = False, rule: bool = False) -> Image.Image:
    image = Image.new("L", (100, 80), color=255)
    draw = ImageDraw.Draw(image)
    if border:
        draw.rectangle((0, 0, 99, 79), outline=0, width=1)
    for y_start, y_end, first_end, second_start, second_end in (
        (20, 24, 28, 32, 55),
        (35, 39, 31, 35, 58),
        (50, 54, 34, 40, 62),
    ):
        draw.rectangle((20, y_start, first_end, y_end), fill=0)
        draw.rectangle((second_start, y_start, second_end, y_end), fill=0)
    if rule:
        draw.line((10, 60, 89, 60), fill=0, width=1)
    return image


def test_extract_candidates_removes_border_and_long_rule(tmp_path: Path) -> None:
    image_path = tmp_path / "rows.png"
    _page_with_rows(border=True, rule=True).save(image_path)

    result = _extract(image_path)

    assert [candidate.box for candidate in result.candidates] == [
        (20, 20, 56, 25),
        (20, 35, 59, 40),
        (20, 50, 63, 55),
    ]
    assert [candidate.band_ordinal for candidate in result.candidates] == [0, 1, 2]
    assert [(item.reason, item.box) for item in result.rejected] == [
        ("long_horizontal_rule", (10, 60, 90, 61)),
        ("page_border", (0, 0, 100, 80)),
    ]
    assert result.probable_multi_column is False
    assert result.gutter is None
    assert result.exclusions == ()


def test_extract_candidates_removes_one_border_across_five_prose_bands(tmp_path: Path) -> None:
    image_path = tmp_path / "five-rows.png"
    image = Image.new("L", (100, 120), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 99, 119), outline=0, width=1)
    bands = tuple(InkBand(y_start=y_start, y_end=y_start + 5) for y_start in range(20, 95, 15))
    for band in bands:
        draw.rectangle((20, band.y_start, 60, band.y_end - 1), fill=0)
    image.save(image_path)

    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    with open_image_snapshot(image_path) as snapshot:
        result = extract_line_candidates(
            snapshot,
            source_frame=CoordinateFrame(width=100, height=120),
            foreground_bounds=(0, 0, 100, 120),
            ink_bands=bands,
        )

    assert [candidate.box for candidate in result.candidates] == [
        (20, 20, 61, 25),
        (20, 35, 61, 40),
        (20, 50, 61, 55),
        (20, 65, 61, 70),
        (20, 80, 61, 85),
    ]
    assert [(item.reason, item.box) for item in result.rejected] == [
        ("page_border", (0, 0, 100, 120)),
    ]


def test_candidate_records_component_measurements_and_normalized_profile(tmp_path: Path) -> None:
    image_path = tmp_path / "rows.png"
    _page_with_rows().save(image_path)

    result = _extract(image_path)

    candidate = result.candidates[0]
    assert candidate.component_count == 2
    assert candidate.foreground_pixels == 165
    assert candidate.width == 36
    assert candidate.height == 5
    assert candidate.fill_ratio == pytest.approx(165 / 180)
    assert len(candidate.horizontal_ink_profile) == candidate.width
    assert sum(candidate.horizontal_ink_profile) == pytest.approx(1.0)
    assert candidate.horizontal_ink_profile[:9] == pytest.approx((1 / 33,) * 9)


def test_extract_candidates_tolerates_one_fragmented_band_and_records_it(tmp_path: Path) -> None:
    image_path = tmp_path / "fragmented.png"
    image = _page_with_rows()
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 35, 78, 39), fill=0)
    image.save(image_path)

    result = _extract(image_path)

    # One fragmented band out of three is a rate of 0.333, under the 0.35 limit.
    # Band 1 keeps both clusters, so its candidate spans out to x=79.
    assert result.exclusions == ()
    assert [candidate.box for candidate in result.candidates] == [
        (20, 20, 56, 25),
        (20, 35, 79, 40),
        (20, 50, 63, 55),
    ]
    assert [(item.reason, item.band_ordinal, item.box) for item in result.rejected] == [
        ("fragmented_band", 1, (20, 35, 59, 40)),
        ("fragmented_band", 1, (70, 35, 79, 40)),
    ]


def test_extract_candidates_sorts_three_fragmented_clusters_by_box(tmp_path: Path) -> None:
    image_path = tmp_path / "three-fragments.png"
    image = _page_with_rows()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 35, 99, 39), fill=255)
    draw.rectangle((20, 35, 28, 39), fill=0)
    draw.rectangle((50, 35, 55, 39), fill=0)
    draw.rectangle((75, 35, 80, 39), fill=0)
    image.save(image_path)

    result = _extract(image_path)

    # Still one fragmented band out of three, so only the gutter excludes the page.
    assert result.exclusions == ("probable_multi_column",)
    assert [(item.band_ordinal, item.box) for item in result.rejected] == [
        (1, (20, 35, 29, 40)),
        (1, (50, 35, 56, 40)),
        (1, (75, 35, 81, 40)),
    ]


def test_extract_candidates_keeps_fragmented_evidence_sorted_when_geometry_is_drawn_reversed(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reversed-fragments.png"
    image = Image.new("L", (100, 80), color=255)
    draw = ImageDraw.Draw(image)
    for y_start, y_end in ((20, 24), (35, 39), (50, 54)):
        draw.rectangle((70, y_start, 78, y_end), fill=0)
        draw.rectangle((45, y_start, 53, y_end), fill=0)
        draw.rectangle((20, y_start, 28, y_end), fill=0)
    image.save(image_path)

    result = _extract(image_path)

    assert result.candidates == ()
    assert result.exclusions == ("fragmented_band", "probable_multi_column")
    assert [(item.band_ordinal, item.box) for item in result.rejected] == [
        (0, (20, 20, 29, 25)),
        (0, (45, 20, 54, 25)),
        (0, (70, 20, 79, 25)),
        (1, (20, 35, 29, 40)),
        (1, (45, 35, 54, 40)),
        (1, (70, 35, 79, 40)),
        (2, (20, 50, 29, 55)),
        (2, (45, 50, 54, 55)),
        (2, (70, 50, 79, 55)),
    ]


def test_join_components_is_stable_when_component_input_order_reverses() -> None:
    from pdomain_pgdp_measure.alignment_image import Component, join_components

    components = (
        Component(ordinal=0, label=0, box=(20, 35, 29, 40), foreground_pixels=45),
        Component(ordinal=1, label=1, box=(50, 35, 56, 40), foreground_pixels=30),
        Component(ordinal=2, label=2, box=(75, 35, 81, 40), foreground_pixels=30),
    )

    expected_boxes = ((20, 35, 29, 40), (50, 35, 56, 40), (75, 35, 81, 40))

    assert tuple(cluster.box for cluster in join_components(components)) == expected_boxes
    assert tuple(cluster.box for cluster in join_components(tuple(reversed(components)))) == (
        expected_boxes
    )


def test_extract_candidates_excludes_an_unexpected_blank_snapshot(tmp_path: Path) -> None:
    image_path = tmp_path / "blank.png"
    Image.new("L", (100, 80), color=255).save(image_path)

    result = _extract(image_path)

    assert result.candidates == ()
    assert result.rejected == ()
    assert result.exclusions == ("blank_page",)


def test_extract_candidates_detects_persistent_vertical_gutter(tmp_path: Path) -> None:
    image_path = tmp_path / "columns.png"
    image = Image.new("L", (100, 80), color=255)
    draw = ImageDraw.Draw(image)
    for y_start, y_end in ((20, 24), (35, 39), (50, 54)):
        draw.rectangle((10, y_start, 38, y_end), fill=0)
        draw.rectangle((61, y_start, 89, y_end), fill=0)
    image.save(image_path)

    result = _extract(image_path)

    assert result.probable_multi_column is True
    assert result.gutter is not None
    assert result.gutter.box == (39, 20, 61, 55)
    assert result.exclusions == ("fragmented_band", "probable_multi_column")


@pytest.mark.parametrize(
    ("inherited_code", "expected_exclusion"),
    [
        ("blank_page", "blank_page"),
        ("one_ink_band", "insufficient_ink_bands"),
        ("no_ink_bands", "insufficient_ink_bands"),
        ("border_dominated", "border_dominated"),
        ("high_foreground_ratio", "high_foreground_ratio"),
        ("image_missing", "image_missing"),
        ("image_unreadable", "image_unreadable"),
        ("image_decode_failed", "image_decode_failed"),
        ("foreground_bounds_unavailable", "foreground_bounds_unavailable"),
        ("", "inherited_profile_diagnostic"),
        ("future_code", "inherited_profile_diagnostic"),
    ],
)
def test_extract_candidates_maps_inherited_profile_diagnostics(
    tmp_path: Path, inherited_code: str, expected_exclusion: str
) -> None:
    image_path = tmp_path / "rows.png"
    _page_with_rows().save(image_path)
    diagnostic = ProfileDiagnostic(code=inherited_code, message="retained evidence")

    result = _extract(image_path, diagnostics=(diagnostic,))

    assert result.candidates == ()
    assert result.exclusions == (expected_exclusion,)
    assert result.inherited_diagnostics == (diagnostic,)


def test_extract_candidates_excludes_missing_bounds_or_insufficient_bands(tmp_path: Path) -> None:
    image_path = tmp_path / "rows.png"
    _page_with_rows().save(image_path)

    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    with open_image_snapshot(image_path) as snapshot:
        no_bounds = extract_line_candidates(
            snapshot,
            source_frame=_SOURCE_FRAME,
            foreground_bounds=None,
            ink_bands=_ROW_BANDS,
        )
        too_few_bands = extract_line_candidates(
            snapshot,
            source_frame=_SOURCE_FRAME,
            foreground_bounds=_FOREGROUND_BOUNDS,
            ink_bands=_ROW_BANDS[:1],
        )

    assert no_bounds.exclusions == ("foreground_bounds_unavailable",)
    assert too_few_bands.exclusions == ("insufficient_ink_bands",)


def test_extract_candidates_excludes_negative_foreground_bounds(tmp_path: Path) -> None:
    image_path = tmp_path / "rows.png"
    _page_with_rows().save(image_path)

    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    with open_image_snapshot(image_path) as snapshot:
        result = extract_line_candidates(
            snapshot,
            source_frame=_SOURCE_FRAME,
            foreground_bounds=(-1, 10, 90, 70),
            ink_bands=_ROW_BANDS,
        )

    assert result.candidates == ()
    assert result.exclusions == ("foreground_bounds_unavailable",)


def test_extract_candidates_uses_snapshot_bytes_after_live_image_mutates(tmp_path: Path) -> None:
    image_path = tmp_path / "changing.png"
    _page_with_rows().save(image_path)
    original_bytes = image_path.read_bytes()
    replacement = Image.new("L", (100, 80), color=255)
    replacement_path = tmp_path / "replacement.png"
    replacement.save(replacement_path)

    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    _ = image_path.write_bytes(original_bytes)
    with open_image_snapshot(image_path) as snapshot:
        _ = image_path.write_bytes(replacement_path.read_bytes())
        result = extract_line_candidates(
            snapshot,
            source_frame=_SOURCE_FRAME,
            foreground_bounds=_FOREGROUND_BOUNDS,
            ink_bands=_ROW_BANDS,
        )

    assert result.source_sha256 == sha256(original_bytes).hexdigest()
    assert [candidate.box for candidate in result.candidates] == [
        (20, 20, 56, 25),
        (20, 35, 59, 40),
        (20, 50, 63, 55),
    ]


def test_extract_candidates_translates_snapshot_replay_read_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(alignment_image, "measure_image_snapshot", _measured_source_frame)
    snapshot = ImageSnapshot(
        sha256="a" * 64,
        source_file=_CandidateReplayReadFailingSnapshot(b"source"),
    )

    with pytest.raises(SnapshotSpoolError, match="temporary scan snapshot"):
        _ = alignment_image.extract_line_candidates(
            snapshot,
            source_frame=_SOURCE_FRAME,
            foreground_bounds=_FOREGROUND_BOUNDS,
            ink_bands=_ROW_BANDS,
        )


def test_extract_candidates_translates_snapshot_replay_seek_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(alignment_image, "measure_image_snapshot", _measured_source_frame)
    snapshot = ImageSnapshot(
        sha256="a" * 64,
        source_file=_CandidateReplaySeekFailingSnapshot(b"source"),
    )

    with pytest.raises(SnapshotSpoolError, match="temporary scan snapshot"):
        _ = alignment_image.extract_line_candidates(
            snapshot,
            source_frame=_SOURCE_FRAME,
            foreground_bounds=_FOREGROUND_BOUNDS,
            ink_bands=_ROW_BANDS,
        )


def test_extract_candidates_accepts_additive_source_frame_evidence(tmp_path: Path) -> None:
    image_path = tmp_path / "rows.png"
    _page_with_rows().save(image_path)

    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    with open_image_snapshot(image_path) as snapshot:
        result = extract_line_candidates(
            snapshot,
            source_frame=CoordinateFrame(
                width=100,
                height=80,
                extensions={"future_measurement": "retained"},
            ),
            foreground_bounds=_FOREGROUND_BOUNDS,
            ink_bands=_ROW_BANDS,
        )

    assert len(result.candidates) == 3


def test_alignment_image_methods_expose_all_fixed_thresholds() -> None:
    from pdomain_pgdp_measure.alignment_image import ALIGNMENT_IMAGE_METHODS

    assert ALIGNMENT_IMAGE_METHODS == {
        "algorithm": "source-frame-components/v4",
        "connectivity": 8,
        "minimum_ink_bands": 2,
        "page_border_edges": 3,
        "long_rule_minimum_width_ratio": 0.8,
        "long_rule_maximum_height_ratio": 0.02,
        "join_minimum_vertical_overlap_ratio": 0.4,
        "join_maximum_vertical_center_distance_ratio": 0.6,
        "join_maximum_horizontal_gap_median_height_ratio": 2.0,
        "merge_horizontally_overlapping_clusters": True,
        "fragmented_band_minor_ink_share": 0.02,
        "fragmented_band_maximum_rate": 0.35,
        "candidate_box_mode": "union",
        "suppress_furniture_bands": True,
        "speck_band_minimum_ink_share": 0.05,
        "speck_band_maximum_height_share": 0.31,
        "rule_band_maximum_height_share": 0.5,
        "rule_band_minimum_ink_density": 0.6,
        "gutter_minimum_width_ratio": 0.03,
        "gutter_minimum_vertical_coverage_ratio": 0.6,
    }


def test_extract_candidates_accepts_exactly_two_ink_bands(tmp_path: Path) -> None:
    image_path = tmp_path / "two-bands.png"
    _page_with_rows().save(image_path)

    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    with open_image_snapshot(image_path) as snapshot:
        result = extract_line_candidates(
            snapshot,
            source_frame=_SOURCE_FRAME,
            foreground_bounds=_FOREGROUND_BOUNDS,
            ink_bands=_ROW_BANDS[:2],
        )

    assert len(result.candidates) == 2
    assert result.exclusions == ()


@pytest.mark.parametrize(
    ("vertical_extent", "expected_page_border"),
    [(41, False), (80, True)],
    ids=("two_edges_retain", "three_edges_remove"),
)
def test_page_border_threshold_requires_exactly_three_source_edges(
    tmp_path: Path,
    vertical_extent: int,
    expected_page_border: bool,
) -> None:
    image_path = tmp_path / f"border-{vertical_extent}.png"
    image = _page_with_rows()
    draw = ImageDraw.Draw(image)
    draw.line((0, 0, 0, vertical_extent - 1), fill=0)
    draw.line((0, 0, 50, 0), fill=0)
    image.save(image_path)

    result = _extract(image_path)

    assert any(item.reason == "page_border" for item in result.rejected) is expected_page_border


@pytest.mark.parametrize(
    ("line_box", "expected_long_rule"),
    [
        ((10, 20, 89, 20), False),
        ((10, 20, 90, 20), True),
        ((10, 20, 90, 21), False),
    ],
    ids=(
        "width_eighty_percent_retain",
        "width_over_eighty_percent_remove",
        "height_two_percent_retain",
    ),
)
def test_long_rule_thresholds_are_strict(
    tmp_path: Path,
    line_box: tuple[int, int, int, int],
    expected_long_rule: bool,
) -> None:
    image_path = tmp_path / "rule.png"
    image = Image.new("L", (100, 100), color=255)
    ImageDraw.Draw(image).rectangle(line_box, fill=0)
    image.save(image_path)

    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    with open_image_snapshot(image_path) as snapshot:
        result = extract_line_candidates(
            snapshot,
            source_frame=CoordinateFrame(width=100, height=100),
            foreground_bounds=(0, 0, 100, 100),
            ink_bands=(InkBand(y_start=20, y_end=25), InkBand(y_start=40, y_end=45)),
        )

    assert (
        any(item.reason == "long_horizontal_rule" for item in result.rejected) is expected_long_rule
    )


@pytest.mark.parametrize(
    ("right_box", "expected_cluster_count"),
    [
        ((20, 6, 30, 26), 1),
        ((20, 7, 30, 27), 2),
        ((20, 6, 30, 16), 1),
        ((20, 7, 30, 17), 2),
        ((15, 0, 20, 5), 1),
        ((16, 0, 21, 5), 2),
    ],
    ids=(
        "vertical_overlap_forty_percent_accept",
        "vertical_overlap_below_forty_percent_reject",
        "center_distance_sixty_percent_accept",
        "center_distance_over_sixty_percent_reject",
        "median_height_gap_two_times_accept",
        "median_height_gap_over_two_times_reject",
    ),
)
def test_component_join_threshold_boundaries(
    right_box: tuple[int, int, int, int], expected_cluster_count: int
) -> None:
    from pdomain_pgdp_measure.alignment_image import Component, join_components

    left = Component(ordinal=0, label=0, box=(0, 0, 10, 10), foreground_pixels=100)
    if right_box[3] - right_box[1] == 5:
        left = Component(ordinal=0, label=0, box=(0, 0, 5, 5), foreground_pixels=25)
    right = Component(ordinal=1, label=1, box=right_box, foreground_pixels=50)

    assert len(join_components((left, right))) == expected_cluster_count


def test_extract_candidates_connects_diagonal_foreground_pixels_with_eight_connectivity(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "diagonal.png"
    image = Image.new("L", (100, 80), color=255)
    draw = ImageDraw.Draw(image)
    for y_start in (20, 35, 50):
        draw.point((20, y_start), fill=0)
        draw.point((21, y_start + 1), fill=0)
    image.save(image_path)

    result = _extract(image_path)

    assert [candidate.component_count for candidate in result.candidates] == [1, 1, 1]
    assert [candidate.foreground_pixels for candidate in result.candidates] == [2, 2, 2]


def _gutter_image(*, width: int, center_ink_rows: tuple[int, ...]) -> Image.Image:
    image = Image.new("L", (100, 80), color=255)
    draw = ImageDraw.Draw(image)
    right_start = 39 + width
    for y_start, y_end in ((20, 24), (35, 39), (50, 54)):
        draw.rectangle((10, y_start, 38, y_end), fill=0)
        draw.rectangle((right_start, y_start, 89, y_end), fill=0)
    for y_index in center_ink_rows:
        draw.point((40, y_index), fill=0)
    return image


@pytest.mark.parametrize(
    ("width", "expected_gutter"),
    [(2, False), (3, True)],
    ids=("width_under_three_percent_reject", "width_at_three_percent_accept"),
)
def test_gutter_minimum_width_threshold(tmp_path: Path, width: int, expected_gutter: bool) -> None:
    image_path = tmp_path / f"gutter-width-{width}.png"
    _gutter_image(width=width, center_ink_rows=()).save(image_path)

    result = _extract(image_path)

    assert (result.gutter is not None) is expected_gutter


@pytest.mark.parametrize(
    ("center_ink_rows", "expected_gutter"),
    [
        ((20, 22, 35, 37, 50, 52), True),
        ((20, 21, 22, 35, 36, 50, 51), False),
    ],
    ids=("empty_coverage_sixty_percent_accept", "empty_coverage_below_sixty_percent_reject"),
)
def test_gutter_vertical_coverage_threshold(
    tmp_path: Path, center_ink_rows: tuple[int, ...], expected_gutter: bool
) -> None:
    image_path = tmp_path / "gutter-coverage.png"
    _gutter_image(width=3, center_ink_rows=center_ink_rows).save(image_path)

    result = _extract(image_path)

    assert (result.gutter is not None) is expected_gutter


def test_gutter_coverage_uses_its_full_vertical_extent(tmp_path: Path) -> None:
    image_path = tmp_path / "gutter-full-extent-coverage.png"
    _gutter_image(width=3, center_ink_rows=(20, 22, 35, 37, 50, 52)).save(image_path)

    result = _extract(image_path)

    assert result.gutter is not None
    assert result.gutter.vertical_coverage == pytest.approx(99 / 105)
    assert result.gutter.vertical_coverage >= 0.6


def test_component_join_uses_the_band_median_height_for_horizontal_gaps(tmp_path: Path) -> None:
    image_path = tmp_path / "median-gap.png"
    image = Image.new("L", (100, 80), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 20, 12, 21), fill=0)
    draw.rectangle((22, 20, 31, 29), fill=0)
    draw.rectangle((40, 20, 49, 29), fill=0)
    draw.rectangle((20, 40, 49, 44), fill=0)
    image.save(image_path)

    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    with open_image_snapshot(image_path) as snapshot:
        result = extract_line_candidates(
            snapshot,
            source_frame=_SOURCE_FRAME,
            foreground_bounds=(10, 20, 50, 45),
            ink_bands=(InkBand(y_start=20, y_end=30), InkBand(y_start=40, y_end=45)),
        )

    assert [(candidate.box, candidate.component_count) for candidate in result.candidates] == [
        ((10, 20, 50, 30), 3),
        ((20, 40, 50, 45), 1),
    ]


def test_merge_overlapping_clusters_unions_a_detached_mark_above_a_row() -> None:
    from pdomain_pgdp_measure.alignment_image import (
        Component,
        join_components,
        merge_horizontally_overlapping_clusters,
    )

    components = (
        Component(ordinal=0, label=0, box=(20, 35, 40, 40), foreground_pixels=100),
        Component(ordinal=1, label=1, box=(25, 30, 30, 33), foreground_pixels=15),
    )
    clusters = join_components(components)
    assert len(clusters) == 2

    merged = merge_horizontally_overlapping_clusters(clusters)

    assert len(merged) == 1
    assert merged[0].box == (20, 30, 40, 40)
    assert merged[0].component_count == 2
    assert merged[0].component_ordinal == 0
    assert merged[0].foreground_pixels == 115


def test_merge_overlapping_clusters_keeps_disjoint_columns_apart() -> None:
    from pdomain_pgdp_measure.alignment_image import (
        Component,
        join_components,
        merge_horizontally_overlapping_clusters,
    )

    components = (
        Component(ordinal=0, label=0, box=(10, 35, 38, 40), foreground_pixels=100),
        Component(ordinal=1, label=1, box=(61, 35, 89, 40), foreground_pixels=100),
    )
    clusters = join_components(components)
    assert len(clusters) == 2

    assert merge_horizontally_overlapping_clusters(clusters) == clusters


def test_merge_overlapping_clusters_follows_a_chain_of_overlaps() -> None:
    from pdomain_pgdp_measure.alignment_image import (
        Component,
        join_components,
        merge_horizontally_overlapping_clusters,
    )

    components = (
        Component(ordinal=0, label=0, box=(20, 35, 30, 40), foreground_pixels=40),
        Component(ordinal=1, label=1, box=(28, 28, 45, 33), foreground_pixels=50),
        Component(ordinal=2, label=2, box=(43, 35, 60, 40), foreground_pixels=60),
    )
    clusters = join_components(components)
    assert len(clusters) == 3

    merged = merge_horizontally_overlapping_clusters(clusters)

    assert len(merged) == 1
    assert merged[0].box == (20, 28, 60, 40)
    assert merged[0].component_count == 3
    assert merged[0].foreground_pixels == 150


def test_merge_overlapping_clusters_returns_a_single_cluster_unchanged() -> None:
    from pdomain_pgdp_measure.alignment_image import (
        Component,
        join_components,
        merge_horizontally_overlapping_clusters,
    )

    components = (Component(ordinal=0, label=0, box=(20, 35, 40, 40), foreground_pixels=100),)
    clusters = join_components(components)

    assert merge_horizontally_overlapping_clusters(clusters) == clusters
    assert merge_horizontally_overlapping_clusters(()) == ()


def test_merge_overlapping_clusters_is_stable_when_cluster_order_reverses() -> None:
    from pdomain_pgdp_measure.alignment_image import (
        Component,
        join_components,
        merge_horizontally_overlapping_clusters,
    )

    components = (
        Component(ordinal=0, label=0, box=(20, 35, 30, 40), foreground_pixels=40),
        Component(ordinal=1, label=1, box=(28, 28, 45, 33), foreground_pixels=50),
        Component(ordinal=2, label=2, box=(70, 35, 89, 40), foreground_pixels=60),
    )
    forward = merge_horizontally_overlapping_clusters(join_components(components))
    reversed_input = merge_horizontally_overlapping_clusters(
        tuple(reversed(join_components(tuple(reversed(components)))))
    )

    assert forward == reversed_input
    assert tuple(cluster.box for cluster in forward) == ((20, 28, 45, 40), (70, 35, 89, 40))


def _unjoinable_pair(main_pixels: int, minor_pixels: int) -> tuple[object, ...]:
    from pdomain_pgdp_measure.alignment_image import Component, join_components

    return join_components(
        (
            Component(ordinal=0, label=0, box=(20, 22, 61, 29), foreground_pixels=main_pixels),
            Component(ordinal=1, label=1, box=(70, 20, 71, 21), foreground_pixels=minor_pixels),
        )
    )


def test_drop_minor_ink_clusters_removes_a_cluster_below_the_share() -> None:
    from pdomain_pgdp_measure.alignment_image import drop_minor_ink_clusters

    clusters = _unjoinable_pair(287, 1)
    assert len(clusters) == 2

    counted, minor = drop_minor_ink_clusters(clusters)

    assert len(counted) == 1
    assert counted[0].foreground_pixels == 287
    assert len(minor) == 1
    assert minor[0].foreground_pixels == 1


def test_drop_minor_ink_clusters_keeps_a_cluster_exactly_at_the_share() -> None:
    from pdomain_pgdp_measure.alignment_image import drop_minor_ink_clusters

    clusters = _unjoinable_pair(98, 2)
    assert len(clusters) == 2

    counted, minor = drop_minor_ink_clusters(clusters)

    assert counted == clusters
    assert minor == ()


def test_drop_minor_ink_clusters_never_empties_a_band() -> None:
    from pdomain_pgdp_measure.alignment_image import (
        Component,
        drop_minor_ink_clusters,
        join_components,
    )

    components = tuple(
        Component(
            ordinal=index,
            label=index,
            box=(index * 10, 20, index * 10 + 2, 22),
            foreground_pixels=1,
        )
        for index in range(60)
    )
    clusters = join_components(components)
    assert len(clusters) == 60

    counted, minor = drop_minor_ink_clusters(clusters)

    assert counted == clusters
    assert minor == ()


def test_extract_candidates_records_minor_ink_clusters_as_evidence(tmp_path: Path) -> None:
    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    image_path = tmp_path / "speck.png"
    image = Image.new("L", (100, 80), color=255)
    draw = ImageDraw.Draw(image)
    for y_start in (20, 40, 60):
        draw.rectangle((20, y_start + 2, 60, y_start + 8), fill=0)
    draw.point((70, 40), fill=0)
    image.save(image_path)

    with open_image_snapshot(image_path) as snapshot:
        result = extract_line_candidates(
            snapshot,
            source_frame=_SOURCE_FRAME,
            foreground_bounds=(10, 10, 90, 75),
            ink_bands=(
                InkBand(y_start=20, y_end=30),
                InkBand(y_start=40, y_end=50),
                InkBand(y_start=60, y_end=70),
            ),
        )

    assert result.exclusions == ()
    assert [candidate.band_ordinal for candidate in result.candidates] == [0, 1, 2]
    assert [candidate.box for candidate in result.candidates] == [
        (20, 22, 61, 29),
        (20, 42, 61, 49),
        (20, 62, 61, 69),
    ]
    assert [(item.reason, item.band_ordinal, item.box) for item in result.rejected] == [
        ("minor_ink_cluster", 1, (70, 40, 71, 41)),
    ]


def _page_with_fragmented_bands(band_count: int, fragmented_count: int) -> Image.Image:
    """Draw `band_count` rows, the first `fragmented_count` of them split in two."""

    image = Image.new("L", (100, 20 * band_count + 20), color=255)
    draw = ImageDraw.Draw(image)
    for index in range(band_count):
        top = 20 * index + 2
        if index < fragmented_count:
            draw.rectangle((20, top, 40, top + 6), fill=0)
            draw.rectangle((70, top, 89, top + 6), fill=0)
        else:
            draw.rectangle((20, top, 60, top + 6), fill=0)
    return image


def _extract_rows(image_path: Path, band_count: int) -> CandidateExtraction:
    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    with open_image_snapshot(image_path) as snapshot:
        return extract_line_candidates(
            snapshot,
            source_frame=CoordinateFrame(width=100, height=20 * band_count + 20),
            foreground_bounds=(10, 10, 95, 20 * band_count + 15),
            ink_bands=tuple(
                InkBand(y_start=20 * index, y_end=20 * index + 15) for index in range(band_count)
            ),
        )


def test_extract_candidates_keeps_a_page_below_the_fragmented_rate(tmp_path: Path) -> None:
    image_path = tmp_path / "under.png"
    _page_with_fragmented_bands(10, 3).save(image_path)

    result = _extract_rows(image_path, 10)

    # 3 of 10 bands is a rate of 0.30, under the 0.35 limit.
    assert "fragmented_band" not in result.exclusions
    assert len(result.candidates) == 10


def test_extract_candidates_keeps_a_page_exactly_at_the_fragmented_rate(tmp_path: Path) -> None:
    image_path = tmp_path / "at-limit.png"
    _page_with_fragmented_bands(20, 7).save(image_path)

    result = _extract_rows(image_path, 20)

    # 7 of 20 bands is exactly 0.35, which the limit admits.
    assert "fragmented_band" not in result.exclusions
    assert len(result.candidates) == 20


def test_extract_candidates_excludes_a_page_above_the_fragmented_rate(tmp_path: Path) -> None:
    image_path = tmp_path / "over.png"
    _page_with_fragmented_bands(10, 4).save(image_path)

    result = _extract_rows(image_path, 10)

    # 4 of 10 bands is a rate of 0.40, past the 0.35 limit.
    assert "fragmented_band" in result.exclusions
    assert result.candidates == ()


def test_extract_candidates_raises_no_fragmented_band_without_measured_bands(
    tmp_path: Path,
) -> None:
    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    image_path = tmp_path / "off-band.png"
    image = Image.new("L", (100, 80), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 60, 26), fill=0)
    image.save(image_path)

    with open_image_snapshot(image_path) as snapshot:
        result = extract_line_candidates(
            snapshot,
            source_frame=_SOURCE_FRAME,
            foreground_bounds=(10, 10, 90, 70),
            ink_bands=(InkBand(y_start=40, y_end=50), InkBand(y_start=55, y_end=65)),
        )

    assert result.candidates == ()
    assert "fragmented_band" not in result.exclusions
    assert [item.reason for item in result.rejected] == ["empty_band", "empty_band"]


@pytest.mark.parametrize(
    ("mode", "expected_box", "expected_components"),
    [
        ("dominant", (20, 2, 41, 9), 1),
        ("union", (20, 2, 90, 9), 2),
        ("union_all", (20, 2, 90, 9), 2),
    ],
)
def test_candidate_box_mode_controls_what_a_fragmented_band_contributes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_box: tuple[int, int, int, int],
    expected_components: int,
) -> None:
    monkeypatch.setattr(alignment_image, "_CANDIDATE_BOX_MODE", mode)
    image_path = tmp_path / f"{mode}.png"
    _page_with_fragmented_bands(10, 3).save(image_path)

    result = _extract_rows(image_path, 10)

    # dominant keeps only the wider left cluster; the unions span the whole row.
    assert result.candidates[0].box == expected_box
    assert result.candidates[0].component_count == expected_components


def test_candidate_box_mode_rejects_an_unknown_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(alignment_image, "_CANDIDATE_BOX_MODE", "nonsense")
    image_path = tmp_path / "unknown-mode.png"
    _page_with_fragmented_bands(10, 3).save(image_path)

    with pytest.raises(ValueError, match="Unsupported candidate box mode"):
        _ = _extract_rows(image_path, 10)


def test_speck_band_is_dropped_before_it_becomes_a_candidate(tmp_path: Path) -> None:
    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    image_path = tmp_path / "speck-band.png"
    image = Image.new("L", (100, 80), color=255)
    draw = ImageDraw.Draw(image)
    for y_start in (20, 60):
        draw.rectangle((20, y_start + 2, 60, y_start + 8), fill=0)
    draw.point((70, 42), fill=0)
    image.save(image_path)

    with open_image_snapshot(image_path) as snapshot:
        result = extract_line_candidates(
            snapshot,
            source_frame=_SOURCE_FRAME,
            foreground_bounds=(10, 10, 90, 75),
            ink_bands=(
                InkBand(y_start=20, y_end=30),
                InkBand(y_start=40, y_end=50),
                InkBand(y_start=60, y_end=70),
            ),
        )

    assert [candidate.band_ordinal for candidate in result.candidates] == [0, 2]
    assert [item.reason for item in result.rejected] == ["speck_band"]
    assert "fragmented_band" not in result.exclusions


def _extract_with_furniture(image_path: Path, furniture: tuple[int, ...]) -> CandidateExtraction:
    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    with open_image_snapshot(image_path) as snapshot:
        return extract_line_candidates(
            snapshot,
            source_frame=_SOURCE_FRAME,
            foreground_bounds=_FOREGROUND_BOUNDS,
            ink_bands=_ROW_BANDS,
            furniture_band_ordinals=furniture,
        )


def test_a_furniture_band_yields_no_candidate_and_is_recorded(tmp_path: Path) -> None:
    image_path = tmp_path / "head.png"
    _page_with_rows().save(image_path)

    result = _extract_with_furniture(image_path, (0,))

    assert [candidate.band_ordinal for candidate in result.candidates] == [1, 2]
    assert [(item.reason, item.band_ordinal) for item in result.rejected] == [
        ("running_head", 0),
    ]
    assert result.exclusions == ()


def test_listing_no_furniture_band_keeps_every_candidate(tmp_path: Path) -> None:
    image_path = tmp_path / "no-head.png"
    _page_with_rows().save(image_path)

    result = _extract_with_furniture(image_path, ())

    assert [candidate.band_ordinal for candidate in result.candidates] == [0, 1, 2]
    assert result.rejected == ()


def test_suppressed_bands_leave_the_fragmented_rate_denominator(tmp_path: Path) -> None:
    image_path = tmp_path / "fragmented-head.png"
    image = _page_with_rows()
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 20, 78, 24), fill=0)
    image.save(image_path)

    fragmented_head = _extract_with_furniture(image_path, ())
    suppressed_head = _extract_with_furniture(image_path, (0,))

    # Band 0 is the only fragmented band. Suppressing it must remove it from
    # both sides of the rate, not just the numerator.
    assert any(item.reason == "fragmented_band" for item in fragmented_head.rejected)
    assert not any(item.reason == "fragmented_band" for item in suppressed_head.rejected)
    assert "fragmented_band" not in suppressed_head.exclusions


def _band_clusters(box: Bounds, foreground_pixels: int) -> tuple[_Cluster, ...]:
    from pdomain_pgdp_measure.alignment_image import Component, join_components

    return join_components(
        (Component(ordinal=0, label=0, box=box, foreground_pixels=foreground_pixels),)
    )


def test_a_dense_thin_band_is_read_as_a_printed_rule() -> None:
    from pdomain_pgdp_measure.alignment_image import is_rule_band

    rule = _band_clusters((505, 811, 713, 824), 1997)

    assert is_rule_band(rule, median_band_height=33.0)


def test_a_short_line_of_small_capitals_is_not_a_printed_rule() -> None:
    """Regression guard for 241.png, whose `towers.` reached a density of 0.68.

    Small capitals in a narrow box carry nearly as much ink per pixel as a rule,
    so density alone would take them. A rule is also far thinner than the rows
    around it, and that is what separates the two.
    """

    from pdomain_pgdp_measure.alignment_image import is_rule_band

    line = _band_clusters((300, 900, 397, 918), 1188)

    assert not is_rule_band(line, median_band_height=33.0)


def test_a_thin_band_of_ordinary_type_is_not_a_printed_rule() -> None:
    from pdomain_pgdp_measure.alignment_image import is_rule_band

    line = _band_clusters((300, 900, 422, 912), 758)

    assert not is_rule_band(line, median_band_height=30.0)


def test_a_rule_band_is_dropped_before_it_becomes_a_candidate(tmp_path: Path) -> None:
    """Regression for 097.png, whose decorative rule took the source line `I.`.

    The rule survives the long-rule test because it spans a fifth of the text
    block, so it reached the aligner as an ordinary candidate and every line
    below it shifted by one.
    """

    from pdomain_pgdp_measure.alignment_image import extract_line_candidates

    image_path = tmp_path / "rule-band.png"
    image = Image.new("L", (100, 80), color=255)
    draw = ImageDraw.Draw(image)
    for y_start in (20, 60):
        draw.rectangle((20, y_start + 2, 60, y_start + 8), fill=0)
    draw.rectangle((35, 43, 55, 45), fill=0)
    image.save(image_path)

    with open_image_snapshot(image_path) as snapshot:
        result = extract_line_candidates(
            snapshot,
            source_frame=_SOURCE_FRAME,
            foreground_bounds=(10, 10, 90, 75),
            ink_bands=(
                InkBand(y_start=20, y_end=30),
                InkBand(y_start=40, y_end=50),
                InkBand(y_start=60, y_end=70),
            ),
        )

    assert [candidate.band_ordinal for candidate in result.candidates] == [0, 2]
    assert [item.reason for item in result.rejected] == ["rule_band"]


def test_a_short_line_of_type_is_not_a_speck_however_little_ink_it_carries() -> None:
    """Regression for 097.png, whose section number `I.` was dropped as dust.

    The number carries about two percent of the ink a full row carries, so the
    ink share alone condemned it. Its source line then bound to the decorative
    rule below it, and every line under that shifted by one.
    """

    from pdomain_pgdp_measure.alignment_image import is_speck_band

    number = _band_clusters((543, 901, 567, 927), 200)

    assert not is_speck_band(number, median_band_ink=9053.0, median_band_height=33.0)


def test_a_fleck_of_dust_is_still_a_speck() -> None:
    from pdomain_pgdp_measure.alignment_image import is_speck_band

    dust = _band_clusters((430, 77, 440, 82), 42)

    assert is_speck_band(dust, median_band_ink=9053.0, median_band_height=33.0)


def test_speck_height_share_is_measured_against_the_page_median_row() -> None:
    """The same band is dust beside tall type and a short line beside small type."""

    from pdomain_pgdp_measure.alignment_image import is_speck_band

    band = _band_clusters((543, 901, 567, 913), 100)

    assert is_speck_band(band, median_band_ink=9053.0, median_band_height=40.0)
    assert not is_speck_band(band, median_band_ink=9053.0, median_band_height=24.0)


def test_a_band_that_is_both_dust_and_solid_is_recorded_as_dust() -> None:
    """A solid fleck of dirt satisfies both tests, and the speck test runs first.

    Both tests reject the band, so only the recorded reason turns on the order.
    Dust is the truer description of a mark this small, however densely it prints.
    """

    from pdomain_pgdp_measure.alignment_image import is_rule_band, is_speck_band

    fleck = _band_clusters((430, 77, 442, 82), 55)

    assert is_speck_band(fleck, median_band_ink=9053.0, median_band_height=33.0)
    assert is_rule_band(fleck, median_band_height=33.0)


def test_each_height_share_is_a_maximum_that_includes_its_boundary() -> None:
    """Both predicates read their share as `at most`, so the exact boundary counts."""

    from pdomain_pgdp_measure.alignment_image import is_rule_band, is_speck_band

    # 31 of 100 is exactly the speck ceiling; 32 is over it.
    assert is_speck_band(
        _band_clusters((0, 0, 200, 31), 1), median_band_ink=1000.0, median_band_height=100.0
    )
    assert not is_speck_band(
        _band_clusters((0, 0, 200, 32), 1), median_band_ink=1000.0, median_band_height=100.0
    )
    # 50 of 100 is exactly the rule ceiling; 51 is over it.
    assert is_rule_band(_band_clusters((0, 0, 200, 50), 9000), median_band_height=100.0)
    assert not is_rule_band(_band_clusters((0, 0, 200, 51), 9000), median_band_height=100.0)


def _load_alignment_fixture_cases() -> tuple[tuple[str, Path, Bounds], ...]:
    """Read `(case_id, image_path, foreground_bounds)` for every committed alignment fixture."""

    manifest = json.loads((_ALIGNMENT_FIXTURE_ROOT / "manifest.json").read_text("utf-8"))
    cases: list[tuple[str, Path, Bounds]] = []
    for case in manifest["cases"]:
        x_start, y_start, x_end, y_end = case["foreground_bounds"]
        cases.append(
            (
                case["case_id"],
                (_ALIGNMENT_FIXTURE_ROOT / case["image_path"]).resolve(),
                (x_start, y_start, x_end, y_end),
            )
        )
    return tuple(cases)


def test_build_candidate_mask_reproduces_every_fixture_candidates_recorded_ink() -> None:
    """A rebuilt mask, cropped to a candidate's box, reproduces its stored ink bit for bit.

    `_extract_band_candidates` computes `foreground_pixels` and
    `horizontal_ink_profile` from `foreground[box]` -- the page mask cropped to
    the candidate's box -- not from cluster membership. `build_candidate_mask`
    rebuilds that same mask from the same three steps, so cropping it to the
    same box must reproduce both stored values exactly, on every committed
    alignment fixture and every candidate it yields.
    """

    reproduced_candidate_count = 0
    for case_id, image_path, foreground_bounds in _load_alignment_fixture_cases():
        with open_image_snapshot(image_path) as snapshot:
            measurement = measure_image_snapshot(snapshot)
            extraction = alignment_image.extract_line_candidates(
                snapshot,
                source_frame=measurement.source_frame,
                foreground_bounds=foreground_bounds,
                ink_bands=tuple(
                    InkBand(y_start=band.y_start, y_end=band.y_end)
                    for band in measurement.ink_bands
                ),
            )
            mask, _rejected = alignment_image.build_candidate_mask(
                snapshot,
                source_frame=measurement.source_frame,
                foreground_bounds=foreground_bounds,
            )
        for candidate in extraction.candidates:
            x_start, y_start, x_end, y_end = candidate.box
            cropped = mask[y_start:y_end, x_start:x_end]
            column_counts = tuple(
                int(np.count_nonzero(cropped[:, column_index]))
                for column_index in range(x_end - x_start)
            )
            rebuilt_ink = sum(column_counts)
            assert rebuilt_ink == candidate.foreground_pixels, case_id
            rebuilt_profile = tuple(count / rebuilt_ink for count in column_counts)
            assert rebuilt_profile == candidate.horizontal_ink_profile, case_id
            reproduced_candidate_count += 1
    assert reproduced_candidate_count > 0
