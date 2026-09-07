"""Reviewed synthetic line fixtures for font-free typographic measurement.

Each case recreates the line geometry measured on a named real page of an admitted book, so
the estimator meets the shapes the corpus actually contains: 15 px x-height on a 4 px stroke
in one book, 10 px on 3 px in another, 17 px on 6 px in a third. None of the bitmaps copies
source pixels or PGDP text, which the manifest records as `provenance` and this file checks by
requiring every `fixture_sha256` to differ from its page's `source_sha256`.

Regenerate with `uv run python tests/fixtures/pgdp_typography/generate_fixtures.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar, Literal

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
)

from pdomain_pgdp_measure.alignment_image import build_candidate_mask
from pdomain_pgdp_measure.image_measurement import measure_image_snapshot, open_image_snapshot
from pdomain_pgdp_measure.typography_measure import (
    LineTypographyMeasurement,
    LineWordMeasurement,
    measure_line_typography,
    measure_line_words,
)

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "pgdp_typography"
_TOLERANCE_PX = 1


class _ExpectedPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    measured: StrictBool
    baseline_row_px: StrictInt
    x_height_top_row_px: StrictInt
    x_height_px: StrictInt
    stroke_width_px: StrictInt
    skew_slope: StrictFloat
    word_run_count: StrictInt
    source_word_count: StrictInt
    words_reconciled: StrictBool


class _CasePayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    case_id: StrictStr
    project_id: StrictStr
    page_name: StrictStr
    image_path: StrictStr
    source_sha256: StrictStr
    fixture_sha256: StrictStr
    provenance: Literal["synthetic_derivative", "real_scan_crop"]
    transform: StrictStr
    box: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    gap_threshold_px: StrictInt
    visible_text: StrictStr
    expected: _ExpectedPayload
    reviewed_on: StrictStr
    review_method: Literal["model-reviewed", "human-reviewed"]


class _ManifestPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    licence_rule: StrictStr
    cases: list[_CasePayload]


_MANIFEST_ADAPTER = TypeAdapter(_ManifestPayload)


@dataclass(frozen=True, slots=True)
class TypographyCase:
    """One reviewed fixture line and the geometry it was drawn from."""

    case_id: str
    project_id: str
    page_name: str
    image_path: str
    source_sha256: str
    fixture_sha256: str
    provenance: str
    transform: str
    box: tuple[int, int, int, int]
    gap_threshold_px: int
    visible_text: str
    expected: _ExpectedPayload
    reviewed_on: str
    review_method: str


def load_typography_cases() -> tuple[TypographyCase, ...]:
    payload = _MANIFEST_ADAPTER.validate_json((_FIXTURE_ROOT / "manifest.json").read_text("utf-8"))
    return tuple(
        TypographyCase(
            case_id=case.case_id,
            project_id=case.project_id,
            page_name=case.page_name,
            image_path=case.image_path,
            source_sha256=case.source_sha256,
            fixture_sha256=case.fixture_sha256,
            provenance=case.provenance,
            transform=case.transform,
            box=case.box,
            gap_threshold_px=case.gap_threshold_px,
            visible_text=case.visible_text,
            expected=case.expected,
            reviewed_on=case.reviewed_on,
            review_method=case.review_method,
        )
        for case in payload.cases
    )


_CASES = load_typography_cases()


def _measure(case: TypographyCase) -> tuple[object, object]:
    image_path = (_FIXTURE_ROOT / case.image_path).resolve()
    if _FIXTURE_ROOT.resolve() not in image_path.parents:
        raise ValueError(f"Fixture image path escapes the fixture root: {case.image_path}")
    if sha256(image_path.read_bytes()).hexdigest() != case.fixture_sha256:
        raise ValueError(f"Fixture hash mismatch: {case.case_id}")
    with open_image_snapshot(image_path) as snapshot:
        measurement = measure_image_snapshot(snapshot)
        bounds = measurement.foreground_bounds
        if bounds is None:
            raise ValueError(f"Fixture has no foreground bounds: {case.case_id}")
        mask, _rejected = build_candidate_mask(
            snapshot, source_frame=measurement.source_frame, foreground_bounds=bounds
        )
    result = measure_line_typography(mask, case.box)
    if not isinstance(result, LineTypographyMeasurement):
        return result, None
    words = measure_line_words(
        mask,
        case.box,
        tuple(_column_profile(mask, case.box)),
        case.visible_text,
        case.gap_threshold_px,
    )
    return result, words


def _column_profile(mask: object, box: tuple[int, int, int, int]) -> list[float]:
    """The candidate profile the alignment report would have stored for this box."""

    import numpy as np

    x_start, y_start, x_end, y_end = box
    cropped = np.asarray(mask)[y_start:y_end, x_start:x_end]
    counts = [int(np.count_nonzero(cropped[:, column])) for column in range(x_end - x_start)]
    total = sum(counts)
    return [count / total for count in counts]


def test_manifest_records_the_licence_decision_for_every_case() -> None:
    assert len(_CASES) >= 8
    assert len({case.case_id for case in _CASES}) == len(_CASES)
    assert len({case.fixture_sha256 for case in _CASES}) == len(_CASES)
    assert len({case.project_id for case in _CASES}) >= 3
    for case in _CASES:
        assert case.provenance == "synthetic_derivative"
        assert case.fixture_sha256 != case.source_sha256
        assert "copies no source pixels" in case.transform
        assert case.reviewed_on == "2026-09-04"
        assert case.review_method == "model-reviewed"
        assert case.gap_threshold_px >= 2


def test_manifest_covers_both_measured_and_excluded_lines() -> None:
    measured = [case for case in _CASES if case.expected.measured]
    excluded = [case for case in _CASES if not case.expected.measured]
    assert len(measured) >= 5
    assert len(excluded) >= 2
    assert any(not case.expected.words_reconciled for case in measured)


@pytest.mark.parametrize("case", _CASES, ids=tuple(case.case_id for case in _CASES))
def test_estimator_recovers_the_drawn_geometry(case: TypographyCase) -> None:
    result, words = _measure(case)

    if not case.expected.measured:
        assert not isinstance(result, LineTypographyMeasurement)
        return
    assert isinstance(result, LineTypographyMeasurement)
    assert abs(result.baseline_row_px - case.expected.baseline_row_px) <= _TOLERANCE_PX
    assert abs(result.x_height_top_row_px - case.expected.x_height_top_row_px) <= _TOLERANCE_PX
    assert abs(result.x_height_px - case.expected.x_height_px) <= _TOLERANCE_PX
    assert abs(result.stroke_width_px - case.expected.stroke_width_px) <= _TOLERANCE_PX
    assert abs(result.skew_slope - case.expected.skew_slope) <= 0.002

    assert isinstance(words, LineWordMeasurement)
    assert words.word_run_count == case.expected.word_run_count
    assert words.source_word_count == case.expected.source_word_count
    assert words.words_reconciled == case.expected.words_reconciled
    if case.expected.words_reconciled:
        assert len(words.words) == case.expected.word_run_count
    else:
        assert words.words == ()


# --- Review overlay ----------------------------------------------------------------------


def _overlay_page(box: tuple[int, int, int, int], scan_path: Path) -> object:
    """A one-line PageAlignment bound to a fixture bitmap, for overlay tests only."""

    from PIL import Image

    from pdomain_pgdp_measure.alignment_models import (
        AlignmentOperation,
        PageAlignment,
        WireLineCandidate,
        WireSourceLine,
    )
    from pdomain_pgdp_measure.profile_models import CoordinateFrame

    with Image.open(scan_path) as image:
        frame = CoordinateFrame(width=image.width, height=image.height)
    return PageAlignment(
        page_name="fixture.pbm",
        f2_sha256="a" * 64,
        scan_sha256=sha256(scan_path.read_bytes()).hexdigest(),
        source_frame=frame,
        source_lines=(
            WireSourceLine(
                ordinal=0,
                byte_start=0,
                byte_end=3,
                content_byte_start=0,
                content_byte_end=2,
                separator_byte_start=2,
                separator_byte_end=3,
                original_text="ab",
                separator_text="\n",
                visible_text="ab",
                matching_eligible=True,
                style_fitting_eligible=True,
            ),
        ),
        formatting_spans=(),
        candidates=(
            WireLineCandidate(
                ordinal=0,
                box=box,
                width=box[2] - box[0],
                height=box[3] - box[1],
                fill_ratio=0.5,
                component_count=1,
                foreground_pixels=1,
            ),
        ),
        operations=(
            AlignmentOperation(kind="match", source_ordinal=0, candidate_ordinal=0, cost=0.0),
        ),
        total_cost=0.0,
        second_best_cost=1.0,
        normalized_cost=0.0,
        matched_count=1,
        matched_ratio=1.0,
        uniqueness_margin=1.0,
        mean_width_residual=0.0,
        mean_indentation_residual=0.0,
        state="accepted",
        accepted=True,
        exclusions=(),
    )


def _overlay_line(case: TypographyCase) -> object:
    from pdomain_pgdp_measure.typography_models import LineTypography

    expected = case.expected
    box = case.box
    return LineTypography(
        candidate_ordinal=0,
        source_ordinal=0,
        box=box,
        line_ink_width_px=box[2] - box[0],
        line_ink_height_px=box[3] - box[1],
        baseline_row_px=expected.baseline_row_px,
        x_height_top_row_px=expected.x_height_top_row_px,
        x_height_px=expected.x_height_px,
        ascender_extent_px=expected.baseline_row_px - box[1],
        descender_extent_px=box[3] - expected.baseline_row_px,
        stroke_width_px=float(expected.stroke_width_px),
        skew_slope=expected.skew_slope,
        segment_count=4,
        word_run_count=expected.word_run_count,
        source_word_count=expected.source_word_count,
        words_reconciled=expected.words_reconciled,
    )


def test_review_overlay_draws_the_baseline_and_x_height_on_a_line_crop(tmp_path: Path) -> None:
    from pdomain_pgdp_measure.alignment_review import render_line_typography_overlay

    case = next(item for item in _CASES if item.case_id == "alpha-body")
    scan_path = _FIXTURE_ROOT / case.image_path
    output = tmp_path / "overlay.png"

    overlay = render_line_typography_overlay(
        scan_path, _overlay_page(case.box, scan_path), _overlay_line(case), output, margin_px=8
    )

    assert output.is_file()
    assert overlay.mode == "RGB"
    assert overlay.width == (case.box[2] - case.box[0]) + 16
    baseline_row = case.expected.baseline_row_px - (case.box[1] - 8)
    x_height_row = case.expected.x_height_top_row_px - (case.box[1] - 8)
    assert overlay.getpixel((0, baseline_row)) == (0, 96, 255)
    assert overlay.getpixel((0, x_height_row)) == (224, 0, 192)


def test_review_overlay_refuses_an_unmeasured_line(tmp_path: Path) -> None:
    from dataclasses import replace

    from pdomain_pgdp_measure.alignment_review import render_line_typography_overlay

    case = next(item for item in _CASES if item.case_id == "alpha-body")
    scan_path = _FIXTURE_ROOT / case.image_path
    unmeasured = replace(
        _overlay_line(case),
        baseline_row_px=None,
        x_height_top_row_px=None,
        x_height_px=None,
        ascender_extent_px=None,
        descender_extent_px=None,
        stroke_width_px=None,
        skew_slope=None,
        word_run_count=None,
        source_word_count=None,
        words_reconciled=None,
        exclusions=("insufficient_segments",),
    )

    with pytest.raises(ValueError, match="requires a measured line"):
        _ = render_line_typography_overlay(
            scan_path, _overlay_page(case.box, scan_path), unmeasured, tmp_path / "overlay.png"
        )


def test_review_overlay_refuses_a_scan_that_does_not_match_the_recorded_hash(
    tmp_path: Path,
) -> None:
    from pdomain_pgdp_measure.alignment_review import render_line_typography_overlay

    case = next(item for item in _CASES if item.case_id == "alpha-body")
    other = next(item for item in _CASES if item.case_id == "beta-small")
    scan_path = _FIXTURE_ROOT / case.image_path

    with pytest.raises(ValueError, match="scan_sha256"):
        _ = render_line_typography_overlay(
            scan_path,
            _overlay_page(case.box, _FIXTURE_ROOT / other.image_path),
            _overlay_line(case),
            tmp_path / "overlay.png",
        )
