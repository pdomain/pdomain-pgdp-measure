"""Reviewed synthetic word fixtures for the column-run glyph cut.

Each case recreates the stroke and x-height geometry measured on a named real page and covers one
outcome the cut can reach: characters clear of one another, characters that touch, ink broken
inside a letter, and a word carrying an apostrophe that stands clear, which is counted as a
position and left unlabelled. None of the bitmaps copies source pixels or PGDP text, which the
manifest records as `provenance` and this file checks by requiring every `fixture_sha256` to
differ from its page's `source_sha256`.

Regenerate with `uv run python tests/fixtures/pgdp_glyphs/generate_fixtures.py`.
"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr, TypeAdapter

from pdomain_pgdp_measure.alignment_image import build_candidate_mask
from pdomain_pgdp_measure.glyph_atlas import cell_extent, render_sheet
from pdomain_pgdp_measure.glyph_cut import cut_word_glyphs
from pdomain_pgdp_measure.glyph_quality import LineBand, measure_glyph_quality
from pdomain_pgdp_measure.image_measurement import measure_image_snapshot, open_image_snapshot

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "pgdp_glyphs"


class _GlyphPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    character: StrictStr
    ordinal: StrictInt
    box: tuple[StrictInt, StrictInt, StrictInt, StrictInt]


class _ExpectedPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    visible_text: StrictStr
    run_count: StrictInt
    position_count: StrictInt
    separable: StrictBool
    reason: StrictStr | None
    glyphs: tuple[_GlyphPayload, ...]


class GlyphCase(BaseModel):
    """One reviewed word fixture and everything the cut is expected to produce from it."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True, frozen=True)

    case_id: StrictStr
    project_id: StrictStr
    page_name: StrictStr
    provenance: StrictStr
    review_method: StrictStr
    reviewed_on: StrictStr
    transform: StrictStr
    source_sha256: StrictStr
    fixture_sha256: StrictStr
    image_path: StrictStr
    atlas_path: StrictStr | None
    atlas_sha256: StrictStr | None
    box: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    baseline_row_px: StrictInt
    x_height_top_row_px: StrictInt
    stroke_px: StrictInt
    x_height_px: StrictInt
    expected: _ExpectedPayload


class _ManifestPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    cases: tuple[GlyphCase, ...]


_MANIFEST = TypeAdapter(_ManifestPayload).validate_json(
    (_FIXTURE_ROOT / "manifest.json").read_bytes()
)
CASES = _MANIFEST.cases


def _mask(case: GlyphCase) -> np.ndarray[tuple[int, int], np.dtype[np.bool]]:
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
    return mask


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_a_fixture_never_hashes_to_the_scan_it_recreates(case: GlyphCase) -> None:
    assert case.fixture_sha256 != case.source_sha256
    assert case.provenance == "synthetic_derivative"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_the_cut_reaches_the_reviewed_outcome(case: GlyphCase) -> None:
    cut = cut_word_glyphs(_mask(case), box=case.box, text=case.expected.visible_text)
    assert (cut.run_count, cut.position_count) == (
        case.expected.run_count,
        case.expected.position_count,
    )
    assert cut.separable == case.expected.separable
    assert cut.reason == case.expected.reason


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_the_cut_yields_the_reviewed_glyph_boxes(case: GlyphCase) -> None:
    cut = cut_word_glyphs(_mask(case), box=case.box, text=case.expected.visible_text)
    assert [(glyph.character, glyph.ordinal, glyph.box) for glyph in cut.glyphs] == [
        (glyph.character, glyph.ordinal, glyph.box) for glyph in case.expected.glyphs
    ]


def test_the_four_reviewed_outcomes_are_all_covered() -> None:
    outcomes = {
        (
            case.expected.separable,
            case.expected.run_count > case.expected.position_count,
            len(case.expected.glyphs) < case.expected.position_count,
        )
        for case in CASES
    }
    assert outcomes == {
        (True, False, False),
        (True, False, True),
        (False, False, True),
        (False, True, True),
    }


@pytest.mark.parametrize(
    "case", [case for case in CASES if case.expected.separable], ids=lambda case: case.case_id
)
def test_a_separable_case_labels_every_alphanumeric_and_no_mark(case: GlyphCase) -> None:
    expected = "".join(character for character in case.expected.visible_text if character.isalnum())
    assert "".join(glyph.character for glyph in case.expected.glyphs) == expected
    assert [glyph.ordinal for glyph in case.expected.glyphs] == [
        position
        for position, character in enumerate(case.expected.visible_text)
        if character.isalnum()
    ]


def test_a_word_carrying_a_mark_cuts_it_and_leaves_it_unlabelled() -> None:
    case = next(case for case in CASES if case.case_id == "punctuated-dont")
    assert case.expected.separable
    assert len(case.expected.glyphs) == case.expected.position_count - 1


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_a_cut_glyph_reads_back_against_the_lines_own_bands(case: GlyphCase) -> None:
    mask = _mask(case)
    cut = cut_word_glyphs(mask, box=case.box, text=case.expected.visible_text)
    band = LineBand(
        box=case.box,
        baseline_row_px=case.baseline_row_px,
        x_height_top_row_px=case.x_height_top_row_px,
    )
    for glyph in cut.glyphs:
        quality = measure_glyph_quality(mask, box=glyph.box, line=band)
        assert quality.ink_density > 0.0
        expects_ascender = glyph.character in "bdfhklt"
        assert ("ascends" in quality.flags) == expects_ascender
        assert ("descends" in quality.flags) == (glyph.character in "gpqy")


@pytest.mark.parametrize(
    "case", [case for case in CASES if case.atlas_path is not None], ids=lambda case: case.case_id
)
def test_the_committed_atlas_grid_still_renders_from_the_cut(case: GlyphCase) -> None:
    mask = _mask(case)
    cut = cut_word_glyphs(mask, box=case.box, text=case.expected.visible_text)
    grayscale = np.where(mask, np.uint8(0), np.uint8(255))
    crops = [
        grayscale[glyph.box[1] : glyph.box[3], glyph.box[0] : glyph.box[2]] for glyph in cut.glyphs
    ]
    cell_width, cell_height = cell_extent(crops)
    payload, _clipped = render_sheet(crops, cell_width=cell_width, cell_height=cell_height)
    assert case.atlas_path is not None
    with Image.open(_FIXTURE_ROOT / case.atlas_path) as committed:
        expected = np.asarray(committed.convert("L"), dtype=np.uint8)
    with Image.open(BytesIO(payload)) as rendered:
        assert np.array_equal(np.asarray(rendered.convert("L"), dtype=np.uint8), expected)
