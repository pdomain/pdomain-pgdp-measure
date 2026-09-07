"""Regenerate the reviewed glyph-cut fixtures, their manifest, and their atlas grids.

Run with `uv run python tests/fixtures/pgdp_glyphs/generate_fixtures.py`. Every bitmap is
self-authored: it copies no source pixels and no PGDP transcription. Each case recreates the
stroke and x-height geometry measured on a named real page, so the column-run cut meets the
shapes the corpus contains rather than round numbers.

Four cases. A separable word yields one glyph per letter. A word whose letters touch yields fewer
runs than characters and is refused. A word whose ink breaks inside a letter yields more runs than
characters and is refused just as firmly, because the direction of the disagreement does not make
the binding safe. And a word carrying an apostrophe that stands clear of its neighbours cuts, with
the mark counted as a position and left unlabelled: that is the case a letters-only count got
wrong, binding every character after the mark to the wrong ink.

The manifest records the real scan's `source_sha256` beside the fixture's own `fixture_sha256`.
The two must differ, which is the licence rule made checkable: a committed fixture that hashed to
a real scan would be a copy of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from pdomain_pgdp_measure.glyph_atlas import cell_extent, render_sheet

FIXTURE_ROOT = Path(__file__).parent
MARGIN_PX = 12
REVIEWED_ON = "2026-09-05"


@dataclass(frozen=True, slots=True)
class LetterSpec:
    """One drawn letter: its label, its bar count, and how far it sits from the last one."""

    character: str
    spacing_px: int
    bars: int = 1
    raised: bool = False
    """A mark that sits above the x-height band, the way an apostrophe does."""


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """One drawn word: the geometry to render and the provenance to record."""

    case_id: str
    project_id: str
    page_name: str
    source_sha256: str
    stroke_px: int
    x_height_px: int
    ascender_px: int
    descender_px: int
    letters: tuple[LetterSpec, ...]
    transform: str


CASES = (
    CaseSpec(
        case_id="separable-nose",
        project_id="projectID609bfa0449bdf",
        page_name="p002.png",
        source_sha256="ed3513095f6283ecb270797bf67a07c35644f35e1ad78694ebe5071ced776fb3",
        stroke_px=4,
        x_height_px=15,
        ascender_px=8,
        descender_px=6,
        letters=(
            LetterSpec("n", 0),
            LetterSpec("o", 3),
            LetterSpec("s", 3),
            LetterSpec("e", 3),
        ),
        transform=(
            "Self-authored PBM recreating the body geometry measured on "
            "projectID609bfa0449bdf/p002.png: 15 px x-height on a 4 px stroke, letters clear of "
            "one another by 3 px. It copies no source pixels and no PGDP transcription."
        ),
    ),
    CaseSpec(
        case_id="touching-shall",
        project_id="projectID657550412c8dc",
        page_name="p010.png",
        source_sha256="4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce",
        stroke_px=6,
        x_height_px=18,
        ascender_px=10,
        descender_px=7,
        letters=(
            LetterSpec("s", 0),
            LetterSpec("h", 4),
            LetterSpec("a", 4),
            LetterSpec("l", 0),
            LetterSpec("l", 4),
        ),
        transform=(
            "Self-authored PBM recreating the body geometry measured on "
            "projectID657550412c8dc/p010.png: 18 px x-height on a 6 px stroke, with one letter "
            "pair set solid so their ink joins. It copies no source pixels and no PGDP "
            "transcription."
        ),
    ),
    CaseSpec(
        case_id="broken-quiet",
        project_id="projectID64a479f51ce5b",
        page_name="p006.png",
        source_sha256="4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
        stroke_px=3,
        x_height_px=10,
        ascender_px=6,
        descender_px=5,
        letters=(
            LetterSpec("q", 0),
            LetterSpec("u", 3, bars=2),
            LetterSpec("i", 3),
            LetterSpec("e", 3),
            LetterSpec("t", 3),
        ),
        transform=(
            "Self-authored PBM recreating the reference-work geometry measured on "
            "projectID64a479f51ce5b/p006.png: 10 px x-height on a 3 px stroke, with one letter "
            "printed broken so its ink makes two runs. It copies no source pixels and no PGDP "
            "transcription."
        ),
    ),
    CaseSpec(
        case_id="punctuated-dont",
        project_id="projectID609bfa0449bdf",
        page_name="p014.png",
        source_sha256="ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d",
        stroke_px=4,
        x_height_px=15,
        ascender_px=8,
        descender_px=6,
        letters=(
            LetterSpec("d", 0),
            LetterSpec("o", 3),
            LetterSpec("n", 3),
            LetterSpec("\u2019", 3, raised=True),  # RIGHT SINGLE QUOTATION MARK
            LetterSpec("t", 3),
        ),
        transform=(
            "Self-authored PBM recreating the body geometry measured on "
            "projectID609bfa0449bdf/p014.png: 15 px x-height on a 4 px stroke, with an apostrophe "
            "standing clear of the letters on either side. It copies no source pixels and no PGDP "
            "transcription."
        ),
    ),
)

_ASCENDERS = frozenset("bdfhklt")
_DESCENDERS = frozenset("gpqy")
_BAR_GAP_PX = 3


def _letter_width(case: CaseSpec, letter: LetterSpec) -> int:
    return letter.bars * case.stroke_px + (letter.bars - 1) * _BAR_GAP_PX


def _draw(case: CaseSpec) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.bool]], int, int]:
    width = MARGIN_PX * 2 + sum(
        _letter_width(case, letter) + letter.spacing_px for letter in case.letters
    )
    baseline_row = MARGIN_PX + case.ascender_px + case.x_height_px
    height = baseline_row + case.descender_px + MARGIN_PX
    mask = np.zeros((height, width), dtype=np.bool_)
    x = MARGIN_PX
    for letter in case.letters:
        x += letter.spacing_px
        top = baseline_row - case.x_height_px
        bottom = baseline_row
        if letter.character in _ASCENDERS:
            top -= case.ascender_px
        if letter.character in _DESCENDERS:
            bottom += case.descender_px
        if letter.raised:
            top -= case.ascender_px
            bottom = top + max(2, case.ascender_px // 2)
        for bar in range(letter.bars):
            left = x + bar * (case.stroke_px + _BAR_GAP_PX)
            mask[top:bottom, left : left + case.stroke_px] = True
        x += _letter_width(case, letter)
    return mask, baseline_row, baseline_row - case.x_height_px


def _write_bitmap(path: Path, mask: np.ndarray[tuple[int, int], np.dtype[np.bool]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.fromarray(~mask) as image, image.convert("1") as bitmap:
        bitmap.save(path, format="PPM")
    return sha256(path.read_bytes()).hexdigest()


def _expected_cut(
    case: CaseSpec, mask: np.ndarray[tuple[int, int], np.dtype[np.bool]]
) -> dict[str, object]:
    from pdomain_pgdp_measure.glyph_cut import cut_word_glyphs

    box = (0, 0, mask.shape[1], mask.shape[0])
    text = "".join(letter.character for letter in case.letters)
    cut = cut_word_glyphs(mask, box=box, text=text)
    return {
        "visible_text": text,
        "run_count": cut.run_count,
        "position_count": cut.position_count,
        "separable": cut.separable,
        "reason": cut.reason,
        "glyphs": [
            {"character": glyph.character, "ordinal": glyph.ordinal, "box": list(glyph.box)}
            for glyph in cut.glyphs
        ],
    }


def _write_atlas(
    path: Path, case: CaseSpec, mask: np.ndarray[tuple[int, int], np.dtype[np.bool]]
) -> str | None:
    from pdomain_pgdp_measure.glyph_cut import cut_word_glyphs

    box = (0, 0, mask.shape[1], mask.shape[0])
    text = "".join(letter.character for letter in case.letters)
    cut = cut_word_glyphs(mask, box=box, text=text)
    if not cut.separable:
        return None
    grayscale = np.where(mask, np.uint8(0), np.uint8(255))
    crops = [
        grayscale[glyph.box[1] : glyph.box[3], glyph.box[0] : glyph.box[2]] for glyph in cut.glyphs
    ]
    cell_width, cell_height = cell_extent(crops)
    payload, _clipped = render_sheet(crops, cell_width=cell_width, cell_height=cell_height)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(payload)
    return sha256(payload).hexdigest()


def main() -> None:
    cases: list[dict[str, object]] = []
    for case in CASES:
        mask, baseline_row, x_height_top_row = _draw(case)
        image_path = f"words/{case.case_id}.pbm"
        atlas_path = f"atlas/{case.case_id}.png"
        fixture_sha256 = _write_bitmap(FIXTURE_ROOT / image_path, mask)
        atlas_sha256 = _write_atlas(FIXTURE_ROOT / atlas_path, case, mask)
        cases.append(
            {
                "case_id": case.case_id,
                "project_id": case.project_id,
                "page_name": case.page_name,
                "provenance": "synthetic_derivative",
                "review_method": "model-reviewed",
                "reviewed_on": REVIEWED_ON,
                "transform": case.transform,
                "source_sha256": case.source_sha256,
                "fixture_sha256": fixture_sha256,
                "image_path": image_path,
                "atlas_path": None if atlas_sha256 is None else atlas_path,
                "atlas_sha256": atlas_sha256,
                "box": [0, 0, mask.shape[1], mask.shape[0]],
                "baseline_row_px": baseline_row,
                "x_height_top_row_px": x_height_top_row,
                "stroke_px": case.stroke_px,
                "x_height_px": case.x_height_px,
                "expected": _expected_cut(case, mask),
            }
        )
    (FIXTURE_ROOT / "manifest.json").write_text(
        json.dumps({"cases": cases}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
