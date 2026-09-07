"""Regenerate the reviewed typography line fixtures and their manifest.

Run with `uv run python tests/fixtures/pgdp_typography/generate_fixtures.py`. Every bitmap is
self-authored: it copies no source pixels and no PGDP transcription. Each case recreates the
line geometry measured on the named real page -- x-height, stroke width, ink height, and word
count -- so the estimator is exercised against the shapes the corpus actually contains rather
than against round numbers.

The manifest records the real scan's `source_sha256` beside the fixture's own
`fixture_sha256`. The two must differ, which is the licence rule made checkable: a committed
fixture that hashed to a real scan would be a copy of it. It also records the
`gap_threshold_px` each case was measured at, because under `ink-profile-word-runs/v2` that
value comes from the book's own gap distribution and cannot be recovered from the drawing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from PIL import Image

FIXTURE_ROOT = Path(__file__).parent
MARGIN_PX = 12
LETTER_SPACING_PX = 2
BOOK_GAP_THRESHOLD_PX = {
    "projectID609bfa0449bdf": 10,
    "projectID64a479f51ce5b": 6,
    "projectID603d7d5e04ca0": 8,
    "projectID657550412c8dc": 15,
}
"""Each book's own measured word-gap threshold under ``ink-profile-word-runs/v2``.

Derived by Otsu over the whole book's pooled gap-length histogram in the ``alignment-t2``
reports, the same value the shipped pipeline computes. A case is measured at its own book's
threshold so the fixture meets the rule the corpus run actually applies to that book, not a
round number.
"""


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """One drawn line: the geometry to render and the provenance to record."""

    case_id: str
    project_id: str
    page_name: str
    source_path: str
    x_height_px: int
    stroke_width_px: int
    ascender_px: int
    descender_px: int
    line_width_px: int
    word_count: int
    gap_px: int
    skew_degrees: float
    source_word_count: int | None
    note: str


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        case_id="alpha-body",
        project_id="projectID609bfa0449bdf",
        page_name="p002.png",
        source_path="projectID609bfa0449bdf/p002.png",
        x_height_px=15,
        stroke_width_px=4,
        ascender_px=9,
        descender_px=8,
        line_width_px=782,
        word_count=9,
        gap_px=10,
        skew_degrees=0.0,
        source_word_count=None,
        note="body line",
    ),
    CaseSpec(
        case_id="alpha-short",
        project_id="projectID609bfa0449bdf",
        page_name="p004.png",
        source_path="projectID609bfa0449bdf/p004.png",
        x_height_px=15,
        stroke_width_px=4,
        ascender_px=9,
        descender_px=8,
        line_width_px=420,
        word_count=5,
        gap_px=10,
        skew_degrees=0.0,
        source_word_count=None,
        note="paragraph-final line below the three-segment floor",
    ),
    CaseSpec(
        case_id="beta-small",
        project_id="projectID64a479f51ce5b",
        page_name="f005.png",
        source_path="projectID64a479f51ce5b/f005.png",
        x_height_px=10,
        stroke_width_px=3,
        ascender_px=7,
        descender_px=6,
        line_width_px=854,
        word_count=12,
        gap_px=6,
        skew_degrees=0.0,
        source_word_count=None,
        note="small body type",
    ),
    CaseSpec(
        case_id="beta-skewed",
        project_id="projectID64a479f51ce5b",
        page_name="f007.png",
        source_path="projectID64a479f51ce5b/f007.png",
        x_height_px=10,
        stroke_width_px=3,
        ascender_px=7,
        descender_px=6,
        line_width_px=854,
        word_count=12,
        gap_px=6,
        skew_degrees=0.5,
        source_word_count=None,
        note="small body type on a half-degree skew",
    ),
    CaseSpec(
        case_id="gamma-heavy",
        project_id="projectID603d7d5e04ca0",
        page_name="053.png",
        source_path="projectID603d7d5e04ca0/053.png",
        x_height_px=17,
        stroke_width_px=6,
        ascender_px=12,
        descender_px=10,
        line_width_px=900,
        word_count=7,
        gap_px=12,
        skew_degrees=0.0,
        source_word_count=None,
        note="heavy inked body line",
    ),
    CaseSpec(
        case_id="gamma-oversplit",
        project_id="projectID603d7d5e04ca0",
        page_name="054.png",
        source_path="projectID603d7d5e04ca0/054.png",
        x_height_px=17,
        stroke_width_px=6,
        ascender_px=12,
        descender_px=10,
        line_width_px=900,
        word_count=9,
        gap_px=12,
        skew_degrees=0.0,
        source_word_count=7,
        note="ink runs outnumber the transcription's words",
    ),
    CaseSpec(
        case_id="delta-wide-gaps",
        project_id="projectID657550412c8dc",
        page_name="p0400.png",
        source_path="projectID657550412c8dc/p0400.png",
        x_height_px=18,
        stroke_width_px=4,
        ascender_px=11,
        descender_px=9,
        line_width_px=852,
        word_count=8,
        gap_px=20,
        skew_degrees=0.0,
        source_word_count=None,
        note="widely spaced setting",
    ),
    CaseSpec(
        case_id="delta-skew-backstop",
        project_id="projectID657550412c8dc",
        page_name="p0420.png",
        source_path="projectID657550412c8dc/p0420.png",
        x_height_px=18,
        stroke_width_px=4,
        ascender_px=11,
        descender_px=9,
        line_width_px=852,
        word_count=8,
        gap_px=20,
        skew_degrees=2.0,
        source_word_count=None,
        note="past the skew backstop",
    ),
)


@dataclass(frozen=True, slots=True)
class DrawnLine:
    """A rendered fixture and the exact geometry it was drawn from."""

    image: Image.Image
    box: tuple[int, int, int, int]
    baseline_row_px: int
    x_height_top_row_px: int
    word_run_count: int
    drawn_width_px: int


def draw(spec: CaseSpec) -> DrawnLine:
    """Draw one line of vertical strokes with a known baseline, x-height, and word gaps.

    Letters sit two pixels apart and words `gap_px` apart. That separation matters: the word
    detector treats a blank run of the book's own gap threshold as a word break, so a letter
    spacing at or above that threshold would split every letter into its own word.
    """

    pitch = spec.stroke_width_px + LETTER_SPACING_PX
    block_target = (spec.line_width_px - (spec.word_count - 1) * spec.gap_px) / spec.word_count
    letters = max(2, round((block_target + LETTER_SPACING_PX) / pitch))
    block_width = letters * pitch - LETTER_SPACING_PX
    width = spec.word_count * block_width + (spec.word_count - 1) * spec.gap_px

    slope = math.tan(math.radians(spec.skew_degrees))
    height = spec.ascender_px + spec.x_height_px + 1 + spec.descender_px
    shift = math.ceil(abs(slope) * width / 2) + 1
    image = Image.new("1", (width + 2 * MARGIN_PX, height + 2 * MARGIN_PX + 2 * shift), 1)

    top = MARGIN_PX + shift
    x_height_top_row = top + spec.ascender_px
    baseline_row = x_height_top_row + spec.x_height_px
    center_x = MARGIN_PX + width / 2
    x = MARGIN_PX
    index = 0
    for word in range(spec.word_count):
        for letter in range(letters):
            offset = round(slope * (x - center_x))
            ascender = index % 3 == 0
            descender = index % 4 == 3
            y_start = (top if ascender else x_height_top_row) + offset
            y_end = (top + height if descender else baseline_row + 1) + offset
            for column in range(x, x + spec.stroke_width_px):
                for row in range(y_start, y_end):
                    image.putpixel((column, row), 0)
            x += pitch if letter < letters - 1 else spec.stroke_width_px
            index += 1
        if word < spec.word_count - 1:
            x += spec.gap_px

    box = (MARGIN_PX, MARGIN_PX, MARGIN_PX + width, image.height - MARGIN_PX)
    gap_threshold = BOOK_GAP_THRESHOLD_PX[spec.project_id]
    if gap_threshold <= LETTER_SPACING_PX:
        raise ValueError(f"Letter spacing would split every letter: {spec.case_id}")
    word_run_count = spec.word_count if spec.gap_px >= gap_threshold else 1
    return DrawnLine(
        image=image,
        box=box,
        baseline_row_px=baseline_row,
        x_height_top_row_px=x_height_top_row,
        word_run_count=word_run_count,
        drawn_width_px=width,
    )


def main() -> None:
    lines_directory = FIXTURE_ROOT / "lines"
    lines_directory.mkdir(parents=True, exist_ok=True)
    corpus_root = Path("/workspaces/pdomain-data/pgdp-corpus")
    previous = _previous_source_hashes()
    cases: list[dict[str, object]] = []
    for spec in CASES:
        drawn = draw(spec)
        image_path = lines_directory / f"{spec.case_id}.pbm"
        drawn.image.save(image_path)
        scan = corpus_root / spec.source_path
        source_sha256 = (
            sha256(scan.read_bytes()).hexdigest() if scan.is_file() else previous[spec.case_id]
        )
        source_word_count = (
            drawn.word_run_count if spec.source_word_count is None else spec.source_word_count
        )
        cases.append(
            {
                "case_id": spec.case_id,
                "project_id": spec.project_id,
                "page_name": spec.page_name,
                "image_path": f"lines/{spec.case_id}.pbm",
                "source_sha256": source_sha256,
                "fixture_sha256": sha256(image_path.read_bytes()).hexdigest(),
                "provenance": "synthetic_derivative",
                "transform": (
                    f"Self-authored PBM recreates the {spec.note} geometry measured on "
                    f"{spec.project_id}/{spec.page_name}: x-height {spec.x_height_px} px, "
                    f"stroke {spec.stroke_width_px} px, drawn {drawn.drawn_width_px} px wide. "
                    "It copies no source pixels and no PGDP transcription."
                ),
                "box": list(drawn.box),
                "gap_threshold_px": BOOK_GAP_THRESHOLD_PX[spec.project_id],
                "visible_text": " ".join(f"w{index}" for index in range(source_word_count)),
                "expected": {
                    "measured": abs(spec.skew_degrees) <= 1.0 and drawn.drawn_width_px >= 600,
                    "baseline_row_px": drawn.baseline_row_px,
                    "x_height_px": spec.x_height_px,
                    "x_height_top_row_px": drawn.x_height_top_row_px,
                    "stroke_width_px": spec.stroke_width_px,
                    "skew_slope": round(math.tan(math.radians(spec.skew_degrees)), 6),
                    "word_run_count": drawn.word_run_count,
                    "source_word_count": source_word_count,
                    "words_reconciled": drawn.word_run_count == source_word_count,
                },
                "reviewed_on": "2026-09-04",
                "review_method": "model-reviewed",
            }
        )
    manifest = {
        "schema_version": 1,
        "licence_rule": (
            "Real scan crops may be committed only where the source licence permits. None are: "
            "every case here is a byte-stable synthetic derivative, recorded as "
            "provenance 'synthetic_derivative', and its fixture_sha256 differs from the "
            "source_sha256 of the real page whose geometry it recreates."
        ),
        "cases": cases,
    }
    _ = (FIXTURE_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _previous_source_hashes() -> dict[str, str]:
    """Keep recorded source hashes when the corpus is not mounted on this machine."""

    manifest_path = FIXTURE_ROOT / "manifest.json"
    if not manifest_path.is_file():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    return {str(case["case_id"]): str(case["source_sha256"]) for case in cases}


if __name__ == "__main__":
    main()
