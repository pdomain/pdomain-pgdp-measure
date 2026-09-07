"""Select the recognized words a page carries outside every matched line box.

Proofers strip running heads and page numbers from F2 entirely, so furniture ink has no human
label anywhere in this corpus. What it does have is a `geometry-v1` record: `pdomain-source-data`
already ran the project's own fine-tuned DocTR checkpoints over every page and wrote one word box
and confidence per read. This module picks the words that fall outside the lines alignment
matched, which is where the running head and the folio sit, and hands them on as the `recognized`
label tier. Nothing here runs OCR or opens a model.

The page's `image_sha256` is checked against the alignment's `scan_sha256` before any box is read,
exactly as the typography witness checks it. Two records that name the same page but not the same
bytes would place boxes on ink that is not there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pdomain_pgdp_measure.ocr_witness import PageWitness

Bounds = tuple[int, int, int, int]

FURNITURE_METHODS: dict[str, float | int | str] = {
    "algorithm": "words-outside-matched-lines/v1",
    "matched_line_overlap_ratio_minimum": 0.5,
}

_MATCHED_LINE_OVERLAP_RATIO_MINIMUM: Final = 0.5
"""How much of a recognized word's own area must fall inside a matched line box to belong to it.

Half, rather than any overlap at all, because a recognized word box carries its own ascender and
descender while a matched line box is tight to the line's ink, so a word on one line routinely
nicks the box of the line above. Requiring the majority of the word's area keeps that nick from
claiming the word, and keeps a running head that happens to sit a pixel inside a first-line box
from being silently dropped.
"""


class FurnitureScanMismatchError(ValueError):
    """The geometry record's page and the alignment's page are not the same scan."""


@dataclass(frozen=True, slots=True)
class FurnitureWord:
    """One recognized word outside every matched line, with the read that labels it."""

    text: str
    box: Bounds
    confidence: float | None
    line_index: int
    word_index: int

    def __post_init__(self) -> None:
        x_start, y_start, x_end, y_end = self.box
        if x_end <= x_start or y_end <= y_start:
            raise ValueError("A furniture word box must have positive width and height.")
        if self.line_index < 0 or self.word_index < 0:
            raise ValueError("A furniture word's ordinals must be nonnegative.")


def witness_matches_scan(page: PageWitness, *, scan_sha256: str) -> bool:
    """Whether this geometry page names the same scan bytes the alignment recorded."""

    return page.image_sha256 == scan_sha256


def page_furniture_words(
    page: PageWitness,
    *,
    scan_sha256: str,
    matched_line_boxes: Sequence[Bounds],
) -> tuple[FurnitureWord, ...]:
    """Every recognized word on this page that no matched line box claims.

    Words come back in the recognizer's own line and word order, so two runs over one record
    produce the same sequence.
    """

    if not witness_matches_scan(page, scan_sha256=scan_sha256):
        wrong_scan = f"is for scan {page.image_sha256}, not the aligned {scan_sha256}"
        raise FurnitureScanMismatchError(f"Geometry page {page.page_name!r} {wrong_scan}.")
    boxes = tuple(matched_line_boxes)
    selected = [
        FurnitureWord(
            text=word.text,
            box=word.box,
            confidence=word.confidence,
            line_index=word.line_index,
            word_index=word.word_index,
        )
        for word in page.words
        if not _claimed_by_a_matched_line(word.box, boxes)
    ]
    selected.sort(key=lambda word: (word.line_index, word.word_index, word.box))
    return tuple(selected)


def _claimed_by_a_matched_line(box: Bounds, matched_line_boxes: Sequence[Bounds]) -> bool:
    x_start, y_start, x_end, y_end = box
    area = (x_end - x_start) * (y_end - y_start)
    if area <= 0:
        return True
    return any(
        _intersection_area(box, line_box) >= _MATCHED_LINE_OVERLAP_RATIO_MINIMUM * area
        for line_box in matched_line_boxes
    )


def _intersection_area(box: Bounds, other: Bounds) -> int:
    x_start, y_start, x_end, y_end = box
    other_x_start, other_y_start, other_x_end, other_y_end = other
    width = min(x_end, other_x_end) - max(x_start, other_x_start)
    height = min(y_end, other_y_end) - max(y_start, other_y_start)
    if width <= 0 or height <= 0:
        return 0
    return width * height


LINE_ADMISSION_METHODS: dict[str, float | int | str] = {
    "algorithm": "recognizer-line-agreement/v1",
    "line_word_agreement_minimum": 0.8,
    "word_overlap_ratio_minimum": 0.5,
}

_LINE_WORD_AGREEMENT_MINIMUM: Final = 0.8
"""How much of a line's transcription the recognizer must also read there to admit the line.

Page acceptance is a page-level judgement and a poor proxy for whether one line bound to the right
text. Measured over three books, lines on accepted pages score 0.926 to 0.976 at this bar, and
lines excluded only for alignment score do about as well: `normalized_cost_exceeds_maximum` runs
0.937 to 0.962 and `uniqueness_margin_below_minimum` 0.887 to 0.934. Lines on illustration pages
do not, at 0.560 to 0.683.

**Accepted lines are checked too.** A line whose candidate box was matched to the wrong source
line still reconciles by word count often enough to cut, and every glyph in it then carries the
wrong character: `They` cut from WILL, `and` from `side,`, `being` from `morning,`. Those lines
score 0.00 to 0.17 here while the lines around them score above 0.8, so one check finds them.
It costs 2.4 to 7.4 percent of accepted lines.

The recognizer only ever rejects here. The label stays the transcription's.
"""


def line_word_agreement(page: PageWitness, *, box: Bounds, visible_text: str) -> float | None:
    """What share of a line's transcription words the recognizer also read inside its box.

    `None` when the line carries no word to check, which is no evidence rather than disagreement.
    """

    words = visible_text.split()
    if not words:
        return None
    read = {
        _normalized(word.text)
        for word in page.words
        if _claimed_by_a_matched_line(word.box, (box,))
    }
    read.discard("")
    return sum(1 for word in words if _normalized(word) in read) / len(words)


def line_is_admissible(agreement: float | None) -> bool:
    """Whether the recognizer agrees enough to admit a line its page was not accepted for."""

    return agreement is not None and agreement >= _LINE_WORD_AGREEMENT_MINIMUM


def _normalized(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())
