"""Cut one word's box into per-glyph boxes by the blank columns between its letters.

A word box drawn by `typography_measure.measure_line_words` is tight to one word's ink, and its
label is the transcription word alignment matched to that line. Splitting the box at every column
carrying no ink gives one run per printed character whenever they do not touch, so the run count
and the word's own character count can be compared. They agree or they do not, and this refuses to
guess: binding four runs to five characters in order would misattribute a box to the wrong
character and poison every page later rendered from it.

**Every printing character counts, and only the alphanumerics are labelled.** A comma or an
apostrophe makes its own ink run as readily as a letter does, so counting letters alone lets a
word match by coincidence: one mark standing alone adds a run while one touching letter pair
removes one, and the two cancel. Measured over four books on 40 pages each, that coincidence
accounted for 1.4 to 15.4 percent of the words a letters-only rule called separable, and in not
one of them did the ink actually carry one run per printed character. Those words shifted every
label in them by a position. Counting all printing characters removes them and, in three of the
four books, raises the yield as well, because a word whose punctuation stands cleanly apart now
cuts instead of being refused.

Measured over the five aligned books, the counts agree for roughly a quarter to four fifths of
reconciled words. The rest fail because characters touch, which is a property of the printing
rather than a defect here, so a rejected word records the two counts and the reason rather than
disappearing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Bounds = tuple[int, int, int, int]
CutRejectReason = Literal["no_labelled_letters", "no_ink", "run_count_disagrees"]

GLYPH_CUT_METHODS: dict[str, float | int | str] = {
    "algorithm": "blank-column-runs/v2",
    "counted_character_class": "printing",
    "labelled_character_class": "alphanumeric",
}


@dataclass(frozen=True, slots=True)
class GlyphCut:
    """One glyph's character and its half-open box in the `source` frame.

    `ordinal` is the character's position among the word's printing characters, so a word whose
    second character is a quotation mark emits its first letter at 0 and its next at 2. The gap is
    the evidence that a mark was cut and deliberately left unlabelled.
    """

    character: str
    ordinal: int
    box: Bounds

    def __post_init__(self) -> None:
        if len(self.character) != 1:
            raise ValueError("A glyph carries exactly one character.")
        if self.ordinal < 0:
            raise ValueError("ordinal must be nonnegative.")
        x_start, y_start, x_end, y_end = self.box
        if x_end <= x_start or y_end <= y_start:
            raise ValueError("A glyph box must have positive width and height.")


@dataclass(frozen=True, slots=True)
class WordCut:
    """What one word's box yielded, or why it yielded nothing.

    `glyphs` is nonempty exactly when `reason` is `None`, so a caller cannot read boxes out of a
    word this refused to cut.
    """

    run_count: int
    position_count: int
    glyphs: tuple[GlyphCut, ...] = ()
    reason: CutRejectReason | None = None

    def __post_init__(self) -> None:
        if self.run_count < 0 or self.position_count < 0:
            raise ValueError("A cut's counts must be nonnegative.")
        object.__setattr__(self, "glyphs", tuple(self.glyphs))
        if (self.reason is None) != bool(self.glyphs):
            raise ValueError("A cut carries glyphs exactly when it names no reason.")
        if self.glyphs and self.run_count != self.position_count:
            raise ValueError("A separable word has one ink run per printing character.")
        if len(self.glyphs) > self.position_count:
            raise ValueError("A word cannot emit more glyphs than it has printing characters.")

    @property
    def separable(self) -> bool:
        return self.reason is None


def label_positions(text: str) -> str:
    """Every printing character of a word, in order, whitespace dropped.

    These are the positions the ink runs are counted against. A mark is counted because it prints;
    it is not labelled because a per-character inventory of a book's type is about its letters and
    figures, and a comma cut from a word box is not a sample of anything a later stage will set.
    """

    return "".join(character for character in text if not character.isspace())


def label_letters(text: str) -> str:
    """The characters of a word this labels: its alphanumerics, in order."""

    return "".join(character for character in text if character.isalnum())


def cut_word_glyphs(
    mask: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    *,
    box: Bounds,
    text: str,
) -> WordCut:
    """Split one word's box at its blank columns and bind the runs to its characters in order.

    `mask` is the full-page ink mask `alignment_image.build_candidate_mask` builds and `box` is a
    half-open box in that same frame. Each glyph's columns come from the run; its rows are
    measured fresh from the run's own projection, so the box is tight to that glyph's ink. A run
    that lands on a punctuation position is cut and discarded, not labelled.
    """

    positions = label_positions(text)
    x_start, y_start, x_end, y_end = box
    runs = ink_column_runs(mask[y_start:y_end, x_start:x_end])
    if not label_letters(text):
        return WordCut(
            run_count=len(runs), position_count=len(positions), reason="no_labelled_letters"
        )
    if not runs:
        return WordCut(run_count=0, position_count=len(positions), reason="no_ink")
    if len(runs) != len(positions):
        return WordCut(
            run_count=len(runs), position_count=len(positions), reason="run_count_disagrees"
        )
    glyphs = tuple(
        GlyphCut(
            character=character,
            ordinal=ordinal,
            box=_run_box(mask, box=box, run=run),
        )
        for ordinal, (character, run) in enumerate(zip(positions, runs, strict=True))
        if character.isalnum()
    )
    return WordCut(run_count=len(runs), position_count=len(positions), glyphs=glyphs)


def ink_column_runs(
    cropped: np.ndarray[tuple[int, int], np.dtype[np.bool]],
) -> tuple[tuple[int, int], ...]:
    """Every maximal run of columns carrying ink, as half-open box-local column ranges."""

    if cropped.size == 0:
        return ()
    padded = np.zeros(cropped.shape[1] + 2, dtype=np.bool_)
    padded[1:-1] = cropped.any(axis=0)
    interior = padded[1:-1]
    starts = np.flatnonzero(interior & ~padded[:-2])
    ends = np.flatnonzero(interior & ~padded[2:])
    return tuple((int(start), int(end) + 1) for start, end in zip(starts, ends, strict=True))


def _run_box(
    mask: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    *,
    box: Bounds,
    run: tuple[int, int],
) -> Bounds:
    x_start, y_start, _, y_end = box
    run_start, run_end = run
    glyph_x_start = x_start + run_start
    glyph_x_end = x_start + run_end
    ink_rows = np.flatnonzero(mask[y_start:y_end, glyph_x_start:glyph_x_end].any(axis=1))
    row_top = int(np.min(ink_rows))
    row_bottom = int(np.max(ink_rows))
    return glyph_x_start, y_start + row_top, glyph_x_end, y_start + row_bottom + 1


def crop_to_ink(
    mask: np.ndarray[tuple[int, int], np.dtype[np.bool]], *, box: Bounds
) -> Bounds | None:
    """The tightest box inside `box` that still holds all its ink, or `None` when there is none.

    A word box measured from a matched line is already tight. A recognized word's box is not: the
    recognizer draws a generous rectangle, so the furniture tier tightens it before cutting.
    """

    x_start, y_start, x_end, y_end = box
    if x_end <= x_start or y_end <= y_start:
        return None
    cropped = mask[y_start:y_end, x_start:x_end]
    ink_columns = np.flatnonzero(cropped.any(axis=0))
    ink_rows = np.flatnonzero(cropped.any(axis=1))
    if ink_columns.size == 0 or ink_rows.size == 0:
        return None
    return (
        x_start + int(np.min(ink_columns)),
        y_start + int(np.min(ink_rows)),
        x_start + int(np.max(ink_columns)) + 1,
        y_start + int(np.max(ink_rows)) + 1,
    )
