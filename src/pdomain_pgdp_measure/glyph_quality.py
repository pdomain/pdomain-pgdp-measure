"""Record what one cut glyph looks like against its line's measured bands.

Three things are recorded and none is enforced: where the glyph's rows sit against the line's
x-height top and baseline, whether the glyph reaches the line box's own top or bottom row, and how
much of its box is ink. A consumer picking exemplars for synthesis wants all three; a gate that
dropped glyphs on any of them would hide the shape of what the corpus actually yields.

The reference is the **line** box, never the word box. A word box is tight to that word's own ink,
so its tallest glyph always touches its top row and its lowest always touches its bottom row. An
earlier check made against the word box read 0.69 to 0.84 of glyphs as clipped across the five
books, which measured nothing but that definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

Bounds = tuple[int, int, int, int]
GlyphQualityFlag = Literal[
    "ascends",
    "flat_ascender",
    "narrow",
    "overtall",
    "unlike_character",
    "wide",
    "descends",
    "short_of_baseline",
    "sparse_ink",
    "touches_line_bottom",
    "touches_line_top",
]

DEFECT_FLAGS: Final = frozenset({"flat_ascender", "narrow", "overtall", "unlike_character", "wide"})
"""The flags that say a glyph may be wrong, as against the ones that merely describe it.

`ascends`, `descends` and the two `touches_line_*` flags are ordinary facts about a letter: every
`h` ascends and every `g` descends. Treating them as defects excludes every ascender and descender
from its own character's reference sample, which is how the shape check first shipped with
references for 11 characters instead of 35.
"""

GLYPH_QUALITY_METHODS: dict[str, float | int | str] = {
    "algorithm": "line-band-agreement/v2",
    "row_extent_tolerance_px": 1,
    "sparse_ink_density_maximum": 0.15,
    "ascender_flatness_maximum": 1.10,
    "overtall_ratio_minimum": 1.35,
    "overtall_line_sample_minimum": 4,
    "narrow_ratio_maximum": 0.6,
    "wide_ratio_minimum": 1.5,
    "narrow_character_sample_minimum": 30,
}

_ROW_EXTENT_TOLERANCE_PX: Final = 1
"""How far a glyph's rows may sit outside a band before the band flag is raised.

One pixel, because the line's baseline and x-height top are medians over column segments and a
round-half-up of a median lands on either side of a true edge by up to a pixel on its own.
"""

_SPARSE_INK_DENSITY_MAXIMUM: Final = 0.15
"""Below this share of ink in its own box, a cut is more likely a speck than a letter."""

ASCENDER_CHARACTERS: Final = frozenset("bdfhklt")
X_HEIGHT_CHARACTERS: Final = frozenset("acemnorsuvwxz")
_ASCENDER_FLATNESS_MAXIMUM: Final = 1.10
"""How short a word's tallest ascender may run, against its own x-height letters, before the
word's ink stops matching what its label claims.

In ordinary lowercase an ascender runs about half again the x-height: measured over the five
books, roman words score 1.46 to 1.58 at the median. A word set in small capitals scores 1.00,
because every capital is the same height, and so does a word bound to the wrong ink. Calibrated
against the 76 words PGDP itself marks `<sc>` in projectID603d7d5e04ca0, a cut at 1.10 catches 74
of them while flagging 1.8 percent of roman words there and 0.27 percent in
projectID609bfa0449bdf.

This is an observation, not a verdict. The flag says the ink is flatter than the label predicts;
it does not say why, and the causes seen so far are unmarked small capitals, a word bound to the
wrong ink, and a badly cut glyph box.
"""


@dataclass(frozen=True, slots=True)
class LineBand:
    """The line measurements a glyph's rows are read against, in the `source` frame."""

    box: Bounds
    baseline_row_px: int
    x_height_top_row_px: int

    def __post_init__(self) -> None:
        _, y_start, _, y_end = self.box
        if y_end <= y_start:
            raise ValueError("A line box must have positive height.")
        if self.x_height_top_row_px > self.baseline_row_px:
            raise ValueError("A line's x-height top must sit at or above its baseline.")


@dataclass(frozen=True, slots=True)
class GlyphQuality:
    """One glyph's recorded quality, with the two offsets the band flags were read from.

    `top_offset_px` is the glyph's first ink row minus the line's x-height top row, so it is
    negative for an ascender. `bottom_offset_px` is the glyph's last ink row minus the baseline
    row, so it is positive for a descender. Both are `None` for a glyph with no line measurement,
    which is every glyph on the `recognized` tier.
    """

    ink_density: float
    row_extent_px: int
    top_offset_px: int | None = None
    bottom_offset_px: int | None = None
    flags: tuple[GlyphQualityFlag, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.ink_density <= 1.0:
            raise ValueError("ink_density must be a share between zero and one.")
        if self.row_extent_px <= 0:
            raise ValueError("row_extent_px must be positive.")
        object.__setattr__(self, "flags", tuple(sorted(set(self.flags))))
        if (self.top_offset_px is None) != (self.bottom_offset_px is None):
            raise ValueError("A glyph's two band offsets are recorded together or not at all.")


def measure_glyph_quality(
    mask: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    *,
    box: Bounds,
    line: LineBand | None,
) -> GlyphQuality:
    """Measure one glyph box against the page ink mask and its line's bands.

    `line` is `None` for a glyph cut from a recognized word, which sits outside every matched line
    and so has no measured baseline to be read against.
    """

    x_start, y_start, x_end, y_end = box
    if x_end <= x_start or y_end <= y_start:
        raise ValueError("A glyph box must have positive width and height.")
    cropped = mask[y_start:y_end, x_start:x_end]
    ink_pixels = int(np.count_nonzero(cropped))
    density = ink_pixels / ((x_end - x_start) * (y_end - y_start))
    flags: list[GlyphQualityFlag] = []
    if density < _SPARSE_INK_DENSITY_MAXIMUM:
        flags.append("sparse_ink")
    if line is None:
        return GlyphQuality(ink_density=density, row_extent_px=y_end - y_start, flags=tuple(flags))

    _, line_y_start, _, line_y_end = line.box
    top_offset = y_start - line.x_height_top_row_px
    bottom_offset = (y_end - 1) - line.baseline_row_px
    if top_offset < -_ROW_EXTENT_TOLERANCE_PX:
        flags.append("ascends")
    if bottom_offset > _ROW_EXTENT_TOLERANCE_PX:
        flags.append("descends")
    if bottom_offset < -_ROW_EXTENT_TOLERANCE_PX:
        flags.append("short_of_baseline")
    if y_start <= line_y_start:
        flags.append("touches_line_top")
    if y_end >= line_y_end:
        flags.append("touches_line_bottom")
    return GlyphQuality(
        ink_density=density,
        row_extent_px=y_end - y_start,
        top_offset_px=top_offset,
        bottom_offset_px=bottom_offset,
        flags=tuple(flags),
    )


def word_ascender_flatness(heights_by_character: Sequence[tuple[str, int]]) -> float | None:
    """How tall one word's ascenders run against its own x-height letters, or `None`.

    `None` when the word carries no ascender letter or no x-height letter, which is most short
    words and every word this cannot say anything about.
    """

    ascenders = [
        height for character, height in heights_by_character if character in ASCENDER_CHARACTERS
    ]
    x_heights = [
        height for character, height in heights_by_character if character in X_HEIGHT_CHARACTERS
    ]
    if not ascenders or not x_heights:
        return None
    reference = median(x_heights)
    if reference <= 0:
        return None
    return max(ascenders) / reference


def ascender_is_flat(flatness: float | None) -> bool:
    """Whether a word's ink is flatter than its label predicts."""

    return flatness is not None and flatness < _ASCENDER_FLATNESS_MAXIMUM


_OVERTALL_RATIO_MINIMUM: Final = 1.35
"""How much taller than its line's own x-height letters a glyph may run before it is flagged.

The reference is the median height of the line's own x-height-only letters, not the line's fitted
x-height. Fitting was tried first and fails: normalised by the fitted value the tail runs to 5.33
and 13 of 16 outliers are correctly labelled glyphs on lines whose x-height estimate was wrong, so
it measures the estimator rather than the glyph. Against the line's own ink the same books read a
median of 1.00 and a 95th percentile of 1.02 to 1.08, and 12 of 16 outliers are genuine label
errors: a box holding several letters, or a capital where the label says lowercase.

At 1.35 this flags 0.03 to 1.2 percent of x-height glyphs, most in the book whose labels measure
worst. It cannot see small capitals, whose whole point is to sit at x-height.
"""

_OVERTALL_LINE_SAMPLE_MINIMUM: Final = 4
"""How many x-height letters a line needs before its median is worth comparing against."""


def line_x_height_reference(heights_by_character: Sequence[tuple[str, int]]) -> float | None:
    """The median height of a line's own x-height-only letters, or `None` if too few to trust."""

    heights = [
        height for character, height in heights_by_character if character in X_HEIGHT_CHARACTERS
    ]
    if len(heights) < _OVERTALL_LINE_SAMPLE_MINIMUM:
        return None
    reference = median(heights)
    return reference if reference > 0 else None


def is_overtall(character: str, height: int, reference: float | None) -> bool:
    """Whether an x-height letter runs taller than its line's other x-height letters."""

    if reference is None or character not in X_HEIGHT_CHARACTERS:
        return False
    return height / reference >= _OVERTALL_RATIO_MINIMUM


_NARROW_RATIO_MAXIMUM: Final = 0.6
"""How narrow a glyph may run against its character's usual width in the book before it is flagged.

A letter built from two strokes joined high up breaks under a light impression: the arch fails and
the stem is left as its own ink run. The word can still be cut if some other pair of letters
touched, so the counts agree and one glyph carries only half a letter. Measured, `h` is the usual
casualty, at 12.5 percent of them in projectID67a80fde44d34 and the worst character in three more
books; `m`, `n` and `r` follow, and `T` loses its arms the same way.

The reference is the median width of that character in that style across the book, so it needs no
per-line estimate. At 0.6 this flags 0.17 to 1.31 percent of glyphs, and nine or ten of ten
reviewed are genuinely half a letter: the `T` of `ETC.` reduced to its stem, the `n` of `furnish`
to its left leg.
"""

_WIDE_RATIO_MINIMUM: Final = 1.5
"""How wide a glyph may run against its character's usual width before it is flagged."""

_NARROW_CHARACTER_SAMPLE_MINIMUM: Final = 30
"""How many of a character a book needs before its median width is worth comparing against."""


def character_width_reference(widths: Sequence[int]) -> float | None:
    """One character's usual width in a book, or `None` when the book carries too few of it."""

    if len(widths) < _NARROW_CHARACTER_SAMPLE_MINIMUM:
        return None
    reference = median(widths)
    return reference if reference > 0 else None


def is_narrow(width: int, reference: float | None) -> bool:
    """Whether a glyph is too narrow to be the whole letter its label names."""

    return reference is not None and width < _NARROW_RATIO_MAXIMUM * reference


def is_wide(width: int, reference: float | None) -> bool:
    """Whether a glyph is too wide to be only the letter its label names.

    The mirror of `is_narrow` and the same evidence. Where a broken arch leaves half a letter, a
    missed blank column leaves a letter and its neighbours in one box: `and` labelled `o`, `ingine`
    labelled `f`, `whe` labelled `t`. Twelve of twelve reviewed at this threshold were genuine
    multi-letter boxes, and it flags 0.06 to 0.91 percent of glyphs.

    It sees only the gross cases. A box holding a letter and a comma, or a letter and one stem of
    its neighbour, sits at 1.06 to 1.15 times the median and inside the ordinary spread, so no
    width test separates it.
    """

    return reference is not None and width >= _WIDE_RATIO_MINIMUM * reference
