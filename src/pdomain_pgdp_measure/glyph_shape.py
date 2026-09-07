"""Score each glyph against the shape its own character usually takes in the same book.

Size gets a long way and then stops. A box holding a letter and one stem of its neighbour, or a
`b` where the label says `h`, sits at an ordinary width and an ordinary height; nothing measured
from the box alone separates it. What does separate it is the ink inside the box.

Each glyph is normalised onto a small grid, and each character in each style gets a reference: the
median of its own clean glyphs. A glyph far from its reference is unlike the letter it claims to
be. Measured, glyphs of one character sit 2.5 to 3.3 from their own reference at the median, while
two different characters' references sit 4.3 to 7.7 apart, so the distance carries real signal.

A glyph in a style the book carries too few of gets no reference and so no check. An italic `e`
is not shaped like a roman one, so comparing the two would manufacture false positives; leaving it
unchecked reports less rather than reporting wrong.

Nothing here is a verdict. The flag says the ink is unlike the character's usual shape, not which
character it really is, and this module never reads or proposes a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

GLYPH_SHAPE_METHODS: dict[str, float | int | str] = {
    "algorithm": "character-median-shape/v1",
    "shape_grid_rows": 16,
    "shape_grid_columns": 12,
    "shape_outlier_ratio": 1.4,
    "shape_reference_sample_minimum": 60,
    "shape_reference_sample_maximum": 250,
}

SHAPE_ROWS: Final = 16
SHAPE_COLUMNS: Final = 12
"""A grid coarse enough to ignore the impression and fine enough to tell letters apart.

Every glyph is resampled onto it, so a comparison never depends on the two being the same size,
which they are not: an inventory spans chapter openings and body text in one book.
"""

_OUTLIER_RATIO: Final = 1.4
"""How far past its character's own 95th percentile a glyph must sit to be called unlike it.

Adaptive rather than absolute, because a book with heavy impression scatters more than a clean
one. At 1.4 this flags 0.5 to 5.1 percent of glyphs per book. Of six reviewed label errors that
survived every size test, four land outside it and one, an `s` cut with its comma, sits at 1.39
and just inside. The threshold is not moved to capture it: six cells cannot calibrate a threshold,
and a bound at roughly twice the character's median spread is the defensible line.
"""

_REFERENCE_SAMPLE_MINIMUM: Final = 60
_REFERENCE_SAMPLE_MAXIMUM: Final = 250
"""Below the minimum a character has no reliable shape; above the maximum the median stops moving."""

ShapeGrid = np.ndarray[tuple[int, int], np.dtype[np.uint8]]


@dataclass(frozen=True, slots=True)
class ShapeReference:
    """One character's usual shape in one style, and how far its own glyphs scatter from it."""

    median: ShapeGrid
    scatter_p95: float
    sample_count: int

    def distance(self, grid: ShapeGrid) -> float:
        difference = grid.astype(np.float32) - self.median.astype(np.float32)
        return float(np.sqrt((difference * difference).sum()) / 255.0)

    def is_outlier(self, grid: ShapeGrid) -> bool:
        return self.scatter_p95 > 0.0 and self.distance(grid) > _OUTLIER_RATIO * self.scatter_p95


def normalise_glyph(crop: np.ndarray[tuple[int, int], np.dtype[np.uint8]]) -> ShapeGrid:
    """Resample one glyph's grayscale crop onto the comparison grid, as ink density."""

    if crop.size == 0:
        return np.zeros((SHAPE_ROWS, SHAPE_COLUMNS), dtype=np.uint8)
    with Image.fromarray(crop, mode="L") as image:
        resized = image.resize((SHAPE_COLUMNS, SHAPE_ROWS), Image.Resampling.BILINEAR)
        return 255 - np.asarray(resized, dtype=np.uint8)


def build_reference(grids: Sequence[ShapeGrid]) -> ShapeReference | None:
    """One character's reference shape, or `None` when the book carries too few of it."""

    if len(grids) < _REFERENCE_SAMPLE_MINIMUM:
        return None
    stack = np.stack([grid.astype(np.float32) for grid in grids[:_REFERENCE_SAMPLE_MAXIMUM]])
    median = np.median(stack, axis=0)
    differences = stack - median
    distances = np.sort(np.sqrt((differences * differences).sum(axis=(1, 2))) / 255.0)
    index = min(len(distances) - 1, int(0.95 * len(distances)))
    return ShapeReference(
        median=median.astype(np.uint8),
        scatter_p95=float(distances[index]),
        sample_count=len(grids),
    )


def build_references(
    grids_by_key: Mapping[tuple[str, str | None], Sequence[ShapeGrid]],
) -> dict[tuple[str, str | None], ShapeReference]:
    """A reference per character and style, skipping those the book carries too few of."""

    return {
        key: reference
        for key, grids in grids_by_key.items()
        if (reference := build_reference(grids)) is not None
    }
