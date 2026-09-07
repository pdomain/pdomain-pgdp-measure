"""Render one book's cut glyphs as per-character grids, so a bad cut is visible at a glance.

The JSONL is the inventory's authority and the atlas is a render of it: open `U+0065-000.png` and
every `e` in the book is on one grid, where a glyph carrying the wrong label or a cut through the
middle of a letter stands out from its neighbours. That is what the label-correctness gate is
reviewed against, built in rather than bolted on.

One grid per character **per style**, because a small-capital `O` and a lowercase `o` share a
character and not a letterform; pooling them would put a capital in the lowercase grid, which is
exactly the defect the grids exist to make visible.

One file per character, not per glyph. A file per glyph is 10,000 files a book and about three
million across the 286-project corpus; per character is 50 to 90 grids a book, sharded so no sheet
grows past the repository's own large-file limit.

Nothing is scaled. A cell is the character's 95th-percentile extent on its tier, so a glyph whose
box ran away does not stretch every other cell on the page; a glyph larger than the cell is
centre-cropped and the sheet records how many that happened to.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from PIL import Image

from pdomain_pgdp_measure.glyph_models import (
    GLYPHS_ATLAS_DIRECTORY,
    GLYPHS_MANIFEST_FILENAME,
    GLYPHS_ROWS_FILENAME,
    AtlasSheet,
    GlyphManifest,
    GlyphPage,
    GlyphRow,
    read_rows,
)
from pdomain_pgdp_measure.image_measurement import open_image_snapshot
from pdomain_pgdp_measure.ordering import natural_page_key

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ATLAS_METHODS: dict[str, float | int | str] = {
    "algorithm": "per-character-grid/v1",
    "atlas_columns": 32,
    "atlas_cells_per_sheet_maximum": 1024,
    "atlas_cell_extent_percentile": 95,
    "atlas_gutter_px": 1,
}

_COLUMNS: Final = 32
_CELLS_PER_SHEET_MAXIMUM: Final = 1024
_CELL_EXTENT_PERCENTILE: Final = 0.95
_GUTTER_PX: Final = 1
_GUTTER_VALUE: Final = 160
UNKNOWN_STYLE: Final = "unknown-style"
"""Directory name for glyphs with no trusted style source, which is the whole recognized tier."""
_PAPER_VALUE: Final = 255


class AtlasError(ValueError):
    """An atlas cannot be rendered from the inputs as given."""


@dataclass(frozen=True, slots=True)
class RenderedSheet:
    """One atlas sheet's record and the PNG bytes it was hashed from."""

    sheet: AtlasSheet
    payload: bytes


def render_atlas(
    rows: Sequence[GlyphRow], *, corpus_root: str | Path, pages: Sequence[GlyphPage]
) -> tuple[RenderedSheet, ...]:
    """Render every character's grid from the rows alone, plus the scans they point at.

    `pages` is the manifest's page table, which is what turns a row's page name into a scan on
    disk. Each scan is opened once and checked against the hash the table records, so a sheet can
    never be drawn from a file that has changed under the inventory.
    """

    root = Path(corpus_root).expanduser().resolve()
    crops = _crop_rows(rows, root=root, pages=pages)
    sheets: list[RenderedSheet] = []
    for (tier, style, character), indexed in sorted(_group(rows, crops).items()):
        cell_width, cell_height = cell_extent(indexed)
        for ordinal, start in enumerate(range(0, len(indexed), _CELLS_PER_SHEET_MAXIMUM)):
            block = indexed[start : start + _CELLS_PER_SHEET_MAXIMUM]
            payload, clipped = render_sheet(block, cell_width=cell_width, cell_height=cell_height)
            sheets.append(
                RenderedSheet(
                    sheet=AtlasSheet(
                        character=character,
                        label_style=None if style == UNKNOWN_STYLE else style,
                        label_tier="recognized" if tier == "recognized" else "transcribed",
                        sheet_ordinal=ordinal,
                        path=sheet_path(
                            tier=tier, style=style, character=character, ordinal=ordinal
                        ),
                        cell_count=len(block),
                        cell_width_px=cell_width,
                        cell_height_px=cell_height,
                        sha256=sha256(payload).hexdigest(),
                        clipped_cell_count=clipped,
                    ),
                    payload=payload,
                )
            )
    return tuple(sheets)


def sheet_path(*, tier: str, style: str, character: str, ordinal: int) -> str:
    """Where one sheet sits inside the inventory directory."""

    return f"{GLYPHS_ATLAS_DIRECTORY}/{tier}/{style}/U+{ord(character):04X}-{ordinal:03d}.png"


def write_atlas(sheets: Sequence[RenderedSheet], directory: str | Path) -> tuple[AtlasSheet, ...]:
    """Write every rendered sheet under the inventory directory and return their records."""

    root = Path(directory).expanduser().resolve()
    for rendered in sheets:
        path = root / rendered.sheet.path
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(rendered.payload)
    return tuple(rendered.sheet for rendered in sheets)


def verify_atlas(directory: str | Path, corpus_root: str | Path) -> tuple[str, ...]:
    """Re-render the atlas from the written inventory and name every sheet that did not match.

    An empty result is the acceptance gate passing: the sheets on disk are exactly what the rows
    say they should be, so the atlas carries no information the JSONL does not.
    """

    root = Path(directory).expanduser().resolve()
    manifest = GlyphManifest.from_json((root / GLYPHS_MANIFEST_FILENAME).read_bytes())
    rows = read_rows((root / GLYPHS_ROWS_FILENAME).read_bytes())
    rendered = render_atlas(rows, corpus_root=corpus_root, pages=manifest.pages)
    recorded = {sheet.path: sheet.sha256 for sheet in manifest.atlas}
    mismatched: list[str] = []
    for sheet in rendered:
        path = root / sheet.sheet.path
        on_disk = sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if recorded.get(sheet.sheet.path) != sheet.sheet.sha256 or on_disk != sheet.sheet.sha256:
            mismatched.append(sheet.sheet.path)
    mismatched.extend(path for path in recorded if path not in {s.sheet.path for s in rendered})
    return tuple(sorted(set(mismatched)))


def _crop_rows(
    rows: Sequence[GlyphRow], *, root: Path, pages: Sequence[GlyphPage]
) -> list[np.ndarray[tuple[int, int], np.dtype[np.uint8]]]:
    """Every row's grayscale crop, in the order the rows were given."""

    by_name = {page.page_name: page for page in pages}
    crops: list[np.ndarray[tuple[int, int], np.dtype[np.uint8]] | None] = [None] * len(rows)
    indices_by_page: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        indices_by_page.setdefault(row.page_name, []).append(index)
    for page_name in sorted(indices_by_page, key=natural_page_key):
        page = by_name.get(page_name)
        if page is None:
            raise AtlasError(f"The manifest page table carries no page {page_name!r}.")
        grayscale = _page_grayscale(root, page=page)
        for index in indices_by_page[page_name]:
            x_start, y_start, x_end, y_end = rows[index].box
            crops[index] = grayscale[y_start:y_end, x_start:x_end]
    return [crop for crop in crops if crop is not None]


def _page_grayscale(
    root: Path, *, page: GlyphPage
) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    path = (root / page.source_path).resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise AtlasError(f"The scan for page {page.page_name!r} is not in the corpus.")
    with open_image_snapshot(path) as snapshot:
        if snapshot.sha256 != page.scan_sha256:
            raise AtlasError(f"The scan for page {page.page_name!r} has changed since harvest.")
        _ = snapshot.source_file.seek(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(snapshot.source_file) as image, _grayscale(image) as converted:
                return np.asarray(converted, dtype=np.uint8)


def _grayscale(image: Image.Image) -> Image.Image:
    """The same conversion `image_measurement` makes, so a crop shows the ink the mask saw."""

    if image.mode in {"LA", "RGBA"} or "transparency" in image.info:
        with (
            image.convert("RGBA") as rgba,
            Image.new("RGBA", image.size, color=(255, 255, 255, 255)) as white,
            Image.alpha_composite(white, rgba) as composite,
        ):
            return composite.convert("L")
    return image.convert("L")


def _group(
    rows: Sequence[GlyphRow],
    crops: Sequence[np.ndarray[tuple[int, int], np.dtype[np.uint8]]],
) -> Mapping[tuple[str, str, str], list[np.ndarray[tuple[int, int], np.dtype[np.uint8]]]]:
    """Bucket crops by tier, style and character, which is what one grid may hold."""

    grouped: dict[tuple[str, str, str], list[np.ndarray[tuple[int, int], np.dtype[np.uint8]]]] = {}
    for row, crop in zip(rows, crops, strict=True):
        grouped.setdefault(
            (row.label_tier, row.label_style or UNKNOWN_STYLE, row.character), []
        ).append(crop)
    return grouped


def cell_extent(
    crops: Sequence[np.ndarray[tuple[int, int], np.dtype[np.uint8]]],
) -> tuple[int, int]:
    """The cell one character's grid uses: its 95th-percentile width and height."""

    widths = sorted(int(crop.shape[1]) for crop in crops)
    heights = sorted(int(crop.shape[0]) for crop in crops)
    return max(1, _percentile(widths)), max(1, _percentile(heights))


def _percentile(values: Sequence[int]) -> int:
    index = min(len(values) - 1, int(_CELL_EXTENT_PERCENTILE * len(values)))
    return values[index]


def render_sheet(
    crops: Sequence[np.ndarray[tuple[int, int], np.dtype[np.uint8]]],
    *,
    cell_width: int,
    cell_height: int,
) -> tuple[bytes, int]:
    """One grid of glyph crops as PNG bytes, with how many were centre-cropped to fit."""

    columns = min(_COLUMNS, len(crops))
    rows_of_cells = -(-len(crops) // columns)
    width = columns * (cell_width + _GUTTER_PX) + _GUTTER_PX
    height = rows_of_cells * (cell_height + _GUTTER_PX) + _GUTTER_PX
    sheet = np.full((height, width), _GUTTER_VALUE, dtype=np.uint8)
    clipped = 0
    for index, crop in enumerate(crops):
        row, column = divmod(index, columns)
        top = row * (cell_height + _GUTTER_PX) + _GUTTER_PX
        left = column * (cell_width + _GUTTER_PX) + _GUTTER_PX
        sheet[top : top + cell_height, left : left + cell_width] = _PAPER_VALUE
        fitted, was_clipped = _fit(crop, cell_width=cell_width, cell_height=cell_height)
        clipped += int(was_clipped)
        offset_y = top + (cell_height - fitted.shape[0]) // 2
        offset_x = left + (cell_width - fitted.shape[1]) // 2
        sheet[offset_y : offset_y + fitted.shape[0], offset_x : offset_x + fitted.shape[1]] = fitted
    buffer = BytesIO()
    with Image.fromarray(sheet, mode="L") as image:
        image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue(), clipped


def _fit(
    crop: np.ndarray[tuple[int, int], np.dtype[np.uint8]], *, cell_width: int, cell_height: int
) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.uint8]], bool]:
    """Centre-crop one glyph to its cell, reporting whether anything was cut away."""

    height, width = crop.shape
    top = max(0, (height - cell_height) // 2)
    left = max(0, (width - cell_width) // 2)
    fitted = crop[top : top + min(height, cell_height), left : left + min(width, cell_width)]
    return fitted, height > cell_height or width > cell_width
