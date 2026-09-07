"""Deterministic source-frame foreground measurements for PGDP scans."""

from __future__ import annotations

import hashlib
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import median
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, BinaryIO, Literal, override

import numpy as np
from PIL import Image

from pdomain_pgdp_measure.profile_models import CoordinateFrame

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_EXIF_ORIENTATION_TAG = 274
_HASH_CHUNK_SIZE = 64 * 1024
_GRAYSCALE_LEVELS = 256

ImageMeasurementDiagnostic = Literal[
    "blank", "one_band", "border_dominated", "high_foreground_ratio"
]


class SnapshotSpoolError(RuntimeError):
    """Temporary scan snapshot storage failed."""


class _SourceImageReadError(OSError):
    """Reading the original image file failed while it was being copied."""


class SnapshotReplayFile(BinaryIO):
    """Translate temporary snapshot I/O failures during Pillow replay."""

    _source_file: BinaryIO

    def __init__(self, source_file: BinaryIO) -> None:
        self._source_file = source_file

    @override
    def readable(self) -> bool:
        return True

    @override
    def seekable(self) -> bool:
        return True

    @override
    def read(self, size: int | None = -1) -> bytes:
        try:
            if size is None:
                return self._source_file.read()
            return self._source_file.read(size)
        except OSError as error:
            raise SnapshotSpoolError("Could not read a temporary scan snapshot.") from error

    @override
    def seek(self, offset: int, whence: int = 0) -> int:
        try:
            return self._source_file.seek(offset, whence)
        except OSError as error:
            raise SnapshotSpoolError("Could not seek a temporary scan snapshot.") from error

    @override
    def tell(self) -> int:
        try:
            return self._source_file.tell()
        except OSError as error:
            raise SnapshotSpoolError(
                "Could not tell the temporary scan snapshot position."
            ) from error


@dataclass(frozen=True, slots=True)
class InkBandMeasurement:
    """One retained horizontal ink band in source-frame pixels."""

    y_start: int
    y_end: int
    x_start: int
    x_end: int
    foreground_pixels: int

    def __post_init__(self) -> None:
        values = (
            ("y_start", self.y_start),
            ("y_end", self.y_end),
            ("x_start", self.x_start),
            ("x_end", self.x_end),
            ("foreground_pixels", self.foreground_pixels),
        )
        for name, value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if self.y_end - self.y_start < 2:
            raise ValueError("Ink bands must be at least two rows high.")
        if self.x_end <= self.x_start:
            raise ValueError("Ink bands must have nonempty horizontal bounds.")
        if self.foreground_pixels == 0:
            raise ValueError("Ink bands must contain foreground pixels.")
        if self.foreground_pixels > (self.y_end - self.y_start) * (self.x_end - self.x_start):
            raise ValueError("Ink band foreground pixels must fit inside its bounds.")


@dataclass(frozen=True, slots=True)
class ImageMeasurement:
    """Observed source-frame geometry for one decoded scan image."""

    sha256: str
    source_frame: CoordinateFrame
    image_mode: str
    exif_orientation: int | None
    grayscale_threshold: int
    foreground_pixels: int
    foreground_bounds: tuple[int, int, int, int] | None
    margins: tuple[int, int, int, int] | None
    ink_bands: tuple[InkBandMeasurement, ...]
    median_ink_band_height: float | None
    median_ink_band_top_pitch: float | None
    diagnostics: tuple[ImageMeasurementDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class ImageSnapshot:
    """One seekable temporary copy of original scan bytes."""

    sha256: str
    source_file: BinaryIO


def measure_image(image_path: str | Path) -> ImageMeasurement:
    """Measure one stored raster without applying its EXIF orientation."""

    with open_image_snapshot(image_path) as snapshot:
        return measure_image_snapshot(snapshot)


def measure_image_snapshot(snapshot: ImageSnapshot) -> ImageMeasurement:
    """Measure one seekable temporary copy of original scan bytes."""

    source_file = SnapshotReplayFile(snapshot.source_file)
    source_file.seek(0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source_file) as image:
                source_frame = CoordinateFrame(width=image.width, height=image.height)
                image_mode = image.mode
                exif_orientation = _exif_orientation(image)
                with _source_grayscale(image) as grayscale_image:
                    grayscale = np.asarray(grayscale_image, dtype=np.uint8)
                    threshold = _otsu_threshold(_histogram(grayscale))
                    foreground = grayscale <= threshold
                    foreground_pixels = int(np.count_nonzero(foreground))
                    bounds = _foreground_bounds(foreground)
                    margins = _margins(bounds, source_frame)
                    ink_bands = extract_ink_bands(foreground)
                    median_band_height = _median_band_height(ink_bands)
                    median_band_top_pitch = _median_band_top_pitch(ink_bands)
                    diagnostics = _diagnostics(
                        foreground,
                        foreground_pixels=foreground_pixels,
                        ink_bands=ink_bands,
                    )
                    del foreground
                    del grayscale
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("Image rejected as a decompression bomb.") from error
    return ImageMeasurement(
        sha256=snapshot.sha256,
        source_frame=source_frame,
        image_mode=image_mode,
        exif_orientation=exif_orientation,
        grayscale_threshold=threshold,
        foreground_pixels=foreground_pixels,
        foreground_bounds=bounds,
        margins=margins,
        ink_bands=ink_bands,
        median_ink_band_height=median_band_height,
        median_ink_band_top_pitch=median_band_top_pitch,
        diagnostics=diagnostics,
    )


@contextmanager
def open_image_snapshot(image_path: str | Path) -> Iterator[ImageSnapshot]:
    """Copy one scan to a seekable temporary file while hashing bounded chunks."""

    path = Path(image_path)
    with path.open("rb") as source_file:
        yielded_snapshot = False
        try:
            with TemporaryFile(mode="w+b") as snapshot_file:
                source_sha256 = _copy_source_to_snapshot(source_file, snapshot_file)
                yielded_snapshot = True
                yield ImageSnapshot(sha256=source_sha256, source_file=snapshot_file)
                yielded_snapshot = False
        except (_SourceImageReadError, SnapshotSpoolError):
            raise
        except OSError as error:
            if yielded_snapshot:
                raise
            raise SnapshotSpoolError("Could not prepare a temporary scan snapshot.") from error


def _copy_source_to_snapshot(source_file: BinaryIO, snapshot_file: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        try:
            chunk = source_file.read(_HASH_CHUNK_SIZE)
        except OSError as error:
            raise _SourceImageReadError("Could not read the source image file.") from error
        if not chunk:
            break
        digest.update(chunk)
        try:
            _ = snapshot_file.write(chunk)
        except OSError as error:
            raise SnapshotSpoolError("Could not write a temporary scan snapshot.") from error
    try:
        _ = snapshot_file.seek(0)
    except OSError as error:
        raise SnapshotSpoolError("Could not rewind a temporary scan snapshot.") from error
    return digest.hexdigest()


def sha256_file(image_path: str | Path) -> str:
    """Return the SHA-256 digest of a file's original stored bytes."""

    path = Path(image_path)
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _exif_orientation(image: Image.Image) -> int | None:
    orientation = image.getexif().get(_EXIF_ORIENTATION_TAG)
    return (
        orientation if isinstance(orientation, int) and not isinstance(orientation, bool) else None
    )


def _source_grayscale(image: Image.Image) -> Image.Image:
    if image.mode in {"LA", "RGBA"} or "transparency" in image.info:
        with (
            image.convert("RGBA") as rgba,
            Image.new("RGBA", image.size, color=(255, 255, 255, 255)) as white,
            Image.alpha_composite(white, rgba) as composite,
        ):
            return composite.convert("L")
    return image.convert("L")


def _histogram(grayscale: np.ndarray[tuple[int, int], np.dtype[np.uint8]]) -> tuple[int, ...]:
    counts = np.bincount(grayscale.ravel(), minlength=_GRAYSCALE_LEVELS)
    return tuple(int(count) for count in counts[:_GRAYSCALE_LEVELS])


def _otsu_threshold(histogram: tuple[int, ...]) -> int:
    total = sum(histogram)
    weighted_total = sum(level * count for level, count in enumerate(histogram))
    background_count = 0
    background_total = 0
    best_threshold = 0
    best_numerator = 0
    best_denominator = 1
    for threshold, count in enumerate(histogram):
        background_count += count
        background_total += threshold * count
        foreground_count = total - background_count
        if background_count == 0 or foreground_count == 0:
            continue
        foreground_total = weighted_total - background_total
        difference = background_total * foreground_count - foreground_total * background_count
        numerator = difference * difference
        denominator = background_count * foreground_count
        if numerator * best_denominator > best_numerator * denominator:
            best_threshold = threshold
            best_numerator = numerator
            best_denominator = denominator
    return best_threshold


def _foreground_bounds(
    foreground: np.ndarray[tuple[int, int], np.dtype[np.bool]],
) -> tuple[int, int, int, int] | None:
    active_rows = foreground.any(axis=1)
    if not bool(active_rows.any()):
        return None
    active_columns = foreground.any(axis=0)
    return (
        int(active_columns.argmax()),
        int(active_rows.argmax()),
        int(active_columns.size - active_columns[::-1].argmax()),
        int(active_rows.size - active_rows[::-1].argmax()),
    )


def _margins(
    bounds: tuple[int, int, int, int] | None, source_frame: CoordinateFrame
) -> tuple[int, int, int, int] | None:
    if bounds is None:
        return None
    x_start, y_start, x_end, y_end = bounds
    return (x_start, y_start, source_frame.width - x_end, source_frame.height - y_end)


def extract_ink_bands(
    foreground: np.ndarray[tuple[int, int], np.dtype[np.bool]],
) -> tuple[InkBandMeasurement, ...]:
    """Extract joined row-projection ink bands from one foreground mask."""

    bounds = _foreground_bounds(foreground)
    if bounds is None:
        return ()
    print_width = bounds[2] - bounds[0]
    active_row_threshold = _active_row_threshold(print_width)
    row_counts = foreground.sum(axis=1)
    active_rows = row_counts >= active_row_threshold
    ranges = _joined_active_ranges(active_rows)
    return tuple(
        band
        for y_start, y_end in ranges
        if (band := _measure_ink_band(foreground, y_start=y_start, y_end=y_end)) is not None
    )


def _active_row_threshold(print_width: int) -> int:
    return max(2, (print_width + 199) // 200)


def _joined_active_ranges(
    active_rows: np.ndarray[tuple[int], np.dtype[np.bool]],
) -> tuple[tuple[int, int], ...]:
    range_start: int | None = None
    last_active_row: int | None = None
    ranges: list[tuple[int, int]] = []
    for row_index, is_active in enumerate(active_rows):
        if bool(is_active):
            if range_start is None:
                range_start = row_index
            last_active_row = row_index
            continue
        if (
            range_start is not None
            and last_active_row is not None
            and row_index - last_active_row > 1
        ):
            ranges.append((range_start, last_active_row + 1))
            range_start = None
            last_active_row = None
    if range_start is not None and last_active_row is not None:
        ranges.append((range_start, last_active_row + 1))
    return tuple(ranges)


def _measure_ink_band(
    foreground: np.ndarray[tuple[int, int], np.dtype[np.bool]], *, y_start: int, y_end: int
) -> InkBandMeasurement | None:
    if y_end - y_start < 2:
        return None
    band_mask = foreground[y_start:y_end]
    active_columns = band_mask.any(axis=0)
    if not bool(active_columns.any()):
        return None
    x_start = int(active_columns.argmax())
    x_end = int(active_columns.size - active_columns[::-1].argmax())
    return InkBandMeasurement(
        y_start=y_start,
        y_end=y_end,
        x_start=x_start,
        x_end=x_end,
        foreground_pixels=int(np.count_nonzero(band_mask)),
    )


def _median_band_height(ink_bands: Sequence[InkBandMeasurement]) -> float | None:
    if not ink_bands:
        return None
    return float(median(band.y_end - band.y_start for band in ink_bands))


def _median_band_top_pitch(ink_bands: Sequence[InkBandMeasurement]) -> float | None:
    if len(ink_bands) < 2:
        return None
    return float(median(later.y_start - earlier.y_start for earlier, later in pairwise(ink_bands)))


def _diagnostics(
    foreground: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    *,
    foreground_pixels: int,
    ink_bands: Sequence[InkBandMeasurement],
) -> tuple[ImageMeasurementDiagnostic, ...]:
    diagnostics: list[ImageMeasurementDiagnostic] = []
    if foreground_pixels == 0:
        diagnostics.append("blank")
    elif len(ink_bands) == 1:
        diagnostics.append("one_band")
    if _is_border_dominated(foreground):
        diagnostics.append("border_dominated")
    if foreground_pixels * 100 > foreground.size * 60:
        diagnostics.append("high_foreground_ratio")
    return tuple(diagnostics)


def _is_border_dominated(
    foreground: np.ndarray[tuple[int, int], np.dtype[np.bool]],
) -> bool:
    height, width = foreground.shape
    row_strip_height = _edge_strip_extent(height)
    column_strip_width = _edge_strip_extent(width)
    horizontal_edges_are_dense = _is_at_least_half_full(
        foreground[:row_strip_height]
    ) and _is_at_least_half_full(foreground[height - row_strip_height :])
    vertical_edges_are_dense = _is_at_least_half_full(
        foreground[:, :column_strip_width]
    ) and _is_at_least_half_full(foreground[:, width - column_strip_width :])
    return horizontal_edges_are_dense or vertical_edges_are_dense


def _edge_strip_extent(dimension: int) -> int:
    return max(1, (dimension + 99) // 100)


def _is_at_least_half_full(
    foreground_strip: np.ndarray[tuple[int, int], np.dtype[np.bool]],
) -> bool:
    return (
        foreground_strip.size > 0
        and int(np.count_nonzero(foreground_strip)) * 2 >= foreground_strip.size
    )
