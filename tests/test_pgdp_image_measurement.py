"""Tests for deterministic source-frame scan measurements."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, override

import pytest
from PIL import Image, ImageDraw

from pdomain_pgdp_measure import image_measurement
from pdomain_pgdp_measure.image_measurement import measure_image, sha256_file

if TYPE_CHECKING:
    from typing import BinaryIO, NoReturn

    from pdomain_pgdp_measure.image_measurement import ImageSnapshot


class _ChunkLimitedSource:
    def __init__(self, data: bytes, *, maximum_chunk_size: int) -> None:
        self._data = data
        self._maximum_chunk_size = maximum_chunk_size
        self._offset = 0
        self.requests: list[int] = []

    def __enter__(self) -> _ChunkLimitedSource:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        assert 0 < size <= self._maximum_chunk_size
        self.requests.append(size)
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class _WriteFailingSnapshot:
    def __enter__(self) -> _WriteFailingSnapshot:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        return None

    def close(self) -> None:
        return None

    def seek(self, _offset: int, _whence: int = 0) -> int:
        return 0

    def write(self, _data: bytes) -> int:
        raise OSError("No space left on device")


class _ReplayReadFailingSnapshot(BytesIO):
    @override
    def read(self, _size: int = -1) -> bytes:
        raise OSError("Input/output error")


class _ReplaySeekFailingSnapshot(BytesIO):
    @override
    def seek(self, _offset: int, _whence: int = 0) -> int:
        raise OSError("Input/output error")


def _record_temporary_file_then_stop(
    snapshot: ImageSnapshot, source_files: list[BinaryIO]
) -> NoReturn:
    source_files.append(snapshot.source_file)
    raise RuntimeError("stop")


def test_open_image_snapshot_copies_source_in_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "source.bin"
    payload = b"x" * (image_measurement._HASH_CHUNK_SIZE * 2 + 1)
    source = _ChunkLimitedSource(
        payload,
        maximum_chunk_size=image_measurement._HASH_CHUNK_SIZE,
    )

    def open_source(path: Path, *_args: object, **_kwargs: object) -> _ChunkLimitedSource:
        assert path == image_path
        return source

    monkeypatch.setattr(Path, "open", open_source)

    with image_measurement.open_image_snapshot(image_path) as snapshot:
        assert snapshot.source_file.read() == payload
        assert snapshot.sha256 == sha256(payload).hexdigest()

    assert source.requests == [
        image_measurement._HASH_CHUNK_SIZE,
        image_measurement._HASH_CHUNK_SIZE,
        image_measurement._HASH_CHUNK_SIZE,
        image_measurement._HASH_CHUNK_SIZE,
    ]


def test_open_image_snapshot_closes_its_temporary_file(tmp_path: Path) -> None:
    image_path = tmp_path / "source.bin"
    image_path.write_bytes(b"source")

    with image_measurement.open_image_snapshot(image_path) as snapshot:
        source_file = snapshot.source_file
        assert not source_file.closed

    assert source_file.closed


def test_open_image_snapshot_closes_its_temporary_file_after_an_error(tmp_path: Path) -> None:
    image_path = tmp_path / "source.bin"
    image_path.write_bytes(b"source")
    source_files: list[BinaryIO] = []

    with (
        pytest.raises(RuntimeError, match="stop"),
        image_measurement.open_image_snapshot(image_path) as snapshot,
    ):
        _record_temporary_file_then_stop(snapshot, source_files)

    assert source_files[0].closed


def test_open_image_snapshot_raises_for_a_temporary_snapshot_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "source.bin"
    image_path.write_bytes(b"source")

    def temporary_file_full(*_args: object, **_kwargs: object) -> _WriteFailingSnapshot:
        return _WriteFailingSnapshot()

    monkeypatch.setattr(image_measurement, "TemporaryFile", temporary_file_full)

    with (
        pytest.raises(image_measurement.SnapshotSpoolError, match="temporary scan snapshot"),
        image_measurement.open_image_snapshot(image_path),
    ):
        pass


def test_measure_image_snapshot_raises_for_a_temporary_snapshot_read_failure() -> None:
    snapshot = image_measurement.ImageSnapshot(
        sha256="a" * 64,
        source_file=_ReplayReadFailingSnapshot(b"source"),
    )

    with pytest.raises(image_measurement.SnapshotSpoolError, match="temporary scan snapshot"):
        _ = image_measurement.measure_image_snapshot(snapshot)


def test_measure_image_snapshot_raises_for_a_temporary_snapshot_seek_failure() -> None:
    snapshot = image_measurement.ImageSnapshot(
        sha256="a" * 64,
        source_file=_ReplaySeekFailingSnapshot(b"source"),
    )

    with pytest.raises(image_measurement.SnapshotSpoolError, match="temporary scan snapshot"):
        _ = image_measurement.measure_image_snapshot(snapshot)


def test_blank_white_image_has_no_foreground_geometry(tmp_path: Path) -> None:
    image_path = tmp_path / "blank.png"
    Image.new("L", (12, 9), color=255).save(image_path)

    result = measure_image(image_path)

    assert result.source_frame.width == 12
    assert result.source_frame.height == 9
    assert result.image_mode == "L"
    assert result.grayscale_threshold == 0
    assert result.foreground_pixels == 0
    assert result.foreground_bounds is None
    assert result.margins is None
    assert result.ink_bands == ()
    assert result.median_ink_band_height is None
    assert result.median_ink_band_top_pitch is None
    assert result.diagnostics == ("blank",)


def test_solid_black_image_uses_the_whole_source_frame(tmp_path: Path) -> None:
    image_path = tmp_path / "solid.png"
    Image.new("L", (7, 5), color=0).save(image_path)

    result = measure_image(image_path)

    assert result.grayscale_threshold == 0
    assert result.foreground_pixels == 35
    assert result.foreground_bounds == (0, 0, 7, 5)
    assert result.margins == (0, 0, 0, 0)
    assert result.ink_bands[0].y_start == 0
    assert result.ink_bands[0].y_end == 5
    assert result.diagnostics == ("one_band", "border_dominated", "high_foreground_ratio")


def test_ink_bands_join_one_row_gaps_and_store_tight_band_geometry(tmp_path: Path) -> None:
    image_path = tmp_path / "bands.png"
    image = Image.new("L", (80, 30), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 3, 25, 5), fill=0)
    draw.rectangle((11, 7, 22, 9), fill=0)
    draw.rectangle((30, 16, 39, 18), fill=0)
    image.save(image_path)

    result = measure_image(image_path)

    assert result.ink_bands == (
        image_measurement.InkBandMeasurement(
            y_start=3,
            y_end=10,
            x_start=10,
            x_end=26,
            foreground_pixels=84,
        ),
        image_measurement.InkBandMeasurement(
            y_start=16,
            y_end=19,
            x_start=30,
            x_end=40,
            foreground_pixels=30,
        ),
    )
    assert result.median_ink_band_height == 5.0
    assert result.median_ink_band_top_pitch == 13.0
    assert result.diagnostics == ()


def test_ink_bands_drop_single_row_runs_and_ignore_single_pixel_row_specks(tmp_path: Path) -> None:
    image_path = tmp_path / "specks.png"
    image = Image.new("L", (100, 30), color=255)
    draw = ImageDraw.Draw(image)
    draw.point((0, 1), fill=0)
    draw.rectangle((25, 10, 34, 12), fill=0)
    draw.point((99, 20), fill=0)
    image.save(image_path)

    result = measure_image(image_path)

    assert result.ink_bands == (
        image_measurement.InkBandMeasurement(
            y_start=10,
            y_end=13,
            x_start=25,
            x_end=35,
            foreground_pixels=30,
        ),
    )
    assert result.median_ink_band_height == 3.0
    assert result.median_ink_band_top_pitch is None
    assert result.diagnostics == ("one_band",)


def test_horizontal_borders_are_diagnostic_without_becoming_ink_bands(tmp_path: Path) -> None:
    image_path = tmp_path / "horizontal-border.png"
    image = Image.new("L", (100, 100), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((0, 0, 99, 0), fill=0)
    draw.line((0, 99, 99, 99), fill=0)
    image.save(image_path)

    result = measure_image(image_path)

    assert result.ink_bands == ()
    assert result.diagnostics == ("border_dominated",)


def test_vertical_borders_are_diagnostic(tmp_path: Path) -> None:
    image_path = tmp_path / "vertical-border.png"
    image = Image.new("L", (100, 100), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((0, 0, 0, 99), fill=0)
    draw.line((99, 0, 99, 99), fill=0)
    image.save(image_path)

    result = measure_image(image_path)

    assert result.ink_bands == (
        image_measurement.InkBandMeasurement(
            y_start=0,
            y_end=100,
            x_start=0,
            x_end=100,
            foreground_pixels=200,
        ),
    )
    assert result.diagnostics == ("one_band", "border_dominated")


def test_one_dark_edge_and_text_near_an_edge_are_not_border_dominated(tmp_path: Path) -> None:
    single_edge_path = tmp_path / "single-edge.png"
    single_edge = Image.new("L", (100, 100), color=255)
    ImageDraw.Draw(single_edge).line((0, 0, 99, 0), fill=0)
    single_edge.save(single_edge_path)

    edge_text_path = tmp_path / "edge-text.png"
    edge_text = Image.new("L", (100, 100), color=255)
    ImageDraw.Draw(edge_text).rectangle((10, 0, 39, 2), fill=0)
    edge_text.save(edge_text_path)

    single_edge_result = measure_image(single_edge_path)
    edge_text_result = measure_image(edge_text_path)

    assert "border_dominated" not in single_edge_result.diagnostics
    assert edge_text_result.ink_bands == (
        image_measurement.InkBandMeasurement(
            y_start=0,
            y_end=3,
            x_start=10,
            x_end=40,
            foreground_pixels=90,
        ),
    )
    assert "border_dominated" not in edge_text_result.diagnostics


def test_high_foreground_ratio_requires_more_than_sixty_percent(tmp_path: Path) -> None:
    image_path = tmp_path / "dense.png"
    image = Image.new("L", (10, 10), color=255)
    ImageDraw.Draw(image).rectangle((0, 0, 9, 5), fill=0)
    image.save(image_path)

    result = measure_image(image_path)

    assert result.foreground_pixels == 60
    assert "high_foreground_ratio" not in result.diagnostics


def test_grayscale_image_hashes_original_bytes_and_uses_otsu_threshold(tmp_path: Path) -> None:
    image_path = tmp_path / "gray.png"
    image = Image.new("L", (6, 4), color=255)
    ImageDraw.Draw(image).rectangle((2, 1, 3, 2), fill=0)
    image.save(image_path)

    result = measure_image(image_path)

    assert result.sha256 == sha256(image_path.read_bytes()).hexdigest()
    assert sha256_file(image_path) == result.sha256
    assert result.grayscale_threshold == 0
    assert result.foreground_pixels == 4
    assert result.foreground_bounds == (2, 1, 4, 3)
    assert result.margins == (2, 1, 2, 1)


def test_rgb_image_uses_luminance_foreground_in_raw_coordinates(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    image = Image.new("RGB", (9, 7), color=(255, 255, 255))
    ImageDraw.Draw(image).rectangle((4, 2, 6, 4), fill=(0, 0, 0))
    image.save(image_path)

    result = measure_image(image_path)

    assert result.image_mode == "RGB"
    assert result.foreground_pixels == 9
    assert result.foreground_bounds == (4, 2, 7, 5)


def test_rgba_image_composites_transparency_over_white(tmp_path: Path) -> None:
    image_path = tmp_path / "rgba.png"
    image = Image.new("RGBA", (8, 6), color=(0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((1, 2, 2, 3), fill=(0, 0, 0, 128))
    image.save(image_path)

    result = measure_image(image_path)

    assert result.image_mode == "RGBA"
    assert result.foreground_pixels == 4
    assert result.foreground_bounds == (1, 2, 3, 4)


def test_palette_image_preserves_mode_and_measures_palette_ink(tmp_path: Path) -> None:
    image_path = tmp_path / "palette.png"
    image = Image.new("P", (8, 6), color=0)
    image.putpalette([255, 255, 255, 0, 0, 0] + [0] * 762)
    ImageDraw.Draw(image).rectangle((3, 1, 4, 4), fill=1)
    image.save(image_path)

    result = measure_image(image_path)

    assert result.image_mode == "P"
    assert result.foreground_pixels == 8
    assert result.foreground_bounds == (3, 1, 5, 5)


def test_two_rectangles_report_half_open_tight_source_bounds(tmp_path: Path) -> None:
    image_path = tmp_path / "rectangles.png"
    image = Image.new("L", (40, 30), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 4, 12, 10), fill=0)
    draw.rectangle((25, 18, 34, 25), fill=0)
    image.save(image_path)

    result = measure_image(image_path)

    assert result.foreground_pixels == 136
    assert result.foreground_bounds == (5, 4, 35, 26)
    assert result.margins == (5, 4, 5, 4)


def test_bounds_avoid_full_page_coordinate_index_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "dense.png"
    Image.new("L", (20, 10), color=0).save(image_path)

    def fail_nonzero(*args: object, **kwargs: object) -> None:
        raise AssertionError("Foreground bounds must not materialize pixel coordinate arrays.")

    monkeypatch.setattr(image_measurement.np, "nonzero", fail_nonzero)

    result = measure_image(image_path)

    assert result.foreground_bounds == (0, 0, 20, 10)


def test_exif_orientation_is_metadata_and_does_not_rotate_source_coordinates(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "oriented.jpg"
    image = Image.new("L", (12, 8), color=255)
    ImageDraw.Draw(image).rectangle((1, 2, 3, 4), fill=0)
    exif = Image.Exif()
    exif[274] = 6
    image.save(image_path, exif=exif, quality=100, subsampling=0)

    result = measure_image(image_path)

    assert result.source_frame.width == 12
    assert result.source_frame.height == 8
    assert result.exif_orientation == 6
    assert result.foreground_bounds == (1, 2, 4, 5)


def test_decompression_bomb_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "small.png"
    Image.new("L", (2, 2), color=255).save(image_path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    with pytest.raises(ValueError, match="decompression bomb"):
        _ = measure_image(image_path)
