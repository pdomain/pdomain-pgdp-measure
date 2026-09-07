"""Tests for rendering and re-checking the per-character glyph atlas."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from pdomain_pgdp_measure.glyph_atlas import (
    AtlasError,
    render_atlas,
    sheet_path,
    verify_atlas,
    write_atlas,
)
from pdomain_pgdp_measure.glyph_models import GlyphManifest, GlyphPage
from pdomain_pgdp_measure.glyphs import build_glyph_inventory, write_glyph_inventory
from tests.test_pgdp_glyphs import Fixture, build_glyph_fixture


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return build_glyph_fixture(tmp_path_factory.mktemp("glyph-atlas"))


@pytest.fixture(scope="module")
def inventory(fixture: Fixture, tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("glyph-atlas-out") / "inventory"
    harvest = build_glyph_inventory(
        fixture.corpus_root,
        fixture.alignment_path,
        fixture.profile_path,
        tool_version="0.0.0",
        geometry_path=fixture.geometry_path,
    )
    _ = write_glyph_inventory(harvest, directory, fixture.corpus_root)
    return directory


def test_a_sheet_path_names_the_tier_the_style_and_the_character() -> None:
    assert sheet_path(tier="transcribed", style="roman", character="e", ordinal=0) == (
        "atlas/transcribed/roman/U+0065-000.png"
    )
    assert sheet_path(tier="transcribed", style="small_caps", character="o", ordinal=2) == (
        "atlas/transcribed/small_caps/U+006F-002.png"
    )


def test_every_character_and_style_gets_a_sheet(inventory: Path) -> None:
    manifest = GlyphManifest.from_json((inventory / "manifest.json").read_bytes())
    buckets = {(row.label_tier, row.label_style, row.character) for row in manifest.coverage}
    assert {
        (sheet.label_tier, sheet.label_style, sheet.character) for sheet in manifest.atlas
    } == buckets


def test_a_sheet_holds_one_cell_per_glyph(inventory: Path) -> None:
    manifest = GlyphManifest.from_json((inventory / "manifest.json").read_bytes())
    cells: dict[tuple[str, str | None, str], int] = {}
    for sheet in manifest.atlas:
        key = (sheet.label_tier, sheet.label_style, sheet.character)
        cells[key] = cells.get(key, 0) + sheet.cell_count
    for row in manifest.coverage:
        assert cells[row.label_tier, row.label_style, row.character] == row.glyph_count


def test_no_sheet_mixes_two_styles_of_one_character(inventory: Path) -> None:
    manifest = GlyphManifest.from_json((inventory / "manifest.json").read_bytes())
    styles = {sheet.label_style for sheet in manifest.atlas if sheet.label_tier == "transcribed"}
    assert "italic" in styles
    assert "roman" in styles
    for sheet in manifest.atlas:
        assert sheet.label_style is None or sheet.label_style in sheet.path


def test_every_sheet_lands_on_disk_with_the_hash_the_manifest_records(inventory: Path) -> None:
    manifest = GlyphManifest.from_json((inventory / "manifest.json").read_bytes())
    for sheet in manifest.atlas:
        payload = (inventory / sheet.path).read_bytes()
        assert sha256(payload).hexdigest() == sheet.sha256


def test_a_sheet_is_a_grid_of_its_recorded_cell_size(inventory: Path) -> None:
    manifest = GlyphManifest.from_json((inventory / "manifest.json").read_bytes())
    sheet = manifest.atlas[0]
    with Image.open(inventory / sheet.path) as image:
        columns = min(32, sheet.cell_count)
        assert image.width == columns * (sheet.cell_width_px + 1) + 1
        assert image.mode == "L"


def test_re_rendering_the_atlas_reproduces_it_byte_for_byte(
    inventory: Path, fixture: Fixture
) -> None:
    assert verify_atlas(inventory, fixture.corpus_root) == ()


def test_a_tampered_sheet_is_named_by_the_check(
    inventory: Path, fixture: Fixture, tmp_path: Path
) -> None:
    copied = tmp_path / "copy"
    copied.mkdir()
    for source in inventory.rglob("*"):
        target = copied / source.relative_to(inventory)
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            _ = target.write_bytes(source.read_bytes())
    manifest = GlyphManifest.from_json((copied / "manifest.json").read_bytes())
    tampered = copied / manifest.atlas[0].path
    _ = tampered.write_bytes(tampered.read_bytes() + b"\x00")
    assert verify_atlas(copied, fixture.corpus_root) == (manifest.atlas[0].path,)


def test_rendering_is_stable_across_two_passes(fixture: Fixture) -> None:
    harvest = build_glyph_inventory(
        fixture.corpus_root,
        fixture.alignment_path,
        fixture.profile_path,
        tool_version="0.0.0",
    )
    first = render_atlas(harvest.rows, corpus_root=fixture.corpus_root, pages=harvest.pages)
    second = render_atlas(harvest.rows, corpus_root=fixture.corpus_root, pages=harvest.pages)
    assert [sheet.payload for sheet in first] == [sheet.payload for sheet in second]


def test_a_page_missing_from_the_table_is_refused(fixture: Fixture) -> None:
    harvest = build_glyph_inventory(
        fixture.corpus_root,
        fixture.alignment_path,
        fixture.profile_path,
        tool_version="0.0.0",
    )
    with pytest.raises(AtlasError, match="carries no page"):
        _ = render_atlas(harvest.rows, corpus_root=fixture.corpus_root, pages=())


def test_a_scan_that_changed_since_harvest_is_refused(fixture: Fixture) -> None:
    harvest = build_glyph_inventory(
        fixture.corpus_root,
        fixture.alignment_path,
        fixture.profile_path,
        tool_version="0.0.0",
    )
    pages = tuple(
        GlyphPage(
            page_name=page.page_name,
            scan_sha256="d" * 64,
            source_path=page.source_path,
            glyph_count=page.glyph_count,
        )
        for page in harvest.pages
    )
    with pytest.raises(AtlasError, match="has changed since harvest"):
        _ = render_atlas(harvest.rows, corpus_root=fixture.corpus_root, pages=pages)


def test_skipping_the_atlas_writes_no_sheets(fixture: Fixture, tmp_path: Path) -> None:
    harvest = build_glyph_inventory(
        fixture.corpus_root,
        fixture.alignment_path,
        fixture.profile_path,
        tool_version="0.0.0",
    )
    directory = tmp_path / "no-atlas"
    manifest = write_glyph_inventory(harvest, directory, fixture.corpus_root, atlas=False)
    assert manifest.atlas == ()
    assert not (directory / "atlas").exists()


def test_written_sheets_carry_the_records_that_were_returned(
    fixture: Fixture, tmp_path: Path
) -> None:
    harvest = build_glyph_inventory(
        fixture.corpus_root,
        fixture.alignment_path,
        fixture.profile_path,
        tool_version="0.0.0",
    )
    rendered = render_atlas(harvest.rows, corpus_root=fixture.corpus_root, pages=harvest.pages)
    directory = tmp_path / "sheets"
    directory.mkdir()
    written = write_atlas(rendered, directory)
    assert written == tuple(sheet.sheet for sheet in rendered)
