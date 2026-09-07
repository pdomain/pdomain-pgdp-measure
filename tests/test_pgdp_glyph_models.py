"""Tests for the `pgdp-glyphs/v1` manifest and per-glyph JSONL contract."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pdomain_pgdp_measure.glyph_models import (
    GLYPHS_ALGORITHM_VERSION,
    GLYPHS_SCHEMA_VERSION,
    AtlasSheet,
    CharacterCoverage,
    GlyphContractError,
    GlyphManifest,
    GlyphPage,
    GlyphRow,
    read_rows,
    render_rows,
)
from pdomain_pgdp_measure.typography_models import find_latent_discipline_violations

MANIFEST_SCHEMA_ARTIFACT = Path("schemas/pgdp-glyphs-v1.schema.json")
ROW_SCHEMA_ARTIFACT = Path("schemas/pgdp-glyphs-row-v1.schema.json")

_SHA = "a" * 64
_OTHER_SHA = "b" * 64


def _row(
    character: str = "e",
    *,
    label_tier: str = "transcribed",
    page_name: str = "p010.png",
    line_ordinal: int = 3,
    word_ordinal: int = 2,
    glyph_ordinal: int = 0,
    label_confidence: float | None = None,
) -> GlyphRow:
    return GlyphRow(
        character=character,
        label_tier="recognized" if label_tier == "recognized" else "transcribed",
        page_name=page_name,
        line_ordinal=line_ordinal,
        word_ordinal=word_ordinal,
        glyph_ordinal=glyph_ordinal,
        box=(10, 20, 18, 34),
        ink_density=0.42,
        row_extent_px=14,
        top_offset_px=0,
        bottom_offset_px=0,
        flags=("ascends",),
        label_confidence=label_confidence,
    )


def _manifest() -> GlyphManifest:
    return GlyphManifest(
        tool_version="0.1.0",
        project_id="projectIDabc",
        alignment_label="alignment.json",
        alignment_sha256=_SHA,
        profile_label="profile.json",
        profile_sha256=_OTHER_SHA,
        rows_label="glyphs.jsonl",
        rows_sha256="c" * 64,
        methods={"algorithm_version": GLYPHS_ALGORITHM_VERSION},
        thresholds={"word_gap_threshold_px": 9},
        page_count=40,
        accepted_page_count=32,
        harvested_page_count=34,
        reconciled_word_count=900,
        separable_word_count=540,
        furniture_word_count=117,
        glyph_count_by_tier={"transcribed": 3, "recognized": 1},
        coverage=(
            CharacterCoverage(character="e", label_tier="transcribed", glyph_count=2),
            CharacterCoverage(character="a", label_tier="transcribed", glyph_count=1),
            CharacterCoverage(character="4", label_tier="recognized", glyph_count=1),
        ),
        quality_flag_counts={"ascends": 2},
        word_reject_counts={"run_count_disagrees": 360},
        pages=(
            GlyphPage(
                page_name="p010.png",
                scan_sha256=_SHA,
                source_path="projectIDabc/p010.png",
                glyph_count=4,
            ),
        ),
    )


def test_a_row_round_trips_through_its_json_line() -> None:
    row = _row()
    assert GlyphRow.from_json(row.to_json()) == row


def test_a_recognized_row_carries_its_read_confidence() -> None:
    row = _row(character="4", label_tier="recognized", label_confidence=0.79)
    assert GlyphRow.from_json(row.to_json()).label_confidence == 0.79


def test_a_transcribed_row_may_not_claim_a_confidence() -> None:
    with pytest.raises(ValueError, match="human proofer"):
        _ = _row(label_confidence=0.9)


def test_rows_render_in_page_then_ordinal_order() -> None:
    rows = [
        _row(page_name="p010.png", glyph_ordinal=1),
        _row(page_name="p002.png", glyph_ordinal=0),
        _row(page_name="p010.png", glyph_ordinal=0),
    ]
    rendered = render_rows(rows)
    read_back = read_rows(rendered)
    assert [(row.page_name, row.glyph_ordinal) for row in read_back] == [
        ("p002.png", 0),
        ("p010.png", 0),
        ("p010.png", 1),
    ]


def test_rendering_is_stable_under_input_order() -> None:
    rows = [_row(glyph_ordinal=index) for index in range(4)]
    assert render_rows(rows) == render_rows(list(reversed(rows)))


def test_two_rows_may_not_claim_the_same_glyph() -> None:
    with pytest.raises(GlyphContractError, match="same page, line, word and ordinal"):
        _ = render_rows([_row(), _row(character="a")])


def test_the_two_tiers_may_share_ordinals_without_colliding() -> None:
    rows = [_row(), _row(character="4", label_tier="recognized", label_confidence=0.8)]
    assert len(read_rows(render_rows(rows))) == 2


def test_a_manifest_round_trips_through_json() -> None:
    manifest = _manifest()
    assert GlyphManifest.from_json(manifest.to_json()) == manifest


def test_a_manifest_reports_its_total_glyph_count() -> None:
    assert _manifest().glyph_count == 4
    assert json.loads(_manifest().to_json())["glyph_count"] == 4


def test_coverage_must_agree_with_the_tier_counts() -> None:
    with pytest.raises(ValueError, match="disagrees with its glyph count"):
        _ = replace(_manifest(), glyph_count_by_tier={"transcribed": 2, "recognized": 1})


def test_a_book_may_not_harvest_more_pages_than_it_has() -> None:
    with pytest.raises(ValueError, match="more pages than it has"):
        _ = replace(_manifest(), harvested_page_count=41)


def test_a_book_may_harvest_more_pages_than_it_accepted() -> None:
    """A line on a non-accepted page is admitted on its own evidence, not its page's."""

    assert _manifest().harvested_page_count > _manifest().accepted_page_count


def test_an_accepted_page_may_not_claim_recognizer_admitted_lines() -> None:
    with pytest.raises(ValueError, match="admitted by its own alignment gate"):
        _ = GlyphPage(
            page_name="p010.png",
            scan_sha256=_SHA,
            source_path="projectIDabc/p010.png",
            glyph_count=4,
            page_state="accepted",
            admitted_line_count=3,
            recognizer_admitted_line_count=1,
        )


def test_a_page_may_not_admit_more_lines_by_recognizer_than_in_total() -> None:
    with pytest.raises(ValueError, match="more lines by recognizer than in total"):
        _ = GlyphPage(
            page_name="p011.png",
            scan_sha256=_SHA,
            source_path="projectIDabc/p011.png",
            glyph_count=4,
            page_state="excluded",
            admitted_line_count=1,
            recognizer_admitted_line_count=2,
        )


def test_an_excluded_page_records_how_its_lines_got_in() -> None:
    page = GlyphPage(
        page_name="p012.png",
        scan_sha256=_SHA,
        source_path="projectIDabc/p012.png",
        glyph_count=9,
        page_state="excluded",
        admitted_line_count=4,
        recognizer_admitted_line_count=4,
    )
    assert page.to_dict()["page_state"] == "excluded"
    assert page.to_dict()["recognizer_admitted_line_count"] == 4


def test_a_book_may_not_separate_more_words_than_it_reconciled() -> None:
    with pytest.raises(ValueError, match="more words than it reconciled"):
        _ = replace(_manifest(), separable_word_count=1000)


def test_a_geometry_record_is_named_and_hashed_together() -> None:
    with pytest.raises(ValueError, match="together or not at all"):
        _ = replace(_manifest(), geometry_label="geometry.jsonl")


def test_a_recognizer_identity_requires_its_geometry_record() -> None:
    with pytest.raises(ValueError, match="requires the geometry record"):
        _ = replace(
            _manifest(), ocr_recognizer={"name": "doctr", "version": "1", "config_sha256": _SHA}
        )


def test_a_manifest_with_a_geometry_record_round_trips() -> None:
    manifest = replace(
        _manifest(),
        geometry_label="geometry.jsonl",
        geometry_sha256=_SHA,
        ocr_recognizer={"name": "doctr", "version": "1", "config_sha256": _OTHER_SHA},
    )
    assert GlyphManifest.from_json(manifest.to_json()).ocr_recognizer == {
        "config_sha256": _OTHER_SHA,
        "name": "doctr",
        "version": "1",
    }


def test_an_atlas_sheet_path_must_stay_inside_the_inventory() -> None:
    with pytest.raises(ValueError, match="relative to the inventory directory"):
        _ = AtlasSheet(
            character="e",
            label_tier="transcribed",
            sheet_ordinal=0,
            path="../U+0065-000.png",
            cell_count=4,
            cell_width_px=8,
            cell_height_px=14,
            sha256=_SHA,
        )


def test_the_manifest_names_nothing_the_design_treats_as_latent() -> None:
    payload = json.loads(_manifest().to_json())
    assert find_latent_discipline_violations(payload) == ()


def test_a_row_names_nothing_the_design_treats_as_latent() -> None:
    assert find_latent_discipline_violations(json.loads(_row().to_json())) == ()


def test_an_unsupported_schema_version_is_refused() -> None:
    with pytest.raises(ValueError, match="Unsupported schema_version"):
        _ = replace(_manifest(), schema_version=2)


def test_the_shipped_manifest_schema_matches_the_model() -> None:
    assert MANIFEST_SCHEMA_ARTIFACT.read_text(encoding="utf-8") == GlyphManifest.json_schema()


def test_the_shipped_row_schema_matches_the_model() -> None:
    assert ROW_SCHEMA_ARTIFACT.read_text(encoding="utf-8") == GlyphRow.json_schema()


def test_the_schema_version_and_algorithm_version_are_pinned() -> None:
    assert (GLYPHS_SCHEMA_VERSION, GLYPHS_ALGORITHM_VERSION) == (1, "pgdp-glyphs/v1")
