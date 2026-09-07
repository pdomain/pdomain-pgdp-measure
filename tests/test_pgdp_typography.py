"""Tests for the typography orchestration stage.

The fixture draws a page whose geometry is known exactly: a 4 px letter stroke on an 8 px
pitch, an x-height band 19 px tall above the baseline, ascenders 8 px higher, descenders 8 px
lower, and a 65 px line pitch. Every asserted number below is that drawing read back.

The alignment report is built by hand from real extracted candidates rather than by running
the aligner. Whether the M15b dynamic program accepts a synthetic page is M15b's question and
has its own tests; what this file tests is that the typography stage measures, verifies, and
pools an accepted page correctly.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

import pytest
from PIL import Image

from pdomain_pgdp_measure.alignment_image import extract_line_candidates
from pdomain_pgdp_measure.alignment_models import (
    AlignmentOperation,
    AlignmentReport,
    PageAlignment,
    ProjectAlignment,
    WireLineCandidate,
    WireSourceLine,
)
from pdomain_pgdp_measure.image_measurement import open_image_snapshot
from pdomain_pgdp_measure.profile_models import ProfileReport
from pdomain_pgdp_measure.typography import ProvenanceError, build_typography_report
from pdomain_pgdp_measure.typography_models import TypographyReport

LINE_COUNT = 6
TEXT_LEFT = 100
TEXT_RIGHT = 1000
LETTER_PITCH = 8
LETTER_WIDTH = 4
FIRST_LINE_TOP = 30
LINE_PITCH = 65
GAP_SLOTS = 2
X_HEIGHT_PX = 19
ASCENDER_EXTENT_PX = 27
DESCENDER_EXTENT_PX = 9
WORD_GAP_PX = GAP_SLOTS * LETTER_PITCH + LETTER_WIDTH


class Fixture(NamedTuple):
    """A drawn corpus with a real profile and a hand-built accepted alignment."""

    corpus_root: Path
    profile_path: Path
    alignment_path: Path
    word_counts: tuple[int, ...]


def _gap_slots(line: int, slot_count: int) -> set[int]:
    """Word-gap slots for one line, staggered so no column is blank on every line.

    Aligned gaps would read as a column gutter and a ragged right edge would read as one
    too, so every line runs the full measure and puts its gaps somewhere different.
    """

    starts = range(9 + (line * 5) % 7, slot_count - 10, 13 + line)
    return {slot for start in starts for slot in range(start, start + GAP_SLOTS)}


def _draw_line(image: Image.Image, top: int, line: int) -> int:
    slot_count = (TEXT_RIGHT - TEXT_LEFT) // LETTER_PITCH
    gaps = _gap_slots(line, slot_count)
    words = 1
    after_gap = False
    for slot in range(slot_count):
        if slot in gaps:
            after_gap = True
            continue
        if after_gap:
            words += 1
            after_gap = False
        y_start = top + (2 if slot % 3 == 0 else 10)
        y_end = top + (38 if slot % 4 == 3 else 30)
        x = TEXT_LEFT + slot * LETTER_PITCH
        for column in range(x, x + LETTER_WIDTH):
            for row in range(y_start, y_end):
                image.putpixel((column, row), 0)
    return words


def _source_line(ordinal: int, text: str, byte_start: int) -> WireSourceLine:
    encoded = len(text.encode("utf-8"))
    content_end = byte_start + encoded
    return WireSourceLine(
        ordinal=ordinal,
        byte_start=byte_start,
        byte_end=content_end + 1,
        content_byte_start=byte_start,
        content_byte_end=content_end,
        separator_byte_start=content_end,
        separator_byte_end=content_end + 1,
        original_text=text,
        separator_text="\n",
        visible_text=text,
        matching_eligible=True,
        style_fitting_eligible=True,
    )


def build_fixture(tmp_path: Path) -> Fixture:
    """Draw two identical pages, profile them for real, and accept them 1:1."""

    from pdomain_pgdp_measure.cli import main

    corpus_root = tmp_path / "corpus"
    project = corpus_root / "projectIDone"
    (project / "rounds").mkdir(parents=True)
    (project / "project.json").write_text(
        json.dumps(
            {
                "projectid": "projectIDone",
                "title": "A book",
                "author": "An author",
                "genre": "Fiction",
                "pages_total": 2,
                "pg_ebook_number": 1,
            }
        ),
        encoding="utf-8",
    )
    image = Image.new("L", (1100, 460), color=255)
    word_counts = tuple(
        _draw_line(image, FIRST_LINE_TOP + line * LINE_PITCH, line) for line in range(LINE_COUNT)
    )
    (project / "rounds" / "F2.json").write_text(
        json.dumps({"p2.png": "placeholder", "p10.png": "placeholder"}), encoding="utf-8"
    )
    for name in ("p2.png", "p10.png"):
        image.save(project / name)

    ranking_path = tmp_path / "ranking.json"
    assert (
        main(
            [
                "rank",
                str(corpus_root),
                "--output",
                str(ranking_path),
                "--project-limit",
                "1",
                "--pages-per-project",
                "2",
            ]
        )
        == 0
    )
    profile_path = tmp_path / "profile.json"
    assert (
        main(
            [
                "profile",
                str(corpus_root),
                "--ranking",
                str(ranking_path),
                "--output",
                str(profile_path),
            ]
        )
        == 0
    )
    alignment_path = tmp_path / "alignment.json"
    _write_accepted_alignment(
        corpus_root=corpus_root,
        profile_path=profile_path,
        alignment_path=alignment_path,
        word_counts=word_counts,
    )
    return Fixture(corpus_root, profile_path, alignment_path, word_counts)


def _write_accepted_alignment(
    *,
    corpus_root: Path,
    profile_path: Path,
    alignment_path: Path,
    word_counts: tuple[int, ...],
) -> None:
    profile_bytes = profile_path.read_bytes()
    profile = ProfileReport.from_json(profile_bytes)
    project_profile = profile.projects[0]
    pages: list[PageAlignment] = []
    for measurement in project_profile.pages:
        source_frame = measurement.source_frame
        bounds = measurement.foreground_bounds
        assert source_frame is not None
        assert bounds is not None
        with open_image_snapshot(corpus_root / measurement.source_path) as snapshot:
            extraction = extract_line_candidates(
                snapshot,
                source_frame=source_frame,
                foreground_bounds=tuple(bounds),
                ink_bands=measurement.ink_bands,
                furniture_band_ordinals=measurement.furniture_band_ordinals,
            )
        candidates: list[WireLineCandidate] = []
        source_lines: list[WireSourceLine] = []
        operations: list[AlignmentOperation] = []
        byte_offset = 0
        for ordinal, candidate in enumerate(extraction.candidates):
            candidates.append(
                WireLineCandidate(
                    ordinal=ordinal,
                    box=candidate.box,
                    width=candidate.width,
                    height=candidate.height,
                    fill_ratio=candidate.fill_ratio,
                    component_count=candidate.component_count,
                    foreground_pixels=candidate.foreground_pixels,
                    band_ordinal=candidate.band_ordinal,
                    component_ordinal=candidate.component_ordinal,
                    horizontal_ink_profile=candidate.horizontal_ink_profile,
                )
            )
            line_index = round((candidate.box[1] - FIRST_LINE_TOP) / LINE_PITCH)
            text = " ".join(f"w{index}" for index in range(word_counts[line_index]))
            source_lines.append(_source_line(ordinal, text, byte_offset))
            byte_offset = source_lines[-1].byte_end
            operations.append(
                AlignmentOperation(
                    kind="match", source_ordinal=ordinal, candidate_ordinal=ordinal, cost=0.0
                )
            )
        pages.append(
            PageAlignment(
                page_name=measurement.page_name,
                f2_sha256="a" * 64,
                scan_sha256=measurement.sha256,
                source_frame=source_frame,
                source_lines=tuple(source_lines),
                formatting_spans=(),
                candidates=tuple(candidates),
                operations=tuple(operations),
                total_cost=0.0,
                second_best_cost=1.0,
                normalized_cost=0.0,
                matched_count=len(operations),
                matched_ratio=1.0,
                uniqueness_margin=1.0,
                mean_width_residual=0.0,
                mean_indentation_residual=0.0,
                state="accepted",
                accepted=True,
                exclusions=(),
            )
        )
    report = AlignmentReport(
        tool_version="0.0.0",
        profile_label=profile_path.name,
        profile_sha256=sha256(profile_bytes).hexdigest(),
        methods={},
        thresholds={},
        projects=(ProjectAlignment(project_id="projectIDone", pages=tuple(pages)),),
    )
    alignment_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _build(fixture: Fixture, **overrides: object) -> TypographyReport:
    arguments: dict[str, object] = {"tool_version": "0.0.0", "evidence_pages": 12}
    arguments.update(overrides)
    return build_typography_report(
        fixture.corpus_root,
        fixture.alignment_path,
        fixture.profile_path,
        tool_version=str(arguments["tool_version"]),
        evidence_pages=int(str(arguments["evidence_pages"])),
    )


def _estimate_value(report: TypographyReport, name: str) -> float | None:
    for estimate in report.books[0].estimates:
        if estimate.name == name:
            return estimate.value
    raise AssertionError(f"No book estimate named {name!r}.")


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return build_fixture(tmp_path_factory.mktemp("typography"))


# --- Provenance --------------------------------------------------------------------------


def test_a_profile_that_does_not_hash_to_the_recorded_value_is_refused(
    fixture: Fixture, tmp_path: Path
) -> None:
    tampered = tmp_path / "profile.json"
    payload = json.loads(fixture.profile_path.read_text(encoding="utf-8"))
    payload["projects"][0]["title"] = "A different book"
    tampered.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="recorded profile_sha256"):
        _ = build_typography_report(
            fixture.corpus_root, fixture.alignment_path, tampered, tool_version="0.0.0"
        )


def test_report_records_both_ends_of_the_hash_chain(fixture: Fixture) -> None:
    report = _build(fixture)
    assert report.alignment_label == "alignment.json"
    assert report.alignment_sha256 == sha256(fixture.alignment_path.read_bytes()).hexdigest()
    assert report.profile_label == "profile.json"
    assert report.profile_sha256 == sha256(fixture.profile_path.read_bytes()).hexdigest()


# --- Measurement -------------------------------------------------------------------------


def test_every_matched_line_on_every_accepted_page_is_measured(fixture: Fixture) -> None:
    book = _build(fixture).books[0]
    assert book.accepted_page_count == 2
    assert book.measured_page_count == 2
    assert book.excluded_page_count == 0
    assert book.matched_line_count == book.measured_line_count > 0
    assert all(page.exclusions == () for page in book.pages)


def test_pooled_values_are_the_geometry_the_fixture_drew(fixture: Fixture) -> None:
    report = _build(fixture)
    assert _estimate_value(report, "x_height_px") == X_HEIGHT_PX
    assert _estimate_value(report, "ascender_extent_px") == ASCENDER_EXTENT_PX
    assert _estimate_value(report, "descender_extent_px") == DESCENDER_EXTENT_PX
    assert _estimate_value(report, "stroke_width_px") == LETTER_WIDTH
    assert _estimate_value(report, "baseline_pitch_px") == LINE_PITCH
    assert _estimate_value(report, "word_gap_px") == WORD_GAP_PX
    assert _estimate_value(report, "line_indent_px") == 0.0


def test_word_runs_reconcile_against_the_transcription(fixture: Fixture) -> None:
    book = _build(fixture).books[0]
    assert book.reconciled_line_count == book.measured_line_count
    page = book.pages[0]
    for line in page.lines:
        assert line.words_reconciled is True
        assert line.word_run_count == line.source_word_count
        assert len(line.words) == line.word_run_count
        assert all(word.following_gap_px == WORD_GAP_PX for word in line.words[:-1])
        assert line.words[-1].following_gap_px is None


def test_baseline_pitch_uses_ordinal_adjacent_pairs_only(fixture: Fixture) -> None:
    page = _build(fixture).books[0].pages[0]
    assert page.baseline_pitch_count == page.measured_line_count - 1


def test_pages_pool_into_the_body_style(fixture: Fixture) -> None:
    book = _build(fixture).books[0]
    assert tuple(style.style for style in book.styles) == ("body",)
    assert book.styles[0].page_count == 2
    assert book.unknown_class_page_count == 0


# --- Evidence bounding -------------------------------------------------------------------


def test_evidence_rows_go_to_the_first_measured_pages_in_natural_order(
    fixture: Fixture,
) -> None:
    book = _build(fixture, evidence_pages=1).books[0]
    assert [page.page_name for page in book.pages] == ["p2.png", "p10.png"]
    assert book.pages[0].evidence_sampled is True
    assert book.pages[1].evidence_sampled is False
    assert book.pages[1].lines == ()


def test_zero_evidence_pages_still_pools_every_page(fixture: Fixture) -> None:
    book = _build(fixture, evidence_pages=0).books[0]
    assert all(page.lines == () for page in book.pages)
    assert book.measured_line_count > 0
    assert _estimate_value(_build(fixture, evidence_pages=0), "x_height_px") == X_HEIGHT_PX


def test_unmeasured_pages_still_carry_their_counts(fixture: Fixture) -> None:
    book = _build(fixture, evidence_pages=1).books[0]
    unsampled = book.pages[1]
    assert unsampled.matched_line_count > 0
    assert unsampled.measured_line_count > 0
    assert unsampled.estimates


# --- Mask reproduction -------------------------------------------------------------------


def test_a_candidate_whose_ink_the_mask_cannot_reproduce_excludes_its_page(
    fixture: Fixture, tmp_path: Path
) -> None:
    tampered = tmp_path / "alignment.json"
    payload = json.loads(fixture.alignment_path.read_text(encoding="utf-8"))
    candidate = payload["projects"][0]["pages"][0]["candidates"][0]
    candidate["foreground_pixels"] += 1
    tampered.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    book = build_typography_report(
        fixture.corpus_root, tampered, fixture.profile_path, tool_version="0.0.0"
    ).books[0]
    page = book.pages[0]
    assert page.exclusions == ("mask_mismatch",)
    assert page.exclusion_evidence[0]["code"] == "foreground_pixels_mismatch"
    assert page.exclusion_evidence[0]["rebuilt"] != page.exclusion_evidence[0]["recorded"]
    assert book.excluded_page_count == 1
    assert book.measured_page_count == 1


def test_a_rebuilt_threshold_that_differs_excludes_its_page(
    fixture: Fixture, tmp_path: Path
) -> None:
    profile_payload = json.loads(fixture.profile_path.read_text(encoding="utf-8"))
    for page in profile_payload["projects"][0]["pages"]:
        page["observations"]["grayscale_threshold"] = 200
    tampered_profile = tmp_path / "profile.json"
    tampered_profile.write_text(
        json.dumps(profile_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    alignment_payload = json.loads(fixture.alignment_path.read_text(encoding="utf-8"))
    alignment_payload["profile_sha256"] = sha256(tampered_profile.read_bytes()).hexdigest()
    tampered_alignment = tmp_path / "alignment.json"
    tampered_alignment.write_text(
        json.dumps(alignment_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    book = build_typography_report(
        fixture.corpus_root, tampered_alignment, tampered_profile, tool_version="0.0.0"
    ).books[0]
    assert all(page.exclusions == ("mask_mismatch",) for page in book.pages)
    assert book.pages[0].exclusion_evidence[0]["code"] == "grayscale_threshold_mismatch"
    assert book.pages[0].exclusion_evidence[0]["rebuilt"] == 0
    assert book.pages[0].exclusion_evidence[0]["recorded"] == 200


# --- Determinism and discipline ----------------------------------------------------------


def test_two_runs_produce_byte_identical_json(fixture: Fixture) -> None:
    assert _build(fixture).to_json() == _build(fixture).to_json()


def test_report_round_trips_and_names_nothing_latent(fixture: Fixture) -> None:
    from pdomain_pgdp_measure.typography_models import find_latent_discipline_violations

    report = _build(fixture)
    assert TypographyReport.from_json(report.to_json()) == report
    assert find_latent_discipline_violations(json.loads(report.to_json())) == ()


def test_a_missing_scan_excludes_its_page_without_failing_the_run(
    fixture: Fixture, tmp_path: Path
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "projectIDone").mkdir()
    report = build_typography_report(
        corpus_root, fixture.alignment_path, fixture.profile_path, tool_version="0.0.0"
    )
    book = report.books[0]
    assert all(page.exclusions == ("image_missing",) for page in book.pages)
    assert book.measured_page_count == 0
    assert _estimate_value(report, "x_height_px") is None
