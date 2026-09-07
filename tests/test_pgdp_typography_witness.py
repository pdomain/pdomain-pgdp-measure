"""Tests for the optional OCR witness on `typography-pgdp`.

The geometry records are written by hand from the fixture's own drawn boxes, so no OCR runs and
no model is opened. What is tested is that the witness reaches the report, that it changes
nothing when absent, and that it refuses a record it cannot prove belongs to the page.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from pdomain_pgdp_measure.alignment_models import AlignmentReport
from pdomain_pgdp_measure.ocr_witness import OcrWitnessError
from pdomain_pgdp_measure.typography import build_typography_report
from pdomain_pgdp_measure.typography_measure import find_word_runs
from pdomain_pgdp_measure.typography_models import find_latent_discipline_violations
from tests.test_pgdp_typography import Fixture, build_fixture

if TYPE_CHECKING:
    from pathlib import Path

_RECOGNIZER = {
    "name": "pdomain-doctr-finetuned",
    "version": "8b7f9b31d55015d16febaa99ae1d42960904028c",
    "config_sha256": "b" * 64,
}


def _geometry(
    fixture: Fixture,
    path: Path,
    *,
    first_word_text: str | None = None,
    image_sha256: str | None = None,
) -> Path:
    """Write a geometry record naming every ink run of every matched line.

    Each run is named with the transcription's own word, so by default the witness agrees
    everywhere. `first_word_text` overrides what the recognizer read at each line's first run,
    which is how a continuation fragment is simulated.
    """

    alignment = AlignmentReport.from_json(fixture.alignment_path.read_bytes())
    records: list[dict[str, object]] = []
    for project in alignment.projects:
        for page in project.pages:
            frame = page.source_frame
            assert frame is not None
            candidates = {candidate.ordinal: candidate for candidate in page.candidates}
            sources = {line.ordinal: line for line in page.source_lines}
            words: list[dict[str, object]] = []
            for index, operation in enumerate(page.operations):
                candidate = candidates[operation.candidate_ordinal or 0]
                source = sources[operation.source_ordinal or 0]
                text = source.visible_text.split()
                runs = find_word_runs(candidate.horizontal_ink_profile, 5)
                x_start, y_start, _x_end, y_end = candidate.box
                for run_index, (run_start, run_end) in enumerate(runs):
                    value = text[run_index] if run_index < len(text) else "extra"
                    if run_index == 0 and first_word_text is not None:
                        value = first_word_text
                    words.append(
                        {
                            "text": value,
                            "bbox": {
                                "top_left": {
                                    "x": x_start + run_start,
                                    "y": y_start,
                                    "is_normalized": False,
                                },
                                "bottom_right": {
                                    "x": x_start + run_end,
                                    "y": y_end,
                                    "is_normalized": False,
                                },
                                "is_normalized": False,
                            },
                            "confidence": 0.97,
                            "line_index": index,
                            "word_index": run_index,
                        }
                    )
            records.append(
                {
                    "schema_version": "1.0",
                    "project_id": project.project_id,
                    "page_name": page.page_name,
                    "image_sha256": image_sha256 or page.scan_sha256,
                    "image_width": frame.width,
                    "image_height": frame.height,
                    "confidence_tier": "silver",
                    "recognizer": dict(_RECOGNIZER),
                    "words": words,
                }
            )
    _ = path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _report(fixture: Fixture, geometry: Path | None):
    return build_typography_report(
        fixture.corpus_root,
        fixture.alignment_path,
        fixture.profile_path,
        tool_version="0.0.0-test",
        evidence_pages=8,
        geometry_path=geometry,
    )


def test_a_run_without_geometry_is_byte_identical(tmp_path: Path) -> None:
    """The gate that confines this change: the witness must perturb nothing when absent."""

    fixture = build_fixture(tmp_path)

    without = _report(fixture, None)
    again = _report(fixture, None)

    assert json.dumps(without.to_dict(), sort_keys=True) == json.dumps(
        again.to_dict(), sort_keys=True
    )
    assert "ocr_witness" not in without.methods
    assert "geometry_sha256" not in without.extensions
    for book in without.books:
        assert "reconciled_after_witness" not in book.extensions
        for page in book.pages:
            for line in page.lines:
                assert "continuation_fragment" not in line.extensions


def test_the_witness_records_its_provenance(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    geometry = _geometry(fixture, tmp_path / "geometry.jsonl")

    report = _report(fixture, geometry)

    assert report.methods["ocr_witness"] == "first-run-correspondence/v1"
    assert report.extensions["geometry_label"] == "geometry.jsonl"
    assert len(str(report.extensions["geometry_sha256"])) == 64
    assert report.extensions["ocr_recognizer"] == _RECOGNIZER


def test_an_agreeing_witness_flags_nothing(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    geometry = _geometry(fixture, tmp_path / "geometry.jsonl")

    book = _report(fixture, geometry).books[0]

    assert book.extensions["continuation_fragment_line_count"] == 0
    assert int(str(book.extensions["witnessed_line_count"])) > 0
    for page in book.pages:
        for line in page.lines:
            assert line.extensions["ocr_first_run_matches_source"] is True
            assert line.extensions["continuation_fragment"] is False


def test_a_disagreeing_first_run_is_recorded_but_flags_only_an_over_split(
    tmp_path: Path,
) -> None:
    """The fixture's lines reconcile, so a disagreeing read is an OCR miss, not a fragment."""

    fixture = build_fixture(tmp_path)
    geometry = _geometry(fixture, tmp_path / "geometry.jsonl", first_word_text="ness")

    book = _report(fixture, geometry).books[0]

    assert book.extensions["continuation_fragment_line_count"] == 0
    lines = [line for page in book.pages for line in page.lines]
    assert lines
    assert all(line.extensions["ocr_first_run_matches_source"] is False for line in lines)
    assert all(line.extensions["ocr_first_run_text"] == "ness" for line in lines)


def test_reconciled_after_witness_is_reported_beside_gate_six(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    geometry = _geometry(fixture, tmp_path / "geometry.jsonl")

    book = _report(fixture, geometry).books[0]

    reconciled = book.reconciled_line_count / book.measured_line_count
    assert book.extensions["reconciled_after_witness"] == pytest.approx(reconciled)


def test_a_page_whose_scan_hash_disagrees_gets_no_witness(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    geometry = _geometry(fixture, tmp_path / "geometry.jsonl", image_sha256="c" * 64)

    book = _report(fixture, geometry).books[0]

    assert book.extensions["witnessed_line_count"] == 0
    for page in book.pages:
        assert page.extensions["ocr_witness"] == "scan_hash_mismatch"
        for line in page.lines:
            assert "continuation_fragment" not in line.extensions


def test_a_geometry_record_for_another_project_is_refused(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    path = tmp_path / "geometry.jsonl"
    _ = _geometry(fixture, path)
    rewritten = [
        json.dumps(json.loads(line) | {"project_id": "projectIDother"})
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    _ = path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    with pytest.raises(OcrWitnessError, match="is for project"):
        _ = _report(fixture, path)


def test_the_witness_keys_pass_the_latent_discipline_check(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    geometry = _geometry(fixture, tmp_path / "geometry.jsonl", first_word_text="ness")

    payload = json.loads(json.dumps(_report(fixture, geometry).to_dict()))

    assert find_latent_discipline_violations(payload) == ()


def test_the_per_gap_error_rate_is_reported_without_geometry(tmp_path: Path) -> None:
    """Gate 6 counts lines and a line carries many gaps, so the per-gap rate is reported too.

    It is a property of the run detector, not of the witness, so it must appear with or without
    a geometry record.
    """

    fixture = build_fixture(tmp_path)

    book = _report(fixture, None).books[0]

    gaps = int(str(book.extensions["word_gap_count"]))
    errors = int(str(book.extensions["word_run_error_count"]))
    assert gaps > 0
    assert errors >= 0
    assert book.extensions["word_gap_error_rate"] == pytest.approx(errors / gaps)


def test_the_per_gap_counts_sum_the_pages(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)

    book = _report(fixture, None).books[0]

    for name in ("word_gap_count", "word_run_error_count"):
        assert int(str(book.extensions[name])) == sum(
            int(str(page.extensions[name])) for page in book.pages
        )


def test_a_fully_reconciled_book_reports_no_run_errors(tmp_path: Path) -> None:
    """The fixture's lines all reconcile, so the detector made no boundary error on it."""

    fixture = build_fixture(tmp_path)

    book = _report(fixture, None).books[0]

    assert book.reconciled_line_count == book.measured_line_count
    assert book.extensions["word_run_error_count"] == 0
    assert book.extensions["word_gap_error_rate"] == 0.0
