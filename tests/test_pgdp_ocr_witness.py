"""Tests for reading OCR geometry records and witnessing a line's first ink run.

Every record here is built inline. Nothing reads the corpus, runs OCR, or opens a model.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from pdomain_pgdp_measure.ocr_witness import (
    OcrWitnessError,
    PageWitness,
    RecognizedWord,
    normalize_word,
    read_book_witness,
    witness_line,
)

if TYPE_CHECKING:
    from pathlib import Path

_SHA = "a" * 64
_RECOGNIZER = {
    "name": "pdomain-doctr-finetuned",
    "version": "8b7f9b31d55015d16febaa99ae1d42960904028c",
    "config_sha256": "b" * 64,
}


def _word(text: str, x_start: int, x_end: int, *, index: int = 0) -> dict[str, object]:
    return {
        "text": text,
        "bbox": {
            "top_left": {"x": x_start, "y": 100, "is_normalized": False},
            "bottom_right": {"x": x_end, "y": 140, "is_normalized": False},
            "is_normalized": False,
        },
        "confidence": 0.98,
        "line_index": 0,
        "word_index": index,
    }


def _record(page_name: str = "p001.png", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "project_id": "projectIDabc",
        "page_name": page_name,
        "image_sha256": _SHA,
        "image_width": 1100,
        "image_height": 1772,
        "confidence_tier": "silver",
        "recognizer": dict(_RECOGNIZER),
        "words": [_word("ness", 10, 90), _word("which", 120, 220, index=1)],
    }
    payload.update(overrides)
    return payload


def _write(tmp_path: Path, *records: dict[str, object]) -> Path:
    path = tmp_path / "book.jsonl"
    _ = path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_reads_pages_and_pins_the_recognizer(tmp_path: Path) -> None:
    path = _write(tmp_path, _record("p001.png"), _record("p002.png"))

    witness = read_book_witness(path, project_id="projectIDabc")

    assert set(witness.pages) == {"p001.png", "p002.png"}
    assert witness.recognizer.name == "pdomain-doctr-finetuned"
    assert witness.recognizer.config_sha256 == "b" * 64
    assert witness.label == "book.jsonl"
    assert len(witness.sha256) == 64
    page = witness.page("p001.png")
    assert page is not None
    assert page.image_sha256 == _SHA
    assert page.words[0].text == "ness"
    assert page.words[0].box == (10, 100, 90, 140)


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "book.jsonl"
    _ = path.write_text(f"\n{json.dumps(_record())}\n\n", encoding="utf-8")

    assert set(read_book_witness(path, project_id="projectIDabc").pages) == {"p001.png"}


def test_refuses_a_record_for_another_project(tmp_path: Path) -> None:
    path = _write(tmp_path, _record(project_id="projectIDother"))

    with pytest.raises(OcrWitnessError, match="is for project"):
        _ = read_book_witness(path, project_id="projectIDabc")


def test_refuses_an_unknown_schema_version(tmp_path: Path) -> None:
    path = _write(tmp_path, _record(schema_version="2.0"))

    with pytest.raises(OcrWitnessError, match="unsupported schema_version"):
        _ = read_book_witness(path, project_id="projectIDabc")


def test_refuses_a_file_that_changes_recognizer(tmp_path: Path) -> None:
    other = dict(_RECOGNIZER) | {"version": "c" * 40}
    path = _write(tmp_path, _record("p001.png"), _record("p002.png", recognizer=other))

    with pytest.raises(OcrWitnessError, match="changes recognizer"):
        _ = read_book_witness(path, project_id="projectIDabc")


def test_refuses_a_repeated_page(tmp_path: Path) -> None:
    path = _write(tmp_path, _record("p001.png"), _record("p001.png"))

    with pytest.raises(OcrWitnessError, match="repeats page"):
        _ = read_book_witness(path, project_id="projectIDabc")


def test_refuses_an_unknown_field(tmp_path: Path) -> None:
    path = _write(tmp_path, _record(surprise="new"))

    with pytest.raises(OcrWitnessError, match="not a readable page record"):
        _ = read_book_witness(path, project_id="projectIDabc")


def test_refuses_a_file_with_no_page(tmp_path: Path) -> None:
    path = tmp_path / "book.jsonl"
    _ = path.write_text("\n", encoding="utf-8")

    with pytest.raises(OcrWitnessError, match="carries no page"):
        _ = read_book_witness(path, project_id="projectIDabc")


def test_reports_a_missing_file_rather_than_raising_oserror(tmp_path: Path) -> None:
    with pytest.raises(OcrWitnessError, match="Could not read"):
        _ = read_book_witness(tmp_path / "absent.jsonl", project_id="projectIDabc")


def test_drops_a_word_carrying_a_normalized_box(tmp_path: Path) -> None:
    normalized = _word("ness", 10, 90)
    box = normalized["bbox"]
    assert isinstance(box, dict)
    box["is_normalized"] = True
    path = _write(tmp_path, _record(words=[normalized, _word("which", 120, 220, index=1)]))

    page = read_book_witness(path, project_id="projectIDabc").page("p001.png")
    assert page is not None
    assert [word.text for word in page.words] == ["which"]


def _page(*words: RecognizedWord) -> PageWitness:
    return PageWitness(
        page_name="p001.png",
        image_sha256=_SHA,
        image_width=1100,
        image_height=1772,
        confidence_tier="silver",
        words=words,
    )


def _recognized(text: str, x_start: int, x_end: int) -> RecognizedWord:
    return RecognizedWord(
        text=text, box=(x_start, 100, x_end, 140), confidence=0.9, line_index=0, word_index=0
    )


def test_flags_a_first_run_the_transcription_does_not_carry() -> None:
    page = _page(_recognized("ness", 10, 90), _recognized("which", 120, 220))

    result = witness_line(
        page,
        box=(10, 100, 400, 145),
        first_run=(0, 80),
        source_first_word="which",
        over_split=True,
    )

    assert result.first_run_text == "ness"
    assert result.first_run_matches_source is False
    assert result.continuation_fragment is True


def test_does_not_flag_a_first_run_the_transcription_carries() -> None:
    page = _page(_recognized("which", 10, 90))

    result = witness_line(
        page,
        box=(10, 100, 400, 145),
        first_run=(0, 80),
        source_first_word="which,",
        over_split=True,
    )

    assert result.first_run_matches_source is True
    assert result.continuation_fragment is False


def test_does_not_flag_a_line_that_is_not_over_split() -> None:
    """A read that disagrees on a line with as many runs as words is an OCR miss, not a
    fragment: a carried-over fragment necessarily produces the extra run.
    """

    page = _page(_recognized("ness", 10, 90))

    result = witness_line(
        page,
        box=(10, 100, 400, 145),
        first_run=(0, 80),
        source_first_word="which",
        over_split=False,
    )

    assert result.first_run_matches_source is False
    assert result.continuation_fragment is False


def test_an_unread_first_run_witnesses_nothing() -> None:
    page = _page(_recognized("elsewhere", 900, 1000))

    result = witness_line(
        page,
        box=(10, 100, 400, 145),
        first_run=(0, 80),
        source_first_word="which",
        over_split=True,
    )

    assert result.first_run_text is None
    assert result.first_run_matches_source is None
    assert result.continuation_fragment is False


def test_a_word_on_another_line_does_not_witness_this_run() -> None:
    page = _page(
        RecognizedWord(
            text="below", box=(10, 300, 90, 340), confidence=0.9, line_index=1, word_index=0
        )
    )

    result = witness_line(
        page,
        box=(10, 100, 400, 145),
        first_run=(0, 80),
        source_first_word="which",
        over_split=True,
    )

    assert result.first_run_text is None


def test_the_most_overlapping_word_wins() -> None:
    page = _page(_recognized("sliver", 70, 90), _recognized("ness", 5, 85))

    result = witness_line(
        page,
        box=(10, 100, 400, 145),
        first_run=(0, 80),
        source_first_word="which",
        over_split=True,
    )

    assert result.first_run_text == "ness"


@pytest.mark.parametrize(
    ("left", "right"),
    [("which,", "which"), ("Which", "which"), ('"Teach', "teach"), ("con-", "con")],
)
def test_normalization_folds_case_and_punctuation(left: str, right: str) -> None:
    assert normalize_word(left) == normalize_word(right)


def test_the_witness_evidence_avoids_the_latent_denylist() -> None:
    """`leading` names the compositor's leading in `pgdp-typography/v1` and is denied there, so
    these keys must say `first_run`.
    """

    from pdomain_pgdp_measure.typography_models import LATENT_KEY_DENYLIST

    page = _page(_recognized("ness", 10, 90))
    keys = witness_line(
        page,
        box=(10, 100, 400, 145),
        first_run=(0, 80),
        source_first_word="which",
        over_split=True,
    ).to_extensions()

    for key in keys:
        for denied in LATENT_KEY_DENYLIST:
            assert denied not in key.lower()
