"""Tests for selecting the recognized words that sit outside every matched line box.

Every record here is built inline. Nothing reads the corpus, runs OCR, or opens a model.
"""

from __future__ import annotations

import pytest

from pdomain_pgdp_measure.glyph_furniture import (
    FurnitureScanMismatchError,
    FurnitureWord,
    line_is_admissible,
    line_word_agreement,
    page_furniture_words,
    witness_matches_scan,
)
from pdomain_pgdp_measure.ocr_witness import PageWitness, RecognizedWord

_SCAN = "a" * 64
_OTHER_SCAN = "b" * 64


def _word(
    text: str,
    box: tuple[int, int, int, int],
    *,
    line_index: int = 0,
    word_index: int = 0,
    confidence: float | None = 0.81,
) -> RecognizedWord:
    return RecognizedWord(
        text=text,
        box=box,
        confidence=confidence,
        line_index=line_index,
        word_index=word_index,
    )


def _page(*words: RecognizedWord, image_sha256: str = _SCAN) -> PageWitness:
    return PageWitness(
        page_name="p001.png",
        image_sha256=image_sha256,
        image_width=1100,
        image_height=1772,
        confidence_tier="silver",
        words=words,
    )


def test_a_word_inside_a_matched_line_is_not_furniture() -> None:
    page = _page(_word("which", (120, 300, 220, 340)))
    assert (
        page_furniture_words(page, scan_sha256=_SCAN, matched_line_boxes=[(100, 295, 900, 345)])
        == ()
    )


def test_a_word_above_every_matched_line_is_furniture() -> None:
    page = _page(_word("14", (520, 60, 560, 100)))
    furniture = page_furniture_words(
        page, scan_sha256=_SCAN, matched_line_boxes=[(100, 295, 900, 345)]
    )
    assert furniture == (
        FurnitureWord(
            text="14", box=(520, 60, 560, 100), confidence=0.81, line_index=0, word_index=0
        ),
    )


def test_a_word_nicking_a_line_box_stays_furniture() -> None:
    """A recognized box carries its own descender, so a small overlap must not claim it."""

    page = _page(_word("CHAPTER", (100, 260, 300, 300)))
    furniture = page_furniture_words(
        page, scan_sha256=_SCAN, matched_line_boxes=[(100, 295, 900, 345)]
    )
    assert [word.text for word in furniture] == ["CHAPTER"]


def test_a_word_mostly_inside_a_line_box_is_claimed() -> None:
    page = _page(_word("wandering", (100, 290, 300, 340)))
    assert (
        page_furniture_words(page, scan_sha256=_SCAN, matched_line_boxes=[(100, 295, 900, 345)])
        == ()
    )


def test_furniture_comes_back_in_recognizer_order() -> None:
    page = _page(
        _word("PAGE", (300, 60, 380, 100), line_index=0, word_index=1),
        _word("THE", (200, 60, 280, 100), line_index=0, word_index=0),
        _word("14", (520, 1700, 560, 1740), line_index=9, word_index=0),
    )
    furniture = page_furniture_words(page, scan_sha256=_SCAN, matched_line_boxes=[])
    assert [word.text for word in furniture] == ["THE", "PAGE", "14"]


def test_no_matched_line_leaves_every_word_as_furniture() -> None:
    page = _page(_word("which", (120, 300, 220, 340)))
    assert len(page_furniture_words(page, scan_sha256=_SCAN, matched_line_boxes=[])) == 1


def test_a_scan_hash_mismatch_is_refused_before_any_box_is_read() -> None:
    page = _page(_word("14", (520, 60, 560, 100)), image_sha256=_OTHER_SCAN)
    assert not witness_matches_scan(page, scan_sha256=_SCAN)
    with pytest.raises(FurnitureScanMismatchError, match="not the aligned"):
        _ = page_furniture_words(page, scan_sha256=_SCAN, matched_line_boxes=[])


def test_a_matching_scan_hash_passes_the_check() -> None:
    assert witness_matches_scan(_page(), scan_sha256=_SCAN)


def test_a_furniture_word_box_must_have_positive_extent() -> None:
    with pytest.raises(ValueError, match="positive width and height"):
        _ = FurnitureWord(
            text="14", box=(10, 10, 10, 20), confidence=None, line_index=0, word_index=0
        )


def test_a_furniture_word_ordinal_must_be_nonnegative() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _ = FurnitureWord(
            text="14", box=(10, 10, 20, 20), confidence=None, line_index=-1, word_index=0
        )


def test_a_line_the_recognizer_reads_the_same_way_agrees_fully() -> None:
    page = _page(
        _word("wandering", (100, 300, 200, 340)),
        _word("alone", (210, 300, 280, 340), word_index=1),
    )
    agreement = line_word_agreement(page, box=(90, 295, 900, 345), visible_text="wandering alone")
    assert agreement == 1.0
    assert line_is_admissible(agreement)


def test_a_line_bound_to_different_text_disagrees_and_is_refused() -> None:
    page = _page(_word("something", (100, 300, 200, 340)))
    agreement = line_word_agreement(
        page, box=(90, 295, 900, 345), visible_text="wandering alone entirely"
    )
    assert agreement == 0.0
    assert not line_is_admissible(agreement)


def test_agreement_ignores_case_and_punctuation() -> None:
    page = _page(_word("Wandering,", (100, 300, 200, 340)))
    assert line_word_agreement(page, box=(90, 295, 900, 345), visible_text="wandering") == 1.0


def test_a_read_outside_the_line_box_does_not_count() -> None:
    page = _page(_word("wandering", (100, 60, 200, 100)))
    assert line_word_agreement(page, box=(90, 295, 900, 345), visible_text="wandering") == 0.0


def test_a_line_with_no_words_is_no_evidence_rather_than_disagreement() -> None:
    assert line_word_agreement(_page(), box=(0, 0, 10, 10), visible_text="   ") is None
    assert not line_is_admissible(None)


def test_the_admission_bar_sits_at_four_words_in_five() -> None:
    page = _page(
        _word("one", (100, 300, 130, 340)),
        _word("two", (140, 300, 170, 340), word_index=1),
        _word("three", (180, 300, 220, 340), word_index=2),
        _word("four", (230, 300, 260, 340), word_index=3),
    )
    agreement = line_word_agreement(
        page, box=(90, 295, 900, 345), visible_text="one two three four five"
    )
    assert agreement == pytest.approx(0.8)
    assert line_is_admissible(agreement)
