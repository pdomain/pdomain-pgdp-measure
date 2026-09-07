"""Read OCR geometry records produced elsewhere and use them to witness one line's first run.

`pdomain-source-data` runs the project's own fine-tuned DocTR checkpoints over PGDP projects and
writes one JSONL record per page: the page's `image_sha256`, the recognizer's identity, and every
recognized word with a pixel box and a confidence. Nothing here runs OCR, opens a model, or
touches the network. It reads that file.

The witness answers one question. PGDP F2 rejoins a word broken across two printed lines, writing
the whole word onto the line where it started, so the next line's first ink run has no F2 word to
bind to and the run count comes out one over. F2 has erased the evidence, so only the ink can say
whether a first run is a continuation fragment. Measured over the five corpus books, that accounts
for 3.3 to 6.8 percent of matched lines and 10 to 31 percent of the remaining word-reconciliation
disagreement.

Keys emitted from here say `first_run`, never `leading_run`: `leading` is on the
`pgdp-typography/v1` latent denylist, where it stands for the compositor's leading.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

Bounds = tuple[int, int, int, int]

OCR_WITNESS_SCHEMA_VERSIONS: Final = frozenset({"1.0"})
"""Record versions this reader understands. An unknown version is refused, not guessed at."""

OCR_WITNESS_METHODS: dict[str, float | int | str] = {
    "algorithm": "first-run-correspondence/v1",
    "row_band_overlap_ratio_minimum": 0.4,
}

_ROW_BAND_OVERLAP_RATIO_MINIMUM = 0.4
"""How much of a recognized word's height must fall inside the line's row band to belong to it.

Set below half because a line box is tight to its own ink while a recognized word box carries its
own ascender and descender, so the two never agree exactly even when they name the same word.
"""


class OcrWitnessError(ValueError):
    """The geometry record handed to this command cannot be used as written."""


@dataclass(frozen=True, slots=True)
class RecognizerIdentity:
    """Which model produced a record, pinned tightly enough to reproduce it."""

    name: str
    version: str
    config_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True, slots=True)
class RecognizedWord:
    """One recognized word and its half-open pixel box in the page's own frame."""

    text: str
    box: Bounds
    confidence: float | None
    line_index: int
    word_index: int

    def __post_init__(self) -> None:
        x_start, y_start, x_end, y_end = self.box
        if x_end <= x_start or y_end <= y_start:
            raise ValueError("A recognized word box must have positive width and height.")


@dataclass(frozen=True, slots=True)
class PageWitness:
    """Every word one page's recognizer read, with the scan hash that pins it to a page."""

    page_name: str
    image_sha256: str
    image_width: int
    image_height: int
    confidence_tier: str
    words: tuple[RecognizedWord, ...]


@dataclass(frozen=True, slots=True)
class BookWitness:
    """One book's geometry records, keyed by page name, plus their shared provenance."""

    project_id: str
    label: str
    sha256: str
    recognizer: RecognizerIdentity
    pages: Mapping[str, PageWitness]

    def page(self, page_name: str) -> PageWitness | None:
        return self.pages.get(page_name)


@dataclass(frozen=True, slots=True)
class LineWitness:
    """What the recognizer read where a matched line's first ink run sits.

    `continuation_fragment` is an inference and `first_run_matches_source` is the observation
    behind it. The flag is set only when the read disagrees with the transcription's first word
    *and* the line carries more runs than words, because a fragment necessarily produces the extra
    run. Both are recorded so a reader can judge the call rather than trust it.
    """

    first_run_text: str | None
    first_run_confidence: float | None
    first_run_matches_source: bool | None
    continuation_fragment: bool

    def to_extensions(self) -> dict[str, object]:
        return {
            "ocr_first_run_text": self.first_run_text,
            "ocr_first_run_confidence": self.first_run_confidence,
            "ocr_first_run_matches_source": self.first_run_matches_source,
            "continuation_fragment": self.continuation_fragment,
        }


class _WitnessModel(BaseModel):
    """Strict base for a foreign record: unknown fields are an error, not a shrug."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True, frozen=True)


class _PointInput(_WitnessModel):
    x: StrictInt
    y: StrictInt
    is_normalized: StrictBool


class _BoxInput(_WitnessModel):
    top_left: _PointInput
    bottom_right: _PointInput
    is_normalized: StrictBool


class _RecognizerInput(_WitnessModel):
    name: StrictStr = Field(min_length=1)
    version: StrictStr = Field(min_length=1)
    config_sha256: StrictStr = Field(min_length=1)


class _WordInput(_WitnessModel):
    text: StrictStr
    bbox: _BoxInput
    confidence: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    line_index: StrictInt = Field(ge=0)
    word_index: StrictInt = Field(ge=0)


class _PageInput(_WitnessModel):
    schema_version: StrictStr
    project_id: StrictStr = Field(min_length=1)
    page_name: StrictStr = Field(min_length=1)
    image_sha256: StrictStr = Field(min_length=64, max_length=64)
    image_width: StrictInt = Field(gt=0)
    image_height: StrictInt = Field(gt=0)
    confidence_tier: StrictStr = Field(min_length=1)
    recognizer: _RecognizerInput
    words: tuple[_WordInput, ...]


def read_book_witness(path: str | Path, *, project_id: str) -> BookWitness:
    """Read one book's geometry JSONL, refusing anything that is not provably this book's.

    Every record must name `project_id`, carry a version this reader knows, and agree with every
    other record on the recognizer. A file that mixes recognizers would produce flags that mean
    different things on different pages, which no later gate could see.
    """

    source = Path(path).expanduser()
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise OcrWitnessError(f"Could not read the geometry record: {source}") from error

    pages: dict[str, PageWitness] = {}
    recognizer: RecognizerIdentity | None = None
    for number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = _parse_record(line, source=source, number=number)
        if record.schema_version not in OCR_WITNESS_SCHEMA_VERSIONS:
            unsupported = f"unsupported schema_version {record.schema_version!r}"
            raise OcrWitnessError(f"Geometry record {source.name} line {number} has {unsupported}.")
        if record.project_id != project_id:
            wrong_book = f"is for project {record.project_id!r}, not {project_id!r}"
            raise OcrWitnessError(f"Geometry record {source.name} line {number} {wrong_book}.")
        identity = RecognizerIdentity(
            name=record.recognizer.name,
            version=record.recognizer.version,
            config_sha256=record.recognizer.config_sha256,
        )
        if recognizer is None:
            recognizer = identity
        elif identity != recognizer:
            raise OcrWitnessError(
                f"Geometry record {source.name} line {number} changes recognizer mid-file."
            )
        if record.page_name in pages:
            raise OcrWitnessError(
                f"Geometry record {source.name} repeats page {record.page_name!r}."
            )
        pages[record.page_name] = _page_witness(record)

    if recognizer is None:
        raise OcrWitnessError(f"Geometry record carries no page: {source}")
    return BookWitness(
        project_id=project_id,
        label=source.name,
        sha256=sha256(payload).hexdigest(),
        recognizer=recognizer,
        pages=MappingProxyType(pages),
    )


def witness_line(
    page: PageWitness,
    *,
    box: Bounds,
    first_run: tuple[int, int],
    source_first_word: str,
    over_split: bool,
) -> LineWitness:
    """Read what the recognizer saw where this line's first ink run sits.

    `first_run` is a box-local half-open column range, so it is translated into the page frame
    before matching. A run no recognized word overlaps gets a witness with no reading rather than
    a guess, and never sets the flag.
    """

    x_start, y_start, _x_end, y_end = box
    run_start, run_end = first_run
    word = _overlapping_word(
        page.words,
        x_start=x_start + run_start,
        x_end=x_start + run_end,
        y_start=y_start,
        y_end=y_end,
    )
    if word is None:
        return LineWitness(
            first_run_text=None,
            first_run_confidence=None,
            first_run_matches_source=None,
            continuation_fragment=False,
        )
    matches = normalize_word(word.text) == normalize_word(source_first_word)
    return LineWitness(
        first_run_text=word.text,
        first_run_confidence=word.confidence,
        first_run_matches_source=matches,
        continuation_fragment=over_split and not matches,
    )


def normalize_word(text: str) -> str:
    """Fold a word to its alphanumerics for comparison.

    Punctuation is dropped on both sides because F2 carries the printed comma or quote and the
    recognizer may not, and a comma is not what distinguishes a fragment from a word.
    """

    return "".join(character for character in text.lower() if character.isalnum())


def _parse_record(line: str, *, source: Path, number: int) -> _PageInput:
    try:
        return _PageInput.model_validate_json(line)
    except ValueError as error:
        raise OcrWitnessError(
            f"Geometry record {source.name} line {number} is not a readable page record: {error}"
        ) from error


def _page_witness(record: _PageInput) -> PageWitness:
    words = tuple(
        RecognizedWord(
            text=word.text,
            box=(
                word.bbox.top_left.x,
                word.bbox.top_left.y,
                word.bbox.bottom_right.x,
                word.bbox.bottom_right.y,
            ),
            confidence=word.confidence,
            line_index=word.line_index,
            word_index=word.word_index,
        )
        for word in record.words
        if _is_usable(word)
    )
    return PageWitness(
        page_name=record.page_name,
        image_sha256=record.image_sha256,
        image_width=record.image_width,
        image_height=record.image_height,
        confidence_tier=record.confidence_tier,
        words=words,
    )


def _is_usable(word: _WordInput) -> bool:
    """Drop a word carrying a normalized or empty box rather than mixing coordinate frames."""

    if word.bbox.is_normalized or word.bbox.top_left.is_normalized:
        return False
    if word.bbox.bottom_right.is_normalized:
        return False
    return (
        word.bbox.bottom_right.x > word.bbox.top_left.x
        and word.bbox.bottom_right.y > word.bbox.top_left.y
    )


def _overlapping_word(
    words: Sequence[RecognizedWord],
    *,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
) -> RecognizedWord | None:
    """The word sharing the most columns with this run, among those on the line's own row band.

    Ties keep the earlier word, so the choice does not depend on iteration luck.
    """

    best: RecognizedWord | None = None
    best_overlap = 0
    for word in words:
        word_x_start, word_y_start, word_x_end, word_y_end = word.box
        row_overlap = min(word_y_end, y_end) - max(word_y_start, y_start)
        if row_overlap <= _ROW_BAND_OVERLAP_RATIO_MINIMUM * (word_y_end - word_y_start):
            continue
        column_overlap = min(word_x_end, x_end) - max(word_x_start, x_start)
        if column_overlap > best_overlap:
            best = word
            best_overlap = column_overlap
    return best
