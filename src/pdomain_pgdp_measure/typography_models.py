"""Versioned wire records for font-free PGDP typographic observables.

`pgdp-typography/v1` carries what `typography_measure` measures from aligned ink: per-line
baseline, x-height, ascender and descender extents, stroke width, skew, word runs, per-word
boxes, and the two-stage pooled estimates built from them. It is a new report family rather
than an extension of `pgdp-profile/v2` or `pgdp-alignment/v3`, so neither of those contracts
changes a byte.

Provenance is a hash chain. A report records the alignment report it measured
(`alignment_label`, `alignment_sha256`) and the profile that alignment recorded
(`profile_label`, `profile_sha256`), so a reader can walk back to the exact geometry and
transcription evidence a number came from. Word rows carry ordinals rather than word text for
the same reason: the transcription already lives in the alignment report, and duplicating it
here would let the two copies drift.

Nothing in this module names a typeface. `LATENT_KEY_DENYLIST` and
`find_latent_discipline_violations` make that enforceable instead of aspirational: the design
lists font identity, nominal point size, advances and side bearings, native word-space width,
tracking and pair kerning, original leading, and physical margins as latent, and a version-1
report may not claim any of them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping as RuntimeMapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from pdomain_pgdp_measure.ordering import natural_page_key
from pdomain_pgdp_measure.profile_models import (
    CoordinateFrame,
    CoordinateFrameWire,
    Estimate,
    EstimateWire,
    ProfileDiagnostic,
    ProfileDiagnosticWire,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


TYPOGRAPHY_SCHEMA_VERSION: Final = 1
TYPOGRAPHY_ALGORITHM_VERSION: Final = "pgdp-typography/v1"
TYPOGRAPHY_CONFIDENCE_KIND: Final = "uncalibrated"
DEFAULT_EVIDENCE_PAGES_PER_BOOK: Final = 12

Bounds = tuple[int, int, int, int] | list[int]
StyleName = Literal["body", "chapter_opening"]

BODY_PAGE_CLASSES: Final = ("normal_recto", "normal_verso")
CHAPTER_OPENING_PAGE_CLASSES: Final = ("chapter_opening",)
UNPOOLED_PAGE_CLASS: Final = "unknown"

TYPOGRAPHY_ESTIMATE_NAMES: Final = (
    "x_height_px",
    "baseline_pitch_px",
    "stroke_width_px",
    "word_gap_px",
    "ascender_extent_px",
    "descender_extent_px",
    "line_ink_height_px",
    "line_ink_width_px",
    "line_indent_px",
)

LATENT_KEY_DENYLIST: Final = frozenset(
    {
        "advance",
        "bearing",
        "em_size",
        "font",
        "glyph_name",
        "kern",
        "leading",
        "point_size",
        "typeface",
        "tracking",
        "word_space",
        "_mm",
        "_pt",
        "_inch",
        "_dpi",
    }
)
"""Substrings a `pgdp-typography/v1` key may never contain.

Each entry stands for one item on the design's latent list. `leading` is covered by
`baseline_pitch_px`, which is a measured pixel distance between two matched baselines and
makes no claim about the leading the compositor set; `word_space` is covered by `word_gap_px`,
which is an observed ink gap and not the font's native word-space advance. The physical-unit
suffixes are here because every value in this contract is in pixels in the `source` frame, so
a millimetre, point, inch, or DPI key would mean an uncalibrated physical claim slipped in.
"""

_JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, JsonValue])


def _require_nonnegative_integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


def _require_integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")


def _require_string(value: object, *, name: str, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if nonempty and not value:
        raise ValueError(f"{name} must not be empty.")
    return value


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _require_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return float(value)


def _require_nonnegative_finite(value: object, *, name: str) -> float:
    result = _require_finite(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _require_sha256(value: object, *, name: str) -> None:
    hexadecimal_characters = "0123456789abcdefABCDEF"
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in hexadecimal_characters for character in value)
    ):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal characters.")


def _require_file_label(value: object, *, name: str) -> str:
    label = _require_string(value, name=name, nonempty=True)
    if label in {".", ".."} or "/" in label or "\\" in label or Path(label).name != label:
        raise ValueError(f"{name} must be a nonempty file name without path components.")
    return label


def _normalize_box(value: Bounds, *, name: str) -> tuple[int, int, int, int]:
    box = tuple(value)
    if len(box) != 4:
        raise ValueError(f"{name} must contain four coordinates.")
    x_start, y_start, x_end, y_end = box
    for coordinate_name, coordinate in zip(
        ("x_start", "y_start", "x_end", "y_end"), box, strict=True
    ):
        _require_nonnegative_integer(coordinate, name=f"{name}.{coordinate_name}")
    if x_end <= x_start or y_end <= y_start:
        raise ValueError(f"{name} must be a nonempty half-open box.")
    return x_start, y_start, x_end, y_end


def _deep_freeze_json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        _ = _require_finite(value, name="JSON value")
        return value
    if isinstance(value, list):
        return tuple(_deep_freeze_json_value(item) for item in value)
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _deep_freeze_json_value(item) for key, item in sorted(value.items())}
        )
    raise ValueError("Extension values must be JSON-native.")


def _thaw_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        _ = _require_finite(value, name="Stored JSON value")
        return value
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    if isinstance(value, MappingProxyType):
        thawed: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Stored JSON object keys must be strings.")
            thawed[key] = _thaw_json_value(item)
        return thawed
    raise TypeError("Stored JSON values must be JSON-native.")


def _normalized_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(values, RuntimeMapping):
        raise TypeError("JSON object values must be mappings.")
    validated = _JSON_OBJECT_ADAPTER.validate_python(dict(values))
    return MappingProxyType(
        {key: _deep_freeze_json_value(value) for key, value in sorted(validated.items())}
    )


def _mapping_dict(values: Mapping[str, object]) -> dict[str, JsonValue]:
    return {key: _thaw_json_value(value) for key, value in sorted(values.items())}


def _merge_payload(
    payload: dict[str, JsonValue], extensions: Mapping[str, object]
) -> dict[str, JsonValue]:
    overlap = set(payload) & set(extensions)
    if overlap:
        raise ValueError(f"extensions cannot replace defined fields: {', '.join(sorted(overlap))}.")
    payload.update(_mapping_dict(extensions))
    return payload


def _normalize_tuple_fields(payload: object, fields: tuple[str, ...]) -> object:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    for field_name in fields:
        value = normalized.get(field_name)
        if isinstance(value, list):
            normalized[field_name] = tuple(value)
    return normalized


def _require_estimates(
    estimates: tuple[Estimate, ...], *, name: str, allowed_pages: frozenset[str] | None
) -> None:
    """Every estimate is a well-formed uncalibrated pixel estimate over known pages."""

    if any(not isinstance(estimate, Estimate) for estimate in estimates):
        raise TypeError(f"{name} must contain Estimate records.")
    names = tuple(estimate.name for estimate in estimates)
    if len(set(names)) != len(names):
        raise ValueError(f"{name} contains duplicate estimate names.")
    for estimate in estimates:
        if estimate.unit != "px":
            raise ValueError(f"{name} estimates must be measured in px.")
        if estimate.frame != "source":
            raise ValueError(f"{name} estimates must stay in the source frame.")
        if estimate.confidence is not None:
            raise ValueError(f"{name} estimates must carry a null confidence.")
        if estimate.confidence_kind != TYPOGRAPHY_CONFIDENCE_KIND:
            raise ValueError(f"{name} estimates must be uncalibrated.")
        if allowed_pages is not None and not set(estimate.evidence_pages) <= allowed_pages:
            raise ValueError(f"{name} estimates must cite pages this record contains.")


def _require_skew(median_value: float | None, mad_value: float | None, *, name: str) -> None:
    """Skew is reported as a bare median and dispersion, never as an `Estimate`.

    A slope is dimensionless, and every `Estimate` in this contract is in px, so pooling skew
    as one would either lie about its unit or force a second unit into a contract whose whole
    point is that it has one. The distribution still gets recorded, because it is the evidence
    that scopes a later rectification slice.
    """

    if (median_value is None) != (mad_value is None):
        raise ValueError(f"{name} median and dispersion must be present or absent together.")
    if median_value is not None:
        _ = _require_finite(median_value, name=f"{name}_median")
    if mad_value is not None:
        _ = _require_nonnegative_finite(mad_value, name=f"{name}_mad")


@dataclass(frozen=True, slots=True)
class WordTypography:
    """One reconciled word's ink box and the gap that follows it, in the `source` frame.

    `ordinal` indexes the word within its line, so a reader joins back to the word's text
    through the alignment report's `visible_text` rather than reading it here.
    `following_gap_px` is the zero-ink column run between this word and the next; the last
    word on a line has none.
    """

    ordinal: int
    box: Bounds
    following_gap_px: int | None = None
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.ordinal, name="Word ordinal")
        object.__setattr__(self, "box", _normalize_box(self.box, name="Word box"))
        if self.following_gap_px is not None:
            _require_nonnegative_integer(self.following_gap_px, name="following_gap_px")
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merge_payload(
            {
                "ordinal": self.ordinal,
                "box": list(self.box),
                "following_gap_px": self.following_gap_px,
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class LineTypography:
    """One matched line's font-free observables, or the record of why it has none.

    The seven measured fields from `baseline_row_px` through `skew_slope` are present or
    absent together: a line the estimator excluded still gets a row, carrying its box, its
    observed width and height, its usable segment count, and the exclusion codes that say
    what happened. Dropping such a line would lose the evidence a later labeler needs, and
    the plan's measurement says 13.9 percent of matched lines fall below the segment floor,
    so exclusions are ordinary rather than exceptional.

    Word fields are present exactly when the measured fields are, because the gap threshold
    is derived from the measured x-height and cannot be computed without it.
    """

    candidate_ordinal: int
    box: Bounds
    line_ink_width_px: int
    line_ink_height_px: int
    segment_count: int
    source_ordinal: int | None = None
    baseline_row_px: int | None = None
    x_height_top_row_px: int | None = None
    x_height_px: int | None = None
    ascender_extent_px: int | None = None
    descender_extent_px: int | None = None
    stroke_width_px: float | None = None
    skew_slope: float | None = None
    line_indent_px: int | None = None
    word_run_count: int | None = None
    source_word_count: int | None = None
    words_reconciled: bool | None = None
    words: tuple[WordTypography, ...] = ()
    exclusions: tuple[str, ...] = ()
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.candidate_ordinal, name="candidate_ordinal")
        if self.source_ordinal is not None:
            _require_nonnegative_integer(self.source_ordinal, name="source_ordinal")
        normalized_box = _normalize_box(self.box, name="Line box")
        object.__setattr__(self, "box", normalized_box)
        _require_nonnegative_integer(self.line_ink_width_px, name="line_ink_width_px")
        _require_nonnegative_integer(self.line_ink_height_px, name="line_ink_height_px")
        if (
            self.line_ink_width_px != normalized_box[2] - normalized_box[0]
            or self.line_ink_height_px != normalized_box[3] - normalized_box[1]
        ):
            raise ValueError("Line ink dimensions must match its source-frame box.")
        _require_nonnegative_integer(self.segment_count, name="segment_count")
        if self.line_indent_px is not None:
            _require_integer(self.line_indent_px, name="line_indent_px")
        object.__setattr__(self, "words", tuple(self.words))
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        for exclusion in self.exclusions:
            _ = _require_string(exclusion, name="Line exclusion", nonempty=True)
        if any(not isinstance(word, WordTypography) for word in self.words):
            raise TypeError("words must contain WordTypography records.")
        self._validate_measurement(normalized_box)
        self._validate_words(normalized_box)
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def _validate_measurement(self, box: tuple[int, int, int, int]) -> None:
        measured = (
            self.baseline_row_px,
            self.x_height_top_row_px,
            self.x_height_px,
            self.ascender_extent_px,
            self.descender_extent_px,
            self.stroke_width_px,
            self.skew_slope,
        )
        present = tuple(value is not None for value in measured)
        if any(present) and not all(present):
            raise ValueError("Line measurement fields must be present or absent together.")
        if not all(present):
            if not self.exclusions:
                raise ValueError("An unmeasured line must record at least one exclusion.")
            return
        baseline_row_px = self.baseline_row_px
        x_height_top_row_px = self.x_height_top_row_px
        x_height_px = self.x_height_px
        ascender_extent_px = self.ascender_extent_px
        descender_extent_px = self.descender_extent_px
        if (
            baseline_row_px is None
            or x_height_top_row_px is None
            or x_height_px is None
            or ascender_extent_px is None
            or descender_extent_px is None
        ):
            return
        for name, value in (
            ("baseline_row_px", baseline_row_px),
            ("x_height_top_row_px", x_height_top_row_px),
            ("x_height_px", x_height_px),
            ("ascender_extent_px", ascender_extent_px),
            ("descender_extent_px", descender_extent_px),
        ):
            _require_nonnegative_integer(value, name=name)
        _ = _require_nonnegative_finite(self.stroke_width_px, name="stroke_width_px")
        _ = _require_finite(self.skew_slope, name="skew_slope")
        _, y_start, _, y_end = box
        if not y_start <= baseline_row_px < y_end:
            raise ValueError("baseline_row_px must lie inside the line box rows.")
        if x_height_px != baseline_row_px - x_height_top_row_px:
            raise ValueError("x_height_px must equal baseline_row_px minus x_height_top_row_px.")
        if ascender_extent_px != baseline_row_px - y_start:
            raise ValueError("ascender_extent_px must equal the baseline's offset from box top.")
        if descender_extent_px != y_end - baseline_row_px:
            raise ValueError("descender_extent_px must equal the box bottom below the baseline.")

    def _validate_words(self, box: tuple[int, int, int, int]) -> None:
        word_presence = (
            self.word_run_count is not None,
            self.source_word_count is not None,
            self.words_reconciled is not None,
        )
        if any(word_presence) and not all(word_presence):
            raise ValueError("Line word-count fields must be present or absent together.")
        word_fields_present = all(word_presence)
        if word_fields_present == (self.baseline_row_px is None):
            raise ValueError(
                "Word counts are present exactly when the line's x-height was measured."
            )
        if not word_fields_present:
            if self.words:
                raise ValueError("A line with no word counts must emit no word rows.")
            return
        word_run_count = self.word_run_count
        source_word_count = self.source_word_count
        words_reconciled = self.words_reconciled
        if word_run_count is None or source_word_count is None or words_reconciled is None:
            return
        _require_nonnegative_integer(word_run_count, name="word_run_count")
        _require_nonnegative_integer(source_word_count, name="source_word_count")
        _ = _require_bool(words_reconciled, name="words_reconciled")
        if words_reconciled != (word_run_count == source_word_count):
            raise ValueError("words_reconciled must match whether the two counts agree.")
        if not words_reconciled and self.words:
            raise ValueError("An unreconciled line must not emit word rows.")
        if self.words:
            if len(self.words) != word_run_count:
                raise ValueError("A reconciled line must emit exactly one row per word run.")
            if tuple(word.ordinal for word in self.words) != tuple(range(len(self.words))):
                raise ValueError("Word ordinals must be consecutive from zero.")
            x_start, y_start, x_end, y_end = box
            for word in self.words:
                word_x_start, word_y_start, word_x_end, word_y_end = word.box
                if (
                    word_x_start < x_start
                    or word_y_start < y_start
                    or word_x_end > x_end
                    or word_y_end > y_end
                ):
                    raise ValueError("Word boxes must stay inside their line box.")

    def to_dict(self) -> dict[str, JsonValue]:
        return _merge_payload(
            {
                "candidate_ordinal": self.candidate_ordinal,
                "source_ordinal": self.source_ordinal,
                "box": list(self.box),
                "line_ink_width_px": self.line_ink_width_px,
                "line_ink_height_px": self.line_ink_height_px,
                "line_indent_px": self.line_indent_px,
                "baseline_row_px": self.baseline_row_px,
                "x_height_top_row_px": self.x_height_top_row_px,
                "x_height_px": self.x_height_px,
                "ascender_extent_px": self.ascender_extent_px,
                "descender_extent_px": self.descender_extent_px,
                "stroke_width_px": self.stroke_width_px,
                "skew_slope": self.skew_slope,
                "segment_count": self.segment_count,
                "word_run_count": self.word_run_count,
                "source_word_count": self.source_word_count,
                "words_reconciled": self.words_reconciled,
                "words": [word.to_dict() for word in self.words],
                "exclusions": list(self.exclusions),
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class PageTypography:
    """One accepted page's line counts, per-page medians, and sampled evidence rows.

    `estimates` holds the first pooling stage: one robust median per observable over this
    page's measured lines, with `sample_count == 1` and this page as its only evidence.
    `lines` is populated only for the bounded evidence sample, so `evidence_sampled` says
    whether an empty `lines` means "not sampled" or "nothing to show".
    """

    page_name: str
    page_class: str
    scan_sha256: str | None
    source_frame: CoordinateFrame | None
    matched_line_count: int
    measured_line_count: int
    excluded_line_count: int
    reconciled_line_count: int
    baseline_pitch_count: int
    evidence_sampled: bool = False
    lines: tuple[LineTypography, ...] = ()
    estimates: tuple[Estimate, ...] = ()
    skew_slope_median: float | None = None
    skew_slope_mad: float | None = None
    exclusions: tuple[str, ...] = ()
    exclusion_evidence: tuple[Mapping[str, object], ...] = ()
    diagnostics: tuple[ProfileDiagnostic, ...] = ()
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ = _require_string(self.page_name, name="Page name", nonempty=True)
        _ = _require_string(self.page_class, name="Page class", nonempty=True)
        if self.scan_sha256 is not None:
            _require_sha256(self.scan_sha256, name="scan_sha256")
        if self.source_frame is not None and not isinstance(self.source_frame, CoordinateFrame):
            raise TypeError("source_frame must be a CoordinateFrame or null.")
        for name in (
            "matched_line_count",
            "measured_line_count",
            "excluded_line_count",
            "reconciled_line_count",
            "baseline_pitch_count",
        ):
            _require_nonnegative_integer(getattr(self, name), name=name)
        if self.measured_line_count + self.excluded_line_count != self.matched_line_count:
            raise ValueError("Measured and excluded lines must account for every matched line.")
        if self.reconciled_line_count > self.measured_line_count:
            raise ValueError("reconciled_line_count cannot exceed measured_line_count.")
        _ = _require_bool(self.evidence_sampled, name="evidence_sampled")
        object.__setattr__(self, "lines", tuple(self.lines))
        object.__setattr__(self, "estimates", tuple(self.estimates))
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(
            self,
            "exclusion_evidence",
            tuple(_normalized_mapping(item) for item in self.exclusion_evidence),
        )
        if any(not isinstance(line, LineTypography) for line in self.lines):
            raise TypeError("lines must contain LineTypography records.")
        if any(not isinstance(exclusion, str) for exclusion in self.exclusions):
            raise TypeError("exclusions must contain strings.")
        if any(not isinstance(diagnostic, ProfileDiagnostic) for diagnostic in self.diagnostics):
            raise TypeError("diagnostics must contain ProfileDiagnostic records.")
        if self.lines and not self.evidence_sampled:
            raise ValueError("Only an evidence-sampled page may emit line rows.")
        if self.evidence_sampled and len(self.lines) != self.matched_line_count:
            raise ValueError("An evidence-sampled page must emit one row per matched line.")
        candidate_ordinals = tuple(line.candidate_ordinal for line in self.lines)
        if len(set(candidate_ordinals)) != len(candidate_ordinals):
            raise ValueError("Line rows must not repeat a candidate ordinal.")
        if candidate_ordinals != tuple(sorted(candidate_ordinals)):
            raise ValueError("Line rows must use increasing candidate ordinals.")
        _require_estimates(
            self.estimates, name="Page typography", allowed_pages=frozenset({self.page_name})
        )
        _require_skew(self.skew_slope_median, self.skew_slope_mad, name="Page skew slope")
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        source_frame = None if self.source_frame is None else self.source_frame.to_dict()
        return _merge_payload(
            {
                "page_name": self.page_name,
                "page_class": self.page_class,
                "scan_sha256": self.scan_sha256,
                "source_frame": source_frame,
                "matched_line_count": self.matched_line_count,
                "measured_line_count": self.measured_line_count,
                "excluded_line_count": self.excluded_line_count,
                "reconciled_line_count": self.reconciled_line_count,
                "baseline_pitch_count": self.baseline_pitch_count,
                "evidence_sampled": self.evidence_sampled,
                "lines": [line.to_dict() for line in self.lines],
                "estimates": [estimate.to_dict() for estimate in self.estimates],
                "skew_slope_median": self.skew_slope_median,
                "skew_slope_mad": self.skew_slope_mad,
                "exclusions": list(self.exclusions),
                "exclusion_evidence": [_mapping_dict(item) for item in self.exclusion_evidence],
                "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class StyleTypography:
    """One pooled style within a book, grouped by the shipped `page_class`.

    `body` pools `normal_recto` and `normal_verso`; `chapter_opening` pools its own class
    alone. Pages classed `unknown` are counted on the book but pooled into no style, because
    a style built from pages that could not be classified would mix templates.
    """

    style: StyleName
    page_classes: tuple[str, ...]
    page_count: int
    line_count: int
    estimates: tuple[Estimate, ...] = ()
    skew_slope_median: float | None = None
    skew_slope_mad: float | None = None
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.style not in {"body", "chapter_opening"}:
            raise ValueError("Typography style is unsupported.")
        object.__setattr__(self, "page_classes", tuple(self.page_classes))
        expected = BODY_PAGE_CLASSES if self.style == "body" else CHAPTER_OPENING_PAGE_CLASSES
        if self.page_classes != expected:
            raise ValueError("Style page_classes must match the style's shipped page classes.")
        _require_nonnegative_integer(self.page_count, name="page_count")
        _require_nonnegative_integer(self.line_count, name="line_count")
        object.__setattr__(self, "estimates", tuple(self.estimates))
        _require_estimates(self.estimates, name="Style typography", allowed_pages=None)
        _require_skew(self.skew_slope_median, self.skew_slope_mad, name="Style skew slope")
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merge_payload(
            {
                "style": self.style,
                "page_classes": list(self.page_classes),
                "page_count": self.page_count,
                "line_count": self.line_count,
                "estimates": [estimate.to_dict() for estimate in self.estimates],
                "skew_slope_median": self.skew_slope_median,
                "skew_slope_mad": self.skew_slope_mad,
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class BookTypography:
    """One PGDP project's pooled typography, its styles, and its per-page aggregates.

    `estimates` is the second pooling stage over every page that belongs to a style, so its
    `sample_count` is a page count and `extensions.line_sample_count` records the lines
    underneath. Pages are in natural page order and page names are unique.
    """

    project_id: str
    pages: tuple[PageTypography, ...] = ()
    styles: tuple[StyleTypography, ...] = ()
    estimates: tuple[Estimate, ...] = ()
    accepted_page_count: int = 0
    measured_page_count: int = 0
    excluded_page_count: int = 0
    unknown_class_page_count: int = 0
    matched_line_count: int = 0
    measured_line_count: int = 0
    reconciled_line_count: int = 0
    skew_slope_median: float | None = None
    skew_slope_mad: float | None = None
    exclusions: tuple[str, ...] = ()
    diagnostics: tuple[ProfileDiagnostic, ...] = ()
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ = _require_string(self.project_id, name="Project ID", nonempty=True)
        object.__setattr__(self, "pages", tuple(self.pages))
        object.__setattr__(self, "styles", tuple(self.styles))
        object.__setattr__(self, "estimates", tuple(self.estimates))
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if any(not isinstance(page, PageTypography) for page in self.pages):
            raise TypeError("pages must contain PageTypography records.")
        if any(not isinstance(style, StyleTypography) for style in self.styles):
            raise TypeError("styles must contain StyleTypography records.")
        if any(not isinstance(exclusion, str) for exclusion in self.exclusions):
            raise TypeError("exclusions must contain strings.")
        if any(not isinstance(diagnostic, ProfileDiagnostic) for diagnostic in self.diagnostics):
            raise TypeError("diagnostics must contain ProfileDiagnostic records.")
        for name in (
            "accepted_page_count",
            "measured_page_count",
            "excluded_page_count",
            "unknown_class_page_count",
            "matched_line_count",
            "measured_line_count",
            "reconciled_line_count",
        ):
            _require_nonnegative_integer(getattr(self, name), name=name)
        if self.measured_page_count + self.excluded_page_count != self.accepted_page_count:
            raise ValueError("Measured and excluded pages must account for every accepted page.")
        if self.measured_line_count > self.matched_line_count:
            raise ValueError("measured_line_count cannot exceed matched_line_count.")
        if self.reconciled_line_count > self.measured_line_count:
            raise ValueError("reconciled_line_count cannot exceed measured_line_count.")
        page_names = tuple(page.page_name for page in self.pages)
        if len(set(page_names)) != len(page_names):
            raise ValueError("Book typography contains duplicate page names.")
        if page_names != tuple(sorted(page_names, key=natural_page_key)):
            raise ValueError("Book typography pages must use natural page order.")
        style_names = tuple(style.style for style in self.styles)
        if len(set(style_names)) != len(style_names):
            raise ValueError("Book typography contains duplicate styles.")
        if style_names != tuple(sorted(style_names)):
            raise ValueError("Book typography styles must be sorted by name.")
        _require_estimates(
            self.estimates, name="Book typography", allowed_pages=frozenset(page_names)
        )
        for style in self.styles:
            _require_estimates(
                style.estimates, name="Style typography", allowed_pages=frozenset(page_names)
            )
        _require_skew(self.skew_slope_median, self.skew_slope_mad, name="Book skew slope")
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merge_payload(
            {
                "project_id": self.project_id,
                "accepted_page_count": self.accepted_page_count,
                "measured_page_count": self.measured_page_count,
                "excluded_page_count": self.excluded_page_count,
                "unknown_class_page_count": self.unknown_class_page_count,
                "matched_line_count": self.matched_line_count,
                "measured_line_count": self.measured_line_count,
                "reconciled_line_count": self.reconciled_line_count,
                "estimates": [estimate.to_dict() for estimate in self.estimates],
                "styles": [style.to_dict() for style in self.styles],
                "pages": [page.to_dict() for page in self.pages],
                "skew_slope_median": self.skew_slope_median,
                "skew_slope_mad": self.skew_slope_mad,
                "exclusions": list(self.exclusions),
                "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class TypographyReport:
    """Stable JSON-safe output from a font-free PGDP typography run."""

    tool_version: str
    alignment_label: str
    alignment_sha256: str
    profile_label: str
    profile_sha256: str
    methods: Mapping[str, object]
    thresholds: Mapping[str, object]
    evidence_pages_per_book: int = DEFAULT_EVIDENCE_PAGES_PER_BOOK
    books: tuple[BookTypography, ...] = ()
    diagnostics: tuple[ProfileDiagnostic, ...] = ()
    schema_version: int = TYPOGRAPHY_SCHEMA_VERSION
    algorithm_version: str = TYPOGRAPHY_ALGORITHM_VERSION
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != TYPOGRAPHY_SCHEMA_VERSION
        ):
            raise ValueError(f"Unsupported schema_version {self.schema_version!r}.")
        if self.algorithm_version != TYPOGRAPHY_ALGORITHM_VERSION:
            raise ValueError(f"Unsupported algorithm_version {self.algorithm_version!r}.")
        _ = _require_string(self.tool_version, name="tool_version", nonempty=True)
        _ = _require_file_label(self.alignment_label, name="alignment_label")
        _ = _require_file_label(self.profile_label, name="profile_label")
        _require_sha256(self.alignment_sha256, name="alignment_sha256")
        _require_sha256(self.profile_sha256, name="profile_sha256")
        _require_nonnegative_integer(self.evidence_pages_per_book, name="evidence_pages_per_book")
        object.__setattr__(self, "methods", _normalized_mapping(self.methods))
        object.__setattr__(self, "thresholds", _normalized_mapping(self.thresholds))
        object.__setattr__(self, "books", tuple(self.books))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if any(not isinstance(book, BookTypography) for book in self.books):
            raise TypeError("books must contain BookTypography records.")
        if any(not isinstance(diagnostic, ProfileDiagnostic) for diagnostic in self.diagnostics):
            raise TypeError("diagnostics must contain ProfileDiagnostic records.")
        project_ids = tuple(book.project_id for book in self.books)
        if len(set(project_ids)) != len(project_ids):
            raise ValueError("Typography report contains duplicate project IDs.")
        if project_ids != tuple(sorted(project_ids, key=natural_page_key)):
            raise ValueError("Typography report books must use natural project order.")
        for book in self.books:
            evidence_pages = sum(1 for page in book.pages if page.evidence_sampled)
            if evidence_pages > self.evidence_pages_per_book:
                raise ValueError("A book may not emit more evidence pages than the report's bound.")
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merge_payload(
            {
                "schema_version": self.schema_version,
                "algorithm_version": self.algorithm_version,
                "tool_version": self.tool_version,
                "alignment_label": self.alignment_label,
                "alignment_sha256": self.alignment_sha256,
                "profile_label": self.profile_label,
                "profile_sha256": self.profile_sha256,
                "methods": _mapping_dict(self.methods),
                "thresholds": _mapping_dict(self.thresholds),
                "evidence_pages_per_book": self.evidence_pages_per_book,
                "books": [book.to_dict() for book in self.books],
                "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            },
            self.extensions,
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, payload: object) -> TypographyReport:
        try:
            return TypographyReportInput.model_validate(payload).to_domain()
        except (TypeError, ValidationError) as error:
            raise ValueError(f"Invalid PGDP typography report: {error}") from error

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> TypographyReport:
        try:
            return TypographyReportInput.model_validate_json(payload).to_domain()
        except (TypeError, ValidationError) as error:
            raise ValueError(f"Invalid PGDP typography report: {error}") from error

    @classmethod
    def json_schema(cls) -> str:
        return (
            json.dumps(TypographyReportInput.model_json_schema(), indent=2, ensure_ascii=False)
            + "\n"
        )


def iter_payload_keys(payload: object, *, path: str = "") -> Iterator[tuple[str, str]]:
    """Every object key in a JSON payload, paired with the path that reached it."""

    if isinstance(payload, RuntimeMapping):
        for key, value in payload.items():
            key_path = f"{path}.{key}" if path else str(key)
            yield str(key), key_path
            yield from iter_payload_keys(value, path=key_path)
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            yield from iter_payload_keys(value, path=f"{path}[{index}]")


def find_latent_discipline_violations(payload: object) -> tuple[str, ...]:
    """Every place a payload claims something `pgdp-typography/v1` treats as latent.

    Returns sorted `"path: reason"` strings, empty when the payload is clean. Two rules are
    checked: no object key may contain a `LATENT_KEY_DENYLIST` substring, and every estimate
    must be an uncalibrated px measurement in the `source` frame. Task 8 runs this against the
    real corpus report, which is why it lives here rather than inside a test.
    """

    violations: list[str] = []
    for key, key_path in iter_payload_keys(payload):
        lowered = key.lower()
        violations.extend(
            f"{key_path}: key contains latent term {term!r}"
            for term in sorted(LATENT_KEY_DENYLIST)
            if term in lowered
        )
    violations.extend(_find_estimate_shape_violations(payload))
    return tuple(sorted(violations))


def _find_estimate_shape_violations(payload: object, *, path: str = "") -> Iterator[str]:
    if isinstance(payload, RuntimeMapping):
        if "estimates" in payload:
            estimates = payload["estimates"]
            if isinstance(estimates, (list, tuple)):
                for index, estimate in enumerate(estimates):
                    estimate_path = f"{path}.estimates[{index}]" if path else f"estimates[{index}]"
                    yield from _check_estimate_shape(estimate, path=estimate_path)
        for key, value in payload.items():
            yield from _find_estimate_shape_violations(
                value, path=f"{path}.{key}" if path else str(key)
            )
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            yield from _find_estimate_shape_violations(value, path=f"{path}[{index}]")


def _check_estimate_shape(estimate: object, *, path: str) -> Iterator[str]:
    if not isinstance(estimate, RuntimeMapping):
        yield f"{path}: estimate is not an object"
        return
    for key, expected in (
        ("unit", "px"),
        ("frame", "source"),
        ("confidence_kind", TYPOGRAPHY_CONFIDENCE_KIND),
    ):
        actual = estimate.get(key)
        if actual != expected:
            yield f"{path}: {key} is {actual!r}, expected {expected!r}"
    if estimate.get("confidence") is not None:
        yield f"{path}: confidence must be null"


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)


def _wire_extensions(model: _WireModel) -> dict[str, JsonValue]:
    extras = model.model_extra
    return {} if extras is None else _JSON_OBJECT_ADAPTER.validate_python(extras)


class WordTypographyInput(_WireModel):
    ordinal: StrictInt = Field(ge=0)
    box: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    following_gap_px: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_sequences(cls, payload: object) -> object:
        return _normalize_tuple_fields(payload, ("box",))

    def to_domain(self) -> WordTypography:
        return WordTypography(
            ordinal=self.ordinal,
            box=self.box,
            following_gap_px=self.following_gap_px,
            extensions=_wire_extensions(self),
        )


class LineTypographyInput(_WireModel):
    candidate_ordinal: StrictInt = Field(ge=0)
    source_ordinal: StrictInt | None = Field(default=None, ge=0)
    box: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    line_ink_width_px: StrictInt = Field(ge=0)
    line_ink_height_px: StrictInt = Field(ge=0)
    line_indent_px: StrictInt | None = None
    baseline_row_px: StrictInt | None = Field(default=None, ge=0)
    x_height_top_row_px: StrictInt | None = Field(default=None, ge=0)
    x_height_px: StrictInt | None = Field(default=None, ge=0)
    ascender_extent_px: StrictInt | None = Field(default=None, ge=0)
    descender_extent_px: StrictInt | None = Field(default=None, ge=0)
    stroke_width_px: StrictFloat | None = Field(default=None, ge=0)
    skew_slope: StrictFloat | None = None
    segment_count: StrictInt = Field(ge=0)
    word_run_count: StrictInt | None = Field(default=None, ge=0)
    source_word_count: StrictInt | None = Field(default=None, ge=0)
    words_reconciled: StrictBool | None = None
    words: tuple[WordTypographyInput, ...] = ()
    exclusions: tuple[StrictStr, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _normalize_sequences(cls, payload: object) -> object:
        return _normalize_tuple_fields(payload, ("box", "words", "exclusions"))

    def to_domain(self) -> LineTypography:
        return LineTypography(
            candidate_ordinal=self.candidate_ordinal,
            source_ordinal=self.source_ordinal,
            box=self.box,
            line_ink_width_px=self.line_ink_width_px,
            line_ink_height_px=self.line_ink_height_px,
            line_indent_px=self.line_indent_px,
            baseline_row_px=self.baseline_row_px,
            x_height_top_row_px=self.x_height_top_row_px,
            x_height_px=self.x_height_px,
            ascender_extent_px=self.ascender_extent_px,
            descender_extent_px=self.descender_extent_px,
            stroke_width_px=self.stroke_width_px,
            skew_slope=self.skew_slope,
            segment_count=self.segment_count,
            word_run_count=self.word_run_count,
            source_word_count=self.source_word_count,
            words_reconciled=self.words_reconciled,
            words=tuple(word.to_domain() for word in self.words),
            exclusions=self.exclusions,
            extensions=_wire_extensions(self),
        )


class PageTypographyInput(_WireModel):
    page_name: StrictStr = Field(min_length=1)
    page_class: StrictStr = Field(min_length=1)
    scan_sha256: StrictStr | None = None
    source_frame: CoordinateFrameWire | None = None
    matched_line_count: StrictInt = Field(ge=0)
    measured_line_count: StrictInt = Field(ge=0)
    excluded_line_count: StrictInt = Field(ge=0)
    reconciled_line_count: StrictInt = Field(ge=0)
    baseline_pitch_count: StrictInt = Field(ge=0)
    evidence_sampled: StrictBool = False
    lines: tuple[LineTypographyInput, ...] = ()
    estimates: tuple[EstimateWire, ...] = ()
    skew_slope_median: StrictFloat | None = None
    skew_slope_mad: StrictFloat | None = Field(default=None, ge=0)
    exclusions: tuple[StrictStr, ...] = ()
    exclusion_evidence: tuple[dict[str, JsonValue], ...] = ()
    diagnostics: tuple[ProfileDiagnosticWire, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _normalize_sequences(cls, payload: object) -> object:
        return _normalize_tuple_fields(
            payload,
            ("lines", "estimates", "exclusions", "exclusion_evidence", "diagnostics"),
        )

    def to_domain(self) -> PageTypography:
        return PageTypography(
            page_name=self.page_name,
            page_class=self.page_class,
            scan_sha256=self.scan_sha256,
            source_frame=None if self.source_frame is None else self.source_frame.to_domain(),
            matched_line_count=self.matched_line_count,
            measured_line_count=self.measured_line_count,
            excluded_line_count=self.excluded_line_count,
            reconciled_line_count=self.reconciled_line_count,
            baseline_pitch_count=self.baseline_pitch_count,
            evidence_sampled=self.evidence_sampled,
            lines=tuple(line.to_domain() for line in self.lines),
            estimates=tuple(estimate.to_domain() for estimate in self.estimates),
            skew_slope_median=self.skew_slope_median,
            skew_slope_mad=self.skew_slope_mad,
            exclusions=self.exclusions,
            exclusion_evidence=self.exclusion_evidence,
            diagnostics=tuple(diagnostic.to_domain() for diagnostic in self.diagnostics),
            extensions=_wire_extensions(self),
        )


class StyleTypographyInput(_WireModel):
    style: Literal["body", "chapter_opening"]
    page_classes: tuple[StrictStr, ...]
    page_count: StrictInt = Field(ge=0)
    line_count: StrictInt = Field(ge=0)
    estimates: tuple[EstimateWire, ...] = ()
    skew_slope_median: StrictFloat | None = None
    skew_slope_mad: StrictFloat | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_sequences(cls, payload: object) -> object:
        return _normalize_tuple_fields(payload, ("page_classes", "estimates"))

    def to_domain(self) -> StyleTypography:
        return StyleTypography(
            style=self.style,
            page_classes=self.page_classes,
            page_count=self.page_count,
            line_count=self.line_count,
            estimates=tuple(estimate.to_domain() for estimate in self.estimates),
            skew_slope_median=self.skew_slope_median,
            skew_slope_mad=self.skew_slope_mad,
            extensions=_wire_extensions(self),
        )


class BookTypographyInput(_WireModel):
    project_id: StrictStr = Field(min_length=1)
    accepted_page_count: StrictInt = Field(default=0, ge=0)
    measured_page_count: StrictInt = Field(default=0, ge=0)
    excluded_page_count: StrictInt = Field(default=0, ge=0)
    unknown_class_page_count: StrictInt = Field(default=0, ge=0)
    matched_line_count: StrictInt = Field(default=0, ge=0)
    measured_line_count: StrictInt = Field(default=0, ge=0)
    reconciled_line_count: StrictInt = Field(default=0, ge=0)
    estimates: tuple[EstimateWire, ...] = ()
    styles: tuple[StyleTypographyInput, ...] = ()
    pages: tuple[PageTypographyInput, ...] = ()
    skew_slope_median: StrictFloat | None = None
    skew_slope_mad: StrictFloat | None = Field(default=None, ge=0)
    exclusions: tuple[StrictStr, ...] = ()
    diagnostics: tuple[ProfileDiagnosticWire, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _normalize_sequences(cls, payload: object) -> object:
        return _normalize_tuple_fields(
            payload, ("estimates", "styles", "pages", "exclusions", "diagnostics")
        )

    def to_domain(self) -> BookTypography:
        return BookTypography(
            project_id=self.project_id,
            accepted_page_count=self.accepted_page_count,
            measured_page_count=self.measured_page_count,
            excluded_page_count=self.excluded_page_count,
            unknown_class_page_count=self.unknown_class_page_count,
            matched_line_count=self.matched_line_count,
            measured_line_count=self.measured_line_count,
            reconciled_line_count=self.reconciled_line_count,
            estimates=tuple(estimate.to_domain() for estimate in self.estimates),
            styles=tuple(style.to_domain() for style in self.styles),
            pages=tuple(page.to_domain() for page in self.pages),
            skew_slope_median=self.skew_slope_median,
            skew_slope_mad=self.skew_slope_mad,
            exclusions=self.exclusions,
            diagnostics=tuple(diagnostic.to_domain() for diagnostic in self.diagnostics),
            extensions=_wire_extensions(self),
        )


class TypographyReportInput(_WireModel):
    schema_version: Literal[1]
    algorithm_version: Literal["pgdp-typography/v1"]
    tool_version: StrictStr = Field(min_length=1)
    alignment_label: StrictStr = Field(min_length=1, pattern=r"^[^/\\]+$")
    alignment_sha256: StrictStr
    profile_label: StrictStr = Field(min_length=1, pattern=r"^[^/\\]+$")
    profile_sha256: StrictStr
    methods: dict[str, JsonValue]
    thresholds: dict[str, JsonValue]
    evidence_pages_per_book: StrictInt = Field(default=DEFAULT_EVIDENCE_PAGES_PER_BOOK, ge=0)
    books: tuple[BookTypographyInput, ...] = ()
    diagnostics: tuple[ProfileDiagnosticWire, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _normalize_sequences(cls, payload: object) -> object:
        normalized = _normalize_tuple_fields(payload, ("books", "diagnostics"))
        if not isinstance(normalized, dict):
            return normalized
        schema_version = normalized.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise TypeError("schema_version must be the exact integer 1.")
        return normalized

    @model_validator(mode="after")
    def _validate_labels(self) -> TypographyReportInput:
        for name in ("alignment_label", "profile_label"):
            _ = _require_file_label(getattr(self, name), name=name)
        return self

    def to_domain(self) -> TypographyReport:
        return TypographyReport(
            schema_version=self.schema_version,
            algorithm_version=self.algorithm_version,
            tool_version=self.tool_version,
            alignment_label=self.alignment_label,
            alignment_sha256=self.alignment_sha256,
            profile_label=self.profile_label,
            profile_sha256=self.profile_sha256,
            methods=self.methods,
            thresholds=self.thresholds,
            evidence_pages_per_book=self.evidence_pages_per_book,
            books=tuple(book.to_domain() for book in self.books),
            diagnostics=tuple(diagnostic.to_domain() for diagnostic in self.diagnostics),
            extensions=_wire_extensions(self),
        )
