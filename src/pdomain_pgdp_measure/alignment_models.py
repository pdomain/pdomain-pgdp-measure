"""Versioned wire records for conservative PGDP source-line alignment."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping as RuntimeMapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Self

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
from pdomain_pgdp_measure.profile_models import CoordinateFrame

if TYPE_CHECKING:
    from collections.abc import Mapping


ALIGNMENT_SCHEMA_VERSION: Final = 1
ALIGNMENT_ALGORITHM_VERSION: Final = "pgdp-alignment/v3"
CONFIDENCE_KIND: Final = "uncalibrated"

AlignmentOperationKind = Literal["match", "skip_image", "skip_text"]
AlignmentState = Literal["accepted", "proposed", "excluded"]
BlockKind = Literal["local", "continued"]
WireStyleKind = Literal[
    "i",
    "b",
    "sc",
    "f",
    "g",
    "tb",
]
Bounds = tuple[int, int, int, int] | list[int]

_JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, JsonValue])


def _require_nonnegative_integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


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


def _require_finite(value: object, *, name: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _require_nonnegative_finite(value: object, *, name: str) -> None:
    if _require_finite(value, name=name) < 0.0:
        raise ValueError(f"{name} must be nonnegative.")


def _require_sha256(value: object, *, name: str) -> None:
    hexadecimal_characters = "0123456789abcdefABCDEF"
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in hexadecimal_characters for character in value)
    ):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal characters.")


def _normalize_box(
    value: Bounds, *, name: str, frame: CoordinateFrame | None = None
) -> tuple[int, int, int, int]:
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
    if frame is not None and (x_end > frame.width or y_end > frame.height):
        raise ValueError(f"{name} must stay inside the source frame.")
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
                raise TypeError("Stored extension object keys must be strings.")
            thawed[key] = _thaw_json_value(item)
        return thawed
    raise ValueError("Stored extension value is not JSON-native.")


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


@dataclass(frozen=True, slots=True)
class AlignmentDiagnostic:
    """One diagnostic retained by the alignment report."""

    code: str
    message: str
    project_id: str | None = None
    page_name: str | None = None
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ = _require_string(self.code, name="Alignment diagnostic code", nonempty=True)
        _ = _require_string(self.message, name="Alignment diagnostic message", nonempty=True)
        if self.project_id is not None:
            _ = _require_string(self.project_id, name="project_id")
        if self.page_name is not None:
            _ = _require_string(self.page_name, name="page_name")
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merge_payload(
            {
                "code": self.code,
                "message": self.message,
                "project_id": self.project_id,
                "page_name": self.page_name,
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class WireSourceLine:
    """Lossless source-line evidence serialized into one page record."""

    ordinal: int
    byte_start: int
    byte_end: int
    content_byte_start: int
    content_byte_end: int
    separator_byte_start: int
    separator_byte_end: int
    original_text: str
    separator_text: str
    visible_text: str
    matching_eligible: bool
    style_fitting_eligible: bool
    operation_ordinals: tuple[int, ...] = ()
    formatting_span_ids: tuple[str, ...] = ()
    continued_block_ids: tuple[str, ...] = ()
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("ordinal", "byte_start", "byte_end"):
            _require_nonnegative_integer(getattr(self, name), name=name)
        if self.byte_end < self.byte_start:
            raise ValueError("Source-line UTF-8 span must be half-open and non-inverted.")
        boundaries: tuple[int, int, int, int] = (
            self.content_byte_start,
            self.content_byte_end,
            self.separator_byte_start,
            self.separator_byte_end,
        )
        for name, value in zip(
            (
                "content_byte_start",
                "content_byte_end",
                "separator_byte_start",
                "separator_byte_end",
            ),
            boundaries,
            strict=True,
        ):
            _require_nonnegative_integer(value, name=name)
        content_start, content_end, separator_start, separator_end = boundaries
        if not (
            self.byte_start
            == content_start
            <= content_end
            == separator_start
            <= separator_end
            == self.byte_end
        ):
            raise ValueError("Source-line spans must partition the line UTF-8 range.")
        _ = _require_string(self.original_text, name="Source-line original_text")
        _ = _require_string(self.separator_text, name="Source-line separator_text")
        _ = _require_string(self.visible_text, name="Source-line visible_text")
        _ = _require_bool(self.matching_eligible, name="matching_eligible")
        _ = _require_bool(self.style_fitting_eligible, name="style_fitting_eligible")
        if len(self.original_text.encode("utf-8")) != content_end - content_start:
            raise ValueError("Source-line original_text must exactly fill its content UTF-8 span.")
        if len(self.separator_text.encode("utf-8")) != separator_end - separator_start:
            raise ValueError("Source-line separator_text must exactly fill its UTF-8 span.")
        object.__setattr__(self, "operation_ordinals", tuple(self.operation_ordinals))
        object.__setattr__(self, "formatting_span_ids", tuple(self.formatting_span_ids))
        object.__setattr__(self, "continued_block_ids", tuple(self.continued_block_ids))
        for operation_ordinal in self.operation_ordinals:
            _require_nonnegative_integer(operation_ordinal, name="operation_ordinal")
        if tuple(sorted(set(self.operation_ordinals))) != self.operation_ordinals:
            raise ValueError("Source-line operation ordinals must be sorted and unique.")
        if any(not isinstance(span_id, str) for span_id in self.formatting_span_ids):
            raise TypeError("Source-line formatting span IDs must be strings.")
        if any(not isinstance(block_id, str) for block_id in self.continued_block_ids):
            raise TypeError("Source-line continued block IDs must be strings.")
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merge_payload(
            {
                "ordinal": self.ordinal,
                "byte_start": self.byte_start,
                "byte_end": self.byte_end,
                "content_byte_start": self.content_byte_start,
                "content_byte_end": self.content_byte_end,
                "separator_byte_start": self.separator_byte_start,
                "separator_byte_end": self.separator_byte_end,
                "original_text": self.original_text,
                "separator_text": self.separator_text,
                "visible_text": self.visible_text,
                "operation_ordinals": list(self.operation_ordinals),
                "formatting_span_ids": list(self.formatting_span_ids),
                "continued_block_ids": list(self.continued_block_ids),
                "matching_eligible": self.matching_eligible,
                "style_fitting_eligible": self.style_fitting_eligible,
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class WireFormattingSpan:
    """One source formatting span retained with byte-addressed evidence."""

    opening_id: str
    block_id: str
    kind: BlockKind
    byte_start: int
    byte_end: int
    closed: bool
    page_name: str | None = None
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ = _require_string(self.kind, name="Formatting span kind")
        _ = _require_string(self.opening_id, name="opening_id", nonempty=True)
        _ = _require_string(self.block_id, name="block_id", nonempty=True)
        if self.kind not in {"local", "continued"}:
            raise ValueError("Formatting span kind is unsupported.")
        if self.page_name is not None:
            _ = _require_string(self.page_name, name="page_name")
        _ = _require_bool(self.closed, name="closed")
        _require_nonnegative_integer(self.byte_start, name="byte_start")
        _require_nonnegative_integer(self.byte_end, name="byte_end")
        if self.byte_end < self.byte_start:
            raise ValueError("Formatting span must be half-open and non-inverted.")
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merge_payload(
            {
                "opening_id": self.opening_id,
                "block_id": self.block_id,
                "kind": self.kind,
                "page_name": self.page_name,
                "byte_start": self.byte_start,
                "byte_end": self.byte_end,
                "closed": self.closed,
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class WireStyleRun:
    """One typed source style run with normalized and UTF-8 byte evidence."""

    kind: WireStyleKind
    normalized_start: int
    normalized_end: int
    byte_start: int
    byte_end: int
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ = _require_string(self.kind, name="Style-run kind")
        if self.kind not in {"i", "b", "sc", "f", "g", "tb"}:
            raise ValueError("Style-run kind is unsupported.")
        for name in ("normalized_start", "normalized_end", "byte_start", "byte_end"):
            _require_nonnegative_integer(getattr(self, name), name=name)
        if self.normalized_end < self.normalized_start:
            raise ValueError("Style-run normalized span must be half-open and non-inverted.")
        if self.byte_end < self.byte_start:
            raise ValueError("Style-run UTF-8 span must be half-open and non-inverted.")
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merge_payload(
            {
                "kind": self.kind,
                "normalized_start": self.normalized_start,
                "normalized_end": self.normalized_end,
                "byte_start": self.byte_start,
                "byte_end": self.byte_end,
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class WireLineCandidate:
    """Source-frame visual line evidence used by a page alignment."""

    ordinal: int
    box: Bounds
    width: int
    height: int
    fill_ratio: float
    component_count: int
    foreground_pixels: int
    band_ordinal: int | None = None
    component_ordinal: int | None = None
    horizontal_ink_profile: tuple[float, ...] = ()
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.ordinal, name="ordinal")
        normalized_box = _normalize_box(self.box, name="Candidate box")
        object.__setattr__(self, "box", normalized_box)
        _require_nonnegative_integer(self.width, name="width")
        _require_nonnegative_integer(self.height, name="height")
        if (
            self.width != normalized_box[2] - normalized_box[0]
            or self.height != normalized_box[3] - normalized_box[1]
        ):
            raise ValueError("Candidate dimensions must match its source-frame box.")
        _ = _require_finite(self.fill_ratio, name="fill_ratio")
        if not 0.0 < self.fill_ratio <= 1.0:
            raise ValueError("Candidate fill_ratio must be in (0, 1].")
        _require_nonnegative_integer(self.component_count, name="component_count")
        _require_nonnegative_integer(self.foreground_pixels, name="foreground_pixels")
        if self.component_count == 0 or self.foreground_pixels == 0:
            raise ValueError("Candidates require components and foreground pixels.")
        if self.band_ordinal is not None:
            _require_nonnegative_integer(self.band_ordinal, name="band_ordinal")
        if self.component_ordinal is not None:
            _require_nonnegative_integer(self.component_ordinal, name="component_ordinal")
        object.__setattr__(self, "horizontal_ink_profile", tuple(self.horizontal_ink_profile))
        if self.horizontal_ink_profile and len(self.horizontal_ink_profile) != self.width:
            raise ValueError("Candidate horizontal profile must match its width.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in self.horizontal_ink_profile
        ):
            raise TypeError("Candidate horizontal profile values must be numbers.")
        if any(not math.isfinite(value) or value < 0.0 for value in self.horizontal_ink_profile):
            raise ValueError("Candidate horizontal profile values must be finite and nonnegative.")
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merge_payload(
            {
                "ordinal": self.ordinal,
                "band_ordinal": self.band_ordinal,
                "component_ordinal": self.component_ordinal,
                "box": list(self.box),
                "component_count": self.component_count,
                "foreground_pixels": self.foreground_pixels,
                "width": self.width,
                "height": self.height,
                "fill_ratio": self.fill_ratio,
                "horizontal_ink_profile": list(self.horizontal_ink_profile),
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class AlignmentOperation:
    """One ordered source-to-image edit operation with fixed cost evidence."""

    kind: AlignmentOperationKind
    source_ordinal: int | None
    candidate_ordinal: int | None
    cost: float
    width_cost: float | None = None
    indentation_cost: float | None = None
    gaps_cost: float | None = None
    style_cost: float | None = None
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ = _require_string(self.kind, name="Alignment operation kind")
        if self.kind not in {"match", "skip_image", "skip_text"}:
            raise ValueError("Alignment operation kind is unsupported.")
        if self.kind == "match" and (self.source_ordinal is None or self.candidate_ordinal is None):
            raise ValueError("Match operations require source and candidate ordinals.")
        if self.kind == "skip_image" and (
            self.source_ordinal is not None or self.candidate_ordinal is None
        ):
            raise ValueError("skip_image operations require only a candidate ordinal.")
        if self.kind == "skip_text" and (
            self.source_ordinal is None or self.candidate_ordinal is not None
        ):
            raise ValueError("skip_text operations require only a source ordinal.")
        if self.source_ordinal is not None:
            _require_nonnegative_integer(self.source_ordinal, name="source_ordinal")
        if self.candidate_ordinal is not None:
            _require_nonnegative_integer(self.candidate_ordinal, name="candidate_ordinal")
        _require_nonnegative_finite(self.cost, name="Alignment operation cost")
        for name in ("width_cost", "indentation_cost", "gaps_cost", "style_cost"):
            value = getattr(self, name)
            if value is not None:
                _require_nonnegative_finite(value, name=name)
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merge_payload(
            {
                "kind": self.kind,
                "source_ordinal": self.source_ordinal,
                "candidate_ordinal": self.candidate_ordinal,
                "cost": self.cost,
                "width_cost": self.width_cost,
                "indentation_cost": self.indentation_cost,
                "gaps_cost": self.gaps_cost,
                "style_cost": self.style_cost,
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class PageAlignment:
    """All immutable alignment evidence and assessment for one source page."""

    page_name: str
    f2_sha256: str | None
    scan_sha256: str | None
    source_frame: CoordinateFrame | None
    source_lines: tuple[WireSourceLine, ...]
    formatting_spans: tuple[WireFormattingSpan, ...]
    candidates: tuple[WireLineCandidate, ...]
    operations: tuple[AlignmentOperation, ...]
    total_cost: float
    second_best_cost: float | None
    normalized_cost: float
    matched_count: int
    matched_ratio: float
    uniqueness_margin: float
    mean_width_residual: float | None
    mean_indentation_residual: float | None
    state: AlignmentState
    accepted: bool
    exclusions: tuple[str, ...]
    diagnostics: tuple[AlignmentDiagnostic, ...] = ()
    style_runs: tuple[WireStyleRun, ...] = ()
    exclusion_evidence: tuple[Mapping[str, object], ...] = ()
    confidence: None = None
    confidence_kind: Literal["uncalibrated"] = CONFIDENCE_KIND
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ = _require_string(self.page_name, name="Page name", nonempty=True)
        if self.f2_sha256 is not None:
            _require_sha256(self.f2_sha256, name="f2_sha256")
        if self.scan_sha256 is not None:
            _require_sha256(self.scan_sha256, name="scan_sha256")
        if self.source_frame is not None and not isinstance(self.source_frame, CoordinateFrame):
            raise TypeError("source_frame must be a CoordinateFrame or null.")
        if self.source_frame is None and (self.candidates or self.scan_sha256 is not None):
            raise ValueError("Candidate or scan evidence requires a source frame.")
        object.__setattr__(self, "source_lines", tuple(self.source_lines))
        object.__setattr__(self, "formatting_spans", tuple(self.formatting_spans))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "style_runs", tuple(self.style_runs))
        object.__setattr__(
            self,
            "exclusion_evidence",
            tuple(_normalized_mapping(item) for item in self.exclusion_evidence),
        )
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        if any(not isinstance(line, WireSourceLine) for line in self.source_lines):
            raise TypeError("source_lines must contain WireSourceLine records.")
        if any(not isinstance(span, WireFormattingSpan) for span in self.formatting_spans):
            raise TypeError("formatting_spans must contain WireFormattingSpan records.")
        if any(not isinstance(candidate, WireLineCandidate) for candidate in self.candidates):
            raise TypeError("candidates must contain WireLineCandidate records.")
        if any(not isinstance(operation, AlignmentOperation) for operation in self.operations):
            raise TypeError("operations must contain AlignmentOperation records.")
        if any(not isinstance(diagnostic, AlignmentDiagnostic) for diagnostic in self.diagnostics):
            raise TypeError("diagnostics must contain AlignmentDiagnostic records.")
        if any(not isinstance(style_run, WireStyleRun) for style_run in self.style_runs):
            raise TypeError("style_runs must contain WireStyleRun records.")
        if any(not isinstance(exclusion, str) for exclusion in self.exclusions):
            raise TypeError("exclusions must contain strings.")
        if self.source_frame is not None:
            for candidate in self.candidates:
                _ = _normalize_box(candidate.box, name="Candidate box", frame=self.source_frame)
        self._validate_ordinals()
        self._validate_source_line_continuity()
        self._validate_formatting_spans()
        self._validate_source_line_formatting_references()
        self._validate_style_runs()
        self._validate_operations()
        for name in ("total_cost", "normalized_cost", "uniqueness_margin"):
            _require_nonnegative_finite(getattr(self, name), name=name)
        _ = _require_finite(self.matched_ratio, name="matched_ratio")
        for name in ("second_best_cost", "mean_width_residual", "mean_indentation_residual"):
            value = getattr(self, name)
            if value is not None:
                _require_nonnegative_finite(value, name=name)
        _require_nonnegative_integer(self.matched_count, name="matched_count")
        if not 0.0 <= self.matched_ratio <= 1.0:
            raise ValueError("matched_ratio must be in [0, 1].")
        if self.second_best_cost is not None and self.second_best_cost < self.total_cost:
            raise ValueError("second_best_cost must be greater than or equal to total_cost.")
        if self.matched_count != sum(operation.kind == "match" for operation in self.operations):
            raise ValueError("matched_count must equal the number of match operations.")
        _ = _require_string(self.state, name="Alignment state")
        if self.state not in {"accepted", "proposed", "excluded"}:
            raise ValueError("Alignment state is unsupported.")
        _ = _require_bool(self.accepted, name="accepted")
        if self.accepted != (self.state == "accepted"):
            raise ValueError("accepted must exactly match the accepted state.")
        if self.f2_sha256 is None and (
            self.state != "excluded" or "f2_missing" not in self.exclusions
        ):
            raise ValueError("f2_sha256 can be null only for an excluded page with f2_missing.")
        if self.confidence is not None or self.confidence_kind != CONFIDENCE_KIND:
            raise ValueError("Alignment confidence must be null and uncalibrated in version 1.")
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def _validate_ordinals(self) -> None:
        source_ordinals = tuple(line.ordinal for line in self.source_lines)
        candidate_ordinals = tuple(candidate.ordinal for candidate in self.candidates)
        if source_ordinals != tuple(range(len(source_ordinals))):
            raise ValueError("Source-line ordinals must be consecutive from zero.")
        if candidate_ordinals != tuple(range(len(candidate_ordinals))):
            raise ValueError("Candidate ordinals must be consecutive from zero.")

    def _validate_operations(self) -> None:
        last_source = -1
        last_candidate = -1
        source_lines = {line.ordinal: line for line in self.source_lines}
        source_ordinals = set(source_lines)
        candidate_ordinals = {candidate.ordinal for candidate in self.candidates}
        required_source_ordinals = {
            line.ordinal
            for line in self.source_lines
            if line.matching_eligible and line.visible_text != ""
        }
        observed_source_ordinals: set[int] = set()
        observed_candidate_ordinals: set[int] = set()
        for operation in self.operations:
            if operation.source_ordinal is not None:
                if (
                    operation.source_ordinal not in source_ordinals
                    or operation.source_ordinal <= last_source
                ):
                    raise ValueError(
                        "Alignment operations must use increasing known source ordinals."
                    )
                source_line = source_lines[operation.source_ordinal]
                if not source_line.matching_eligible or source_line.visible_text == "":
                    raise ValueError(
                        "Alignment operations may reference only matching-eligible nonblank "
                        + "source lines."
                    )
                last_source = operation.source_ordinal
                observed_source_ordinals.add(operation.source_ordinal)
            if operation.candidate_ordinal is not None:
                if (
                    operation.candidate_ordinal not in candidate_ordinals
                    or operation.candidate_ordinal <= last_candidate
                ):
                    raise ValueError(
                        "Alignment operations must use increasing known candidate ordinals."
                    )
                last_candidate = operation.candidate_ordinal
                observed_candidate_ordinals.add(operation.candidate_ordinal)
        if observed_candidate_ordinals != candidate_ordinals:
            raise ValueError(
                "Alignment operations must cover every candidate ordinal exactly once."
            )
        if observed_source_ordinals != required_source_ordinals:
            raise ValueError(
                "Alignment operations must cover every matching-eligible nonblank source "
                + "line ordinal exactly once."
            )

    def _validate_source_line_continuity(self) -> None:
        previous_end = 0
        for source_line in self.source_lines:
            if source_line.byte_start != previous_end:
                raise ValueError("Source-line UTF-8 evidence must be contiguous from byte zero.")
            previous_end = source_line.byte_end

    def _validate_style_runs(self) -> None:
        normalized_visible_text_length = self._normalized_visible_text_length()
        if any(
            style_run.normalized_start > normalized_visible_text_length
            or style_run.normalized_end > normalized_visible_text_length
            for style_run in self.style_runs
        ):
            raise ValueError("Style-run spans must stay inside normalized visible text.")
        source_boundaries = self._source_utf8_boundaries()
        if any(
            style_run.byte_start not in source_boundaries
            or style_run.byte_end not in source_boundaries
            for style_run in self.style_runs
        ):
            raise ValueError(
                "Style-run spans must use UTF-8 codepoint boundary offsets in source evidence."
            )

    def _validate_formatting_spans(self) -> None:
        source_boundaries = self._source_utf8_boundaries()
        for formatting_span in self.formatting_spans:
            if (
                formatting_span.byte_start not in source_boundaries
                or formatting_span.byte_end not in source_boundaries
            ):
                raise ValueError(
                    "Formatting span endpoints must use UTF-8 codepoint boundary offsets."
                )
            if (
                formatting_span.page_name is not None
                and formatting_span.page_name != self.page_name
            ):
                raise ValueError("Formatting span page_name must match its page alignment.")

    def _validate_source_line_formatting_references(self) -> None:
        formatting_block_ids = {
            formatting_span.block_id for formatting_span in self.formatting_spans
        }
        continued_opening_ids = {
            formatting_span.opening_id
            for formatting_span in self.formatting_spans
            if formatting_span.kind == "continued"
        }
        for source_line in self.source_lines:
            if not set(source_line.formatting_span_ids) <= formatting_block_ids:
                raise ValueError(
                    "Source-line formatting_span_ids must resolve to page formatting block IDs."
                )
            if not set(source_line.continued_block_ids) <= continued_opening_ids:
                raise ValueError(
                    "Source-line continued_block_ids must resolve to continued opening IDs."
                )

    def _source_utf8_boundaries(self) -> frozenset[int]:
        source_text = "".join(
            source_line.original_text + source_line.separator_text
            for source_line in self.source_lines
        )
        byte_offset = 0
        boundaries = {byte_offset}
        for character in source_text:
            byte_offset += len(character.encode("utf-8"))
            boundaries.add(byte_offset)
        return frozenset(boundaries)

    def _normalized_visible_text_length(self) -> int:
        normalized_visible_text = "".join(
            source_line.visible_text + ("\n" if source_line.separator_text else "")
            for source_line in self.source_lines
        )
        return len(normalized_visible_text)

    def to_dict(self) -> dict[str, JsonValue]:
        source_frame = None if self.source_frame is None else self.source_frame.to_dict()
        return _merge_payload(
            {
                "page_name": self.page_name,
                "f2_sha256": self.f2_sha256,
                "scan_sha256": self.scan_sha256,
                "source_frame": source_frame,
                "source_lines": [line.to_dict() for line in self.source_lines],
                "formatting_spans": [span.to_dict() for span in self.formatting_spans],
                "style_runs": [style_run.to_dict() for style_run in self.style_runs],
                "candidates": [candidate.to_dict() for candidate in self.candidates],
                "operations": [operation.to_dict() for operation in self.operations],
                "total_cost": self.total_cost,
                "second_best_cost": self.second_best_cost,
                "normalized_cost": self.normalized_cost,
                "matched_count": self.matched_count,
                "matched_ratio": self.matched_ratio,
                "uniqueness_margin": self.uniqueness_margin,
                "mean_width_residual": self.mean_width_residual,
                "mean_indentation_residual": self.mean_indentation_residual,
                "state": self.state,
                "accepted": self.accepted,
                "exclusions": list(self.exclusions),
                "exclusion_evidence": [_mapping_dict(item) for item in self.exclusion_evidence],
                "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
                "confidence": self.confidence,
                "confidence_kind": self.confidence_kind,
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class ProjectAlignment:
    """Naturally ordered page alignments from one PGDP project."""

    project_id: str
    pages: tuple[PageAlignment, ...] = ()
    title: str | None = None
    author: str | None = None
    genre: str | None = None
    exclusions: tuple[str, ...] = ()
    diagnostics: tuple[AlignmentDiagnostic, ...] = ()
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ = _require_string(self.project_id, name="Project ID", nonempty=True)
        object.__setattr__(self, "pages", tuple(self.pages))
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        for name, value in (("title", self.title), ("author", self.author), ("genre", self.genre)):
            if value is not None:
                _ = _require_string(value, name=name)
        if any(not isinstance(page, PageAlignment) for page in self.pages):
            raise TypeError("pages must contain PageAlignment records.")
        if any(not isinstance(exclusion, str) for exclusion in self.exclusions):
            raise TypeError("exclusions must contain strings.")
        if any(not isinstance(diagnostic, AlignmentDiagnostic) for diagnostic in self.diagnostics):
            raise TypeError("diagnostics must contain AlignmentDiagnostic records.")
        page_names = tuple(page.page_name for page in self.pages)
        if len(set(page_names)) != len(page_names):
            raise ValueError("Project alignment contains duplicate page names.")
        if page_names != tuple(sorted(page_names, key=natural_page_key)):
            raise ValueError("Project alignment pages must use natural page order.")
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merge_payload(
            {
                "project_id": self.project_id,
                "title": self.title,
                "author": self.author,
                "genre": self.genre,
                "pages": [page.to_dict() for page in self.pages],
                "exclusions": list(self.exclusions),
                "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """Stable JSON-safe output from a PGDP source-line alignment run."""

    tool_version: str
    profile_label: str
    profile_sha256: str
    methods: Mapping[str, object]
    thresholds: Mapping[str, object]
    projects: tuple[ProjectAlignment, ...] = ()
    diagnostics: tuple[AlignmentDiagnostic, ...] = ()
    schema_version: int = ALIGNMENT_SCHEMA_VERSION
    algorithm_version: str = ALIGNMENT_ALGORITHM_VERSION
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != ALIGNMENT_SCHEMA_VERSION
        ):
            raise ValueError(f"Unsupported schema_version {self.schema_version!r}.")
        if self.algorithm_version != ALIGNMENT_ALGORITHM_VERSION:
            raise ValueError(f"Unsupported algorithm_version {self.algorithm_version!r}.")
        _ = _require_string(self.algorithm_version, name="algorithm_version")
        _ = _require_string(self.tool_version, name="tool_version", nonempty=True)
        _ = _require_string(self.profile_label, name="profile_label", nonempty=True)
        if (
            not self.profile_label
            or self.profile_label in {".", ".."}
            or "/" in self.profile_label
            or "\\" in self.profile_label
            or Path(self.profile_label).name != self.profile_label
        ):
            raise ValueError("profile_label must be a nonempty file name without path components.")
        _require_sha256(self.profile_sha256, name="profile_sha256")
        object.__setattr__(self, "methods", _normalized_mapping(self.methods))
        object.__setattr__(self, "thresholds", _normalized_mapping(self.thresholds))
        object.__setattr__(self, "projects", tuple(self.projects))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if any(not isinstance(project, ProjectAlignment) for project in self.projects):
            raise TypeError("projects must contain ProjectAlignment records.")
        if any(not isinstance(diagnostic, AlignmentDiagnostic) for diagnostic in self.diagnostics):
            raise TypeError("diagnostics must contain AlignmentDiagnostic records.")
        project_ids = tuple(project.project_id for project in self.projects)
        if len(set(project_ids)) != len(project_ids):
            raise ValueError("Alignment report contains duplicate project IDs.")
        if project_ids != tuple(sorted(project_ids, key=natural_page_key)):
            raise ValueError("Alignment report projects must use natural project order.")
        object.__setattr__(self, "extensions", _normalized_mapping(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merge_payload(
            {
                "schema_version": self.schema_version,
                "algorithm_version": self.algorithm_version,
                "tool_version": self.tool_version,
                "profile_label": self.profile_label,
                "profile_sha256": self.profile_sha256,
                "methods": _mapping_dict(self.methods),
                "thresholds": _mapping_dict(self.thresholds),
                "projects": [project.to_dict() for project in self.projects],
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
    def from_dict(cls, payload: object) -> AlignmentReport:
        try:
            return AlignmentReportInput.model_validate(payload).to_domain()
        except (TypeError, ValidationError) as error:
            raise ValueError(f"Invalid PGDP alignment report: {error}") from error

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> AlignmentReport:
        try:
            return AlignmentReportInput.model_validate_json(payload).to_domain()
        except (TypeError, ValidationError) as error:
            raise ValueError(f"Invalid PGDP alignment report: {error}") from error

    @classmethod
    def json_schema(cls) -> str:
        return (
            json.dumps(AlignmentReportInput.model_json_schema(), indent=2, ensure_ascii=False)
            + "\n"
        )


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)


def _wire_extensions(model: _WireModel) -> dict[str, JsonValue]:
    extras = model.model_extra
    return {} if extras is None else _JSON_OBJECT_ADAPTER.validate_python(extras)


class AlignmentDiagnosticInput(_WireModel):
    code: StrictStr = Field(min_length=1)
    message: StrictStr = Field(min_length=1)
    project_id: StrictStr | None = None
    page_name: StrictStr | None = None

    def to_domain(self) -> AlignmentDiagnostic:
        return AlignmentDiagnostic(
            code=self.code,
            message=self.message,
            project_id=self.project_id,
            page_name=self.page_name,
            extensions=_wire_extensions(self),
        )


class WireSourceLineInput(_WireModel):
    ordinal: StrictInt = Field(ge=0)
    byte_start: StrictInt = Field(ge=0)
    byte_end: StrictInt = Field(ge=0)
    content_byte_start: StrictInt = Field(ge=0)
    content_byte_end: StrictInt = Field(ge=0)
    separator_byte_start: StrictInt = Field(ge=0)
    separator_byte_end: StrictInt = Field(ge=0)
    original_text: StrictStr
    separator_text: StrictStr
    visible_text: StrictStr
    matching_eligible: StrictBool
    style_fitting_eligible: StrictBool
    operation_ordinals: tuple[StrictInt, ...] = ()
    formatting_span_ids: tuple[StrictStr, ...] = ()
    continued_block_ids: tuple[StrictStr, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _normalize_sequences(cls, payload: object) -> object:
        return _normalize_tuple_fields(
            payload, ("operation_ordinals", "formatting_span_ids", "continued_block_ids")
        )

    def to_domain(self) -> WireSourceLine:
        return WireSourceLine(
            ordinal=self.ordinal,
            byte_start=self.byte_start,
            byte_end=self.byte_end,
            content_byte_start=self.content_byte_start,
            content_byte_end=self.content_byte_end,
            separator_byte_start=self.separator_byte_start,
            separator_byte_end=self.separator_byte_end,
            original_text=self.original_text,
            separator_text=self.separator_text,
            visible_text=self.visible_text,
            matching_eligible=self.matching_eligible,
            style_fitting_eligible=self.style_fitting_eligible,
            operation_ordinals=self.operation_ordinals,
            formatting_span_ids=self.formatting_span_ids,
            continued_block_ids=self.continued_block_ids,
            extensions=_wire_extensions(self),
        )


class WireFormattingSpanInput(_WireModel):
    opening_id: StrictStr = Field(min_length=1)
    block_id: StrictStr = Field(min_length=1)
    kind: Literal["local", "continued"]
    page_name: StrictStr | None = None
    byte_start: StrictInt = Field(ge=0)
    byte_end: StrictInt = Field(ge=0)
    closed: StrictBool

    def to_domain(self) -> WireFormattingSpan:
        return WireFormattingSpan(
            opening_id=self.opening_id,
            block_id=self.block_id,
            kind=self.kind,
            page_name=self.page_name,
            byte_start=self.byte_start,
            byte_end=self.byte_end,
            closed=self.closed,
            extensions=_wire_extensions(self),
        )


class WireStyleRunInput(_WireModel):
    kind: Literal[
        "i",
        "b",
        "sc",
        "f",
        "g",
        "tb",
    ]
    normalized_start: StrictInt = Field(ge=0)
    normalized_end: StrictInt = Field(ge=0)
    byte_start: StrictInt = Field(ge=0)
    byte_end: StrictInt = Field(ge=0)

    def to_domain(self) -> WireStyleRun:
        return WireStyleRun(
            kind=self.kind,
            normalized_start=self.normalized_start,
            normalized_end=self.normalized_end,
            byte_start=self.byte_start,
            byte_end=self.byte_end,
            extensions=_wire_extensions(self),
        )


class WireLineCandidateInput(_WireModel):
    ordinal: StrictInt = Field(ge=0)
    band_ordinal: StrictInt | None = Field(default=None, ge=0)
    component_ordinal: StrictInt | None = Field(default=None, ge=0)
    box: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    component_count: StrictInt = Field(ge=0)
    foreground_pixels: StrictInt = Field(ge=0)
    width: StrictInt = Field(ge=0)
    height: StrictInt = Field(ge=0)
    fill_ratio: StrictFloat
    horizontal_ink_profile: tuple[StrictFloat, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _normalize_sequences(cls, payload: object) -> object:
        return _normalize_tuple_fields(payload, ("box", "horizontal_ink_profile"))

    def to_domain(self) -> WireLineCandidate:
        return WireLineCandidate(
            ordinal=self.ordinal,
            band_ordinal=self.band_ordinal,
            component_ordinal=self.component_ordinal,
            box=self.box,
            component_count=self.component_count,
            foreground_pixels=self.foreground_pixels,
            width=self.width,
            height=self.height,
            fill_ratio=self.fill_ratio,
            horizontal_ink_profile=self.horizontal_ink_profile,
            extensions=_wire_extensions(self),
        )


class AlignmentOperationInput(_WireModel):
    kind: Literal["match", "skip_image", "skip_text"]
    source_ordinal: StrictInt | None = Field(default=None, ge=0)
    candidate_ordinal: StrictInt | None = Field(default=None, ge=0)
    cost: StrictFloat = Field(ge=0)
    width_cost: StrictFloat | None = Field(default=None, ge=0)
    indentation_cost: StrictFloat | None = Field(default=None, ge=0)
    gaps_cost: StrictFloat | None = Field(default=None, ge=0)
    style_cost: StrictFloat | None = Field(default=None, ge=0)

    def to_domain(self) -> AlignmentOperation:
        return AlignmentOperation(
            kind=self.kind,
            source_ordinal=self.source_ordinal,
            candidate_ordinal=self.candidate_ordinal,
            cost=self.cost,
            width_cost=self.width_cost,
            indentation_cost=self.indentation_cost,
            gaps_cost=self.gaps_cost,
            style_cost=self.style_cost,
            extensions=_wire_extensions(self),
        )


class CoordinateFrameInput(_WireModel):
    name: Literal["source"] = "source"
    width: StrictInt = Field(ge=0)
    height: StrictInt = Field(ge=0)

    def to_domain(self) -> CoordinateFrame:
        return CoordinateFrame(
            name=self.name, width=self.width, height=self.height, extensions=_wire_extensions(self)
        )


class PageAlignmentInput(_WireModel):
    model_config = ConfigDict(
        json_schema_extra={
            "oneOf": [
                {
                    "title": "Available F2 source",
                    "properties": {"f2_sha256": {"type": "string"}},
                    "required": ["f2_sha256"],
                },
                {
                    "title": "Unavailable F2 source",
                    "properties": {
                        "f2_sha256": {"type": "null"},
                        "state": {"const": "excluded"},
                        "exclusions": {
                            "type": "array",
                            "contains": {"const": "f2_missing"},
                        },
                    },
                    "required": ["f2_sha256", "state", "exclusions"],
                },
            ]
        }
    )
    page_name: StrictStr = Field(min_length=1)
    f2_sha256: StrictStr | None
    scan_sha256: StrictStr | None = None
    source_frame: CoordinateFrameInput | None = None
    source_lines: tuple[WireSourceLineInput, ...]
    formatting_spans: tuple[WireFormattingSpanInput, ...]
    style_runs: tuple[WireStyleRunInput, ...] = ()
    candidates: tuple[WireLineCandidateInput, ...]
    operations: tuple[AlignmentOperationInput, ...]
    total_cost: StrictFloat = Field(ge=0)
    second_best_cost: StrictFloat | None = Field(default=None, ge=0)
    normalized_cost: StrictFloat = Field(ge=0)
    matched_count: StrictInt = Field(ge=0)
    matched_ratio: StrictFloat = Field(ge=0, le=1)
    uniqueness_margin: StrictFloat = Field(ge=0)
    mean_width_residual: StrictFloat | None = Field(default=None, ge=0)
    mean_indentation_residual: StrictFloat | None = Field(default=None, ge=0)
    state: Literal["accepted", "proposed", "excluded"]
    accepted: StrictBool
    exclusions: tuple[StrictStr, ...]
    exclusion_evidence: tuple[dict[str, JsonValue], ...] = ()
    diagnostics: tuple[AlignmentDiagnosticInput, ...] = ()
    confidence: None
    confidence_kind: Literal["uncalibrated"]

    @model_validator(mode="before")
    @classmethod
    def _normalize_sequences(cls, payload: object) -> object:
        return _normalize_tuple_fields(
            payload,
            (
                "source_lines",
                "formatting_spans",
                "style_runs",
                "candidates",
                "operations",
                "exclusions",
                "exclusion_evidence",
                "diagnostics",
            ),
        )

    @model_validator(mode="after")
    def _validate_unavailable_f2_hash(self) -> Self:
        if self.f2_sha256 is None and (
            self.state != "excluded" or "f2_missing" not in self.exclusions
        ):
            raise ValueError("f2_sha256 can be null only for an excluded page with f2_missing.")
        return self

    def to_domain(self) -> PageAlignment:
        return PageAlignment(
            page_name=self.page_name,
            f2_sha256=self.f2_sha256,
            scan_sha256=self.scan_sha256,
            source_frame=None if self.source_frame is None else self.source_frame.to_domain(),
            source_lines=tuple(line.to_domain() for line in self.source_lines),
            formatting_spans=tuple(span.to_domain() for span in self.formatting_spans),
            style_runs=tuple(style_run.to_domain() for style_run in self.style_runs),
            candidates=tuple(candidate.to_domain() for candidate in self.candidates),
            operations=tuple(operation.to_domain() for operation in self.operations),
            total_cost=self.total_cost,
            second_best_cost=self.second_best_cost,
            normalized_cost=self.normalized_cost,
            matched_count=self.matched_count,
            matched_ratio=self.matched_ratio,
            uniqueness_margin=self.uniqueness_margin,
            mean_width_residual=self.mean_width_residual,
            mean_indentation_residual=self.mean_indentation_residual,
            state=self.state,
            accepted=self.accepted,
            exclusions=self.exclusions,
            exclusion_evidence=self.exclusion_evidence,
            diagnostics=tuple(diagnostic.to_domain() for diagnostic in self.diagnostics),
            confidence=self.confidence,
            confidence_kind=self.confidence_kind,
            extensions=_wire_extensions(self),
        )


class ProjectAlignmentInput(_WireModel):
    project_id: StrictStr = Field(min_length=1)
    title: StrictStr | None = None
    author: StrictStr | None = None
    genre: StrictStr | None = None
    pages: tuple[PageAlignmentInput, ...]
    exclusions: tuple[StrictStr, ...] = ()
    diagnostics: tuple[AlignmentDiagnosticInput, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _normalize_sequences(cls, payload: object) -> object:
        return _normalize_tuple_fields(payload, ("pages", "exclusions", "diagnostics"))

    def to_domain(self) -> ProjectAlignment:
        return ProjectAlignment(
            project_id=self.project_id,
            title=self.title,
            author=self.author,
            genre=self.genre,
            pages=tuple(page.to_domain() for page in self.pages),
            exclusions=self.exclusions,
            diagnostics=tuple(diagnostic.to_domain() for diagnostic in self.diagnostics),
            extensions=_wire_extensions(self),
        )


class AlignmentReportInput(_WireModel):
    schema_version: Literal[1]
    algorithm_version: Literal["pgdp-alignment/v3"]
    tool_version: StrictStr = Field(min_length=1)
    profile_label: StrictStr = Field(min_length=1, pattern=r"^[^/\\]+$")
    profile_sha256: StrictStr
    methods: dict[str, JsonValue]
    thresholds: dict[str, JsonValue]
    projects: tuple[ProjectAlignmentInput, ...] = ()
    diagnostics: tuple[AlignmentDiagnosticInput, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _normalize_sequences(cls, payload: object) -> object:
        normalized = _normalize_tuple_fields(payload, ("projects", "diagnostics"))
        if not isinstance(normalized, dict):
            return normalized
        schema_version = normalized.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise TypeError("schema_version must be the exact integer 1.")
        return normalized

    @model_validator(mode="after")
    def _validate_profile_label(self) -> AlignmentReportInput:
        if (
            self.profile_label in {".", ".."}
            or "/" in self.profile_label
            or "\\" in self.profile_label
            or Path(self.profile_label).name != self.profile_label
        ):
            raise ValueError("profile_label must be a nonempty file name without path components.")
        return self

    def to_domain(self) -> AlignmentReport:
        return AlignmentReport(
            schema_version=self.schema_version,
            algorithm_version=self.algorithm_version,
            tool_version=self.tool_version,
            profile_label=self.profile_label,
            profile_sha256=self.profile_sha256,
            methods=self.methods,
            thresholds=self.thresholds,
            projects=tuple(project.to_domain() for project in self.projects),
            diagnostics=tuple(diagnostic.to_domain() for diagnostic in self.diagnostics),
            extensions=_wire_extensions(self),
        )
