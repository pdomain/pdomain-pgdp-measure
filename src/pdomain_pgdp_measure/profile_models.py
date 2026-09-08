"""Versioned, evidence-preserving PGDP source-geometry profile records."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from pdomain_pgdp_measure.paths import UnsafePathError, require_canonical_relative_reference

if TYPE_CHECKING:
    from collections.abc import Mapping


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "pgdp-profile/v2"
SOURCE_FRAME = "source"
UNCALIBRATED_CONFIDENCE_KIND = "uncalibrated"
_UNAVAILABLE_IMAGE_DIAGNOSTIC_CODES = frozenset({"image_missing", "image_unreadable"})
_DECODE_FAILURE_DIAGNOSTIC_CODE = "image_decode_failed"
_BLANK_PAGE_DIAGNOSTIC_CODE = "blank_page"

TruthClass = Literal["observed", "derived", "pooled"]
ConfidenceKind = Literal["uncalibrated"]
Bounds = tuple[int, int, int, int] | list[int]
Margins = tuple[int | None, int | None, int | None, int | None] | list[int | None]

_JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, JsonValue])


def _require_nonnegative_integer(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


def _require_finite(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _require_sha256(value: object) -> None:
    hexadecimal_characters = "0123456789abcdefABCDEF"
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in hexadecimal_characters for character in value)
    ):
        raise ValueError("SHA-256 must contain exactly 64 hexadecimal characters.")


def _require_corpus_relative_source_path(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("Source path must be a string.")
    try:
        _ = require_canonical_relative_reference(value=value, label="source path")
    except UnsafePathError as error:
        raise ValueError(
            f"Source path must be a canonical corpus-relative POSIX path: {value!r}."
        ) from error
    return value


def _deep_freeze_json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        _require_finite(value, name="JSON value")
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
        _require_finite(value, name="JSON value")
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


def _normalized_extensions(values: Mapping[str, object]) -> Mapping[str, object]:
    validated = _JSON_OBJECT_ADAPTER.validate_python(dict(values))
    return MappingProxyType(
        {key: _deep_freeze_json_value(value) for key, value in sorted(validated.items())}
    )


def _extensions_dict(values: Mapping[str, object]) -> dict[str, JsonValue]:
    return {key: _thaw_json_value(value) for key, value in sorted(values.items())}


def _merged_payload(
    payload: dict[str, JsonValue], extensions: Mapping[str, object]
) -> dict[str, JsonValue]:
    overlap = set(payload) & set(extensions)
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise ValueError(f"extensions cannot replace defined fields: {joined}.")
    payload.update(_extensions_dict(extensions))
    return payload


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    """The dimensions of the source-image coordinate frame."""

    width: int
    height: int
    name: Literal["source"] = SOURCE_FRAME
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name != SOURCE_FRAME:
            raise ValueError(f"Coordinate frame must be {SOURCE_FRAME!r} in profile version 1.")
        _require_nonnegative_integer(self.width, name="width")
        _require_nonnegative_integer(self.height, name="height")
        object.__setattr__(self, "extensions", _normalized_extensions(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merged_payload(
            {"name": self.name, "width": self.width, "height": self.height}, self.extensions
        )


@dataclass(frozen=True, slots=True)
class InkBand:
    """A half-open horizontal row-projection ink interval."""

    y_start: int
    y_end: int
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.y_start, name="y_start")
        _require_nonnegative_integer(self.y_end, name="y_end")
        if self.y_end < self.y_start:
            raise ValueError("Ink band must have non-inverted half-open bounds.")
        if self.y_end - self.y_start < 2:
            raise ValueError("Ink band must be at least two rows high.")
        object.__setattr__(self, "extensions", _normalized_extensions(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merged_payload({"y_start": self.y_start, "y_end": self.y_end}, self.extensions)


@dataclass(frozen=True, slots=True)
class Estimate:
    """One measured, derived, or pooled value with page-level evidence."""

    name: str
    value: float | None
    unit: str
    truth_class: TruthClass
    method: str
    sample_count: int
    evidence_pages: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    frame: Literal["source"] = SOURCE_FRAME
    confidence: None = None
    confidence_kind: ConfidenceKind = UNCALIBRATED_CONFIDENCE_KIND
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Estimate name must not be empty.")
        if not self.unit:
            raise ValueError("Estimate unit must not be empty.")
        if not self.method:
            raise ValueError("Estimate method must not be empty.")
        if self.truth_class not in {"observed", "derived", "pooled"}:
            raise ValueError("Estimate truth_class is unsupported.")
        if self.frame != SOURCE_FRAME:
            raise ValueError(f"Estimate frame must be {SOURCE_FRAME!r} in profile version 1.")
        if self.confidence is not None:
            raise ValueError("Profile version 1 confidence must be null.")
        if self.confidence_kind != UNCALIBRATED_CONFIDENCE_KIND:
            raise ValueError("Profile version 1 confidence_kind must be 'uncalibrated'.")
        _require_nonnegative_integer(self.sample_count, name="sample_count")
        if self.value is not None:
            _require_finite(self.value, name="Estimate value")
        object.__setattr__(self, "evidence_pages", tuple(self.evidence_pages))
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        if len(set(self.evidence_pages)) != len(self.evidence_pages):
            raise ValueError("Estimate evidence_pages contains duplicate pages.")
        if self.sample_count != len(self.evidence_pages):
            raise ValueError("Estimate sample_count must match evidence_pages.")
        if self.value is not None and self.sample_count == 0:
            raise ValueError("A numeric estimate requires page evidence.")
        object.__setattr__(self, "extensions", _normalized_extensions(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merged_payload(
            {
                "name": self.name,
                "value": self.value,
                "unit": self.unit,
                "truth_class": self.truth_class,
                "frame": self.frame,
                "method": self.method,
                "sample_count": self.sample_count,
                "evidence_pages": list(self.evidence_pages),
                "exclusions": list(self.exclusions),
                "confidence": self.confidence,
                "confidence_kind": self.confidence_kind,
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class ProfileDiagnostic:
    """A non-fatal issue collected while creating a geometry profile."""

    code: str
    message: str
    project_id: str | None = None
    page_name: str | None = None
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extensions", _normalized_extensions(self.extensions))

    def to_dict(self) -> dict[str, JsonValue]:
        return _merged_payload(
            {
                "code": self.code,
                "message": self.message,
                "project_id": self.project_id,
                "page_name": self.page_name,
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class PageMeasurement:
    """The source-frame geometry observed and derived for one scan."""

    page_name: str
    source_path: str
    sha256: str | None
    source_frame: CoordinateFrame | None
    image_mode: str | None
    grayscale_threshold: int | None
    foreground_pixels: int | None
    foreground_bounds: Bounds | None
    margins: Margins | None
    ink_bands: tuple[InkBand, ...] | None = ()
    derived_estimates: tuple[Estimate, ...] = ()
    exclusions: tuple[str, ...] = ()
    diagnostics: tuple[ProfileDiagnostic, ...] = ()
    page_class: str = "unknown"
    template_residual_px: int | None = None
    furniture_band_ordinals: tuple[int, ...] = ()
    extensions: Mapping[str, object] = field(default_factory=dict)
    image_extensions: Mapping[str, object] = field(default_factory=dict)
    observation_extensions: Mapping[str, object] = field(default_factory=dict)
    margin_extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.page_name:
            raise ValueError("Page name must not be empty.")
        _ = _require_corpus_relative_source_path(self.source_path)
        self._validate_image_metadata()
        if self.grayscale_threshold is not None:
            _require_nonnegative_integer(self.grayscale_threshold, name="grayscale_threshold")
            if self.grayscale_threshold > 255:
                raise ValueError("grayscale_threshold must be in the range 0 through 255.")
        self._normalize_geometry()
        if self.foreground_pixels is not None:
            _require_nonnegative_integer(self.foreground_pixels, name="foreground_pixels")
        self._validate_bounds()
        self._validate_margins()
        object.__setattr__(self, "derived_estimates", tuple(self.derived_estimates))
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "extensions", _normalized_extensions(self.extensions))
        object.__setattr__(self, "image_extensions", _normalized_extensions(self.image_extensions))
        object.__setattr__(
            self, "observation_extensions", _normalized_extensions(self.observation_extensions)
        )
        object.__setattr__(
            self, "margin_extensions", _normalized_extensions(self.margin_extensions)
        )
        self._validate_foreground_coherence()
        if any(estimate.truth_class != "derived" for estimate in self.derived_estimates):
            raise ValueError("Page derived_estimates must use the derived truth_class.")

    def _normalize_geometry(self) -> None:
        if self.foreground_bounds is not None:
            normalized_bounds = tuple(self.foreground_bounds)
            if len(normalized_bounds) != 4:
                raise ValueError("Foreground bounds must contain four coordinates.")
            object.__setattr__(self, "foreground_bounds", normalized_bounds)
        if self.margins is not None:
            normalized_margins = tuple(self.margins)
            if len(normalized_margins) != 4:
                raise ValueError("Margins must contain left, top, right, and bottom values.")
            object.__setattr__(self, "margins", normalized_margins)
        if self.ink_bands is not None:
            object.__setattr__(self, "ink_bands", tuple(self.ink_bands))

    def _validate_image_metadata(self) -> None:
        metadata = (self.sha256, self.source_frame, self.image_mode)
        has_decoded_metadata = all(value is not None for value in metadata)
        has_byte_hash_only = (
            self.sha256 is not None and self.source_frame is None and self.image_mode is None
        )
        has_unavailable_metadata = all(value is None for value in metadata)
        if not (has_decoded_metadata or has_byte_hash_only or has_unavailable_metadata):
            raise ValueError(
                "Image metadata must be decoded, original-byte hash only, or unavailable."
            )
        if has_decoded_metadata or has_byte_hash_only:
            if self.sha256 is None:
                raise ValueError("Measured image metadata must be complete.")
            _require_sha256(self.sha256)
        if has_decoded_metadata:
            if self.image_mode is None or not self.image_mode:
                raise ValueError("Image mode must not be empty.")
        elif self.image_extensions:
            raise ValueError("Unavailable image metadata cannot have image extensions.")

    def _validate_foreground_coherence(self) -> None:
        if self.foreground_pixels is None:
            if (
                self.source_frame is not None
                or self.image_mode is not None
                or self.grayscale_threshold is not None
            ):
                raise ValueError(
                    "Unavailable foreground measurements must not have decoded image metadata."
                )
            if (
                self.foreground_bounds is not None
                or self.margins is not None
                or self.ink_bands is not None
            ):
                raise ValueError(
                    "Unavailable foreground measurements must have unavailable geometry."
                )
            if self.derived_estimates:
                raise ValueError(
                    "Unavailable foreground measurements must not have derived estimates."
                )
            if self.sha256 is None:
                self._require_diagnostic(
                    _UNAVAILABLE_IMAGE_DIAGNOSTIC_CODES,
                    "Unavailable image measurements require an image_missing or image_unreadable "
                    "diagnostic.",
                )
            else:
                self._require_diagnostic(
                    frozenset({_DECODE_FAILURE_DIAGNOSTIC_CODE}),
                    "Readable undecodable images require an image_decode_failed diagnostic.",
                )
            return
        if self.sha256 is None or self.source_frame is None or self.image_mode is None:
            raise ValueError("Measured foreground measurements require image metadata.")
        if self.grayscale_threshold is None:
            raise ValueError("Measured foreground measurements require a grayscale threshold.")
        if self.foreground_pixels == 0:
            if self.foreground_bounds is not None or self.margins is not None:
                raise ValueError("A blank page must not have foreground bounds or margins.")
            if self.ink_bands != ():
                raise ValueError("A blank page requires an empty ink bands tuple.")
            if self.derived_estimates:
                raise ValueError("A blank page must not have derived_estimates.")
            self._require_diagnostic(
                frozenset({_BLANK_PAGE_DIAGNOSTIC_CODE}),
                "A blank page requires a blank_page diagnostic.",
            )
            return
        if self.foreground_bounds is None or self.margins is None or self.ink_bands is None:
            raise ValueError("Measured foreground pixels require bounds, margins, and ink bands.")
        x_start, y_start, x_end, y_end = self.foreground_bounds
        if x_end == x_start or y_end == y_start:
            raise ValueError("Measured foreground pixels require nonempty bounds.")
        bounds_area = (x_end - x_start) * (y_end - y_start)
        if self.foreground_pixels > bounds_area:
            raise ValueError("Measured foreground pixels must not exceed the bounds area.")
        if any(margin is None for margin in self.margins):
            raise ValueError("Measured foreground pixels require four numeric margins.")
        source_frame = self.source_frame
        expected_margins = (
            x_start,
            y_start,
            source_frame.width - x_end,
            source_frame.height - y_end,
        )
        if self.margins != expected_margins:
            raise ValueError("Measured foreground margins must match the bounds and source frame.")
        if any(
            band.y_start > source_frame.height or band.y_end > source_frame.height
            for band in self.ink_bands
        ):
            raise ValueError("Ink bands must stay inside the source frame height.")

    def _require_diagnostic(self, required_codes: frozenset[str], message: str) -> None:
        if not any(diagnostic.code in required_codes for diagnostic in self.diagnostics):
            raise ValueError(message)

    def _validate_bounds(self) -> None:
        if self.foreground_bounds is None:
            return
        if self.source_frame is None:
            raise ValueError("Unavailable foreground measurements must have unavailable geometry.")
        x_start, y_start, x_end, y_end = self.foreground_bounds
        for name, value in (
            ("foreground_bounds.x_start", x_start),
            ("foreground_bounds.y_start", y_start),
            ("foreground_bounds.x_end", x_end),
            ("foreground_bounds.y_end", y_end),
        ):
            _require_nonnegative_integer(value, name=name)
        if x_end < x_start or y_end < y_start:
            raise ValueError("Foreground bounds must be non-inverted half-open bounds.")
        if x_end > self.source_frame.width or y_end > self.source_frame.height:
            raise ValueError("Foreground bounds must stay inside the source frame.")

    def _validate_margins(self) -> None:
        if self.margins is None:
            return
        if len(self.margins) != 4:
            raise ValueError("Margins must contain left, top, right, and bottom values.")
        for margin in self.margins:
            if margin is not None:
                _require_nonnegative_integer(margin, name="margin")

    def to_dict(self) -> dict[str, JsonValue]:
        foreground_bounds: list[JsonValue] | None = (
            None if self.foreground_bounds is None else list(self.foreground_bounds)
        )
        margins: dict[str, JsonValue] | None
        if self.margins is None:
            margins = None
        else:
            margins = _merged_payload(
                {
                    "left": self.margins[0],
                    "top": self.margins[1],
                    "right": self.margins[2],
                    "bottom": self.margins[3],
                },
                self.margin_extensions,
            )
        ink_bands: list[JsonValue] | None = (
            None
            if self.ink_bands is None
            else list[JsonValue](band.to_dict() for band in self.ink_bands)
        )
        observations = _merged_payload(
            {
                "grayscale_threshold": self.grayscale_threshold,
                "foreground_pixels": self.foreground_pixels,
                "foreground_bounds": foreground_bounds,
                "margins": margins,
                "ink_bands": ink_bands,
            },
            self.observation_extensions,
        )
        source_frame: dict[str, JsonValue] | None = (
            None if self.source_frame is None else self.source_frame.to_dict()
        )
        image: dict[str, JsonValue] | None = (
            None
            if self.image_mode is None
            else _merged_payload({"mode": self.image_mode}, self.image_extensions)
        )
        return _merged_payload(
            {
                "page_name": self.page_name,
                "source_path": self.source_path,
                "sha256": self.sha256,
                "source_frame": source_frame,
                "image": image,
                "observations": observations,
                "derived_estimates": [estimate.to_dict() for estimate in self.derived_estimates],
                "exclusions": list(self.exclusions),
                "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
                "page_class": self.page_class,
                "template_residual_px": self.template_residual_px,
                "furniture_band_ordinals": list(self.furniture_band_ordinals),
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class PageTemplateRecord:
    """One measured page shape a book repeats."""

    page_class: str
    first_band_top_px: int
    text_left_px: int
    text_right_px: int
    band_count: int
    page_count: int
    page_share: float
    first_band_spread_px: int = 0

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "page_class": self.page_class,
            "first_band_top_px": self.first_band_top_px,
            "first_band_spread_px": self.first_band_spread_px,
            "text_left_px": self.text_left_px,
            "text_right_px": self.text_right_px,
            "band_count": self.band_count,
            "page_count": self.page_count,
            "page_share": self.page_share,
        }


@dataclass(frozen=True, slots=True)
class ProjectProfile:
    """Ordered page observations and pooled estimates for one PGDP project."""

    project_id: str
    title: str | None
    author: str | None
    genre: str | None
    pages: tuple[PageMeasurement, ...] = ()
    pooled_estimates: tuple[Estimate, ...] = ()
    exclusions: tuple[str, ...] = ()
    diagnostics: tuple[ProfileDiagnostic, ...] = ()
    page_templates: tuple[PageTemplateRecord, ...] = ()
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.project_id:
            raise ValueError("Project ID must not be empty.")
        object.__setattr__(self, "pages", tuple(self.pages))
        object.__setattr__(self, "pooled_estimates", tuple(self.pooled_estimates))
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "page_templates", tuple(self.page_templates))
        object.__setattr__(self, "extensions", _normalized_extensions(self.extensions))
        page_names = tuple(page.page_name for page in self.pages)
        if len(set(page_names)) != len(page_names):
            raise ValueError("Project profile contains duplicate page names.")
        if any(estimate.truth_class != "pooled" for estimate in self.pooled_estimates):
            raise ValueError("Project pooled_estimates must use the pooled truth_class.")

    def to_dict(self) -> dict[str, JsonValue]:
        return _merged_payload(
            {
                "project_id": self.project_id,
                "title": self.title,
                "author": self.author,
                "genre": self.genre,
                "pages": [page.to_dict() for page in self.pages],
                "pooled_estimates": [estimate.to_dict() for estimate in self.pooled_estimates],
                "exclusions": list(self.exclusions),
                "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
                "page_templates": [item.to_dict() for item in self.page_templates],
            },
            self.extensions,
        )


@dataclass(frozen=True, slots=True)
class ProfileReport:
    """Stable JSON-safe output from a local PGDP geometry-profile run."""

    source_ranking: Mapping[str, object]
    methods: Mapping[str, object]
    projects: tuple[ProjectProfile, ...] = ()
    diagnostics: tuple[ProfileDiagnostic, ...] = ()
    schema_version: int = SCHEMA_VERSION
    algorithm_version: str = ALGORITHM_VERSION
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError(f"Unsupported schema_version {self.schema_version!r}.")
        if self.algorithm_version != ALGORITHM_VERSION:
            raise ValueError(f"Unsupported algorithm_version {self.algorithm_version!r}.")
        object.__setattr__(self, "source_ranking", _normalized_extensions(self.source_ranking))
        object.__setattr__(self, "methods", _normalized_extensions(self.methods))
        object.__setattr__(self, "projects", tuple(self.projects))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "extensions", _normalized_extensions(self.extensions))
        project_ids = tuple(project.project_id for project in self.projects)
        if len(set(project_ids)) != len(project_ids):
            raise ValueError("Profile report contains duplicate project IDs.")

    def to_dict(self) -> dict[str, JsonValue]:
        return _merged_payload(
            {
                "schema_version": self.schema_version,
                "algorithm_version": self.algorithm_version,
                "source_ranking": _extensions_dict(self.source_ranking),
                "methods": _extensions_dict(self.methods),
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
    def from_dict(cls, payload: object) -> ProfileReport:
        try:
            return ProfileReportWire.model_validate(payload).to_domain()
        except ValidationError as error:
            raise ValueError(f"Invalid PGDP geometry profile: {error}") from error

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> ProfileReport:
        try:
            return ProfileReportWire.model_validate_json(payload).to_domain()
        except ValidationError as error:
            raise ValueError(f"Invalid PGDP geometry profile: {error}") from error


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


def _wire_extensions(model: _WireModel) -> dict[str, JsonValue]:
    extras = model.model_extra
    if extras is None:
        return {}
    return _JSON_OBJECT_ADAPTER.validate_python(extras)


class CoordinateFrameWire(_WireModel):
    name: Literal["source"] = SOURCE_FRAME
    width: StrictInt = Field(ge=0)
    height: StrictInt = Field(ge=0)

    def to_domain(self) -> CoordinateFrame:
        return CoordinateFrame(
            name=self.name,
            width=self.width,
            height=self.height,
            extensions=_wire_extensions(self),
        )


class InkBandWire(_WireModel):
    y_start: StrictInt = Field(ge=0)
    y_end: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _validate_minimum_height(self) -> Self:
        if self.y_end - self.y_start < 2:
            raise ValueError("Ink band must be at least two rows high.")
        return self

    def to_domain(self) -> InkBand:
        return InkBand(y_start=self.y_start, y_end=self.y_end, extensions=_wire_extensions(self))


class EstimateWire(_WireModel):
    name: str = Field(min_length=1)
    value: float | None = None
    unit: str = Field(min_length=1)
    truth_class: Literal["observed", "derived", "pooled"]
    frame: Literal["source"] = SOURCE_FRAME
    method: str = Field(min_length=1)
    sample_count: StrictInt = Field(ge=0)
    evidence_pages: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    confidence: None
    confidence_kind: Literal["uncalibrated"]

    @field_validator("value", mode="before")
    @classmethod
    def _validate_numeric_value(cls, value: object) -> object:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("Estimate value must be a JSON number or null.")
        return value

    def to_domain(self) -> Estimate:
        return Estimate(
            name=self.name,
            value=self.value,
            unit=self.unit,
            truth_class=self.truth_class,
            frame=self.frame,
            method=self.method,
            sample_count=self.sample_count,
            evidence_pages=self.evidence_pages,
            exclusions=self.exclusions,
            confidence=self.confidence,
            confidence_kind=self.confidence_kind,
            extensions=_wire_extensions(self),
        )


class ProfileDiagnosticWire(_WireModel):
    code: str
    message: str
    project_id: str | None = None
    page_name: str | None = None

    def to_domain(self) -> ProfileDiagnostic:
        return ProfileDiagnostic(
            code=self.code,
            message=self.message,
            project_id=self.project_id,
            page_name=self.page_name,
            extensions=_wire_extensions(self),
        )


class ImageMetadataWire(_WireModel):
    mode: str = Field(min_length=1)


class MarginsWire(_WireModel):
    left: StrictInt | None = Field(default=None, ge=0)
    top: StrictInt | None = Field(default=None, ge=0)
    right: StrictInt | None = Field(default=None, ge=0)
    bottom: StrictInt | None = Field(default=None, ge=0)


class ObservationsWire(_WireModel):
    grayscale_threshold: StrictInt | None = Field(..., ge=0, le=255)
    foreground_pixels: StrictInt | None = Field(ge=0)
    foreground_bounds: tuple[StrictInt, StrictInt, StrictInt, StrictInt] | None
    margins: MarginsWire | None
    ink_bands: tuple[InkBandWire, ...] | None

    @model_validator(mode="after")
    def _validate_blank_ink_bands(self) -> Self:
        if self.foreground_pixels == 0 and self.ink_bands != ():
            raise ValueError("A blank page requires an empty ink bands tuple.")
        return self


class PageMeasurementWire(_WireModel):
    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        json_schema_extra={
            "oneOf": [
                {
                    "title": "Unavailable image",
                    "properties": {
                        "sha256": {"type": "null"},
                        "source_frame": {"type": "null"},
                        "image": {"type": "null"},
                        "derived_estimates": {"type": "array", "maxItems": 0},
                        "observations": {
                            "properties": {
                                "grayscale_threshold": {"type": "null"},
                                "foreground_pixels": {"type": "null"},
                                "foreground_bounds": {"type": "null"},
                                "margins": {"type": "null"},
                                "ink_bands": {"type": "null"},
                            },
                            "required": [
                                "grayscale_threshold",
                                "foreground_pixels",
                                "foreground_bounds",
                                "margins",
                                "ink_bands",
                            ],
                        },
                        "diagnostics": {
                            "type": "array",
                            "contains": {
                                "properties": {
                                    "code": {
                                        "enum": ["image_missing", "image_unreadable"],
                                    },
                                },
                                "required": ["code"],
                            },
                        },
                    },
                    "required": ["diagnostics"],
                },
                {
                    "title": "Readable undecodable image",
                    "properties": {
                        "sha256": {"type": "string"},
                        "source_frame": {"type": "null"},
                        "image": {"type": "null"},
                        "derived_estimates": {"type": "array", "maxItems": 0},
                        "observations": {
                            "properties": {
                                "grayscale_threshold": {"type": "null"},
                                "foreground_pixels": {"type": "null"},
                                "foreground_bounds": {"type": "null"},
                                "margins": {"type": "null"},
                                "ink_bands": {"type": "null"},
                            },
                            "required": [
                                "grayscale_threshold",
                                "foreground_pixels",
                                "foreground_bounds",
                                "margins",
                                "ink_bands",
                            ],
                        },
                        "diagnostics": {
                            "type": "array",
                            "contains": {
                                "properties": {
                                    "code": {"const": "image_decode_failed"},
                                },
                                "required": ["code"],
                            },
                        },
                    },
                    "required": ["diagnostics"],
                },
                {
                    "title": "Blank measured image",
                    "properties": {
                        "sha256": {"type": "string"},
                        "source_frame": {"type": "object"},
                        "image": {"type": "object"},
                        "derived_estimates": {"type": "array", "maxItems": 0},
                        "observations": {
                            "properties": {
                                "grayscale_threshold": {"type": "integer"},
                                "foreground_pixels": {"const": 0},
                                "foreground_bounds": {"type": "null"},
                                "margins": {"type": "null"},
                                "ink_bands": {"type": "array", "maxItems": 0},
                            },
                            "required": [
                                "grayscale_threshold",
                                "foreground_pixels",
                                "foreground_bounds",
                                "margins",
                                "ink_bands",
                            ],
                        },
                        "diagnostics": {
                            "type": "array",
                            "contains": {
                                "properties": {
                                    "code": {"const": "blank_page"},
                                },
                                "required": ["code"],
                            },
                        },
                    },
                    "required": ["diagnostics"],
                },
                {
                    "title": "Foreground measured image",
                    "properties": {
                        "sha256": {"type": "string"},
                        "source_frame": {"type": "object"},
                        "image": {"type": "object"},
                        "observations": {
                            "properties": {
                                "grayscale_threshold": {"type": "integer"},
                                "foreground_pixels": {"type": "integer", "minimum": 1},
                                "foreground_bounds": {"type": "array"},
                                "margins": {"type": "object"},
                                "ink_bands": {"type": "array"},
                            },
                            "required": [
                                "grayscale_threshold",
                                "foreground_pixels",
                                "foreground_bounds",
                                "margins",
                                "ink_bands",
                            ],
                        },
                    },
                },
            ],
        },
    )
    page_name: str = Field(min_length=1)
    source_path: str = Field(
        min_length=1,
        json_schema_extra={"format": "pgdp-corpus-relative-posix-path"},
    )
    sha256: StrictStr | None = Field(
        ..., min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$"
    )
    source_frame: CoordinateFrameWire | None
    image: ImageMetadataWire | None
    observations: ObservationsWire
    derived_estimates: tuple[EstimateWire, ...] = ()
    exclusions: tuple[str, ...] = ()
    diagnostics: tuple[ProfileDiagnosticWire, ...] = ()
    page_class: str = "unknown"
    template_residual_px: StrictInt | None = None
    furniture_band_ordinals: tuple[StrictInt, ...] = ()

    @field_validator("source_path")
    @classmethod
    def _validate_source_path(cls, source_path: str) -> str:
        return _require_corpus_relative_source_path(source_path)

    @field_validator("derived_estimates")
    @classmethod
    def _validate_derived_estimates(
        cls, estimates: tuple[EstimateWire, ...]
    ) -> tuple[EstimateWire, ...]:
        if any(estimate.truth_class != "derived" for estimate in estimates):
            raise ValueError("Page derived_estimates must use the derived truth_class.")
        return estimates

    @model_validator(mode="after")
    def _validate_page_coherence(self) -> Self:
        metadata = (self.sha256, self.source_frame, self.image)
        has_decoded_metadata = all(value is not None for value in metadata)
        has_byte_hash_only = (
            self.sha256 is not None and self.source_frame is None and self.image is None
        )
        has_unavailable_metadata = all(value is None for value in metadata)
        if not (has_decoded_metadata or has_byte_hash_only or has_unavailable_metadata):
            raise ValueError(
                "Image metadata must be decoded, original-byte hash only, or unavailable."
            )
        observations = self.observations
        if observations.foreground_pixels is None:
            if has_decoded_metadata:
                raise ValueError(
                    "Unavailable foreground measurements must not have decoded image metadata."
                )
            if (
                observations.grayscale_threshold is not None
                or observations.foreground_bounds is not None
                or observations.margins is not None
                or observations.ink_bands is not None
            ):
                raise ValueError(
                    "Unavailable foreground measurements must have unavailable geometry."
                )
            if self.derived_estimates:
                raise ValueError(
                    "Unavailable foreground measurements must not have derived estimates."
                )
            if has_unavailable_metadata:
                self._require_diagnostic(
                    _UNAVAILABLE_IMAGE_DIAGNOSTIC_CODES,
                    "Unavailable image measurements require an image_missing or image_unreadable "
                    "diagnostic.",
                )
            else:
                self._require_diagnostic(
                    frozenset({_DECODE_FAILURE_DIAGNOSTIC_CODE}),
                    "Readable undecodable images require an image_decode_failed diagnostic.",
                )
            return self
        if not has_decoded_metadata:
            raise ValueError("Measured foreground measurements require image metadata.")
        if observations.foreground_pixels == 0:
            if self.derived_estimates:
                raise ValueError("A blank page must not have derived estimates.")
            self._require_diagnostic(
                frozenset({_BLANK_PAGE_DIAGNOSTIC_CODE}),
                "A blank page requires a blank_page diagnostic.",
            )
        self.to_domain()
        return self

    def _require_diagnostic(self, required_codes: frozenset[str], message: str) -> None:
        if not any(diagnostic.code in required_codes for diagnostic in self.diagnostics):
            raise ValueError(message)

    def to_domain(self) -> PageMeasurement:
        observations = self.observations
        margins = observations.margins
        page_margins: tuple[int | None, int | None, int | None, int | None] | None
        margin_extensions: dict[str, JsonValue]
        if margins is None:
            page_margins = None
            margin_extensions = {}
        else:
            page_margins = (margins.left, margins.top, margins.right, margins.bottom)
            margin_extensions = _wire_extensions(margins)
        source_frame = self.source_frame
        image = self.image
        return PageMeasurement(
            page_name=self.page_name,
            source_path=self.source_path,
            sha256=self.sha256,
            source_frame=None if source_frame is None else source_frame.to_domain(),
            image_mode=None if image is None else image.mode,
            grayscale_threshold=observations.grayscale_threshold,
            foreground_pixels=observations.foreground_pixels,
            foreground_bounds=observations.foreground_bounds,
            margins=page_margins,
            ink_bands=(
                None
                if observations.ink_bands is None
                else tuple(band.to_domain() for band in observations.ink_bands)
            ),
            derived_estimates=tuple(estimate.to_domain() for estimate in self.derived_estimates),
            exclusions=self.exclusions,
            diagnostics=tuple(diagnostic.to_domain() for diagnostic in self.diagnostics),
            page_class=self.page_class,
            template_residual_px=self.template_residual_px,
            furniture_band_ordinals=self.furniture_band_ordinals,
            extensions=_wire_extensions(self),
            image_extensions={} if image is None else _wire_extensions(image),
            observation_extensions=_wire_extensions(observations),
            margin_extensions=margin_extensions,
        )


class PageTemplateWire(_WireModel):
    page_class: str = Field(min_length=1)
    first_band_top_px: StrictInt
    first_band_spread_px: StrictInt = 0
    text_left_px: StrictInt
    text_right_px: StrictInt
    band_count: StrictInt = Field(ge=0)
    page_count: StrictInt = Field(ge=0)
    page_share: float = Field(ge=0.0, le=1.0)

    def to_domain(self) -> PageTemplateRecord:
        return PageTemplateRecord(
            page_class=self.page_class,
            first_band_top_px=self.first_band_top_px,
            first_band_spread_px=self.first_band_spread_px,
            text_left_px=self.text_left_px,
            text_right_px=self.text_right_px,
            band_count=self.band_count,
            page_count=self.page_count,
            page_share=self.page_share,
        )


class ProjectProfileWire(_WireModel):
    project_id: str = Field(min_length=1)
    title: str | None = None
    author: str | None = None
    genre: str | None = None
    pages: tuple[PageMeasurementWire, ...] = ()
    pooled_estimates: tuple[EstimateWire, ...] = ()
    exclusions: tuple[str, ...] = ()
    diagnostics: tuple[ProfileDiagnosticWire, ...] = ()
    page_templates: tuple[PageTemplateWire, ...] = ()

    @field_validator("pooled_estimates")
    @classmethod
    def _validate_pooled_estimates(
        cls, estimates: tuple[EstimateWire, ...]
    ) -> tuple[EstimateWire, ...]:
        if any(estimate.truth_class != "pooled" for estimate in estimates):
            raise ValueError("Project pooled_estimates must use the pooled truth_class.")
        return estimates

    def to_domain(self) -> ProjectProfile:
        return ProjectProfile(
            project_id=self.project_id,
            title=self.title,
            author=self.author,
            genre=self.genre,
            pages=tuple(page.to_domain() for page in self.pages),
            pooled_estimates=tuple(estimate.to_domain() for estimate in self.pooled_estimates),
            exclusions=self.exclusions,
            diagnostics=tuple(diagnostic.to_domain() for diagnostic in self.diagnostics),
            page_templates=tuple(item.to_domain() for item in self.page_templates),
            extensions=_wire_extensions(self),
        )


class ProfileReportWire(_WireModel):
    schema_version: Literal[1]
    algorithm_version: Literal["pgdp-profile/v2"]
    source_ranking: dict[str, JsonValue]
    methods: dict[str, JsonValue]
    projects: tuple[ProjectProfileWire, ...] = ()
    diagnostics: tuple[ProfileDiagnosticWire, ...] = ()

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int) or value != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version {value!r}.")
        return value

    @field_validator("algorithm_version")
    @classmethod
    def _validate_algorithm_version(cls, value: Literal["pgdp-profile/v1"]) -> str:
        if value != ALGORITHM_VERSION:
            raise ValueError(f"Unsupported algorithm_version {value!r}.")
        return value

    @classmethod
    def from_domain(cls, report: ProfileReport) -> ProfileReportWire:
        return cls.model_validate(report.to_dict())

    def to_domain(self) -> ProfileReport:
        return ProfileReport(
            schema_version=self.schema_version,
            algorithm_version=self.algorithm_version,
            source_ranking=self.source_ranking,
            methods=self.methods,
            projects=tuple(project.to_domain() for project in self.projects),
            diagnostics=tuple(diagnostic.to_domain() for diagnostic in self.diagnostics),
            extensions=_wire_extensions(self),
        )


def profile_schema_json() -> str:
    """Generate the canonical version-1 JSON Schema artifact."""

    return json.dumps(ProfileReportWire.model_json_schema(), indent=2, sort_keys=False) + "\n"
