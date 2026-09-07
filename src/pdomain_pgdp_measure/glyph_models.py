"""Versioned wire records for the per-book PGDP glyph inventory.

`pgdp-glyphs/v1` is two files. `glyphs.jsonl` carries one row per cut glyph: its character, the
tier that labelled it, the page, line, word and glyph ordinals that place it, its `source`-frame
box, and its recorded quality. `manifest.json` carries the provenance, the method versions, the
per-character coverage split by tier, the quality tallies, and the page table the boxes are read
against. The JSONL is the authority and the atlas is a render of it.

Two label tiers never mix. `transcribed` glyphs come from a word on a line alignment reconciled
against F2, so the character is a human proofer's. `recognized` glyphs come from a word outside
every matched line, where PGDP carries no text at all and the label is a DocTR read with its own
confidence. Every row names its tier, so a consumer can take digits from `recognized` while
training characters only on `transcribed`.

Nothing here names a typeface. `find_latent_discipline_violations` is the same check
`pgdp-typography/v1` runs, and it applies to this manifest unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from pdomain_pgdp_measure.ordering import natural_page_key

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

Bounds = tuple[int, int, int, int]
LabelTier = Literal["transcribed", "recognized"]

GLYPHS_SCHEMA_VERSION: Final = 1
GLYPHS_ALGORITHM_VERSION: Final = "pgdp-glyphs/v1"
GLYPHS_MANIFEST_FILENAME: Final = "manifest.json"
GLYPHS_ROWS_FILENAME: Final = "glyphs.jsonl"
GLYPHS_ATLAS_DIRECTORY: Final = "atlas"
LABEL_TIERS: Final = ("transcribed", "recognized")

_JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, JsonValue])
_HEXADECIMAL: Final = "0123456789abcdefABCDEF"


class GlyphContractError(ValueError):
    """A payload does not satisfy the `pgdp-glyphs/v1` contract."""


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in _HEXADECIMAL for character in value):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal characters.")


def _require_file_label(value: str, *, name: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{name} must be a nonempty file name without path components.")
    if Path(value).name != value:
        raise ValueError(f"{name} must be a nonempty file name without path components.")


def _require_tier(value: str, *, name: str) -> None:
    if value not in LABEL_TIERS:
        raise ValueError(f"{name} must name a known label tier.")


def _counts_dict(counts: Mapping[str, int], *, name: str) -> dict[str, int]:
    ordered: dict[str, int] = {}
    for key in sorted(counts):
        count = counts[key]
        if isinstance(count, bool) or count < 0:
            raise ValueError(f"{name} values must be nonnegative integers.")
        ordered[key] = count
    return ordered


def _mapping_dict(values: Mapping[str, object]) -> dict[str, JsonValue]:
    return _JSON_OBJECT_ADAPTER.validate_python(dict(values))


@dataclass(frozen=True, slots=True)
class GlyphRow:
    """One cut glyph, its label, where it came from, and how it measured.

    `line_ordinal` is the matched candidate's ordinal on the `transcribed` tier and the
    recognizer's own line index on the `recognized` tier, so the two tiers are never comparable
    on it and `label_tier` is part of a row's identity.
    """

    character: str
    label_tier: LabelTier
    page_name: str
    line_ordinal: int
    word_ordinal: int
    glyph_ordinal: int
    box: Bounds
    ink_density: float
    row_extent_px: int
    top_offset_px: int | None = None
    bottom_offset_px: int | None = None
    flags: tuple[str, ...] = ()
    label_confidence: float | None = None
    label_style: str | None = None
    source_line_ordinal: int | None = None

    def __post_init__(self) -> None:
        if len(self.character) != 1:
            raise ValueError("A glyph row carries exactly one character.")
        _require_tier(self.label_tier, name="label_tier")
        if not self.page_name:
            raise ValueError("page_name must not be empty.")
        for name, ordinal in (
            ("line_ordinal", self.line_ordinal),
            ("word_ordinal", self.word_ordinal),
            ("glyph_ordinal", self.glyph_ordinal),
        ):
            if ordinal < 0:
                raise ValueError(f"{name} must be nonnegative.")
        object.__setattr__(self, "box", tuple(self.box))
        x_start, y_start, x_end, y_end = self.box
        if x_end <= x_start or y_end <= y_start:
            raise ValueError("A glyph box must have positive width and height.")
        if not 0.0 <= self.ink_density <= 1.0:
            raise ValueError("ink_density must be a share between zero and one.")
        if self.row_extent_px <= 0:
            raise ValueError("row_extent_px must be positive.")
        if (self.top_offset_px is None) != (self.bottom_offset_px is None):
            raise ValueError("A row's two band offsets are recorded together or not at all.")
        object.__setattr__(self, "flags", tuple(sorted(set(self.flags))))
        if self.label_confidence is not None and not 0.0 <= self.label_confidence <= 1.0:
            raise ValueError("label_confidence must be a share between zero and one.")
        if self.label_tier == "transcribed" and self.label_confidence is not None:
            raise ValueError("A transcribed label is a human proofer's and carries no confidence.")
        if self.label_style is not None and not self.label_style:
            raise ValueError("label_style must be a nonempty name or null.")
        if self.label_tier == "recognized" and self.label_style is not None:
            raise ValueError("A recognized label has no F2 markup to take a style from.")
        if self.source_line_ordinal is not None and self.source_line_ordinal < 0:
            raise ValueError("source_line_ordinal must be nonnegative.")

    @property
    def identity(self) -> tuple[str, str, int, int, int]:
        """What makes this row one glyph, and what two rows may never share."""

        return (
            self.page_name,
            self.label_tier,
            self.line_ordinal,
            self.word_ordinal,
            self.glyph_ordinal,
        )

    @property
    def sort_key(self) -> tuple[tuple[object, ...], str, int, int, int]:
        return (
            natural_page_key(self.page_name),
            self.label_tier,
            self.line_ordinal,
            self.word_ordinal,
            self.glyph_ordinal,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "box": list(self.box),
            "bottom_offset_px": self.bottom_offset_px,
            "character": self.character,
            "flags": list(self.flags),
            "glyph_ordinal": self.glyph_ordinal,
            "ink_density": self.ink_density,
            "label_confidence": self.label_confidence,
            "label_style": self.label_style,
            "label_tier": self.label_tier,
            "line_ordinal": self.line_ordinal,
            "page_name": self.page_name,
            "row_extent_px": self.row_extent_px,
            "source_line_ordinal": self.source_line_ordinal,
            "top_offset_px": self.top_offset_px,
            "word_ordinal": self.word_ordinal,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> GlyphRow:
        try:
            return GlyphRowInput.model_validate_json(payload).to_domain()
        except (TypeError, ValidationError) as error:
            raise GlyphContractError(f"Invalid glyph row: {error}") from error

    @classmethod
    def json_schema(cls) -> str:
        return json.dumps(GlyphRowInput.model_json_schema(), indent=2, ensure_ascii=False) + "\n"


@dataclass(frozen=True, slots=True)
class CharacterCoverage:
    """How many glyphs one character carries on one tier in one style.

    Style is part of the key, not a tally beside it. A small-capital `O` and a lowercase `o` are
    different letterforms under the same character, so pooling them would report coverage the
    inventory does not actually have.
    """

    character: str
    label_tier: LabelTier
    glyph_count: int
    label_style: str | None = None

    def __post_init__(self) -> None:
        if len(self.character) != 1:
            raise ValueError("Coverage is counted one character at a time.")
        _require_tier(self.label_tier, name="label_tier")
        if self.glyph_count <= 0:
            raise ValueError("A coverage row counts at least one glyph.")
        if self.label_style is not None and not self.label_style:
            raise ValueError("label_style must be a nonempty name or null.")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "character": self.character,
            "code_point": f"U+{ord(self.character):04X}",
            "glyph_count": self.glyph_count,
            "label_style": self.label_style,
            "label_tier": self.label_tier,
        }


@dataclass(frozen=True, slots=True)
class GlyphPage:
    """One harvested page, and the scan its boxes are read against."""

    page_name: str
    scan_sha256: str
    source_path: str
    glyph_count: int
    page_state: str = "accepted"
    admitted_line_count: int = 0
    recognizer_admitted_line_count: int = 0
    page_class: str | None = None
    line_x_height_median_px: int | None = None
    line_x_height_spread_px: int | None = None

    def __post_init__(self) -> None:
        if not self.page_name or not self.source_path:
            raise ValueError("A page row names both a page and its scan path.")
        _require_sha256(self.scan_sha256, name="scan_sha256")
        if self.glyph_count < 0:
            raise ValueError("glyph_count must be nonnegative.")
        if not self.page_state:
            raise ValueError("page_state must name the alignment state the page was in.")
        if self.admitted_line_count < 0 or self.recognizer_admitted_line_count < 0:
            raise ValueError("A page's admitted line counts must be nonnegative.")
        if self.recognizer_admitted_line_count > self.admitted_line_count:
            raise ValueError("A page cannot admit more lines by recognizer than in total.")
        if self.page_state == "accepted" and self.recognizer_admitted_line_count:
            raise ValueError("An accepted page's lines are admitted by its own alignment gate.")
        if self.page_class is not None and not self.page_class:
            raise ValueError("page_class must be a nonempty name or null.")
        if self.line_x_height_median_px is not None and self.line_x_height_median_px <= 0:
            raise ValueError("line_x_height_median_px must be positive.")
        if self.line_x_height_spread_px is not None and self.line_x_height_spread_px < 0:
            raise ValueError("line_x_height_spread_px must be nonnegative.")
        if (self.line_x_height_median_px is None) != (self.line_x_height_spread_px is None):
            raise ValueError("A page's x-height median and spread are recorded together.")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "admitted_line_count": self.admitted_line_count,
            "glyph_count": self.glyph_count,
            "line_x_height_median_px": self.line_x_height_median_px,
            "line_x_height_spread_px": self.line_x_height_spread_px,
            "page_class": self.page_class,
            "page_name": self.page_name,
            "page_state": self.page_state,
            "recognizer_admitted_line_count": self.recognizer_admitted_line_count,
            "scan_sha256": self.scan_sha256,
            "source_path": self.source_path,
        }


@dataclass(frozen=True, slots=True)
class AtlasSheet:
    """One rendered atlas sheet, hashed so a re-render can be compared byte for byte."""

    character: str
    label_tier: LabelTier
    sheet_ordinal: int
    path: str
    cell_count: int
    cell_width_px: int
    cell_height_px: int
    sha256: str
    clipped_cell_count: int = 0
    label_style: str | None = None

    def __post_init__(self) -> None:
        if len(self.character) != 1:
            raise ValueError("An atlas sheet holds one character.")
        _require_tier(self.label_tier, name="label_tier")
        if self.sheet_ordinal < 0:
            raise ValueError("sheet_ordinal must be nonnegative.")
        if not self.path or self.path.startswith("/") or ".." in Path(self.path).parts:
            raise ValueError("An atlas path must be relative to the inventory directory.")
        if self.cell_count <= 0:
            raise ValueError("An atlas sheet holds at least one cell.")
        if self.cell_width_px <= 0 or self.cell_height_px <= 0:
            raise ValueError("An atlas cell must have positive width and height.")
        if not 0 <= self.clipped_cell_count <= self.cell_count:
            raise ValueError("clipped_cell_count may not exceed the sheet's cell count.")
        if self.label_style is not None and not self.label_style:
            raise ValueError("label_style must be a nonempty name or null.")
        _require_sha256(self.sha256, name="sha256")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "cell_count": self.cell_count,
            "cell_height_px": self.cell_height_px,
            "cell_width_px": self.cell_width_px,
            "character": self.character,
            "clipped_cell_count": self.clipped_cell_count,
            "label_style": self.label_style,
            "label_tier": self.label_tier,
            "path": self.path,
            "sha256": self.sha256,
            "sheet_ordinal": self.sheet_ordinal,
        }


@dataclass(frozen=True, slots=True)
class GlyphManifest:
    """Stable JSON-safe provenance and coverage for one book's glyph inventory."""

    tool_version: str
    project_id: str
    alignment_label: str
    alignment_sha256: str
    profile_label: str
    profile_sha256: str
    rows_label: str
    rows_sha256: str
    methods: Mapping[str, object]
    thresholds: Mapping[str, object]
    page_count: int
    accepted_page_count: int
    harvested_page_count: int
    reconciled_word_count: int
    separable_word_count: int
    furniture_word_count: int
    glyph_count_by_tier: Mapping[str, int]
    coverage: tuple[CharacterCoverage, ...] = ()
    quality_flag_counts: Mapping[str, int] = field(default_factory=dict)
    word_reject_counts: Mapping[str, int] = field(default_factory=dict)
    page_exclusion_counts: Mapping[str, int] = field(default_factory=dict)
    pages: tuple[GlyphPage, ...] = ()
    atlas: tuple[AtlasSheet, ...] = ()
    geometry_label: str | None = None
    geometry_sha256: str | None = None
    ocr_recognizer: Mapping[str, str] | None = None
    schema_version: int = GLYPHS_SCHEMA_VERSION
    algorithm_version: str = GLYPHS_ALGORITHM_VERSION
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != GLYPHS_SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version {self.schema_version!r}.")
        if self.algorithm_version != GLYPHS_ALGORITHM_VERSION:
            raise ValueError(f"Unsupported algorithm_version {self.algorithm_version!r}.")
        if not self.tool_version or not self.project_id:
            raise ValueError("A manifest names both its tool version and its project.")
        for name, label in (
            ("alignment_label", self.alignment_label),
            ("profile_label", self.profile_label),
            ("rows_label", self.rows_label),
        ):
            _require_file_label(label, name=name)
        for name, digest in (
            ("alignment_sha256", self.alignment_sha256),
            ("profile_sha256", self.profile_sha256),
            ("rows_sha256", self.rows_sha256),
        ):
            _require_sha256(digest, name=name)
        if (self.geometry_label is None) != (self.geometry_sha256 is None):
            raise ValueError("A geometry record is named and hashed together or not at all.")
        if self.geometry_label is not None:
            _require_file_label(self.geometry_label, name="geometry_label")
        if self.geometry_sha256 is not None:
            _require_sha256(self.geometry_sha256, name="geometry_sha256")
        if self.geometry_label is None and self.ocr_recognizer is not None:
            raise ValueError("A recognizer identity requires the geometry record that carries it.")
        for name, count in (
            ("page_count", self.page_count),
            ("accepted_page_count", self.accepted_page_count),
            ("harvested_page_count", self.harvested_page_count),
            ("reconciled_word_count", self.reconciled_word_count),
            ("separable_word_count", self.separable_word_count),
            ("furniture_word_count", self.furniture_word_count),
        ):
            if count < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if self.harvested_page_count > self.page_count:
            raise ValueError("A book cannot harvest more pages than it has.")
        if self.separable_word_count > self.reconciled_word_count:
            raise ValueError("A book cannot separate more words than it reconciled.")
        object.__setattr__(
            self,
            "glyph_count_by_tier",
            _counts_dict(self.glyph_count_by_tier, name="glyph_count_by_tier"),
        )
        for tier in self.glyph_count_by_tier:
            _require_tier(tier, name="glyph_count_by_tier key")
        object.__setattr__(
            self,
            "quality_flag_counts",
            _counts_dict(self.quality_flag_counts, name="quality_flag_counts"),
        )
        object.__setattr__(
            self,
            "word_reject_counts",
            _counts_dict(self.word_reject_counts, name="word_reject_counts"),
        )
        object.__setattr__(
            self,
            "page_exclusion_counts",
            _counts_dict(self.page_exclusion_counts, name="page_exclusion_counts"),
        )
        object.__setattr__(
            self,
            "coverage",
            tuple(
                sorted(
                    self.coverage,
                    key=lambda row: (row.label_tier, row.label_style or "", row.character),
                )
            ),
        )
        object.__setattr__(
            self,
            "pages",
            tuple(sorted(self.pages, key=lambda page: natural_page_key(page.page_name))),
        )
        object.__setattr__(self, "atlas", tuple(sorted(self.atlas, key=lambda sheet: sheet.path)))
        page_names = tuple(page.page_name for page in self.pages)
        if len(set(page_names)) != len(page_names):
            raise ValueError("A manifest may not repeat a page.")
        coverage_totals: dict[str, int] = {}
        for row in self.coverage:
            coverage_totals[row.label_tier] = (
                coverage_totals.get(row.label_tier, 0) + row.glyph_count
            )
        for tier, total in coverage_totals.items():
            if self.glyph_count_by_tier.get(tier, 0) != total:
                raise ValueError(f"Coverage for tier {tier!r} disagrees with its glyph count.")
        if self.ocr_recognizer is not None:
            object.__setattr__(self, "ocr_recognizer", dict(sorted(self.ocr_recognizer.items())))
        object.__setattr__(self, "methods", dict(sorted(self.methods.items())))
        object.__setattr__(self, "thresholds", dict(sorted(self.thresholds.items())))
        object.__setattr__(self, "extensions", dict(sorted(self.extensions.items())))

    @property
    def glyph_count(self) -> int:
        return sum(self.glyph_count_by_tier.values())

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "accepted_page_count": self.accepted_page_count,
            "algorithm_version": self.algorithm_version,
            "alignment_label": self.alignment_label,
            "alignment_sha256": self.alignment_sha256,
            "atlas": [sheet.to_dict() for sheet in self.atlas],
            "coverage": [row.to_dict() for row in self.coverage],
            "furniture_word_count": self.furniture_word_count,
            "geometry_label": self.geometry_label,
            "geometry_sha256": self.geometry_sha256,
            "glyph_count": self.glyph_count,
            "glyph_count_by_tier": dict(self.glyph_count_by_tier),
            "harvested_page_count": self.harvested_page_count,
            "page_count": self.page_count,
            "methods": _mapping_dict(self.methods),
            "ocr_recognizer": None if self.ocr_recognizer is None else dict(self.ocr_recognizer),
            "page_exclusion_counts": dict(self.page_exclusion_counts),
            "pages": [page.to_dict() for page in self.pages],
            "profile_label": self.profile_label,
            "profile_sha256": self.profile_sha256,
            "project_id": self.project_id,
            "quality_flag_counts": dict(self.quality_flag_counts),
            "reconciled_word_count": self.reconciled_word_count,
            "rows_label": self.rows_label,
            "rows_sha256": self.rows_sha256,
            "schema_version": self.schema_version,
            "separable_word_count": self.separable_word_count,
            "thresholds": _mapping_dict(self.thresholds),
            "tool_version": self.tool_version,
            "word_reject_counts": dict(self.word_reject_counts),
        }
        payload.update(_mapping_dict(self.extensions))
        return payload

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> GlyphManifest:
        try:
            return GlyphManifestInput.model_validate_json(payload).to_domain()
        except (TypeError, ValidationError) as error:
            raise GlyphContractError(f"Invalid PGDP glyph manifest: {error}") from error

    @classmethod
    def json_schema(cls) -> str:
        return (
            json.dumps(GlyphManifestInput.model_json_schema(), indent=2, ensure_ascii=False) + "\n"
        )


def render_rows(rows: Sequence[GlyphRow]) -> str:
    """Every row as one JSONL document, in the inventory's own deterministic order."""

    ordered = sorted(rows, key=lambda row: row.sort_key)
    identities = [row.identity for row in ordered]
    if len(set(identities)) != len(identities):
        raise GlyphContractError("Two glyph rows claim the same page, line, word and ordinal.")
    return "".join(f"{row.to_json()}\n" for row in ordered)


def read_rows(payload: str | bytes) -> tuple[GlyphRow, ...]:
    """Read a `glyphs.jsonl` document back into rows, refusing a repeated identity."""

    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    rows = tuple(GlyphRow.from_json(line) for line in text.splitlines() if line.strip())
    identities = [row.identity for row in rows]
    if len(set(identities)) != len(identities):
        raise GlyphContractError("Two glyph rows claim the same page, line, word and ordinal.")
    return rows


class _WireModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", frozen=True, strict=True)


def _wire_extensions(model: _WireModel) -> dict[str, JsonValue]:
    extras = model.model_extra
    return {} if extras is None else _JSON_OBJECT_ADAPTER.validate_python(extras)


def _normalize_tuple_fields(payload: object, fields: tuple[str, ...]) -> object:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    for name in fields:
        value = normalized.get(name)
        if isinstance(value, list):
            normalized[name] = tuple(value)
    return normalized


class GlyphRowInput(_WireModel):
    character: StrictStr = Field(min_length=1, max_length=1)
    label_tier: Literal["transcribed", "recognized"]
    page_name: StrictStr = Field(min_length=1)
    line_ordinal: StrictInt = Field(ge=0)
    word_ordinal: StrictInt = Field(ge=0)
    glyph_ordinal: StrictInt = Field(ge=0)
    box: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    ink_density: StrictFloat = Field(ge=0.0, le=1.0)
    row_extent_px: StrictInt = Field(gt=0)
    top_offset_px: StrictInt | None = None
    bottom_offset_px: StrictInt | None = None
    flags: tuple[StrictStr, ...] = ()
    label_confidence: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    label_style: StrictStr | None = Field(default=None, min_length=1)
    source_line_ordinal: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_sequences(cls, payload: object) -> object:
        return _normalize_tuple_fields(payload, ("box", "flags"))

    def to_domain(self) -> GlyphRow:
        return GlyphRow(
            character=self.character,
            label_tier=self.label_tier,
            page_name=self.page_name,
            line_ordinal=self.line_ordinal,
            word_ordinal=self.word_ordinal,
            glyph_ordinal=self.glyph_ordinal,
            box=self.box,
            ink_density=self.ink_density,
            row_extent_px=self.row_extent_px,
            top_offset_px=self.top_offset_px,
            bottom_offset_px=self.bottom_offset_px,
            flags=self.flags,
            label_confidence=self.label_confidence,
            label_style=self.label_style,
            source_line_ordinal=self.source_line_ordinal,
        )


class CharacterCoverageInput(_WireModel):
    character: StrictStr = Field(min_length=1, max_length=1)
    label_tier: Literal["transcribed", "recognized"]
    glyph_count: StrictInt = Field(gt=0)
    label_style: StrictStr | None = Field(default=None, min_length=1)

    def to_domain(self) -> CharacterCoverage:
        return CharacterCoverage(
            character=self.character,
            label_tier=self.label_tier,
            glyph_count=self.glyph_count,
            label_style=self.label_style,
        )


class GlyphPageInput(_WireModel):
    page_name: StrictStr = Field(min_length=1)
    scan_sha256: StrictStr = Field(min_length=64, max_length=64)
    source_path: StrictStr = Field(min_length=1)
    glyph_count: StrictInt = Field(ge=0)
    page_state: StrictStr = Field(default="accepted", min_length=1)
    admitted_line_count: StrictInt = Field(default=0, ge=0)
    recognizer_admitted_line_count: StrictInt = Field(default=0, ge=0)
    page_class: StrictStr | None = Field(default=None, min_length=1)
    line_x_height_median_px: StrictInt | None = Field(default=None, gt=0)
    line_x_height_spread_px: StrictInt | None = Field(default=None, ge=0)

    def to_domain(self) -> GlyphPage:
        return GlyphPage(
            page_name=self.page_name,
            scan_sha256=self.scan_sha256,
            source_path=self.source_path,
            glyph_count=self.glyph_count,
            page_state=self.page_state,
            admitted_line_count=self.admitted_line_count,
            recognizer_admitted_line_count=self.recognizer_admitted_line_count,
            page_class=self.page_class,
            line_x_height_median_px=self.line_x_height_median_px,
            line_x_height_spread_px=self.line_x_height_spread_px,
        )


class AtlasSheetInput(_WireModel):
    character: StrictStr = Field(min_length=1, max_length=1)
    label_tier: Literal["transcribed", "recognized"]
    sheet_ordinal: StrictInt = Field(ge=0)
    path: StrictStr = Field(min_length=1)
    cell_count: StrictInt = Field(gt=0)
    cell_width_px: StrictInt = Field(gt=0)
    cell_height_px: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(min_length=64, max_length=64)
    clipped_cell_count: StrictInt = Field(default=0, ge=0)
    label_style: StrictStr | None = Field(default=None, min_length=1)

    def to_domain(self) -> AtlasSheet:
        return AtlasSheet(
            character=self.character,
            label_tier=self.label_tier,
            sheet_ordinal=self.sheet_ordinal,
            path=self.path,
            cell_count=self.cell_count,
            cell_width_px=self.cell_width_px,
            cell_height_px=self.cell_height_px,
            sha256=self.sha256,
            clipped_cell_count=self.clipped_cell_count,
            label_style=self.label_style,
        )


class GlyphManifestInput(_WireModel):
    schema_version: StrictInt
    algorithm_version: StrictStr
    tool_version: StrictStr = Field(min_length=1)
    project_id: StrictStr = Field(min_length=1)
    alignment_label: StrictStr = Field(min_length=1)
    alignment_sha256: StrictStr = Field(min_length=64, max_length=64)
    profile_label: StrictStr = Field(min_length=1)
    profile_sha256: StrictStr = Field(min_length=64, max_length=64)
    rows_label: StrictStr = Field(min_length=1)
    rows_sha256: StrictStr = Field(min_length=64, max_length=64)
    methods: dict[str, JsonValue] = Field(default_factory=dict)
    thresholds: dict[str, JsonValue] = Field(default_factory=dict)
    page_count: StrictInt = Field(ge=0)
    accepted_page_count: StrictInt = Field(ge=0)
    harvested_page_count: StrictInt = Field(ge=0)
    reconciled_word_count: StrictInt = Field(ge=0)
    separable_word_count: StrictInt = Field(ge=0)
    furniture_word_count: StrictInt = Field(ge=0)
    glyph_count_by_tier: dict[str, StrictInt] = Field(default_factory=dict)
    coverage: tuple[CharacterCoverageInput, ...] = ()
    quality_flag_counts: dict[str, StrictInt] = Field(default_factory=dict)
    word_reject_counts: dict[str, StrictInt] = Field(default_factory=dict)
    page_exclusion_counts: dict[str, StrictInt] = Field(default_factory=dict)
    pages: tuple[GlyphPageInput, ...] = ()
    atlas: tuple[AtlasSheetInput, ...] = ()
    geometry_label: StrictStr | None = None
    geometry_sha256: StrictStr | None = Field(default=None, min_length=64, max_length=64)
    ocr_recognizer: dict[str, StrictStr] | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_sequences(cls, payload: object) -> object:
        return _normalize_tuple_fields(payload, ("coverage", "pages", "atlas"))

    def to_domain(self) -> GlyphManifest:
        extensions = _wire_extensions(self)
        _ = extensions.pop("glyph_count", None)
        return GlyphManifest(
            tool_version=self.tool_version,
            project_id=self.project_id,
            alignment_label=self.alignment_label,
            alignment_sha256=self.alignment_sha256,
            profile_label=self.profile_label,
            profile_sha256=self.profile_sha256,
            rows_label=self.rows_label,
            rows_sha256=self.rows_sha256,
            methods=self.methods,
            thresholds=self.thresholds,
            page_count=self.page_count,
            accepted_page_count=self.accepted_page_count,
            harvested_page_count=self.harvested_page_count,
            reconciled_word_count=self.reconciled_word_count,
            separable_word_count=self.separable_word_count,
            furniture_word_count=self.furniture_word_count,
            glyph_count_by_tier=self.glyph_count_by_tier,
            coverage=tuple(row.to_domain() for row in self.coverage),
            quality_flag_counts=self.quality_flag_counts,
            word_reject_counts=self.word_reject_counts,
            page_exclusion_counts=self.page_exclusion_counts,
            pages=tuple(page.to_domain() for page in self.pages),
            atlas=tuple(sheet.to_domain() for sheet in self.atlas),
            geometry_label=self.geometry_label,
            geometry_sha256=self.geometry_sha256,
            ocr_recognizer=self.ocr_recognizer,
            schema_version=self.schema_version,
            algorithm_version=self.algorithm_version,
            extensions=extensions,
        )
