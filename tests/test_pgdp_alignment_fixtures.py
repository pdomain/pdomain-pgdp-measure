"""Reviewed synthetic geometry fixtures for PGDP source-line alignment."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar, Literal

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
)

from pdomain_pgdp_measure.alignment_dp import AlignmentResult, align_sequences
from pdomain_pgdp_measure.alignment_image import extract_line_candidates
from pdomain_pgdp_measure.alignment_source import tokenize_f2_pages
from pdomain_pgdp_measure.image_measurement import measure_image_snapshot, open_image_snapshot
from pdomain_pgdp_measure.profile_models import InkBand

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "pgdp_alignment"


class _ExpectedOperationPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    kind: Literal["match", "skip_image", "skip_text"]
    source_ordinal: StrictInt | None
    candidate_ordinal: StrictInt | None
    cost: StrictFloat


class _AlignmentCasePayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    case_id: StrictStr
    project_id: StrictStr
    page_name: StrictStr
    f2_text: StrictStr
    image_path: StrictStr
    source_sha256: StrictStr
    fixture_sha256: StrictStr
    transform: StrictStr
    foreground_bounds: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    expected_candidates: list[tuple[StrictInt, StrictInt, StrictInt, StrictInt]]
    reviewed_on: StrictStr
    accepted: StrictBool
    exclusions: list[StrictStr]
    operations: list[_ExpectedOperationPayload]


class _ManifestPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    cases: list[_AlignmentCasePayload]


_MANIFEST_ADAPTER = TypeAdapter(_ManifestPayload)


@dataclass(frozen=True, slots=True)
class ExpectedOperation:
    kind: Literal["match", "skip_image", "skip_text"]
    source_ordinal: int | None
    candidate_ordinal: int | None
    cost: float


@dataclass(frozen=True, slots=True)
class AlignmentCase:
    case_id: str
    project_id: str
    page_name: str
    f2_text: str
    image_path: str
    source_sha256: str
    fixture_sha256: str
    transform: str
    foreground_bounds: tuple[int, int, int, int]
    expected_candidates: tuple[tuple[int, int, int, int], ...]
    reviewed_on: str
    accepted: bool
    exclusions: tuple[str, ...]
    operations: tuple[ExpectedOperation, ...]


def load_alignment_cases() -> tuple[AlignmentCase, ...]:
    """Load the exact reviewed fixture-manifest shape."""

    payload = _MANIFEST_ADAPTER.validate_json((_FIXTURE_ROOT / "manifest.json").read_text("utf-8"))
    if len(payload.cases) != 10:
        raise ValueError("The reviewed alignment manifest must contain exactly ten cases.")
    return tuple(
        AlignmentCase(
            case_id=item.case_id,
            project_id=item.project_id,
            page_name=item.page_name,
            f2_text=item.f2_text,
            image_path=item.image_path,
            source_sha256=item.source_sha256,
            fixture_sha256=item.fixture_sha256,
            transform=item.transform,
            foreground_bounds=item.foreground_bounds,
            expected_candidates=tuple(item.expected_candidates),
            reviewed_on=item.reviewed_on,
            accepted=item.accepted,
            exclusions=tuple(item.exclusions),
            operations=tuple(
                ExpectedOperation(
                    kind=operation.kind,
                    source_ordinal=operation.source_ordinal,
                    candidate_ordinal=operation.candidate_ordinal,
                    cost=operation.cost,
                )
                for operation in item.operations
            ),
        )
        for item in payload.cases
    )


def run_alignment_case(case: AlignmentCase) -> AlignmentResult:
    """Run public source tokenization, extraction, and alignment for one fixture."""

    image_path = (_FIXTURE_ROOT / case.image_path).resolve()
    if _FIXTURE_ROOT.resolve() not in image_path.parents:
        raise ValueError(f"Fixture image path escapes the fixture root: {case.image_path}")
    fixture_bytes = image_path.read_bytes()
    fixture_sha256 = sha256(fixture_bytes).hexdigest()
    if fixture_sha256 != case.fixture_sha256:
        raise ValueError(f"Fixture hash mismatch: {case.case_id}")
    source_page = tokenize_f2_pages({case.page_name: case.f2_text}).pages[0]
    with open_image_snapshot(image_path) as snapshot:
        measurement = measure_image_snapshot(snapshot)
        if measurement.foreground_bounds != case.foreground_bounds:
            raise ValueError(f"Fixture foreground bounds changed: {case.case_id}")
        extraction = extract_line_candidates(
            snapshot,
            source_frame=measurement.source_frame,
            foreground_bounds=case.foreground_bounds,
            ink_bands=tuple(
                InkBand(y_start=band.y_start, y_end=band.y_end) for band in measurement.ink_bands
            ),
        )
    if extraction.source_sha256 != case.fixture_sha256:
        raise ValueError(f"Fixture extraction hash mismatch: {case.case_id}")
    return align_sequences(
        source_page.lines,
        extraction.candidates,
        foreground_bounds=case.foreground_bounds,
        inherited_exclusions=extraction.exclusions,
        probable_multi_column=extraction.probable_multi_column,
        source_page=source_page,
    )


def _result_operations(result: AlignmentResult) -> tuple[ExpectedOperation, ...]:
    return tuple(
        ExpectedOperation(
            kind=operation.kind,
            source_ordinal=operation.source_ordinal,
            candidate_ordinal=operation.candidate_ordinal,
            cost=operation.cost,
        )
        for operation in result.operations
    )


def test_manifest_is_strict_and_reviewed_on_the_declared_date() -> None:
    cases = load_alignment_cases()

    assert [case.case_id for case in cases] == [
        "alpha-prose",
        "beta-indented",
        "beta-styled",
        "gamma-unicode",
        "gamma-ragged",
        "delta-columns",
        "delta-table-rule",
        "epsilon-too-few",
        "epsilon-source-diagnostic",
        "zeta-illustration",
    ]
    assert all(case.reviewed_on == "2026-08-24" for case in cases)
    assert all(case.source_sha256 != case.fixture_sha256 for case in cases)
    assert len({case.project_id for case in cases if case.accepted}) >= 3
    assert all(len(box) == 4 for case in cases for box in case.expected_candidates)


@pytest.mark.parametrize(
    "case",
    load_alignment_cases(),
    ids=tuple(case.case_id for case in load_alignment_cases()),
)
def test_reviewed_alignment_case(case: AlignmentCase) -> None:
    result = run_alignment_case(case)

    assert tuple(candidate.box for candidate in result.candidates) == case.expected_candidates
    assert result.accepted is case.accepted
    assert result.exclusions == case.exclusions
    assert _result_operations(result) == case.operations
