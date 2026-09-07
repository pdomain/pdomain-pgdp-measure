"""Deterministically align eligible PGDP source lines to scan candidates."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from statistics import median
from typing import TYPE_CHECKING, Final, Literal

from pdomain_pgdp_measure.f2 import ParsedF2Page
from pdomain_pgdp_measure.features import extract_page_features

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pdomain_pgdp_measure.alignment_image import Bounds, LineCandidate
    from pdomain_pgdp_measure.alignment_source import (
        FormattingSpan,
        SourceLine,
        StyleRun,
        TokenizedSourcePage,
    )


OperationKind = Literal["match", "skip_image", "skip_text"]
AlignmentState = Literal["accepted", "proposed", "excluded"]

TIE_PRIORITY: Final[dict[OperationKind, int]] = {
    "match": 0,
    "skip_image": 1,
    "skip_text": 2,
}
SKIP_COST: Final = 1.0
MINIMUM_SOURCE_LINES: Final = 4
MAXIMUM_SOURCE_LINES: Final = 80
MINIMUM_MATCHED_RATIO: Final = 0.90
MAXIMUM_NORMALIZED_COST: Final = 0.22
MINIMUM_UNIQUENESS_MARGIN: Final = 0.15
_RANK_FEATURE_METHOD: Final = "pgdp-rank/v1"
_ALIGNED_FIELDS_PATTERN: Final = re.compile(r"\S(?:.*?\S)? {3,}\S(?:.*\S)?$")
_ILLUSTRATION_OR_ORNAMENT_PATTERN: Final = re.compile(
    r"\[(?:illustration|decoration)\b[^\]]*\]", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class MatchCost:
    """The four fixed weak visual costs for one source-to-image match."""

    width: float
    indent: float
    gaps: float
    style: float

    @property
    def total(self) -> float:
        """Return the fixed weighted binary64 match cost."""

        return 0.55 * self.width + 0.25 * self.indent + 0.15 * self.gaps + 0.05 * self.style


@dataclass(frozen=True, slots=True)
class EligibleSourceLine:
    """One nonblank source line retained for sequence alignment."""

    line: SourceLine
    blank_lines_before: int
    leading_spaces: int
    styled: bool


@dataclass(frozen=True, slots=True)
class AlignmentOperation:
    """One monotone edit operation with all cost evidence retained."""

    kind: OperationKind
    source_ordinal: int | None
    candidate_ordinal: int | None
    cost: float
    match_cost: MatchCost | None = None


@dataclass(frozen=True, slots=True)
class AlignmentPath:
    """One complete monotone operation path."""

    operations: tuple[AlignmentOperation, ...]
    total_cost: float


@dataclass(frozen=True, slots=True)
class SourceFeatureExclusion:
    """Verified source evidence for a ranking-feature page exclusion."""

    code: Literal["table_like", "illustration_marker"]
    predicate_method: Literal["pgdp-rank/v1"]
    source_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source_ordinals:
            raise ValueError("Feature exclusions require verified source ordinals.")
        if tuple(sorted(set(self.source_ordinals))) != self.source_ordinals:
            raise ValueError("Feature exclusion source ordinals must be sorted and unique.")


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """A best and second-best path plus conservative page-level assessment."""

    eligible_source_lines: tuple[EligibleSourceLine, ...]
    candidates: tuple[LineCandidate, ...]
    operations: tuple[AlignmentOperation, ...]
    best: AlignmentPath
    second_best: AlignmentPath | None
    total_cost: float
    normalized_cost: float
    matched_count: int
    matched_ratio: float
    uniqueness_margin: float
    mean_width_residual: float | None
    mean_indentation_residual: float | None
    state: AlignmentState
    accepted: bool
    exclusions: tuple[str, ...]
    exclusion_evidence: tuple[SourceFeatureExclusion, ...]
    confidence: None = None
    confidence_kind: Literal["uncalibrated"] = "uncalibrated"


@dataclass(frozen=True, slots=True)
class _PathNode:
    """One retained dynamic-programming path step linked to its predecessor."""

    parent: _PathNode | None
    operation: AlignmentOperation | None
    lexical_rank: int
    rank_depth: int
    total_cost: float


def align_sequences(
    source_lines: Sequence[SourceLine],
    candidates: Sequence[LineCandidate],
    *,
    foreground_bounds: Bounds,
    inherited_exclusions: Iterable[str] = (),
    probable_multi_column: bool = False,
    source_page: TokenizedSourcePage,
) -> AlignmentResult:
    """Return deterministic monotone source-to-candidate alignment evidence.

    ``source_lines`` retains blank records so their count can participate in the
    vertical-gap cost.  Only matching-eligible nonblank lines enter the DP.
    """

    source = tuple(source_lines)
    image_candidates = tuple(candidates)
    if source != source_page.lines:
        raise ValueError("source_page lines must be the exact source sequence used for alignment.")
    feature_exclusions = derive_source_feature_exclusions(source_page)
    eligible = _eligible_lines(source, source_page.style_runs)
    malformed_control = any(not line.matching_eligible for line in source)
    count_exclusions = _pre_dp_count_exclusions(len(eligible), len(image_candidates))
    if count_exclusions:
        return _excluded_without_dp(
            eligible,
            image_candidates,
            exclusion_evidence=feature_exclusions,
            count_exclusions=count_exclusions,
            inherited_exclusions=inherited_exclusions,
            probable_multi_column=probable_multi_column,
            malformed_control=malformed_control,
        )
    foreground_x0, foreground_width = _foreground_geometry(foreground_bounds)
    costs = _match_costs(eligible, image_candidates, foreground_x0, foreground_width)
    best, second_best = _align(eligible, image_candidates, costs)
    denominator = max(len(eligible), len(image_candidates), 1)
    normalized_cost = best.total_cost / denominator
    matched_operations = tuple(
        operation for operation in best.operations if operation.kind == "match"
    )
    matched_count = len(matched_operations)
    matched_ratio = matched_count / len(eligible) if eligible else 0.0
    uniqueness_margin = (
        abs(second_best.total_cost - best.total_cost) if second_best is not None else 0.0
    )
    width_residuals = tuple(
        operation.match_cost.width
        for operation in matched_operations
        if operation.match_cost is not None
    )
    indent_residuals = tuple(
        operation.match_cost.indent
        for operation in matched_operations
        if operation.match_cost is not None
    )
    preliminary = AlignmentResult(
        eligible_source_lines=eligible,
        candidates=image_candidates,
        operations=best.operations,
        best=best,
        second_best=second_best,
        total_cost=best.total_cost,
        normalized_cost=normalized_cost,
        matched_count=matched_count,
        matched_ratio=matched_ratio,
        uniqueness_margin=uniqueness_margin,
        mean_width_residual=_mean_or_none(width_residuals),
        mean_indentation_residual=_mean_or_none(indent_residuals),
        state="proposed",
        accepted=False,
        exclusions=(),
        exclusion_evidence=feature_exclusions,
    )
    return assess_alignment(
        preliminary,
        inherited_exclusions=inherited_exclusions,
        probable_multi_column=probable_multi_column,
        malformed_control=malformed_control,
    )


def assess_alignment(
    result: AlignmentResult,
    *,
    inherited_exclusions: Iterable[str] = (),
    probable_multi_column: bool = False,
    malformed_control: bool = False,
) -> AlignmentResult:
    """Classify one DP result using the fixed precision-first eligibility gate."""

    exclusions = list(dict.fromkeys(inherited_exclusions))
    source_count = len(result.eligible_source_lines)
    candidate_count = len(result.candidates)
    pre_alignment_codes = set(exclusions)
    if probable_multi_column:
        exclusions.append("persistent_gutter")
        pre_alignment_codes.add("persistent_gutter")
    for evidence in result.exclusion_evidence:
        exclusions.append(evidence.code)
        pre_alignment_codes.add(evidence.code)
    if malformed_control:
        exclusions.append("malformed_control")
        pre_alignment_codes.add("malformed_control")
    if not MINIMUM_SOURCE_LINES <= source_count <= MAXIMUM_SOURCE_LINES:
        exclusions.append("line_count_out_of_range")
        pre_alignment_codes.add("line_count_out_of_range")
    if pre_alignment_codes:
        return replace(
            result,
            state="excluded",
            accepted=False,
            exclusions=tuple(dict.fromkeys(exclusions)),
        )

    if result.matched_ratio < MINIMUM_MATCHED_RATIO:
        exclusions.append("matched_ratio_below_minimum")
    maximum_count_difference = max(2, math.ceil(0.10 * source_count))
    if abs(source_count - candidate_count) > maximum_count_difference:
        exclusions.append("line_count_difference_exceeds_maximum")
    if result.normalized_cost > MAXIMUM_NORMALIZED_COST:
        exclusions.append("normalized_cost_exceeds_maximum")
    if result.uniqueness_margin < MINIMUM_UNIQUENESS_MARGIN:
        exclusions.append("uniqueness_margin_below_minimum")
    if exclusions:
        return replace(
            result,
            state="proposed",
            accepted=False,
            exclusions=tuple(dict.fromkeys(exclusions)),
        )
    return replace(result, state="accepted", accepted=True)


def _pre_dp_count_exclusions(source_count: int, candidate_count: int) -> tuple[str, ...]:
    """Return count exclusions that make DP allocation unnecessary."""

    exclusions: list[str] = []
    if not MINIMUM_SOURCE_LINES <= source_count <= MAXIMUM_SOURCE_LINES:
        exclusions.append("line_count_out_of_range")
    maximum_count_difference = max(2, math.ceil(0.10 * source_count))
    if abs(source_count - candidate_count) > maximum_count_difference:
        exclusions.append("line_count_difference_exceeds_maximum")
    return tuple(exclusions)


def _excluded_without_dp(
    source_lines: tuple[EligibleSourceLine, ...],
    candidates: tuple[LineCandidate, ...],
    *,
    exclusion_evidence: tuple[SourceFeatureExclusion, ...],
    count_exclusions: tuple[str, ...],
    inherited_exclusions: Iterable[str],
    probable_multi_column: bool,
    malformed_control: bool,
) -> AlignmentResult:
    """Return complete deterministic skip evidence without building a DP grid."""

    operations = _skip_operations(source_lines, candidates)
    total_cost = sum(operation.cost for operation in operations)
    best = AlignmentPath(operations=operations, total_cost=total_cost)
    exclusions = list(dict.fromkeys(inherited_exclusions))
    if probable_multi_column:
        exclusions.append("persistent_gutter")
    exclusions.extend(evidence.code for evidence in exclusion_evidence)
    if malformed_control:
        exclusions.append("malformed_control")
    exclusions.extend(count_exclusions)
    return AlignmentResult(
        eligible_source_lines=source_lines,
        candidates=candidates,
        operations=operations,
        best=best,
        second_best=None,
        total_cost=total_cost,
        normalized_cost=total_cost / max(len(source_lines), len(candidates), 1),
        matched_count=0,
        matched_ratio=0.0,
        uniqueness_margin=0.0,
        mean_width_residual=None,
        mean_indentation_residual=None,
        state="excluded",
        accepted=False,
        exclusions=tuple(dict.fromkeys(exclusions)),
        exclusion_evidence=exclusion_evidence,
    )


def _skip_operations(
    source_lines: tuple[EligibleSourceLine, ...], candidates: tuple[LineCandidate, ...]
) -> tuple[AlignmentOperation, ...]:
    """Return the deterministic complete skip path for an unaligned page."""

    return (
        *(
            AlignmentOperation(
                kind="skip_image",
                source_ordinal=None,
                candidate_ordinal=candidate_index,
                cost=SKIP_COST,
            )
            for candidate_index in range(len(candidates))
        ),
        *(
            AlignmentOperation(
                kind="skip_text",
                source_ordinal=source.line.ordinal,
                candidate_ordinal=None,
                cost=SKIP_COST,
            )
            for source in source_lines
        ),
    )


def derive_source_feature_exclusions(
    page: TokenizedSourcePage,
) -> tuple[SourceFeatureExclusion, ...]:
    """Return ranking-feature exclusions with immutable source-line evidence."""

    features = extract_page_features(to_feature_page(page))
    evidence: list[SourceFeatureExclusion] = []
    if features.table_like:
        ordinals = _table_source_ordinals(page)
        if len(ordinals) < 3:
            raise ValueError("Table-like feature evidence lacks three matching source lines.")
        evidence.append(SourceFeatureExclusion("table_like", _RANK_FEATURE_METHOD, ordinals))
    if features.illustration_or_ornament:
        ordinals = tuple(
            line.ordinal
            for line in page.lines
            if _ILLUSTRATION_OR_ORNAMENT_PATTERN.search(line.original_text) is not None
        )
        if not ordinals:
            raise ValueError("Illustration feature evidence lacks a matching source line.")
        evidence.append(
            SourceFeatureExclusion("illustration_marker", _RANK_FEATURE_METHOD, ordinals)
        )
    return tuple(evidence)


def to_feature_page(page: TokenizedSourcePage) -> ParsedF2Page:
    """Adapt one immutable tokenized page to the existing feature-input record.

    The adapter uses only source bytes and the tokenizer's closed formatting
    spans.  It deliberately does not read an F2 file or call the legacy parser.
    """

    source = page.source_utf8
    valid_spans = tuple(span for span in page.formatting_spans if span.closed)
    removed_ranges = tuple(
        removal_range
        for span in valid_spans
        for removal_range in _formatting_control_ranges(span, source_length=len(source))
    )
    feature_bytes = _remove_ranges(source, removed_ranges)
    special_blocks = tuple(
        source[span.byte_start : span.byte_end].decode("utf-8") for span in valid_spans
    )
    return ParsedF2Page(
        name=page.page_name,
        source_text=source.decode("utf-8"),
        feature_text=feature_bytes.decode("utf-8"),
        special_blocks=special_blocks,
        has_special_format=bool(special_blocks),
    )


def _eligible_lines(
    source_lines: tuple[SourceLine, ...], style_runs: Sequence[StyleRun]
) -> tuple[EligibleSourceLine, ...]:
    eligible: list[EligibleSourceLine] = []
    previous_eligible_index: int | None = None
    for index, line in enumerate(source_lines):
        if not line.matching_eligible or line.visible_text == "":
            continue
        blank_lines_before = 0
        if previous_eligible_index is not None:
            blank_lines_before = sum(
                prior.visible_text == ""
                for prior in source_lines[previous_eligible_index + 1 : index]
            )
        eligible.append(
            EligibleSourceLine(
                line=line,
                blank_lines_before=blank_lines_before,
                leading_spaces=_leading_spaces(line.visible_text),
                styled=_line_is_styled(line, style_runs),
            )
        )
        previous_eligible_index = index
    return tuple(eligible)


def _table_source_ordinals(page: TokenizedSourcePage) -> tuple[int, ...]:
    valid_spans = tuple(span for span in page.formatting_spans if span.closed)
    return tuple(
        line.ordinal
        for line in page.lines
        if _line_overlaps_a_span(line, valid_spans)
        and _ALIGNED_FIELDS_PATTERN.fullmatch(line.original_text.strip()) is not None
    )


def _line_overlaps_a_span(line: SourceLine, spans: Sequence[FormattingSpan]) -> bool:
    return any(
        span.byte_start < line.content_byte_end and line.content_byte_start < span.byte_end
        for span in spans
    )


def _formatting_control_ranges(
    span: FormattingSpan, *, source_length: int
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    if span.byte_start >= 2:
        ranges.append((span.byte_start - 2, span.byte_start))
    if span.byte_end < source_length:
        ranges.append((span.byte_end, span.byte_end + 2))
    return tuple(ranges)


def _remove_ranges(source: bytes, ranges: Sequence[tuple[int, int]]) -> bytes:
    result = bytearray()
    position = 0
    for start, end in sorted(ranges):
        if start < position:
            continue
        result.extend(source[position:start])
        position = end
    result.extend(source[position:])
    return bytes(result)


def _line_is_styled(line: SourceLine, style_runs: Sequence[StyleRun]) -> bool:
    if not line.style_fitting_eligible:
        return False
    return any(
        style.kind != "tb"
        and style.byte_start < line.content_byte_end
        and line.content_byte_start < style.byte_end
        for style in style_runs
    )


def _match_costs(
    source_lines: tuple[EligibleSourceLine, ...],
    candidates: tuple[LineCandidate, ...],
    foreground_x0: int,
    foreground_width: int,
) -> tuple[tuple[MatchCost, ...], ...]:
    max_text_length = max((len(item.line.visible_text) for item in source_lines), default=0)
    max_leading_spaces = max((item.leading_spaces for item in source_lines), default=0)
    gap_before = _candidate_gaps(candidates)
    max_candidate_gap = max(gap_before, default=0)
    candidate_heights = tuple(item.height for item in candidates)
    median_candidate_height = median(candidate_heights) if candidate_heights else 0
    return tuple(
        tuple(
            _match_cost(
                source,
                candidate,
                gap_before[candidate_index],
                max_text_length=max_text_length,
                max_leading_spaces=max_leading_spaces,
                foreground_x0=foreground_x0,
                foreground_width=foreground_width,
                max_candidate_gap=max_candidate_gap,
                median_candidate_height=median_candidate_height,
            )
            for candidate_index, candidate in enumerate(candidates)
        )
        for source in source_lines
    )


def _match_cost(
    source: EligibleSourceLine,
    candidate: LineCandidate,
    gap_before: int,
    *,
    max_text_length: int,
    max_leading_spaces: int,
    foreground_x0: int,
    foreground_width: int,
    max_candidate_gap: int,
    median_candidate_height: float,
) -> MatchCost:
    width = abs(
        len(source.line.visible_text) / _nonzero_denominator(max_text_length)
        - candidate.width / _nonzero_denominator(foreground_width)
    )
    indent = abs(
        source.leading_spaces / _nonzero_denominator(max_leading_spaces)
        - (candidate.box[0] - foreground_x0) / _nonzero_denominator(foreground_width)
    )
    candidate_gap_ratio = gap_before / _nonzero_denominator(max_candidate_gap)
    gaps = abs(min(source.blank_lines_before, 3) / 3 - min(max(candidate_gap_ratio, 0.0), 1.0))
    expected_height = 1.10 if source.styled else 1.0
    style = min(
        abs(candidate.height / _nonzero_denominator(median_candidate_height) - expected_height),
        1.0,
    )
    return MatchCost(width=width, indent=indent, gaps=gaps, style=style)


def _align(
    source_lines: tuple[EligibleSourceLine, ...],
    candidates: tuple[LineCandidate, ...],
    costs: tuple[tuple[MatchCost, ...], ...],
) -> tuple[AlignmentPath, AlignmentPath | None]:
    rows = len(source_lines)
    columns = len(candidates)
    rank_radix, rank_bits = _rank_encoding(source_lines, candidates)
    cells: list[list[tuple[_PathNode, ...]]] = [
        [() for _ in range(columns + 1)] for _ in range(rows + 1)
    ]
    cells[0][0] = (
        _PathNode(
            parent=None,
            operation=None,
            lexical_rank=0,
            rank_depth=0,
            total_cost=0.0,
        ),
    )
    for source_index in range(rows + 1):
        for candidate_index in range(columns + 1):
            if source_index == 0 and candidate_index == 0:
                continue
            options: list[_PathNode] = []
            if source_index and candidate_index:
                cost = costs[source_index - 1][candidate_index - 1]
                options.extend(
                    _append_operation(
                        path,
                        AlignmentOperation(
                            kind="match",
                            source_ordinal=source_lines[source_index - 1].line.ordinal,
                            candidate_ordinal=candidate_index - 1,
                            cost=cost.total,
                            match_cost=cost,
                        ),
                        rank_radix=rank_radix,
                        rank_bits=rank_bits,
                    )
                    for path in cells[source_index - 1][candidate_index - 1]
                )
            if candidate_index:
                options.extend(
                    _append_operation(
                        path,
                        AlignmentOperation(
                            kind="skip_image",
                            source_ordinal=None,
                            candidate_ordinal=candidate_index - 1,
                            cost=SKIP_COST,
                        ),
                        rank_radix=rank_radix,
                        rank_bits=rank_bits,
                    )
                    for path in cells[source_index][candidate_index - 1]
                )
            if source_index:
                options.extend(
                    _append_operation(
                        path,
                        AlignmentOperation(
                            kind="skip_text",
                            source_ordinal=source_lines[source_index - 1].line.ordinal,
                            candidate_ordinal=None,
                            cost=SKIP_COST,
                        ),
                        rank_radix=rank_radix,
                        rank_bits=rank_bits,
                    )
                    for path in cells[source_index - 1][candidate_index]
                )
            cells[source_index][candidate_index] = _best_two(options, rank_bits=rank_bits)
    paths = cells[rows][columns]
    best = _to_alignment_path(paths[0])
    second_best = _to_alignment_path(paths[1]) if len(paths) > 1 else None
    return best, second_best


def _append_operation(
    path: _PathNode,
    operation: AlignmentOperation,
    *,
    rank_radix: int,
    rank_bits: int,
) -> _PathNode:
    return _PathNode(
        parent=path,
        operation=operation,
        lexical_rank=(path.lexical_rank << rank_bits)
        | _operation_rank_code(operation, rank_radix=rank_radix),
        rank_depth=path.rank_depth + 1,
        total_cost=path.total_cost + operation.cost,
    )


def _best_two(paths: Sequence[_PathNode], *, rank_bits: int) -> tuple[_PathNode, ...]:
    """Retain two distinct lowest-cost nodes without reconstructing paths per cell."""

    distinct: list[_PathNode] = []
    ordered = sorted(paths, key=lambda path: path.total_cost)
    start = 0
    while start < len(ordered) and len(distinct) < 2:
        end = start + 1
        while end < len(ordered) and ordered[end].total_cost == ordered[start].total_cost:
            end += 1
        equal_cost_paths = ordered[start:end]
        greatest_rank_depth = max(path.rank_depth for path in equal_cost_paths)
        for path in sorted(
            equal_cost_paths,
            key=lambda node: (
                node.lexical_rank << (rank_bits * (greatest_rank_depth - node.rank_depth))
            ),
        ):
            if not any(
                path.lexical_rank == retained.lexical_rank
                and path.rank_depth == retained.rank_depth
                for retained in distinct
            ):
                distinct.append(path)
            if len(distinct) == 2:
                break
        start = end
    return tuple(distinct)


def _to_alignment_path(node: _PathNode) -> AlignmentPath:
    return AlignmentPath(
        operations=operations_from_node(node),
        total_cost=node.total_cost,
    )


def operations_from_node(node: _PathNode) -> tuple[AlignmentOperation, ...]:
    """Reconstruct one final alignment path from its retained backpointers."""

    reverse_operations: list[AlignmentOperation] = []
    current = node
    while current.operation is not None:
        reverse_operations.append(current.operation)
        parent = current.parent
        if parent is None:
            raise RuntimeError("Alignment path operation lacks a predecessor.")
        current = parent
    return tuple(reversed(reverse_operations))


def _rank_encoding(
    source_lines: tuple[EligibleSourceLine, ...], candidates: tuple[LineCandidate, ...]
) -> tuple[int, int]:
    """Return the radix and digit width for exact operation-sequence ordering."""

    maximum_ordinal = max(
        max((source.line.ordinal for source in source_lines), default=-1),
        len(candidates) - 1,
    )
    rank_radix = maximum_ordinal + 2
    maximum_code = 3 * rank_radix * rank_radix
    return (rank_radix, maximum_code.bit_length())


def _operation_rank_code(operation: AlignmentOperation, *, rank_radix: int) -> int:
    """Encode one operation tuple as a positive digit in the page-local rank radix."""

    source_ordinal = 0 if operation.source_ordinal is None else operation.source_ordinal + 1
    candidate_ordinal = (
        0 if operation.candidate_ordinal is None else operation.candidate_ordinal + 1
    )
    return (
        (TIE_PRIORITY[operation.kind] * rank_radix + source_ordinal) * rank_radix
        + candidate_ordinal
        + 1
    )


def _candidate_gaps(candidates: tuple[LineCandidate, ...]) -> tuple[int, ...]:
    gaps: list[int] = []
    for index, candidate in enumerate(candidates):
        if index == 0:
            gaps.append(0)
        else:
            gaps.append(max(0, candidate.box[1] - candidates[index - 1].box[3]))
    return tuple(gaps)


def _foreground_geometry(bounds: Bounds) -> tuple[int, int]:
    width = bounds[2] - bounds[0]
    if width < 0:
        raise ValueError("Foreground bounds must not be inverted.")
    return (bounds[0], width)


def _nonzero_denominator(value: int | float) -> int | float:
    return value if value != 0 else 1


def _leading_spaces(text: str) -> int:
    return len(text) - len(text.lstrip(" "))


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
