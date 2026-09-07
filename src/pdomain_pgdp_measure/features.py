from __future__ import annotations

import re
from statistics import median
from typing import TYPE_CHECKING

from pdomain_pgdp_measure.models import PageFeatures, PageScore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pdomain_pgdp_measure.f2 import ParsedF2Page


_OPENING_TAG_PATTERNS: dict[str, re.Pattern[str]] = {
    "italic": re.compile(r"<i(?:\s+[^>]*)?/?>", re.IGNORECASE),
    "bold": re.compile(r"<b(?:\s+[^>]*)?/?>", re.IGNORECASE),
    "small_caps": re.compile(r"<sc(?:\s+[^>]*)?/?>", re.IGNORECASE),
}
_DOT_LEADERS_PATTERN = re.compile(r"\.(?: *\.){2,}")
_ALIGNED_FIELDS_PATTERN = re.compile(r"\S(?:.*?\S)? {3,}\S(?:.*\S)?$")
_MULTIPART_NAME_PATTERN = re.compile(r"[LMR]\.[^.]+$")
_ILLUSTRATION_OR_ORNAMENT_PATTERN = re.compile(
    r"\[(?:illustration|decoration)\b[^\]]*\]", re.IGNORECASE
)
_UNCERTAINTY_NOTE_PATTERN = re.compile(r"\[\*\*")


def extract_page_features(page: ParsedF2Page) -> PageFeatures:
    """Extract deterministic PGDP rank version 1 evidence from one parsed F2 page."""

    block_lines = tuple(_nonblank_lines((block,)) for block in page.special_blocks)
    aligned_row_counts = tuple(
        sum(_is_aligned_fields_line(line) for line in lines) for lines in block_lines
    )
    return PageFeatures(
        special_format=page.has_special_format,
        italic_tags=len(_OPENING_TAG_PATTERNS["italic"].findall(page.feature_text)),
        bold_tags=len(_OPENING_TAG_PATTERNS["bold"].findall(page.feature_text)),
        small_caps_tags=len(_OPENING_TAG_PATTERNS["small_caps"].findall(page.feature_text)),
        dot_leaders=any(
            _DOT_LEADERS_PATTERN.search(body) is not None for body in page.special_blocks
        ),
        aligned_fields=any(count >= 2 for count in aligned_row_counts),
        table_like=any(count >= 3 for count in aligned_row_counts),
        poetry_like=any(_is_poetry_like(lines, page.source_text) for lines in block_lines),
        quotation_like=_is_quotation_like(page.special_blocks),
        multipart_name=_MULTIPART_NAME_PATTERN.search(page.name) is not None,
        illustration_or_ornament=_ILLUSTRATION_OR_ORNAMENT_PATTERN.search(page.source_text)
        is not None,
        uncertainty_note=_UNCERTAINTY_NOTE_PATTERN.search(page.source_text) is not None,
    )


def score_page(features: PageFeatures) -> PageScore:
    """Score one page's PGDP rank version 1 evidence."""

    components = {
        "special_format": 5 * features.special_format,
        "italic_tags": min(features.italic_tags, 10),
        "bold_tags": min(features.bold_tags, 10),
        "small_caps_tags": min(features.small_caps_tags, 10),
        "leaders_or_fields": 8 * (features.dot_leaders or features.aligned_fields),
        "table_like": 8 * features.table_like,
        "poetry_like": 8 * features.poetry_like,
        "quotation_like": 6 * features.quotation_like,
        "multipart_name": 6 * features.multipart_name,
        "illustration_or_ornament": 4 * features.illustration_or_ornament,
        "uncertainty_note": 2 * features.uncertainty_note,
    }
    return PageScore(total=sum(components.values()), components=components)


def _nonblank_lines(blocks: Sequence[str]) -> tuple[str, ...]:
    return tuple(line for body in blocks for line in body.splitlines() if line.strip())


def _is_aligned_fields_line(line: str) -> bool:
    return _ALIGNED_FIELDS_PATTERN.fullmatch(line.strip()) is not None


def _is_poetry_like(body_lines: Sequence[str], source_text: str) -> bool:
    if len(body_lines) < 4:
        return False
    page_lengths = tuple(len(line.strip()) for line in source_text.splitlines() if line.strip())
    if not page_lengths:
        return False
    body_lengths = tuple(len(line.strip()) for line in body_lines)
    body_median = median(body_lengths)
    return (
        body_median < median(page_lengths) * 0.7 and _median_absolute_deviation(body_lengths) != 0
    )


def _median_absolute_deviation(lengths: Sequence[int]) -> float:
    center = median(lengths)
    return float(median(tuple(abs(length - center) for length in lengths)))


def _is_quotation_like(blocks: Sequence[str]) -> bool:
    return any(_has_quoted_run(tuple(body.splitlines())) for body in blocks) or any(
        _has_common_indent(body) for body in blocks
    )


def _has_quoted_run(lines: Sequence[str]) -> bool:
    consecutive_lines = 0
    for line in lines:
        if _is_quoted_line(line):
            consecutive_lines += 1
            if consecutive_lines >= 4:
                return True
        else:
            consecutive_lines = 0
    return False


def _is_quoted_line(line: str) -> bool:
    stripped = line.strip()
    return (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("\u201c") and stripped.endswith("\u201d")
    )


def _has_common_indent(body: str) -> bool:
    lines = tuple(line for line in body.splitlines() if line.strip())
    if len(lines) < 2:
        return False
    first_indent = _leading_indent(lines[0])
    return bool(first_indent) and all(_leading_indent(line) == first_indent for line in lines[1:])


def _leading_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]
