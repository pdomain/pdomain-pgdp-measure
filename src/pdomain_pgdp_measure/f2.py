from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from pdomain_pgdp_measure.models import Diagnostic
from pdomain_pgdp_measure.ordering import natural_page_key

if TYPE_CHECKING:
    from collections.abc import Mapping


class F2LoadError(ValueError):
    """Raised when an F2 file is unreadable or has an invalid JSON shape."""


class _BlockKind(StrEnum):
    LOCAL = "local"
    CONTINUED = "continued"


class _Control(StrEnum):
    LOCAL_OPEN = "/*"
    LOCAL_CLOSE = "*/"
    CONTINUED_OPEN = "/#"
    CONTINUED_CLOSE = "#/"


_OPENING_KINDS: dict[_Control, _BlockKind] = {
    _Control.LOCAL_OPEN: _BlockKind.LOCAL,
    _Control.CONTINUED_OPEN: _BlockKind.CONTINUED,
}
_CLOSING_KINDS: dict[_Control, _BlockKind] = {
    _Control.LOCAL_CLOSE: _BlockKind.LOCAL,
    _Control.CONTINUED_CLOSE: _BlockKind.CONTINUED,
}
_F2_PAGES_ADAPTER = TypeAdapter(dict[str, str])


@dataclass(frozen=True, slots=True)
class ParsedF2Page:
    """Lossless source text and valid formatting content for one F2 page."""

    name: str
    source_text: str
    feature_text: str
    special_blocks: tuple[str, ...]
    has_special_format: bool


@dataclass(frozen=True, slots=True)
class F2ParseResult:
    """Naturally ordered F2 pages and recoverable parse diagnostics."""

    pages: tuple[ParsedF2Page, ...]
    diagnostics: tuple[Diagnostic, ...]


@dataclass(slots=True)
class _OpenBlock:
    kind: _BlockKind
    opened_page_name: str
    is_valid: bool = True
    bodies: dict[str, list[str]] = field(default_factory=dict)

    def append(self, *, page_name: str, text: str) -> None:
        self.bodies.setdefault(page_name, []).append(text)


def load_f2(path: str | Path) -> F2ParseResult:
    """Load and parse one UTF-8 F2 JSON file."""

    source = Path(path)
    try:
        payload = _F2_PAGES_ADAPTER.validate_json(source.read_text(encoding="utf-8"), strict=True)
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise F2LoadError(f"Could not load F2 JSON from {source}.") from error

    return parse_f2_pages(payload)


def parse_f2_pages(source_pages: Mapping[str, str]) -> F2ParseResult:
    """Parse valid local and continued formatting blocks without changing source text."""

    page_blocks: dict[str, list[str]] = {name: [] for name in source_pages}
    feature_text_parts: dict[str, list[str]] = {name: [] for name in source_pages}
    diagnostics: list[Diagnostic] = []
    active_block: _OpenBlock | None = None

    for page_name in sorted(source_pages, key=natural_page_key):
        source_text = source_pages[page_name]
        active_block = _scan_page(
            source_text=source_text,
            page_name=page_name,
            active_block=active_block,
            page_blocks=page_blocks,
            feature_text_parts=feature_text_parts,
            diagnostics=diagnostics,
        )
        if active_block is not None and active_block.kind is _BlockKind.LOCAL:
            diagnostics.append(_unmatched_opener(active_block))
            active_block = None

    if active_block is not None:
        diagnostics.append(_unmatched_opener(active_block))

    pages = tuple(
        ParsedF2Page(
            name=page_name,
            source_text=source_pages[page_name],
            feature_text="".join(feature_text_parts[page_name]),
            special_blocks=tuple(page_blocks[page_name]),
            has_special_format=bool(page_blocks[page_name]),
        )
        for page_name in sorted(source_pages, key=natural_page_key)
    )
    return F2ParseResult(pages=pages, diagnostics=tuple(diagnostics))


def _scan_page(
    *,
    source_text: str,
    page_name: str,
    active_block: _OpenBlock | None,
    page_blocks: dict[str, list[str]],
    feature_text_parts: dict[str, list[str]],
    diagnostics: list[Diagnostic],
) -> _OpenBlock | None:
    position = 0
    while next_control := _find_next_control(source_text, start=position):
        control_position, control = next_control
        if active_block is not None:
            active_block.append(page_name=page_name, text=source_text[position:control_position])
        else:
            feature_text_parts[page_name].append(source_text[position:control_position])

        if control in _OPENING_KINDS:
            active_block = _handle_opener(
                control=control,
                page_name=page_name,
                active_block=active_block,
                diagnostics=diagnostics,
            )
        else:
            active_block = _handle_closer(
                control=control,
                page_name=page_name,
                active_block=active_block,
                page_blocks=page_blocks,
                feature_text_parts=feature_text_parts,
                diagnostics=diagnostics,
            )
        position = control_position + len(control)

    if active_block is not None:
        active_block.append(page_name=page_name, text=source_text[position:])
    else:
        feature_text_parts[page_name].append(source_text[position:])
    return active_block


def _find_next_control(source_text: str, *, start: int) -> tuple[int, _Control] | None:
    positions = (
        (source_text.find(_Control.LOCAL_OPEN, start), _Control.LOCAL_OPEN),
        (source_text.find(_Control.LOCAL_CLOSE, start), _Control.LOCAL_CLOSE),
        (source_text.find(_Control.CONTINUED_OPEN, start), _Control.CONTINUED_OPEN),
        (source_text.find(_Control.CONTINUED_CLOSE, start), _Control.CONTINUED_CLOSE),
    )
    found_positions = tuple(position for position in positions if position[0] >= 0)
    if not found_positions:
        return None
    return min(found_positions, key=lambda position: position[0])


def _handle_opener(
    *,
    control: _Control,
    page_name: str,
    active_block: _OpenBlock | None,
    diagnostics: list[Diagnostic],
) -> _OpenBlock:
    kind = _OPENING_KINDS[control]
    if active_block is None:
        return _OpenBlock(kind=kind, opened_page_name=page_name)

    code = "nested_format_control" if active_block.kind is kind else "overlapping_format_control"
    diagnostics.append(
        Diagnostic(
            code=code,
            message=f"{code.replace('_', ' ').capitalize()} '{control.value}'.",
            page_name=page_name,
        )
    )
    active_block.is_valid = False
    return active_block


def _handle_closer(
    *,
    control: _Control,
    page_name: str,
    active_block: _OpenBlock | None,
    page_blocks: dict[str, list[str]],
    feature_text_parts: dict[str, list[str]],
    diagnostics: list[Diagnostic],
) -> _OpenBlock | None:
    kind = _CLOSING_KINDS[control]
    if active_block is None:
        diagnostics.append(
            Diagnostic(
                code="unmatched_format_closer",
                message=f"Unmatched formatting control closer '{control.value}'.",
                page_name=page_name,
            )
        )
        return None
    if active_block.kind is not kind:
        diagnostics.append(
            Diagnostic(
                code="overlapping_format_control",
                message=f"Overlapping formatting control closer '{control.value}'.",
                page_name=page_name,
            )
        )
        active_block.is_valid = False
        return active_block

    if active_block.is_valid:
        for block_page_name, parts in active_block.bodies.items():
            body = "".join(parts)
            page_blocks[block_page_name].append(body)
            feature_text_parts[block_page_name].append(body)
    return None


def _unmatched_opener(active_block: _OpenBlock) -> Diagnostic:
    control = (
        _Control.LOCAL_OPEN if active_block.kind is _BlockKind.LOCAL else _Control.CONTINUED_OPEN
    )
    return Diagnostic(
        code="unmatched_format_opener",
        message=f"Unmatched formatting control opener '{control.value}'.",
        page_name=active_block.opened_page_name,
    )
