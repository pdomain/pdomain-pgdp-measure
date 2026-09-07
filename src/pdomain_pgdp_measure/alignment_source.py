from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pdomain_pgdp_measure.ordering import natural_page_key

if TYPE_CHECKING:
    from collections.abc import Mapping

NormalizationKind = Literal[
    "copy",
    "delete_control",
    "delete_tag",
    "delete_note",
    "normalize_newline",
]
StyleKind = Literal["i", "b", "sc", "f", "g", "tb"]
BlockKind = Literal["local", "continued"]

_CONTROL_PAIRS = {"/*": "*/", "/#": "#/"}
_CONTROL_CLOSERS = frozenset(_CONTROL_PAIRS.values())


@dataclass(frozen=True, slots=True)
class NormalizationOperation:
    """One source range and its normalized visible output."""

    kind: NormalizationKind
    byte_start: int
    byte_end: int
    output: str


@dataclass(frozen=True, slots=True)
class StyleRun:
    """A recognized style body in normalized and source coordinates."""

    kind: StyleKind
    normalized_start: int
    normalized_end: int
    byte_start: int
    byte_end: int


@dataclass(frozen=True, slots=True)
class SourceDiagnostic:
    """A recoverable source-tokenization issue with byte evidence."""

    code: str
    message: str
    page_name: str
    byte_start: int
    byte_end: int


@dataclass(frozen=True, slots=True)
class FormattingSpan:
    """One page-local segment of an F2 formatting block."""

    opening_id: str
    block_id: str
    kind: BlockKind
    page_name: str
    byte_start: int
    byte_end: int
    closed: bool


@dataclass(frozen=True, slots=True)
class SourceLine:
    """One original source line and its visible normalized form."""

    page_name: str
    ordinal: int
    byte_start: int
    byte_end: int
    content_byte_start: int
    content_byte_end: int
    separator_byte_start: int
    separator_byte_end: int
    original_text: str
    visible_text: str
    operation_ordinals: tuple[int, ...]
    formatting_span_ids: tuple[str, ...]
    continued_block_ids: tuple[str, ...]
    matching_eligible: bool
    style_fitting_eligible: bool


@dataclass(frozen=True, slots=True)
class TokenizedSourcePage:
    """Lossless source bytes and their visible normalization view."""

    page_name: str
    source_utf8: bytes
    visible_text: str
    lines: tuple[SourceLine, ...]
    operations: tuple[NormalizationOperation, ...]
    style_runs: tuple[StyleRun, ...]
    diagnostics: tuple[SourceDiagnostic, ...]
    formatting_spans: tuple[FormattingSpan, ...] = ()


@dataclass(frozen=True, slots=True)
class TokenizedF2:
    """Naturally ordered F2 source pages and cross-page control evidence."""

    pages: tuple[TokenizedSourcePage, ...]
    diagnostics: tuple[SourceDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class _Lexeme:
    kind: Literal["tag", "note", "control", "newline", "malformed"]
    character_start: int
    character_end: int
    name: str | None = None
    closing: bool = False
    self_closing: bool = False


@dataclass(frozen=True, slots=True)
class _OpenStyle:
    kind: StyleKind
    byte_start: int
    normalized_start: int


@dataclass(frozen=True, slots=True)
class _ControlParseResult:
    deleted_starts: tuple[int, ...]
    diagnostics: tuple[SourceDiagnostic, ...]
    invalid_byte_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class _SpanDraft:
    page_name: str
    byte_start: int
    byte_end: int


@dataclass(slots=True)
class _OpenFormattingBlock:
    kind: BlockKind
    opening_page_name: str
    opening_byte_start: int
    opening_character_start: int
    segment_byte_start: int
    is_valid: bool = True
    segments: list[_SpanDraft] = field(default_factory=list)

    @property
    def opening_id(self) -> str:
        return f"{self.kind}:{self.opening_page_name}:{self.opening_byte_start}"

    def append_segment(self, *, page_name: str, byte_end: int) -> None:
        self.segments.append(_SpanDraft(page_name, self.segment_byte_start, byte_end))


@dataclass(slots=True)
class _ControlCursor:
    starts: tuple[int, ...]
    index: int = 0

    def take_within(self, character_start: int, character_end: int) -> tuple[int, ...]:
        while self.index < len(self.starts) and self.starts[self.index] < character_start:
            self.index += 1
        first = self.index
        while self.index < len(self.starts) and self.starts[self.index] + 2 <= character_end:
            self.index += 1
        return self.starts[first : self.index]


@dataclass(slots=True)
class _RangeCursor:
    ranges: tuple[tuple[int, int], ...]
    index: int = 0

    def overlaps(self, byte_start: int, byte_end: int) -> bool:
        while self.index < len(self.ranges):
            range_start, range_end = self.ranges[self.index]
            if (range_start == range_end and range_start < byte_start) or (
                range_start != range_end and range_end <= byte_start
            ):
                self.index += 1
            else:
                break
        if self.index == len(self.ranges):
            return False
        range_start, range_end = self.ranges[self.index]
        if range_start == range_end:
            return range_start <= byte_end
        return range_start < byte_end


def tokenize_source_page(page_name: str, source_text: str) -> TokenizedSourcePage:
    """Tokenize one F2 page without changing its source bytes."""

    source_utf8 = source_text.encode("utf-8")
    byte_offsets = _utf8_offsets(source_text)
    control_result = _valid_controls(
        page_name=page_name,
        raw_controls=_raw_controls(source_text),
        byte_offsets=byte_offsets,
    )
    return _tokenize_source_page(
        page_name=page_name,
        source_text=source_text,
        source_utf8=source_utf8,
        byte_offsets=byte_offsets,
        control_result=control_result,
        formatting_spans=(),
    )


def tokenize_f2_pages(source_pages: Mapping[str, str]) -> TokenizedF2:
    """Tokenize naturally ordered F2 pages with cross-page control spans."""

    ordered_page_names = tuple(sorted(source_pages, key=natural_page_key))
    control_results, formatting_spans = _scan_f2_controls(source_pages, ordered_page_names)
    pages = tuple(
        _tokenize_source_page(
            page_name=page_name,
            source_text=source_pages[page_name],
            source_utf8=source_pages[page_name].encode("utf-8"),
            byte_offsets=_utf8_offsets(source_pages[page_name]),
            control_result=control_results[page_name],
            formatting_spans=formatting_spans[page_name],
        )
        for page_name in ordered_page_names
    )
    diagnostics = tuple(diagnostic for page in pages for diagnostic in page.diagnostics)
    return TokenizedF2(pages=pages, diagnostics=diagnostics)


def _tokenize_source_page(
    *,
    page_name: str,
    source_text: str,
    source_utf8: bytes,
    byte_offsets: tuple[int, ...],
    control_result: _ControlParseResult,
    formatting_spans: tuple[FormattingSpan, ...],
) -> TokenizedSourcePage:
    lexemes = _lex(source_text)
    diagnostics: list[SourceDiagnostic] = []
    diagnostics.extend(control_result.diagnostics)
    operations: list[NormalizationOperation] = []
    style_runs: list[StyleRun] = []
    matching_ineligible_ranges = list(control_result.invalid_byte_ranges)
    style_ineligible_ranges = list(matching_ineligible_ranges)
    control_cursor = _ControlCursor(control_result.deleted_starts)
    open_styles: list[_OpenStyle] = []
    visible_length = 0
    position = 0

    for lexeme in lexemes:
        enclosed_control_starts = control_cursor.take_within(
            lexeme.character_start, lexeme.character_end
        )
        if position < lexeme.character_start:
            output = source_text[position : lexeme.character_start]
            operation = _copy_operation(byte_offsets, position, lexeme.character_start, output)
            operations.append(operation)
            visible_length += len(output)

        byte_start = byte_offsets[lexeme.character_start]
        byte_end = byte_offsets[lexeme.character_end]
        if lexeme.kind == "newline":
            operations.append(
                NormalizationOperation("normalize_newline", byte_start, byte_end, "\n")
            )
            visible_length += 1
        elif lexeme.kind == "note":
            visible_length += _append_fragmented_operation(
                source_text=source_text,
                byte_offsets=byte_offsets,
                lexeme=lexeme,
                outer_kind="delete_note",
                control_starts=enclosed_control_starts,
                operations=operations,
            )
        elif lexeme.kind == "control":
            if enclosed_control_starts:
                operations.append(
                    NormalizationOperation("delete_control", byte_start, byte_end, "")
                )
            else:
                output = source_text[lexeme.character_start : lexeme.character_end]
                operations.append(
                    _copy_operation(
                        byte_offsets, lexeme.character_start, lexeme.character_end, output
                    )
                )
                visible_length += len(output)
        elif lexeme.kind == "tag":
            tag_name = _style_kind(lexeme.name)
            if tag_name is not None:
                visible_length += _append_fragmented_operation(
                    source_text=source_text,
                    byte_offsets=byte_offsets,
                    lexeme=lexeme,
                    outer_kind="delete_tag",
                    control_starts=enclosed_control_starts,
                    operations=operations,
                )
                if tag_name == "tb":
                    style_runs.append(
                        StyleRun("tb", visible_length, visible_length, byte_end, byte_end)
                    )
                elif lexeme.closing:
                    _close_style(
                        page_name=page_name,
                        tag_name=tag_name,
                        byte_start=byte_start,
                        byte_end=byte_end,
                        open_styles=open_styles,
                        style_runs=style_runs,
                        diagnostics=diagnostics,
                        style_ineligible_ranges=style_ineligible_ranges,
                        normalized_end=visible_length,
                    )
                elif not lexeme.self_closing:
                    open_styles.append(_OpenStyle(tag_name, byte_end, visible_length))
            else:
                visible_length += _append_fragmented_operation(
                    source_text=source_text,
                    byte_offsets=byte_offsets,
                    lexeme=lexeme,
                    outer_kind="copy",
                    control_starts=enclosed_control_starts,
                    operations=operations,
                )
                diagnostic = _diagnostic(
                    "unknown_tag",
                    f"Unknown F2 tag '{lexeme.name}'.",
                    page_name,
                    byte_start,
                    byte_end,
                )
                diagnostics.append(diagnostic)
                style_ineligible_ranges.append((byte_start, byte_end))
        else:
            visible_length += _append_fragmented_operation(
                source_text=source_text,
                byte_offsets=byte_offsets,
                lexeme=lexeme,
                outer_kind="copy",
                control_starts=enclosed_control_starts,
                operations=operations,
            )
            diagnostic = _diagnostic(
                "malformed_markup",
                "Malformed F2 markup remains visible.",
                page_name,
                byte_start,
                byte_end,
            )
            diagnostics.append(diagnostic)
            style_ineligible_ranges.append((byte_start, byte_end))
        position = lexeme.character_end

    if position < len(source_text):
        output = source_text[position:]
        operations.append(_copy_operation(byte_offsets, position, len(source_text), output))
        visible_length += len(output)

    for open_style in open_styles:
        diagnostic = _diagnostic(
            "unmatched_style_tag",
            f"Unmatched opening F2 tag '{open_style.kind}'.",
            page_name,
            open_style.byte_start,
            open_style.byte_start,
        )
        diagnostics.append(diagnostic)
        style_ineligible_ranges.append((open_style.byte_start, open_style.byte_start))

    visible_text = "".join(operation.output for operation in operations)
    lines = _source_lines(
        page_name=page_name,
        source_text=source_text,
        byte_offsets=byte_offsets,
        operations=operations,
        matching_ineligible_ranges=matching_ineligible_ranges,
        style_ineligible_ranges=style_ineligible_ranges,
        formatting_spans=formatting_spans,
    )
    return TokenizedSourcePage(
        page_name=page_name,
        source_utf8=source_utf8,
        visible_text=visible_text,
        lines=lines,
        operations=tuple(operations),
        style_runs=tuple(style_runs),
        diagnostics=tuple(diagnostics),
        formatting_spans=formatting_spans,
    )


def _utf8_offsets(source_text: str) -> tuple[int, ...]:
    offsets = [0]
    for character in source_text:
        offsets.append(offsets[-1] + len(character.encode("utf-8")))
    return tuple(offsets)


def _style_kind(name: str | None) -> StyleKind | None:
    match name:
        case "i":
            return "i"
        case "b":
            return "b"
        case "sc":
            return "sc"
        case "f":
            return "f"
        case "g":
            return "g"
        case "tb":
            return "tb"
        case _:
            return None


def _lex(source_text: str) -> tuple[_Lexeme, ...]:
    lexemes: list[_Lexeme] = []
    position = 0
    line_end = _line_end(source_text, position)
    while position < len(source_text):
        newline_end = _newline_end(source_text, position)
        if newline_end is not None:
            lexemes.append(_Lexeme("newline", position, newline_end))
            position = newline_end
            line_end = _line_end(source_text, position)
            continue
        control = source_text[position : position + 2]
        if control in _CONTROL_PAIRS or control in _CONTROL_CLOSERS:
            lexemes.append(_Lexeme("control", position, position + 2, control))
            position += 2
            continue
        if source_text.startswith("[**", position):
            note_end = _proof_note_end(source_text, position, line_end)
            if note_end is None:
                lexemes.append(_Lexeme("malformed", position, line_end))
                position = line_end
            else:
                lexemes.append(_Lexeme("note", position, note_end))
                position = note_end
            continue
        if source_text[position] == "<":
            tag = _tag_lexeme(source_text, position, line_end)
            if tag is not None:
                lexemes.append(tag)
                position = tag.character_end
                continue
        position += 1
    return tuple(lexemes)


def _raw_controls(source_text: str) -> tuple[_Lexeme, ...]:
    controls: list[_Lexeme] = []
    position = 0
    while position < len(source_text):
        control = source_text[position : position + 2]
        if control in _CONTROL_PAIRS or control in _CONTROL_CLOSERS:
            controls.append(_Lexeme("control", position, position + 2, control))
            position += 2
        else:
            position += 1
    return tuple(controls)


def _newline_end(source_text: str, position: int) -> int | None:
    character = source_text[position]
    if character == "\n":
        return position + 1
    if character == "\r":
        return position + 2 if source_text[position + 1 : position + 2] == "\n" else position + 1
    return None


def _proof_note_end(source_text: str, position: int, line_end: int) -> int | None:
    closing = source_text.find("]", position + 3, line_end)
    return None if closing == -1 else closing + 1


def _line_end(source_text: str, position: int) -> int:
    candidates = [
        candidate
        for candidate in (source_text.find("\n", position), source_text.find("\r", position))
        if candidate != -1
    ]
    return min(candidates) if candidates else len(source_text)


def _tag_lexeme(source_text: str, position: int, line_end: int) -> _Lexeme | None:
    closing_bracket = source_text.find(">", position + 1, line_end)
    if closing_bracket == -1:
        remainder = source_text[position + 1 : line_end].lstrip()
        if remainder.startswith("/"):
            remainder = remainder[1:].lstrip()
        if remainder[:1].isalpha() and remainder[:1].isascii():
            return _Lexeme("malformed", position, line_end)
        return None
    body = source_text[position + 1 : closing_bracket].strip()
    closing = body.startswith("/")
    if closing:
        body = body[1:].lstrip()
    if not body or not body[0].isalpha() or not body[0].isascii():
        return _Lexeme("malformed", position, closing_bracket + 1)
    name_end = 1
    while name_end < len(body) and (body[name_end].isalnum() or body[name_end] in "_:-"):
        name_end += 1
    name = body[:name_end]
    attribute_text = body[name_end:]
    self_closing = not closing and attribute_text.rstrip().endswith("/")
    if self_closing:
        attribute_text = attribute_text.rstrip()[:-1]
    if attribute_text and not attribute_text[0].isspace():
        return _Lexeme("malformed", position, closing_bracket + 1)
    return _Lexeme(
        "tag",
        position,
        closing_bracket + 1,
        name.lower(),
        closing,
        self_closing,
    )


def _valid_controls(
    *,
    page_name: str,
    raw_controls: tuple[_Lexeme, ...],
    byte_offsets: tuple[int, ...],
) -> _ControlParseResult:
    deleted: set[int] = set()
    diagnostics: list[SourceDiagnostic] = []
    invalid_byte_ranges: list[tuple[int, int]] = []
    open_control: _Lexeme | None = None
    open_control_is_valid = True
    for control_lexeme in raw_controls:
        if control_lexeme.name is None:
            continue
        control = control_lexeme.name
        if control in _CONTROL_PAIRS:
            if open_control is None:
                open_control = control_lexeme
                open_control_is_valid = True
            else:
                open_control_is_valid = False
                diagnostics.append(
                    _diagnostic(
                        "malformed_control",
                        f"Nested or overlapping formatting control '{control}'.",
                        page_name,
                        byte_offsets[control_lexeme.character_start],
                        byte_offsets[control_lexeme.character_end],
                    )
                )
        elif open_control is not None and _CONTROL_PAIRS[open_control.name or ""] == control:
            if open_control_is_valid:
                deleted.add(open_control.character_start)
                deleted.add(control_lexeme.character_start)
            else:
                invalid_byte_ranges.append(
                    (
                        byte_offsets[open_control.character_start],
                        byte_offsets[control_lexeme.character_end],
                    )
                )
            open_control = None
        else:
            if open_control is not None:
                open_control_is_valid = False
            else:
                invalid_byte_ranges.append(
                    (
                        byte_offsets[control_lexeme.character_start],
                        byte_offsets[control_lexeme.character_end],
                    )
                )
            diagnostics.append(
                _diagnostic(
                    "malformed_control",
                    f"Unmatched formatting control '{control}'.",
                    page_name,
                    byte_offsets[control_lexeme.character_start],
                    byte_offsets[control_lexeme.character_end],
                )
            )
    if open_control is not None:
        invalid_byte_ranges.append((byte_offsets[open_control.character_start], byte_offsets[-1]))
        diagnostics.append(
            _diagnostic(
                "malformed_control",
                f"Unmatched formatting control '{open_control.name}'.",
                page_name,
                byte_offsets[open_control.character_start],
                byte_offsets[open_control.character_end],
            )
        )
    return _ControlParseResult(
        deleted_starts=tuple(sorted(deleted)),
        diagnostics=tuple(diagnostics),
        invalid_byte_ranges=tuple(invalid_byte_ranges),
    )


def _scan_f2_controls(
    source_pages: Mapping[str, str], ordered_page_names: tuple[str, ...]
) -> tuple[dict[str, _ControlParseResult], dict[str, tuple[FormattingSpan, ...]]]:
    deleted_starts: dict[str, set[int]] = {page_name: set() for page_name in ordered_page_names}
    diagnostics: dict[str, list[SourceDiagnostic]] = {
        page_name: [] for page_name in ordered_page_names
    }
    invalid_ranges: dict[str, list[tuple[int, int]]] = {
        page_name: [] for page_name in ordered_page_names
    }
    page_spans: dict[str, list[FormattingSpan]] = {
        page_name: [] for page_name in ordered_page_names
    }
    active_block: _OpenFormattingBlock | None = None

    for page_name in ordered_page_names:
        source_text = source_pages[page_name]
        byte_offsets = _utf8_offsets(source_text)
        for control_lexeme in _raw_controls(source_text):
            control = control_lexeme.name
            if control is None:
                continue
            byte_start = byte_offsets[control_lexeme.character_start]
            byte_end = byte_offsets[control_lexeme.character_end]
            if control in _CONTROL_PAIRS:
                active_block = _open_f2_control(
                    control=control,
                    page_name=page_name,
                    byte_start=byte_start,
                    byte_end=byte_end,
                    character_start=control_lexeme.character_start,
                    active_block=active_block,
                    diagnostics=diagnostics,
                )
            else:
                active_block = _close_f2_control(
                    control=control,
                    page_name=page_name,
                    byte_start=byte_start,
                    byte_end=byte_end,
                    character_start=control_lexeme.character_start,
                    active_block=active_block,
                    source_pages=source_pages,
                    ordered_page_names=ordered_page_names,
                    deleted_starts=deleted_starts,
                    diagnostics=diagnostics,
                    invalid_ranges=invalid_ranges,
                    page_spans=page_spans,
                )

        if active_block is not None and active_block.kind == "local":
            active_block.append_segment(
                page_name=page_name, byte_end=len(source_text.encode("utf-8"))
            )
            _finish_invalid_block(
                active_block=active_block,
                end_page_name=page_name,
                end_byte=len(source_text.encode("utf-8")),
                source_pages=source_pages,
                ordered_page_names=ordered_page_names,
                diagnostics=diagnostics,
                invalid_ranges=invalid_ranges,
                page_spans=page_spans,
                reason="Unmatched formatting control opener '/*'.",
            )
            active_block = None
        elif active_block is not None:
            active_block.append_segment(
                page_name=page_name, byte_end=len(source_text.encode("utf-8"))
            )
            active_block.segment_byte_start = 0

    if active_block is not None:
        end_page_name = ordered_page_names[-1]
        _finish_invalid_block(
            active_block=active_block,
            end_page_name=end_page_name,
            end_byte=len(source_pages[end_page_name].encode("utf-8")),
            source_pages=source_pages,
            ordered_page_names=ordered_page_names,
            diagnostics=diagnostics,
            invalid_ranges=invalid_ranges,
            page_spans=page_spans,
            reason="Unmatched formatting control opener '/#'.",
        )

    results = {
        page_name: _ControlParseResult(
            deleted_starts=tuple(sorted(deleted_starts[page_name])),
            diagnostics=tuple(diagnostics[page_name]),
            invalid_byte_ranges=tuple(invalid_ranges[page_name]),
        )
        for page_name in ordered_page_names
    }
    spans: dict[str, tuple[FormattingSpan, ...]] = {
        page_name: tuple(page_spans[page_name]) for page_name in ordered_page_names
    }
    return results, spans


def _open_f2_control(
    *,
    control: str,
    page_name: str,
    byte_start: int,
    byte_end: int,
    character_start: int,
    active_block: _OpenFormattingBlock | None,
    diagnostics: dict[str, list[SourceDiagnostic]],
) -> _OpenFormattingBlock:
    if active_block is None:
        kind: BlockKind = "local" if control == "/*" else "continued"
        return _OpenFormattingBlock(kind, page_name, byte_start, character_start, byte_end)

    active_block.is_valid = False
    diagnostics[page_name].append(
        _diagnostic(
            "malformed_control",
            f"Nested or overlapping formatting control '{control}'.",
            page_name,
            byte_start,
            byte_end,
        )
    )
    return active_block


def _close_f2_control(
    *,
    control: str,
    page_name: str,
    byte_start: int,
    byte_end: int,
    character_start: int,
    active_block: _OpenFormattingBlock | None,
    source_pages: Mapping[str, str],
    ordered_page_names: tuple[str, ...],
    deleted_starts: dict[str, set[int]],
    diagnostics: dict[str, list[SourceDiagnostic]],
    invalid_ranges: dict[str, list[tuple[int, int]]],
    page_spans: dict[str, list[FormattingSpan]],
) -> _OpenFormattingBlock | None:
    if active_block is None:
        kind: BlockKind = "local" if control == "*/" else "continued"
        opening_id = f"{kind}:{page_name}:{byte_start}"
        page_spans[page_name].append(
            FormattingSpan(opening_id, opening_id, kind, page_name, byte_start, byte_end, False)
        )
        invalid_ranges[page_name].append((byte_start, byte_end))
        diagnostics[page_name].append(
            _diagnostic(
                "malformed_control",
                f"Unmatched formatting control '{control}'.",
                page_name,
                byte_start,
                byte_end,
            )
        )
        return None

    expected_control = "*/" if active_block.kind == "local" else "#/"
    if control != expected_control:
        active_block.is_valid = False
        diagnostics[page_name].append(
            _diagnostic(
                "malformed_control",
                f"Overlapping formatting control '{control}'.",
                page_name,
                byte_start,
                byte_end,
            )
        )
        return active_block

    active_block.append_segment(page_name=page_name, byte_end=byte_start)
    if active_block.is_valid:
        deleted_starts[active_block.opening_page_name].add(active_block.opening_character_start)
        deleted_starts[page_name].add(character_start)
        block_id = f"{active_block.opening_id}:{page_name}:{byte_end}"
        _append_formatting_spans(
            active_block=active_block,
            block_id=block_id,
            closed=True,
            page_spans=page_spans,
        )
    else:
        _finish_invalid_block(
            active_block=active_block,
            end_page_name=page_name,
            end_byte=byte_end,
            source_pages=source_pages,
            ordered_page_names=ordered_page_names,
            diagnostics=diagnostics,
            invalid_ranges=invalid_ranges,
            page_spans=page_spans,
            reason=None,
        )
    return None


def _finish_invalid_block(
    *,
    active_block: _OpenFormattingBlock,
    end_page_name: str,
    end_byte: int,
    source_pages: Mapping[str, str],
    ordered_page_names: tuple[str, ...],
    diagnostics: dict[str, list[SourceDiagnostic]],
    invalid_ranges: dict[str, list[tuple[int, int]]],
    page_spans: dict[str, list[FormattingSpan]],
    reason: str | None,
) -> None:
    _append_formatting_spans(
        active_block=active_block,
        block_id=active_block.opening_id,
        closed=False,
        page_spans=page_spans,
    )
    opening_index = ordered_page_names.index(active_block.opening_page_name)
    ending_index = ordered_page_names.index(end_page_name)
    for page_name in ordered_page_names[opening_index : ending_index + 1]:
        page_length = len(source_pages[page_name].encode("utf-8"))
        range_start = (
            active_block.opening_byte_start if page_name == active_block.opening_page_name else 0
        )
        range_end = end_byte if page_name == end_page_name else page_length
        invalid_ranges[page_name].append((range_start, range_end))
    if reason is not None:
        diagnostics[active_block.opening_page_name].append(
            _diagnostic(
                "malformed_control",
                reason,
                active_block.opening_page_name,
                active_block.opening_byte_start,
                active_block.opening_byte_start + 2,
            )
        )


def _append_formatting_spans(
    *,
    active_block: _OpenFormattingBlock,
    block_id: str,
    closed: bool,
    page_spans: dict[str, list[FormattingSpan]],
) -> None:
    for segment in active_block.segments:
        page_spans[segment.page_name].append(
            FormattingSpan(
                active_block.opening_id,
                block_id,
                active_block.kind,
                segment.page_name,
                segment.byte_start,
                segment.byte_end,
                closed,
            )
        )


def _copy_operation(
    byte_offsets: tuple[int, ...], character_start: int, character_end: int, output: str
) -> NormalizationOperation:
    return NormalizationOperation(
        "copy", byte_offsets[character_start], byte_offsets[character_end], output
    )


def _append_fragmented_operation(
    *,
    source_text: str,
    byte_offsets: tuple[int, ...],
    lexeme: _Lexeme,
    outer_kind: NormalizationKind,
    control_starts: tuple[int, ...],
    operations: list[NormalizationOperation],
) -> int:
    visible_length = 0
    position = lexeme.character_start
    for control_start in control_starts:
        visible_length += _append_outer_operation(
            source_text=source_text,
            byte_offsets=byte_offsets,
            character_start=position,
            character_end=control_start,
            kind=outer_kind,
            operations=operations,
        )
        operations.append(
            NormalizationOperation(
                "delete_control",
                byte_offsets[control_start],
                byte_offsets[control_start + 2],
                "",
            )
        )
        position = control_start + 2
    visible_length += _append_outer_operation(
        source_text=source_text,
        byte_offsets=byte_offsets,
        character_start=position,
        character_end=lexeme.character_end,
        kind=outer_kind,
        operations=operations,
    )
    return visible_length


def _append_outer_operation(
    *,
    source_text: str,
    byte_offsets: tuple[int, ...],
    character_start: int,
    character_end: int,
    kind: NormalizationKind,
    operations: list[NormalizationOperation],
) -> int:
    if character_start == character_end:
        return 0
    output = source_text[character_start:character_end] if kind == "copy" else ""
    operations.append(
        NormalizationOperation(
            kind,
            byte_offsets[character_start],
            byte_offsets[character_end],
            output,
        )
    )
    return len(output)


def _close_style(
    *,
    page_name: str,
    tag_name: StyleKind,
    byte_start: int,
    byte_end: int,
    open_styles: list[_OpenStyle],
    style_runs: list[StyleRun],
    diagnostics: list[SourceDiagnostic],
    style_ineligible_ranges: list[tuple[int, int]],
    normalized_end: int,
) -> None:
    if open_styles and open_styles[-1].kind == tag_name:
        open_style = open_styles.pop()
        style_runs.append(
            StyleRun(
                tag_name,
                open_style.normalized_start,
                normalized_end,
                open_style.byte_start,
                byte_start,
            )
        )
        return
    diagnostic = _diagnostic(
        "unmatched_style_tag",
        f"Unmatched closing F2 tag '{tag_name}'.",
        page_name,
        byte_start,
        byte_end,
    )
    diagnostics.append(diagnostic)
    style_ineligible_ranges.append((byte_start, byte_end))


def _source_lines(
    *,
    page_name: str,
    source_text: str,
    byte_offsets: tuple[int, ...],
    operations: list[NormalizationOperation],
    matching_ineligible_ranges: list[tuple[int, int]],
    style_ineligible_ranges: list[tuple[int, int]],
    formatting_spans: tuple[FormattingSpan, ...],
) -> tuple[SourceLine, ...]:
    lines: list[SourceLine] = []
    matching_ranges = _RangeCursor(tuple(sorted(matching_ineligible_ranges)))
    style_ranges = _RangeCursor(tuple(sorted(style_ineligible_ranges)))
    content_start = 0
    ordinal = 0
    operation_start = 0
    position = 0
    while position < len(source_text):
        newline_end = _newline_end(source_text, position)
        if newline_end is None:
            position += 1
            continue
        line, operation_start = _make_line(
            page_name=page_name,
            ordinal=ordinal,
            source_text=source_text,
            byte_offsets=byte_offsets,
            content_start=content_start,
            content_end=position,
            separator_end=newline_end,
            operations=operations,
            operation_start=operation_start,
            matching_ranges=matching_ranges,
            style_ranges=style_ranges,
            formatting_spans=formatting_spans,
        )
        lines.append(line)
        ordinal += 1
        content_start = newline_end
        position = newline_end
    final_line, _ = _make_line(
        page_name=page_name,
        ordinal=ordinal,
        source_text=source_text,
        byte_offsets=byte_offsets,
        content_start=content_start,
        content_end=len(source_text),
        separator_end=len(source_text),
        operations=operations,
        operation_start=operation_start,
        matching_ranges=matching_ranges,
        style_ranges=style_ranges,
        formatting_spans=formatting_spans,
    )
    lines.append(final_line)
    return tuple(lines)


def _make_line(
    *,
    page_name: str,
    ordinal: int,
    source_text: str,
    byte_offsets: tuple[int, ...],
    content_start: int,
    content_end: int,
    separator_end: int,
    operations: list[NormalizationOperation],
    operation_start: int,
    matching_ranges: _RangeCursor,
    style_ranges: _RangeCursor,
    formatting_spans: tuple[FormattingSpan, ...],
) -> tuple[SourceLine, int]:
    byte_start = byte_offsets[content_start]
    content_byte_end = byte_offsets[content_end]
    byte_end = byte_offsets[separator_end]
    operation_end = operation_start
    while operation_end < len(operations) and operations[operation_end].byte_end <= byte_end:
        operation_end += 1
    operation_ordinals = tuple(range(operation_start, operation_end))
    visible_text = "".join(
        operations[index].output
        for index in operation_ordinals
        if operations[index].byte_end <= content_byte_end
    )
    matching_eligible = not matching_ranges.overlaps(byte_start, content_byte_end)
    style_fitting_eligible = not style_ranges.overlaps(byte_start, content_byte_end)
    line_span_ids = tuple(
        span.block_id
        for span in formatting_spans
        if _ranges_overlap(byte_start, byte_end, span.byte_start, span.byte_end)
    )
    continued_block_ids = tuple(
        span.opening_id
        for span in formatting_spans
        if span.kind == "continued"
        and _ranges_overlap(byte_start, byte_end, span.byte_start, span.byte_end)
    )
    return (
        SourceLine(
            page_name=page_name,
            ordinal=ordinal,
            byte_start=byte_start,
            byte_end=byte_end,
            content_byte_start=byte_start,
            content_byte_end=content_byte_end,
            separator_byte_start=content_byte_end,
            separator_byte_end=byte_end,
            original_text=source_text[content_start:content_end],
            visible_text=visible_text,
            operation_ordinals=operation_ordinals,
            formatting_span_ids=line_span_ids,
            continued_block_ids=continued_block_ids,
            matching_eligible=matching_eligible,
            style_fitting_eligible=style_fitting_eligible,
        ),
        operation_end,
    )


def _diagnostic(
    code: str, message: str, page_name: str, byte_start: int, byte_end: int
) -> SourceDiagnostic:
    return SourceDiagnostic(code, message, page_name, byte_start, byte_end)


def _ranges_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    if left_start == left_end:
        return right_start <= left_start <= right_end
    if right_start == right_end:
        return left_start <= right_start <= left_end
    return left_start < right_end and right_start < left_end
