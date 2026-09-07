"""Resolve what face each cut glyph was printed in, from PGDP's own F2 markup.

A per-character inventory that mixes faces is not one inventory. PGDP transcribes small capitals
in mixed case, so `<sc>Lowther Street</sc>` yields the letters `Lowther Street` and a capital `O`
lands in the lowercase `o` bucket carrying a lowercase label. Nothing that compares a label to a
character can see that, because the character is right and only the letterform is wrong. Italic is
the same problem one degree milder.

The evidence is already in the alignment report. Every page carries a `style_run` per `<i>`, `<b>`
and `<sc>` span with offsets into that page's normalized visible text, so nothing here reads the
corpus, re-parses a transcription, or takes a second input. A page's line offsets are rebuilt from
its own source lines, each line separator normalizing to a single newline; that reconstruction was
checked against the tokenizer on all 1,385 pages of the five aligned books and matched every one.

`None` and `"roman"` are different answers. `"roman"` means the transcription marks no style on
that glyph. `None` means no style source could be trusted for it, which is every glyph on the
`recognized` tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pdomain_pgdp_measure.alignment_models import PageAlignment, ProjectAlignment

GLYPH_STYLE_METHODS: dict[str, float | int | str] = {
    "algorithm": "alignment-style-runs/v1",
    "style_source": "pgdp-f2-markup",
}

ROMAN: Final = "roman"
"""The transcription marks no style on this glyph. Not the same as an unknown style."""

STYLE_NAMES: Final = MappingProxyType(
    {
        "i": "italic",
        "b": "bold",
        "sc": "small_caps",
        "g": "gesperrt",
        "f": "f",
        "tb": "thought_break",
    }
)
"""F2 tag names spelled out. `gesperrt` is letter-spaced setting, which PGDP marks as `<g>`."""


@dataclass(frozen=True, slots=True)
class BookStyle:
    """One book's style spans, keyed by page name and source-line ordinal.

    Spans are half-open character ranges relative to the start of that line's own visible text,
    which is the same string the alignment report carries for the line.
    """

    project_id: str
    spans: Mapping[tuple[str, int], tuple[tuple[int, int, str], ...]]
    visible: Mapping[tuple[str, int], str]

    @property
    def styled_line_count(self) -> int:
        return len(self.spans)

    def line_agrees(self, page_name: str, ordinal: int, visible_text: str) -> bool:
        """Whether the line asked about is the one the report carries at that ordinal."""

        return self.visible.get((page_name, ordinal)) == visible_text

    def style_at(self, page_name: str, ordinal: int, *, character_offset: int) -> str:
        """The style covering one character of one line, or `roman` when none does."""

        kinds = sorted(
            {
                STYLE_NAMES.get(kind, kind)
                for start, end, kind in self.spans.get((page_name, ordinal), ())
                if start <= character_offset < end
            }
        )
        return "+".join(kinds) if kinds else ROMAN


def read_alignment_style(project: ProjectAlignment) -> BookStyle:
    """Build one book's style lookup from the alignment report it was aligned into."""

    spans: dict[tuple[str, int], tuple[tuple[int, int, str], ...]] = {}
    visible: dict[tuple[str, int], str] = {}
    for page in project.pages:
        for ordinal, start, line_visible in _line_offsets(page):
            end = start + len(line_visible)
            visible[page.page_name, ordinal] = line_visible
            covering = tuple(
                (
                    max(0, run.normalized_start - start),
                    min(end, run.normalized_end) - start,
                    run.kind,
                )
                for run in page.style_runs
                if run.normalized_start < end and run.normalized_end > start
            )
            if covering:
                spans[page.page_name, ordinal] = covering
    return BookStyle(
        project_id=project.project_id,
        spans=MappingProxyType(spans),
        visible=MappingProxyType(visible),
    )


def word_character_offsets(visible_text: str) -> tuple[int, ...]:
    """Where each whitespace-split word starts in the line's own visible text.

    `str.split()` collapses whitespace runs and drops the edges, so the offsets cannot be found by
    counting separators; they are walked instead. The order matches `visible_text.split()`, which
    is the order the word boxes are reconciled in.
    """

    offsets: list[int] = []
    inside = False
    for index, character in enumerate(visible_text):
        if character.isspace():
            inside = False
        elif not inside:
            offsets.append(index)
            inside = True
    return tuple(offsets)


def styles_for_word(
    book_style: BookStyle | None,
    *,
    page_name: str,
    source_ordinal: int,
    visible_text: str,
    word_ordinal: int,
    positions: Sequence[int],
) -> tuple[str | None, ...]:
    """One style per printing-character position of a reconciled word.

    `positions` are the offsets, inside the word, of the characters that were cut. Returns `None`
    for every position when no trusted style source covers this line, so a caller can tell "no
    style marked" from "no style known".
    """

    if book_style is None or not book_style.line_agrees(page_name, source_ordinal, visible_text):
        return tuple(None for _ in positions)
    starts = word_character_offsets(visible_text)
    if word_ordinal >= len(starts):
        return tuple(None for _ in positions)
    word_start = starts[word_ordinal]
    return tuple(
        book_style.style_at(page_name, source_ordinal, character_offset=word_start + position)
        for position in positions
    )


def _line_offsets(page: PageAlignment) -> tuple[tuple[int, int, str], ...]:
    """Each source line's ordinal, offset in the page's visible text, and its own visible text.

    A line separator is one or two source bytes but always normalizes to a single newline, so its
    contribution to the offset is one character whenever the line has a separator at all.
    """

    offsets: list[tuple[int, int, str]] = []
    position = 0
    for line in page.source_lines:
        offsets.append((line.ordinal, position, line.visible_text))
        position += len(line.visible_text) + (1 if line.separator_text else 0)
    return tuple(offsets)
