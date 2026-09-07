from __future__ import annotations

type NaturalPagePart = tuple[int, int, int, str]
type NaturalPageKey = tuple[NaturalPagePart, ...]


def _is_ascii_digit(character: str) -> bool:
    return "0" <= character <= "9"


def _split_natural_parts(name: str) -> tuple[str, ...]:
    if not name:
        return ()

    parts: list[str] = []
    start = 0
    is_digit_run = _is_ascii_digit(name[0])
    for index in range(1, len(name)):
        next_is_digit = _is_ascii_digit(name[index])
        if next_is_digit != is_digit_run:
            parts.append(name[start:index])
            start = index
            is_digit_run = next_is_digit
    parts.append(name[start:])
    return tuple(parts)


def natural_page_key(name: str) -> NaturalPageKey:
    """Return a deterministic natural-order key for a PGDP page name."""

    parts: list[NaturalPagePart] = []
    for part in _split_natural_parts(name):
        if _is_ascii_digit(part[0]):
            parts.append((2, int(part), len(part), ""))
        else:
            parts.append((1, 0, 0, part.casefold()))
    parts.append((0, 0, 0, name))
    return tuple(parts)
