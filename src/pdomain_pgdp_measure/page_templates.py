"""Fit per-book page templates and classify pages against them.

A letterpress book is set once, so its text block holds the same position across
the volume. That makes a band's distance from the top of the page evidence for
what the band is: a running head sits where every page's first band sits, while
a chapter opening starts far below it.

Two classes named in the design are absent here on purpose. Front matter has no
measured discriminator, and one illustration is not enough to separate a plate
from a table with tall bands. Both fall to `unknown`, which suppresses nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pdomain_pgdp_measure.profile_models import PageMeasurement

PageClass = Literal[
    "normal_recto",
    "normal_verso",
    "chapter_opening",
    "unknown",
]

PAGE_TEMPLATE_METHOD = "first-band-templates/v3"


def page_template_methods() -> dict[str, float | int | str | bool]:
    """Return the versioned constants page classification is fixed to."""

    return {
        "algorithm": PAGE_TEMPLATE_METHOD,
        "head_mad_maximum_px": _HEAD_MAD_MAXIMUM_PX,
        "head_window_minimum_px": _HEAD_WINDOW_MINIMUM_PX,
        "head_window_mad_factor": _HEAD_WINDOW_MAD_FACTOR,
        "chapter_sink_minimum_px": _CHAPTER_SINK_MINIMUM_PX,
        "side_minimum_share": _SIDE_MINIMUM_SHARE,
    }


# A head that wanders is not a head that is missing. The ceiling is set by the window it produces:
# at 25 the widest window is 3 x 25 = 75px, which stays below the 76px at which genuine top-of-page
# text begins, so a qualifying book can never call body text a running head.
_HEAD_MAD_MAXIMUM_PX = 25
_CHAPTER_SINK_MINIMUM_PX = 150
# How steady a book's first band must be is a different question from how far a
# single page may sit from it. Running heads were measured 0 to 4px off their
# book's median, while the nearest page whose first band is genuine text sat
# 76px below. The window is flat between 4 and 20px on all three steady books,
# so 8 is double the observed spread and an order of magnitude clear of text.
_HEAD_WINDOW_MINIMUM_PX = 8
_HEAD_WINDOW_MAD_FACTOR = 3
# A binding shift divides a book roughly in half. A group far below that is an oddity, such as one
# indented page, and letting it define a side would give it a template of its own.
_SIDE_MINIMUM_SHARE = 0.2

_TRAILING_DIGITS = re.compile(r"(\d+)(?!.*\d)")


@dataclass(frozen=True, slots=True)
class PageTemplate:
    """One measured page shape a book repeats."""

    page_class: PageClass
    first_band_top_px: int
    first_band_spread_px: int
    text_left_px: int
    text_right_px: int
    band_count: int
    page_count: int
    page_share: float


@dataclass(frozen=True, slots=True)
class BookTemplates:
    """The page shapes fitted for one book, with the evidence for trusting them."""

    first_band_median_px: float | None
    first_band_mad_px: float | None
    head_window_px: float | None
    templates: tuple[PageTemplate, ...]
    classified_share: float

    @property
    def has_type_page(self) -> bool:
        """Whether the book's first band is steady enough to classify against."""

        return bool(self.templates)


@dataclass(frozen=True, slots=True)
class PageClassification:
    """One page's class and how far it sat from its template."""

    page_name: str
    page_class: PageClass
    template_residual_px: int | None
    furniture_band_ordinals: tuple[int, ...]
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class _PageGeometry:
    page_name: str
    band_tops: tuple[int, ...]
    text_left: int
    text_right: int
    band_count: int
    recto: bool

    @property
    def first_band_top(self) -> int:
        """The top of the topmost ink band, whether or not it is type."""

        return self.band_tops[0]


def fit_book_templates(pages: Sequence[PageMeasurement]) -> BookTemplates:
    """Fit the page shapes one book repeats, or none when it has no type page."""

    geometry = tuple(item for item in map(_page_geometry, pages) if item is not None)
    if not geometry:
        return BookTemplates(None, None, None, (), 0.0)

    tops = [item.first_band_top for item in geometry]
    center = float(median(tops))
    deviation = float(median(abs(top - center) for top in tops))
    if deviation > _HEAD_MAD_MAXIMUM_PX:
        # A book whose first band wanders has no single text block to describe.
        # Widening the window to fit it would classify everything and mean
        # nothing, so the book gets no templates at all.
        return BookTemplates(center, deviation, None, (), 0.0)

    window = max(float(_HEAD_WINDOW_MINIMUM_PX), _HEAD_WINDOW_MAD_FACTOR * deviation)
    normal = [item for item in geometry if abs(item.first_band_top - center) <= window]
    sunk = [item for item in geometry if item.first_band_top - center >= _CHAPTER_SINK_MINIMUM_PX]

    templates: list[PageTemplate] = []
    total = len(geometry)
    recto_side, verso_side = _split_sides(normal)
    groups: tuple[tuple[PageClass, list[_PageGeometry]], ...] = (
        ("normal_recto", recto_side),
        ("normal_verso", verso_side),
        ("chapter_opening", sunk),
    )
    for page_class, group in groups:
        template = _fit_template(page_class, group, total=total)
        if template is not None:
            templates.append(template)

    classified = sum(template.page_count for template in templates)
    return BookTemplates(
        first_band_median_px=center,
        first_band_mad_px=deviation,
        head_window_px=window,
        templates=tuple(templates),
        classified_share=classified / total if total else 0.0,
    )


def classify_pages(
    pages: Sequence[PageMeasurement], templates: BookTemplates
) -> tuple[PageClassification, ...]:
    """Classify each page and name the bands the aligner should ignore."""

    by_class: dict[PageClass, PageTemplate] = {
        template.page_class: template for template in templates.templates
    }
    results: list[PageClassification] = []
    for page in pages:
        geometry = _page_geometry(page)
        if geometry is None or not templates.has_type_page:
            results.append(PageClassification(page.page_name, "unknown", None, ()))
            continue
        results.append(_classify_page(geometry, templates=templates, by_class=by_class))
    return tuple(results)


def _classify_page(
    geometry: _PageGeometry,
    *,
    templates: BookTemplates,
    by_class: dict[PageClass, PageTemplate],
) -> PageClassification:
    center = templates.first_band_median_px
    window = templates.head_window_px
    if center is None or window is None:
        return PageClassification(geometry.page_name, "unknown", None, ())

    head_ordinal = _head_band_ordinal(geometry, center=center, window=window)
    head_top = geometry.band_tops[head_ordinal]
    offset = head_top - center
    if abs(offset) <= window:
        template = _nearest_normal(geometry, by_class)
        if template is None or abs(geometry.text_left - template.text_left_px) > window:
            return PageClassification(geometry.page_name, "unknown", None, ())
        # The running head is printed but deleted from F2, so the aligner must not
        # match a source line to it. Nothing is printed above the head, so every
        # band down to and including it is furniture.
        residual = int(abs(head_top - template.first_band_top_px))
        return PageClassification(
            geometry.page_name,
            template.page_class,
            residual,
            tuple(range(head_ordinal + 1)),
            _confidence_from_residual(residual, template),
        )

    if offset >= _CHAPTER_SINK_MINIMUM_PX and "chapter_opening" in by_class:
        template = by_class["chapter_opening"]
        residual = int(abs(head_top - template.first_band_top_px))
        return PageClassification(
            geometry.page_name,
            "chapter_opening",
            residual,
            (),
            _confidence_from_residual(residual, template),
        )

    # Between the head window and the chapter sink the evidence is thin, so the
    # page is left unclassified rather than forced into a template.
    return PageClassification(geometry.page_name, "unknown", None, ())


def _confidence_from_residual(residual_px: int, template: PageTemplate) -> float:
    """Invert and scale a template residual into a 0..1 confidence.

    ``template_residual_px`` is a raw pixel distance: unbounded, and lower is
    better, the opposite polarity of every other confidence in the suite. The
    scale is the spread of the template's own fitted group
    (``first_band_spread_px``), floored at ``_HEAD_WINDOW_MINIMUM_PX`` so a
    single-page group — whose spread is 0 — never divides by zero and never
    collapses every nonzero residual straight to 0. A normal page's spread is
    bounded by the book's head window; a chapter opening's is measured against
    pages that sank by widely differing amounts, so each template scores
    against its own spread rather than one shared constant.
    """

    scale_px = max(template.first_band_spread_px, _HEAD_WINDOW_MINIMUM_PX)
    return max(0.0, 1.0 - (residual_px / scale_px))


def _head_band_ordinal(geometry: _PageGeometry, *, center: float, window: float) -> int:
    """Ordinal of the topmost band that could be the page's running head.

    A fleck of dust can hold an ink band of its own above the head, and the
    profile keeps it because dropping it needs the ink the profile does not
    carry. Such a band sits higher than anything the press printed, so the first
    band at or below the head window is the first that can be type. When every
    band is above the window the page is nowhere near its book's text block, and
    the topmost band is returned so the caller reaches the same `unknown` it
    would have reached anyway.
    """

    floor = center - window
    for ordinal, top in enumerate(geometry.band_tops):
        if top >= floor:
            return ordinal
    return 0


def _nearest_normal(
    geometry: _PageGeometry, by_class: dict[PageClass, PageTemplate]
) -> PageTemplate | None:
    """Pick the binding side whose text block sits closest to this page's own."""

    sides = [by_class[name] for name in ("normal_recto", "normal_verso") if name in by_class]
    if not sides:
        return None
    return min(
        sides, key=lambda item: (abs(geometry.text_left - item.text_left_px), item.page_class)
    )


def _split_sides(
    normal: Sequence[_PageGeometry],
) -> tuple[list[_PageGeometry], list[_PageGeometry]]:
    """Split normal pages into binding sides by text-block left edge."""

    if not normal:
        return ([], [])
    center = median(item.text_left for item in normal)
    low = [item for item in normal if item.text_left <= center]
    high = [item for item in normal if item.text_left > center]
    if min(len(low), len(high)) < _SIDE_MINIMUM_SHARE * len(normal):
        # No usable shift to find, so the file name decides after all. It cannot place a head box
        # wrong here, because both sides then share almost the same measurements.
        return (
            [item for item in normal if item.recto],
            [item for item in normal if not item.recto],
        )
    # File names are not always numbered by folio, so the side a name implies can be wrong for a
    # whole book. Geometry defines the two groups; the name only picks which label goes on which.
    if _recto_share(high) > _recto_share(low):
        return (high, low)
    return (low, high)


def _recto_share(group: Sequence[_PageGeometry]) -> float:
    return sum(item.recto for item in group) / len(group) if group else 0.0


def _fit_template(
    page_class: PageClass, group: Sequence[_PageGeometry], *, total: int
) -> PageTemplate | None:
    if not group:
        return None
    tops = [item.first_band_top for item in group]
    center = median(tops)
    return PageTemplate(
        page_class=page_class,
        first_band_top_px=int(center),
        first_band_spread_px=int(median(abs(top - center) for top in tops)),
        text_left_px=int(median(item.text_left for item in group)),
        text_right_px=int(median(item.text_right for item in group)),
        band_count=int(median(item.band_count for item in group)),
        page_count=len(group),
        page_share=len(group) / total if total else 0.0,
    )


def _page_geometry(page: PageMeasurement) -> _PageGeometry | None:
    bounds = page.foreground_bounds
    if not page.ink_bands or bounds is None:
        return None
    return _PageGeometry(
        page_name=page.page_name,
        band_tops=tuple(band.y_start for band in page.ink_bands),
        text_left=bounds[0],
        text_right=bounds[2],
        band_count=len(page.ink_bands),
        recto=_is_recto(page.page_name),
    )


def _is_recto(page_name: str) -> bool:
    """Guess the binding side from the page number in the file name.

    The folio sits on the outer edge, so the text block shifts between the two
    sides in books with a wide binding margin. A page with no number in its name
    is treated as recto, which only costs a template split.
    """

    match = _TRAILING_DIGITS.search(page_name)
    return bool(int(match.group(1)) % 2) if match else True
