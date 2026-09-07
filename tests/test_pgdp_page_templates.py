"""Tests for PGDP per-book page templates and page classification."""

from __future__ import annotations

from dataclasses import replace

from pdomain_pgdp_measure.page_templates import (
    classify_pages,
    fit_book_templates,
)
from pdomain_pgdp_measure.profile_models import (
    CoordinateFrame,
    InkBand,
    PageMeasurement,
    ProfileDiagnostic,
)

_WIDTH = 1000
_HEIGHT = 1600


def _page(
    page_name: str,
    *,
    first_band_top: int,
    left: int = 80,
    right: int = 920,
    bands: int = 24,
) -> PageMeasurement:
    """Build a measured page with evenly pitched bands from a given top."""

    ink_bands = tuple(
        InkBand(y_start=first_band_top + index * 40, y_end=first_band_top + index * 40 + 26)
        for index in range(bands)
    )
    bottom = ink_bands[-1].y_end
    bounds = (left, first_band_top, right, bottom)
    return PageMeasurement(
        page_name=page_name,
        source_path=f"projectID1/{page_name}",
        sha256="a" * 64,
        source_frame=CoordinateFrame(width=_WIDTH, height=_HEIGHT),
        image_mode="L",
        grayscale_threshold=127,
        foreground_pixels=1000,
        foreground_bounds=bounds,
        margins=(left, first_band_top, _WIDTH - right, _HEIGHT - bottom),
        ink_bands=ink_bands,
        derived_estimates=(),
        diagnostics=(),
    )


def _blank(page_name: str) -> PageMeasurement:
    return PageMeasurement(
        page_name=page_name,
        source_path=f"projectID1/{page_name}",
        sha256="a" * 64,
        source_frame=CoordinateFrame(width=_WIDTH, height=_HEIGHT),
        image_mode="L",
        grayscale_threshold=127,
        foreground_pixels=0,
        foreground_bounds=None,
        margins=None,
        ink_bands=(),
        derived_estimates=(),
        diagnostics=(ProfileDiagnostic(code="blank_page", message="blank", page_name=page_name),),
    )


def _with_leading_band(page: PageMeasurement, band: InkBand) -> PageMeasurement:
    """Put one extra ink band above everything the page already measured."""

    bands = page.ink_bands
    assert bands is not None
    return replace(page, ink_bands=(band, *bands))


def _steady_book() -> tuple[PageMeasurement, ...]:
    """A book whose recto and verso differ by a 60px binding shift."""

    pages: list[PageMeasurement] = []
    for number in range(1, 21):
        recto = bool(number % 2)
        pages.append(
            _page(
                f"p{number:03d}.png",
                first_band_top=71 if recto else 72,
                left=80 if recto else 140,
                right=920 if recto else 980,
            )
        )
    return tuple(pages)


def test_steady_book_fits_separate_recto_and_verso_templates() -> None:
    templates = fit_book_templates(_steady_book())

    assert templates.has_type_page
    assert templates.first_band_median_px == 71.5
    assert templates.first_band_mad_px == 0.5
    assert templates.head_window_px == 8.0
    by_class = {item.page_class: item for item in templates.templates}
    assert set(by_class) == {"normal_recto", "normal_verso"}
    assert by_class["normal_recto"].text_left_px == 80
    assert by_class["normal_verso"].text_left_px == 140
    assert by_class["normal_recto"].page_count == 10
    assert by_class["normal_verso"].page_share == 0.5
    assert by_class["normal_recto"].band_count == 24


def test_book_without_a_binding_shift_still_fits_both_sides() -> None:
    pages = tuple(
        _page(f"p{number:03d}.png", first_band_top=67, left=90, right=910)
        for number in range(1, 11)
    )

    templates = fit_book_templates(pages)

    by_class = {item.page_class: item for item in templates.templates}
    assert by_class["normal_recto"].text_left_px == by_class["normal_verso"].text_left_px


def test_side_split_survives_file_names_numbered_by_ten() -> None:
    """A book naming pages at ten times the folio must still split by binding side."""

    pages: list[PageMeasurement] = []
    for folio in range(1, 21):
        recto = bool(folio % 2)
        pages.append(
            _page(
                f"p{folio * 10:04d}.png",
                first_band_top=111,
                left=120 if recto else 185,
                right=880 if recto else 945,
            )
        )

    templates = fit_book_templates(tuple(pages))

    by_class = {item.page_class: item for item in templates.templates}
    assert set(by_class) == {"normal_recto", "normal_verso"}
    assert {by_class["normal_recto"].text_left_px, by_class["normal_verso"].text_left_px} == {
        120,
        185,
    }
    assert by_class["normal_recto"].page_count == 10
    assert by_class["normal_verso"].page_count == 10


def test_pages_of_a_book_numbered_by_ten_all_classify() -> None:
    pages: list[PageMeasurement] = []
    for folio in range(1, 21):
        recto = bool(folio % 2)
        pages.append(
            _page(
                f"p{folio * 10:04d}.png",
                first_band_top=111,
                left=120 if recto else 185,
                right=880 if recto else 945,
            )
        )

    templates = fit_book_templates(tuple(pages))
    classified = {item.page_name: item for item in classify_pages(tuple(pages), templates)}

    assert all(item.page_class != "unknown" for item in classified.values())
    assert all(item.furniture_band_ordinals == (0,) for item in classified.values())


def test_book_with_a_noisy_head_still_fits_templates() -> None:
    """A first band that wanders by about 16px is unsteady, not absent."""

    offsets = (-18, -12, -6, 0, 6, 12, 18, -16, 16, 0)
    pages = tuple(
        _page(
            f"p{number:03d}.png",
            first_band_top=185 + offsets[number % len(offsets)],
            left=80 if number % 2 else 140,
            right=920 if number % 2 else 980,
        )
        for number in range(1, 41)
    )

    templates = fit_book_templates(pages)

    assert templates.has_type_page
    assert templates.first_band_mad_px is not None
    assert 2 < templates.first_band_mad_px <= 25
    assert templates.head_window_px is not None
    assert templates.head_window_px <= 75


def test_book_whose_head_wanders_past_the_ceiling_fits_nothing() -> None:
    pages = tuple(
        _page(f"p{number:03d}.png", first_band_top=185 + (number % 2) * 60)
        for number in range(1, 21)
    )

    templates = fit_book_templates(pages)

    assert not templates.has_type_page
    assert templates.templates == ()


def test_scattered_book_fits_no_templates() -> None:
    pages = tuple(
        _page(f"p{number:03d}.png", first_band_top=top)
        for number, top in enumerate((60, 90, 120, 150, 180, 210), start=1)
    )

    templates = fit_book_templates(pages)

    assert templates.has_type_page is False
    assert templates.templates == ()
    assert templates.first_band_mad_px is not None
    assert templates.first_band_mad_px > 2
    assert templates.head_window_px is None


def test_chapter_openings_fit_their_own_template() -> None:
    pages = list(_steady_book())
    pages.append(_page("p021.png", first_band_top=367, bands=20))
    pages.append(_page("p022.png", first_band_top=368, left=140, right=980, bands=20))

    templates = fit_book_templates(tuple(pages))

    by_class = {item.page_class: item for item in templates.templates}
    assert by_class["chapter_opening"].page_count == 2
    assert by_class["chapter_opening"].first_band_top_px == 367
    assert by_class["chapter_opening"].band_count == 20


def test_a_book_with_no_measurable_page_fits_nothing() -> None:
    templates = fit_book_templates((_blank("p001.png"),))

    assert templates.has_type_page is False
    assert templates.first_band_median_px is None
    assert templates.classified_share == 0.0


def test_normal_pages_classify_by_side_and_name_their_head_band() -> None:
    pages = _steady_book()
    templates = fit_book_templates(pages)

    classified = {item.page_name: item for item in classify_pages(pages, templates)}

    recto = classified["p001.png"]
    verso = classified["p002.png"]
    assert recto.page_class == "normal_recto"
    assert verso.page_class == "normal_verso"
    assert recto.furniture_band_ordinals == (0,)
    assert verso.furniture_band_ordinals == (0,)
    assert recto.template_residual_px == 0


def test_a_chapter_opening_keeps_every_band() -> None:
    pages = (*_steady_book(), _page("p021.png", first_band_top=367, bands=20))
    templates = fit_book_templates(pages)

    classified = {item.page_name: item for item in classify_pages(pages, templates)}

    opening = classified["p021.png"]
    assert opening.page_class == "chapter_opening"
    assert opening.furniture_band_ordinals == ()


def test_a_page_in_the_trough_classifies_unknown() -> None:
    trough = _page("p021.png", first_band_top=151)
    pages = (*_steady_book(), trough)
    templates = fit_book_templates(pages)

    classified = {item.page_name: item for item in classify_pages(pages, templates)}

    assert classified["p021.png"].page_class == "unknown"
    assert classified["p021.png"].furniture_band_ordinals == ()
    assert classified["p021.png"].template_residual_px is None


def test_every_page_of_a_scattered_book_classifies_unknown() -> None:
    pages = tuple(
        _page(f"p{number:03d}.png", first_band_top=top)
        for number, top in enumerate((60, 90, 120, 150, 180, 210), start=1)
    )
    templates = fit_book_templates(pages)

    classified = classify_pages(pages, templates)

    assert {item.page_class for item in classified} == {"unknown"}
    assert all(item.furniture_band_ordinals == () for item in classified)


def test_a_blank_page_classifies_unknown() -> None:
    pages = (*_steady_book(), _blank("p021.png"))
    templates = fit_book_templates(pages)

    classified = {item.page_name: item for item in classify_pages(pages, templates)}

    assert classified["p021.png"].page_class == "unknown"


def test_a_page_whose_text_block_is_misplaced_classifies_unknown() -> None:
    stray = _page("p021.png", first_band_top=71, left=300, right=920)
    pages = (*_steady_book(), stray)
    templates = fit_book_templates(pages)

    classified = {item.page_name: item for item in classify_pages(pages, templates)}

    assert classified["p021.png"].page_class == "unknown"
    assert classified["p021.png"].furniture_band_ordinals == ()


def test_a_head_four_pixels_off_the_median_is_still_a_head() -> None:
    """Regression for p179.png, whose running head sat 4px below its book median.

    A window of 2px classified it `unknown`, so its head was never suppressed and
    the page stayed misaligned. Running heads were measured 0 to 4px off, while
    the nearest genuine top-of-page text sat 76px below.
    """

    pages = (*_steady_book(), _page("p021.png", first_band_top=75))
    templates = fit_book_templates(pages)

    classified = {item.page_name: item for item in classify_pages(pages, templates)}

    assert classified["p021.png"].page_class == "normal_recto"
    assert classified["p021.png"].furniture_band_ordinals == (0,)


def test_genuine_top_of_page_text_is_not_taken_for_a_head() -> None:
    """Regression for a005.png, a dedication 76px below its book median."""

    pages = (*_steady_book(), _page("p021.png", first_band_top=147, bands=18))
    templates = fit_book_templates(pages)

    classified = {item.page_name: item for item in classify_pages(pages, templates)}

    assert classified["p021.png"].page_class == "unknown"
    assert classified["p021.png"].furniture_band_ordinals == ()


def test_a_speck_above_the_head_window_does_not_hide_the_running_head() -> None:
    """Regression for 166.png, which carried a dirt speck 80px above its head.

    The profile keeps the speck because dropping it needs ink the profile does
    not carry, so band 0 sat 63px above the book median. The page fell outside
    the head window, classified `unknown`, and its running head was never
    suppressed, which let a body source line bind to the head.
    """

    page = _page("p021.png", first_band_top=71)
    speckled = _with_leading_band(page, InkBand(y_start=20, y_end=26))
    pages = (*_steady_book(), speckled)
    templates = fit_book_templates(pages)

    classified = {item.page_name: item for item in classify_pages(pages, templates)}

    assert classified["p021.png"].page_class == "normal_recto"
    assert classified["p021.png"].furniture_band_ordinals == (0, 1)


def test_a_speck_inside_the_head_window_still_names_only_the_first_band() -> None:
    """A band that could itself be the head is taken as the head, as before.

    A thin roman folio sits exactly where a running head sits, so position
    cannot tell it from dust and height must not be asked to. Only a band above
    the window is skipped.
    """

    page = _page("p021.png", first_band_top=71)
    speckled = _with_leading_band(page, InkBand(y_start=69, y_end=72))
    pages = (*_steady_book(), speckled)
    templates = fit_book_templates(pages)

    classified = {item.page_name: item for item in classify_pages(pages, templates)}

    assert classified["p021.png"].furniture_band_ordinals == (0,)


def test_a_speck_above_a_chapter_opening_still_leaves_every_band() -> None:
    """The speck must not suppress a band on a page that has no running head.

    The book's chapter-opening template is fitted from the topmost band of every
    page, so it takes clean openings to exist at all. A speckled opening is then
    classified against it.
    """

    openings = tuple(
        _page(f"c{number:03d}.png", first_band_top=400, bands=16) for number in range(1, 5)
    )
    page = _page("p021.png", first_band_top=400, bands=16)
    speckled = _with_leading_band(page, InkBand(y_start=20, y_end=26))
    pages = (*_steady_book(), *openings, speckled)
    templates = fit_book_templates(pages)

    classified = {item.page_name: item for item in classify_pages(pages, templates)}

    assert classified["p021.png"].page_class == "chapter_opening"
    assert classified["p021.png"].furniture_band_ordinals == ()
    # The residual measures the opening against its template, so it must come
    # from the band that opens the chapter and not from the speck above it.
    assert classified["p021.png"].template_residual_px == 0
