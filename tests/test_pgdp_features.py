from __future__ import annotations

import pytest

from pdomain_pgdp_measure.f2 import ParsedF2Page, parse_f2_pages
from pdomain_pgdp_measure.models import PageFeatures


def _page(name: str, source_text: str) -> ParsedF2Page:
    return parse_f2_pages({name: source_text}).pages[0]


@pytest.mark.parametrize(
    ("tag", "attribute", "feature_name"),
    [
        ("I", ' class="em"', "italic_tags"),
        ("b", ' data-weight="bold"', "bold_tags"),
        ("Sc", ' lang="la"', "small_caps_tags"),
    ],
)
def test_extract_page_features_counts_case_insensitive_opening_tags(
    tag: str, attribute: str, feature_name: str
) -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    page = _page(
        "p1.png",
        f"outside <{tag}{attribute}>one</{tag}> /* <{tag}>two</{tag}> */ <{tag}/>",
    )

    features = extract_page_features(page)

    assert getattr(features, feature_name) == 3


def test_extract_page_features_counts_a_tag_followed_by_literal_greater_than() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    features = extract_page_features(_page("p1.png", "<i>>literal</i>"))

    assert features.italic_tags == 1


def test_extract_page_features_attributes_each_continued_body_to_its_page() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    result = parse_f2_pages(
        {
            "p10.png": "third field   value #/ outside <i>text</i>",
            "p2.png": "before /# <i>first</i>\nfirst field   value",
            "p3.png": "<b>middle</b>\nsecond field   value",
        }
    )

    features_by_page = {page.name: extract_page_features(page) for page in result.pages}

    assert features_by_page["p2.png"].italic_tags == 1
    assert features_by_page["p2.png"].bold_tags == 0
    assert features_by_page["p2.png"].aligned_fields is False
    assert features_by_page["p3.png"].italic_tags == 0
    assert features_by_page["p3.png"].bold_tags == 1
    assert features_by_page["p3.png"].aligned_fields is False
    assert features_by_page["p10.png"].italic_tags == 1
    assert features_by_page["p10.png"].aligned_fields is False


def test_extract_page_features_detects_dot_leaders_in_special_bodies() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    features = extract_page_features(_page("p1.png", "outside ... /* title . . . 12 */"))

    assert features.dot_leaders is True


def test_extract_page_features_detects_two_aligned_field_rows_without_table() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    features = extract_page_features(_page("p1.png", "/* name   value\nother   12 */"))

    assert features.aligned_fields is True
    assert features.table_like is False


def test_extract_page_features_detects_three_table_like_rows() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    features = extract_page_features(_page("p1.png", "/* name   value\nother   12\nthird   19 */"))

    assert features.table_like is True


def test_extract_page_features_does_not_combine_aligned_rows_from_separate_blocks() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    features = extract_page_features(_page("p1.png", "/* first   1 */ plain /* second   2 */"))

    assert features.aligned_fields is False
    assert features.table_like is False


def test_extract_page_features_does_not_combine_table_rows_from_separate_blocks() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    features = extract_page_features(
        _page("p1.png", "/* first   1\nsecond   2 */ plain /* third   3 */")
    )

    assert features.aligned_fields is True
    assert features.table_like is False


def test_extract_page_features_detects_poetry_line_shape() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    long_line = "ordinary prose " * 12
    source_text = f"{long_line}\n{long_line}\n{long_line}\n{long_line}\n/* a\nbb\nccc\ndddd */"

    features = extract_page_features(_page("p1.png", source_text))

    assert features.poetry_like is True


def test_extract_page_features_does_not_combine_poetry_lines_from_separate_blocks() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    long_line = "ordinary prose " * 12
    source_text = (
        f"{long_line}\n{long_line}\n{long_line}\n{long_line}\n/* a\nbb */ plain /* ccc\ndddd */"
    )

    features = extract_page_features(_page("p1.png", source_text))

    assert features.poetry_like is False


def test_extract_page_features_detects_four_quoted_body_lines() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    features = extract_page_features(_page("p1.png", '/* "one"\n"two"\n"three"\n"four" */'))

    assert features.quotation_like is True


def test_extract_page_features_requires_quoted_lines_to_be_consecutive() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    features = extract_page_features(_page("p1.png", '/* "one"\n"two"\n\n"three"\n"four" */'))

    assert features.quotation_like is False


def test_extract_page_features_does_not_join_quoted_runs_from_separate_blocks() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    features = extract_page_features(
        _page("p1.png", '/* "one"\n"two" */ plain /* "three"\n"four" */')
    )

    assert features.quotation_like is False


def test_extract_page_features_detects_common_special_body_indent() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    features = extract_page_features(_page("p1.png", "/*   one\n   two */"))

    assert features.quotation_like is True


@pytest.mark.parametrize("name", ["p12L.png", "p12M.jpg", "p12R.tif"])
def test_extract_page_features_detects_multipart_page_names(name: str) -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    features = extract_page_features(_page(name, "plain text"))

    assert features.multipart_name is True


@pytest.mark.parametrize("marker", ["[Illustration: frontispiece]", "[Decoration: fleur]"])
def test_extract_page_features_detects_illustration_or_ornament_markers(marker: str) -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    features = extract_page_features(_page("p1.png", marker))

    assert features.illustration_or_ornament is True


def test_extract_page_features_detects_uncertainty_note() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    features = extract_page_features(_page("p1.png", "[** damaged source]"))

    assert features.uncertainty_note is True


def test_extract_page_features_excludes_invalid_blocks_from_structural_predicates() -> None:
    from pdomain_pgdp_measure.features import extract_page_features

    features = extract_page_features(_page("p1.png", "/* title . . . 12\nname   value"))

    assert features.special_format is False
    assert features.dot_leaders is False
    assert features.aligned_fields is False


def test_extract_page_features_excludes_typography_tags_in_unmatched_blocks() -> None:
    from pdomain_pgdp_measure.features import extract_page_features, score_page

    features = extract_page_features(
        _page("p1.png", "<i>valid</i> /* <b>invalid</b><sc>invalid</sc>")
    )

    assert features.italic_tags == 1
    assert features.bold_tags == 0
    assert features.small_caps_tags == 0
    assert score_page(features).total == 1


def test_page_score_exposes_every_component() -> None:
    from pdomain_pgdp_measure.features import score_page

    features = PageFeatures(
        special_format=True,
        italic_tags=12,
        bold_tags=2,
        small_caps_tags=1,
        dot_leaders=True,
        aligned_fields=True,
        table_like=True,
        poetry_like=True,
        quotation_like=True,
        multipart_name=True,
        illustration_or_ornament=True,
        uncertainty_note=True,
    )

    score = score_page(features)

    assert score.total == 60
    assert score.components == {
        "special_format": 5,
        "italic_tags": 10,
        "bold_tags": 2,
        "small_caps_tags": 1,
        "leaders_or_fields": 8,
        "table_like": 8,
        "poetry_like": 8,
        "quotation_like": 6,
        "multipart_name": 6,
        "illustration_or_ornament": 4,
        "uncertainty_note": 2,
    }


def test_page_score_does_not_double_count_overlapping_leaders_and_fields() -> None:
    from pdomain_pgdp_measure.features import score_page

    score = score_page(PageFeatures(dot_leaders=True, aligned_fields=True))

    assert score.total == 8
    assert score.components["leaders_or_fields"] == 8
