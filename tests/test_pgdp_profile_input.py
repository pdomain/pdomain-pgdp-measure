from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdomain_pgdp_measure.profile_input import load_profile_input


def _write_ranking(
    path: Path,
    *,
    schema_version: object = 1,
    algorithm_version: object = "pgdp-rank/v1",
    projects: object = (),
    payload: object | None = None,
) -> None:
    ranking_payload = (
        _ranking_payload(
            schema_version=schema_version,
            algorithm_version=algorithm_version,
            projects=projects,
        )
        if payload is None
        else payload
    )
    path.write_text(json.dumps(ranking_payload), encoding="utf-8")


def _ranking_payload(
    *,
    schema_version: object = 1,
    algorithm_version: object = "pgdp-rank/v1",
    projects: object = (),
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "algorithm_version": algorithm_version,
        "limits": {"projects": 50, "pages_per_project": 12},
        "corpus": {"projects_seen": 1, "projects_ranked": 1},
        "diagnostics": [],
        "projects": projects,
    }


def _project(project_id: str, page_names: tuple[str, ...]) -> dict[str, object]:
    return {
        "project_id": project_id,
        "title": f"Title {project_id}",
        "author": "Author",
        "genre": "Fiction",
        "pages_total": 1,
        "pg_ebook_number": 1,
        "score": 0,
        "score_components": {"top_ten_page_scores": 0, "group_bonus": 0},
        "pages": [_page(page_name) for page_name in page_names],
    }


def _page(name: str) -> dict[str, object]:
    return {
        "name": name,
        "transcription_available": False,
        "image_available": False,
        "features": {
            "special_format": False,
            "italic_tags": 0,
            "bold_tags": 0,
            "small_caps_tags": 0,
            "dot_leaders": False,
            "aligned_fields": False,
            "table_like": False,
            "poetry_like": False,
            "quotation_like": False,
            "multipart_name": False,
            "illustration_or_ornament": False,
            "uncertainty_note": False,
        },
        "score": {"total": 0, "components": {"special_format": 0}},
        "matched_groups": [],
    }


def _diagnostic() -> dict[str, object]:
    return {
        "code": "issue",
        "message": "Issue",
        "project_id": None,
        "page_name": None,
    }


def _remove_path(payload: dict[str, object], path: str) -> None:
    current: dict[str, object] | list[object] = payload
    parts = path.split(".")
    for part in parts[:-1]:
        child = _container_child(current, part)
        if not isinstance(child, (dict, list)):
            raise AssertionError(f"Expected a JSON container at {part!r}.")
        current = child
    if not isinstance(current, dict):
        raise AssertionError(f"Expected an object before {parts[-1]!r}.")
    del current[parts[-1]]


def _container_child(container: dict[str, object] | list[object], part: str) -> object:
    if isinstance(container, dict):
        return container[part]
    return container[int(part)]


def _make_project(corpus_root: Path, project_id: str) -> Path:
    project_directory = corpus_root / project_id
    project_directory.mkdir(parents=True)
    return project_directory


def test_profile_input_preserves_report_order_and_records_root_image_path(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    project_b = _make_project(corpus_root, "projectIDb")
    project_a = _make_project(corpus_root, "projectIDa")
    (project_b / "p2.png").touch()
    (project_b / "p1.png").touch()
    (project_a / "p3.png").touch()
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(
        ranking_path,
        projects=(
            _project("projectIDb", ("p2.png", "p1.png")),
            _project("projectIDa", ("p3.png",)),
        ),
    )

    ranking_input = load_profile_input(ranking_path, corpus_root=corpus_root)

    assert [project.project_id for project in ranking_input.projects] == [
        "projectIDb",
        "projectIDa",
    ]
    assert [page.name for page in ranking_input.projects[0].pages] == ["p2.png", "p1.png"]
    page = ranking_input.projects[0].pages[0]
    assert page.source_path == "projectIDb/p2.png"
    assert page.image_path == project_b / "p2.png"


def test_profile_input_rejects_unknown_algorithm(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(ranking_path, algorithm_version="pgdp-rank/v2")

    with pytest.raises(ValueError, match="pgdp-rank/v1"):
        _ = load_profile_input(ranking_path, corpus_root=corpus_root)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[1]", "valid ranking JSON"),
        ("{", "valid ranking JSON"),
        (
            json.dumps(
                {"schema_version": "1", "algorithm_version": "pgdp-rank/v1", "projects": []}
            ),
            "valid ranking JSON",
        ),
    ],
)
def test_profile_input_rejects_malformed_ranking_json(
    tmp_path: Path, payload: str, message: str
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _ = load_profile_input(ranking_path, corpus_root=corpus_root)


@pytest.mark.parametrize(
    "missing_path",
    [
        "schema_version",
        "algorithm_version",
        "limits",
        "limits.projects",
        "limits.pages_per_project",
        "corpus",
        "corpus.projects_seen",
        "corpus.projects_ranked",
        "diagnostics",
        "diagnostics.0.code",
        "diagnostics.0.message",
        "diagnostics.0.project_id",
        "diagnostics.0.page_name",
        "projects",
        "projects.0.project_id",
        "projects.0.title",
        "projects.0.author",
        "projects.0.genre",
        "projects.0.pages_total",
        "projects.0.pg_ebook_number",
        "projects.0.score",
        "projects.0.score_components",
        "projects.0.pages",
        "projects.0.pages.0.name",
        "projects.0.pages.0.transcription_available",
        "projects.0.pages.0.image_available",
        "projects.0.pages.0.features",
        "projects.0.pages.0.features.special_format",
        "projects.0.pages.0.features.italic_tags",
        "projects.0.pages.0.features.bold_tags",
        "projects.0.pages.0.features.small_caps_tags",
        "projects.0.pages.0.features.dot_leaders",
        "projects.0.pages.0.features.aligned_fields",
        "projects.0.pages.0.features.table_like",
        "projects.0.pages.0.features.poetry_like",
        "projects.0.pages.0.features.quotation_like",
        "projects.0.pages.0.features.multipart_name",
        "projects.0.pages.0.features.illustration_or_ornament",
        "projects.0.pages.0.features.uncertainty_note",
        "projects.0.pages.0.score",
        "projects.0.pages.0.score.total",
        "projects.0.pages.0.score.components",
        "projects.0.pages.0.matched_groups",
    ],
)
def test_profile_input_rejects_each_missing_required_m14_field(
    tmp_path: Path, missing_path: str
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    payload = _ranking_payload(projects=[_project("projectIDone", ("p1.png",))])
    if missing_path.startswith("diagnostics.0"):
        payload["diagnostics"] = [_diagnostic()]
    _remove_path(payload, missing_path)
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(ranking_path, payload=payload)

    with pytest.raises(ValueError, match="valid ranking JSON"):
        _ = load_profile_input(ranking_path, corpus_root=corpus_root)


def test_profile_input_rejects_unknown_schema_version(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(ranking_path, schema_version=2)

    with pytest.raises(ValueError, match="schema version 1"):
        _ = load_profile_input(ranking_path, corpus_root=corpus_root)


def test_profile_input_rejects_duplicate_project_ids(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    _make_project(corpus_root, "projectIDone")
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(
        ranking_path,
        projects=(
            _project("projectIDone", ()),
            _project("projectIDone", ()),
        ),
    )

    with pytest.raises(ValueError, match="Duplicate project_id"):
        _ = load_profile_input(ranking_path, corpus_root=corpus_root)


def test_profile_input_rejects_duplicate_page_names(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    _make_project(corpus_root, "projectIDone")
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(ranking_path, projects=(_project("projectIDone", ("p1.png", "p1.png")),))

    with pytest.raises(ValueError, match="Duplicate page name"):
        _ = load_profile_input(ranking_path, corpus_root=corpus_root)


@pytest.mark.parametrize("project_alias", ["projectIDone/.", "projectIDone//"])
def test_profile_input_rejects_noncanonical_project_aliases(
    tmp_path: Path, project_alias: str
) -> None:
    corpus_root = tmp_path / "corpus"
    _make_project(corpus_root, "projectIDone")
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(
        ranking_path,
        projects=(
            _project("projectIDone", ()),
            _project(project_alias, ()),
        ),
    )

    with pytest.raises(ValueError, match=r"(?i)(canonical|unsafe)"):
        _ = load_profile_input(ranking_path, corpus_root=corpus_root)


@pytest.mark.parametrize("page_alias", ["./p1.png", "nested/../p1.png", "p1.png/"])
def test_profile_input_rejects_noncanonical_page_aliases(tmp_path: Path, page_alias: str) -> None:
    corpus_root = tmp_path / "corpus"
    project_directory = _make_project(corpus_root, "projectIDone")
    (project_directory / "p1.png").touch()
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(
        ranking_path,
        projects=(_project("projectIDone", ("p1.png", page_alias)),),
    )

    with pytest.raises(ValueError, match=r"(?i)(canonical|unsafe)"):
        _ = load_profile_input(ranking_path, corpus_root=corpus_root)


def test_profile_input_rejects_a_missing_project_directory(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(ranking_path, projects=(_project("projectIDmissing", ()),))

    with pytest.raises(ValueError, match="does not exist"):
        _ = load_profile_input(ranking_path, corpus_root=corpus_root)


@pytest.mark.parametrize(
    "project_id",
    ["/absolute", "../traversal", "projectID\x00bad", "projectIDone\\alias"],
)
def test_profile_input_rejects_unsafe_project_references(tmp_path: Path, project_id: str) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(ranking_path, projects=(_project(project_id, ()),))

    with pytest.raises(ValueError, match=r"(?i)unsafe"):
        _ = load_profile_input(ranking_path, corpus_root=corpus_root)


@pytest.mark.parametrize(
    "page_name",
    ["/absolute.png", "../traversal.png", "p1\x00.png", ".", "nested\\p1.png"],
)
def test_profile_input_rejects_unsafe_page_references(tmp_path: Path, page_name: str) -> None:
    corpus_root = tmp_path / "corpus"
    _make_project(corpus_root, "projectIDone")
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(ranking_path, projects=(_project("projectIDone", (page_name,)),))

    with pytest.raises(ValueError, match=r"(?i)unsafe"):
        _ = load_profile_input(ranking_path, corpus_root=corpus_root)


def test_profile_input_rejects_symlink_escape(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    project_directory = _make_project(corpus_root, "projectIDone")
    outside = tmp_path / "outside.png"
    outside.touch()
    try:
        (project_directory / "p1.png").symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Symlinks are unavailable: {error}")
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(ranking_path, projects=(_project("projectIDone", ("p1.png",)),))

    with pytest.raises(ValueError, match="escapes"):
        _ = load_profile_input(ranking_path, corpus_root=corpus_root)


def test_profile_input_uses_root_precedence_when_root_and_images_candidates_exist(
    tmp_path: Path,
) -> None:
    corpus_root = tmp_path / "corpus"
    project_directory = _make_project(corpus_root, "projectIDone")
    root_image = project_directory / "p1.png"
    root_image.touch()
    images_image = project_directory / "images" / "p1.png"
    images_image.parent.mkdir()
    images_image.touch()
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(ranking_path, projects=(_project("projectIDone", ("p1.png",)),))

    ranking_input = load_profile_input(ranking_path, corpus_root=corpus_root)

    page = ranking_input.projects[0].pages[0]
    assert page.image_path == root_image
    assert page.source_path == "projectIDone/p1.png"


def test_profile_input_keeps_missing_image_page_for_later_measurement_diagnostics(
    tmp_path: Path,
) -> None:
    corpus_root = tmp_path / "corpus"
    _make_project(corpus_root, "projectIDone")
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(ranking_path, projects=(_project("projectIDone", ("missing.png",)),))

    ranking_input = load_profile_input(ranking_path, corpus_root=corpus_root)

    page = ranking_input.projects[0].pages[0]
    assert page.image_path is None
    assert page.source_path is None


def test_whole_book_measures_every_page_but_emits_only_ranked_pages(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    project = _make_project(corpus_root, "projectIDa")
    for name in ("p1.png", "p2.png", "p3.png", "p4.png"):
        (project / name).touch()
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(ranking_path, projects=(_project("projectIDa", ("p2.png", "p4.png")),))

    selection = load_profile_input(ranking_path, corpus_root=corpus_root, whole_book=True)

    pages = selection.projects[0].pages
    assert [page.name for page in pages] == ["p1.png", "p2.png", "p3.png", "p4.png"]
    assert [page.name for page in pages if page.emit] == ["p2.png", "p4.png"]


def test_whole_book_is_off_by_default(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    project = _make_project(corpus_root, "projectIDa")
    for name in ("p1.png", "p2.png"):
        (project / name).touch()
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(ranking_path, projects=(_project("projectIDa", ("p2.png",)),))

    selection = load_profile_input(ranking_path, corpus_root=corpus_root)

    pages = selection.projects[0].pages
    assert [page.name for page in pages] == ["p2.png"]
    assert all(page.emit for page in pages)


def test_whole_book_keeps_a_ranked_page_missing_from_disk(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    project = _make_project(corpus_root, "projectIDa")
    (project / "p1.png").touch()
    ranking_path = tmp_path / "ranking.json"
    _write_ranking(ranking_path, projects=(_project("projectIDa", ("p9.png",)),))

    selection = load_profile_input(ranking_path, corpus_root=corpus_root, whole_book=True)

    pages = selection.projects[0].pages
    assert [page.name for page in pages] == ["p1.png", "p9.png"]
    missing = next(page for page in pages if page.name == "p9.png")
    assert missing.emit is True
    assert missing.image_path is None
    unranked = next(page for page in pages if page.name == "p1.png")
    assert unranked.emit is False
