"""Tests for the local PGDP ranking command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pdomain_ocr_synth.cli import build_parser, main


def _write_project(
    corpus_root: Path,
    project_id: str,
    *,
    f2_contents: str,
) -> None:
    project_directory = corpus_root / project_id
    project_directory.mkdir()
    (project_directory / "project.json").write_text(
        json.dumps(
            {
                "projectid": project_id,
                "title": f"Title {project_id}",
                "author": "Author",
                "genre": "Fiction",
                "pages_total": 1,
                "pg_ebook_number": 99,
            }
        ),
        encoding="utf-8",
    )
    rounds_directory = project_directory / "rounds"
    rounds_directory.mkdir()
    (rounds_directory / "F2.json").write_text(f2_contents, encoding="utf-8")
    (project_directory / "pages.json").write_text(
        json.dumps({"p1.png": "text"}),
        encoding="utf-8",
    )


def test_rank_pgdp_parser_accepts_the_pinned_interface() -> None:
    args = build_parser().parse_args(
        [
            "rank-pgdp",
            "corpus",
            "--output",
            "ranking.json",
            "--project-limit",
            "4",
            "--pages-per-project",
            "3",
        ]
    )

    assert args.command == "rank-pgdp"
    assert args.corpus_root == "corpus"
    assert args.output == "ranking.json"
    assert args.project_limit == 4
    assert args.pages_per_project == 3


def test_rank_pgdp_parser_uses_pinned_defaults() -> None:
    args = build_parser().parse_args(["rank-pgdp", "corpus"])

    assert args.output == "./pgdp-ranking.json"
    assert args.project_limit == 50
    assert args.pages_per_project == 12


def test_rank_pgdp_writes_a_report_and_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    _write_project(
        corpus_root,
        "projectIDone",
        f2_contents=json.dumps({"p1.png": "/* <i>text</i> */"}),
    )
    output = tmp_path / "ranking.json"

    rc = main(
        [
            "rank-pgdp",
            str(corpus_root),
            "--output",
            str(output),
            "--project-limit",
            "1",
            "--pages-per-project",
            "1",
        ]
    )

    assert rc == 0
    assert json.loads(output.read_text(encoding="utf-8"))["corpus"] == {
        "projects_seen": 1,
        "projects_ranked": 1,
    }
    summary = capsys.readouterr().out
    assert "projects seen: 1" in summary
    assert "projects ranked: 1" in summary
    assert "diagnostics: 0" in summary
    assert "selected pages: 1" in summary


@pytest.mark.parametrize(
    ("root_kind", "expected_error"),
    [("missing", "does not exist"), ("file", "not a directory")],
)
def test_rank_pgdp_rejects_missing_or_non_directory_corpus_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    root_kind: str,
    expected_error: str,
) -> None:
    corpus_root = tmp_path / "corpus"
    if root_kind == "file":
        corpus_root.write_text("not a directory\n", encoding="utf-8")

    rc = main(["rank-pgdp", str(corpus_root)])

    captured = capsys.readouterr()
    assert rc == 2
    assert expected_error in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    ("option", "value"),
    [("--project-limit", "0"), ("--pages-per-project", "-1")],
)
def test_rank_pgdp_rejects_nonpositive_limits(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()

    rc = main(["rank-pgdp", str(corpus_root), option, value])

    captured = capsys.readouterr()
    assert rc == 2
    assert "must be positive" in captured.err
    assert captured.out == ""


def test_rank_pgdp_rejects_output_inside_the_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()

    rc = main(["rank-pgdp", str(corpus_root), "--output", str(corpus_root / "ranking.json")])

    captured = capsys.readouterr()
    assert rc == 2
    assert "outside the corpus root" in captured.err
    assert captured.out == ""


def test_rank_pgdp_rejects_an_existing_non_file_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    output = tmp_path / "ranking.json"
    output.mkdir()

    rc = main(["rank-pgdp", str(corpus_root), "--output", str(output)])

    captured = capsys.readouterr()
    assert rc == 2
    assert "output is not a file" in captured.err
    assert captured.out == ""


def test_rank_pgdp_writes_a_partial_report_for_malformed_projects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    _write_project(corpus_root, "projectIDbroken", f2_contents="not JSON")
    _write_project(
        corpus_root,
        "projectIDvalid",
        f2_contents=json.dumps({"p1.png": "/* <i>text</i> */"}),
    )
    output = tmp_path / "ranking.json"

    rc = main(["rank-pgdp", str(corpus_root), "--output", str(output)])

    assert rc == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert [project["project_id"] for project in report["projects"]] == ["projectIDvalid"]
    assert report["diagnostics"] == [
        {
            "code": "malformed_f2",
            "message": "Could not load rounds/F2.json as a string-to-string JSON object.",
            "page_name": None,
            "project_id": "projectIDbroken",
        }
    ]
    assert "projects seen: 2" in capsys.readouterr().out
