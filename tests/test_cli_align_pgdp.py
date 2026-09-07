"""Tests for the ``align-pgdp`` command."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from pdomain_pgdp_measure.cli import DESTINATION_EXIT, VALIDATION_EXIT, build_parser, main


def _write_line_page(path: Path) -> None:
    image = Image.new("L", (120, 100), color=255)
    for row in (10, 30, 50, 70):
        for x in range(20, 100):
            for y in range(row, row + 5):
                image.putpixel((x, y), 0)
    image.save(path)


def build_alignment_fixture(tmp_path: Path) -> tuple[Path, Path]:
    corpus_root = tmp_path / "corpus"
    project = corpus_root / "projectIDone"
    rounds = project / "rounds"
    rounds.mkdir(parents=True)
    (project / "project.json").write_text(
        json.dumps(
            {
                "projectid": "projectIDone",
                "title": "A book",
                "author": "An author",
                "genre": "Fiction",
                "pages_total": 2,
                "pg_ebook_number": 1,
            }
        ),
        encoding="utf-8",
    )
    (rounds / "F2.json").write_text(
        json.dumps({"p10.png": "ten\nten\nten\nten", "p2.png": "two\ntwo\ntwo\ntwo"}),
        encoding="utf-8",
    )
    _write_line_page(project / "p10.png")
    _write_line_page(project / "p2.png")
    ranking_path = tmp_path / "ranking.json"
    assert (
        main(
            [
                "rank",
                str(corpus_root),
                "--output",
                str(ranking_path),
                "--project-limit",
                "1",
                "--pages-per-project",
                "2",
            ]
        )
        == 0
    )
    profile_path = tmp_path / "profile.json"
    assert (
        main(
            [
                "profile",
                str(corpus_root),
                "--ranking",
                str(ranking_path),
                "--output",
                str(profile_path),
            ]
        )
        == 0
    )
    return corpus_root, profile_path


def test_align_pgdp_parser_accepts_the_pinned_interface() -> None:
    args = build_parser().parse_args(
        ["align", "corpus", "--profile", "profile.json", "--output", "alignment.json"]
    )

    assert args.command == "align"
    assert args.corpus_root == "corpus"
    assert args.profile == "profile.json"
    assert args.output == "alignment.json"


@pytest.mark.parametrize("arguments", [[], ["--profile", "profile.json"], ["--output", "out.json"]])
def test_align_pgdp_parser_requires_profile_and_output(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        _ = build_parser().parse_args(["align", "corpus", *arguments])

    assert error.value.code == 2


def test_align_pgdp_writes_snapshot_safe_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)
    _ = capsys.readouterr()
    output = tmp_path / "alignment.json"

    result = main(
        [
            "align",
            str(corpus_root),
            "--profile",
            str(profile_path),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["algorithm_version"] == "pgdp-alignment/v3"
    assert payload["profile_label"] == profile_path.name
    assert payload["profile_sha256"] == sha256(profile_path.read_bytes()).hexdigest()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("profile_contents", ["{", "[]", '{"algorithm_version":"pgdp-profile/v2"}'])
def test_align_pgdp_rejects_malformed_or_unsupported_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], profile_contents: str
) -> None:
    corpus_root, _ = build_alignment_fixture(tmp_path)
    _ = capsys.readouterr()
    profile_path = tmp_path / "invalid-profile.json"
    profile_path.write_text(profile_contents, encoding="utf-8")

    result = main(
        [
            "align",
            str(corpus_root),
            "--profile",
            str(profile_path),
            "--output",
            str(tmp_path / "alignment.json"),
        ]
    )

    assert result == VALIDATION_EXIT
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("output_kind", ["inside_corpus", "equal_profile", "directory"])
def test_align_pgdp_rejects_unsafe_destinations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], output_kind: str
) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)
    _ = capsys.readouterr()
    output = tmp_path / "alignment.json"
    if output_kind == "inside_corpus":
        output = corpus_root / "alignment.json"
    elif output_kind == "equal_profile":
        output = profile_path
    else:
        output.mkdir()

    result = main(
        [
            "align",
            str(corpus_root),
            "--profile",
            str(profile_path),
            "--output",
            str(output),
        ]
    )

    assert result == DESTINATION_EXIT
    assert capsys.readouterr().out == ""
