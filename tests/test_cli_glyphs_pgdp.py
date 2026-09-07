"""Tests for the ``glyphs-pgdp`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdomain_pgdp_measure.cli import (
    DESTINATION_EXIT,
    USAGE_EXIT,
    VALIDATION_EXIT,
    build_parser,
    main,
)
from pdomain_pgdp_measure.glyph_models import GlyphManifest, read_rows
from tests.test_pgdp_glyphs import Fixture, build_glyph_fixture


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return build_glyph_fixture(tmp_path_factory.mktemp("glyphs-cli"))


def _run(fixture: Fixture, output: Path, *extra: str) -> int:
    return main(
        [
            "glyphs",
            str(fixture.corpus_root),
            "--alignment",
            str(fixture.alignment_path),
            "--profile",
            str(fixture.profile_path),
            "--output",
            str(output),
            *extra,
        ]
    )


def test_parser_accepts_the_pinned_interface() -> None:
    args = build_parser().parse_args(
        [
            "glyphs",
            "corpus",
            "--alignment",
            "alignment.json",
            "--profile",
            "profile.json",
            "--output",
            "glyphs/",
            "--geometry",
            "geometry.jsonl",
        ]
    )

    assert args.command == "glyphs"
    assert args.corpus_root == "corpus"
    assert args.alignment == "alignment.json"
    assert args.profile == "profile.json"
    assert args.output == "glyphs/"
    assert args.geometry == "geometry.jsonl"


def test_geometry_defaults_to_off() -> None:
    args = build_parser().parse_args(
        ["glyphs", "corpus", "--alignment", "a.json", "--profile", "p.json", "--output", "g"]
    )

    assert args.geometry is None


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--alignment", "a.json"],
        ["--alignment", "a.json", "--profile", "p.json"],
    ],
)
def test_the_three_required_arguments_are_required(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        _ = build_parser().parse_args(["glyphs", "corpus", *arguments])


def test_a_missing_corpus_root_is_a_usage_error(fixture: Fixture, tmp_path: Path) -> None:
    assert (
        main(
            [
                "glyphs",
                str(tmp_path / "absent"),
                "--alignment",
                str(fixture.alignment_path),
                "--profile",
                str(fixture.profile_path),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == USAGE_EXIT
    )


def test_a_corpus_root_that_is_a_file_is_a_usage_error(fixture: Fixture, tmp_path: Path) -> None:
    not_a_directory = tmp_path / "corpus.txt"
    not_a_directory.write_text("", encoding="utf-8")
    assert (
        main(
            [
                "glyphs",
                str(not_a_directory),
                "--alignment",
                str(fixture.alignment_path),
                "--profile",
                str(fixture.profile_path),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == USAGE_EXIT
    )


def test_an_output_naming_a_file_is_a_destination_error(fixture: Fixture, tmp_path: Path) -> None:
    output = tmp_path / "not-a-directory"
    output.write_text("", encoding="utf-8")
    assert _run(fixture, output) == DESTINATION_EXIT


def test_an_unreadable_alignment_is_a_validation_error(fixture: Fixture, tmp_path: Path) -> None:
    assert (
        main(
            [
                "glyphs",
                str(fixture.corpus_root),
                "--alignment",
                str(tmp_path / "absent.json"),
                "--profile",
                str(fixture.profile_path),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == VALIDATION_EXIT
    )


def test_a_profile_that_does_not_match_the_alignment_is_a_validation_error(
    fixture: Fixture, tmp_path: Path
) -> None:
    tampered = tmp_path / "profile.json"
    payload = json.loads(fixture.profile_path.read_text(encoding="utf-8"))
    payload["tool_version"] = "tampered"
    tampered.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert (
        main(
            [
                "glyphs",
                str(fixture.corpus_root),
                "--alignment",
                str(fixture.alignment_path),
                "--profile",
                str(tampered),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == VALIDATION_EXIT
    )


def test_a_run_writes_a_readable_inventory(fixture: Fixture, tmp_path: Path) -> None:
    output = tmp_path / "inventory"
    assert _run(fixture, output, "--geometry", str(fixture.geometry_path)) == 0
    manifest = GlyphManifest.from_json((output / "manifest.json").read_bytes())
    rows = read_rows((output / "glyphs.jsonl").read_bytes())
    assert manifest.glyph_count == len(rows)
    assert manifest.project_id == "projectIDone"
    assert set(manifest.glyph_count_by_tier) == {"transcribed", "recognized"}


def test_a_run_without_geometry_harvests_only_the_transcribed_tier(
    fixture: Fixture, tmp_path: Path
) -> None:
    output = tmp_path / "inventory"
    assert _run(fixture, output) == 0
    manifest = GlyphManifest.from_json((output / "manifest.json").read_bytes())
    assert set(manifest.glyph_count_by_tier) == {"transcribed"}
    assert manifest.geometry_label is None


def test_an_output_inside_the_corpus_is_a_destination_error(fixture: Fixture) -> None:
    assert _run(fixture, fixture.corpus_root / "inventory") == DESTINATION_EXIT


def test_the_command_reports_what_it_cut(
    fixture: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(fixture, tmp_path / "inventory") == 0
    printed = capsys.readouterr().out
    assert "glyphs:" in printed
    assert "transcribed:" in printed
    assert "pages harvested: 1/2" in printed
