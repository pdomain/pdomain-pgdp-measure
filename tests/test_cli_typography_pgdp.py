"""Tests for the ``typography-pgdp`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdomain_pgdp_measure.cli import (
    _TYPOGRAPHY_EVIDENCE_PAGES_DEFAULT,
    DESTINATION_EXIT,
    USAGE_EXIT,
    VALIDATION_EXIT,
    build_parser,
    main,
)
from pdomain_pgdp_measure.typography_models import DEFAULT_EVIDENCE_PAGES_PER_BOOK
from tests.test_pgdp_typography import Fixture, build_fixture


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return build_fixture(tmp_path_factory.mktemp("typography-cli"))


def _run(fixture: Fixture, output: Path, *extra: str) -> int:
    return main(
        [
            "typography",
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
            "typography",
            "corpus",
            "--alignment",
            "alignment.json",
            "--profile",
            "profile.json",
            "--output",
            "typography.json",
            "--evidence-pages",
            "3",
        ]
    )

    assert args.command == "typography"
    assert args.corpus_root == "corpus"
    assert args.alignment == "alignment.json"
    assert args.profile == "profile.json"
    assert args.output == "typography.json"
    assert args.evidence_pages == 3


def test_evidence_pages_defaults_to_twelve() -> None:
    args = build_parser().parse_args(
        [
            "typography",
            "corpus",
            "--alignment",
            "a.json",
            "--profile",
            "p.json",
            "--output",
            "t.json",
        ]
    )

    assert args.evidence_pages == DEFAULT_EVIDENCE_PAGES_PER_BOOK
    assert _TYPOGRAPHY_EVIDENCE_PAGES_DEFAULT == DEFAULT_EVIDENCE_PAGES_PER_BOOK


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--alignment", "a.json"],
        ["--alignment", "a.json", "--profile", "p.json"],
        ["--profile", "p.json", "--output", "t.json"],
    ],
)
def test_parser_requires_every_input(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        _ = build_parser().parse_args(["typography", "corpus", *arguments])

    assert error.value.code == 2


def test_writes_a_deterministic_report_and_prints_no_json(
    fixture: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "typography.json"
    second = tmp_path / "typography-again.json"

    assert _run(fixture, first) == 0
    _ = capsys.readouterr()
    assert _run(fixture, second) == 0
    captured = capsys.readouterr()

    assert first.read_bytes() == second.read_bytes()
    assert captured.out == ""
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["algorithm_version"] == "pgdp-typography/v1"
    assert payload["schema_version"] == 1


def test_report_is_written_outside_the_corpus_root(fixture: Fixture) -> None:
    inside = fixture.corpus_root / "typography.json"

    assert _run(fixture, inside) == DESTINATION_EXIT
    assert not inside.exists()


def test_a_missing_corpus_root_is_a_usage_error(fixture: Fixture, tmp_path: Path) -> None:
    result = main(
        [
            "typography",
            str(tmp_path / "absent"),
            "--alignment",
            str(fixture.alignment_path),
            "--profile",
            str(fixture.profile_path),
            "--output",
            str(tmp_path / "typography.json"),
        ]
    )

    assert result == USAGE_EXIT


def test_a_negative_evidence_page_count_is_a_usage_error(fixture: Fixture, tmp_path: Path) -> None:
    assert _run(fixture, tmp_path / "typography.json", "--evidence-pages", "-1") == USAGE_EXIT


def test_output_may_not_overwrite_an_input(fixture: Fixture) -> None:
    assert _run(fixture, fixture.profile_path) == DESTINATION_EXIT
    assert _run(fixture, fixture.alignment_path) == DESTINATION_EXIT


def test_a_mismatched_profile_is_a_validation_error(
    fixture: Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tampered = tmp_path / "profile.json"
    payload = json.loads(fixture.profile_path.read_text(encoding="utf-8"))
    payload["projects"][0]["title"] = "A different book"
    tampered.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = main(
        [
            "typography",
            str(fixture.corpus_root),
            "--alignment",
            str(fixture.alignment_path),
            "--profile",
            str(tampered),
            "--output",
            str(tmp_path / "typography.json"),
        ]
    )

    assert result == VALIDATION_EXIT
    assert "recorded profile_sha256" in capsys.readouterr().err


def test_an_unreadable_alignment_report_is_a_validation_error(
    fixture: Fixture, tmp_path: Path
) -> None:
    result = main(
        [
            "typography",
            str(fixture.corpus_root),
            "--alignment",
            str(tmp_path / "absent.json"),
            "--profile",
            str(fixture.profile_path),
            "--output",
            str(tmp_path / "typography.json"),
        ]
    )

    assert result == VALIDATION_EXIT


def test_evidence_pages_bounds_the_emitted_rows(fixture: Fixture, tmp_path: Path) -> None:
    output = tmp_path / "typography.json"

    assert _run(fixture, output, "--evidence-pages", "1") == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    book = payload["books"][0]
    assert sum(page["evidence_sampled"] for page in book["pages"]) == 1
    assert book["measured_page_count"] == 2


def test_geometry_defaults_to_absent(fixture: Fixture, tmp_path: Path) -> None:
    parsed = build_parser().parse_args(
        [
            "typography",
            str(fixture.corpus_root),
            "--alignment",
            str(fixture.alignment_path),
            "--profile",
            str(fixture.profile_path),
            "--output",
            str(tmp_path / "typography.json"),
        ]
    )

    assert parsed.geometry is None


def test_geometry_output_must_differ_from_its_input(fixture: Fixture, tmp_path: Path) -> None:
    geometry = tmp_path / "geometry.jsonl"
    _ = geometry.write_text("\n", encoding="utf-8")

    result = _run(fixture, geometry, "--geometry", str(geometry))

    assert result == DESTINATION_EXIT


def test_an_unreadable_geometry_record_is_a_validation_error(
    fixture: Fixture, tmp_path: Path
) -> None:
    result = _run(
        fixture, tmp_path / "typography.json", "--geometry", str(tmp_path / "absent.jsonl")
    )

    assert result == VALIDATION_EXIT
