"""Tests for the local PGDP source-geometry profiling command."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pdomain_ocr_synth.cli import build_parser, main
from PIL import Image

from pdomain_pgdp_measure import image_measurement


def _write_project(corpus_root: Path, *, image_bytes: bytes | None = None) -> None:
    project = corpus_root / "projectIDone"
    project.mkdir(parents=True)
    (project / "project.json").write_text(
        json.dumps(
            {
                "projectid": "projectIDone",
                "title": "A scanned book",
                "author": "An author",
                "genre": "Fiction",
                "pages_total": 1,
                "pg_ebook_number": 1,
            }
        ),
        encoding="utf-8",
    )
    rounds = project / "rounds"
    rounds.mkdir()
    (rounds / "F2.json").write_text(
        json.dumps({"p1.png": "/* <i>Measured text</i> */"}), encoding="utf-8"
    )
    (project / "pages.json").write_text(json.dumps({"p1.png": "Measured text"}), encoding="utf-8")
    image_path = project / "p1.png"
    if image_bytes is not None:
        image_path.write_bytes(image_bytes)
        return
    image = Image.new("L", (40, 30), color=255)
    for x in range(5, 35):
        for y in range(10, 15):
            image.putpixel((x, y), 0)
    image.save(image_path)


def _ranking_for(corpus_root: Path, ranking_path: Path) -> None:
    rc = main(
        [
            "rank-pgdp",
            str(corpus_root),
            "--output",
            str(ranking_path),
            "--project-limit",
            "1",
            "--pages-per-project",
            "1",
        ]
    )
    assert rc == 0


def _prepared_ranking(tmp_path: Path, *, image_bytes: bytes | None = None) -> tuple[Path, Path]:
    corpus_root = tmp_path / "corpus"
    _write_project(corpus_root, image_bytes=image_bytes)
    ranking_path = tmp_path / "ranking.json"
    _ranking_for(corpus_root, ranking_path)
    return corpus_root, ranking_path


def test_profile_pgdp_parser_requires_the_pinned_options() -> None:
    args = build_parser().parse_args(
        ["profile-pgdp", "corpus", "--ranking", "ranking.json", "--output", "profile.json"]
    )

    assert args.command == "profile-pgdp"
    assert args.corpus_root == "corpus"
    assert args.ranking == "ranking.json"
    assert args.output == "profile.json"


@pytest.mark.parametrize(
    "arguments", [[], ["--ranking", "ranking.json"], ["--output", "profile.json"]]
)
def test_profile_pgdp_parser_requires_ranking_and_output(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        _ = build_parser().parse_args(["profile-pgdp", "corpus", *arguments])

    assert error.value.code == 2


def test_profile_pgdp_rejects_a_missing_ranking_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()

    rc = main(
        [
            "profile-pgdp",
            str(corpus_root),
            "--ranking",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "profile.json"),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "valid ranking JSON" in captured.err
    assert captured.out == ""


def test_profile_pgdp_rejects_malformed_ranking_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text("not JSON", encoding="utf-8")

    rc = main(
        [
            "profile-pgdp",
            str(corpus_root),
            "--ranking",
            str(ranking_path),
            "--output",
            str(tmp_path / "profile.json"),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "valid ranking JSON" in captured.err
    assert captured.out == ""


def test_profile_pgdp_keeps_a_malformed_scan_as_a_diagnostic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_root, ranking_path = _prepared_ranking(tmp_path, image_bytes=b"not an image")
    output = tmp_path / "profile.json"

    rc = main(
        [
            "profile-pgdp",
            str(corpus_root),
            "--ranking",
            str(ranking_path),
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    page = payload["projects"][0]["pages"][0]
    assert rc == 0
    assert [item["code"] for item in page["diagnostics"]] == ["image_decode_failed"]
    assert page["source_path"] == "projectIDone/p1.png"
    assert str(corpus_root) not in output.read_text(encoding="utf-8")
    summary = capsys.readouterr().out
    assert "pages measured: 0" in summary
    assert "pages excluded: 1" in summary
    assert "diagnostics: 1" in summary


def test_profile_pgdp_aborts_when_temporary_scan_snapshot_storage_is_full(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_root, ranking_path = _prepared_ranking(tmp_path)
    _ = capsys.readouterr()
    output = tmp_path / "profile.json"

    class WriteFailingSnapshot:
        def __enter__(self) -> WriteFailingSnapshot:
            return self

        def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
            return None

        def close(self) -> None:
            return None

        def seek(self, _offset: int, _whence: int = 0) -> int:
            return 0

        def write(self, _data: bytes) -> int:
            raise OSError("No space left on device")

    def temporary_file_full(*_args: object, **_kwargs: object) -> WriteFailingSnapshot:
        return WriteFailingSnapshot()

    monkeypatch.setattr(image_measurement, "TemporaryFile", temporary_file_full)

    rc = main(
        [
            "profile-pgdp",
            str(corpus_root),
            "--ranking",
            str(ranking_path),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 6
    assert "temporary scan snapshot" in captured.err
    assert captured.out == ""
    assert not output.exists()


@pytest.mark.parametrize("destination", ["inside-corpus", "ranking-input", "directory"])
def test_profile_pgdp_rejects_unsafe_destinations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], destination: str
) -> None:
    corpus_root, ranking_path = _prepared_ranking(tmp_path)
    _ = capsys.readouterr()
    if destination == "inside-corpus":
        output = corpus_root / "profile.json"
    elif destination == "ranking-input":
        output = ranking_path
    else:
        output = tmp_path / "output-directory"
        output.mkdir()

    rc = main(
        [
            "profile-pgdp",
            str(corpus_root),
            "--ranking",
            str(ranking_path),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 6
    assert "output" in captured.err
    assert captured.out == ""


def test_profile_pgdp_preserves_existing_output_after_interrupted_replacement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_root, ranking_path = _prepared_ranking(tmp_path)
    output = tmp_path / "profile.json"
    output.write_text("existing report\n", encoding="utf-8")

    def interrupted_replace(_source: Path, _destination: Path) -> Path:
        raise OSError("interrupted replacement")

    monkeypatch.setattr(Path, "replace", interrupted_replace)

    rc = main(
        [
            "profile-pgdp",
            str(corpus_root),
            "--ranking",
            str(ranking_path),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 6
    assert "interrupted replacement" in captured.err
    assert output.read_text(encoding="utf-8") == "existing report\n"
    assert list(tmp_path.glob(".profile.json.*.tmp")) == []


def test_profile_pgdp_reports_both_replacement_and_cleanup_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_root, ranking_path = _prepared_ranking(tmp_path)
    output = tmp_path / "profile.json"

    def interrupted_replace(_source: Path, _destination: Path) -> Path:
        raise OSError("interrupted replacement")

    def interrupted_cleanup(_path: Path, *, missing_ok: bool = False) -> None:
        _ = missing_ok
        raise OSError("interrupted cleanup")

    monkeypatch.setattr(Path, "replace", interrupted_replace)
    monkeypatch.setattr(Path, "unlink", interrupted_cleanup)

    rc = main(
        [
            "profile-pgdp",
            str(corpus_root),
            "--ranking",
            str(ranking_path),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 6
    assert "Writing the report" in captured.err
    assert captured.out.endswith("selected pages: 1\n")


def test_profile_pgdp_is_byte_deterministic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus_root, ranking_path = _prepared_ranking(tmp_path)
    output_a = tmp_path / "profile-a.json"
    output_b = tmp_path / "profile-b.json"

    first = main(
        [
            "profile-pgdp",
            str(corpus_root),
            "--ranking",
            str(ranking_path),
            "--output",
            str(output_a),
        ]
    )
    second = main(
        [
            "profile-pgdp",
            str(corpus_root),
            "--ranking",
            str(ranking_path),
            "--output",
            str(output_b),
        ]
    )

    assert first == second == 0
    assert output_a.read_bytes() == output_b.read_bytes()
    assert capsys.readouterr().err == ""


def test_profile_pgdp_hashes_and_parses_one_ranking_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_root, ranking_path = _prepared_ranking(tmp_path)
    output = tmp_path / "profile.json"
    original_read_bytes = Path.read_bytes
    snapshot = ranking_path.read_bytes()
    reads = 0

    def read_snapshot_then_mutate(path: Path) -> bytes:
        nonlocal reads
        if path == ranking_path:
            reads += 1
            result = original_read_bytes(path)
            path.write_text("not JSON", encoding="utf-8")
            return result
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_snapshot_then_mutate)

    rc = main(
        [
            "profile-pgdp",
            str(corpus_root),
            "--ranking",
            str(ranking_path),
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert rc == 0
    assert reads == 1
    assert payload["source_ranking"]["sha256"] == hashlib.sha256(snapshot).hexdigest()
    assert "pages measured: 1" in capsys.readouterr().out


def _add_extra_page(corpus_root: Path, name: str, *, top: int) -> None:
    """Add a page image that the ranking will not select."""

    image = Image.new("L", (40, 30), color=255)
    for x in range(5, 35):
        for y in range(top, top + 5):
            image.putpixel((x, y), 0)
    image.save(corpus_root / "projectIDone" / name)


def test_whole_book_pools_every_page_but_emits_only_the_ranked_page(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    _write_project(corpus_root)
    _add_extra_page(corpus_root, "p2.png", top=12)
    _add_extra_page(corpus_root, "p3.png", top=14)
    ranking_path = tmp_path / "ranking.json"
    _ranking_for(corpus_root, ranking_path)
    output = tmp_path / "profile.json"

    rc = main(
        [
            "profile-pgdp",
            str(corpus_root),
            "--ranking",
            str(ranking_path),
            "--output",
            str(output),
            "--whole-book",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    project = payload["projects"][0]
    assert [page["page_name"] for page in project["pages"]] == ["p1.png"]
    pooled = {estimate["name"]: estimate for estimate in project["pooled_estimates"]}
    assert pooled["margin_top_px"]["sample_count"] == 3
    assert pooled["margin_top_px"]["evidence_pages"] == ["p1.png", "p2.png", "p3.png"]


def test_default_mode_pools_only_the_ranked_page(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    _write_project(corpus_root)
    _add_extra_page(corpus_root, "p2.png", top=12)
    ranking_path = tmp_path / "ranking.json"
    _ranking_for(corpus_root, ranking_path)
    output = tmp_path / "profile.json"

    rc = main(
        [
            "profile-pgdp",
            str(corpus_root),
            "--ranking",
            str(ranking_path),
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    project = payload["projects"][0]
    assert [page["page_name"] for page in project["pages"]] == ["p1.png"]
    pooled = {estimate["name"]: estimate for estimate in project["pooled_estimates"]}
    assert pooled["margin_top_px"]["sample_count"] == 1
    assert pooled["margin_top_px"]["evidence_pages"] == ["p1.png"]
