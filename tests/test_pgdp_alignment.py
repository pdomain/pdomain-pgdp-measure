"""Tests for snapshot-safe PGDP source-line alignment orchestration."""

from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from shutil import copytree, rmtree
from typing import TYPE_CHECKING

import pytest
from PIL import Image
from pydantic import TypeAdapter

from pdomain_pgdp_measure import __version__, alignment
from pdomain_pgdp_measure.alignment import build_alignment_report
from pdomain_pgdp_measure.cli import main
from pdomain_pgdp_measure.image_measurement import open_image_snapshot
from pdomain_pgdp_measure.paths import (
    ImageResolution,
    resolve_image_candidate,
    resolve_project_directory,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import TypeGuard

    from pdomain_pgdp_measure.image_measurement import ImageSnapshot


_PROFILE_PAYLOAD_ADAPTER = TypeAdapter(dict[str, object])


def test_build_alignment_report_is_public() -> None:
    assert callable(build_alignment_report)


def _write_line_page(path: Path) -> None:
    image = Image.new("L", (120, 100), color=255)
    for row in (10, 30, 50, 70):
        for x in range(20, 100):
            for y in range(row, row + 5):
                image.putpixel((x, y), 0)
    image.save(path)


def _build_profile(corpus_root: Path, profile_path: Path) -> None:
    ranking_path = profile_path.with_name("ranking.json")
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


def build_alignment_fixture(tmp_path: Path) -> tuple[Path, Path]:
    corpus_root = tmp_path / "corpus"
    project = corpus_root / "projectIDone"
    rounds = project / "rounds"
    rounds.mkdir(parents=True)
    _ = (project / "project.json").write_text(
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
    _ = (rounds / "F2.json").write_text(
        json.dumps(
            {
                "p10.png": "ten\nten\nten\nten",
                "p2.png": "two\ntwo\ntwo\ntwo",
            }
        ),
        encoding="utf-8",
    )
    _write_line_page(project / "p10.png")
    _write_line_page(project / "p2.png")
    profile_path = tmp_path / "profile.json"
    _build_profile(corpus_root, profile_path)
    return corpus_root, profile_path


def test_build_alignment_report_keeps_natural_page_order(tmp_path: Path) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)

    report = build_alignment_report(corpus_root, profile_path, tool_version=__version__)

    assert [page.page_name for page in report.projects[0].pages] == ["p2.png", "p10.png"]
    assert report.profile_label == "profile.json"


def test_build_alignment_report_excludes_profile_scan_hash_mismatches(tmp_path: Path) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)
    payload = _read_profile_payload(profile_path)
    _profile_page(payload, "p2.png")["sha256"] = "0" * 64
    _ = profile_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_alignment_report(corpus_root, profile_path, tool_version=__version__)

    page = report.projects[0].pages[0]
    assert page.exclusions == ("scan_hash_mismatch",)
    assert page.state == "excluded"


def test_build_alignment_report_discards_project_evidence_when_f2_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)
    f2_path = corpus_root / "projectIDone" / "rounds" / "F2.json"
    original_read = Path.read_bytes
    reads = 0

    def read_then_change(path: Path) -> bytes:
        nonlocal reads
        if path == f2_path:
            reads += 1
            return original_read(path) if reads == 1 else b'{"p2.png":"changed"}'
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read_then_change)

    report = build_alignment_report(corpus_root, profile_path, tool_version=__version__)

    assert reads == 2
    assert all(page.exclusions == ("source_changed",) for page in report.projects[0].pages)
    assert all(not page.candidates and not page.operations for page in report.projects[0].pages)


def test_build_alignment_report_prefers_source_changed_after_profile_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)
    payload = _read_profile_payload(profile_path)
    _profile_page(payload, "p2.png")["sha256"] = "0" * 64
    _ = profile_path.write_text(json.dumps(payload), encoding="utf-8")
    original_open_snapshot = open_image_snapshot

    @contextmanager
    def snapshot_then_change(path: Path) -> Generator[ImageSnapshot]:
        with original_open_snapshot(path) as snapshot:
            yield snapshot
            _ = path.write_bytes(b"changed after snapshot")

    monkeypatch.setattr(alignment, "open_image_snapshot", snapshot_then_change)

    report = build_alignment_report(corpus_root, profile_path, tool_version=__version__)

    assert report.projects[0].pages[0].exclusions == ("source_changed",)


def test_build_alignment_report_rejects_profile_scan_paths_outside_page_candidates(
    tmp_path: Path,
) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)
    _ = copytree(corpus_root / "projectIDone", corpus_root / "projectIDtwo")
    payload = _read_profile_payload(profile_path)
    _profile_page(payload, "p2.png")["source_path"] = "projectIDtwo/p2.png"
    _ = profile_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_alignment_report(corpus_root, profile_path, tool_version=__version__)

    page = report.projects[0].pages[0]
    assert page.exclusions == ("image_missing",)
    assert page.diagnostics[0].code == "source_path_mismatch"


def test_build_alignment_report_sorts_unordered_profile_projects(tmp_path: Path) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)
    _add_second_project_to_profile(corpus_root, profile_path)

    report = build_alignment_report(corpus_root, profile_path, tool_version=__version__)

    assert [project.project_id for project in report.projects] == ["projectIDone", "projectIDtwo"]


def test_build_alignment_report_continues_when_one_project_disappears_after_f2_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)
    _add_second_project_to_profile(corpus_root, profile_path)
    project_one_checks = 0

    def project_directory_then_disappear(root: Path, project_id: str) -> Path:
        nonlocal project_one_checks
        if project_id == "projectIDone":
            project_one_checks += 1
            if project_one_checks > 1:
                raise ValueError("Project directory disappeared after F2 snapshot.")
        return resolve_project_directory(corpus_root=root, project_id=project_id)

    monkeypatch.setattr(alignment, "_project_directory", project_directory_then_disappear)

    report = build_alignment_report(corpus_root, profile_path, tool_version=__version__)

    assert [project.project_id for project in report.projects] == ["projectIDone", "projectIDtwo"]
    assert all(page.exclusions == ("image_missing",) for page in report.projects[0].pages)
    assert all(page.exclusions != ("image_missing",) for page in report.projects[1].pages)
    assert all(
        page.diagnostics[0].code == "scan_resolution_failed" for page in report.projects[0].pages
    )


def test_build_alignment_report_excludes_candidate_that_moves_outside_corpus_during_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)
    outside_scan = tmp_path / "outside.png"
    _write_line_page(outside_scan)

    def resolve_then_move_candidate(
        *, project_directory: Path, page_name: str, corpus_root: Path
    ) -> ImageResolution:
        resolution = resolve_image_candidate(
            project_directory=project_directory,
            page_name=page_name,
            corpus_root=corpus_root,
        )
        if page_name == "p2.png" and resolution.image_path is not None:
            resolution.image_path.unlink()
            _ = resolution.image_path.symlink_to(outside_scan)
        return resolution

    monkeypatch.setattr(alignment, "resolve_image_candidate", resolve_then_move_candidate)

    report = build_alignment_report(corpus_root, profile_path, tool_version=__version__)

    page = report.projects[0].pages[0]
    assert page.exclusions == ("image_missing",)
    assert page.diagnostics[0].code == "scan_resolution_failed"


def test_build_alignment_report_continues_after_a_scan_symlink_loop(tmp_path: Path) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)
    _add_second_project_to_profile(corpus_root, profile_path)
    looping_scan = corpus_root / "projectIDone" / "p2.png"
    looping_scan.unlink()
    _ = looping_scan.symlink_to("p2.png")

    report = build_alignment_report(corpus_root, profile_path, tool_version=__version__)

    first_page = next(page for page in report.projects[0].pages if page.page_name == "p2.png")
    second_page = next(page for page in report.projects[1].pages if page.page_name == "p2.png")
    assert first_page.exclusions == ("image_missing",)
    assert first_page.diagnostics[0].code == "scan_resolution_failed"
    assert second_page.exclusions != ("image_missing",)


def test_build_alignment_report_excludes_profile_pages_absent_from_f2(tmp_path: Path) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)
    payload = _read_profile_payload(profile_path)
    missing = deepcopy(_profile_page(payload, "p2.png"))
    missing["page_name"] = "missing.png"
    pages = _profile_project(payload).get("pages")
    if not _is_json_list(pages):
        raise AssertionError("Expected profile pages.")
    pages.append(missing)
    _ = profile_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_alignment_report(corpus_root, profile_path, tool_version=__version__)

    missing_page = next(
        page for page in report.projects[0].pages if page.page_name == "missing.png"
    )
    assert missing_page.exclusions == ("f2_page_missing",)


@pytest.mark.parametrize("failure", ["missing_project", "unreadable_f2", "malformed_f2"])
def test_build_alignment_report_keeps_profile_pages_when_project_f2_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)
    project_directory = corpus_root / "projectIDone"
    f2_path = project_directory / "rounds" / "F2.json"
    if failure == "missing_project":
        rmtree(project_directory)
    elif failure == "unreadable_f2":
        original_read_bytes = Path.read_bytes

        def unreadable_f2(path: Path) -> bytes:
            if path == f2_path:
                raise OSError("F2.json cannot be read")
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", unreadable_f2)
    else:
        _ = f2_path.write_text("{", encoding="utf-8")

    report = build_alignment_report(corpus_root, profile_path, tool_version=__version__)

    assert [project.project_id for project in report.projects] == ["projectIDone"]
    pages = report.projects[0].pages
    assert [page.page_name for page in pages] == ["p2.png", "p10.png"]
    assert all(page.exclusions == ("f2_missing",) for page in pages)
    assert all(
        [(diagnostic.code, diagnostic.page_name) for diagnostic in page.diagnostics]
        == [("malformed_f2", page.page_name)]
        for page in pages
    )
    expected_f2_sha256 = sha256(b"{").hexdigest() if failure == "malformed_f2" else None
    assert {page.f2_sha256 for page in pages} == {expected_f2_sha256}
    assert [diagnostic.code for diagnostic in report.projects[0].diagnostics] == ["malformed_f2"]
    assert [diagnostic.code for diagnostic in report.diagnostics] == ["malformed_f2"]


def test_build_alignment_report_keeps_available_f2_hash(tmp_path: Path) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)
    f2_path = corpus_root / "projectIDone" / "rounds" / "F2.json"

    report = build_alignment_report(corpus_root, profile_path, tool_version=__version__)

    assert {page.f2_sha256 for page in report.projects[0].pages} == {
        sha256(f2_path.read_bytes()).hexdigest()
    }


@pytest.mark.parametrize(
    ("profile_code", "expected_exclusion"),
    [
        ("blank_page", "blank_page"),
        ("one_ink_band", "insufficient_ink_bands"),
        ("no_ink_bands", "insufficient_ink_bands"),
        ("border_dominated", "border_dominated"),
        ("high_foreground_ratio", "high_foreground_ratio"),
        ("image_missing", "image_missing"),
        ("image_unreadable", "image_unreadable"),
        ("image_decode_failed", "image_decode_failed"),
        ("foreground_bounds_unavailable", "foreground_bounds_unavailable"),
        ("", "inherited_profile_diagnostic"),
        ("future_profile_code", "inherited_profile_diagnostic"),
    ],
)
def test_build_alignment_report_maps_every_inherited_profile_diagnostic_before_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_code: str,
    expected_exclusion: str,
) -> None:
    corpus_root, profile_path = build_alignment_fixture(tmp_path)
    payload = _read_profile_payload(profile_path)
    page_payload = _profile_page(payload, "p2.png")
    page_payload["diagnostics"] = [
        {
            "code": profile_code,
            "message": "Inherited issue",
            "project_id": "projectIDone",
            "page_name": "p2.png",
        }
    ]
    _profile_project(payload)["pages"] = [page_payload]
    _ = profile_path.write_text(json.dumps(payload), encoding="utf-8")

    def candidates_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("candidate extraction must not run for inherited exclusions")

    monkeypatch.setattr(alignment, "extract_line_candidates", candidates_must_not_run)

    report = build_alignment_report(corpus_root, profile_path, tool_version=__version__)

    page = report.projects[0].pages[0]
    assert page.exclusions == (expected_exclusion,)
    if expected_exclusion == "inherited_profile_diagnostic":
        assert page.exclusion_evidence == ({"code": profile_code, "message": "Inherited issue"},)


def _profile_page(payload: object, page_name: str) -> dict[str, object]:
    project = _profile_project(payload)
    pages = project.get("pages")
    if not _is_json_list(pages):
        raise AssertionError("Expected profile pages.")
    for page in pages:
        if _is_json_object(page) and page.get("page_name") == page_name:
            return page
    raise AssertionError(f"Could not find profile page: {page_name}")


def _profile_project(payload: object) -> dict[str, object]:
    if not _is_json_object(payload):
        raise AssertionError("Expected profile JSON object.")
    projects = payload.get("projects")
    if not _is_json_list(projects) or not projects:
        raise AssertionError("Expected at least one profile project.")
    project = projects[0]
    if not _is_json_object(project):
        raise AssertionError("Expected profile project object.")
    return project


def _read_profile_payload(path: Path) -> dict[str, object]:
    return _PROFILE_PAYLOAD_ADAPTER.validate_json(path.read_bytes())


def _add_second_project_to_profile(corpus_root: Path, profile_path: Path) -> None:
    first_project = corpus_root / "projectIDone"
    second_project = corpus_root / "projectIDtwo"
    _ = copytree(first_project, second_project)
    payload = _read_profile_payload(profile_path)
    first = _profile_project(payload)
    second = deepcopy(first)
    second["project_id"] = "projectIDtwo"
    second_pages = second.get("pages")
    if not _is_json_list(second_pages):
        raise AssertionError("Expected profile pages.")
    for page in second_pages:
        if not _is_json_object(page):
            raise AssertionError("Expected profile page object.")
        source_path = page.get("source_path")
        if not isinstance(source_path, str):
            raise AssertionError("Expected profile source_path.")
        page["source_path"] = source_path.replace("projectIDone", "projectIDtwo")
    payload["projects"] = [second, first]
    _ = profile_path.write_text(json.dumps(payload), encoding="utf-8")


def _is_json_object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _is_json_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)
