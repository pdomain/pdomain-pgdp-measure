from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from pdomain_pgdp_measure.models import RankingReportData
from pdomain_pgdp_measure.paths import (
    UnsafePathError,
    corpus_relative_path,
    require_canonical_relative_reference,
    resolve_image_candidate,
    resolve_project_directory,
)

if TYPE_CHECKING:
    from pdomain_pgdp_measure.models import RankedPageData

_SCHEMA_VERSION = 1
_ALGORITHM_VERSION = "pgdp-rank/v1"
_RANKING_REPORT_ADAPTER = TypeAdapter(RankingReportData)


@dataclass(frozen=True, slots=True)
class ProfileInputPage:
    """One selected page with its current safe image resolution."""

    name: str
    image_path: Path | None
    source_path: str | None
    emit: bool = True
    """Whether the page appears in the emitted report.

    Whole-book mode measures every page of a book so its templates rest on the
    whole volume, but emits only the pages the M14 ranking selected. Alignment
    costs about a hundred times more per page than profiling, and the fixed
    review selection must not move.
    """


@dataclass(frozen=True, slots=True)
class ProfileInputProject:
    """One selected report project in report order."""

    project_id: str
    title: str | None
    author: str | None
    genre: str | None
    pages: tuple[ProfileInputPage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pages", tuple(self.pages))


@dataclass(frozen=True, slots=True)
class ProfileInput:
    """Validated ranking selection ready for scan measurement."""

    projects: tuple[ProfileInputProject, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "projects", tuple(self.projects))


def load_profile_input(
    ranking_path: str | Path, *, corpus_root: str | Path, whole_book: bool = False
) -> ProfileInput:
    """Validate an M14 report and resolve its selected images safely."""

    ranking_snapshot = read_profile_snapshot(ranking_path)
    return load_profile_snapshot(ranking_snapshot, corpus_root=corpus_root, whole_book=whole_book)


def read_profile_snapshot(ranking_path: str | Path) -> bytes:
    """Read one stable ranking-report byte snapshot."""

    try:
        return Path(ranking_path).read_bytes()
    except OSError as error:
        raise ValueError("Ranking report must be valid ranking JSON.") from error


def load_profile_snapshot(
    ranking_snapshot: bytes, *, corpus_root: str | Path, whole_book: bool = False
) -> ProfileInput:
    """Validate one immutable M14 report snapshot and resolve its selected images safely."""

    payload = _load_ranking_payload(ranking_snapshot)
    _validate_version(payload)
    root = Path(corpus_root)
    project_ids: set[str] = set()
    projects: list[ProfileInputProject] = []
    for project_payload in payload["projects"]:
        project_id = project_payload["project_id"]
        if project_id in project_ids:
            raise ValueError(f"Duplicate project_id in ranking report: {project_id!r}.")
        project_ids.add(project_id)
        project_directory = resolve_project_directory(corpus_root=root, project_id=project_id)
        pages = _load_project_pages(
            project_id=project_id,
            page_payloads=project_payload["pages"],
            project_directory=project_directory,
            corpus_root=root,
        )
        if whole_book:
            pages = _with_unranked_pages(
                ranked=pages,
                project_directory=project_directory,
                corpus_root=root,
            )
        projects.append(
            ProfileInputProject(
                project_id=project_id,
                title=project_payload["title"],
                author=project_payload["author"],
                genre=project_payload["genre"],
                pages=pages,
            )
        )
    return ProfileInput(projects=tuple(projects))


def _with_unranked_pages(
    *,
    ranked: tuple[ProfileInputPage, ...],
    project_directory: Path,
    corpus_root: Path,
) -> tuple[ProfileInputPage, ...]:
    """Add every unranked page image so book statistics rest on the whole book.

    Ranked pages keep `emit` set, so the emitted report still holds exactly the
    M14 selection. The added pages are measured and pooled, then dropped.
    """

    ranked_names = {page.name for page in ranked}
    extra: list[ProfileInputPage] = []
    for image_path in sorted(project_directory.glob("*.png")):
        page_name = image_path.name
        if page_name in ranked_names:
            continue
        resolution = resolve_image_candidate(
            project_directory=project_directory,
            page_name=page_name,
            corpus_root=corpus_root,
        )
        if resolution.unsafe_candidate or resolution.image_path is None:
            continue
        extra.append(
            ProfileInputPage(
                name=page_name,
                image_path=resolution.image_path,
                source_path=corpus_relative_path(
                    path=resolution.image_path, corpus_root=corpus_root
                ),
                emit=False,
            )
        )
    return tuple(sorted(ranked + tuple(extra), key=lambda page: page.name))


def _load_ranking_payload(ranking_snapshot: bytes) -> RankingReportData:
    try:
        return _RANKING_REPORT_ADAPTER.validate_json(ranking_snapshot, strict=True)
    except ValidationError as error:
        raise ValueError("Ranking report must be valid ranking JSON.") from error


def _validate_version(payload: RankingReportData) -> None:
    if payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("Ranking report must use schema version 1.")
    if payload["algorithm_version"] != _ALGORITHM_VERSION:
        raise ValueError("Ranking report must use algorithm pgdp-rank/v1.")


def _load_project_pages(
    *,
    project_id: str,
    page_payloads: list[RankedPageData],
    project_directory: Path,
    corpus_root: Path,
) -> tuple[ProfileInputPage, ...]:
    page_names: set[str] = set()
    pages: list[ProfileInputPage] = []
    for page_payload in page_payloads:
        page_name = page_payload["name"]
        _ = require_canonical_relative_reference(value=page_name, label="page reference")
        if page_name in page_names:
            raise ValueError(f"Duplicate page name for project {project_id!r}: {page_name!r}.")
        page_names.add(page_name)
        resolution = resolve_image_candidate(
            project_directory=project_directory,
            page_name=page_name,
            corpus_root=corpus_root,
        )
        if resolution.unsafe_candidate:
            raise UnsafePathError(
                f"Page image candidate escapes the project or corpus: {page_name!r}."
            )
        source_path = (
            corpus_relative_path(path=resolution.image_path, corpus_root=corpus_root)
            if resolution.image_path is not None
            else None
        )
        pages.append(
            ProfileInputPage(
                name=page_name,
                image_path=resolution.image_path,
                source_path=source_path,
            )
        )
    return tuple(pages)
