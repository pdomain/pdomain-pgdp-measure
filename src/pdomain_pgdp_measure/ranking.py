from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict

from pydantic import TypeAdapter, ValidationError

from pdomain_pgdp_measure.f2 import F2LoadError, ParsedF2Page, load_f2
from pdomain_pgdp_measure.features import extract_page_features, score_page
from pdomain_pgdp_measure.models import (
    CorpusSummary,
    Diagnostic,
    PageFeatures,
    PageScore,
    RankedPage,
    RankedProject,
    RankingLimits,
    RankingReport,
)
from pdomain_pgdp_measure.ordering import natural_page_key
from pdomain_pgdp_measure.paths import UnsafePathError, resolve_image_candidate

if TYPE_CHECKING:
    from collections.abc import Callable


_PROJECT_PREFIX = "projectID"
_TARGET_GROUP_ORDER = (
    "mixed_typography",
    "table_like_structure",
    "poetry",
    "quotation",
    "multipart_layout",
)


class _ProjectMetadataPayload(TypedDict):
    projectid: str
    title: NotRequired[str]
    author: NotRequired[str]
    genre: NotRequired[str]
    pages_total: NotRequired[int]
    pg_ebook_number: NotRequired[int]


_PROJECT_METADATA_ADAPTER = TypeAdapter(_ProjectMetadataPayload)
_PAGES_ADAPTER = TypeAdapter(dict[str, str])


@dataclass(frozen=True, slots=True)
class _ProjectMetadata:
    project_id: str
    title: str | None
    author: str | None
    genre: str | None
    pages_total: int | None
    pg_ebook_number: int | None


@dataclass(frozen=True, slots=True)
class _PageCandidate:
    name: str
    transcription_available: bool
    image_available: bool
    features: PageFeatures
    score: PageScore
    matched_groups: tuple[str, ...]


def rank_corpus(
    corpus_root: str | Path, *, project_limit: int = 50, pages_per_project: int = 12
) -> RankingReport:
    """Rank immediate PGDP projects and select bounded review pages."""

    if project_limit <= 0 or pages_per_project <= 0:
        raise ValueError("project_limit and pages_per_project must be positive.")

    root = Path(corpus_root)
    diagnostics: list[Diagnostic] = []
    projects: list[RankedProject] = []
    project_directories = tuple(
        directory
        for directory in sorted(root.iterdir(), key=lambda path: path.name)
        if directory.is_dir() and directory.name.startswith(_PROJECT_PREFIX)
    )

    for project_directory in project_directories:
        ranked_project = _rank_project(
            project_directory=project_directory,
            pages_per_project=pages_per_project,
            diagnostics=diagnostics,
        )
        if ranked_project is not None:
            projects.append(ranked_project)

    projects.sort(key=lambda project: (-project.score, project.project_id))
    return RankingReport(
        limits=RankingLimits(projects=project_limit, pages_per_project=pages_per_project),
        corpus=CorpusSummary(
            projects_seen=len(project_directories),
            projects_ranked=len(projects),
        ),
        diagnostics=tuple(sorted(diagnostics, key=_diagnostic_key)),
        projects=tuple(projects[:project_limit]),
    )


def _rank_project(
    *,
    project_directory: Path,
    pages_per_project: int,
    diagnostics: list[Diagnostic],
) -> RankedProject | None:
    directory_project_id = project_directory.name
    f2_path = project_directory / "rounds" / "F2.json"
    if not f2_path.is_file():
        diagnostics.append(
            Diagnostic(
                code="missing_f2",
                message="Project has no rounds/F2.json file.",
                project_id=directory_project_id,
            )
        )
        return None

    metadata = _load_metadata(
        path=project_directory / "project.json",
        directory_project_id=directory_project_id,
        diagnostics=diagnostics,
    )
    if metadata is None:
        return None

    try:
        parsed_f2 = load_f2(f2_path)
    except F2LoadError:
        diagnostics.append(
            Diagnostic(
                code="malformed_f2",
                message="Could not load rounds/F2.json as a string-to-string JSON object.",
                project_id=metadata.project_id,
            )
        )
        return None

    diagnostics.extend(
        Diagnostic(
            code=diagnostic.code,
            message=diagnostic.message,
            project_id=metadata.project_id,
            page_name=diagnostic.page_name,
        )
        for diagnostic in parsed_f2.diagnostics
    )
    transcriptions = _load_transcriptions(
        path=project_directory / "pages.json",
        project_id=metadata.project_id,
        diagnostics=diagnostics,
    )
    candidates = _rank_pages(
        parsed_pages=parsed_f2.pages,
        project_directory=project_directory,
        project_id=metadata.project_id,
        transcriptions=transcriptions,
        diagnostics=diagnostics,
    )
    top_ten_page_scores = sum(candidate.score.total for candidate in candidates[:10])
    group_bonus = 20 * len(
        {group for candidate in candidates for group in candidate.matched_groups}
    )
    selected_pages = _select_review_pages(candidates, pages_per_project=pages_per_project)

    return RankedProject(
        project_id=metadata.project_id,
        title=metadata.title,
        author=metadata.author,
        genre=metadata.genre,
        pages_total=metadata.pages_total,
        pg_ebook_number=metadata.pg_ebook_number,
        score=top_ten_page_scores + group_bonus,
        score_components={
            "top_ten_page_scores": top_ten_page_scores,
            "group_bonus": group_bonus,
        },
        pages=tuple(
            RankedPage(
                name=candidate.name,
                transcription_available=candidate.transcription_available,
                image_available=candidate.image_available,
                features=candidate.features,
                score=candidate.score,
                matched_groups=candidate.matched_groups,
            )
            for candidate in selected_pages
        ),
    )


def _load_metadata(
    *, path: Path, directory_project_id: str, diagnostics: list[Diagnostic]
) -> _ProjectMetadata | None:
    try:
        payload = _PROJECT_METADATA_ADAPTER.validate_json(
            path.read_text(encoding="utf-8"), strict=True
        )
    except (OSError, UnicodeDecodeError, ValidationError):
        diagnostics.append(
            Diagnostic(
                code="malformed_project_metadata",
                message="Could not load project.json with required typed metadata.",
                project_id=directory_project_id,
            )
        )
        return None
    if payload["projectid"] != directory_project_id:
        diagnostics.append(
            Diagnostic(
                code="project_id_mismatch",
                message="project.json projectid does not match the project directory.",
                project_id=directory_project_id,
            )
        )
        return None
    return _ProjectMetadata(
        project_id=payload["projectid"],
        title=payload.get("title"),
        author=payload.get("author"),
        genre=payload.get("genre"),
        pages_total=payload.get("pages_total"),
        pg_ebook_number=payload.get("pg_ebook_number"),
    )


def _load_transcriptions(
    *, path: Path, project_id: str, diagnostics: list[Diagnostic]
) -> dict[str, str]:
    if not path.is_file():
        diagnostics.append(
            Diagnostic(
                code="missing_pages_json",
                message="Project has no pages.json file.",
                project_id=project_id,
            )
        )
        return {}
    try:
        return _PAGES_ADAPTER.validate_json(path.read_text(encoding="utf-8"), strict=True)
    except (OSError, UnicodeDecodeError, ValidationError):
        diagnostics.append(
            Diagnostic(
                code="malformed_pages_json",
                message="Could not load pages.json as a string-to-string JSON object.",
                project_id=project_id,
            )
        )
        return {}


def _rank_pages(
    *,
    parsed_pages: tuple[ParsedF2Page, ...],
    project_directory: Path,
    project_id: str,
    transcriptions: dict[str, str],
    diagnostics: list[Diagnostic],
) -> list[_PageCandidate]:
    candidates = [
        _page_candidate(
            page=page,
            project_directory=project_directory,
            project_id=project_id,
            transcriptions=transcriptions,
            diagnostics=diagnostics,
        )
        for page in parsed_pages
    ]
    candidates.sort(
        key=lambda candidate: (-candidate.score.total, natural_page_key(candidate.name))
    )
    return candidates


def _page_candidate(
    *,
    page: ParsedF2Page,
    project_directory: Path,
    project_id: str,
    transcriptions: dict[str, str],
    diagnostics: list[Diagnostic],
) -> _PageCandidate:
    features = extract_page_features(page)
    return _PageCandidate(
        name=page.name,
        transcription_available=page.name in transcriptions,
        image_available=_image_available(
            project_directory=project_directory,
            page_name=page.name,
            project_id=project_id,
            diagnostics=diagnostics,
        ),
        features=features,
        score=score_page(features),
        matched_groups=_matched_groups(features),
    )


def _image_available(
    *, project_directory: Path, page_name: str, project_id: str, diagnostics: list[Diagnostic]
) -> bool:
    page_path = Path(page_name)
    if not page_name or page_path.is_absolute() or ".." in page_path.parts:
        return False
    try:
        resolution = resolve_image_candidate(
            project_directory=project_directory,
            page_name=page_path.as_posix(),
        )
    except (OSError, UnsafePathError):
        diagnostics.append(
            Diagnostic(
                code="unresolvable_image_path",
                message="Could not resolve page image path.",
                project_id=project_id,
                page_name=page_name,
            )
        )
        return False
    else:
        return resolution.image_path is not None


def _matched_groups(features: PageFeatures) -> tuple[str, ...]:
    predicates: tuple[tuple[str, Callable[[PageFeatures], bool]], ...] = (
        (
            "mixed_typography",
            lambda value: value.italic_tags > 0 or value.bold_tags > 0 or value.small_caps_tags > 0,
        ),
        (
            "table_like_structure",
            lambda value: value.table_like or value.dot_leaders or value.aligned_fields,
        ),
        ("poetry", lambda value: value.poetry_like),
        ("quotation", lambda value: value.quotation_like),
        ("multipart_layout", lambda value: value.multipart_name),
    )
    return tuple(group for group, predicate in predicates if predicate(features))


def _select_review_pages(
    candidates: list[_PageCandidate], *, pages_per_project: int
) -> tuple[_PageCandidate, ...]:
    selected: list[_PageCandidate] = []
    selected_names: set[str] = set()
    for group in _TARGET_GROUP_ORDER:
        if len(selected) == pages_per_project:
            break
        candidate = next(
            (
                candidate
                for candidate in candidates
                if group in candidate.matched_groups and candidate.name not in selected_names
            ),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
            selected_names.add(candidate.name)

    for candidate in candidates:
        if len(selected) == pages_per_project:
            break
        if candidate.name not in selected_names:
            selected.append(candidate)
            selected_names.add(candidate.name)
    return tuple(selected)


def _diagnostic_key(diagnostic: Diagnostic) -> tuple[str, str, str, str]:
    return (
        diagnostic.code,
        diagnostic.project_id or "",
        diagnostic.page_name or "",
        diagnostic.message,
    )
