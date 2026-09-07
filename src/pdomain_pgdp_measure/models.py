from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "pgdp-rank/v1"


class DiagnosticData(TypedDict):
    code: str
    message: str
    project_id: str | None
    page_name: str | None


class PageFeaturesData(TypedDict):
    special_format: bool
    italic_tags: int
    bold_tags: int
    small_caps_tags: int
    dot_leaders: bool
    aligned_fields: bool
    table_like: bool
    poetry_like: bool
    quotation_like: bool
    multipart_name: bool
    illustration_or_ornament: bool
    uncertainty_note: bool


class PageScoreData(TypedDict):
    total: int
    components: dict[str, int]


class RankedPageData(TypedDict):
    name: str
    transcription_available: bool
    image_available: bool
    features: PageFeaturesData
    score: PageScoreData
    matched_groups: list[str]


class RankedProjectData(TypedDict):
    project_id: str
    title: str | None
    author: str | None
    genre: str | None
    pages_total: int | None
    pg_ebook_number: int | None
    score: int
    score_components: dict[str, int]
    pages: list[RankedPageData]


class CorpusSummaryData(TypedDict):
    projects_seen: int
    projects_ranked: int


class RankingLimitsData(TypedDict):
    projects: int
    pages_per_project: int


class RankingReportData(TypedDict):
    schema_version: int
    algorithm_version: str
    limits: RankingLimitsData
    corpus: CorpusSummaryData
    diagnostics: list[DiagnosticData]
    projects: list[RankedProjectData]


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A non-fatal issue found while inspecting one project or page."""

    code: str
    message: str
    project_id: str | None = None
    page_name: str | None = None

    def to_dict(self) -> DiagnosticData:
        return DiagnosticData(
            code=self.code,
            message=self.message,
            project_id=self.project_id,
            page_name=self.page_name,
        )


@dataclass(frozen=True, slots=True)
class PageFeatures:
    """Deterministic F2 typography and layout evidence for one page."""

    special_format: bool = False
    italic_tags: int = 0
    bold_tags: int = 0
    small_caps_tags: int = 0
    dot_leaders: bool = False
    aligned_fields: bool = False
    table_like: bool = False
    poetry_like: bool = False
    quotation_like: bool = False
    multipart_name: bool = False
    illustration_or_ornament: bool = False
    uncertainty_note: bool = False

    def to_dict(self) -> PageFeaturesData:
        return PageFeaturesData(
            special_format=self.special_format,
            italic_tags=self.italic_tags,
            bold_tags=self.bold_tags,
            small_caps_tags=self.small_caps_tags,
            dot_leaders=self.dot_leaders,
            aligned_fields=self.aligned_fields,
            table_like=self.table_like,
            poetry_like=self.poetry_like,
            quotation_like=self.quotation_like,
            multipart_name=self.multipart_name,
            illustration_or_ornament=self.illustration_or_ornament,
            uncertainty_note=self.uncertainty_note,
        )


@dataclass(frozen=True, slots=True)
class PageScore:
    """Versioned score and component totals for one ranked page."""

    total: int
    components: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))

    def to_dict(self) -> PageScoreData:
        return PageScoreData(total=self.total, components=dict(self.components))


@dataclass(frozen=True, slots=True)
class RankedPage:
    """One selected page and the local evidence that ranked it."""

    name: str
    transcription_available: bool
    image_available: bool
    features: PageFeatures
    score: PageScore
    matched_groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_groups", tuple(self.matched_groups))

    def to_dict(self) -> RankedPageData:
        return RankedPageData(
            name=self.name,
            transcription_available=self.transcription_available,
            image_available=self.image_available,
            features=self.features.to_dict(),
            score=self.score.to_dict(),
            matched_groups=list(self.matched_groups),
        )


@dataclass(frozen=True, slots=True)
class RankedProject:
    """One ranked PGDP project and its bounded review queue."""

    project_id: str
    title: str | None
    author: str | None
    genre: str | None
    pages_total: int | None
    pg_ebook_number: int | None
    score: int
    score_components: Mapping[str, int]
    pages: tuple[RankedPage, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "score_components", MappingProxyType(dict(self.score_components)))
        object.__setattr__(self, "pages", tuple(self.pages))

    def to_dict(self) -> RankedProjectData:
        return RankedProjectData(
            project_id=self.project_id,
            title=self.title,
            author=self.author,
            genre=self.genre,
            pages_total=self.pages_total,
            pg_ebook_number=self.pg_ebook_number,
            score=self.score,
            score_components=dict(self.score_components),
            pages=[page.to_dict() for page in self.pages],
        )


@dataclass(frozen=True, slots=True)
class CorpusSummary:
    """Counts collected before report-level project truncation."""

    projects_seen: int
    projects_ranked: int

    def to_dict(self) -> CorpusSummaryData:
        return CorpusSummaryData(
            projects_seen=self.projects_seen,
            projects_ranked=self.projects_ranked,
        )


@dataclass(frozen=True, slots=True)
class RankingLimits:
    """Effective caps for projects and selected pages."""

    projects: int
    pages_per_project: int

    def to_dict(self) -> RankingLimitsData:
        return RankingLimitsData(
            projects=self.projects,
            pages_per_project=self.pages_per_project,
        )


@dataclass(frozen=True, slots=True)
class RankingReport:
    """Stable JSON-safe output from a local PGDP ranking run."""

    limits: RankingLimits
    corpus: CorpusSummary
    diagnostics: tuple[Diagnostic, ...] = ()
    projects: tuple[RankedProject, ...] = ()
    schema_version: int = SCHEMA_VERSION
    algorithm_version: str = ALGORITHM_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "projects", tuple(self.projects))

    @classmethod
    def empty(cls, *, project_limit: int = 50, pages_per_project: int = 12) -> RankingReport:
        return cls(
            limits=RankingLimits(projects=project_limit, pages_per_project=pages_per_project),
            corpus=CorpusSummary(projects_seen=0, projects_ranked=0),
        )

    def to_dict(self) -> RankingReportData:
        return RankingReportData(
            schema_version=self.schema_version,
            algorithm_version=self.algorithm_version,
            limits=self.limits.to_dict(),
            corpus=self.corpus.to_dict(),
            diagnostics=[diagnostic.to_dict() for diagnostic in self.diagnostics],
            projects=[project.to_dict() for project in self.projects],
        )
