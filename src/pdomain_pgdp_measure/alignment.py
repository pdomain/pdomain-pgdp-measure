"""Build snapshot-safe PGDP source-line alignment reports."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from stat import S_ISREG
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from pdomain_pgdp_measure.alignment_dp import (
    MAXIMUM_NORMALIZED_COST,
    MAXIMUM_SOURCE_LINES,
    MINIMUM_MATCHED_RATIO,
    MINIMUM_SOURCE_LINES,
    MINIMUM_UNIQUENESS_MARGIN,
    SKIP_COST,
    align_sequences,
)
from pdomain_pgdp_measure.alignment_image import ALIGNMENT_IMAGE_METHODS, extract_line_candidates
from pdomain_pgdp_measure.alignment_models import (
    AlignmentDiagnostic,
    AlignmentOperation,
    AlignmentReport,
    PageAlignment,
    ProjectAlignment,
    WireFormattingSpan,
    WireLineCandidate,
    WireSourceLine,
    WireStyleRun,
)
from pdomain_pgdp_measure.alignment_source import TokenizedSourcePage, tokenize_f2_pages
from pdomain_pgdp_measure.image_measurement import open_image_snapshot, sha256_file
from pdomain_pgdp_measure.ordering import natural_page_key
from pdomain_pgdp_measure.paths import (
    corpus_relative_path,
    require_canonical_relative_reference,
    resolve_image_candidate,
)
from pdomain_pgdp_measure.profile_models import (
    CoordinateFrame,
    PageMeasurement,
    ProfileDiagnostic,
    ProfileReport,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pdomain_pgdp_measure.alignment_dp import AlignmentResult
    from pdomain_pgdp_measure.alignment_image import CandidateExtraction, LineCandidate
    from pdomain_pgdp_measure.alignment_source import (
        FormattingSpan,
        SourceDiagnostic,
        SourceLine,
        StyleRun,
    )
    from pdomain_pgdp_measure.profile_models import ProjectProfile


_F2_PAGES_ADAPTER = TypeAdapter(dict[str, str])
_PASSTHROUGH_DIAGNOSTICS = frozenset(
    {
        "blank_page",
        "border_dominated",
        "foreground_bounds_unavailable",
        "high_foreground_ratio",
        "image_decode_failed",
        "image_missing",
        "image_unreadable",
    }
)


def build_alignment_report(
    corpus_root: str | Path,
    profile_path: str | Path,
    *,
    tool_version: str,
) -> AlignmentReport:
    """Build one alignment report from a stable M15a profile and local corpus."""

    root = Path(corpus_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Corpus root is not a directory: {root}")
    profile = Path(profile_path).expanduser()
    profile_bytes = _read_profile(profile)
    profile_report = ProfileReport.from_json(profile_bytes)
    projects: list[ProjectAlignment] = []
    diagnostics: list[AlignmentDiagnostic] = []
    for project in sorted(
        profile_report.projects, key=lambda project: natural_page_key(project.project_id)
    ):
        f2_bytes: bytes | None = None
        try:
            project_directory = _project_directory(root, project.project_id)
            f2_path = project_directory / "rounds" / "F2.json"
            f2_bytes = _read_f2_bytes(f2_path)
            tokenized_pages = _tokenize_f2_bytes(f2_bytes)
        except ValueError as error:
            diagnostics.append(
                AlignmentDiagnostic(
                    code="malformed_f2",
                    message=str(error),
                    project_id=project.project_id,
                )
            )
            projects.append(
                _f2_unavailable_project(
                    project=project,
                    message=str(error),
                    f2_bytes=f2_bytes,
                )
            )
            continue
        f2_sha256 = sha256(f2_bytes).hexdigest()
        tokenized_by_name = {page.page_name: page for page in tokenized_pages}
        pages = [
            _align_profile_page(
                root=root,
                project_id=project.project_id,
                measurement=measurement,
                source_page=tokenized_by_name.get(measurement.page_name),
                f2_sha256=f2_sha256,
            )
            for measurement in sorted(
                project.pages, key=lambda page: natural_page_key(page.page_name)
            )
        ]
        try:
            f2_changed = _read_f2_bytes(f2_path) != f2_bytes
        except ValueError:
            f2_changed = True
        if f2_changed:
            pages = [_source_changed(page) for page in pages]
        projects.append(
            ProjectAlignment(
                project_id=project.project_id,
                title=project.title,
                author=project.author,
                genre=project.genre,
                pages=tuple(pages),
                diagnostics=tuple(
                    _profile_diagnostic(diagnostic) for diagnostic in project.diagnostics
                ),
            )
        )
    return AlignmentReport(
        tool_version=tool_version,
        profile_label=profile.name,
        profile_sha256=sha256(profile_bytes).hexdigest(),
        methods={
            "source_normalization": "pgdp-f2-visible/v1",
            "image_candidates": ALIGNMENT_IMAGE_METHODS["algorithm"],
            "source_features": "pgdp-rank/v1",
            "alignment": "monotone-dp/v1",
        },
        thresholds={
            "minimum_source_lines": MINIMUM_SOURCE_LINES,
            "maximum_source_lines": MAXIMUM_SOURCE_LINES,
            "minimum_matched_ratio": MINIMUM_MATCHED_RATIO,
            "maximum_normalized_cost": MAXIMUM_NORMALIZED_COST,
            "minimum_uniqueness_margin": MINIMUM_UNIQUENESS_MARGIN,
            **ALIGNMENT_IMAGE_METHODS,
        },
        projects=tuple(projects),
        diagnostics=tuple(diagnostics),
    )


def _f2_unavailable_project(
    *,
    project: ProjectProfile,
    message: str,
    f2_bytes: bytes | None,
) -> ProjectAlignment:
    """Retain selected profile pages when their project F2 source is unusable."""

    f2_sha256 = None if f2_bytes is None else sha256(f2_bytes).hexdigest()
    project_diagnostic = AlignmentDiagnostic(
        code="malformed_f2",
        message=message,
        project_id=project.project_id,
    )
    pages = tuple(
        _excluded_page(
            page_name=measurement.page_name,
            f2_sha256=f2_sha256,
            scan_sha256=measurement.sha256,
            source_frame=measurement.source_frame,
            exclusions=("f2_missing",),
            diagnostics=(
                AlignmentDiagnostic(
                    code="malformed_f2",
                    message=message,
                    project_id=project.project_id,
                    page_name=measurement.page_name,
                ),
            ),
        )
        for measurement in sorted(project.pages, key=lambda page: natural_page_key(page.page_name))
    )
    return ProjectAlignment(
        project_id=project.project_id,
        title=project.title,
        author=project.author,
        genre=project.genre,
        pages=pages,
        diagnostics=(
            *(_profile_diagnostic(diagnostic) for diagnostic in project.diagnostics),
            project_diagnostic,
        ),
    )


def _read_profile(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError("Profile must be valid pgdp-profile/v1 JSON.") from error


def _project_directory(root: Path, project_id: str) -> Path:
    reference = require_canonical_relative_reference(
        value=project_id,
        label="project reference",
        direct_child=True,
    )
    directory = (root / reference).resolve()
    if not directory.is_dir() or not directory.is_relative_to(root):
        raise ValueError(f"Project directory does not exist: {project_id!r}.")
    return directory


def _read_f2_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError("Could not read rounds/F2.json.") from error


def _tokenize_f2_bytes(source: bytes) -> tuple[TokenizedSourcePage, ...]:
    try:
        pages = _F2_PAGES_ADAPTER.validate_json(source, strict=True)
    except ValidationError as error:
        raise ValueError(
            "Could not load rounds/F2.json as a string-to-string JSON object."
        ) from error
    return tokenize_f2_pages(pages).pages


def _align_profile_page(
    *,
    root: Path,
    project_id: str,
    measurement: PageMeasurement,
    source_page: TokenizedSourcePage | None,
    f2_sha256: str,
) -> PageAlignment:
    if source_page is None:
        return _excluded_page(
            page_name=measurement.page_name,
            f2_sha256=f2_sha256,
            scan_sha256=measurement.sha256,
            source_frame=measurement.source_frame,
            exclusions=("f2_page_missing",),
        )
    inherited_exclusions, inherited_diagnostics = _map_profile_diagnostics(measurement.diagnostics)
    if inherited_exclusions:
        return _excluded_page(
            page_name=measurement.page_name,
            f2_sha256=f2_sha256,
            scan_sha256=measurement.sha256,
            source_frame=measurement.source_frame,
            source_page=source_page,
            exclusions=inherited_exclusions,
            diagnostics=inherited_diagnostics,
            exclusion_evidence=_inherited_evidence(measurement.diagnostics),
        )
    if measurement.source_frame is None:
        return _excluded_page(
            page_name=measurement.page_name,
            f2_sha256=f2_sha256,
            scan_sha256=measurement.sha256,
            source_page=source_page,
            exclusions=("foreground_bounds_unavailable",),
        )
    image_path, path_diagnostic = _resolve_scan(
        root,
        project_id=project_id,
        page_name=measurement.page_name,
        source_path=measurement.source_path,
    )
    if image_path is None:
        diagnostics = ()
        if path_diagnostic is not None:
            message = (
                "Could not resolve a permitted scan candidate for this page."
                if path_diagnostic == "scan_resolution_failed"
                else "Profile source_path is not a permitted scan candidate for this page."
            )
            diagnostics = (
                AlignmentDiagnostic(
                    path_diagnostic,
                    message,
                    project_id,
                    measurement.page_name,
                ),
            )
        return _excluded_page(
            page_name=measurement.page_name,
            f2_sha256=f2_sha256,
            scan_sha256=measurement.sha256,
            source_frame=measurement.source_frame,
            source_page=source_page,
            exclusions=("image_missing",),
            diagnostics=diagnostics,
        )
    foreground_bounds = _foreground_bounds(measurement)
    snapshot_sha256: str | None = None
    provisional: PageAlignment | None = None
    try:
        with open_image_snapshot(image_path) as snapshot:
            snapshot_sha256 = snapshot.sha256
            if measurement.sha256 != snapshot.sha256:
                provisional = _excluded_page(
                    page_name=measurement.page_name,
                    f2_sha256=f2_sha256,
                    scan_sha256=snapshot.sha256,
                    source_frame=measurement.source_frame,
                    source_page=source_page,
                    exclusions=("scan_hash_mismatch",),
                )
            elif foreground_bounds is None:
                provisional = _excluded_page(
                    page_name=measurement.page_name,
                    f2_sha256=f2_sha256,
                    scan_sha256=snapshot.sha256,
                    source_frame=measurement.source_frame,
                    source_page=source_page,
                    exclusions=("foreground_bounds_unavailable",),
                )
            else:
                try:
                    extraction = extract_line_candidates(
                        snapshot,
                        source_frame=measurement.source_frame,
                        foreground_bounds=foreground_bounds,
                        ink_bands=measurement.ink_bands,
                        furniture_band_ordinals=measurement.furniture_band_ordinals,
                    )
                    result = align_sequences(
                        source_page.lines,
                        extraction.candidates,
                        foreground_bounds=foreground_bounds,
                        inherited_exclusions=extraction.exclusions,
                        probable_multi_column=extraction.probable_multi_column,
                        source_page=source_page,
                    )
                    provisional = _result_page(
                        page_name=measurement.page_name,
                        f2_sha256=f2_sha256,
                        scan_sha256=snapshot.sha256,
                        source_frame=measurement.source_frame,
                        source_page=source_page,
                        extraction=extraction,
                        result=result,
                    )
                except (ValueError, RuntimeError) as error:
                    provisional = _excluded_page(
                        page_name=measurement.page_name,
                        f2_sha256=f2_sha256,
                        scan_sha256=snapshot.sha256,
                        source_frame=measurement.source_frame,
                        source_page=source_page,
                        exclusions=("image_unreadable",),
                        diagnostics=(
                            AlignmentDiagnostic(
                                "image_unreadable", str(error), project_id, measurement.page_name
                            ),
                        ),
                    )
    except (OSError, ValueError, RuntimeError) as error:
        if snapshot_sha256 is None:
            return _excluded_page(
                page_name=measurement.page_name,
                f2_sha256=f2_sha256,
                scan_sha256=measurement.sha256,
                source_frame=measurement.source_frame,
                source_page=source_page,
                exclusions=("image_unreadable",),
                diagnostics=(
                    AlignmentDiagnostic(
                        "image_unreadable", str(error), project_id, measurement.page_name
                    ),
                ),
            )
        provisional = _excluded_page(
            page_name=measurement.page_name,
            f2_sha256=f2_sha256,
            scan_sha256=snapshot_sha256,
            source_frame=measurement.source_frame,
            source_page=source_page,
            exclusions=("image_unreadable",),
            diagnostics=(
                AlignmentDiagnostic(
                    "image_unreadable", str(error), project_id, measurement.page_name
                ),
            ),
        )
    try:
        source_changed = sha256_file(image_path) != snapshot_sha256
    except OSError:
        source_changed = True
    if source_changed:
        return _excluded_page(
            page_name=measurement.page_name,
            f2_sha256=f2_sha256,
            scan_sha256=snapshot_sha256,
            source_frame=measurement.source_frame,
            source_page=source_page,
            exclusions=("source_changed",),
        )
    return provisional


def _resolve_scan(
    root: Path, *, project_id: str, page_name: str, source_path: str
) -> tuple[Path | None, str | None]:
    try:
        reference = require_canonical_relative_reference(value=source_path, label="source_path")
    except ValueError:
        return None, "source_path_mismatch"
    try:
        path = (root / reference).resolve()
        path_status = path.stat()
    except FileNotFoundError:
        return None, "source_path_mismatch"
    except (OSError, RuntimeError):
        return None, "scan_resolution_failed"
    if not S_ISREG(path_status.st_mode) or not path.is_relative_to(root):
        return None, "source_path_mismatch"
    try:
        project_directory = _project_directory(root, project_id)
        resolution = resolve_image_candidate(
            project_directory=project_directory,
            page_name=page_name,
            corpus_root=root,
        )
    except (OSError, RuntimeError, ValueError):
        return None, "scan_resolution_failed"
    if resolution.image_path is None:
        return None, None
    try:
        expected_path = corpus_relative_path(path=resolution.image_path, corpus_root=root)
    except (OSError, RuntimeError, ValueError):
        return None, "scan_resolution_failed"
    if source_path != expected_path:
        return None, "source_path_mismatch"
    return resolution.image_path, None


def _foreground_bounds(measurement: PageMeasurement) -> tuple[int, int, int, int] | None:
    bounds = measurement.foreground_bounds
    if bounds is None or len(bounds) != 4:
        return None
    x_start, y_start, x_end, y_end = bounds
    return x_start, y_start, x_end, y_end


def _map_profile_diagnostics(
    diagnostics: tuple[ProfileDiagnostic, ...],
) -> tuple[tuple[str, ...], tuple[AlignmentDiagnostic, ...]]:
    exclusions: list[str] = []
    converted: list[AlignmentDiagnostic] = []
    for diagnostic in diagnostics:
        code = diagnostic.code
        exclusion = (
            "insufficient_ink_bands"
            if code in {"one_ink_band", "no_ink_bands"}
            else code
            if code in _PASSTHROUGH_DIAGNOSTICS
            else "inherited_profile_diagnostic"
        )
        if exclusion not in exclusions:
            exclusions.append(exclusion)
        converted.append(_profile_diagnostic(diagnostic))
    return tuple(exclusions), tuple(converted)


def _profile_diagnostic(diagnostic: ProfileDiagnostic) -> AlignmentDiagnostic:
    return AlignmentDiagnostic(
        code=diagnostic.code or "inherited_profile_diagnostic",
        message=diagnostic.message or "Inherited M15a profile diagnostic.",
        project_id=diagnostic.project_id,
        page_name=diagnostic.page_name,
    )


def _inherited_evidence(
    diagnostics: tuple[ProfileDiagnostic, ...],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {"code": diagnostic.code, "message": diagnostic.message}
        for diagnostic in diagnostics
        if diagnostic.code not in _PASSTHROUGH_DIAGNOSTICS
        and diagnostic.code not in {"one_ink_band", "no_ink_bands"}
    )


def _source_changed(page: PageAlignment) -> PageAlignment:
    return _excluded_page(
        page_name=page.page_name,
        f2_sha256=page.f2_sha256,
        scan_sha256=page.scan_sha256,
        source_frame=page.source_frame,
        exclusions=("source_changed",),
    )


def _excluded_page(
    *,
    page_name: str,
    f2_sha256: str | None,
    scan_sha256: str | None,
    source_frame: CoordinateFrame | None = None,
    source_page: TokenizedSourcePage | None = None,
    exclusions: tuple[str, ...],
    diagnostics: tuple[AlignmentDiagnostic, ...] = (),
    exclusion_evidence: tuple[Mapping[str, object], ...] = (),
) -> PageAlignment:
    lines = (
        ()
        if source_page is None
        else tuple(_wire_source_line(line, source_page) for line in source_page.lines)
    )
    spans = (
        ()
        if source_page is None
        else tuple(_wire_formatting_span(span) for span in source_page.formatting_spans)
    )
    styles = (
        ()
        if source_page is None
        else tuple(_wire_style_run(style) for style in source_page.style_runs)
    )
    operations = tuple(
        AlignmentOperation("skip_text", line.ordinal, None, SKIP_COST)
        for line in lines
        if line.matching_eligible and line.visible_text
    )
    return PageAlignment(
        page_name=page_name,
        f2_sha256=f2_sha256,
        scan_sha256=scan_sha256 if source_frame is not None else None,
        source_frame=source_frame,
        source_lines=lines,
        formatting_spans=spans,
        style_runs=styles,
        candidates=(),
        operations=operations,
        total_cost=sum(operation.cost for operation in operations),
        second_best_cost=None,
        normalized_cost=1.0 if operations else 0.0,
        matched_count=0,
        matched_ratio=0.0,
        uniqueness_margin=0.0,
        mean_width_residual=None,
        mean_indentation_residual=None,
        state="excluded",
        accepted=False,
        exclusions=exclusions,
        diagnostics=diagnostics,
        exclusion_evidence=exclusion_evidence,
    )


def _result_page(
    *,
    page_name: str,
    f2_sha256: str,
    scan_sha256: str,
    source_frame: CoordinateFrame,
    source_page: TokenizedSourcePage,
    extraction: CandidateExtraction,
    result: AlignmentResult,
) -> PageAlignment:
    return PageAlignment(
        page_name=page_name,
        f2_sha256=f2_sha256,
        scan_sha256=scan_sha256,
        source_frame=source_frame,
        source_lines=tuple(_wire_source_line(line, source_page) for line in source_page.lines),
        formatting_spans=tuple(
            _wire_formatting_span(span) for span in source_page.formatting_spans
        ),
        style_runs=tuple(_wire_style_run(style) for style in source_page.style_runs),
        candidates=tuple(
            _wire_candidate(index, candidate)
            for index, candidate in enumerate(extraction.candidates)
        ),
        operations=tuple(_wire_operation(operation) for operation in result.operations),
        total_cost=result.total_cost,
        second_best_cost=None if result.second_best is None else result.second_best.total_cost,
        normalized_cost=result.normalized_cost,
        matched_count=result.matched_count,
        matched_ratio=result.matched_ratio,
        uniqueness_margin=result.uniqueness_margin,
        mean_width_residual=result.mean_width_residual,
        mean_indentation_residual=result.mean_indentation_residual,
        state=result.state,
        accepted=result.accepted,
        exclusions=result.exclusions,
        diagnostics=tuple(_source_diagnostic(diagnostic) for diagnostic in source_page.diagnostics),
        exclusion_evidence=tuple(
            {
                "code": evidence.code,
                "predicate_method": evidence.predicate_method,
                "source_ordinals": list(evidence.source_ordinals),
            }
            for evidence in result.exclusion_evidence
        ),
    )


def _wire_source_line(line: SourceLine, page: TokenizedSourcePage) -> WireSourceLine:
    return WireSourceLine(
        ordinal=line.ordinal,
        byte_start=line.byte_start,
        byte_end=line.byte_end,
        content_byte_start=line.content_byte_start,
        content_byte_end=line.content_byte_end,
        separator_byte_start=line.separator_byte_start,
        separator_byte_end=line.separator_byte_end,
        original_text=line.original_text,
        separator_text=page.source_utf8[line.separator_byte_start : line.separator_byte_end].decode(
            "utf-8"
        ),
        visible_text=line.visible_text,
        operation_ordinals=line.operation_ordinals,
        formatting_span_ids=line.formatting_span_ids,
        continued_block_ids=line.continued_block_ids,
        matching_eligible=line.matching_eligible,
        style_fitting_eligible=line.style_fitting_eligible,
    )


def _wire_formatting_span(span: FormattingSpan) -> WireFormattingSpan:
    return WireFormattingSpan(
        opening_id=span.opening_id,
        block_id=span.block_id,
        kind=span.kind,
        page_name=span.page_name,
        byte_start=span.byte_start,
        byte_end=span.byte_end,
        closed=span.closed,
    )


def _wire_style_run(style: StyleRun) -> WireStyleRun:
    return WireStyleRun(
        kind=style.kind,
        normalized_start=style.normalized_start,
        normalized_end=style.normalized_end,
        byte_start=style.byte_start,
        byte_end=style.byte_end,
    )


def _wire_candidate(ordinal: int, candidate: LineCandidate) -> WireLineCandidate:
    return WireLineCandidate(
        ordinal=ordinal,
        band_ordinal=candidate.band_ordinal,
        component_ordinal=candidate.component_ordinal,
        box=candidate.box,
        component_count=candidate.component_count,
        foreground_pixels=candidate.foreground_pixels,
        width=candidate.width,
        height=candidate.height,
        fill_ratio=candidate.fill_ratio,
        horizontal_ink_profile=candidate.horizontal_ink_profile,
    )


def _wire_operation(operation: object) -> AlignmentOperation:
    from pdomain_pgdp_measure.alignment_dp import AlignmentOperation as DynamicOperation

    if not isinstance(operation, DynamicOperation):
        raise TypeError("Unexpected dynamic alignment operation.")
    costs = operation.match_cost
    return AlignmentOperation(
        kind=operation.kind,
        source_ordinal=operation.source_ordinal,
        candidate_ordinal=operation.candidate_ordinal,
        cost=operation.cost,
        width_cost=None if costs is None else costs.width,
        indentation_cost=None if costs is None else costs.indent,
        gaps_cost=None if costs is None else costs.gaps,
        style_cost=None if costs is None else costs.style,
    )


def _source_diagnostic(diagnostic: SourceDiagnostic) -> AlignmentDiagnostic:
    return AlignmentDiagnostic(
        code=diagnostic.code,
        message=diagnostic.message,
        page_name=diagnostic.page_name,
        extensions={"byte_start": diagnostic.byte_start, "byte_end": diagnostic.byte_end},
    )
