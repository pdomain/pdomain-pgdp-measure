"""Build snapshot-safe font-free PGDP typography reports.

`build_typography_report` reads a shipped `pgdp-alignment/v3` report together with the
`pgdp-profile/v2` profile that alignment recorded, rebuilds each accepted page's ink mask,
measures every matched line, and pools the results into a `pgdp-typography/v1` report.

Two checks run before any measurement is trusted. The command refuses to start unless the
profile it is handed hashes to the value the alignment report recorded, so the two inputs are
provably the pair that produced each other. Then, on every page, the rebuilt mask must
reproduce each matched candidate's stored `foreground_pixels` and `horizontal_ink_profile`
exactly and the rebuilt Otsu threshold must equal the recorded one; a page that fails is
excluded as `mask_mismatch` carrying both values, because a mask that has drifted would make
every number measured from it wrong in a way no later gate could see.

Nothing here opens a font, renders text, or leaves the `source` frame.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pdomain_pgdp_measure.alignment_image import build_candidate_mask
from pdomain_pgdp_measure.alignment_models import AlignmentReport
from pdomain_pgdp_measure.image_measurement import (
    measure_image_snapshot,
    open_image_snapshot,
    sha256_file,
)
from pdomain_pgdp_measure.ocr_witness import (
    OCR_WITNESS_METHODS,
    BookWitness,
    read_book_witness,
    witness_line,
)
from pdomain_pgdp_measure.ordering import natural_page_key
from pdomain_pgdp_measure.paths import (
    corpus_relative_path,
    require_canonical_relative_reference,
    resolve_image_candidate,
)
from pdomain_pgdp_measure.profile_models import ProfileDiagnostic, ProfileReport
from pdomain_pgdp_measure.typography_measure import (
    TYPOGRAPHY_MEASURE_METHODS,
    WORD_SEGMENTATION_METHODS,
    LineTypographyMeasurement,
    WordGapThreshold,
    derive_word_gap_threshold,
    find_word_runs,
    measure_baseline_pitches,
    measure_line_typography,
    measure_line_words,
    word_gap_lengths,
    x_height_gap_threshold,
)
from pdomain_pgdp_measure.typography_models import (
    DEFAULT_EVIDENCE_PAGES_PER_BOOK,
    TYPOGRAPHY_ALGORITHM_VERSION,
    BookTypography,
    LineTypography,
    PageTypography,
    TypographyReport,
    WordTypography,
)
from pdomain_pgdp_measure.typography_pooling import (
    TYPOGRAPHY_POOLING_METHODS,
    pool_book,
    pool_page,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pdomain_pgdp_measure.alignment_models import (
        PageAlignment,
        ProjectAlignment,
        WireLineCandidate,
    )
    from pdomain_pgdp_measure.ocr_witness import LineWitness, PageWitness
    from pdomain_pgdp_measure.profile_models import PageMeasurement, ProjectProfile
    from pdomain_pgdp_measure.typography_measure import (
        LineTypographyExclusion,
    )


class ProvenanceError(ValueError):
    """The profile handed to this command is not the one the alignment report recorded."""


def build_typography_report(
    corpus_root: str | Path,
    alignment_path: str | Path,
    profile_path: str | Path,
    *,
    tool_version: str,
    evidence_pages: int = DEFAULT_EVIDENCE_PAGES_PER_BOOK,
    geometry_path: str | Path | None = None,
) -> TypographyReport:
    """Measure typography for every accepted page in one alignment report.

    `geometry_path` is optional. Without it this behaves exactly as it did before the witness
    existed, which is the property the acceptance gate checks byte for byte. With it, each
    matched line's first ink run is checked against what the recognizer read there, so a line
    whose first run is a continuation fragment F2 dropped can be marked.
    """

    root = Path(corpus_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Corpus root is not a directory: {root}")
    if evidence_pages < 0:
        raise ValueError("evidence_pages must be nonnegative.")

    alignment_file = Path(alignment_path).expanduser()
    alignment_bytes = _read_bytes(alignment_file, label="alignment report")
    alignment = AlignmentReport.from_json(alignment_bytes)

    profile_file = Path(profile_path).expanduser()
    profile_bytes = _read_bytes(profile_file, label="profile")
    profile_sha256 = sha256(profile_bytes).hexdigest()
    if profile_sha256 != alignment.profile_sha256:
        raise ProvenanceError(
            "Profile does not match the alignment report's recorded profile_sha256: "
            f"{profile_sha256} != {alignment.profile_sha256}."
        )
    profile = ProfileReport.from_json(profile_bytes)
    profiles_by_id = {project.project_id: project for project in profile.projects}

    ordered_projects = sorted(
        alignment.projects, key=lambda project: natural_page_key(project.project_id)
    )
    witness = _read_witness(geometry_path, projects=ordered_projects)
    books = tuple(
        _measure_project(
            root,
            project=project,
            project_profile=profiles_by_id.get(project.project_id),
            evidence_pages=evidence_pages,
            witness=witness,
        )
        for project in ordered_projects
    )
    return TypographyReport(
        tool_version=tool_version,
        alignment_label=alignment_file.name,
        alignment_sha256=sha256(alignment_bytes).hexdigest(),
        profile_label=alignment.profile_label,
        profile_sha256=profile_sha256,
        methods={
            "algorithm_version": TYPOGRAPHY_ALGORITHM_VERSION,
            "line_measurement": TYPOGRAPHY_MEASURE_METHODS["algorithm"],
            "word_segmentation": WORD_SEGMENTATION_METHODS["algorithm"],
            **({} if witness is None else {"ocr_witness": OCR_WITNESS_METHODS["algorithm"]}),
            **TYPOGRAPHY_POOLING_METHODS,
        },
        thresholds={
            **TYPOGRAPHY_MEASURE_METHODS,
            **WORD_SEGMENTATION_METHODS,
            "word_gap_threshold_px_by_book": {
                book.project_id: book.extensions["word_gap_threshold_px"] for book in books
            },
            "word_gap_threshold_rule_by_book": {
                book.project_id: book.extensions["word_gap_threshold_rule"] for book in books
            },
        },
        evidence_pages_per_book=evidence_pages,
        books=books,
        extensions=_witness_provenance(witness),
    )


def _book_gap_accuracy(pages: Sequence[PageTypography]) -> dict[str, object]:
    """The book's per-gap error rate, reported beside Gate 6's per-line rate.

    The denominator is the transcription's own word boundaries, so a one-word line that
    over-splits contributes an error against no boundary. That is rare and the alternative,
    dropping such lines, would hide a real error.
    """

    gaps = sum(_extension_count(page, "word_gap_count") for page in pages)
    errors = sum(_extension_count(page, "word_run_error_count") for page in pages)
    return {
        "word_gap_count": gaps,
        "word_run_error_count": errors,
        "word_gap_error_rate": None if gaps == 0 else errors / gaps,
    }


def _book_witness_totals(
    pages: Sequence[PageTypography], *, witness: BookWitness | None
) -> dict[str, object]:
    """Roll the per-page witness tallies up to the book.

    `reconciled_after_witness` is reported beside Gate 6's own rate and does not replace it. Gate
    6 is a tripwire for a broken detector, and folding a known cause into it would make it blind
    to that cause returning.
    """

    if witness is None:
        return {}
    measured = sum(page.measured_line_count for page in pages)
    totals = {
        name: sum(_extension_count(page, name) for page in pages)
        for name in (
            "witnessed_line_count",
            "continuation_fragment_line_count",
            "reconciled_after_witness_line_count",
        )
    }
    reconciled_after = totals["reconciled_after_witness_line_count"]
    return {
        **totals,
        "reconciled_after_witness": None if measured == 0 else reconciled_after / measured,
    }


def _extension_count(page: PageTypography, name: str) -> int:
    value = page.extensions.get(name)
    return value if isinstance(value, int) else 0


def _read_witness(
    geometry_path: str | Path | None, *, projects: Sequence[ProjectAlignment]
) -> BookWitness | None:
    """Read the geometry record, refusing one that is not for the single book being measured.

    A geometry file names one project, so a report covering several books has no single witness
    to apply. Refusing is better than silently witnessing one book and not the others.
    """

    if geometry_path is None:
        return None
    if len(projects) != 1:
        raise ValueError("A geometry record can only be applied to a single-project alignment.")
    return read_book_witness(geometry_path, project_id=projects[0].project_id)


@dataclass(frozen=True, slots=True)
class _GapTally:
    """One page's word boundaries and how many the run detector got wrong.

    Reported because Gate 6 counts lines, and a line carries six to twelve gaps, so one bad gap
    fails the whole line. Measured over the corpus, line agreement runs 0.59 to 0.81 while
    per-gap accuracy runs 0.915 to 0.968. A consumer of word boxes is buying gaps, not lines.
    """

    gaps: int = 0
    run_errors: int = 0

    def to_extensions(self) -> dict[str, object]:
        return {"word_gap_count": self.gaps, "word_run_error_count": self.run_errors}


@dataclass(frozen=True, slots=True)
class _WitnessTally:
    """How many of a page's matched lines the recognizer spoke for, and what it said."""

    witnessed: int = 0
    continuation_fragments: int = 0
    reconciled_after_witness: int = 0

    def to_extensions(self) -> dict[str, object]:
        return {
            "witnessed_line_count": self.witnessed,
            "continuation_fragment_line_count": self.continuation_fragments,
            "reconciled_after_witness_line_count": self.reconciled_after_witness,
        }


def _witness_provenance(witness: BookWitness | None) -> dict[str, object]:
    if witness is None:
        return {}
    return {
        "geometry_label": witness.label,
        "geometry_sha256": witness.sha256,
        "ocr_recognizer": witness.recognizer.to_dict(),
    }


def _measure_project(
    root: Path,
    *,
    project: ProjectAlignment,
    project_profile: ProjectProfile | None,
    evidence_pages: int,
    witness: BookWitness | None,
) -> BookTypography:
    """Measure one book's accepted pages and pool them.

    Evidence rows go to the first `evidence_pages` pages that measure successfully, in
    natural page order. Sampling measured pages rather than accepted ones means the sample
    always carries the rows it promises: an accepted page excluded as `mask_mismatch` has no
    lines to show.
    """

    measurements = (
        {} if project_profile is None else {page.page_name: page for page in project_profile.pages}
    )
    text_left_by_class = (
        {}
        if project_profile is None
        else {
            template.page_class: template.text_left_px
            for template in project_profile.page_templates
        }
    )
    book_gap, page_gaps = _word_gap_thresholds(project)
    diagnostics: list[ProfileDiagnostic] = []
    pages: list[PageTypography] = []
    evidence_remaining = evidence_pages
    for page in sorted(project.pages, key=lambda page: natural_page_key(page.page_name)):
        if not page.accepted:
            continue
        measured = _measure_page(
            root,
            project_id=project.project_id,
            page=page,
            measurement=measurements.get(page.page_name),
            text_left_by_class=text_left_by_class,
            wants_evidence=evidence_remaining > 0,
            book_gap=book_gap,
            page_gap=page_gaps.get(page.page_name),
            witness=witness,
        )
        if measured.evidence_sampled:
            evidence_remaining -= 1
        diagnostics.extend(measured.diagnostics)
        pages.append(measured)

    ordered = tuple(pages)
    pooled = pool_book(ordered)
    excluded_pages = sum(1 for page in ordered if page.exclusions)
    return BookTypography(
        project_id=project.project_id,
        pages=ordered,
        styles=pooled.styles,
        estimates=pooled.estimates,
        accepted_page_count=len(ordered),
        measured_page_count=len(ordered) - excluded_pages,
        excluded_page_count=excluded_pages,
        unknown_class_page_count=pooled.unknown_class_page_count,
        matched_line_count=sum(page.matched_line_count for page in ordered),
        measured_line_count=sum(page.measured_line_count for page in ordered),
        reconciled_line_count=sum(page.reconciled_line_count for page in ordered),
        skew_slope_median=pooled.skew.median,
        skew_slope_mad=pooled.skew.median_absolute_deviation,
        diagnostics=tuple(diagnostics),
        extensions={
            "word_gap_threshold_px": book_gap.threshold_px,
            "word_gap_threshold_rule": book_gap.rule,
            "word_gap_valley_ratio": book_gap.valley_ratio,
            "word_gap_sample_count": book_gap.sample_count,
            **_book_gap_accuracy(ordered),
            **_book_witness_totals(ordered, witness=witness),
        },
    )


def book_word_gap_threshold(project: ProjectAlignment) -> WordGapThreshold:
    """One book's word-gap threshold, for a stage that needs word boxes but not a full report."""

    return _word_gap_thresholds(project)[0]


def _word_gap_thresholds(
    project: ProjectAlignment,
) -> tuple[WordGapThreshold, dict[str, WordGapThreshold]]:
    """One book's word-gap threshold and every accepted page's own, from stored ink alone.

    This runs before any scan is opened. Every gap length it counts comes from the
    `horizontal_ink_profile` the alignment report already carries, so the whole pre-pass costs
    one walk of a report that is being read anyway.

    The book value is what segmentation uses. Page values are recorded as evidence only: they
    measure within 1 point of the book value in every book, so a per-page threshold would buy
    nothing and would make one page's short line count move another page's word boxes.
    """

    book_counts: Counter[int] = Counter()
    page_counts_by_name: dict[str, WordGapThreshold] = {}
    for page in project.pages:
        if not page.accepted:
            continue
        counts: Counter[int] = Counter()
        for _, _, candidate, visible_text in matched_lines(page):
            if not visible_text.split() or not candidate.horizontal_ink_profile:
                continue
            counts.update(word_gap_lengths(candidate.horizontal_ink_profile))
        if not counts:
            continue
        page_counts_by_name[page.page_name] = derive_word_gap_threshold(counts)
        book_counts.update(counts)
    return derive_word_gap_threshold(book_counts), page_counts_by_name


def _measure_page(
    root: Path,
    *,
    project_id: str,
    page: PageAlignment,
    measurement: PageMeasurement | None,
    text_left_by_class: Mapping[str, int],
    wants_evidence: bool,
    book_gap: WordGapThreshold,
    page_gap: WordGapThreshold | None,
    witness: BookWitness | None,
) -> PageTypography:
    page_class = "unknown" if measurement is None else measurement.page_class
    gap_evidence = _page_gap_evidence(page_gap)
    page_witness = None if witness is None else witness.page(page.page_name)
    if page_witness is not None and page_witness.image_sha256 != page.scan_sha256:
        page_witness = None
        gap_evidence = {**gap_evidence, "ocr_witness": "scan_hash_mismatch"}
    if measurement is None:
        return _excluded_page(
            page,
            page_class=page_class,
            extensions=gap_evidence,
            exclusions=("profile_page_missing",),
        )
    source_frame = page.source_frame
    foreground_bounds = _foreground_bounds(measurement)
    if source_frame is None or foreground_bounds is None:
        return _excluded_page(
            page,
            page_class=page_class,
            extensions=gap_evidence,
            exclusions=("foreground_bounds_unavailable",),
        )

    image_path = _resolve_scan(root, project_id=project_id, measurement=measurement)
    if image_path is None:
        return _excluded_page(
            page, page_class=page_class, extensions=gap_evidence, exclusions=("image_missing",)
        )

    snapshot_sha256: str | None = None
    try:
        with open_image_snapshot(image_path) as snapshot:
            snapshot_sha256 = snapshot.sha256
            if snapshot.sha256 != page.scan_sha256:
                return _excluded_page(
                    page,
                    page_class=page_class,
                    extensions=gap_evidence,
                    exclusions=("scan_hash_mismatch",),
                )
            rebuilt_threshold = measure_image_snapshot(snapshot).grayscale_threshold
            if rebuilt_threshold != measurement.grayscale_threshold:
                return _excluded_page(
                    page,
                    page_class=page_class,
                    extensions=gap_evidence,
                    exclusions=("mask_mismatch",),
                    evidence=(
                        {
                            "code": "grayscale_threshold_mismatch",
                            "recorded": measurement.grayscale_threshold,
                            "rebuilt": rebuilt_threshold,
                        },
                    ),
                )
            mask, _rejected = build_candidate_mask(
                snapshot, source_frame=source_frame, foreground_bounds=foreground_bounds
            )
    except (OSError, ValueError, RuntimeError) as error:
        return _excluded_page(
            page,
            page_class=page_class,
            extensions=gap_evidence,
            exclusions=("image_unreadable",),
            diagnostics=(
                ProfileDiagnostic(
                    code="image_unreadable",
                    message=str(error),
                    project_id=project_id,
                    page_name=page.page_name,
                ),
            ),
        )

    try:
        if sha256_file(image_path) != snapshot_sha256:
            return _excluded_page(
                page, page_class=page_class, extensions=gap_evidence, exclusions=("source_changed",)
            )
    except OSError:
        return _excluded_page(
            page, page_class=page_class, extensions=gap_evidence, exclusions=("source_changed",)
        )

    matched = matched_lines(page)
    mismatch = _mask_mismatch(mask, matched)
    if mismatch is not None:
        return _excluded_page(
            page,
            page_class=page_class,
            extensions=gap_evidence,
            exclusions=("mask_mismatch",),
            evidence=(mismatch,),
        )

    text_left = text_left_by_class.get(page_class)
    lines, pitches, tally, gap_tally = _measure_lines(
        mask,
        matched=matched,
        text_left=text_left,
        gap_threshold_px=book_gap.threshold_px,
        page_witness=page_witness,
    )
    gap_evidence = {**gap_evidence, **gap_tally.to_extensions()}
    if witness is not None:
        gap_evidence = {**gap_evidence, **tally.to_extensions()}
    pooling = pool_page(page.page_name, lines, pitches)
    return PageTypography(
        page_name=page.page_name,
        page_class=page_class,
        scan_sha256=page.scan_sha256,
        source_frame=source_frame,
        matched_line_count=len(lines),
        measured_line_count=sum(1 for line in lines if line.baseline_row_px is not None),
        excluded_line_count=sum(1 for line in lines if line.baseline_row_px is None),
        reconciled_line_count=sum(1 for line in lines if line.words_reconciled),
        baseline_pitch_count=len(pitches),
        evidence_sampled=wants_evidence,
        lines=lines if wants_evidence else (),
        estimates=pooling.estimates,
        skew_slope_median=pooling.skew.median,
        skew_slope_mad=pooling.skew.median_absolute_deviation,
        extensions=gap_evidence,
    )


def _page_gap_evidence(page_gap: WordGapThreshold | None) -> dict[str, object]:
    """One page's own Otsu split, recorded so a later slice can see within-book variation.

    Segmentation never reads these: the book value is what runs. A page whose matched lines
    carry no gap at all gets no keys rather than a fabricated zero.
    """

    if page_gap is None:
        return {}
    return {
        "word_gap_otsu_threshold_px": page_gap.threshold_px,
        "word_gap_otsu_valley_ratio": page_gap.valley_ratio,
        "word_gap_sample_count": page_gap.sample_count,
    }


def _measure_lines(
    mask: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    *,
    matched: Sequence[tuple[int, int, WireLineCandidate, str]],
    text_left: int | None,
    gap_threshold_px: int | None,
    page_witness: PageWitness | None,
) -> tuple[tuple[LineTypography, ...], tuple[int, ...], _WitnessTally, _GapTally]:
    results: list[tuple[int, LineTypographyMeasurement | LineTypographyExclusion]] = []
    lines: list[LineTypography] = []
    witnessed = fragments = reconciled_after = 0
    gaps = run_errors = 0
    for candidate_ordinal, source_ordinal, candidate, visible_text in matched:
        box = _box(candidate)
        result = measure_line_typography(mask, box)
        results.append((candidate_ordinal, result))
        row = _line_row(
            mask,
            candidate_ordinal=candidate_ordinal,
            source_ordinal=source_ordinal,
            candidate=candidate,
            visible_text=visible_text,
            result=result,
            text_left=text_left,
            gap_threshold_px=gap_threshold_px,
            page_witness=page_witness,
        )
        lines.append(row)
        if row.word_run_count is not None and row.source_word_count is not None:
            gaps += max(0, row.source_word_count - 1)
            run_errors += abs(row.word_run_count - row.source_word_count)
        witnessed += row.extensions.get("ocr_first_run_text") is not None
        fragments += row.extensions.get("continuation_fragment") is True
        reconciled_after += _reconciles_after_witness(row)
    pitches = tuple(pitch.pitch_px for pitch in measure_baseline_pitches(results))
    tally = _WitnessTally(
        witnessed=witnessed,
        continuation_fragments=fragments,
        reconciled_after_witness=reconciled_after,
    )
    return tuple(lines), pitches, tally, _GapTally(gaps=gaps, run_errors=run_errors)


def _reconciles_after_witness(row: LineTypography) -> bool:
    """Whether this line agrees with its transcription once a witnessed fragment is discounted.

    A flagged line's first run has no transcription word, so the comparison that matters is one
    run fewer. A line that already reconciles counts as it stands.
    """

    if row.words_reconciled:
        return True
    if row.extensions.get("continuation_fragment") is not True:
        return False
    if row.word_run_count is None or row.source_word_count is None:
        return False
    return row.word_run_count - 1 == row.source_word_count


def _line_row(
    mask: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    *,
    candidate_ordinal: int,
    source_ordinal: int,
    candidate: WireLineCandidate,
    visible_text: str,
    result: LineTypographyMeasurement | LineTypographyExclusion,
    text_left: int | None,
    gap_threshold_px: int | None,
    page_witness: PageWitness | None,
) -> LineTypography:
    """One matched line's wire row, measured or excluded.

    An excluded line still gets a row. It keeps its box, its observed width and height, the
    usable segment count, and the exclusion code, because the measured 13.9 percent line loss
    is the evidence a later labeler needs rather than noise to discard.
    """

    box = _box(candidate)
    indent = None if text_left is None else box[0] - text_left
    if not isinstance(result, LineTypographyMeasurement):
        return LineTypography(
            candidate_ordinal=candidate_ordinal,
            source_ordinal=source_ordinal,
            box=box,
            line_ink_width_px=candidate.width,
            line_ink_height_px=candidate.height,
            line_indent_px=indent,
            segment_count=result.segment_count,
            exclusions=(result.reason,),
        )
    threshold = (
        x_height_gap_threshold(result.x_height_px) if gap_threshold_px is None else gap_threshold_px
    )
    reconciled = measure_line_words(
        mask, box, candidate.horizontal_ink_profile, visible_text, threshold
    )
    line_witness = _witness_first_run(
        page_witness,
        box=box,
        profile=candidate.horizontal_ink_profile,
        gap_threshold_px=threshold,
        visible_text=visible_text,
        over_split=reconciled.word_run_count > reconciled.source_word_count,
    )
    return LineTypography(
        candidate_ordinal=candidate_ordinal,
        source_ordinal=source_ordinal,
        box=box,
        line_ink_width_px=candidate.width,
        line_ink_height_px=candidate.height,
        line_indent_px=indent,
        baseline_row_px=result.baseline_row_px,
        x_height_top_row_px=result.x_height_top_row_px,
        x_height_px=result.x_height_px,
        ascender_extent_px=result.ascender_extent_px,
        descender_extent_px=result.descender_extent_px,
        stroke_width_px=result.stroke_width_px,
        skew_slope=result.skew_slope,
        segment_count=result.segment_count,
        word_run_count=reconciled.word_run_count,
        source_word_count=reconciled.source_word_count,
        words_reconciled=reconciled.words_reconciled,
        words=tuple(
            WordTypography(ordinal=index, box=word.box, following_gap_px=word.following_gap_px)
            for index, word in enumerate(reconciled.words)
        ),
        extensions={} if line_witness is None else line_witness.to_extensions(),
    )


def _witness_first_run(
    page_witness: PageWitness | None,
    *,
    box: tuple[int, int, int, int],
    profile: Sequence[float],
    gap_threshold_px: int,
    visible_text: str,
    over_split: bool,
) -> LineWitness | None:
    """Ask the recognizer what sits where this line's first ink run does.

    A line with no transcription word and a line with no ink get no witness, because there is
    nothing for the read to disagree with.
    """

    if page_witness is None:
        return None
    words = visible_text.split()
    runs = find_word_runs(profile, gap_threshold_px)
    if not words or not runs:
        return None
    return witness_line(
        page_witness,
        box=box,
        first_run=runs[0],
        source_first_word=words[0],
        over_split=over_split,
    )


def matched_lines(
    page: PageAlignment,
) -> tuple[tuple[int, int, WireLineCandidate, str], ...]:
    """Every match operation as `(candidate_ordinal, source_ordinal, candidate, text)`.

    Candidate ordinal, not source ordinal, is what `baseline_pitch_px` uses for adjacency:
    two candidates one ordinal apart are physically consecutive lines on the page, which is
    the thing a pitch is supposed to measure.
    """

    candidates = {candidate.ordinal: candidate for candidate in page.candidates}
    source_lines = {line.ordinal: line for line in page.source_lines}
    matched: list[tuple[int, int, WireLineCandidate, str]] = []
    for operation in page.operations:
        if operation.kind != "match":
            continue
        candidate_ordinal = operation.candidate_ordinal
        source_ordinal = operation.source_ordinal
        if candidate_ordinal is None or source_ordinal is None:
            continue
        candidate = candidates.get(candidate_ordinal)
        source_line = source_lines.get(source_ordinal)
        if candidate is None or source_line is None:
            continue
        matched.append((candidate_ordinal, source_ordinal, candidate, source_line.visible_text))
    matched.sort(key=lambda item: item[0])
    return tuple(matched)


def _mask_mismatch(
    mask: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    matched: Sequence[tuple[int, int, WireLineCandidate, str]],
) -> dict[str, object] | None:
    """The first matched candidate the rebuilt mask fails to reproduce, if any.

    The reproduction formula is `_extract_band_candidates`'s own: column counts over
    `mask[box]`, their sum as `foreground_pixels`, and each count divided by that sum as the
    profile. Any difference means the two stages no longer share one definition of ink.
    """

    for candidate_ordinal, _, candidate, _ in matched:
        x_start, y_start, x_end, y_end = _box(candidate)
        cropped = mask[y_start:y_end, x_start:x_end]
        column_counts = tuple(
            int(np.count_nonzero(cropped[:, column_index]))
            for column_index in range(x_end - x_start)
        )
        rebuilt_ink = sum(column_counts)
        if rebuilt_ink != candidate.foreground_pixels:
            return {
                "code": "foreground_pixels_mismatch",
                "candidate_ordinal": candidate_ordinal,
                "recorded": candidate.foreground_pixels,
                "rebuilt": rebuilt_ink,
            }
        rebuilt_profile = tuple(count / rebuilt_ink for count in column_counts)
        if rebuilt_profile != candidate.horizontal_ink_profile:
            return {
                "code": "horizontal_ink_profile_mismatch",
                "candidate_ordinal": candidate_ordinal,
                "recorded_length": len(candidate.horizontal_ink_profile),
                "rebuilt_length": len(rebuilt_profile),
            }
    return None


def _excluded_page(
    page: PageAlignment,
    *,
    page_class: str,
    exclusions: tuple[str, ...],
    extensions: Mapping[str, object],
    evidence: tuple[Mapping[str, object], ...] = (),
    diagnostics: tuple[ProfileDiagnostic, ...] = (),
) -> PageTypography:
    return PageTypography(
        page_name=page.page_name,
        page_class=page_class,
        scan_sha256=page.scan_sha256,
        source_frame=page.source_frame,
        matched_line_count=0,
        measured_line_count=0,
        excluded_line_count=0,
        reconciled_line_count=0,
        baseline_pitch_count=0,
        exclusions=exclusions,
        exclusion_evidence=evidence,
        diagnostics=diagnostics,
        extensions=extensions,
    )


def _box(candidate: WireLineCandidate) -> tuple[int, int, int, int]:
    x_start, y_start, x_end, y_end = candidate.box
    return x_start, y_start, x_end, y_end


def _foreground_bounds(measurement: PageMeasurement) -> tuple[int, int, int, int] | None:
    bounds = measurement.foreground_bounds
    if bounds is None or len(bounds) != 4:
        return None
    x_start, y_start, x_end, y_end = bounds
    return x_start, y_start, x_end, y_end


def _resolve_scan(root: Path, *, project_id: str, measurement: PageMeasurement) -> Path | None:
    """Resolve the page's scan the way `alignment.py` does, or `None` if it is not there."""

    try:
        reference = require_canonical_relative_reference(
            value=measurement.source_path, label="source_path"
        )
        path = (root / reference).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            return None
        project_reference = require_canonical_relative_reference(
            value=project_id, label="project reference", direct_child=True
        )
        project_directory = (root / project_reference).resolve()
        if not project_directory.is_dir():
            return None
        resolution = resolve_image_candidate(
            project_directory=project_directory,
            page_name=measurement.page_name,
            corpus_root=root,
        )
        if resolution.image_path is None:
            return None
        expected = corpus_relative_path(path=resolution.image_path, corpus_root=root)
    except (OSError, RuntimeError, ValueError):
        return None
    if measurement.source_path != expected:
        return None
    return resolution.image_path


def _read_bytes(path: Path, *, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"Could not read the {label}: {path}") from error
