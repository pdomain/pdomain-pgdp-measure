"""Command line for pdomain-pgdp-measure.

Five subcommands, one per measurement stage:

- ``rank`` — rank a local PGDP corpus and write a review report.
- ``profile`` — measure the scan geometry of the ranked pages.
- ``align`` — align PGDP F2 source lines against that geometry.
- ``typography`` — measure font-free typographic observables.
- ``glyphs`` — cut a labelled glyph inventory.

They were ``rank-pgdp`` through ``glyphs-pgdp`` in ``pdomain-ocr-synth``. The suffix is
dropped because the package name already carries it. Flags and positional arguments are
unchanged, and so are the wire contracts the reports declare.

Every handler imports what it needs inside the function, so ``--help`` costs no heavy import.
"""

from __future__ import annotations

import argparse
import sys
from hashlib import sha256
from pathlib import Path

from pdomain_pgdp_measure import __version__

USAGE_EXIT = 2
VALIDATION_EXIT = 3
DESTINATION_EXIT = 6

#: Mirrors ``typography_models.DEFAULT_EVIDENCE_PAGES_PER_BOOK`` so building the parser
#: costs no measurement import. ``tests/test_cli_typography_pgdp.py`` pins the two together.
_TYPOGRAPHY_EVIDENCE_PAGES_DEFAULT = 12


def build_parser() -> argparse.ArgumentParser:
    """Build the ``pgdp-measure`` parser with its five subcommands."""
    parser = argparse.ArgumentParser(
        prog="pgdp-measure",
        description="Font-free measurement of PGDP scans.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_rank = subparsers.add_parser(
        "rank",
        help="rank local PGDP projects and select review pages",
    )
    _ = p_rank.add_argument("corpus_root", help="local PGDP corpus root directory")
    _ = p_rank.add_argument(
        "--output",
        default="./pgdp-ranking.json",
        help="report path (default: ./pgdp-ranking.json)",
    )
    _ = p_rank.add_argument(
        "--project-limit",
        type=int,
        default=50,
        help="maximum ranked projects to report (default: 50)",
    )
    _ = p_rank.add_argument(
        "--pages-per-project",
        type=int,
        default=12,
        help="maximum selected review pages per project (default: 12)",
    )

    p_profile = subparsers.add_parser(
        "profile",
        help="measure selected local PGDP scan geometry",
    )
    _ = p_profile.add_argument("corpus_root", help="local PGDP corpus root directory")
    _ = p_profile.add_argument(
        "--ranking",
        required=True,
        help="M14 ranking report JSON path",
    )
    _ = p_profile.add_argument(
        "--output",
        required=True,
        help="write the source-geometry profile JSON here",
    )
    _ = p_profile.add_argument(
        "--whole-book",
        action="store_true",
        help="measure every page of each ranked project, but emit only the ranked pages",
    )

    p_align = subparsers.add_parser(
        "align",
        help="align local PGDP F2 source lines with measured scan geometry",
    )
    _ = p_align.add_argument("corpus_root", help="local PGDP corpus root directory")
    _ = p_align.add_argument(
        "--profile",
        required=True,
        help="M15a source-geometry profile JSON path",
    )
    _ = p_align.add_argument(
        "--output",
        required=True,
        help="write the source-line alignment JSON here",
    )

    p_typography = subparsers.add_parser(
        "typography",
        help="measure font-free typographic observables from an alignment report",
    )
    _ = p_typography.add_argument("corpus_root", help="local PGDP corpus root directory")
    _ = p_typography.add_argument(
        "--alignment",
        required=True,
        help="M15b source-line alignment JSON path",
    )
    _ = p_typography.add_argument(
        "--profile",
        required=True,
        help="M15a source-geometry profile JSON path the alignment recorded",
    )
    _ = p_typography.add_argument(
        "--output",
        required=True,
        help="write the typography JSON here",
    )
    _ = p_typography.add_argument(
        "--geometry",
        default=None,
        help=(
            "optional OCR geometry JSONL for this book; marks a line whose first ink run is a "
            "continuation fragment the transcription does not carry"
        ),
    )
    _ = p_typography.add_argument(
        "--evidence-pages",
        type=int,
        default=_TYPOGRAPHY_EVIDENCE_PAGES_DEFAULT,
        help=(
            "emit per-line and per-word rows for this many measured pages per book "
            f"(default {_TYPOGRAPHY_EVIDENCE_PAGES_DEFAULT})"
        ),
    )

    p_glyphs = subparsers.add_parser(
        "glyphs",
        help="cut a per-book labelled glyph inventory from an alignment report",
    )
    _ = p_glyphs.add_argument("corpus_root", help="local PGDP corpus root directory")
    _ = p_glyphs.add_argument(
        "--alignment",
        required=True,
        help="M15b source-line alignment JSON path for one book",
    )
    _ = p_glyphs.add_argument(
        "--profile",
        required=True,
        help="M15a source-geometry profile JSON path the alignment recorded",
    )
    _ = p_glyphs.add_argument(
        "--output",
        required=True,
        help="write the pgdp-glyphs/v1 inventory directory here",
    )
    _ = p_glyphs.add_argument(
        "--geometry",
        default=None,
        help=(
            "optional OCR geometry JSONL for this book; harvests the running head and folio "
            "as the recognized label tier"
        ),
    )
    _ = p_glyphs.add_argument(
        "--no-atlas",
        dest="atlas",
        action="store_false",
        help="write the JSONL and manifest without rendering the per-character atlas",
    )

    return parser


def _cmd_rank(
    corpus_root: str,
    *,
    output: str,
    project_limit: int,
    pages_per_project: int,
) -> int:
    """Rank a local PGDP corpus and write its review report."""

    from pdomain_pgdp_measure import rank_corpus, write_report

    root = Path(corpus_root).expanduser()
    if not root.exists():
        print(f"error: corpus root does not exist: {root}", file=sys.stderr)
        return USAGE_EXIT
    if not root.is_dir():
        print(f"error: corpus root is not a directory: {root}", file=sys.stderr)
        return USAGE_EXIT
    if project_limit <= 0:
        print("error: --project-limit must be positive", file=sys.stderr)
        return USAGE_EXIT
    if pages_per_project <= 0:
        print("error: --pages-per-project must be positive", file=sys.stderr)
        return USAGE_EXIT

    output_path = Path(output).expanduser()
    if output_path.resolve().is_relative_to(root.resolve()):
        print("error: report output must be outside the corpus root", file=sys.stderr)
        return USAGE_EXIT
    if output_path.exists() and not output_path.is_file():
        print(f"error: report output is not a file: {output_path}", file=sys.stderr)
        return USAGE_EXIT

    report = rank_corpus(
        root,
        project_limit=project_limit,
        pages_per_project=pages_per_project,
    )
    try:
        write_report(report, output_path, root)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE_EXIT

    selected_pages = sum(len(project.pages) for project in report.projects)
    print(f"report: {output_path}")
    print(f"projects seen: {report.corpus.projects_seen}")
    print(f"projects ranked: {report.corpus.projects_ranked}")
    print(f"diagnostics: {len(report.diagnostics)}")
    print(f"selected pages: {selected_pages}")
    return 0


def _cmd_profile(corpus_root: str, *, ranking: str, output: str, whole_book: bool = False) -> int:
    """Measure selected local PGDP scans and write a geometry profile."""

    from pdomain_pgdp_measure.image_measurement import SnapshotSpoolError
    from pdomain_pgdp_measure.profile_input import load_profile_snapshot, read_profile_snapshot
    from pdomain_pgdp_measure.profile_models import ProfileReport
    from pdomain_pgdp_measure.profiling import profile_methods, profile_selection
    from pdomain_pgdp_measure.report import write_report

    root = Path(corpus_root).expanduser()
    if not root.exists():
        print(f"error: corpus root does not exist: {root}", file=sys.stderr)
        return USAGE_EXIT
    if not root.is_dir():
        print(f"error: corpus root is not a directory: {root}", file=sys.stderr)
        return USAGE_EXIT

    ranking_path = Path(ranking).expanduser()
    output_path = Path(output).expanduser()
    if output_path.resolve().is_relative_to(root.resolve()):
        print("error: profile output must be outside the corpus root", file=sys.stderr)
        return DESTINATION_EXIT
    if output_path.resolve() == ranking_path.resolve():
        print("error: profile output must differ from the ranking input", file=sys.stderr)
        return DESTINATION_EXIT
    if output_path.exists() and not output_path.is_file():
        print(f"error: profile output is not a file: {output_path}", file=sys.stderr)
        return DESTINATION_EXIT

    try:
        ranking_snapshot = read_profile_snapshot(ranking_path)
        selection = load_profile_snapshot(ranking_snapshot, corpus_root=root, whole_book=whole_book)
        ranking_sha256 = sha256(ranking_snapshot).hexdigest()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return USAGE_EXIT

    try:
        projects = profile_selection(selection)
    except SnapshotSpoolError as error:
        print(f"error: {error}", file=sys.stderr)
        return DESTINATION_EXIT
    report = ProfileReport(
        source_ranking={"algorithm_version": "pgdp-rank/v1", "sha256": ranking_sha256},
        methods=profile_methods(),
        projects=projects,
    )
    try:
        write_report(report, output_path, root)
    except (OSError, ValueError, ExceptionGroup) as error:
        print(f"error: {error}", file=sys.stderr)
        return DESTINATION_EXIT

    pages = tuple(page for project in report.projects for page in project.pages)
    measured_pages = sum(page.foreground_pixels is not None for page in pages)
    diagnostics = len(report.diagnostics) + sum(
        len(project.diagnostics) + sum(len(page.diagnostics) for page in project.pages)
        for project in report.projects
    )
    print(f"report: {output_path}")
    print(f"projects profiled: {len(report.projects)}")
    print(f"pages measured: {measured_pages}")
    print(f"pages excluded: {len(pages) - measured_pages}")
    if whole_book:
        # Whole-book mode pools over pages it does not emit, so report both.
        pooled_pages = sum(
            max(
                (estimate.sample_count for estimate in project.pooled_estimates),
                default=0,
            )
            for project in report.projects
        )
        print(f"pages pooled: {pooled_pages}")
    print(f"diagnostics: {diagnostics}")
    return 0


def _cmd_align(corpus_root: str, *, profile: str, output: str) -> int:
    """Align local PGDP source lines and write a snapshot-safe report."""

    from pdomain_pgdp_measure.alignment import build_alignment_report
    from pdomain_pgdp_measure.report import write_report

    root = Path(corpus_root).expanduser()
    if not root.exists():
        print(f"error: corpus root does not exist: {root}", file=sys.stderr)
        return USAGE_EXIT
    if not root.is_dir():
        print(f"error: corpus root is not a directory: {root}", file=sys.stderr)
        return USAGE_EXIT
    profile_path = Path(profile).expanduser()
    output_path = Path(output).expanduser()
    if output_path.resolve() == profile_path.resolve():
        print("error: alignment output must differ from the profile input", file=sys.stderr)
        return DESTINATION_EXIT
    if output_path.exists() and not output_path.is_file():
        print(f"error: alignment output is not a file: {output_path}", file=sys.stderr)
        return DESTINATION_EXIT
    try:
        report = build_alignment_report(root, profile_path, tool_version=__version__)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return VALIDATION_EXIT
    try:
        write_report(report, output_path, root)
    except (OSError, ValueError, ExceptionGroup) as error:
        print(f"error: {error}", file=sys.stderr)
        return DESTINATION_EXIT
    return 0


def _cmd_typography(
    corpus_root: str,
    *,
    alignment: str,
    profile: str,
    output: str,
    evidence_pages: int,
    geometry: str | None = None,
) -> int:
    """Measure font-free typographic observables and write a snapshot-safe report."""

    from pdomain_pgdp_measure.report import write_report
    from pdomain_pgdp_measure.typography import build_typography_report

    root = Path(corpus_root).expanduser()
    if not root.exists():
        print(f"error: corpus root does not exist: {root}", file=sys.stderr)
        return USAGE_EXIT
    if not root.is_dir():
        print(f"error: corpus root is not a directory: {root}", file=sys.stderr)
        return USAGE_EXIT
    if evidence_pages < 0:
        print("error: --evidence-pages must be nonnegative", file=sys.stderr)
        return USAGE_EXIT
    alignment_path = Path(alignment).expanduser()
    profile_path = Path(profile).expanduser()
    output_path = Path(output).expanduser()
    geometry_path = None if geometry is None else Path(geometry).expanduser()
    sources = [("alignment", alignment_path), ("profile", profile_path)]
    if geometry_path is not None:
        sources.append(("geometry", geometry_path))
    for label, source in sources:
        if output_path.resolve() == source.resolve():
            print(f"error: typography output must differ from the {label} input", file=sys.stderr)
            return DESTINATION_EXIT
    if output_path.exists() and not output_path.is_file():
        print(f"error: typography output is not a file: {output_path}", file=sys.stderr)
        return DESTINATION_EXIT
    try:
        report = build_typography_report(
            root,
            alignment_path,
            profile_path,
            tool_version=__version__,
            evidence_pages=evidence_pages,
            geometry_path=geometry_path,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return VALIDATION_EXIT
    try:
        write_report(report, output_path, root)
    except (OSError, ValueError, ExceptionGroup) as error:
        print(f"error: {error}", file=sys.stderr)
        return DESTINATION_EXIT
    return 0


def _cmd_glyphs(
    corpus_root: str,
    *,
    alignment: str,
    profile: str,
    output: str,
    atlas: bool = True,
    geometry: str | None = None,
) -> int:
    """Cut one book's labelled glyph inventory and write it as a directory."""

    from pdomain_pgdp_measure.glyphs import build_glyph_inventory, write_glyph_inventory

    root = Path(corpus_root).expanduser()
    if not root.exists():
        print(f"error: corpus root does not exist: {root}", file=sys.stderr)
        return USAGE_EXIT
    if not root.is_dir():
        print(f"error: corpus root is not a directory: {root}", file=sys.stderr)
        return USAGE_EXIT
    alignment_path = Path(alignment).expanduser()
    profile_path = Path(profile).expanduser()
    output_path = Path(output).expanduser()
    if output_path.exists() and not output_path.is_dir():
        print(f"error: glyph inventory output is not a directory: {output_path}", file=sys.stderr)
        return DESTINATION_EXIT
    geometry_path = None if geometry is None else Path(geometry).expanduser()
    try:
        harvest = build_glyph_inventory(
            root,
            alignment_path,
            profile_path,
            tool_version=__version__,
            geometry_path=geometry_path,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return VALIDATION_EXIT
    try:
        manifest = write_glyph_inventory(harvest, output_path, root, atlas=atlas)
    except (OSError, ValueError, ExceptionGroup) as error:
        print(f"error: {error}", file=sys.stderr)
        return DESTINATION_EXIT
    print(f"glyphs: {manifest.glyph_count}")
    for tier, count in sorted(manifest.glyph_count_by_tier.items()):
        print(f"  {tier}: {count}")
    print(f"pages harvested: {manifest.harvested_page_count}/{manifest.page_count}")
    print(f"atlas sheets: {len(manifest.atlas)}")
    return 0


_HANDLERS = {
    "rank": lambda args: _cmd_rank(
        args.corpus_root,
        output=args.output,
        project_limit=args.project_limit,
        pages_per_project=args.pages_per_project,
    ),
    "profile": lambda args: _cmd_profile(
        args.corpus_root,
        ranking=args.ranking,
        output=args.output,
        whole_book=args.whole_book,
    ),
    "align": lambda args: _cmd_align(
        args.corpus_root,
        profile=args.profile,
        output=args.output,
    ),
    "typography": lambda args: _cmd_typography(
        args.corpus_root,
        alignment=args.alignment,
        profile=args.profile,
        output=args.output,
        geometry=args.geometry,
        evidence_pages=args.evidence_pages,
    ),
    "glyphs": lambda args: _cmd_glyphs(
        args.corpus_root,
        alignment=args.alignment,
        profile=args.profile,
        output=args.output,
        geometry=args.geometry,
        atlas=args.atlas,
    ),
}


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(_HANDLERS[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
