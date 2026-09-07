# PGDP ranking and review queue

## Agent Index

- **Kind:** architecture
- **Status:** active
- **Owner:** CT
- **Created:** 2026-08-24
- **Last verified:** 2026-08-24
- **Provenance:** verified from shipped code, tests, commits, and corpus runs during the 2026-08-24 docgraph migration
- **Disposition:** Current truth promoted from the implemented M14 ranking and review-queue plan.
- **Promotes:** implemented `2026-08-22-pgdp-ranking-review-queue` plan.

`rank` builds a deterministic, bounded review queue from local PGDP
metadata and F2 source. It ranks projects and pages without opening scan images.

## Corpus input stays local and preserves F2 source

The command reads only immediate `projectID*` directories under the corpus
root. Within each directory, it reads `project.json`, `pages.json`, and
`rounds/F2.json`. It does not use a network service, large language model, OCR
engine, or scan measurement.

F2 loading preserves every decoded source body without loss. The parser
recognizes exact `/* */` local controls and `/# #/` continued controls. It
processes them in natural page order. Malformed, nested, overlapping, or
unmatched controls produce stable diagnostics. An invalid body remains
preserved as evidence, but it does not drive block-scoped content predicates.
Source-wide illustration, decoration, and uncertainty markers can still
contribute when they are present.

Missing or malformed metadata produces diagnostics. When available project and
page data permit a record, missing-image status remains recorded on the page.
The ranker checks image availability without decoding the image.

## Page features produce explicit component scores

Each page score records its component contributions. The feature weights and
caps are:

- Special format contributes 5.
- Italic, bold, and small-cap opening tags contribute up to 10 each.
- Dot leaders or aligned fields contribute 8 once.
- Table-like structure contributes 8.
- Poetry-like structure contributes 8.
- Quotation-like structure contributes 6.
- A multipart `L`, `M`, or `R` name contributes 6.
- An illustration or decoration contributes 4.
- An uncertainty note contributes 2.

Natural page ordering governs F2 parsing and page-score ties. It keeps page
names such as `2`, `10`, and their multipart forms in human reading order.
Diagnostics and selected pages use their own stable deterministic ordering.

## Project ranking rewards both strength and coverage

A project score is the sum of its ten highest page scores plus 20 points for
each target group present anywhere in the ranked pages. The target groups are
mixed typography, table-like structure, poetry, quotation, and multipart
layout. Projects sort by descending score and then by project ID.

Page selection visits those five target groups in that fixed order. It selects
an eligible page for each group, then fills remaining slots from the ranked
page list. One page occupies one slot even when it matches several groups.

By default, the command returns at most 50 projects and 12 pages per project.
The immutable `pgdp-rank/v1` report records those limits, corpus counts,
diagnostics, project and page scores, component scores, matched groups, and
selected pages.

## Report output is deterministic and corpus-safe

The report contains no absolute paths, timestamps, or host identifiers. It
uses stable natural ordering and stable diagnostics. Serialization writes
sorted, indented UTF-8 JSON followed by one newline.

The writer replaces a sibling temporary file atomically. It refuses an output
path inside the corpus, so report generation cannot modify corpus contents.

## Boundaries and limits

Ranking detects source patterns that make a page useful for review. It does not
assert semantic correctness or visual quality. A high score is evidence of
interesting F2 structure, not a measurement of the printed page.

The command does not open scans, measure geometry, recognize image text, infer
fonts, or call remote systems. Later profiling stages consume its bounded page
selection and perform scan work under separate versioned contracts.

## Evidence

- Code: `src/pdomain_pgdp_measure/models.py`,
  `src/pdomain_pgdp_measure/ordering.py`,
  `src/pdomain_pgdp_measure/f2.py`,
  `src/pdomain_pgdp_measure/features.py`,
  `src/pdomain_pgdp_measure/ranking.py`,
  `src/pdomain_pgdp_measure/report.py`, and
  `src/pdomain_pgdp_measure/cli.py`.
- Tests: `tests/test_pgdp_f2.py`, `tests/test_pgdp_features.py`,
  `tests/test_pgdp_ranking.py`, and `tests/test_cli_rank_pgdp.py`.
- Commits: `7a454f1`, `66cd619`, `41385e6`, `48027ee`, `da9ff31`,
  `0d118cc`, `9774904`, and `93507c9`.
- Corpus verification: two runs produced byte-identical reports. The ranker saw
  286 projects, ranked 285, emitted 1,777 stable diagnostics, and selected 600
  pages across 50 projects.
- Verified: 2026-08-22 in `docs/context/current-state.md`, then promoted on
  2026-08-24 during the docgraph migration.
