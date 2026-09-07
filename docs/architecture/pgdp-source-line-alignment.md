# PGDP source-line alignment

## Agent Index

- **Kind:** architecture
- **Status:** active
- **Owner:** CT
- **Created:** 2026-09-06
- **Last verified:** 2026-09-06
- **Provenance:** verified from shipped code, schemas, tests, and the measured five-book corpus runs
- **Disposition:** Current truth promoted from the implemented M15b source-line alignment,
  fragmented-band correction, and page-classification plans.
- **Promotes:** implemented `2026-08-23-pgdp-source-line-alignment`,
  `2026-08-31-pgdp-fragmented-band-correction`, and `2026-08-31-pgdp-page-classification` plans.
- **Read when:** implementing or reviewing `align`, PGDP line candidates, F2 source spans,
  running-head suppression, or scan-to-text alignment.
- **Search terms:** PGDP, M15b, F2 tokenizer, line candidate, gutter, dynamic programming,
  alignment, fragmented band, running head, page class.

`align` conservatively aligns lossless F2 source lines with source-frame scan line candidates
on eligible single-column PGDP pages. It reads a `pgdp-profile/v2` report and writes a separate
`pgdp-alignment/v3` report. It keeps rectification and typographic fitting out of scope.

## Input and output

The command is:

```text
pgdp-measure align CORPUS_ROOT --profile PROFILE_PATH --output OUTPUT_PATH
```

Both flags are required. `PROFILE_PATH` is a
[`pgdp-profile/v2`](pgdp-observed-geometry-profiling.md) report from `profile`. The output must
be a file outside `CORPUS_ROOT` and must not resolve to the profile input. The report is
`pgdp-alignment/v3`, written atomically as sorted, indented UTF-8 JSON.

The command reads project F2 text and scan images directly from the corpus. It never opens a font,
renders text, or reaches a network service.

## Eligible source lines come from a lossless F2 tokenizer

The tokenizer reads each F2 page string as Unicode and computes offsets in its UTF-8 encoding. It
splits on `\n`, `\r\n`, and `\r`, and it records each separator's own byte span. Every `SourceLine`
carries the page name, its ordinal, full and content byte spans, and original and normalized text.

Normalization is an analysis view, not a replacement string. It keeps visible case, punctuation,
spacing, and Unicode code points. It deletes recognized F2 control markers, proof notes, and `i`,
`b`, `sc`, `f`, `g`, and `tb` tags, and it turns `\r\n` and `\r` into `\n`. Every deletion and copy
is recorded as an ordered operation over a UTF-8 byte range, so concatenating the operations
reproduces the normalized text exactly.

Local `/* ... */` spans and cross-page `/# ... #/` spans get a stable identifier from their block
kind, opening page, and opening byte offset. An open continued block keeps that identity on every
page it spans. A page whose control markers are unmatched, nested, or overlapping stays diagnosed
but does not lose its other lines. Blank visible lines stay in the source record but drop out of the
sequence the aligner matches, so their absence is explicit rather than silent.

## Image line candidates come from connected components in two dimensions

Candidate extraction starts from the validated `pgdp-profile/v2` foreground bounds and ink bands. A
page needs at least two retained bands. Inside each band's y-range, the extractor derives a tight
foreground x-range, labels eight-connected components, and removes two kinds of noise before joining
what is left: full-frame components touching at least three source-frame edges (page border), and
components spanning at least 80 percent of foreground width at under 2 percent of foreground height
(long horizontal rules).

Surviving components are joined within their band by vertical overlap, center distance, and
horizontal gap. At most one candidate represents a band, storing its band ordinal, box, component
count, foreground pixel count, and horizontal ink profile.

### Version 1 excluded ordinary text pages over punctuation and dust

Version 1 rejected a page outright whenever a band's join left more than one cluster. That rule
could not tell a genuine second column from a stray period or a fleck of dust sitting apart from
its line, so it excluded ordinary single-column prose pages along with real multi-column ones.

Visual review of the first 25-page corpus run found this directly: `fragmented_band` excluded 23 of
25 pages, and 12 of them were plainly alignable prose. See [the corpus review
result](#measured-results) below for the run that found this.

### Version 2 added three fixes that made ordinary pages eligible again

`source-frame-components/v2` replaced the all-or-nothing rule with three additions, all published
together as `pgdp-alignment/v2`:

- **Cluster merge.** Clusters whose horizontal ranges overlap are merged until no pair overlaps,
  before fragmentation is counted.
- **Minor-ink filter.** A cluster holding under 2 percent of its band's ink is dropped from the
  fragmentation count, recorded as a `minor_ink_cluster` rejection, unless every cluster in the band
  falls under that share, in which case none are dropped.
- **Dominant-cluster candidate and rate limit.** The surviving cluster with the most ink becomes the
  band's candidate instead of the band contributing nothing. A page is excluded as `fragmented_band`
  only when the fraction of its measured bands that still fragment exceeds 0.35, not on any single
  fragmented band.

On the fixed 25-page review set, `fragmented_band` exclusions fell from 23 to 5, matching the 12
pages the earlier visual review judged alignable.

### The three band-identification fixes are the substance of v3

Correcting fragmentation was not enough. Every accepted page under `pgdp-alignment/v2` still risked
one defect: a page whose candidate count equalled its eligible source-line count could bind a
printed but F2-deleted running head to source line 0, shifting every match beneath it. Two accepted
pages in the first review, `p008.png` and `p179.png`, were shifted exactly this way, and
accepted-line precision measured near 62 percent.

`pgdp-alignment/v3` and its `source-frame-components/v4` extractor fix three related defects in how
a band is identified. All three had to ship together because fixing one alone made another page
worse:

- **Head-band selection by position, not by ordinal.** The classifier no longer assumes band 0 is
  the running head. A fleck of dust can hold an ink band of its own above the real head, and the
  profile wire format carries only a band's `y_start` and `y_end`, not its ink, so ink cannot
  separate dust from a genuine short line at classification time. `_head_band_ordinal` instead picks
  the first band at or below `center - window`, the first band that could plausibly be type, and
  every ordinal up to and including it is named as furniture. See
  `src/pdomain_pgdp_measure/page_templates.py`.
- **Decorative rules read as rules, not as match candidates.** `_remove_long_rules` only removed
  rules spanning 80 percent of foreground width, so a decorative rule under a chapter title survived
  and took a source line. `is_rule_band` now rejects a band under half the median band height and
  over 60 percent ink by area as a `rule_band`. Text rows reached 0.40 density at the 99th
  percentile over 17,205 matched candidates; measured rules ran 0.61 to 0.78.
- **The speck test narrowed at the same time.** Ink share alone could not separate dust from a
  short genuine line such as a section number. `is_speck_band` now also requires a band to be at
  most 31 percent of the page's median band height before it is dropped as a `speck_band`. Removing
  the rule fix alone, without narrowing the speck test, made a held-out page worse: wrong matches on
  `379.png` would have risen from about two to about twenty-nine, because the rule band had been
  absorbing the line a wrongly dropped speck test had displaced.

Neither the alignment wire version nor the profile wire version changed for this fix; only the two
method identifiers moved, to `source-frame-components/v4` and `first-band-templates/v3`. See
`docs/context/decisions.md`, "Geometry alone decides what a band is, and three tests share the job."

## The dynamic-programming alignment is deterministic

The aligner operates on visible nonblank source lines and ordered image candidates. It emits three
operations: `match` (cost 0 plus weak agreement penalties), `skip_text`, and `skip_image` (cost 1.0
each). A match's cost blends width, indentation, blank-line-gap, and style agreement between the
source line and the candidate box, using fixed weights of 0.55, 0.25, 0.15, and 0.05.

The selected path has the lowest total cost. Equal-cost transitions prefer `match`, then
`skip_image`, then `skip_text`, and source and candidate ordinals break any remaining tie. This
makes repeated runs over the same input byte-identical; two `align` runs over the fixed
25-page selection and over five whole books have both reproduced this.

Path cost is normalized by the larger of source count, candidate count, or 1. The uniqueness margin
is the absolute cost difference between the best path and the next-best path with a different
operation sequence. It is not divided by line count.

The algorithm never compares candidate boxes to recognized scan text. `align` does not use OCR
output as its own verification signal.

## A page is accepted only when its evidence clears every gate

A page needs between 4 and 80 eligible source lines. It must carry no border-dominated,
high-foreground, persistent-gutter, table-like, illustration-marker, or malformed-control exclusion.
Given that, it is accepted only when:

- at least 90 percent of its eligible source lines participate in a match;
- the absolute source-line and candidate count difference is at most the larger of 2 and 10 percent
  of the source-line count;
- normalized path cost is at most 0.22;
- the uniqueness margin is at least 0.15.

These constants are unchanged from the original M15b design; the fragmented-band and
band-identification fixes changed only which pages produce enough clean candidate evidence to reach
them. Confidence is always `null` and `confidence_kind` is always `"uncalibrated"`.

## A book, not the whole corpus, is what qualifies for synthesis

The original design gated the corpus on 70 percent eligible-page coverage. That gate was measured
on the 25-page review set, which M14 selects for typographic interest: against the other 1,360
pages of the same five books, the review set over-represents table-like structure 4.3 times, poetry
7.4 times, and aligned fields 5.7 times. Coverage measured on it could not represent an ordinary
book.

The 70 percent coverage gate was withdrawn on 2026-08-31 and replaced by per-book admission. A book
is admitted to synthesis when `align`, run with the profile's `--whole-book` mode, yields at
least 30 accepted pages (`MINIMUM_ACCEPTED_PAGES_PER_BOOK` in
`src/pdomain_pgdp_measure/alignment_review.py`). This threshold is an uncalibrated seed: nothing
downstream yet states how many pages a per-book typography fit needs, so corpus review is expected
to recalibrate it once one does.

Eligible-page coverage remains a reported per-book statistic. Nothing fails on it. The corpus-wide
gate keeps its other two rules: accepted-line precision must be at least 98 percent, and zero
accepted pages may be declared complex. See
`docs/specs/2026-08-31-pgdp-whole-book-yield-gate-design.md`.

## Every reviewed page reconciles into one of four counts

Corpus review classifies each reviewed selected page into exactly one of four disjoint counts, using
a fixed priority: source changed, unavailable or malformed, declared complex, then eligible.

`declared_complex` covers `probable_multi_column`, `persistent_gutter`, `table_like`,
`illustration_marker`, `border_dominated`, and `high_foreground_ratio`. `unavailable_or_malformed`
covers missing or unreadable F2 and images, missing profile pages, unavailable foreground bounds,
insufficient ink bands, blank pages, fragmented bands, out-of-range line counts, malformed controls,
and scan hash mismatches. `source_changed` is its own count. Everything else, including `proposed`
and alignment-quality exclusions, is `eligible`. Eligible-page coverage is accepted eligible pages
divided by all eligible pages.

## Measured results

The five-book corpus review moved through three stages as each defect was fixed.

| stage | fragmented pages | accepted pages | precision |
|---|---:|---:|---:|
| `pgdp-alignment/v1` (25-page review) | 23 of 25 | 0 | undefined |
| `pgdp-alignment/v2` extractor, 25-page review | 5 of 25 | 3 | undefined |
| `pgdp-alignment/v3`, running heads suppressed, whole books | — | 367 of 971 eligible | 0.9716 |
| after the two page-template fitting fixes | — | 665 across five books | 0.9974 |
| after the three band-identification fixes | — | 713 across five books | 1.0000 |

The final measurement judged 760 matched lines across a stratified sample of 26 accepted pages
from all five books, all correct, against the 0.98 gate. No accepted page was declared complex,
and all five books cleared the 30-page admission minimum with 713 accepted pages between them.
This ledger was built by a model reading rendered line crops against each source line, at the
owner's direction, not signed off row by row by a person. See `docs/context/decisions.md` and
`docs/context/current-state.md` for the full run-by-run detail.

An earlier measurement in the same sequence, after the two page-template fitting fixes but before
the three band-identification fixes, reported 665 accepted pages and precision 0.9974 over 781
matched lines, with 1,165 of 1,385 pages classified. The band-identification fixes then moved 35
more pages from `unknown` into a real class and raised accepted pages to 713. No later document
states a revised total classified-page count.

## Boundaries

Alignment reports source-frame geometry. It does not rectify rotation, perspective, or page warp;
that is M15c. It does not measure true baselines, x-height, or cap height as typography; that is a
separate, later contract (see `docs/architecture/pgdp-observed-geometry-profiling.md` for the
geometry it builds on).

It does not read multi-column reading order. A page carrying a probable column gutter is excluded,
not reordered. It does not transcribe or verify against OCR text. It does not identify typefaces,
fit fonts, or infer kerning, tracking, or word spacing. It does not read folio digits or
running-head text; page classification uses only band position. It does not use adjacent pages:
every measurement comes from one page's own geometry.

## Evidence

- Code: `src/pdomain_pgdp_measure/alignment_source.py`,
  `src/pdomain_pgdp_measure/alignment_image.py`,
  `src/pdomain_pgdp_measure/alignment_dp.py`,
  `src/pdomain_pgdp_measure/alignment_models.py`,
  `src/pdomain_pgdp_measure/alignment.py`,
  `src/pdomain_pgdp_measure/alignment_review.py`,
  `src/pdomain_pgdp_measure/page_templates.py`, and
  `src/pdomain_pgdp_measure/cli.py`.
- Schema: `schemas/pgdp-alignment-v1.schema.json`, `schemas/pgdp-alignment-v2.schema.json`, and
  `schemas/pgdp-alignment-v3.schema.json`.
- Tests: `tests/test_pgdp_alignment_source.py`, `tests/test_pgdp_alignment_image.py`,
  `tests/test_pgdp_alignment_dp.py`, `tests/test_pgdp_alignment_models.py`,
  `tests/test_pgdp_alignment.py`, `tests/test_pgdp_alignment_review.py`,
  `tests/test_pgdp_alignment_fixtures.py`, `tests/test_pgdp_page_templates.py`, and
  `tests/test_cli_align_pgdp.py`.
- Plans: retired on 2026-09-06 with tombstones in decisions (the decision log in `pdomain-ocr-synth`);
  they were `2026-08-23-pgdp-source-line-alignment`,
  `2026-08-31-pgdp-fragmented-band-correction`, and `2026-08-31-pgdp-page-classification`.
- Specs: `docs/specs/2026-08-23-pgdp-source-line-alignment-design.md`,
  `docs/specs/2026-08-31-pgdp-whole-book-page-templates-design.md`, and
  `docs/specs/2026-08-31-pgdp-whole-book-yield-gate-design.md`.
- Corpus verification: five whole-book runs over 1,385 pages measured deterministic replay,
  accepted-page counts per book, and accepted-line precision at three stages, ending at 1.0000
  over 760 rows from 26 pages across five books with 713 accepted pages and zero accepted
  declared-complex pages.
- Verified: measured counts recorded in `docs/context/current-state.md` and
  `docs/context/decisions.md`, promoted on 2026-09-06.
