# PGDP observed geometry profiling

## Agent Index

- **Kind:** architecture
- **Status:** active
- **Owner:** CT
- **Created:** 2026-08-24
- **Last verified:** 2026-09-06
- **Provenance:** verified from shipped code, schema, tests, commits, and corpus runs during the
  2026-08-24 docgraph migration; re-verified on 2026-09-06 against the shipped page-classification
  work that moved the report to `pgdp-profile/v2`
- **Disposition:** Current truth promoted from the implemented first M15 observed-geometry
  profiling plan and the implemented page-classification plan.
- **Promotes:** implemented `2026-08-22-pgdp-observed-geometry-profiler` and
  `2026-08-31-pgdp-page-classification` plans.

`profile` converts a `pgdp-rank/v1` review queue into a deterministic
`pgdp-profile/v2` report. It measures observed scan geometry in original raster
coordinates and keeps failures as evidence instead of numeric zeroes. Its
`--whole-book` mode also fits each book's page templates and classifies every
page against them.

## Input validation preserves ranking and scan evidence

A strict Pydantic boundary validates the complete ranking report before corpus
work begins. Corpus references resolve only through safe relative paths. The
profiler snapshots and hashes the original ranking bytes and the original bytes
of each available selected scan. A missing scan has a `null` SHA-256 value.

The profiler handles one full-resolution page at a time. Before decoding, it
copies the source bytes into bounded temporary-file spooling. It does not apply
EXIF orientation. Dimensions, bounds, margins, and bands therefore use source
raster coordinates. Each successfully decoded page records its original-byte
hash, decoded dimensions, image mode, and orientation metadata when present.

Missing, corrupt, blank, or otherwise unusable scans retain page records and
diagnostics. A failed measurement never becomes a numeric zero.

## Foreground measurement is integer and deterministic

Each decoded image becomes 8-bit grayscale. A deterministic 256-bin integer
Otsu calculation chooses the lowest threshold among all maximizing thresholds.
Pixels at or below that threshold are foreground.

Foreground bounds use half-open coordinates. Four margins are derived from
those bounds in the same source frame. Blank pages have no bounds or margins.

The active-row threshold is the larger of two pixels and 0.5% of print width,
rounded up. Runs separated by at most one inactive row join. The profiler
discards joined runs shorter than two rows.

The report stores every retained raw ink band. It also derives median band
height and median pitch between successive band tops. A one-band page
contributes height but has no pitch sample.

## Whole-book mode measures every page while emitting only the ranked ones

By default the profiler measures only the pages `rank` selected. The `--whole-book` flag
measures every `*.png` file directly under each ranked project directory instead, in sorted order,
while the report still emits only the ranked pages. Pooled estimates and page templates are fit
over every measured page, and each project records how many pages were measured against how many
were emitted, so a reader can see that book-level statistics rest on more evidence than the report
lists.

This keeps per-page processing unchanged. The profiler still measures one full-resolution image at
a time; `--whole-book` only widens which pages that loop visits.

## Per-book page templates classify each page against its own type page

A letterpress book is set once, so its text block holds the same position across the whole volume.
Whole-book profiling uses that fact. It pools the first ink-band top across every measured page in
a book with the existing `median-mad/v1` method, then fits a small set of page templates from that
pooled position: `normal_recto`, `normal_verso`, and `chapter_opening`. Recto and verso are fit as
separate templates because a book's binding margin can shift the text block between the two sides
by tens of pixels.

Each page is classified against those templates by how far its own first band sits from them. A
page whose offset falls between the head window and the chapter-opening sink is classified
`unknown` rather than forced into either template. Front matter and plates have no measured
discriminator of their own in this corpus, so they also fall to `unknown`. A book whose first-band
position is not steady enough to trust yields no templates, and every one of its pages is
`unknown`.

`unknown` suppresses nothing. Only `normal_recto` and `normal_verso` pages, and only when their
first band falls inside the book's head window, name a furniture band. `pgdp-alignment/v3` reads
that list and drops those bands before candidate extraction, so a running head or folio never
becomes a line candidate. See [PGDP source-line
alignment](pgdp-source-line-alignment.md#the-three-band-identification-fixes-are-the-substance-of-v3)
for how the aligner uses the class and for the head-band-selection fix that followed it.

### Running heads exist because F2 deletes them, not the scan

PGDP proofers strip running heads and folios from F2 transcription, so a page's printed head has no
matching source line. Before classification, a page whose candidate count matched its eligible
source-line count could bind that head to source line 0, shifting every match beneath it. Page
classification exists to stop exactly that failure at its source.

### Two page-template fitting defects held classification back

The first shipped fitting logic underperformed on ordinary books, for two unrelated reasons.

The recto/verso split originally read the trailing digit of the file name to guess binding side.
One book in the corpus numbers its page files at ten times the printed folio, so nearly every file
name ended in zero and read as the same side. Splitting normal pages by their text-block left edge
instead, and using the file name only to label which geometric group is which, fixed this.

The book-steadiness ceiling was `2` pixels of median absolute deviation in the first-band top.
Two books in the corpus have a running head that is clearly present and periodic but noisy, with a
deviation around 16 pixels, and the 2px ceiling refused templates for both. Raising the ceiling to
`25` admits them while keeping the resulting head window, `max(8, 3 x deviation)`, below the 76px
offset at which genuine top-of-page text was observed to begin.

Fixing both raised classified pages from 527 of 1,385 to 1,165, and raised `pgdp-alignment/v3`
accepted pages from 367 to 665 across the same five whole books. See [PGDP source-line
alignment](pgdp-source-line-alignment.md#measured-results) for the fuller measurement sequence,
including the later band-identification fixes that raised accepted pages further, to 713.

## Exclusions keep observations while limiting estimates

Truth classes distinguish observed, derived, and pooled values. Every estimate
states its unit, coordinate frame, method, sample count, evidence, and
exclusions. Confidence is `null`, and `confidence_kind` is `uncalibrated`.

Project estimates pool eligible page values with the median and median absolute
deviation. Pooling does not remove outliers. Exclusions are metric-specific. A
page can contribute to one estimate while remaining ineligible for another.

Blank and corrupt pages contribute no numeric values to pooled estimates.
Blank pages can retain raw observations such as zero foreground pixels and a
threshold. One-band pages contribute band height but not pitch.
Border-dominated pages and pages with more than 60% foreground keep their raw
observations, but do not contribute margins, height, or pitch to project pooling.

A page is border-dominated when foreground occupies at least 50% of both
opposing outer strips on either axis. Each strip is the larger of one pixel and
1% of the corresponding dimension, rounded up.

## Report output separates measurement from ranking

The profile report is distinct from its ranking input. Serialization is
deterministic, uses atomic replacement, and requires output outside the corpus.
Invalid input exits with status 2. Completed measurement, including results
with diagnostics, exits with status 0. Output failures exit with status 6.

## Boundaries and limits

The profiler reports pixel observations in the unmodified source raster. It
does not rectify rotation, perspective, or page warp. It does not estimate
physical units or verified baselines.

The profiler does not detect columns, interpret semantics, measure text
alignment, infer fonts or styles, or choose renderer settings. Those tasks need
additional evidence and separate contracts.

Page classification is geometry only. It reads no folio digit and no running-head text, and it
does not read adjacent pages. A book whose type page is not steady enough to trust yields no
templates, and a page whose offset falls in the trough between the head window and the
chapter-opening sink stays `unknown` rather than being forced into a class.

## Evidence

- Code: `src/pdomain_pgdp_measure/profile_models.py`,
  `src/pdomain_pgdp_measure/profile_input.py`,
  `src/pdomain_pgdp_measure/paths.py`,
  `src/pdomain_pgdp_measure/image_measurement.py`,
  `src/pdomain_pgdp_measure/profiling.py`,
  `src/pdomain_pgdp_measure/page_templates.py`,
  `src/pdomain_pgdp_measure/report.py`, and
  `src/pdomain_pgdp_measure/cli.py`.
- Schema: `schemas/pgdp-profile-v1.schema.json` (retained as schema and git history; a version 1
  payload no longer loads) and `schemas/pgdp-profile-v2.schema.json` (current).
- Tests: `tests/test_pgdp_profile_models.py`,
  `tests/test_pgdp_profile_input.py`,
  `tests/test_pgdp_image_measurement.py`, `tests/test_pgdp_profiling.py`,
  `tests/test_pgdp_page_templates.py`,
  `tests/test_cli_profile_pgdp.py`, and
  `tests/test_pgdp_geometry_fixtures.py`.
- Commits: `c97b108`, `4b13fca`, `8f01f2a`, `895e355`, `83e0a09`,
  `ed86d52`, `f182774`, `5d0ea55`, `a7e5565`, `c058327`, `c9fadb5`,
  `a176747`, and `f29661f` (original M15a slice).
- Plans: `2026-08-31-pgdp-page-classification`, retired on 2026-09-06 with a tombstone in
  decisions (the decision log in `pdomain-ocr-synth`).
- Specs: `docs/specs/2026-08-31-pgdp-whole-book-page-templates-design.md`.
- Corpus verification: two five-project, 60-page profiles were byte-identical,
  and all pages were measured. Three one-band pages were excluded only from
  pooled pitch. Ten reviewed overlays had no false bounds merge or split.
  Reviewed fixtures had zero bounds-edge error and mean band intersection over
  union of 1.0, with the expected border and blank exclusions. Five whole-book
  runs over 1,385 pages showed classified pages rising from 527 to 1,165 after
  the two template-fitting fixes, and reproduced byte-identical profiles on
  replay.
- Verified: corpus results recorded in `docs/context/current-state.md` and
  `docs/context/decisions.md`; promoted on 2026-08-24 during the docgraph
  migration and re-verified on 2026-09-06 against the shipped page-classification
  work.
