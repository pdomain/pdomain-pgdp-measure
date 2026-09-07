# PGDP font-free typography

## Agent Index

- **Kind:** architecture
- **Status:** active
- **Owner:** CT
- **Created:** 2026-09-06
- **Last verified:** 2026-09-06
- **Provenance:** verified from shipped code, schemas, tests, and the measured five-book corpus
  runs
- **Disposition:** Current truth promoted from the complete M15d font-free typographic
  observables and M15e OCR witness plans.
- **Promotes:** complete `2026-09-02-pgdp-font-free-typographic-observables` and
  `2026-09-04-pgdp-ocr-witness-for-continuation-fragments` plans.
- **Read when:** measuring scan-derived typography, changing word segmentation, wiring the OCR
  witness, or judging what a consumer of `pgdp-typography/v1` can rely on.
- **Search terms:** PGDP, M15d, M15e, typography, x-height, baseline, word gap, stroke width,
  continuation fragment, OCR witness, DocTR, word reconciliation.

`typography` reads a `pgdp-alignment/v3` report together with the `pgdp-profile/v2` profile
that alignment recorded. It rebuilds each accepted page's ink mask, measures per-line typographic
observables, and pools them into per-book styles. It writes a deterministic `pgdp-typography/v1`
report and opens no font. An optional `--geometry` input adds an OCR witness that flags a line
whose leading ink run is a word fragment the transcription does not carry.

## The command refuses to run on a profile that does not match the alignment

The typography report records `alignment_label`, `alignment_sha256`, `profile_label`, and
`profile_sha256`. Before doing any work, `build_typography_report` hashes the profile file it was
handed and compares that hash against `profile_sha256` recorded in the alignment report. A
mismatch raises a `ValueError` and the command exits with the validation status. The check exists
because the profile is where the page ink mask and the Otsu threshold come from, and a stale or
swapped profile would silently mismeasure every line on top of it.

The refusal discipline goes further than the file hash. On every accepted page, the command
rebuilds the same page ink mask the alignment extractor built and checks two things: that the
rebuilt mask reproduces each matched candidate's stored `foreground_pixels` and
`horizontal_ink_profile` exactly, and that the rebuilt Otsu threshold equals the one alignment
recorded. A page that fails either check is excluded as `mask_mismatch`, carrying both values as
evidence, rather than measured on drifted ink.

Across 665 accepted pages in five books, all 17,205 matched candidates reproduced their recorded
values and no page was excluded as `mask_mismatch`. That is not a fixture result. It is the whole
corpus.

## Nothing here opens a font, renders text, or leaves the source frame

Per matched line, the command measures: baseline row, x-height, ascender and descender extents,
stroke width, skew slope, word runs, and a per-word ink box. Every value stays in the `source`
coordinate frame, the same unrectified raster frame the profile and alignment reports use, and
every unit is pixels.

This is deliberately font-free. The command imports no font module, opens no font file, and
renders no text. That discipline is the point of the slice, not an incidental limit. The
milestone's eventual target is inverse rendering: ranking candidate fonts by re-rendering a known
transcription and scoring the render against observed ink. Scoring a candidate needs a measured
target to score against, and nothing in this repo had measured where a baseline sits, how tall an
x-height runs, or how thick a stroke is, before this slice shipped. Building font candidates first
would have meant searching a space with no measured target.

The discipline is also enforced, not just stated. `typography_models.py` carries a denylist of
substrings a report key may never contain, covering font identity, point size, font advances, side
bearings, tracking, native word-space width, and any physical-unit suffix such as `_mm`, `_pt`,
`_inch`, or `_dpi`. A test asserts the real corpus report names none of them.

## Pooling runs in two stages so one dense page cannot dominate a book

Estimates pool page medians first, then pool those page medians into a book estimate, grouped by
the page's shipped `page_class`. This is not only a robustness choice. The shipped `Estimate` type
requires `sample_count == len(evidence_pages)` with no duplicate pages, and a book carries roughly
30 matched lines per page, so pooling raw lines directly into a book estimate would break that
invariant. Two-stage pooling keeps `sample_count` equal to the contributing page count and records
the underlying line count separately in `extensions`.

Emitting a row for every line and word would also be its own problem. Five books at roughly 21,000
matched lines works out to on the order of 150,000 word-level records, which would dwarf the
report's own product. So per-line and per-word rows are emitted only for the first
`--evidence-pages` accepted pages of each book in natural page order, defaulting to 12. Pooled
book estimates and per-page aggregates still cover every accepted page; only the row-level detail
is bounded.

## Word segmentation moved to v2 because the v1 gap threshold was too low

Splitting the stored `horizontal_ink_profile` into word runs needs a gap threshold: a zero-ink
column run at least that wide counts as a word gap rather than a letter gap. The first version,
`max(2, round(0.25 * x_height_px))`, sat below the true optimum in all five books, in one book by
as much as 6 px. That meant it split inside words at ordinary letter gaps, not just at word
boundaries. See the word-gap threshold finding (`pdomain-ocr-synth` research: the word-gap threshold is too low)
for the sweep that found it.

`ink-profile-word-runs/v2` fixes this per book instead of by one fixed coefficient. A pre-pass
builds each book's own gap-length histogram from the `horizontal_ink_profile` values the
alignment report already carries, so it opens no image. Otsu then splits that histogram into a
letter-gap population and a word-gap population. The measured thresholds across the five books
are 10, 6, 8, 15, and 9 px.

Otsu assumes two populations and will split a single hump down the middle if handed one, so the
split is checked before it is trusted. The count at the chosen threshold must fall to at most half
the smaller class's peak count. When it does not, the book falls back to the v1 x-height ratio
rule applied per line, and the report records which rule ran. Across the corpus this guard fired
on no book and on 14 of 665 pages, and every one of those 14 pages was evidence-only, so no pooled
book estimate was affected.

## Word reconciliation clears its floor in every book under v2

Gate 6 measures the share of measured lines where the ink-run count matches the transcription's
word count. Under v2 it reads 0.805, 0.742, 0.593, 0.800, and 0.782 across the five books, all
above the 0.50 floor. Under v1 the same gate failed in two books, at 0.476 and 0.208.

## A line fails on one bad gap even when most of its gaps are read correctly

Gate 6 counts whole lines, and a line typically carries six to twelve word gaps. One bad gap
fails the entire line even if every other gap on it is correct. The report also carries per-gap
counts: `word_gap_count` and `word_run_error_count` per page and per book, with
`word_gap_error_rate` computed from their ratio. Read this way, the detector gets individual word
gaps right 0.923 to 0.972 of the time, well above Gate 6's 0.593 to 0.805 line-level rate. A
consumer that wants word boxes is buying gaps, not lines, and the per-gap number is the one that
describes what it is buying.

## The optional OCR witness flags a line whose leading fragment F2 does not carry

`typography --geometry <book>.jsonl` adds an optional input: a JSONL file of OCR geometry
records, one per page, each carrying recognized words with pixel boxes and confidences. Without
`--geometry` the command behaves exactly as before, byte for byte, on all five books. With it,
each matched line records what the recognizer read at its first ink run, and whether that run is
a continuation fragment.

The mechanism is a known gap in the PGDP transcription. F2 rejoins a word broken across two
printed lines onto the line where it started, so the next line's leading ink run has no F2 word to
bind to and the run count comes out one over. F2 has erased that evidence, so only the ink can say
whether a leading run is a fragment. See
F2 silently joins line-break hyphens (`pdomain-ocr-synth` research: F2 line-break hyphens) for
the mechanism, and
what word reconciliation still misses (`pdomain-ocr-synth` research: what word reconciliation still misses)
for the measurement this witness acts on.

Nothing in this repo runs OCR or opens a model. The geometry records are produced elsewhere, by
`pdomain-source-data`, using the project's own fine-tuned DocTR checkpoints. This repo only reads
the JSONL file that process writes, so its base install stays torch-free.
`src/pdomain_pgdp_measure/ocr_witness.py` verifies each accepted page's `image_sha256` against
the alignment report's `scan_sha256` before using a page's witness, on the same reasoning as
`mask_mismatch`: a witness read from a different scan would be wrong in a way no gate could see.

The witness flags 893 lines across the corpus, 3.7 to 7.1 percent of matched lines. Discounting
a flagged line's leading run lifts agreement with the transcription by 3.0 to 7.0 points, to
0.860, 0.772, 0.632, 0.870, and 0.847. That number, `reconciled_after_witness`, is reported beside
Gate 6 and does not replace it: Gate 6 stays a tripwire for a broken detector, and folding a known
cause into its definition would make it blind to that cause coming back.

## Pooled geometry settles in a handful of pages, but word gap needs more

The report's Task 8 calibration answers how many accepted body pages a book-level estimate needs
before it stops moving. Pooled x-height, baseline pitch, and stroke width all settle within 0.5 px
by 5 body pages in four of the five books, and by 20 pages in the fifth. Word gap is less stable.
Under v2 it settles at 5, 50, 5, 30, and 5 body pages across the five books, an improvement over
v1, where one book did not converge within the 100-page sweep. `MINIMUM_ACCEPTED_PAGES_PER_BOOK`
stays at 30, since that ceiling is comfortably above what the three steadier observables need and
still covers four of the five books' word-gap answer.

## What this slice does not build

Font candidates, inverse rendering, typeface ranking, rectification, rectified coordinate frames,
and change-point detection over page index remain unimplemented. So does any point-size or
leading claim. The design's latent-discipline rule holds across both plans this document
promotes: no report key may claim font identity, nominal point size, font advance, side bearing,
or tracking. `baseline_pitch_px` is a measured pixel distance between two matched baselines, and
`word_gap_px` is an observed ink gap; neither claims the leading or word-space value the
compositor set.

## Evidence

- Code: `src/pdomain_pgdp_measure/typography.py`,
  `src/pdomain_pgdp_measure/typography_measure.py`,
  `src/pdomain_pgdp_measure/typography_models.py`,
  `src/pdomain_pgdp_measure/typography_pooling.py`,
  `src/pdomain_pgdp_measure/ocr_witness.py`,
  `src/pdomain_pgdp_measure/alignment_image.py` (shared mask builder), and
  `src/pdomain_pgdp_measure/cli.py`.
- Schema: `schemas/pgdp-typography-v1.schema.json`.
- Tests: `tests/test_pgdp_typography.py`, `tests/test_pgdp_typography_measure.py`,
  `tests/test_pgdp_typography_models.py`, `tests/test_pgdp_typography_pooling.py`,
  `tests/test_pgdp_typography_fixtures.py`, `tests/test_pgdp_typography_witness.py`,
  `tests/test_cli_typography_pgdp.py`, and `tests/test_pgdp_alignment_image.py`.
- Corpus verification: two runs per book were byte-identical. Across 665 accepted pages in five
  books, all 17,205 matched candidates reproduced their recorded mask values and zero pages were
  excluded as `mask_mismatch`. Word reconciliation under v2 read 0.805, 0.742, 0.593, 0.800, and
  0.782, all above the 0.50 floor. Per-gap accuracy read 0.923 to 0.972. The OCR witness flagged
  893 lines, 3.7 to 7.1 percent of matched lines, and reconciliation after discounting them read
  0.860, 0.772, 0.632, 0.870, and 0.847.
- Verified: corpus results recorded in `docs/context/current-state.md`, then promoted on
  2026-09-06.

## Related

- The two plans this document promotes, `2026-09-02-pgdp-font-free-typographic-observables`
  and `2026-09-04-pgdp-ocr-witness-for-continuation-fragments`, were retired on 2026-09-06.
  Their tombstones are in decisions (the decision log in `pdomain-ocr-synth`).
- The word-gap threshold finding (`pdomain-ocr-synth` research: the word-gap threshold is too low).
- What word reconciliation still misses (`pdomain-ocr-synth` research: what word reconciliation still misses).
- F2 silently joins line-break hyphens (`pdomain-ocr-synth` research: F2 line-break hyphens).
- [PGDP observed geometry profiling](pgdp-observed-geometry-profiling.md), the profile this
  command consumes.
