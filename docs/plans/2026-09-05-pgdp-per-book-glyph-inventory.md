---
Status: shipped-gate-open
Owner: CT
Created: 2026-09-05
Last verified: 2026-09-05
Kind: plan
---

# PGDP Per-Book Glyph Inventory

## Agent Index

- **Kind:** plan
- **Status:** shipped-gate-open
- **Owner:** CT
- **Created:** 2026-09-05
- **Last verified:** 2026-09-05
- **Provenance:** scoped on 2026-09-05, with coverage measured by harvesting all 665 accepted
  pages of the five aligned books, then shipped and re-measured the same day
- **Disposition:** Shipped, with Gate 3 open. All eight tasks shipped and nine of ten gates pass;
  Gate 3 fails on one book under a random sample. See "Gate 3 fails on one book". All three open
  decisions were taken by CT on 2026-09-05, and one of them forced a two-tier label design; see
  "Decisions taken". Review of the first corpus run overturned the scoping rule for what a word's
  ink runs are counted against; see "What the corpus run changed".
- **Read when:** building or consuming the glyph inventory, or asking what synthesis can render
  in a book's own type.
- **Search terms:** glyph inventory, per-book, harvest, atlas, coverage, synthesis, M15f.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
  (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
  checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut a labelled glyph inventory from each aligned book's own scans, so synthesis can
render arbitrary text in that book's actual type with per-character ground truth.

## What this is for

**Nothing else in the pipeline produces glyph-level ground truth.** PGDP gives line text.
Alignment gives line boxes. M15d gives word boxes on reconciled lines. A recognition model wants
per-character boxes, and only this makes them.

Four things follow. Training pairs stop being bounded by what PGDP has proofed, because any text
can be set in the book's own glyphs at its measured x-height, word gap and baseline pitch.
Degradation gets a known original, which a real scan never has. The observed in-situ glyph
positions give an empirical spacing distribution for the book, which is a measurement of ink
rather than a claim about a typeface. And the ranking slice gets real shapes to score revival
candidates against.

**Body-text labels come from F2, which is human-proofed, not from OCR.** A glyph is cut only from
a word on an exactly reconciled line, so its character is the transcription's. That is what keeps
this from being a model trained on its own output.

Furniture is different and is kept in a separate tier, because proofers strip running heads and
page numbers from F2 entirely, so nothing there has a human label. See "Decisions taken".

## What the corpus run changed

**Punctuation prints, so it is counted.** The scoping pass counted a word's ink runs against its
alphanumeric characters alone. A comma or an apostrophe makes its own run as readily as a letter,
so that let a word match by coincidence: one mark standing alone adds a run while one touching
letter pair removes one, and the two cancel. Every label in such a word then bound to the wrong
ink. Measured over four books on 40 pages each, the coincidence covered 1.4 to 15.4 percent of the
words the letters-only rule called separable, and in not one of those 927 cases did the ink
actually carry one run per printed character. Visual review of 210 glyphs in
`projectID609bfa0449bdf` found seven wrong labels, among them a `t` and an apostrophe both
labelled `n`.

Runs are now counted against every printing character and only the alphanumerics are labelled, so
a mark is cut and discarded rather than shifting its neighbours. In three of the four measured
books this also raised the yield, because a word whose punctuation stands cleanly apart now cuts
instead of being refused. The shipped rule is `blank-column-runs/v2`.

The scoping numbers below were reproduced exactly before the rule changed, which is how the
defect was isolated rather than confused with a regression. The shipped numbers are in
"Measured on 2026-09-05: what shipped".

## Measured at scoping: what the five books yield under the letters-only rule

Every accepted page of all five books was harvested. A glyph is cut when, inside a reconciled
word's box, the count of ink runs separated by a blank column equals the word's letter count, so
every glyph in that word is separable with a known label.

| book | pages | glyphs | per page | separable words | distinct chars | lowercase | uppercase | digits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| projectID657550412c8dc | 155 | 70,916 | 458 | 0.799 | 67 | 69,343 | 1,511 | 62 |
| projectID609bfa0449bdf | 226 | 61,701 | 273 | 0.534 | 51 | 59,958 | 1,743 | 0 |
| projectID64a479f51ce5b | 75 | 30,905 | 412 | 0.345 | 63 | 29,498 | 1,032 | 375 |
| projectID603d7d5e04ca0 | 177 | 13,663 | 77 | 0.256 | 61 | 12,351 | 1,285 | 27 |
| projectID67a80fde44d34 | 32 | 10,654 | 333 | 0.601 | 49 | 10,455 | 197 | 2 |

187,839 glyphs from 665 pages, at roughly 1.6 seconds a page. The shipped command reproduced
every one of these counts exactly before the counting rule changed.

## Measured on 2026-09-05: what shipped

The same 665 accepted pages, harvested by `glyphs` under `blank-column-runs/v2` with the
`recognized` tier on.

| book | pages | glyphs | transcribed | recognized | per page | separable | digits | reviewed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| projectID657550412c8dc | 155 | 84,733 | 82,321 | 2,412 | 547 | 0.886 | 564 | 1.000 |
| projectID609bfa0449bdf | 226 | 64,417 | 61,692 | 2,725 | 285 | 0.543 | 563 | 1.000 |
| projectID64a479f51ce5b | 75 | 31,578 | 30,669 | 909 | 421 | 0.355 | 1,879 | 0.990 |
| projectID603d7d5e04ca0 | 177 | 15,367 | 12,748 | 2,619 | 87 | 0.248 | 306 | 0.981 |
| projectID67a80fde44d34 | 32 | 11,233 | 11,159 | 74 | 351 | 0.624 | 83 | 1.000 |

207,328 glyphs from 665 pages in 8 minutes 10 seconds, 198,589 on the `transcribed` tier and
8,739 on `recognized`. Every page of every book harvested; none was excluded.

**Furniture closed the digit gap, as measured.** `projectID609bfa0449bdf` yields 8 digits from
body prose across 226 pages and 555 from its running heads and folios. Recognition confidence on
the `recognized` tier runs 0.77 to 0.85 mean.

Four things follow.

**Every book carries all 26 lowercase, with none missing.** The floors are `q` at 41, `z` at 10,
`z` at 1, and accented rarities at 1. Lowercase is what body prose gives you and it gives it
deeply.

**Uppercase is thin, and digits are scarce in narrative prose rather than absent.** 197 to 1,743
uppercase samples per book. `projectID609bfa0449bdf` yields **zero digits across 226 pages**, but
its transcription contains only 22 digit characters in 5,053 matched lines, so that is scarcity,
not a defect. A reference work fares far better: `projectID64a479f51ce5b` has 3,111 digit
characters in F2 and yields 375. Furniture closes the gap; see "Decisions taken".

**Yield varies nearly sixfold per page, and separability is why.** `projectID603d7d5e04ca0`
separates only 0.256 of its words and yields 77 glyphs a page against
`projectID657550412c8dc`'s 0.799 and 458. Column-run splitting fails where letters touch.

**Per book is forced, not tidy.** X-heights run 10 to 18 px across these five and glyph pixel
sizes scale with them, so a pooled inventory would be a chimera of sizes.

## The shape

One inventory per book, from the same pinned inputs `typography` takes, so provenance is the
existing discipline rather than a new one.

```text
pgdp-measure glyphs <corpus_root> --alignment a.json --profile p.json --out <dir>/
  manifest.json      pgdp-glyphs/v1: input hashes, method version, per-character
                     coverage per tier, quality tallies
  glyphs.jsonl       one row per glyph: character, label tier, page, line ordinal,
                     word ordinal, glyph ordinal, source-frame box, quality flags
  atlas/U+0065.png   derived: every 'e' in the book on one grid
```

**Boxes are the contract; the atlas is a derived render that also ships.** The JSONL stays the
authority, re-cuttable from the corpus and small. The atlas is regenerated from it and committed,
per decision 1, which makes the inventory usable without the corpus mounted.

**One atlas per character, not per glyph.** A file per glyph is 10,000 files a book and three
million across 286 projects. Per character is 50 to 90 files a book, and it is self-reviewing:
open `U+0065.png` and every `e` in the book is on one grid, where a bad cut is obvious. That is
the Gate 4 trick, built in rather than bolted on.

## Tasks

- [x] **0. Harvest furniture through the OCR witness.** Words outside every matched line box get
  their label and box from the `geometry-v1` record, tagged `recognized`. The page's
  `image_sha256` must match the alignment's `scan_sha256` first, the same check the witness makes.
- [x] **1. Cut glyphs from a reconciled word.** Given a word box and its transcription word, split
  by blank columns and emit one record per glyph when the counts agree. Emit nothing and record
  the reason when they do not.
- [x] **2. Quality flags, recorded not enforced.** Whether the glyph's row extent agrees with the
  line's measured x-height, ascender and descender; whether it touches the **line** box edge;
  its ink density. Not the word box: a word box is tight to its own ink, so its tallest and
  lowest glyphs always touch it, which makes a word-box test meaningless.
- [x] **3. The `pgdp-glyphs/v1` contract and its schema.** Input hashes, method version, per-
  character counts, quality tallies, and the per-glyph JSONL.
- [x] **4. The `glyphs` command.** Refuses a profile that does not hash to the alignment's
  recorded value, the same as `typography`.
- [x] **5. The per-character atlas renderer,** deterministic and regenerable from the JSONL alone.
- [x] **6. Reviewed fixtures** covering a separable word, a touching word, and a word whose
  letters split into the wrong count.
- [x] **7. Harvest all five books, both tiers, and measure every gate below.**

## Acceptance gates

All nine pass, measured on 2026-09-05 over the whole five-book corpus. Evidence is in
`/workspaces/pdomain/.m15f-evidence/`: `gate-summary.json`, one `gates-<book>.json` per book, and
the review sheets the label gates were read from.

| Gate | Floor | Measured | Verdict |
| --- | --- | --- | --- |
| 1. Determinism | byte-identical | five of five inventory directories identical across two runs, atlas PNGs included | pass |
| 2. Provenance | exact | every row unique on page, tier, line, word and glyph ordinal; every row's page in the manifest table; a profile that does not hash to the alignment's value is refused | pass |
| 3. Label correctness, `transcribed` | 0.98 over 200+ across 3+ books | 1,050 glyphs drawn at random across all five books, 23 wrong, **0.978 pooled**; per book 1.000, 0.995, 0.990, **0.962**, **0.943**. Filtering the three quality flags gives 0.992 pooled and every book passing | **fail** |
| 3b. Label correctness, `recognized` | 0.90 over 100+ | 340 furniture glyphs reviewed across two books, 0 wrong, 1.000 | pass |
| 4. Lowercase coverage | all 26 per book | no book misses a letter | pass |
| 4b. Digit coverage | 20 per book | 83 to 1,879 per book | pass |
| 5. Yield floor | 50 per harvested page | 68 to 308 | pass |
| 3c. Label correctness, recognizer-admitted | 0.98 | 420 glyphs reviewed across two books on non-accepted pages, 5 wrong, 0.988 | pass |
| 6. Atlas reproduction | exact | re-rendering from the JSONL reproduced all 900 sheets byte for byte | pass |
| 7. Latent discipline | none | no manifest or row key claims font identity, point size, advance, side bearing or tracking | pass |

## Gate 3 fails

Reviewed on 2026-09-06 at a legible scale, 30 glyphs each for `a e h n o s t` per book drawn at
random under seed 20260905:

| book | wrong | rate | filtered | verdict |
| --- | ---: | ---: | ---: | --- |
| projectID657550412c8dc | 0 | 1.000 | 1.000 | pass |
| projectID609bfa0449bdf | 1 | 0.995 | 1.000 | pass |
| projectID64a479f51ce5b | 2 | 0.990 | 1.000 | pass |
| projectID603d7d5e04ca0 | 8 | **0.962** | 0.981 | **fail** |
| projectID67a80fde44d34 | 12 | **0.943** | 0.990 | **fail** |
| pooled | 23 | **0.978** | 0.994 | **fail** |

Filtering costs 23 of 1,050 sampled glyphs and carries six false positives.

Two books fail and the pooled figure fails, at 0.978 against a floor of 0.98.

**An earlier pass of the same sample scored this too high, and the cause was the review, not the
inventory.** Rendered thirty cells to a row, a bare stem cut from an `h` is indistinguishable from
a whole letter; at fifteen to a row it is obvious. `projectID67a80fde44d34` scored 4 wrong at the
coarse scale and 12 at the legible one. Any later review should render at most fifteen cells a row.

**The review is still scoring three of the 23 wrong, for the same reason.** The sheet shows a
reviewer the character without `label_style`, so a small-capital N whose record reads
`character: n, label_style: small_caps` is marked wrong when the record is right. Show the style
and rescore before deciding this gate. Re-running the measurement chain does not help: all 23
marked glyphs survive the band-identification fixes unchanged. See
the Gate 3 recheck in `pdomain-ocr-synth`.

**The five quality flags catch 17 of the 23, with six false positives across 1,050 cells.**
Dropping flagged glyphs takes the pooled figure to 0.994 and every book over the floor, at a cost
of 23 of 1,050 sampled glyphs. That is not the gate passing: the gate measures the inventory as
shipped, and moving it to measure a filtered subset would be weakening it. It is what a consumer
gets for one line of filtering.

**A fifth flag, `unlike_character`, compares ink rather than boxes**, resampling each glyph onto a
16 by 12 grid and scoring it against the median shape its own character takes in that book and
style. Glyphs of one character sit 2.5 to 3.3 from their reference while two different characters'
references sit 4.3 to 7.7 apart, so the distance carries signal, and the flag costs 0.5 to 5.1
percent of glyphs.

**It has a measured blind list, and a finer grid does not fix it.** Fourteen character pairs in
`projectID603d7d5e04ca0` have references closer together than the characters' own scatter, so no
threshold separates them. The count holds at 14, 15 and 14 across grids of 16 by 12, 24 by 18 and
32 by 24: the distances scale with the grid and the separation does not. The pairs say why. Four
are a capital and its lowercase of the same outline, `I`/`l`, `O`/`o`, `W`/`w`, `S`/`s`, which
normalising size away deliberately erases, and that is the small-capital problem restated in
another form. The rest differ by a few pixels of outline: `b`/`h` by whether the bowl closes,
`c`/`e` by the crossbar. Measured, the `b` reference sits 3.34 from the `h` reference while `h`
scatters 3.46 from its own, so a `b` labelled `h` is inside ordinary `h` variation.

Scaling each glyph against its line's x-height instead of onto a fixed grid would restore the
capital-versus-lowercase distinction, at the cost of depending on a per-line x-height that this
milestone has already found unreliable on badly printed pages. That is the next thing to try.

The six that survive filtering split two ways. Three are a box holding a letter and just enough of
its neighbour to stay inside the ordinary width spread, at 1.00 to 1.15 times the character's
median: `s12` cut from an `s` and its comma, `a10` from an `a` and one stem, `e5` from `ler`.
Three are the right width but the wrong letterform: `h21` reading `b`, `n16` reading `N`, `e13`
reading `E`. No size test reaches either group, and that is the floor of what geometry can do.

Gate 3 is model-reviewed, not human-reviewed.

Gate 3 is model-reviewed, not human-reviewed. Sheets of 30 samples each for `a e h n o s t` were
rendered from the scans at 7x and read by eye; the six errors are a `g` labelled `a`, a `Th`
labelled `h`, an `re` labelled `e`, a `th` labelled `h`, a `ho` labelled `o`, and a near-empty
cell labelled `t`. All six are residual coincidences inside pure-alphanumeric words, where one
letter broke into two runs while another pair touched. The sheets are kept so CT can confirm.

## What review found after the gates passed, and what it cost

**Faces were pooled under one character, and now they are not.** PGDP transcribes small capitals
in mixed case, so `<sc>Lowther Street</sc>` yields the letters `Lowther Street` and a capital `O`
landed in the lowercase `o` bucket carrying a lowercase label. Nothing comparing a label to a
character could see it: the character was right and only the letterform was wrong, which is why
every gate passed over it. Rows now carry `label_style`, and coverage and the atlas are keyed by
it, so `atlas/transcribed/small_caps/U+0065-000.png` holds capital `E` forms and the roman grid
holds lowercase `e`.

**The evidence was already in the alignment report.** Every page carries a style run per `<i>`,
`<b>` and `<sc>` span, matching F2 exactly. The command reads no extra input.

**Measured, the contamination was far smaller than a first estimate suggested.** Dividing each
book's F2-wide styled characters by its F2-wide total put `projectID657550412c8dc` at 15.8 percent
styled. The inventory does not draw from F2 at large: it draws from accepted pages' matched,
reconciled, separable words, and that book's styled text is almost all front matter the alignment
never accepted. Only 52 of its 3,232 style runs fall on an accepted page. The real figures:

| book | italic glyphs | small-cap glyphs | styled share of transcribed |
| --- | ---: | ---: | ---: |
| projectID603d7d5e04ca0 | 118 | 607 | 0.0569 |
| projectID64a479f51ce5b | 192 | 83 | 0.0090 |
| projectID609bfa0449bdf | 30 | 22 | 0.0008 |
| projectID657550412c8dc | 49 | 0 | 0.0006 |
| projectID67a80fde44d34 | 0 | 0 | 0.0000 |

The separation is still right, and it is now measured rather than assumed. It does not catch
unmarked styling: `projectID609bfa0449bdf` p060 prints `LOWTHER STREET` in small capitals with a
proofer's unresolved query `[**mark place in <sc>?]` and no markup, so those 11 capital
letterforms still sit in lowercase buckets labelled roman. Only review finds that class.

## Every page is harvested, not only the accepted ones

**Page acceptance turned out to be a poor proxy for whether one line bound to the right text.**
Measured against the recognizer over three books, lines excluded only for alignment score agree
about as well as accepted ones: `normalized_cost_exceeds_maximum` scores 0.937 to 0.962 and
`uniqueness_margin_below_minimum` 0.887 to 0.934, against 0.926 to 0.976 for accepted. Lines on
illustration pages score 0.560 to 0.683, which is what the page filter was really earning.

So a line on a non-accepted page is admitted when the recognizer reads at least four of every
five of its transcription words inside its own box. That needs `--geometry`; without it those
pages are skipped as before. The recognizer only ever rejects, and the label stays the
transcription's.

The result is 1,367 of 1,385 pages harvested against 665, and 258,790 `transcribed` glyphs
against 198,589, a 30 percent gain. Reviewed on the newly admitted population alone, 420 glyphs
across two books, 5 were wrong, 0.988 against the same 0.98 floor.

**Furniture stays on accepted pages.** Harvesting every page first pushed the `recognized` tier
from 8,739 glyphs to 165,670, because on a page alignment did not accept most words fall outside
every matched line simply because few lines matched. That stops meaning the running head and the
folio and starts meaning ordinary body text, which is not what the tier documents and not what its
review covers. Extending it is a separate decision with its own measurement to do first.

## Decisions taken

All three were settled by CT on 2026-09-05. Two went against the recommendation, and one of those
forced a design change.

**1. Atlases ship.** They are committed rather than kept as local build artifacts. Two
consequences to size for: `check-added-large-files` is set to 1000 kB, so an atlas must be split
per character and probably per size band to stay under it. And at the measured rate the full
286-project corpus would reach roughly 16 million glyphs, so shipping atlases needs a stated
policy well before then. Shipping the five aligned books is comfortable; shipping 286 is not, and
that limit should be written down when the second book lands rather than discovered at the
fiftieth.

**2. Headers and page numbers are harvested, in a separate tier.** This was measured before it was
written in, and the measurement changed the design. Proofers strip running heads from F2: across
all five books there is **not one bare page-number line** in the transcription, while furniture
bands run almost exactly one per accepted page. So furniture ink has no human label and can only
be labelled by OCR.

The `geometry-v1` records already cover it. Words falling outside every matched line box number
117 to 1,082 per book, carrying **65 to 560 digit characters** each at 0.77 to 0.82 mean
recognition confidence. `projectID609bfa0449bdf`, which yields no digits at all from body text,
has 560 in its furniture. That is the digit gap closed.

So the inventory carries two label tiers and never mixes them:

| tier | source | label from | corpus scale |
| --- | --- | --- | ---: |
| `transcribed` | reconciled words on matched lines | F2, human-proofed | 187,839 glyphs |
| `recognized` | words outside every matched line | DocTR `geometry-v1` | about 2,905 words |

The manifest counts them separately and every glyph row names its tier, so a consumer can take
digits from `recognized` while training characters only on `transcribed`. The safety property was
never "no OCR labels"; it is that you always know which is which.

**3. A thin book emits and declares.** A floor that silently drops books hides what it dropped,
which is the reasoning Gate 6 was written under. No book came close to the floor: the thinnest,
`projectID603d7d5e04ca0`, yields 87 glyphs an accepted page against a floor of 50.

## What this does not do

It does not build a font. No outlines, no autotracing, no shaping tables. For synthesis you
compose at the measured size from samples taken at that size, and a traced outline would be
noisier than the bitmap it came from.

It does not label furniture from anything but OCR. There is no human transcription of a running
head in this corpus, so the `recognized` tier cannot be promoted by better parsing, only by
review.

It does not render anything. Composing pages from an inventory is a later slice and needs its own
sampling rule, because reusing one exemplar per character would look mechanical in a way real
setting never does.

It does not fix separability. Connected-component labelling would rescue touching letters and
lift `projectID603d7d5e04ca0` most, but the column-run baseline is measured first so the
improvement can be scored against it. That book separates 0.248 of its reconciled words and is
where the headroom is.

It does not remove the last coincidence. A pure-alphanumeric word can still match by luck when one
letter breaks into two runs while another pair touches, which is where all six reviewed label
errors came from, at roughly 0.006 of reviewed glyphs. Refusing a word whose glyph widths vary
implausibly against the book's own distribution would catch most of them, and is worth its own
slice once there is a reason to want a tighter number than 0.994.

It reaches only books with alignment reports: five of 286. Alignment is the bottleneck this sits
behind.

## Related

- PGDP font-free typography in `pdomain-pgdp-measure`
  — the word boxes this cuts from, and the latent discipline it inherits.
- PGDP font-free typography, OCR witness in `pdomain-pgdp-measure`
  — the same provenance pattern, and the reason reconciliation is trustworthy.
- `/workspaces/pdomain/.m15f-evidence/` — the measured per-book coverage this plan is written
  against, and `next-loop-prompt.md`, the execution brief for an unattended run.
- What word reconciliation still misses in `pdomain-ocr-synth`
  — why a reconciled word is a safe place to cut from.
