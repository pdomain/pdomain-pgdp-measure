# PGDP per-book glyph inventory

## Agent Index

- **Kind:** architecture
- **Status:** active
- **Owner:** CT
- **Created:** 2026-09-06
- **Last verified:** 2026-09-06
- **Provenance:** verified from shipped code, schemas, tests, and the measured five-book corpus
  runs, with Gate 3 recorded as failing
- **Disposition:** Current truth promoted from the M15f per-book glyph inventory plan, which
  shipped with Gate 3 open.
- **Promotes:** `2026-09-05-pgdp-per-book-glyph-inventory` plan, status shipped-gate-open.
- **Read when:** building or consuming the glyph inventory, deciding whether its labels are
  trustworthy enough for a task, or asking what synthesis can render in a book's own type.
- **Search terms:** glyph inventory, per-book, harvest, atlas, transcribed, recognized, quality
  flags, Gate 3, M15f.

`glyphs` cuts a labelled, per-character glyph inventory from one aligned book's own scans.
It is the first thing in this pipeline that produces glyph-level ground truth: PGDP gives line
text, alignment gives line boxes, and word-level typography gives word boxes, but nothing before
this cuts a box around one letter and names it.

## Gate 3 is open, and label correctness fails on two of five books

**The milestone shipped with an acceptance gate failing.** Gate 3 measures how often a glyph on
the `transcribed` label tier carries the right character. The floor is 0.98. The measured pooled
result is **0.978**, drawn from 1,050 glyphs sampled at random across all five books and read by
eye at a legible scale.

| book | wrong | rate | verdict |
| --- | ---: | ---: | --- |
| projectID657550412c8dc | 0 | 1.000 | pass |
| projectID609bfa0449bdf | 1 | 0.995 | pass |
| projectID64a479f51ce5b | 2 | 0.990 | pass |
| projectID603d7d5e04ca0 | 8 | **0.962** | **fail** |
| projectID67a80fde44d34 | 12 | **0.943** | **fail** |
| pooled | 23 | **0.978** | **fail** |

Three books pass comfortably. Two do not, and the pooled figure misses the floor by two tenths of
a point.

**Filtering is not the gate passing.** Dropping every glyph the five quality flags below mark
takes the pooled figure to 0.994, with every book over the floor, for a cost of 23 of the 1,050
sampled glyphs and six false positives. That is what one line of filtering buys a consumer. It is
not a measurement of the inventory as shipped, and treating it as the gate result would weaken the
gate. Both numbers are real. Keep them apart: **0.978 pooled is the gate result, and it fails.
0.994 pooled is what filtering buys, and it is a separate, optional step.**

See "Two measured ceilings mark where geometry stops" for why the two failing books resist a fix,
and "Six reviewed errors survive filtering" for what even the filtered inventory still gets wrong.

## What the command writes

```text
pgdp-measure glyphs <corpus_root> --alignment a.json --profile p.json --out <dir>/
  manifest.json           pgdp-glyphs/v1: input hashes, method versions, per-character
                          coverage per tier, page table, quality and reject tallies
  glyphs.jsonl            one row per glyph: character, label tier, page, line ordinal,
                          word ordinal, glyph ordinal, source-frame box, quality flags
  atlas/<tier>/<style>/U+0065-000.png
                          derived: every roman lowercase 'e' in the book on one grid
```

The command takes the same pinned inputs `typography` does: a `pgdp-alignment/v3` report and
the `pgdp-profile/v2` profile it recorded. It refuses to start unless the profile hashes to the
value the alignment report names, and it refuses an alignment covering more than one book, because
x-heights run 10 to 18 px across the five corpus books and a pooled inventory would mix sizes.
`--geometry` is optional and unlocks the `recognized` label tier and the recognizer's admission
check on non-accepted pages; without it the command still runs, on the `transcribed` tier and
accepted pages only.

**The JSONL is the authority, and the atlas is a derived render that also ships.** `glyphs.jsonl`
is small and re-cuttable from the corpus. The atlas is regenerated from the JSONL rows and
committed alongside it, so the inventory is reviewable without the corpus mounted: open one
character's grid and every glyph the book yields for it sits on one sheet, where a bad cut stands
out from its neighbours. Gate 6 confirmed the atlas re-renders byte for byte from the rows alone.

The atlas path names its tier, style, and character: `atlas/transcribed/small_caps/U+0065-000.png`
holds capital `E` forms transcribed as small capitals, and `atlas/transcribed/roman/U+0065-000.png`
holds ordinary lowercase `e`. A character too large for one sheet gets more than one, hence the
`-000` ordinal.

## Two label tiers never mix, and each names its own source

A glyph's `label_tier` is either `transcribed` or `recognized`, and a consumer must never treat
them the same way.

**`transcribed` comes from a word on a line reconciled against PGDP's F2 transcription.** The
character is a human proofer's. This is the tier a model should train characters on.

**`recognized` comes from a word a DocTR read finds outside every matched line box** — the
running heads and folios that PGDP does not transcribe at all. Proofers strip that furniture from
F2 entirely: across all five aligned books, **not one bare page-number line appears in the
transcription**. Furniture ink therefore has no human label anywhere in this corpus, and the only
label it can carry is a machine read.

Both tiers exist because that gap is real and each book has it. The safety property this design
gives is not "no OCR labels appear in the inventory." It is that **every row names which kind of
label it carries**, so a consumer can take digit samples from `recognized` while training
characters only on `transcribed`, and never accidentally mix the two.

`glyph_furniture.py` implements the `recognized` tier as `words-outside-matched-lines/v1`: a
recognized word belongs to a matched line only when at least half its own area falls inside that
line's box, checked after confirming the geometry record's page hash matches the alignment's
scanned page. Gate 3b measured this tier at 340 reviewed furniture glyphs across two books, zero
wrong.

## The cutting rule counts every printed mark, and labels only the letters and digits

A word is cut by splitting its box at blank columns. Each resulting ink run is matched, in order,
against one position in the word. When the run count equals the position count, every run is a
known character and the word is separable; otherwise nothing is cut, and the word's rejection
reason is recorded instead. The shipped rule, `blank-column-runs/v2` in `glyph_cut.py`, counts
**every printing character** — letters, digits, and punctuation alike — against the runs, but
only labels the alphanumerics. A run that lands on a punctuation mark is cut and discarded rather
than shifting every later position onto the wrong letter.

**The first version of this rule counted only alphanumerics, and it was wrong.** A comma or an
apostrophe makes its own ink run as readily as a letter does. Counting letters alone let a word
match by coincidence: one mark standing alone on a line adds a run the letters-only count does not
expect, while one pair of touching letters removes a run the same count does expect, and the two
errors cancelled. Measured over four books on 40 pages each, that coincidence covered 1.4 to 15.4
percent of the words the letters-only rule called separable, and in none of those 927 cases did
the ink actually carry one run per printed character. Visual review of 210 glyphs in one book
found seven wrong labels from exactly this cause, among them a `t` and an apostrophe both labelled
`n`.

Counting every printing character removes that coincidence, and in three of the four measured
books it also raised the yield, because a word whose punctuation stands cleanly apart now cuts
instead of being refused for a false mismatch.

## Every page is harvested, not only the pages alignment accepted

Early in the milestone, only a book's alignment-accepted pages were harvested: 665 pages across
the five books. **Page acceptance turned out to be a poor proxy for whether one line bound to the
right text.** Measured against the recognizer over three books, lines excluded only for alignment
score agree with the recognizer about as well as accepted lines do — 0.937 to 0.962 for one
exclusion reason and 0.887 to 0.934 for another, against 0.926 to 0.976 for accepted lines. Lines
on illustration pages score far lower, 0.560 to 0.683, which is what page acceptance was really
screening for.

So a line on a non-accepted page is now admitted when the recognizer reads at least four of every
five of its transcription words inside its own box. That check needs `--geometry`; without it,
non-accepted pages are skipped as before. **The same check now runs on accepted pages too**, which
is what caught lines whose candidate box had matched the wrong source line: cases like `They` cut
from `WILL`, or `and` from `side,`. Those mis-bound lines score 0.00 to 0.17 on the recognizer
check while the lines around them score above 0.8, and the check costs 2.4 to 7.4 percent of
accepted lines. The recognizer only ever rejects here; a line it agrees with still takes its
label from the transcription, never from the recognizer's own read.

**Furniture stays on accepted pages only.** Extending the recognizer-admission check to furniture
was tried and reverted: on a page most lines do not match, most words fall outside every matched
line simply because so few lines matched at all, and the `recognized` tier would stop meaning the
running head and the folio and start meaning ordinary body text that never got matched. That is a
different claim from the one this tier is reviewed against, so it needs its own measurement before
it can be trusted.

## The corpus currently yields 266,549 glyphs across five books

Coverage was measured three times as the harvesting rule changed, and each figure belongs to a
different run:

- **Scoping, letters-only cutting rule, 665 accepted pages:** 187,839 glyphs. This measurement
  found the punctuation-coincidence defect and was superseded before shipping.
- **First shipped run, `blank-column-runs/v2`, 665 accepted pages, both tiers:** 207,328 glyphs —
  198,589 `transcribed` and 8,739 `recognized` — in 8 minutes 10 seconds.
- **Current corpus state, every page harvested, recognizer admission on both accepted and
  non-accepted pages:** **266,549 glyphs from 1,367 of 1,385 pages**, across the five aligned
  books. Harvesting every page raised the `transcribed` count by about 30 percent over the
  accepted-pages-only run.

**Every book carries all 26 lowercase letters, with none missing.** Rare letters such as `q` and
`z` sit at the low end and common letters run into the tens of thousands; lowercase is what body
prose gives, and it gives it deeply.

**Uppercase and digits are thin in narrative prose, not absent, and furniture closes the digit
gap.** One book yields zero digits from 226 pages of body text because its transcription itself
contains only 22 digit characters across more than 5,000 matched lines — that is scarcity in the
source, not a defect in the harvest. Its running heads and folios carry hundreds of digit
characters instead, at 0.77 to 0.85 mean recognition confidence, which is why the digit-coverage
gate is measured across both tiers together.

**Yield per page varies nearly sixfold, and separability is why.** The book that separates only a
quarter of its reconciled words yields under a hundred glyphs a page; the book that separates four
out of five words yields several hundred. Column-run splitting fails wherever letters touch, and
that is a property of the printing, not a defect in the cutting rule.

## Faces are tracked, because a per-book inventory that mixes them is not one inventory

PGDP transcribes small capitals in mixed case: `<sc>Lowther Street</sc>` yields the plain letters
`Lowther Street`, so a capital `O` set as a small capital lands in the ordinary lowercase `o`
bucket carrying a lowercase label. Nothing that compares a label to a character can see that
defect, because the character is right and only the letterform is wrong.

`glyph_style.py` resolves each glyph's face from the alignment report's own `style_run` records,
which already carry one span per `<i>`, `<b>`, and `<sc>` tag matched against F2. No corpus file is
re-read to get it. Every row carries a `label_style` — `roman`, `italic`, `small_caps`, and so
on — and both coverage counts and the atlas are keyed by it, so a small-capital `E` and a roman
`e` never share a grid. `label_style` is `None`, distinct from `roman`, for every glyph on the
`recognized` tier, where no style source can be trusted at all.

Measured against each book's F2-wide styled character share, styled text turned out to be a small
fraction of what actually gets cut, because the inventory draws only from accepted, reconciled,
separable words, and most styled text in these books sits in front matter the alignment never
accepted. Styled glyphs run from none up to about 5.7 percent of one book's `transcribed` glyphs.
This does not catch small-capital text PGDP leaves unmarked: one page in one book prints a phrase
in small capitals with an unresolved proofer's query and no markup, and those letterforms still
sit in the lowercase bucket labelled roman. Only manual review finds that class of error.

## Five quality flags record observations, never verdicts

`DEFECT_FLAGS` in `glyph_quality.py` names five flags that mark a glyph as possibly wrong. None of
them changes a label or drops a row; each is recorded so a consumer can filter on it.

| flag | says | catches | cost |
| --- | --- | --- | ---: |
| `flat_ascender` | the word's ascenders run no taller than its x-height letters | unmarked small capitals, a word bound to the wrong ink | 0.2 to 2.8% of words |
| `narrow` | the glyph runs under 0.6x its character's usual width in the book | a broken arch leaving half a letter | 0.17 to 1.31% |
| `wide` | the glyph runs at or over 1.5x that width | a missed blank column leaving several letters in one box | 0.06 to 0.91% |
| `overtall` | the glyph runs taller than its own line's x-height letters | a capital where the label says lowercase | 0.03 to 1.2% |
| `unlike_character` | the glyph's ink sits far from its character's median shape in this book | wrong letterforms and multi-letter boxes that size alone cannot see | 0.5 to 5.1% |

A row can also carry `ascends`, `descends`, `touches_line_top`, and `touches_line_bottom`. **These
four are ordinary facts about a letter, not defects.** Every `h` ascends and every `g` descends;
treating those as defects would exclude every ascender and descender from its own character's
shape reference, which is a defect this design deliberately avoids.

`unlike_character` is the newest flag and works differently from the other four: it resamples each
glyph onto a fixed 16-by-12 grid and scores its distance from the median shape its own character
takes in that book and style, rather than comparing box dimensions. Glyphs of one character sit 2.5
to 3.3 units from their own reference at the median; two different characters' references sit 4.3
to 7.7 units apart. That gap is what the flag reads as signal.

## Nine of ten gates pass

| Gate | Result | Verdict |
| --- | --- | --- |
| 1. Determinism | five of five inventory trees byte-identical across two runs, atlas included | pass |
| 2. Provenance | every row unique on page, tier, line, word, and glyph ordinal; a mismatched profile refused | pass |
| 3. Labels, `transcribed` | 23 wrong of 1,050 reviewed at random, 0.978 pooled; filtered 0.994 | **fail** |
| 3b. Labels, `recognized` | 340 furniture glyphs reviewed, 0 wrong | pass |
| 3c. Labels, recognizer-admitted | 420 glyphs reviewed on non-accepted pages, 5 wrong, 0.988 | pass |
| 4. Lowercase coverage | all 26 letters present in every book | pass |
| 4b. Digit coverage | 128 to 2,837 per book, against a floor of 20 | pass |
| 5. Yield floor | 68 to 308 glyphs per harvested page, against a floor of 50 | pass |
| 6. Atlas reproduction | every sheet re-rendered byte for byte from the JSONL alone | pass |
| 7. Latent discipline | zero violations — no manifest or row field claims font identity, point size, advance, side bearing, or tracking | pass |

Gate 3 is reviewed by eye at a legible render scale, not by an automated check: an earlier pass at
a coarser render scale scored one book's error count at 4 wrong where a legible pass on the same
sample found 12, because a bare stem cut from an `h` is indistinguishable from a whole letter when
thirty cells share one row. Any future review of this gate should render at most fifteen cells to a
row.

## Two measured ceilings mark where geometry stops

Two findings are worth keeping even though neither fixes anything, because both close off a
direction someone would otherwise try next.

**Small capitals cannot be found from geometry alone.** Checked against lines PGDP itself marks
`<sc>`, a line's x-height reads 1.00 where its roman lines also read 1.00 in one book, and 1.08
against 1.08 in another. Descender share separates no better, and it points the opposite way in
the two books. No geometric signal tried so far tells a small capital from an ordinary capital.

**The shape check has a blind list, and a finer comparison grid does not shrink it.** Fourteen
character pairs in the worst-affected book have shape references closer to each other than a
single character's own glyphs scatter from their own reference — so no distance threshold can
ever separate them. The count holds at 14, 15, and 14 pairs across grids of 16 by 12, 24 by 18,
and 32 by 24 cells: distances scale up with a finer grid, but the pairs' separation does not
scale with them.

Four of the fourteen pairs are a capital and its lowercase of the same outline — `I`/`l`,
`O`/`o`, `W`/`w`, `S`/`s` — which the shape check's own size normalization deliberately erases;
that is the small-capital ceiling restated in another form. The rest differ by only a few pixels:
`b` from `h` by whether the bowl closes, `c` from `e` by the crossbar. Measured, the `b` reference
sits 3.34 units from the `h` reference, while `h`'s own glyphs scatter 3.46 units from their own
reference — so a `b` mislabelled `h` sits well inside ordinary `h` variation, and no threshold on
this grid can catch it.

Scaling each glyph against its own line's x-height, instead of onto one fixed grid, would restore
the capital-versus-lowercase distinction. That is the named next experiment, at the cost of
depending on a per-line x-height estimate this milestone already found unreliable on badly printed
pages — the same pages where these errors concentrate.

## Six reviewed errors survive filtering

Of the 23 wrong labels the Gate 3 review found, the five quality flags catch 17 and cost six false
positives, at a rate of 23 of 1,050 sampled glyphs. Six wrong labels pass every flag unflagged, and
they split into two groups.

**Three are a box holding a letter and just enough of a neighbour to stay inside the ordinary
width spread**, at 1.00 to 1.15 times the character's median width: one glyph cut from an `s` and
its trailing comma, one from an `a` and one stray stem, and one that swallowed three letters into a
box labelled as one. No width test reaches this group, because the extra ink is small enough to
sit inside normal variation.

**Three are the right width but the wrong letterform**: a glyph shaped like a `b` labelled `h`, one
shaped like an `N` labelled `n`, and one shaped like an `E` labelled `e`. These are exactly the
capital-versus-lowercase and outline-confusion pairs the shape check's blind list names above, so
no size test and no current shape test reaches them either.

That is the floor of what geometry-only checking can do against this corpus.

## What this does not do

**It does not build a font.** No outlines, no autotracing, no shaping tables. Synthesis composes
at the size samples were measured at; a traced outline would be noisier than the bitmap it came
from.

**It does not render or compose pages.** Composing arbitrary text from an inventory is a separate,
later slice, and needs its own rule for how exemplars get reused so a rendered page does not look
mechanically repetitive.

**It does not fix separability.** Connected-component labelling would rescue touching letters and
would help the least-separable book the most, but the column-run baseline is measured first so any
future improvement has a number to beat.

**It does not remove the last labelling coincidence.** A pure-alphanumeric word can still match by
luck when one letter breaks into two ink runs while another pair touches, which is the source of
the six errors above, at roughly 0.006 of reviewed glyphs.

**It reaches only books with an alignment report — five of 286 total corpus projects.** Alignment
is the bottleneck this inventory sits behind; it cannot be run on a book that has not been aligned.

## Evidence

- Code: `src/pdomain_pgdp_measure/glyphs.py`, `glyph_cut.py`, `glyph_quality.py`,
  `glyph_shape.py`, `glyph_style.py`, `glyph_furniture.py`, `glyph_atlas.py`, `glyph_models.py`,
  and `src/pdomain_pgdp_measure/cli.py`.
- Schemas: `schemas/pgdp-glyphs-v1.schema.json` (manifest) and
  `schemas/pgdp-glyphs-row-v1.schema.json` (one glyph row).
- Tests: `tests/test_pgdp_glyphs.py`, `tests/test_pgdp_glyph_cut.py`,
  `tests/test_pgdp_glyph_quality.py`, `tests/test_pgdp_glyph_shape.py`,
  `tests/test_pgdp_glyph_style.py`, `tests/test_pgdp_glyph_furniture.py`,
  `tests/test_pgdp_glyph_atlas.py`, `tests/test_pgdp_glyph_models.py`,
  `tests/test_pgdp_glyph_fixtures.py`, and `tests/test_cli_glyphs_pgdp.py`.
- Reviewed cut fixtures: `tests/fixtures/pgdp_glyphs/` — a separable word, a punctuated word, a
  touching word, and a word whose run count disagrees with its letter count.
- Commits: `8ed91d9`, `19d5783`, `f89277a`, `6b6700d`, `1123467`, `995b602`, `1ed50b3`,
  `b72478d`, `84d9f7e`, `d68d807`, `912b316`, `4eba005`, `32785ce`, `262dff7`, `eaa0989`,
  `872ac6f`, `7ecb71e`, `66d00d2`, and `36b9a77`.
- Corpus verification: 266,549 glyphs harvested from 1,367 of 1,385 pages across the five aligned
  books, deterministic across two full runs with the atlas included, atlas re-rendered byte for
  byte from the JSONL on all sheets, and Gate 3 measured at 0.978 pooled label correctness on the
  `transcribed` tier against a floor of 0.98.
- Verified: state recorded on 2026-09-06 in
  `docs/handoff/2026-09-06-140411-m15f-gate3-open-and-five-quality-flags.md`, and promoted here the
  same day.
