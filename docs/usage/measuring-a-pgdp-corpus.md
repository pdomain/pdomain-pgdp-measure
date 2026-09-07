# Measuring a PGDP Corpus

Five subcommands take a local PGDP corpus to a labelled glyph inventory, each one reading the
report the previous one wrote.

```text
rank  ->  profile  ->  align  ->  typography
                          \-->  glyphs
```

Every stage writes a deterministic report that declares its own wire contract, and every stage is
gated on producing byte-identical output across two runs.

| Command | Purpose |
|---------|---------|
| `rank <corpus_root>` | Rank local PGDP projects and select bounded review pages |
| `profile <corpus_root>` | Measure selected local PGDP scan geometry |
| `align <corpus_root>` | Align PGDP F2 source lines with measured scan lines |
| `typography <corpus_root>` | Measure font-free typographic observables from an alignment |
| `glyphs <corpus_root>` | Cut a per-book labelled glyph inventory from an alignment |

Run them from the corpus root, and always write reports outside it. Each command refuses an output
path inside the corpus.

## Rank a local PGDP corpus

```bash
pgdp-measure rank /path/to/pgdp-corpus \
  --output ./pgdp-ranking.json \
  --project-limit 50 \
  --pages-per-project 12
```

`rank` reads projects from the local `corpus_root` positional argument. It
does not use the network, an LLM, or a recipe. The command does not modify the
corpus, and the report path must be outside the corpus root. `--project-limit`
and `--pages-per-project` must be positive integers.

The command writes deterministic JSON to `--output`, which defaults to
`./pgdp-ranking.json`. The report contains schema and algorithm versions, the
requested limits, corpus counts, stable diagnostics, ranked projects, selected
pages, feature evidence, and score components. It does not contain absolute
source paths. Re-running the command with the same corpus and limits produces
the same report bytes.

## Profile selected PGDP scans

```bash
pgdp-measure profile /path/to/pgdp-corpus \
  --ranking ./pgdp-ranking.json \
  --output ./pgdp-profile.json
```

`profile` reads the selected pages in an M14 `rank` report. Both
`--ranking` and `--output` are required. The profile output must be outside the
corpus root, differ from the ranking input, and name a file rather than a
directory.

The command measures scan geometry locally and writes deterministic JSON. It
stores corpus-relative source paths, source-byte hashes, observed ink geometry,
and non-fatal diagnostics. Missing or undecodable scans remain in the report as
excluded pages with diagnostics. A completed measurement returns exit code 0
even when diagnostics are present. Input errors return 2 and output errors
return 6.

## Align PGDP source lines with scan rows

```bash
pgdp-measure align /path/to/pgdp-corpus \
  --profile ./pgdp-profile.json \
  --output ./pgdp-alignment.json
```

`align` reads F2 source and the scans referenced by a
`pgdp-profile/v2` report. It writes a deterministic `pgdp-alignment/v3` report.
The output must be outside the corpus root, differ from the profile input, and
name a file rather than a directory. Source or scan changes remain explicit
exclusions instead of being silently realigned.

The report keeps UTF-8 source byte spans, normalized visible text, formatting
spans, source-frame candidate boxes, monotone alignment operations, costs,
residuals, and exclusion evidence. Accepted alignments require a single-column
page and four to 80 nonblank source lines eligible for matching. Matched
operations must cover at least 90 percent of those lines. Normalized cost must
be at most 0.22, and uniqueness margin at least 0.15. Confidence is always null
with `confidence_kind` set to `uncalibrated` in version 1.

## Measure typography from an alignment

```bash
pgdp-measure typography /path/to/pgdp-corpus \
  --alignment ./pgdp-alignment.json \
  --profile ./pgdp-profile.json \
  --output ./pgdp-typography.json
```

`typography` measures per-line baseline row, x-height, ascender and
descender extents, stroke width, skew slope, word runs, and per-word ink boxes
from the accepted pages of a `pgdp-alignment/v3` report, then pools them per
page, per book, and per page class into a deterministic `pgdp-typography/v1`
report.

The atlas is a render of the JSONL, not a second source. One grid per
character per tier holds every sample of that character in the book, sharded at
1,024 cells a sheet, so opening `atlas/transcribed/U+0065-000.png` shows every
`e` at once and a bad cut is obvious next to its neighbours. Cells are the
character's 95th-percentile extent and nothing is scaled; a larger glyph is
centre-cropped and its sheet records how many that happened to. Re-rendering
from the JSONL reproduces every sheet byte for byte, which is what makes the
atlas safe to ship. `--no-atlas` skips the render.

The command refuses to run unless the profile hashes to the value the alignment
report recorded, so the two inputs are provably the pair that produced each
other. On every page it rebuilds the same ink mask the alignment extractor
built and checks that the mask reproduces each matched candidate's recorded
`foreground_pixels` and `horizontal_ink_profile` exactly and that the rebuilt
Otsu threshold equals the recorded one. A page that fails is excluded as
`mask_mismatch` carrying both values.

Every value is a pixel count in the `source` frame with `confidence` null and
`confidence_kind` set to `uncalibrated`. The report names nothing the design
treats as latent: no font identity, nominal point size, advance, side bearing,
native word-space width, tracking, kerning, leading, or physical margin.

`--geometry` is optional and off by default. Without it the command behaves
exactly as it does without the flag existing. With it, the command reads an OCR
geometry JSONL produced elsewhere and checks what the recognizer read where each
matched line's first ink run sits. PGDP F2 rejoins a word broken across two
printed lines onto the line where it started, so the next line's first run has
no transcription word to bind to; the flag makes that visible. A page whose
geometry `image_sha256` disagrees with the alignment's `scan_sha256` gets no
witness. Nothing here runs OCR or opens a model.

Pooled estimates and per-page aggregates cover every accepted page. Per-line
and per-word rows are emitted only for the first `--evidence-pages` measured
pages of each book, which keeps a five-book run from emitting roughly 150,000
word records.

Exclusions cover missing or changed inputs, malformed F2 controls, unusable ink
bands, implausible line counts, persistent gutters, probable columns, tables,
illustrations, borders, and high foreground ratios. An excluded or proposed page
is evidence for review, not an accepted mapping. Candidate boxes remain in the
original scan frame. The command does not infer baselines, reading order across
columns, typography, font identity, semantics, rotation, or dewarping, and it
does not use OCR output as verification.

## Cut a glyph inventory

```bash
pgdp-measure glyphs /path/to/pgdp-corpus \
  --alignment ./pgdp-alignment.json \
  --profile ./pgdp-profile.json \
  --geometry ./geometry-v1.jsonl \
  --output ./pgdp-glyphs/
```

`glyphs` cuts one book's glyphs from its own scans and writes an inventory
directory: `glyphs.jsonl`, one row per glyph, and `manifest.json`, the
provenance and per-character coverage. The JSONL is the authority.

Every page of the book is harvested, not only the pages alignment accepted.
Acceptance is a page-level judgement and a poor proxy for whether one line bound
to the right text: measured over three books, lines excluded only for alignment
score agree with the recognizer about as well as accepted ones, at 0.887 to
0.962 against 0.926 to 0.976, while lines on illustration pages agree at 0.560
to 0.683. So every line is admitted only when the recognizer reads at
least four of every five of its transcription words inside its own box. That
needs `--geometry`; without it, non-accepted pages are skipped and accepted ones
go unchecked as before.

The check matters most on accepted pages, where nothing else looks. A line whose
candidate box was matched to the wrong source line still reconciles by word
count often enough to cut, and every glyph in it then carries the wrong
character: `They` cut from WILL, `and` from `side,`, `being` from `morning,`.
Those lines score 0.00 to 0.17 while the lines around them score above 0.8. The
check costs 2.4 to 7.4 percent of accepted lines and the manifest counts what it
rejected. The recognizer only ever rejects, and the label stays the
transcription's. The manifest's page table records each page's alignment state
and how many of its lines were admitted that way, plus what the profile calls
the page and its lines' x-height median and spread.

Those last two separate two different problems. A page whose line x-heights
spread more than about 8 px mixes type sizes: measured, 43 to 67 percent of such
pages are chapter openings against a 2 to 7 percent base rate, and glyphs on
chapter openings are flagged about three times as often as glyphs elsewhere. A
page whose median runs above the book's is not a bigger size at all. The tallest
in `projectID603d7d5e04ca0` is ordinary verse printed with heavy ink that has
spread every letter, which is also why glyphs there merge and get flagged. One
signal is for classification, the other for quality.

An inventory is per book. X-heights run 10 to 18 px across the five aligned
corpus books and glyph pixel sizes scale with them, so a pooled inventory would
be a chimera of sizes; the command refuses an alignment report covering more
than one book.

Glyphs carry one of two label tiers and the two never mix. A `transcribed`
glyph comes from a word on a line reconciled against PGDP F2, so its character
is a human proofer's. A `recognized` glyph comes from a word outside every
matched line **on an accepted page**, which is where the running head and the
folio sit; on a page alignment did not accept, most words fall outside every
matched line simply because few lines matched, so the tier would stop meaning
furniture and start meaning ordinary body text; proofers strip
both from F2, so the only label available is the OCR read that `--geometry`
supplies, and the row carries that read's confidence. Every row names its tier,
so a consumer can take digits from `recognized` while training characters only
on `transcribed`.

A glyph is cut only when the word's box splits at blank columns into exactly as
many ink runs as the word has printing characters. Punctuation is counted and
not labelled: a comma or an apostrophe makes its own ink run as readily as a
letter, so counting letters alone lets a word match by coincidence when one mark
stands alone and one letter pair touches. Measured over four books, that
coincidence covered 1.4 to 15.4 percent of the words a letters-only rule called
separable, and every one of them bound its labels to the wrong ink. The
remaining rejections are letters that touch, and the manifest records how many
failed and why rather than dropping them silently.

Every glyph names the face it was printed in, taken from PGDP's own `<i>`,
`<b>` and `<sc>` markup. This matters most for small capitals: PGDP transcribes
them in mixed case, so `<sc>Lowther Street</sc>` yields the letters `Lowther
Street` and a capital `O` would otherwise land in the lowercase `o` bucket
carrying a lowercase label, which no check comparing a label to a character can
see. Coverage and the atlas are keyed by style, so a style is never pooled with
another under one character.

`label_style` is `roman` when the transcription marks no style and `null` when
no style source could be trusted, which is every glyph on the `recognized` tier.
The evidence comes from the alignment report itself, which already carries a
style run per span, so the command reads no extra input and re-parses no
transcription.

A word whose ink runs flatter than its label predicts is flagged `flat_ascender`
and listed in the manifest for review. In ordinary lowercase an ascender runs
about half again the x-height, 1.46 to 1.58 at the median across the five books.
A word set in small capitals scores 1.00, because every capital is the same
height, and so does a word bound to the wrong ink. Calibrated against the words
PGDP itself marks `<sc>`, a cut at 1.10 catches 74 of 76 while flagging 1.8
percent of roman words in the same book.

An x-height letter that runs at least 1.35 times the median height of its
line's other x-height letters is flagged `overtall`. That catches a box holding
several letters, or a capital where the label says lowercase. The reference is
the line's own ink, not its fitted x-height: normalised by the fitted value the
tail runs to 5.33 and most outliers are correctly labelled glyphs on lines whose
estimate was wrong, which measures the estimator rather than the glyph. Against
the line's own letters the median is 1.00 and the 95th percentile 1.02 to 1.08,
and 12 of 16 outliers reviewed were genuine label errors. It cannot see small
capitals, whose whole point is to sit at x-height.

A glyph narrower than 0.6 times its character's usual width in the book is
flagged `narrow`. A letter built from two strokes joined high up breaks under a
light impression: the arch fails and the stem becomes its own ink run, and the
word can still be cut if some other pair happened to touch, so one glyph carries
half a letter. `h` is the usual casualty, at 12.5 percent of them in
`projectID67a80fde44d34`; `m`, `n` and `r` follow, and `T` loses its arms the
same way. This is the one measure taken across the whole book, because a
half-cut letter looks unremarkable beside its neighbours and only the
character's usual width elsewhere gives it away.

The mirror case is flagged `wide`: a glyph at or above 1.5 times its
character's usual width, which is a missed blank column leaving a letter and its
neighbours in one box. Twelve of twelve reviewed were genuine multi-letter
boxes: `and` labelled `o`, `ingine` labelled `f`, `whe` labelled `t`. It flags
0.06 to 0.91 percent of glyphs. It sees only the gross cases; a box holding a
letter and a comma sits at 1.06 to 1.15 times the median, inside the ordinary
spread, so no width test separates it.

Size gets that far and then stops. A box holding a letter and one stem of its
neighbour, or a `b` where the label says `h`, sits at an ordinary width and an
ordinary height. What separates those is the ink itself, so every glyph is also
resampled onto a 16 by 12 grid and compared against the median shape its own
character takes in that book and style. A glyph more than 1.4 times its
character's own 95th-percentile scatter from that median is flagged
`unlike_character`. Measured, glyphs of one character sit 2.5 to 3.3 from their
reference while two different characters' references sit 4.3 to 7.7 apart, so
the distance carries real signal; the flag costs 0.5 to 5.1 percent of glyphs.
It never proposes a label, only observes that the ink is unlike the one claimed.

All five flags are observations, not verdicts: it says the ink is flatter than the
label predicts, not why. The causes found so far are small capitals PGDP never
marked, a word bound to the wrong ink, and a badly cut glyph box. Only looking
settles which, so the manifest carries a bounded queue of suspects rather than
acting on them.

Quality is recorded and never enforced. Each row carries where its rows sit
against the line's x-height top and baseline, whether it reaches the line box's
own top or bottom row, and its ink density. The reference is the line box, not
the word box: a word box is tight to its own ink, so its tallest and lowest
glyphs always touch it.

The atlas is a render of the JSONL, not a second source. One grid per
character per tier holds every sample of that character in the book, sharded at
1,024 cells a sheet, so opening `atlas/transcribed/U+0065-000.png` shows every
`e` at once and a bad cut is obvious next to its neighbours. Cells are the
character's 95th-percentile extent and nothing is scaled; a larger glyph is
centre-cropped and its sheet records how many that happened to. Re-rendering
from the JSONL reproduces every sheet byte for byte, which is what makes the
atlas safe to ship. `--no-atlas` skips the render.

The command refuses to run unless the profile hashes to the value the alignment
report recorded, the same check `typography` makes. A page whose geometry
`image_sha256` disagrees with the alignment's `scan_sha256` yields no furniture,
and the manifest counts those pages. The output directory gets an empty
`.nobackup` marker, because the whole inventory can be rebuilt from the corpus
and the two reports.

## Per-subcommand flags

### `rank <corpus_root>`

The positional `corpus_root` is the local PGDP corpus directory.

| Flag | Meaning |
|------|---------|
| `--output PATH` | Write the JSON report here (default: `./pgdp-ranking.json`); the path must be outside the corpus root |
| `--project-limit N` | Limit the number of ranked projects in the report (default: 50; must be positive) |
| `--pages-per-project N` | Limit selected review pages per reported project (default: 12; must be positive) |

### `profile <corpus_root>`

The positional `corpus_root` is the local PGDP corpus directory. The command
profiles only the selected pages in the supplied M14 ranking report, unless
`--whole-book` widens what it measures.

| Flag | Meaning |
|------|---------|
| `--ranking PATH` | Read this required `pgdp-rank/v1` JSON report |
| `--output PATH` | Write the required `pgdp-profile/v2` JSON report here; it must be outside the corpus root, differ from `--ranking`, and not name a directory |
| `--whole-book` | Measure every page image in each ranked project directory, but emit only the ranked pages |

`--whole-book` separates what is measured from what is reported. Book-level
estimates describe a book, so they are pooled over every page of it. The
emitted page list is unchanged, so a later `align` run still sees the
ranking's selection. Profiling costs about 9.5ms per page against alignment's
1s, which is why the two stages take different page sets.

The run prints `pages pooled` alongside `pages measured` in this mode, and each
pooled estimate records the pages that contributed in `evidence_pages`.

Whole-book measurement also fits per-book page templates. A book whose first ink
band holds a steady position across the volume gets templates for its normal
recto, normal verso, and chapter-opening pages, and every page is classified
against them as `normal_recto`, `normal_verso`, `chapter_opening`, or `unknown`.
A book whose first band wanders gets no templates and every page classifies
`unknown`.

The class decides which bands `align` ignores. A normal page's first band
is its running head, which is printed but deleted from F2, so no candidate is
emitted for it. Every other class suppresses nothing. Front matter and plates
have no class of their own because the corpus offers no discriminator for them.

### `align <corpus_root>`

The `corpus_root` positional argument is the local PGDP corpus directory.
The command aligns F2 source lines with candidates derived from the referenced
corpus scans. Those scans must still be available and match the supplied M15a
`profile` report.

| Flag | Meaning |
|------|---------|
| `--profile PATH` | Read this required `pgdp-profile/v2` JSON report |
| `--output PATH` | Write the required `pgdp-alignment/v1` JSON report here; it must be outside the corpus root, differ from `--profile`, and not name a directory |

### `typography <corpus_root>`

The `corpus_root` positional argument is the local PGDP corpus directory. The
command measures typographic observables from the accepted pages of an
alignment report, and it opens no font, renders no text, and never leaves the
`source` coordinate frame.

| Flag | Meaning |
|------|---------|
| `--alignment PATH` | Read this required `pgdp-alignment/v3` JSON report |
| `--profile PATH` | Read this required `pgdp-profile/v2` JSON report; it must hash to the value the alignment recorded |
| `--output PATH` | Write the required `pgdp-typography/v1` JSON report here; it must be outside the corpus root, differ from both inputs, and not name a directory |
| `--evidence-pages N` | Emit per-line and per-word rows for this many measured pages per book (default 12) |
| `--geometry PATH` | Optional OCR geometry JSONL for this book. Marks a line whose first ink run is a continuation fragment the transcription does not carry |

### `glyphs <corpus_root>`

The `corpus_root` positional argument is the local PGDP corpus directory. The
command cuts one book's glyph inventory from its own scans and writes it as a
directory. It opens no font, renders no text, runs no OCR, and never leaves the
`source` coordinate frame.

| Flag | Meaning |
|------|---------|
| `--alignment PATH` | Read this required `pgdp-alignment/v3` JSON report; it must cover exactly one book |
| `--profile PATH` | Read this required `pgdp-profile/v2` JSON report; it must hash to the value the alignment recorded |
| `--output PATH` | Write the required `pgdp-glyphs/v1` inventory directory here; it must be outside the corpus root and must not name a file |
| `--geometry PATH` | Optional OCR geometry JSONL for this book. Harvests the running head and the folio as the `recognized` label tier |
| `--no-atlas` | Write the JSONL and manifest without rendering the per-character atlas |
