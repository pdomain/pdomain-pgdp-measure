---
Status: active
Owner: CT
Created: 2026-11-06
Last verified: 2026-11-12
Kind: process
---

# Remaining work

## Agent Index

- **Kind:** process
- **Status:** active
- **Owner:** CT
- **Created:** 2026-11-06
- **Last verified:** 2026-11-12
- **Provenance:** checked the architecture docs, the 2026-09-07 Gate 3 recheck in
  `pdomain-ocr-synth`, and the package boundary in `pyproject.toml` on 2026-11-12
- **Disposition:** The only open-work list for this package. Shipped behavior is a link.
- **Read when:** deciding what is left to do, or what should not be built here.
- **Search terms:** remaining work, roadmap, Gate 3, glyph labels, admission floor, separate repo

Two things are still open in this package. Gate 3 still fails on two books after the small-caps
correction. The 30-page book admission floor is still undecided.

The five measurement commands stay in this package. Page composition, font outlining, scan
rectification, and a whole-corpus batch runner belong elsewhere.

This package has no issue file. The index is [docs/issues/README.md](issues/README.md).

## Gate 3 still fails on two books

The original review scored 0.978 pooled: 23 of 1,050 glyphs wrong. Two of five books failed.

A recheck on 2026-09-07 showed that three of those 23 marks are correct small-caps records. The
sheet had hidden `label_style`.

Counting them as correct moves the pool from 0.9781 to 0.9810, and
moves `projectID603d7d5e04ca0` from 0.9619 to 0.9762. `projectID67a80fde44d34` stays at 0.9429.
Gate 3 is judged per book, so both books still fail.

The study is in `pdomain-ocr-synth` at
`docs/research/2026-09-07-gate3-recheck-on-the-713-page-alignment.md`. Current behavior is in
[PGDP per-book glyph inventory](architecture/pgdp-glyph-inventory.md).

Three decisions are still open.

The owner re-reads cells `e13`, `h21`, and `a10`. One reader thought those three were cut
correctly. That reading is not a verdict.

Someone chooses how the gate is judged. The choices named by the recheck are the filtered
inventory, a lower unfiltered number, or a fresh sample from the 713-page pool read by a person.
The recheck did not draw that sample.

Filtering the five quality flags takes the original review
to 0.994 pooled, with every book over the floor. That number is what filtering buys. It is not the
gate result.

Someone measures how often a word gap is missed in italic and script faces. Cell `e5` is one miss
in a script italic. No current flag covers that class. The measurement belongs in this
package, next to the word-gap work typography already does.

Re-running alignment does not close the gate. All 23 marks survive the band-identification fixes.

## The 30-page admission floor is not decided

`MINIMUM_ACCEPTED_PAGES_PER_BOOK` is 30. The code comment calls it a seed.

[PGDP source-line alignment](architecture/pgdp-source-line-alignment.md) says corpus review should
recalibrate it once typography states a page count.

[PGDP font-free typography](architecture/pgdp-font-free-typography.md) says the measurement was
made and the floor stays at 30.

No test in this repo records that calibration.

The sources are compared in the [intent map](context/intent-map.md).

Both architecture docs stay as written until that decision is made.

## Shipped behavior

- [PGDP ranking and review queue](architecture/pgdp-ranking-and-review-queue.md)
- [PGDP observed geometry profiling](architecture/pgdp-observed-geometry-profiling.md)
- [PGDP source-line alignment](architecture/pgdp-source-line-alignment.md)
- [PGDP font-free typography](architecture/pgdp-font-free-typography.md)
- [PGDP per-book glyph inventory](architecture/pgdp-glyph-inventory.md), except the open gate above

## Work that belongs in another repository

Do not split `rank`, `profile`, `align`, `typography`, or `glyphs` out of this package. The
extraction put the measurement half here so the synthesizer can depend on one library. Runtime
dependencies stay pydantic, Pillow, and numpy.

Page composition from glyph samples does not belong here. This package does not compose pages.
`pdomain-ocr-synth` already renders synthetic pages, and the README names it as the consumer.

Font outlines and autotracing do not belong here. This package does not build fonts.
`pdomain-fonts` stores font files that have no stable download URL. A traced outline is a
different job. The glyph architecture says it would be noisier than the bitmap it came from.
Build that only if that judgment changes. Build it in its own repository.

Scan rectification does not belong here. Profile and alignment both refuse rotation, perspective,
and page warp. Both call that a separate contract. No repository owns it. A rectification
library, with its own report, is the clearest new repository. This package would consume it later.

Training export does not belong here. Turning `glyphs.jsonl` into trainer input belongs in
`pdomain-ocr-training`.

A review application would overlap `pdomain-ocr-labeler-spa`. The Gate 3
sheets stay scripts. The recheck scripts already import this package.

A whole-corpus batch runner does not belong in this git tree. This repository commits atlases for
five books. The pre-commit hook refuses a file over 1000 kB.

The glyph plan estimated about 16
million glyphs if the 286 projects then in view all shipped atlases. The extraction check later
saw the corpus grow from 330 projects to 390.

`pdomain-ops` is suite plumbing and GPU dispatch. A runner
that stores reports outside git fits there, or in a small repository of its own.

Do not start another F2 parser, and do not replace this package's loader. The loader carries a
continued `/#` … `#/` block from one page onto the next. The tokenizer in
`pdomain-book-contracts` reads one page at a time. It warns `unclosed_block` when a block is still
open at the end of that page. `pdomain-book-tools` re-exports that tokenizer.

`pdomain-book-tools` also depends on OpenCV and pins numpy below 2.5. This package requires numpy
2.5 or newer. Depending on either package would add a fourth runtime dependency, which the
extraction rule forbids.

Connected-component splitting of touching letters stays here. This package measures the column-run
baseline it has to beat.
