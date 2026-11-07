---
Status: active
Owner: CT
Created: 2026-11-06
Last verified: 2026-11-06
Kind: process
---

# Remaining work

## Agent Index

- **Kind:** process
- **Status:** active
- **Owner:** CT
- **Created:** 2026-11-06
- **Last verified:** 2026-11-06
- **Provenance:** checked the live plan, the architecture docs, and the code and tests on 2026-11-06
- **Disposition:** The open list for this package. Shipped behavior is a link, not a story.
- **Read when:** deciding what is left to do.
- **Search terms:** remaining work, roadmap, Gate 3, glyph labels, admission floor

One acceptance gate is still open.

The five measurement commands are shipped.

## Transcribed glyph labels are still short of the 0.98 floor

Transcribed glyph labels are still short of the 0.98 floor.

The pooled score is 0.978 (23 of 1,050 wrong).

Two of five books fail.

Three of those 23 may be right records marked wrong because the sheet hid `label_style`.

The example is a small-cap N stored as character `n`.

Rescore with the style shown, at most fifteen cells a row, before the gate can close. [Per-book glyph inventory plan](plans/2026-09-05-pgdp-per-book-glyph-inventory.md)

## Needs a decision

- The 30-page book admission floor is still called an uncalibrated seed in `src/pdomain_pgdp_measure/alignment_review.py` and in [PGDP source-line alignment](architecture/pgdp-source-line-alignment.md).
- [PGDP font-free typography](architecture/pgdp-font-free-typography.md) says that floor stays at 30 after calibration.
- No test in this repo proves the calibration.
- [Intent map](context/intent-map.md)

## Shipped behavior

- [PGDP ranking and review queue](architecture/pgdp-ranking-and-review-queue.md)
- [PGDP observed geometry profiling](architecture/pgdp-observed-geometry-profiling.md)
- [PGDP source-line alignment](architecture/pgdp-source-line-alignment.md)
- [PGDP font-free typography](architecture/pgdp-font-free-typography.md)
- [PGDP per-book glyph inventory](architecture/pgdp-glyph-inventory.md), except the open gate above
