---
Status: active
Owner: CT
Created: 2026-11-06
Last verified: 2026-11-12
Kind: process
---

# The 30 accepted-page admission floor is not settled in the sources that name it

## Agent Index

- **Kind:** process
- **Status:** active
- **Owner:** CT
- **Created:** 2026-11-06
- **Last verified:** 2026-11-12
- **Provenance:** compared the alignment code comment, the alignment architecture, and the typography architecture on 2026-11-06. Rechecked on 2026-11-12. The disagreement is unchanged.
- **Disposition:** Records the one claim this repo does not prove. Gate 3's open judgment is on the roadmap, not here.
- **Read when:** deciding whether the 30-page admission floor is still open.
- **Search terms:** intent map, needs a decision, admission floor, MINIMUM_ACCEPTED_PAGES_PER_BOOK

## Needs a decision

Gate 3's remaining judgment is on the [roadmap](../roadmap.md). This file is only the admission
floor.

The 30 accepted-page admission floor is not settled in the sources that name it.

`MINIMUM_ACCEPTED_PAGES_PER_BOOK` in `src/pdomain_pgdp_measure/alignment_review.py` is commented as a seed.

The comment says nothing downstream had yet stated how many pages a per-book style fit needs.

[PGDP source-line alignment](../architecture/pgdp-source-line-alignment.md) says corpus review should recalibrate the floor once typography states the page count.

It also repeats that comment.

[PGDP font-free typography](../architecture/pgdp-font-free-typography.md) says the measurement was made.

Pooled x-height, baseline pitch, and stroke width settle within 0.5 px by 5 body pages in four of the five books. They settle within 0.5 px by 20 pages in the fifth book.

Word gap settles at 5, 50, 5, 30, and 5 body pages across those books.

The floor stays at 30.

That covers the three steadier observables and four of the five word-gap answers.

The code constant is still 30.

This repo has no test that records the calibration.

Both architecture docs stay as written until someone decides whether the floor is still open.
