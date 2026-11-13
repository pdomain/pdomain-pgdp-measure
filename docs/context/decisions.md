---
Status: active
Owner: CT
Created: 2026-11-12
Last verified: 2026-11-12
Kind: process
---

# Decisions

## Agent Index

- **Kind:** process
- **Status:** active
- **Owner:** CT
- **Created:** 2026-11-12
- **Last verified:** 2026-11-12
- **Provenance:** written when the glyph inventory plan was retired on 2026-11-12
- **Disposition:** Tombstones for docs retired in this repository. Earlier measurement decisions
  stay in `pdomain-ocr-synth`.
- **Read when:** asking where a retired plan went.
- **Search terms:** decisions, tombstone, glyph inventory plan, retired

Decisions made before this package was extracted stay in `pdomain-ocr-synth` at
`docs/context/decisions.md`. This file records retirements made here. Entries are appended. An
older entry is not rewritten.

### 2026-11-12 Retired: PGDP per-book glyph inventory plan

- Old path: `docs/plans/2026-09-05-pgdp-per-book-glyph-inventory.md`
- Outcome: implemented, with Gate 3 still open
- Superseded by: `docs/architecture/pgdp-glyph-inventory.md`
- Removal commit: (fill after commit)
- Rationale kept: current behavior, the atlas scale limit, and the 2026-09-07 recheck are in that
  architecture doc. The study itself stays in `pdomain-ocr-synth` at
  `docs/research/2026-09-07-gate3-recheck-on-the-713-page-alignment.md`.
- Remaining work: [the roadmap](../roadmap.md). Gate 3 still fails on two books after the
  small-caps correction. The 30-page admission floor is still open.

### 2026-11-13 Enabled the documentation workflow

- Context: this repository had architecture and context docs but no docgraph configuration.
- Decision: use the canonical docgraph bootstrap, with workflow enforcement and legacy detection
  enabled, `master` as the comparison base, and a strict check at the end of `just ci`.
- Rationale: the owner authorized the minimal bootstrap for this governed migration. The earlier
  instruction against adding docgraph applied to the completed Makefile migration.
- Evidence: `docgraph.toml`, `DOCGRAPH.md`, and `justfile`; the architecture audit checked all
  eleven previously tracked Markdown documents against code, tests, and history.
- Remaining work: the open Gate 3 judgments and admission-floor calibration stay on the
  [roadmap](../roadmap.md) and [intent map](intent-map.md). No plan or spec remained to retire.
