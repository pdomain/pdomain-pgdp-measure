---
kind: process
status: active
owner: CT
created: 2026-11-13
last_verified: 2026-11-13
---

# Measurement contracts and release preparation

The five measurement commands are shipped. Their current contracts are linked from the
[roadmap](../roadmap.md#shipped-behavior). The architecture audit refreshed input integrity,
template spread, and typography sampling details against source and tests.

The remaining product work stays on the [roadmap](../roadmap.md). Gate 3 still fails on two books.
The 30-page admission floor remains an owner decision in the [intent map](intent-map.md).
This audit did not rerun historical corpus measurements or settle either question.

The documentation migration is on `docs/architecture-retirement-measure`, owned by this
session. All 1,088 tests passed in the isolated worktree on 2026-11-13. No failing or flaky test
was observed in that run.

Documentation now uses the generated `docgraph.toml` workflow with `master` as its comparison
base. `just ci` ends with `docgraph check --strict`. See the [bootstrap decision](decisions.md#2026-11-13-enabled-the-documentation-workflow).

## Release remains held

The latest published release is `0.1.0`; the proposed patch release is `0.1.1`.
The candidate requires NumPy 2.5 or newer. Labeler must keep using the published measurement
package until the compatible book-tools dependency chain is published and checked.
Publication awaits the coordinator's staged authorization.

## Evidence

- Code and tests: the evidence sections of the linked architecture docs.
- Open work: [roadmap](../roadmap.md) and [intent map](intent-map.md).
- Local build-tool changes: commits `65936c9`, `3dde34d`, and `fd854f5`.
- Prior retirement: [decisions](decisions.md). No tracked plan or spec remains to retire.
