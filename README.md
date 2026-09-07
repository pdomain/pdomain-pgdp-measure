# pdomain-pgdp-measure

Font-free measurement of PGDP scans: geometry, alignment, typography, and glyphs.

Extracted from `pdomain-ocr-synth`, which measured scans and rendered synthetic pages
from one package. This package owns the measurement half. The synthesizer consumes what
it produces.

The console script is `pgdp-measure`, with subcommands `rank`, `profile`, `align`,
`typography`, and `glyphs`.

Runtime dependencies are exactly pydantic, Pillow, and numpy. This package must stay
installable without cv2, torch, or doctr, and a test enforces that.

Extraction plan and acceptance criteria live in `pdomain-ocr-synth` at
`docs/plans/2026-09-06-extract-pgdp-measurement-library.md`.

## The move is verified byte for byte

This package reproduces `pdomain-ocr-synth`'s measurement output exactly. All 927 report files
across the five aligned books match the baseline captured at `pdomain-ocr-synth` commit `f9fdfef`,
before anything moved.

Two provenance fields are normalized before comparing, and only two. `tool_version` is the
package's own VCS version, so it necessarily changes when the package changes. `alignment_sha256`
hashes `alignment.json`, which carries `tool_version`, so it moves for that reason alone. That was
proven rather than assumed: on all five books the recorded `alignment_sha256` equals the raw sha of
`alignment.json`, and the two alignment files hash identically once `tool_version` is normalized.
`profile_sha256`, `rows_sha256`, and `geometry_sha256` are left alone, because the files they hash
carry no version field.

Every measured value is identical. In `projectID67a80fde44d34`'s typography report, 2 of 29,680
leaf values differ, and both are those provenance fields. Every atlas PNG and every `glyphs.jsonl`
matched with no normalization at all.

`rank` is checked separately, because the acceptance run consumes a frozen ranking and never
exercises it. All 328 projects shared with the baseline are byte-identical; the corpus itself grew
from 330 projects to 390 in the interval.

The baseline and the acceptance scripts live outside both repositories, in
`/workspaces/pdomain/.extraction-baseline/` and `/workspaces/pdomain/.extraction-verify/`.
