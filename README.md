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
