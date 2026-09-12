"""Create the staging file for an atomic write, at a mode others can read.

`tempfile.mkstemp` hardcodes 0600 and ignores the umask. That is right for a
private scratch file and wrong for one about to be published, because a rename
preserves the mode: the result is unreadable to any other uid, the host's
restic backup included. That once put 52,575 files totalling 6.9 GiB outside
every backup snapshot.

Passing the mode to `os.open` instead lets the kernel apply the umask exactly
as it does for a plain `open()`, so there is no chmod to forget and no umask to
read. The rule lives in shared-devtools at
`docs/process/shared-file-permissions.md`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from pathlib import Path


def open_staged(path: Path) -> tuple[int, Path]:
    """Return a write file descriptor and the staging path beside *path*.

    The staging file is a sibling so the later rename stays on one filesystem,
    which is what makes it atomic. 0666 rather than 0777: nothing published
    this way is a program. ``O_EXCL`` keeps ``mkstemp``'s guarantee that
    creation fails rather than opening an existing file or following a symlink
    into one.
    """
    while True:
        staged = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        try:
            return os.open(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666), staged
        except FileExistsError:  # pragma: no cover - needs a uuid4 collision
            continue
