"""The file mode a staged write must publish at.

``tempfile.mkstemp`` creates at 0600 and ignores the umask by design, and a
rename preserves whatever mode the file already has. Every write-to-temp-then-
rename therefore publishes a file no other uid can read unless it chmods first.
That once put 52,575 files totalling 6.9 GiB outside every backup snapshot,
because the host's restic runs as a different uid.

The rule and its worked examples live in shared-devtools at
``docs/process/shared-file-permissions.md``.
"""

from __future__ import annotations

import os


def shared_file_mode() -> int:
    """Return the mode a plain ``open()`` would produce: 0666 minus the umask.

    ``os.umask`` has no read-only form, so reading the umask means setting it
    to zero and putting it back, which is process-global. Calling this per
    write would expose a zero umask to every other thread for those two
    syscalls, so call it once at import while the module is single-threaded.

    0666 rather than 0777: nothing published this way is a program. Deriving
    the mode from the umask rather than hardcoding one keeps a caller who runs
    a private umask private: at 0o077 this still returns 0600.
    """
    value = os.umask(0)
    _ = os.umask(value)
    return 0o666 & ~value


FILE_MODE = shared_file_mode()
