"""The mode a staged write publishes at.

`tempfile.mkstemp` creates at 0600 and ignores the umask, and a rename keeps
that mode, so a staged write must chmod before publishing or no other uid can
read the result. See shared-devtools docs/process/shared-file-permissions.md.
"""

from __future__ import annotations

import os

import pytest

from pdomain_pgdp_measure.file_mode import FILE_MODE, shared_file_mode


@pytest.mark.parametrize(("mask", "expected"), [(0o002, 0o664), (0o022, 0o644), (0o077, 0o600)])
def test_mode_is_0666_minus_the_umask(mask: int, expected: int) -> None:
    """Deriving from the umask keeps a caller who runs a private umask private."""
    previous = os.umask(mask)
    try:
        assert shared_file_mode() == expected
    finally:
        _ = os.umask(previous)


def test_never_marks_a_file_executable() -> None:
    """0666, not 0777: nothing published this way is a program."""
    for mask in (0o000, 0o002, 0o022, 0o077):
        previous = os.umask(mask)
        try:
            assert not shared_file_mode() & 0o111
        finally:
            _ = os.umask(previous)


def test_reading_the_mode_leaves_the_umask_alone() -> None:
    """Reading a umask requires setting it; the helper must put it back."""
    before = os.umask(0o022)
    _ = os.umask(before)
    try:
        _ = shared_file_mode()
        after = os.umask(0o022)
        _ = os.umask(after)
        assert after == before
    finally:
        _ = os.umask(before)


def test_module_constant_is_captured_once() -> None:
    """Import-time capture avoids a process-global umask write per file write."""
    current = os.umask(0)
    _ = os.umask(current)

    assert (0o666 & ~current) == FILE_MODE
