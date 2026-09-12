"""The staging file an atomic write publishes from.

`tempfile.mkstemp` hardcodes 0600 and ignores the umask, and a rename preserves
that mode, so a staged write built on it publishes a file no other uid can
read. Creating the file with an explicit mode lets the kernel apply the umask,
so there is no chmod to forget. See shared-devtools
docs/process/shared-file-permissions.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pdomain_pgdp_measure.staged_write import open_staged


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


@pytest.mark.parametrize(("mask", "expected"), [(0o002, 0o664), (0o022, 0o644), (0o077, 0o600)])
def test_mode_is_whatever_the_umask_allows(tmp_path: Path, mask: int, expected: int) -> None:
    """The kernel applies the umask, so a caller with a private umask stays private."""
    previous = os.umask(mask)
    try:
        descriptor, staged = open_staged(tmp_path / "target.json")
    finally:
        _ = os.umask(previous)
    os.close(descriptor)

    assert _mode(staged) == expected


def test_never_creates_an_executable_file(tmp_path: Path) -> None:
    """0666, not 0777: nothing published this way is a program."""
    previous = os.umask(0)
    try:
        descriptor, staged = open_staged(tmp_path / "target.json")
    finally:
        _ = os.umask(previous)
    os.close(descriptor)

    assert not _mode(staged) & 0o111


def test_refuses_to_reuse_an_existing_path(tmp_path: Path) -> None:
    """O_EXCL is what makes the staged name safe without mkstemp."""
    descriptor, staged = open_staged(tmp_path / "target.json")
    os.close(descriptor)

    with pytest.raises(FileExistsError):
        _ = os.open(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)


def test_stages_a_hidden_sibling(tmp_path: Path) -> None:
    """A sibling keeps the rename on one filesystem, which is what makes it atomic."""
    target = tmp_path / "target.json"
    descriptor, staged = open_staged(target)
    os.close(descriptor)

    assert staged.parent == target.parent
    assert staged.name.startswith(".target.json.")
    assert staged.name.endswith(".tmp")


def test_does_not_touch_the_umask(tmp_path: Path) -> None:
    """Nothing here reads the umask, so nothing can disturb another thread."""
    before = os.umask(0o022)
    _ = os.umask(before)
    try:
        descriptor, _staged = open_staged(tmp_path / "target.json")
        os.close(descriptor)

        after = os.umask(0o022)
        _ = os.umask(after)
        assert after == before
    finally:
        _ = os.umask(before)
