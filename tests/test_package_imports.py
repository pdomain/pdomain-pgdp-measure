"""The package imports, and it drags in nothing heavy."""

from __future__ import annotations

import subprocess
import sys


def test_package_imports() -> None:
    import pdomain_pgdp_measure

    assert pdomain_pgdp_measure.__name__ == "pdomain_pgdp_measure"


def test_package_is_real_not_a_namespace_package() -> None:
    """An empty src directory satisfies a bare import, so check for __init__.py.

    Without this, the import test above passes against nothing at all: Python
    treats any directory on the path as an implicit namespace package.
    """
    import pdomain_pgdp_measure

    assert pdomain_pgdp_measure.__file__ is not None
    assert pdomain_pgdp_measure.__file__.endswith("__init__.py")


def test_import_pulls_no_heavy_dependency() -> None:
    """cv2, torch, and doctr must never become dependencies of this package."""
    code = (
        "import sys, pdomain_pgdp_measure;"
        "bad=[m for m in ('cv2','torch','doctr','nicegui') if m in sys.modules];"
        "print(','.join(bad))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
