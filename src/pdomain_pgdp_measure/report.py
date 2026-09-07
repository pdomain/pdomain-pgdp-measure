from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Protocol


class _JsonReport(Protocol):
    """A report that supplies a JSON-safe deterministic payload."""

    def to_dict(self) -> object: ...


def write_report(report: _JsonReport, output_path: str | Path, corpus_root: str | Path) -> None:
    """Write a deterministic report without modifying the corpus."""

    output = Path(output_path).resolve()
    root = Path(corpus_root).resolve()
    if output.is_relative_to(root):
        raise ValueError("Report output must be outside the corpus root.")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as temporary_file:
            json.dump(
                report.to_dict(),
                temporary_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            _ = temporary_file.write("\n")
            temporary_file.flush()
            _ = os.fsync(temporary_file.fileno())
        _ = temporary_path.replace(output)
    except Exception as write_error:
        try:
            _remove_temporary_file(temporary_path)
        except OSError as cleanup_error:
            raise ExceptionGroup(
                "Writing the report and removing its temporary file both failed.",
                [write_error, cleanup_error],
            ) from None
        raise

    directory_descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        _ = os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _remove_temporary_file(temporary_path: Path) -> None:
    try:
        temporary_path.unlink()
    except FileNotFoundError:
        return
