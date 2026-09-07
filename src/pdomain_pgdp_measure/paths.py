from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class UnsafePathError(ValueError):
    """A PGDP report reference cannot be resolved safely."""


@dataclass(frozen=True, slots=True)
class ImageResolution:
    """The first safe image candidate and whether another candidate escaped."""

    image_path: Path | None
    unsafe_candidate: bool


def resolve_project_directory(*, corpus_root: Path, project_id: str) -> Path:
    """Resolve one direct project directory inside a corpus root."""

    root = corpus_root.resolve()
    reference = require_canonical_relative_reference(
        value=project_id,
        label="project reference",
        direct_child=True,
    )
    project_directory = (root / reference).resolve()
    if not project_directory.is_relative_to(root):
        raise UnsafePathError(f"Project reference escapes the corpus root: {project_id!r}.")
    if not project_directory.is_dir():
        raise ValueError(f"Project directory does not exist: {project_id!r}.")
    return project_directory


def resolve_image_candidate(
    *, project_directory: Path, page_name: str, corpus_root: Path | None = None
) -> ImageResolution:
    """Find the first safe root-then-images candidate for one page name."""

    page_path = _validate_relative_reference(value=page_name, label="page reference")
    resolved_corpus_root = corpus_root.resolve() if corpus_root is not None else None
    unsafe_candidate = False
    for candidate_root in (project_directory, project_directory / "images"):
        resolved_root = candidate_root.resolve()
        candidate = (candidate_root / page_path).resolve()
        if not candidate.is_relative_to(resolved_root):
            unsafe_candidate = True
            continue
        if resolved_corpus_root is not None and not candidate.is_relative_to(resolved_corpus_root):
            unsafe_candidate = True
            continue
        if candidate.is_file():
            return ImageResolution(image_path=candidate, unsafe_candidate=unsafe_candidate)
    return ImageResolution(image_path=None, unsafe_candidate=unsafe_candidate)


def corpus_relative_path(*, path: Path, corpus_root: Path) -> str:
    """Return an already-safe path in portable corpus-relative form."""

    root = corpus_root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(root):
        raise UnsafePathError(f"Image path escapes the corpus root: {path!s}.")
    return resolved_path.relative_to(root).as_posix()


def require_canonical_relative_reference(
    *, value: str, label: str, direct_child: bool = False
) -> Path:
    """Require the portable spelling of a safe relative report reference."""

    reference = _validate_relative_reference(
        value=value,
        label=label,
        direct_child=direct_child,
    )
    if reference.as_posix() != value:
        raise UnsafePathError(f"Noncanonical {label}: {value!r}.")
    return reference


def _validate_relative_reference(*, value: str, label: str, direct_child: bool = False) -> Path:
    if not value or "\x00" in value:
        raise UnsafePathError(f"Unsafe {label}: {value!r}.")
    if "\\" in value or "." in value.split("/"):
        raise UnsafePathError(f"Unsafe {label}: {value!r}.")
    reference = Path(value)
    if reference.is_absolute() or ".." in reference.parts:
        raise UnsafePathError(f"Unsafe {label}: {value!r}.")
    if direct_child and len(reference.parts) != 1:
        raise UnsafePathError(f"Unsafe {label}: {value!r}.")
    return reference
