"""Utilities for listing files and preparing text chunks."""

from __future__ import annotations

import glob
import os
from typing import Iterable, List, Sequence, Tuple

from .config import SUPPORTED_EXTS


def read_text(path: str) -> str:
    """Read text from *path* using UTF-8 with ignore fallback."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def list_code_files(root: str, *, extensions: Iterable[str] | None = None) -> List[str]:
    """Return all supported source files under *root* sorted deterministically."""
    exts: Sequence[str] = tuple(extensions or SUPPORTED_EXTS)
    files: List[str] = []
    for ext in exts:
        pattern = os.path.join(root, f"**/*{ext}")
        files.extend(glob.glob(pattern, recursive=True))
    # Remove duplicates while keeping deterministic order
    return sorted(set(files))


def make_line_chunks(text: str, chunk_lines: int, overlap: int) -> List[Tuple[int, str]]:
    """Split *text* into ``(start_line, chunk)`` pairs for embedding."""
    lines = text.splitlines()
    chunks: List[Tuple[int, str]] = []
    index = 0
    step = max(1, (chunk_lines - overlap) if chunk_lines > overlap else chunk_lines)
    while index < len(lines):
        piece = lines[index : index + chunk_lines]
        if not piece:
            break
        chunks.append((index + 1, "\n".join(piece)))
        index += step
    return chunks
