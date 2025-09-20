"""Generic helpers used across the Code Smell server."""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Iterable, List, Tuple


def repo_hash(path: str) -> str:
    """Return a deterministic short hash for *path* used in Chroma collections."""
    return hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:16]


def deduplicate_findings(findings: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Remove duplicated findings based on ``(file, line, type)``."""
    seen: set[Tuple[object, object, object]] = set()
    unique: List[Dict[str, object]] = []
    for finding in findings:
        key = (
            finding.get("file"),
            finding.get("line"),
            finding.get("type"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(finding))
    return unique
