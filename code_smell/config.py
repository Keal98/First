"""Configuration constants for the Code Smell MCP server."""

from __future__ import annotations

import os
from typing import Set

CHROMA_DIR: str = os.getenv("CHROMA_DIR", ".chroma_code_smell")
COLLECTION_PREFIX: str = "code_smell_chunks__"
CHUNK_LINES: int = int(os.getenv("CHUNK_LINES", "80"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "10"))
SUPPORTED_EXTS: Set[str] = {
    ".py",
    ".js",
    ".ts",
    ".cs",
    ".java",
    ".go",
    ".rb",
    ".php",
    ".cpp",
    ".c",
    ".rs",
}
EMBED_MODEL: str = "text-embedding-004"
LLM_MODEL: str = "gemini-1.5-pro"
DEFAULT_RETRIEVAL_QUERY: str = (
    "potential security smells: sql injection, hardcoded secrets, insecure cryptography, "
    "command injection, unsafe deserialization, deprecated libs, missing validation"
)
