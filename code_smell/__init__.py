"""Core package for the Code Smell MCP server."""

from .server import MCPServer, run_stdio_server
from .skill import CodeSmellSkill
from .google import ensure_genai_configured

__all__ = [
    "CodeSmellSkill",
    "MCPServer",
    "ensure_genai_configured",
    "run_stdio_server",
]
