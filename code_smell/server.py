"""Minimal MCP stdio server that exposes the CodeSmellSkill."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

from semantic_kernel.kernel import Kernel
from tqdm import tqdm

from .google import ensure_genai_configured
from .skill import CodeSmellSkill
from .utils import deduplicate_findings, repo_hash


class MCPServer:
    """JSON-RPC server that bridges MCP tools to the skill implementation."""

    def __init__(self):
        ensure_genai_configured()
        self.kernel = Kernel()
        self.skills_cache: Dict[str, CodeSmellSkill] = {}

    def _tools_list(self) -> Dict[str, Any]:
        return {
            "tools": [
                {
                    "name": "index_repo",
                    "description": "Build/refresh index for a repo",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
                {
                    "name": "scan_repo",
                    "description": "Retrieve suspicious chunks and analyze them with Gemini",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "top_k": {"type": "integer", "default": 20},
                        },
                        "required": ["path"],
                    },
                },
                {
                    "name": "scan_snippet",
                    "description": "Analyze a single in-memory snippet",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "code": {"type": "string"},
                            "start_line": {"type": "integer", "default": 1},
                        },
                        "required": ["filename", "code"],
                    },
                },
            ]
        }

    def _get_skill(self, repo_path: str) -> CodeSmellSkill:
        key = os.path.abspath(repo_path)
        skill = self.skills_cache.get(key)
        if skill is None:
            skill = CodeSmellSkill(key)
            self.kernel.add_plugin(skill, f"smell::{repo_hash(key)}")
            self.skills_cache[key] = skill
        return skill

    def handle_initialize(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": msg["id"], "result": {"capabilities": {"tools": {}}}}

    def handle_tools_list(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": msg["id"], "result": self._tools_list()}

    def handle_tools_call(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        params = msg.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {}) or {}

        if name == "index_repo":
            path = args["path"]
            skill = self._get_skill(path)
            result = skill.index_repo()
            return {"id": msg["id"], "result": {"content": result}}

        if name == "scan_repo":
            path = args["path"]
            top_k = int(args.get("top_k", 20))
            skill = self._get_skill(path)
            if skill.coll.count() == 0:
                skill.index_repo()
            retrieved = json.loads(skill.retrieve_chunks(top_k))
            findings: List[Dict[str, Any]] = []
            for item in tqdm(retrieved, desc="Analyzing"):
                response_json = skill.analyze_with_gemini(
                    item["file"],
                    item["start_line"],
                    item["code"],
                )
                try:
                    findings.extend(json.loads(response_json))
                except Exception:
                    continue
            deduped = deduplicate_findings(findings)
            return {"id": msg["id"], "result": {"content": deduped}}

        if name == "scan_snippet":
            filename = args["filename"]
            code = args["code"]
            start_line = int(args.get("start_line", 1))
            temp_skill = CodeSmellSkill(repo_path=os.getcwd())
            response_json = temp_skill.analyze_with_gemini(filename, start_line, code)
            try:
                content = json.loads(response_json)
            except Exception:
                content = []
            return {"id": msg["id"], "result": {"content": content}}

        return {
            "id": msg.get("id"),
            "error": {"code": -32601, "message": f"Unknown tool: {name}"},
        }

    def run(self) -> None:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue

            method = message.get("method")
            try:
                if method == "initialize":
                    response = self.handle_initialize(message)
                elif method == "tools/list":
                    response = self.handle_tools_list(message)
                elif method == "tools/call":
                    response = self.handle_tools_call(message)
                else:
                    response = {
                        "id": message.get("id"),
                        "error": {
                            "code": -32601,
                            "message": f"Unknown method {method}",
                        },
                    }
            except Exception as exc:
                response = {
                    "id": message.get("id"),
                    "error": {"code": -32000, "message": str(exc)},
                }

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            time.sleep(0.01)


def run_stdio_server() -> None:
    """Entry point used by :mod:`main` and ``python -m code_smell``."""
    try:
        MCPServer().run()
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": {"message": f"Fatal: {exc}"}}) + "\n")
        sys.stdout.flush()
        sys.exit(1)
