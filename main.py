#!/usr/bin/env python3
"""
MCP Server (stdio) for Code Smell Detection using:
- Semantic Kernel (orchestration via native functions)
- Chroma (vector storage)
- Gemini (embeddings + analysis)

Tools exposed:
  1) index_repo(path: str) -> str
  2) scan_repo(path: str, top_k: int = 20) -> List[Finding]
  3) scan_snippet(filename: str, code: str, start_line: int = 1) -> List[Finding]

Finding schema:
  {
    "file": str,
    "line": int,
    "type": str,
    "severity": "Low"|"Medium"|"High"|"Critical",
    "description": str,
    "suggested_fix": str
  }

Env:
  GOOGLE_API_KEY (required)
  CHUNK_LINES (default 80)
  CHUNK_OVERLAP (default 10)
  CHROMA_DIR (default ".chroma_code_smell")

Run (standalone debug): python mcp_server_sk_code_smell.py
Wire to an Agent (example):
  tools.mcp_server(
      name="code_smell_mcp",
      transport="stdio",
      command=["python", "mcp_server_sk_code_smell.py"]
  )
"""

import os, sys, json, glob, time, threading
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# --- Dependencies ---
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import google.generativeai as genai

# Semantic Kernel
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions.native_function_decorator import kernel_function

# ---------- Config ----------
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma_code_smell")
COLLECTION_PREFIX = "code_smell_chunks__"  # one collection per repo path hash
CHUNK_LINES = int(os.getenv("CHUNK_LINES", "80"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "10"))
SUPPORTED_EXTS = {".py", ".js", ".ts", ".cs", ".java", ".go", ".rb", ".php", ".cpp", ".c", ".rs"}
EMBED_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-1.5-pro"

# ---------- Helpers ----------
def _ensure_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    genai.configure(api_key=api_key)

def _read_text(path: str) -> str:
    return open(path, "r", encoding="utf-8", errors="ignore").read()

def _list_code_files(root: str) -> List[str]:
    files = []
    for ext in SUPPORTED_EXTS:
        files += glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True)
    return sorted(set(files))

def _make_line_chunks(text: str, chunk_lines: int, overlap: int) -> List[Tuple[int, str]]:
    lines = text.splitlines()
    out, i = [], 0
    step = max(1, (chunk_lines - overlap) if chunk_lines > overlap else chunk_lines)
    while i < len(lines):
        piece = lines[i:i+chunk_lines]
        if not piece:
            break
        out.append((i + 1, "\n".join(piece)))
        i += step
    return out

def _embed_batch(texts: List[str]) -> List[List[float]]:
    # Batched when possible; fall back to per-item
    try:
        r = genai.embed_content(model=EMBED_MODEL, content=texts, task_type="retrieval_document")
        embs = r.get("embedding")
        if isinstance(embs, list):
            if embs and isinstance(embs[0], dict) and "values" in embs[0]:
                return [e["values"] for e in embs]
            if embs and isinstance(embs[0], list):
                return embs
        if isinstance(embs, dict) and "values" in embs:
            return [embs["values"]]
    except Exception:
        pass
    out = []
    for t in texts:
        rr = genai.embed_content(model=EMBED_MODEL, content=t, task_type="retrieval_document")
        out.append(rr["embedding"]["values"])
    return out

def _repo_hash(path: str) -> str:
    # simple deterministic hash for collection naming (avoid long names)
    import hashlib
    return hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:16]

# ---------- Semantic Kernel Skill ----------
class CodeSmellSkill:
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
        self.collection_name = f"{COLLECTION_PREFIX}{_repo_hash(self.repo_path)}"
        self.coll = self.client.get_or_create_collection(self.collection_name)

    @kernel_function(name="index_repo", description="Index repository code into Chroma with embeddings")
    def index_repo(self) -> str:
        if not os.path.isdir(self.repo_path):
            return f"Path not found or not a directory: {self.repo_path}"
        files = _list_code_files(self.repo_path)
        if not files:
            return f"No supported code files under {self.repo_path}"

        # Clear existing (fresh index)
        if self.coll.count() > 0:
            self.client.delete_collection(self.collection_name)
            self.coll = self.client.get_or_create_collection(self.collection_name)

        ids, docs, metas = [], [], []
        for fp in tqdm(files, desc="Indexing"):
            text = _read_text(fp)
            for (start, chunk) in _make_line_chunks(text, CHUNK_LINES, CHUNK_OVERLAP):
                ids.append(f"{fp}:{start}")
                docs.append(chunk)
                metas.append({"file": fp, "start_line": start})
                if len(docs) >= 200:
                    embs = _embed_batch(docs)
                    self.coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                    ids, docs, metas = [], [], []
        if docs:
            embs = _embed_batch(docs)
            self.coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        return "Index build complete"

    @kernel_function(name="retrieve_chunks", description="Retrieve top-k likely-smelly chunks")
    def retrieve_chunks(self, top_k: int = 20) -> str:
        if self.coll.count() == 0:
            return json.dumps([])
        query = ("potential security smells: sql injection, hardcoded secrets, insecure cryptography, "
                 "command injection, unsafe deserialization, deprecated libs, missing validation")
        q_emb = _embed_batch([query])[0]
        res = self.coll.query(query_embeddings=[q_emb], n_results=max(1, int(top_k)))
        items = []
        if res and res.get("documents"):
            for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                items.append({"file": meta["file"], "start_line": meta["start_line"], "code": doc})
        return json.dumps(items, ensure_ascii=False)

    @kernel_function(name="analyze_with_gemini", description="Analyze a chunk with Gemini and return JSON findings")
    def analyze_with_gemini(self, file: str, start_line: int, code: str) -> str:
        sys_prompt = (
            "You are a strict secure-code auditor. Analyze the code and return ONLY a JSON array. "
            "Each item MUST have: file, line, type, severity (Low|Medium|High|Critical), "
            "description, suggested_fix. If no issues, return []"
        )
        payload = {
            "file": file, "start_line": start_line, "code": code,
            "checklist": [
                "SQL injection", "Hardcoded secrets", "Insecure crypto (MD5/SHA1/TripleDES)",
                "Command injection", "Unsafe deserialization", "Deprecated/vulnerable libs",
                "Missing input validation/sanitization"
            ]
        }
        model = genai.GenerativeModel(LLM_MODEL)
        resp = model.generate_content([
            {"role": "user", "parts": sys_prompt},
            {"role": "user", "parts": json.dumps(payload)}
        ])
        text = (getattr(resp, "text", "") or "").strip().strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
        try:
            arr = json.loads(text)
            if isinstance(arr, dict):
                arr = [arr]
        except Exception:
            arr = []
        # Normalize defaults
        out = []
        for f in arr:
            f.setdefault("file", file)
            f.setdefault("line", start_line)
            out.append(f)
        return json.dumps(out, ensure_ascii=False)

# ---------- MCP Protocol (very small, stdio JSON-RPC) ----------
class MCPServer:
    def __init__(self):
        _ensure_api_key()
        self.kernel = Kernel()
        self.skills_cache: Dict[str, CodeSmellSkill] = {}  # key = abs(repo_path)

    # --- Tool registry presented to clients ---
    def _tools_list(self):
        return {
            "tools": [
                {
                    "name": "index_repo",
                    "description": "Build/refresh index for a repo",
                    "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                },
                {
                    "name": "scan_repo",
                    "description": "Retrieve suspicious chunks and analyze them with Gemini",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "top_k": {"type": "integer", "default": 20}},
                        "required": ["path"]
                    }
                },
                {
                    "name": "scan_snippet",
                    "description": "Analyze a single in-memory snippet",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "code": {"type": "string"},
                            "start_line": {"type": "integer", "default": 1}
                        },
                        "required": ["filename", "code"]
                    }
                }
            ]
        }

    def _get_skill(self, repo_path: str) -> CodeSmellSkill:
        key = os.path.abspath(repo_path)
        skill = self.skills_cache.get(key)
        if skill is None:
            skill = CodeSmellSkill(key)
            self.kernel.add_plugin(skill, f"smell::{_repo_hash(key)}")
            self.skills_cache[key] = skill
        return skill

    # --- MCP Handlers ---
    def handle_initialize(self, msg):
        return {"id": msg["id"], "result": {"capabilities": {"tools": {}}}}

    def handle_tools_list(self, msg):
        return {"id": msg["id"], "result": self._tools_list()}

    def handle_tools_call(self, msg):
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
            # ensure index exists
            if skill.coll.count() == 0:
                skill.index_repo()
            retrieved = json.loads(skill.retrieve_chunks(top_k))
            findings: List[Dict[str, Any]] = []
            for item in tqdm(retrieved, desc="Analyzing"):
                res_json = skill.analyze_with_gemini(item["file"], item["start_line"], item["code"])
                try:
                    findings.extend(json.loads(res_json))
                except Exception:
                    pass
            # de-dup
            seen = set(); dedup = []
            for f in findings:
                k = (f.get("file"), f.get("line"), f.get("type"))
                if k not in seen:
                    seen.add(k); dedup.append(f)
            return {"id": msg["id"], "result": {"content": dedup}}

        if name == "scan_snippet":
            filename = args["filename"]
            code = args["code"]
            start_line = int(args.get("start_line", 1))
            # use a temporary skill context for snippet-only analysis (no index)
            # direct call to analyzer:
            temp_skill = CodeSmellSkill(repo_path=os.getcwd())  # dummy for constructor deps
            res_json = temp_skill.analyze_with_gemini(filename, start_line, code)
            try:
                content = json.loads(res_json)
            except Exception:
                content = []
            return {"id": msg["id"], "result": {"content": content}}

        # Unknown tool
        return {"id": msg["id"], "error": {"code": -32601, "message": f"Unknown tool: {name}"}}

    # --- Run loop (stdio) ---
    def run(self):
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            method = msg.get("method")
            try:
                if method == "initialize":
                    resp = self.handle_initialize(msg)
                elif method == "tools/list":
                    resp = self.handle_tools_list(msg)
                elif method == "tools/call":
                    resp = self.handle_tools_call(msg)
                else:
                    resp = {"id": msg.get("id"), "error": {"code": -32601, "message": f"Unknown method {method}"}}
            except Exception as e:
                resp = {"id": msg.get("id"), "error": {"code": -32000, "message": str(e)}}

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            time.sleep(0.01)

# ---------- Entry ----------
if __name__ == "__main__":
    try:
        _ensure_api_key()
        MCPServer().run()
    except Exception as e:
        sys.stdout.write(json.dumps({"error": {"message": f"Fatal: {e}"}}) + "\n")
        sys.stdout.flush()
        sys.exit(1)
