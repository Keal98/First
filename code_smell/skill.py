"""Semantic Kernel skill that implements the code smell tooling."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from semantic_kernel.functions.native_function_decorator import kernel_function

from .config import (
    CHROMA_DIR,
    CHUNK_LINES,
    CHUNK_OVERLAP,
    COLLECTION_PREFIX,
    DEFAULT_RETRIEVAL_QUERY,
)
from .files import list_code_files, make_line_chunks, read_text
from .google import embed_batch, make_generative_model
from .utils import repo_hash


class CodeSmellSkill:
    """Expose repository indexing and Gemini analysis as Semantic Kernel functions."""

    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(allow_reset=False),
        )
        self.collection_name = f"{COLLECTION_PREFIX}{repo_hash(self.repo_path)}"
        self.coll = self.client.get_or_create_collection(self.collection_name)

    @kernel_function(name="index_repo", description="Index repository code into Chroma with embeddings")
    def index_repo(self) -> str:
        if not os.path.isdir(self.repo_path):
            return f"Path not found or not a directory: {self.repo_path}"
        files = list_code_files(self.repo_path)
        if not files:
            return f"No supported code files under {self.repo_path}"

        # Reset the collection to avoid stale chunks.
        if self.coll.count() > 0:
            self.client.delete_collection(self.collection_name)
            self.coll = self.client.get_or_create_collection(self.collection_name)

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        for file_path in tqdm(files, desc="Indexing"):
            text = read_text(file_path)
            for start_line, chunk in make_line_chunks(text, CHUNK_LINES, CHUNK_OVERLAP):
                ids.append(f"{file_path}:{start_line}")
                docs.append(chunk)
                metas.append({"file": file_path, "start_line": start_line})
                if len(docs) >= 200:
                    embeddings = embed_batch(docs)
                    self.coll.add(
                        ids=ids,
                        documents=docs,
                        metadatas=metas,
                        embeddings=embeddings,
                    )
                    ids, docs, metas = [], [], []
        if docs:
            embeddings = embed_batch(docs)
            self.coll.add(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeddings,
            )
        return "Index build complete"

    @kernel_function(name="retrieve_chunks", description="Retrieve top-k likely-smelly chunks")
    def retrieve_chunks(self, top_k: int = 20) -> str:
        if self.coll.count() == 0:
            return json.dumps([])
        query_embedding = embed_batch([DEFAULT_RETRIEVAL_QUERY])[0]
        result = self.coll.query(
            query_embeddings=[query_embedding],
            n_results=max(1, int(top_k)),
        )
        items: List[Dict[str, Any]] = []
        if result and result.get("documents"):
            for document, metadata in zip(result["documents"][0], result["metadatas"][0]):
                items.append(
                    {
                        "file": metadata["file"],
                        "start_line": metadata["start_line"],
                        "code": document,
                    }
                )
        return json.dumps(items, ensure_ascii=False)

    @kernel_function(name="analyze_with_gemini", description="Analyze a chunk with Gemini and return JSON findings")
    def analyze_with_gemini(self, file: str, start_line: int, code: str) -> str:
        system_prompt = (
            "You are a strict secure-code auditor. Analyze the code and return ONLY a JSON array. "
            "Each item MUST have: file, line, type, severity (Low|Medium|High|Critical), "
            "description, suggested_fix. If no issues, return []"
        )
        payload = {
            "file": file,
            "start_line": start_line,
            "code": code,
            "checklist": [
                "SQL injection",
                "Hardcoded secrets",
                "Insecure crypto (MD5/SHA1/TripleDES)",
                "Command injection",
                "Unsafe deserialization",
                "Deprecated/vulnerable libs",
                "Missing input validation/sanitization",
            ],
        }
        model = make_generative_model()
        response = model.generate_content(
            [
                {"role": "user", "parts": system_prompt},
                {"role": "user", "parts": json.dumps(payload)},
            ]
        )
        text = (getattr(response, "text", "") or "").strip().strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
        try:
            findings = json.loads(text)
            if isinstance(findings, dict):
                findings = [findings]
        except Exception:
            findings = []

        normalised: List[Dict[str, Any]] = []
        for finding in findings:
            finding.setdefault("file", file)
            finding.setdefault("line", start_line)
            normalised.append(finding)
        return json.dumps(normalised, ensure_ascii=False)
