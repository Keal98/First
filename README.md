# Code Smell MCP Server

This repository contains a MCP server that exposes
code-smell detection tools backed by Semantic Kernel orchestration, Chroma for
vector storage, and Google Gemini for embeddings and LLM-powered analysis.

## Features

- **Repository indexing** via Chroma with Gemini embeddings to enable fast
  retrieval of code chunks for inspection.
- **Automated smell detection** across a repository, returning structured
  findings with file, line, type, severity, description, and suggested fix.
- **Snippet scanning** for ad-hoc analysis without indexing an entire project.
- **Semantic Kernel integration** so the MCP server can be wired directly into
  agents that speak the MCP stdio protocol.

## Dependencies

The server depends on the following Python packages:

- `chromadb`
- `tqdm`
- `google-generativeai`
- `semantic-kernel`

Install them with:

```bash
pip install chromadb tqdm google-generativeai semantic-kernel
```

## Environment Variables

- `GOOGLE_API_KEY` (required): API key used to access Google Gemini models.
- `CHUNK_LINES` (optional, default `80`): Number of lines per indexed chunk.
- `CHUNK_OVERLAP` (optional, default `10`): Overlap between successive chunks.
- `CHROMA_DIR` (optional, default `.chroma_code_smell`): Directory for Chroma's
  persistent storage.

## Usage

Run the server directly for local debugging:

```bash
python main.py
```

On startup the server validates `GOOGLE_API_KEY` and then listens for MCP stdio
messages. Three tools are available through the MCP interface:

1. `index_repo(path: str) -> str`
2. `scan_repo(path: str, top_k: int = 20) -> List[Finding]`
3. `scan_snippet(filename: str, code: str, start_line: int = 1) -> List[Finding]`

A `Finding` includes the following fields:

```json
{
  "file": "path/to/file.py",
  "line": 42,
  "type": "SmellType",
  "severity": "Low" | "Medium" | "High" | "Critical",
  "description": "What was detected",
  "suggested_fix": "How to remediate"
}
```

To connect the server to an MCP-compatible agent, configure the tool like:

```python
tools.mcp_server(
    name="code_smell_mcp",
    transport="stdio",
    command=["python", "main.py"],
)
```

This will allow the agent to invoke the indexing and scanning tools to surface
code smells in repositories or ad-hoc snippets.
