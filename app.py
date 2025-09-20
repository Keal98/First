# app.py
"""
FastAPI UI for OpenAI Agent + MCP Code Smell server.

Endpoints:
  POST /run/scan_repo   { "path": "...", "top_k": 20 }
  POST /run/scan_snippet { "filename": "x.py", "code": "...", "start_line": 1 }

Prereqs:
  pip install fastapi uvicorn openai openai-agents pydantic==2.* python-multipart

Env:
  OPENAI_API_KEY=...
  GOOGLE_API_KEY=...      # used by the MCP server (Gemini)
"""

import os
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from openai import OpenAI
from openai_agents import Agent, run, tools

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
if not GOOGLE_API_KEY:
    # MCP server (Gemini) needs this. We just validate early to avoid 500 later.
    raise RuntimeError("GOOGLE_API_KEY is not set (required by MCP server)")

# --- Build the Agent wired to your MCP server ---
client = OpenAI()

# Launches your MCP server via stdio. The entry point lives in ``main.py``.
MCP_COMMAND = ["python", "main.py"]

smell_tool = tools.mcp_server(
    name="code_smell_mcp",
    transport="stdio",
    command=MCP_COMMAND,
)

agent = Agent(
    model="gpt-4o-mini",
    name="Security Smell Agent",
    instructions=(
        "You detect security code smells. "
        "Always call the MCP tools before answering. "
        "Return a compact JSON array of findings with fields: "
        "[file, line, type, severity, description, suggested_fix]."
    ),
    tools=[smell_tool],
    temperature=0
)

app = FastAPI(title="Code Smell Agent UI")

# CORS for local testing / simple static pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------- Pydantic models ----------
class ScanRepoBody(BaseModel):
    path: str
    top_k: int = 20

class ScanSnippetBody(BaseModel):
    filename: str
    code: str
    start_line: int = 1

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def home():
    # Serve the UI from here (see the HTML below)
    return INDEX_HTML

@app.post("/run/scan_repo")
def run_scan_repo(body: ScanRepoBody):
    # We instruct the agent what tool to call and how
    prompt = (
        f'Call tool scan_repo with {{"path":"{body.path}","top_k":{body.top_k}}} '
        f'and return ONLY the JSON findings array.'
    )
    result = run(agent, prompt)
    # If the agent returned text, try to parse JSON; else just forward text
    text = (result.output_text or "").strip()
    # Loosen: accept plain JSON or quoted/markdown-fenced JSON
    import json
    raw = text.strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        data = json.loads(raw)
        return JSONResponse(data)
    except Exception:
        return JSONResponse({"raw": text})

@app.post("/run/scan_snippet")
def run_scan_snippet(body: ScanSnippetBody):
    # Ask the agent to call scan_snippet with the given args
    import json
    args = {"filename": body.filename, "code": body.code, "start_line": body.start_line}
    prompt = (
        "Call tool scan_snippet with " + json.dumps(args) +
        " and return ONLY the JSON findings array."
    )
    result = run(agent, prompt)
    text = (result.output_text or "").strip()
    raw = text.strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    import json as _json
    try:
        data = _json.loads(raw)
        return JSONResponse(data)
    except Exception:
        return JSONResponse({"raw": text})
    

# ---------- Inline HTML (served at GET /) ----------
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Code Smell Agent UI</title>
<style>
  :root { color-scheme: dark light; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }
  h1 { margin-bottom: .25rem; }
  .card { border: 1px solid #ddd; border-radius: 14px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 10px rgba(0,0,0,.04); }
  label { display: block; font-weight: 600; margin-top: .5rem; }
  input[type=text], textarea, input[type=number] {
      width: 100%; padding: .6rem; border-radius: 10px; border: 1px solid #bbb; font-family: inherit;
  }
  button { padding: .6rem 1rem; border-radius: 10px; border: 1px solid #444; cursor: pointer; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  pre { background: #111; color: #eee; padding: 1rem; border-radius: 12px; overflow:auto; }
  .pill { display:inline-block; border:1px solid #aaa; padding: .25rem .5rem; border-radius: 999px; font-size: .8rem; margin-right: .5rem;}
</style>
</head>
<body>
  <h1>🛡️ Code Smell Agent</h1>
  <div style="margin-bottom:1rem;color:#666;">OpenAI Agent + MCP Server + Gemini + Chroma</div>

  <div class="card">
    <div class="pill">Scan Repo</div>
    <form id="repoForm">
      <label>Repo Path</label>
      <input type="text" id="repoPath" placeholder="./src" value="./src" />
      <label>Top K Chunks</label>
      <input type="number" id="topK" value="20" min="1" max="200" />
      <div style="margin-top: .75rem;">
        <button type="submit">Run Scan</button>
      </div>
    </form>
  </div>

  <div class="card">
    <div class="pill">Scan Snippet</div>
    <form id="snippetForm">
      <label>Filename (for reporting)</label>
      <input type="text" id="fileName" placeholder="auth.py" />
      <label>Start line</label>
      <input type="number" id="startLine" value="1" />
      <label>Code</label>
      <textarea id="code" rows="8" placeholder="Paste code here..."></textarea>
      <div style="margin-top: .75rem;">
        <button type="submit">Analyze Snippet</button>
      </div>
    </form>
  </div>

  <div class="card">
    <div class="pill">Results</div>
    <pre id="out">Waiting…</pre>
  </div>

<script>
const out = document.getElementById("out");

function show(obj){
  out.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}

document.getElementById("repoForm").addEventListener("submit", async (e)=>{
  e.preventDefault();
  show("Scanning repo…");
  const path = document.getElementById("repoPath").value.trim();
  const topK = parseInt(document.getElementById("topK").value, 10) || 20;
  const res = await fetch("/run/scan_repo", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ path, top_k: topK })
  });
  const data = await res.json();
  show(data);
});

document.getElementById("snippetForm").addEventListener("submit", async (e)=>{
  e.preventDefault();
  show("Analyzing snippet…");
  const filename = document.getElementById("fileName").value.trim() || "snippet.txt";
  const start_line = parseInt(document.getElementById("startLine").value,10) || 1;
  const code = document.getElementById("code").value;
  const res = await fetch("/run/scan_snippet", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ filename, code, start_line })
  });
  const data = await res.json();
  show(data);
});
</script>
</body>
</html>
"""
