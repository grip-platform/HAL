#!/usr/bin/env python3
"""HAL Web Demo — lightweight HTTP server + interactive UI.

Single file. No dependencies. Serves the demo UI and proxies LLM requests.

Usage:
    python3 ~/.hal/lib/hal/web_demo.py           # Start on port 3847
    python3 ~/.hal/lib/hal/web_demo.py 8080     # Custom port

Then open http://localhost:3847 in your browser.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

_hal_path = os.environ.get("HAL_HOME", os.path.expanduser("~/.hal"))
sys.path.insert(0, _hal_path)

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3847

# Provider configs
PROVIDERS = {
    "groq": {
        "name": "Groq (Llama 3.3 70B)",
        "type": "cloud",
        "speed": "750 tok/s",
        "privacy": "Cloud API",
    },
    "gemini": {
        "name": "Gemini 2.5 Flash",
        "type": "cloud",
        "speed": "Fast",
        "privacy": "Cloud API",
    },
    "ollama": {
        "name": "Ollama (Local)",
        "type": "local",
        "speed": "7-12 tok/s",
        "privacy": "FULLY PRIVATE",
    },
}


def _query_groq(prompt: str) -> dict:
    from lib.providers.groq import GroqProvider
    start = time.monotonic()
    r = GroqProvider().generate(prompt, max_tokens=150)
    ms = (time.monotonic() - start) * 1000
    from lib.hal.cost_tracker import calculate_cost
    cost = calculate_cost(
        "llama-3.3-70b-versatile",
        input_tokens=r.usage.get("prompt_tokens", 0),
        output_tokens=r.usage.get("completion_tokens", 0),
    )
    return {"text": r.text, "model": r.model, "latency_ms": round(ms),
            "cost_usd": cost, "tokens": r.usage, "provider": "groq"}


def _query_gemini(prompt: str) -> dict:
    from lib.providers.gemini import GeminiProvider
    start = time.monotonic()
    r = GeminiProvider().generate(prompt, max_tokens=512)
    ms = (time.monotonic() - start) * 1000
    return {"text": r.text, "model": r.model, "latency_ms": round(ms),
            "cost_usd": 0.0, "tokens": r.usage, "provider": "gemini"}


def _query_ollama(prompt: str) -> dict:
    models = _get_ollama_models()
    model = models[0] if models else "mistral:7b"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    ms = (time.monotonic() - start) * 1000
    return {
        "text": data["choices"][0]["message"]["content"],
        "model": model, "latency_ms": round(ms),
        "cost_usd": 0.0, "tokens": data.get("usage", {}),
        "provider": "ollama", "private": True,
    }


def _get_ollama_models() -> list:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _get_status() -> dict:
    from lib.hal.skill_loader import SkillLoader
    from lib.hal.agent_runner import AgentRunner
    from lib.hal.memory_engine import MemoryEngine
    loader = SkillLoader()
    skills = loader.discover()
    modes = loader.list_modes()
    agents = AgentRunner().list_agents()
    memory = MemoryEngine()
    entries = memory.list_entries()
    ollama_models = _get_ollama_models()
    return {
        "skills": len(skills), "modes": len(modes), "agents": len(agents),
        "memory_entries": len(entries), "local_models": ollama_models,
        "providers": PROVIDERS,
    }


QUERY_DISPATCH = {"groq": _query_groq, "gemini": _query_gemini, "ollama": _query_ollama}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == "/api/status":
            self._json_response(_get_status())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/query":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            provider = body.get("provider", "groq")
            prompt = body.get("prompt", "")
            fn = QUERY_DISPATCH.get(provider)
            if not fn:
                self._json_response({"error": f"Unknown provider: {provider}"}, 400)
                return
            try:
                result = fn(prompt)
                self._json_response(result)
            except Exception as e:
                self._json_response({"error": str(e)[:300]}, 500)
        else:
            self.send_error(404)

    def _json_response(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        pass  # Suppress request logging


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HAL — Knowledge Engine</title>
<style>
  :root { --bg: #0a0a0f; --card: #12121a; --border: #1e1e2e; --text: #e0e0e0;
          --dim: #666; --accent: #e94560; --green: #00d26a; --blue: #4fc3f7; --orange: #ff9800; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: var(--bg); color: var(--text);
         min-height: 100vh; padding: 2rem; line-height: 1.6; }
  h1 { font-size: 2rem; color: var(--accent); margin-bottom: 0.25rem; letter-spacing: 2px; }
  .subtitle { color: var(--dim); margin-bottom: 2rem; font-size: 0.85rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; }
  .card h2 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; color: var(--dim);
             margin-bottom: 1rem; }
  .stat { font-size: 2rem; font-weight: 700; color: var(--accent); }
  .stat-label { font-size: 0.7rem; color: var(--dim); text-transform: uppercase; letter-spacing: 1px; }
  .stat-row { display: flex; gap: 2rem; margin-bottom: 0.5rem; }
  .provider-btn { background: var(--card); border: 2px solid var(--border); border-radius: 6px;
                  padding: 0.75rem 1.25rem; color: var(--text); cursor: pointer; font-family: inherit;
                  font-size: 0.85rem; transition: all 0.2s; }
  .provider-btn:hover { border-color: var(--accent); }
  .provider-btn.active { border-color: var(--accent); background: rgba(233,69,96,0.1); }
  .provider-btn.local { border-color: var(--green); }
  .provider-btn.local.active { background: rgba(0,210,106,0.1); }
  .provider-row { display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
  .badge { font-size: 0.65rem; padding: 2px 6px; border-radius: 3px; font-weight: 600; }
  .badge-cloud { background: rgba(79,195,247,0.15); color: var(--blue); }
  .badge-local { background: rgba(0,210,106,0.15); color: var(--green); }
  .badge-private { background: rgba(233,69,96,0.15); color: var(--accent); }
  textarea { width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
             color: var(--text); padding: 0.75rem; font-family: inherit; font-size: 0.85rem;
             resize: vertical; min-height: 80px; }
  textarea:focus { outline: none; border-color: var(--accent); }
  .send-btn { background: var(--accent); border: none; color: white; padding: 0.6rem 1.5rem;
              border-radius: 6px; cursor: pointer; font-family: inherit; font-weight: 600;
              font-size: 0.85rem; margin-top: 0.75rem; transition: opacity 0.2s; }
  .send-btn:hover { opacity: 0.85; }
  .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .response-box { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
                  padding: 1rem; margin-top: 1rem; min-height: 60px; max-height: 300px;
                  overflow-y: auto; white-space: pre-wrap; word-wrap: break-word;
                  font-size: 0.85rem; }
  .meta { display: flex; gap: 1.5rem; margin-top: 0.5rem; font-size: 0.7rem; color: var(--dim); }
  .meta .cost { color: var(--green); font-weight: 600; }
  .meta .private-tag { color: var(--accent); font-weight: 700; text-transform: uppercase; }
  .models-list { font-size: 0.75rem; color: var(--dim); }
  .cost-bar { height: 6px; border-radius: 3px; margin: 2px 0; transition: width 0.5s; }
  .cost-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; font-size: 0.75rem; }
  .cost-label { width: 140px; color: var(--dim); }
  .cost-value { width: 60px; text-align: right; }
  .full-width { grid-column: 1 / -1; }
  .sentinel-chain { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; font-size: 0.8rem; }
  .sentinel-node { padding: 0.3rem 0.6rem; border-radius: 4px; border: 1px solid var(--border); }
  .sentinel-node.active { border-color: var(--green); color: var(--green); }
  .sentinel-node.limited { border-color: var(--accent); color: var(--accent); text-decoration: line-through; }
  .sentinel-arrow { color: var(--dim); }
  .session-cost { position: fixed; top: 1rem; right: 1rem; background: var(--card); border: 1px solid var(--border);
                  border-radius: 6px; padding: 0.5rem 1rem; font-size: 0.75rem; }
  @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } body { padding: 1rem; } }
</style>
</head>
<body>

<h1>HAL</h1>
<p class="subtitle">Harness Abstraction Layer — Any model. Any provider. Any domain. Zero limits.</p>

<div class="session-cost">Session: $<span id="session-cost">0.000000</span></div>

<div class="grid">
  <div class="card">
    <h2>System Status</h2>
    <div class="stat-row">
      <div><div class="stat" id="skill-count">—</div><div class="stat-label">Skills</div></div>
      <div><div class="stat" id="mode-count">—</div><div class="stat-label">Modes</div></div>
      <div><div class="stat" id="agent-count">—</div><div class="stat-label">Agents</div></div>
      <div><div class="stat" id="memory-count">—</div><div class="stat-label">Memories</div></div>
    </div>
    <div class="models-list" id="local-models"></div>
  </div>

  <div class="card">
    <h2>Cost Comparison (10K in + 5K out)</h2>
    <div id="cost-bars"></div>
  </div>

  <div class="card full-width">
    <h2>Limit Sentinel — Zero Limits</h2>
    <div class="sentinel-chain" id="sentinel-chain"></div>
    <p style="font-size:0.7rem;color:var(--dim);margin-top:0.5rem">
      When any provider hits its limit, HAL auto-routes to the next. Local models have zero external limits.</p>
  </div>

  <div class="card full-width">
    <h2>Live Query</h2>
    <div class="provider-row" id="provider-btns"></div>
    <textarea id="prompt" placeholder="Ask anything...">In one sentence, why is private on-premises AI essential for law firms handling confidential client data?</textarea>
    <button class="send-btn" id="send-btn" onclick="sendQuery()">Send Query</button>
    <div class="response-box" id="response">Responses appear here...</div>
    <div class="meta" id="meta"></div>
  </div>
</div>

<script>
let activeProvider = 'groq';
let sessionCost = 0;
const costs = [
  { name: 'Claude Opus 4.7', cost: 0.175, color: '#e94560' },
  { name: 'Claude Sonnet 4.6', cost: 0.105, color: '#ff6b6b' },
  { name: 'Groq Llama 70B', cost: 0.0099, color: '#4fc3f7' },
  { name: 'Gemini Flash', cost: 0.0045, color: '#00d26a' },
  { name: 'Local (Ollama)', cost: 0, color: '#ff9800' },
];
const sentinelChain = [
  { name: 'Anthropic', status: 'active' },
  { name: 'Groq', status: 'active' },
  { name: 'Gemini', status: 'active' },
  { name: 'OpenRouter', status: 'active' },
  { name: 'Ollama (local)', status: 'active' },
];

async function init() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('skill-count').textContent = d.skills;
    document.getElementById('mode-count').textContent = d.modes;
    document.getElementById('agent-count').textContent = d.agents;
    document.getElementById('memory-count').textContent = d.memory_entries;
    if (d.local_models.length > 0) {
      document.getElementById('local-models').textContent = 'Local models: ' + d.local_models.join(', ');
    }
    renderProviderBtns(d.providers, d.local_models.length > 0);
  } catch(e) { console.error(e); }
  renderCostBars();
  renderSentinel();
}

function renderProviderBtns(providers, hasLocal) {
  const el = document.getElementById('provider-btns');
  el.innerHTML = '';
  for (const [key, p] of Object.entries(providers)) {
    if (key === 'ollama' && !hasLocal) continue;
    const isLocal = p.type === 'local';
    const btn = document.createElement('button');
    btn.className = 'provider-btn' + (isLocal ? ' local' : '') + (key === activeProvider ? ' active' : '');
    btn.innerHTML = p.name + ' <span class="badge ' + (isLocal ? 'badge-private' : 'badge-cloud') + '">'
      + (isLocal ? 'PRIVATE' : 'CLOUD') + '</span>';
    btn.onclick = () => { activeProvider = key; document.querySelectorAll('.provider-btn').forEach(b => b.classList.remove('active')); btn.classList.add('active'); };
    el.appendChild(btn);
  }
}

function renderCostBars() {
  const el = document.getElementById('cost-bars');
  const max = costs[0].cost;
  el.innerHTML = costs.map(c => `<div class="cost-row">
    <span class="cost-label">${c.name}</span>
    <div style="flex:1;background:var(--bg);border-radius:3px;overflow:hidden">
      <div class="cost-bar" style="width:${Math.max(1, (c.cost/max)*100)}%;background:${c.color}"></div>
    </div>
    <span class="cost-value">$${c.cost.toFixed(4)}</span>
  </div>`).join('');
}

function renderSentinel() {
  const el = document.getElementById('sentinel-chain');
  el.innerHTML = sentinelChain.map((n, i) =>
    `<span class="sentinel-node ${n.status}">${n.name}</span>` +
    (i < sentinelChain.length - 1 ? '<span class="sentinel-arrow">&rarr;</span>' : '')
  ).join('');
}

async function sendQuery() {
  const btn = document.getElementById('send-btn');
  const prompt = document.getElementById('prompt').value;
  const resp = document.getElementById('response');
  const meta = document.getElementById('meta');
  if (!prompt.trim()) return;

  btn.disabled = true; btn.textContent = 'Querying ' + activeProvider + '...';
  resp.textContent = '...'; meta.innerHTML = '';

  try {
    const r = await fetch('/api/query', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ provider: activeProvider, prompt }),
    });
    const d = await r.json();
    if (d.error) { resp.textContent = 'Error: ' + d.error; return; }
    resp.textContent = d.text;
    sessionCost += d.cost_usd || 0;
    document.getElementById('session-cost').textContent = sessionCost.toFixed(6);
    const parts = [
      `<span>${d.latency_ms}ms</span>`,
      `<span>${d.model}</span>`,
      `<span class="cost">$${(d.cost_usd||0).toFixed(6)}</span>`,
    ];
    if (d.private) parts.push('<span class="private-tag">zero data exposure</span>');
    if (d.tokens) parts.push(`<span>${d.tokens.prompt_tokens||0} in / ${d.tokens.completion_tokens||0} out</span>`);
    meta.innerHTML = parts.join('');
  } catch(e) { resp.textContent = 'Error: ' + e.message; }
  finally { btn.disabled = false; btn.textContent = 'Send Query'; }
}

document.getElementById('prompt').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) sendQuery();
});

init();
</script>
</body>
</html>"""

if __name__ == "__main__":
    print(f"\n  HAL Web Demo — http://localhost:{PORT}\n")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
