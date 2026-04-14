"""
nexari.preview.server
──────────────────────
Minimal FastAPI server + single-page UI for live model preview.
Launched automatically after deployment.
"""

from __future__ import annotations

import json
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from nexari.config import HF_TOKEN, PREVIEW_HOST, PREVIEW_PORT

app = FastAPI(title="Nexari Preview")

# State injected at startup
_endpoint_url: str = ""
_metadata: dict = {}


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(req: PredictRequest):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            _endpoint_url,
            json={"inputs": req.text},
            headers=headers,
        )
    if response.status_code != 200:
        return JSONResponse({"error": response.text}, status_code=response.status_code)
    return response.json()


@app.get("/", response_class=HTMLResponse)
async def index():
    domain = _metadata.get("domain", "model")
    labels = list(_metadata.get("label2id", {}).keys())
    labels_html = "".join(f"<span class='label'>{l}</span>" for l in labels)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nexari Preview — {domain}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f0f0f; color: #e0e0e0; min-height: 100vh;
            display: flex; align-items: center; justify-content: center; }}
    .container {{ max-width: 640px; width: 100%; padding: 2rem; }}
    .header {{ margin-bottom: 2rem; }}
    .header h1 {{ font-size: 1.5rem; font-weight: 600; color: #fff; }}
    .header h1 span {{ color: #7c6aff; }}
    .header p {{ color: #666; font-size: 0.9rem; margin-top: 0.25rem; }}
    .labels {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0; }}
    .label {{ background: #1a1a2e; color: #7c6aff; border: 1px solid #7c6aff33;
              padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.8rem; }}
    textarea {{ width: 100%; background: #1a1a1a; border: 1px solid #333;
                color: #e0e0e0; padding: 1rem; border-radius: 8px; font-size: 0.95rem;
                resize: vertical; min-height: 120px; outline: none; }}
    textarea:focus {{ border-color: #7c6aff; }}
    button {{ margin-top: 1rem; width: 100%; background: #7c6aff; color: #fff;
              border: none; padding: 0.75rem; border-radius: 8px; font-size: 1rem;
              cursor: pointer; font-weight: 500; transition: background 0.2s; }}
    button:hover {{ background: #6a58e0; }}
    button:disabled {{ background: #333; color: #666; cursor: not-allowed; }}
    .result {{ margin-top: 1.5rem; background: #1a1a1a; border: 1px solid #333;
               border-radius: 8px; padding: 1rem; display: none; }}
    .result.show {{ display: block; }}
    .result h3 {{ font-size: 0.8rem; color: #666; text-transform: uppercase;
                  letter-spacing: 0.05em; margin-bottom: 0.75rem; }}
    .prediction {{ display: flex; align-items: center; justify-content: space-between;
                   padding: 0.5rem 0; border-bottom: 1px solid #222; }}
    .prediction:last-child {{ border-bottom: none; }}
    .pred-label {{ font-weight: 500; }}
    .pred-score {{ color: #7c6aff; font-size: 0.9rem; }}
    .bar {{ height: 4px; background: #222; border-radius: 2px; margin-top: 0.25rem; }}
    .bar-fill {{ height: 100%; background: #7c6aff; border-radius: 2px; transition: width 0.3s; }}
    .error {{ color: #ff6b6b; font-size: 0.9rem; }}
    .powered {{ margin-top: 2rem; text-align: center; color: #444; font-size: 0.8rem; }}
    .powered a {{ color: #7c6aff; text-decoration: none; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1><span>nexari</span> preview</h1>
      <p>Live inference · {domain} classifier</p>
      <div class="labels">{labels_html}</div>
    </div>
    <textarea id="input" placeholder="Type something to classify..."></textarea>
    <button id="btn" onclick="predict()">Classify</button>
    <div class="result" id="result">
      <h3>Predictions</h3>
      <div id="predictions"></div>
    </div>
    <div class="powered">Powered by <a href="https://github.com/jdavidaguil/nexari">nexari</a></div>
  </div>
  <script>
    async function predict() {{
      const text = document.getElementById('input').value.trim();
      if (!text) return;
      const btn = document.getElementById('btn');
      btn.disabled = true; btn.textContent = 'Classifying...';
      const result = document.getElementById('result');
      const preds = document.getElementById('predictions');
      try {{
        const res = await fetch('/predict', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{text}})
        }});
        const data = await res.json();
        if (data.error) {{ preds.innerHTML = `<p class="error">${{data.error}}</p>`; }}
        else {{
          const items = Array.isArray(data) ? (Array.isArray(data[0]) ? data[0] : data) : [data];
          const sorted = items.sort((a,b) => b.score - a.score);
          preds.innerHTML = sorted.map(p => `
            <div class="prediction">
              <div>
                <div class="pred-label">${{p.label}}</div>
                <div class="bar"><div class="bar-fill" style="width:${{(p.score*100).toFixed(1)}}%"></div></div>
              </div>
              <div class="pred-score">${{(p.score*100).toFixed(1)}}%</div>
            </div>`).join('');
        }}
        result.classList.add('show');
      }} catch(e) {{ preds.innerHTML = `<p class="error">${{e.message}}</p>`; result.classList.add('show'); }}
      btn.disabled = false; btn.textContent = 'Classify';
    }}
    document.getElementById('input').addEventListener('keydown', e => {{
      if (e.key === 'Enter' && e.metaKey) predict();
    }});
  </script>
</body>
</html>"""


def start_server(url: str, metadata: dict):
    global _endpoint_url, _metadata
    _endpoint_url = url
    _metadata = metadata
    import uvicorn
    uvicorn.run(app, host=PREVIEW_HOST, port=PREVIEW_PORT, log_level="warning")