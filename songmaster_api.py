#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
songmaster_api.py – FastAPI service for Songmaster + private tools

Features:
- Guest UI (Pornhub-inspired dark theme): query → suggestions → pick → name → thank-you
- Performer UI (live-updating): shows request queue, auto-refreshes every 5s
- JSON API:
    POST /api/search      { "prompt": "..." }           → { results: [...] }
    POST /api/request     { "prompt": "...", ... }      → { status: "ok", id: ... }
    GET  /api/requests    → [ request records... ]

- Discord notifications on each new request via webhook.

- Scaffolded private tools (for future expansion):
    JeffGPT:
      GET  /journal?code=Jeff
      POST /api/jm/chat

    Music CRM:
      GET  /crm?code=Jeff
      POST /api/crm/draft_email

    Portfolio helper (read-only advisory):
      GET  /portfolio?code=Jeff
      POST /api/portfolio/plan

Environment variables:
- OPENAI_API_KEY                 – used indirectly by SongBrain
- SONGMASTER_DISCORD_WEBHOOK_URL – Discord webhook URL for notifications
- SONGMASTER_PERFORMER_CODE      – optional; if set, private views require ?code=...
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from songbrain import SongBrain

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
REQUESTS_PATH = BASE_DIR / "song_requests.jsonl"

PERFORMER_CODE = os.environ.get("SONGMASTER_PERFORMER_CODE", "").strip()
DISCORD_WEBHOOK_URL = os.environ.get("SONGMASTER_DISCORD_WEBHOOK_URL", "").strip()

# ── APP SETUP ────────────────────────────────────────────────────────────────

app = FastAPI(title="Songmaster API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SongBrain once (full intelligence: embeddings + GPT rerank)
try:
    songbrain = SongBrain()
except Exception as e:
    raise RuntimeError(f"Failed to initialize SongBrain: {e}") from e


# ── MODELS – SONGMASTER ──────────────────────────────────────────────────────

class SongResult(BaseModel):
    song_idx: int
    key: str
    title: str
    artist: str
    category: str
    songbook: str
    group: str


class SearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)


class SearchResponse(BaseModel):
    results: List[SongResult]


class RequestPayload(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    song_idx: int
    name: Optional[str] = Field(default=None, max_length=100)


class RequestRecord(BaseModel):
    id: int
    timestamp: float
    prompt: str
    song_idx: int
    song_title: str
    song_key: str
    songbook: str
    name: Optional[str]
    status: str  # "pending", "accepted", "played", "skipped"


# ── REQUEST QUEUE UTILS ─────────────────────────────────────────────────────

def load_requests() -> List[RequestRecord]:
    records: List[RequestRecord] = []
    if not REQUESTS_PATH.exists():
        return records
    with REQUESTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(RequestRecord(**data))
            except Exception:
                continue
    return records


def append_request(rec: RequestRecord) -> None:
    with REQUESTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec.dict(), ensure_ascii=False) + "\n")


# ── DISCORD NOTIFICATIONS ───────────────────────────────────────────────────

def send_discord_notification(rec: RequestRecord) -> None:
    """
    Send a notification to a Discord channel via webhook.

    Uses SONGMASTER_DISCORD_WEBHOOK_URL from the environment.
    Fails silently if webhook is not configured or if the request errors out.
    """
    if not DISCORD_WEBHOOK_URL:
        return

    # Human-readable time
    try:
        import datetime as _dt
        ts_str = _dt.datetime.fromtimestamp(rec.timestamp).strftime("%H:%M:%S")
    except Exception:
        ts_str = "time unknown"

    name_part = rec.name or "Anonymous"
    song_part = f"[{rec.song_key or '-'}] {rec.song_title}"
    prompt_part = rec.prompt.strip().replace("\n", " ")

    content = (
        f"🎹 **New song request #{rec.id}**\n"
        f"**From:** {name_part}\n"
        f"**Song:** {song_part} ({rec.songbook or 'Songbook unknown'})\n"
        f"**Prompt:** “{prompt_part}”\n"
        f"**Time:** {ts_str}"
    )

    payload = {"content": content}
    try:
        requests.post(
            DISCORD_WEBHOOK_URL,
            json=payload,
            timeout=5,
        )
    except Exception:
        # We deliberately don't crash the request flow if Discord is down.
        pass


# ── PERFORMER GUARD ─────────────────────────────────────────────────────────

def require_performer(request: Request):
    """
    Simple guard for performer/private views: ?code=<SECRET>

    - If SONGMASTER_PERFORMER_CODE is empty, no protection.
    - Otherwise, compares code case-insensitively, trimming whitespace.
    """
    if not PERFORMER_CODE:
        return
    expected = PERFORMER_CODE.strip().lower()
    if not expected:
        return
    code = request.query_params.get("code", "").strip().lower()
    if code != expected:
        raise HTTPException(status_code=403, detail="Forbidden")
    return


# ── API ENDPOINTS – SONGMASTER SEARCH / REQUESTS ────────────────────────────

@app.post("/api/search", response_model=SearchResponse)
async def api_search(req: SearchRequest):
    """
    Guest or performer search. Returns top N matching songs, full intelligence.
    """
    results_raw = songbrain.search(req.prompt, top_k=25)
    results = [SongResult(**r) for r in results_raw]
    return SearchResponse(results=results)


@app.post("/api/request")
async def api_request_song(payload: RequestPayload):
    """
    Guest requests a single song based on a prior search.
    """
    songs = songbrain.songs
    if payload.song_idx < 0 or payload.song_idx >= len(songs):
        raise HTTPException(status_code=400, detail="Invalid song_idx")

    s = songs[payload.song_idx]
    existing = load_requests()
    next_id = (max((r.id for r in existing), default=0) + 1) if existing else 1

    rec = RequestRecord(
        id=next_id,
        timestamp=time.time(),
        prompt=payload.prompt,
        song_idx=payload.song_idx,
        song_title=s.title,
        song_key=s.key,
        songbook=s.songbook,
        name=(payload.name or "").strip() or None,
        status="pending",
    )
    append_request(rec)
    send_discord_notification(rec)
    return JSONResponse({"status": "ok", "id": rec.id})


@app.get("/api/requests", response_model=List[RequestRecord])
async def api_get_requests(dep=Depends(require_performer)):
    """
    JSON view of all requests, newest first.
    Used by the live-updating performer dashboard.
    """
    records = load_requests()
    records = sorted(records, key=lambda r: r.timestamp, reverse=True)
    return records


# ── GUEST HTML UI (PORNHUB-INSPIRED THEME) ──────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def guest_page():
    """
    Guest UI:
      - Prompt input
      - Shows search results
      - Lets guest pick one song + name
      - Posts to /api/request and shows thank-you
    Dark theme with orange accent, Pornhub-inspired but not identical.
    """
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Songmaster • Request a Song</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg-main: #0f0f0f;
      --bg-card: #1b1b1b;
      --bg-card-alt: #141414;
      --accent: #ff9900;
      --accent-soft: #ffbb33;
      --text-main: #f5f5f5;
      --text-muted: #aaaaaa;
      --border-subtle: #262626;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 1.5rem 1rem 2rem;
      font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
      background: radial-gradient(circle at top, #222 0, #0b0b0b 48%, #000 100%);
      color: var(--text-main);
      display: flex;
      justify-content: center;
    }
    .shell {
      width: 100%;
      max-width: 640px;
    }
    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }
    .logo {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 1.25rem;
    }
    .logo-main {
      color: #ffffff;
    }
    .logo-pill {
      background: var(--accent);
      color: #000;
      padding: 0.15rem 0.55rem;
      border-radius: 999px;
    }
    .subtitle {
      font-size: 0.8rem;
      color: var(--text-muted);
    }
    .card {
      background: var(--bg-card);
      border-radius: 12px;
      border: 1px solid var(--border-subtle);
      padding: 1rem;
      box-shadow: 0 14px 40px rgba(0,0,0,0.6);
    }
    label {
      font-size: 0.85rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    textarea, input[type="text"] {
      width: 100%;
      margin-top: 0.35rem;
      margin-bottom: 0.9rem;
      padding: 0.6rem 0.7rem;
      border-radius: 8px;
      border: 1px solid var(--border-subtle);
      background: var(--bg-card-alt);
      color: var(--text-main);
      font-size: 0.95rem;
      resize: vertical;
      min-height: 2.6rem;
    }
    textarea::placeholder, input::placeholder {
      color: #555;
    }
    button {
      padding: 0.55rem 1.1rem;
      border-radius: 999px;
      border: none;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      transition: background 0.12s ease, transform 0.05s ease, box-shadow 0.12s ease;
    }
    button:disabled {
      opacity: 0.6;
      cursor: default;
      box-shadow: none;
      transform: none;
    }
    .btn-primary {
      background: var(--accent);
      color: #000;
      box-shadow: 0 8px 18px rgba(255,153,0,0.35);
    }
    .btn-primary:hover:not(:disabled) {
      background: var(--accent-soft);
      transform: translateY(-1px);
      box-shadow: 0 10px 22px rgba(255,153,0,0.45);
    }
    .small {
      font-size: 0.8rem;
      color: var(--text-muted);
    }
    .results {
      margin-top: 1rem;
    }
    .results-title {
      font-size: 0.83rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      margin-bottom: 0.4rem;
    }
    .song-option {
      border-radius: 8px;
      border: 1px solid var(--border-subtle);
      padding: 0.6rem 0.7rem;
      margin-bottom: 0.45rem;
      background: #161616;
      cursor: pointer;
      transition: border 0.12s ease, background 0.12s ease, transform 0.04s ease;
    }
    .song-option:hover {
      border-color: var(--accent-soft);
    }
    .song-option.selected {
      border-color: var(--accent);
      background: #22190a;
      transform: translateY(-1px);
    }
    .song-title {
      font-size: 0.95rem;
    }
    .song-meta {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 0.15rem;
    }
    .badge-key {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 1.8rem;
      height: 1.2rem;
      border-radius: 999px;
      background: #252525;
      border: 1px solid var(--border-subtle);
      font-size: 0.75rem;
      margin-right: 0.4rem;
    }
    .section {
      margin-top: 0.9rem;
    }
    .thankyou {
      margin-top: 0.85rem;
      padding: 0.7rem 0.75rem;
      border-radius: 10px;
      background: rgba(0,120,40,0.18);
      border: 1px solid rgba(0,180,80,0.55);
      font-size: 0.86rem;
    }
    .thankyou b {
      color: #b7ffb7;
    }
    .error {
      margin-top: 0.85rem;
      padding: 0.7rem 0.75rem;
      border-radius: 10px;
      background: rgba(200,40,0,0.18);
      border: 1px solid rgba(255,80,40,0.5);
      font-size: 0.86rem;
    }
    @media (max-width: 480px) {
      body { padding: 1rem 0.6rem 1.6rem; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <div>
        <div class="logo">
          <span class="logo-main">Song</span>
          <span class="logo-pill">master</span>
        </div>
        <div class="subtitle">Live request line</div>
      </div>
    </div>

    <div class="card">
      <label for="prompt">Describe what you want to hear</label>
      <textarea id="prompt" rows="3" placeholder="Example: something from the 80s that both grandparents and teenagers would recognize"></textarea>

      <button id="searchBtn" class="btn-primary">
        <span>Find song ideas</span>
      </button>

      <div id="results" class="results"></div>

      <div id="pickSection" class="section" style="display:none;">
        <label for="name">Name for the shout-out (optional)</label>
        <input id="name" type="text" placeholder="e.g. Sarah, Table 4">

        <button id="submitBtn" class="btn-primary">
          <span>Send request</span>
        </button>
      </div>

      <div id="message"></div>
    </div>

    <p class="small" style="margin-top:0.6rem; text-align:center; color:#777;">
      Requests are suggestions, not guarantees — but Jeff sees every one. If you’re enjoying the music, a tip or a smile goes a long way. 🎹
    </p>
  </div>

<script>
const searchBtn = document.getElementById('searchBtn');
const submitBtn = document.getElementById('submitBtn');
const promptEl = document.getElementById('prompt');
const resultsEl = document.getElementById('results');
const pickSection = document.getElementById('pickSection');
const nameEl = document.getElementById('name');
const messageEl = document.getElementById('message');

let selectedSongIdx = null;

function clearMessage() {
  messageEl.innerHTML = '';
}

function renderResults(results) {
  resultsEl.innerHTML = '';
  selectedSongIdx = null;
  pickSection.style.display = 'none';

  if (!results || results.length === 0) {
    resultsEl.innerHTML = '<p class="small">No suggestions yet. Try describing it a little differently — mention an artist, movie, decade, or mood.</p>';
    return;
  }

  const heading = document.createElement('div');
  heading.className = 'results-title';
  heading.innerText = 'Tap one of these:';
  resultsEl.appendChild(heading);

  results.forEach((song) => {
    const div = document.createElement('div');
    div.className = 'song-option';
    div.dataset.songIdx = song.song_idx;
    div.innerHTML = `
      <div class="song-title">
        <span class="badge-key">${song.key || '-'}</span>
        <span>${song.title}</span>
      </div>
      <div class="song-meta">
        ${song.artist || ''} ${song.songbook ? ' • ' + song.songbook : ''}
      </div>
    `;
    div.onclick = () => {
      document.querySelectorAll('.song-option').forEach(el => el.classList.remove('selected'));
      div.classList.add('selected');
      selectedSongIdx = song.song_idx;
      pickSection.style.display = 'block';
      clearMessage();
    };
    resultsEl.appendChild(div);
  });
}

searchBtn.onclick = async () => {
  const prompt = promptEl.value.trim();
  if (!prompt) return;
  searchBtn.disabled = true;
  clearMessage();
  resultsEl.innerHTML = '<p class="small">Thinking...</p>';
  pickSection.style.display = 'none';
  selectedSongIdx = null;

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    if (!res.ok) throw new Error('Error from server');
    const data = await res.json();
    renderResults(data.results || []);
  } catch (err) {
    resultsEl.innerHTML = '';
    messageEl.innerHTML = '<div class="error">Sorry, something went wrong. Try again in a moment.</div>';
  } finally {
    searchBtn.disabled = false;
  }
};

submitBtn.onclick = async () => {
  if (selectedSongIdx === null) return;
  const prompt = promptEl.value.trim();
  const name = nameEl.value.trim();

  submitBtn.disabled = true;
  clearMessage();

  try {
    const res = await fetch('/api/request', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, song_idx: selectedSongIdx, name })
    });
    if (!res.ok) throw new Error('Error from server');
    await res.json();
    messageEl.innerHTML = `
      <div class="thankyou">
        <b>Thanks!</b> Your request is in Jeff’s queue.<br/>
        He can’t promise the exact order, but he <i>does</i> see it.<br/>
        If you’re enjoying the music, don’t forget to say thanks (or leave a tip 💖).
      </div>
    `;
    pickSection.style.display = 'none';
  } catch (err) {
    messageEl.innerHTML = '<div class="error">Sorry, something went wrong. Please try again.</div>';
  } finally {
    submitBtn.disabled = false;
  }
};
</script>
</body>
</html>
"""
    return HTMLResponse(html)


# ── PERFORMER VIEW (LIVE-UPDATING) ──────────────────────────────────────────

@app.get("/performer", response_class=HTMLResponse)
async def performer_page(dep=Depends(require_performer)):
    """
    Performer dashboard:
      - Uses JS to poll /api/requests every 5s
      - Highlights newly arrived requests
    """
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Songmaster • Performer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg-main: #050505;
      --bg-card: #141414;
      --accent: #ff9900;
      --text-main: #f3f3f3;
      --text-muted: #9a9a9a;
      --border-subtle: #262626;
      --row-alt: #111111;
      --row-new: #1e1305;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 0.75rem;
      font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
      background: radial-gradient(circle at top, #292929 0, #050505 55%, #000000 100%);
      color: var(--text-main);
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 0.7rem;
    }
    .logo {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 1rem;
    }
    .logo-main {
      color: #ffffff;
    }
    .logo-pill {
      background: var(--accent);
      color: #000;
      padding: 0.1rem 0.5rem;
      border-radius: 999px;
      font-size: 0.85rem;
    }
    .pill-live {
      border-radius: 999px;
      padding: 0.1rem 0.55rem;
      font-size: 0.75rem;
      border: 1px solid rgba(0,255,120,0.7);
      color: #b8ffcf;
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
    }
    .pill-live-dot {
      width: 0.4rem;
      height: 0.4rem;
      border-radius: 999px;
      background: #00ff88;
      box-shadow: 0 0 8px rgba(0,255,120,0.7);
    }
    .card {
      background: var(--bg-card);
      border-radius: 10px;
      border: 1px solid var(--border-subtle);
      padding: 0.5rem 0.75rem;
      overflow-x: auto;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      font-size: 0.78rem;
    }
    th, td {
      border-bottom: 1px solid #1f1f1f;
      padding: 0.3rem 0.4rem;
      vertical-align: top;
    }
    th {
      position: sticky;
      top: 0;
      background: #161616;
      z-index: 1;
    }
    th {
      text-align: left;
      color: var(--text-muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.07em;
    }
    tbody tr:nth-child(even) {
      background: var(--row-alt);
    }
    tbody tr.new-row {
      background: var(--row-new);
      animation: fadeNew 6s ease-out forwards;
    }
    @keyframes fadeNew {
      0% { background-color: var(--row-new); }
      100% { background-color: transparent; }
    }
    .meta {
      color: var(--text-muted);
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="logo">
      <span class="logo-main">Song</span>
      <span class="logo-pill">master</span>
    </div>
    <div class="pill-live">
      <span class="pill-live-dot"></span>
      <span>Live queue</span>
    </div>
  </div>

  <div class="card">
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Time</th>
          <th>Name</th>
          <th>Song</th>
          <th>Book</th>
          <th>Prompt</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody id="tbody">
        <tr><td colspan="7" class="meta">Loading…</td></tr>
      </tbody>
    </table>
  </div>

<script>
let lastSeenId = 0;

function fmtTime(ts) {
  try {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
  } catch (e) {
    return '';
  }
}

async function fetchRequests() {
  try {
    const res = await fetch('/api/requests');
    if (!res.ok) throw new Error('Failed');
    const data = await res.json();
    renderTable(data || []);
  } catch (err) {
    // leave previous table as-is
  }
}

function renderTable(records) {
  const tbody = document.getElementById('tbody');
  if (!records.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="meta">No requests yet.</td></tr>';
    return;
  }

  let html = '';
  let maxIdThisBatch = lastSeenId;

  records.forEach((r) => {
    if (r.id > maxIdThisBatch) maxIdThisBatch = r.id;
    const isNew = r.id > lastSeenId;
    html += `
      <tr class="${isNew ? 'new-row' : ''}">
        <td>${r.id}</td>
        <td>${fmtTime(r.timestamp)}</td>
        <td>${r.name || '&nbsp;'}</td>
        <td>[${r.song_key || '-'}] ${r.song_title}</td>
        <td>${r.songbook || '&nbsp;'}</td>
        <td>${r.prompt}</td>
        <td>${r.status}</td>
      </tr>
    `;
  });

  tbody.innerHTML = html;

  if (maxIdThisBatch > lastSeenId) {
    lastSeenId = maxIdThisBatch;
  }
}

// initial load
fetchRequests();
// poll every 5 seconds
setInterval(fetchRequests, 5000);
</script>
</body>
</html>
"""
    return HTMLResponse(html)


# ── JEFFGPT (JOURNALMASTER ONLINE) – SCAFFOLD ────────────────────────────────

class JMChatTurn(BaseModel):
    role: str
    content: str


class JMChatRequest(BaseModel):
    history: List[JMChatTurn] = Field(default_factory=list)
    message: str = Field(..., min_length=1, max_length=4000)


class JMChatResponse(BaseModel):
    reply: str


@app.get("/journal", response_class=HTMLResponse)
async def journal_page(dep=Depends(require_performer)):
    """
    Placeholder page for JeffGPT (Journalmaster Online).
    Later this becomes a full chat UI talking to your personal corpus.
    """
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>JeffGPT • Journalmaster Online</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      margin: 0;
      padding: 1rem;
      font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
      background: radial-gradient(circle at top, #1d1d1d 0, #050505 60%, #000 100%);
      color: #f5f5f5;
    }
    .shell {
      max-width: 720px;
      margin: 0 auto;
    }
    .logo {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 1.1rem;
    }
    .logo-main {
      color: #ffffff;
    }
    .logo-pill {
      background: #ff9900;
      color: #000;
      padding: 0.1rem 0.55rem;
      border-radius: 999px;
    }
    .card {
      margin-top: 0.8rem;
      background: #151515;
      border-radius: 10px;
      border: 1px solid #2a2a2a;
      padding: 0.9rem;
    }
    p {
      font-size: 0.9rem;
      color: #c3c3c3;
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="logo">
      <span class="logo-main">Jeff</span>
      <span class="logo-pill">GPT</span>
    </div>
    <div class="card">
      <p>This will be your private, memory-augmented reflection partner.</p>
      <p>Plan:</p>
      <ul>
        <li>Load 10+ years of journals and transcripts (exported from Loom/Journalmaster).</li>
        <li>Use semantic search to pull in relevant memories for each question.</li>
        <li>Call an LLM with that context to have “therapy-like” conversations (without pretending to be a therapist).</li>
      </ul>
      <p>Right now this is just a placeholder. The API endpoint <code>/api/jm/chat</code> is wired as a stub and can be expanded once the JeffGPT brain is built.</p>
    </div>
  </div>
</body>
</html>
"""
    return HTMLResponse(html)


@app.post("/api/jm/chat", response_model=JMChatResponse)
async def api_jm_chat(req: JMChatRequest, dep=Depends(require_performer)):
    """
    Scaffold endpoint for JeffGPT chat.

    For now this just echoes that it's not implemented yet.
    Later it will:
      - run retrieval over your journal corpus
      - call OpenAI with those snippets + conversation history
      - return a reflective response
    """
    reply = (
        "JeffGPT isn’t wired up yet in this deployment, "
        "but this is where your journal-aware responses will come from.\n\n"
        f"You said: {req.message}"
    )
    return JMChatResponse(reply=reply)


# ── MUSIC CRM – SCAFFOLD ─────────────────────────────────────────────────────

class CRMDraftRequest(BaseModel):
    contact_name: Optional[str] = Field(default=None, max_length=200)
    venue_type: Optional[str] = Field(default=None, max_length=200)
    notes: Optional[str] = Field(
        default=None,
        description="Free-form notes about the context, e.g. 'upscale restaurant in Durango, wants mellow jazz'.",
    )


class CRMDraftResponse(BaseModel):
    subject: str
    body: str


@app.get("/crm", response_class=HTMLResponse)
async def crm_page(dep=Depends(require_performer)):
    """
    Placeholder page for a future music-contacts CRM assistant.
    """
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Music CRM • Outreach Assistant</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      margin: 0;
      padding: 1rem;
      font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
      background: radial-gradient(circle at top, #1d1d1d 0, #050505 60%, #000 100%);
      color: #f5f5f5;
    }
    .shell {
      max-width: 720px;
      margin: 0 auto;
    }
    .logo {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 1.1rem;
    }
    .logo-main {
      color: #ffffff;
    }
    .logo-pill {
      background: #ff9900;
      color: #000;
      padding: 0.1rem 0.55rem;
      border-radius: 999px;
    }
    .card {
      margin-top: 0.8rem;
      background: #151515;
      border-radius: 10px;
      border: 1px solid #2a2a2a;
      padding: 0.9rem;
    }
    p {
      font-size: 0.9rem;
      color: #c3c3c3;
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="logo">
      <span class="logo-main">Music</span>
      <span class="logo-pill">CRM</span>
    </div>
    <div class="card">
      <p>This will become your gig/outreach assistant:</p>
      <ul>
        <li>Knows your contacts and venue history.</li>
        <li>Suggests which template + links to use when emailing managers.</li>
        <li>Drafts emails like “reach out to the manager of RestaurantXYZ”.</li>
      </ul>
      <p>Right now <code>/api/crm/draft_email</code> is a stub endpoint; it just returns a placeholder draft.</p>
    </div>
  </div>
</body>
</html>
"""
    return HTMLResponse(html)


@app.post("/api/crm/draft_email", response_model=CRMDraftResponse)
async def api_crm_draft_email(payload: CRMDraftRequest, dep=Depends(require_performer)):
    """
    Scaffold endpoint for CRM email drafting.

    For now:
      - Ignores payload details and returns a simple placeholder draft.
    Later:
      - Will look up the contact and venue history
      - Choose templates / links
      - Call an LLM to draft a tailored email
    """
    contact = payload.contact_name or "there"
    subject = "Live music inquiry"
    body_lines = [
        f"Hi {contact},",
        "",
        "This is a placeholder from your future music CRM assistant.",
        "Once wired, this will draft a tailored email using your templates,",
        "venue type, prior notes, and links to your best reels.",
        "",
        "— Jeff",
    ]
    return CRMDraftResponse(subject=subject, body="\n".join(body_lines))


# ── PORTFOLIO HELPER – SCAFFOLD (READ-ONLY ADVISORY) ─────────────────────────

class Holding(BaseModel):
    symbol: str
    basis_price: Optional[float] = None
    current_price: Optional[float] = None
    allocation_pct: Optional[float] = None


class PortfolioPlanRequest(BaseModel):
    holdings: List[Holding] = Field(
        default_factory=list,
        description="Current positions you care about (symbol, basis, current, allocation).",
    )
    risk_notes: Optional[str] = Field(
        default=None,
        description="Free text about your risk tolerance, e.g. 'slightly more conservative than big players; avoid huge drawdowns.'",
    )


class PortfolioPlanResponse(BaseModel):
    summary: str


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(dep=Depends(require_performer)):
    """
    Placeholder page for a future portfolio helper (read-only advisory).
    """
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Portfolio Helper • Defensive Planning</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      margin: 0;
      padding: 1rem;
      font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
      background: radial-gradient(circle at top, #1d1d1d 0, #050505 60%, #000 100%);
      color: #f5f5f5;
    }
    .shell {
      max-width: 720px;
      margin: 0 auto;
    }
    .logo {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 1.1rem;
    }
    .logo-main {
      color: #ffffff;
    }
    .logo-pill {
      background: #ff9900;
      color: #000;
      padding: 0.1rem 0.55rem;
      border-radius: 999px;
    }
    .card {
      margin-top: 0.8rem;
      background: #151515;
      border-radius: 10px;
      border: 1px solid #2a2a2a;
      padding: 0.9rem;
    }
    p {
      font-size: 0.9rem;
      color: #c3c3c3;
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="logo">
      <span class="logo-main">Portfolio</span>
      <span class="logo-pill">Helper</span>
    </div>
    <div class="card">
      <p>This will eventually help you think through defensive rules:</p>
      <ul>
        <li>Summarize your current positions.</li>
        <li>Propose rules-of-thumb for protective exits (read-only; no auto-trading).</li>
        <li>Help you stay a bit more conservative than big players in selloffs.</li>
      </ul>
      <p>Endpoint <code>/api/portfolio/plan</code> currently returns a stub summary and will be wired to an LLM later.</p>
    </div>
  </div>
</body>
</html>
"""
    return HTMLResponse(html)


@app.post("/api/portfolio/plan", response_model=PortfolioPlanResponse)
async def api_portfolio_plan(payload: PortfolioPlanRequest, dep=Depends(require_performer)):
    """
    Scaffold endpoint for portfolio planning helper.

    For now:
      - Returns a static summary acknowledging the request.
    Later:
      - Will call an LLM with your holdings + risk notes to suggest defensive rules.
    """
    symbols = ", ".join(sorted({h.symbol.upper() for h in payload.holdings})) or "no symbols provided"
    summary = (
        "This is a placeholder for your portfolio planning helper.\n\n"
        f"Symbols mentioned: {symbols}\n\n"
        "Once wired, this endpoint will:\n"
        "- Read your positions and risk notes.\n"
        "- Suggest protective rules (e.g., trailing stops, max drawdowns) as *advice only*.\n"
        "- Leave actual order entry to you in Vanguard/Kraken/etc."
    )
    return PortfolioPlanResponse(summary=summary)
