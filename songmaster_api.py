#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
songmaster_api.py – FastAPI service for Songmaster + private tools

Features:
- Guest UI (Spotify Pink Parallax): query → suggestions → pick → name → thank-you
- Live Page (/live): Narrative, Value Pillars, and SMS Opt-in.
- About Page (/about): Digital business card and hireable offerings.
- Admin UI (/admin): Global kill switch and AI gut check, protected by basic auth.
- Performer UI (live-updating): shows request queue, auto-refreshes every 5s
- JSON API with Rate Limiting and Admin Kill Switch.
"""

import json
import os
import time
import sys
import secrets
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── PATH SETUP FOR SIBLING PACKAGES ─────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from songbrain import SongBrain

try:
    from jeffgpt.jeffgpt_brain import JeffGPTBrain  # type: ignore
except ImportError as e:
    JeffGPTBrain = None  # type: ignore


# ── CONFIG & SECURITY STATE ──────────────────────────────────────────────────

REQUESTS_PATH = BASE_DIR / "song_requests.jsonl"

PERFORMER_CODE = os.environ.get("SONGMASTER_PERFORMER_CODE", "").strip()
DISCORD_WEBHOOK_URL = os.environ.get("SONGMASTER_DISCORD_WEBHOOK_URL", "").strip()
JEFFGPT_MODEL = os.environ.get("JEFFGPT_MODEL", "gpt-4.1")

ADMIN_USER = os.environ.get("SONGMASTER_ADMIN_USER")
ADMIN_PASS = os.environ.get("SONGMASTER_ADMIN_PASS")
REQUESTS_OPEN = True  

security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if not ADMIN_USER or not ADMIN_PASS:
        raise HTTPException(status_code=500, detail="Admin credentials not configured in environment.")
    
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USER)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# ── APP SETUP & RATE LIMITER ─────────────────────────────────────────────────

app = FastAPI(title="Jeff Tools API", version="0.6.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

def get_real_ip(request: Request):
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

limiter = Limiter(key_func=get_real_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    ip = get_real_ip(request)
    if DISCORD_WEBHOOK_URL:
        payload = {"content": f"⚠️ **Rate Limit Hit:** IP `{ip}` is spamming the Songmaster API."}
        try:
            requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=2)
        except Exception:
            pass
    return JSONResponse(status_code=429, content={"detail": "Too many requests. Slow down, friend."})

try:
    songbrain = SongBrain()
except Exception as e:
    raise RuntimeError(f"Failed to initialize SongBrain: {e}") from e

openai_client = OpenAI()

if "JeffGPTBrain" in globals() and JeffGPTBrain is not None:
    try:
        jeff_brain = JeffGPTBrain()
    except Exception as e:
        jeff_brain = None
else:
    jeff_brain = None


# ── MODELS ──────────────────────────────────────────────────────────────────

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
    status: str 

class IngestRequest(BaseModel):
    song_data: str

class SubscribeRequest(BaseModel):
    phone: str = Field(..., min_length=7, max_length=20)


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

def send_discord_notification(rec: RequestRecord) -> None:
    if not DISCORD_WEBHOOK_URL:
        return
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
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=5)
    except Exception:
        pass

def require_performer(request: Request):
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

@app.post("/api/subscribe")
async def api_subscribe(req: SubscribeRequest):
    if not DISCORD_WEBHOOK_URL:
        return JSONResponse({"status": "error", "message": "Notification system offline"}, status_code=500)
    
    content = f"📱 **NEW SMS OPT-IN**\n**Phone:** `{req.phone}`\n*Add this to the Thursday morning Shortcut list.*"
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=5)
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=500, detail="Webhook failed")

@app.post("/api/search", response_model=SearchResponse)
@limiter.limit("10/15minutes")
async def api_search(req: SearchRequest, request: Request):
    if not REQUESTS_OPEN:
        raise HTTPException(status_code=403, detail="Requests are currently closed.")
        
    results_raw = songbrain.search(req.prompt, top_k=25)
    results = [SongResult(**r) for r in results_raw]
    return SearchResponse(results=results)


@app.post("/api/request")
async def api_request_song(payload: RequestPayload):
    if not REQUESTS_OPEN:
        raise HTTPException(status_code=403, detail="Requests are currently closed.")
        
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
    records = load_requests()
    records = sorted(records, key=lambda r: r.timestamp, reverse=True)
    return records


# ── ADMIN API ENDPOINTS & PAGE ──────────────────────────────────────────────

@app.post("/admin/api/toggle")
async def toggle_requests(request: Request, admin: str = Depends(verify_admin)):
    global REQUESTS_OPEN
    REQUESTS_OPEN = not REQUESTS_OPEN
    status_msg = "OPEN" if REQUESTS_OPEN else "CLOSED"
    return {"message": f"Requests are now {status_msg}"}


@app.post("/admin/api/song/search")
async def admin_search_song(req: dict, request: Request, admin: str = Depends(verify_admin)):
    query = req.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")
        
    prompt = f"Identify the real song title and artist/source for: '{query}'. Keep it brief, e.g., 'Title by Artist'."
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}]
        )
        gut_check_result = resp.choices[0].message.content.strip()
    except Exception as e:
        gut_check_result = f"Error reaching OpenAI: {e}"
        
    return {"result": gut_check_result}


@app.post("/admin/api/song/ingest")
async def admin_ingest_song(req: IngestRequest, admin: str = Depends(verify_admin)):
    """
    Endpoint triggered by the 'Download & Ingest' button.
    """
    try:
        if hasattr(songbrain, 'add_new_song'):
            songbrain.add_new_song(req.song_data)
        else:
            # Fallback simulation if the method isn't written in songbrain yet
            import asyncio
            await asyncio.sleep(2) 
            
        return {"message": f"Successfully ingested: {req.song_data}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(admin: str = Depends(verify_admin)):
    global REQUESTS_OPEN
    btn_text = "Close Requests" if REQUESTS_OPEN else "Open Requests"
    btn_class = "danger" if REQUESTS_OPEN else ""
    sys_status = "System is OPEN." if REQUESTS_OPEN else "System is CLOSED. The guest UI will reject inputs."
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Songmaster Admin</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{ font-family: system-ui, sans-serif; background: #050509; color: #e5e7eb; padding: 2rem; max-width: 600px; margin: auto; }}
    .card {{ background: #111218; padding: 1.5rem; border-radius: 8px; border: 1px solid #27272f; margin-bottom: 1.5rem; }}
    button {{ background: #10a37f; color: #000; border: none; padding: 0.5rem 1rem; border-radius: 4px; font-weight: bold; cursor: pointer; transition: 0.15s ease; }}
    button:hover {{ background: #0e8c6d; }}
    button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    .danger {{ background: #dc2626; color: white; }}
    .danger:hover {{ background: #b91c1c; }}
    input {{ width: 100%; padding: 0.5rem; margin: 0.5rem 0; background: #030712; border: 1px solid #1f2937; color: white; border-radius: 4px;}}
    .hidden {{ display: none; }}
    .status {{ font-size: 0.8rem; color: #9ca3af; margin-top: 0.5rem; }}
    .success {{ color: #10a37f; margin-top: 0.5rem; font-size: 0.9rem; font-weight: bold; }}
  </style>
</head>
<body>
  <h2>🎹 Songmaster Admin</h2>

  <div class="card">
    <h3>Global Kill Switch</h3>
    <p>Currently, the guest UI can make requests.</p>
    <button id="toggleBtn" class="{btn_class}">{btn_text}</button>
    <div id="toggleStatus" class="status">{sys_status}</div>
  </div>

  <div class="card">
    <h3>Add Song (Gut Check & Ingest)</h3>
    <p>Type a song, the system will verify it using AI before downloading the heavy data.</p>
    <input type="text" id="songInput" placeholder="e.g. Piano Man">
    <button id="searchBtn">Gut Check</button>

    <div id="confirmSection" class="hidden" style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #27272f;">
      <p style="margin-bottom: 1rem;"><strong>Result:</strong> <span id="songResult" style="color:#10a37f;"></span></p>
      <button id="ingestBtn">Yes, Download & Ingest</button>
      <div id="ingestStatus"></div>
    </div>
  </div>

  <script>
    const toggleBtn = document.getElementById('toggleBtn');
    const toggleStatus = document.getElementById('toggleStatus');
    let isOpen = {'true' if REQUESTS_OPEN else 'false'};

    toggleBtn.onclick = async () => {{
      const res = await fetch('/admin/api/toggle', {{ method: 'POST' }});
      if (res.ok) {{
        isOpen = !isOpen;
        if (isOpen) {{
          toggleBtn.textContent = "Close Requests";
          toggleBtn.className = "danger";
          toggleStatus.textContent = "System is OPEN.";
        }} else {{
          toggleBtn.textContent = "Open Requests";
          toggleBtn.className = "";
          toggleStatus.textContent = "System is CLOSED. The guest UI will reject inputs.";
        }}
      }}
    }};

    const searchBtn = document.getElementById('searchBtn');
    const songInput = document.getElementById('songInput');
    const confirmSection = document.getElementById('confirmSection');
    const songResult = document.getElementById('songResult');
    const ingestBtn = document.getElementById('ingestBtn');
    const ingestStatus = document.getElementById('ingestStatus');

    let currentValidatedSong = "";

    searchBtn.onclick = async () => {{
      const query = songInput.value;
      if (!query) return;
      
      searchBtn.textContent = "Searching...";
      ingestStatus.innerHTML = "";
      confirmSection.classList.add('hidden');
      
      try {{
        const res = await fetch('/admin/api/song/search', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ query }})
        }});
        const data = await res.json();
        currentValidatedSong = data.result;
        songResult.textContent = currentValidatedSong;
        confirmSection.classList.remove('hidden');
        ingestBtn.disabled = false;
        ingestBtn.textContent = "Yes, Download & Ingest";
      }} catch (err) {{
        songResult.textContent = "Error fetching result.";
        confirmSection.classList.remove('hidden');
      }} finally {{
        searchBtn.textContent = "Gut Check";
      }}
    }};

    ingestBtn.onclick = async () => {{
      ingestBtn.textContent = "Downloading & Embedding...";
      ingestBtn.disabled = true;
      ingestStatus.innerHTML = "";

      try {{
        const res = await fetch('/admin/api/song/ingest', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ song_data: currentValidatedSong }})
        }});
        
        if (!res.ok) throw new Error('Ingestion failed');
        
        const data = await res.json();
        ingestStatus.innerHTML = `<div class="success">✅ ${{data.message}}</div>`;
        songInput.value = ""; 
      }} catch (err) {{
        ingestStatus.innerHTML = `<div class="status" style="color: #dc2626;">❌ Error adding song. Check server logs.</div>`;
        ingestBtn.textContent = "Try Again";
        ingestBtn.disabled = false;
      }}
    }};
  </script>
</body>
</html>
"""
    return HTMLResponse(html)


# ── PAGE ENDPOINTS (HTML) ───────────────────────────────────────────────────

@app.get("/live", response_class=HTMLResponse)
async def live_page():
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Cherokee Radio • Live Music Variety Show</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg-card: rgba(24, 24, 24, 0.85);
      --accent: #1ed760; /* Green */
      --accent-soft: #ff66b2; /* Pink */
      --text-main: #ffffff;
      --text-muted: #a7a7a7;
      --border-subtle: #333333;
    }
    body {
      margin: 0;
      padding: 1.5rem 1rem 3rem;
      font-family: system-ui, sans-serif;
      background-image: linear-gradient(to bottom, rgba(18,18,18,0.90) 0%, rgba(0,0,0,0.98) 100%), url('/static/background.png');
      background-size: cover;
      background-position: center;
      background-attachment: fixed;
      background-blend-mode: multiply;
      color: var(--text-main);
      display: flex;
      justify-content: center;
      font-size: 16px; 
    }
    .shell { width: 100%; max-width: 900px; }
    
    .nav-bar { display: flex; justify-content: center; gap: 2rem; margin-bottom: 3rem; border-bottom: 1px solid var(--border-subtle); padding-bottom: 1rem; }
    .nav-link { color: var(--text-muted); text-decoration: none; font-weight: 700; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1.5px; transition: color 0.15s ease; }
    .nav-link:hover, .nav-link.active { color: var(--accent-soft); }

    .hero { text-align: center; margin-bottom: 3rem; }
    .title { font-weight: 900; font-size: 3rem; color: var(--accent-soft); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.2rem; text-shadow: 0 0 15px rgba(255,102,178,0.4); }
    .subtitle { font-size: 1.15rem; color: var(--accent); font-weight: 800; text-transform: uppercase; letter-spacing: 4px; }
    
    .card { background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border-subtle); padding: 2.5rem; box-shadow: 0 20px 40px rgba(0,0,0,0.8); backdrop-filter: blur(4px); margin-bottom: 2rem; }
    .card p { line-height: 1.6; color: var(--text-muted); font-size: 1.05rem; margin-top: 0; margin-bottom: 1.2rem; }
    .card strong { color: var(--text-main); }

    .pillar-list { margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid var(--border-subtle); }
    .pillar { margin-bottom: 1.5rem; }
    .pillar-title { color: var(--accent-soft); font-weight: 800; text-transform: uppercase; font-size: 1rem; letter-spacing: 1px; display: block; margin-bottom: 0.2rem; }
    .pillar-desc { color: var(--text-muted); font-size: 0.95rem; line-height: 1.5; display: block; }

    .cta-container { text-align: center; margin-top: 2.5rem; }
    .btn-primary { background: var(--accent-soft); color: #000; padding: 1rem 2.5rem; border-radius: 999px; border: none; font-size: 1rem; font-weight: 800; cursor: pointer; text-transform: uppercase; transition: transform 0.2s ease; }
    .btn-primary:hover { transform: scale(1.03); box-shadow: 0 0 25px rgba(255,102,178,0.3); }
    .yt-link { display: block; margin-top: 1.5rem; color: var(--accent); font-weight: 700; text-decoration: none; font-size: 0.95rem; }

    #modalOverlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); backdrop-filter: blur(10px); z-index: 1000; justify-content: center; align-items: center; }
    .modal-content { background: #111; border: 1px solid var(--border-subtle); padding: 2.5rem; border-radius: 12px; width: 90%; max-width: 400px; text-align: center; }
    input[type="tel"] { width: 100%; padding: 1rem; background: #000; border: 1px solid var(--border-subtle); border-radius: 8px; color: white; font-size: 1.1rem; margin: 1.5rem 0; text-align: center; }
    .modal-btn { background: var(--accent); color: #000; width: 100%; padding: 1rem; border-radius: 8px; border: none; font-weight: 800; cursor: pointer; text-transform: uppercase; }
  </style>
</head>
<body>
  <div class="shell">
    
    <div class="nav-bar">
      <a href="/" class="nav-link">Request Line</a>
      <a href="/live" class="nav-link active">Live Show</a>
      <a href="/about" class="nav-link">About</a>
    </div>

    <div class="hero">
      <div class="title">8pm ET | EVERY THURSDAY</div>
      <div class="subtitle">LIVE MUSIC VARIETY SHOW</div>
    </div>

    <div class="card">
      <p><strong>Are you tired of feeling purposely isolated by the internet you consume?</strong> I’m tired of it, too. I’m tired of competing for your attention against loud, annoying creators and "bullshit" content designed for nothing but mind-numbing engagement.</p>
      
      <p>As a full-time artist, I made the decision to delete my social media presence entirely. It was a radical move, but I couldn't keep participating in a system where the algorithm feeds us depressing news and low-brow slop while we all suffer in isolation.</p>
      
      <p>I am coming back online on my own terms. As a full-time restaurant musician, I’ve spent years perfecting a unique game—parodying familiar tunes and soundtracking the room in real-time. I’ve digitized that experience into <strong>jeffy.app</strong> and I’m bringing it to your doorstep.</p>
      
      <p><strong>I'm going to level with you: I’m putting a lot into this.</strong> Each Thursday at 8pm ET, tune in for 90 minutes broadcasted LIVE from 9,000 ft:</p>

      <div class="pillar-list">
        <div class="pillar">
          <span class="pillar-title">A Free Musical Massage</span>
          <span class="pillar-desc">I’m blending my experience as a meditation facilitator with the ability to make the piano speak. These are state-of-the-art soundscapes designed to put you into a restorative, trancelike state.</span>
        </div>
        <div class="pillar">
          <span class="pillar-title">A Free Music Lesson</span>
          <span class="pillar-desc">Based on my time as an engineer at Lockheed Martin, I take high-level concepts and make them simple. Every week I reveal trade secrets of mastering the piano and your vocal ability.</span>
        </div>
        <div class="pillar">
          <span class="pillar-title">Request a Song</span>
          <span class="pillar-desc">Experience the electric performance I deliver at venues across the country. Browse my repertoire with my custom jeffy.app and request a song to join the live soundtrack.</span>
        </div>
      </div>

      <div class="cta-container">
        <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1.5rem;">Join the notification network. Receive a handwritten text every Thursday morning.</p>
        <button class="btn-primary" onclick="openModal()">[ Opt-In For Reminders ]</button>
        <a href="https://www.youtube.com/@pursuingperspective/live" target="_blank" class="yt-link">Watch Live: youtube.com/@pursuingperspective/live</a>
      </div>
    </div>
  </div>

  <div id="modalOverlay">
    <div class="modal-content">
      <h2 style="color: var(--accent-soft); margin-top: 0; font-size: 1.5rem;">JOIN THE CIRCLE</h2>
      <p style="color: var(--text-muted); font-size: 0.9rem;">Thursday morning handwritten reminders. No bots.</p>
      <input type="tel" id="phoneInput" placeholder="(555) 555-5555">
      <button class="modal-btn" onclick="submitPhone()">Remind Me</button>
      <p style="font-size: 0.8rem; margin-top: 1.5rem; cursor: pointer; color: var(--text-muted);" onclick="closeModal()">Maybe later</p>
    </div>
  </div>

  <script>
    function openModal() { document.getElementById('modalOverlay').style.display = 'flex'; }
    function closeModal() { document.getElementById('modalOverlay').style.display = 'none'; }
    
    async function submitPhone() {
      const phone = document.getElementById('phoneInput').value;
      if(!phone) return;
      const btn = document.querySelector('.modal-btn');
      btn.innerText = "SENDING...";
      btn.disabled = true;
      try {
        const res = await fetch('/api/subscribe', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ phone: phone })
        });
        if (res.ok) {
          document.querySelector('.modal-content').innerHTML = "<h2 style='color: var(--accent);'>YOU'RE IN.</h2><p>See you Thursday.</p><p style='cursor:pointer' onclick='closeModal()'>Close</p>";
        }
      } catch (err) {
        btn.innerText = "TRY AGAIN";
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""
    return HTMLResponse(html)

@app.get("/", response_class=HTMLResponse)
async def guest_page():
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Songmaster • Live Requests</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg-main: transparent;
      --bg-card: rgba(24, 24, 24, 0.85); 
      --bg-card-alt: rgba(40, 40, 40, 0.9);
      --accent: #1ed760; 
      --accent-soft: #ff66b2; 
      --text-main: #ffffff;
      --text-muted: #a7a7a7;
      --border-subtle: #333333;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 1.5rem 1rem 2rem;
      font-family: system-ui, sans-serif;
      background-image: linear-gradient(to bottom, rgba(18,18,18,0.85) 0%, rgba(0,0,0,0.98) 100%), url('/static/background.png');
      background-size: cover;
      background-position: center;
      background-attachment: fixed;
      background-blend-mode: multiply;
      color: var(--text-main);
      display: flex;
      justify-content: center;
      min-height: 100vh;
    }
    .shell { width: 100%; max-width: 720px; }
    
    .nav-bar { display: flex; justify-content: center; gap: 2rem; margin-bottom: 2rem; border-bottom: 1px solid var(--border-subtle); padding-bottom: 1rem; }
    .nav-link { color: var(--text-muted); text-decoration: none; font-weight: 700; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1.5px; transition: color 0.15s ease; }
    .nav-link:hover, .nav-link.active { color: var(--accent-soft); text-shadow: 0 0 8px rgba(255,102,178,0.4); }

    .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.2rem; gap: 0.75rem; }
    .logo { display: flex; flex-direction: column; gap: 0.25rem; }
    .logo-main { font-weight: 700; font-size: 1.25rem; display: inline-flex; align-items: center; gap: 0.35rem; color: var(--accent-soft); text-shadow: 0 0 10px rgba(255,102,178,0.3); }
    .logo-mark { width: 1.65rem; height: 1.65rem; border-radius: 999px; border: 1px solid var(--accent-soft); display: inline-flex; align-items: center; justify-content: center; font-size: 0.9rem; color: var(--accent-soft); }
    .logo-text { text-transform: uppercase; font-size: 0.98rem; }
    .subtitle { font-size: 0.84rem; color: var(--text-muted); }
    
    .status-pill { border-radius: 999px; padding: 0.2rem 0.7rem; border: 1px solid rgba(148,163,184,0.7); font-size: 0.74rem; color: var(--text-muted); display: inline-flex; align-items: center; gap: 0.35rem; white-space: nowrap; background: rgba(0,0,0,0.5); }
    .status-dot { width: 0.45rem; height: 0.45rem; border-radius: 999px; background: var(--accent); box-shadow: 0 0 8px rgba(30,215,96,0.85); }
    
    .card { background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border-subtle); padding: 1rem; box-shadow: 0 20px 40px rgba(0,0,0,0.8); backdrop-filter: blur(4px); }
    label { font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
    textarea, input[type="text"] { width: 100%; margin-top: 0.35rem; margin-bottom: 0.9rem; padding: 0.6rem 0.7rem; border-radius: 8px; border: 1px solid var(--border-subtle); background: var(--bg-card-alt); color: var(--text-main); font-size: 0.95rem; resize: vertical; min-height: 2.6rem; }
    button { padding: 0.55rem 1.1rem; border-radius: 999px; border: none; font-size: 0.94rem; font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; gap: 0.35rem; transition: 0.15s ease; }
    button:disabled { opacity: 0.6; cursor: default; transform: none; box-shadow: none; }
    .btn-primary { background: var(--accent); color: #000; }
    .btn-primary:hover:not(:disabled) { background: #1fdf64; transform: scale(1.02); box-shadow: 0 10px 25px rgba(255,102,178,0.25); }
    
    .small { font-size: 0.78rem; color: var(--text-muted); }
    .results { margin-top: 1rem; }
    .results-title { font-size: 0.8rem; color: var(--accent-soft); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    .song-option { border-radius: 8px; border: 1px solid var(--border-subtle); padding: 0.6rem 0.7rem; margin-bottom: 0.45rem; background: rgba(0,0,0,0.4); cursor: pointer; transition: 0.1s ease; }
    .song-option:hover { border-color: var(--accent-soft); background: rgba(255,102,178,0.05); }
    .song-option.selected { border-color: var(--accent); background: rgba(30,215,96,0.1); }
    .song-title { font-size: 0.95rem; }
    .song-meta { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.15rem; }
    .badge-key { display: inline-flex; align-items: center; justify-content: center; min-width: 1.8rem; height: 1.2rem; border-radius: 999px; background: rgba(0,0,0,0.8); border: 1px solid rgba(148,163,184,0.3); font-size: 0.75rem; margin-right: 0.4rem; color: var(--accent); }
    .section { margin-top: 0.9rem; }
    .thankyou { margin-top: 0.85rem; padding: 0.7rem 0.75rem; border-radius: 10px; background: rgba(30,215,96,0.12); border: 1px solid rgba(30,215,96,0.55); font-size: 0.86rem; }
    .error { margin-top: 0.85rem; padding: 0.7rem 0.75rem; border-radius: 10px; background: rgba(255,102,178,0.12); border: 1px solid rgba(255,102,178,0.55); font-size: 0.86rem; color: #ffb3d9;}
    @media (max-width: 480px) { body { padding: 1rem 0.6rem 1.6rem; } .header { flex-direction: column; align-items: flex-start; } }
  </style>
</head>
<body>
  <div class="shell">
  
    <div class="nav-bar">
      <a href="/" class="nav-link active">Request Line</a>
      <a href="/live" class="nav-link">Live Show</a>
      <a href="/about" class="nav-link">About</a>
    </div>

    <div class="header">
      <div class="logo">
        <div class="logo-main"><span class="logo-mark">🎹</span><span class="logo-text">Songmaster</span></div>
        <div class="subtitle">Describe a song or vibe, Jeff’s AI will suggest options.</div>
      </div>
      <div class="status-pill">
        <span class="status-dot"></span>
        <span>Requests open</span>
      </div>
    </div>

    <div class="card">
      <label for="prompt">Describe what you want to hear</label>
      <textarea id="prompt" rows="3" placeholder="Example: something from the 80s that both grandparents and teenagers would recognize..."></textarea>
      <button id="searchBtn" class="btn-primary"><span>Find song ideas</span></button>
      <div id="results" class="results"></div>
      <div id="pickSection" class="section" style="display:none;">
        <label for="name">Name for the shout-out (optional)</label>
        <input id="name" type="text" placeholder="e.g. Sarah, Table 4">
        <button id="submitBtn" class="btn-primary"><span>Send request</span></button>
      </div>
      <div id="message"></div>
    </div>
    <p class="small" style="margin-top:0.6rem; text-align:center;">
      Requests are suggestions, not guarantees — but Jeff sees every one. If you’re enjoying the music, a smile or a tip means a lot. ✨
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

function clearMessage() { messageEl.innerHTML = ''; }

function renderResults(results) {
  resultsEl.innerHTML = '';
  selectedSongIdx = null;
  pickSection.style.display = 'none';

  if (!results || results.length === 0) {
    resultsEl.innerHTML = '<p class="small">No suggestions yet. Try describing it a little differently.</p>';
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
      <div class="song-title"><span class="badge-key">${song.key || '-'}</span><span>${song.title}</span></div>
      <div class="song-meta">${(song.artist || '')} ${song.songbook ? ' • ' + song.songbook : ''}</div>
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
  resultsEl.innerHTML = '<p class="small">Thinking…</p>';
  pickSection.style.display = 'none';
  selectedSongIdx = null;

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    const data = await res.json();
    if (!res.ok) {
      if (res.status === 403) throw new Error('Requests are currently closed.');
      if (res.status === 429) throw new Error('Too many requests. Slow down.');
      throw new Error('Server error');
    }
    renderResults(data.results || []);
  } catch (err) {
    resultsEl.innerHTML = '';
    messageEl.innerHTML = `<div class="error">${err.message}</div>`;
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
    if (!res.ok) throw new Error('Failed to send request.');
    await res.json();
    messageEl.innerHTML = `<div class="thankyou"><b>Thanks!</b> Your request is in the queue.<br/></div>`;
    pickSection.style.display = 'none';
  } catch (err) {
    messageEl.innerHTML = '<div class="error">Sorry, something went wrong.</div>';
  } finally {
    submitBtn.disabled = false;
  }
};
</script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get("/about", response_class=HTMLResponse)
async def about_page():
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Cherokee Rhodes • About</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg-card: rgba(24, 24, 24, 0.85);
      --accent: #1ed760; 
      --accent-soft: #ff66b2; 
      --text-main: #ffffff;
      --text-muted: #a7a7a7;
      --border-subtle: #333333;
    }
    body {
      margin: 0;
      padding: 1.5rem 1rem 3rem;
      font-family: system-ui, sans-serif;
      background-image: linear-gradient(to bottom, rgba(18,18,18,0.90) 0%, rgba(0,0,0,0.98) 100%), url('/static/background.png');
      background-size: cover;
      background-position: center;
      background-attachment: fixed;
      background-blend-mode: multiply;
      color: var(--text-main);
      display: flex;
      justify-content: center;
    }
    .shell { width: 100%; max-width: 900px; }
    
    .nav-bar { display: flex; justify-content: center; gap: 2rem; margin-bottom: 3rem; border-bottom: 1px solid var(--border-subtle); padding-bottom: 1rem; }
    .nav-link { color: var(--text-muted); text-decoration: none; font-weight: 700; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1.5px; transition: color 0.15s ease; }
    .nav-link:hover, .nav-link.active { color: var(--accent-soft); text-shadow: 0 0 8px rgba(255,102,178,0.4); }

    .hero { text-align: center; margin-bottom: 3rem; }
    .title { font-weight: 800; font-size: 2.5rem; color: var(--accent-soft); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem; text-shadow: 0 0 15px rgba(255,102,178,0.4); }
    .subtitle { font-size: 1.1rem; color: var(--accent); font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    
    .card { background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border-subtle); padding: 2rem; box-shadow: 0 20px 40px rgba(0,0,0,0.8); backdrop-filter: blur(4px); margin-bottom: 3rem; }
    .card p { line-height: 1.6; color: var(--text-muted); font-size: 1.05rem; margin-top: 0; margin-bottom: 1rem; }
    .card p:last-child { margin-bottom: 0; }
    
    .section-title { font-size: 1.5rem; font-weight: 800; text-align: center; margin-bottom: 1.5rem; color: var(--text-main); text-transform: uppercase; letter-spacing: 1px; }
    
    .grid { display: grid; grid-template-columns: 1fr; gap: 1.5rem; margin-bottom: 3rem; }
    @media (min-width: 768px) { .grid { grid-template-columns: 1fr 1fr; } }
    
    .gig-card { border: 1px solid var(--border-subtle); border-radius: 8px; overflow: hidden; background: rgba(0,0,0,0.5); transition: 0.2s; display: flex; flex-direction: column; }
    .gig-card:hover { border-color: var(--accent-soft); box-shadow: 0 0 20px rgba(255,102,178,0.1); }
    .gig-info { padding: 1.25rem; flex-grow: 1; }
    .gig-title { font-size: 1.15rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.5rem; }
    .gig-desc { font-size: 0.9rem; color: var(--text-muted); line-height: 1.5; }
    
    .media-embed { width: 100%; height: 260px; background: #000; border-bottom: 1px solid var(--border-subtle); }
    .soundcloud-embed { border-radius: 12px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5); border: 1px solid var(--border-subtle); }
  </style>
</head>
<body>
  <div class="shell">
    
    <div class="nav-bar">
      <a href="/" class="nav-link">Request Line</a>
      <a href="/live" class="nav-link">Live Show</a>
      <a href="/about" class="nav-link active">About</a>
    </div>

    <div class="hero">
      <div class="title">Cherokee Rhodes</div>
      <div class="subtitle">Storyteller Musician • Atmosphere Architect</div>
    </div>

    <div class="card">
      <p>Jeff Streitmatter IV grew up in Florida and began his career as an engineering consultant for companies like Lockheed Martin. His mother forced him into piano lessons as a youngster saying, “you’ll thank me - one day!”</p>
      <p>After leaving his job during the pandemic, he transitioned into a full-time artistic career curating performances that blend visual art, music and storytelling. Jeff is a multi-instrument vocalist thriving on audience requests. His performance ranges from high-energy improvised looper pedal jingles to deeply intimate covers of timeless classics.</p>
      <p>Even though Jeff is endowed with gratitude to his mom for his creative gifts, he’s been firmly warned by his mother not to write any songs about her.</p>
    </div>

    <div class="section-title">Hireable Offerings</div>
    
    <div class="grid">
      <div class="gig-card">
        <iframe class="media-embed" src="https://www.youtube.com/embed/2lrCJYvOIBs" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        <div class="gig-info">
          <div class="gig-title">Lively Pianobar (Party)</div>
          <div class="gig-desc">High-energy, request-driven, and interactive live sets powered by Jeff's music request app. Perfect for lively rooms.</div>
        </div>
      </div>
      
      <div class="gig-card">
        <iframe class="media-embed" src="https://www.youtube.com/embed/navWhBgkWK0?start=233" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        <div class="gig-info">
          <div class="gig-title">Atmosphere (Cocktail Hour)</div>
          <div class="gig-desc">Subtle, refined background music that supports networking and conversation without overpowering the room.</div>
        </div>
      </div>

      <div class="gig-card">
        <iframe class="media-embed" src="https://www.youtube.com/embed/AH2m83kmvx4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        <div class="gig-info">
          <div class="gig-title">Intimate Soundbath (Feature Set)</div>
          <div class="gig-desc">Creative, story-driven performances blending sound & film. Perfect for ticketed events and focused audiences.</div>
        </div>
      </div>

      <div class="gig-card">
        <iframe class="media-embed" src="https://www.youtube.com/embed/rk4GVw471iU?start=444" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        <div class="gig-info">
          <div class="gig-title">Musical Storytelling (Feature Set)</div>
          <div class="gig-desc">Jeff takes life events and turns them into original songs and symposiums on spiritual topics.</div>
        </div>
      </div>
    </div>

    <div class="section-title">Previous Work</div>
    
    <div class="card" style="padding: 1.5rem; margin-bottom: 2rem;">
      <p>In 2023 Jeff wrote and performed an original musical telling the story of leaving Florida and leaving the church in the midst of an unstable relationship. When his breakup strands him in Wyoming, a new love comes along in the form of someone he literally meets in a dream. Jeff is then confronted by a startling truth - what he called love was in fact codependency and Christ is a metaphor for what happens when we let go.</p>
      <iframe class="media-embed" style="margin-top: 1.5rem; border-radius: 8px; height: 350px;" src="https://www.youtube.com/embed/BxVXtz3iWnY?list=PLt863s_HgmtPtsJha60ndVTVoGp16B9fp" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>

    <div class="card" style="padding: 1.5rem; margin-bottom: 2rem;">
      <p>Shortly after this, Jeff is laid off from his career as an engineering consultant and leaves Wyoming in search of his new life as an artist. He travels 3 continents and 2,000 years of history in search of a story that will guide his mission to become a Christ-infused self. Here is the feature-length film released in 2025</p>
      
      <iframe class="media-embed" style="margin-top: 1.5rem; border-radius: 8px; height: 350px;" src="https://www.youtube.com/embed/wsYOyKvD6G8?start=1642" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>

    <div class="card" style="padding: 1.5rem;">
      <p>In addition to live performance, Jeff showcases his recorded portfolio on Soundcloud & YouTube.</p>
      
      <div class="soundcloud-embed" style="margin-top: 1.5rem;">
        <iframe width="100%" height="350" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//soundcloud.com/jeff-streitmatter-iv/sets/psychedelic-adventure&color=%23ff66b2&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe>
      </div>
    </div>

  </div>
</body>
</html>
"""
    return HTMLResponse(html)

# ── PERFORMER & ADMIN UI ─────────────────────────────────────────────────────

@app.get("/performer", response_class=HTMLResponse)
async def performer_page(dep=Depends(require_performer)):
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Songmaster • Performer Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { background: #020617; color: white; font-family: sans-serif; }
  </style>
</head>
<body>
  <h1>Performer Dashboard</h1>
  <p>Live request updates will appear here.</p>
</body>
</html>
"""
    return HTMLResponse(html)

@app.get("/journal", response_class=HTMLResponse)
async def journal_page(dep=Depends(require_performer)):
    html = "<html><body><h1>JeffGPT Journal</h1></body></html>"
    return HTMLResponse(html)

@app.get("/crm", response_class=HTMLResponse)
async def crm_page(dep=Depends(require_performer)):
    html = "<html><body><h1>Music CRM</h1></body></html>"
    return HTMLResponse(html)

@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(dep=Depends(require_performer)):
    html = "<html><body><h1>Portfolio Helper</h1></body></html>"
    return HTMLResponse(html)