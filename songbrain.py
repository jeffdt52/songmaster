#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
songbrain.py – Core Songmaster brain

Responsibilities:
- Load "Songbook Data.xlsx" into Song objects.
- Load enriched metadata (songbook_enriched.json) if present.
- Load or build embeddings (songbook_embeddings.npy).
- Provide a simple API:

    brain = SongBrain()
    results = brain.search("query text", top_k=10)

Each result is a dict:
    {
      "song_idx": int,
      "key": str,
      "title": str,
      "artist": str,
      "category": str,
      "songbook": str,
      "group": str,
    }

Config via env vars:
- SONGMASTER_XLSX_PATH       – override Excel path (optional)
- SONGMASTER_USE_GPT_RERANK  – "true"/"false" (default: true)
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv  # <<< NEW

# Load .env so OPENAI_API_KEY etc. are available, like your CLI
load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_EXCEL_PATH = SCRIPT_DIR / "Songbook Data.xlsx"
EXCEL_PATH = Path(os.environ.get("SONGMASTER_XLSX_PATH", DEFAULT_EXCEL_PATH))

BASE_DIR = SCRIPT_DIR
EMBED_PATH = BASE_DIR / "songbook_embeddings.npy"
META_PATH = BASE_DIR / "songbook_embeddings_meta.json"
ENRICHED_PATH = BASE_DIR / "songbook_enriched.json"

EMBED_MODEL = "text-embedding-3-small"
ENRICHED_MODE = "enriched_lyrics_v2"
BASIC_MODE = "basic_v1"
ENRICHED_VERSION = 2  # keep in sync with your build-db

MAX_LYRICS_CHARS = 1200
DEFAULT_TOP_K = 10
CANDIDATE_POOL_SIZE = 60

GPT_RERANK_MODEL = "gpt-4.1-mini"


# ── DATA MODEL ────────────────────────────────────────────────────────────────

@dataclass
class Song:
    idx: int
    key: str
    title: str
    category: str
    songbook: str
    artist: str
    group: str
    notes: str
    lyrics: str

    def descriptor_basic(self) -> str:
        lyrics_slice = (self.lyrics or "").strip()
        if len(lyrics_slice) > MAX_LYRICS_CHARS:
            lyrics_slice = lyrics_slice[:MAX_LYRICS_CHARS]

        parts = [
            f"Song: {self.title}",
            f"Artist: {self.artist}" if self.artist else "",
            f"Key: {self.key}" if self.key else "",
            f"Category: {self.category}" if self.category else "",
            f"Songbook: {self.songbook}" if self.songbook else "",
            f"Group: {self.group}" if self.group else "",
            f"Notes: {self.notes}" if self.notes else "",
            f"Lyrics (fragment): {lyrics_slice}" if lyrics_slice else "",
        ]
        return ". ".join(p for p in parts if p)


# ── LOADING SONGBOOK ─────────────────────────────────────────────────────────

def load_songbook(path: Path) -> List[Song]:
    if not path.exists():
        raise FileNotFoundError(f"Songbook Excel not found at: {path}")

    try:
        df = pd.read_excel(path)
    except ImportError as e:
        raise RuntimeError(
            "Could not read Excel file. You probably need:\n"
            "  pip install openpyxl\n"
        ) from e

    required_cols = ["Key", "Song", "Genre", "Songbook", "Artist", "Group", "Notes"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Songbook is missing columns: {', '.join(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    has_lyrics_col = "Lyrics" in df.columns

    songs: List[Song] = []
    for i, row in df.iterrows():
        def safe_str(val) -> str:
            if pd.isna(val):
                return ""
            return str(val).strip()

        lyrics_val = safe_str(row["Lyrics"]) if has_lyrics_col else ""
        title_raw = safe_str(row["Song"])
        title_caps = title_raw.upper()

        songs.append(
            Song(
                idx=i,
                key=safe_str(row["Key"]),
                title=title_caps,
                category=safe_str(row["Genre"]),
                songbook=safe_str(row["Songbook"]),
                artist=safe_str(row["Artist"]),
                group=safe_str(row["Group"]),
                notes=safe_str(row["Notes"]),
                lyrics=lyrics_val,
            )
        )
    return songs


# ── ENRICHED METADATA ─────────────────────────────────────────────────────────

def load_enriched_metadata() -> Dict[str, Any]:
    if not ENRICHED_PATH.exists():
        return {}
    try:
        with ENRICHED_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def enriched_descriptor_for_song(song: Song, enriched: Dict[str, Any]) -> str:
    entry = enriched.get(str(song.idx)) or {}
    meta = entry.get("data", entry)

    mood_words = ", ".join(meta.get("mood_words", []))
    energy = meta.get("energy", "")
    tempo_label = meta.get("tempo_label", "")
    tempo_bpm = meta.get("tempo_bpm_guess", "")
    primary_era = meta.get("primary_era", "")
    release_decade = meta.get("release_decade", "")
    themes = ", ".join(meta.get("lyrical_themes", []))
    vibe_tags = ", ".join(meta.get("vibe_tags", []))
    style_tags = ", ".join(meta.get("style_tags", []))
    context = meta.get("recommended_set_context", "")
    comment = meta.get("comment_for_performer", "")
    emotional_arc = meta.get("emotional_arc", "")
    difficulty = meta.get("difficulty", "")
    tech_diff = meta.get("tech_difficulty_solo_piano", "")
    lyrics_summary = meta.get("lyrics_summary", "")
    time_signature = meta.get("time_signature", "")
    is_waltz = meta.get("is_waltz", False)
    is_movie = meta.get("is_movie_or_musical_song", False)
    fam_teens = meta.get("familiarity_teens", "")
    fam_seniors = meta.get("familiarity_seniors", ""

    )

    lyrics_slice = (song.lyrics or "").strip()
    if len(lyrics_slice) > MAX_LYRICS_CHARS:
        lyrics_slice = lyrics_slice[:MAX_LYRICS_CHARS]

    parts = [
        f"Song: {song.title}",
        f"Artist: {song.artist}" if song.artist else "",
        f"Key: {song.key}" if song.key else "",
        f"Category: {song.category}" if song.category else "",
        f"Songbook: {song.songbook}" if song.songbook else "",
        f"Group: {song.group}" if song.group else "",
        f"Primary era: {primary_era}" if primary_era else "",
        f"Release decade: {release_decade}" if release_decade else "",
        f"Mood: {mood_words}" if mood_words else "",
        f"Energy: {energy}" if energy else "",
        f"Tempo: {tempo_label} around {tempo_bpm} BPM" if (tempo_label or tempo_bpm) else "",
        f"Themes: {themes}" if themes else "",
        f"Vibe: {vibe_tags}" if vibe_tags else "",
        f"Style tags: {style_tags}" if style_tags else "",
        f"Emotional arc: {emotional_arc}" if emotional_arc else "",
        f"Difficulty: {difficulty}" if difficulty else "",
        f"Technical difficulty solo piano: {tech_diff}" if tech_diff != "" else "",
        f"Time signature: {time_signature}" if time_signature else "",
        f"Is waltz: {is_waltz}" if time_signature or is_waltz else "",
        f"Is movie or musical song: {is_movie}" if is_movie else "",
        f"Familiarity teens: {fam_teens}" if fam_teens != "" else "",
        f"Familiarity seniors: {fam_seniors}" if fam_seniors != "" else "",
        f"Lyrics summary: {lyrics_summary}" if lyrics_summary else "",
        f"Lyrics (fragment): {lyrics_slice}" if lyrics_slice else "",
        f"Set context: {context}" if context else "",
        f"Comment: {comment}" if comment else "",
        f"Original notes: {song.notes}" if song.notes else "",
    ]
    return ". ".join(p for p in parts if p)


# ── EMBEDDINGS ───────────────────────────────────────────────────────────────

def compute_embeddings_from_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    batch_size = 64
    vectors: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for d in resp.data:
            vectors.append(d.embedding)

    mat = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return mat


def load_or_build_embeddings(
    client: OpenAI,
    songs: List[Song],
    enriched: Dict[str, Any],
) -> np.ndarray:
    excel_mtime = EXCEL_PATH.stat().st_mtime
    num_songs = len(songs)

    enriched_available = bool(enriched)
    expected_mode = ENRICHED_MODE if enriched_available else BASIC_MODE

    if EMBED_PATH.exists() and META_PATH.exists():
        try:
            with META_PATH.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            if (
                meta.get("num_songs") == num_songs
                and abs(meta.get("excel_mtime", 0) - excel_mtime) < 1
                and meta.get("embed_model") == EMBED_MODEL
                and meta.get("mode") == expected_mode
            ):
                mat = np.load(EMBED_PATH)
                if mat.shape[0] == num_songs:
                    return mat
        except Exception:
            pass

    if enriched_available:
        texts = [enriched_descriptor_for_song(s, enriched) for s in songs]
        mode = ENRICHED_MODE
    else:
        texts = [s.descriptor_basic() for s in songs]
        mode = BASIC_MODE

    mat = compute_embeddings_from_texts(client, texts)

    meta = {
        "num_songs": num_songs,
        "excel_mtime": excel_mtime,
        "embed_model": EMBED_MODEL,
        "mode": mode,
    }
    np.save(EMBED_PATH, mat)
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return mat


# ── GPT RE-RANK (FULL INTELLIGENCE) ──────────────────────────────────────────

def gpt_rerank_candidates(
    client: OpenAI,
    query: str,
    songs: List[Song],
    candidate_indices: List[int],
    enriched: Dict[str, Any],
) -> List[int]:
    if not candidate_indices or not enriched:
        return candidate_indices

    candidates = []
    id_to_global: Dict[int, int] = {}

    for local_id, global_idx in enumerate(candidate_indices):
        id_to_global[local_id] = global_idx
        s = songs[global_idx]
        entry = enriched.get(str(s.idx)) or {}
        meta = entry.get("data", entry)

        candidates.append({
            "candidate_id": local_id,
            "title": s.title,
            "artist": s.artist,
            "key": s.key,
            "category": s.category,
            "songbook": s.songbook,
            "group": s.group,
            "notes": s.notes,
            "mood_words": meta.get("mood_words", []),
            "lyrical_themes": meta.get("lyrical_themes", []),
            "vibe_tags": meta.get("vibe_tags", []),
            "style_tags": meta.get("style_tags", []),
            "time_signature": meta.get("time_signature", ""),
            "is_waltz": meta.get("is_waltz", False),
            "primary_era": meta.get("primary_era", ""),
            "release_decade": meta.get("release_decade", ""),
            "is_movie_or_musical_song": meta.get("is_movie_or_musical_song", False),
            "familiarity_teens": meta.get("familiarity_teens", 0),
            "familiarity_seniors": meta.get("familiarity_seniors", 0),
            "tech_difficulty_solo_piano": meta.get("tech_difficulty_solo_piano", 0),
            "emotional_arc": meta.get("emotional_arc", ""),
            "lyrics_summary": meta.get("lyrics_summary", ""),
        })

    payload = {
        "query": query,
        "candidates": candidates,
    }

    system_msg = (
        "You are an expert musician and setlist curator for a solo pianist. "
        "Given a free-text query and a list of candidate songs with metadata, "
        "choose and order the best songs. "
        "Honor constraints like era, difficulty, waltz/3-4, folk, hair rock, "
        "teen/senior familiarity, 'not depressing', etc. "
        "Return JSON with key 'ordered_candidate_ids' listing the candidate_id values. "
        "Do not invent songs; only use the given candidates."
    )

    user_msg = json.dumps(payload, ensure_ascii=False)

    try:
        resp = client.chat.completions.create(
            model=GPT_RERANK_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        text = resp.choices[0].message.content.strip()
        data = json.loads(text)
        ordered_ids = data.get("ordered_candidate_ids") or data.get("candidate_ids")
        if not isinstance(ordered_ids, list):
            return candidate_indices

        new_global: List[int] = []
        seen = set()
        for cid in ordered_ids:
            try:
                cid_int = int(cid)
            except Exception:
                continue
            if cid_int in id_to_global:
                gi = id_to_global[cid_int]
                if gi not in seen:
                    new_global.append(gi)
                    seen.add(gi)

        return new_global or candidate_indices
    except Exception:
        # On any error, fall back to embedding order
        return candidate_indices


# ── SONGBRAIN CLASS ──────────────────────────────────────────────────────────

class SongBrain:
    """
    Reusable core object:

        brain = SongBrain()
        results = brain.search("songs teenagers and old people would know", top_k=25)

    GPT re-rank is ON by default; disable only if you intentionally set
    SONGMASTER_USE_GPT_RERANK=false in the environment.
    """

    def __init__(
        self,
        excel_path: Optional[Path] = None,
        use_gpt_rerank: Optional[bool] = None,
    ):
        self.client = OpenAI()
        self.excel_path = excel_path or EXCEL_PATH

        env_flag = os.environ.get("SONGMASTER_USE_GPT_RERANK", "").strip().lower()
        if use_gpt_rerank is not None:
            self.use_gpt_rerank = use_gpt_rerank
        else:
            # default: TRUE (full intelligence)
            self.use_gpt_rerank = env_flag not in ("0", "false", "no")

        self.songs: List[Song] = load_songbook(self.excel_path)
        self.enriched: Dict[str, Any] = load_enriched_metadata()
        self.embeddings: np.ndarray = load_or_build_embeddings(
            self.client, self.songs, self.enriched
        )

    def _search_core(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Tuple[int, float]]:
        if not query.strip():
            return []

        resp = self.client.embeddings.create(model=EMBED_MODEL, input=[query])
        q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            q_norm = 1.0
        q_vec = q_vec / q_norm

        scores = self.embeddings @ q_vec
        num_songs = len(self.songs)

        if CANDIDATE_POOL_SIZE >= num_songs:
            candidate_indices = list(np.argsort(-scores))
        else:
            cand_idx = np.argpartition(-scores, CANDIDATE_POOL_SIZE)[:CANDIDATE_POOL_SIZE]
            candidate_indices = list(cand_idx[np.argsort(-scores[cand_idx])])

        if self.use_gpt_rerank and self.enriched:
            candidate_indices = gpt_rerank_candidates(
                self.client, query, self.songs, candidate_indices, self.enriched
            )

        results = [(idx, float(scores[idx])) for idx in candidate_indices]

        if len(results) > top_k:
            results = results[:top_k]

        return results

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Public API: returns list of song dicts, sorted for display.
        """
        pairs = self._search_core(query, top_k=top_k)

        decorated = []
        for idx, score in pairs:
            s = self.songs[idx]
            decorated.append((
                s.key or "",
                s.category or "",
                -score,
                s.title or "",
                idx,
            ))
        decorated.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

        results: List[Dict[str, Any]] = []
        for _key, _cat, _ns, _title, idx in decorated:
            s = self.songs[idx]
            results.append({
                "song_idx": idx,
                "key": s.key,
                "title": s.title,
                "artist": s.artist,
                "category": s.category,
                "songbook": s.songbook,
                "group": s.group,
            })
        return results
