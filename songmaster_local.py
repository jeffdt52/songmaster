#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_songmaster.py

CLI Songmaster agent on top of your "Songbook Data.xlsx".

Two modes:

1) Enrichment / DB build (run occasionally):
   python agent_songmaster.py build-db

   - Loads songs from Excel.
   - For each song, calls GPT to infer mood/era/themes/etc., plus structured fields:
       * time_signature, is_waltz
       * release_decade, style_tags
       * familiarity_teens, familiarity_seniors
       * tech_difficulty_solo_piano
       * is_movie_or_musical_song
   - Creates a compact lyrics summary (no direct quotes).
   - Saves to songbook_enriched.json
   - Builds embeddings from rich text (title/artist/category + lyrics + summary + tags)
     and caches them.

2) Interactive CLI (run frequently):
   python agent_songmaster.py

   - Loads songs from Excel.
   - Loads enriched DB and embeddings if present.
   - For each query:
       * Uses embeddings to grab a candidate pool.
       * Calls GPT to re-rank/filter candidates intelligently based on your query
         (audience, decade, style, difficulty, etc.).
       * Displays results grouped by KEY then CATEGORY.
       * POPULAR songs (Songbook containing 'Popular') are bolded.
       * Movie & video Games are colored PINK (bright magenta).
       * Alternative / Weird is left uncolored (white).

Commands:
    query <text>
    group <idx...> <group_name>
    groups
    show
    show <group_name>
    help
    exit / quit

Dependencies (install in your venv):
    pip install openai pandas numpy openpyxl
"""

import os
import sys
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from openai import OpenAI

# ── ANSI COLORS ───────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

FG_BLACK = "\033[30m"
FG_RED = "\033[31m"
FG_GREEN = "\033[32m"
FG_YELLOW = "\033[33m"
FG_BLUE = "\033[34m"
FG_MAGENTA = "\033[35m"
FG_CYAN = "\033[36m"
FG_WHITE = "\033[37m"

# Bright variants
FG_BRIGHT_RED = "\033[91m"
FG_BRIGHT_GREEN = "\033[92m"
FG_BRIGHT_YELLOW = "\033[93m"
FG_BRIGHT_BLUE = "\033[94m"
FG_BRIGHT_MAGENTA = "\033[95m"  # PINK-ish
FG_BRIGHT_CYAN = "\033[96m"
FG_BRIGHT_WHITE = "\033[97m"

FG_PINK = FG_BRIGHT_MAGENTA  # alias for clarity

# Category color mapping
# NOTE: Alternative / Weird intentionally left as RESET (no color)
#       Movie & video Games gets PINK
CATEGORY_COLORS = {
    "Alternative / Weird": RESET,         # no color
    "Classic Rock & Pop hits": FG_CYAN,   # POPULAR = will be bolded separately
    "Country, Oldies & Folk": FG_RED,
    "Jazz & Soul": FG_GREEN,
    "Movie & video Games": FG_PINK,       # movie music highlighted
    "Ragtime, Gospel & Patriotic": FG_BLUE,
}

# ── CONFIG ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

# Default path; you can override with SONGMASTER_XLSX_PATH env var
DEFAULT_EXCEL_PATH = SCRIPT_DIR / "Songbook Data.xlsx"
EXCEL_PATH = Path(os.environ.get("SONGMASTER_XLSX_PATH", DEFAULT_EXCEL_PATH))

# Where to store embedding cache and enrichment (next to this script)
BASE_DIR = SCRIPT_DIR
EMBED_PATH = BASE_DIR / "songbook_embeddings.npy"
META_PATH = BASE_DIR / "songbook_embeddings_meta.json"
ENRICHED_PATH = BASE_DIR / "songbook_enriched.json"

EMBED_MODEL = "text-embedding-3-small"
ENRICHED_MODE = "enriched_lyrics_v2"
BASIC_MODE = "basic_v1"
ENRICHED_VERSION = 2  # bump when schema changes

# How many results to show for a query
DEFAULT_TOP_K = 25

# Candidate pool size for GPT re-rank
CANDIDATE_POOL_SIZE = 60

# Limit lyrics length fed into embeddings (to keep cost reasonable)
MAX_LYRICS_CHARS = 1200

# GPT model for enrichment and re-ranking
GPT_ENRICH_MODEL = "gpt-4.1-mini"
GPT_RERANK_MODEL = "gpt-4.1-mini"

# ── DATA MODEL ────────────────────────────────────────────────────────────────

@dataclass
class Song:
    idx: int                # index in the songs list (row index)
    key: str
    title: str
    category: str
    songbook: str
    artist: str
    group: str
    notes: str
    lyrics: str             # optional, from Excel "Lyrics" column

    def descriptor_basic(self) -> str:
        """
        Basic text description used for embeddings when enriched data is absent.
        Includes lyrics if present (trimmed).
        """
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


# ── OPENAI CLIENT ─────────────────────────────────────────────────────────────

def make_client() -> OpenAI:
    # Uses OPENAI_API_KEY from environment
    return OpenAI()


# ── LOADING SONGBOOK ─────────────────────────────────────────────────────────

def load_songbook(path: Path) -> List[Song]:
    if not path.exists():
        print(f"{FG_RED}ERROR:{RESET} Songbook Excel not found at:\n  {path}")
        print("Move 'Songbook Data.xlsx' next to agent_songmaster.py,")
        print("or set SONGMASTER_XLSX_PATH to the full path.")
        sys.exit(1)

    try:
        df = pd.read_excel(path)
    except ImportError:
        print(f"{FG_RED}ERROR:{RESET} Could not read Excel file.")
        print("You probably need to install the Excel engine dependency:")
        print("  pip install openpyxl")
        print()
        raise

    # Expected core columns
    required_cols = ["Key", "Song", "Genre", "Songbook", "Artist", "Group", "Notes"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"{FG_RED}ERROR:{RESET} Songbook is missing columns: {', '.join(missing)}")
        print(f"Columns found: {list(df.columns)}")
        sys.exit(1)

    # Optional lyrics column
    has_lyrics_col = "Lyrics" in df.columns

    songs: List[Song] = []
    for i, row in df.iterrows():
        def safe_str(val) -> str:
            if pd.isna(val):
                return ""
            return str(val).strip()

        lyrics_val = safe_str(row["Lyrics"]) if has_lyrics_col else ""

        # Normalize title to ALL CAPS so everything is consistent
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


def save_enriched_metadata(enriched: Dict[str, Any]) -> None:
    with ENRICHED_PATH.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)


def enriched_descriptor_for_song(song: Song, enriched: Dict[str, Any]) -> str:
    """
    Build a rich descriptor string from base song info + enriched metadata.
    This is what we embed for semantic search.

    No lexical flags — just dense text:
    - title/artist/key/category/songbook
    - mood words, themes, emotional arc
    - style_tags, release_decade, time_signature, difficulty, familiarity
    - paraphrased lyrics summary
    - optional actual lyrics text (truncated) if present
    """
    entry = enriched.get(str(song.idx)) or {}
    meta = entry.get("data", entry)  # tolerant if we ever store flat dicts

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
    fam_seniors = meta.get("familiarity_seniors", "")

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


# ── EMBEDDING UTILITIES ──────────────────────────────────────────────────────

def compute_embeddings_from_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    vectors: List[List[float]] = []
    batch_size = 64

    print(f"{DIM}Computing embeddings for {len(texts)} items (model: {EMBED_MODEL})...{RESET}")

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for d in resp.data:
            vectors.append(d.embedding)

        done = min(start + batch_size, len(texts))
        pct = (done / len(texts)) * 100
        print(f"{DIM}  -> {done}/{len(texts)} ({pct:.1f}%) done{RESET}", end="\r", flush=True)

    print()
    mat = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return mat


def load_or_build_embeddings(
    client: OpenAI,
    songs: List[Song],
) -> np.ndarray:
    """
    Decide whether to use enriched mode or basic mode, and either:
    - load cached embeddings if they match, or
    - compute and cache new embeddings.
    """
    excel_mtime = EXCEL_PATH.stat().st_mtime
    num_songs = len(songs)

    enriched_available = ENRICHED_PATH.exists()
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

    if enriched_available and expected_mode == ENRICHED_MODE:
        print(f"{DIM}Using enriched lyrics-centric metadata for embeddings...{RESET}")
        enriched = load_enriched_metadata()
        texts = [enriched_descriptor_for_song(s, enriched) for s in songs]
        mat = compute_embeddings_from_texts(client, texts)
        meta = {
            "num_songs": num_songs,
            "excel_mtime": excel_mtime,
            "embed_model": EMBED_MODEL,
            "mode": ENRICHED_MODE,
        }
    else:
        print(f"{DIM}No enriched DB; using basic descriptors (with lyrics if present)...{RESET}")
        texts = [s.descriptor_basic() for s in songs]
        mat = compute_embeddings_from_texts(client, texts)
        meta = {
            "num_songs": num_songs,
            "excel_mtime": excel_mtime,
            "embed_model": EMBED_MODEL,
            "mode": BASIC_MODE,
        }

    np.save(EMBED_PATH, mat)
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return mat


# ── GPT RE-RANK LAYER ────────────────────────────────────────────────────────

def gpt_rerank_candidates(
    client: OpenAI,
    query: str,
    songs: List[Song],
    candidate_indices: List[int],
    enriched: Dict[str, Any],
) -> List[int]:
    """
    Second layer: given a query + candidate songs (from embeddings),
    call GPT to re-rank/filter them like a smart music brain.

    We send compact metadata plus enriched fields and ask GPT for
    an ordered list of candidate_ids (0..N-1).
    """
    if not candidate_indices:
        return candidate_indices

    # Build candidates list with small local IDs
    candidates = []
    id_to_global: Dict[int, int] = {}
    for local_id, global_idx in enumerate(candidate_indices):
        id_to_global[local_id] = global_idx
        s = songs[global_idx]
        entry = enriched.get(str(s.idx)) or {}
        meta = entry.get("data", entry)

        cand = {
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
        }
        candidates.append(cand)

    payload = {
        "query": query,
        "candidates": candidates,
    }

    system_msg = (
        "You are an expert musician and setlist curator for a solo pianist. "
        "You know music history, genres, decades, styles, and how different audiences respond "
        "to songs. You are helping choose songs from the user's personal catalog.\n\n"
        "Rules:\n"
        "- Read the user's free-text query carefully. Honor ALL explicit constraints if possible.\n"
        "- Use the provided metadata fields (style_tags, release_decade, time_signature, "
        "is_waltz, is_movie_or_musical_song, familiarity_teens, familiarity_seniors, "
        "tech_difficulty_solo_piano, lyrical_themes, etc.) to filter.\n"
        "- When the query names a specific show or work (e.g. 'Phantom of the Opera') and "
        "does NOT say 'similar' or 'like', only return songs actually from that work if any exist.\n"
        "- If the query mentions 'waltz' or '3/4', only return songs that are waltzes "
        "(time_signature like 3/4 or 6/8, or is_waltz == true).\n"
        "- If the query mentions '80s' or '1980s', strongly prefer songs with release_decade == '1980s'. "
        "Only relax this if there are essentially zero matches.\n"
        "- If the query mentions 'hair rock' or 'hair metal', prefer style_tags containing those phrases "
        "or '80s rock', 'arena rock', 'glam rock'.\n"
        "- If the query mentions 'folk', prefer style_tags including 'folk', 'folk ballad', "
        "'americana', etc.\n"
        "- If the query mentions 'technically difficult' or 'hard for solo piano', "
        "filter for high tech_difficulty_solo_piano and exclude obviously easy tunes.\n"
        "- If the query mentions 'teenagers and old people', prefer songs where both "
        "familiarity_teens and familiarity_seniors are high.\n"
        "- If the query mentions 'not depressing' or 'not too sad', avoid songs whose "
        "emotional_arc is very dark or heavy.\n"
        "- It's better to return a narrower, more accurate set than a huge vague one.\n\n"
        "Output: a single JSON object with a list field named 'ordered_candidate_ids' "
        "containing the candidate_ids in the desired order. "
        "Do not invent songs; only choose from the provided candidates."
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
        ordered_ids = (
            data.get("ordered_candidate_ids")
            or data.get("chosen_ids")
            or data.get("ordered_ids")
            or data.get("candidate_ids")
        )

        if not isinstance(ordered_ids, list):
            raise ValueError("No ordered ids in GPT response")

        new_global_indices: List[int] = []
        seen = set()
        for cid in ordered_ids:
            try:
                cid_int = int(cid)
            except Exception:
                continue
            if cid_int in id_to_global:
                gi = id_to_global[cid_int]
                if gi not in seen:
                    new_global_indices.append(gi)
                    seen.add(gi)

        # Fallback if GPT returned empty or nonsense
        if not new_global_indices:
            return candidate_indices

        return new_global_indices

    except Exception:
        # On any error, just use the original candidate order
        return candidate_indices


# ── SEARCH / QUERY ────────────────────────────────────────────────────────────

def sort_results_for_display(
    songs: List[Song],
    results: List[Tuple[int, float]],
) -> List[Tuple[int, float]]:
    """
    Sort results for display:
    1) by Key
    2) by Category
    3) by descending score
    4) by Title
    """
    decorated = []
    for (song_idx, score) in results:
        s = songs[song_idx]
        key = s.key or ""
        category = s.category or ""
        title = s.title or ""
        decorated.append((key, category, -score, title, song_idx, score))

    decorated.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return [(song_idx, score) for (_k, _c, _ns, _t, song_idx, score) in decorated]


def search_songs(
    client: OpenAI,
    songs: List[Song],
    embeddings: np.ndarray,
    query: str,
    enriched: Dict[str, Any],
    top_k: int = DEFAULT_TOP_K,
) -> List[Tuple[int, float]]:
    """
    Two-layer search:

    1) Embeddings over whole catalog to get a candidate pool.
    2) GPT re-rank/filter using rich metadata and your query.
    3) Sort final results by key, then category, then score for display.

    Returns list of (song_idx, score).
    """
    if not query.strip():
        return []

    # Embedding for query
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        q_norm = 1.0
    q_vec = q_vec / q_norm

    scores = embeddings @ q_vec  # shape: (num_songs,)

    num_songs = len(songs)
    if CANDIDATE_POOL_SIZE >= num_songs:
        candidate_indices = list(np.argsort(-scores))
    else:
        cand_idx = np.argpartition(-scores, CANDIDATE_POOL_SIZE)[:CANDIDATE_POOL_SIZE]
        candidate_indices = list(cand_idx[np.argsort(-scores[cand_idx])])

    # GPT re-rank layer if we have enriched metadata
    if enriched:
        candidate_indices = gpt_rerank_candidates(
            client=client,
            query=query,
            songs=songs,
            candidate_indices=candidate_indices,
            enriched=enriched,
        )

    # Now build (song_idx, score) pairs from the re-ranked list
    results: List[Tuple[int, float]] = [(idx, float(scores[idx])) for idx in candidate_indices]

    # Limit to top_k
    if len(results) > top_k:
        results = results[:top_k]

    # Sort for display: by key, category, score, title
    results = sort_results_for_display(songs, results)
    return results


# ── PRINT HELP / HEADERS ─────────────────────────────────────────────────────

def print_header():
    print("───────────────────────────────────────────────")
    print(f"{BOLD}              SONGMASTER{RESET}")
    print(f"{DIM}   Lyrics-aware Semantic Songbook Brain{RESET}")
    print("───────────────────────────────────────────────")
    print("Type 'help' to see available commands.")
    print("Type 'exit' or 'quit' to leave Songmaster.")
    print("───────────────────────────────────────────────")
    print()


def print_help():
    msg = f"""
{BOLD}COMMANDS{RESET}
  {BOLD}query <text>{RESET}
      Semantic search + GPT re-rank over your songbook.
      Example: query songs teenagers and old people would know
      Example: query folk love songs about lost love but not depressing
      Example: query waltzes
      Example: query 80s music also in movies
      Example: query songs which are technically difficult for a single piano player

  {BOLD}group <idx...> <group_name>{RESET}
      Create/extend a group with songs from the last query results.
      Indices refer to the numbered results shown by the last query.
      Example: group 3 4 6 7 countrysongs

  {BOLD}groups{RESET}
      List all group names and the number of songs in each.

  {BOLD}show{RESET}
      Show all groups, with songs ordered by key and grouped
      with a blank line between key changes. Color-coded by Category.
      POPULAR songs (Songbook containing 'Popular') are bold.

  {BOLD}show <group_name>{RESET}
      Show only that specific group.

  {BOLD}help{RESET}
      Show this help.

  {BOLD}exit{RESET} / {BOLD}quit{RESET}
      Leave Songmaster.

{BOLD}BUILD DB MODE{RESET}
  Run this occasionally (not every day):

      python agent_songmaster.py build-db

  This will:
    - Call GPT once per song to infer mood/era/themes etc.
    - Add structured fields like time_signature, release_decade, difficulty, familiarity.
    - Create a compact lyrics summary (no direct quotes).
    - Save songbook_enriched.json
    - Build enriched embeddings using lyrics + summary + context.
"""
    print(textwrap.dedent(msg).strip())
    print()


# ── DISPLAY UTILITIES ────────────────────────────────────────────────────────

def color_for_category(category: str) -> str:
    return CATEGORY_COLORS.get(category, RESET)


def print_query_results(
    songs: List[Song],
    results: List[Tuple[int, float]],
):
    if not results:
        print(f"{FG_RED}No matches found.{RESET}")
        return

    print(f"{BOLD}Results ({len(results)}):{RESET}")
    print(f"{DIM}[Idx]  Key   Song Name                               Songbook{RESET}")
    print(f"{DIM}---------------------------------------------------------------{RESET}")

    prev_key = None
    for display_idx, (song_idx, score) in enumerate(results, start=1):
        s = songs[song_idx]
        key_str = s.key or "-"
        songbook_str = s.songbook or ""
        title_str = s.title or ""

        if prev_key is not None and key_str != prev_key:
            # double space between different key signatures
            print()

        base_color = color_for_category(s.category)

        # NEW: POPULAR songs (Songbook contains 'Popular') get bold
        style = base_color
        if "Popular" in songbook_str:
            style += BOLD

        line = f"{display_idx:>4}.  {key_str:<4}  {title_str:<40}  {songbook_str:<20}"
        print(style + line + RESET)
        prev_key = key_str

    print()


def sort_songs_chronological(songs: List[Song], indices: List[int]) -> List[Song]:
    """Sort by Key then by title."""
    subset = [songs[i] for i in indices]
    subset.sort(key=lambda s: (s.key, s.title))
    return subset


def print_group(name: str, songs: List[Song], indices: List[int]):
    if not indices:
        print(f"{DIM}Group '{name}' is empty.{RESET}")
        return

    print(f"{BOLD}Group: {name}{RESET}  ({len(indices)} songs)")
    print(f"{DIM}Key   Song Name                               Songbook{RESET}")
    print(f"{DIM}--------------------------------------------------------{RESET}")

    ordered = sort_songs_chronological(songs, indices)

    prev_key = None
    for s in ordered:
        key_str = s.key or "-"
        title_str = s.title or ""
        songbook_str = s.songbook or ""

        if prev_key is not None and key_str != prev_key:
            print()

        base_color = color_for_category(s.category)
        style = base_color
        if "Popular" in songbook_str:
            style += BOLD

        line = f"  {key_str:<4}  {title_str:<40}  {songbook_str:<20}"
        print(style + line + RESET)
        prev_key = key_str
    print()


def print_all_groups(songs: List[Song], groups: Dict[str, List[int]]):
    if not groups:
        print(f"{DIM}No groups defined yet. Use 'group' after a query.{RESET}")
        return

    for i, (name, indices) in enumerate(groups.items(), start=1):
        print_group(name, songs, indices)
        if i < len(groups):
            print("───────────────────────────────────────────────")


# ── PARSING COMMANDS ─────────────────────────────────────────────────────────

def parse_group_command(cmd: str) -> Tuple[List[int], str]:
    """
    Parse 'group 3 4 6 7 mygroup name here'
    Returns (indices_list, group_name_str).
    """
    parts = cmd.strip().split()
    if len(parts) < 3:
        raise ValueError("Usage: group <idx...> <group_name>")

    nums: List[int] = []
    i = 1
    while i < len(parts):
        try:
            n = int(parts[i])
            nums.append(n)
            i += 1
        except ValueError:
            break

    if not nums:
        raise ValueError("You must provide at least one index from the last query.")

    if i >= len(parts):
        raise ValueError("You must provide a group name.")

    group_name = " ".join(parts[i:])
    return nums, group_name


# ── ENRICHMENT (BUILD-DB) LOGIC ──────────────────────────────────────────────

def enrich_song_with_gpt(client: OpenAI, song: Song) -> Dict[str, Any]:
    """
    Call GPT once for a single song to infer rich metadata.

    IMPORTANT: We do NOT ask for or store full lyrics from GPT, to avoid any
    copyright issues. Instead we ask for a *paraphrased summary* of the lyrics.
    Your own Lyrics column (if you add it) is used directly in embeddings.
    """
    system_msg = (
        "You are an expert musicologist and performance coach helping a live pianist. "
        "Given minimal song metadata (title, artist, key, category, group, notes), "
        "infer high-level characteristics and paraphrase the lyrics. "
        "Respond ONLY with a single valid JSON object, no extra commentary."
    )

    user_msg = f"""
Song metadata from my personal songbook:

- Title: {song.title or "Unknown"}
- Artist: {song.artist or "Unknown"}
- Key: {song.key or "Unknown"}
- Category: {song.category or "Unknown"}   (e.g. Alternative / Weird, Classic Rock & Pop hits, Movie & video Games)
- Group: {song.group or "None"}           (sometimes a show like "Mary Poppins" or "Phantom")
- Notes: {song.notes or "None"}

Constraints:
- Treat the given Artist field as canonical for this song in my dataset.
- Do NOT list additional artists who have covered the song; do not replace the artist.
- Do NOT quote lyrics verbatim. Instead, paraphrase what the lyrics describe
  (story, imagery, emotional content), in your own words.

Return a JSON object with EXACTLY these keys:

{{
  "mood_words": [ "short mood word", "..." ],
  "energy": "low" | "medium" | "high",
  "tempo_label": "slow" | "medium" | "fast",
  "tempo_bpm_guess": 0,
  "primary_era": "pre-1950s" | "1950s" | "1960s" | "1970s" | "1980s" | "1990s" | "2000s" | "2010s" | "2020s",
  "release_decade": "pre-1950s" | "1950s" | "1960s" | "1970s" | "1980s" | "1990s" | "2000s" | "2010s" | "2020s",
  "lyrical_themes": [ "theme1", "theme2", "..." ],
  "vibe_tags": [ "tag1", "tag2", "..." ],
  "style_tags": [ "folk", "hair rock", "bossa nova", "showtune", "jazz standard", "80s rock", ... ],
  "suitable_for_background": true or false,
  "suitable_for_showpiece": true or false,
  "difficulty": "easy" | "medium" | "hard",
  "tech_difficulty_solo_piano": 0,
  "emotional_arc": "short phrase describing emotional journey",
  "time_signature": "4/4" or "3/4" or "6/8" or similar,
  "is_waltz": true or false,
  "is_movie_or_musical_song": true or false,
  "familiarity_teens": 0,
  "familiarity_seniors": 0,
  "lyrics_summary": "2-4 sentences paraphrasing the lyrics, no direct quotes",
  "recommended_set_context": "one short sentence about when to play this in a set",
  "comment_for_performer": "one or two short sentences of advice or nuance"
}}

Rules:
- All strings should be short and punchy.
- Do NOT include direct lyric lines; always paraphrase in your own words.
- tech_difficulty_solo_piano is how demanding this is for a solo pianist on a 0-10 scale.
- familiarity_teens and familiarity_seniors are 0-10 rough guesses of how likely
  teenagers or seniors in the US would recognize the song.
"""

    resp = client.chat.completions.create(
        model=GPT_ENRICH_MODEL,
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    text = resp.choices[0].message.content.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

    # Fallback: minimal structure
    return {
        "mood_words": [],
        "energy": "medium",
        "tempo_label": "medium",
        "tempo_bpm_guess": 0,
        "primary_era": "unknown",
        "release_decade": "unknown",
        "lyrical_themes": [],
        "vibe_tags": [],
        "style_tags": [],
        "suitable_for_background": True,
        "suitable_for_showpiece": False,
        "difficulty": "medium",
        "tech_difficulty_solo_piano": 5,
        "emotional_arc": "",
        "time_signature": "",
        "is_waltz": False,
        "is_movie_or_musical_song": False,
        "familiarity_teens": 0,
        "familiarity_seniors": 0,
        "lyrics_summary": "",
        "recommended_set_context": "",
        "comment_for_performer": "",
    }


def build_db():
    """
    Build/update the enriched metadata DB and enriched embeddings.

    Run:
        python agent_songmaster.py build-db
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    client = make_client()
    songs = load_songbook(EXCEL_PATH)
    print(f"{BOLD}Building enriched Songmaster DB for {len(songs)} songs...{RESET}")
    print()

    enriched = load_enriched_metadata()

    updated_count = 0
    for idx, song in enumerate(songs, start=1):
        key = str(song.idx)
        existing = enriched.get(key)
        if isinstance(existing, dict) and existing.get("_version") == ENRICHED_VERSION:
            continue  # already enriched with current schema

        print(f"{DIM}Enriching {idx}/{len(songs)}:{RESET} {song.title}  ({song.artist})")
        data = enrich_song_with_gpt(client, song)
        enriched[key] = {
            "_version": ENRICHED_VERSION,
            "title": song.title,
            "artist": song.artist,
            "data": data,
        }
        updated_count += 1

    if updated_count == 0:
        print(f"{DIM}No songs needed enrichment; using existing songbook_enriched.json (v{ENRICHED_VERSION}).{RESET}")
    else:
        save_enriched_metadata(enriched)
        print(f"{BOLD}Enriched metadata updated for {updated_count} song(s).{RESET}")

    print()
    print(f"{DIM}Building enriched embeddings from songbook_enriched.json...{RESET}")
    texts = [enriched_descriptor_for_song(s, enriched) for s in songs]
    mat = compute_embeddings_from_texts(client, texts)

    excel_mtime = EXCEL_PATH.stat().st_mtime
    meta = {
        "num_songs": len(songs),
        "excel_mtime": excel_mtime,
        "embed_model": EMBED_MODEL,
        "mode": ENRICHED_MODE,
    }
    np.save(EMBED_PATH, mat)
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"{BOLD}Build complete.{RESET} Enriched DB and embeddings are ready.")
    print("You can now run:")
    print("  python agent_songmaster.py")
    print("for smart, GPT-re-ranked semantic queries.")
    print()


# ── MAIN REPL ────────────────────────────────────────────────────────────────

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    client = make_client()
    songs = load_songbook(EXCEL_PATH)
    embeddings = load_or_build_embeddings(client, songs)
    enriched = load_enriched_metadata()

    last_results: List[Tuple[int, float]] = []   # list of (song_idx, score)
    groups: Dict[str, List[int]] = {}

    print_header()

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("exit", "quit", "q"):
            break

        if cmd == "help":
            print_help()
            continue

        if cmd.startswith("query "):
            query_text = raw[len("query "):].strip()
            if not query_text:
                print(f"{FG_RED}Please provide some text after 'query'.{RESET}")
                continue

            last_results = search_songs(client, songs, embeddings, query_text, enriched, DEFAULT_TOP_K)
            print_query_results(songs, last_results)
            continue

        if cmd == "groups":
            if not groups:
                print(f"{DIM}No groups defined yet.{RESET}")
            else:
                print(f"{BOLD}Groups:{RESET}")
                for name, indices in groups.items():
                    print(f"  {name} {DIM}({len(indices)} songs){RESET}")
            print()
            continue

        if cmd == "show":
            print_all_groups(songs, groups)
            continue

        if cmd.startswith("show "):
            group_name = raw[len("show "):].strip()
            if group_name not in groups:
                print(f"{FG_RED}No such group:{RESET} {group_name}")
            else:
                print_group(group_name, songs, groups[group_name])
            continue

        if cmd.startswith("group "):
            if not last_results:
                print(f"{FG_RED}No last query results.{RESET} Run 'query' first.")
                continue
            try:
                display_indices, group_name = parse_group_command(raw)
            except ValueError as e:
                print(f"{FG_RED}ERROR:{RESET} {e}")
                continue

            song_indices: List[int] = []
            for di in display_indices:
                if di < 1 or di > len(last_results):
                    print(f"{FG_RED}Index out of range:{RESET} {di}")
                    break
                song_idx, _score = last_results[di - 1]
                song_indices.append(song_idx)
            else:
                if group_name not in groups:
                    groups[group_name] = []
                existing = set(groups[group_name])
                for si in song_indices:
                    if si not in existing:
                        groups[group_name].append(si)
                        existing.add(si)

                print(
                    f"{BOLD}Added {len(song_indices)} song(s){RESET} "
                    f"to group '{group_name}'. Now has {len(groups[group_name])} songs."
                )
            print()
            continue

        print(f"{FG_RED}Unknown command.{RESET} Type 'help' for options.\n")

    print("Goodbye.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("build-db", "build_db", "--build-db"):
        build_db()
    else:
        main()
