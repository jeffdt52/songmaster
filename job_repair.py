#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
job_repairArtist.py

Repairs artist fields and normalizes song titles to ALL CAPS in:

  - Songbook Data.xlsx
  - songbook_enriched.json

Use this when GPT guessed the wrong artist, you’ve corrected them, or you want
to enforce consistent ALL-CAPS titles.

Workflow:
  1. Edit the CORRECTIONS list below if you need to tweak any mapping.
  2. Run:
        python job_repairArtist.py
  3. Then rebuild the enriched DB / embeddings:
        python agent_songmaster.py build-db
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "Songbook Data.xlsx"
ENRICHED_PATH = BASE_DIR / "songbook_enriched.json"

# Backup suffix for Excel; we'll write e.g. "Songbook Data.xlsx.bak"
BACKUP_SUFFIX = ".bak"

# If True, when we fix an entry in songbook_enriched.json, we clear its "data"
# so the next build-db run will re-enrich that song.
CLEAR_ENRICHED_DATA_ON_FIX = True

# ── CORRECTIONS ──────────────────────────────────────────────────────────────
# Each dict describes how to fix one song.
#
# Fields:
#   - match_titles: list of possible current titles to match (ALL CAPS)
#   - new_title:    desired canonical title (will be uppercased)
#   - new_artist:   desired Artist field (or None if only changing title)
#   - extra_updates: optional dict of other column updates, e.g. {"Songbook": "My Book"}
#
# These 15 corrections follow YOUR intended pattern:
#   [artists] Cage the Elephant, Kodaline, Israel Kamakawiwoʻole, Queen, Supertramp,
#             Lukas Nelson, Willie Nelson, B.B. King, Louis Armstrong, Willie Nelson,
#             Duke Ellington, Susan Tedeschi, Stan Getz, Bill Withers, Fats Waller
#   ↔
#   [songs]   HYPOCRITE, HIGH HOPES, KAULANA KAWAIHAE, WHO NEEDS YOU, GOODBYE STRANGER,
#            JUST OUTSIDE OF AUSTIN, BLUE SKIES, IS YOU OR IS YOU AIN'T (MY BABY),
#            C'EST SI BON, STARDUST, CARAVAN, BLUES ON A HOLIDAY, GIRL FROM IPANEMA,
#            SUNNY, MARGIE

CORRECTIONS: List[Dict[str, Any]] = [
    {
        "match_titles": ["HYPOCRITE"],
        "new_title": "HYPOCRITE",
        "new_artist": "CAGE THE ELEPHANT",
    },
    {
        "match_titles": ["HIGH HOPES"],
        "new_title": "HIGH HOPES",
        "new_artist": "KODALINE",
    },
    {
        "match_titles": ["KAULANA KAWAIHAE"],
        "new_title": "KAULANA KAWAIHAE",
        # using a canonical spelling for the name, all caps for consistency
        "new_artist": "ISRAEL KAMAKAWIWO'OLE",
    },
    {
        "match_titles": ["WHO NEEDS YOU"],
        "new_title": "WHO NEEDS YOU",
        "new_artist": "QUEEN",
    },
    {
        # handle both GOODBYE and possible GOODYBE typos
        "match_titles": ["GOODBYE STRANGER", "GOODYBE STRANGER"],
        "new_title": "GOODBYE STRANGER",
        "new_artist": "SUPERTRAMP",
    },
    {
        "match_titles": ["JUST OUTSIDE OF AUSTIN"],
        "new_title": "JUST OUTSIDE OF AUSTIN",
        "new_artist": "LUKAS NELSON",
    },
    {
        "match_titles": ["BLUE SKIES"],
        "new_title": "BLUE SKIES",
        "new_artist": "WILLIE NELSON",
    },
    {
        "match_titles": [
            "IS YOU OR IS YOU AINT (MY BABY)",
            "IS YOU OR IS YOU AIN'T (MY BABY)",
        ],
        "new_title": "IS YOU OR IS YOU AIN'T (MY BABY)",
        "new_artist": "B.B. KING",
    },
    {
        "match_titles": ["C'EST SI BON", "C’EST SI BON"],
        "new_title": "C'EST SI BON",
        "new_artist": "LOUIS ARMSTRONG",
    },
    {
        "match_titles": ["STARDUST"],
        "new_title": "STARDUST",
        "new_artist": "WILLIE NELSON",
    },
    {
        "match_titles": ["CARAVAN"],
        "new_title": "CARAVAN",
        "new_artist": "DUKE ELLINGTON",
    },
    {
        "match_titles": ["BLUES ON A HOLIDAY"],
        "new_title": "BLUES ON A HOLIDAY",
        "new_artist": "SUSAN TEDESCHI",
    },
    {
        "match_titles": ["GIRL FROM IPANEMA", "THE GIRL FROM IPANEMA"],
        "new_title": "GIRL FROM IPANEMA",
        "new_artist": "STAN GETZ",
    },
    {
        "match_titles": ["SUNNY"],
        "new_title": "SUNNY",
        "new_artist": "BILL WITHERS",
    },
    {
        "match_titles": ["MARGIE"],
        "new_title": "MARGIE",
        "new_artist": "FATS WALLER",
    },
]

# ── HELPERS ──────────────────────────────────────────────────────────────────

def ensure_excel_exists():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Songbook Excel not found at {EXCEL_PATH}")


def backup_excel():
    backup_path = EXCEL_PATH.with_suffix(EXCEL_PATH.suffix + BACKUP_SUFFIX)
    print(f"Creating backup: {backup_path.name}")
    shutil.copy2(EXCEL_PATH, backup_path)


def fix_excel():
    """
    - Load Songbook Data.xlsx
    - Normalize all Song titles to ALL CAPS
    - Apply CORRECTIONS to Song + Artist columns
    - Save the Excel back (after making a .bak backup)
    """
    ensure_excel_exists()
    df = pd.read_excel(EXCEL_PATH)

    if "Song" not in df.columns:
        raise KeyError("Expected a 'Song' column in Songbook Data.xlsx")
    if "Artist" not in df.columns:
        raise KeyError("Expected an 'Artist' column in Songbook Data.xlsx")

    # Normalize EVERYTHING in Song column to ALL CAPS
    df["Song"] = df["Song"].astype(str).str.upper()

    # Apply corrections
    for corr in CORRECTIONS:
        match_titles = [str(t).upper() for t in corr.get("match_titles", [])]
        new_title_upper = (
            str(corr.get("new_title", "")).upper() if corr.get("new_title") else None
        )
        new_artist = corr.get("new_artist")
        extra_updates = corr.get("extra_updates", {})

        if not match_titles:
            continue

        mask = df["Song"].isin(match_titles)
        if not mask.any():
            print(f"[Excel] WARNING: No rows found matching titles {match_titles}")
            continue

        if new_title_upper:
            df.loc[mask, "Song"] = new_title_upper

        if new_artist is not None:
            df.loc[mask, "Artist"] = new_artist

        for col, val in extra_updates.items():
            if col in df.columns:
                df.loc[mask, col] = val

        count = int(mask.sum())
        print(f"[Excel] Updated {count} row(s) matching {match_titles}")

    # Backup then save
    backup_excel()
    df.to_excel(EXCEL_PATH, index=False)
    print(f"[Excel] Saved updated Excel to {EXCEL_PATH.name}")


def fix_enriched_json():
    """
    - Load songbook_enriched.json (if present)
    - Normalize title fields to ALL CAPS for everything
    - Apply CORRECTIONS to title + artist
    - Optionally clear 'data' for corrected entries so build-db re-enriches them
    """
    if not ENRICHED_PATH.exists():
        print("[JSON] No songbook_enriched.json found; skipping JSON repair.")
        return

    with ENRICHED_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print("[JSON] Unexpected format in songbook_enriched.json; expected dict.")
        return

    # First pass: normalize all titles to ALL CAPS
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        title = entry.get("title")
        if title is not None:
            entry["title"] = str(title).upper()

    # Second pass: apply corrections
    for corr in CORRECTIONS:
        match_titles = [str(t).upper() for t in corr.get("match_titles", [])]
        new_title_upper = (
            str(corr.get("new_title", "")).upper() if corr.get("new_title") else None
        )
        new_artist = corr.get("new_artist")

        if not match_titles:
            continue

        matched_any = False
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            title = str(entry.get("title", "")).upper()
            if title in match_titles:
                matched_any = True
                if new_title_upper:
                    entry["title"] = new_title_upper
                if new_artist is not None:
                    entry["artist"] = new_artist

                if CLEAR_ENRICHED_DATA_ON_FIX:
                    entry.pop("data", None)
                    entry["_version"] = 0  # force re-enrich

        if matched_any:
            print(f"[JSON] Updated enriched entry/entries for titles {match_titles}")
        else:
            print(f"[JSON] WARNING: No enriched entries found matching {match_titles}")

    # Save back
    with ENRICHED_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[JSON] Saved updated JSON to {ENRICHED_PATH.name}")


def main():
    print("── job_repairArtist.py ───────────────────────")
    print(f"Excel:    {EXCEL_PATH}")
    print(f"Enriched: {ENRICHED_PATH}")
    print("Normalizing titles to ALL CAPS and applying corrections...")
    print()

    fix_excel()
    print()
    fix_enriched_json()
    print()
    print("Done. Now run:")
    print("  python agent_songmaster.py build-db")
    print("to re-enrich any corrected songs and rebuild embeddings.")


if __name__ == "__main__":
    main()
