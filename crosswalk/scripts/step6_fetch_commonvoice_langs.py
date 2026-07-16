"""
step6_fetch_commonvoice_langs.py

Fetches Mozilla Common Voice's language stats via their public API.
Writes commonvoice_languages_raw.json, the same file
step7_join_commonvoice.py expects to read.

Run this BEFORE step7_join_commonvoice.py.
"""
import requests
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"

API_URL = "https://commonvoice.mozilla.org/api/v1/stats/languages"
OUTPUT_FILE = RAW_DIR / "commonvoice_languages_raw.json"


def fetch_commonvoice_languages():
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    data = fetch_commonvoice_languages()
    print(f"Found {len(data)} entries")
    print(data[:5])

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")
