"""
step4_fetch_huggingface_tags.py

Fetches Hugging Face Hub's full language tag vocabulary via the Hub's
public tags API. Writes hf_language_tags.json, the same file
step5_join_huggingface.py already expects to read.

Run this BEFORE step5_join_huggingface.py.
"""
import requests
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"

API_URL = "https://huggingface.co/api/datasets-tags-by-type"
OUTPUT_FILE = RAW_DIR / "hf_language_tags.json"


def fetch_hf_language_tags():
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    language_tags = data.get("language", [])
    return language_tags


if __name__ == "__main__":
    tags = fetch_hf_language_tags()
    print(f"Found {len(tags)} language tags on Hugging Face Hub")
    print(tags[:10])

    with open(OUTPUT_FILE, "w") as f:
        json.dump(tags, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")
