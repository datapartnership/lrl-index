"""
step2_fetch_fineweb2_labels.py

Fetches FineWeb-2's published language-script labels via the Hugging
Face datasets-server API. Writes fineweb2_labels.txt, the same file
step3_join_fineweb2.py expects to read.

Run this BEFORE step3_join_fineweb2.py.
"""
import requests

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"

DATASET = "HuggingFaceFW/fineweb-2"
API_URL = f"https://datasets-server.huggingface.co/splits?dataset={DATASET}"
OUTPUT_FILE = RAW_DIR / "fineweb2_labels.txt"

def fetch_fineweb2_labels():
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    labels = sorted({entry["config"] for entry in data["splits"]})
    return labels


if __name__ == "__main__":
    labels = fetch_fineweb2_labels()
    print(f"Found {len(labels)} language-script labels")
    print(labels[:10])

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(labels))
    print(f"Saved to {OUTPUT_FILE}")
