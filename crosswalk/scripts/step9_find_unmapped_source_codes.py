"""
step9_find_unmapped_source_codes.py

REVERSE DIRECTION CHECK.

Steps 3, 5, 7 only check: "does this ISO code exist in source X?"
This finds the opposite: codes/labels that EXIST in a source but
don't correspond to any known ISO entry at all.

"known codes" includes ISO 639-2 (both bibliographic and
terminological forms) alongside ISO 639-3 and ISO 639-1, consistent
with the matching tiers.
"""
import json
import pandas as pd
import langcodes
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
PROCESSED_DIR = SCRIPT_DIR.parent / "data" / "processed"

ISO_REF_FILE = PROCESSED_DIR / "full_language_reference.csv"
FW2_LABELS_FILE = RAW_DIR / "fineweb2_labels.txt"
HF_TAGS_FILE = RAW_DIR / "hf_language_tags.json"
CV_RAW_FILE = RAW_DIR / "commonvoice_languages_raw.json"


def get_macrolanguage_info(code):
    try:
        lang = langcodes.Language.get(code)
        if not lang.is_valid():
            return False, None
        macro = str(lang.prefer_macrolanguage())
        if macro != code:
            return True, macro
        return False, None
    except Exception:
        return False, None


def get_known_codes(iso_ref_df):
    """
    Known codes now include ISO 639-3, ISO 639-1, AND both ISO 639-2
    forms (bibliographic, terminological) - matching the matching
    tiers used in the join scripts. A code that only matches as an
    ISO 639-2 form should not be treated as an orphan.
    """
    iso3 = set(iso_ref_df["iso_639_3"].dropna())
    iso1 = set(iso_ref_df["iso_639_1"].dropna())
    iso2b = set(iso_ref_df["iso_639_2b"].dropna()) if "iso_639_2b" in iso_ref_df.columns else set()
    iso2t = set(iso_ref_df["iso_639_2t"].dropna()) if "iso_639_2t" in iso_ref_df.columns else set()
    return iso3 | iso1 | iso2b | iso2t


def find_unmapped_fineweb2(known_codes, fw2_labels_path):
    with open(fw2_labels_path) as f:
        labels = [l.strip() for l in f if l.strip()]

    unmapped = []
    for label in labels:
        code, script = label.rsplit("_", 1)
        if code not in known_codes:
            has_macro, macro_code = get_macrolanguage_info(code)
            unmapped.append({
                "fineweb2_orphan_label": label,
                "orphan_code": code,
                "has_macrolanguage_family": has_macro,
                "macrolanguage_code": macro_code,
            })
    return unmapped


def find_unmapped_huggingface(known_codes, hf_tags_path):
    with open(hf_tags_path) as f:
        tags = json.load(f)
    tag_ids = [t["id"].replace("language:", "") for t in tags]

    unmapped = []
    for t in tag_ids:
        if t not in known_codes:
            has_macro, macro_code = get_macrolanguage_info(t)
            unmapped.append({
                "hf_orphan_tag": t,
                "has_macrolanguage_family": has_macro,
                "macrolanguage_code": macro_code,
            })
    return unmapped


def find_unmapped_commonvoice(known_codes, cv_raw_path):
    with open(cv_raw_path) as f:
        data = json.load(f)

    unmapped = []
    for entry in data:
        locale = entry["locale"]
        base = locale.split("-")[0] if "-" in locale else locale
        if base not in known_codes:
            has_macro, macro_code = get_macrolanguage_info(base)
            unmapped.append({
                "locale": locale,
                "english_name": entry.get("english_name"),
                "has_macrolanguage_family": has_macro,
                "macrolanguage_code": macro_code,
            })
    return unmapped


if __name__ == "__main__":
    iso_ref = pd.read_csv(ISO_REF_FILE)
    known = get_known_codes(iso_ref)
    print(f"{len(known)} known codes (ISO 639-3 + ISO 639-1 + ISO 639-2b/2t)")
    print()

    fw2_orphans = find_unmapped_fineweb2(known, FW2_LABELS_FILE)
    fw2_df = pd.DataFrame(fw2_orphans)
    print(f"=== FineWeb-2: {len(fw2_orphans)} labels with no matching ISO entry ===")
    if len(fw2_df):
        print(f"  of which {fw2_df['has_macrolanguage_family'].sum()} belong to a known macrolanguage family")
    print()

    hf_orphans = find_unmapped_huggingface(known, HF_TAGS_FILE)
    hf_df = pd.DataFrame(hf_orphans)
    print(f"=== Hugging Face: {len(hf_orphans)} tags with no matching ISO entry ===")
    if len(hf_df):
        print(f"  of which {hf_df['has_macrolanguage_family'].sum()} belong to a known macrolanguage family")
    print()

    cv_orphans = find_unmapped_commonvoice(known, CV_RAW_FILE)
    cv_df = pd.DataFrame(cv_orphans)
    print(f"=== Common Voice: {len(cv_orphans)} locales with no matching ISO entry ===")
    if len(cv_df):
        print(f"  of which {cv_df['has_macrolanguage_family'].sum()} belong to a known macrolanguage family")

    fw2_df.to_csv(PROCESSED_DIR / "orphans_fineweb2.csv", index=False)
    hf_df.to_csv(PROCESSED_DIR / "orphans_huggingface.csv", index=False)
    cv_df.to_csv(PROCESSED_DIR / "orphans_commonvoice.csv", index=False)
    print(f"\nSaved orphan CSVs to {PROCESSED_DIR}")
