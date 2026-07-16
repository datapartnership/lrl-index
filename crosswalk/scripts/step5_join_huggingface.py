"""
step5_join_huggingface.py

Build a lookup from Hugging Face Hub's language tags, then join it
onto the master table. Records ALL matched tags when multiple exist
(e.g. "swh;sw"), rather than stopping at the first.

Matching tiers, in order:
  1. Exact ISO 639-3
  2. ISO 639-1 (BCP-47)
  3. ISO 639-2 - both bibliographic and terminological forms, tried
     only if neither tier 1 nor 2 matched. Most languages don't have
     a distinct ISO 639-2 code that differs from what's already been
     tried, so this tier is expected to rarely add new matches - but
     is included for completeness and to surface cases where it does.
"""
import json
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
PROCESSED_DIR = SCRIPT_DIR.parent / "data" / "processed"

MASTER_FILE = PROCESSED_DIR / "full_language_reference.csv"
HF_TAGS_FILE = RAW_DIR / "hf_language_tags.json"


def load_hf_tags(path):
    with open(path) as f:
        tags = json.load(f)
    return {t["id"].replace("language:", "") for t in tags}


def match_hf_tag(iso_639_3, iso_639_1, iso_639_2b, iso_639_2t, hf_tag_set):
    """
    Checks all known codes for a language independently, and records
    EVERY one that's actually present in HF's tag vocabulary - not
    just the first found.
    """
    matches = []
    tiers_hit = []

    if iso_639_3 in hf_tag_set:
        matches.append(iso_639_3)
        tiers_hit.append("iso3")

    if pd.notna(iso_639_1) and iso_639_1 in hf_tag_set:
        matches.append(iso_639_1)
        tiers_hit.append("iso1")

    # Tier 3: ISO 639-2, only meaningful to try as a genuinely NEW
    # tier if it wasn't already covered by tiers 1-2 above.
    if pd.notna(iso_639_2b) and iso_639_2b not in matches and iso_639_2b in hf_tag_set:
        matches.append(iso_639_2b)
        tiers_hit.append("iso2b")

    if pd.notna(iso_639_2t) and iso_639_2t not in matches and iso_639_2t in hf_tag_set:
        matches.append(iso_639_2t)
        tiers_hit.append("iso2t")

    if not matches:
        return [], "no_match"

    if len(matches) > 1:
        return matches, "matched_multiple:" + "+".join(tiers_hit)

    tier_label = {"iso3": "exact_iso3", "iso1": "matched_via_bcp47",
                  "iso2b": "matched_via_iso639_2b", "iso2t": "matched_via_iso639_2t"}
    return matches, tier_label[tiers_hit[0]]


def join_huggingface(master_df, hf_tag_set):
    df = master_df.copy()

    results = df.apply(
        lambda row: match_hf_tag(
            row["iso_639_3"], row["iso_639_1"],
            row.get("iso_639_2b"), row.get("iso_639_2t"),
            hf_tag_set
        ),
        axis=1
    )
    df["hf_tag"] = results.apply(lambda r: ";".join(r[0]) if r[0] else None)
    df["hf_match_status"] = results.apply(lambda r: r[1])
    return df


if __name__ == "__main__":
    master = pd.read_csv(MASTER_FILE)
    hf_tags = load_hf_tags(HF_TAGS_FILE)

    result = join_huggingface(master, hf_tags)

    print(f"Hugging Face has {len(hf_tags)} language tags")
    print()
    print(result["hf_match_status"].value_counts())

    iso2_matches = result[result["hf_match_status"].str.contains("iso2", na=False)]
    print(f"\nMatches that relied on ISO 639-2 (new tier): {len(iso2_matches)}")
    if len(iso2_matches):
        print(iso2_matches[["iso_639_3", "language_name", "hf_match_status"]].to_string(index=False))

    result.to_csv(MASTER_FILE, index=False)
    print(f"\nUpdated {MASTER_FILE} with Hugging Face columns")
