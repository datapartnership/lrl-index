"""
step3_join_fineweb2.py

Joins FineWeb-2's language-script labels onto the full ISO table.
No patching - mismatches are flagged as "no_match" and left as-is.

Matching tiers, in order:
  1. Exact ISO 639-3 (FineWeb-2's normal convention)
  2. ISO 639-1 - tried only as a fallback, since FineWeb-2 labels are
     consistently {ISO 639-3 code}_{Script}, so this is expected to
     rarely fire
  3. ISO 639-2 (bibliographic and terminological) - same expectation,
     included for completeness
"""
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
PROCESSED_DIR = SCRIPT_DIR.parent / "data" / "processed"

MASTER_FILE = PROCESSED_DIR / "full_language_reference.csv"
FW2_LABELS_FILE = RAW_DIR / "fineweb2_labels.txt"


def load_fineweb2_labels(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_fw2_lookup(labels):
    lookup = {}
    for label in labels:
        iso_code, script = label.rsplit("_", 1)
        lookup.setdefault(iso_code, []).append(label)
    return lookup


def match_fineweb2(iso_639_3, iso_639_1, iso_639_2b, iso_639_2t, fw2_lookup):
    """
    Tries each code tier in order. FineWeb-2's labels are keyed by
    whatever code shows up before the script suffix, so any of the
    4 candidate codes could in principle match a label, even though
    in practice tier 1 (ISO 639-3) is expected to dominate.
    """
    candidates = [
        (iso_639_3, "iso3"),
        (iso_639_1, "iso1"),
        (iso_639_2b, "iso2b"),
        (iso_639_2t, "iso2t"),
    ]

    for code, tier in candidates:
        if pd.isna(code):
            continue
        matches = fw2_lookup.get(code, [])
        if matches:
            status = {
                "iso3": "exact" if len(matches) == 1 else "multi_script",
                "iso1": "matched_via_bcp47",
                "iso2b": "matched_via_iso639_2b",
                "iso2t": "matched_via_iso639_2t",
            }[tier]
            return ";".join(matches), status

    return None, "no_match"


def join_fineweb2(master_df, fw2_lookup):
    df = master_df.copy()

    results = df.apply(
        lambda row: match_fineweb2(
            row["iso_639_3"], row["iso_639_1"],
            row.get("iso_639_2b"), row.get("iso_639_2t"),
            fw2_lookup
        ),
        axis=1
    )
    df["fineweb2_labels"] = results.apply(lambda r: r[0])
    df["fineweb2_match_status"] = results.apply(lambda r: r[1])
    return df


if __name__ == "__main__":
    master = pd.read_csv(MASTER_FILE)
    labels = load_fineweb2_labels(FW2_LABELS_FILE)
    fw2_lookup = build_fw2_lookup(labels)

    result = join_fineweb2(master, fw2_lookup)

    print(f"FineWeb-2 has {len(fw2_lookup)} unique ISO codes across {len(labels)} labels")
    print()
    print(result["fineweb2_match_status"].value_counts())

    iso2_matches = result[result["fineweb2_match_status"].str.contains("iso2", na=False)]
    print(f"\nMatches that relied on ISO 639-2 (new tier): {len(iso2_matches)}")
    if len(iso2_matches):
        print(iso2_matches[["iso_639_3", "language_name", "fineweb2_match_status"]].to_string(index=False))

    result.to_csv(MASTER_FILE, index=False)
    print(f"\nUpdated {MASTER_FILE} with FineWeb-2 columns")
