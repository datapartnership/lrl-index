"""
step7_join_commonvoice.py

Build a lookup from Common Voice's locale codes, then join onto the
master table. Records ALL matched locales when multiple exist.

Matching tiers, in order:
  1. Exact ISO 639-3
  2. ISO 639-1 (BCP-47)
  3. ISO 639-2 (bibliographic and terminological forms)
  4. Region-variant: strip a hyphenated region suffix (e.g. zh-CN)
     and compare the base to ISO 639-1 OR either ISO 639-2 form

Does not record hours - matching only.
"""
import json
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
PROCESSED_DIR = SCRIPT_DIR.parent / "data" / "processed"

MASTER_FILE = PROCESSED_DIR / "full_language_reference.csv"
CV_RAW_FILE = RAW_DIR / "commonvoice_languages_raw.json"


def load_commonvoice_locales(path):
    with open(path) as f:
        data = json.load(f)
    return {entry["locale"]: entry for entry in data}


def match_cv_locale(iso_639_3, iso_639_1, iso_639_2b, iso_639_2t, cv_lookup):
    matches = []
    statuses = []

    if iso_639_3 in cv_lookup:
        matches.append(iso_639_3)
        statuses.append("exact_iso3")

    if pd.notna(iso_639_1) and iso_639_1 in cv_lookup:
        matches.append(iso_639_1)
        statuses.append("matched_via_bcp47")

    if pd.notna(iso_639_2b) and iso_639_2b not in matches and iso_639_2b in cv_lookup:
        matches.append(iso_639_2b)
        statuses.append("matched_via_iso639_2b")

    if pd.notna(iso_639_2t) and iso_639_2t not in matches and iso_639_2t in cv_lookup:
        matches.append(iso_639_2t)
        statuses.append("matched_via_iso639_2t")

    base_codes_to_check = {c for c in [iso_639_1, iso_639_2b, iso_639_2t] if pd.notna(c)}
    if base_codes_to_check:
        for locale in cv_lookup:
            if "-" in locale:
                base = locale.split("-")[0]
                if base in base_codes_to_check and locale not in matches:
                    matches.append(locale)
                    statuses.append("matched_via_bcp47_region_variant")

    if not matches:
        return [], "no_match"

    if len(matches) > 1:
        return matches, "matched_multiple:" + "+".join(sorted(set(statuses)))

    return matches, statuses[0]


def join_commonvoice(master_df, cv_lookup):
    df = master_df.copy()

    if "cv_validated_hours" in df.columns:
        df = df.drop(columns=["cv_validated_hours"])

    results = df.apply(
        lambda row: match_cv_locale(
            row["iso_639_3"], row["iso_639_1"],
            row.get("iso_639_2b"), row.get("iso_639_2t"),
            cv_lookup
        ),
        axis=1
    )
    df["cv_locale"] = results.apply(lambda r: ";".join(r[0]) if r[0] else None)
    df["cv_match_status"] = results.apply(lambda r: r[1])

    return df


if __name__ == "__main__":
    master = pd.read_csv(MASTER_FILE)
    cv_lookup = load_commonvoice_locales(CV_RAW_FILE)

    result = join_commonvoice(master, cv_lookup)

    print(f"Common Voice has {len(cv_lookup)} locales")
    print()
    print(result["cv_match_status"].value_counts())

    iso2_matches = result[result["cv_match_status"].str.contains("iso2", na=False)]
    print(f"\nMatches that relied on ISO 639-2 (new tier): {len(iso2_matches)}")
    if len(iso2_matches):
        print(iso2_matches[["iso_639_3", "language_name", "cv_match_status"]].to_string(index=False))

    result.to_csv(MASTER_FILE, index=False)
    print(f"\nUpdated {MASTER_FILE} with Common Voice columns")
