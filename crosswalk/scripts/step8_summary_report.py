"""
step8_summary_report.py

Produces a coverage summary across all 3 sources, since eyeballing
~7,929 rows directly isn't practical the way it was for the curated
191-language list.

Updated to also report how many matches across all 3 sources relied
specifically on the new ISO 639-2 tier - useful for judging whether
that tier was worth adding, once run against real data.
"""
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR.parent / "data" / "processed"

MASTER_FILE = PROCESSED_DIR / "full_language_reference.csv"


def build_coverage_by_source(df):
    rows = []
    source_cols = {
        "FineWeb-2": "fineweb2_match_status",
        "Hugging Face": "hf_match_status",
        "Common Voice": "cv_match_status",
    }
    for source_name, col in source_cols.items():
        counts = df[col].value_counts()
        total = len(df)
        matched = (df[col] != "no_match").sum()
        iso2_matched = df[col].str.contains("iso2", na=False).sum()
        for status, count in counts.items():
            rows.append({
                "source": source_name,
                "match_status": status,
                "count": count,
                "pct_of_total": round(count / total * 100, 2),
            })
        rows.append({
            "source": source_name,
            "match_status": "TOTAL_MATCHED (any status != no_match)",
            "count": matched,
            "pct_of_total": round(matched / total * 100, 2),
        })
        rows.append({
            "source": source_name,
            "match_status": "TOTAL_RELYING_ON_ISO_639_2_TIER",
            "count": iso2_matched,
            "pct_of_total": round(iso2_matched / total * 100, 2),
        })
        rows.append({
            "source": source_name,
            "match_status": "TOTAL_LANGUAGES",
            "count": total,
            "pct_of_total": 100.0,
        })
    return pd.DataFrame(rows)


def build_overlap_categories(df):
    out = df[["iso_639_3", "language_name", "is_macrolanguage"]].copy()

    out["matched_fineweb2"] = df["fineweb2_match_status"] != "no_match"
    out["matched_huggingface"] = df["hf_match_status"] != "no_match"
    out["matched_commonvoice"] = df["cv_match_status"] != "no_match"

    out["num_sources_matched"] = (
        out["matched_fineweb2"].astype(int)
        + out["matched_huggingface"].astype(int)
        + out["matched_commonvoice"].astype(int)
    )

    out["any_match_via_iso639_2"] = (
        df["fineweb2_match_status"].str.contains("iso2", na=False)
        | df["hf_match_status"].str.contains("iso2", na=False)
        | df["cv_match_status"].str.contains("iso2", na=False)
    )

    def label_row(row):
        sources = []
        if row["matched_fineweb2"]:
            sources.append("FineWeb-2")
        if row["matched_huggingface"]:
            sources.append("HuggingFace")
        if row["matched_commonvoice"]:
            sources.append("CommonVoice")
        return "+".join(sources) if sources else "NONE"

    out["matched_sources_label"] = out.apply(label_row, axis=1)
    return out


if __name__ == "__main__":
    df = pd.read_csv(MASTER_FILE)

    print(f"Total ISO 639-3 languages: {len(df)}")
    print(f"  - Macrolanguages: {df['is_macrolanguage'].sum()}")
    print(f"  - Individual languages: {(~df['is_macrolanguage']).sum()}")
    print()

    for source_name, col in [("FineWeb-2", "fineweb2_match_status"),
                               ("Hugging Face", "hf_match_status"),
                               ("Common Voice", "cv_match_status")]:
        print(f"=== {source_name} coverage ===")
        print(df[col].value_counts())
        matched = (df[col] != "no_match").sum()
        iso2 = df[col].str.contains("iso2", na=False).sum()
        print(f"Coverage: {matched} / {len(df)} ({matched/len(df)*100:.1f}%)")
        print(f"  of which relying on the new ISO 639-2 tier: {iso2}")
        print()

    none_matched = df[
        (df["fineweb2_match_status"] == "no_match") &
        (df["hf_match_status"] == "no_match") &
        (df["cv_match_status"] == "no_match")
    ]
    print(f"=== Languages with NO match in any of the 3 sources: {len(none_matched)} ===")
    print(none_matched[["iso_639_3", "language_name", "is_macrolanguage"]].head(20).to_string(index=False))

    all_matched = df[
        (df["fineweb2_match_status"] != "no_match") &
        (df["hf_match_status"] != "no_match") &
        (df["cv_match_status"] != "no_match")
    ]
    print(f"\n=== Languages matched in ALL 3 sources: {len(all_matched)} ===")
    print(all_matched[["iso_639_3", "language_name"]].head(20).to_string(index=False))

    coverage_by_source = build_coverage_by_source(df)
    coverage_by_source.to_csv(PROCESSED_DIR / "summary_coverage_by_source.csv", index=False)

    overlap = build_overlap_categories(df)
    overlap.to_csv(PROCESSED_DIR / "summary_overlap_categories.csv", index=False)

    none_matched_full = overlap[overlap["num_sources_matched"] == 0]
    none_matched_full.to_csv(PROCESSED_DIR / "summary_no_coverage_languages.csv", index=False)

    full_matched_full = overlap[overlap["num_sources_matched"] == 3]
    full_matched_full.to_csv(PROCESSED_DIR / "summary_full_coverage_languages.csv", index=False)

    iso2_reliant = overlap[overlap["any_match_via_iso639_2"]]
    iso2_reliant.to_csv(PROCESSED_DIR / "summary_iso639_2_reliant_languages.csv", index=False)

    print(f"\n--- CSV files written to {PROCESSED_DIR} ---")
    print("summary_coverage_by_source.csv         - match counts/% per source, incl. ISO 639-2 tier usage")
    print("summary_overlap_categories.csv         - per-language matched-sources label, full table")
    print("summary_no_coverage_languages.csv      - full list, zero matches across all 3 sources")
    print("summary_full_coverage_languages.csv    - full list, matched in all 3 sources")
    print("summary_iso639_2_reliant_languages.csv - languages where any source's match needed the new tier")
