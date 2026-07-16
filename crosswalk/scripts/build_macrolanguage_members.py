"""
build_macrolanguage_members.py

Builds a CSV listing every ISO 639-3 macrolanguage alongside ALL of its
individual-language members - the reverse direction from what
langcodes' prefer_macrolanguage() gives you (which only goes
individual -> macro, one at a time).

Source: SIL's official Macrolanguage Mappings table, a separate
download from the main ISO 639-3 code table:
https://iso639-3.sil.org/code_tables/macrolanguage_mappings/data

Download the tab-delimited file from that page (commonly named
something like iso-639-3-macrolanguages.tab) and place it in this
directory before running this script.

The source table has 3 columns:
  M_Id    - ISO 639-3 code of the macrolanguage
  I_Id    - ISO 639-3 code of the individual member language
  I_Status - 'A' (Active) or 'R' (Retired)
"""
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
PROCESSED_DIR = SCRIPT_DIR.parent / "data" / "processed"

MACRO_MAPPINGS_FILE = RAW_DIR / "iso-639-3-macrolanguages.tab"
ISO_REF_FILE = PROCESSED_DIR / "full_language_reference.csv"
OUTPUT_FILE = PROCESSED_DIR / "macrolanguage_to_members.csv"

def load_macrolanguage_mappings(path):
    df = pd.read_csv(path, sep="\t")
    # Column names can vary slightly by download vintage - normalize them
    df.columns = [c.strip() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("m_id", "mid"):
            rename_map[c] = "macrolanguage_code"
        elif cl in ("i_id", "iid"):
            rename_map[c] = "member_code"
        elif "status" in cl:
            rename_map[c] = "member_status"
    df = df.rename(columns=rename_map)
    return df


def build_macro_to_members_table(mappings_df, iso_ref_df=None, active_only=True):
    """
    Returns one row per macrolanguage, with all its members listed
    together in a single semicolon-separated column - same convention
    used throughout the rest of this crosswalk pipeline.
    """
    df = mappings_df.copy()

    if active_only and "member_status" in df.columns:
        df = df[df["member_status"].str.upper().str.startswith("A")]

    # Defensive check: drop any row with a missing macrolanguage_code or
    # member_code before grouping. A blank/NaN value here would otherwise
    # crash sorted() (mixing float NaN with strings) or silently create
    # a bogus group. Report what's dropped so it's visible, not silent.
    before = len(df)
    bad_rows = df[df["macrolanguage_code"].isna() | df["member_code"].isna()]
    if len(bad_rows):
        print(f"WARNING: dropping {len(bad_rows)} row(s) with missing macrolanguage_code or member_code:")
        print(bad_rows.to_string(index=False))
    df = df.dropna(subset=["macrolanguage_code", "member_code"])
    df["macrolanguage_code"] = df["macrolanguage_code"].astype(str)
    df["member_code"] = df["member_code"].astype(str)

    grouped = (
        df.groupby("macrolanguage_code")["member_code"]
        .apply(lambda codes: ";".join(sorted(codes)))
        .reset_index()
        .rename(columns={"member_code": "member_codes"})
    )
    grouped["member_count"] = grouped["member_codes"].apply(lambda s: len(s.split(";")) if s else 0)

    if iso_ref_df is not None:
        name_lookup = iso_ref_df.set_index("iso_639_3")["language_name"].to_dict()
        grouped["macrolanguage_name"] = grouped["macrolanguage_code"].map(name_lookup)

        def member_names(codes_str):
            codes = codes_str.split(";")
            names = [name_lookup.get(c, c) for c in codes]
            return ";".join(names)

        grouped["member_names"] = grouped["member_codes"].apply(member_names)

    cols = ["macrolanguage_code"]
    if "macrolanguage_name" in grouped.columns:
        cols.append("macrolanguage_name")
    cols += ["member_count", "member_codes"]
    if "member_names" in grouped.columns:
        cols.append("member_names")

    return grouped[cols].sort_values("macrolanguage_code").reset_index(drop=True)


if __name__ == "__main__":
    mappings = load_macrolanguage_mappings(MACRO_MAPPINGS_FILE)
    print(f"Loaded {len(mappings)} macrolanguage-member mapping rows")
    print(f"Covering {mappings['macrolanguage_code'].nunique()} distinct macrolanguages")
    print()

    try:
        iso_ref = pd.read_csv(ISO_REF_FILE)
    except FileNotFoundError:
        print(f"Note: {ISO_REF_FILE} not found - proceeding without language names")
        iso_ref = None

    result = build_macro_to_members_table(mappings, iso_ref)

    print(f"\nBuilt table: {len(result)} macrolanguages")
    print(result.head(10).to_string(index=False))

    result.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
