"""
step1_load_iso_table.py

Loads the FULL ISO 639-3 reference table from SIL - all ~7,929 entries,
no filtering to any curated list. This is the anchor for the
all-languages crosswalk.
"""
import pandas as pd

ISO_FILE = "iso-639-3.tab"


def load_iso_table(path):
    df = pd.read_csv(path, sep="\t")
    return df


def build_reference_table(raw_df):
    ref = raw_df[["Id", "Part1", "Scope", "Language_Type", "Ref_Name"]].copy()
    ref = ref.rename(columns={
        "Id": "iso_639_3",
        "Part1": "iso_639_1",
        "Ref_Name": "language_name",
    })
    ref["is_macrolanguage"] = ref["Scope"] == "M"
    ref["is_living"] = ref["Language_Type"] == "L"
    return ref


if __name__ == "__main__":
    raw = load_iso_table(ISO_FILE)
    ref = build_reference_table(raw)

    print(f"Loaded {len(ref)} total ISO 639-3 entries (full table, unfiltered)")
    print()
    print("Scope breakdown:")
    print(raw["Scope"].value_counts())
    print()
    print(f"Macrolanguages: {ref['is_macrolanguage'].sum()}")
    print(f"Living languages: {ref['is_living'].sum()}")

    ref.to_csv("full_language_reference.csv", index=False)
    print("\nSaved to full_language_reference.csv")
