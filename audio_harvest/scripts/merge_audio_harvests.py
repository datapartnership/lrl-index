"""
merge_audio_harvest.py

Simple combiner - NOT an aggregation. Takes the existing harvest
outputs, selects only the columns they have in common, and stacks
the rows into one master file. No summing, no deduplication, no
joining on iso_639_3 - every row from each source is kept as-is.

--------------------------------------------------------------------
COLUMNS KEPT (the overlap between all sources' outputs)
--------------------------------------------------------------------
iso_639_3, source, category, hours, license_tier, license, source_type,
who_transcribed, transcript_exists

Source-specific columns (cv_locale, validated_hours_only, clips,
speakers, corpus_version, vp_code, num_recordings, num_missing_duration,
total_seconds, etc.) are DROPPED here - they only exist in one source's
output, not all three, so they don't fit a combined table. Go back to
the individual harvest files if you need those.

Lingua Libre's own output (from lingua_libre_duration_summing.py) uses a
different column naming scheme (Iso_639_3, total_hours, Category, etc.)
and doesn't have source / license_tier / source_type columns at all, so
it's mapped into the shared schema below rather than read directly with
COLUMNS_TO_KEEP like the other two sources:
    - source is hardcoded to "lingua_libre"
    - source_type is left blank - Lingua Libre's source_type varies
      per-recording (different word lists), so there's no single
      per-language value to put here; see the per-recording harvest
      CSV (lingua_libre_harvest.csv) if that detail is needed
    - license_tier is NOT yet confirmed - see LINGUA_LIBRE_LICENSE_TIER
      below, fill it in once the tiering scheme is confirmed for a
      CC BY-SA 4.0 (share-alike) source, since it isn't CC0 like the
      other two sources

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
Run common_voice_harvest.py, voxpopuli_harvest.py, lingua_libre_harvest.py,
and lingua_libre_duration_summing.py first before running this.

--------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------
pip install pandas
"""
from pathlib import Path

import pandas as pd

# ---- CONFIG ----
COMMON_VOICE_FILE = "../data/processed/common_voice/common_voice_hours.csv"
VOXPOPULI_FILE = "../data/processed/vox_populi/voxpopuli_hours.csv"
LINGUA_LIBRE_FILE = "../data/processed/lingua_libre/lingua_libre_language_summary.csv"

OUTPUT_FILE = "../data/processed/audio_harvest_all.csv"

COLUMNS_TO_KEEP = [
    "iso_639_3", "source", "category", "hours",
    "license_tier", "license", "source_type", "who_transcribed", "transcript_exists",
]

# NOT YET CONFIRMED: Lingua Libre is CC BY-SA 4.0, not CC0 like the other
# two sources, so it may not belong in the same license_tier as them.
# Fill this in once the tiering scheme is confirmed - left as None
# (blank in the output) rather than guessed.
LINGUA_LIBRE_LICENSE_TIER = None


def load_and_select(filepath, source_label):
    """For sources whose output already matches COLUMNS_TO_KEEP directly."""
    path = Path(filepath)
    if not path.exists():
        print(f"  {filepath} not found - skipping {source_label} (run its harvest script first)")
        return None

    df = pd.read_csv(filepath)
    missing = [c for c in COLUMNS_TO_KEEP if c not in df.columns]
    if missing:
        raise RuntimeError(f"{filepath} is missing expected column(s) {missing} - "
                            f"has this source's harvest script's output schema changed?")

    print(f"  {filepath}: {len(df)} row(s)")
    return df[COLUMNS_TO_KEEP]


def load_lingua_libre(filepath):
    """
    Lingua Libre's per-language summary (lingua_libre_duration_summing.py's
    output) uses a different schema than the other two sources, so it's
    mapped into COLUMNS_TO_KEEP here rather than read directly.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"  {filepath} not found - skipping Lingua Libre "
              f"(run lingua_libre_harvest.py and lingua_libre_duration_summing.py first)")
        return None

    df = pd.read_csv(filepath)

    expected_source_cols = [
        "Iso_639_3", "total_hours", "Category",
        "License", "Who_transcribed", "Transcript_exists",
    ]
    missing = [c for c in expected_source_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{filepath} is missing expected column(s) {missing} - "
            f"has lingua_libre_duration_summing.py's output schema changed?"
        )

    mapped = pd.DataFrame({
        "iso_639_3": df["Iso_639_3"],
        "source": "lingua_libre",
        "category": df["Category"],
        "hours": df["total_hours"],
        "license_tier": LINGUA_LIBRE_LICENSE_TIER,
        "license": df["License"],
        "source_type": pd.NA,  # varies per-recording - see module docstring
        "who_transcribed": df["Who_transcribed"],
        "transcript_exists": df["Transcript_exists"],
    })

    print(f"  {filepath}: {len(mapped)} row(s)")
    return mapped[COLUMNS_TO_KEEP]


def main():
    print("Loading harvest outputs...")
    frames = []
    for filepath, label in [(COMMON_VOICE_FILE, "Common Voice"), (VOXPOPULI_FILE, "VoxPopuli")]:
        df = load_and_select(filepath, label)
        if df is not None:
            frames.append(df)

    lingua_libre_df = load_lingua_libre(LINGUA_LIBRE_FILE)
    if lingua_libre_df is not None:
        frames.append(lingua_libre_df)

    if not frames:
        print("No source files found - nothing to combine.")
        return

    combined = pd.concat(frames, ignore_index=True)

    if LINGUA_LIBRE_LICENSE_TIER is None and "lingua_libre" in combined["source"].values:
        print(
            "\nNOTE: LINGUA_LIBRE_LICENSE_TIER is not set - license_tier is "
            "blank for all lingua_libre rows in the output. Fill in the "
            "constant at the top of this script once confirmed."
        )

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {len(combined)} row(s) total to {OUTPUT_FILE}")
    print(f"By source:\n{combined['source'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
