"""
combine_audio_harvest.py

Simple combiner - NOT an aggregation. Takes the two existing harvest
outputs (common_voice_harvest_v2.py, voxpopuli_harvest.py), selects
only the columns they have in common, and stacks the rows into one
master file. No summing, no deduplication, no joining on iso_639_3 -
every row from both sources is kept as-is, side by side.

--------------------------------------------------------------------
COLUMNS KEPT (the overlap between both sources' outputs)
--------------------------------------------------------------------
iso_639_3, source, category, hours, license_tier, license, source_type,
who_transcribed, transcript_exists

Source-specific columns (cv_locale, validated_hours_only, clips,
speakers, corpus_version, vp_code, etc.) are DROPPED here - they only
exist in one source's output, not both, so they don't fit a combined
table. Go back to the individual harvest files if you need those.

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
Run common_voice_harvest_v2.py and voxpopuli_harvest.py first (or
already have their output files present), then run this.

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

OUTPUT_FILE = "../data/processed/audio_harvest_all.csv"

COLUMNS_TO_KEEP = [
    "iso_639_3", "source", "category", "hours",
    "license_tier", "license", "source_type", "who_transcribed", "transcript_exists",
]


def load_and_select(filepath, source_label):
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


def main():
    print("Loading harvest outputs...")
    frames = []
    for filepath, label in [(COMMON_VOICE_FILE, "Common Voice"), (VOXPOPULI_FILE, "VoxPopuli")]:
        df = load_and_select(filepath, label)
        if df is not None:
            frames.append(df)

    if not frames:
        print("No source files found - nothing to combine.")
        return

    combined = pd.concat(frames, ignore_index=True)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {len(combined)} row(s) total to {OUTPUT_FILE}")
    print(f"By source:\n{combined['source'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
