"""
lingua_libre_duration_summing.py
=================================

Reads the per-recording harvest CSV (output of
lingua_libre_duration_harvester.py) and produces a per-language summary:
total duration in seconds and hours, recording counts, and the fixed
provenance/transcript fields that apply to every Lingua Libre recording.

USAGE:
    python lingua_libre_summarize.py
    python lingua_libre_summarize.py --input path/to/harvest.csv --output path/to/summary.csv

By default:
    INPUT  = lrl-index/audio_harvest/data/processed/lingua_libre/lingua_libre_harvest.csv
    OUTPUT = lrl-index/audio_harvest/data/processed/lingua_libre/lingua_libre_language_summary.csv
Both paths are anchored to this script's own file location (assumes it
lives in .../audio_harvest/scripts/), not the current working directory,
so it runs correctly regardless of where you invoke it from.

OUTPUT SCHEMA (one row per iso_639_3):
    Iso_639_3
    num_recordings       - total rows for this language in the harvest
    num_missing_duration - rows where Duration (s) was blank/unparseable
                            (these are excluded from the sums below, not
                            treated as 0 - see note in main())
    total_seconds
    total_hours           = total_seconds / 3600
    Category              - fixed: "transcribed" (see methodology doc)
    License                - fixed: "CC BY-SA 4.0"
    Who_transcribed        - fixed: "no transcription - pre-set word"
    Transcript_exists      - fixed: TRUE

NOTE: Source_type is NOT included here, since it varies per recording
(different word lists) rather than being a single fixed value per
language - it stays in the per-recording harvest CSV, not this summary.
"""

import argparse
import csv
import os

# Fixed provenance/transcript constants - identical to those written into
# every row by lingua_libre_duration_harvester.py, since these are
# constant across the whole Lingua Libre corpus by construction.
CATEGORY_VALUE = "transcribed"
LICENSE_VALUE = "CC BY-SA 4.0"
WHO_TRANSCRIBED_VALUE = "no transcription \u2013 pre-set word"
TRANSCRIPT_EXISTS_VALUE = "TRUE"

SUMMARY_FIELDNAMES = [
    "Iso_639_3",
    "num_recordings",
    "num_missing_duration",
    "total_seconds",
    "total_hours",
    "Category",
    "License",
    "Who_transcribed",
    "Transcript_exists",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "data", "processed", "lingua_libre")
)
DEFAULT_INPUT = os.path.join(DATA_DIR, "lingua_libre_harvest.csv")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "lingua_libre_language_summary.csv")


def summarize(input_path, output_path):
    totals = {}       # iso_639_3 -> running total seconds
    counts = {}        # iso_639_3 -> total row count
    missing = {}        # iso_639_3 -> count of rows with no parseable duration

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iso = row.get("Iso_639_3", "").strip()
            if not iso:
                continue

            counts[iso] = counts.get(iso, 0) + 1

            raw_duration = row.get("Duration (s)", "")
            try:
                seconds = float(raw_duration)
            except (TypeError, ValueError):
                missing[iso] = missing.get(iso, 0) + 1
                continue

            totals[iso] = totals.get(iso, 0.0) + seconds

    summary_rows = []
    for iso in sorted(counts):
        total_seconds = totals.get(iso, 0.0)
        summary_rows.append(
            {
                "Iso_639_3": iso,
                "num_recordings": counts[iso],
                "num_missing_duration": missing.get(iso, 0),
                "total_seconds": round(total_seconds, 2),
                "total_hours": round(total_seconds / 3600, 4),
                "Category": CATEGORY_VALUE,
                "License": LICENSE_VALUE,
                "Who_transcribed": WHO_TRANSCRIBED_VALUE,
                "Transcript_exists": TRANSCRIPT_EXISTS_VALUE,
            }
        )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)

    total_missing = sum(missing.values())
    print(f"Summarized {len(summary_rows)} language(s) from {input_path}")
    if total_missing:
        print(
            f"NOTE: {total_missing} recording(s) across all languages had no "
            f"parseable duration and were excluded from the sums (see "
            f"num_missing_duration per language in the output)."
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize Lingua Libre harvest durations by language."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to the per-recording harvest CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to write the per-language summary CSV. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()
    summarize(args.input, args.output)
