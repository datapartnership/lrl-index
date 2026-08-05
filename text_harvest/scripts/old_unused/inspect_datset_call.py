"""
inspect_dataset.py

Diagnostic tool, not part of any harvest pipeline: loads one specific
(dataset_id, language_code) config from Hugging Face and prints out
what's actually there - available configs/splits, the feature schema,
and a handful of real sample rows. Also supports summing real total
duration across chosen splits (see --sum-splits).

--------------------------------------------------------------------
WHY STREAMING BY DEFAULT
--------------------------------------------------------------------
streaming=True (the default here) avoids downloading the full dataset
just to look at a few rows - critical for audio datasets, which can
be many GB even for one language config.

--------------------------------------------------------------------
WHY AUDIO DECODING IS DISABLED
--------------------------------------------------------------------
Any Audio-typed column is cast to decode=False before use, both for
inspection and for the duration-summing mode - this tool only ever
needs metadata (paths, durations, transcripts), never the actual
decoded waveform, so it never needs torchcodec/soundfile installed.

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
Inspect (peek at sample rows):
  python inspect_dataset.py <dataset_id> <language_code> [--split SPLIT] [--num-examples N]

Sum real total duration across chosen splits:
  python inspect_dataset.py <dataset_id> <language_code> --sum-splits train dev test

Examples:
  python inspect_dataset.py facebook/multilingual_librispeech german
  python inspect_dataset.py facebook/multilingual_librispeech german --sum-splits train dev test

--------------------------------------------------------------------
IMPORTANT: CHOOSE --sum-splits DELIBERATELY, DON'T JUST SUM EVERYTHING
--------------------------------------------------------------------
Some datasets have splits that are SUBSETS of a larger split, not
additional data - e.g. Multilingual LibriSpeech's "9_hours" and
"1_hours" splits are curated subsets carved out of "train" for
low-resource experiments, not separate audio. Summing train + 9_hours
+ 1_hours would DOUBLE-COUNT that data. This script prints a warning
for any split name matching a number+time-unit pattern, but does NOT
auto-exclude anything - you decide which splits are genuinely
disjoint for the dataset you're looking at (check its README/paper
if unsure) and pass exactly those to --sum-splits.

--------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------
pip install datasets
(HF_TOKEN env var recommended for gated datasets)
"""
import argparse
import os
import re
import sys

from datasets import Audio, get_dataset_config_names, get_dataset_split_names, load_dataset, load_dataset_builder

MAX_FIELD_DISPLAY_CHARS = 300  # truncate long fields (raw audio bytes, full document text) for readability
SUSPICIOUS_SUBSET_PATTERN = re.compile(r"^\d+_(hour|min|hr)s?$", re.IGNORECASE)


def _disable_audio_decoding(ds):
    """Casts every Audio-typed column to decode=False. Returns the
    (possibly) modified dataset and the list of audio column names
    found, for logging."""
    audio_columns = [name for name, feat in ds.features.items() if isinstance(feat, Audio)]
    for col in audio_columns:
        ds = ds.cast_column(col, Audio(decode=False))
    return ds, audio_columns


def inspect(dataset_id, language_code, split=None, num_examples=3, streaming=True):
    print(f"Dataset: {dataset_id}")
    print(f"Requested config (language code): {language_code}")
    print("-" * 60)

    # ---- Step 1: does this config actually exist? ----
    try:
        available_configs = get_dataset_config_names(dataset_id)
    except Exception as e:
        print(f"Could not list configs for {dataset_id}: {type(e).__name__}: {e}")
        available_configs = None

    if available_configs is not None:
        preview = available_configs[:30]
        suffix = " ..." if len(available_configs) > 30 else ""
        print(f"Available configs ({len(available_configs)}): {preview}{suffix}")

        if language_code not in available_configs:
            print(f"\n'{language_code}' is NOT an exact match among available configs.")
            near_matches = [c for c in available_configs if language_code.lower() in c.lower()]
            if near_matches:
                print(f"Possible near-matches: {near_matches}")
            else:
                print("No obvious near-matches found either - double check the code against the list above.")
            print("Proceeding anyway in case load_dataset resolves it some other way...")
    print("-" * 60)

    # ---- Step 2: what splits exist for this config? ----
    try:
        splits = get_dataset_split_names(dataset_id, language_code)
        print(f"Available splits for this config: {splits}")
    except Exception as e:
        print(f"Could not list splits: {type(e).__name__}: {e}")
        splits = None

    target_split = split or (splits[0] if splits else "train")
    print(f"Will load split: {target_split!r} (streaming={streaming})")
    print("-" * 60)

    # ---- Step 3: actually load it ----
    try:
        ds = load_dataset(
            dataset_id, language_code, split=target_split, streaming=streaming,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as e:
        print(f"load_dataset() failed: {type(e).__name__}: {e}")
        print("\nCommon causes: wrong config/split name (see lists above), a gated dataset needing "
              "license acceptance + HF_TOKEN, or a dataset using a legacy loading script that needs "
              "trust_remote_code=True (not enabled here by default).")
        sys.exit(1)

    # ---- Step 4: feature schema ----
    try:
        print("Feature schema:")
        for name, feat in ds.features.items():
            print(f"  {name}: {feat}")
    except Exception as e:
        print(f"Could not read .features directly: {type(e).__name__}: {e}")
    print("-" * 60)

    # ---- Step 4b: row count (free, from builder info - no download needed) ----
    try:
        builder = load_dataset_builder(dataset_id, language_code, token=os.environ.get("HF_TOKEN"))
        split_info = builder.info.splits.get(target_split) if builder.info.splits else None
        if split_info is not None and split_info.num_examples is not None:
            print(f"num_rows for split {target_split!r}: {split_info.num_examples}")
        else:
            print(f"num_rows for split {target_split!r}: not available from precomputed builder info "
                  f"(dataset may not publish this metadata, or use a legacy loading script)")
    except Exception as e:
        print(f"Could not resolve num_rows from builder info: {type(e).__name__}: {e}")
    print("-" * 60)

    # ---- Step 4c: skip audio decoding for inspection purposes ----
    try:
        ds, audio_columns = _disable_audio_decoding(ds)
        if audio_columns:
            print(f"Audio column(s) detected ({audio_columns}) - decoding disabled for inspection "
                  f"(install torchcodec separately if you need actual decoded waveforms later).")
    except Exception as e:
        print(f"Could not inspect/adjust Audio columns: {type(e).__name__}: {e}")
    print("-" * 60)

    # ---- Step 5: real sample rows ----
    print(f"First {num_examples} example(s):")
    try:
        iterator = iter(ds)
        for i in range(num_examples):
            example = next(iterator)
            print(f"\n--- example {i} ---")
            for key, value in example.items():
                value_repr = repr(value)
                if len(value_repr) > MAX_FIELD_DISPLAY_CHARS:
                    value_repr = (value_repr[:MAX_FIELD_DISPLAY_CHARS]
                                  + f"... [truncated, full repr length {len(value_repr)} chars]")
                print(f"  {key}: {value_repr}")
    except StopIteration:
        print("  (dataset exhausted before reaching the requested example count)")
    except Exception as e:
        print(f"Error while iterating examples: {type(e).__name__}: {e}")


def sum_total_duration(dataset_id, language_code, splits_to_sum, duration_field="audio_duration",
                        duration_unit="seconds", progress_every=5000):
    """
    Streams through each split in splits_to_sum and sums duration_field
    to get a REAL, exact total (not an estimate) - audio is never
    decoded, only the duration field is read, so this is cheap despite
    iterating every row.

    Does NOT auto-detect overlapping splits (e.g. MLS's 9_hours/1_hours
    subsets of train) - prints a warning for suspicious split names,
    but the choice of which splits are genuinely disjoint is yours.
    """
    for split in splits_to_sum:
        if SUSPICIOUS_SUBSET_PATTERN.match(split):
            print(f"  WARNING: split {split!r} looks like it could be a curated SUBSET of a larger "
                  f"split (e.g. MLS's 9_hours/1_hours are subsets of train, not additional data) - "
                  f"summing it alongside a split it overlaps with will double-count. Proceed only if "
                  f"you're deliberately choosing this split and NOT also summing its parent split.")

    grand_total_seconds = 0.0
    per_split_totals = {}

    for split in splits_to_sum:
        print(f"\nSumming split {split!r}...")
        try:
            ds = load_dataset(dataset_id, language_code, split=split, streaming=True,
                               token=os.environ.get("HF_TOKEN"))
        except Exception as e:
            print(f"  could not load split {split!r}: {type(e).__name__}: {e}")
            continue

        ds, _ = _disable_audio_decoding(ds)

        if duration_field not in ds.features:
            print(f"  '{duration_field}' not found in this split's features "
                  f"({list(ds.features.keys())}) - skipping. Pass --duration-field to point at the "
                  f"right column name if it's called something else here.")
            continue

        split_total_seconds = 0.0
        row_count = 0
        for example in ds:
            value = example.get(duration_field)
            if value is not None:
                split_total_seconds += value if duration_unit == "seconds" else value * 60
            row_count += 1
            if row_count % progress_every == 0:
                print(f"    ...{row_count} row(s) processed so far, running total "
                      f"{split_total_seconds / 3600:.2f} hrs")

        split_total_hours = split_total_seconds / 3600
        per_split_totals[split] = {"rows": row_count, "hours": split_total_hours}
        grand_total_seconds += split_total_seconds
        print(f"  {split}: {row_count} row(s), {split_total_hours:.2f} hours")

    print("\n=== SUMMARY ===")
    for split, totals in per_split_totals.items():
        print(f"  {split}: {totals['rows']} rows, {totals['hours']:.2f} hours")
    print(f"  TOTAL across summed splits: {grand_total_seconds / 3600:.2f} hours")
    print(f"  (only the splits you explicitly listed in --sum-splits were included)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect or total-duration-sum a (dataset, language) config from Hugging Face.")
    parser.add_argument("dataset_id", help="e.g. facebook/multilingual_librispeech")
    parser.add_argument("language_code", help="the config/language name to load, e.g. german, de, deu")
    parser.add_argument("--split", default=None, help="[inspect mode] split to load (default: first available)")
    parser.add_argument("--num-examples", type=int, default=3, help="[inspect mode] number of sample rows to print")
    parser.add_argument("--no-streaming", action="store_true", help="[inspect mode] disable streaming")
    parser.add_argument("--sum-splits", nargs="+", default=None,
                         help="[sum mode] one or more DISJOINT split names to sum real total duration across, "
                              "e.g. --sum-splits train dev test")
    parser.add_argument("--duration-field", default="audio_duration",
                         help="column name holding per-row duration (default: audio_duration)")
    parser.add_argument("--duration-unit", choices=["seconds", "minutes"], default="seconds",
                         help="unit of the duration field (default: seconds)")
    args = parser.parse_args()

    if args.sum_splits:
        sum_total_duration(
            args.dataset_id, args.language_code, args.sum_splits,
            duration_field=args.duration_field, duration_unit=args.duration_unit,
        )
    else:
        inspect(
            args.dataset_id, args.language_code, split=args.split,
            num_examples=args.num_examples, streaming=not args.no_streaming,
        )
