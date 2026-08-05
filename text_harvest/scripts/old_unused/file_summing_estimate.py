"""
file_summing_estimate.py

Application step: for a language with NO per-language HF config (the
fallback case - see resolve_config_names in tier_a_harvest_v3.py),
estimates size by isolating that language's files and summing them,
using the CHEAPEST correct method per file type - NEVER downloading
the actual dataset content. Two different methods, by format:

- Parquet / Arrow: EXACT row count via a footer-only read (a tiny
  HTTP range request, not a download) - reuses
  get_parquet_row_count/get_arrow_row_count from tier_a_harvest_v2.py.
  No calibration needed, no estimate - this is a real count.
- JSONL / CSV / TXT: no footer/index exists for these formats, so an
  exact count would require reading the whole file - not feasible at
  harvest scale (confirmed by direct testing - see conversation).
  Instead, uses the CALIBRATED bytes-per-row/bytes-per-word ratio from
  calibrate_text_byte_ratios.py, applied to the file's byte size
  (already free, from the Hub's own file-listing metadata) - an
  ESTIMATE, explicitly flagged as such, never presented as exact.
- Any other extension with no calibration data at all: byte total is
  recorded, flagged needs_manual_review, no unit conversion attempted.

--------------------------------------------------------------------
LANGUAGE FILE ISOLATION
--------------------------------------------------------------------
Reuses isolate_language_files from tier_a_harvest_v2.py - path-segment
matching (splits on common separators, keeps files where the language
code appears as a COMPLETE segment).

--------------------------------------------------------------------
NEVER MERGE UNITS
--------------------------------------------------------------------
Row-based estimates (Parquet/Arrow/JSONL/CSV) and word-based estimates
(TXT) are kept STRICTLY SEPARATE in the output - same "never sum
different units into one number" principle used throughout this
project. A language with both Parquet and TXT files gets two
figures, not one merged one.

--------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------
pip install pandas huggingface_hub
"""
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from tier_a_harvest_v2 import (
    api, isolate_language_files, get_parquet_row_count, get_arrow_row_count,
    is_excluded_file, is_archive_file,
)

# ---- CONFIG ----
CALIBRATION_FILE = "../data/calibration/text_byte_ratio_calibration.csv"


def load_calibration():
    """Returns {extension: {"unit":..., "bytes_per_unit_median":...}}."""
    if not Path(CALIBRATION_FILE).exists():
        print(f"WARNING: {CALIBRATION_FILE} not found - run calibrate_text_byte_ratios.py first. "
              f"JSONL/CSV/TXT files will be recorded as byte-only, flagged needs_manual_review.")
        return {}
    df = pd.read_csv(CALIBRATION_FILE)
    return {row["extension"]: row.to_dict() for _, row in df.iterrows()}


def estimate_language_size(dataset_id, lang_code, calibration):
    """
    Returns a dict: {row_based_units, row_based_source (exact/estimated),
    word_based_units, byte_only_bytes (unconverted), needs_manual_review,
    flag_reason, isolated_file_count}. Never downloads dataset content -
    only file-listing metadata (free) plus Parquet/Arrow footer reads
    (tiny range requests, not full downloads).

    Returns a flagged dict (not a crash) if the dataset_id itself
    doesn't resolve at all - e.g. a typo, a renamed/deleted repo, or a
    gated repo you're not authenticated for. See flag_reason.
    """
    try:
        info = api.dataset_info(dataset_id, files_metadata=True)
    except Exception as e:
        print(f"Could not resolve dataset {dataset_id!r}: {type(e).__name__}: {e}")
        return {
            "dataset_id": dataset_id, "lang_code": lang_code,
            "isolated_file_count": None,
            "row_based_units": None, "row_based_source": None,
            "word_based_units": None, "word_based_source": None,
            "byte_only_bytes": None,
            "needs_manual_review": True,
            "flag_reason": f"dataset_not_resolvable: {type(e).__name__}",
        }

    siblings = info.siblings

    lang_files = isolate_language_files(siblings, lang_code)
    lang_files = [f for f in lang_files if not is_excluded_file(f) and not is_archive_file(f)]

    sibling_by_name = {s.rfilename: s for s in siblings}
    base_url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/"

    row_based_units = 0
    row_based_is_exact = True  # flips to False if any row-based file needed a calibrated estimate
    word_based_units = 0
    byte_only_bytes = 0
    byte_only_extensions = set()

    for fname in lang_files:
        ext = "." + fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
        file_size = sibling_by_name[fname].size or 0

        if ext == ".parquet":
            count = get_parquet_row_count(base_url + fname)
            if count is not None:
                row_based_units += count
                continue
            # footer read failed - fall through to byte-only below
        elif ext in (".arrow", ".feather"):
            count = get_arrow_row_count(base_url + fname)
            if count is not None:
                row_based_units += count
                continue

        elif ext in (".jsonl", ".csv") and ext in calibration:
            ratio = calibration[ext]["bytes_per_unit_median"]
            row_based_units += file_size / ratio
            row_based_is_exact = False
            continue

        elif ext == ".txt" and ext in calibration:
            ratio = calibration[ext]["bytes_per_unit_median"]
            word_based_units += file_size / ratio
            continue

        # nothing above matched - byte-only fallback, flagged
        byte_only_bytes += file_size
        byte_only_extensions.add(ext)

    needs_manual_review = byte_only_bytes > 0
    flag_reason = (f"no_calibration_or_footer_method_for: {';'.join(sorted(byte_only_extensions))}"
                   if byte_only_extensions else None)

    return {
        "dataset_id": dataset_id, "lang_code": lang_code,
        "isolated_file_count": len(lang_files),
        "row_based_units": row_based_units if row_based_units else None,
        "row_based_source": ("exact" if row_based_is_exact else "calibrated_estimate") if row_based_units else None,
        "word_based_units": word_based_units if word_based_units else None,
        "word_based_source": "calibrated_estimate" if word_based_units else None,
        "byte_only_bytes": byte_only_bytes if byte_only_bytes else None,
        "needs_manual_review": needs_manual_review,
        "flag_reason": flag_reason,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate a language's size within a dataset via free metadata + calibrated ratios.")
    parser.add_argument("dataset_id")
    parser.add_argument("lang_code")
    args = parser.parse_args()

    calibration = load_calibration()
    result = estimate_language_size(args.dataset_id, args.lang_code, calibration)
    for k, v in result.items():
        print(f"{k}: {v}")
