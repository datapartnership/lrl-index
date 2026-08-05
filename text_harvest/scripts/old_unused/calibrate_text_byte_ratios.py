"""
calibrate_text_byte_ratios.py

Combined discovery + calibration step (run ONCE, on a small sample)
for the byte-size-summing fallback used when a language has no
per-language HF config. Per conversation: exact row counts for
Parquet/Arrow are already free via footer reads
(get_parquet_row_count/get_arrow_row_count in tier_a_harvest_v2.py) -
no calibration needed there. This script is ONLY for formats with no
footer/index to read cheaply - JSONL, CSV, and plain TXT - where an
exact count requires reading the whole file, which doesn't scale
across a full harvest. The calibrated ratio here lets you estimate
size from a file's byte count alone (free metadata, no download)
instead.

--------------------------------------------------------------------
TWO STAGES, ONE SCRIPT
--------------------------------------------------------------------
1. DISCOVERY: scans a small set of known dataset IDs (pulled from your
   config_availability_by_language.csv experiment output by default)
   for real .jsonl/.csv/.txt files, using free file-listing metadata
   only (no download at this stage).
2. CALIBRATION: downloads JUST the discovered sample files (not whole
   datasets), measures real bytes-per-row (.jsonl/.csv) or
   bytes-per-word (.txt), and writes the resulting ratios.

--------------------------------------------------------------------
WHAT GETS MEASURED
--------------------------------------------------------------------
- .jsonl, .csv: bytes-per-ROW (row = line, for these formats)
- .txt: bytes-per-WORD (no row concept for unstructured plain text -
  word count via whitespace split is the defensible comparable unit,
  per HPLT's own precedent of using wc-style word counts as the
  standard cross-language corpus statistic - see conversation)

--------------------------------------------------------------------
METHODOLOGY
--------------------------------------------------------------------
Per the literature crosscheck (ROOTS corpus, arXiv:2303.03915;
cross-lingual "information content per byte" calibration in
arXiv:1506.00572): a SMALL, real sample is downloaded ONCE to measure
the true ratio empirically, rather than assuming a fixed constant.
Median AND mean are recorded, since format overhead can be skewed by
outliers (e.g. one file with very long lines).

--------------------------------------------------------------------
OUTPUT
--------------------------------------------------------------------
CALIBRATION_OUTPUT_FILE - one row per extension:
  extension, unit (rows or words), sample_size, bytes_per_unit_mean,
  bytes_per_unit_median, sample_file_urls (for audit)

This file is what file_summing_estimate.py reads to convert a byte
count into an estimated row/word count - CALIBRATE ONCE, reuse across
the entire harvest, recalibrate periodically rather than per dataset.

--------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------
pip install requests pandas huggingface_hub
"""
import time
from pathlib import Path
from statistics import mean, median

import pandas as pd
import requests
from huggingface_hub import HfApi

# ---- CONFIG ----
CONFIG_AVAILABILITY_FILE = "../data/experiments/config_availability_by_language.csv"  # default source of dataset IDs to scan
CALIBRATION_OUTPUT_FILE = "../data/calibration/text_byte_ratio_calibration.csv"

MAX_DATASETS_TO_SCAN = 30       # discovery stage: keep this small and controlled, not a full crawl
SAMPLE_SIZE_PER_EXTENSION = 30  # calibration stage: per literature precedent (ROOTS-style), a small real sample
MAX_FILE_BYTES_TO_SAMPLE = 50_000_000  # skip files bigger than this - don't need a 5GB file to calibrate a ratio

EXTENSION_UNITS = {".jsonl": "rows", ".csv": "rows", ".tsv": "rows", ".txt": "words"}

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT = 60

api = HfApi()


def _with_retry(fn, description, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"    [retry {attempt + 1}/{max_retries}] {description}: {e} - waiting {wait}s")
            time.sleep(wait)
    print(f"    giving up on {description} after {max_retries} retries")
    return None


# ---- STAGE 1: DISCOVERY (free metadata only, no download) ----
def get_dataset_ids_to_scan(filepath=CONFIG_AVAILABILITY_FILE, max_datasets=MAX_DATASETS_TO_SCAN, explicit_ids=None):
    if explicit_ids:
        return explicit_ids[:max_datasets]
    if not Path(filepath).exists():
        print(f"{filepath} not found and no --dataset-ids given - nothing to scan.")
        return []
    df = pd.read_csv(filepath)
    unique_ids = df["dataset_id"].dropna().unique().tolist()
    print(f"Found {len(unique_ids)} unique dataset ID(s) in {filepath}")
    return unique_ids[:max_datasets]


def discover_candidate_files(dataset_ids):
    """
    Returns {extension: [(dataset_id, file_path), ...]} across all
    scanned datasets - free file-listing metadata only (no download).
    """
    by_extension = {ext: [] for ext in EXTENSION_UNITS}
    for dataset_id in dataset_ids:
        print(f"[{dataset_id}] scanning for .jsonl/.csv/.txt files...")
        try:
            info = api.dataset_info(dataset_id, files_metadata=True)
        except Exception as e:
            print(f"  could not inspect {dataset_id}: {type(e).__name__}: {e}")
            continue

        found = 0
        for sibling in info.siblings:
            fname = sibling.rfilename
            ext = "." + fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
            size = sibling.size or 0
            if ext in EXTENSION_UNITS and 0 < size <= MAX_FILE_BYTES_TO_SAMPLE:
                by_extension[ext].append((dataset_id, fname))
                found += 1
        print(f"  found {found} candidate file(s)")
    return by_extension


# ---- STAGE 2: CALIBRATION (downloads only the sampled files) ----
def download_and_measure(dataset_id, file_path, extension):
    """Downloads ONE sampled file and measures real byte size vs.
    real row/word count. Returns (bytes, unit_count) or None."""
    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{file_path}"

    def _fetch():
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.content

    content = _with_retry(_fetch, f"download {dataset_id}/{file_path}", max_retries=2)
    if content is None:
        return None

    num_bytes = len(content)
    if num_bytes == 0 or num_bytes > MAX_FILE_BYTES_TO_SAMPLE:
        return None

    text = content.decode("utf-8", errors="ignore")
    if extension in (".jsonl", ".csv", ".tsv"):
        unit_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    else:  # .txt
        unit_count = len(text.split())

    if unit_count == 0:
        return None
    return num_bytes, unit_count


def calibrate_extension(extension, dataset_file_pairs):
    """Samples up to SAMPLE_SIZE_PER_EXTENSION real pairs, measures
    each, returns a summary dict or None if nothing was measurable."""
    ratios = []
    sample_urls = []
    for dataset_id, file_path in dataset_file_pairs[:SAMPLE_SIZE_PER_EXTENSION]:
        print(f"    sampling {dataset_id}/{file_path}...")
        result = download_and_measure(dataset_id, file_path, extension)
        if result is None:
            continue
        num_bytes, unit_count = result
        ratios.append(num_bytes / unit_count)
        sample_urls.append(f"{dataset_id}/{file_path}")

    if not ratios:
        print(f"    no usable samples for {extension} - could not calibrate")
        return None

    return {
        "extension": extension,
        "unit": EXTENSION_UNITS[extension],
        "sample_size": len(ratios),
        "bytes_per_unit_mean": mean(ratios),
        "bytes_per_unit_median": median(ratios),
        "sample_file_urls": ";".join(sample_urls[:10]),
    }


def main(dataset_ids=None):
    ids_to_scan = get_dataset_ids_to_scan(explicit_ids=dataset_ids)
    if not ids_to_scan:
        return

    candidates_by_extension = discover_candidate_files(ids_to_scan)

    results = []
    for extension, pairs in candidates_by_extension.items():
        print(f"\n[{extension}] {len(pairs)} candidate file(s) discovered, sampling up to {SAMPLE_SIZE_PER_EXTENSION}...")
        result = calibrate_extension(extension, pairs)
        if result:
            results.append(result)
            print(f"    calibrated: {result['bytes_per_unit_median']:.1f} bytes/{result['unit'][:-1]} "
                  f"(median, n={result['sample_size']})")

    if not results:
        print("\nNo extensions calibrated - nothing to write.")
        return

    Path(CALIBRATION_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(CALIBRATION_OUTPUT_FILE, index=False)
    print(f"\nWrote calibration for {len(results)} extension(s) to {CALIBRATION_OUTPUT_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Discover and calibrate text byte-to-unit ratios in one pass.")
    parser.add_argument("--dataset-ids", nargs="+", default=None,
                         help="Specific dataset IDs to scan (default: pulled from config_availability_by_language.csv)")
    args = parser.parse_args()

    main(dataset_ids=args.dataset_ids)
