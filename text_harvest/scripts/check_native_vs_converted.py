"""
check_native_vs_converted.py

Quantifies how many datasets in a sample are NATIVE Parquet (uploaded
directly as Parquet, at risk of non-standardized internal structure)
vs AUTO-CONVERTED by Hugging Face's own pipeline (standardized
structure, since HF's conversion process is consistent).

--------------------------------------------------------------------
HOW THIS IS CHECKED
--------------------------------------------------------------------
Per HF's own docs (https://huggingface.co/docs/dataset-viewer/en/parquet#parquet-native-datasets):
"When the dataset is already in Parquet format, the data are not
converted and the files in refs/convert/parquet are links to the
original files."

It calls api.dataset_info(dataset_id), which returns the dataset's main branch file listing —
.siblings is the list of every file sitting in the repo on main, not on the special
refs/convert/parquet branch HF generates. It collects every distinct file extension present among
those main-branch files (.json, .csv, .parquet, .txt, whatever's actually there).
The test itself is just: does .parquet show up in that set of main-branch extensions?

If yes → "native"
If no → "auto_converted"

--------------------------------------------------------------------
SAMPLING
--------------------------------------------------------------------
Pulls a broad sample of dataset IDs directly from the Hub via
api.list_datasets(limit=N) - NOT scoped to your language crosswalk,
just a general cross-section of datasets on Hugging Face.

--------------------------------------------------------------------
DEPENDENCIES / SETUP
--------------------------------------------------------------------
pip install pandas huggingface_hub
export HF_TOKEN=hf_...
"""
import os
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

# ---- CONFIG ----
OUTPUT_FILE = "../data/experiments/native_vs_converted_check.csv"

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 5

api = HfApi(token=os.environ.get("HF_TOKEN"))


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


def check_one_dataset(dataset_id):
    """
    Fetches the main branch's file listing and checks for .parquet
    files directly. Returns a row dict.
    """
    info = _with_retry(lambda: api.dataset_info(dataset_id, files_metadata=False),
                        f"dataset_info({dataset_id})")
    if info is None:
        return {"dataset_id": dataset_id, "resolved": False,
                "native_or_converted": None, "main_branch_extensions": None}

    main_branch_files = [s.rfilename for s in info.siblings]
    main_branch_extensions = set()
    for f in main_branch_files:
        if "." in f:
            main_branch_extensions.add("." + f.lower().rsplit(".", 1)[-1])

    native_status = "native" if ".parquet" in main_branch_extensions else "auto_converted"

    return {
        "dataset_id": dataset_id,
        "resolved": True,
        "native_or_converted": native_status,
        "main_branch_extensions": ";".join(sorted(main_branch_extensions)) if main_branch_extensions else None,
    }


def sample_dataset_ids(n, sort=None):
    """Pulls n dataset IDs directly from the Hub - a general
    cross-section, not scoped to any particular language or project."""
    def _fetch():
        return list(api.list_datasets(limit=n, sort=sort))

    results = _with_retry(_fetch, f"list_datasets(limit={n}, sort={sort})", max_retries=2)
    if results is None:
        return []
    return [d.id for d in results]


def main(n=100, sort=None):
    print(f"Sampling {n} dataset(s) from the Hub" + (f" (sorted by {sort})" if sort else "") + "...")
    dataset_ids = sample_dataset_ids(n, sort=sort)
    print(f"Got {len(dataset_ids)} dataset ID(s)")

    results = []
    for dataset_id in dataset_ids:
        print(f"\n[{dataset_id}] checking...")
        result = check_one_dataset(dataset_id)
        results.append(result)
        print(f"    resolved={result['resolved']}, native_or_converted={result['native_or_converted']}, "
              f"extensions={result['main_branch_extensions']}")

    result_df = pd.DataFrame(results)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n=== SUMMARY (n={len(result_df)}) ===")
    print(f"Could not resolve: {(~result_df['resolved']).sum()}")

    resolved = result_df[result_df["resolved"] == True]
    if len(resolved):
        print(f"\nAmong {len(resolved)} resolved dataset(s):")
        print(resolved["native_or_converted"].value_counts())
        print(f"\n  ({resolved['native_or_converted'].value_counts(normalize=True).round(3) * 100})")

    print(f"\nWrote full detail to {OUTPUT_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check native vs auto-converted Parquet status across a broad Hub sample.")
    parser.add_argument("--n", type=int, default=100, help="How many datasets to sample from the Hub (default: 100)")
    parser.add_argument("--sort", type=str, default=None,
                         help="Hub sort order, e.g. 'downloads' for popular datasets (default: Hub's own default order)")
    args = parser.parse_args()

    main(n=args.n, sort=args.sort)
