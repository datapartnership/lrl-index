"""
check_partial_conversion.py

Quantifies how many datasets in a sample have PARTIAL Parquet
conversion - HF only converts up to ~5GB of a dataset, so anything
bigger only gets partially converted, with the remainder never
represented in the Parquet files at all.

--------------------------------------------------------------------
HOW THIS IS CHECKED
--------------------------------------------------------------------
The Datasets Server /parquet endpoint response has a TOP-LEVEL
"partial": true/false field, confirmed directly from HF's own docs
(https://huggingface.co/docs/dataset-viewer/en/parquet#partially-converted-datasets).
Partially-converted datasets ALSO get their split names prefixed
"partial-" (e.g. "partial-train" instead of "train") - checked here
as a second, independent confirmation signal, not just the boolean
alone.

--------------------------------------------------------------------
SAMPLING
--------------------------------------------------------------------
Pulls a broad sample of dataset IDs directly from the Hub via
api.list_datasets(limit=N) - NOT scoped to your language crosswalk,
just a general cross-section of datasets on Hugging Face. Default
sort is the Hub's own default ordering; use --sort to change it (e.g.
"downloads" for popular datasets specifically).

--------------------------------------------------------------------
NOT YET VALIDATED AGAINST LIVE DATA
--------------------------------------------------------------------
No network access to huggingface.co / datasets-server.huggingface.co
from the environment this was written in. Response parsing IS
validated against HF's own documented example response - see
conversation - but the live call itself is untested. Run --n 5 first.

--------------------------------------------------------------------
DEPENDENCIES / SETUP
--------------------------------------------------------------------
pip install pandas requests huggingface_hub
export HF_TOKEN=hf_...
"""
import os
import time
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import HfApi

# ---- CONFIG ----
OUTPUT_FILE = "../data/experiments/partial_conversion_check.csv"

DATASETS_SERVER_PARQUET_URL = "https://datasets-server.huggingface.co/parquet"

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT = 30

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


def fetch_parquet_info(dataset_id, hf_token):
    """
    Returns (data_or_None, status_code_or_None).

    Three possible outcomes, kept distinct rather than lumped together:
    - (json_data, None): success (200) - real Parquet info.
    - (None, 404 or 501): HF's own determination that this dataset has
      no Parquet representation - a genuine "not convertible" case.
    - (None, some_other_code): the REQUEST itself failed for a reason
      unrelated to whether the dataset is convertible - auth issues
      (401/403), server errors (500/502/503), rate limiting (429), etc.
      Recorded as its own status_code rather than silently treated the
      same as "not convertible", since these mean "we don't actually
      know" rather than "HF says no".
    """
    def _fetch():
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        return requests.get(DATASETS_SERVER_PARQUET_URL, params={"dataset": dataset_id},
                             headers=headers, timeout=REQUEST_TIMEOUT)

    resp = _with_retry(_fetch, f"fetch /parquet for {dataset_id}", max_retries=2)
    if resp is None:
        return None, "request_failed_after_retries"  # connection-level failure, not an HTTP status

    if resp.status_code == 200:
        return resp.json(), None
    return None, resp.status_code


def check_one_dataset(dataset_id, hf_token):
    data, status_code = fetch_parquet_info(dataset_id, hf_token)

    if data is None:
        # Genuine "not convertible" - HF's own determination
        if status_code in (404, 501):
            return {"dataset_id": dataset_id, "parquet_convertible": False, "status_code": status_code,
                    "partial": None, "partial_split_names_seen": None, "num_shards": None}
        # Anything else (401, 403, 429, 500, 502, connection failure, etc.) -
        # the request itself failed, we genuinely don't know convertibility.
        return {"dataset_id": dataset_id, "parquet_convertible": None, "status_code": status_code,
                "partial": None, "partial_split_names_seen": None, "num_shards": None}

    parquet_files = data.get("parquet_files", [])
    is_partial = data.get("partial", False)
    partial_splits = [f["split"] for f in parquet_files if f.get("split", "").startswith("partial-")]

    return {
        "dataset_id": dataset_id,
        "parquet_convertible": True,
        "status_code": 200,
        "partial": is_partial,
        "partial_split_names_seen": ";".join(partial_splits) if partial_splits else None,
        "num_shards": len(parquet_files),
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
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Sampling {n} dataset(s) from the Hub" + (f" (sorted by {sort})" if sort else "") + "...")
    dataset_ids = sample_dataset_ids(n, sort=sort)
    print(f"Got {len(dataset_ids)} dataset ID(s)")

    results = []
    for dataset_id in dataset_ids:
        print(f"\n[{dataset_id}] checking...")
        result = check_one_dataset(dataset_id, hf_token)
        results.append(result)
        print(f"    parquet_convertible={result['parquet_convertible']}, partial={result['partial']}")

    result_df = pd.DataFrame(results)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n=== SUMMARY (n={len(result_df)}) ===")
    not_convertible = result_df[result_df["parquet_convertible"] == False]
    unknown = result_df[result_df["parquet_convertible"].isna()]
    print(f"Not Parquet-convertible (genuine 404/501 from HF): {len(not_convertible)}")
    print(f"Unknown - request failed for another reason: {len(unknown)}")
    if len(unknown):
        print(f"  Status code breakdown:\n{unknown['status_code'].value_counts()}")

    convertible = result_df[result_df["parquet_convertible"] == True]
    if len(convertible):
        print(f"\nAmong {len(convertible)} Parquet-convertible dataset(s):")
        print(f"  Partial conversion: {convertible['partial'].sum()} ({convertible['partial'].mean():.1%})")

    print(f"\nWrote full detail to {OUTPUT_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check partial Parquet conversion status across a broad Hub sample.")
    parser.add_argument("--n", type=int, default=100, help="How many datasets to sample from the Hub (default: 100)")
    parser.add_argument("--sort", type=str, default=None,
                         help="Hub sort order, e.g. 'downloads' for popular datasets (default: Hub's own default order)")
    args = parser.parse_args()

    main(n=args.n, sort=args.sort)
