"""
fineweb2_yield_multiplier.py

Computes the Tier C "raw -> usable yield" multiplier per language, per
the Week 3 plan: yield(lang) = clean FineWeb-2 bytes (matched to your
existing raw Common Crawl data's crawl coverage) / your raw CC bytes
over that same matched set of crawls.

APPROXIMATE approach: FineWeb-2 does not publish a per-crawl byte
breakdown, only a per-language(-script) AGGREGATE total (see
fineweb2-language-distribution.csv). Per-crawl information only lives
inside individual document rows, via the `dump` field. Rather than
reading actual row/text data (expensive, exact), this script queries
Hugging Face's Datasets Server /statistics endpoint on the `dump`
column - a metadata-only call that returns ROW COUNT per dump value.
The aggregate utf8_bytes total is then APPORTIONED across dumps
proportionally to each dump's row-count share. This assumes average
document size is roughly uniform across crawls within a language -
the yield_coverage_fraction column is your signal to flag a language
for manual review if that assumption looks shaky (low coverage from
very few matched dumps).

--------------------------------------------------------------------
TWO SEPARATE THINGS THAT CAN GO WRONG, TRACKED SEPARATELY
--------------------------------------------------------------------
1. "not_in_cc_data" - the language code isn't present in your Common
   Crawl harvest at all (see tier_c_common_crawl_harvest.py). Likely
   cause: Common Crawl's own language stats use CLD2 (~160 languages
   covered), while FineWeb-2 uses GlotLID (2000+ languages) - a real,
   structural coverage gap, not a bug. Checked BEFORE any API call is
   made, so these never touch Hugging Face's servers at all.
2. "fw2_api_unavailable" - the language IS in your CC data, but HF's
   Datasets Server /statistics call returned 404 or 500 for that
   config. Observed cause for at least some cases: Datasets Server
   appears to run Parquet-conversion jobs per CONFIG (covering all its
   splits together), so a config whose `test` split is absent or has
   only a handful of documents can fail conversion for the whole
   config - including `train`, even though `train` itself has good
   data. Not retried (both 404 and 500 are treated as non-transient
   here) - see fetch_dump_row_distribution().

Both categories are written to SKIPPED_LOG_FILE with their reason, and
are NOT included in the final yield output table - see main().

--------------------------------------------------------------------
DEPENDENCIES / SETUP
--------------------------------------------------------------------
pip install pandas requests
export HF_TOKEN=hf_...   (recommended - higher rate limits on Datasets Server)

--------------------------------------------------------------------
VALIDATED
--------------------------------------------------------------------
Confirmed against live data: the 500-on-config-conversion failure mode
was reproduced directly against the real Datasets Server API (config
abn_Latn) during development of this script. The CLD2/GlotLID coverage
gap theory (aai_Latn, aak_Latn, aau_Latn, aaz_Latn showing 0 matched
bytes) is plausible but NOT yet directly confirmed against the raw CC
CSV - worth a quick grep check the first time this runs for real (see
conversation) to confirm those codes are genuinely absent from your CC
data rather than something else going on.
"""
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# ---- CONFIG ----
FW2_LANG_DIST_CSV = "https://raw.githubusercontent.com/huggingface/fineweb-2/main/fineweb2-language-distribution.csv"
FW2_DATASET_ID = "HuggingFaceFW/fineweb-2"
DATASETS_SERVER_STATISTICS_URL = "https://datasets-server.huggingface.co/statistics"

# Points at tier_c_common_crawl_harvest.py's GRANULAR output (language x
# crawl, one row per pair) - NOT the cumulative or latest-per-language
# tables, since dump-level matching needs the crawl_id breakdown.
CC_RAW_BYTES_FILE = "../data/crawl/processed/tier_c_language_bytes_by_crawl.csv"
CC_LANGUAGE_COL = "language_iso_639_3"   # already ISO 639-3 - matches FineWeb-2's `code` column directly
CC_CRAWL_ID_COL = "crawl_id"             # e.g. "CC-MAIN-2026-25" - same CC-MAIN-YYYY-WW format FineWeb-2's `dump` uses
CC_BYTES_COL = "approx_language_bytes"   # NOTE: already an estimate (page_share x total bytes downloaded),
                                          # and measures uncompressed HTTP content (HTML+markup), not extracted
                                          # text - so the yield computed here bakes in HTML->text extraction
                                          # ratio on top of quality filtering, not quality filtering alone.
                                          # State this plainly in the methods note.

OUTPUT_FILE = "../data/crawl/processed/yield_multiplier_by_language.csv"
SKIPPED_LOG_FILE = "../data/crawl/processed/yield_multiplier_skipped_languages.csv"

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT = 30

HF_TOKEN_HEADER = {}  # populated at runtime if HF_TOKEN is set - see main()


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


# ---- STEP 1: FineWeb-2's published aggregate table ----
def load_fineweb2_language_distribution():
    """
    Downloads and parses fineweb2-language-distribution.csv. Keeps
    only split == "train" and subsets that are NOT "_removed" (the
    filtered/clean data actually used for training). utf8_bytes is
    coerced to int; rows where it's missing ("-") are dropped.

    Returns a DataFrame with columns: subset, code, script, utf8_bytes.
    """
    def _fetch():
        resp = requests.get(FW2_LANG_DIST_CSV, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text

    csv_text = _with_retry(_fetch, "fetch fineweb2-language-distribution.csv", max_retries=3)
    if csv_text is None:
        raise RuntimeError("could not fetch FineWeb-2 language distribution CSV - aborting")

    df = pd.read_csv(StringIO(csv_text))
    df = df[(df["split"] == "train") & (~df["subset"].str.endswith("_removed"))].copy()
    df["utf8_bytes"] = pd.to_numeric(df["utf8_bytes"], errors="coerce")
    df = df.dropna(subset=["utf8_bytes"])
    df["utf8_bytes"] = df["utf8_bytes"].astype(int)
    return df[["subset", "code", "script", "utf8_bytes"]]


# ---- STEP 2: per-dump row-count distribution via Datasets Server ----
def fetch_dump_row_distribution(subset, split="train"):
    """
    Queries the /statistics endpoint for the `dump` column of one
    FineWeb-2 config. Returns a dict {dump_id: row_count}, or None if
    the config is unavailable (404) or its Parquet-conversion job
    failed (500 - observed to correlate with an absent/tiny `test`
    split for that config, which appears to break conversion for the
    WHOLE config including `train` - see module docstring). Neither
    is retried: both are treated as non-transient for this endpoint.
    """
    def _fetch():
        params = {"dataset": FW2_DATASET_ID, "config": subset, "split": split}
        resp = requests.get(DATASETS_SERVER_STATISTICS_URL, params=params,
                             headers=HF_TOKEN_HEADER, timeout=REQUEST_TIMEOUT)
        if resp.status_code in (404, 500):
            return None
        resp.raise_for_status()
        return resp.json()

    data = _with_retry(_fetch, f"fetch /statistics for {subset}", max_retries=MAX_RETRIES)
    if not data:
        return None

    for col in data.get("statistics", []):
        if col.get("column_name") == "dump":
            freqs = col.get("column_statistics", {}).get("frequencies")
            if freqs:
                return freqs
    print(f"    no categorical 'dump' column found in /statistics response for {subset}")
    return None


def apportion_bytes_by_dump(total_utf8_bytes, dump_row_counts):
    """
    Apportions total_utf8_bytes across dumps proportionally to each
    dump's share of total rows. Returns {dump_id: apportioned_bytes}.
    """
    total_rows = sum(dump_row_counts.values())
    if total_rows == 0:
        return {}
    return {
        dump: total_utf8_bytes * (row_count / total_rows)
        for dump, row_count in dump_row_counts.items()
    }


# ---- STEP 3: your existing raw CC byte data ----
def load_cc_raw_bytes():
    """
    Loads your Tier C raw CC byte data (tier_c_common_crawl_harvest.py's
    granular output). Returns a DataFrame indexed by (code, dump) with
    a raw_bytes column, summed if there are duplicate rows.
    """
    df = pd.read_csv(CC_RAW_BYTES_FILE)
    df = df.rename(columns={
        CC_LANGUAGE_COL: "code",
        CC_CRAWL_ID_COL: "dump",
        CC_BYTES_COL: "raw_bytes",
    })
    return df.groupby(["code", "dump"], as_index=False)["raw_bytes"].sum()


# ---- STEP 4-5: match crawls, compute yield ----
def compute_yield_for_subset(subset, code, total_utf8_bytes, cc_raw_by_dump_for_code):
    """
    For one FineWeb-2 subset (language-script) that we already know
    has at least some presence in the CC data (see main() - the
    not_in_cc_data check happens before this is ever called), fetches
    its dump distribution, apportions bytes, matches against the raw
    CC data's dumps, and returns a row dict - or None if the dump
    distribution couldn't be resolved (404/500 - flagged as
    "fw2_api_unavailable" by the caller).
    """
    dump_row_counts = fetch_dump_row_distribution(subset)
    if dump_row_counts is None:
        return None

    apportioned = apportion_bytes_by_dump(total_utf8_bytes, dump_row_counts)

    cc_dumps_for_code = set(cc_raw_by_dump_for_code["dump"])
    matched_dumps = set(apportioned.keys()) & cc_dumps_for_code

    matched_clean_bytes = sum(apportioned[d] for d in matched_dumps)
    matched_raw_bytes = cc_raw_by_dump_for_code[
        cc_raw_by_dump_for_code["dump"].isin(matched_dumps)
    ]["raw_bytes"].sum()

    return {
        "subset": subset,
        "code": code,
        "total_utf8_bytes": total_utf8_bytes,
        "fw2_dumps_seen": len(apportioned),
        "matched_dump_count": len(matched_dumps),
        "matched_clean_bytes": matched_clean_bytes,
        "matched_raw_bytes": matched_raw_bytes,
        "yield_coverage_fraction": (matched_clean_bytes / total_utf8_bytes) if total_utf8_bytes else None,
    }


def aggregate_to_language_level(subset_rows_df):
    """
    Sums matched_clean_bytes / matched_raw_bytes / total_utf8_bytes
    across scripts sharing the same language code, then computes the
    final yield ratio at the LANGUAGE level (per the plan).
    """
    grouped = subset_rows_df.groupby("code", as_index=False).agg(
        total_utf8_bytes=("total_utf8_bytes", "sum"),
        matched_clean_bytes=("matched_clean_bytes", "sum"),
        matched_raw_bytes=("matched_raw_bytes", "sum"),
        matched_dump_count=("matched_dump_count", "sum"),
        script_count=("subset", "count"),
    )
    grouped["yield"] = grouped.apply(
        lambda r: (r["matched_clean_bytes"] / r["matched_raw_bytes"]) if r["matched_raw_bytes"] else None,
        axis=1,
    )
    grouped["yield_coverage_fraction"] = grouped.apply(
        lambda r: (r["matched_clean_bytes"] / r["total_utf8_bytes"]) if r["total_utf8_bytes"] else None,
        axis=1,
    )
    return grouped


def main(test_n=None):
    if not HF_TOKEN_HEADER:
        import os
        if os.environ.get("HF_TOKEN"):
            HF_TOKEN_HEADER["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"

    print("Loading FineWeb-2 language distribution...")
    fw2_dist = load_fineweb2_language_distribution()

    print("Loading your raw CC byte data...")
    cc_raw = load_cc_raw_bytes()
    cc_codes_present = set(cc_raw["code"].unique())

    # Split the work up front: codes with no presence in your CC data
    # at all never touch the HF API - logged immediately as
    # not_in_cc_data, saving API calls/retries for languages that can
    # never produce a non-zero match anyway.
    in_cc = fw2_dist[fw2_dist["code"].isin(cc_codes_present)].copy()
    not_in_cc = fw2_dist[~fw2_dist["code"].isin(cc_codes_present)].copy()
    print(f"{len(in_cc)} subset(s) have a matching language code in your CC data; "
          f"{len(not_in_cc)} do not (logged, not queried).")

    if test_n is not None:
        in_cc = in_cc.head(test_n)
        print(f"TEST RUN: limited to first {test_n} subset(s) with CC data present")

    skipped_rows = [
        {"subset": row["subset"], "code": row["code"], "reason": "not_in_cc_data"}
        for _, row in not_in_cc.iterrows()
    ]

    subset_rows = []
    for _, row in in_cc.iterrows():
        subset, code = row["subset"], row["code"]
        print(f"\n[{subset}] fetching dump distribution...")
        cc_for_code = cc_raw[cc_raw["code"] == code]
        result = compute_yield_for_subset(subset, code, row["utf8_bytes"], cc_for_code)
        if result is None:
            skipped_rows.append({"subset": subset, "code": code, "reason": "fw2_api_unavailable"})
            print(f"    skipped - FineWeb-2 API unavailable for this config (404/500)")
            continue
        subset_rows.append(result)
        if result["yield_coverage_fraction"]:
            print(f"    matched {result['matched_dump_count']} dump(s), "
                  f"coverage={result['yield_coverage_fraction']:.2%}")
        else:
            print(f"    matched 0 bytes despite CC data being present for this language")

    # Write the skip log FIRST and separately - this is backend
    # bookkeeping, not part of the final yield table.
    if skipped_rows:
        Path(SKIPPED_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(skipped_rows).to_csv(SKIPPED_LOG_FILE, index=False)
        by_reason = pd.DataFrame(skipped_rows)["reason"].value_counts()
        print(f"\nWrote {len(skipped_rows)} skipped subset(s) to {SKIPPED_LOG_FILE}:")
        for reason, count in by_reason.items():
            print(f"    {reason}: {count}")

    if not subset_rows:
        print("No subsets resolved to a real yield - nothing to write to the main output.")
        return

    subset_df = pd.DataFrame(subset_rows)
    language_df = aggregate_to_language_level(subset_df)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    language_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {len(language_df)} language row(s) to {OUTPUT_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Tier C yield multiplier per language.")
    parser.add_argument("--test", type=int, metavar="N", default=None,
                         help="Test run: only process the first N FineWeb-2 subsets with CC data present.")
    args = parser.parse_args()

    main(test_n=args.test)
