"""
tier_a_config_availability_experiment.py

EXPERIMENT, not a production harvest step - measures, across every
dataset found per crosswalk language code, whether the dataset
actually has a per-language config at all (either the bare code, or
the code appearing as a complete segment ANYWHERE in a config name,
delimited by _, -, ., /), versus datasets that only expose a single
flat config (commonly "default") with no per-language split.

This is the direct evidence for whether the Tier 2 Parquet-column
fallback is actually necessary, and how much coverage you'd be
missing without it - if a large share of tagged datasets DON'T have a
per-language config, that's the argument for Tier 2; if most do,
Tier 2 may be a smaller marginal gain than expected.

--------------------------------------------------------------------
MATCHING LOGIC - matches tier_a_harvest_v6.py exactly
--------------------------------------------------------------------
UPDATED from an earlier version of this script, which only checked
for the code as a PREFIX ("^code[_-./]"). That would have MISSED
real per-language configs where the code appears later in the string
- e.g. "eng-aai" (code is the second segment) or
"eng_Latn-zac_Latn" (code buried after multiple delimiters) - both
confirmed real patterns seen in this project's actual harvest runs.
This version splits each config name on every delimiter and checks
for an EXACT segment match anywhere in the result, matching
match_language_configs() in tier_a_harvest_v5.py/v6.py exactly, so
this experiment's percentage is consistent with what the real harvest
would actually find.

--------------------------------------------------------------------
WHAT THIS RECORDS, PER (dataset, query_code) PAIR
--------------------------------------------------------------------
dataset_id, hf_language_tag, total_config_count, configs_sample (first
15, for manual inspection), exact_match (bare code is a real config),
script_suffixed_matches (any segment-anywhere matches - name kept for
continuity with earlier output, now covers the broader match logic),
has_per_language_config (exact_match OR script_suffixed_matches - the
main yes/no this experiment is measuring), likely_single_default_config
(flag: total_config_count == 1 AND that one config is "default" or
matches the dataset's own name - a strong signal language lives in the
data, not the config list, i.e. a Tier 2 candidate).

--------------------------------------------------------------------
DEPENDENCIES / SETUP
--------------------------------------------------------------------
pip install datasets huggingface_hub pandas
export HF_TOKEN=hf_...  (recommended - higher rate limits)
"""
import re
import time
from pathlib import Path

import pandas as pd
from datasets import get_dataset_config_names
from huggingface_hub import HfApi

# ---- CONFIG ----
CROSSWALK_FILE = "../../crosswalk/data/processed/full_language_reference.csv"
OUTPUT_FILE = "../data/experiments/config_availability_by_language.csv"
SKIPPED_LOG_FILE = "../data/experiments/config_availability_skipped.csv"

SEARCH_LIMIT = 30
CONFIG_SAMPLE_SIZE = 15  # how many config names to keep for manual inspection when there's no match

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 5

api = HfApi(token=None)  # set token below from env if present


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


def _list_datasets_with_retry(query_code, limit):
    for attempt in range(MAX_RETRIES):
        try:
            return list(api.list_datasets(language=query_code, limit=limit)), False
        except Exception as e:
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"    [retry {attempt + 1}/{MAX_RETRIES}] search '{query_code}': {e} - waiting {wait}s")
            time.sleep(wait)
    print(f"    giving up on search '{query_code}' after {MAX_RETRIES} retries - NOT recording as zero-result")
    return [], True


DELIMITER_PATTERN = re.compile(r"[_\-./]")


def match_language_configs(hf_language_tag, all_configs):
    """Matches whenever the code appears as a COMPLETE SEGMENT
    anywhere in the config name (split on _, -, ., /) - not just as a
    prefix. Identical logic to tier_a_harvest_v5.py/v6.py, kept in
    sync deliberately so this experiment's numbers are trustworthy."""
    if not all_configs:
        return []
    matches = []
    for config in all_configs:
        segments = DELIMITER_PATTERN.split(config)
        if hf_language_tag in segments:
            matches.append(config)
    return matches


def check_dataset_config_availability(dataset_id, hf_language_tag, hf_token):
    """Returns a row dict, or None if the config list itself couldn't
    be fetched at all after retries (skip, don't fabricate a row)."""
    all_configs = _with_retry(
        lambda: get_dataset_config_names(dataset_id, token=hf_token),
        f"get_dataset_config_names({dataset_id})",
    )
    if all_configs is None:
        return None

    exact_match = hf_language_tag in all_configs
    segment_matches = match_language_configs(hf_language_tag, all_configs)
    has_match = exact_match or bool(segment_matches)

    likely_single_default = (
        len(all_configs) == 1
        and (all_configs[0] == "default" or all_configs[0].lower() == dataset_id.split("/")[-1].lower())
    )

    return {
        "dataset_id": dataset_id,
        "hf_language_tag": hf_language_tag,
        "total_config_count": len(all_configs),
        "configs_sample": ";".join(all_configs[:CONFIG_SAMPLE_SIZE]),
        "exact_match": exact_match,
        "script_suffixed_matches": ";".join(segment_matches) if segment_matches else None,
        "has_per_language_config": has_match,
        "likely_single_default_config": likely_single_default,
    }


def run_all(max_languages=None, search_limit=SEARCH_LIMIT):
    import os
    hf_token = os.environ.get("HF_TOKEN")
    global api
    api = HfApi(token=hf_token)

    crosswalk = pd.read_csv(CROSSWALK_FILE)
    if max_languages is not None:
        crosswalk = crosswalk.head(max_languages)
        print(f"TEST RUN: limited to first {max_languages} language row(s)")

    output_path = Path(OUTPUT_FILE)
    already_done_keys = set()
    if output_path.exists():
        prior = pd.read_csv(output_path)
        already_done_keys = set(zip(prior["dataset_id"], prior["hf_language_tag"]))
        print(f"Resuming: {len(already_done_keys)} (dataset, code) pair(s) already in {OUTPUT_FILE}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()

    skipped_rows = []
    query_codes_seen = set()
    for _, row in crosswalk.iterrows():
        if row.get("hf_match_status") == "no_match" or pd.isna(row.get("hf_tag")):
            continue
        for code in [c.strip() for c in str(row["hf_tag"]).split(";") if c.strip()]:
            query_codes_seen.add(code)

    for hf_language_tag in sorted(query_codes_seen):
        print(f"\n[{hf_language_tag}] searching...")
        search_results, search_failed = _list_datasets_with_retry(hf_language_tag, search_limit)
        if search_failed:
            continue

        print(f"  {len(search_results)} dataset(s) found")
        for ds in search_results:
            key = (ds.id, hf_language_tag)
            if key in already_done_keys:
                continue

            print(f"    checking {ds.id}...")
            row = check_dataset_config_availability(ds.id, hf_language_tag, hf_token)
            if row is None:
                print(f"    skipping {ds.id} - could not fetch config list")
                skipped_rows.append({"dataset_id": ds.id, "hf_language_tag": hf_language_tag,
                                      "reason": "config_list_fetch_failed"})
                continue

            pd.DataFrame([row]).to_csv(output_path, mode="a", header=write_header, index=False)
            write_header = False
            status = "MATCH" if row["has_per_language_config"] else (
                "single default config" if row["likely_single_default_config"] else "no match")
            print(f"    wrote row for {ds.id} ({status}, {row['total_config_count']} config(s) total)")

    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(SKIPPED_LOG_FILE, index=False)
        print(f"\n{len(skipped_rows)} dataset(s) skipped - logged to {SKIPPED_LOG_FILE}")

    print_summary(output_path)


def print_summary(output_path):
    if not Path(output_path).exists():
        return
    df = pd.read_csv(output_path)
    total = len(df)
    if total == 0:
        print("\nNo rows recorded yet.")
        return

    has_match = df["has_per_language_config"].sum()
    single_default = df["likely_single_default_config"].sum()
    other_no_match = total - has_match - single_default

    print(f"\n=== SUMMARY ===")
    print(f"Total (dataset, language-code) pairs checked: {total}")
    print(f"Has per-language config (exact or segment-anywhere match): {has_match} ({has_match / total:.1%})")
    print(f"Likely single 'default' config (language probably a data column - Tier 2 candidate): {single_default} ({single_default / total:.1%})")
    print(f"No match, multiple configs but unrecognized naming: {other_no_match} ({other_no_match / total:.1%})")
    print(f"Unique datasets involved: {df['dataset_id'].nunique()}")
    print(f"\nThis is the direct evidence for whether Tier 2 is necessary: "
          f"{100 - has_match / total * 100:.1f}% of (dataset, language) pairs have NO usable per-language "
          f"config and would be entirely missed without the Tier 2 Parquet-column fallback.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment: how many HF datasets have a genuine per-language config?")
    parser.add_argument("--test", type=int, metavar="N", default=None,
                         help="Test run: only process the first N languages from the crosswalk.")
    parser.add_argument("--search-limit", type=int, default=SEARCH_LIMIT)
    args = parser.parse_args()

    run_all(max_languages=args.test, search_limit=args.search_limit)
