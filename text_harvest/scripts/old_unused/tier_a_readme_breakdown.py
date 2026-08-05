"""
tier_a_readme_breakdown_experiment.py

EXPERIMENT, not a production harvest step - per your supervisor
conversation: before investing further in the README-heuristic path
(search_readme_for_language_size in tier_a_harvest_v2.py), measure
how often multilingual datasets actually HAVE a per-language size
breakdown in their README at all.

Location: lrl-index/text_harvest/scripts/tier_a_readme_breakdown_experiment.py
(sibling to tier_a_harvest_v2.py, which it imports from - reuses card
extraction, linguality classification, and the README search heuristic
rather than reimplementing them)

--------------------------------------------------------------------
SCOPE
--------------------------------------------------------------------
For every HF query code in the crosswalk's hf_tag field (same source
as the existing Tier A wrapper), searches Hugging Face for matching
datasets. For each dataset:
  1. Extracts card fields (language_codes, etc.) - same as production.
  2. Classifies linguality - same as production.
  3. ONLY MULTILINGUAL datasets are relevant to this experiment - a
     monolingual dataset has no "breakdown" to look for by definition,
     so those are recorded but not further processed (see
     RELEVANCE_LOG - kept distinct from "processed, no breakdown
     found" so you can see the denominator this experiment is
     actually measuring against).
  4. For multilingual datasets: fetches the README body and runs the
     EXACT SAME search_readme_for_language_size heuristic used in
     production, recording whether a match was found and what the
     matched line(s) say if so.

--------------------------------------------------------------------
OUTPUT: ONE ROW PER (dataset, query_code) PAIR
--------------------------------------------------------------------
dataset_id, hf_language_tag (the query code this dataset was found
under), other_languages_in_dataset (the dataset's full language tag
list, minus the query code), is_multilingual, breakdown_found_in_readme
(bool), matched_readme_lines (the raw matched line(s) if found, else
None).

Note the SAME dataset can appear multiple times if it matches more
than one query code (e.g. a dataset tagged for both "fra" and "por"
shows up once under each) - this is intentional, matching the
existing Tier A wrapper's convention, since "does this dataset have a
breakdown for THIS language" is meaningfully a per-code question even
when the dataset itself is the same.

--------------------------------------------------------------------
WHAT THIS DOES NOT DO
--------------------------------------------------------------------
No size resolution, no total-size decision tree, no modality counts -
this experiment ONLY answers "is there a README breakdown, yes/no,
and what does it say" for multilingual datasets. Not wired into the
production harvest pipeline.

--------------------------------------------------------------------
DEPENDENCIES / SETUP
--------------------------------------------------------------------
Same as tier_a_harvest_v2.py:
pip install pyarrow requests mutagen huggingface_hub anthropic pypdf
export HF_TOKEN=hf_...   (recommended given per-dataset request volume)
"""
import time
from pathlib import Path

import pandas as pd

from tier_a_harvest_v2 import (
    api, fetch_card_data, extract_card_fields, classify_linguality,
    fetch_readme_body_text, search_readme_for_language_size,
)

# ---- CONFIG ----
CROSSWALK_FILE = "../../crosswalk/data/processed/full_language_reference.csv"
OUTPUT_FILE = "../data/experiments/readme_breakdown_by_language.csv"
SKIPPED_LOG_FILE = "../data/experiments/readme_breakdown_skipped.csv"

SEARCH_LIMIT = 30  # datasets fetched per query code - same default as the production wrapper

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 5


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
    """Same shape as the production wrapper: returns (results, search_failed) -
    a rate-limited search must never be recorded as a confirmed-empty result."""
    for attempt in range(MAX_RETRIES):
        try:
            return list(api.list_datasets(language=query_code, limit=limit)), False
        except Exception as e:
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"    [retry {attempt + 1}/{MAX_RETRIES}] search '{query_code}': {e} - waiting {wait}s")
            time.sleep(wait)
    print(f"    giving up on search '{query_code}' after {MAX_RETRIES} retries - NOT recording as zero-result")
    return [], True


def process_one_dataset(dataset_id, hf_language_tag):
    """
    Returns a row dict, or None if the dataset_info fetch itself fails
    after retries (skip, don't fabricate a row - same principle as
    the production script).
    """
    info = _with_retry(lambda: api.dataset_info(dataset_id, files_metadata=False),
                        f"dataset_info({dataset_id})")
    if info is None:
        return None

    tags = info.tags or []
    card_data = fetch_card_data(dataset_id)
    card_fields = extract_card_fields(dataset_id, tags, card_data)
    linguality = classify_linguality(card_fields["language_codes"])

    other_languages = None
    if card_fields["language_codes"]:
        codes = [c for c in card_fields["language_codes"].split(";") if c != hf_language_tag]
        other_languages = ";".join(codes) if codes else None

    row = {
        "dataset_id": dataset_id,
        "hf_language_tag": hf_language_tag,
        "other_languages_in_dataset": other_languages,
        "linguality": linguality,
        "is_multilingual": linguality == "multilingual",
        "breakdown_found_in_readme": None,
        "matched_readme_lines": None,
    }

    if linguality != "multilingual":
        return row  # recorded, but not relevant to the experiment's core question - see docstring

    readme_text = fetch_readme_body_text(dataset_id)
    matches = search_readme_for_language_size(readme_text, hf_language_tag)
    row["breakdown_found_in_readme"] = matches is not None
    row["matched_readme_lines"] = ";".join(matches) if matches else None
    return row


def run_all(max_languages=None, search_limit=SEARCH_LIMIT):
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
            print(f"    search failed after retries - skipping this code for this run, will retry next run")
            continue

        print(f"  {len(search_results)} dataset(s) found")
        for ds in search_results:
            key = (ds.id, hf_language_tag)
            if key in already_done_keys:
                continue

            print(f"    processing {ds.id}...")
            row = process_one_dataset(ds.id, hf_language_tag)
            if row is None:
                print(f"    skipping {ds.id} - dataset_info fetch failed after retries")
                skipped_rows.append({"dataset_id": ds.id, "hf_language_tag": hf_language_tag,
                                      "reason": "dataset_info_fetch_failed"})
                continue

            pd.DataFrame([row]).to_csv(output_path, mode="a", header=write_header, index=False)
            write_header = False
            status = ("BREAKDOWN FOUND" if row["breakdown_found_in_readme"]
                      else "no breakdown" if row["is_multilingual"] else "monolingual - n/a")
            print(f"    wrote row for {ds.id} ({status})")

    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(SKIPPED_LOG_FILE, index=False)
        print(f"\n{len(skipped_rows)} dataset(s) skipped - logged to {SKIPPED_LOG_FILE}")

    print_summary(output_path)


def print_summary(output_path):
    """Quick console summary - the actual denominator you and your
    supervisor care about: of MULTILINGUAL datasets specifically, what
    fraction have a README breakdown."""
    if not Path(output_path).exists():
        return
    df = pd.read_csv(output_path)
    multi = df[df["is_multilingual"] == True]
    if multi.empty:
        print("\nNo multilingual dataset rows recorded yet.")
        return
    found = multi["breakdown_found_in_readme"].sum()
    total = len(multi)
    print(f"\n=== SUMMARY ===")
    print(f"Multilingual dataset x language-code rows: {total}")
    print(f"README breakdown found: {found} ({found / total:.1%})")
    print(f"Unique datasets involved: {multi['dataset_id'].nunique()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment: how often do multilingual datasets have a README language breakdown?")
    parser.add_argument("--test", type=int, metavar="N", default=None,
                         help="Test run: only process the first N languages from the crosswalk.")
    parser.add_argument("--search-limit", type=int, default=SEARCH_LIMIT,
                         help=f"Datasets fetched per query code (default {SEARCH_LIMIT}).")
    args = parser.parse_args()

    run_all(max_languages=args.test, search_limit=args.search_limit)
