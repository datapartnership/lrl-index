"""
tier_a_harvest_v2_all_languages.py

Crosswalk-scale wrapper around tier_a_harvest_v2.py, following the
same resumability/retry/caching patterns as
tier_a_harvest_all_languages.py (the Tier A v1 wrapper).

Location: lrl-index/harvest/scripts/tier_a_harvest_v2_all_languages.py
(sibling to tier_a_harvest_v2.py, which it imports from)

--------------------------------------------------------------------
SCOPE OF THIS WRAPPER
--------------------------------------------------------------------
For every HF query code in the crosswalk's hf_tag field (same source
as the v1 wrapper), searches Hugging Face for matching datasets, then
runs the full v2 per-dataset pipeline on each: card field extraction,
the total-size decision tree, the language breakdown layer (isolating
files and checking for a language-specific size when the dataset is
multilingual), and modality-specific structured counts (num_rows /
num_hours, file-metadata-only, capped per MAX_FILES_TO_PROBE_PER_MODALITY
for efficiency - see that constant's docstring in tier_a_harvest_v2.py).

Each output row is stamped with hf_language_tag, the literal code
queried - same convention as the v1 wrapper, for the same reason
(crosswalk to a canonical ISO code happens downstream, not here).

--------------------------------------------------------------------
WHAT'S DIFFERENT FROM THE V1 WRAPPER
--------------------------------------------------------------------
V1 harvested Tier A metadata directly from HF's search/tags API with
no per-dataset publication lookups or language-specific size
resolution - it was comprehensive across ALL datasets matching a
code, with lightweight per-dataset processing.

V2 does deep, expensive per-dataset processing (card fetch, README
fetch, Parquet footer reads, audio header probes) - each dataset
costs many more requests than a v1 row. At full crosswalk scale
(~8,000 codes), this is a MUCH heavier run than v1. Consider whether
you need every dataset per code, or whether SEARCH_LIMIT (imported
from tier_a_harvest_v2) should be capped for a full run - unlike v1,
where "no cap" was the recommended default, v2's per-dataset cost
makes a cap worth genuinely considering here, not just for testing.

--------------------------------------------------------------------
DEPENDENCIES / SETUP (same as tier_a_harvest_v2.py)
--------------------------------------------------------------------
pip install pyarrow requests mutagen huggingface_hub anthropic pypdf
export HF_TOKEN=hf_...   (strongly recommended given per-dataset request volume)
export ANTHROPIC_API_KEY=...   (only needed if the publication lookup is re-enabled - currently paused)

--------------------------------------------------------------------
NOT YET VALIDATED AT SCALE
--------------------------------------------------------------------
The underlying tier_a_harvest_v2.py functions are validated against
real datasets (see its module docstring / your own testing). This
wrapper's LOOPING, RESUMABILITY, and RETRY logic has NOT been run
against a real multi-dataset, multi-code crosswalk pass yet - start
with --test on a small number of languages before a full run.
"""
import json
import time
from pathlib import Path

import pandas as pd

from tier_a_harvest_v2 import (
    api, fetch_card_data, extract_card_fields, resolve_language_specific_size,
    get_modality_structured_counts, classify_linguality,
)

# ---- CONFIG ----
CROSSWALK_FILE = "../../crosswalk/data/processed/full_language_reference.csv"
OUTPUT_FILE = "../data/harvest_v2/tier_a_v2_by_language.csv"
SKIPPED_LOG = "../data/harvest_v2/tier_a_v2_skipped_languages.csv"

SEARCH_LIMIT = 30  # datasets fetched per query code - see module docstring on why v2 defaults to a cap, unlike v1

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
    """Same shape as the v1 wrapper: returns (results, search_failed).
    search_failed=True means never write a fake zero-result row - see
    the v1 wrapper's identical logic and reasoning for why this
    distinction matters (a rate-limited search must never be recorded
    as a confirmed-empty result)."""
    for attempt in range(MAX_RETRIES):
        try:
            return list(api.list_datasets(language=query_code, limit=limit)), False
        except Exception as e:
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"    [retry {attempt + 1}/{MAX_RETRIES}] search '{query_code}': {e} - waiting {wait}s")
            time.sleep(wait)
    print(f"    giving up on search '{query_code}' after {MAX_RETRIES} retries - NOT recording as zero-result")
    return [], True


def harvest_one_dataset(dataset_id, hf_language_tag):
    """
    Runs the full v2 per-dataset pipeline on one dataset for one query
    code. Returns a single row dict, or None if the dataset_info fetch
    itself fails after retries (skip, don't fabricate a row).
    """
    info = _with_retry(lambda: api.dataset_info(dataset_id, files_metadata=True),
                        f"dataset_info({dataset_id})")
    if info is None:
        return None

    tags = info.tags or []
    card_data = fetch_card_data(dataset_id)
    card_fields = extract_card_fields(dataset_id, tags, card_data)

    size_result = resolve_language_specific_size(card_fields, info.siblings, dataset_id, hf_language_tag)

    linguality = classify_linguality(card_fields["language_codes"])
    if linguality == "multilingual":
        from tier_a_harvest_v2 import isolate_language_files
        target_files = isolate_language_files(info.siblings, hf_language_tag)
    else:
        target_files = None  # unscoped - whole dataset

    modality_counts = get_modality_structured_counts(info.siblings, dataset_id, target_files=target_files)

    row = {
        "dataset_id": dataset_id,
        "hf_language_tag": hf_language_tag,
        "language_codes_on_card": card_fields["language_codes"],
        "linguality": linguality,
        "modalities": card_fields["modalities"],
        "tasks": card_fields["tasks"],
        "license": card_fields["license"],
        "arxiv_ids": card_fields["arxiv_ids"],
        "doi_ids": card_fields["doi_ids"],
        "total_size_bytes": size_result.get("total_size_bytes"),
        "total_num_rows": size_result.get("total_num_rows"),
        "total_size_method": size_result.get("total_size_method"),
        "reference_full_dataset_size_bytes": size_result.get("reference_full_dataset_size_bytes"),
        "reference_full_dataset_num_rows": size_result.get("reference_full_dataset_num_rows"),
        "reference_full_dataset_size_method": size_result.get("reference_full_dataset_size_method"),
        "readme_matched_lines": size_result.get("readme_matched_lines"),
        "isolated_file_count": size_result.get("isolated_file_count"),
        "needs_manual_review": size_result.get("needs_manual_review"),
        "flag_reason": size_result.get("flag_reason"),
        "language_scope": size_result.get("language_scope"),
    }

    for modality in ("text", "audio", "image"):
        m = modality_counts.get(modality, {})
        row[f"{modality}_num_rows" if modality != "audio" else "audio_num_hours"] = (
            m.get("num_rows") if modality != "audio" else m.get("num_hours")
        )
        row[f"{modality}_count_method"] = m.get("method")
        row[f"{modality}_file_types"] = m.get("file_types")

    return row


def run_all(max_languages=None, search_limit=SEARCH_LIMIT):
    """
    max_languages: if set, only the first N rows (languages) of the
    crosswalk are used - for a test run. Leave as None for a full run.
    """
    crosswalk = pd.read_csv(CROSSWALK_FILE)
    if max_languages is not None:
        crosswalk = crosswalk.head(max_languages)
        print(f"TEST RUN: limited to first {max_languages} language row(s)")

    output_path = Path(OUTPUT_FILE)
    already_done_keys = set()  # (dataset_id, hf_language_tag) pairs already in output
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
            skipped_rows.append({"iso_639_3": row.get("iso_639_3"), "language_name": row.get("language_name"),
                                  "reason": "no_match_in_crosswalk"})
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
            row = harvest_one_dataset(ds.id, hf_language_tag)
            if row is None:
                print(f"    skipping {ds.id} - dataset_info fetch failed after retries")
                continue

            pd.DataFrame([row]).to_csv(output_path, mode="a", header=write_header, index=False)
            write_header = False
            print(f"    wrote row for {ds.id} (total_size_method={row['total_size_method']}, "
                  f"needs_manual_review={row['needs_manual_review']}, flag_reason={row['flag_reason']})")

    if skipped_rows:
        pd.DataFrame(skipped_rows).drop_duplicates().to_csv(SKIPPED_LOG, index=False)
        print(f"\n{len(skipped_rows)} language(s) skipped (no hf_tag) - logged to {SKIPPED_LOG}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tier A v2 crosswalk-scale harvest.")
    parser.add_argument("--test", type=int, metavar="N", default=None,
                         help="Test run: only process the first N languages from the crosswalk.")
    parser.add_argument("--search-limit", type=int, default=SEARCH_LIMIT,
                         help=f"Datasets fetched per query code (default {SEARCH_LIMIT}).")
    args = parser.parse_args()

    run_all(max_languages=args.test, search_limit=args.search_limit)
