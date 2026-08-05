"""
tier_a_harvest_v5.py

Clean rewrite of the simplified Tier A harvest, single self-contained
file. Per explicit spec (see conversation).

--------------------------------------------------------------------
FIELDS RECORDED
--------------------------------------------------------------------
dataset_id, language_code, languages_in_dataset, linguality,
license, modality, tasks, num_rows, provenance, synthetic_flag,
arxiv_ids, doi_ids, manual_review.

--------------------------------------------------------------------
LINGUALITY DETERMINATION
--------------------------------------------------------------------
1. Check the dataset's own "multilinguality" tag first (HF's own
   explicit tag category, e.g. multilinguality:monolingual /
   multilinguality:multilingual) - if present, trust it directly.
2. Fall back to counting language tags (1 = monolingual, >1 =
   multilingual, 0 = no_language_tag) if no multilinguality tag exists.

--------------------------------------------------------------------
Num_rows RESOLUTION
--------------------------------------------------------------------
Monolingual: (a) check the dataset's OVERALL metadata (no config
specified) for num_rows first; (b) if not there, fall back to
checking the 'default' configuration specifically.
Multilingual: check for a per-language configuration (bare code, or
{code}{delimiter}{script} - delimiters _, -, ., /) and read num_rows
from THAT configuration. No per-language config = out of scope for
sizing (goes to the catch-all output, see below).

--------------------------------------------------------------------
PROVENANCE (3 categories only)
--------------------------------------------------------------------
human_generated / machine_generated / unclear - README keyword
heuristic, consolidated to three outcomes (both "no signal found" and
"conflicting signals found" collapse into "unclear").
synthetic_flag: True if provenance == machine_generated, else False.

--------------------------------------------------------------------
MANUAL_REVIEW
--------------------------------------------------------------------
"404 - need to grant access to dataset": access issue - checked
  FIRST, before anything else.
"check unclear provenance": provenance heuristic couldn't determine
  human vs. machine - checked SECOND, only if access wasn't the issue.
Blank/None: neither applies.

--------------------------------------------------------------------
FOUR OUTPUT FILES, MUTUALLY EXCLUSIVE (priority order below)
--------------------------------------------------------------------
1. FULL_CLEAN_FILE - num_rows successfully resolved AND provenance is
   NOT unclear.
2. MANUAL_REVIEW_FILE - (a) access issues (404-style), checked FIRST,
   OR (b) num_rows WAS successfully resolved but provenance is unclear.
3. NO_PRECOMPUTED_METADATA_FILE - a per-language config was found (or
   the dataset is monolingual/'default'), but the actual load_dataset()
   call failed for a non-access reason (e.g. legacy loading script
   requiring trust_remote_code=True, deliberately NOT enabled here).
4. EVERYTHING_ELSE_FILE - catch-all: NO per-language config existed at
   all for a multilingual dataset - regardless of provenance.

--------------------------------------------------------------------
CONFIGURATION NAMES RECORDED
--------------------------------------------------------------------
matched_configurations: the specific config string(s) actually used -
  "overall", "default", the real per-language config name(s), or None.
all_configurations: full list of every config the dataset has
  (multilingual case only) - capped at 20 shown inline, true count in
  all_configurations_count.

--------------------------------------------------------------------
DEPENDENCIES / SETUP
--------------------------------------------------------------------
pip install pandas huggingface_hub datasets
export HF_TOKEN=hf_...
"""
import os
import re
import time
from pathlib import Path

import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import DatasetCard, HfApi

# ---- CONFIG ----
CROSSWALK_FILE = "../../crosswalk/data/processed/full_language_reference.csv"
FULL_CLEAN_FILE = "../data/tier_a_v5/tier_a_v5_full_clean.csv"
MANUAL_REVIEW_FILE = "../data/tier_a_v5/tier_a_v5_manual_review.csv"
NO_PRECOMPUTED_METADATA_FILE = "../data/tier_a_v5/tier_a_v5_no_precomputed_metadata.csv"
EVERYTHING_ELSE_FILE = "../data/tier_a_v5/tier_a_v5_everything_else.csv"

SEARCH_LIMIT = None
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 5
MAX_CONFIGS_TO_DISPLAY = 20  # some datasets have hundreds+ of configs - cap what's shown, always record the true count

api = HfApi(token=os.environ.get("HF_TOKEN"))

ACCESS_ISSUE_KEYWORDS = {
    "need to manually accept dataset access": ["gated", "access request", "must agree", "accept the"],
    "private dataset": ["private", "does not exist", "repository not found", "404"],
}


def classify_access_issue(response_text):
    """Best-effort classification of an access-related error. Not a
    guaranteed-accurate diagnosis - a convenience label for triage."""
    text_lower = (response_text or "").lower()
    for reason, keywords in ACCESS_ISSUE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return reason
    return "other"


# ---- Provenance heuristic (3 categories) ----
# NOTE on false-positive risk: "written by"/"authored by" are the two
# riskiest entries here - a README saying "this dataset card was
# written by the HF team" describes documentation authorship, not
# data provenance, and would still trigger a human_generated hit. Kept
# as specified; worth spot-checking human_generated results that hit
# ONLY on these two phrases and nothing else in the list.
PROVENANCE_KEYWORDS = {
    "human_generated": [
        "human-generat", "human annotat", "crowdsourc", "crowd-sourc",
        "human-translat", "human translat",
        "native speaker", "manually annotat", "manually transcrib",
        "manually translat", "hand-annotat", "hand annotat",
        "volunteer", "expert annotat", "professionally translat",
        "collected from speakers", "collected by speakers",
        "transcribed by", "annotated by humans", "human-curat",
        "human curat", "written by", "authored by",
    ],
    "machine_generated": [
        "machine-generat", "machine generat", "synthetic",
        "llm-generat", "llm generat", "machine-translat", "machine translat",
        "auto-generat", "auto generat", "automatically generat",
        "automatically translat", "ai-generat", "ai generat",
        "model-generat", "model generat", "gpt-generat",
        "back-translat", "back translat", "machine translation system",
        "neural machine translation", "nmt-generat",
    ],
}


def classify_provenance(readme_text):
    """
    Returns 'human_generated', 'machine_generated', or 'unclear'.
    Plain substring matching. Both 'no signal at all' and 'conflicting
    signals' collapse into 'unclear'.
    """
    if not readme_text:
        return "unclear"
    text_lower = readme_text.lower()

    human_hit = any(kw in text_lower for kw in PROVENANCE_KEYWORDS["human_generated"])
    machine_hit = any(kw in text_lower for kw in PROVENANCE_KEYWORDS["machine_generated"])

    if machine_hit and not human_hit:
        return "machine_generated"
    if human_hit and not machine_hit:
        return "human_generated"
    return "unclear"


RATE_LIMIT_WAIT_SECONDS = 300  # 5 minutes - HF's rate-limit window; short exponential backoff never gets close to this


def is_rate_limit_error(e):
    text = str(e).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def call_with_rate_limit_handling(fn, description):
    """
    Retries INDEFINITELY, but ONLY for rate-limit errors specifically -
    waits the full 5-minute window rather than a short exponential
    backoff, since retrying sooner just re-hits the same quota and
    wastes the retry budget. Any OTHER exception is re-raised
    immediately, unchanged - this only adds rate-limit handling on top
    of whatever error handling already exists at the call site.
    """
    while True:
        try:
            return fn()
        except Exception as e:
            if is_rate_limit_error(e):
                print(f"    [rate limited] {description}: waiting {RATE_LIMIT_WAIT_SECONDS}s (5 min) before retrying...")
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
                continue
            raise


def _with_retry(fn, description, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            return call_with_rate_limit_handling(fn, description)
        except Exception as e:
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"    [retry {attempt + 1}/{max_retries}] {description}: {e} - waiting {wait}s")
            time.sleep(wait)
    print(f"    giving up on {description} after {max_retries} retries")
    return None


def _list_datasets_with_retry(query_code, limit):
    for attempt in range(MAX_RETRIES):
        try:
            results = call_with_rate_limit_handling(
                lambda: list(api.list_datasets(language=query_code, limit=limit)),
                f"search '{query_code}'",
            )
            return results, False
        except Exception as e:
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"    [retry {attempt + 1}/{MAX_RETRIES}] search '{query_code}': {e} - waiting {wait}s")
            time.sleep(wait)
    print(f"    giving up on search '{query_code}' after {MAX_RETRIES} retries - NOT recording as zero-result")
    return [], True


# ---- Card field extraction ----
def fetch_card_data(dataset_id):
    def _load():
        card = DatasetCard.load(dataset_id)
        return card.data.to_dict()
    return _with_retry(_load, f"load card for {dataset_id}", max_retries=2)


def fetch_readme_body_text(dataset_id):
    def _load():
        return DatasetCard.load(dataset_id).text
    return _with_retry(_load, f"load README body for {dataset_id}", max_retries=2)


def extract_card_fields(tags):
    language_tags = [t.replace("language:", "") for t in tags if t.startswith("language:")]
    task_tags = [t.replace("task_categories:", "") for t in tags if t.startswith("task_categories:")]
    license_tags = [t.replace("license:", "") for t in tags if t.startswith("license:")]
    # Checks BOTH "modality:" and "modalities:" prefixes - unconfirmed which one HF
    # actually uses for a given dataset (the site UI section header "Modalities"
    # doesn't necessarily match the literal tag string). Verify directly against a
    # real dataset's tags if you want certainty - see conversation.
    modality_tags = (
        [t.replace("modality:", "") for t in tags if t.startswith("modality:")]
        + [t.replace("modalities:", "") for t in tags if t.startswith("modalities:")]
    )
    multilinguality_tags = [t.replace("multilinguality:", "") for t in tags if t.startswith("multilinguality:")]
    arxiv_ids = [t.replace("arxiv:", "") for t in tags if t.startswith("arxiv:")]
    doi_ids = [t.replace("doi:", "") for t in tags if t.startswith("doi:")]

    return {
        "language_tags": language_tags,
        "tasks": ";".join(task_tags) if task_tags else None,
        "license": ";".join(license_tags) if license_tags else "license_unstated",
        "modalities": ";".join(modality_tags) if modality_tags else "modality_unstated",
        "multilinguality_tags": multilinguality_tags,
        "arxiv_ids": ";".join(arxiv_ids) if arxiv_ids else None,
        "doi_ids": ";".join(doi_ids) if doi_ids else None,
    }


def determine_linguality(card_fields):
    """1. Trust an explicit multilinguality tag if present.
    2. Fall back to counting language tags."""
    mtags = card_fields["multilinguality_tags"]
    if "monolingual" in mtags:
        return "monolingual"
    if "multilingual" in mtags:
        return "multilingual"

    num_languages = len(card_fields["language_tags"])
    if num_languages == 0:
        return "no_language_tag"
    return "monolingual" if num_languages == 1 else "multilingual"


# ---- Config resolution ----
def try_list_all_configs(dataset_id, hf_token):
    """Returns (configs_or_None, error_str_or_None). Rate-limit errors
    get an indefinite 5-minute-wait retry; other errors fail immediately."""
    try:
        configs = call_with_rate_limit_handling(
            lambda: get_dataset_config_names(dataset_id, token=hf_token),
            f"get_dataset_config_names({dataset_id})",
        )
        return configs, None
    except Exception as e:
        return None, str(e)


DELIMITER_PATTERN = re.compile(r"[_\-./]")


def match_language_configs(hf_language_tag, all_configs):
    """
    Matches whenever hf_language_tag appears as a COMPLETE SEGMENT
    anywhere in the config name, when split on common delimiters
    (_, -, ., /) - not just as a prefix. Handles:
      - simple {code}{delimiter}{script}: "aai_Latn" for code "aai"
      - pair-direction-style configs: "eng-aai" for code "aai" (code
        is the SECOND segment, not the first)
      - multi-segment combinations: "eng_Latn-zac_Latn" for code "zac"
        (code is buried in the middle after multiple delimiters)
      - a bare exact match: "aai" for code "aai" (trivial single-
        segment case, subsumed by the same logic)
    Same principle as isolate_language_files's path-segment matching
    elsewhere in this project - split fully, then check for an EXACT
    segment match, never a substring match (avoids the "en" matching
    inside "sentence"-style false positives from earlier in this
    project).
    """
    if not all_configs:
        return []
    matches = []
    for config in all_configs:
        segments = DELIMITER_PATTERN.split(config)
        if hf_language_tag in segments:
            matches.append(config)
    return matches


def format_config_list_for_display(configs):
    """Capped inline display - true count always recorded separately."""
    if not configs:
        return None
    shown = configs[:MAX_CONFIGS_TO_DISPLAY]
    suffix = f" ...({len(configs)} total)" if len(configs) > MAX_CONFIGS_TO_DISPLAY else ""
    return ";".join(shown) + suffix


def get_num_rows(dataset_id, config_name, hf_token):
    """
    Returns (num_rows_or_None, error_str_or_None). Uses an ACTUAL
    load_dataset() call - this DOWNLOADS the data, per explicit choice
    to drop the free precomputed-metadata check entirely. Rate-limit
    errors specifically get an indefinite 5-minute-wait retry (see
    call_with_rate_limit_handling) - other errors fail immediately,
    same as before.

    load_dataset() without split= returns a DatasetDict covering every
    split - summed here.
    """
    try:
        if config_name is not None:
            ds = call_with_rate_limit_handling(
                lambda: load_dataset(dataset_id, name=config_name, token=hf_token),
                f"load_dataset({dataset_id}, {config_name})",
            )
        else:
            ds = call_with_rate_limit_handling(
                lambda: load_dataset(dataset_id, token=hf_token),
                f"load_dataset({dataset_id})",
            )
    except Exception as e:
        return None, str(e)

    total = sum(split.num_rows for split in ds.values())
    return (total if total else None), None


# ---- Main per-dataset processing ----
def process_one_dataset(dataset_id, hf_language_tag, hf_token):
    """Returns (row_dict, bucket) where bucket is one of:
    'full_clean', 'manual_review', 'no_precomputed_metadata', 'everything_else'."""

    info = _with_retry(lambda: api.dataset_info(dataset_id, files_metadata=False),
                        f"dataset_info({dataset_id})")
    if info is None:
        return {
            "dataset_id": dataset_id, "language_code": hf_language_tag,
            "languages_in_dataset": None, "linguality": None, "license": None, "modalities": None, "tasks": None,
            "num_rows": None, "provenance": None, "synthetic_flag": None,
            "arxiv_ids": None, "doi_ids": None,
            "matched_configurations": None, "all_configurations": None, "all_configurations_count": None,
            "manual_review": "404 - need to grant access to dataset",
        }, "manual_review"

    tags = info.tags or []
    card_fields = extract_card_fields(tags)
    linguality = determine_linguality(card_fields)

    readme_text = fetch_readme_body_text(dataset_id)
    provenance = classify_provenance(readme_text)
    synthetic_flag = (provenance == "machine_generated")

    base_row = {
        "dataset_id": dataset_id,
        "language_code": hf_language_tag,
        "languages_in_dataset": ";".join(card_fields["language_tags"]) if card_fields["language_tags"] else None,
        "linguality": linguality,
        "license": card_fields["license"],
        "modalities": card_fields["modalities"],
        "tasks": card_fields["tasks"],
        "provenance": provenance,
        "synthetic_flag": synthetic_flag,
        "arxiv_ids": card_fields["arxiv_ids"],
        "doi_ids": card_fields["doi_ids"],
    }

    # ---- Monolingual (or no_language_tag) ----
    if linguality != "multilingual":
        num_rows, err = get_num_rows(dataset_id, None, hf_token)
        matched_config = "overall"
        if num_rows is None and not (err and classify_access_issue(err) != "other"):
            num_rows, err = get_num_rows(dataset_id, "default", hf_token)
            matched_config = "default"
        config_fields = {"matched_configurations": matched_config if num_rows is not None else None,
                          "all_configurations": None, "all_configurations_count": None}

        if err and classify_access_issue(err) != "other":
            return {**base_row, "num_rows": None, **config_fields,
                    "manual_review": "404 - need to grant access to dataset"}, "manual_review"

        if num_rows is not None:
            if provenance == "unclear":
                return {**base_row, "num_rows": num_rows, **config_fields,
                        "manual_review": "check unclear provenance"}, "manual_review"
            return {**base_row, "num_rows": num_rows, **config_fields, "manual_review": None}, "full_clean"

        return {**base_row, "num_rows": None, "matched_configurations": None,
                "all_configurations": None, "all_configurations_count": None,
                "manual_review": None}, "no_precomputed_metadata"

    # ---- Multilingual ----
    all_configs, config_err = try_list_all_configs(dataset_id, hf_token)
    if config_err and classify_access_issue(config_err) != "other":
        return {**base_row, "num_rows": None,
                "matched_configurations": None, "all_configurations": None, "all_configurations_count": None,
                "manual_review": "404 - need to grant access to dataset"}, "manual_review"

    config_fields = {
        "all_configurations": format_config_list_for_display(all_configs),
        "all_configurations_count": len(all_configs) if all_configs is not None else None,
    }
    resolved_configs = match_language_configs(hf_language_tag, all_configs)

    if not resolved_configs:
        return {**base_row, "num_rows": None, "matched_configurations": None, **config_fields,
                "manual_review": None}, "everything_else"

    total_rows = 0
    any_resolved = False
    for config_name in resolved_configs:
        num_rows, _ = get_num_rows(dataset_id, config_name, hf_token)
        if num_rows is not None:
            total_rows += num_rows
            any_resolved = True

    matched_fields = {"matched_configurations": ";".join(resolved_configs), **config_fields}

    if any_resolved:
        if provenance == "unclear":
            return {**base_row, "num_rows": total_rows, **matched_fields,
                    "manual_review": "check unclear provenance"}, "manual_review"
        return {**base_row, "num_rows": total_rows, **matched_fields, "manual_review": None}, "full_clean"

    return {**base_row, "num_rows": None, **matched_fields, "manual_review": None}, "no_precomputed_metadata"


def run_all(max_languages=None, search_limit=SEARCH_LIMIT):
    hf_token = os.environ.get("HF_TOKEN")

    crosswalk = pd.read_csv(CROSSWALK_FILE)
    if max_languages is not None:
        crosswalk = crosswalk.head(max_languages)
        print(f"TEST RUN: limited to first {max_languages} language row(s)")

    output_paths = {
        "full_clean": Path(FULL_CLEAN_FILE),
        "manual_review": Path(MANUAL_REVIEW_FILE),
        "no_precomputed_metadata": Path(NO_PRECOMPUTED_METADATA_FILE),
        "everything_else": Path(EVERYTHING_ELSE_FILE),
    }
    for p in output_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    already_done_keys = set()
    write_headers = {}
    for bucket, path in output_paths.items():
        write_headers[bucket] = not path.exists()
        if path.exists():
            prior = pd.read_csv(path)
            already_done_keys |= set(zip(prior["dataset_id"], prior["language_code"]))
    if already_done_keys:
        print(f"Resuming: {len(already_done_keys)} (dataset, code) pair(s) already processed")

    query_codes_seen = set()
    for _, row in crosswalk.iterrows():
        if row.get("hf_match_status") == "no_match" or pd.isna(row.get("hf_tag")):
            continue
        for code in [c.strip() for c in str(row["hf_tag"]).split(";") if c.strip()]:
            query_codes_seen.add(code)

    bucket_counts = {b: 0 for b in output_paths}

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

            print(f"    processing {ds.id}...")
            row, bucket = process_one_dataset(ds.id, hf_language_tag, hf_token)

            path = output_paths[bucket]
            pd.DataFrame([row]).to_csv(path, mode="a", header=write_headers[bucket], index=False)
            write_headers[bucket] = False
            bucket_counts[bucket] += 1
            print(f"    -> {bucket}")

    print("\n=== SUMMARY ===")
    for bucket, count in bucket_counts.items():
        print(f"{bucket}: {count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tier A harvest v5 - clean spec, 4 output buckets.")
    parser.add_argument("--test", type=int, metavar="N", default=None)
    parser.add_argument("--search-limit", type=int, default=SEARCH_LIMIT)
    args = parser.parse_args()

    run_all(max_languages=args.test, search_limit=args.search_limit)
