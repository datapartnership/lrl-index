"""
tier_a_harvest_v4.py

Simplified Tier A harvest - fully self-contained, single file. The
scoped alternative proposed to your supervisor: only datasets where
num_rows can be obtained EXACTLY and CHEAPLY get a real size figure.
No calibration, no estimation, no file-size summing.

--------------------------------------------------------------------
SCOPE
--------------------------------------------------------------------
- Monolingual datasets: dataset-level IS language-level. num_rows for
  the whole dataset is used directly.
- Multilingual datasets WITH a per-language config (bare code, or
  {code}{delimiter}{script} - delimiters: _, -, ., /):
  num_rows for that SPECIFIC config is used.
- Multilingual datasets WITHOUT a per-language config: recorded with
  every field we CAN collect (linguality, modality, license,
  provenance, publication_link, synthetic_flag, num_languages_in_dataset)
  but num_rows/retrieval_method left BLANK - written to
  NO_CONFIG_LOG_FILE, not silently dropped. This is a known,
  documented scope limitation (see supervisor email), not a bug -
  there is more training data on Hugging Face than this pass captures
  a size figure for, but we still want a record that the dataset
  exists and matched this language.
- Genuinely unresolvable datasets (access issues, dataset_info fetch
  failure - i.e. we couldn't even get card metadata) go to a separate,
  minimal SKIPPED_LOG_FILE, since there's nothing to record for those
  beyond the ID and the reason.

--------------------------------------------------------------------
FIELDS RECORDED (both OUTPUT_FILE and NO_CONFIG_LOG_FILE)
--------------------------------------------------------------------
dataset_id, hf_language_tag, num_languages_in_dataset, linguality,
modality, license, provenance, publication_link, synthetic_flag,
retrieval_method, num_rows.

--------------------------------------------------------------------
provenance / synthetic_flag
--------------------------------------------------------------------
Lightweight README keyword heuristic - same "prose-parsing, not a
determination" caveat as every other README-based heuristic in this
project. Defaults to "unknown" rather than assuming non-synthetic when
no signal is found.

--------------------------------------------------------------------
retrieval_method
--------------------------------------------------------------------
"precomputed_metadata_monolingual" - whole-dataset builder info
"precomputed_metadata_per_language_config" - per-config builder info
"no_precomputed_metadata_available" - config resolved, but builder
  info didn't have num_examples for it
(blank) - no per-language config existed at all - see NO_CONFIG_LOG_FILE

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
from datasets import get_dataset_config_names, load_dataset_builder
from huggingface_hub import DatasetCard, HfApi

# ---- CONFIG ----
CROSSWALK_FILE = "../../crosswalk/data/processed/full_language_reference.csv"
OUTPUT_FILE = "../data/tier_a_v4/tier_a_v4_by_language.csv"
NO_CONFIG_LOG_FILE = "../data/tier_a_v4/tier_a_v4_no_per_language_config.csv"
SKIPPED_LOG_FILE = "../data/tier_a_v4/tier_a_v4_skipped_unresolvable.csv"

SEARCH_LIMIT = 30
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 5

api = HfApi(token=os.environ.get("HF_TOKEN"))

# ---- Access-issue classification (from the original Tier A pipeline) ----
ACCESS_ISSUE_KEYWORDS = {
    "need to manually accept dataset access": ["gated", "access request", "must agree", "accept the"],
    "private dataset": ["private", "does not exist", "repository not found"],
}
ACCESS_ISSUE_REASONS = set(ACCESS_ISSUE_KEYWORDS.keys()) | {"other"}


def classify_access_issue(response_text):
    """Best-effort classification of WHY a dataset couldn't be
    accessed, based on keyword matching against the error text. Not a
    guaranteed-accurate diagnosis - a convenience label for triage."""
    text_lower = (response_text or "").lower()
    for reason, keywords in ACCESS_ISSUE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return reason
    return "other"


# ---- Provenance / synthetic-flag keyword heuristic ----
SYNTHETIC_KEYWORDS = [
    r"\bmachine.translat", r"\bmachine.generat", r"\bmt.generated", r"\bllm.generated",
    r"\bgenerated (?:by|using|with) (?:gpt|claude|llm)",
    r"\bsynthetic(?:ally)? generated", r"\bai.generated text",
]
HUMAN_KEYWORDS = [
    r"\bhuman.written", r"\bhuman.translat", r"\bhuman.authored", r"\bnative speaker",
    r"\bcrowdsourc", r"\bvolunteer",
]


def classify_provenance(readme_text):
    """Returns (provenance, synthetic_flag). provenance is 'human',
    'synthetic', 'mixed_signals_needs_review', or 'unknown' - never
    guessed when there's no keyword signal. synthetic_flag is
    True/False/None (None = unknown, propagated not defaulted to False)."""
    if not readme_text:
        return "unknown", None
    text_lower = readme_text.lower()

    synthetic_hit = any(re.search(p, text_lower) for p in SYNTHETIC_KEYWORDS)
    human_hit = any(re.search(p, text_lower) for p in HUMAN_KEYWORDS)

    if synthetic_hit and not human_hit:
        return "synthetic", True
    if human_hit and not synthetic_hit:
        return "human", False
    if synthetic_hit and human_hit:
        return "mixed_signals_needs_review", None
    return "unknown", None


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


# ---- Card field extraction ----
def fetch_card_data(dataset_id):
    """Loads a dataset's README card and returns its parsed YAML front
    matter as a dict, or None on failure - never raises."""
    def _load():
        card = DatasetCard.load(dataset_id)
        return card.data.to_dict()

    return _with_retry(_load, f"load card for {dataset_id}", max_retries=2)


def fetch_readme_body_text(dataset_id):
    """Loads a dataset's README BODY text (prose, not YAML). Returns
    None on failure rather than raising."""
    def _load():
        return DatasetCard.load(dataset_id).text

    return _with_retry(_load, f"load README body for {dataset_id}", max_retries=2)


def extract_card_fields(dataset_id, tags, card_data):
    """Pulls language_code(s), modalities, tasks, license, arxiv_id(s),
    doi(s) from tags/card YAML - never from summing files."""
    language_codes = [t.replace("language:", "") for t in tags if t.startswith("language:")]
    modality_tags = [t.replace("modality:", "") for t in tags if t.startswith("modality:")]
    task_tags = [t.replace("task_categories:", "") for t in tags if t.startswith("task_categories:")]
    license_tags = [t.replace("license:", "") for t in tags if t.startswith("license:")]
    arxiv_ids = [t.replace("arxiv:", "") for t in tags if t.startswith("arxiv:")]
    doi_ids = [t.replace("doi:", "") for t in tags if t.startswith("doi:")]

    return {
        "dataset_id": dataset_id,
        "language_codes": ";".join(language_codes) if language_codes else None,
        "modalities": ";".join(modality_tags) if modality_tags else None,
        "tasks": ";".join(task_tags) if task_tags else None,
        "license": ";".join(license_tags) if license_tags else "license_unstated",
        "arxiv_ids": ";".join(arxiv_ids) if arxiv_ids else None,
        "doi_ids": ";".join(doi_ids) if doi_ids else None,
    }


def classify_linguality(language_codes_str):
    """Returns 'monolingual', 'multilingual', or 'no_language_tag'."""
    if not language_codes_str:
        return "no_language_tag"
    codes = language_codes_str.split(";")
    return "monolingual" if len(codes) == 1 else "multilingual"


def build_publication_links(card_fields):
    """Builds clickable arXiv/DOI links for manual review, without
    fetching/analyzing their content. Returns None if neither present."""
    links = []
    if card_fields["arxiv_ids"]:
        links += [f"https://arxiv.org/abs/{a}" for a in card_fields["arxiv_ids"].split(";")]
    if card_fields["doi_ids"]:
        links += [f"https://doi.org/{d}" for d in card_fields["doi_ids"].split(";")]
    return ";".join(links) if links else None


# ---- Per-language config resolution ----
MAX_CONFIGS_TO_DISPLAY = 20  # some datasets have hundreds+ of configs - cap what's shown inline, always record the true count


def list_all_configs(dataset_id, hf_token):
    """Fetches the full config list ONCE per dataset - reused for both
    matching and recording, so we don't hit the API twice for the same
    thing. Returns (list_or_None). None means the fetch itself failed."""
    try:
        return get_dataset_config_names(dataset_id, token=hf_token)
    except Exception as e:
        print(f"      could not list configs at all: {type(e).__name__}: {e}")
        return None


def match_language_configs(hf_language_tag, all_configs):
    """
    Pure matching logic (no API call - takes an already-fetched config
    list). Most datasets use the bare code directly. Some use a
    {code}{delimiter}{script} naming convention instead (delimiters:
    _, -, ., /) - e.g. query code 'aae' only exists as config
    'aae_Latn'. Returns [] if nothing resolves - caller treats this as
    out of scope, not an error.
    """
    if all_configs is None:
        return []
    if hf_language_tag in all_configs:
        return [hf_language_tag]

    suffix_pattern = re.compile(rf"^{re.escape(hf_language_tag)}[_\-./]")
    script_suffixed = [c for c in all_configs if suffix_pattern.match(c)]
    if script_suffixed:
        print(f"      '{hf_language_tag}' not a direct config - found delimiter-suffixed match(es): {script_suffixed}")
        return script_suffixed

    return []


def format_config_list_for_display(configs):
    """Semicolon-joined, capped at MAX_CONFIGS_TO_DISPLAY with a
    truncation note - full true count is always in a separate field,
    this is just for eyeballing, not a complete record."""
    if not configs:
        return None
    shown = configs[:MAX_CONFIGS_TO_DISPLAY]
    suffix = f" ...({len(configs)} total)" if len(configs) > MAX_CONFIGS_TO_DISPLAY else ""
    return ";".join(shown) + suffix


def get_num_rows_for_config(dataset_id, config_name, hf_token):
    """Free, no-download row count via builder info. Returns
    (num_rows_or_None, method)."""
    try:
        builder = load_dataset_builder(dataset_id, config_name, token=hf_token)
    except Exception as e:
        return None, f"builder_error: {type(e).__name__}"

    if not builder.info.splits:
        return None, "no_split_info_available"

    total = sum(s.num_examples for s in builder.info.splits.values() if s.num_examples is not None)
    return (total if total else None), "precomputed_metadata"


# ---- Main per-dataset processing ----
def process_one_dataset(dataset_id, hf_language_tag, hf_token):
    """
    Returns (row_dict, category) where category is 'sized' (goes to
    OUTPUT_FILE), 'no_config' (goes to NO_CONFIG_LOG_FILE, full
    metadata but blank size), or None (genuinely unresolvable - caller
    logs a minimal SKIPPED_LOG_FILE entry).
    """
    info = _with_retry(lambda: api.dataset_info(dataset_id, files_metadata=False),
                        f"dataset_info({dataset_id})")
    if info is None:
        return None, None

    tags = info.tags or []
    card_data = fetch_card_data(dataset_id)
    card_fields = extract_card_fields(dataset_id, tags, card_data)
    linguality = classify_linguality(card_fields["language_codes"])
    num_languages = len(card_fields["language_codes"].split(";")) if card_fields["language_codes"] else 0

    readme_text = fetch_readme_body_text(dataset_id)
    provenance, synthetic_flag = classify_provenance(readme_text)
    publication_link = build_publication_links(card_fields)

    base_row = {
        "dataset_id": dataset_id,
        "hf_language_tag": hf_language_tag,
        "num_languages_in_dataset": num_languages,
        "linguality": linguality,
        "modality": card_fields["modalities"],
        "license": card_fields["license"],
        "provenance": provenance,
        "publication_link": publication_link,
        "synthetic_flag": synthetic_flag,
    }

    if linguality != "multilingual":
        all_configs = list_all_configs(dataset_id, hf_token)
        num_rows, method = get_num_rows_for_config(dataset_id, None, hf_token)
        return {
            **base_row,
            "num_rows": num_rows,
            "retrieval_method": "precomputed_metadata_monolingual" if num_rows else method,
            "all_configurations": format_config_list_for_display(all_configs),
            "all_configurations_count": len(all_configs) if all_configs is not None else None,
            "matched_configurations": None,  # not applicable - monolingual uses the default config, not a matched one
        }, "sized"

    # Multilingual - only proceed to a real size figure if a per-language config resolves
    all_configs = list_all_configs(dataset_id, hf_token)
    resolved_configs = match_language_configs(hf_language_tag, all_configs)

    config_fields = {
        "all_configurations": format_config_list_for_display(all_configs),
        "all_configurations_count": len(all_configs) if all_configs is not None else None,
    }

    if not resolved_configs:
        # Out of scope for sizing, but still recorded with everything else we know.
        return {**base_row, "num_rows": None, "retrieval_method": None,
                "matched_configurations": None, **config_fields}, "no_config"

    total_rows = 0
    any_resolved = False
    for config_name in resolved_configs:
        num_rows, method = get_num_rows_for_config(dataset_id, config_name, hf_token)
        if num_rows is not None:
            total_rows += num_rows
            any_resolved = True

    return {
        **base_row,
        "num_rows": total_rows if any_resolved else None,
        "retrieval_method": "precomputed_metadata_per_language_config" if any_resolved else "no_precomputed_metadata_available",
        "matched_configurations": ";".join(resolved_configs),
        **config_fields,
    }, "sized"


def run_all(max_languages=None, search_limit=SEARCH_LIMIT):
    hf_token = os.environ.get("HF_TOKEN")

    crosswalk = pd.read_csv(CROSSWALK_FILE)
    if max_languages is not None:
        crosswalk = crosswalk.head(max_languages)
        print(f"TEST RUN: limited to first {max_languages} language row(s)")

    output_path = Path(OUTPUT_FILE)
    no_config_path = Path(NO_CONFIG_LOG_FILE)
    skipped_path = Path(SKIPPED_LOG_FILE)

    already_done_keys = set()
    for path, key_cols in [(output_path, ("dataset_id", "hf_language_tag")),
                            (no_config_path, ("dataset_id", "hf_language_tag"))]:
        if path.exists():
            prior = pd.read_csv(path)
            already_done_keys |= set(zip(prior[key_cols[0]], prior[key_cols[1]]))
    if already_done_keys:
        print(f"Resuming: {len(already_done_keys)} (dataset, code) pair(s) already processed")

    for p in (output_path, no_config_path, skipped_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    write_header_output = not output_path.exists()
    write_header_no_config = not no_config_path.exists()

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

            print(f"    processing {ds.id}...")
            row, category = process_one_dataset(ds.id, hf_language_tag, hf_token)

            if category is None:
                print(f"    skipping {ds.id} - genuinely unresolvable (no card metadata at all)")
                skipped_rows.append({"dataset_id": ds.id, "hf_language_tag": hf_language_tag,
                                      "reason": "dataset_info_fetch_failed"})
                continue

            if category == "sized":
                pd.DataFrame([row]).to_csv(output_path, mode="a", header=write_header_output, index=False)
                write_header_output = False
                print(f"    wrote row for {ds.id} (num_rows={row['num_rows']}, method={row['retrieval_method']})")
            else:  # no_config
                pd.DataFrame([row]).to_csv(no_config_path, mode="a", header=write_header_no_config, index=False)
                write_header_no_config = False
                print(f"    wrote row for {ds.id} to NO_CONFIG_LOG (no per-language config - size left blank)")

    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(skipped_path, index=False)
        print(f"\n{len(skipped_rows)} dataset(s) genuinely unresolvable - logged to {SKIPPED_LOG_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simplified Tier A harvest - exact num_rows for monolingual/per-language-config datasets; full metadata (blank size) for the rest.")
    parser.add_argument("--test", type=int, metavar="N", default=None)
    parser.add_argument("--search-limit", type=int, default=SEARCH_LIMIT)
    args = parser.parse_args()

    run_all(max_languages=args.test, search_limit=args.search_limit)
