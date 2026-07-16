"""
tier_a_harvest_v6.py

Full successor to tier_a_harvest_v5.py - same crosswalk-driven search,
same card/provenance/linguality extraction, same field set, but with
an entirely new THREE-TIER size resolution strategy replacing v5's
load_dataset()-based approach (which had real runtime problems on
large datasets - see conversation).

--------------------------------------------------------------------
THREE-TIER SIZE RESOLUTION (replaces v5's get_num_rows entirely)
--------------------------------------------------------------------
TIER 1 - exact, via Datasets Server /size endpoint, NO DOWNLOAD:
  - Monolingual: tries the OVERALL dataset first (no config), falls
    back to the 'default' config specifically - matches your
    instruction to pull num_examples/num_bytes from 'default' for
    monolingual datasets.
  - Multilingual WITH a per-language config (see match_language_configs
    - matches the code as a complete segment ANYWHERE in the config
    name, not just as a prefix): queries /size for that config
    directly. Summed across all matched configs if more than one.

TIER 2 - exact-or-near-exact, via a language-COLUMN query on the
  auto-converted Parquet files, for multilingual datasets with NO
  per-language config: checks the Datasets Server /parquet endpoint
  for file URLs, reads one file's schema (footer-only, no download) to
  find a language-identifying column, then runs a column-pruned DuckDB
  query directly against the remote Parquet file(s) - only the
  language column and (if found) the text column are actually read,
  not the whole file, even for a multi-GB dataset. Byte size uses
  octet_length(encode(text_col)) - NOT length()/LENGTH, which counts
  UTF-8 characters, not bytes, and would silently undercount non-Latin
  scripts. Confirmed via direct testing against real Parquet data with
  Japanese text - see conversation.

TIER 3 - genuine residual: not Parquet-convertible at all (old
  script-based loader), or Parquet but no language-identifying column
  anywhere in the schema. Flagged, not estimated.

--------------------------------------------------------------------
FIELDS RECORDED
--------------------------------------------------------------------
dataset_id, language_code, languages_in_dataset, linguality, license,
modalities, tasks, num_examples, num_bytes, resolution_tier,
matched_configurations, all_configurations, all_configurations_count,
lang_column_used, text_column_used, provenance, synthetic_flag,
arxiv_ids, doi_ids, manual_review.

--------------------------------------------------------------------
LINGUALITY / PROVENANCE / MANUAL_REVIEW - unchanged from v5
--------------------------------------------------------------------
See tier_a_harvest_v5.py for the full rationale - multilinguality tag
first then language-tag count; 3-category README keyword heuristic;
manual_review = "404 - need to grant access to dataset" (access
issues, checked first) or "check unclear provenance" (resolved size
but unclear provenance).

--------------------------------------------------------------------
FOUR OUTPUT FILES, MUTUALLY EXCLUSIVE
--------------------------------------------------------------------
1. FULL_CLEAN_FILE - size resolved (any tier) AND provenance NOT unclear.
2. MANUAL_REVIEW_FILE - access issues, OR size resolved but unclear provenance.
3. EVERYTHING_ELSE_FILE - Tier 3 residual: no per-language config AND
   not resolvable via Tier 2 either - regardless of provenance.
4. SKIPPED_NON_TEXT_FILE - modality explicitly stated, no "text" present.

--------------------------------------------------------------------
NOT YET VALIDATED AGAINST LIVE DATA
--------------------------------------------------------------------
No network access to huggingface.co / datasets-server.huggingface.co
from the environment this was written in. The /size and /parquet
endpoint response parsing, and the DuckDB-over-HTTP query, are the
least-tested pieces here - the DuckDB query STRING and the
octet_length(encode()) byte-counting fix ARE validated against a real
local Parquet file (see conversation), but the actual remote-URL fetch
via httpfs is not. Run --test 5 first.

--------------------------------------------------------------------
DEPENDENCIES / SETUP
--------------------------------------------------------------------
pip install pandas huggingface_hub requests duckdb pyarrow
export HF_TOKEN=hf_...
"""
import io
import os
import re
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import requests
from datasets import load_dataset_builder
from huggingface_hub import DatasetCard, HfApi

# ---- CONFIG ----
CROSSWALK_FILE = "../../crosswalk/data/processed/full_language_reference.csv"
FULL_CLEAN_FILE = "../data/tier_a_v6/tier_a_v6_full_clean.csv"
MANUAL_REVIEW_FILE = "../data/tier_a_v6/tier_a_v6_manual_review.csv"
EVERYTHING_ELSE_FILE = "../data/tier_a_v6/tier_a_v6_everything_else.csv"
SKIPPED_NON_TEXT_FILE = "../data/tier_a_v6/tier_a_v6_skipped_non_text_modality.csv"

SEARCH_LIMIT = 30
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 5
MAX_CONFIGS_TO_DISPLAY = 20
RATE_LIMIT_WAIT_SECONDS = 300

DATASETS_SERVER_SIZE_URL = "https://datasets-server.huggingface.co/size"
DATASETS_SERVER_PARQUET_URL = "https://datasets-server.huggingface.co/parquet"

LANGUAGE_COLUMN_CANDIDATES = {
    "iso3", "iso_639_3", "iso639_3", "language", "lang", "language_code",
    "lang_code", "locale", "language_id", "lang_id",
}
TEXT_COLUMN_CANDIDATES = {"text", "content", "sentence", "document", "raw_text", "body"}

api = HfApi(token=os.environ.get("HF_TOKEN"))
_duckdb_con = None

ACCESS_ISSUE_KEYWORDS = {
    "need to manually accept dataset access": ["gated", "access request", "must agree", "accept the"],
    "private dataset": ["private", "does not exist", "repository not found", "404"],
}


def classify_access_issue(response_text):
    text_lower = (response_text or "").lower()
    for reason, keywords in ACCESS_ISSUE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return reason
    return "other"


# ---- Rate limit handling ----
def is_rate_limit_error(e):
    text = str(e).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def call_with_rate_limit_handling(fn, description):
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


# ---- Provenance heuristic (3 categories) - unchanged from v5 ----
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


# ---- Card field extraction - unchanged from v5 ----
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
    mtags = card_fields["multilinguality_tags"]
    if "monolingual" in mtags:
        return "monolingual"
    if "multilingual" in mtags:
        return "multilingual"
    num_languages = len(card_fields["language_tags"])
    if num_languages == 0:
        return "no_language_tag"
    return "monolingual" if num_languages == 1 else "multilingual"


# ---- Config resolution - unchanged from v5 (segment-anywhere matching) ----
def try_list_all_configs(dataset_id, hf_token):
    try:
        return get_dataset_config_names_safe(dataset_id, hf_token), None
    except Exception as e:
        return None, str(e)


def get_dataset_config_names_safe(dataset_id, hf_token):
    from datasets import get_dataset_config_names
    return call_with_rate_limit_handling(
        lambda: get_dataset_config_names(dataset_id, token=hf_token),
        f"get_dataset_config_names({dataset_id})",
    )


DELIMITER_PATTERN = re.compile(r"[_\-./]")


def match_language_configs(hf_language_tag, all_configs):
    """Matches whenever the code appears as a COMPLETE SEGMENT anywhere
    in the config name (split on _, -, ., /) - not just as a prefix.
    Handles "aai_Latn", "eng-aai", "eng_Latn-zac_Latn", and bare "aai"."""
    if not all_configs:
        return []
    matches = []
    for config in all_configs:
        segments = DELIMITER_PATTERN.split(config)
        if hf_language_tag in segments:
            matches.append(config)
    return matches


def format_config_list_for_display(configs):
    if not configs:
        return None
    shown = configs[:MAX_CONFIGS_TO_DISPLAY]
    suffix = f" ...({len(configs)} total)" if len(configs) > MAX_CONFIGS_TO_DISPLAY else ""
    return ";".join(shown) + suffix


# ---- TIER 1: Datasets Server /size endpoint (exact, no download) ----
def get_size_via_api(dataset_id, config_name, hf_token):
    """
    GET /size?dataset=X[&config=Y] - returns exact num_examples/num_bytes,
    precomputed server-side, no download.

    CONFIRMED response shape (via HF's own docs, ibm/duorc example):
    data["size"]["dataset"] is ALWAYS the WHOLE DATASET's aggregate
    total across every config, REGARDLESS of whether &config=Y was
    passed in the request - it does NOT scope down. The real
    per-config figures live in data["size"]["configs"], a LIST of
    per-config objects each carrying their own "config" key - you have
    to find the matching entry yourself. The original version of this
    function always read data["size"]["dataset"], which meant every
    per-language-config lookup was silently wrong (either returning
    the whole dataset's total, or failing to find data at all) - see
    conversation for how this was confirmed against real HF docs.

    Returns dict {"num_examples":..., "num_bytes":...} or None if
    unavailable.
    """
    def _fetch():
        params = {"dataset": dataset_id}
        if config_name:
            params["config"] = config_name
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        resp = requests.get(DATASETS_SERVER_SIZE_URL, params=params, headers=headers, timeout=30)
        if resp.status_code in (404, 501):
            return None
        resp.raise_for_status()
        return resp.json()

    data = _with_retry(_fetch, f"fetch /size for {dataset_id} config={config_name}", max_retries=2)
    if not data or "size" not in data:
        return None

    if config_name is None:
        # No config specified - the whole-dataset aggregate IS what we want here.
        d = data["size"].get("dataset")
    else:
        # Find the matching entry in the per-config list - NOT data["size"]["dataset"].
        configs_list = data["size"].get("configs", [])
        d = next((c for c in configs_list if c.get("config") == config_name), None)

    if not d:
        return None
    num_examples = d.get("num_rows")
    num_bytes = d.get("num_bytes_original_files")
    if num_bytes is None:
        num_bytes = d.get("num_bytes_parquet_files")
    if num_examples is None and num_bytes is None:
        return None
    return {"num_examples": num_examples, "num_bytes": num_bytes}


def get_size_via_builder(dataset_id, config_name, hf_token):
    """
    FALLBACK path for Tier 1: load_dataset_builder(...).info.splits
    [split].num_examples - free, no download. Uses a different code
    path than /size (the `datasets` library's own resolution logic,
    not the Datasets Server API), so coverage can genuinely differ -
    confirmed on real data from a live run (tier_a_v5_metadata.py):
    24 of 31 real datasets got a genuine num_examples value this way,
    including several where /size might plausibly have nothing cached
    yet. Only gives num_examples cleanly - byte size at this level is
    coarse (download_size is whole-config, not split-scoped) so NOT
    returned here to avoid mixing a differently-scoped number in with
    /size's split-scoped figure.

    Returns dict {"num_examples":..., "num_bytes": None} or None.
    """
    try:
        builder = call_with_rate_limit_handling(
            lambda: load_dataset_builder(dataset_id, config_name, token=hf_token),
            f"load_dataset_builder({dataset_id}, {config_name})",
        )
    except Exception:
        return None

    if not builder.info.splits:
        return None
    total = sum(s.num_examples for s in builder.info.splits.values() if s.num_examples is not None)
    if not total:
        return None
    return {"num_examples": total, "num_bytes": None}


def get_size_tier1(dataset_id, config_name, hf_token):
    """
    Combined Tier 1 entry point: try /size first (gives both
    num_examples AND num_bytes, validated against real HF docs - see
    get_size_via_api). If that comes back empty, fall back to
    load_dataset_builder (num_examples only, but a genuinely different
    resolution path that can succeed where /size doesn't - confirmed
    on real data, see get_size_via_builder). Returns
    (result_dict_or_None, method_used) where method_used is
    "size_api", "builder_metadata", or None if both failed.
    """
    result = get_size_via_api(dataset_id, config_name, hf_token)
    if result is not None:
        return result, "size_api"

    result = get_size_via_builder(dataset_id, config_name, hf_token)
    if result is not None:
        return result, "builder_metadata"

    return None, None


# ---- TIER 2: Parquet + language-column query via DuckDB ----
def get_duckdb_connection():
    global _duckdb_con
    if _duckdb_con is None:
        _duckdb_con = duckdb.connect()
        _duckdb_con.execute("INSTALL httpfs; LOAD httpfs;")
    return _duckdb_con


def get_parquet_file_urls(dataset_id, hf_token):
    def _fetch():
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        resp = requests.get(DATASETS_SERVER_PARQUET_URL, params={"dataset": dataset_id},
                             headers=headers, timeout=30)
        if resp.status_code in (404, 501):
            return None
        resp.raise_for_status()
        return resp.json()

    data = _with_retry(_fetch, f"fetch /parquet for {dataset_id}", max_retries=2)
    if not data or "parquet_files" not in data or not data["parquet_files"]:
        return []
    return [f["url"] for f in data["parquet_files"] if "url" in f]


FOOTER_SIZE_ERROR_PATTERN = re.compile(r"smaller than the size reported by footer's \((\d+)\s*bytes?\)")


def get_parquet_schema_columns(file_url, hf_token=None, initial_range_bytes=1_000_000):
    """
    Footer-only read to get column names. Two things this handles that
    a naive version doesn't:
    1. Sends the auth token - some datasets need it just to read raw
       file bytes even when /parquet's own API call succeeded (see
       conversation: sil-ai/bloom-vist 401'd without this).
    2. If the footer metadata itself is bigger than the initial range
       fetched (rare, but real - seen on lbourdois/panlex, footer
       reported as 3.78MB vs the 1MB initially fetched, likely from a
       huge schema/column-statistics section), parses the REQUIRED
       size directly out of PyArrow's own error message and retries
       with a range big enough to cover it - rather than uselessly
       re-requesting the same too-small range twice.
    """
    headers_base = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    range_bytes = initial_range_bytes
    last_error = None

    for attempt in range(3):
        try:
            headers = {**headers_base, "Range": f"bytes=-{range_bytes}"}
            resp = requests.get(file_url, headers=headers, timeout=30)
            if resp.status_code not in (200, 206):
                raise RuntimeError(f"unexpected status {resp.status_code}")
            return pq.ParquetFile(io.BytesIO(resp.content)).schema.names
        except Exception as e:
            last_error = e
            match = FOOTER_SIZE_ERROR_PATTERN.search(str(e))
            if match:
                required = int(match.group(1))
                range_bytes = required + 100_000  # small buffer over the exact required size
                print(f"    footer larger than expected ({required} bytes) - retrying with a {range_bytes}-byte range")
                continue
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"    [retry {attempt + 1}/3] read parquet schema for {file_url}: {e} - waiting {wait}s")
            time.sleep(wait)

    print(f"    giving up on read parquet schema for {file_url} after retries: {last_error}")
    return None


def detect_column(columns, candidates):
    if not columns:
        return None
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def query_language_stats(file_urls, lang_column, text_column, target_lang_code):
    """
    Column-pruned DuckDB query over remote Parquet file(s) - only the
    needed columns/row-groups are read, not the whole file.
    octet_length(encode(...)) gives TRUE UTF-8 byte length - confirmed
    via direct testing against real Japanese text (see module docstring).
    """
    con = get_duckdb_connection()
    url_list_sql = "[" + ", ".join(f"'{u}'" for u in file_urls) + "]"

    if text_column:
        query = f"""
            SELECT count(*) AS n_rows, sum(octet_length(encode("{text_column}"))) AS n_bytes
            FROM read_parquet({url_list_sql})
            WHERE "{lang_column}" = '{target_lang_code}'
        """
    else:
        query = f"""
            SELECT count(*) AS n_rows, NULL AS n_bytes
            FROM read_parquet({url_list_sql})
            WHERE "{lang_column}" = '{target_lang_code}'
        """

    result = _with_retry(lambda: con.execute(query).fetchone(), f"DuckDB query for {target_lang_code}", max_retries=2)
    if result is None:
        return None, None
    return result


def resolve_via_tier2(dataset_id, hf_language_tag, hf_token):
    """Returns (num_examples, num_bytes, lang_col, text_col) or
    (None, None, None, None) with a reason if Tier 2 doesn't apply."""
    file_urls = get_parquet_file_urls(dataset_id, hf_token)
    if not file_urls:
        return None, None, None, None, "not_parquet_convertible"

    columns = get_parquet_schema_columns(file_urls[0], hf_token=hf_token)
    lang_column = detect_column(columns, LANGUAGE_COLUMN_CANDIDATES)
    if not lang_column:
        return None, None, None, None, "no_language_column_in_schema"

    text_column = detect_column(columns, TEXT_COLUMN_CANDIDATES)
    n_rows, n_bytes = query_language_stats(file_urls, lang_column, text_column, hf_language_tag)
    if n_rows is None:
        return None, None, lang_column, text_column, "duckdb_query_failed"
    return n_rows, n_bytes, lang_column, text_column, None


# ---- Main per-dataset processing ----
def process_one_dataset(dataset_id, hf_language_tag, hf_token):
    """Returns (row_dict, bucket) where bucket is one of:
    'full_clean', 'manual_review', 'everything_else'."""

    info = _with_retry(lambda: api.dataset_info(dataset_id, files_metadata=False),
                        f"dataset_info({dataset_id})")
    if info is None:
        return {
            "dataset_id": dataset_id, "language_code": hf_language_tag,
            "languages_in_dataset": None, "linguality": None, "license": None, "modalities": None, "tasks": None,
            "num_examples": None, "num_bytes": None, "resolution_tier": None,
            "matched_configurations": None, "all_configurations": None, "all_configurations_count": None,
            "lang_column_used": None, "text_column_used": None, "tier1_size_source": None,
            "provenance": None, "synthetic_flag": None, "arxiv_ids": None, "doi_ids": None,
            "manual_review": "404 - need to grant access to dataset",
        }, "manual_review"

    tags = info.tags or []
    card_fields = extract_card_fields(tags)

    # ---- Text-modality filter, checked early ----
    if card_fields["modalities"] != "modality_unstated":
        modality_list = card_fields["modalities"].split(";")
        if "text" not in modality_list:
            print(f"    skipping {dataset_id} - modality is {card_fields['modalities']!r}, no 'text' present")
            return {"dataset_id": dataset_id, "language_code": hf_language_tag,
                    "modalities": card_fields["modalities"]}, "skipped_non_text_modality"

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

    def finalize(num_examples, num_bytes, tier, matched_config, all_cfgs, all_cfgs_count,
                 lang_col=None, text_col=None, size_source=None):
        row = {
            **base_row, "num_examples": num_examples, "num_bytes": num_bytes,
            "resolution_tier": tier, "matched_configurations": matched_config,
            "all_configurations": all_cfgs, "all_configurations_count": all_cfgs_count,
            "lang_column_used": lang_col, "text_column_used": text_col,
            "tier1_size_source": size_source,
        }
        if num_examples is not None:
            if provenance == "unclear":
                return {**row, "manual_review": "check unclear provenance"}, "manual_review"
            return {**row, "manual_review": None}, "full_clean"
        return {**row, "manual_review": None}, "everything_else"

    # ---- TIER 1: monolingual (overall, then 'default') - /size first, builder fallback ----
    if linguality != "multilingual":
        result, size_source = get_size_tier1(dataset_id, None, hf_token)
        matched_config = "overall"
        if result is None:
            result, size_source = get_size_tier1(dataset_id, "default", hf_token)
            matched_config = "default"
        if result is not None:
            return finalize(result["num_examples"], result["num_bytes"], "tier1_monolingual",
                             matched_config, None, None, size_source=size_source)
        # No Tier 1 data for monolingual - genuinely nothing else to try (no language column
        # makes sense to query on a single-language dataset) - straight to residual.
        return finalize(None, None, None, None, None, None)

    # ---- Multilingual: TIER 1 first (per-language config, /size first then builder fallback) ----
    all_configs, config_err = try_list_all_configs(dataset_id, hf_token)
    if config_err and classify_access_issue(config_err) != "other":
        return {**base_row, "num_examples": None, "num_bytes": None, "resolution_tier": None,
                "matched_configurations": None, "all_configurations": None, "all_configurations_count": None,
                "lang_column_used": None, "text_column_used": None, "tier1_size_source": None,
                "manual_review": "404 - need to grant access to dataset"}, "manual_review"

    config_display = format_config_list_for_display(all_configs)
    config_count = len(all_configs) if all_configs is not None else None
    resolved_configs = match_language_configs(hf_language_tag, all_configs)

    if resolved_configs:
        total_examples, total_bytes = 0, 0
        any_resolved = False
        size_source_used = None
        for config_name in resolved_configs:
            result, size_source = get_size_tier1(dataset_id, config_name, hf_token)
            if result is not None:
                total_examples += result["num_examples"] or 0
                total_bytes += result["num_bytes"] or 0
                any_resolved = True
                size_source_used = size_source  # last-resolved method wins if configs used different sources
        if any_resolved:
            return finalize(total_examples, total_bytes, "tier1_per_language_config",
                             ";".join(resolved_configs), config_display, config_count, size_source=size_source_used)
        # Config(s) matched but Tier 1 had nothing (neither /size nor builder) - fall through
        # to Tier 2 anyway, since the dataset may still be Parquet-convertible with a language column.

    # ---- TIER 2: Parquet + language column query ----
    n_examples, n_bytes, lang_col, text_col, tier2_reason = resolve_via_tier2(dataset_id, hf_language_tag, hf_token)
    if n_examples is not None:
        return finalize(n_examples, n_bytes, "tier2_column_query",
                         None, config_display, config_count, lang_col, text_col)

    # ---- TIER 3: residual ----
    print(f"    Tier 3 residual for {dataset_id}: {tier2_reason}")
    return finalize(None, None, None, None, config_display, config_count, lang_col, text_col)


def run_all(max_languages=None, search_limit=SEARCH_LIMIT):
    hf_token = os.environ.get("HF_TOKEN")

    crosswalk = pd.read_csv(CROSSWALK_FILE)
    if max_languages is not None:
        crosswalk = crosswalk.head(max_languages)
        print(f"TEST RUN: limited to first {max_languages} language row(s)")

    output_paths = {
        "full_clean": Path(FULL_CLEAN_FILE),
        "manual_review": Path(MANUAL_REVIEW_FILE),
        "everything_else": Path(EVERYTHING_ELSE_FILE),
        "skipped_non_text_modality": Path(SKIPPED_NON_TEXT_FILE),
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
    tier_counts = {}

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

            tier = row.get("resolution_tier")
            if tier:
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            print(f"    -> {bucket} (tier={tier})")

    print("\n=== SUMMARY ===")
    for bucket, count in bucket_counts.items():
        print(f"{bucket}: {count}")
    print("\n=== RESOLUTION TIER BREAKDOWN (within full_clean + manual_review) ===")
    for tier, count in tier_counts.items():
        print(f"{tier}: {count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tier A harvest v6 - crosswalk-driven, 3-tier size resolution.")
    parser.add_argument("--test", type=int, metavar="N", default=None)
    parser.add_argument("--search-limit", type=int, default=SEARCH_LIMIT)
    args = parser.parse_args()

    run_all(max_languages=args.test, search_limit=args.search_limit)
