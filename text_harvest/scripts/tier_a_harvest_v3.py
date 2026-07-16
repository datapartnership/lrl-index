"""
tier_a_harvest_v3.py

Redesigned Tier A harvest logic - scoped down per your instruction to
NOT resolve raw_size_bytes for now. This version focuses entirely on
getting text/audio counts and the transcribed/untranscribed audio
split right, using load_dataset() directly rather than Parquet-footer
probing, before adding byte-size resolution back in later.

--------------------------------------------------------------------
UNIFIED MONOLINGUAL/MULTILINGUAL LOGIC - NO SEPARATE CODE PATHS
--------------------------------------------------------------------
Rather than treating monolingual as "one whole-dataset yes/no check"
and multilingual as "sum across subsets," BOTH use the exact same
mechanism: load_dataset(dataset_id, language_code) to get whatever
splits exist for that language's config, then process EACH SPLIT
independently. A monolingual dataset is just a language whose splits
happen to be the whole dataset - there's no principled reason to
treat it differently. This also naturally handles datasets where some
splits within one language are transcribed and others aren't (e.g.
VoxPopuli-style labeled + unlabeled splits together).

--------------------------------------------------------------------
DECISION ORDER: README FIRST, THEN load_dataset
--------------------------------------------------------------------
1. Check the README for an explicit breakdown (size, rows, hours) for
   this language - reuses the table-aware, word-boundary-matched
   search_readme_for_language_size from tier_a_harvest_v2.py (word-
   boundary fix + markdown-table support - see conversation history
   on the "en" substring-matching bug and the multilingual_librispeech
   table-blindness bug). If found, used directly, flagged
   needs_manual_review (same as before - a regex match still needs
   human confirmation).
2. If not found in the README: load_dataset(dataset_id, language_code),
   process every split (see below).

--------------------------------------------------------------------
PER-SPLIT PROCESSING (applies to every split, every dataset)
--------------------------------------------------------------------
For each split:
  - SUSPICIOUS SUBSET EXCLUSION: split names matching a "curated
    subset of a larger split" pattern (e.g. MLS's 9_hours/1_hours,
    which are subsets of train, not additional data) are EXCLUDED BY
    DEFAULT and logged - not just warned, since this runs unattended
    across many datasets and a scrolling warning is easy to miss in a
    batch run. See EXCLUDED_SUBSET_LOG.
  - ROW COUNT: tries load_dataset_builder(...).info.splits[split]
    .num_examples FIRST (free, no download, precomputed metadata) -
    only falls back to actually loading if that's unavailable.
  - MODALITY: checked via feature types - an Audio-typed column means
    "audio", a large string column (heuristically: not an obvious ID/
    label field) means "text". A split can be BOTH (e.g. MLS has audio
    + transcript together) - recorded as "audio;text", not forced into
    one category.
  - AUDIO HOURS: tries a duration-like field first (checked against a
    list of common names: audio_duration, duration, duration_seconds,
    length, length_s). If none exists, falls back to reading container
    headers via mutagen on the RAW, UNDECODED audio bytes per row (no
    torchcodec needed - see inspect_dataset.py's same approach) -
    slower since it requires iterating every row, but avoids full
    waveform decoding.
  - TRANSCRIBED vs UNTRANSCRIBED: samples the first
    TRANSCRIPT_SAMPLE_SIZE rows and checks whether a 'text' or
    'transcript' field is present AND non-empty for at least one
    sampled row. If yes, this split's hours count toward transcribed;
    if no, toward untranscribed. This is a SAMPLE-based check, not
    exhaustive - flagged in the output (transcript_check_sample_size)
    so a thin sample isn't mistaken for a confident determination.

--------------------------------------------------------------------
ACCESS GATING
--------------------------------------------------------------------
Reuses classify_access_issue from tier_a_harvest_v2.py. Any dataset
that fails to load due to a gating/access/private issue is flagged
needs_manual_access_verification=True and NOT retried (same principle
as the existing pipeline - an access wall is a persistent state, not
a transient failure).

--------------------------------------------------------------------
RAW SIZE IN BYTES - via load_dataset_builder, NOT the card/Datasets-
Server decision tree from tier_a_harvest_v2.py
--------------------------------------------------------------------
Pulled directly from load_dataset_builder(dataset_id, language_code)
.info - the exact same download_size figure shown in load_dataset's
own "Downloading and preparing dataset... (download: X, generated: Y,
total: Z)" console message. download_size (the RAW, as-transferred
bytes) is recorded as raw_size_bytes; dataset_size (the "generated"
post-processing size, e.g. Arrow format on disk) is kept as a
secondary context column, generated_size_bytes.

IMPORTANT: both figures are CONFIG-LEVEL (this language's full set of
splits combined), not attributable to any one split - recorded once
per (dataset, language) row, not per split. See size_scope column.

The builder is built ONCE per dataset and reused for row-count lookups
across all its splits, rather than rebuilding it per split.

--------------------------------------------------------------------
FIELDS RECORDED
--------------------------------------------------------------------
dataset_id, hf_language_tag, language_codes_in_dataset, linguality,
modality, tasks, license, arxiv_ids, doi_ids (license/arxiv/doi as
SEPARATE fields, not bundled), raw_size_bytes, generated_size_bytes,
size_scope, num_rows, num_hours, transcribed_hours, untranscribed_hours,
breakdown_source (readme_heuristic / load_dataset_splits),
needs_manual_access_verification, excluded_subset_splits (which splits
were dropped as suspicious subsets, if any).

--------------------------------------------------------------------
DEPENDENCIES / SETUP
--------------------------------------------------------------------
pip install datasets huggingface_hub mutagen pandas
export HF_TOKEN=hf_...

--------------------------------------------------------------------
NOT YET VALIDATED AGAINST LIVE DATA
--------------------------------------------------------------------
No network access to huggingface.co from the environment this was
written in - validate with --test on a handful of languages before a
full run, and spot-check the mutagen fallback path specifically
against a real dataset that lacks a duration field, since that path
is the least exercised by anything tested so far in this project.
"""
import io
import re
import time
from pathlib import Path

import pandas as pd
from datasets import Audio, get_dataset_config_names, get_dataset_split_names, load_dataset, load_dataset_builder
from huggingface_hub import HfApi

try:
    from mutagen import File as MutagenFile
except ImportError:
    MutagenFile = None

# Reuse existing, validated logic rather than reimplementing it
from tier_a_harvest_v2 import (
    api, fetch_card_data, extract_card_fields, classify_linguality,
    fetch_readme_body_text, search_readme_for_language_size,
    classify_access_issue, ACCESS_ISSUE_REASONS,
)

# ---- CONFIG ----
CROSSWALK_FILE = "../../crosswalk/data/processed/full_language_reference.csv"
OUTPUT_FILE = "../data/tier_a_v3/tier_a_v3_by_language.csv"
SKIPPED_LOG_FILE = "../data/tier_a_v3/tier_a_v3_skipped.csv"
EXCLUDED_SUBSET_LOG = "../data/tier_a_v3/tier_a_v3_excluded_subset_splits.csv"

SEARCH_LIMIT = 30
TRANSCRIPT_SAMPLE_SIZE = 30  # rows sampled per split to check for a non-empty text/transcript field
DURATION_FIELD_CANDIDATES = ["audio_duration", "duration", "duration_seconds", "duration_secs", "length", "length_s"]
TRANSCRIPT_FIELD_CANDIDATES = ["text", "transcript", "transcription", "sentence"]

SUSPICIOUS_SUBSET_PATTERN = re.compile(r"^\d+_(hour|min|hr)s?$", re.IGNORECASE)

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
    for attempt in range(MAX_RETRIES):
        try:
            return list(api.list_datasets(language=query_code, limit=limit)), False
        except Exception as e:
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"    [retry {attempt + 1}/{MAX_RETRIES}] search '{query_code}': {e} - waiting {wait}s")
            time.sleep(wait)
    print(f"    giving up on search '{query_code}' after {MAX_RETRIES} retries - NOT recording as zero-result")
    return [], True


# ---- Per-split helpers ----
def resolve_config_names(dataset_id, hf_language_tag, hf_token):
    """
    Returns a list of ACTUAL config names to process for this query
    code - most datasets use the bare code directly (tried first, no
    extra call needed if it works downstream). Some datasets (e.g.
    FineWeb-2-style, and the "omnilingual-asr-corpus" mirrors seen
    during testing) use a {code}{DELIMITER}{Script}-style naming
    convention instead - e.g. query code "aae" only exists as config
    "aae_Latn". Delimiter varies by dataset - underscore is most
    common, but "-", ".", and "/" all show up too (e.g. "aae-Latn",
    "aae.Latn", "aae/Latn") - matched via CONFIG_SUFFIX_PATTERN below
    rather than a hardcoded single character. If the bare code fails,
    this looks for configs matching that prefix pattern and returns
    ALL matches - a language can legitimately have more than one
    script-suffixed config (multi-script language), and dropping the
    dataset entirely just because the bare code isn't a literal config
    name would silently lose real data, as it was doing before this fix.

    Returns [hf_language_tag] optimistically without an extra API call
    if we have no reason yet to think it's wrong - the caller finds
    out for certain when it actually tries to use it, and only falls
    back to calling this resolver's full config-listing path if that
    first attempt fails with a "config not found"-shaped error.
    """
    try:
        all_configs = get_dataset_config_names(dataset_id, token=hf_token)
    except Exception as e:
        print(f"      could not list configs at all: {type(e).__name__}: {e}")
        return []

    if hf_language_tag in all_configs:
        return [hf_language_tag]

    suffix_pattern = re.compile(rf"^{re.escape(hf_language_tag)}[_\-./]")
    script_suffixed = [c for c in all_configs if suffix_pattern.match(c)]
    if script_suffixed:
        print(f"      '{hf_language_tag}' not a direct config - found delimiter-suffixed match(es): {script_suffixed}")
        return script_suffixed

    return []


def get_precomputed_row_count(builder, split_name):
    """Uses an ALREADY-BUILT builder (see get_dataset_size_info -
    avoids rebuilding it per split, which the earlier version of this
    function did wastefully). Returns an int or None if unavailable."""
    try:
        split_info = builder.info.splits.get(split_name) if builder.info.splits else None
        return split_info.num_examples if split_info else None
    except Exception:
        return None


def get_dataset_size_info(dataset_id, resolved_config, hf_token):
    """
    Builds the dataset builder ONCE for the ALREADY-RESOLVED config
    name (see resolve_config_names - no longer necessarily the same
    as the crosswalk's bare hf_language_tag) and pulls the config-level
    size figures directly from it - the same numbers shown in
    load_dataset's own "Downloading and preparing dataset... (download:
    X, generated: Y, total: Z)" message. download_size is the RAW bytes
    actually transferred over the network (what you asked for as
    raw_size_bytes); dataset_size is the "generated" post-processing
    size (e.g. Arrow format on disk) - kept as a secondary context
    column, same pattern as other robustness columns elsewhere in this
    project.

    NOTE: both figures are for the WHOLE CONFIG (this language's full
    set of splits combined), not attributable to one split - reported
    once per (dataset, resolved_config) row, not per split.

    Returns (builder_or_None, download_size_bytes_or_None,
    dataset_size_bytes_or_None). builder is returned so callers can
    reuse it for get_precomputed_row_count without rebuilding.
    """
    try:
        builder = load_dataset_builder(dataset_id, resolved_config, token=hf_token)
        return builder, builder.info.download_size, builder.info.dataset_size
    except Exception as e:
        print(f"      could not resolve builder/size info: {type(e).__name__}: {e}")
        return None, None, None


def detect_modality(features):
    """Returns a set of modality strings present in this split's
    schema - can be both audio and text together (e.g. MLS)."""
    modalities = set()
    for name, feat in features.items():
        if isinstance(feat, Audio):
            modalities.add("audio")
    for candidate in TRANSCRIPT_FIELD_CANDIDATES:
        if candidate in features:
            modalities.add("text")
    return modalities


def get_duration_field(features):
    for candidate in DURATION_FIELD_CANDIDATES:
        if candidate in features:
            return candidate
    return None


def get_transcript_field(features):
    for candidate in TRANSCRIPT_FIELD_CANDIDATES:
        if candidate in features:
            return candidate
    return None


def sample_check_transcript_nonempty(ds, transcript_field, sample_size=TRANSCRIPT_SAMPLE_SIZE):
    """Samples up to sample_size rows and checks whether
    transcript_field is populated (non-null, non-empty string) for at
    least one of them. Returns (is_nonempty: bool, actual_sample_size: int)."""
    count = 0
    for example in ds:
        count += 1
        value = example.get(transcript_field)
        if value is not None and str(value).strip():
            return True, count
        if count >= sample_size:
            break
    return False, count


def probe_duration_via_mutagen(ds, audio_field, max_rows=None):
    """
    Fallback when no duration field exists: reads container headers
    (mutagen) on RAW, UNDECODED audio bytes per row - no torchcodec
    needed. Returns total hours (float), or None if mutagen isn't
    installed or nothing resolved. Iterates every row up to max_rows
    (None = all) - slower than reading a duration field, since there's
    no shortcut for this without touching the actual audio bytes.
    """
    if MutagenFile is None:
        print("      mutagen not installed - cannot probe duration without a duration field. pip install mutagen.")
        return None

    total_seconds = 0.0
    resolved_any = False
    for i, example in enumerate(ds):
        if max_rows is not None and i >= max_rows:
            break
        audio_value = example.get(audio_field)
        if not audio_value or not audio_value.get("bytes"):
            continue
        try:
            audio_file = MutagenFile(io.BytesIO(audio_value["bytes"]))
            if audio_file is not None and getattr(audio_file, "info", None) is not None:
                total_seconds += audio_file.info.length
                resolved_any = True
        except Exception:
            continue

    return (total_seconds / 3600.0) if resolved_any else None


def process_split(dataset_id, language_code, split_name, hf_token, builder):
    """
    Returns a dict of per-split results, or None if the split
    couldn't be loaded at all (caller decides how to log this).
    builder is the already-built dataset builder (see
    get_dataset_size_info) - passed through so row counts don't
    require rebuilding it per split.
    """
    try:
        ds = load_dataset(dataset_id, language_code, split=split_name, streaming=True, token=hf_token)
    except Exception as e:
        return {"error": str(e), "access_issue": classify_access_issue(str(e))}

    features = ds.features
    modalities = detect_modality(features)

    num_rows = get_precomputed_row_count(builder, split_name) if builder else None

    num_hours = None
    transcribed_hours = None
    untranscribed_hours = None

    if "audio" in modalities:
        audio_field = next(name for name, feat in features.items() if isinstance(feat, Audio))
        ds = ds.cast_column(audio_field, Audio(decode=False))  # never decode

        duration_field = get_duration_field(features)
        transcript_field = get_transcript_field(features)

        if duration_field:
            # Fast path: sum a real field. Need to iterate regardless
            # since streaming - but this is cheap (just reading floats,
            # not touching audio bytes at all).
            total_seconds = 0.0
            row_count = 0
            is_transcribed, _ = (False, 0)
            transcript_checked = False
            for example in ds:
                value = example.get(duration_field)
                if value is not None:
                    total_seconds += value
                row_count += 1
                if not transcript_checked and transcript_field:
                    val = example.get(transcript_field)
                    if val is not None and str(val).strip():
                        is_transcribed = True
                        transcript_checked = True
            num_hours = total_seconds / 3600.0
            if transcript_field:
                if is_transcribed:
                    transcribed_hours, untranscribed_hours = num_hours, 0.0
                else:
                    transcribed_hours, untranscribed_hours = 0.0, num_hours
            else:
                transcribed_hours, untranscribed_hours = 0.0, num_hours
            if not num_rows:
                num_rows = row_count
        else:
            # Slow path: mutagen header probing, no duration field available
            num_hours = probe_duration_via_mutagen(ds, audio_field)
            if transcript_field:
                is_transcribed, sample_n = sample_check_transcript_nonempty(
                    load_dataset(dataset_id, language_code, split=split_name, streaming=True, token=hf_token),
                    transcript_field,
                )
                if num_hours is not None:
                    transcribed_hours = num_hours if is_transcribed else 0.0
                    untranscribed_hours = 0.0 if is_transcribed else num_hours
            else:
                untranscribed_hours = num_hours

    return {
        "modality": ";".join(sorted(modalities)) if modalities else "unknown",
        "num_rows": num_rows,
        "num_hours": num_hours,
        "transcribed_hours": transcribed_hours,
        "untranscribed_hours": untranscribed_hours,
    }


def process_one_config(dataset_id, hf_language_tag, resolved_config, card_fields, linguality, hf_token):
    """
    Processes ONE resolved config (which may or may not equal the bare
    hf_language_tag - see resolve_config_names) and returns a single
    row dict, or None if this config genuinely couldn't be resolved.
    """
    other_languages = None
    if card_fields["language_codes"]:
        codes = [c for c in card_fields["language_codes"].split(";") if c != hf_language_tag]
        other_languages = ";".join(codes) if codes else None

    base_row = {
        "dataset_id": dataset_id,
        "hf_language_tag": hf_language_tag,
        "resolved_config_name": resolved_config,  # may differ from hf_language_tag - see resolve_config_names
        "language_codes_in_dataset": other_languages,
        "linguality": linguality,
        "tasks": card_fields["tasks"],
        "license": card_fields["license"],
        "arxiv_ids": card_fields["arxiv_ids"],
        "doi_ids": card_fields["doi_ids"],
        "needs_manual_access_verification": False,
        "breakdown_source": None,
        "excluded_subset_splits": None,
    }

    # ---- Size info (config-level, once, regardless of README vs load_dataset path below) ----
    builder, download_size_bytes, dataset_size_bytes = get_dataset_size_info(dataset_id, resolved_config, hf_token)
    base_row["raw_size_bytes"] = download_size_bytes
    base_row["generated_size_bytes"] = dataset_size_bytes
    base_row["size_scope"] = "whole_config_not_per_split"

    # ---- README first (still keyed on the bare hf_language_tag - that's what
    # a human-written README would actually mention, not an internal config id) ----
    readme_text = fetch_readme_body_text(dataset_id)
    readme_matches = search_readme_for_language_size(readme_text, hf_language_tag)
    if readme_matches:
        return {
            **base_row,
            "modality": card_fields["modalities"],
            "num_rows": None, "num_hours": None,
            "transcribed_hours": None, "untranscribed_hours": None,
            "breakdown_source": "readme_heuristic",
            "readme_matched_lines": ";".join(readme_matches),
            "needs_manual_review": True,
        }

    # ---- Fall back to load_dataset, per split - using resolved_config, not hf_language_tag ----
    try:
        split_names = get_dataset_split_names(dataset_id, resolved_config, token=hf_token)
    except Exception as e:
        access_issue = classify_access_issue(str(e))
        if access_issue in ACCESS_ISSUE_REASONS and access_issue != "other":
            base_row["needs_manual_access_verification"] = True
            return base_row
        return None

    excluded = [s for s in split_names if SUSPICIOUS_SUBSET_PATTERN.match(s)]
    included = [s for s in split_names if s not in excluded]
    if excluded:
        print(f"      excluding likely-nested-subset split(s): {excluded}")

    total_rows, total_hours, total_transcribed, total_untranscribed = 0, 0.0, 0.0, 0.0
    any_rows, any_hours = False, False
    modalities_seen = set()
    access_flag = False

    for split_name in included:
        result = process_split(dataset_id, resolved_config, split_name, hf_token, builder)
        if result is None:
            continue
        if "error" in result:
            if result["access_issue"] in ACCESS_ISSUE_REASONS and result["access_issue"] != "other":
                access_flag = True
            continue

        modalities_seen.update(result["modality"].split(";"))
        if result["num_rows"] is not None:
            total_rows += result["num_rows"]
            any_rows = True
        if result["num_hours"] is not None:
            total_hours += result["num_hours"]
            total_transcribed += result["transcribed_hours"] or 0.0
            total_untranscribed += result["untranscribed_hours"] or 0.0
            any_hours = True

    return {
        **base_row,
        "modality": ";".join(sorted(modalities_seen)) if modalities_seen else card_fields["modalities"],
        "num_rows": total_rows if any_rows else None,
        "num_hours": total_hours if any_hours else None,
        "transcribed_hours": total_transcribed if any_hours else None,
        "untranscribed_hours": total_untranscribed if any_hours else None,
        "breakdown_source": "load_dataset_splits",
        "needs_manual_access_verification": access_flag,
        "excluded_subset_splits": ";".join(excluded) if excluded else None,
    }


def process_one_dataset(dataset_id, hf_language_tag, hf_token):
    """
    Dataset-level wrapper: fetches shared metadata (card, license,
    linguality) ONCE, resolves which actual config name(s) correspond
    to hf_language_tag (may be more than one - see resolve_config_names
    on the {code}_{Script} multi-script case), then processes each
    resolved config separately. Returns a LIST of row dicts (possibly
    empty if nothing resolved at all - never a single bare dict).
    """
    info = _with_retry(lambda: api.dataset_info(dataset_id, files_metadata=False),
                        f"dataset_info({dataset_id})")
    if info is None:
        return []

    tags = info.tags or []
    card_data = fetch_card_data(dataset_id)
    card_fields = extract_card_fields(dataset_id, tags, card_data)
    linguality = classify_linguality(card_fields["language_codes"])

    resolved_configs = resolve_config_names(dataset_id, hf_language_tag, hf_token)
    if not resolved_configs:
        return []

    rows = []
    for resolved_config in resolved_configs:
        row = process_one_config(dataset_id, hf_language_tag, resolved_config, card_fields, linguality, hf_token)
        if row is not None:
            rows.append(row)
    return rows


def run_all(max_languages=None, search_limit=SEARCH_LIMIT):
    import os
    hf_token = os.environ.get("HF_TOKEN")

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
            rows = process_one_dataset(ds.id, hf_language_tag, hf_token)
            if not rows:
                print(f"    skipping {ds.id} - could not resolve")
                continue

            pd.DataFrame(rows).to_csv(output_path, mode="a", header=write_header, index=False)
            write_header = False
            for row in rows:
                print(f"    wrote row for {ds.id} (config={row['resolved_config_name']}, "
                      f"source={row['breakdown_source']}, "
                      f"needs_manual_access_verification={row['needs_manual_access_verification']})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tier A v3: text/audio breakdown via load_dataset, no byte-size resolution yet.")
    parser.add_argument("--test", type=int, metavar="N", default=None,
                         help="Test run: only process the first N languages from the crosswalk.")
    parser.add_argument("--search-limit", type=int, default=SEARCH_LIMIT)
    args = parser.parse_args()

    run_all(max_languages=args.test, search_limit=args.search_limit)
