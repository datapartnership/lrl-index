"""
tier_a_harvest_all_languages.py

Location: lrl-index/text_harvest/scripts/tier_a_harvest_all_languages.py
(sibling to tier_a_harvest_single_language.py, which it imports from)

Runs the Tier A text harvest across every HF query code in the
crosswalk (lrl-index/crosswalk/data/processed/full_language_reference.csv),
reusing all per-dataset logic from tier_a_harvest_single_language.py
unchanged. This wrapper only adds what's needed to go from one
sample language to ~7,900 languages safely:

  - Queries every code in hf_tag independently. hf_tag can hold more
    than one HF query code per language (e.g. "aar;aa" for a language
    matched on both iso3 and iso1) - each is queried and treated as
    its own harvest pass. NO crosswalk to iso_639_3 happens here: each
    output row is tagged with hf_language_tag, the literal code that
    was queried, not a canonical ISO code. That mapping is deferred to
    a separate downstream step (crosswalk join happens later, outside
    this script).
  - Skips codes with hf_match_status == "no_match" or a missing
    hf_tag - logged to a separate file, not silently dropped, since a
    future crosswalk update could resolve them.
  - Caches dataset_info() by dataset_id across the ENTIRE run (not
    per-code), since a multilingual dataset tagged with dozens of
    languages would otherwise be re-fetched once per code that hits it.
  - Retries transient API errors with exponential backoff, for both
    the search call and the detailed-info call.
  - Writes incrementally to one master CSV (append per code) and
    skips codes already present in that file on restart, so a crash
    partway through a multi-hour run doesn't lose completed work.

Does not change: modality detection, license extraction, size
resolution, provenance detection, or the definition of what counts as
excluded modality. All of that stays exactly as validated in the
single-language script. Size resolution's filename segment matching
still uses whichever literal code was queried (see harvest_one_code),
since that's what actually appears in dataset file paths - not a
canonical form.

Field naming matches the current single-language script: the
language-specific size figure is total_lang_size_bytes/
total_lang_size_method (renamed from total_size_bytes/total_size_method
to disambiguate from the new dataset_total_size_bytes/
dataset_total_size_method fields, which record the whole dataset's
size across all languages and modalities, unfiltered - see
get_dataset_total_size() in the single-language script).
"""
import os
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from tier_a_huggingface_harvest import (
    classify_linguality,
    detect_modality,
    is_excluded_modality,
    extract_license,
    get_language_specific_size,
    get_dataset_total_size,
    detect_provenance,
    extract_publication_link,
    extract_multilinguality_tag,
    needs_review,
    SEARCH_LIMIT,
)

# Authenticated requests get a much higher HF rate limit than the
# shared-IP anonymous limit (500 req/300s), which a full-crosswalk run
# blows through fast. Reads HF_TOKEN from the environment - set it
# with `export HF_TOKEN=hf_...` before running (get a free token at
# https://huggingface.co/settings/tokens). Falls back to anonymous
# access if unset, but expect frequent 429s at that tier.
api = HfApi(token=os.environ.get("HF_TOKEN"))

# ---- CONFIG ----
CROSSWALK_FILE = "../../crosswalk/data/processed/full_language_reference.csv"
OUTPUT_FILE = "../data/harvest/tier_a_harvest_all_languages.csv"
SKIPPED_LOG = "../data/harvest/tier_a_skipped_languages.csv"
MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 5
MAX_WAIT_SECONDS = 300  # safety cap even if a Retry-After header asks for longer
RATE_LIMIT_FALLBACK_WAIT_SECONDS = 300  # HF's advertised window size (see docstring below)

_dataset_info_cache = {}  # dataset_id -> HfApi.dataset_info() result, shared across all languages


def _get_wait_seconds(exception, attempt):
    """
    Determines how long to sleep before retrying.

    HF's 429 responses are NOT consistent in shape - sometimes the
    error includes an explicit "Retry after N seconds" in the message
    body (or a Retry-After header), sometimes it's just a bare
    "429 Client Error: Too Many Requests..." with neither. Relying
    only on parsing means real rate-limit hits silently fall through
    to a short exponential guess (5s, 10s, 20s...) that's nowhere near
    HF's actual ~300-second rate-limit window - every retry just fails
    again for no visible reason.

    Resolution order:
      1. Explicit Retry-After header, if present -> use it exactly.
      2. "Retry after N seconds" parsed from the error body, if
         present -> use it exactly.
      3. If the failure was confirmed a 429 (rate limit) but neither
         of the above was parseable, fall back to
         RATE_LIMIT_FALLBACK_WAIT_SECONDS (HF's documented window
         size), NOT a short exponential guess - a 429 with no
         explicit wait time still means "you're rate limited," and a
         5s guess is actively wrong information to act on.
      4. Only for non-429 failures (network blips, timeouts, etc.)
         does this fall back to a short exponential backoff, since
         those are more likely to be transient and unrelated to a
         rate-limit window.
    Always capped at MAX_WAIT_SECONDS.
    """
    response = getattr(exception, "response", None)
    if response is not None:
        header_value = response.headers.get("Retry-After") or response.headers.get("retry-after")
        if header_value:
            try:
                return min(float(header_value) + 2, MAX_WAIT_SECONDS)  # small buffer past the deadline
            except ValueError:
                pass

    # HF's 429 responses put "Retry after N seconds" in the error body
    # even when no Retry-After header is present - parse it from there
    # as a fallback before resorting to a blind exponential guess.
    import re
    match = re.search(r"Retry after (\d+) seconds", str(exception))
    if match:
        return min(int(match.group(1)) + 2, MAX_WAIT_SECONDS)

    # Confirmed 429 (rate limit) but neither header nor body gave an
    # explicit wait time - this still means "you are rate limited,"
    # so use HF's documented window size rather than a short
    # exponential guess that would just fail again immediately.
    status_code = getattr(response, "status_code", None)
    if status_code == 429 or "429" in str(exception):
        return min(RATE_LIMIT_FALLBACK_WAIT_SECONDS, MAX_WAIT_SECONDS)

    # Non-429 failure (network blip, timeout, etc.) - short exponential
    # backoff is reasonable here since these are more likely transient
    # and unrelated to a rate-limit window.
    return min(BASE_BACKOFF_SECONDS * (2 ** attempt), MAX_WAIT_SECONDS)


def _dataset_info_with_retry(dataset_id):
    """Cached, retrying wrapper around api.dataset_info(). Returns
    None (never raises) if all retries are exhausted, so one stuck
    dataset can't kill a multi-hour run - the caller skips it and
    moves on."""
    if dataset_id in _dataset_info_cache:
        return _dataset_info_cache[dataset_id]

    for attempt in range(MAX_RETRIES):
        try:
            info = api.dataset_info(dataset_id, files_metadata=True)
            _dataset_info_cache[dataset_id] = info
            return info
        except Exception as e:
            wait = _get_wait_seconds(e, attempt)
            print(f"    [retry {attempt + 1}/{MAX_RETRIES}] dataset_info({dataset_id}): {e} - waiting {wait:.0f}s")
            time.sleep(wait)

    print(f"    giving up on dataset_info({dataset_id}) after {MAX_RETRIES} retries")
    return None


def _list_datasets_with_retry(query_code, limit):
    """
    Retrying wrapper around api.list_datasets(). Returns a tuple
    (results, search_failed):
      - (list_of_datasets, False) on success, even if that list is
        genuinely empty (0 real datasets found for this code).
      - ([], True) if all retries were exhausted without a real
        answer. This is NOT the same as a genuine zero-result search -
        callers must check search_failed and treat it as "unknown,
        needs retry," never as "confirmed zero," or a rate-limited
        code silently gets recorded as having no data forever.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return list(api.list_datasets(language=query_code, limit=limit)), False
        except Exception as e:
            wait = _get_wait_seconds(e, attempt)
            print(f"    [retry {attempt + 1}/{MAX_RETRIES}] search '{query_code}': {e} - waiting {wait:.0f}s")
            time.sleep(wait)

    print(f"    giving up on search '{query_code}' after {MAX_RETRIES} retries - "
          f"NOT recording as zero-result, will retry on next run")
    return [], True


def harvest_one_code(hf_language_tag, limit=SEARCH_LIMIT):
    """
    Same per-dataset extraction as harvest_language() in the single-
    language script, run for exactly one literal HF query code (no
    merging across a language's multiple codes, no crosswalk to
    iso_639_3 - that happens downstream, outside this script).

    Each output row is stamped with hf_language_tag, the exact string
    that was queried. If the same underlying dataset also gets pulled
    in by a different code for the "same" language (e.g. both "aar"
    and "aa"), it will appear twice in the output - once per code -
    which is intentional here so the crosswalk step downstream has
    the full matched_query_codes-equivalent information per row to
    work with, rather than this script guessing at equivalence.

    Returns (DataFrame, search_failed). search_failed=True means the
    search itself never succeeded (e.g. rate-limited past all
    retries) - the caller must NOT write this to the output file as a
    zero-result row, since that would falsely and permanently mark
    the code as "done, zero datasets" when the truth is "unknown."
    """
    search_results, search_failed = _list_datasets_with_retry(hf_language_tag, limit)
    print(f"[{hf_language_tag}] {len(search_results)} dataset(s) found in search")

    rows = []
    excluded_count = 0

    for ds in search_results:
        full_info = _dataset_info_with_retry(ds.id)
        if full_info is None:
            continue  # already logged by the retry helper

        tags = full_info.tags or []
        language_tags = [t for t in tags if t.startswith("language:")]
        linguality = classify_linguality(language_tags)
        modality_value, modality_method = detect_modality(tags, full_info.siblings)

        if is_excluded_modality(modality_value):
            excluded_count += 1
            continue

        license_value = extract_license(tags)
        # Segment matching uses the literal queried code, since that's
        # what actually appears in dataset file paths (Wikipedia/OPUS-
        # style per-language folders use whatever code the dataset
        # author picked, not a canonical iso_639_3 form).
        size_result = get_language_specific_size(full_info.siblings, hf_language_tag, linguality)
        # Tracking-only field: text/unclear-modality portion of the
        # repo size, unfiltered by language (see get_dataset_total_size
        # in the single-language script for the exact file-level
        # scope - confirmed audio/image files are excluded).
        # dataset_info() is called unconditionally above for every
        # dataset in this wrapper (unlike the single-language script,
        # which skips it for confirmed-excluded modalities), so
        # full_info.siblings is always populated here.
        dataset_total_size_bytes, dataset_total_size_method = get_dataset_total_size(full_info.siblings)
        provenance = detect_provenance(tags, getattr(full_info, "description", None))
        publication_link = extract_publication_link(tags) if provenance == "provenance_unknown" else None
        multilinguality_tag = extract_multilinguality_tag(tags)

        rows.append({
            "dataset_id": ds.id,
            "hf_language_tag": hf_language_tag,
            "num_languages_in_dataset": len(language_tags),
            "linguality": linguality,
            "multilinguality_tag": multilinguality_tag,
            "modality": modality_value,
            "modality_method": modality_method,
            "license": license_value,
            "dataset_total_size_bytes": dataset_total_size_bytes,
            "dataset_total_size_method": dataset_total_size_method,
            "total_lang_size_bytes": size_result["total_size_bytes"],
            "total_lang_size_method": size_result["total_size_method"],
            "text_size_bytes": size_result["text_size_bytes"],
            "text_size_method": size_result["text_size_method"],
            "provenance": provenance,
            "publication_link": publication_link,
            "synthetic_flag": provenance == "machine-generated",
            "needs_review": needs_review(provenance),
            "retrieval_method": "huggingface_hub_api",
        })

    print(f"    excluded {excluded_count} audio/image-only dataset(s)")
    return pd.DataFrame(rows), search_failed


def run_all(crosswalk_path=CROSSWALK_FILE, output_path=OUTPUT_FILE,
            skipped_log_path=SKIPPED_LOG, limit=SEARCH_LIMIT, max_languages=None):
    """
    max_languages: if set, only the FIRST N ROWS of the crosswalk CSV
    are used (i.e. first N languages, not N query codes - a language
    with a multi-code hf_tag like "aar;aa" still only counts as one
    row/one language toward this limit, even though it produces two
    query codes). Intended for test runs - leave as None for a full run.
    """
    crosswalk = pd.read_csv(crosswalk_path)
    if max_languages is not None:
        crosswalk = crosswalk.head(max_languages)
        print(f"TEST RUN: limited to first {max_languages} language row(s) from {crosswalk_path}")

    output_file = Path(output_path)
    already_done = set()
    if output_file.exists():
        prior = pd.read_csv(output_path)
        already_done = set(prior["hf_language_tag"].dropna().unique())
        print(f"Resuming: {len(already_done)} code(s) already in {output_path}, skipping those")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_file.exists()
    skipped_rows = []

    # Flatten hf_tag into one query code per row up front, so each
    # code is its own unit of work (and its own resume checkpoint) -
    # not nested inside a per-language loop.
    query_codes_seen = set()
    for _, row in crosswalk.iterrows():
        if row["hf_match_status"] == "no_match" or pd.isna(row["hf_tag"]):
            skipped_rows.append({
                "iso_639_3": row.get("iso_639_3"), "language_name": row.get("language_name"),
                "reason": "no_match_in_crosswalk",
            })
            continue

        for code in [c.strip() for c in row["hf_tag"].split(";") if c.strip()]:
            query_codes_seen.add(code)

    remaining_codes = [c for c in sorted(query_codes_seen) if c not in already_done]
    total_to_run = len(remaining_codes)
    print(f"\n{len(query_codes_seen)} total code(s) in crosswalk, "
          f"{len(already_done)} already done, {total_to_run} remaining\n")

    start_time = time.time()
    failed_codes = []

    for i, hf_language_tag in enumerate(remaining_codes, start=1):
        df, search_failed = harvest_one_code(hf_language_tag, limit=limit)

        if search_failed:
            # Do NOT write anything to output for this code - it must
            # stay absent from already_done so a future run retries it
            # for real, instead of treating a rate-limited failure as
            # a confirmed zero-result language forever.
            failed_codes.append(hf_language_tag)
            print(f"    [{hf_language_tag}] search failed after retries - "
                  f"skipping write, will retry on next run")
        else:
            if df.empty:
                # Explicit zero-result marker row, so "not yet run"
                # (absent from output) and "run, genuinely found
                # nothing" (present, null dataset_id) stay
                # distinguishable on resume/audit. Only reached when
                # search_failed is False, i.e. this is a REAL zero,
                # not a disguised failure.
                df = pd.DataFrame([{
                    "dataset_id": None, "hf_language_tag": hf_language_tag,
                    "num_languages_in_dataset": None, "linguality": None,
                    "multilinguality_tag": None, "modality": None, "modality_method": None,
                    "license": None, "dataset_total_size_bytes": None, "dataset_total_size_method": None,
                    "total_lang_size_bytes": None, "total_lang_size_method": None,
                    "text_size_bytes": None, "text_size_method": None, "provenance": None,
                    "publication_link": None, "synthetic_flag": None, "needs_review": None,
                    "retrieval_method": "huggingface_hub_api_zero_results",
                }])

            df.to_csv(output_path, mode="a", header=write_header, index=False)
            write_header = False

        # Progress summary every 25 codes (not every code - that would
        # flood the log) - elapsed/rate/ETA, so a `tail -f` on the log
        # file gives a real sense of how far along a multi-hour run is,
        # not just a stream of per-code dataset counts.
        if i % 25 == 0 or i == total_to_run:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = total_to_run - i
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            print(f"\n--- progress: {i}/{total_to_run} codes done this run "
                  f"({rate:.2f} codes/sec, elapsed {elapsed / 60:.1f} min, "
                  f"ETA {eta_seconds / 60:.1f} min, {len(failed_codes)} failed so far) ---\n")

    if failed_codes:
        print(f"\n{len(failed_codes)} code(s) failed after retries and were NOT written "
              f"to output - re-run this script to retry them: {failed_codes}")

    if skipped_rows:
        pd.DataFrame(skipped_rows).drop_duplicates().to_csv(skipped_log_path, index=False)
        print(f"\n{len(skipped_rows)} language(s) skipped (no hf_tag) - logged to {skipped_log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tier A Hugging Face text harvest across the language crosswalk.")
    parser.add_argument(
        "--test", type=int, metavar="N", default=None,
        help="Test run: only process the first N languages from the crosswalk "
             "(writes to a separate tier_a_harvest_test.csv / tier_a_skipped_test.csv "
             "so it never mixes with a full run's output).",
    )
    parser.add_argument(
        "--search-limit", type=int, default=None,
        help="Cap datasets fetched per query code (default: no cap, fetch everything). "
             "Useful to keep a --test run fast.",
    )
    args = parser.parse_args()

    if args.test:
        run_all(
            output_path="../data/harvest/tier_a_harvest_test.csv",
            skipped_log_path="../data/harvest/tier_a_skipped_test.csv",
            max_languages=args.test,
            limit=args.search_limit,
        )
    else:
        run_all(limit=args.search_limit if args.search_limit is not None else SEARCH_LIMIT)
