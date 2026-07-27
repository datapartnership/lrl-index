"""
tier_c_fw2_token_ratio.py

per-LANGUAGE calibration: every language in your curated
set gets its own sample and its own ratios.

--------------------------------------------------------------------
TWO DIFFERENT METRICS, BOTH FROM THE SAME SAMPLE
--------------------------------------------------------------------
1. tokens_per_byte / bytes_per_token - the actual multiplier used
   downstream in tier_c_tokens.py's
   tokens = raw_bytes x yield x tokens_per_byte. Raw CC data is
   measured in bytes, so this is the unit-correct conversion factor.
2. tokens_per_word (fertility) - tokens produced per WORD (whitespace split),
   a standard tokenizer-fertility metric and a much
   more direct signal of morphological complexity than bytes_per_token
   alone (an agglutinative language can have totally normal
   bytes_per_token while still fragmenting individual words into many
   more subword pieces than an analytic language would). This does NOT
   feed the final token count, since "words" aren't a unit raw_bytes can be
   converted into without knowing bytes-per-word too.

Both are computed from the same tokenized sample - no extra cost to
reporting both.

--------------------------------------------------------------------
CAVEAT: WORD COUNTING BREAKS FOR NON-SPACE-DELIMITED SCRIPTS
--------------------------------------------------------------------
tokens_per_word uses a text.split() (whitespace) word count.
This is meaningless for scripts that don't delimit words with spaces
(Japanese, Thai, Khmer, Lao, Myanmar, Han-derived scripts) - "word"
isn't even a well-defined unit there without a real segmenter, which
this script does NOT run. Every row is flagged via
whitespace_delimited_script; tokens_per_word for a False row should be
treated as unreliable / not meaningfully comparable to a True row's
figure. bytes_per_token is NOT affected by this caveat (it doesn't
depend on word boundaries) and remains valid for these languages.

--------------------------------------------------------------------
MULTI-SCRIPT LANGUAGES
--------------------------------------------------------------------
A language with more than one script subset in FineWeb-2 (e.g. ben,
hin, kan - see script_count in your yield output) has ITS candidate
pool gathered across ALL of its subsets, dominant (largest
utf8_bytes) subset first, until the pool target is reached or every
subset is exhausted. This means the sample can mix scripts for these
languages - see subsets_used in the output to see exactly which
contributed.

--------------------------------------------------------------------
DEPENDENCIES / SETUP
--------------------------------------------------------------------
pip install pandas requests transformers sentencepiece
export HF_TOKEN=hf_...   REQUIRED (Datasets Server rate limits +
                          gated Gemma tokenizer - see prior script)

--------------------------------------------------------------------
RUNTIME NOTE
--------------------------------------------------------------------
This runs per-LANGUAGE (~122 in curated set) instead of per-
SCRIPT (~26) - roughly 4-5x the API calls. If you hit rate limiting even
with HF_TOKEN set, raise BASE_BACKOFF_SECONDS or add a manual sleep
between languages.
"""
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# ---- CONFIG ----
FW2_LANG_DIST_CSV = "https://raw.githubusercontent.com/huggingface/fineweb-2/main/fineweb2-language-distribution.csv"
FW2_DATASET_ID = "HuggingFaceFW/fineweb-2"
DATASETS_SERVER_FILTER_URL = "https://datasets-server.huggingface.co/filter"
DATASETS_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"

YIELD_OUTPUT_FILE = "../data/crawl/processed/yield_multiplier_by_language.csv"  # restricts which codes are in scope

GEMMA_MODEL_ID = "google/gemma-2-9b"  # gated - see module docstring

TOP_N_PER_LANGUAGE = 25
POOL_TARGET_SIZE = 100
LANGUAGE_SCORE_THRESHOLDS = [0.99, 0.95, 0.90, 0.75]
MAX_ROWS_PER_API_CALL = 100
MAX_PAGES_PER_THRESHOLD = 3

# Scripts where whitespace does NOT reliably delimit words - tokens_per_word
# is flagged unreliable for languages using any of these. Not exhaustive -
# add to this if your curated set includes others you know of.
NON_WHITESPACE_DELIMITED_SCRIPTS = {
    "Jpan", "Hani", "Hans", "Hant", "Thai", "Khmr", "Laoo", "Mymr",
}

OUTPUT_FILE = "../data/crawl/processed/tier_c_token_ratio_by_language.csv"
SKIPPED_LOG_FILE = "../data/crawl/processed/tier_c_token_ratio_skipped_languages.csv"

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT = 30

HF_TOKEN_HEADER = {}


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


# ---- STEP 1: curated languages -> their FineWeb-2 subset(s) ----
def load_yield_language_codes():
    df = pd.read_csv(YIELD_OUTPUT_FILE)
    codes = list(df["code"].unique())
    print(f"{len(codes)} language code(s) loaded from {YIELD_OUTPUT_FILE}")
    return codes


def load_fineweb2_distribution():
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
    return df


def get_language_subset_map(fw2_dist, codes):
    """
    For each code, returns its list of (subset, script, utf8_bytes)
    tuples sorted by utf8_bytes DESCENDING (dominant subset first -
    see module docstring on multi-script pooling order). Returns a
    dict {code: [(subset, script, utf8_bytes), ...]}. Codes with no
    matching subset at all are simply absent from the dict (caller
    logs these as skipped).
    """
    scoped = fw2_dist[fw2_dist["code"].isin(codes)]
    out = {}
    for code, group in scoped.groupby("code"):
        rows = group.sort_values("utf8_bytes", ascending=False)
        out[code] = list(zip(rows["subset"], rows["script"], rows["utf8_bytes"]))
    return out


# ---- STEP 2: candidate pool + top-N ranking (per subset, then combined across a language's subsets) ----
def fetch_rows_page(subset, offset, length, threshold=None, split="train"):
    def _fetch():
        if threshold is not None:
            params = {
                "dataset": FW2_DATASET_ID, "config": subset, "split": split,
                "where": f'"language_score">={threshold}',
                "offset": offset, "length": length,
            }
            url = DATASETS_SERVER_FILTER_URL
        else:
            params = {"dataset": FW2_DATASET_ID, "config": subset, "split": split,
                      "offset": offset, "length": length}
            url = DATASETS_SERVER_ROWS_URL

        resp = requests.get(url, params=params, headers=HF_TOKEN_HEADER, timeout=REQUEST_TIMEOUT)
        if resp.status_code in (404, 500):
            return None
        resp.raise_for_status()
        return resp.json()

    desc = f"fetch {'filtered' if threshold else 'unfiltered'} rows for {subset} (offset={offset})"
    return _with_retry(_fetch, desc)


def _extract_text_and_score(rows):
    out = []
    for r in rows:
        row = r.get("row", {})
        if "text" in row and "language_score" in row and row["language_score"] is not None:
            out.append((row["text"], row["language_score"]))
    return out


def build_candidate_pool_for_subset(subset, pool_target):
    """Same threshold-ladder + fallback logic as before, scoped to
    ONE subset. Returns (pool: list[(text, score)], threshold_used)."""
    for threshold in LANGUAGE_SCORE_THRESHOLDS:
        pool = []
        for page in range(MAX_PAGES_PER_THRESHOLD):
            data = fetch_rows_page(subset, offset=page * MAX_ROWS_PER_API_CALL,
                                    length=MAX_ROWS_PER_API_CALL, threshold=threshold)
            if data is None:
                break
            rows = data.get("rows", [])
            pool.extend(_extract_text_and_score(rows))
            if len(rows) < MAX_ROWS_PER_API_CALL:
                break
        if len(pool) >= pool_target:
            return pool, threshold

    pool = []
    for page in range(MAX_PAGES_PER_THRESHOLD):
        data = fetch_rows_page(subset, offset=page * MAX_ROWS_PER_API_CALL,
                                length=MAX_ROWS_PER_API_CALL, threshold=None)
        if data is None:
            break
        rows = data.get("rows", [])
        pool.extend(_extract_text_and_score(rows))
        if len(rows) < MAX_ROWS_PER_API_CALL:
            break
    return pool, None


def build_candidate_pool_for_language(subsets, pool_target=POOL_TARGET_SIZE):
    """
    Gathers candidates across ALL of a language's subsets, dominant
    first (subsets is already sorted by utf8_bytes descending - see
    get_language_subset_map), stopping as soon as the combined pool
    reaches pool_target. Returns (combined_pool, subsets_used: list of
    (subset, threshold_used) for whichever subsets actually contributed).
    """
    combined_pool = []
    subsets_used = []
    for subset, script, _ in subsets:
        if len(combined_pool) >= pool_target:
            break
        remaining = pool_target - len(combined_pool)
        pool, threshold_used = build_candidate_pool_for_subset(subset, pool_target=remaining)
        if pool:
            combined_pool.extend(pool)
            subsets_used.append((subset, script, threshold_used, len(pool)))
        if len(subsets) > 1:
            print(f"    subset {subset}: +{len(pool)} candidate(s) (threshold={threshold_used}), "
                  f"pool now {len(combined_pool)}/{pool_target}")
    return combined_pool, subsets_used


def sample_top_n_by_score(subsets, top_n=TOP_N_PER_LANGUAGE, pool_target=POOL_TARGET_SIZE):
    pool, subsets_used = build_candidate_pool_for_language(subsets, pool_target)
    if not pool:
        return [], subsets_used, 0, False, None, None

    pool.sort(key=lambda pair: pair[1], reverse=True)
    top = pool[:top_n]
    texts = [t for t, _ in top]
    scores = [s for _, s in top]
    return texts, subsets_used, len(pool), len(pool) >= pool_target, min(scores), max(scores)


# ---- STEP 3: tokenization + word count ----
def load_gemma_tokenizer():
    from transformers import AutoTokenizer
    import os
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set - required to load the gated Gemma tokenizer. See module docstring.")
    return AutoTokenizer.from_pretrained(GEMMA_MODEL_ID, token=token)


def compute_ratios(tokenizer, texts):
    """
    Aggregate ratios (sum of totals, not average of per-doc ratios -
    see prior script's reasoning, applies equally here). Word count is
    a naive whitespace split - see module docstring caveat.
    Returns (total_bytes, total_tokens, total_words, bytes_per_token,
    tokens_per_word).
    """
    total_bytes, total_tokens, total_words = 0, 0, 0
    for text in texts:
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(tokenizer.encode(text))
        total_words += len(text.split())
    bytes_per_token = (total_bytes / total_tokens) if total_tokens else None
    tokens_per_word = (total_tokens / total_words) if total_words else None
    return total_bytes, total_tokens, total_words, bytes_per_token, tokens_per_word


def main(test_n=None, only_code=None):
    import os
    if os.environ.get("HF_TOKEN"):
        HF_TOKEN_HEADER["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    else:
        print("WARNING: HF_TOKEN not set - /filter calls will be rate-limited harder, "
              "and loading the Gemma tokenizer will fail outright. See module docstring.")

    codes = load_yield_language_codes()
    if only_code is not None:
        codes = [c for c in codes if c == only_code]
        print(f"Filtered to code == {only_code!r}: {len(codes)} language(s)")

    print("Loading FineWeb-2 language distribution...")
    fw2_dist = load_fineweb2_distribution()
    subset_map = get_language_subset_map(fw2_dist, codes)

    missing_codes = [c for c in codes if c not in subset_map]
    codes = [c for c in codes if c in subset_map]

    if test_n is not None:
        codes = codes[:test_n]
        print(f"TEST RUN: limited to first {test_n} language(s)")

    print("\nLoading Gemma tokenizer (this can take a minute the first time)...")
    tokenizer = load_gemma_tokenizer()

    result_rows = []
    skipped_rows = [{"code": c, "reason": "no_fineweb2_subset_found"} for c in missing_codes]

    for code in codes:
        subsets = subset_map[code]
        scripts_involved = sorted(set(s for _, s, _ in subsets))
        print(f"\n[{code}] {len(subsets)} subset(s) across script(s) {scripts_involved} - building pool...")

        texts, subsets_used, pool_size, reached_pool_target, min_score, max_score = \
            sample_top_n_by_score(subsets)

        if not texts:
            skipped_rows.append({"code": code, "reason": "no_documents_retrieved"})
            print(f"    skipped - could not retrieve any documents")
            continue

        total_bytes, total_tokens, total_words, bytes_per_token, tokens_per_word = \
            compute_ratios(tokenizer, texts)

        is_whitespace_delimited = not any(s in NON_WHITESPACE_DELIMITED_SCRIPTS for s in scripts_involved)

        result_rows.append({
            "code": code,
            "scripts_involved": ";".join(scripts_involved),
            "subsets_used": ";".join(f"{s}(thr={t})" for s, _, t, _ in subsets_used),
            "num_docs_selected": len(texts),
            "candidate_pool_size": pool_size,
            "reached_pool_target": reached_pool_target,
            "min_score_in_selection": min_score,
            "max_score_in_selection": max_score,
            "total_utf_8_bytes_sampled": total_bytes,
            "total_tokens_sampled": total_tokens,
            "total_words_sampled": total_words,
            "bytes_per_token": bytes_per_token,
            "tokens_per_byte": (1 / bytes_per_token) if bytes_per_token else None,
            "tokens_per_word_fertility": tokens_per_word,
            "whitespace_delimited_script": is_whitespace_delimited,
        })
        fert_str = f"{tokens_per_word:.2f}" if tokens_per_word else "n/a"
        bpt_str = f"{bytes_per_token:.3f}" if bytes_per_token else "n/a"
        print(f"    {len(texts)} doc(s) selected from pool of {pool_size} "
              f"(scores {min_score:.4f}-{max_score:.4f}) | "
              f"bytes/token={bpt_str} | fertility(tokens/word)={fert_str}"
              f"{'' if is_whitespace_delimited else '  [NOT whitespace-delimited - fertility unreliable]'}")

    if skipped_rows:
        Path(SKIPPED_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(skipped_rows).to_csv(SKIPPED_LOG_FILE, index=False)
        print(f"\nWrote {len(skipped_rows)} skipped language(s) to {SKIPPED_LOG_FILE}")
        print(pd.DataFrame(skipped_rows)["reason"].value_counts())

    if not result_rows:
        print("No languages resolved to a ratio - nothing to write.")
        return

    result_df = pd.DataFrame(result_rows)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {len(result_df)} language row(s) to {OUTPUT_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute per-language fertility + byte-to-token ratio.")
    parser.add_argument("--test", type=int, metavar="N", default=None,
                         help="Test run: only process the first N languages.")
    parser.add_argument("--code", type=str, default=None,
                         help='Only run for one ISO 639-3 language code, e.g. "jpn".')
    args = parser.parse_args()

    main(test_n=args.test, only_code=args.code)
