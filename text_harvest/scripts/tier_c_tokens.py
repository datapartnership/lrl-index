"""
fineweb2_tier_c_tokens.py

Computes final Tier C token estimates per language:

    tokens(lang) = latest_raw_cc_bytes(lang) x yield(lang) x tokens_per_byte(lang)

Joins three separate outputs already produced:
  1. tier_c_common_crawl_harvest.py's LATEST_OUTPUT_FILE - one row per
     language, from whichever crawl it most recently appeared in (NOT
     summed across crawls - uses the single latest value).
  2. fineweb2_yield_multiplier.py's OUTPUT_FILE - one row per language,
     the yield ratio.
  3. fineweb2_language_fertility.py's OUTPUT_FILE - one row per
     LANGUAGE (not script), the tokens_per_byte ratio.

--------------------------------------------------------------------
NO MORE SCRIPT-MISMATCH APPROXIMATION
--------------------------------------------------------------------
The previous version of this script joined against a SCRIPT-level
token ratio (fineweb2_token_ratio.py's output) and had to approximate
a "dominant script" for every language to bridge language-level raw
bytes/yield against script-level token ratios - an extra approximation
layer, and a fetch of FineWeb-2's full distribution CSV just to
compute it.

Now that fineweb2_language_fertility.py produces a token ratio
directly PER LANGUAGE (already correctly handling multi-script
languages internally via its own multi-subset pooling), this join is
a straightforward three-way merge on `code` across all three files.
No script mapping, no dominant-script proxying, no FineWeb-2 CSV
re-fetch needed here at all.

--------------------------------------------------------------------
WHAT GETS SKIPPED, AND WHY (see SKIPPED_LOG_FILE)
--------------------------------------------------------------------
- "missing_from_cc_data": language has a yield ratio but no row in
  the latest-CC-bytes table (shouldn't happen if yield was computed
  FROM that CC data, but checked rather than assumed).
- "missing_from_yield_data": language has CC bytes but no yield ratio
  (e.g. it was in the "not_in_cc_data" or "fw2_api_unavailable" skip
  logs from the yield step).
- "missing_from_token_ratio_data": language has raw bytes and a yield
  ratio, but no row in the per-language token ratio output (e.g. that
  step was run with --test/--code restricted to a subset of languages,
  or hit "no_documents_retrieved" there).
- "null_value_in_one_of_the_three_inputs": all three inputs were
  found, but at least one came back null rather than a usable number.

--------------------------------------------------------------------
DEPENDENCIES / SETUP
--------------------------------------------------------------------
pip install pandas
(no requests/HF_TOKEN needed anymore - this script only reads your
own local outputs, no external fetch required)
"""
from pathlib import Path

import pandas as pd

# ---- CONFIG ----
LATEST_CC_BYTES_FILE = "../data/crawl/processed/tier_c_language_bytes_latest.csv"
YIELD_FILE = "../data/crawl/processed/yield_multiplier_by_language.csv"
TOKEN_RATIO_FILE = "../data/crawl/processed/tier_c_token_ratio_by_language.csv"

OUTPUT_FILE = "../data/crawl/processed/tier_c_tokens_by_language.csv"
SKIPPED_LOG_FILE = "../data/crawl/processed/tier_c_tokens_skipped_languages.csv"


def load_latest_cc_bytes():
    df = pd.read_csv(LATEST_CC_BYTES_FILE)
    df = df.rename(columns={"language_iso_639_3": "code", "approx_language_bytes": "raw_bytes"})
    return df[["code", "raw_bytes", "crawl_id", "crawl_date"]]


def load_yield():
    df = pd.read_csv(YIELD_FILE)
    return df[["code", "yield", "script_count", "yield_coverage_fraction"]]


def load_token_ratio():
    """
    Reads fineweb2_language_fertility.py's per-language output
    directly - no script mapping step needed. Keeps the columns
    useful for judging how trustworthy a given language's ratio is
    (see fineweb2_language_fertility.py's own output-columns notes).
    """
    df = pd.read_csv(TOKEN_RATIO_FILE)
    return df[[
        "code", "tokens_per_byte", "tokens_per_word_fertility",
        "scripts_involved", "subsets_used",
        "num_docs_selected", "candidate_pool_size", "reached_pool_target",
        "min_score_in_selection", "max_score_in_selection",
        "whitespace_delimited_script",
    ]]


def main():
    cc_bytes = load_latest_cc_bytes()
    yield_df = load_yield()
    token_ratio = load_token_ratio()

    all_codes = set(cc_bytes["code"]) | set(yield_df["code"])
    skipped_rows = []
    result_rows = []

    for code in sorted(all_codes):
        cc_row = cc_bytes[cc_bytes["code"] == code]
        yield_row = yield_df[yield_df["code"] == code]

        if cc_row.empty:
            skipped_rows.append({"code": code, "reason": "missing_from_cc_data"})
            continue
        if yield_row.empty:
            skipped_rows.append({"code": code, "reason": "missing_from_yield_data"})
            continue

        ratio_row = token_ratio[token_ratio["code"] == code]
        if ratio_row.empty:
            skipped_rows.append({"code": code, "reason": "missing_from_token_ratio_data"})
            continue

        raw_bytes = cc_row.iloc[0]["raw_bytes"]
        yield_ratio = yield_row.iloc[0]["yield"]
        tokens_per_byte = ratio_row.iloc[0]["tokens_per_byte"]

        if pd.isna(raw_bytes) or pd.isna(yield_ratio) or pd.isna(tokens_per_byte):
            skipped_rows.append({"code": code, "reason": "null_value_in_one_of_the_three_inputs"})
            continue

        estimated_tokens = raw_bytes * yield_ratio * tokens_per_byte

        result_rows.append({
            "code": code,
            "estimated_tokens": estimated_tokens,
            "raw_bytes_latest_crawl": raw_bytes,
            "crawl_id_used": cc_row.iloc[0]["crawl_id"],
            "crawl_date": cc_row.iloc[0]["crawl_date"],
            "yield": yield_ratio,
            "yield_coverage_fraction": yield_row.iloc[0]["yield_coverage_fraction"],
            "tokens_per_byte": tokens_per_byte,
            "tokens_per_word_fertility": ratio_row.iloc[0]["tokens_per_word_fertility"],
            "scripts_involved": ratio_row.iloc[0]["scripts_involved"],
            "whitespace_delimited_script": ratio_row.iloc[0]["whitespace_delimited_script"],
            "token_ratio_sample_size": ratio_row.iloc[0]["num_docs_selected"],
            "token_ratio_pool_size": ratio_row.iloc[0]["candidate_pool_size"],
            "token_ratio_reached_pool_target": ratio_row.iloc[0]["reached_pool_target"],
            "token_ratio_min_score": ratio_row.iloc[0]["min_score_in_selection"],
            "token_ratio_max_score": ratio_row.iloc[0]["max_score_in_selection"],
        })

    if skipped_rows:
        Path(SKIPPED_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(skipped_rows).to_csv(SKIPPED_LOG_FILE, index=False)
        print(f"Wrote {len(skipped_rows)} skipped language(s) to {SKIPPED_LOG_FILE}")
        print(pd.DataFrame(skipped_rows)["reason"].value_counts())

    if not result_rows:
        print("No languages resolved to a token estimate - nothing to write.")
        return

    result_df = pd.DataFrame(result_rows).sort_values("estimated_tokens", ascending=False)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {len(result_df)} language row(s) to {OUTPUT_FILE}")
    print(f"\n{result_df[['code', 'estimated_tokens', 'scripts_involved']].head(10).to_string(index=False)}")


if __name__ == "__main__":
    main()
