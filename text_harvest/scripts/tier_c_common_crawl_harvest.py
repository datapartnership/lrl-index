"""
tier_c_common_crawl_harvest.py

Tier C harvest: estimates raw bytes per language, per Common Crawl
monthly archive, aggregated into a cumulative total across all crawls.

Location: lrl-index/crawl_harvest/scripts/tier_c_common_crawl_harvest.py
(suggested - adjust paths below to match wherever this actually lives)

--------------------------------------------------------------------
WHY THIS IS AN ESTIMATE, NOT AN EXACT FIGURE
--------------------------------------------------------------------
Common Crawl does not publish an exact "bytes of text for language X"
figure anywhere. What IS published, per crawl, is each language's
SHARE OF PAGES (see below) and the crawl's total bytes downloaded
(see further below). This script combines the two:

    approx_bytes(lang, crawl) = page_share(lang, crawl) * total_bytes_downloaded(crawl)

This assumes average page size is roughly uniform across languages
within a crawl, which is not exactly true (verbose scripts, heavier
markup, etc. will skew this somewhat) - it is a scoping-level
approximation, not a precise measurement. This same approach is used
in published work extracting low-resource languages from Common
Crawl (Tessema et al., "UnifiedCrawl", arXiv:2411.14343), which
validated the approximation against an actual filtered extraction
and found it consistent.

--------------------------------------------------------------------
HOW LANGUAGE PAGE SHARES ARE OBTAINED
--------------------------------------------------------------------
Pulled directly from plots/languages.csv in the cc-crawl-statistics
GitHub repo (the exact source file Common Crawl uses to generate the
language percentages shown on their own published statistics site) -
fetched here via raw.githubusercontent.com. Columns: crawl,
primary_language, pages, urls, %pages/crawl.

The CSV's <unknown> row (pages where CLD2 could not
determine a language) is included as its own proper row - it is NOT
in this script's per-language output (see run_all()), since it isn't
a real language, but it correctly remains part of the page-share
denominator.

--------------------------------------------------------------------
HOW TOTAL CRAWL BYTES ARE OBTAINED
--------------------------------------------------------------------
Common Crawl's cc-crawl-statistics project publishes a per-crawl
crawler operational-metrics file: stats/crawler/{crawl_id}.json,
hosted in the same GitHub repo (fetched here via
raw.githubusercontent.com). It contains the key
["crawl_status", "fetcher:bytes_downloaded_total", crawl_id] - the
ACTUAL total bytes downloaded by Common Crawl's crawler for that
crawl, as measured by Common Crawl itself, not estimated here.

IMPORTANT CAVEAT: this figure represents bytes downloaded over HTTP
during fetching - closer in magnitude to the "TiB of uncompressed
content" figure Common Crawl states in its monthly blog announcements
than to the compressed, on-disk size of the .warc.gz files (which
would be smaller, since those are gzip-compressed when written).
Anyone using this figure downstream should treat it as an
uncompressed-content-scale total, not a compressed-WARC-disk-size
total - worth stating explicitly in any methods write-up.

The only remaining approximation layer in this pipeline is the
language PAGE SHARE assumption at the very top of this docstring.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ---- CONFIG ----
LANGUAGES_CSV_URL = "https://raw.githubusercontent.com/commoncrawl/cc-crawl-statistics/master/plots/languages.csv"
CRAWLER_STATS_URL_TEMPLATE = "https://raw.githubusercontent.com/commoncrawl/cc-crawl-statistics/master/stats/crawler/{crawl}.json"

OUTPUT_FILE = "../data/crawl/processed/tier_c_language_bytes_by_crawl.csv"          # language x crawl, granular
CUMULATIVE_OUTPUT_FILE = "../data/crawl/processed/tier_c_language_bytes_cumulative.csv"  # language, summed across crawls
LATEST_OUTPUT_FILE = "../data/crawl/processed/tier_c_language_bytes_latest.csv"    # language, most recent crawl it appears in
CRAWL_BYTES_CACHE_FILE = "../data/crawl/raw/tier_c_crawl_bytes_cache.json"    # crawl -> total bytes downloaded, cached
LANGUAGES_CSV_CACHE_FILE = "../data/crawl/raw/tier_c_languages_csv_cache.csv"  # local cache of the downloaded CSV

MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 5


# ---- RETRY HELPER (same pattern as the Tier A harvest scripts) ----
def _with_retry(fn, description, max_retries=MAX_RETRIES):
    """Runs fn() with exponential backoff on failure. Returns None
    (never raises) if all retries are exhausted, so one bad crawl
    can't kill a run across all ~200+ crawls."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"  [retry {attempt + 1}/{max_retries}] {description}: {e} - waiting {wait}s")
            time.sleep(wait)
    print(f"  giving up on {description} after {max_retries} retries")
    return None


# ---- STEP 1 + 2: LANGUAGE PAGE SHARES (from Common Crawl's own published CSV) ----
def fetch_languages_csv():
    """
    Downloads plots/languages.csv from the cc-crawl-statistics GitHub
    repo - the exact file Common Crawl uses to generate the language
    percentages on their own published statistics site. Cached to
    local disk (LANGUAGES_CSV_CACHE_FILE) since it's a moderately
    large file (~12,000 rows covering all 125 crawls x ~200 languages
    each) that's cheap to keep fresh by just re-downloading in full
    each run - unlike the per-crawl caches elsewhere in this script,
    this one is NOT split per-crawl, since the whole file is one
    single download regardless of how many crawls you actually need.

    Returns a pandas DataFrame with columns: crawl, primary_language,
    pages, urls, %pages/crawl. Raises RuntimeError if the download
    fails after retries - this file is required, there's no
    per-language fallback if it's unavailable.
    """
    def _fetch():
        resp = requests.get(LANGUAGES_CSV_URL, timeout=60)
        resp.raise_for_status()
        return resp.text

    csv_text = _with_retry(_fetch, "fetch languages.csv")
    if csv_text is None:
        raise RuntimeError("Could not fetch languages.csv - cannot proceed without it.")

    Path(LANGUAGES_CSV_CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(LANGUAGES_CSV_CACHE_FILE, "w") as f:
        f.write(csv_text)

    from io import StringIO
    df = pd.read_csv(StringIO(csv_text))
    print(f"Fetched languages.csv: {len(df)} row(s) across {df['crawl'].nunique()} crawl(s)")
    return df


def get_language_page_shares(languages_df, crawl_id):
    """
    Filters the full languages.csv DataFrame down to one crawl and
    returns {language_code: page_share} as a fraction (0-1, not a
    percentage), excluding the <unknown> row - <unknown> correctly
    stays part of the page-share DENOMINATOR (it's real pages, just
    ones CLD2 couldn't identify a language for), but isn't a language
    itself, so it's excluded from the per-language OUTPUT rows.
    Uses the file's own %pages/crawl column directly rather than
    recomputing pages/total, since that's exactly what Common Crawl
    itself publishes and what was verified against their live site.
    """
    crawl_rows = languages_df[languages_df["crawl"] == crawl_id]
    crawl_rows = crawl_rows[crawl_rows["primary_language"] != "<unknown>"]
    return dict(zip(crawl_rows["primary_language"], crawl_rows["%pages/crawl"] / 100.0)), \
           dict(zip(crawl_rows["primary_language"], crawl_rows["pages"]))


# ---- STEP 3: PER-CRAWL TOTAL BYTES DOWNLOADED (cached) ----
def load_crawl_bytes_cache():
    path = Path(CRAWL_BYTES_CACHE_FILE)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_crawl_bytes_cache(cache):
    Path(CRAWL_BYTES_CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(CRAWL_BYTES_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_crawler_stats_lines(crawl_id):
    """
    Downloads the crawler operational-metrics file for one crawl from
    the cc-crawl-statistics GitHub repo. This is a DIFFERENT file from
    the language stats file (Step 2) - plain JSON-lines text, not
    gzip-compressed (unlike the stats/part-00000.gz language file).
    Returns a list of raw text lines, or None on failure.
    """
    def _fetch():
        resp = requests.get(CRAWLER_STATS_URL_TEMPLATE.format(crawl=crawl_id), timeout=30)
        resp.raise_for_status()
        return resp.text.splitlines()

    return _with_retry(_fetch, f"fetch crawler stats for {crawl_id}")


def get_crawl_total_bytes_downloaded(crawl_id, cache):
    """
    Returns Common Crawl's own measured total bytes downloaded for
    this crawl - read directly from the

    ["crawl_status", "fetcher:bytes_downloaded_total", crawl_id]
    OR
    ["crawl_status", "fetcher:bytes_downloaded", crawl_id]

    entry in the crawler stats file. Not derived, estimated, or sampled -
    this is their reported figure. See module docstring for the
    uncompressed-vs-compressed caveat.

    Results are cached to disk by crawl_id and never recomputed once
    cached - a published crawl's totals don't change after release.

    Returns None if the fetch fails, or if the expected key isn't
    found in the file (rather than silently returning 0 or guessing).
    """
    if crawl_id in cache:
        return cache[crawl_id]

    lines = fetch_crawler_stats_lines(crawl_id)
    if lines is None:
        return None

    target_key_total = ["crawl_status", "fetcher:bytes_downloaded_total", crawl_id]
    target_key_plain = ["crawl_status", "fetcher:bytes_downloaded", crawl_id]

    for line in lines:
        try:
            key_part, value_part = line.split("\t", 1)
            key = json.loads(key_part)
            value = json.loads(value_part)
        except (ValueError, json.JSONDecodeError):
            continue
        if key == target_key_total or key == target_key_plain:
            cache[crawl_id] = value
            save_crawl_bytes_cache(cache)
            print(f"  {crawl_id}: {key[1]} = {value / 1e9:.1f} GB (Common Crawl's own reported figure)")
            return value
    print(f"  fetcher:bytes_downloaded_total nor bytes_downloaded not found in crawler stats for {crawl_id} - "
        f"file structure may differ for this crawl, skipping")
    return None


# ---- STEP 4 + 5: COMBINE AND AGGREGATE ----
def run_all(max_crawls=None):
    """
    max_crawls: if set, only process the first N crawls (newest
    first, per languages.csv's crawl ordering) - for a quick test
    run. Leave as None for a full run across every available crawl.
    """
    languages_df = fetch_languages_csv()

    # Crawl list is now derived from languages.csv itself (its own
    # 'crawl' column), and guarantees we only ever try crawls
    # this file actually has language data for.
    crawl_ids = sorted(languages_df["crawl"].unique(), reverse=True)
    print(f"Found {len(crawl_ids)} crawl(s) in languages.csv")

    if max_crawls is not None:
        crawl_ids = crawl_ids[:max_crawls]
        print(f"TEST RUN: limited to the {max_crawls} most recent crawl(s)")

    output_path = Path(OUTPUT_FILE)
    already_done = set()
    if output_path.exists():
        prior = pd.read_csv(output_path)
        already_done = set(prior["crawl_id"].unique())
        print(f"Resuming: {len(already_done)} crawl(s) already in {OUTPUT_FILE}, skipping those")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    crawl_bytes_cache = load_crawl_bytes_cache()

    for crawl_id in crawl_ids:
        if crawl_id in already_done:
            continue

        print(f"\n[{crawl_id}] processing...")

        page_shares, page_counts = get_language_page_shares(languages_df, crawl_id)
        if not page_shares:
            print(f"  skipping {crawl_id} - no language rows found in languages.csv for this crawl")
            continue

        total_bytes_downloaded = get_crawl_total_bytes_downloaded(crawl_id, crawl_bytes_cache)
        if total_bytes_downloaded is None:
            print(f"  skipping {crawl_id} - could not get total bytes downloaded after retries")
            continue

        rows = []
        for lang_code, page_share in page_shares.items():
            approx_bytes = page_share * total_bytes_downloaded
            rows.append({
                "crawl_id": crawl_id,
                "language_iso_639_3": lang_code,
                "page_count": page_counts[lang_code],
                "page_share": page_share,
                "crawl_total_bytes_downloaded": total_bytes_downloaded,
                "approx_language_bytes": approx_bytes,
                "estimation_method": "page_share_times_reported_bytes_downloaded_total",
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, mode="a", header=write_header, index=False)
        write_header = False
        print(f"  wrote {len(df)} language row(s) for {crawl_id} "
              f"(crawl total bytes downloaded: {total_bytes_downloaded / 1e9:.1f} GB)")

    build_cumulative_table(output_path, CUMULATIVE_OUTPUT_FILE)
    build_latest_per_language_table(output_path, LATEST_OUTPUT_FILE)


def crawl_id_to_date(crawl_id):
    """
    Converts a crawl ID like "CC-MAIN-2026-25" into an approximate
    calendar date - the Monday of that ISO year/week. Common Crawl's
    crawl IDs ARE ISO year-week identifiers, so this conversion is
    exact in the sense of "which ISO week," but is an APPROXIMATION
    of the actual crawl date: Common Crawl's blog announcements state
    an exact date range per crawl (e.g. "crawled between March 5th
    and March 17th"), which spans roughly two weeks and doesn't
    align perfectly with a single ISO week's Monday. Using this
    computed date avoids an extra network call / blog-scraping
    fragility per crawl (see the total-bytes docstring section above
    for why scraping blog prose was avoided elsewhere in this
    script) - if exact crawl date ranges are ever needed, they would
    have to come from each crawl's blog post individually.

    A small number of very old crawl IDs (pre-2013, e.g.
    "CC-MAIN-2008-2009", "CC-MAIN-2012") don't follow the YYYY-WW
    pattern at all - these return None rather than a guessed date.
    """
    parts = crawl_id.replace("CC-MAIN-", "").split("-")
    if len(parts) != 2 or not (parts[0].isdigit() and parts[1].isdigit()):
        return None
    year, week = parts
    try:
        return datetime.strptime(f"{year}-W{int(week):02d}-1", "%G-W%V-%u").date()
    except ValueError:
        return None


def build_latest_per_language_table(granular_path, latest_path):
    """
    For every language that has EVER appeared in any crawl, keeps
    only its row from the MOST RECENT crawl it appears in - not
    necessarily the single most recent crawl overall. A language
    absent from the latest crawl (e.g. a very low-resource language
    that didn't cross CLD2's detection threshold that particular
    month) still gets a row here, carrying its stats from whichever
    earlier crawl it was last seen in, rather than being dropped.

    Adds a crawl_date column (see crawl_id_to_date()) so it's clear,
    per language, how current or stale its figures are - a language
    last seen in a crawl from two years ago should visibly read as
    such, not look identical to one from last month.

    Recomputed fresh from the full granular file every time, same as
    build_cumulative_table() - always consistent regardless of how
    many resumed runs built the granular file.
    """
    if not Path(granular_path).exists():
        print("No granular output yet - skipping latest-per-language table.")
        return

    df = pd.read_csv(granular_path)
    df["crawl_date"] = df["crawl_id"].apply(crawl_id_to_date)

    # Rows with an unparseable crawl_date (very old non-YYYY-WW crawl
    # IDs) are still eligible to be "most recent" for a language that
    # ONLY ever appeared in one of those - sort them first (oldest/
    # unknown) so any real dated crawl always wins the tiebreak, but
    # don't drop them outright.
    df_sorted = df.sort_values("crawl_date", na_position="first")
    latest = df_sorted.drop_duplicates(subset="language_iso_639_3", keep="last")
    latest = latest.sort_values("approx_language_bytes", ascending=False)

    latest.to_csv(latest_path, index=False)
    print(f"Wrote latest-per-language table ({len(latest)} language(s)) to {latest_path}")


def build_cumulative_table(granular_path, cumulative_path):
    """
    Recomputes the cumulative (summed across all crawls) table fresh
    from the full granular output every time - rather than
    incrementally maintaining a running total - so it's always
    exactly consistent with whatever's in the granular file,
    regardless of how many runs/resumes it took to build it.
    """
    if not Path(granular_path).exists():
        print("No granular output yet - skipping cumulative table.")
        return

    df = pd.read_csv(granular_path)
    cumulative = (
        df.groupby("language_iso_639_3")
        .agg(
            total_approx_bytes_all_crawls=("approx_language_bytes", "sum"),
            num_crawls_present_in=("crawl_id", "nunique"),
        )
        .reset_index()
        .sort_values("total_approx_bytes_all_crawls", ascending=False)
    )
    cumulative.to_csv(cumulative_path, index=False)
    print(f"\nWrote cumulative table ({len(cumulative)} language(s)) to {cumulative_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tier C Common Crawl per-language byte estimation harvest.")
    parser.add_argument(
        "--test", type=int, metavar="N", default=None,
        help="Test run: only process the N most recent crawls.",
    )
    args = parser.parse_args()

    run_all(max_crawls=args.test)
