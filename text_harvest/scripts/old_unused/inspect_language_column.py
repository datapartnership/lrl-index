"""
inspect_language_column.py

Diagnostic tool: for a given dataset, shows the FULL Parquet schema
(all column names, not just the ones matching our candidate lists)
and pulls the DISTINCT values + counts from whichever column looks
like a language identifier - so you can see the REAL format of the
data (case, whitespace, code scheme) instead of guessing why a query
returned 0 matches.

Usage:
  python inspect_language_column.py <dataset_id> [--column NAME] [--limit N]

Examples:
  python inspect_language_column.py espnet/mms_ulab_v2
  python inspect_language_column.py espnet/mms_ulab_v2 --column iso3 --limit 30
"""
import argparse
import io
import os

import duckdb
import pyarrow.parquet as pq
import requests

DATASETS_SERVER_PARQUET_URL = "https://datasets-server.huggingface.co/parquet"
LANGUAGE_COLUMN_CANDIDATES = {
    "iso3", "iso_639_3", "iso639_3", "language", "lang", "language_code",
    "lang_code", "locale", "language_id", "lang_id",
}


def get_parquet_file_urls(dataset_id, hf_token):
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    resp = requests.get(DATASETS_SERVER_PARQUET_URL, params={"dataset": dataset_id}, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [f["url"] for f in data.get("parquet_files", []) if "url" in f]


def get_schema(file_url, hf_token, range_bytes=1_000_000):
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    headers["Range"] = f"bytes=-{range_bytes}"
    resp = requests.get(file_url, headers=headers, timeout=30)
    resp.raise_for_status()
    return pq.ParquetFile(io.BytesIO(resp.content)).schema.names


def main(dataset_id, column_override=None, limit=30, max_files=5, show_all=False):
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Dataset: {dataset_id}")
    print("-" * 60)

    file_urls = get_parquet_file_urls(dataset_id, hf_token)
    print(f"Found {len(file_urls)} Parquet file(s)")
    if not file_urls:
        print("No Parquet files - can't inspect schema.")
        return

    columns = get_schema(file_urls[0], hf_token)
    print(f"\nFULL schema ({len(columns)} column(s)):")
    for c in columns:
        print(f"  {c}")

    # Figure out which column(s) to inspect
    if column_override:
        target_columns = [column_override] if column_override in columns else []
        if not target_columns:
            print(f"\n'{column_override}' not found in schema - check spelling against the list above.")
            return
    else:
        lower_map = {c.lower(): c for c in columns}
        target_columns = [lower_map[c] for c in LANGUAGE_COLUMN_CANDIDATES if c in lower_map]
        if not target_columns:
            print("\nNo column matched the known language-column candidates - "
                  "pass --column NAME explicitly using one of the schema columns above.")
            return

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    files_to_scan = file_urls if show_all else file_urls[:max_files]
    url_list_sql = "[" + ", ".join(f"'{u}'" for u in files_to_scan) + "]"
    print(f"\nScanning {len(files_to_scan)} of {len(file_urls)} Parquet file(s)"
          + (" (ALL files)" if show_all else f" (use --all-files to scan all {len(file_urls)})"))

    display_limit = 100_000 if limit is None else limit  # effectively "no cap" for a language-code column

    for col in target_columns:
        print(f"\n=== ALL distinct values in '{col}' ===")
        query = f"""
            SELECT "{col}" AS value, count(*) AS n
            FROM read_parquet({url_list_sql})
            GROUP BY "{col}"
            ORDER BY n DESC
            LIMIT {display_limit}
        """
        try:
            rows = con.execute(query).fetchall()
            print(f"({len(rows)} distinct value(s) found)")
            for value, n in rows:
                print(f"  {value!r}: {n:,} row(s)")
        except Exception as e:
            print(f"  query failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect real schema + language-column values for a dataset.")
    parser.add_argument("dataset_id")
    parser.add_argument("--column", default=None, help="Force a specific column name instead of auto-detecting")
    parser.add_argument("--limit", type=int, default=None, help="Cap on distinct values shown (default: no cap)")
    parser.add_argument("--max-files", type=int, default=5, help="How many Parquet files to sample (ignored if --all-files)")
    parser.add_argument("--all-files", action="store_true", help="Scan ALL Parquet files, not just a sample - slower but complete")
    args = parser.parse_args()

    main(args.dataset_id, column_override=args.column, limit=args.limit,
         max_files=args.max_files, show_all=args.all_files)
