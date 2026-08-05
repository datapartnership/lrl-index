"""
debug_parquet_raw.py

Bypasses the retry wrapper entirely to show the REAL exception (full
traceback) from fetching + parsing a Parquet file's footer, since the
retry wrapper's error printing has been showing just the URL with no
actual error detail - something is being swallowed or mis-stringified.
"""
import io
import traceback

import requests
import pyarrow.parquet as pq

URL = "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/main/squad_v2/train-00000-of-00001.parquet"

print(f"Requesting: {URL}\n")

try:
    resp = requests.get(URL, headers={"Range": "bytes=-1000000"}, timeout=30)
    print("Status code:", resp.status_code)
    print("Response headers:")
    for k, v in resp.headers.items():
        print(f"  {k}: {v}")
    print("\nContent length received:", len(resp.content))
    print("First 300 bytes of content (repr):")
    print(repr(resp.content[:300]))
except Exception:
    print("!!! Exception during requests.get - full traceback: !!!\n")
    traceback.print_exc()
    raise SystemExit(1)

print("\n--- Attempting pyarrow parse ---")
try:
    pf = pq.ParquetFile(io.BytesIO(resp.content))
    print("SUCCESS - num_rows:", pf.metadata.num_rows)
except Exception:
    print("!!! Exception during pyarrow parse - full traceback: !!!\n")
    traceback.print_exc()
    