"""
debug_datasets_server_raw.py

Bypasses the retry wrapper to show the REAL response (status code,
headers, full body) from the Datasets Server /size endpoint - same
diagnostic pattern used earlier for the Parquet footer issue, since
values silently not populating usually means an error is being
swallowed somewhere rather than the logic being wrong.
"""
import os
import traceback

import requests

DATASET_ID = "sil-ai/bloom-lm"
URL = f"https://datasets-server.huggingface.co/size?dataset={DATASET_ID}"

token = os.environ.get("HF_TOKEN")
headers = {"Authorization": f"Bearer {token}"} if token else {}
print(f"HF_TOKEN set: {bool(token)}")
print(f"Requesting: {URL}\n")

try:
    resp = requests.get(URL, headers=headers, timeout=30)
    print("Status code:", resp.status_code)
    print("Response headers:")
    for k, v in resp.headers.items():
        print(f"  {k}: {v}")
    print("\nFull response body:")
    print(resp.text[:3000])
except Exception:
    print("!!! Exception during requests.get - full traceback: !!!\n")
    traceback.print_exc()
