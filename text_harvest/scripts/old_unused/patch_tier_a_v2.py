"""
patch_tier_a_v2.py

Patches tier_a_harvest_v2.py IN PLACE to replace the fsspec-based
Parquet footer reader with the fixed requests+pyarrow version, and
removes the now-unused fsspec import. Run this once from the same
directory as tier_a_harvest_v2.py:

    python3 patch_tier_a_v2.py

It will print exactly what it changed, and print an error (without
touching the file) if it can't find the expected old code - meaning
the file already has different content than expected and needs a
manual look rather than this patch.
"""
from pathlib import Path

TARGET = Path("tier_a_harvest_v2.py")

if not TARGET.exists():
    raise SystemExit(f"ERROR: {TARGET} not found in the current directory. "
                      f"Run this from the same folder as tier_a_harvest_v2.py.")

content = TARGET.read_text()
original_content = content
changes_made = []

# 1. Remove the fsspec import
old_import = "import fsspec\nimport pyarrow.parquet as pq"
new_import = "import pyarrow.parquet as pq"
if old_import in content:
    content = content.replace(old_import, new_import)
    changes_made.append("Removed 'import fsspec'")
else:
    print("WARNING: could not find the expected fsspec import line to remove - "
          "may already be removed, or file differs from expected. Skipping this change.")

# 2. Replace the fsspec-based get_parquet_row_count function with the fixed version
old_function = '''def get_parquet_row_count(file_url):
    """
    Reads ONLY the Parquet footer (a small trailer at the end of the
    file containing row-group metadata, including total row count) -
    not the full file. fsspec's HTTP filesystem performs this as a
    range request if the server supports it (Hugging Face's CDN
    does), so this is cheap regardless of the file's actual size.

    Returns an int row count, or None if the read fails for any
    reason (never guesses, never partial-counts).
    """
    def _read():
        with fsspec.open(file_url, mode="rb") as f:
            return pq.ParquetFile(f).metadata.num_rows

    return _with_retry(_read, f"read parquet footer for {file_url}", max_retries=2)'''

new_function = '''def get_parquet_row_count(file_url, tail_bytes=1_000_000):
    """
    Reads ONLY the Parquet footer - not the full file. A Parquet
    file's footer (containing row-group metadata, including total row
    count) is self-contained at the END of the file; nothing at the
    start of the file is needed to read metadata.num_rows. This fetches
    just the last `tail_bytes` (default ~1MB, generously larger than
    any realistic footer) via a single HTTP Range request and hands
    that buffer directly to pyarrow - pyarrow treats the buffer's own
    length as "the file size" and reads backward from there, so as
    long as the true footer fits within the fetched tail (it does, for
    any normal Parquet file), this works correctly without needing the
    file's true total size or its first bytes at all.

    Returns an int row count, or None if the read/parse fails for any
    reason (never guesses, never partial-counts).
    """
    def _read():
        resp = requests.get(file_url, headers={"Range": f"bytes=-{tail_bytes}"}, timeout=30)
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"unexpected status {resp.status_code}")
        return pq.ParquetFile(io.BytesIO(resp.content)).metadata.num_rows

    return _with_retry(_read, f"read parquet footer for {file_url}", max_retries=2)'''

if old_function in content:
    content = content.replace(old_function, new_function)
    changes_made.append("Replaced get_parquet_row_count() with the fsspec-free version")
else:
    print("WARNING: could not find the expected old get_parquet_row_count() function - "
          "may already be patched, or file differs from expected. Skipping this change.")

# 3. Update the pip install comment line (drop fsspec/aiohttp, add pypdf)
old_pip = "pip install pyarrow fsspec aiohttp requests mutagen huggingface_hub anthropic"
new_pip = "pip install pyarrow requests mutagen huggingface_hub anthropic pypdf"
if old_pip in content:
    content = content.replace(old_pip, new_pip)
    changes_made.append("Updated the pip install dependency comment line")
else:
    print("WARNING: could not find the expected pip install comment line - "
          "may already be updated, or file differs from expected. Skipping this change.")

if content == original_content:
    print("\nNo changes were made - file already matches expected state, or patch "
          "targets weren't found (see warnings above). Nothing was written.")
else:
    TARGET.write_text(content)
    print(f"\nPatched {TARGET} successfully:")
    for c in changes_made:
        print(f"  - {c}")
    print("\nVerify with: grep -n fsspec tier_a_harvest_v2.py   (should print nothing now)")
