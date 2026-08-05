"""
debug_tier_a_v2_squad.py

Quick sanity check: does resolve_total_size() correctly pick up
squad_v2's card-level dataset_info YAML, without touching the
publication or file-summing fallback paths at all?
"""
from tier_a_harvest_v2 import (
    api, fetch_card_data, extract_card_fields, resolve_total_size,
    get_modality_structured_counts,
)

DATASET_ID = "rajpurkar/squad_v2"

print(f"Fetching dataset_info for {DATASET_ID}...")
info = api.dataset_info(DATASET_ID, files_metadata=True)
tags = info.tags or []

print("Loading card data...")
card_data = fetch_card_data(DATASET_ID)

fields = extract_card_fields(DATASET_ID, tags, card_data)
print("\n--- Card fields ---")
for k, v in fields.items():
    print(f"  {k}: {v}")

result = resolve_total_size(fields, info.siblings, DATASET_ID)
print("\n--- Total size resolution ---")
for k, v in result.items():
    print(f"  {k}: {v}")

print("\nExpected: total_size_method == 'card_reported', total_size_bytes == 128393116")

# Modality structured counts - squad_v2 is text-only, should attempt
# Parquet row counts. Limiting to a couple of files for speed.
text_files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet")][:3]
print(f"\nChecking Parquet row counts on {len(text_files)} file(s): {text_files}")
counts = get_modality_structured_counts(info.siblings, DATASET_ID, target_files=text_files)
print("\n--- Modality structured counts (limited sample) ---")
print(counts)
