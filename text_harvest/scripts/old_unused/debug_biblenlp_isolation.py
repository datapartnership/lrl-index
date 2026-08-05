"""
debug_biblenlp_isolation.py

Diagnoses why isolate_language_files() reports no matches for
davidstap/biblenlp-corpus-mmteb, despite folders like "eng-{iso}"
existing. Prints the real file listing and tests isolation against
several candidate code forms, to pin down whether this is a 2-letter
vs 3-letter ISO code mismatch or something else entirely.
"""
from tier_a_harvest_v2 import api, isolate_language_files

DATASET_ID = "davidstap/biblenlp-corpus-mmteb"

info = api.dataset_info(DATASET_ID, files_metadata=True)
all_files = [s.rfilename for s in info.siblings]
print(f"{len(all_files)} total file(s) in {DATASET_ID}\n")

print("First 30 files (raw, unfiltered):")
for f in all_files[:30]:
    print(" ", f)

# Pull out the distinct top-level folder names to see the real naming
# convention directly, rather than guessing from a partial file list
folders = sorted(set(f.split("/")[0] for f in all_files if "/" in f))
print(f"\n{len(folders)} distinct top-level folder(s) - first 20:")
for folder in folders[:20]:
    print(" ", folder)

# Test isolation against the actual code that failed in the real run
test_codes = ["aai"]
print(f"\nTesting isolate_language_files() against candidate codes: {test_codes}")
for code in test_codes:
    matches = isolate_language_files(info.siblings, code)
    print(f"  code='{code}': {len(matches)} match(es)" + (f" - e.g. {matches[0]}" if matches else ""))

# Also do a loose substring search (not our real matching logic - just
# for diagnosis) to see if "aai" appears ANYWHERE in the file listing,
# under any naming variant, even if our strict segment-match doesn't
# catch it
print("\nLoose substring search for 'aai' anywhere in file paths (diagnostic only):")
loose_matches = [f for f in all_files if "aai" in f.lower()]
for f in loose_matches[:20]:
    print(" ", f)
if not loose_matches:
    print("  (no matches at all - 'aai' doesn't appear anywhere in the file listing)")
