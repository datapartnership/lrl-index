import sys
sys.path.insert(0, '.')
from tier_c_common_crawl_harvest import fetch_stats_file_lines, parse_language_page_counts

lines = fetch_stats_file_lines("CC-MAIN-2026-25")
language_pages, total_pages = parse_language_page_counts(lines)

for code in ["eng", "rus", "deu", "un", "unk"]:
    if code in language_pages:
        print(f"{code}: {language_pages[code]} pages, {100 * language_pages[code] / total_pages:.4f}%")

print("total_pages (our denominator):", total_pages)

# also check for anything that might represent "unknown"
for line in lines:
    if '"un"' in line or '"unk"' in line or "unknown" in line.lower():
        print("possible unknown entry:", line)

import sys
sys.path.insert(0, '.')
from tier_c_common_crawl_harvest import fetch_stats_file_lines
import json

lines = fetch_stats_file_lines("CC-MAIN-2026-25")

# Find the total HTML page count - likely under "mimetype"
for line in lines:
    if '"mimetype","text/html"' in line or '"mimetype","text\\/html"' in line:
        print("HTML mimetype line:", line)

# Also dump the full list of "languages" category codes so we can see
# if there's a sentinel/non-ISO code we're missing entirely
lang_codes = []
for line in lines:
    try:
        key_part, value_part = line.split("\t", 1)
        key = json.loads(key_part)
    except Exception:
        continue
    if isinstance(key, list) and len(key) == 3 and key[0] == "languages":
        lang_codes.append(key[1])

print("total distinct 'languages' key entries:", len(lang_codes))
print("any non-comma, non-3-letter codes (possible sentinels)?")
for c in set(lang_codes):
    if "," not in c and len(c) != 3:
        print(" ", repr(c))
