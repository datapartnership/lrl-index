"""
voxpopuli_harvest.py

Week 4 audio harvest: VoxPopuli. Parses the real "Unlabelled and
transcribed data" table directly from facebookresearch/voxpopuli's
README (confirmed live structure - see conversation), rather than an
API (VoxPopuli doesn't have one - this is a static, versioned table in
the repo's own documentation).

--------------------------------------------------------------------
TABLE STRUCTURE (confirmed against the live README)
--------------------------------------------------------------------
| Language | Code | Unlabelled Hours (v1/v2) | Transcribed Hours | Transcribed Speakers | Transcribed Tokens | LM Tokens |

- Code is a 2-letter code, capitalized (e.g. "En", "De") - lowercased
  here before matching against your crosswalk's ISO 639-1 column.
- "Unlabelled Hours (v1/v2)" is a COMPOUND field like "4.5K/24.1K" -
  v1 and v2 separated by a slash. Per your instruction, only the v2
  figure is used (untranscribed hours).
- "Transcribed Hours" is a plain number, no K-suffix in the live data.
  Some languages show "-" here (Portuguese, Bulgarian, Greek, Latvian,
  Maltese, Swedish, Danish, as of this table's current contents) -
  meaning NO transcribed data exists for that language at all, not
  zero. No transcribed row is produced for these - only untranscribed.
- A "Total" row exists at the end of the table and is explicitly
  excluded - it's an aggregate, not a language.

--------------------------------------------------------------------
YOUR RULES FOR THIS SOURCE
--------------------------------------------------------------------
- Unlabelled v2 hours -> untranscribed_hours, transcript_exists=False
- Transcribed Hours -> transcribed_hours, transcript_exists=True
- license: CC0
- source_type: "european parliament recording"
- who_transcribed: "european parliament human volunteers"

--------------------------------------------------------------------
JOIN KEY: ISO 639-1 (2-letter), NOT 639-3
--------------------------------------------------------------------
VoxPopuli's Code column is ISO 639-1. See the ADAPT block in CONFIG -
CROSSWALK_ISO2_COL is a guess at your crosswalk's 2-letter-code column
name and needs confirming. CROSSWALK_ISO3_COL is the canonical output
code (same convention as the Common Voice harvest).

--------------------------------------------------------------------
NO CLEAN VERSION TRACKING
--------------------------------------------------------------------
Unlike Common Voice's dated, versioned release JSON files, this table
is just "whatever's currently in the README" - there's no version
number attached to it in a way this script can track automatically.
retrieval_date (today's date, recorded at run time) is included so at
least WHEN this was pulled is on record, even though WHICH version of
the table isn't.

--------------------------------------------------------------------
DEPENDENCIES / SETUP
--------------------------------------------------------------------
pip install pandas requests
(no auth needed - plain public README fetch)
"""
import re
from pathlib import Path

import pandas as pd
import requests

# ---- CONFIG ----
VOXPOPULI_README_URL = "https://raw.githubusercontent.com/facebookresearch/voxpopuli/main/README.md"

# ---- ADAPT THIS: confirm these match your actual crosswalk file ----
CROSSWALK_FILE = "../../crosswalk/data/processed/full_language_reference.csv"
CROSSWALK_ISO3_COL = "iso_639_3"   # canonical output code
CROSSWALK_ISO2_COL = "iso_639_1"   # VoxPopuli's join key - CONFIRM this column name exists in your crosswalk

OUTPUT_FILE = "../data/audio/processed/voxpopuli_hours.csv"
SKIPPED_LOG_FILE = "../data/audio/processed/voxpopuli_skipped_codes.csv"

SOURCE_TYPE = "european parliament recording"
WHO_TRANSCRIBED = "european parliament human volunteers"
LICENSE = "CC0-1.0"
LICENSE_TIER = "A"

HOURS_VALUE_PATTERN = re.compile(r"^([\d.]+)(K)?$", re.IGNORECASE)


def parse_hours_value(raw):
    """
    Parses a table cell like '4.5K', '543', or '-' into a float number
    of hours. '-' (missing data) is treated as 0, not omitted - per
    your instruction, every language gets a row for both categories
    even when no transcribed data exists.
    """
    raw = raw.strip()
    if raw in ("-", "", "—"):
        return 0.0
    match = HOURS_VALUE_PATTERN.match(raw)
    if not match:
        print(f"    could not parse hours value: {raw!r} - treating as 0")
        return 0.0
    number = float(match.group(1))
    if match.group(2):  # 'K' suffix
        number *= 1000
    return number


def fetch_readme_text():
    resp = requests.get(VOXPOPULI_README_URL, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_unlabelled_transcribed_table(readme_text):
    """
    Extracts and parses the "Unlabelled and transcribed data" table
    specifically (the repo README has a second table further down,
    speech-to-speech interpretation data, which is NOT what we want -
    scoped out by only reading between this table's own markdown
    boundaries).

    Returns a list of dicts: {code, unlabelled_v1_hours,
    unlabelled_v2_hours, transcribed_hours}. The "Total" row is
    excluded. Raises RuntimeError if the table can't be located at all
    (structure changed) - better to fail loudly than silently return
    nothing.
    """
    lines = readme_text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("| Language | Code | Unlabelled Hours"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not find the Unlabelled and transcribed data table header - "
                            "README structure may have changed since this script was written.")

    rows = []
    # Table rows start 2 lines after the header (header + markdown separator line)
    for line in lines[header_idx + 2:]:
        line = line.strip()
        if not line.startswith("|"):
            break  # table ended
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        language_name, code = cells[0], cells[1]
        if language_name == "Total" or not code:
            continue  # aggregate row, not a language

        unlabelled_raw = cells[2]
        if "/" in unlabelled_raw:
            v1_raw, v2_raw = unlabelled_raw.split("/", 1)
        else:
            v1_raw, v2_raw = unlabelled_raw, unlabelled_raw  # defensive - shouldn't happen in practice

        rows.append({
            "language_name": language_name,
            "code": code.lower(),
            "unlabelled_v1_hours": parse_hours_value(v1_raw),
            "unlabelled_v2_hours": parse_hours_value(v2_raw),
            "transcribed_hours": parse_hours_value(cells[3]),
            "transcribed_speakers": parse_hours_value(cells[4]) if len(cells) > 4 else None,
        })

    if not rows:
        raise RuntimeError("Table header found but no data rows parsed - check table format hasn't changed.")
    return rows


def load_crosswalk_iso2_map():
    df = pd.read_csv(CROSSWALK_FILE)
    if CROSSWALK_ISO2_COL not in df.columns:
        raise RuntimeError(
            f"Column {CROSSWALK_ISO2_COL!r} not found in crosswalk - "
            f"available columns: {list(df.columns)}. Update CROSSWALK_ISO2_COL."
        )
    code_to_iso3 = {}
    for _, row in df.iterrows():
        iso3 = row.get(CROSSWALK_ISO3_COL)
        iso2 = row.get(CROSSWALK_ISO2_COL)
        if pd.isna(iso3) or pd.isna(iso2):
            continue
        code_to_iso3[str(iso2).strip().lower()] = iso3
    print(f"{len(code_to_iso3)} ISO 639-1 code(s) mapped from crosswalk")
    return code_to_iso3


def main():
    print("Fetching VoxPopuli README...")
    readme_text = fetch_readme_text()
    table_rows = parse_unlabelled_transcribed_table(readme_text)
    print(f"Parsed {len(table_rows)} language row(s) from the table")

    code_to_iso3 = load_crosswalk_iso2_map()

    output_rows = []
    skipped_rows = []

    for row in table_rows:
        iso3 = code_to_iso3.get(row["code"])
        if iso3 is None:
            skipped_rows.append({"code": row["code"], "language_name": row["language_name"],
                                  "reason": "not_in_crosswalk"})
            continue

        # Untranscribed row - always produced (0 if data was missing)
        output_rows.append({
            "iso_639_3": iso3, "vp_code": row["code"],
            "source": "voxpopuli", "category": "untranscribed",
            "hours": row["unlabelled_v2_hours"],
            "license_tier": LICENSE_TIER, "license": LICENSE,
            "source_type": SOURCE_TYPE, "who_transcribed": WHO_TRANSCRIBED,
            "transcript_exists": False,
        })

        # Transcribed row - always produced (0 if no transcribed data exists for this language)
        output_rows.append({
            "iso_639_3": iso3, "vp_code": row["code"],
            "source": "voxpopuli", "category": "transcribed",
            "hours": row["transcribed_hours"],
            "license_tier": LICENSE_TIER, "license": LICENSE,
            "source_type": SOURCE_TYPE, "who_transcribed": WHO_TRANSCRIBED,
            "transcript_exists": True,
        })

    if skipped_rows:
        Path(SKIPPED_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(skipped_rows).to_csv(SKIPPED_LOG_FILE, index=False)
        print(f"\nWrote {len(skipped_rows)} skipped code(s) to {SKIPPED_LOG_FILE}")

    if not output_rows:
        print("No rows produced - nothing to write.")
        return

    FINAL_COLUMNS = [
        "iso_639_3", "vp_code", "source", "category", "hours",
        "license_tier", "license", "source_type", "who_transcribed", "transcript_exists",
    ]
    df = pd.DataFrame(output_rows)[FINAL_COLUMNS].sort_values(["iso_639_3", "category"])
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {len(df)} row(s) to {OUTPUT_FILE}")
    print(f"\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
