"""
lingua_libre_harvest.py
================================

Pulls every audio file title for each language in the language crosswalk
from its Wikimedia Commons category, fetches duration metadata in batches
via the Commons API, and writes the initial harvest CSV with the schema
below (matching lrl-index/audio_harvest/scripts conventions).

INPUT:
    Language crosswalk CSV, e.g.:
        lrl-index/crosswalk/data/processed/full_language_reference.csv
    Only the `iso_639_3` column is used to build Commons category names;
    all other crosswalk columns are ignored by this script.

OUTPUT (initial harvest CSV), one row per audio file:
    Iso_639_3        - from the crosswalk
    Source           - the audio file name (Commons File: title)
    Category         - constant: "transcribed"
                       (every Lingua Libre recording is paired with a
                       preset word by construction, so this is fixed
                       rather than derived per-file)
    Duration (s)     - from Commons file metadata (img_metadata /
                       imageinfo), in seconds
    License          - constant: "CC BY-SA 4.0"
    Source_type      - the specific word list the recording came from,
                       derived from Commons "fileusage": if the file is
                       used/embedded on a page matching
                       "Commons:Lingua_Libre/List/<iso>/<list name>",
                       the <list name> segment is used (e.g. "Unilex
                       common words 1"). NOTE: the "LL-Q####" number in
                       the filename is the Wikidata QID for the LANGUAGE
                       (e.g. Q150 = French), not a word-list ID - it is
                       not used for this derivation.
                       Left blank when no matching list page is found
                       (some recordings are made ad hoc, not from a
                       tracked list - this is expected, not a bug).
                       If a file matches more than one list page, all
                       matches are joined with "; ".
    Who_transcribed  - constant: "no transcription – pre-set word"
    Transcript_exists - constant: TRUE

USAGE:
    python lingua_libre_duration_harvester.py full_language_reference.csv
    python lingua_libre_duration_harvester.py full_language_reference.csv \
        --user-agent "YourName-LRLIndex/1.0 (you@example.org)" \
        --output /custom/path/output.csv

    Designed to be run from lrl-index/audio_harvest/scripts/ (or anywhere
    else) - the default output path is anchored to this script's own file
    location, not the current working directory, so it always resolves to:
        lrl-index/audio_harvest/data/processed/lingua_libre/lingua_libre_harvest.csv
    regardless of where you invoke it from. The folder is created
    automatically if it doesn't exist yet. Pass --output to override.

    USER-AGENT: this script does NOT ship with a hardcoded User-Agent -
    Wikimedia requires each requester to identify themselves, so whoever
    runs this needs to supply their own. Resolution order:
        1. --user-agent flag
        2. COMMONS_USER_AGENT environment variable
        3. interactive prompt (asked at runtime if neither above is set)

REQUIREMENTS:
    pip install requests --break-system-packages

NOTES / CAVEATS :
  - Calls the public Commons API (commons.wikimedia.org/w/api.php).
    Requires internet access from wherever you run it.
  - Set a real, descriptive User-Agent below (COMMONS_USER_AGENT) before
    running at scale - Wikimedia blocks unidentified traffic.
  - Category naming assumes the modern pattern:
        Category:Lingua_Libre_pronunciation-<iso_639_3>
  - Metadata format is INCONSISTENT across files:
    newer .wav files return nested JSON with "playtime_seconds"; older
    .ogg files return a flatter structure with "length". extract_duration()
    tries several known key names and falls back to a recursive search -
    if you see a lot of blank durations, that's the signal a new format
    variant showed up and needs another branch added.
  - Rate limiting: batches of 50 titles per imageinfo call, with a small
    delay and retry-with-backoff on HTTP 429. Adjust SLEEP_BETWEEN_CALLS
    and BATCH_SIZE if you have an authenticated account (higher tier) or
    hit limits with an anonymous one.
  - Uses POST (not GET) for Commons API calls, since batches containing
    long titles (e.g. Arabic filenames that are full phrases rather than
    single words) can push a GET request's URL past the server's length
    limit (HTTP 414 error). If you still hit 414s with very long titles,
    lower BATCH_SIZE.
  - RESUME / FAULT TOLERANCE: results are written to the output CSV
    incrementally, one language at a time, and a failure on one language
    (network error, unexpected data, etc.) is logged and skipped rather
    than crashing the whole run. If the script stops or crashes, just
    rerun it the same way - it detects which languages are already in
    the output file and skips them, re-attempting only what's missing.
"""

import argparse
import csv
import os
import time
import requests

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
BATCH_SIZE = 50
SLEEP_BETWEEN_CALLS = 0.5  # seconds; raise this if you get 429s
MAX_RETRIES = 5

# Populated at runtime in main() - see get_user_agent(). Do NOT hardcode a
# value here; each person running this script should supply their own,
# since Wikimedia ties rate limits/blocking decisions to this identifier.
HEADERS = {}

# Constants for the harvest CSV schema
CATEGORY_VALUE = "transcribed"
LICENSE_VALUE = "CC BY-SA 4.0"
WHO_TRANSCRIBED_VALUE = "no transcription \u2013 pre-set word"
TRANSCRIPT_EXISTS_VALUE = "TRUE"

LIST_PAGE_PREFIX = "Commons:Lingua_Libre/List/"

HARVEST_FIELDNAMES = [
    "Iso_639_3",
    "Source",
    "Category",
    "Duration (s)",
    "License",
    "Source_type",
    "Who_transcribed",
    "Transcript_exists",
]

def get_user_agent(cli_value=None):
    """
    Resolve the User-Agent to send with every Commons API request.
    Priority: --user-agent flag > COMMONS_USER_AGENT env var > interactive
    prompt. Wikimedia requires a real, identifying User-Agent (project
    name + contact info/email) - generic or missing ones get blocked with
    403/429. Each person running this script should supply their own.
    """
    if cli_value:
        return cli_value
    env_value = os.environ.get("COMMONS_USER_AGENT")
    if env_value:
        return env_value
    print(
        "No User-Agent provided. Wikimedia requires a real, identifying "
        "User-Agent for API access (project name + contact email), e.g.:\n"
        "  LRLIndexHarvester/1.0 (kirstenmayer@example.org)\n"
    )
    while True:
        value = input("Enter your User-Agent string: ").strip()
        if value:
            return value
        print("User-Agent cannot be blank - please enter a value.")


def api_get(params, retries=MAX_RETRIES):
    """
    POST against the Commons API with basic 429 backoff.
    Uses POST (not GET) because batches of long titles - e.g. Arabic
    filenames that are full phrases rather than single words - can push a
    GET request's URL past the server's length limit (HTTP 414). POST puts
    parameters in the request body instead, avoiding that ceiling.
    """
    params = {**params, "format": "json"}
    for attempt in range(retries):
        resp = requests.post(COMMONS_API, data=params, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 5))
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Exceeded retries on Commons API call")


def get_category_files(iso_639_3):
    """Return all File: titles in Category:Lingua_Libre_pronunciation-<iso_639_3>."""
    category = f"Category:Lingua_Libre_pronunciation-{iso_639_3}"
    titles = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "500",
            "cmtype": "file",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(params)
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)
        if "continue" in data:
            cmcontinue = data["continue"]["cmcontinue"]
            time.sleep(SLEEP_BETWEEN_CALLS)
        else:
            break
    return titles


def extract_duration(metadata_blob):
    """
    Try to find a duration-in-seconds value inside whatever shape the
    Commons API's iiprop=metadata returns. Handles the known shapes seen
    during testing; falls back to a generic recursive search for
    plausible key names.
    """
    if metadata_blob is None:
        return None

    known_keys = ("playtime_seconds", "length", "duration")

    # Legacy API shape: list of {"name": ..., "value": ...} dicts
    if isinstance(metadata_blob, list):
        for item in metadata_blob:
            if isinstance(item, dict) and item.get("name") in known_keys:
                try:
                    return float(item["value"])
                except (TypeError, ValueError):
                    pass
        for item in metadata_blob:
            if isinstance(item, dict):
                found = extract_duration(item.get("value"))
                if found is not None:
                    return found
        return None

    # Dict shape (nested JSON, e.g. getID3-style)
    if isinstance(metadata_blob, dict):
        for key in known_keys:
            if key in metadata_blob:
                try:
                    return float(metadata_blob[key])
                except (TypeError, ValueError):
                    pass
        for value in metadata_blob.values():
            found = extract_duration(value)
            if found is not None:
                return found
        return None

    return None


def parse_source_type(fileusage_entries):
    """
    Given a list of {"title": ...} dicts from prop=fileusage, return the
    word-list name for any page matching Commons:Lingua_Libre/List/<iso>/<name>,
    joining multiple matches with "; ". Returns "" if none match.
    """
    if not fileusage_entries:
        return ""
    list_names = []
    for entry in fileusage_entries:
        title = entry.get("title", "")
        if title.startswith(LIST_PAGE_PREFIX):
            remainder = title[len(LIST_PAGE_PREFIX):]  # "<iso>/<list name>"
            parts = remainder.split("/", 1)
            if len(parts) == 2:
                list_name = parts[1].replace("_", " ")
                if list_name not in list_names:
                    list_names.append(list_name)
    return "; ".join(list_names)


def get_file_metadata(titles):
    """
    Batch-fetch duration and source_type for a list of File: titles.
    Returns dict title -> {"duration": seconds|None, "source_type": str}.
    """
    results = {}
    for i in range(0, len(titles), BATCH_SIZE):
        batch = titles[i : i + BATCH_SIZE]
        params = {
            "action": "query",
            "prop": "imageinfo|fileusage",
            "iiprop": "size|metadata",
            "fuprop": "title",
            "fulimit": "500",
            "titles": "|".join(batch),
        }
        data = api_get(params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title")
            imageinfo = page.get("imageinfo", [])
            duration = None
            if imageinfo:
                duration = extract_duration(imageinfo[0].get("metadata"))
            source_type = parse_source_type(page.get("fileusage", []))
            # NOTE: if a file has >500 usages, fileusage will be truncated
            # here (no continuation handled) - a page with "fucontinue" in
            # the raw response is the signal this happened for that file.
            results[title] = {"duration": duration, "source_type": source_type}
        time.sleep(SLEEP_BETWEEN_CALLS)
        print(f"  ...{min(i + BATCH_SIZE, len(titles))}/{len(titles)} files processed")
    return results


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Script is expected to live in .../audio_harvest/scripts/ ; data output
# lives as a sibling under .../audio_harvest/data/processed/lingua_libre/
DEFAULT_OUTPUT_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "data", "processed", "lingua_libre")
)
DEFAULT_OUTPUT_FILENAME = "lingua_libre_harvest.csv"


def main(crosswalk_path, output_path=None, user_agent=None):
    HEADERS["User-Agent"] = get_user_agent(user_agent)

    if output_path is None:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILENAME)
    else:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    with open(crosswalk_path, newline="", encoding="utf-8") as f:
        crosswalk = list(csv.DictReader(f))

    if not crosswalk or "iso_639_3" not in crosswalk[0]:
        raise ValueError(
            "Crosswalk CSV must have an 'iso_639_3' column "
            "(matching lrl-index/crosswalk/data/processed/full_language_reference.csv)"
        )

    # RESUME SUPPORT: if the output file already exists (e.g. from a run
    # that crashed partway through), figure out which languages already
    # have rows written and skip them, rather than starting over. A
    # language is only ever written once it's fully processed (see below),
    # so anything present in the file is safe to treat as "done".
    already_done = set()
    file_exists = os.path.isfile(output_path)
    if file_exists:
        with open(output_path, newline="", encoding="utf-8") as f:
            for existing_row in csv.DictReader(f):
                already_done.add(existing_row.get("Iso_639_3", ""))
        if already_done:
            print(
                f"Resuming: found {len(already_done)} language(s) already in "
                f"{output_path}, will skip them: {sorted(already_done)}"
            )

    out_f = open(output_path, "a" if file_exists else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=HARVEST_FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        out_f.flush()

    total_rows_written = 0
    failed_languages = []

    try:
        for row in crosswalk:
            iso_639_3 = row["iso_639_3"].strip()
            if not iso_639_3 or iso_639_3 in already_done:
                continue

            print(f"\n=== {iso_639_3} ===")

            try:
                print("Fetching file list from category...")
                titles = get_category_files(iso_639_3)
                print(f"  Found {len(titles)} files.")

                if not titles:
                    continue

                print("Fetching durations and source_type...")
                file_metadata = get_file_metadata(titles)

                language_rows = []
                found_count = 0
                no_source_type_count = 0
                for title, info in file_metadata.items():
                    seconds = info["duration"]
                    source_type = info["source_type"]
                    language_rows.append(
                        {
                            "Iso_639_3": iso_639_3,
                            "Source": title,
                            "Category": CATEGORY_VALUE,
                            "Duration (s)": seconds if seconds is not None else "",
                            "License": LICENSE_VALUE,
                            "Source_type": source_type,
                            "Who_transcribed": WHO_TRANSCRIBED_VALUE,
                            "Transcript_exists": TRANSCRIPT_EXISTS_VALUE,
                        }
                    )
                    if seconds is not None:
                        found_count += 1
                    if not source_type:
                        no_source_type_count += 1

                if no_source_type_count:
                    print(
                        f"  NOTE: {no_source_type_count} files had no matching list "
                        f"page (likely recorded ad hoc, not from a tracked word list)."
                    )

                missing = len(titles) - found_count
                if missing:
                    print(f"  WARNING: {missing} files had no parseable duration.")

                # Write and flush this language's rows immediately, so a
                # crash on a LATER language doesn't lose this one's work.
                writer.writerows(language_rows)
                out_f.flush()
                total_rows_written += len(language_rows)
                print(f"  Wrote {len(language_rows)} rows for {iso_639_3}.")

            except Exception as exc:  # noqa: BLE001 - intentionally broad
                print(f"  ERROR processing {iso_639_3}: {exc}")
                print(f"  Skipping {iso_639_3} and continuing with the next language.")
                failed_languages.append(iso_639_3)
                continue
    finally:
        out_f.close()

    print(f"\nDone. Wrote {total_rows_written} new rows to {output_path}.")
    if failed_languages:
        print(
            f"The following language(s) failed and were skipped - rerun the "
            f"script (it will resume and retry only these): {failed_languages}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Harvest Lingua Libre recording durations/source_type by language."
    )
    parser.add_argument(
        "crosswalk_path",
        help="Path to the language crosswalk CSV (needs an 'iso_639_3' column).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "lrl-index/audio_harvest/data/processed/lingua_libre/lingua_libre_harvest.csv"
        ),
    )
    parser.add_argument(
        "--user-agent",
        default=None,
        help=(
            "Your Commons API User-Agent, e.g. "
            "'YourName-LRLIndex/1.0 (you@example.org)'. If omitted, checks "
            "the COMMONS_USER_AGENT environment variable, then prompts "
            "interactively."
        ),
    )
    args = parser.parse_args()
    main(args.crosswalk_path, output_path=args.output, user_agent=args.user_agent)
