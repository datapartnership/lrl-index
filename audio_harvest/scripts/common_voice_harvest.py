"""
common_voice_harvest.py

Week 4 audio harvest: Common Voice - REVISED to pull from actual
DATASET RELEASE statistics (Scripted Speech + Spontaneous Speech),
not the live contribution-platform API used in the first version of
this script.

--------------------------------------------------------------------
DATA SOURCE: cv-dataset GitHub repo, not MDC, not the live stats API
--------------------------------------------------------------------
Common Voice's own `cv-dataset` repo (common-voice/cv-dataset) hosts
structured per-release, per-locale statistics as plain JSON files,
committed to the repo - no auth, no API key, no MDC download quota.
Confirmed live structure (see conversation):

  datasets/scripted-speech/cv-corpus-{version}-{date}.json
  datasets/spontaneous-speech/sps-corpus-{version}-{date}.json

Each file has top-level keys including `locales` (a dict keyed by
locale code) plus corpus-wide totals. Fetched here by downloading the
whole repo as a tarball via codeload.github.com (confirmed to avoid
the much stricter api.github.com unauthenticated rate limit hit
during development) rather than the GitHub Contents API.

--------------------------------------------------------------------
YOUR TRANSCRIBED / UNTRANSCRIBED RULE, PER DATASET TYPE
--------------------------------------------------------------------
Scripted Speech (SCS): EVERY clip counts as transcribed, validated or
not - because the "transcript" is the known prompt sentence the
speaker was asked to read, so it's correct by construction regardless
of whether the RECORDING was validated as a good-quality match.
  transcribed_hours = locale["totalHrs"]   (validated + unvalidated combined)
  (locale["validHrs"] is also recorded as a robustness/context column)

Spontaneous Speech (SPS): only VALIDATED clips count as transcribed,
because here the transcript itself is user-generated after the fact
and may be wrong until community-validated.
  transcribed_hours   = locale["duration"]["validated_hrs"]
  untranscribed_hours = locale["duration"]["total_hrs"] - locale["duration"]["validated_hrs"]

This produces up to 3 output rows per language (long format - one row
per language x source x category), not one wide row per language:
  (lang, common_voice_scripted,    transcribed,   hours=totalHrs)
  (lang, common_voice_spontaneous, transcribed,   hours=validated_hrs)
  (lang, common_voice_spontaneous, untranscribed, hours=total_hrs - validated_hrs)

--------------------------------------------------------------------
LICENSE
--------------------------------------------------------------------
Confirmed CC0 for BOTH dataset types directly from their respective
READMEs in the repo (see conversation) - Tier A for every row.

--------------------------------------------------------------------
SCOPE: ONLY LOCALES IN YOUR CROSSWALK
--------------------------------------------------------------------
Only languages present in your full_language_reference crosswalk are
processed - NOT every locale in the CV release files. See the ADAPT
block in CONFIG: the crosswalk's Common Voice locale column name and
ISO code column name are GUESSED and need to be confirmed/edited
before running. A language can have more than one CV locale (e.g. a
macrolanguage matched to several regional variants) - semicolon-
separated, same convention as the Tier A crosswalk's hf_tag field.

--------------------------------------------------------------------
DEPENDENCIES / SETUP
--------------------------------------------------------------------
pip install pandas
(no requests/API key needed - uses plain tar download + local parsing)
"""
import json
import re
import tarfile
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

# ---- COMMON VOICE PROVENANCE (hardcoded - Common Voice's methodology is
# fully documented, so this never needs a README-guessing fallback like a
# general/unknown dataset would) ----
COMMON_VOICE_PROVENANCE = {
    "common_voice_scripted_transcribed": {
        # Full pipeline: text sourced from a website page -> that page is
        # TRANSLATED into the target language by volunteers -> resulting
        # sentences are given to (possibly different) volunteers to read
        # aloud -> recorded -> a volunteer verifies the audio and
        # transcript actually match.
        "source_type": "website text",
        "who_transcribed": "human volunteer",
        "transcript_exists": True,
    },
    "common_voice_spontaneous_transcribed": {
        # Full pipeline: a question is devised and posed to a volunteer ->
        # volunteer answers via free-speech audio recording (no pre-
        # existing text prompt at all - the speech is generated on the
        # spot) -> a (possibly different) volunteer validates the
        # recording is high quality AND generates the transcript as part
        # of that same step. This is the VALIDATED portion.
        "source_type": "no source - spontaneous speech",
        "who_transcribed": "human volunteer",
        "transcript_exists": True,
    },
    "common_voice_spontaneous_untranscribed": {
        # Same free-speech recording pipeline, but this clip hasn't been
        # through the validate+transcribe step (or didn't pass it) -
        # transcript_exists is False for this unvalidated portion.
        "source_type": "no source - spontaneous speech",
        "who_transcribed": "human volunteer",
        "transcript_exists": False,
    },
}


def get_provenance(registry_key):
    """Looks up a hardcoded CV provenance entry. Never guesses -
    registry_key must be one of the three keys in
    COMMON_VOICE_PROVENANCE above."""
    return COMMON_VOICE_PROVENANCE[registry_key]

# ---- CONFIG ----
CV_DATASET_TARBALL_URL = "https://codeload.github.com/common-voice/cv-dataset/tar.gz/refs/heads/main"

# ---- ADAPT THIS: confirm these match your actual crosswalk file ----
CROSSWALK_FILE = "../../crosswalk/data/processed/full_language_reference.csv"
CROSSWALK_ISO_COL = "iso_639_3"          # column holding the canonical ISO 639-3 code
CROSSWALK_CV_LOCALE_COL = "cv_locale"  # column holding this language's Common Voice locale code(s)

OUTPUT_FILE = "../data/processed/common_voice/common_voice_hours.csv"
SKIPPED_LOG_FILE = "../data/processed/common_voice/common_voice_skipped_locales.csv"

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT = 60

VERSION_PATTERN = re.compile(r"-(\d+(?:\.\d+)?)-")


def _with_retry(fn, description, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"  [retry {attempt + 1}/{max_retries}] {description}: {e} - waiting {wait}s")
            time.sleep(wait)
    print(f"  giving up on {description} after {max_retries} retries")
    return None


def fetch_cv_dataset_tarball():
    """
    Downloads the whole cv-dataset repo as a tarball. Deliberately NOT
    using the GitHub Contents/Trees API here - confirmed during
    development that unauthenticated api.github.com calls hit rate
    limits quickly, while codeload.github.com's tarball endpoint does
    not have the same restriction. Returns an open tarfile.TarFile.
    """
    def _fetch():
        resp = requests.get(CV_DATASET_TARBALL_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.content

    content = _with_retry(_fetch, "download cv-dataset tarball", max_retries=3)
    if content is None:
        raise RuntimeError("could not download cv-dataset tarball - aborting")
    return tarfile.open(fileobj=BytesIO(content), mode="r:gz")


def _version_key(filename):
    """Extracts the numeric version (e.g. 26.0) from a filename like
    cv-corpus-26.0-2026-06-12.json for proper numeric sort - a plain
    string sort would put '9.0' after '26.0' incorrectly."""
    match = VERSION_PATTERN.search(filename)
    return float(match.group(1)) if match else -1.0


def find_latest_release_files(tar):
    """
    Finds the latest non-delta, non-singleword release JSON for each
    dataset type. Returns (scs_member_name, sps_member_name).
    """
    names = tar.getnames()
    scs_candidates = [
        n for n in names
        if "/datasets/scripted-speech/cv-corpus-" in n
        and n.endswith(".json")
        and "delta" not in n
        and "singleword" not in n
    ]
    sps_candidates = [
        n for n in names
        if "/datasets/spontaneous-speech/sps-corpus-" in n
        and n.endswith(".json")
        and "delta" not in n
    ]
    if not scs_candidates or not sps_candidates:
        raise RuntimeError(f"could not find release files - scs={len(scs_candidates)}, sps={len(sps_candidates)}")

    scs_latest = max(scs_candidates, key=_version_key)
    sps_latest = max(sps_candidates, key=_version_key)
    print(f"Latest Scripted Speech release: {scs_latest}")
    print(f"Latest Spontaneous Speech release: {sps_latest}")
    return scs_latest, sps_latest


def load_release_json(tar, member_name):
    f = tar.extractfile(member_name)
    return json.load(f)


# ---- Crosswalk ----
def load_crosswalk_cv_locales():
    """
    Returns a dict {cv_locale: iso_639_3} built from the crosswalk,
    handling semicolon-separated multi-locale entries. Rows with no
    CV locale are skipped (not every language has a Common Voice
    presence).
    """
    df = pd.read_csv(CROSSWALK_FILE)
    if CROSSWALK_CV_LOCALE_COL not in df.columns:
        raise RuntimeError(
            f"Column {CROSSWALK_CV_LOCALE_COL!r} not found in crosswalk - "
            f"available columns: {list(df.columns)}. Update CROSSWALK_CV_LOCALE_COL."
        )

    locale_to_iso = {}
    for _, row in df.iterrows():
        iso = row.get(CROSSWALK_ISO_COL)
        cv_field = row.get(CROSSWALK_CV_LOCALE_COL)
        if pd.isna(iso) or pd.isna(cv_field):
            continue
        for locale in str(cv_field).split(";"):
            locale = locale.strip()
            if locale:
                locale_to_iso[locale] = iso
    print(f"{len(locale_to_iso)} Common Voice locale(s) mapped from crosswalk")
    return locale_to_iso


def build_provenance_url(member_name):
    """Raw GitHub URL for the specific release file used, for citation/audit."""
    path = member_name.split("/", 1)[1]  # strip the cv-dataset-main/ prefix
    return f"https://raw.githubusercontent.com/common-voice/cv-dataset/main/{path}"


def main():
    print("Downloading cv-dataset repo...")
    tar = fetch_cv_dataset_tarball()

    scs_member, sps_member = find_latest_release_files(tar)
    scs_data = load_release_json(tar, scs_member)
    sps_data = load_release_json(tar, sps_member)
    scs_provenance = build_provenance_url(scs_member)
    sps_provenance = build_provenance_url(sps_member)
    scs_version = re.search(r"cv-corpus-([\d.]+)", scs_member).group(1)
    sps_version = re.search(r"sps-corpus-([\d.]+)", sps_member).group(1)

    locale_to_iso = load_crosswalk_cv_locales()

    rows = []
    skipped_rows = []

    for cv_locale, iso in locale_to_iso.items():
        found_in_scs = cv_locale in scs_data["locales"]
        found_in_sps = cv_locale in sps_data["locales"]

        if not found_in_scs and not found_in_sps:
            skipped_rows.append({"cv_locale": cv_locale, "iso_639_3": iso,
                                  "reason": "locale_not_in_either_release"})
            continue

        if found_in_scs:
            loc = scs_data["locales"][cv_locale]
            provenance = get_provenance("common_voice_scripted_transcribed")
            rows.append({
                "iso_639_3": iso, "cv_locale": cv_locale,
                "source": "common_voice_scripted", "category": "transcribed",
                "hours": loc.get("totalHrs"),
                "validated_hours_only": loc.get("validHrs"),  # robustness/context column
                "clips": loc.get("clips"), "speakers": loc.get("users"),
                "license_tier": "A", "license": "CC0-1.0",
                "corpus_version": scs_version,
                **provenance,
            })

        if found_in_sps:
            loc = sps_data["locales"][cv_locale]
            total_hrs = loc.get("duration", {}).get("total_hrs")
            validated_hrs = loc.get("duration", {}).get("validated_hrs")
            unvalidated_hrs = (
                total_hrs - validated_hrs if total_hrs is not None and validated_hrs is not None else None
            )

            transcribed_provenance = get_provenance("common_voice_spontaneous_transcribed")
            rows.append({
                "iso_639_3": iso, "cv_locale": cv_locale,
                "source": "common_voice_spontaneous", "category": "transcribed",
                "hours": validated_hrs,
                "validated_hours_only": validated_hrs,
                "clips": loc.get("clips"), "speakers": loc.get("users"),
                "license_tier": "A", "license": "CC0-1.0",
                "corpus_version": sps_version,
                **transcribed_provenance,
            })

            untranscribed_provenance = get_provenance("common_voice_spontaneous_untranscribed")
            rows.append({
                "iso_639_3": iso, "cv_locale": cv_locale,
                "source": "common_voice_spontaneous", "category": "untranscribed",
                "hours": unvalidated_hrs,
                "validated_hours_only": None,  # not applicable - this row IS the unvalidated portion
                "clips": loc.get("clips"), "speakers": loc.get("users"),
                "license_tier": "A", "license": "CC0-1.0",
                "corpus_version": sps_version,
                **untranscribed_provenance,
            })

    if skipped_rows:
        Path(SKIPPED_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(skipped_rows).to_csv(SKIPPED_LOG_FILE, index=False)
        print(f"\nWrote {len(skipped_rows)} skipped locale(s) to {SKIPPED_LOG_FILE}")

    if not rows:
        print("No rows produced - nothing to write.")
        return

    FINAL_COLUMNS = [
        "iso_639_3", "cv_locale", "source", "category", "hours", "validated_hours_only",
        "clips", "speakers", "license_tier", "license", "corpus_version",
        "source_type", "who_transcribed", "transcript_exists",
    ]
    df = pd.DataFrame(rows)[FINAL_COLUMNS].sort_values(["iso_639_3", "source", "category"])
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {len(df)} row(s) to {OUTPUT_FILE}")
    print(f"\n{df.head(15).to_string(index=False)}")


if __name__ == "__main__":
    main()
