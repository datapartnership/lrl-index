"""
tier_a_harvest_v2.py

Redesigned Tier A single-dataset harvest logic, per the revised spec
from the Week 3 supervisor review. This REPLACES the size-resolution
approach in tier_a_harvest_single_language.py / tier_a_harvest_all_languages.py
with a card-first, decision-tree-driven approach. Text/modality
detection, license/provenance extraction, and the general "never
guess, record unresolved explicitly" philosophy carry over from those
scripts unchanged.

--------------------------------------------------------------------
WHAT'S PULLED FROM THE DATASET CARD / TAGS (not summed from files)
--------------------------------------------------------------------
dataset_id, language_code(s), modalities, task(s), license,
arxiv_id(s), doi(s), total_file_size (from the card's dataset_info
YAML block, if present), num_rows (from dataset_info, if present),
num_hours (from dataset_info, if present - rare, but some audio
dataset cards do state this).

--------------------------------------------------------------------
TWO SEPARATE THINGS THIS SCRIPT TRACKS PER DATASET
--------------------------------------------------------------------
1. TOTAL DATASET SIZE - one figure, resolved via the decision tree
   below (card -> publication/Claude -> file-summing fallback).
2. PER-MODALITY STRUCTURED COUNTS (num_rows for text/image, num_hours
   for audio) - ALWAYS attempted independently, straight from file
   metadata only (Parquet footer row counts; audio container header
   duration). These are NEVER derived or estimated (no file-size/
   bitrate math) - if a file's format doesn't expose this in its
   metadata, it is recorded not_found_in_metadata, full stop.

--------------------------------------------------------------------
TOTAL SIZE DECISION TREE
--------------------------------------------------------------------
1. Card's dataset_info YAML has a total size / dataset_size figure?
   -> use it directly. method = "card_reported".
2. Not on card -> check Hugging Face's own Datasets Server /size API
   (see fetch_datasets_server_size()) - a size figure HF's backend
   computes automatically for most datasets via auto-conversion to
   Parquet, independent of whether the README has a hand-written
   dataset_info block. If available -> method =
   "datasets_server_computed" (or "datasets_server_computed_partial"
   if HF's own response flags the count as partial for a very large
   dataset - still flagged needs_manual_review in that case).
3. Neither on card nor from the Datasets Server -> flag
   size_not_on_card. Does the dataset have an arxiv/doi tag?
   a. Yes -> [API-BASED LOOKUP CURRENTLY PAUSED, see note above
      fetch_arxiv_pdf_text() below - re-enable by wiring
      resolve_size_via_publication() back into resolve_total_size()]
      For now: the publication link(s) are recorded for manual
      review instead, and an interim file-summed number (3b) is
      computed in parallel so there's still something to work with
      while the publication sits in the review queue.
   b. No link, or Claude couldn't find it (once re-enabled) -> fall
      back to summing individual file sizes:
        - Task is ASR-related -> sum ONLY audio file sizes (hours).
          method = "file_summed_asr_audio_only"
        - Otherwise -> sum num_rows and audio hours SEPARATELY (not
          added together into one number - a row and an hour are not
          the same unit). method = "file_summed_unmerged_units" -
          flagged for later unification via a GB-conversion step
          that does not exist yet (out of scope for this script).

Every dataset with a publication link but no card/server size, AND
every dataset resolved via the file-summing fallback with no
publication link either, is flagged needs_manual_review - not
filtered out.

--------------------------------------------------------------------
FILE HANDLING
--------------------------------------------------------------------
Excluded from all size/row/hour calculations: .py, README.md (any
case), .gitattributes.
Flagged, NOT counted: any archive/container format (.zip, .tar,
.tar.gz, .tgz, .7z, .rar) - contents aren't enumerable from the file
listing alone, so including their compressed size would misrepresent
actual content volume. See has_compressed_files_excluded.

--------------------------------------------------------------------
LANGUAGE BREAKDOWN
--------------------------------------------------------------------
Monolingual: the full pipeline above runs unscoped (dataset-level IS
language-level).
Multilingual: language-specific files are isolated first (same
path-segment matching as the existing single-language script), then:
  - README is checked for a language-specific size statement (regex
    heuristic, LOW CONFIDENCE - this is prose-parsing, not a
    Claude-assisted lookup like the publication step; flagged as
    such). If found, used as this language's total size.
  - If not found:
      - Card has a dataset-wide total size -> flag
        language_breakdown_not_found_total_size_present (we know the
        whole dataset's size, just not this language's slice of it)
      - Card has NO dataset-wide total size either -> flag
        language_breakdown_not_found_total_size_missing (compounding
        gap - flagged distinctly so it doesn't look the same as the
        first case on review)
  - Modality-specific num_rows/num_hours are still attempted from
    file metadata, scoped only to the language-isolated files.

--------------------------------------------------------------------
DEPENDENCIES / SETUP REQUIRED
--------------------------------------------------------------------
pip install pyarrow requests mutagen huggingface_hub anthropic pypdf
export ANTHROPIC_API_KEY=sk-ant-...   (required for the publication-lookup step)
export HF_TOKEN=hf_...                 (optional, same as the other Tier A scripts)

--------------------------------------------------------------------
NOT YET VALIDATED AGAINST REAL DATA
--------------------------------------------------------------------
Unlike the rest of this pipeline, this script has NOT been run
against a real Hugging Face dataset from this environment (no network
access to huggingface.co / arxiv.org from this sandbox). The Parquet
footer-read and audio-header-read approaches are standard techniques,
but should be spot-checked against a handful of real datasets -
including at least one ASR dataset, one multi-config dataset, and one
dataset with a paywalled (non-arXiv) DOI - before trusting this at
scale. Treat this as a first implementation to validate, not a
finished, battle-tested harvester like the other Tier A scripts.
"""
import io
import json
import os
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.ipc as ipc
import requests
from huggingface_hub import HfApi, DatasetCard

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from mutagen import File as MutagenFile
except ImportError:
    MutagenFile = None

# ---- CONFIG ----
EXCLUDED_FILENAMES = {"readme.md", ".gitattributes"}
EXCLUDED_EXTENSIONS = {".py"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".rar", ".gz.tar"}

ASR_TASK_CATEGORIES = {"automatic-speech-recognition"}  # expand if other namings show up in practice

TEXT_EXTENSIONS = {".txt", ".csv", ".tsv", ".json", ".jsonl", ".parquet", ".xml", ".arrow"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

# Configurable, per your instruction - tune once real size distribution is visible.
MANUAL_REVIEW_SIZE_THRESHOLD_BYTES = None  # e.g. set to 50_000_000 (50MB) once you have a real distribution to judge from
MAX_FILES_TO_PROBE_PER_MODALITY = 20  # cap per dataset per modality for get_modality_structured_counts - see its docstring for why this yields a PARTIAL sum, not an estimate, when a modality exceeds this

ANTHROPIC_MODEL = "claude-sonnet-4-6"
AUDIO_PROBE_BYTES = 2_000_000  # first ~2MB - enough to cover header/duration metadata for WAV/FLAC/OGG/most MP3s with a Xing header

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 5

api = HfApi(token=os.environ.get("HF_TOKEN"))
_anthropic_client = anthropic.Anthropic() if anthropic and os.environ.get("ANTHROPIC_API_KEY") else None


def _with_retry(fn, description, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"    [retry {attempt + 1}/{max_retries}] {description}: {e} - waiting {wait}s")
            time.sleep(wait)
    print(f"    giving up on {description} after {max_retries} retries")
    return None


# ---- FILE-LEVEL HELPERS ----
def is_excluded_file(filename):
    """.py, README.md (any case), .gitattributes - never counted toward any size/row/hour figure."""
    lower = filename.lower()
    if lower in EXCLUDED_FILENAMES or lower.split("/")[-1] in EXCLUDED_FILENAMES:
        return True
    return any(lower.endswith(ext) for ext in EXCLUDED_EXTENSIONS)


def is_archive_file(filename):
    return any(filename.lower().endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def get_file_modality(filename):
    lower = filename.lower()
    if any(lower.endswith(ext) for ext in TEXT_EXTENSIONS):
        return "text"
    if any(lower.endswith(ext) for ext in AUDIO_EXTENSIONS):
        return "audio"
    if any(lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
        return "image"
    return None


def list_file_extensions(file_list):
    """Same approach as the existing Tier A scripts: literal extensions
    present, not a category. Unknown extensions fall back to their raw
    suffix rather than being dropped."""
    if not file_list:
        return "no_files"
    known = TEXT_EXTENSIONS | AUDIO_EXTENSIONS | IMAGE_EXTENSIONS | ARCHIVE_EXTENSIONS
    exts = set()
    for fname in file_list:
        lower = fname.lower()
        matched = None
        for ext in known:
            if lower.endswith(ext) and (matched is None or len(ext) > len(matched)):
                matched = ext
        exts.add(matched if matched else ("." + lower.rsplit(".", 1)[-1] if "." in lower else "(no_extension)"))
    return ";".join(sorted(exts))


# ---- FILE-METADATA-ONLY ROW / HOUR EXTRACTION (never derived/estimated) ----
def get_parquet_column_names(file_url, tail_bytes=1_000_000):
    """
    Reads ONLY the Parquet footer to get the file's SCHEMA (column
    names) - the same footer read as get_parquet_row_count(), just
    returning a different piece of the same metadata. Column names are
    part of a Parquet file's structural metadata, stored in the
    footer alongside row-group statistics - reading them requires no
    more access than reading num_rows does, and involves NO reading of
    actual row/column DATA.

    Used to detect datasets that store a language identifier as a
    DATA COLUMN (e.g. a column literally named "iso3", one row per
    audio clip) rather than as separate per-language files/folders.
    This pipeline does not read column DATA to get per-language row
    counts for that case (see LANGUAGE_COLUMN_NAME_CANDIDATES below
    and its use in resolve_language_specific_size) - only the fact
    that such a column EXISTS is metadata-derivable; the per-language
    breakdown itself is not, and is flagged rather than computed.

    Returns a list of column names (strings), or None if the read
    fails for any reason.
    """
    def _read():
        resp = requests.get(file_url, headers={"Range": f"bytes=-{tail_bytes}"}, timeout=30)
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"unexpected status {resp.status_code}")
        return pq.ParquetFile(io.BytesIO(resp.content)).schema.names

    return _with_retry(_read, f"read parquet schema for {file_url}", max_retries=2)


LANGUAGE_COLUMN_NAME_CANDIDATES = {
    "iso3", "iso_639_3", "iso639_3", "language", "lang", "language_code",
    "lang_code", "locale", "language_id", "lang_id",
}


def detect_language_stored_as_column(siblings, dataset_id):
    """
    Checks whether any Parquet file in the dataset has a column whose
    name matches a common language-identifier naming pattern (see
    LANGUAGE_COLUMN_NAME_CANDIDATES) - metadata-only (schema names
    from the footer), no row data read. Only checks the FIRST Parquet
    file found (schema is normally uniform across a dataset's shards -
    checking every shard would be redundant metadata reads for no
    added information in the normal case).

    Returns the matched column name if found, else None. This does
    NOT attempt to compute per-language counts from that column - see
    resolve_language_specific_size() for how this is used to flag,
    not resolve, this case.
    """
    parquet_files = [s.rfilename for s in siblings if s.rfilename.lower().endswith(".parquet")]
    if not parquet_files:
        return None

    base_url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/"
    columns = get_parquet_column_names(base_url + parquet_files[0])
    if not columns:
        return None

    lower_columns = {c.lower(): c for c in columns}
    for candidate in LANGUAGE_COLUMN_NAME_CANDIDATES:
        if candidate in lower_columns:
            return lower_columns[candidate]
    return None


class _HTTPRangeFile:
    """
    Minimal seekable, read-only file-like object over HTTP range
    requests - lets pyarrow lazily read only the specific byte ranges
    it actually needs (an Arrow IPC file's trailing footer, then each
    record batch's small header at its own offset - never the actual
    column data buffers), without depending on fsspec. fsspec's HTTP
    filesystem was tried earlier in this pipeline for Parquet footer
    reads and had unreliable behavior across environments (see
    get_parquet_row_count()'s docstring/history) - this small custom
    class sidesteps that by implementing only the minimal seek/tell/
    read interface pyarrow's IPC reader actually needs, nothing more.

    Every read() call is one HTTP range request - for a typical Arrow
    file with few record batches (often just one per shard), this
    means a small, bounded number of requests (a footer read plus one
    read per batch), not a full-file download.
    """
    def __init__(self, url):
        self._url = url
        self._pos = 0
        self.closed = False
        head = requests.head(url, timeout=30)
        head.raise_for_status()
        self._size = int(head.headers["Content-Length"])

    def close(self):
        self.closed = True

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        return self._pos

    def tell(self):
        return self._pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n is None or n < 0:
            end = self._size - 1
        else:
            end = min(self._pos + n, self._size) - 1
        if self._pos > end:
            return b""
        resp = requests.get(self._url, headers={"Range": f"bytes={self._pos}-{end}"}, timeout=30)
        resp.raise_for_status()
        data = resp.content
        self._pos += len(data)
        return data


def get_arrow_row_count(file_url):
    """
    Reads an Arrow IPC file's (.arrow/.feather) row count via its
    footer and per-record-batch headers - metadata only, no column
    data ever read. Unlike Parquet, an Arrow IPC footer alone only
    gives each record batch's byte OFFSET, not its row count directly
    - the row count lives in that batch's own small message header at
    that offset. pyarrow's ipc.open_file() handles this transparently
    given a seekable source: it reads the footer first, then seeks to
    and reads each batch's header as needed - see _HTTPRangeFile for
    how that's done via HTTP range requests instead of a full download.

    Returns an int row count (summed across all record batches in the
    file), or None if the read/parse fails for any reason (never
    guesses, never partial-counts).
    """
    def _read():
        f = _HTTPRangeFile(file_url)
        reader = ipc.open_file(f)
        return sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))

    return _with_retry(_read, f"read arrow footer for {file_url}", max_retries=2)


def get_parquet_row_count(file_url, tail_bytes=1_000_000):
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

    If the server returns the whole file (some servers ignore Range
    headers and return 200 with the full body instead of 206 partial
    content), this still works correctly - just less efficient for
    that one file, not a correctness problem.

    Returns an int row count, or None if the read/parse fails for any
    reason (never guesses, never partial-counts).
    """
    def _read():
        resp = requests.get(file_url, headers={"Range": f"bytes=-{tail_bytes}"}, timeout=30)
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"unexpected status {resp.status_code}")
        return pq.ParquetFile(io.BytesIO(resp.content)).metadata.num_rows

    return _with_retry(_read, f"read parquet footer for {file_url}", max_retries=2)


def get_audio_duration_hours_from_metadata(file_url):
    """
    Reads only the first AUDIO_PROBE_BYTES of an audio file and asks
    mutagen to extract duration from its container header metadata -
    NOT a full download+decode, and NOT a file-size/bitrate estimate.
    Works reliably for WAV, FLAC, OGG (duration is explicit in their
    headers) and MP3s that have a Xing/VBRI header (common for
    encoder-written VBR/CBR files, not guaranteed for every MP3).

    Returns hours (float), or None if duration isn't recoverable from
    the probed header bytes - this is intentionally NOT retried with
    progressively larger downloads or a full-file fallback, per the
    instruction to never derive/estimate what isn't in metadata.
    """
    if MutagenFile is None:
        print("    mutagen not installed - cannot read audio metadata. pip install mutagen.")
        return None

    def _probe():
        resp = requests.get(file_url, headers={"Range": f"bytes=0-{AUDIO_PROBE_BYTES - 1}"}, timeout=30)
        if resp.status_code not in (200, 206):
            return None
        audio = MutagenFile(io.BytesIO(resp.content))
        if audio is not None and getattr(audio, "info", None) is not None and hasattr(audio.info, "length"):
            return audio.info.length / 3600.0
        return None

    return _with_retry(_probe, f"probe audio duration for {file_url}", max_retries=2)


def get_modality_structured_counts(siblings, dataset_id, target_files=None, max_files_per_modality=MAX_FILES_TO_PROBE_PER_MODALITY):
    """
    For each modality present (among target_files if given, i.e. a
    language-isolated subset, else all siblings), attempts to get
    num_rows (text/image, via Parquet footers only) or num_hours
    (audio, via header metadata only). Never sums file sizes as a
    substitute count, and NEVER extrapolates or scales a sample up to
    estimate a full-dataset total - that would be a derived/estimated
    number, which contradicts this pipeline's explicit rule that
    num_rows/num_hours are recorded ONLY when directly read from file
    metadata, nothing else.

    EFFICIENCY NOTE: a dataset can have thousands of files per
    modality - probing every single one at crosswalk scale would mean
    an enormous number of HTTP requests per dataset. If a modality has
    more than max_files_per_modality in-scope files, only the first
    max_files_per_modality are probed, and the result is reported as a
    PARTIAL sum (sum of just the probed files, not scaled up) with a
    method flag making that explicit - not a substitute for the true
    total. Set max_files_per_modality to None to disable the cap
    entirely and probe every file (only advisable for small datasets
    or targeted validation runs, not a full crosswalk pass).

    Returns a dict like:
      {"text": {"num_rows": 12345, "method": "found_in_metadata", "file_types": ".parquet"},
       "audio": {"num_hours": None, "method": "not_found_in_metadata", "file_types": ".mp3;.wav"},
       "image": {"num_rows": None, "method": "no_image_files"}}
    Only includes keys for modalities that have at least one
    in-scope, non-excluded, non-archive file. file_types is computed
    from EVERY in-scope file for that modality (not limited to the
    sampled subset used for row/hour probing - see
    max_files_per_modality below), since listing extensions is just
    filename string matching, not an extra network call.
    """
    files = target_files if target_files is not None else [s.rfilename for s in siblings]
    files = [f for f in files if not is_excluded_file(f) and not is_archive_file(f)]

    by_modality = {"text": [], "audio": [], "image": []}
    for fname in files:
        modality = get_file_modality(fname)
        if modality:
            by_modality[modality].append(fname)

    result = {}
    base_url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/"

    for modality, flist in by_modality.items():
        if not flist:
            continue

        total_file_count = len(flist)
        if max_files_per_modality is not None and total_file_count > max_files_per_modality:
            probe_list = flist[:max_files_per_modality]
            is_sampled = True
        else:
            probe_list = flist
            is_sampled = False

        # File types are recorded from the FULL in-scope file list for
        # this modality (flist), not just probe_list - listing
        # extensions is just string matching on filenames already in
        # hand, no extra network calls, so there's no reason to limit
        # it to the same sample used for the (expensive) row/hour
        # probing above.
        file_types = list_file_extensions(flist)

        if modality in ("text", "image"):
            total_rows, any_resolved = 0, False
            for fname in probe_list:
                lower_fname = fname.lower()
                if lower_fname.endswith(".parquet"):
                    count = get_parquet_row_count(base_url + fname)
                elif lower_fname.endswith(".arrow") or lower_fname.endswith(".feather"):
                    count = get_arrow_row_count(base_url + fname)
                else:
                    continue  # only Parquet/Arrow expose row count as metadata - other text/image formats: not_found
                if count is not None:
                    total_rows += count
                    any_resolved = True
            if any_resolved and is_sampled:
                result[modality] = {
                    "num_rows": total_rows,  # PARTIAL - only the probed files, not scaled to the full total
                    "method": f"partial_metadata_sample_{len(probe_list)}_of_{total_file_count}_files",
                    "file_types": file_types,
                }
            else:
                result[modality] = {
                    "num_rows": total_rows if any_resolved else None,
                    "method": "found_in_metadata" if any_resolved else "not_found_in_metadata",
                    "file_types": file_types,
                }

        elif modality == "audio":
            total_hours, any_resolved = 0.0, False
            for fname in probe_list:
                hours = get_audio_duration_hours_from_metadata(base_url + fname)
                if hours is not None:
                    total_hours += hours
                    any_resolved = True
            if any_resolved and is_sampled:
                result[modality] = {
                    "num_hours": total_hours,  # PARTIAL - only the probed files, not scaled to the full total
                    "method": f"partial_metadata_sample_{len(probe_list)}_of_{total_file_count}_files",
                    "file_types": file_types,
                }
            else:
                result[modality] = {
                    "num_hours": total_hours if any_resolved else None,
                    "method": "found_in_metadata" if any_resolved else "not_found_in_metadata",
                    "file_types": file_types,
                }

    return result


# ---- CARD-LEVEL FIELD EXTRACTION ----
def extract_card_fields(dataset_id, tags, card_data):
    """
    Pulls dataset_id, language_code(s), modalities, task(s), license,
    arxiv_id(s), doi(s), and card-level size figures (total size,
    num_rows, num_hours if the card's dataset_info YAML states them)
    - all from tags or the card's structured YAML block, never from
    summing files. card_data is the dict form of a DatasetCard's
    .data (may be None if no card / unparseable card).
    """
    language_codes = [t.replace("language:", "") for t in tags if t.startswith("language:")]
    modality_tags = [t.replace("modality:", "") for t in tags if t.startswith("modality:")]
    task_tags = [t.replace("task_categories:", "") for t in tags if t.startswith("task_categories:")]
    license_tags = [t.replace("license:", "") for t in tags if t.startswith("license:")]
    arxiv_ids = [t.replace("arxiv:", "") for t in tags if t.startswith("arxiv:")]
    doi_ids = [t.replace("doi:", "") for t in tags if t.startswith("doi:")]

    card_total_size_bytes = None
    card_num_rows = None
    card_num_hours = None  # rare on cards, but checked for completeness

    if card_data:
        dataset_info = card_data.get("dataset_info")
        info_list = dataset_info if isinstance(dataset_info, list) else ([dataset_info] if dataset_info else [])
        total_size, total_rows, found_any = 0, 0, False
        for info in info_list:
            if not isinstance(info, dict):
                continue
            if "dataset_size" in info:
                total_size += info["dataset_size"]
                found_any = True
            for split in info.get("splits", []) or []:
                if "num_examples" in split:
                    total_rows += split["num_examples"]
                    found_any = True
        if found_any:
            card_total_size_bytes = total_size or None
            card_num_rows = total_rows or None

    return {
        "dataset_id": dataset_id,
        "language_codes": ";".join(language_codes) if language_codes else None,
        "modalities": ";".join(modality_tags) if modality_tags else None,
        "tasks": ";".join(task_tags) if task_tags else None,
        "license": ";".join(license_tags) if license_tags else "license_unstated",
        "arxiv_ids": ";".join(arxiv_ids) if arxiv_ids else None,
        "doi_ids": ";".join(doi_ids) if doi_ids else None,
        "card_total_size_bytes": card_total_size_bytes,
        "card_num_rows": card_num_rows,
        "card_num_hours": card_num_hours,
        "task_list_raw": task_tags,  # kept for the ASR check downstream
    }


def fetch_card_data(dataset_id):
    """Loads a dataset's README card and returns its parsed YAML
    front-matter as a dict, or None if there's no card / it fails to
    parse. Never raises - a missing/broken card just means card-level
    fields stay unresolved, not a harvest failure."""
    def _load():
        card = DatasetCard.load(dataset_id)
        return card.data.to_dict()

    return _with_retry(_load, f"load card for {dataset_id}", max_retries=2)


DATASETS_SERVER_SIZE_URL = "https://datasets-server.huggingface.co/size?dataset={dataset_id}"


ACCESS_ISSUE_KEYWORDS = {
    "need to manually accept dataset access": ["gated", "access request", "must agree", "accept the"],
    "private dataset": ["private", "does not exist", "repository not found"],
}
# Every possible return value of classify_access_issue(), used to
# detect (elsewhere in this module) when a flag_reason represents a
# genuine access problem rather than a data-availability one - see
# resolve_language_specific_size()'s final fallback branch.
ACCESS_ISSUE_REASONS = set(ACCESS_ISSUE_KEYWORDS.keys()) | {"other"}


def classify_access_issue(response_text):
    """
    Best-effort classification of WHY a dataset couldn't be accessed,
    based on keyword matching against HF's own error response text.
    Not exhaustive - HF's exact wording can vary or change - falls
    back to "other" when nothing recognizable matches. This is a
    convenience label for manual review triage, not a guaranteed-
    accurate diagnosis.
    """
    text_lower = (response_text or "").lower()
    for reason, keywords in ACCESS_ISSUE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return reason
    return "other"


def fetch_datasets_server_size(dataset_id):
    """
    Queries Hugging Face's Datasets Server /size endpoint - the same
    backend that powers the "Number of rows" / "Total file size" block
    shown on a dataset's own page. This is populated automatically by
    HF's backend (via auto-conversion to Parquet) for most datasets,
    REGARDLESS of whether the README has a hand-written dataset_info
    YAML block - it's a genuinely separate, often-more-available
    source than fetch_card_data()/extract_card_fields()'s YAML parsing.

    Sends HF_TOKEN as a bearer token if set (same env var used
    elsewhere in this pipeline) - needed for gated datasets your
    account has access to. A 501 (dataset not supported by the
    viewer, e.g. script-based datasets) or 404 is treated as an
    immediate "nothing available" result - not transient failures,
    so retrying wastes time without changing the outcome. A 401/403
    is ALSO treated as immediate/non-retried, but additionally
    classified via classify_access_issue() so the caller can surface
    WHY access failed (gated vs. private vs. unclear).

    Returns a tuple (data_or_None, access_issue_reason_or_None):
      - data: dict {"num_bytes_original_files", "num_rows",
        "num_bytes_parquet_files", "is_partial"}, or None if
        unavailable for any reason.
      - access_issue_reason: set only when the failure was a 401/403
        (see classify_access_issue()); None for 404/501/success/other
        failures, since those aren't an "access" problem specifically.

    is_partial reflects HF's own documented caveat: for very large
    datasets, the reported num_rows/bytes can be a PARTIAL count, not
    the true total (HF's UI separately shows an "estimated" full-size
    figure in that case, which this endpoint's response does not
    itself provide) - callers must not treat a partial count as exact.
    """
    access_issue_holder = {"reason": None}

    def _fetch():
        headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"} if os.environ.get("HF_TOKEN") else {}
        resp = requests.get(DATASETS_SERVER_SIZE_URL.format(dataset_id=dataset_id), headers=headers, timeout=30)
        if resp.status_code == 404:
            # IMPORTANT: HF's datasets-server uses 404 for BOTH "this
            # dataset genuinely doesn't exist / isn't viewer-compatible"
            # AND "this dataset exists but is private/gated and you
            # don't have access" - confirmed via a real response with
            # header x-error-code: ExternalAuthenticatedError and body
            # "...is not accessible with the current credentials
            # (private or gated)...". These need different handling:
            # the access-restricted case should be classified and
            # reported, not silently treated as "nothing to see here."
            if resp.headers.get("x-error-code") == "ExternalAuthenticatedError":
                access_issue_holder["reason"] = classify_access_issue(resp.text)
                print(f"    datasets-server returned 404 for {dataset_id} but with "
                      f"x-error-code: ExternalAuthenticatedError - this is an access "
                      f"restriction, not a genuine 'not found'. Classified as: "
                      f"{access_issue_holder['reason']} - skipping, not retrying")
                return None
            return None  # genuine "not viewer-compatible" - not a retry-able failure
        if resp.status_code == 501:
            print(f"    datasets-server has no support for {dataset_id} (501 Not Implemented - "
                  f"likely a script-based dataset) - skipping, not retrying")
            return None
        if resp.status_code in (401, 403):
            access_issue_holder["reason"] = classify_access_issue(resp.text)
            print(f"    datasets-server returned {resp.status_code} for {dataset_id} - "
                  f"classified as: {access_issue_holder['reason']} - skipping, not retrying")
            return None
        resp.raise_for_status()
        return resp.json()

    data = _with_retry(_fetch, f"fetch datasets-server size for {dataset_id}", max_retries=2)
    if not data or "size" not in data or "dataset" not in data["size"]:
        if data is not None:
            # Request succeeded (200) but the response didn't have the
            # expected shape - print it so this isn't a silent, un-
            # diagnosable failure. Common real cause: the dataset is
            # still "pending" processing on HF's backend (present in
            # the response's own "pending"/"failed" arrays) rather than
            # an access problem at all - worth seeing the raw response
            # to tell those apart.
            print(f"    datasets-server returned 200 for {dataset_id} but with an unexpected "
                  f"response shape - raw response: {data}")
        return None, access_issue_holder["reason"]

    d = data["size"]["dataset"]
    return {
        "num_bytes_original_files": d.get("num_bytes_original_files"),
        "num_bytes_parquet_files": d.get("num_bytes_parquet_files"),
        "num_rows": d.get("num_rows"),
        "is_partial": d.get("partial", False),  # field name per HF docs; defaults False if absent
    }, None


# ---- PUBLICATION LINKS (API lookup PAUSED - see resolve_total_size) ----
def build_publication_links(card_fields):
    """
    Builds human-clickable links from arxiv_ids/doi_ids for manual
    review, without fetching or analyzing their content. Returns None
    if neither is present.
    """
    links = []
    if card_fields["arxiv_ids"]:
        links += [f"https://arxiv.org/abs/{a}" for a in card_fields["arxiv_ids"].split(";")]
    if card_fields["doi_ids"]:
        links += [f"https://doi.org/{d}" for d in card_fields["doi_ids"].split(";")]
    return ";".join(links) if links else None


# ---- PAUSED: API-BASED PUBLICATION SIZE LOOKUP ----
# The functions below (fetch_arxiv_pdf_text, fetch_doi_landing_text,
# ask_claude_for_dataset_size, resolve_size_via_publication) implement
# a real Anthropic API call to read a publication and extract its
# stated dataset size. This path is CURRENTLY NOT CALLED from
# resolve_total_size() - paused to avoid API costs while the rest of
# the pipeline is being validated. The functions are left in place,
# fully working, for easy re-enabling later: swap the manual-review
# branch in resolve_total_size() back to calling
# resolve_size_via_publication() when ready to turn this back on.
def fetch_arxiv_pdf_text(arxiv_id):
    """
    Downloads an arXiv paper's PDF and extracts its text. arXiv PDFs
    are freely downloadable at a predictable URL with no auth needed.
    Returns extracted text, or None on failure.
    """
    from pypdf import PdfReader

    def _fetch_and_extract():
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        reader = PdfReader(io.BytesIO(resp.content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return _with_retry(_fetch_and_extract, f"fetch/extract arXiv PDF {arxiv_id}", max_retries=2)


def fetch_doi_landing_text(doi):
    """
    Best-effort fetch of a DOI's landing page text. LOWER CONFIDENCE
    than the arXiv path: many publishers paywall full text behind the
    landing page (which may only expose an abstract), and extraction
    quality varies enormously by publisher site structure. A short or
    abstract-only result is a real risk here - flagged accordingly by
    the caller rather than silently trusted as "the full paper."
    """
    def _fetch():
        resp = requests.get(f"https://doi.org/{doi}", timeout=30, allow_redirects=True)
        resp.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", resp.text)  # crude tag strip - not a real HTML parser
        return re.sub(r"\s+", " ", text)

    return _with_retry(_fetch, f"fetch DOI landing page {doi}", max_retries=2)


def ask_claude_for_dataset_size(publication_text, dataset_id):
    """
    Real Anthropic API call. Asks Claude whether the given publication
    text states the dataset's total size, and to return only that
    figure verbatim if so. Requires ANTHROPIC_API_KEY to be set.

    Returns (size_text_or_None, method_flag). method_flag is one of:
      "found_via_publication", "claude_could_not_find_in_publication",
      "anthropic_api_not_configured"
    """
    if _anthropic_client is None:
        print("    ANTHROPIC_API_KEY not set / anthropic package not installed - skipping publication lookup.")
        return None, "anthropic_api_not_configured"

    prompt = (
        f"Below is text extracted from a research paper that introduces or describes "
        f"the dataset '{dataset_id}'. Does this text state the TOTAL SIZE of the dataset "
        f"(e.g. in GB, TB, number of hours, or number of examples/rows/sentences)?\n\n"
        f"If yes, respond with ONLY the exact size figure and unit as stated in the paper "
        f"(e.g. \"1.2 TB\" or \"500,000 examples\"), nothing else.\n"
        f"If no such figure is stated, respond with exactly: NOT_FOUND\n\n"
        f"Paper text:\n{publication_text[:100000]}"
    )

    def _call():
        response = _anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    answer = _with_retry(_call, "Claude publication size lookup", max_retries=2)
    if answer is None:
        return None, "claude_could_not_find_in_publication"
    if answer == "NOT_FOUND":
        return None, "claude_could_not_find_in_publication"
    return answer, "found_via_publication"


def resolve_size_via_publication(card_fields, dataset_id):
    """
    Tries arXiv first (higher confidence, reliably fetchable), then
    DOI as a fallback (lower confidence - see fetch_doi_landing_text).
    Returns (size_text_or_None, method_flag).
    """
    if card_fields["arxiv_ids"]:
        first_arxiv = card_fields["arxiv_ids"].split(";")[0]
        text = fetch_arxiv_pdf_text(first_arxiv)
        if text:
            return ask_claude_for_dataset_size(text, dataset_id)

    if card_fields["doi_ids"]:
        first_doi = card_fields["doi_ids"].split(";")[0]
        text = fetch_doi_landing_text(first_doi)
        if text:
            size_text, method = ask_claude_for_dataset_size(text, dataset_id)
            if method == "found_via_publication":
                method = "found_via_publication_doi_lower_confidence"
            return size_text, method

    return None, "no_publication_link_available"


# ---- FILE-SUMMING FALLBACK (last resort, always flagged) ----
def sum_files_fallback(siblings, task_list, target_files=None):
    """
    Last-resort size determination when nothing is on the card and no
    publication size was found. Returns a dict describing what was
    summed and how - never merges rows and hours into one number.
    """
    files = target_files if target_files is not None else [s.rfilename for s in siblings]
    sibling_by_name = {s.rfilename: s for s in siblings}
    in_scope = [f for f in files if not is_excluded_file(f) and not is_archive_file(f)]
    has_archive = any(is_archive_file(f) for f in files)

    is_asr = bool(ASR_TASK_CATEGORIES & set(task_list))

    if is_asr:
        audio_files = [f for f in in_scope if get_file_modality(f) == "audio"]
        audio_bytes = sum((sibling_by_name[f].size or 0) for f in audio_files if f in sibling_by_name)
        return {
            "method": "file_summed_asr_audio_only",
            "summed_audio_bytes": audio_bytes,
            "summed_rows": None,
            "has_compressed_files_excluded": has_archive,
        }

    # Non-ASR: sum rows (text/image files' byte sizes as a proxy - NOTE
    # this is still a BYTE size sum, not a row count; row counts
    # remain metadata-only per get_modality_structured_counts) and
    # audio hours separately, never combined.
    text_image_files = [f for f in in_scope if get_file_modality(f) in ("text", "image")]
    audio_files = [f for f in in_scope if get_file_modality(f) == "audio"]
    text_image_bytes = sum((sibling_by_name[f].size or 0) for f in text_image_files if f in sibling_by_name)
    audio_bytes = sum((sibling_by_name[f].size or 0) for f in audio_files if f in sibling_by_name)

    return {
        "method": "file_summed_unmerged_units_needs_gb_conversion",
        "summed_text_image_bytes": text_image_bytes,
        "summed_audio_bytes": audio_bytes,
        "has_compressed_files_excluded": has_archive,
    }


# ---- TOTAL SIZE DECISION TREE ----
def resolve_total_size(card_fields, siblings, dataset_id):
    """
    Runs the decision tree from the module docstring, WITH THE
    API-BASED PUBLICATION LOOKUP CURRENTLY PAUSED (see note above
    fetch_arxiv_pdf_text). Instead of calling Claude to read the
    publication, a dataset with a publication link but no card-level
    size gets:
      - publication_links populated (arXiv/DOI URLs, for a human to
        check manually)
      - needs_manual_review = True
      - an INTERIM file-summed number computed anyway (via
        sum_files_fallback), so there's still something to work with
        while the publication sits in the manual-review queue - not
        left blank just because a possibly-better number might exist
        in the paper.

    ORDER: Datasets Server API is now checked FIRST, card YAML SECOND
    - the Datasets Server figure is HF's own auto-computed value from
    the actual converted Parquet files, which is at least as reliable
    as (and often more current than) a hand-written card figure that
    may go stale if a dataset is updated without the card being
    edited. The card YAML is still checked as a fallback for datasets
    the Datasets Server can't process (gated, private, script-based).

    Returns a dict with total_size_bytes (or None), total_size_method
    (flag), total_num_rows (or None), publication_links (or None),
    needs_manual_review, and flag_reason - a short human-readable
    string explaining WHY needs_manual_review is True (None when it's
    False). See FLAG_REASON values used throughout this function.
    """
    # Step 1: HF's own auto-computed size, via the Datasets Server API
    # - works for most datasets even when the README has no hand-
    # written dataset_info YAML (see fetch_datasets_server_size()).
    server_size, access_issue_reason = fetch_datasets_server_size(dataset_id)
    if server_size and server_size["num_bytes_original_files"] is not None:
        is_partial = server_size["is_partial"]
        return {
            "total_size_bytes": server_size["num_bytes_original_files"],
            "total_size_method": "datasets_server_computed_partial" if is_partial else "datasets_server_computed",
            "total_num_rows": server_size["num_rows"],
            "publication_links": None,
            "needs_manual_review": is_partial,
            "flag_reason": "partial_count_from_datasets_server" if is_partial else None,
        }

    # Step 2: card's hand-written dataset_info YAML, if the Datasets
    # Server didn't have (or couldn't access) a figure.
    if card_fields["card_total_size_bytes"] is not None:
        return {
            "total_size_bytes": card_fields["card_total_size_bytes"],
            "total_size_method": "card_reported",
            "total_num_rows": card_fields["card_num_rows"],
            "publication_links": None,
            "needs_manual_review": False,
            "flag_reason": None,
        }

    publication_links = build_publication_links(card_fields)
    fallback = sum_files_fallback(siblings, card_fields["task_list_raw"])
    result = {"total_size_bytes": None, "total_size_method": fallback["method"], "total_num_rows": None,
              "publication_links": publication_links, **fallback}

    # flag_reason priority: an access issue (gated/private/other) is
    # the most useful root-cause explanation when present, since it
    # explains WHY neither of the two good sources worked - shown even
    # if a publication link also exists, since the access issue is
    # still the more actionable thing for a reviewer to look at first.
    if access_issue_reason:
        flag_reason = access_issue_reason
        needs_review = True
    elif publication_links:
        # A publication exists that MIGHT state a better size than our
        # file-summed interim number - always flagged for review,
        # regardless of the size threshold below, since a human
        # checking the paper is a separate concern from "is this
        # dataset small enough to warrant scrutiny."
        flag_reason = "check_publication_for_size"
        needs_review = True
    elif MANUAL_REVIEW_SIZE_THRESHOLD_BYTES is not None:
        summed = fallback.get("summed_audio_bytes", 0) + fallback.get("summed_text_image_bytes", 0)
        needs_review = summed < MANUAL_REVIEW_SIZE_THRESHOLD_BYTES
        flag_reason = "file_summed_below_review_threshold" if needs_review else None
    else:
        needs_review = True  # no threshold set yet - flag ALL file-summed datasets until one is configured
        flag_reason = "file_summed_no_threshold_configured"

    result["needs_manual_review"] = needs_review
    result["flag_reason"] = flag_reason
    return result


# ====================================================================
# LANGUAGE BREAKDOWN LAYER
# ====================================================================
# Monolingual: the total-size decision tree above runs unscoped -
# dataset-level IS language-level, nothing extra needed.
#
# Multilingual: language-specific files are isolated first (same
# path-segment matching used in the original single-language script -
# split each file's path on common separators, check whether the
# language code appears as a COMPLETE segment), then:
#   1. README body text (not the YAML card, the actual prose) is
#      searched for a line mentioning this language near a size-like
#      pattern (GB/TB/MB/hours/rows/examples). LOW CONFIDENCE - this
#      is regex prose-parsing, not a Claude-assisted lookup like the
#      publication step. Always flagged needs_manual_review=True when
#      matched, since a human needs to confirm the match is real and
#      actually refers to total size (not e.g. a per-file size, a
#      different metric, or an unrelated number that happens to sit
#      near the language name).
#   2. If nothing found in the README, and isolation DID find real
#      per-language files: this is flagged as
#      "language_breakdown_found_no_summary_size_metric" - distinct
#      from "no breakdown found at all," since the per-language data
#      genuinely exists (isolated_file_count > 0 proves it). No
#      overall size/row figure is reported here on purpose: summing
#      isolated files' raw byte sizes was tried and deliberately
#      rejected, since a byte total isn't a meaningful stand-in for a
#      row count, and conflating the two would misrepresent what's
#      actually known. The real per-modality row/hour counts, when
#      extractable at all (Parquet footers, audio headers), are
#      reported separately via get_modality_structured_counts() - this
#      flag only means "no single total figure to report here," it
#      does NOT mean the per-modality columns are necessarily empty.
#   3. If isolation found NOTHING and the README had nothing either:
#      flag distinctly depending on whether the dataset has ANY
#      card/server-level total size at all - "we know the whole
#      dataset's size, just not this language's slice" is a different,
#      less-bad situation than "we don't know either number," and
#      collapsing them into one flag would hide that distinction from
#      whoever does the manual review. If the dataset-wide resolution
#      itself hit an
#      access issue (gated/private/other), that root cause is
#      surfaced here instead of the generic message - see
#      ACCESS_ISSUE_REASONS.
# ====================================================================
LANGUAGE_SIZE_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:GB|TB|MB|KB|hours?|hrs?|rows?|examples?|sentences?|utterances?)",
    re.IGNORECASE,
)


def classify_linguality(language_codes_str):
    """Same classification as the original single-language script,
    adapted to the semicolon-joined string format used in this
    script's card fields. Returns "monolingual", "multilingual", or
    "no_language_tag"."""
    if not language_codes_str:
        return "no_language_tag"
    codes = language_codes_str.split(";")
    if len(codes) == 1:
        return "monolingual"
    return "multilingual"


def isolate_language_files(siblings, lang_code):
    """
    Same path-segment matching as the original single-language
    script's find_language_segment_matches(): splits each file's path
    on common separators (/, _, -, .) and keeps files where the
    language code appears as a COMPLETE segment - not a substring
    match. Catches Wikipedia-style per-language folders and OPUS-style
    language-pair folders with one rule. Returns a list of matching
    filenames (not full sibling objects), or an empty list if none
    match (not guaranteed every multilingual dataset follows a
    recognizable naming convention).
    """
    path_separators = re.compile(r"[/_\-.]")
    return [
        s.rfilename for s in siblings
        if lang_code in path_separators.split(s.rfilename)
    ]


def fetch_readme_body_text(dataset_id):
    """
    Loads a dataset's README and returns its BODY text (the prose
    after the YAML front matter) - distinct from fetch_card_data(),
    which returns the parsed YAML metadata only. Returns None on
    failure rather than raising - a missing/unparseable README just
    means the heuristic search below finds nothing, not a harvest
    failure.
    """
    def _load():
        return DatasetCard.load(dataset_id).text

    return _with_retry(_load, f"load README body for {dataset_id}", max_retries=2)


def search_readme_for_language_size(readme_text, lang_code, language_name=None):
    """
    HEURISTIC, LOW CONFIDENCE - regex prose-parsing, not Claude-
    assisted. Scans the README body line by line for lines that
    mention the language (by code or, if given, by name) AND contain
    a size-like pattern (e.g. "5.2 GB", "10000 rows", "3.5 hours").

    Returns a list of matching raw lines (for human review - NOT
    parsed into a normalized byte figure, same "record verbatim for
    manual confirmation" approach used for publication lookups
    elsewhere in this script), or None if nothing matched.
    """
    if not readme_text:
        return None

    search_terms = [lang_code]
    if language_name:
        search_terms.append(language_name)

    matches = []
    for line in readme_text.splitlines():
        lower_line = line.lower()
        if any(term.lower() in lower_line for term in search_terms) and LANGUAGE_SIZE_PATTERN.search(line):
            matches.append(line.strip())

    return matches if matches else None



def resolve_language_specific_size(card_fields, siblings, dataset_id, lang_code, language_name=None):
    """
    Top-level entry point for the language breakdown layer.

    IMPORTANT: total_size_bytes / total_num_rows are STRICTLY
    language-specific on every row - never the whole dataset's
    combined total. For monolingual datasets, dataset-level IS
    language-level, so these are populated directly from
    resolve_total_size(). For multilingual datasets, they are ONLY
    populated when a genuine language-specific figure is resolved
    (currently: via a parsed README match, if one existed - see
    search_readme_for_language_size) - otherwise they stay None. They
    are deliberately NEVER filled in with the whole dataset's combined
    size, even as a placeholder or approximation - showing the whole-
    dataset figure in a per-language field would misrepresent it as
    that language's own size.

    The whole-dataset resolution (card_reported -> datasets_server ->
    fallback, the full resolve_total_size() chain) is still computed
    for every dataset regardless of linguality, but is exposed under
    the clearly separate reference_full_dataset_size_bytes /
    reference_full_dataset_size_method / reference_full_dataset_num_rows
    fields - ONLY for multilingual rows, where it's needed to decide
    which "not found" flag applies (see below), and to give a reviewer
    context on whether the dataset has a known total at all even
    though this language's slice of it isn't known. For monolingual
    rows these reference_ fields are left None, since total_size_bytes
    already IS that figure - showing it twice would be redundant.
    """
    dataset_wide = resolve_total_size(card_fields, siblings, dataset_id)
    linguality = classify_linguality(card_fields["language_codes"])

    if linguality != "multilingual":
        dataset_wide["reference_full_dataset_size_bytes"] = None
        dataset_wide["reference_full_dataset_size_method"] = None
        dataset_wide["reference_full_dataset_num_rows"] = None
        dataset_wide["language_scope"] = linguality  # "monolingual" or "no_language_tag"
        return dataset_wide

    lang_files = isolate_language_files(siblings, lang_code)
    readme_text = fetch_readme_body_text(dataset_id)
    readme_matches = search_readme_for_language_size(readme_text, lang_code, language_name)

    base = {
        "reference_full_dataset_size_bytes": dataset_wide["total_size_bytes"],
        "reference_full_dataset_size_method": dataset_wide["total_size_method"],
        "reference_full_dataset_num_rows": dataset_wide["total_num_rows"],
        "language_scope": "multilingual_isolated",
        "isolated_file_count": len(lang_files),
    }

    if readme_matches:
        return {
            **base,
            "total_size_bytes": None,
            "total_size_method": "language_size_found_in_readme_heuristic",
            "readme_matched_lines": ";".join(readme_matches),
            "needs_manual_review": True,  # heuristic match always needs human confirmation
            "flag_reason": "readme_heuristic_needs_confirmation",
        }

    # Path-based isolation found nothing AND no README size mention -
    # before falling through to the generic "not found" flags, check
    # (metadata-only - see detect_language_stored_as_column) whether
    # this dataset actually stores its language identifier as a DATA
    # COLUMN (one row per item, e.g. an "iso3" column) rather than as
    # separate per-language files. This is a genuinely different
    # situation from "no recognizable naming convention was used" -
    # here, per-language data unambiguously EXISTS in the file, it's
    # just not something this pipeline can isolate without reading
    # actual row data, which is out of scope. Flagged distinctly so a
    # reviewer doesn't confuse this with a dataset that simply lacks
    # a language breakdown at all.
    if not lang_files:
        language_column = detect_language_stored_as_column(siblings, dataset_id)
        if language_column:
            return {
                **base,
                "total_size_bytes": None,
                "total_size_method": "language_stored_as_data_column_not_isolatable_via_metadata",
                "detected_language_column_name": language_column,
                "needs_manual_review": True,
                "flag_reason": "language_stored_as_data_column",
            }

    # "Present vs missing" now reflects the FULL dataset-wide
    # resolution (card OR datasets-server OR fallback), not just the
    # raw card YAML value - a dataset with no hand-written card size
    # but a resolved datasets-server figure correctly counts as
    # "present," not "missing."
    if lang_files:
        # Isolation SUCCEEDED (real per-language files exist - see
        # isolated_file_count) but neither the README nor a byte-sum
        # gives a trustworthy language-specific total: a breakdown
        # genuinely exists, this is a fundamentally different, better
        # situation than "no breakdown found at all," and must not be
        # reported with the same generic flag - that would tell a
        # reviewer the data doesn't exist when it actually does, just
        # not in a form this pipeline can summarize into one number.
        # Deliberately NOT summing isolated files' raw byte sizes as a
        # substitute (that was tried and rejected - see git history/
        # conversation - since a byte total without a matching row
        # count is not a meaningful "size" for this purpose). The real
        # per-modality row/hour counts, when extractable at all (e.g.
        # from Parquet footers or audio headers), live in the separate
        # text_num_rows/audio_num_hours/image_num_rows columns computed
        # by get_modality_structured_counts() - this flag only means
        # "no single overall figure to report here," not "no data."
        method = "language_breakdown_found_no_summary_size_metric"
        flag_reason = "language_breakdown_found_but_row_count_not_in_file_metadata"
    else:
        if dataset_wide["total_size_bytes"] is not None:
            method = "language_breakdown_not_found_total_size_present"
        else:
            method = "language_breakdown_not_found_total_size_missing"

        # If the DATASET-WIDE resolution's own flag_reason was an
        # access problem (gated/private/other - see
        # ACCESS_ISSUE_REASONS), that's the real root cause of why
        # nothing was found at any level - not "no language
        # breakdown," which would misleadingly suggest the data
        # exists but just isn't organized in a way this pipeline can
        # parse. Surface the access issue instead of overwriting it
        # with the generic message in that case.
        if dataset_wide.get("flag_reason") in ACCESS_ISSUE_REASONS:
            flag_reason = dataset_wide["flag_reason"]
        else:
            flag_reason = "no_language_breakdown_found"

    return {
        **base,
        "total_size_bytes": None,
        "total_size_method": method,
        "needs_manual_review": True,
        "flag_reason": flag_reason,
    }


if __name__ == "__main__":
    print("This module provides functions for the redesigned Tier A per-dataset "
          "harvest logic. It is not yet wired into a full crosswalk-scale runner - "
          "see the accompanying notes for suggested next steps (validate against a "
          "handful of real datasets, then integrate into a wrapper matching the "
          "existing tier_a_harvest_all_languages.py resumability pattern).\n\n"
          "NOTE: the API-based publication size lookup (ask_claude_for_dataset_size) "
          "is currently PAUSED to avoid API costs during validation - see the note "
          "above fetch_arxiv_pdf_text() for how to re-enable it.")
