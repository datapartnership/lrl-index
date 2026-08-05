"""
tier_a_harvest_single_language.py

Full Tier A TEXT harvest pipeline for ONE sample language, end to end:
search -> modality filter (text only) -> license -> size -> provenance
-> multilinguality -> one row per dataset, written to CSV.

Scope: TEXT ONLY. Datasets confirmed as audio or image are filtered
out entirely and do not appear in the output. This keeps this script
modular - audio/image harvesting belongs to a separate script.
Datasets where modality cannot be determined (modality_unstated) are
still kept, since excluding them would silently lose potentially
relevant text data - they are recorded as unstated rather than
assumed to be text or non-text.

Language classification: a dataset is "monolingual" if it carries
exactly one language: tag, "multilingual" if it carries more than
one. This is recorded explicitly as its own column, not just used
internally for size resolution.

Provenance: recorded as "provenance_unknown" (not assumed) whenever
neither human-generated nor machine-generated keywords are found.

Size resolution handles monolingual and multilingual datasets
differently:
  - Monolingual: the whole dataset counts, even if no filename
    mentions the language code at all.
  - Multilingual: isolates just the portion matching this language
    via filename pattern - only works if the dataset happens to
    split files per language (e.g. Wikipedia-style naming). Returns
    size_unknown otherwise.
"""
import pandas as pd
import re
from huggingface_hub import HfApi

api = HfApi()

PATH_SEPARATORS = re.compile(r"[/_\-.]")

# ---- CONFIG ----
SAMPLE_LANGUAGE = "fmp"   # swap this for any language code to test
SEARCH_LIMIT = None
OUTPUT_FILE = f"tier_a_harvest_{SAMPLE_LANGUAGE}.csv"

TEXT_EXTENSIONS = {".txt", ".csv", ".tsv", ".json", ".jsonl", ".parquet", ".xml", ".arrow"}
# Compressed variants - many real text datasets (e.g. allenai/c4) ship
# as gzip-compressed JSON, which a plain ".json" check misses entirely
# since the filename actually ends in ".gz". Confirmed against real
# C4 filenames like "c4-af.tfrecord-00000-of-00064.json.gz". Only
# specific compressed TEXT extensions are added, not a bare ".gz"
# catch-all, since audio/image files could in principle be gzipped
# too and a bare ".gz" check would wrongly classify those as text.
TEXT_EXTENSIONS |= {ext + ".gz" for ext in TEXT_EXTENSIONS}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

# NOTE: WebDataset format (.tar archives) is NOT included in
# TEXT_EXTENSIONS, deliberately. Per HF's own documentation,
# WebDataset is "designed for multimodal datasets, i.e. for image,
# audio and/or video datasets" - text/captions may be bundled inside
# the tar alongside media files, but a bare ".tar" extension cannot
# tell us the modality mix without opening the archive, which this
# pipeline does not do. A dataset using this format will currently
# fall through to size_unknown_no_text_files_found, which is the
# correct conservative behavior - not a guess in either direction.

PROVENANCE_KEYWORDS = {
    "human-generated": [
        "human-generat", "human annotat", "crowdsourc", "crowd-sourc",
        "human-translat", "human translat",
        "native speaker", "manually annotat", "manually transcrib",
        "manually translat", "hand-annotat", "hand annotat",
        "volunteer", "expert annotat", "professionally translat",
        "collected from speakers", "collected by speakers",
        "transcribed by", "annotated by humans", "human-curat",
        "human curat", "written by", "authored by",
    ],
    "machine-generated": [
        "machine-generat", "machine generat", "synthetic",
        "llm-generat", "llm generat", "machine-translat", "machine translat",
        "auto-generat", "auto generat", "automatically generat",
        "automatically translat", "ai-generat", "ai generat",
        "model-generat", "model generat", "gpt-generat",
        "back-translat", "back translat", "machine translation system",
        "neural machine translation", "nmt-generat",
    ],
}


def _loose_match(text_to_search, keywords):
    """
    Checks whether any keyword ROOT appears anywhere in the text,
    rather than requiring an exact phrase match. This is intentionally
    simple substring matching on shortened root forms (e.g.
    "crowdsourc" instead of the full word "crowdsourced") - not
    stemming via an NLP library, not semantic/embedding-based
    similarity. This keeps the logic transparent and auditable, which
    matters since flagged/unmatched datasets are meant to go to manual
    review - a reviewer should be able to look at this list directly
    and understand exactly what would or wouldn't have matched.
    """
    return any(kw in text_to_search for kw in keywords)


# ---- MODALITY ----
def detect_modality(tags, siblings=None):
    """Tag first (free). File extension fallback (costs an extra API
    call upstream to populate siblings). Never guesses - returns
    modality_unstated if neither resolves it."""
    modality_tags = [t for t in tags if t.startswith("modality:")]
    if modality_tags:
        return ";".join(t.replace("modality:", "") for t in modality_tags), "from_tag"

    if siblings:
        extensions_seen = {
            ext for s in siblings
            for ext in (TEXT_EXTENSIONS | AUDIO_EXTENSIONS | IMAGE_EXTENSIONS)
            if s.rfilename.lower().endswith(ext)
        }
        detected = []
        if extensions_seen & TEXT_EXTENSIONS:
            detected.append("text")
        if extensions_seen & AUDIO_EXTENSIONS:
            detected.append("audio")
        if extensions_seen & IMAGE_EXTENSIONS:
            detected.append("image")
        if detected:
            return ";".join(detected), "inferred_from_file_extension"

    return "modality_unstated", "unresolved"


def is_excluded_modality(modality_value):
    """
    Returns True only when modality is CONFIRMED to be audio and/or
    image with NO text component at all. modality_unstated is NOT
    excluded - an unresolved case should still be kept, since
    dropping it could silently lose real text data.
    """
    if modality_value == "modality_unstated":
        return False
    values = set(modality_value.split(";"))
    has_text = "text" in values
    has_only_excluded = values.issubset(AUDIO_IMAGE_MODALITIES) and not has_text
    return has_only_excluded


AUDIO_IMAGE_MODALITIES = {"audio", "image", "video"}


# ---- LANGUAGE CLASSIFICATION ----
def classify_linguality(language_tags):
    """
    Explicit classification based purely on the count of language:*
    tags - exactly one means monolingual, more than one means
    multilingual. (Zero is theoretically possible if a dataset has no
    language tag at all, despite matching the search - recorded
    separately so it isn't silently miscounted as either.)
    """
    if len(language_tags) == 1:
        return "monolingual"
    elif len(language_tags) > 1:
        return "multilingual"
    else:
        return "no_language_tag"


# ---- LICENSE ----
def extract_license(tags):
    license_tags = [t.replace("license:", "") for t in tags if t.startswith("license:")]
    if license_tags:
        return ";".join(license_tags)
    return "license_unstated"


# ---- SIZE (handles monolingual vs multilingual differently) ----
def filter_to_text_files(siblings):
    """
    Restricts a siblings list to files that look like text data,
    based on file extension. This matters specifically for datasets
    tagged with multiple modalities (e.g. modality:text;modality:audio)
    - such a dataset correctly passes the dataset-level modality
    filter (is_excluded_modality) since it DOES contain real text,
    but without this step, size summation would also include the
    audio/image files sitting in the same repo, inflating the
    reported "text" size with non-text bytes.
    """
    return [s for s in siblings if any(s.rfilename.lower().endswith(ext) for ext in TEXT_EXTENSIONS)]

def get_dataset_total_size(siblings):
    """
    Tracking-only field: total size in bytes of EVERY sibling file in
    the dataset repo, regardless of language or modality (text, audio,
    image, everything summed together). Not used in any language- or
    modality-specific calculation - purely so a reviewer can see, at a
    glance, how big the overall dataset is relative to the
    language-specific slice recorded in total_size_bytes/
    text_size_bytes. E.g. distinguishes a 500GB multilingual corpus
    with a 5MB slice for this language from a 5MB dataset that's
    almost entirely this language.

    Returns (None, "size_unknown_no_siblings") if there are no
    siblings at all, and (None, "size_unknown_no_size_data") if
    siblings exist but none carry a size - never guesses.
    """
    if not siblings:
        return None, "size_unknown_no_siblings"

    sized = [s.size for s in siblings if s.size]
    if not sized:
        return None, "size_unknown_no_size_data"

    return sum(sized), "exact_full_dataset_all_languages_all_modalities"

def find_language_segment_matches(siblings, lang_code):
    """
    Generalized file-path matcher. Splits each file's path into
    segments on common separators (/, _, -, .) and checks whether the
    target language code appears as a COMPLETE segment anywhere in
    the path - not just a substring match.

    This catches multiple real naming conventions with one approach:
      - Wikipedia: "20231101.sw/train-...parquet" -> segment "sw" found
      - OPUS-100:  "en-ha/train-...parquet" -> segment "ha" found
                   (language-PAIR folders, hyphen-joined)

    Segment-based matching also avoids false positives from substring
    matching - e.g. searching for "en" will not wrongly match inside
    an unrelated longer token, since "en" must appear as its own
    complete segment after splitting.
    """
    return [s for s in siblings if lang_code in PATH_SEPARATORS.split(s.rfilename)]


def get_language_specific_size(siblings, lang_code, linguality):
    """
    Returns a dict with two parallel size figures, each with its own
    status reason:
      - total_size_bytes / total_size_method: every matching file,
        regardless of modality (text, audio, image, etc. all included)
      - text_size_bytes / text_size_method: same matching logic, but
        restricted to text-extension files only (see
        filter_to_text_files)

    Both follow the same monolingual/multilingual matching rules:
      1. Monolingual: the WHOLE dataset is already specific to this
         language, even if no filename mentions the language code at
         all. Sum every matching file's size.
      2. Multilingual: isolate just the portion for this language
         using find_language_segment_matches(). Works for Wikipedia-
         style per-language folders AND OPUS-style language-pair
         folders, since both contain the language code as a distinct
         path segment - not guaranteed for every naming convention.

    Returns size_unknown with a specific reason when a figure can't
    be resolved - never guesses.
    """
    if not siblings:
        empty = (None, "size_unknown_no_siblings")
        return {
            "total_size_bytes": empty[0], "total_size_method": empty[1],
            "text_size_bytes": empty[0], "text_size_method": empty[1],
        }

    def resolve(file_list):
        if linguality == "monolingual":
            sized = [s.size for s in file_list if s.size]
            if not sized:
                return None, "size_unknown_no_size_data"
            return sum(sized), "exact_full_dataset_monolingual"

        matching = find_language_segment_matches(file_list, lang_code)
        if not matching:
            return None, "size_unknown_no_language_split_in_multilingual_dataset"
        sized = [s.size for s in matching if s.size]
        if not sized:
            return None, "size_unknown_no_size_data"
        return sum(sized), "exact_from_siblings_multilingual_split"

    total_bytes, total_method = resolve(siblings)

    text_siblings = filter_to_text_files(siblings)
    if not text_siblings:
        text_bytes, text_method = None, "size_unknown_no_text_files_found"
    else:
        text_bytes, text_method = resolve(text_siblings)

    return {
        "total_size_bytes": total_bytes, "total_size_method": total_method,
        "text_size_bytes": text_bytes, "text_size_method": text_method,
    }


# ---- PROVENANCE ----
def detect_provenance(tags, description):
    text_to_search = " ".join(tags or []).lower() + " " + (description or "").lower()
    for category, keywords in PROVENANCE_KEYWORDS.items():
        if _loose_match(text_to_search, keywords):
            return category
    return "provenance_unknown"


# ---- PUBLICATION LINK (for manual/future LLM-assisted provenance review) ----
def extract_publication_link(tags):
    """
    Looks for an arxiv:* tag on the dataset and constructs the real
    paper URL. This does NOT attempt to read or search the paper -
    that is a deliberately separate, later step. This only records
    the link so a flagged dataset can be manually (or later,
    LLM-assisted) reviewed against its source publication.

    Only checks for arxiv tags currently, since that's the only
    publication-link convention confirmed from real Hugging Face tag
    data so far (e.g. "arxiv:2606.21661" seen on a real dataset).
    Other publication link conventions, if they exist, are not yet
    handled and would need separate confirmation before adding.
    """
    links = []

    arxiv_tags = [t for t in tags if t.startswith("arxiv:")]
    if arxiv_tags:
        arxiv_id = arxiv_tags[0].replace("arxiv:", "")
        links.append(f"https://arxiv.org/abs/{arxiv_id}")

    doi_tags = [t for t in tags if t.startswith("doi:")]
    if doi_tags:
        doi_id = doi_tags[0].replace("doi:", "")
        links.append(f"https://doi.org/{doi_id}")

    return ";".join(links) if links else None

# ---- MULTILINGUALITY TAG (HF's own controlled vocabulary) ----
def extract_multilinguality_tag(tags):
    """
    Separate from classify_linguality() above. This reads HF's own
    multilinguality:* tag (monolingual / multilingual / translation /
    other-...), which is a dataset-author-declared label and may not
    always agree with the language tag count - both are kept as
    independent fields rather than collapsed into one.
    """
    type_tags = [t.replace("multilinguality:", "") for t in tags if t.startswith("multilinguality:")]
    if type_tags:
        return ";".join(type_tags)
    return "type_unstated"


def needs_review(provenance):
    """
    Flags any dataset where provenance could not be determined,
    regardless of linguality. This was previously narrower - only
    flagging unknown-provenance datasets that were ALSO multilingual
    or translation-type - but is now simplified to flag every
    unknown-provenance case.
    """
    return provenance == "provenance_unknown"


# ---- MAIN HARVEST ----
def harvest_language(lang_code, limit=SEARCH_LIMIT):
    rows = []
    excluded_count = 0
    search_results = list(api.list_datasets(language=lang_code, limit=limit))
    print(f"[{lang_code}] {len(search_results)} datasets found in search\n")

    for ds in search_results:
        tags = ds.tags or []
        language_tags = [t for t in tags if t.startswith("language:")]
        linguality = classify_linguality(language_tags)

        modality_value, modality_method = detect_modality(tags)

        # Only pay for the detailed lookup (files_metadata=True) when
        # modality is confirmed text, OR unresolved (since we still
        # want to try the file-extension fallback for those before
        # deciding whether to exclude).
        siblings = None
        if "text" in modality_value or modality_value == "modality_unstated":
            try:
                full_info = api.dataset_info(ds.id, files_metadata=True)
                siblings = full_info.siblings
                if modality_value == "modality_unstated":
                    modality_value, modality_method = detect_modality(tags, siblings)
            except Exception as e:
                print(f"  Could not fetch detailed info for {ds.id}: {e}")

        # TEXT-ONLY SCOPE: drop anything confirmed as audio/image with
        # no text component. modality_unstated is kept (see
        # is_excluded_modality docstring).
        if is_excluded_modality(modality_value):
            excluded_count += 1
            continue

        license_value = extract_license(tags)
        size_result = get_language_specific_size(siblings, lang_code, linguality)
        dataset_total_size_bytes, dataset_total_size_method = get_dataset_total_size(siblings)
        provenance = detect_provenance(tags, getattr(ds, "description", None))

        # Only bother recording a publication link when provenance
        # couldn't be determined - if provenance is already known,
        # there's nothing to manually verify against a paper.
        publication_link = extract_publication_link(tags) if provenance == "provenance_unknown" else None

        multilinguality_tag = extract_multilinguality_tag(tags)
        review_flag = needs_review(provenance)

        rows.append({
            "dataset_id": ds.id,
            "language_queried": lang_code,
            "num_languages_in_dataset": len(language_tags),
            "linguality": linguality,
            "multilinguality_tag": multilinguality_tag,
            "modality": modality_value,
            "modality_method": modality_method,
            "license": license_value,
            "dataset_total_size_bytes": dataset_total_size_bytes,
            "dataset_total_size_method": dataset_total_size_method,
            "total_lang_size_bytes": size_result["total_size_bytes"],
            "total_lang_size_method": size_result["total_size_method"],
            "text_size_bytes": size_result["text_size_bytes"],
            "text_size_method": size_result["text_size_method"],
            "provenance": provenance,
            "publication_link": publication_link,
            "synthetic_flag": provenance == "machine-generated",
            "needs_review": review_flag,
            "retrieval_method": "huggingface_hub_api",
        })

    print(f"Excluded {excluded_count} dataset(s) confirmed as audio/image only (out of scope for this text harvest)")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = harvest_language(SAMPLE_LANGUAGE)

    print(f"\nHarvested {len(df)} text-scoped rows for language '{SAMPLE_LANGUAGE}'\n")
    print(df.to_string(index=False))

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
