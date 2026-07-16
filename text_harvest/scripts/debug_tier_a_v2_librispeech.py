"""
debug_tier_a_v2_librispeech.py

Quick sanity check: does the ASR task detection work, does audio
duration extraction work on real files, and (bonus) does this dataset
ALSO have card-level dataset_info, letting us cross-check the
file-summed fallback against a known-correct card value?

NOTE: originally written against mozilla-foundation/common_voice_11_0,
which has been removed from Hugging Face entirely (confirmed via
search - Common Voice moved to Mozilla's own Data Collective
platform). Swapped to openslr/librispeech_asr, confirmed to still
exist, non-gated, real ASR task tag, real audio files.
"""
from tier_a_harvest_v2 import (
    api, fetch_card_data, extract_card_fields, resolve_total_size,
    get_modality_structured_counts, get_audio_duration_hours_from_metadata,
)

DATASET_ID = "openslr/librispeech_asr"
SAMPLE_SIZE = 5  # keep this small - just enough to validate the logic works

print(f"Fetching dataset_info for {DATASET_ID}...")
info = api.dataset_info(DATASET_ID, files_metadata=True)
tags = info.tags or []

print("Loading card data...")
card_data = fetch_card_data(DATASET_ID)

fields = extract_card_fields(DATASET_ID, tags, card_data)
print("\n--- Card fields ---")
for k, v in fields.items():
    print(f"  {k}: {v}")

# Grab just a small sample of audio files rather than the whole dataset
all_audio_files = [s.rfilename for s in info.siblings if s.rfilename.lower().endswith((".mp3", ".wav", ".flac", ".ogg"))]
print(f"\nFound {len(all_audio_files)} standalone audio file(s) among siblings")

if not all_audio_files:
    print(
        "\n*** No standalone audio files found in the file listing. ***\n"
        "This likely means this dataset's audio is embedded inside .parquet files\n"
        "(as an Arrow audio column) rather than shipped as separate .flac/.mp3/.wav\n"
        "files - common after a dataset gets 'converted to Parquet' on the Hub.\n"
        "get_audio_duration_hours_from_metadata() reads headers of STANDALONE audio\n"
        "files via HTTP range requests - it does NOT decode audio embedded inside a\n"
        "Parquet column, which is a different extraction problem not yet built.\n"
        "Skipping the audio duration probe for this dataset - this is expected\n"
        "behavior given the file structure, not a bug.\n"
    )
    sample_files = []
else:
    sample_files = all_audio_files[:SAMPLE_SIZE]
    print("Sample:", sample_files)

result = resolve_total_size(fields, info.siblings, DATASET_ID)
print("\n--- Total size resolution ---")
for k, v in result.items():
    print(f"  {k}: {v}")
print("\nNote: librispeech_asr has real card-level dataset_info, so expect "
      "total_size_method == 'card_reported' here, NOT the ASR fallback path - "
      "the card check runs first in the decision tree. This still validates "
      "the ASR task tag was correctly parsed into tasks/task_list_raw above, "
      "just not the fallback branch itself.")

if sample_files:
    print(f"\n--- Probing audio duration metadata on {len(sample_files)} sample file(s) only ---")
    counts = get_modality_structured_counts(info.siblings, DATASET_ID, target_files=sample_files)
    print(counts)

    print("\n--- Per-file breakdown (to see which files succeeded/failed the metadata probe) ---")
    base_url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/"
    for fname in sample_files:
        hours = get_audio_duration_hours_from_metadata(base_url + fname)
        print(f"  {fname}: {hours if hours is not None else 'not_found_in_metadata'}")
