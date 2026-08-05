"""
debug_tier_a_v2_audio_duration.py

Targeted test of get_audio_duration_hours_from_metadata() against
known-real, standalone .flac files - confirmed to exist at these
exact paths via multiple current Hugging Face documentation pages
(used as literal example audio inputs in their transformers docs).
Unlike librispeech_asr, this dataset was NOT converted to Parquet.
"""
from tier_a_harvest_v2 import get_audio_duration_hours_from_metadata

DATASET_ID = "Narsil/asr_dummy"
KNOWN_FILES = ["mlk.flac", "1.flac"]

base_url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/"

print(f"Testing audio duration extraction against {len(KNOWN_FILES)} known real file(s) "
      f"from {DATASET_ID}\n")

for fname in KNOWN_FILES:
    url = base_url + fname
    print(f"Probing: {url}")
    hours = get_audio_duration_hours_from_metadata(url)
    if hours is not None:
        print(f"  SUCCESS: {hours:.6f} hours ({hours * 3600:.2f} seconds)")
    else:
        print(f"  not_found_in_metadata (mutagen could not extract duration from the probed header bytes)")
    print()
