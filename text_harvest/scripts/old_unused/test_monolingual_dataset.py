"""
test_monolingual_real_datasets.py

Verifies the monolingual size-resolution fix against 3 real,
confirmed-monolingual Swahili datasets found via live search:
  - uestc-swahili/swahili
  - community-datasets/swahili_news
  - flax-community/swahili-safi

Also re-tests wikimedia/wikipedia as the multilingual comparison
case, so both code paths get verified side by side in one run.
"""
from huggingface_hub import HfApi
from sample_tier_a_harvest import get_language_specific_size, detect_modality, extract_license

api = HfApi()

MONOLINGUAL_TEST_IDS = [
    "uestc-swahili/swahili",
    "community-datasets/swahili_news",
    "flax-community/swahili-safi",
]
MULTILINGUAL_TEST_ID = "wikimedia/wikipedia"
LANG_CODE = "sw"


def test_dataset(dataset_id, lang_code):
    print(f"--- {dataset_id} ---")
    try:
        info = api.dataset_info(dataset_id, files_metadata=True)
    except Exception as e:
        print(f"  ERROR fetching info: {e}\n")
        return

    tags = info.tags or []
    language_tags = [t for t in tags if t.startswith("language:")]
    modality, modality_method = detect_modality(tags)
    license_value = extract_license(tags)
    size_bytes, size_method = get_language_specific_size(info.siblings, lang_code, language_tags)

    print(f"  language_tags: {language_tags}")
    print(f"  modality: {modality} ({modality_method})")
    print(f"  license: {license_value}")
    print(f"  size: {size_bytes} bytes ({size_bytes / 1e6:.2f} MB)" if size_bytes else f"  size: UNKNOWN ({size_method})")
    print(f"  size_method: {size_method}")
    print(f"  total files in repo: {len(info.siblings) if info.siblings else 0}")
    print()


if __name__ == "__main__":
    print("=== MONOLINGUAL TEST CASES ===\n")
    for dataset_id in MONOLINGUAL_TEST_IDS:
        test_dataset(dataset_id, LANG_CODE)

    print("=== MULTILINGUAL COMPARISON CASE ===\n")
    test_dataset(MULTILINGUAL_TEST_ID, LANG_CODE)
