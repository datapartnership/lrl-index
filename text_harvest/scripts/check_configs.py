from datasets import get_dataset_config_names

datasets_to_check = [
    "lbourdois/language_tags",
    "lbourdois/panlex",
    "espnet/mms_ulab_v2",
    "amine-khelif/mms_ulab_v2",
    "coml/mmsulab",
    "sundram1996/mms_ulab_v2",
]

for ds_id in datasets_to_check:
    print(f"\n=== {ds_id} ===")
    try:
        configs = get_dataset_config_names(ds_id)
        print(f"  {len(configs)} config(s): {configs[:20]}{' ...' if len(configs) > 20 else ''}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

load_dataset("lbourdois/language_tags", 'language')
