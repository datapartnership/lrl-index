"""
inspect_builder_info.py

Diagnostic tool: dumps the FULL load_dataset_builder(...).info object
for a given dataset/config - not just num_examples (what
get_num_rows_for_config checks), everything else too - so you can see
exactly what metadata is and isn't available, rather than guessing
why num_rows came back blank.

Usage:
  python inspect_builder_info.py <dataset_id> <config_name>

Examples:
  python inspect_builder_info.py openbmb/DCAD-2000 aaa_Latn
  python inspect_builder_info.py ec5ug/chikhapo aab_eng
"""
import argparse
import os

from datasets import load_dataset_builder


def inspect(dataset_id, config_name):
    print(f"Dataset: {dataset_id}")
    print(f"Config: {config_name}")
    print("-" * 60)

    try:
        builder = load_dataset_builder(dataset_id, config_name, token=os.environ.get("HF_TOKEN"))
    except Exception as e:
        print(f"load_dataset_builder() failed: {type(e).__name__}: {e}")
        return

    info = builder.info

    print(f"description: {(info.description or '')[:200]!r}")
    print(f"download_size: {info.download_size}")
    print(f"dataset_size: {info.dataset_size}")
    print(f"size_in_bytes: {info.size_in_bytes}")
    print(f"features: {list(info.features.keys()) if info.features else None}")
    print("-" * 60)

    if not info.splits:
        print("splits: None / empty - this dataset's builder has NO split metadata at all for this "
              "config. This is the actual reason num_rows came back blank - there's nothing to read, "
              "not a bug in how it's being read.")
    else:
        print(f"splits found: {list(info.splits.keys())}")
        for split_name, split_info in info.splits.items():
            print(f"  [{split_name}] num_examples={split_info.num_examples}, "
                  f"num_bytes={split_info.num_bytes}")
        if all(s.num_examples is None for s in info.splits.values()):
            print("\nAll splits exist but num_examples is None for every one of them - the split "
                  "STRUCTURE is known, but row counts specifically weren't populated. Different from "
                  "the 'no splits at all' case above, but same practical result: nothing to read for now.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump full builder.info metadata for a dataset/config.")
    parser.add_argument("dataset_id")
    parser.add_argument("config_name")
    args = parser.parse_args()

    inspect(args.dataset_id, args.config_name)
