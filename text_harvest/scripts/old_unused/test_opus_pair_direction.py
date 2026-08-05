"""
test_opus_pair_direction.py

Tests whether the OPUS API treats source/target as order-sensitive,
or whether it normalizes pairs internally regardless of which
language is passed as "source" vs "target".

This matters directly for harvester design: if the API is
order-sensitive, every language pair must be queried in BOTH
directions to avoid silently missing real data. If it normalizes
internally, querying one direction per pair is sufficient.

Run this on your own machine - opus.nlpl.eu is not reachable from
some sandboxed environments.
"""
import requests
import json

BASE_URL = "https://opus.nlpl.eu/opusapi/"


def query_pair(source, target, preprocessing="xml", version="latest"):
    params = {
        "source": source,
        "target": target,
        "preprocessing": preprocessing,
        "version": version,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def inspect_raw_structure(data, label):
    """Print the raw structure of the API response so we can see the
    ACTUAL field names, rather than assuming they match the docs."""
    print(f"\n--- RAW STRUCTURE: {label} ---")
    print(f"Top-level keys: {list(data.keys())}")
    corpora = data.get("corpora", [])
    print(f"Number of items in 'corpora': {len(corpora)}")
    if corpora:
        print(f"Keys in FIRST corpus entry: {list(corpora[0].keys())}")
        print(f"FULL first entry (raw): {json.dumps(corpora[0], indent=2)}")
        print(f"FULL second entry (raw): {json.dumps(corpora[1], indent=2) if len(corpora) > 1 else 'N/A'}")


def summarize(label, data):
    corpora = data.get("corpora", [])
    print(f"{label}: {len(corpora)} corpora found")
    for c in corpora[:5]:
        print(f"  - {c.get('name')} | source={c.get('source')} target={c.get('target')} "
              f"| src_tokens={c.get('src_tokens')} trg_tokens={c.get('trg_tokens')}")
    if len(corpora) > 5:
        print(f"  ... and {len(corpora) - 5} more")
    return {c.get("name") for c in corpora}


if __name__ == "__main__":
    test_pairs = [
        ("sw", "en"),  # Swahili-English, a real pair we expect to exist
        ("ha", "en"),  # Hausa-English, another real pair (OPUS-100 confirmed this earlier)
    ]

    for lang_a, lang_b in test_pairs:
        print("=" * 70)
        print(f"TESTING: {lang_a} <-> {lang_b}")
        print("=" * 70)

        print(f"\nQuery 1: source={lang_a}, target={lang_b}")
        data_forward = query_pair(lang_a, lang_b)

        if lang_a == test_pairs[0][0]:  # only dump raw structure once, for the first pair
            inspect_raw_structure(data_forward, "forward query")

        names_forward = summarize("Forward", data_forward)

        print(f"\nQuery 2: source={lang_b}, target={lang_a}")
        data_reverse = query_pair(lang_b, lang_a)
        names_reverse = summarize("Reverse", data_reverse)

        print()
        if names_forward == names_reverse and names_forward:
            print("RESULT: SAME corpora found in both directions -> API appears ORDER-INSENSITIVE for this pair")
        elif names_forward and not names_reverse:
            print("RESULT: Forward direction has data, reverse does NOT -> API appears ORDER-SENSITIVE")
        elif names_reverse and not names_forward:
            print("RESULT: Reverse direction has data, forward does NOT -> API appears ORDER-SENSITIVE")
        elif names_forward and names_reverse and names_forward != names_reverse:
            print("RESULT: BOTH directions return data, but DIFFERENT corpora -> partial overlap, query both directions to be safe")
        else:
            print("RESULT: Neither direction returned data for this pair - may not exist in OPUS, try a different pair")
        print()
