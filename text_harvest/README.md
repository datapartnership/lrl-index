# Text Harvest

Measures how much AI training-relevant text data is available per language, across three tiers of source: Hugging Face Hub datasets (Tier A), rights-cleared Language Data Trust content (Tier B), and Common Crawl web text (Tier C).

## Tier A vs. Tier B vs. Tier C

**Tier A — Hugging Face Hub Harvest**
Pulls per-language text dataset metadata (size, provenance, license, modality) directly from the Hugging Face Hub. Size is resolved through a three-tier strategy of its own: **Tier 1** gets an exact count with no download, via Hugging Face's Datasets Server `/size` endpoint (falling back to `load_dataset_builder()` metadata when `/size` has nothing); **Tier 2** covers datasets with no per-language configuration by running a column-pruned query against auto-converted Parquet files, filtering to the target language directly rather than downloading anything; **Tier 3** is the residual — datasets that resolve through neither path are flagged for manual review, never estimated or silently dropped.

**Tier B — Rights-Cleared Data (Language Data Trusts)**
Proprietary and licensed content — primarily from media and government sources — made available through national Language Data Trusts, developed in partnership with DDP-partner country governments. Not yet actively harvested in this repository; this tier gets populated as individual trusts are formed. See `docs/` for the Language Data Trust Initiative context document.

**Tier C — Common Crawl Harvest**
Estimates usable web text volume per language via a four-stage pipeline: raw per-language byte volume from Common Crawl → a yield multiplier (calibrated against FineWeb-2, estimating clean-text survival after filtering) → per-language tokenizer fertility (bytes-per-token, calibrated on a real sample) → a final token estimate combining all three prior stages. Methodology fully documented in `docs/LRL Tier C Common Crawl Harvest Methodology.docx`.

## Structure

```
text_harvest/
├── scripts/          Active pipeline scripts
│   └── old_unused/   Superseded harvest versions and one-off debugging/test scripts —
│                      kept for reference and audit history, not part of the active pipeline
├── data/
│   ├── crawl/         Tier C raw inputs and processed outputs
│   ├── experiments/    Methodology-validation experiment outputs
│   └── tier_a_v6/      Current Tier A harvest output
└── docs/              Methodology writeups
```

## Scripts

| Script | What it does |
|---|---|
| `tier_a_harvest_v6.py` | The current Tier A harvest: full crosswalk-driven search with card/provenance/linguality extraction, using the three-tier size-resolution strategy (replacing earlier versions' slower, download-based approach) |
| `tier_a_config_availability_experiment.py` | Experiment, not a production step — measures, across every dataset found per crosswalk language code, whether a real per-language configuration exists. This is the direct evidence for whether the Tier 2 Parquet-column fallback is actually necessary |
| `check_configs.py` | Debugging script — lists the available configurations for datasets specified directly in the script |
| `check_native_vs_converted.py` | Quantifies how many datasets in a sample are natively uploaded as Parquet (at risk of non-standardized internal structure) versus auto-converted by Hugging Face's own pipeline (standardized) |
| `check_partial_conversion.py` | Quantifies how many datasets in a sample only received partial Parquet conversion — Hugging Face only converts up to ~5GB of a dataset, so anything larger is missing from the Parquet files entirely |
| `tier_c_common_crawl_harvest.py` | Tier C Stage 1 — estimates raw bytes per language, per Common Crawl monthly archive, aggregated into a cumulative total across all crawls |
| `tier_c_fw2_yield_multiplier.py` | Tier C Stage 2 — computes the yield multiplier per language (clean FineWeb-2 bytes ÷ raw Common Crawl bytes, matched to the same crawls), approximated via Hugging Face's `/statistics` endpoint on FineWeb-2's `dump` column |
| `tier_c_fw2_token_ratio.py` | Tier C Stage 3 — calibrates bytes-per-token and tokens-per-word (fertility) per language, not per script, so every language in the curated set gets its own sample and its own ratios |
| `tier_c_tokens.py` | Tier C Stage 4 — computes final token estimates per language (latest raw bytes × yield × tokens-per-byte), joining the outputs of the three prior Tier C scripts |

## Data Files

### `data/crawl/` — Tier C

| File | What it represents |
|---|---|
| `raw/tier_c_languages_csv_cache.csv` | Cached copy of Common Crawl's own `languages.csv` (page-share per language, per crawl) |
| `processed/tier_c_crawl_bytes_cache.json` | Cached total-bytes-downloaded figures per crawl, from Common Crawl's own crawl-level stats |
| `processed/tier_c_language_bytes_by_crawl.csv` | Stage 1 output: raw estimated bytes per language, per individual crawl |
| `processed/tier_c_language_bytes_cumulative.csv` | Stage 1 output: raw estimated bytes per language, summed across all crawls |
| `processed/tier_c_language_bytes_latest.csv` | Stage 1 output: raw estimated bytes per language, most recent crawl only |
| `processed/yield_multiplier_by_language.csv` | Stage 2 output: the calibrated yield multiplier (clean-text survival rate) per language |
| `processed/yield_multiplier_skipped_languages.csv` | Stage 2 skip log — languages not covered by Common Crawl's own language stats, or where the FineWeb-2 API was unavailable |
| `processed/tier_c_token_ratio_by_language.csv` | Stage 3 output: bytes-per-token and tokens-per-word (fertility) per language |
| `processed/tier_c_tokens_by_language.csv` | Stage 4 output: final estimated token count per language |
| `processed/tier_c_tokens_skipped_languages.csv` | Stage 4 skip log — languages missing one or more required inputs from Stages 1–3 |

### `data/experiments/` — Methodology Validation

| File | What it represents |
|---|---|
| `config_availability_by_language.csv` | Per (dataset, language) check of whether a real per-language Hugging Face configuration exists — the evidence behind the Tier 2 fallback decision |
| `config_availability_skipped.csv` | Skip log for the above — datasets whose config list couldn't be fetched at all |
| `native_vs_converted_check.csv` | Per-dataset check of whether its Parquet files are native (uploaded as Parquet directly) or auto-converted by Hugging Face's own pipeline |
| `partial_conversion_check.csv` | Per-dataset check of whether Hugging Face's Parquet auto-conversion was partial (i.e. the dataset exceeded the ~5GB conversion limit) |
| `readme_breakdown_by_language.csv` | From an earlier (now archived) experiment measuring how often dataset READMEs include a per-language size breakdown |

### `data/tier_a_v6/` — Current Tier A Harvest Output

| File | What it represents |
|---|---|
| `tier_a_v6_full_clean.csv` | Datasets with a resolved size (Tier 1 or Tier 2) and clear provenance — the "ready to use" output |
| `tier_a_v6_manual_review.csv` | Datasets needing human review: either an access issue (gated/private/404), or a resolved size with unclear provenance |
| `tier_a_v6_everything_else.csv` | Datasets that resolved through neither Tier 1 nor Tier 2 — flagged residual, not estimated |
