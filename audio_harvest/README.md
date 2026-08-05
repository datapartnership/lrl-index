# Audio Harvest

Measures transcribed and untranscribed speech hours available per language, across multiple audio sources, feeding into the LRL Index alongside the text harvest. Each source has its own dedicated script, since each platform has its own release mechanism, its own definition of "transcribed," and its own known collection methodology — meaning provenance can be recorded with confidence, not inferred via a README heuristic the way it sometimes has to be for text data.

Full methodology is documented in `docs/Audio Harvest Methodology.docx`.

## Audio Sources

**Common Voice** — Two dataset types, each pulled from Common Voice's own structured GitHub release files: Scripted Speech (read-aloud, translated prompts — always counted transcribed, since the prompt is known in advance) and Spontaneous Speech (free-speech answers — only the validated portion counts as transcribed).

**Lingua Libre** — Community-recorded pronunciations, pulled via the Wikimedia Commons API (since the actual audio files live on Commons, not on lingualibre.org directly). Every recording is a speaker reading a pre-set word, so every recording is transcribed by construction — there's no separate transcription step or untranscribed category for this source.

**VoxPopuli** — European Parliament recordings. No API exists for this source; hours are parsed directly from a statistics table published in the source repository's own README.

**Living Tongues** — Community-built, community-maintained multimedia dictionaries via the Living Dictionaries platform (all entries uploaded and maintained by speakers of that language). No public API — the current data (`data/raw/living-dictionaries-language-stats-2026-07-29.csv`) came directly from their team by email, and does not yet have a dedicated harvest script converting it into the shared output schema.

**Other candidate sources** — additional audio sources have been identified for potential future inclusion; see `docs/Audio Data Sources.xlsx`.

## Structure

```
audio_harvest/
├── scripts/     Harvest scripts, one per source, plus the merge script
├── data/
│   ├── raw/       Source files that can't be fetched programmatically (e.g. Living Tongues' emailed spreadsheet)
│   └── processed/ Per-source harvest outputs, plus the combined master file
└── docs/        Methodology writeup and the candidate-sources spreadsheet
```

## Scripts

| Script | What it does |
|---|---|
| `common_voice_harvest.py` | Pulls Scripted Speech and Spontaneous Speech statistics from Common Voice's actual dataset release files (`cv-dataset` GitHub repo) — revised from an earlier version that used the live contribution-platform API, which had no distinction between speech types and no untranscribed category at all |
| `voxpopuli_harvest.py` | Parses the "Unlabelled and transcribed data" table directly from the VoxPopuli repository's README — there is no API for this source, so this is a static, versioned table in the repo's own documentation |
| `lingua_libre_harvest.py` | Pulls every audio file for each crosswalk language from its Wikimedia Commons category, fetches duration in batches via the Commons API, and derives `source_type` from the Commons page(s) that use or embed each file — one row per recording, not per language |
| `lingua_libre_duration_summing.py` | Aggregates the per-recording harvest CSV into one row per language — total duration, recording counts, and the fixed provenance fields that apply to every Lingua Libre recording. Recordings with missing or unparseable duration are excluded from the sum and counted separately, not treated as zero |
| `merge_audio_harvests.py` | Combines each source's output into one master table, keeping only the fields they have in common — a simple union, not an aggregation, with no summing, deduplication, or joining across sources |

## Data Files

### `data/raw/`

| File | What it represents |
|---|---|
| `living-dictionaries-language-stats-2026-07-29.csv` | Language × transcribed audio hours from Living Tongues, provided directly by their team via email (no API for this source) — not yet processed into the shared output schema |

### `data/processed/`

| File | What it represents |
|---|---|
| `common_voice/common_voice_hours.csv` | Common Voice harvest output — transcribed/untranscribed hours per language, both Scripted and Spontaneous Speech |
| `common_voice/common_voice_skipped_locales.csv` | Common Voice locales that couldn't be matched to a crosswalk language |
| `vox_populi/voxpopuli_hours.csv` | VoxPopuli harvest output — transcribed/untranscribed hours per language |
| `lingua_libre/lingua_libre_harvest.csv` | Lingua Libre per-recording table (one row per recording, not per language) |
| `lingua_libre/lingua_libre_language_summary.csv` | Lingua Libre hours aggregated to one row per language, via `lingua_libre_duration_summing.py` |
| `audio_harvest_all.csv` | The combined master file — every source's rows stacked together on their common fields only |
