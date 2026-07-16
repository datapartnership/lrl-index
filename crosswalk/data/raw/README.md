# Raw Inputs

Place the following files here before running the pipeline scripts:

- `iso-639-3.tab`
- `iso-639-3-macrolanguages.tab`
- `fineweb2_labels.txt`
- `hf_language_tags.json`
- `commonvoice_languages_raw.json`

See the main README for download/fetch instructions for each.

These files are intentionally not committed to version control (see `.gitignore`), since two of them are live API snapshots that go stale, and the SIL tables are easy to re-download from the authoritative source directly.
