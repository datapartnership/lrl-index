# Low Resource Language Data Readiness Index

A reproducible, automatically-updatable index measuring, for every major world language, how much non-synthetic, clearly-licensed text and audio data exists for training generative AI models — and how close each language is to the data thresholds real applications require.

Most AI language tools work well for a handful of high-resource languages and poorly, or not at all, for thousands of others — and this isn't just a function of speaker population: a language can have millions of native speakers and still be low-resource if it has limited representation online or in digital datasets. This index is designed to answer two practical questions: *how close is language X to being AI-ready?* and *where should future investment be directed* — toward collecting more data, improving data quality, or developing models? It's built as a World Bank Development Data Partnership initiative.

## New to this project? Start here.

If you're picking up this project without prior context, read the documents in **`Supplementary Information/`** before anything else:

- **`Index Overview.docx`** — a high-level overview of what the index measures and why, and how the whole project is structured.
- **`Next Steps.docx`** — outstanding work across every part of the project, with concrete next actions per section.
- **`Challenges.docx`** — major challenges already encountered, written up as problem/solution pairs, so you're not rediscovering the same issues from scratch.
- **`Contact Information.docx`** — who to reach out to for questions about different parts of the project.
- **`language-data-index-intern-workplan.docx`** — the original workplan this project was built against.

Together, these should give you the full picture of what this index is, what's been done, what's left, and who can help — before you go digging into individual tier folders.

## Project structure

```
.
├── Supplementary Information/  project overview, next steps, challenges, contacts, workplan
├── crosswalk/        language code crosswalk (ISO 639-3 x FineWeb-2, HuggingFace, Common Voice)
├── population/       Ethnologue-derived population baseline + curated language list
├── text_harvest/      Tier A/B/C text volume harvesting
├── audio_harvest/     transcribed/untranscribed audio hours harvesting
├── gates/             data threshold ("gate") methodology
└── index/              final index construction, refresh pipeline, visualization
```

Each subfolder has its own README with setup instructions, scripts, and data — see those for component-specific detail.

## Setup

```bash
pip install -r requirements.txt
```

## Status

This repo was under active development as part of a 7-week internship project. See individual subfolder READMEs for what's complete vs. in progress.
