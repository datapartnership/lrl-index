# [Project Name] — Language Data Readiness Index

A reproducible, automatically-updatable index measuring, for every major world language, how much non-synthetic, clearly-licensed text and audio data exists for training generative AI models — and how close each language is to the data thresholds real applications require.

## Project structure

```
.
├── crosswalk/        Week 2 — language code crosswalk (ISO 639-3 x FineWeb-2, HuggingFace, Common Voice)
├── population/       Week 2 — Ethnologue-derived population baseline + curated language list
├── text_harvest/      Week 3 — Tier A/B/C text volume harvesting
├── audio_harvest/     Week 4 — transcribed/untranscribed audio hours harvesting
├── gates/             Week 5 — data threshold ("gate") methodology
└── index/              Week 6-7 — final index construction, refresh pipeline, visualization
```

Each subfolder has its own README with setup instructions, scripts, and data — see those for component-specific detail.

## Setup

```bash
pip install -r requirements.txt
```

## Status

This repo is under active development as part of a 7-week internship project. See individual subfolder READMEs for what's complete vs. in progress.
