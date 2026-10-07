---
title: Limitless Footage Checker
emoji: 🎬
colorFrom: yellow
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

Visual verification service for the Limitless footage finder: given a narration line and candidate clip URLs, it downloads each clip, samples real frames, CLIP-scores them against the line, and returns a ranked strong / ok / weak verdict.
