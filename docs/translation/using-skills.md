---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Sample prompts and a short guide for productive agent-assisted sessions with nemotron steps run translate/nemo_curator."
topics: ["Translation", "FAITH", "NeMo Curator"]
tags: ["Translation", "Documentation"]
content:
  type: "How-To"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

(translation-using-skills)=
# Tips for Translation With Agents

This page is for you if you want a productive session with a coding agent without a long back-and-forth.
The mechanics of the `nemotron steps run translate/nemo_curator` are shown in {doc}`getting-started` and the how-to guides.

## Sample Prompts You Can Paste

Paste one block as the first message in a new session.
Fill in bracketed parts or paths before you send.

### Example: Small Bilingual for SFT and FAITH Filtering

```text
I need help running nemotron steps run translate/nemo_curator for a multilingual supervised fine-tuning experiment.

Context:
- Repo: Nemotron at the machine root where I run commands.
- Input: [SOURCE-JSONL]
- Target language: ISO 639-1 code [TARGET], for example hi or de.
- I want translated shards I can inspect before scaling up.

Constraints:
- Use the checked-in config at src/nemotron/steps/translate/nemo_curator/config/default.yaml with -c default and CLI dotlist overrides only.
- Keep FAITH enabled with sensible training defaults from that tutorial unless you see a conflict; I want low-quality rows filtered, not every row kept.
- output_dir: [MY_OUTPUT_DIR] under the repo, for example ./output/translation-agent-1.

Please:
- Give me the exact uv run nemotron steps run translate/nemo_curator command with input_path, output_dir, source_language=en, target_language=[TARGET], and server.model filled in.
- Point me to the next doc page if I need to change text_field, output_mode, or segmentation.
```

## Next Steps

- First hands-on run: {doc}`getting-started`
- Field paths and `output_mode`: {doc}`how-to/configure-fields-and-output`
- FAITH tuning after exploration: {doc}`how-to/run-faith-evaluation`
- Full CLI: {doc}`reference/cli-translation`