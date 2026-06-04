---
paper: "NVIDIA Nemotron 3 Ultra v3 Tech Report (2026-06-03)"
model: "nemotron-ultra"
section: "recipes-overview"
title: "Released recipe status for Nemotron 3 Ultra"
currency: "tracking"
---

# Recipe status: not yet released as runnable stage code

Unlike `nemotron-nano3` and `nemotron-super3`, there is **no `ultra` recipe tree** in
`src/nemotron/recipes/` at the time of writing. This file is a **release tracker**, not a
runnable path. Do not fabricate config names, commands, or source paths for Ultra.

## What exists in the repo today

| Artifact | Path | Notes |
|---|---|---|
| Base-model usage cookbook | `usage-cookbook/Nemotron-3-Ultra-Base/README.md` | Identity, base benchmark table, availability framing. Base checkpoint only. |
| Stage recipe code | `src/nemotron/recipes/ultra/…` | **Absent.** No released pretrain/SFT/MOPD/eval stages. |

## Release framing (from the cookbook)

- Initial public artifact is the **base checkpoint**; not instruction-tuned or aligned.
- Full Nemotron 3 Ultra release (post-trained + NVFP4) expected **1H 2026**.
- Positioned as a starting point for customization (domain fine-tuning, RL post-training).

## How to answer reproduction questions

1. Say plainly that no public Ultra recipe stages exist yet.
2. Point to the **paper chunks** (`paper/*.md`, `paper/mopd/*.md`) for methodology.
3. For runnable analogs, the closest released code is the **Super3** recipe stack
   (`/nemotron-super3` → `recipes/`), since Ultra reuses the Super3 architecture and
   NVFP4 pretraining recipe. Note MOPD has no Super3 analog.
4. Hand procedural/build work to `/nemotron-customize`.

## When this file should change

Update this tracker when `src/nemotron/recipes/ultra/` lands, then add per-stage recipe
summaries mirroring the Super3 `recipes/` layout.
