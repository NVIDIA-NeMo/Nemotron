# TODO before publishing

Checklist of things to finish before sharing this cookbook publicly. The notebook/scripts are
genericized and verified free of environment-specific values; the items below are what's left.

## Blockers — must resolve before publish

- [ ] **Container image.** Replace the placeholder with the official, publicly available Ultra
      container. Touches two spots that should agree: the notebook `CONTAINER_IMAGE` placeholder
      (`<CONTAINER_IMAGE>`) and the README prerequisite (`<<REPLACE_WITH_OFFICIAL_CONTAINER>>`).
      If no public image exists at publish time, document how to build one instead (NeMo base +
      Megatron-Bridge + the `[ssm]` extra: mamba-ssm + causal-conv1d).
- [ ] **Base-checkpoint access.** The tutorial assumes the Ultra base weights are already downloaded
      at `HF_MODEL_PATH`. Add public instructions / a link for obtaining them (NGC or HF), and
      confirm the checkpoint is actually public by the time this goes out.
- [ ] **Dataset availability + licensing.** Confirm both BIRD splits used in data prep are public and
      that their licenses permit this use, and add attribution:
      - no-reasoning: `xu3kev/BIRD-SQL-data-train`
      - reasoning: `meowterspace45/bird-sql-train-with-reasoning`

## Cleanup / correctness

- [ ] **Keep internal-only docs out of the published tree.** `CLUSTER_ENV.md`, `FEEDBACK.md`, and
      `HANDOFF_PROMPT.md` live in the repo parent and are internal — publish only this cookbook
      directory, not those.
- [ ] **Decide what to do with `config.env`.** It currently ships as a genericized placeholder
      template (no real values). Either keep it as a documented template or delete it and let the
      notebook's setup cell generate it. (Verified: no leaked values either way.)
- [ ] **Clear notebook outputs/metadata** so the published `.ipynb` has clean cells and no execution
      counts.
- [ ] **Verify the Super cross-link** in the README (`../../Nemotron-3-Super/lora-text2sql`) resolves
      from the public repo layout.

## Nice to have

- [ ] **License/copyright headers** on the `.py` files if the target repo requires them.
- [ ] **Final read-through** of `README.md` and `SKILL.md` for tone and any lingering internal
      phrasing.
- [ ] **Mention non-4-GPU nodes** briefly in the README — node count auto-derives from
      `GPUS_PER_NODE` (world size stays 48 GPUs), so other tray sizes work.
