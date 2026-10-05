# Nemotron Curation

Turn a raw corpus into a filtered one you can defend: what was removed, on whose
threshold, and against which measurement.

## The shape of it

Curating with quality thresholds takes **two runs**, because the numbers you need
do not exist until the corpus has been read. Run 1 measures. You choose. Run 2
applies what you chose.

```text
raw parquet/JSONL
  -> curate/ingest          mint a content-derived id, normalise to JSONL
  -> curate/profile         RUN 1 ONLY: what would each threshold cost?
                            -> profile_summary.md      read this
                            -> candidate_policies.yaml approved: false
  ---- a person picks thresholds and signs for them ----
  -> curate/nemo_curator    the only step that drops rows
  -> curate/audit           independently recount; refuse a silent loss
  -> curate/decontamination overlap against a holdout
  -> curate/subset          nested token-budget tiers, cut from what is left
```

Nothing here approves a threshold on your behalf. A distribution says what a gate
removes; it never says whether removing it is right.

## Steps

Listed in the order the flow runs them. Types are the ones each step declares in
its `step.toml` and registers in [`../types.toml`](../types.toml) — this table is
generated from nothing, so it is checked against them by `tests/steps/test_types.py`
rather than trusted.

| Need | Step | Input | Output |
|---|---|---|---|
| Raw parquet/JSONL in, curatable JSONL with a stable id out | [`curate/ingest`](nemo_curator/ingest/README.md) | `raw_jsonl` | `prepared_jsonl` |
| Measure what each threshold would remove, before removing anything | [`curate/profile`](nemo_curator/profile/README.md) | `raw_jsonl` \| `prepared_jsonl` | `profile_report`, `filter_policy` |
| Language, length, domain and approved-policy gating | [`curate/nemo_curator`](nemo_curator/README.md) | `raw_jsonl` \| `prepared_jsonl` (or HF snapshot), `filter_policy` | `filtered_jsonl`, `curation_manifest`, `curation_ledger` |
| Prove no records went missing without being counted | [`curate/audit`](nemo_curator/audit/README.md) | `filtered_jsonl`, `curation_manifest`, `curation_ledger` | `curation_report` |
| Overlap against an evaluation holdout | [`curate/decontamination`](nemo_curator/decontamination/README.md) | `filtered_jsonl` | `decontaminated_jsonl`, `decontamination_report` |
| Nested token-budget tiers from one corpus | [`curate/subset`](nemo_curator/subset/README.md) | `filtered_jsonl` \| `decontaminated_jsonl` | `filtered_jsonl`, `subset_plan`, `subset_report` |

Two of those types exist only to make a dependency statable. `prepared_jsonl` is a
`raw_jsonl` with a guaranteed document id, so every step that reads a raw corpus
accepts it; `decontaminated_jsonl` is a `filtered_jsonl` with the holdout overlap
removed, so everything downstream accepts it and `curate/subset` can say it must
read the corpus *after* decontamination rather than before. Both were identity
edges — `raw_jsonl -> raw_jsonl`, `filtered_jsonl -> filtered_jsonl` — which state
that a step changed nothing.

`curation_manifest` and `curation_ledger` are optional inputs to `curate/audit`:
without them it still runs, and reports completeness as informational and
attribution as unavailable rather than reporting a clean result.

Each is a registered step and runs on its own:

```bash
uv run nemotron steps run curate/profile -c default
```

## Running all six from one config

The flow derives every cross-step path, so a producer and its consumer cannot
disagree, and refuses a misconfigured run before any step does work. It is a
script rather than a registered step, so it takes a config path:

```bash
uv run --extra curate --extra xenna \
  python -m nemotron.steps.curate.nemo_curator.scripts.run_flow \
  --config src/nemotron/steps/curate/nemo_curator/config/vi_c4_measure.yaml
```

The two extras are not optional and a plain `uv run` will not do. Curator executes
the filter on Ray, and Ray resolves the *worker* interpreter independently of the
one you launched: without `xenna` on that side the run reaches the executor and
dies with `ModuleNotFoundError: No module named 'cosmos_xenna'` from inside a
worker, several minutes in, after the flow has already reported its plan. Neither
extra is in the default dependency set, because installing Curator and Ray is not
something every user of this repository should be made to do.

Steps that never start Ray — `curate/profile`, `curate/subset`,
`curate/decontamination` with `skip_similarity: true` — run under `--extra curate`
alone.

Two worked examples show the two halves, and are meant to be copied:

- [`vi_c4_measure.yaml`](nemo_curator/config/vi_c4_measure.yaml) — run 1, `approve: null`
- [`vi_c4_apply.yaml`](nemo_curator/config/vi_c4_apply.yaml) — run 2, a filled approve block

`--plan` prints the derived per-step configs without running anything.

## Developer Journey

1. Identify the raw source: local parquet/JSONL or a Hugging Face snapshot.
2. Run `vi_c4_measure` with your paths. Nothing is gated on a policy nobody read.
3. Read `output/profile/profile_summary.md`: the language composition, then each
   signal's `gate at` table, then `Policy simulation` for what the gates cost
   *together* — they overlap, so the union is smaller than the sum.
4. Copy the `Approve block` section into `vi_c4_apply.yaml`, edit the thresholds,
   and write down in `evidence` what you actually looked at.
5. Delete `filtered_jsonl/`, `audit/` and `subset/`, then run `vi_c4_apply`.
6. Check `flow_report.json`: `policy_applied`, `audit_passed`, and the warnings.
7. Check the thresholds kept the right documents, not merely a plausible number
   of them: label a held-out set and run
   [`curate/evaluate`](#checking-a-policy-against-labelled-documents). Retention
   alone cannot tell a gate that dropped 7% of junk from one that dropped 7% of
   your best material.

## Extending It

- A language the shipped packs do not cover:
  [ADDING_A_LANGUAGE.md](nemo_curator/data/langpacks/ADDING_A_LANGUAGE.md)
- A measurement the shipped signals do not make, including a custom domain
  filter: [ADDING_A_SIGNAL.md](nemo_curator/ADDING_A_SIGNAL.md)

## Checking A Policy Against Labelled Documents

`curate/profile` reports what a threshold *removes*. Nothing in the six steps
reports whether removing it was *right* — that needs documents a person has
labelled, and two rates that fail in opposite directions:

- **false rejection** — of documents labelled `keep`, how many the policy dropped
- **noise removal** — of documents labelled `drop`, how many the policy dropped

A gate that removes nothing is perfect on the first; one that removes everything
is perfect on the second. Either number alone is worthless.

```bash
uv run --extra curate python -m nemotron.steps.curate.nemo_curator.scripts.run_evaluate \
  --policy ./output/vi/policy/approved_policy.yaml \
  --labelled ./eval/vi.jsonl \
  --langpack-dir ./src/nemotron/steps/curate/nemo_curator/data/langpacks \
  --report ./output/vi/evaluation.json
```

Like the flow, this is a module rather than a registered step: it measures a
policy the steps produce rather than being one of them, and it runs without NeMo
Curator so it works in CI.

Read the per-phenomenon table rather than the headline — an aggregate over a
mostly-clean set reports a good number while rejecting every OCR-noised document
in it. The per-signal table names the threshold to move.

The labelled format, the six failure modes it must cover, and worked example sets
are in
[ADDING_A_LANGUAGE.md](nemo_curator/data/langpacks/ADDING_A_LANGUAGE.md) § 8.

## Guardrails

- Signals fall into three groups by language dependence — agnostic, pack-backed,
  and those that assume whitespace word segmentation or an ASCII alphabet without
  saying so. The third group is the one that silently stops meaning what it meant.
  See [Language dependence](nemo_curator/profile/README.md#language-dependence).
- Profile the **unfiltered** corpus. Profiling the filtered output measures the
  gates after they have already run.
- `corpus.language` has no default. A wrong default silently produces wrong
  numbers, which is worse than an error.
- No production language pack is bundled. Three opt-in example packs ship
  (`en`, `vi`, `hi`); supply a reviewed pack root for anything else. To build one, follow
  [ADDING_A_LANGUAGE.md](nemo_curator/data/langpacks/ADDING_A_LANGUAGE.md) — the
  end-to-end procedure from an empty directory to an approved threshold.
- Gate one thing in one place. Declaring the same gate in both
  `quality_filters` and the policy builds two stages for it, and the per-gate
  breakdown is then discarded as `unattributed` rather than published wrong.
- Empty or tiny output usually means a filter is too aggressive — read the
  ledger's `filtered_by_reason` before changing thresholds.
