# banking_vn oracle pack: file map, commands, and release record

This document sits outside
[`data/banking_vn_oracle_pack/`](../data/banking_vn_oracle_pack/README.md) on
purpose. An eval verifies that the pack it executes is byte-for-byte the pack
that certified the gold traces it is scoring, and that check hashes *every* file
in the pack directory — documentation included, because a fingerprint blind to a
file cannot tell a comment from a policy the backend reads. A published
benchmark therefore freezes the whole directory: editing even the pack's
`README.md` makes every eval of `banking-vn-gold-v1-1392` fail with
`eval_source_oracle_pack_drift` until the bytes are restored.

Notes about the pack that are not themselves pack inputs belong here, where they
can be revised freely. The pack's own `README.md` is frozen at the revision
generation recorded.

That freeze extends to links. The frozen `README.md` points at
`config/banking_vn.gold.yaml` and `config/banking_vn.gold.paraphrase.yaml`, the
names those configs carried at publication; they now ship as
`publication.example.yaml` and `publication.paraphrase.example.yaml`. Repointing
the links would be a byte change like any other, so the mapping is recorded here
instead. Only the filenames, the commentary, and the declared `expt_name` and
`output_dir` changed; every key that decides what gets generated is the same.

## File map

- `manifest.yaml`: identity, Vietnamese prompts, frozen clock, primary keys,
  absent IDs, and authoritative paths.
- `tools.json`: nine strict schemas, including mutation and confirmation flags.
- `fixtures.json`: deterministic account, card, transaction, transfer, VietQR,
  and dispute state plus template slot inventory.
- `backend.py`: isolated reset/state, deterministic IDs/results, business errors,
  confirmation, and mutation semantics.
- `task_templates.yaml`: 42 templates spanning every supported turn policy.
- `assertions.py`: result, state, dependency, confirmation, error, and no-tool checks.
- `validation_cases.yaml`: direct success, invalid, not-found, safe confirmation,
  and committed-mutation probes for the tools.
- `README.md`: slot inventory, publication scale, and the declared mixes.

## Validate, smoke, and generate

From the repository root:

```bash
SMOKE="$(pwd)/src/nemotron/steps/byob/bfcl/config/smoke.example.yaml"
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config "$SMOKE" \
  --output-dir /tmp/bfcl-banking-vn-validation
python -m nemotron.steps.byob.scripts.run \
  --config "$SMOKE" \
  --stage prepare
python -m nemotron.steps.byob.scripts.run \
  --config "$SMOKE" \
  --stage generate
```

Validation exits zero only after schema, direct probes, deterministic replay,
assertions, reset, and confirmation safety pass. Inspect
`oracle_validation_report.json`, stage reports, both Parquet files, and
`run_manifest.json` beneath the `output_dir` the config declares, which is
`/tmp/bfcl/smoke_validation/` unless you change it.

The deterministic Gold profile targets 232 tasks in each of six categories:

```bash
python -m nemotron.steps.byob.scripts.run \
  --config "$(pwd)/src/nemotron/steps/byob/bfcl/config/publication.example.yaml" \
  --stage all
```

The paraphrased profile is separate because it requires model credentials and
records model-exposure provenance:

```bash
export NGC_API_KEY=REPLACE_ME
python -m nemotron.steps.byob.scripts.run \
  --config "$(pwd)/src/nemotron/steps/byob/bfcl/config/publication.paraphrase.example.yaml" \
  --stage all
```

Any of these that regenerate into the pack directory, or that are run with an
edited pack, produce a benchmark certified against different bytes. That is
correct behaviour, not a nuisance: the resulting benchmark is a different
artifact and cannot be scored against the old one.

## Reference release

The scale claims in the pack's `README.md` are not projections. This pack
produced a gold release under `publication.paraphrase.example.yaml`, and the
figures below were read from that run's `run_manifest.json`.

The release identifies itself by content rather than by path, so a copy of it is
verifiable wherever it is mounted. The identity is recorded here so that a run
you produce from these files can be compared against it:

| Identity | Value |
| --- | --- |
| `run_id` | `bfcl_banking_vn_gold_paraphrase_v1_1392-20260901T233211455629Z-277950452534-cb1472102ab04cadb74be58666e1160b` |
| pack `content_hash` | `sha256:c5bb5c39033c28dd4c6e5d7197e88db2801e85887458457088a16cccf7e740eb` |
| `benchmark.parquet` | `sha256:d40ba8d3ec5fd7778a42a0f4359feacfec14e6be08cf4a26de14c3ef922e58f6` |
| `benchmark_raw.parquet` | `sha256:e988c246dccbafbf5a2c3638f2204a8de1c10da97f85160bffb3ddbc01ae1d94` |
| `generation_config_hash` | `sha256:9dec917235992be2b2d888016ca27ecf9daab7a1f141abf530b7522ad577be48` |

The pack in this checkout still hashes to the `content_hash` above, so this
release is scoreable as it stands. `tests/steps/byob/test_bfcl_published_pack_fingerprint.py`
pins that fact per file and fails at the commit that breaks it, rather than
hours later inside someone else's eval.

Of the rest, only the pack `content_hash` reproduces from this checkout. The
config that produced this release has since been renamed and now declares an
example `expt_name` and `output_dir` instead of the release's own, and that name
reaches further than it looks: into the `run_id`, into the
`generation_config_hash` that covers the whole config document, and into each
row's `metadata` column and therefore both Parquet hashes. Restore the
`expt_name` and `output_dir` recorded in that run's `run_manifest.json` if you
need the identical artifact rather than an equivalent one. Nothing about what
gets generated changed, so a run under the example names yields the same tasks,
calls, and mixes.

The pack `content_hash` is the one that matters when reading these numbers as
evidence about the pack files: the manifest recorded it from the eight pack
files that generation actually loaded, so it ties those rows to that directory
rather than to a copy that had drifted from it. Those eight files still hash to
that value, and they do so only as long as the directory is left alone — which
is why this document is not in it.

Every count below comes from that run's `run_manifest.json`: `stage_counts` for
the funnel and `semantic_deduplication.report.actual_counts` for the realized
mixes.

Stage 4 supplied 2,824 canonical candidates, one guarded Vietnamese variant was
requested for each eligible binding (2,568 requested, 2,542 accepted, 26
rejected), and every one of the 5,366 expanded rows passed replay, schema, and
surface-quality validation. Stage 11 then selected exactly 1,392 rows and
reported no unmet target, which is the concrete form of the fail-closed claim
the pack's `README.md` makes. The realized `turn_policy` counts are
`single_turn` 265, `irrelevant` 232, `dependent_call` 209, `clarify_only` 144,
`missing_slot` 139, `correction` 111, `confirmation` 98, `multi_tool` 97, and
`negative_path` 97 — so slightly over four fifths of the release exercises
something other than a plain lookup. Difficulty landed at 348 easy, 418 medium,
and 626 hard, turns at 974 single and 418 multi, and all six categories at 232.

A release of this pack is archived as a snapshot rather than as a full run
directory, conventionally `banking-vn-gold-v1-1392/`, and reading one is the
quickest way to see what each artifact actually contains before committing to a
generation run. A snapshot keeps the published data and its manifest and nothing
else, so its layout is flatter than the `output_dir/expt_name/` layout a live run
writes: `benchmark/` holds both parquets, `run_manifest.json` sits at the top
level because everything else is traceable to it, and `exports/` holds the
`bfcl_json` question/answer pair, the six-file `nemo_evaluator_bundle`, and
`export_validation_report.json`.

The `stage_cache` tables, the separate stage reports, and the evaluation
artifacts are absent by choice. They are working state, and the manifest already
carries the parts of them a reader needs: `stage_counts` holds the full
generation funnel and `semantic_deduplication.report.actual_counts` holds the
realized category, difficulty, turn, and policy mixes, so the numbers a release
claim rests on survive without the tables that produced them.

What remains is enough to verify the published data offline. Both parquets are
byte-identical to the hashes recorded above, so `shasum -a 256
benchmark/*.parquet` is a complete integrity check that needs no network and no
pipeline. Verifying the *oracle* behind the rows is the separate question the
pack fingerprint answers, and the pack directory in this checkout is still the
revision that produced the release. A snapshot is a generation artifact either
way, so it
demonstrates nothing about scoring on its own; reproducing the recorded score
means re-running the evaluation against its `benchmark.parquet`.

The release has also been scored end to end rather than only for trace
plausibility, which is the point of shipping an executable pack: `gpt-oss-120b`
reached 53.4% task success (744 of 1,392) in combined `trace` and `executable`
mode. Treat that as a recorded result rather than a reproducible claim. The
evaluation artifacts are not kept in this repository, so auditing the figure
means re-running the evaluation against `benchmark.parquet` yourself.
