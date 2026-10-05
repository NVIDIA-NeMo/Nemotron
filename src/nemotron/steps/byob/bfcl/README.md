# BYOB BFCL

The `bfcl` benchmark family builds function-calling benchmark artifacts — Apache
Parquet rows plus a manifest that pins their provenance — from an executable
oracle pack, and it evaluates candidate models against the result.

One thing separates it from the MCQ family: generation is driven by the pack
rather than by a model. The pack's templates define the conversation, and the
pack's backend or HTTPS endpoint, together with its assertions, establish what
correct tool behavior is. Every task is replayed against that oracle before it is
allowed into the benchmark, so every expected tool call is one the oracle actually
reproduced. That is why a published row can state why its answer is correct
instead of asserting that a model once agreed with it.

This file is an entry point. Operator guides live on the documentation site and
the normative contracts live in [`../references/`](../references/); both are
indexed under [Where to go next](#where-to-go-next).

## Quick Start

Install the BYOB dependencies:

```bash
uv sync --extra byob
```

Run the bundled tiny reference pack from the repository root:

```bash
nemotron steps run byob/bfcl -c src/nemotron/steps/byob/bfcl/config/tiny.yaml stage=all family=bfcl
```

`config/tiny.yaml` reads [`../data/tiny_oracle_pack`](../data/tiny_oracle_pack/README.md),
generates four rows, and writes them under `/tmp/bfcl/tiny_out`. It pins
`lineage.policy: smoke_no_publication`, so its output is deliberately not
publication-eligible; it exists to prove that the plumbing, the process
isolation, and the output paths work before a real pack does.

Paths in a generation config resolve against the BYOB root,
`src/nemotron/steps/byob`, rather than against your shell's working directory or
the config file's own location, so run the command from the repository root or
make the paths absolute.

To validate a pack without generating rows, use `stage=prepare`, or run the same
report outside the step, which is the faster loop while a pack is not yet
gold-eligible:

```bash
python -m nemotron.steps.byob.scripts.validate_oracle_pack --config <CONFIG>
```

## Pipeline

`stage` selects what a run does. Translation and evaluation are separate runs over
a benchmark that was already published, not stages of generation.

| `stage` | What the run does |
| --- | --- |
| `prepare` | Normalize and validate the oracle pack, then write the validation report. No benchmark rows are produced. |
| `generate` | Require a gold-eligible pack, generate tasks, replay them against the oracle, and publish artifacts. |
| `translate` | Localize an already published benchmark without changing oracle truth. |
| `eval` | Score candidate models using a separate evaluation configuration. |
| `all` | Run `prepare` followed by `generate`. It does not implicitly translate or evaluate. |

A full generation run is twelve stages. Stage 1 prepares the pack and Stage 2 is
the gold-eligibility gate that decides whether generation may proceed at all. The
remaining ten turn templates into published rows: `reference_profile`, `expand`,
`state_machine`, `render`, `expected_trace`, `schema_validation`,
`executable_replay`, the optional `surface_quality` and `dedup_balancing`, and
`final_output`. A disabled optional stage is bypassed rather than run as a no-op,
so it leaves behind no artifact a later reader could mistake for a verdict it
never reached.

Each stage writes one table under `stage_cache/` keyed by `task_id`, plus a
verified checkpoint. `skip_until=<stage>` resumes by running the named stage and
every later enabled stage, after recursively verifying the immediate enabled
predecessor's manifest, state, artifact hashes, task order, config identity, and
pack and endpoint identities. Any drift fails closed.

Three authoring routes produce a reviewed pack before Stage 1 — writing one by
hand, drafting one from a conventional Python package or HTTPS service with model
assistance, or onboarding a running MCP server — and all three converge on the
same generation stages and the same gold gate, so the authoring route never earns
a weaker guarantee.

Both shapes are drawn rather than only described:
[authoring flows](../../../../../docs/build-benchmarks/function-calling/explanation/authoring-flows.md)
diagrams the three routes converging on the gold gate, and
[pipeline overview](../../../../../docs/build-benchmarks/function-calling/explanation/pipeline-overview.md)
expands the twelve stages with the artifact each one writes.
The adjacent
[worked example](../../../../../docs/build-benchmarks/function-calling/explanation/pipeline-worked-example.md)
follows one missing-slot task from its template through binding, conversation planning,
executable replay, selection, and publication.

## Bundled Configurations

`config/` holds nine runnable files: a tiny plumbing run, an annotated template whose
pack path is a placeholder so it cannot publish an example domain by omission, a
domain-sized smoke run, two publication profiles, three evaluation envelopes, and a
localization config. Each one is described in
[the configuration reference](../../../../../docs/build-benchmarks/function-calling/reference/index.md#bundled-configurations).

Every number in those files is a worked example for the pack it points at, not a
framework default. Copying another pack's task counts and diversity limits is the
most common way to make the balancing stage infeasible.

Model roles are opt-in and disabled in the shipped templates. An enabled role is
routed by Data Designer, so the provider it names must exist in that installation
and the environment variable named by `api_key_env` must be exported; the config
records the route identity and does not create the provider.

## Outputs

Artifacts are written to `output_dir/expt_name/`. Three of them carry most of the
weight:

- `benchmark.parquet` is the published benchmark. `benchmark_raw.parquet` beside
  it holds every schema-valid, replay-valid row, and the difference between the
  two is a selection and never a rewrite: a published row is byte-identical to
  its raw counterpart across every column.
- `run_manifest.json` is the publication commit marker, moved into place last. If
  it is absent, any parquet or export beside it is unpublished bytes whatever the
  file names say. It pins the pack's content hash, the config hashes, the seeds,
  the stage counts, and one content hash per artifact.
- The `stage_cache/` tables are how a run is diagnosed. Every table carries the
  same `task_id` set, so joining two adjacent ones shows exactly which stage
  dropped a task and therefore which part of the pack to fix. A task present in
  `expected_traces.parquet` but missing from `replay_validated_tasks.parquet` is
  the most informative case: the pack claimed a behavior its own backend did not
  reproduce.

Optional compatibility exports and the artifacts an evaluation writes are
documented in the reference pages below.

## Where to Go Next

Operator guides on the documentation site, ordered from first run to release:

| Guide | What you will do |
| --- | --- |
| [About building function-calling benchmarks](../../../../../docs/build-benchmarks/function-calling/index.md) | Orient yourself and pick a path |
| [Getting started](../../../../../docs/build-benchmarks/function-calling/getting-started.md) | Run the tiny pack end to end and inspect what it wrote |
| [From domain assets to an Oracle Pack](../../../../../docs/build-benchmarks/function-calling/how-to/start-from-domain-data.md) | Choose manual or model-assisted authoring for an existing interface, records, implementation, or behavior specification |
| [Hand-author an oracle pack](../../../../../docs/build-benchmarks/function-calling/how-to/author-a-pack.md) | Scaffold, fill in, validate, and smoke-run a pack of your own |
| [Assisted authoring](../../../../../docs/build-benchmarks/function-calling/how-to/assisted-authoring.md) | Draft a pack from a Python package or HTTPS service with model assistance |
| [Onboard an MCP server](../../../../../docs/build-benchmarks/function-calling/how-to/mcp-server.md) | Use a running MCP server as the oracle |
| [Publish a release](../../../../../docs/build-benchmarks/function-calling/how-to/publish-a-release.md) | Choose a budget and challenge mix, then verify the artifacts |
| [Translate a published benchmark](../../../../../docs/build-benchmarks/function-calling/how-to/translate.md) | Localize approved model-facing text without changing oracle truth |
| [Evaluate a candidate model](../../../../../docs/build-benchmarks/function-calling/how-to/run-evaluation.md) | Score models and read the report |
| [Oracle Pack inputs](../../../../../docs/build-benchmarks/function-calling/reference/oracle-pack-inputs.md) | Find the required files, manifest fields, fill order, and cross-file tool lineage |
| [Domain brief](../../../../../docs/build-benchmarks/function-calling/reference/domain-brief.md) | Prepare the human-owned domain context required by assisted authoring |
| [Probe plan](../../../../../docs/build-benchmarks/function-calling/reference/probe-plan.md) | Prepare the intake cases that earn certification tier A2 |
| [Manifest fields](../../../../../docs/build-benchmarks/function-calling/reference/manifest.md) | Configure identity, paths, languages, shared text, and confirmation vocabulary |
| [Tool catalog and fixtures](../../../../../docs/build-benchmarks/function-calling/reference/tools-and-fixtures.md) | Define public schemas and deterministic reset records |
| [Python backend contract](../../../../../docs/build-benchmarks/function-calling/reference/python-backend.md) | Implement the four required local-oracle callables |
| [Task template fields](../../../../../docs/build-benchmarks/function-calling/reference/task-templates.md) | Look up slots, milestones, policies, and worked conversation shapes |
| [Assertion contract](../../../../../docs/build-benchmarks/function-calling/reference/assertions.md) | Define deterministic replay predicates and capabilities |
| [Validation case fields](../../../../../docs/build-benchmarks/function-calling/reference/validation-cases.md) | Define direct success, rejection, and confirmation probes |
| [Endpoint configuration](../../../../../docs/build-benchmarks/function-calling/reference/endpoint-config.md) | Pin endpoint identity, credentials, TLS, and attestation |
| [Held-out policy](../../../../../docs/build-benchmarks/function-calling/reference/held-out-policy.md) | Reserve fixture and template identities from generation |
| [Generation config reference](../../../../../docs/build-benchmarks/function-calling/reference/generate-config.md) | Look up a generation YAML field |
| [Evaluation config reference](../../../../../docs/build-benchmarks/function-calling/reference/eval-config.md) | Look up an evaluation YAML field |
| [Output files](../../../../../docs/build-benchmarks/function-calling/reference/output-files.md) | Find what every written path contains |
| [Troubleshooting](../../../../../docs/build-benchmarks/function-calling/reference/troubleshooting.md) | Map a refusal message to its fix |

Concept pages explain why the pipeline is shaped the way it is:
[pipeline overview](../../../../../docs/build-benchmarks/function-calling/explanation/pipeline-overview.md),
[pipeline worked example](../../../../../docs/build-benchmarks/function-calling/explanation/pipeline-worked-example.md),
[the oracle pack](../../../../../docs/build-benchmarks/function-calling/explanation/oracle-pack.md),
[authoring flows](../../../../../docs/build-benchmarks/function-calling/explanation/authoring-flows.md),
and [evaluation](../../../../../docs/build-benchmarks/function-calling/explanation/evaluation.md).

The normative contracts are the engineering-facing source of truth. Where a docs
page and a contract disagree, the contract is authoritative, because the pipeline
content-hashes some of these documents into the identity of what it publishes.

| Contract | Defines |
| --- | --- |
| [`../references/bfcl-oracle-pack.md`](../references/bfcl-oracle-pack.md) | The complete pack contract: file layout, manifest keys, backend and endpoint contracts, template and surface requirements, turn policies, slot sources, validation cases, tiers, every generation stage including surface quality, deduplication and balancing, and held-out enforcement, the compatibility exports, and the eval config and source-verification rules |
| [`../references/bfcl-eval-scoring-contract.md`](../references/bfcl-eval-scoring-contract.md) | What a score means: argument matching, call selection, grouping and order, how candidate output is observed, how a conversation advances, every gate and its attribution, trace and executable aggregation, private held-out generalization, the native adapter, CLI orchestration, task success, repair, determinism, and contamination |
| [`../references/bfcl-bias-audit-contract.md`](../references/bfcl-bias-audit-contract.md) | The read-only post-release audit: its evidence binding, the metric per audit dimension, and the complete command |
| [`../references/bfcl-authoring-support-matrix.md`](../references/bfcl-authoring-support-matrix.md) | Which assisted-authoring surfaces are supported, experimental, or unimplemented, with the test that evidences each one |
| [`../references/bfcl-mcp-support-matrix.md`](../references/bfcl-mcp-support-matrix.md) | The same for MCP transport behavior |

Also useful: [`../references/bfcl-manual-oracle-pack-flow.md`](../references/bfcl-manual-oracle-pack-flow.md)
for the hand-authored lifecycle end to end, from a new pack through publication to
evaluation, including endpoint identity pins;
[`../references/bfcl-endpoint-config.example.yaml`](../references/bfcl-endpoint-config.example.yaml)
for a complete endpoint-backed pack configuration; and
[`../references/bfcl-authoring-user-guide.md`](../references/bfcl-authoring-user-guide.md)
as the index to the assisted-authoring contracts.

Assisted authoring takes three reviewed inputs a model may not write for you, and each
ships as a worked example taken from a published release: a domain brief
([`../references/bfcl-domain-brief.example.txt`](../references/bfcl-domain-brief.example.txt)),
a probe plan that must reach the certification tier you need
([`../references/bfcl-probe-plan.example.json`](../references/bfcl-probe-plan.example.json)),
and the organizational defaults a guided session should not ask for twice
([`../references/bfcl-authoring-policy.example.yaml`](../references/bfcl-authoring-policy.example.yaml)).

Two packs ship under [`../data/`](../data/). `tiny_oracle_pack` is the smallest
working example. `banking_vn_oracle_pack` is the reference pack: it declares a
template for every conversation policy the pipeline supports, and no template
narrows `tools_present`, so every row must select its calls out of the full tool
catalog. Read it as a worked example of the pack contract rather than as a set of
defaults. Its file map, commands, and release record are held outside the pack
directory, in
[`../references/bfcl-banking-vn-pack-operations.md`](../references/bfcl-banking-vn-pack-operations.md),
so that editing the notes is not an edit to the pack — publishing a benchmark
freezes every byte of the pack it was generated from.

## Capability Matrix

A capability that is not wired is gated: generation and evaluation refuse a
configuration that asks for one rather than accepting the key and ignoring it.

| Capability | Availability | What it covers |
| --- | --- | --- |
| Reference profiling | Supported | Normalizing content-addressed style samples into a cached profile without exposing oracle truth. |
| Model paraphrasing | Supported | Requesting one structural surface style per binding, under Python guards that preserve values, hidden slots, tool-name boundaries, turn shape, and variant distinctness. |
| Surface quality validation | Supported, optional stage | The six-check contract, the surface-only judge, its advisory or drop authority, and publication-row filtering. |
| Semantic deduplication and balancing | Supported, optional stage | Masked-surface projection, clustering, coverage-safe representative selection, declared mixes and diversity caps, and selection-rank publication order. |
| Held-out enforcement | Supported | Refusing reserved templates and fixture rows at binding time, re-scanning every row before publication, and recording the policy in run lineage. |
| Compatibility exports | Supported | Emitting, reading back, validating, and transactionally publishing the BFCL JSON pair and the NeMo Evaluator input bundle from one canonical projection. |
| Stage resume | Supported | Resuming Stages 3 through 12 from a recursively verified predecessor checkpoint. |
| Evaluation and scoring | Supported | Config resolution, source verification, contamination gating, native function-calling transport, deterministic trace driving and scoring, process-isolated executable oracle sessions with pack assertions, run-level aggregation, immutable scope-stamped artifacts, the NeMo Evaluator native bridge, and CLI orchestration. |
| Bias audit | Supported | Recomputing every audit dimension from frozen release and evaluation evidence, without modifying the source artifacts. |
| Translation and localization | Partial | Localizing benchmark surfaces while preserving executable calls and oracle assertions. It never filters rows and does not support `skip_until`. |
