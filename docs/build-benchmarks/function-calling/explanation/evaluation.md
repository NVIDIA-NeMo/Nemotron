<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Evaluation

Evaluation is a separate run over a benchmark that was already published, and it has its own configuration file.
Generation settings and evaluation settings never share a document: `stage=generate` refuses an evaluation block outright, and the evaluation input is excluded from the generation lineage hashes.

The reason is a single sentence worth stating plainly: evaluating a new candidate must not change the identity of the benchmark it was scored on.
If one file described both the benchmark and the model under test, then swapping the model would fork the benchmark's hash, and two scores taken on the same rows would no longer be provably comparable.

The configuration names a `source_run_manifest`, never a bare Parquet file, because `run_manifest.json` is the publication commit marker and it already states which table was published, whether the run was gold-eligible, and which oracle kind produced it.
Nothing in the file defaults. Every scoring gate, runtime limit, and decoding parameter is stated, because each one changes what the resulting number means — a model cut off at two turns did not answer the same question as one given ten — and quoted booleans and numbers are refused rather than coerced, since a `"false"` that became `true` would silently switch off a correctness gate.
Resolution ends in one `eval_config_hash` taken over the configuration's meaning: referenced files enter as content hashes, and absolute paths, output locations, and secret values are absent.
Refer to {doc}`../reference/eval-config` for every key.

## Two Modes

A report always says which mode produced a number.

| Mode | What it measures | What it needs |
| --- | --- | --- |
| `trace` | Whether the model proposed the right calls, compared against the gold calls the benchmark recorded. | The published benchmark only. No oracle runs. |
| `executable` | Additionally replays the proposed calls against a live oracle session and evaluates the pack's success assertions. | A gold-eligible source run plus the exact pack manifest and its concrete backend or endpoint configuration. |

A trace score says the model asked for the right thing, not that the request would have worked.
That distinction is not academic: a model can emit well-formed calls on every attempt and still fail a large share of tasks once the oracle and the pack's assertions are in the loop, because whether a call *succeeds* depends on state the call text cannot express.
Running both modes is therefore not redundant, and the artifact contract keeps them apart: every aggregate declares the scope it measured, a report whose aggregates mix scopes is refused, and a trace-only artifact set never stands in for an executable one.

Executable mode additionally requires the oracle pack to still fingerprint to what generation certified, across every file in its tree.
Replaying against an oracle that changed since generation cannot confirm the gold trace it certified, so such a run is refused rather than reported with a caveat.

## The Fail-Closed Sequence

Each step below produces a handle that the next step needs, and there is no way to obtain a later handle without passing the earlier gate.
That is what turns the sequence into a guarantee instead of a convention: "the runner scored an unpublished table" is not a reachable state, because the only way a runner receives paths is from a verified source, and the only task list it has is the one on an authorized plan.

1. **Load the configuration.** Parsing, resolution, and hashing happen before any candidate is contacted, so an invalid configuration fails before any model request is issued. Validation is also the one place that does not stop at the first refusal: the sections constrain unrelated things and no request is sent, so every independent violation is reported in one pass.
2. **Verify the source.** The manifest is re-read and held to the hash the configuration resolved, both tables are hashed against every declaration the manifest makes about them, the publication relationship between the raw and published tables is replayed on disk, and the published rows are decoded into a unique addressable task index. A row the evaluator cannot decode aborts verification rather than being skipped, because skipping it would change the task set. For executable mode the pack fingerprint is recomputed and the backend is probed in a throwaway process worker.
3. **Check contamination.** Every model that read a published row while it was being built is named in the manifest together with the rows it read. Each candidate is compared against each exposure, strongest evidence first. A match is a violation; a comparison that cannot settle the question is recorded as unresolved and never guessed either way. The result is the eligible task plan.
4. **Re-assert the source and the plan.** Both are recomputed immediately before the first request. Verification and use are separated in time, and that gap is exactly where a source gets replaced — a regeneration into the same directory, a pack edited to make a failing task pass, a plan widened after it was authorized.
5. **Run per-candidate episodes.** Candidates run sequentially; within one candidate, authorized tasks run in publication order with a bounded number in flight. Each executable task owns one oracle session.
6. **Score each episode.** Scoring is a pure projection of recorded evidence: it contacts no provider, executes no tool, reads no clock, and re-parses no provider bytes. Re-scoring the same episode under the same authorized rules reproduces the same score identity, which is what makes a published number auditable after the endpoint it came from is gone.
7. **Aggregate.** One candidate's authorized task set, in plan order, rolls into one aggregate. A partial, reordered, or cross-candidate input is refused rather than averaged.
8. **Write artifacts.** The report, task table, manifest, and required caches are published as one immutable set.

:::{note}
Contamination policy only ever narrows. Refusing the run is the locked publication setting; dropping only the exposed rows and keeping per-candidate task sets are debug behaviors that report what was actually scored and are not publishable.
:::

## What the Metrics Say

A trace score names every gate the scoring contract defines, says whether that gate applied to the row, and, when a gate failed, which assistant turn to look at.
A gate that does not apply is reported as such rather than omitted, because a report that silently dropped the ordering gate on single-call rows could not be told apart from one where ordering was checked.
Run-level trace metrics are per-gate task rates plus `task_success_rate`; executable scores emit a separate fixed per-task taxonomy of call-level and state-level rates.
The two taxonomies use deliberately different names — a rate of tasks whose argument gate passed is not the same measurement as a rate of calls whose arguments matched — so that publishing one under the other's name cannot make two incomparable numbers look alike.

Every metric carries an integer numerator and denominator alongside its value, and a zero denominator reports a null value with a stable reason code rather than a vacuous zero or one.
Evidence that an infrastructure stop prevented from being produced makes its metric not applicable rather than a candidate failure: a broken oracle is not a wrong answer.

That accounting is also why the exported evaluator bundle declares only the metrics its own files can support.
It declares `tool_selection` and `arguments`, and `call_ordering` only when some task actually expects more than one call, since an ordering metric over single-call rows would report a perfect score for something it never measured.
It does not declare `results` or `task_success`, because both would require the pack's tools to be re-executed against oracle state, and no file in a dataset bundle provides that.
A recorded oracle result is provenance, not an answer key: scoring against a snapshot of one backend revision would measure agreement with that snapshot instead of whether the call worked.

## Calls and Text Use Different Matching Rules

There is no single exact-match rule for a conversation:

- A predicted tool name must match the expected public function name.
- Arguments first pass the declared schema and are then compared as canonical JSON.
  JSON types remain meaningful: `500000`, `"500000"`, and `500000.0` are different
  argument values.
- Publication scoring uses `intermediate_text_matching: structural` for a
  clarification, confirmation, decline, or other text-only assistant turn. The
  candidate must answer in non-empty words and must not make a tool call on that turn;
  it does not need to reproduce the pack's sentence verbatim.
- `verbatim` text matching exists as a debug setting, not the locked publication
  behavior.

The distinction matters for no-call tasks. `clarify_only` ends by asking for a slot,
and `irrelevant` ends by declining; neither expects a tool call. A candidate that
correctly makes no call passes tool selection, while the argument gate is skipped
because there are no arguments to compare. The text-turn gate still requires the
expected text behavior.

For `missing_slot` and confirmation conversations, the next scripted user message is
released only after the candidate passes the preceding text turn. A model that calls
straight through therefore never receives the hidden slot or confirmation it failed
to request.

Structural matching proves that the candidate respected the text-versus-call shape. It
does not by itself judge whether a clarification asked for the semantically correct
slot or used domain-appropriate wording. Executable assertions can add domain behavior
checks; any stronger language-quality judgment must be declared separately rather than
reported as exact match. Refer to {doc}`pipeline-worked-example` for a missing-slot
conversation and {doc}`../reference/eval-config` for the scoring fields.

## Artifacts

Both modes publish through one writer, and the file set is immutable.

| Artifact | Holds |
| --- | --- |
| `eval_report.json` | Run-level and per-candidate aggregates, each stamped with the scope it measured. |
| `eval_task_results.parquet` | One row per scored task with its gate verdicts, terminal episode status, non-candidate-stop flag, and structured failure records. |
| `eval_manifest.json` | Binds the source, configuration, and plan identities to the candidate aggregate hashes, the output byte hashes, and the byte hashes of the required caches. |
| `candidate_io_cache.jsonl` | Append-only, hash-verified request and completion records for replayable provider completions. It is validly empty when every call ended in a retryable provider or transport failure; executable episode evidence retains those failures. |
| `tool_trace_cache.jsonl` | Append-only, hash-verified complete executable episodes, written by executable runs. |

Two more files record the gates themselves. A pass writes `source_verification_report.json` and `contamination_report.json`; a refusal writes `source_verification_failure.json` or `contamination_failure.json` instead, under a different name so no reader can mistake a diagnosis for a pass by seeing which artifact is present.

The caches are replay evidence, not an optimization. A committed completion replays without network access or credentials, while an interrupted sequence is preserved as crash evidence and fails closed, so a resumed run can never read an interruption as the model's answer.
Executable episodes are cached whole rather than per call: skipping one mutating call would not reproduce the state that dependent calls, final state, and assertions depend on.
The output directory must sit outside the generation publication tree, so an evaluation run cannot overwrite `run_manifest.json` or the benchmark it scores.

## Two Boundaries the Evaluator Does Not Cross

**Pack Python never enters the evaluator process.** Both the local-backend and endpoint adapters keep reset, ordered calls, state reads, and assertions inside one task-local process worker, and endpoint sessions are deleted on every normal and exceptional exit.
Keeping the pack out of the evaluator is what makes assertion and oracle failures separable from candidate failures: an assertion that could not be imported or executed is recorded as an infrastructure outcome, never counted as a pass or a failure for the model.
It is also the only place a hanging tool can be stopped on a deadline, and a mutating call whose response is lost becomes unknown commit state rather than being retried, because losing a response does not prove the state change was rolled back.

**The gold trace never enters the prompt.** A candidate's prompt starts as the leading system messages plus the first user request. From there the only material that may enter is an assistant turn the candidate itself produced, a tool result the driver decided to release, and a scripted user request.
The conversation object exposes no general append method and re-audits provenance before every send.
A recorded tool result is released only to a call that matches the trace and is addressed to the id the candidate's own call carried, which is what makes a clarification or confirmation policy meaningful: a model that calls straight through never receives the slot value it failed to ask for.
Nothing repairs a candidate's output, either — no model, including a judge, may rewrite or complete a tool call before it is scored, because repair turns a wrong answer into a right one and makes the score a property of the repairer.

## Related Information

- {doc}`../how-to/run-evaluation` for running an evaluation end to end.
- {doc}`../reference/eval-config` for every evaluation YAML key.
- {doc}`../reference/output-files` for artifact locations.
- [`src/nemotron/steps/byob/references/bfcl-eval-scoring-contract.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/byob/references/bfcl-eval-scoring-contract.md) for the normative definition of what a score means.
- {doc}`pipeline-overview` and {doc}`oracle-pack` for how the benchmark and its oracle were built.
