# Walking the assisted authoring flow end to end

`scripts/bfcl_assisted_authoring_demo.py` runs the entire assisted authoring flow in one
command: a reviewed source package is certified by live probes, an authoring model drafts
what that source can support, the drafts plus reviewed semantics become a candidate pack,
and the pack is validated, reviewed, frozen, published into a benchmark, and scored by a
real evaluation run.

Its purpose is to make the flow observable end to end, so that you can watch each gate and
each artifact instead of reading about them. It is a demonstration, not a proof: the
guarantee that a gate holds comes from the test that owns it, named in
[bfcl-authoring-support-matrix.md](bfcl-authoring-support-matrix.md).

This page orients you. To run the flow yourself command by command, use
[bfcl-assisted-authoring-runbook.md](bfcl-assisted-authoring-runbook.md): it spells out
every refusal and recovery, along with setup, live-model configuration, per-stage
artifacts, an intentionally failing scorer run, and evaluation against an independent
candidate endpoint.

```shell
export BFCL_DEMO_ROOT="${TMPDIR:-/tmp}/bfcl-assisted-authoring-walkthrough"
uv run python scripts/bfcl_assisted_authoring_demo.py --workdir "$BFCL_DEMO_ROOT"
```

The run takes a few minutes, most of it real probe sessions and two unmocked validation
passes. It refuses a workdir that already exists, because a run that reuses state is not
the flow being demonstrated.

## What is simulated, and what is not

Two things stand in for people, and both announce themselves in the output.

The authoring model is scripted by default, so the demo needs no credentials: the drafting
prompts are built and cached exactly as they would be, and canned stage answers come back.
`--author-model live` sends those same prompts to a real endpoint instead.

The four human review points are answered from constants, each printed as
`[simulated human review]` with the decision a reviewer would have been making:

| Step | What a person decides |
| --- | --- |
| Authorize | whether this source may be exposed to a model at all |
| Approve evidence | whether the normalized bundle describes the source they reviewed |
| Reviewed semantics | slot bindings, turn policies, and per-language user turns, which no draft schema may express |
| Release checklist | semantics, descriptions, held-out policy, assumptions, validation evidence, certification, pre-model authorization, and answered questions |

Nothing else is faked. Intake probes a real package in a real worker, certification
signatures are real, both validation passes run unmocked and derive their own tier, freeze
really seals the pack read-only, and evaluation runs the shipped scorer.

## The nine steps

1. **Intake.** `bfcl_author author` probes the source package and certifies it at A2,
   requiring a held-out decision before the first model call.
2. **Authorize.** The model-exposure boundary, bound to the resolved configuration digest.
3. **Approve evidence.** The evidence boundary, bound to both bundle digests.
4. **Draft.** `bfcl_author draft` produces the coverage plan, validation cases, task
   templates, and assertion specs, cached immutably for replay.
5. **Assemble.** `bfcl_author assemble` binds drafts, evidence, and the reviewed
   supplement into a candidate pack, and records every digest.
6. **Validate.** Unmocked `prepare_bfcl` derives the tier; the demo stops unless the pack
   is Gold-eligible.
7. **Review, approve, freeze.** The review packet, the release approval it pins, and the
   frozen pack the approval seals.
8. **Publish.** A fresh Gold validation, then benchmark generation, committed by
   `run_manifest.json`.
9. **Evaluate.** A real evaluation run against a candidate served on loopback, in
   `mode: [trace]`; the demo scores recorded traces and does not execute a live oracle.

## Reading the evaluation

The demo's candidate answers from the benchmark's own recorded assistant turns, keyed by
what the user last said and how many tool results have come back. A clean run therefore
scores 1.0 across every metric, which is the point: it shows the flow produced a benchmark
that is passable, with assertions that fire and traces that complete.

A scorer that only ever reports success proves nothing, so the same published benchmark
can be re-scored with one task sabotaged — the candidate answers it with text where a call
was expected:

```shell
uv run python scripts/bfcl_assisted_authoring_demo.py --workdir "$BFCL_DEMO_ROOT" \
    --stage eval --wrong-answer-task <task_id>
```

That run drops `task_success_rate` and names the failure codes with their attribution, so
the difference between a benchmark that cannot fail and a candidate that did is visible in
one comparison. Each `--stage eval` run writes a fresh `eval-N` directory, because a
committed eval report belongs to the run that wrote it.

The demo's candidate is a scripted loopback server, so there is no commit and no weight
digest to point at. It is declared unpinned rather than given a plausible-looking one, and
the report says so: `canonical_id` ends in `@provider_managed`, `publication_allowed` is
false, and `non_publication_reasons` names the candidate. Its `model_identity` restates the
serving route, which is all an unpinned candidate is allowed to claim, and the config
declares `schema_version: "1.2"`, the first schema that reads an unpinned candidate. That is
the same shape a hosted frontier model takes when its provider publishes neither value — the
score is real, and it does not pretend to be reproducible. Use
`python -m nemotron.steps.byob.scripts.resolve_bfcl_model_identity` to produce the block
for a candidate that *can* be pinned.

Frozen release artifacts are read-only by design, so removing a finished workdir needs
write permission restored first (`chmod -R u+w`).
