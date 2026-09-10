<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Author a Pack with Model Assistance

Use this guide to produce a reviewed Oracle Pack from a conventional source package, with a model drafting the parts a model is allowed to propose. The guided command is `bfcl_author.py`. It is stateful: each subcommand binds its output into a session, and when a gate refuses progress it prints the next safe command.

A model in this flow may propose a tool coverage plan, validation cases, task-template plans, and declarative assertion specifications. It may not change the backend, the endpoint behavior, the tool schemas, or the fixtures, and it may not certify its own output, invent fixture bindings, approve anything, or bypass executable Gold validation. Everything it proposes passes through the same replay and Gold gate as a hand-written pack, which is why {doc}`author-a-pack` and this guide converge on one publication contract.

If you are starting from domain records and behavior rather than an already prepared
source package, begin with {doc}`start-from-domain-data`. It lists the interface,
state, behavior, and conversation inputs to identify, then compares the manual and
model-assisted paths.

During supplement and candidate-pack review, use the artifact index in
{doc}`../reference/oracle-pack-inputs`. Its dedicated references cover the manifest,
tool catalog and fixtures, backend or endpoint, templates, assertions, validation
cases, and held-out policy with the same create-contract-example-validate structure.

This page is the walkthrough. `src/nemotron/steps/byob/references/bfcl-authoring-user-guide.md` is the matching command-level reference: it lists every subcommand and refusal code, and its invocations are executed as smoke cases by the test suite, so consult it when you need exact arguments rather than the shape of the flow.

:::{tip}
To watch the whole flow run before you prepare a source of your own, use the
credential-free assisted-authoring walkthrough. It needs no credentials and no
endpoint: the authoring model is scripted and the candidate is served on loopback,
while intake probes a real package and validation derives its own tier unmocked. Run
it from the repository root, because the script path is relative:

```bash
uv run python scripts/bfcl_assisted_authoring_demo.py --workdir /tmp/bfcl-demo
```

It certifies a source, drafts and assembles a candidate pack, validates, reviews,
freezes, publishes a benchmark, and scores it, printing at each human gate what a
reviewer would have been deciding. The `_demo.py` suffix correctly marks it as a
non-normative walkthrough rather than a production launcher. Pass
`--author-model live` to send the same prompts to a real endpoint instead.
:::

## Before You Start

- Install the BYOB dependencies with `uv sync --extra byob`, and prepare a source package in one of the two supported layouts below.
- Prepare a domain brief, a reviewed statement of what the source is for, which is sanitized and bound into the evidence. Copy `src/nemotron/steps/byob/references/bfcl-domain-brief.skeleton.txt` and replace every bracketed `BFCL-SKELETON` block; intake rejects the copy while even one remains. `bfcl-domain-brief.example.txt` is the same form filled in, being the brief a published release was authored from. Prefer the skeleton for a new source, since copying the example tends to carry its banking framing across with it. See {doc}`../reference/domain-brief` for its content and safety contract.
- Prepare a probe plan, which you need for certification tier A1 or A2 and therefore
  for a Gold release. See {doc}`../reference/probe-plan`.
  `src/nemotron/steps/byob/references/bfcl-probe-plan.example.json`
  is a complete A2-shaped banking example: copy its structure, then replace its tools,
  fixture ids, cases, and domain assumptions.
- Have a certification key pair and its allowlisted key identifier available.
- Organizational defaults that should not be retyped per session belong in a reviewed policy file; see `src/nemotron/steps/byob/references/bfcl-authoring-policy.example.yaml`.
- Configure the authoring model through NeMo Data Designer. See
  {doc}`../reference/data-designer-provider` for creating `DATA_DESIGNER_HOME`,
  registering a provider, referencing credentials, and pinning model identity.

### Inputs, Outputs, and Ownership by Step

Use this map to decide what an operator must supply and what the pipeline writes:

1. **Prepare the source.** The operator supplies `tools.json`, a domain brief, and
   either a reviewed executable source or enough independently reviewed behavior and
   state to complete a scaffold. Scaffolding may write `backend.py`, `fixtures.json`,
   and `dependency-lock.json`; the operator owns the resulting behavior.
2. **Prepare probes.** The operator reviews `probe-plan.json`.
   `check_probe_plan` writes a readiness verdict; it does not certify A2.
3. **Run intake.** `author` reads the reviewed source, brief, probe plan, held-out
   decision, and certification key. It writes fingerprinted observations,
   certification, sanitized evidence, and a model-exposure subject.
4. **Cross the pre-model boundary.** A human or organizational policy authorizes
   exposure, then a human approves the exact evidence used for drafting.
5. **Draft proposals.** `draft` writes bounded coverage, validation-case,
   task-template, and assertion proposals with model-call provenance. A human reviews
   grounding, coverage, compilation, and unresolved blockers.
6. **Assemble and validate.** The operator supplies `reviewed-supplement.yaml`.
   `assemble` writes a candidate Oracle Pack and provenance; candidate validation
   writes a Gold or non-Gold verdict.
7. **Review, approve, and freeze.** `review` writes a packet for the exact candidate
   and current validation. A reviewer approves that packet; `freeze` seals its bytes.
8. **Publish.** `publish` reruns Gold validation and writes `benchmark.parquet`,
   `benchmark_raw.parquet`, and `run_manifest.json`.

Generated evidence, provenance, reports, frozen bytes, Parquet tables, and manifests
are pipeline-owned. Correct the upstream operator-owned source, plan, supplement, or
configuration and rerun its gate; do not patch generated outputs.

### Source layouts

A `local_python` source requires `backend.py` as its only import-closure root, a reviewed `tools.json`, and a canonical `dependency-lock.json`, and may add `fixtures.json`. An `http_package` source requires a strict, secret-free `endpoint_config.yaml` and its companion `tools.json`. `src/nemotron/steps/byob/references/bfcl-conventional-source-packages.md` is the normative description of both, including the dependency-lock format and the static-inspection rules.

### Create and pre-check authoring inputs

The domain brief is human-owned. Copy the skeleton, remove every bracketed marker, and
write only reviewed, non-secret domain context:

```bash
cp src/nemotron/steps/byob/references/bfcl-domain-brief.skeleton.txt \
  /srv/sources/domain-brief.txt
```

There is no standalone brief validator. Intake checks size, encoding, skeleton markers,
and sanitization before binding the brief into evidence. The complete requirements and
a domain-neutral example are in {doc}`../reference/domain-brief`.

A probe plan may be written from the reviewed source contract or drafted by a model
for human review:

```text
python -m nemotron.steps.byob.scripts.draft_probe_plan \
  --source /srv/sources/warehouse-package \
  --domain-brief /srv/sources/domain-brief.txt \
  --output /srv/sources/probe-plan.json \
  --clock <FROZEN_ISO_8601_TIME> \
  <pinned model identity and provider arguments>
```

The model returns a structured draft; the script validates its shape before writing.
It does not certify the plan or its source. Check the reviewed plan without executing
probes:

```bash
python -m nemotron.steps.byob.scripts.check_probe_plan \
  --source /srv/sources/warehouse-package \
  --probe-plan /srv/sources/probe-plan.json
```

The check reports static coverage gaps that would block A2. Intake remains
authoritative because only it executes the probes and observes reset, isolation,
confirmation, timeout cleanup, and result behavior.

### Optionally Scaffold A Local Source

If no independently implemented local source exists, generate the mechanical
four-function interface and fill its domain decisions manually:

```bash
python -m nemotron.steps.byob.scripts.scaffold_source_package \
  --tools /srv/sources/warehouse-package/tools.json \
  --output /srv/sources/warehouse-package \
  --collection assets \
  --dependency-lock
```

That writes the interface, one raising handler per published tool, and a fixture row
per value a published call has to be given. Every generated file carries `BFCL-TODO`
on each decision the catalogue could not make, and intake refuses the source while one
remains.

`--draft-with-model`, together with `--domain-brief` and pinned model identity flags,
is an optional fallback. It lets a model propose fixture data and behavior in a fixed
declarative vocabulary, which the command compiles rather than accepting model-written
Python.

:::{caution}
Do not prefer model-assisted source drafting over an existing domain-owned
implementation or independently authored backend. Authoring-model priors can shape
fixtures, errors, transitions, and vocabulary, biasing the resulting task distribution
or favoring familiar conventions. Gold validation proves deterministic consistency,
not semantic neutrality. Review proposals against independent specifications, records,
and tests; do not let one model be the sole source of both oracle behavior and task
semantics.
:::

Check the result against its own catalogue before spending an intake run on it:

```bash
python -m nemotron.steps.byob.scripts.check_source_package \
  --source /srv/sources/warehouse-package
```

It exits `0` when the source agrees with `tools.json`, and `2` with a finding per disagreement — a missing `get_state`, a name `list_tools` publishes that the catalogue does not, a placeholder still in place. It cannot tell you the behaviour is right; only the probes do that.

### Enable the adapter

Live source intake is disabled for every adapter by default.

```bash
export BFCL_ENABLE_LOCAL_PYTHON=1
```

The equivalent variable for the HTTP adapter is `BFCL_ENABLE_HTTP_PACKAGE`. Without the variable, or an equivalent reviewed `adapter_rollout` policy block, intake fails with `adapter_rollout_disabled`. Accepted values are `1`, `true`, `yes`, `0`, `false`, and `no`, case-insensitive. Offline artifact verification, review, approval, and freeze do not require the flag; only operations that inspect or execute a source do.

:::{important}
Publication currently supports `local_python`. An `http_package` source reaches intake, drafting, review, and freeze, but publication is deliberately refused until an independently verified publication adapter exists. Plan for that limitation before you onboard an HTTP source, because the refusal arrives at the last step.
:::

## Step 1: Intake and Certification

`author` runs source intake and produces transport-neutral evidence. Two decisions are required at this first command rather than deferred, because evidence that has already been collected cannot be retroactively declared clean.

```bash
python -m nemotron.steps.byob.scripts.bfcl_author \
  --ci author \
  --workspace /srv/bfcl/authoring/warehouse \
  --source /srv/sources/warehouse-package \
  --brief /srv/sources/domain-brief.txt \
  --adapter local_python \
  --pack-id warehouse_assets \
  --pack-version 0.1.0 \
  --required-tier A2 \
  --held-out-not-applicable-reason "The asset catalog is public reference data." \
  --held-out-reviewed-by reviewer@example.test \
  --certification-private-key /srv/bfcl/keys/certification-private.pem \
  --certification-key-id warehouse-authoring \
  --probe-plan /srv/sources/probe-plan.json
```

The first decision is held-out status: supply either `--held-out-policy` or `--held-out-not-applicable-reason`, always together with `--held-out-reviewed-by`. The second is the probe plan, because A1 and A2 are earned from observed probe outcomes and nothing else can supply them. `--adapter auto` is the default and recognizes any supported layout, though naming the adapter is clearer in automation; `--ci` guarantees the command never prompts; and flags after the guided ones, including the certification key flags above, are delegated to the underlying intake command.

The probe plan is one transport-neutral document. It names a case per published tool, at least one structured error if the source has error codes, and a case the tool cannot finish inside the deadline; without that last case, timeout cleanup stays unobserved and certification cannot reach A2. Certification then awards one of three tiers. A0 proves source identity and catalog integrity. A1 adds bounded read-only observation. A2 adds deterministic reset, episode isolation, confirmation safety, mutation declaration, timeout cleanup, and result coverage. Stop if the report does not record the tier you need: drafting may inspect lower tiers, but a Gold freeze requires A2, and no approval can raise a certification tier.

## Step 2: Answer the Open Questions

Intake may raise digest-bound open questions about the source. Apply the reviewed answers as a new evidence revision:

```bash
python -m nemotron.steps.byob.scripts.bfcl_author answer \
  --workspace /srv/bfcl/authoring/warehouse \
  --evidence <EVIDENCE_BUNDLE_JSON> \
  --questions <OPEN_QUESTIONS_JSON> \
  --answers <REVIEWED_ANSWERS_JSON>
```

## Step 3: Authorize Exposure, Then Approve the Evidence

A named human or organizational policy first authorizes the exact redacted evidence
subject for exposure to a model. This is the pre-model boundary, and it is the reason
the evidence bundle is sanitized before anything is sent anywhere. Supply either
`--authorized-by` or `--organizational-policy-digest`, not both; `--output` writes the
record to a path you choose.

```bash
python -m nemotron.steps.byob.scripts.bfcl_author authorize \
  --workspace /srv/bfcl/authoring/warehouse \
  --subject <MODEL_EXPOSURE_SUBJECT_JSON> \
  --authorized-by reviewer@example.test

python -m nemotron.steps.byob.scripts.bfcl_author approve \
  --workspace /srv/bfcl/authoring/warehouse \
  --boundary evidence \
  --approved-by reviewer@example.test \
  --source-bundle-digest <SOURCE_EVIDENCE_BUNDLE_DIGEST> \
  --normalized-bundle-digest <NORMALIZED_EVIDENCE_BUNDLE_DIGEST> \
  --output /srv/bfcl/authoring/warehouse/evidence_approval.json
```

Read each digest from the `bundle_digest` field of the corresponding verified
evidence-bundle artifact; do not substitute a filesystem checksum. For a normal
non-migration intake, the source and normalized bundle are the same artifact, so both
arguments use the `bundle_digest` from `intake/evidence_bundle.json`. For a migrated
intake they can differ: `--source-bundle-digest` names the original source evidence,
while `--normalized-bundle-digest` names the normalized evidence approved for drafting.
The `--output` path stores the immutable approval record that the later `draft` command
must bind.

:::{important}
Evidence approval and release approval are different decisions, and the first cannot be replaced by the second. Evidence approval says a reviewer inspected this exact source and normalized evidence and considers it fit to draft from. Release approval, later, says a reviewer inspected the finished pack and its fresh validation and considers it fit to publish. Approving the release does not retroactively authorize the model exposure that already happened, so the command sequence requires both in order.

The workflow enforces separate decision records and digests, not separate identities.
The example deliberately uses `reviewer@example.test` at both pre-model gates because
the implementation does not require two people. An organization may instead assign
different source owners, evidence reviewers, and release reviewers when its own
separation-of-duties policy requires that.
:::

Both approvals are digest-bound, so if the source, brief, redaction, observations, certification, or resolved authoring configuration changes afterwards, the approval goes stale and must be redone against the new digest.

:::{note}
From drafting onward, the short command blocks below show the guided subcommand and
operator-owned arguments, not a standalone invocation to copy without session output.
The current session supplies some paths while the delegated command still requires
artifact, model, key, digest, or output arguments printed by the previous gate. Use
`src/nemotron/steps/byob/references/bfcl-authoring-user-guide.md` for the exact argument
contract, or run `scripts/bfcl_assisted_authoring_demo.py` for a complete executable
sequence.
:::

## Step 4: Draft

```text
python -m nemotron.steps.byob.scripts.bfcl_author draft \
  --workspace /srv/bfcl/authoring/warehouse \
  <artifact, certification, approval, key, output, and model arguments>
```

Drafting issues bounded, cached, structured model requests and stops at proposals: it writes them beside a pack rather than into one. Unknown tools, unsupported assertions, ungrounded arguments, malformed output, and cache conflicts all fail closed.

## Step 5: Assemble the Candidate Pack

```bash
python -m nemotron.steps.byob.scripts.bfcl_author assemble \
  --workspace /srv/bfcl/authoring/warehouse \
  --supplement /srv/bfcl/authoring/warehouse/reviewed-supplement.yaml \
  --output /srv/bfcl/authoring/warehouse/candidate-pack
```

The assembler derives everything mechanical from evidence that is already trusted: pack identity and the manifest from the verified bundle, `tools.json` from the certified catalog, and `assertions.py` from drafts that compiled without a blocker. For a `local_python` source, `backend.py` and `fixtures.json` are copied byte for byte from the fingerprinted tree; for a session-based source there is nothing to copy, so the pack names the certified endpoint and takes its fixtures from the reviewed probe plan those sessions were opened with. What remains is what a model that has only read a catalog must not state: slot bindings to fixture columns, turn policies, per-language user turns, and the validation cases that decide the tier. Those arrive in one reviewed `bfcl-candidate-pack-supplement-v1` YAML file, and every tool and assertion it names is checked back against the evidence and the compiled assertions, so a supplement cannot introduce a tool the source never published or an assertion nobody compiled. For a pack assembled outside a guided session, `assemble_candidate_pack.py` takes the same inputs explicitly through `--evidence`, `--source`, `--drafts`, `--supplement`, `--output`, and `--probe-plan`.

If candidate validation finds a supplement defect, correct the supplement and run
`bfcl_author assemble` again with a new output path before review. The guided session
retains the previous candidate as immutable history and binds the replacement candidate
in a child revision.

## Step 6: Build the Review Packet

```text
python -m nemotron.steps.byob.scripts.bfcl_author review \
  --workspace /srv/bfcl/authoring/warehouse \
  --adapter-kind local_python \
  <review-packet input and output arguments>
```

Review assembles independently verified certification, fresh validation, the answered questions, and the complete candidate pack into one deterministic packet. `--adapter-kind` defaults to `mcp_mode_a`, so name your own adapter explicitly.

## Step 7: Approve the Release, Then Freeze

```text
python -m nemotron.steps.byob.scripts.bfcl_author approve \
  --workspace /srv/bfcl/authoring/warehouse \
  --boundary release \
  --approved-by reviewer@example.test \
  <review packet, checklist, and output arguments>

python -m nemotron.steps.byob.scripts.bfcl_author freeze \
  --workspace /srv/bfcl/authoring/warehouse \
  <freeze inputs, approval, and output arguments>
```

Release approval binds the exact review packet, so a rebuilt packet needs its new digest approved. `--acknowledge-warning` and `--acknowledge-finding` may each be repeated to record an acknowledged item. Freeze then seals the pack and all reviewed sidecars as immutable bytes; never edit frozen bytes.

## Step 8: Publish

```text
python -m nemotron.steps.byob.scripts.bfcl_author publish \
  --workspace /srv/bfcl/authoring/warehouse \
  --adapter-kind local_python \
  <frozen release, generation config, and output arguments>
```

Publication reruns fresh Gold validation against the frozen pack and then runs the ordinary generation pipeline as `stage=all`; it does not trust the validation evidence recorded during review. Choose the release budget and the mixes with {doc}`publish-a-release`.

## Verify Success

The certification report records the tier you required, `A2` for a Gold release, and the exposure authorization and both approvals reference current digests. Assembly wrote `candidate_pack_provenance.json` recording the digest of every input and every file it produced. Fresh validation of the candidate pack reports `gold_eligible: true`, and publication wrote `run_manifest.json` beside `benchmark.parquet` and `benchmark_raw.parquet`.

## End-to-End Example: From Two Inputs to a Gold Publication

This compact example starts with a tool catalog and domain brief and uses the optional
model-assisted scaffold. Replace the model variables with a route registered according
to {doc}`../reference/data-designer-provider`.

```bash
export SOURCE=/srv/sources/warehouse-package
export BRIEF=/srv/sources/domain-brief.txt
export PROBE_PLAN=/srv/sources/probe-plan.json
export AUTHOR_PROVIDER=nvidia_inference_api
export AUTHOR_MODEL="<served model identifier>"
export AUTHOR_MODEL_CANONICAL="<immutable or reviewed provider-managed identity>"

mkdir -p "$SOURCE"
```

The operator places the two reviewed inputs:

```text
/srv/sources/
├── warehouse-package/
│   └── tools.json
└── domain-brief.txt
```

Generate the source proposal:

```bash
uv run python -m nemotron.steps.byob.scripts.scaffold_source_package \
  --tools "$SOURCE/tools.json" \
  --output "$SOURCE" \
  --dependency-lock \
  --draft-with-model \
  --domain-brief "$BRIEF" \
  --model-alias author \
  --model-provider "$AUTHOR_PROVIDER" \
  --model "$AUTHOR_MODEL" \
  --model-canonical-id "$AUTHOR_MODEL_CANONICAL" \
  --seed 7 \
  --temperature 0 \
  --request-timeout 600
```

The pipeline adds `backend.py`, `fixtures.json`, `dependency-lock.json`, and
`source-draft.json`. The operator completes and reviews the executable behavior and
fixture state, then checks the source:

```bash
uv run python -m nemotron.steps.byob.scripts.check_source_package \
  --source "$SOURCE"
```

Draft a probe-plan proposal, review it, and statically check the result:

```bash
uv run python -m nemotron.steps.byob.scripts.draft_probe_plan \
  --source "$SOURCE" \
  --domain-brief "$BRIEF" \
  --output "$PROBE_PLAN" \
  --clock 2026-03-02T02:00:00Z \
  --model-alias author \
  --model-provider "$AUTHOR_PROVIDER" \
  --model "$AUTHOR_MODEL" \
  --model-canonical-id "$AUTHOR_MODEL_CANONICAL" \
  --seed 7 \
  --temperature 0 \
  --request-timeout 600

uv run python -m nemotron.steps.byob.scripts.check_probe_plan \
  --source "$SOURCE" \
  --probe-plan "$PROBE_PLAN"
```

Continue only when the static check reports `attainable_tier: A2`; intake must still
execute the probes to earn A2. Create the certification key and held-out decision
described above before starting intake.

```text
reviewed source + brief + probe plan + held-out decision + certification key
  → author: A2 certification and sanitized evidence
  → authorize + evidence approval: pre-model boundary
  → draft: bounded model proposals and compiled assertions
  → operator: reviewed-supplement.yaml
  → assemble + candidate validation: Gold candidate
  → review + release approval + freeze: immutable release
  → publish: benchmark.parquet + benchmark_raw.parquet + run_manifest.json
```

The two initial files are enough to enter the scaffold lane, not enough to bypass
source, probe, supplement, or release review.

## Common Failures

Model-drafted arguments must remain grounded in the certified schema and fixture
evidence. Bind unconstrained strings such as identifiers and dates to reviewed
fixtures; reserve `literal` for schema-pinned enums and booleans. Encode boolean and
numeric literal fields as strings such as `"true"` and `"2"`, and never add an
argument absent from the tool's parameter schema.

The assertion compiler prefixes exported callables with `assert_`. For example,
assertion id `tool_called_lookup` compiles as `assert_tool_called_lookup`. If a
supplement uses the unprefixed id, correct it and assemble a new candidate path rather
than editing the generated candidate.

| Reported code | What to do |
| --- | --- |
| `adapter_rollout_disabled` | Enable the reviewed adapter policy or its environment variable. Do not bypass the gate downstream. |
| `adapter_under_certified` | Collect the missing A2 observations. Approval cannot raise certification, and the probe plan needs its timeout case. |
| `source_identity_mismatch` | The source tree changed after certification. Assemble the exact revision intake certified, or rerun intake. |
| `supplement_assertion_unknown` | The supplement names an assertion drafting never compiled. Draft and compile that specification first. |
| `review_approval_stale` | Rebuild the review packet and approve its new digest. |
| `session_binding_drift` | Restore the bound artifact, or resume with `bfcl_author resume --workspace <DIR> --next <STEP>`. |

## Next Steps

- Onboard a running MCP server instead of a package: {doc}`mcp-server`, or take the frozen pack to publication scale with {doc}`publish-a-release`.
- Read where the authorization boundaries sit and why: {doc}`../explanation/authoring-flows`.
