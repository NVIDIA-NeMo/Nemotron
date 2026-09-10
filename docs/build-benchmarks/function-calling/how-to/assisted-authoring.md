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
   and `dependency-lock.json`; the operator owns the resulting behavior and removes
   no review marker without implementing its decision.
2. **Prepare probes.** The operator reviews `probe-plan.json`.
   `check_probe_plan` writes a readiness verdict; it does not certify A2.
3. **Run intake.** `author` reads the reviewed source, brief, probe plan, policy, and
   certification key. It writes fingerprinted source observations, certification,
   sanitized evidence, and model-exposure subjects. Continue only when the measured
   tier is A2 for a Gold release.
4. **Cross the pre-model boundary.** `apply-policy` normally reads the policy bound
   at intake and writes separate exposure-authorization and evidence-approval
   records. Exceptional evidence uses explicit human `authorize` and
   `approve --boundary evidence` commands.
5. **Draft proposals.** `draft` reads only approved, sanitized evidence plus pinned
   model configuration. It writes coverage, validation-case, task-template, and
   assertion proposals with model-call provenance. A human reviewer checks grounding,
   coverage, and compilation; the model cannot edit the certified source.
6. **Assemble and validate.** The operator supplies
   `reviewed-supplement.yaml`. `assemble` writes a candidate Oracle Pack and
   provenance; candidate validation writes a Gold or non-Gold verdict. Correct the
   supplement and assemble to a new output rather than editing the candidate.
7. **Review and release.** `review` binds the exact candidate, current certification,
   and fresh validation into a packet. After human review, `release` records approval,
   freezes the exact bytes, reruns Gold validation, and publishes
   `benchmark.parquet`, `benchmark_raw.parquet`, and `run_manifest.json`.

Generated evidence, provenance, reports, frozen bytes, Parquet tables, and manifests
are pipeline-owned. Correct the upstream operator-owned source, plan, supplement, or
configuration and rerun the named gate; do not patch generated outputs.

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

For the normal one-approval release flow, bind adapter rollout and the two pre-model
decisions to one reviewed project policy:

```yaml
schema_version: bfcl-authoring-policy-v1
pack_id: warehouse_assets
pack_version: 0.1.0
required_certification_tier: A2
adapter_rollout:
  local_python: true
pre_model:
  exposure_authorization: organizational_policy
  clean_evidence_approval: organizational_policy
held_out:
  not_applicable_reason: The asset catalog is public reference data.
  reviewed_by: policy-owner@example.test
release:
  signing_key_env: BFCL_RELEASE_SIGNING_KEY
  signing_key_id: warehouse-release
  seal_issuer: benchmark-release-team
  seal_public_key: keys/warehouse-release-public.pem
```

Pass that file as `--policy /srv/bfcl/policies/warehouse-authoring.yaml`. Its canonical
digest is bound to the exposure authorization and clean-evidence approval generated
later. Review the policy when the allowed model, data boundary, or source scope changes,
not once per unchanged authoring run. `release` resolves the public-key path relative
to this policy file and reads only the private-key file path from
`BFCL_RELEASE_SIGNING_KEY`; a secret manager should populate that environment variable.

For an isolated manual run without that policy, enable only the adapter:

```bash
export BFCL_ENABLE_LOCAL_PYTHON=1
```

The equivalent variable for the HTTP adapter is `BFCL_ENABLE_HTTP_PACKAGE`. Without the variable, or an equivalent reviewed `adapter_rollout` policy block, intake fails with `adapter_rollout_disabled`. Accepted values are `1`, `true`, `yes`, `0`, `false`, and `no`, case-insensitive. Offline artifact verification, review, approval, and freeze do not require the flag; only operations that inspect or execute a source do.

:::{important}
Publication currently supports `local_python`. An `http_package` source reaches intake, drafting, review, and freeze, but publication is deliberately refused until an independently verified publication adapter exists. Plan for that limitation before you onboard an HTTP source, because the refusal arrives at the last step.
:::

## Step 1: Intake and Certification

`author` runs source intake and produces transport-neutral evidence. Held-out status and
the probe plan must be settled at this first command rather than deferred. The reviewed
project policy may supply held-out status, so the operator does not restate it per run.

**Operator supplies:** reviewed source, domain brief, probe plan, policy, pack identity,
and certification key. **Pipeline writes:** the intake directory containing source
observations, certification, sanitized evidence, redaction reports, and the
model-exposure subject. **Human check:** confirm the measured tier is A2 and resolve
every open question before model exposure.

```bash
python -m nemotron.steps.byob.scripts.bfcl_author \
  --ci author \
  --workspace /srv/bfcl/authoring/warehouse \
  --source /srv/sources/warehouse-package \
  --brief /srv/sources/domain-brief.txt \
  --adapter local_python \
  --policy /srv/bfcl/policies/warehouse-authoring.yaml \
  --pack-id warehouse_assets \
  --pack-version 0.1.0 \
  --required-tier A2 \
  --certification-private-key /srv/bfcl/keys/certification-private.pem \
  --certification-key-id warehouse-authoring \
  --probe-plan /srv/sources/probe-plan.json
```

Without policy defaults, supply either `--held-out-policy` or
`--held-out-not-applicable-reason`, always together with
`--held-out-reviewed-by`. A1 and A2 are earned from observed probe outcomes, so the
probe plan still belongs to intake. `--adapter auto` is the default and recognizes any
supported layout; `--ci` never prompts; and adapter-specific flags after the guided
ones are delegated to the underlying intake command.

The probe plan is one transport-neutral document. It names a case per published tool, at least one structured error if the source has error codes, and a case the tool cannot finish inside the deadline; without that last case, timeout cleanup stays unobserved and certification cannot reach A2. Certification then awards one of three tiers. A0 proves source identity and catalog integrity. A1 adds bounded read-only observation. A2 adds deterministic reset, episode isolation, confirmation safety, mutation declaration, timeout cleanup, and result coverage. Stop if the report does not record the tier you need: drafting may inspect lower tiers, but a Gold freeze requires A2, and no approval can raise a certification tier.

## Step 2: Answer the Open Questions

Intake may raise digest-bound open questions about the source. Apply the reviewed answers as a new evidence revision:

**Operator supplies:** reviewed answers to the exact question artifact. **Pipeline
writes:** a child evidence revision with a new digest. **Human check:** confirm that
the answers describe existing source truth rather than inventing behavior to satisfy
the benchmark.

```bash
python -m nemotron.steps.byob.scripts.bfcl_author answer \
  --workspace /srv/bfcl/authoring/warehouse \
  --evidence <EVIDENCE_BUNDLE_JSON> \
  --questions <OPEN_QUESTIONS_JSON> \
  --answers <REVIEWED_ANSWERS_JSON>
```

## Step 3: Apply Pre-Model Policy

For clean native-v2 evidence, apply the reviewed policy once:

**Operator supplies:** the policy already bound during intake. **Pipeline writes:**
separate exposure-authorization and clean-evidence-approval records. **Human check:**
use the policy path only for clean evidence within its reviewed scope; exceptional
findings require the explicit fallback below.

```bash
python -m nemotron.steps.byob.scripts.bfcl_author apply-policy \
  --workspace /srv/bfcl/authoring/warehouse
```

`apply-policy` verifies that the policy digest is the one bound during intake, refuses
unresolved gaps, migrations, and domain-brief advisory findings, and writes both
digest-bound pre-model records. It does not ask the operator for a per-run approval.
Source certification remains automatic, and the finished review packet still requires
one human release approval in Step 7.

:::{important}
Evidence approval and release approval are different decisions, and the first cannot
be replaced by the second. Evidence approval records that the exact normalized
evidence is fit for drafting, either under the reviewed clean-evidence policy or after
explicit human review. Release approval records that a reviewer inspected the finished
pack and fresh validation and considers the pack fit to publish. Approving the release
does not retroactively authorize model exposure that already happened, so the command
sequence requires both records in order.

The streamlined flow preserves those separate records but lets a reusable organizational
policy produce the pre-model records. A reviewer approves only the final release.
:::

If intake reports open questions, migrated evidence, or advisory findings, policy cannot
silently accept them. Use the granular `authorize` and `approve --boundary evidence`
commands after human review. If the source, brief, redaction, observations,
certification, or resolved configuration changes, start a new intake so the policy is
bound to the new digests.

The manual fallback remains:

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
evidence-bundle artifact; do not substitute a filesystem checksum.

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

**Operator supplies:** approved evidence and a pinned Data Designer model route.
**Pipeline writes:** coverage, validation-case, task-template, and assertion drafts,
compiled assertions, model I/O cache, and provenance. **Human check:** inspect
grounding, direct tool coverage, compilation refusals, and unresolved blockers.

```text
python -m nemotron.steps.byob.scripts.bfcl_author draft \
  --workspace /srv/bfcl/authoring/warehouse \
  <artifact, certification, approval, key, output, and model arguments>
```

Drafting issues bounded, cached, structured model requests and stops at proposals: it writes them beside a pack rather than into one. Unknown tools, unsupported assertions, ungrounded arguments, malformed output, and cache conflicts all fail closed.

## Step 5: Assemble the Candidate Pack

**Operator supplies:** the reviewed semantic supplement and a fresh output path; the
session supplies certified evidence, source, and compiled drafts. **Pipeline writes:**
the candidate pack and `candidate_pack_provenance.json`. **Human check:** validate the
candidate at Gold and reassemble to a new path after any supplement correction.

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

**Operator supplies:** the exact candidate, current certification, fresh Gold
validation, and their provenance records. **Pipeline writes:** a deterministic review
packet and freeze inputs. **Human check:** inspect domain semantics, validation
evidence, assumptions, and every finding before release approval.

```text
python -m nemotron.steps.byob.scripts.bfcl_author review \
  --workspace /srv/bfcl/authoring/warehouse \
  --adapter-kind local_python \
  <review-packet input and output arguments>
```

Review assembles independently verified certification, fresh validation, the answered questions, and the complete candidate pack into one deterministic packet. `--adapter-kind` defaults to `mcp_mode_a`, so name your own adapter explicitly.

## Step 7: Review Once, Then Release

**Operator supplies:** a reviewer identity, the freeze handoff, signing identity, and
reviewed publication configuration. **Pipeline writes:** a digest-bound approval, an
immutable frozen release, fresh validation evidence, `benchmark_raw.parquet`,
`benchmark.parquet`, and `run_manifest.json`. **Human check:** inspect the bound review
packet and confirm its semantics, descriptions, assumptions, held-out treatment, and
reported risks once.

```text
python -m nemotron.steps.byob.scripts.bfcl_author release \
  --workspace /srv/bfcl/authoring/warehouse \
  --approved-by reviewer@example.test \
  --freeze-inputs /srv/bfcl/authoring/warehouse/freeze_inputs.json \
  --signing-key <RELEASE_PRIVATE_KEY> \
  --signing-key-id <RELEASE_KEY_ID> \
  --seal-issuer <RELEASE_ISSUER> \
  --seal-public-key <RELEASE_PUBLIC_KEY> \
  --config <PUBLICATION_CONFIG>
```

The command loads the exact review packet from the verified session. The approval
defaults to `<workspace>/release_approval.json`, the frozen release to
`<workspace>/release`, and omitted `--freeze-inputs` and `--config` paths to
`<workspace>/freeze_inputs.json` and `<workspace>/publication.yaml`. It automatically
checks the packet digest, blockers, Gold validation evidence, independently verified
A2 certification, pre-model authorization, question state, and the exact risk set, and
refuses before asking the reviewer anything if one of them does not hold.

What remains is the judgment a machine cannot make: domain semantics, descriptions and
snapshots, stated assumptions, held-out treatment, and the reported risks. Interactive
use prints one summary of those and asks for one confirmation.
`--confirm-reviewed-content` records that same decision without a prompt and is
required under `--ci`, so pass it only after the review has happened. Because `--ci`
is a dispatcher option, place it before the subcommand:
`bfcl_author --ci release ... --confirm-reviewed-content`.

Prepare `publication.yaml` with {doc}`publish-a-release`; the guided `release` command
runs that reviewed configuration after it freezes the approved pack.

One seal identity covers both steps: `--signing-key-id` seals the frozen release and
verifies it again at publication, so the granular flow's separate `--seal-key-id` is
not repeated here.

Projects may also put `signing_key_id`, `seal_issuer`, `seal_public_key`, and a
`signing_key_env` reference under `release` in the reviewed authoring policy. The
private key itself must remain in a secret store: mount it as a file and set the named
environment variable to that path. When those defaults are present, omit the four
signing flags above. Explicit flags remain available for exceptional runs.

After confirmation, `release` records the approval time in UTC, acknowledges the
packet's current risk IDs, freezes the approved bytes, and publishes them. Each
intermediate phase remains separately committed, so a freeze or publication failure
can resume from the last successful phase without asking for the same approval again.
Publication reruns fresh Gold validation and does not trust the validation evidence
recorded during review.

The streamlined command writes a v3 approval whose `checklist_sources` maps each item
to `machine` or `human`; the signed digest therefore preserves how every decision was
made. The granular approval command continues to write and freeze compatible v2
records, so existing releases do not need migration.

Approval, freeze, and publication use separate short workspace leases: each next phase
re-verifies the committed bindings, and a long publication run does not turn the whole
sequence into one unrecoverable transaction. Delegated progress is streamed to stderr
while stdout remains one machine-readable combined JSON verdict.

The granular `approve --boundary release`, `freeze`, and `publish` commands remain
available for custom signing, revocation, or operational workflows. A rebuilt review
packet always has a new digest and therefore requires a new approval.

## Verify Success

The certification report records the tier you required, `A2` for a Gold release.
Exposure authorization and evidence approval records, whether policy-generated or
manual, reference current evidence digests; release approval binds the current review
packet digest. Assembly wrote `candidate_pack_provenance.json` recording the digest of
every input and every file it produced. Fresh validation of the candidate pack reports
`gold_eligible: true`, and publication wrote `run_manifest.json` beside
`benchmark.parquet` and `benchmark_raw.parquet`.

## End-to-End Example: From Two Inputs to a Gold Publication

This example starts with only a tool catalog and domain brief, uses the optional
model-assisted source scaffold, and shows the handoff into the guided flow. Replace
the model variables with an identity registered according to
{doc}`../reference/data-designer-provider`.

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
`source-draft.json`. The operator reviews and completes `backend.py` and
`fixtures.json`, then checks the source:

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

If model drafting cannot ground an argument, the operator corrects its binding from
reviewed fixture state and reruns `check_probe_plan`. Continue only when it reports
`attainable_tier: A2`; intake still has to execute the probes to earn A2.

Create the certification key and reviewed authoring policy described above before
starting intake. Prepare the publication configuration before release.

The remaining handoff is:

```text
operator-reviewed source + brief + probe plan + policy + certification key
  → author: A2 certification and sanitized evidence
  → apply-policy: exposure authorization and evidence approval
  → draft: bounded model proposals and compiled assertions
  → operator: reviewed-supplement.yaml
  → assemble: candidate pack and provenance
  → candidate validation: Gold verdict
  → review: review packet and freeze inputs
  → release: human approval + immutable freeze + Gold publication
  → output: benchmark.parquet + benchmark_raw.parquet + run_manifest.json
```

The procedure above gives the command and review contract for each handoff. The two
initial files are enough to enter the scaffold lane, not enough to bypass source,
probe, supplement, or release review.

## Common Failures

Model-drafted arguments must remain grounded in the certified schema and fixture
evidence. A string parameter without an enum cannot use an arbitrary model-selected
literal merely because the value looks plausible. Bind it to a reviewed fixture;
reserve `literal` for schema-pinned enums and booleans, and encode the draft schema's
boolean and numeric literal fields as strings such as `"true"` and `"2"`. Never add
an argument absent from the tool's parameter schema. If drafting is refused, correct
the operator-owned input or drafting instruction and retry the command with a fresh
output path; no failed draft becomes approved evidence.

The assertion compiler exports callable names with an `assert_` prefix. A supplement
must reference the compiled callable, not only the model's assertion id. For example,
an assertion id `tool_called_lookup` compiles as `assert_tool_called_lookup`. If the
supplement names the wrong form, correct it and assemble to a new candidate path:

```bash
python -m nemotron.steps.byob.scripts.bfcl_author assemble \
  --workspace /srv/bfcl/authoring/warehouse \
  --supplement /srv/bfcl/authoring/warehouse/reviewed-supplement.yaml \
  --output /srv/bfcl/authoring/warehouse/candidate-pack-v2
```

| Reported code | What to do |
| --- | --- |
| `adapter_rollout_disabled` | Enable the reviewed adapter policy or its environment variable. Do not bypass the gate downstream. |
| `adapter_under_certified` | Collect the missing A2 observations. Approval cannot raise certification, and the probe plan needs its timeout case. |
| `source_identity_mismatch` | The source tree changed after certification. Assemble the exact revision intake certified, or rerun intake. |
| `supplement_assertion_unknown` | The supplement names an assertion drafting never compiled. Draft and compile that specification first. |
| `review_approval_stale` | Rebuild the review packet and approve its new digest. |
| `session_binding_drift` | Restore the bound artifact, or resume with `bfcl_author resume --workspace <DIR> --next <STEP>`. |

## Next Steps

- Onboard a running MCP server instead of a package with {doc}`mcp-server`.
- To republish an existing Gold pack or tune publication budgets outside a guided
  session, use {doc}`publish-a-release`.
- Read where the authorization boundaries sit and why: {doc}`../explanation/authoring-flows`.
