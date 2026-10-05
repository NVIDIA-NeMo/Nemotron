# BFCL assisted-authoring user guide

This guide covers assisted authoring: producing a reviewed Oracle Pack from one source declaration
and one domain brief. The manual Oracle Pack path remains unchanged and is documented in
[bfcl-oracle-pack.md](bfcl-oracle-pack.md). Current adapter and publication status is
test-linked in [bfcl-authoring-support-matrix.md](bfcl-authoring-support-matrix.md). To
watch the whole flow run once before reading it step by step, see
[bfcl-assisted-authoring-walkthrough.md](bfcl-assisted-authoring-walkthrough.md).

This file is the command-level reference: every invocation below is a smoke case the test
suite executes, so it stays in step with the CLI. For the narrative walkthrough of the same
flow, read
[the assisted-authoring task guide](../../../../../docs/build-benchmarks/function-calling/how-to/assisted-authoring.md)
first and return here for exact arguments and refusal codes.

## Install and inspect the CLI

Install the BYOB dependencies. MCP transport users also install the isolated `bfcl-mcp`
extra. The guided command is intentionally stateful and prints the next safe command when
a gate refuses progress.

<!-- doc-smoke: bfcl-author-help -->
```shell
python -m nemotron.steps.byob.scripts.bfcl_author --help
```

## Inputs and adapter selection

`author` accepts `--source`, `--brief`, and a workspace. `--adapter auto` recognizes a
local Python source package, reviewed HTTP package, or MCP Mode A intake. Pack ID and
version come from policy or explicit confirmation; CI never guesses or prompts.

Held-out status and the probe plan must be settled at this first command rather than
deferred. A reviewed policy may supply `held_out`; otherwise pass either
`--held-out-policy` or `--held-out-not-applicable-reason` with
`--held-out-reviewed-by`. A source reaching A1 or A2 also needs `--probe-plan`, since
those tiers are earned from observed probe outcomes and nothing else can supply them.

The probe plan is one document for every transport. It names a case per published tool,
at least one structured error if the source has error codes, and a case the tool cannot
finish inside the deadline; the same plan then drives a local package through a child
process, an HTTP endpoint through one session per episode, or an MCP Mode A gateway
through one gateway session per episode. Any session-based plan — HTTP or MCP — must
carry `fixtures`, because a session is handed its world at open rather than reading it
from a reviewed file. MCP plans are supplied to `build_mcp_intake.py --probe-plan` and
are accepted for Mode A only, since Mode A is the only mode whose reset and state are
control tools; without a plan MCP intake certifies A0, exactly as the other transports
do.

[`bfcl-probe-plan.example.json`](bfcl-probe-plan.example.json) is a complete plan of that
shape from a published release, with a success per published tool, a state-changing case
for each mutating tool, structured errors naming their codes, and the timeout case A2
requires; its `fixtures` block is abridged to the records its own cases reach.
[`bfcl-domain-brief.example.txt`](bfcl-domain-brief.example.txt) is the matching `--brief`.
Organizational defaults that policy should supply rather than each invocation are shown in
[`bfcl-authoring-policy.example.yaml`](bfcl-authoring-policy.example.yaml).
Set both `pre_model.exposure_authorization` and
`pre_model.clean_evidence_approval` to `organizational_policy` to use the streamlined
path. The policy is reviewed once; each intake binds its canonical digest.

<!-- doc-smoke: bfcl-author-author-help -->
```shell
python -m nemotron.steps.byob.scripts.bfcl_author author --help
```

Source layouts are defined in
[bfcl-conventional-source-packages.md](bfcl-conventional-source-packages.md) and
[bfcl-mcp-user-guide.md](bfcl-mcp-user-guide.md). Live adapters are fail-closed behind
the policy described in [bfcl-authoring-rollout.md](bfcl-authoring-rollout.md)
([`test_bfcl_authoring_rollout_policy.py`](../../../../../tests/steps/byob/test_bfcl_authoring_rollout_policy.py)).

## One Per-Run Approval

For clean native-v2 evidence, the normal command sequence is:

1. `author` creates transport-neutral evidence.
2. `answer` applies any digest-bound open questions.
3. `apply-policy` produces separate exposure and evidence records from the reviewed
   organizational policy.
4. `draft` runs bounded, cached structured model calls.
5. `assemble` binds those drafts into a loadable pack.
6. `review` assembles independently verified certification, fresh validation, answered
   questions, and the complete candidate pack.
7. `release` asks for the one per-run semantic/risk confirmation, records the
   digest-bound approval, seals the reviewed bytes, and reruns fresh Gold validation
   plus `stage=all`. Approval, freeze, and publication remain separate audit phases.

`apply-policy` is fail-closed: it rejects a changed policy, unresolved gaps, migrated
evidence, and domain-brief advisory findings. Those cases use the granular `authorize`
and `approve --boundary evidence` commands after human review. This fallback requires
two decisions at different times because pre-model authorization cannot be replaced
retroactively by final release approval.

<!-- doc-smoke: bfcl-author-apply-policy-help -->
```shell
python -m nemotron.steps.byob.scripts.bfcl_author apply-policy --help
```

Session ordering and stale-binding refusal are exercised by
[`test_bfcl_authoring_cli.py`](../../../../../tests/steps/byob/test_bfcl_authoring_cli.py)
and
[`test_bfcl_authoring_e2e.py`](../../../../../tests/steps/byob/test_bfcl_authoring_e2e.py).

Use each subcommand’s help as the authoritative argument synopsis:

<!-- doc-smoke: bfcl-author-review-help -->
```shell
python -m nemotron.steps.byob.scripts.bfcl_author review --help
```

<!-- doc-smoke: bfcl-author-publish-help -->
```shell
python -m nemotron.steps.byob.scripts.bfcl_author publish --help
```

<!-- doc-smoke: bfcl-author-release-help -->
```shell
python -m nemotron.steps.byob.scripts.bfcl_author release --help
```

## Assembling the candidate pack

Drafting stops at proposals, and it writes them beside a pack rather than into one. The
assembler turns them into a loadable pack and derives everything mechanical from evidence
that is already trusted: pack identity and the manifest from the verified bundle,
`tools.json` from the catalog certification observed, and `assertions.py` from drafts that
compiled without a blocker.

Where the oracle itself comes from depends on what kind of source was certified. A
`local_python` source is a tree, so `backend.py` and `fixtures.json` are copied into the
pack byte for byte from the tree certification fingerprinted. An `http_package` or MCP
source is a session, and there is nothing to copy: the pack names the certified endpoint,
pinned to the identity and TLS bundle intake verified, and takes its fixtures from the
reviewed probe plan those sessions were opened with. Either way the fixtures a pack claims
must be the fixtures the source was certified against, so an endpoint that pins another
oracle, a catalog that drifted, or fixtures the plan does not name are all refused
([`test_bfcl_mcp_pack_assembly.py`](../../../../../tests/steps/byob/test_bfcl_mcp_pack_assembly.py)).

What remains is what a model that has only read a catalog must not state: slots bound to
fixture columns, turn policies, per-language user turns, and the validation cases that
decide the tier. Those arrive in one reviewed `bfcl-candidate-pack-supplement-v1` YAML
file, and every tool and assertion it names is checked back against the evidence and the
compiled assertions, so a supplement cannot introduce a tool the source never published
or an assertion nobody compiled. Assembly writes `candidate_pack_provenance.json`
recording the digest of every input and every file it produced.

Inside a guided session, assembly is a step of the flow rather than a separate tool:
`bfcl_author assemble` reads the drafts and evidence the session already bound, binds the
pack it writes back into the session, and leaves `review` as the next command. The
standalone assembler remains available for a pack assembled outside a session, where the
evidence, drafts, and probe plan are passed explicitly.

Candidate validation may expose a defect in the reviewed supplement, such as incomplete
positive/negative coverage. While the session is still at `pack_assembled`, correct the
supplement and run `bfcl_author assemble` again with a new `--output` path. The previous
candidate remains immutable in the session history and the new candidate becomes the one
`review` binds. Re-run validation against that new pack before continuing.

<!-- doc-smoke: guide-author-assemble-help -->
```shell
python -m nemotron.steps.byob.scripts.bfcl_author assemble --help
```

<!-- doc-smoke: guide-assemble-help -->
```shell
python -m nemotron.steps.byob.scripts.assemble_candidate_pack --help
```

One test drives this whole path — source declaration and domain brief through A2 intake,
both authorization boundaries, drafting, assembly, and unmocked validation — and fails
unless the result is Gold-eligible
([`test_bfcl_authoring_gold_e2e.py`](../../../../../tests/steps/byob/test_bfcl_authoring_gold_e2e.py)).
Each binding the assembler refuses is owned by
[`test_bfcl_authoring_pack_assembly.py`](../../../../../tests/steps/byob/test_bfcl_authoring_pack_assembly.py).

## Certification and publication

A0 proves identity and catalog integrity, A1 adds bounded read-only observation, and A2
adds deterministic reset, isolation, confirmation safety, mutation declaration, timeout
cleanup, and result coverage. Drafting may inspect lower tiers, but Gold freeze requires
A2 ([`test_bfcl_authoring_release.py`](../../../../../tests/steps/byob/test_bfcl_authoring_release.py)).

Local Python and MCP Mode A have publication adapters. HTTP packages share intake,
review, and freeze but publication is intentionally refused until an independently
verified publication adapter exists
([`test_bfcl_authoring_e2e.py`](../../../../../tests/steps/byob/test_bfcl_authoring_e2e.py)).

The release contract is
[bfcl-authoring-release-v2.md](bfcl-authoring-release-v2.md). Legacy MCP v1 releases
remain verifiable but are not byte-identical to v2.

## Operations

- Structured events: [bfcl-authoring-events.md](bfcl-authoring-events.md), verified by
  [`test_bfcl_authoring_events.py`](../../../../../tests/steps/byob/test_bfcl_authoring_events.py).
- Credential references and rotation:
  [bfcl-authoring-credentials.md](bfcl-authoring-credentials.md), verified by
  [`test_bfcl_authoring_credentials.py`](../../../../../tests/steps/byob/test_bfcl_authoring_credentials.py).
- Model-cache retention:
  [bfcl-authoring-cache-retention.md](bfcl-authoring-cache-retention.md), verified by
  [`test_bfcl_authoring_retention.py`](../../../../../tests/steps/byob/test_bfcl_authoring_retention.py).
- Release revocation:
  [bfcl-authoring-revocation.md](bfcl-authoring-revocation.md), verified by
  [`test_bfcl_release_revocation.py`](../../../../../tests/steps/byob/test_bfcl_release_revocation.py).

<!-- doc-smoke: guide-purge-help -->
```shell
python -m nemotron.steps.byob.scripts.bfcl_author purge-cache --help
```

<!-- doc-smoke: guide-revoke-help -->
```shell
python -m nemotron.steps.byob.scripts.revoke_authoring_release --help
```

## Troubleshooting

- `adapter_rollout_disabled`: explicitly enable the reviewed adapter policy; do not
  bypass it in a downstream command.
- `credential_context_stale`: rerun intake and certification after principal,
  permission, or credential-reference drift.
- `session_binding_drift`: restore the bound artifact or resume from the last verified
  session.
- `adapter_under_certified`: collect the missing A2 observations; approval cannot raise
  certification. Every source needs `--probe-plan`, including a case the tool cannot
  finish in time, or timeout cleanup stays unobserved; for MCP the plan is passed to
  `build_mcp_intake.py`, and it is accepted for Mode A only.
- `source_identity_mismatch`: the source tree changed after certification; assemble the
  exact revision intake certified, or rerun intake.
- `supplement_assertion_unknown`: the supplement names an assertion drafting never
  compiled; draft and compile that specification first.
- `review_approval_stale`: rebuild review and approve its new digest.
- `release_revoked`: consult the authenticated registry and replacement fingerprint;
  never edit frozen bytes.

## Contract index

- Adapter declaration and evidence:
  [bfcl-transport-neutral-intake.md](bfcl-transport-neutral-intake.md)
- Certification profiles:
  [bfcl-source-adapter-certification-profiles.md](bfcl-source-adapter-certification-profiles.md)
- Adding an adapter:
  [bfcl-adapter-authoring-certification-guide.md](bfcl-adapter-authoring-certification-guide.md)
- MCP transport:
  [bfcl-mcp-oracle-contract.md](bfcl-mcp-oracle-contract.md)
- Release:
  [bfcl-authoring-release-v2.md](bfcl-authoring-release-v2.md)
- Workflow acceptance criteria and their owning tests:
  [bfcl-workflow-acceptance-matrix.md](bfcl-workflow-acceptance-matrix.md)
