# Authoring and certifying a BFCL source adapter

This is the implementation checklist for a new built-in assisted-authoring source adapter.
Runtime installation of third-party adapters is **unimplemented**; the registry is a
static allowlist
([`test_bfcl_source_adapter_registry.py`](../../../../../tests/steps/byob/test_bfcl_source_adapter_registry.py)).

## 1. Define declarations, not trust

Implement `OracleSourceAdapter` from `runtime/source_adapters/contract.py`. Its
`AdapterDescriptor` declares capabilities, fixture access, probe safety, and cleanup,
but never an attained tier. Unknown fields and self-certification are rejected by
[`test_bfcl_source_adapter_contract.py`](../../../../../tests/steps/byob/test_bfcl_source_adapter_contract.py).

Add a versioned `SourceDeclaration` kind and a side-effect-free resolver in
`runtime/source_adapters/registry.py`. Detection markers must be unique; importing the
registry must not import adapter implementations
([`test_bfcl_source_adapter_registry.py`](../../../../../tests/steps/byob/test_bfcl_source_adapter_registry.py)).

## 2. Publish the certification profile first

Add an inert profile to `PUBLISHED_CERTIFICATION_PROFILES` in
`runtime/source_adapters/certification.py`. Specify applicable and required probes for
A0, A1, and A2 before writing the collector. Add every stable refusal code to
[bfcl-source-adapter-certification-profiles.md](bfcl-source-adapter-certification-profiles.md).

Profile immutability, generic-tier derivation, signatures, and refusal-code/document
parity are enforced by
[`test_bfcl_source_adapter_certification.py`](../../../../../tests/steps/byob/test_bfcl_source_adapter_certification.py)
and
[`test_bfcl_source_intake.py`](../../../../../tests/steps/byob/test_bfcl_source_intake.py).

## 3. Collect observations without issuing certificates

The adapter returns identity, reviewed tools, and `AdapterProbeObservation` records.
It must not receive the certification private key or construct
`AdapterCertificationReport`. BFCL’s intake orchestrator projects observations against
the published profile and signs the result.

Use the local adapter tests for process isolation and least privilege
([`test_bfcl_local_authoring_adapter.py`](../../../../../tests/steps/byob/test_bfcl_local_authoring_adapter.py)),
the HTTP tests for reviewed-schema/live-identity parity
([`test_bfcl_http_authoring_adapter.py`](../../../../../tests/steps/byob/test_bfcl_http_authoring_adapter.py)),
and the MCP tests for paginated discovery and control separation
([`test_bfcl_mcp_discovery.py`](../../../../../tests/steps/byob/test_bfcl_mcp_discovery.py)).

Credentials must use `CredentialReference`; resolved values remain in memory and the
source identity binds the non-secret authorization context
([`test_bfcl_authoring_credentials.py`](../../../../../tests/steps/byob/test_bfcl_authoring_credentials.py)).

## 4. Join the transport-neutral intake

Wire collection through `finalize_source_intake` or `run_conventional_intake` in
`runtime/source_adapters/intake.py`. The output must be
`bfcl-source-evidence-v2`, bind the domain brief, held-out decision, redaction proof,
resolved configuration, independent certificate, and exposure subject.

Add the adapter to the parameterized A0/A1/A2/refusal matrix in
[`test_bfcl_source_intake.py`](../../../../../tests/steps/byob/test_bfcl_source_intake.py).
If migration from a persisted format is required, add an explicit normalized record and
coverage in
[`test_bfcl_source_evidence_migration.py`](../../../../../tests/steps/byob/test_bfcl_source_evidence_migration.py).

## 5. Add release hooks deliberately

Implement `ReleaseAdapter` for review and freeze contributions. Implement
`PublicationAdapter` only if the adapter can bind a config, rerun authoritative BFCL
prepare, require fresh Gold, generate `stage=all`, and verify publication lineage.
Review/freeze support does not imply publication support.

Hook contracts and v1 compatibility are covered by
[`test_bfcl_authoring_release.py`](../../../../../tests/steps/byob/test_bfcl_authoring_release.py).
Every adapter must join the shared-envelope test; publishable adapters must also join
the real handoff test in
[`test_bfcl_authoring_e2e.py`](../../../../../tests/steps/byob/test_bfcl_authoring_e2e.py).

## 6. Add rollout, UX, and operations evidence

Add the kind to the closed rollout-policy vocabulary and fail closed when omitted,
false, malformed, or conflicting
([`test_bfcl_authoring_rollout_policy.py`](../../../../../tests/steps/byob/test_bfcl_authoring_rollout_policy.py)).
Update guided detection and state-specific recovery in `scripts/bfcl_author.py`
([`test_bfcl_authoring_cli.py`](../../../../../tests/steps/byob/test_bfcl_authoring_cli.py)).

Before claiming support, add:

1. contract, registry, profile, and refusal tests;
2. transport-specific identity and probe tests;
3. the shared intake/certification matrix row;
4. review/freeze and, if applicable, publication tests;
5. a row in [bfcl-authoring-support-matrix.md](bfcl-authoring-support-matrix.md);
6. any new CLI example to
   [`test_bfcl_authoring_documentation.py`](../../../../../tests/steps/byob/test_bfcl_authoring_documentation.py).

Until all six exist, label the adapter **unimplemented** rather than experimental or
supported.
