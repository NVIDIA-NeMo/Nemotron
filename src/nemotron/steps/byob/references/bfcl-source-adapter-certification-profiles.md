# BFCL source-adapter certification profiles

This document describes `bfcl-adapter-certification-profile-v1`, the BFCL-owned
contract used by assisted authoring. Adapter declarations and raw observations are
inputs to certification; they are not certification authority.

## Authority boundary

1. The adapter returns `AdapterProbeObservation` values. That schema has no issuer,
   input digest, tier, signature, or Gold field.
2. BFCL resolves one profile from the static
   `PUBLISHED_CERTIFICATION_PROFILES` registry.
3. BFCL chooses the execution policy, records calls, elapsed time, and cleanup in a
   `ProbeExecutionRecord`, derives applicability, computes the exact
   descriptor/identity/profile input digest, and projects executions into
   `ProbeOutcome` values.
4. BFCL derives the attained tier, applies the descriptor capability ceiling, and
   signs `bfcl-adapter-certification-report-v1`.
5. A consumer independently rebinds the report to the descriptor, identity, profile,
   trusted signing key, and required tier.

An adapter cannot add a profile, expand a profile's allowed reasons, or return a
certification report through the adapter contract.

## Common probe ladder

- `A0 / identity_integrity`: identity-only; detects `identity_drift`.
- `A0 / catalog_integrity`: identity-only; detects `catalog_mismatch` and
  `reviewed_schema_missing`.
- `A1 / executable_observation`: read-only; detects unsafe execution, timeout, and
  unknown commit state.
- `A1 / structured_error_shape`: read-only and conditional only when BFCL proves
  there is no reviewed structured-error case.
- `A2 / reset_determinism`: reset-isolated; detects `reset_nondeterministic`.
- `A2 / episode_isolation`: reset-isolated; detects `episode_state_leakage`.
- `A2 / confirmation_safety`: reset-isolated and conditional only when BFCL proves
  there are no confirmation-gated tools.
- `A2 / timeout_cleanup`: reset-isolated; detects timeout, cleanup failure, and
  unknown commit state.
- `A2 / mutation_declaration`: reset-isolated; detects mutation declaration drift.
- `A2 / result_shape_coverage`: reset-isolated; detects incomplete observed result
  coverage.

Every probe profile carries the BFCL executor and evidence issuer, input-binding and
outcome schema versions, maximum calls, timeout, cleanup kind, cleanup timeout, and
the closed failure-reason set. Passing and not-applicable outcomes require
digest-bound evidence. Missing observations become `probe_missing`; they never become
`not_applicable`.

## Published transport profiles

### `mcp_mode_a` / `mcp-mode-a-v1`

- Maximum total calls: 128 across the same bounded plan the local and endpoint profiles
  accept.
- Maximum summed probe timeout: 600 seconds.
- Per-probe timeout and cleanup timeout: 60 seconds. A Mode A probe crosses two hops, the
  gateway and the MCP server behind it, and every episode pays a session open on both, so
  the endpoint budgets apply here for the same reason.
- Cleanup boundary: MCP session.
- Without a reviewed probe plan intake pins an identity-only descriptor and only the
  discovery projection applies, which caps the attained tier at A0. With a plan, Mode A
  runs the same probe choreography as the conventional transports over the gateway's
  BFCL Oracle HTTP v1 routes. Only Mode A is probeable, because only Mode A exposes reset
  and state as control tools.

### `local_python` / `local-python-v1`

- Maximum total calls: 128 across a bounded plan of at most 16 success, 8
  structured-error, and 1 timeout case.
- Maximum summed probe timeout: 100 seconds.
- Per-probe timeout and cleanup timeout: 10 seconds.
- Cleanup boundary: none for static A0 inspection; worker process for A1/A2.
- A0 identity includes the reviewed source closure and runtime identity. A1/A2 use a
  digest-distinct descriptor and a process worker.

### `http_package` / `http-package-v1`

- Maximum total calls: 128 across the same bounded plan the local profile accepts.
- Maximum summed probe timeout: 600 seconds.
- Per-probe timeout and cleanup timeout: 60 seconds. A probe here is several episodes,
  and one episode costs a worker start and a fresh session, so a budget written for two
  read-only calls would refuse an honest endpoint for being remote. The deadline that
  still carries meaning is the per-call one the probe runner enforces.
- Cleanup boundary: endpoint session. A call that passes its deadline has its session
  deleted before the observation is recorded, so a leaked session is a failed cleanup.
- Catalog integrity requires the reviewed companion schema; schema inference from
  names or observed values is never allowed. Endpoint execution runs the same probe
  choreography as the local transport.

## Conditional applicability

Only these profile-owned reasons are accepted:

- `no_structured_error_case`
- `no_confirmation_tools`

The BFCL verifier derives both conditions from reviewed artifacts. Adapter prose or a
source-provided reason cannot establish non-applicability.

## Stable refusal taxonomy

The v1 taxonomy is `CertificationRefusalCode`:

- `adapter_under_certified`
- `applicability_mismatch`
- `attestation_mismatch`
- `catalog_mismatch`
- `cleanup_failed`
- `cross_origin_redirect`
- `dependency_lock_invalid`
- `dependency_lock_missing`
- `dynamic_import`
- `episode_state_leakage`
- `fixture_metadata_invalid`
- `identity_drift`
- `import_path_ambiguous`
- `mutation_declaration_mismatch`
- `namespace_package_ambiguous`
- `probe_evidence_invalid`
- `probe_failed`
- `probe_missing`
- `probe_timeout`
- `probe_unsafe`
- `profile_mismatch`
- `reset_nondeterministic`
- `response_too_large`
- `result_shape_incomplete`
- `reviewed_schema_invalid`
- `reviewed_schema_missing`
- `reviewed_schema_too_large`
- `source_encoding_invalid`
- `source_package_invalid`
- `source_path_escape`
- `source_syntax_invalid`
- `structured_error_mismatch`
- `undeclared_import`
- `unknown_commit_state`
- `unsupported_auth`

Unknown codes and codes outside a probe's profile fail closed.

## Tier timing

- A0 is required before source evidence can be issued.
- A1 is required before observed live results become model-visible.
- A2 is required at the shared assisted-authoring freeze and fresh-Gold boundary.
- Catalog-only drafting may use A0 only when behavior remains explicitly unresolved
  and model exposure has separate authorization.
