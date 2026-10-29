# BFCL MCP Integration — Architecture Decision

Status: accepted.
Supersedes nothing. Superseded by nothing.

## Decision

An MCP server reaches BFCL through a **gateway that speaks BFCL Oracle HTTP v1**.
The oracle pack keeps declaring `endpoint_config.yaml`; the gateway translates that
contract into MCP `tools/list`, `tools/call`, and the control operations defined in
`bfcl-mcp-oracle-contract.md`. Generation stages do not learn what MCP is, and
`generate_bfcl` is not modified. The Gold Gate gains one provider-neutral endpoint
attestation check: it verifies what an endpoint claims it has proved, without
implementing MCP discovery, transport, or result mapping itself.

```
prepare → Gold Gate → existing ProcessWorker → EndpointOracleClient
                                                   │
                                      BFCL Oracle HTTP v1
                                                   ▼
                                        MCP-to-BFCL gateway
                                                   │
                                              MCP client
                                                   │
                                                JSON-RPC
                                                   ▼
                                               MCP server

Gold Gate ── GET /v1/conformance + pinned digest ──▶ gateway
```

MCP is additionally used as an **intake source** for authoring: the normalized tool
catalog feeds `tools.json` and the evidence bundle that the assisted authoring flow reads.
Intake and oracle execution are separate concerns and ship in that order.

## Context

BFCL certifies a pack by executing it. `run_oracle_validation` needs four operations
per episode — reset from fixtures, call a tool, read final state, and stop a hung
call — and it needs the same episode replayed twice from a fresh reset with an
identical `RunContext` (`clock`, `seed`, `timeout_s`, `task_id`, `turn_index`).

The MCP `2026-07-28` specification standardizes discovery and invocation only.
`tools/list` and `tools/call` are stateless and transport-agnostic by design; the
protocol defines no fixture reset, no per-task isolation, no final-state read, and no
machine-readable error code. Tool execution errors are reported as `isError: true`
with human-readable content, and `annotations` such as mutation hints are explicitly
untrusted. A conforming MCP server is therefore **not** a conforming BFCL oracle.

Two integration shapes were available: teach the BFCL runtime to speak MCP natively,
or place an adapter in front of the existing endpoint contract.

## Rationale

### The endpoint contract already is the adapter boundary

`resolve_pack_paths` requires exactly one of `backend.py` or `endpoint_config.yaml`
and refuses a pack that declares both. `EndpointOracleClient` already implements the
callable surface the worker expects (`list_tools`, `reset`, `call_tool`, `get_state`,
`close`), plus HTTPS-only origins, environment-referenced credentials, an allowlisted
CA bundle, request and response size caps, no redirects, and no retry of mutating
requests. Those controls protect the **BFCL-to-gateway** hop. The gateway separately
implements the equivalent transport policy for the **gateway-to-MCP-server** hop;
security does not propagate across an adapter boundary. A native MCP oracle kind
would still duplicate the first set inside BFCL.

### A native oracle kind is not a local change

`oracle.kind` in `run_manifest.json` is written as `"endpoint"` or `"python"`, and the
evaluation path holds that value to a closed set: `eval/config.py` rejects any
`source_oracle.kind` outside `{python, endpoint}`, and `eval/source_verification.py`
refuses a source manifest whose `oracle.kind` is not one of the same two. Introducing
`kind: mcp` therefore reaches the eval config schema, source verification, the oracle
resource model, and the published manifest — for a benchmark that would be scored
identically either way. Behind a gateway the pack is an endpoint pack, the manifest
stays truthful, and the eval path needs no change at all.

### The sanitized worker cannot hold MCP credentials

`_sanitize_pack_environment()` clears the environment inside the oracle worker and
keeps only `LANG`, `LC_ALL`, `LC_CTYPE`, `TMPDIR`, `TEMP`, and `TMP`. Endpoint
credentials work today because the parent resolves them with
`resolve_endpoint_headers()` and passes headers into the worker. An MCP client
launched inside that worker — a stdio server especially — would find no environment
to authenticate with, and relaxing the sanitizer would hand every pack the host
session. Keeping the MCP client in the gateway leaves the sanitizer untouched.

### Registering a family too early buys obligations, not capability

`scripts/validate.py` discovers a family from any directory under
`runtime/benchmark_families/` that contains `family.py` or `pipeline.py`, and then
requires `step.toml`, `step.py`, and `config/{default,tiny,translate}.yaml` for it.
MCP work needs none of those to be useful. The gateway and the authoring adapter live
outside `benchmark_families/`, and family registration stays a later decision.

### Determinism is a server property, not a protocol property

Because the gap between MCP and BFCL is behavioral rather than syntactic, it has to be
stated as a contract a server operator can implement and a validator can check. That
contract is the BFCL MCP Oracle Profile. Whether it is satisfied by the server itself
or by a shim the gateway owns is an operator choice; either way the same probes decide
eligibility. Building the gateway first makes the profile testable without touching
certified code paths.

## Consequences

- The pack still declares `endpoint_config.yaml`; `oracle.kind` stays `endpoint`, and
  no eval-side change is required.
- The gateway must publish a stable `expected.oracle_id`, `oracle_version`, and
  `sha256:` `content_digest`, because identity is compared during prepare, session
  creation, generation, and publication. This is an **effective** digest over the
  server, normalized catalog, gateway and shim artifacts, and any snapshot — not a
  server version string. Digest derivation is specified in
  `bfcl-mcp-oracle-contract.md`.
- The endpoint config pins a conformance-attestation digest. Prepare retrieves
  `GET /v1/conformance`, verifies that digest, records the document in
  `oracle_validation_report.json`, and permits publication only for level `L2`.
  This is the only Gold-Gate change and is provider-neutral.
- One BFCL session maps to exactly one isolated MCP episode. Session lifetime,
  cleanup, and concurrency become gateway responsibilities.
- A server that cannot reset fixtures, cannot expose deterministic final state, or
  cannot return machine-readable error codes is diagnosed as ineligible rather than
  coerced into a Gold claim.
- Control operations are never exposed to the evaluated model: `tools.json` carries
  business tools only.
- Adding a transport (stdio, Streamable HTTP) is a gateway-local change and does not
  alter the pack contract.
- One extra service boundary sits in the execution path (it may be a local process or
  a remote service), so gateway faults must surface as infrastructure failures rather
  than as business outcomes.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| Native `oracle.kind: mcp` in the BFCL runtime | Reaches eval config, source verification, oracle resource schema, and the manifest; duplicates TLS, auth, size, and retry policy already in `endpoint.py`; blocked by worker environment sanitization for stdio. |
| Generate a `backend.py` that wraps an MCP client per pack | Puts network and credential handling inside fingerprinted pack code that runs in the sanitized worker, and copies the same client into every pack. |
| Use MCP for intake only, keep `backend.py` as the oracle | This is what ships. It lowers authoring effort but does not remove the requirement to write an executable oracle. |
| Treat MCP `annotations` as the mutation and confirmation contract | The specification requires clients to treat annotations as untrusted. BFCL check `M1` compares declared mutation against observed state change, so an unverified hint could certify a wrong claim. |
| Accept text-only tool results and parse error prose | Check `D2` requires `error.code` on every observed structured error and check 5 compares `expect.error_code`. Deriving codes from prose would make a Gold verdict depend on wording. |
| Support the MCP Tasks extension and `InputRequiredResult` in v1 | Both turn one BFCL call into a multi-round-trip exchange with server-side lifecycle. The expected-trace and replay contracts assume one call, one result. Deferred with an explicit rejection instead of a partial implementation. |

## Scope of this decision

**In scope for v1:** synchronous tools; stdio and Streamable HTTP transports; an
accepted *set* of MCP revisions rather than one pinned version; object-shaped
`structuredContent`; a reviewed structured error envelope; reset, state, and
end-episode control; per-episode isolation; pinned server and catalog identity;
MCP-derived `tools.json` and authoring evidence.

Three server situations are in scope, because requiring a purpose-built server would
make the integration useless against anything already deployed: a cooperative server
that implements the control operations, a server that cannot be modified and is driven
through a gateway-owned shim, and a read-only third-party server whose fixtures are
snapshotted from it rather than pushed to it. The profile states them as modes `A`,
`B`, and `C`, and grades what each can earn as levels `L0` (discovery only), `L1`
(executable, non-publishable) and `L2` (certifiable). A level describes what the MCP
side contributed; it never substitutes for BFCL's own gold gate.

**Out of scope for v1:** MCP resources and prompts as oracle truth; the Tasks
extension; `InputRequiredResult`; `notifications/tools/list_changed` driven catalog
refresh during a run; server-side sampling; multi-server tool aggregation; native
MCP support inside `runtime/benchmark_families/bfcl`. Each exclusion is recorded with
its unblocking condition in the contract's deferred-extensions table, so a later
reader can tell a decision from an oversight.

## Follow-on contracts

- `bfcl-mcp-oracle-contract.md` — profile, `mcp_oracle.yaml` schema, tool
  normalization, and result mapping.
- `bfcl-mcp-threat-model.md` — trust boundaries and required controls.
- `bfcl-oracle-pack.md` — the pack, backend, and endpoint contracts this decision
  deliberately leaves unchanged.
