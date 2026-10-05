# BFCL MCP Oracle Contract

Scope: the profile, the `mcp_oracle.yaml` schema, tool normalization, and result and
error mapping.

Profile id: `bfcl-mcp-oracle-v1`.
Read with `bfcl-oracle-pack.md`, which defines the pack, backend, and endpoint
contracts this document maps onto, and with `bfcl-mcp-architecture-decision.md`,
which explains why the mapping happens in a gateway.

The profile is written for three situations, not one: a server the operator built for
BFCL, a server the operator cannot modify, and a read-only third-party server with
static data. Requirements are therefore stated as a **level** (how much trust the
server has earned) crossed with a **mode** (how the episode control plane is obtained).
A requirement that applies to only one mode says so.

## 1. Conformance Levels

| Level | Claim | Requires | Publication |
| --- | --- | --- | --- |
| `L0` discovery | The catalog can be normalized into `tools.json` | Probes `P1`–`P3`: initialize succeeds, every catalog page is retrieved, and the selected tools normalize (§7) | Authoring intake only. No oracle, no execution claim. |
| `L1` executable | Episodes can be reset, called, and read | A working control plane (§4), the result mapping total over every included tool (§8), probes `P1`–`P4` | `lineage.policy: smoke_no_publication` only |
| `L2` certifiable | The oracle is trustworthy enough for BFCL to certify | `L1` plus every applicable target probe, gateway conformance tests, complete observable state, and effective identity (§9–§10) | Publishable if BFCL's own gold gate independently verifies the `L2` attestation and passes |

`L2` is a precondition, not a verdict. A pack on an `L2` server still has to pass
`run_oracle_validation`; the levels only describe what the MCP side contributed.
A level is recorded in the gateway's conformance artifact and in the pack's
provenance, so a pack can never be published on the strength of an unstated
assumption about its server.

### Enforcement boundary

The level is not advisory metadata. An MCP-backed endpoint config pins
`attestation.expected_digest`; during prepare, the endpoint client retrieves
`GET /v1/conformance`, verifies the document against that digest, and copies the
verified document into `oracle_validation_report.json`. `derive_pack_tier` treats all
of the following as publication blockers:

- the attestation is missing, malformed, or does not match the pinned digest;
- `level` is not `L2`;
- `effective_content_digest` differs from `/v1/metadata.content_digest`;
- a required or applicable probe is not `pass`;
- a gateway conformance test refers to a different gateway artifact digest.

This is a provider-neutral extension of endpoint validation: the Gold Gate reads a
strict attestation schema and digests, not MCP messages. An `L1` gateway can still be
used by a pack whose lineage is `smoke_no_publication`; changing lineage does not
upgrade the attestation.

## 2. Operating Modes

| Mode | Situation | Reset strategy | State strategy | Fixtures | Reachable level |
| --- | --- | --- | --- | --- | --- |
| `A` cooperative | The operator controls the server and can add the control tools of §4 | `control_tool` | `control_tool` | pushed to the server | `L2` |
| `B` shimmed | The server cannot be changed; the gateway owns a wrapper that launches, seeds, and inspects it | `process_restart` or `namespace` | `read_only_projection` or a shim-provided control tool | pushed by the shim | `L2` when the shim can seed and isolate |
| `C` read-only snapshot | A third-party server with data the operator cannot seed, exposing non-mutating tools only | `no_op_verified` | `read_only_projection` | **snapshotted from** the server at discovery | `L1` by default; `L2` only with an enforceable read-only boundary, complete state projection, and effective identity |

Mode `C` inverts the usual direction of fixtures. Nothing is pushed; `fixtures.json`
is a snapshot of the server's own data taken during discovery, so template slots bind
to values that exist upstream. Its digest is part of `content_digest`, which means
upstream data drift is detected as an identity change rather than as a run of
mysteriously failing tasks. The cost is stated plainly: mode `C` exposes no mutating
tool, so it cannot cover the `confirmation`, `correction`-of-a-mutation, or
mutation-assertion task categories, and a pack built on it is narrower by
construction rather than by accident. Calling a credential or tool “read-only” is
not enough for `L2`: the boundary must be enforced by the upstream authorization
scope, an immutable snapshot sandbox, or an equivalent mechanism the conformance
artifact names. If hidden state can change without appearing in the state projection,
the mode remains `L1`.

Mode `B` is the one that makes this contract usable against servers in the wild. The
shim is gateway-side code, not pack code, so it is not fingerprinted as pack content;
its exact artifact digest enters effective identity instead (§9).

## 3. Gateway Obligations

The gateway exposes BFCL Oracle HTTP v1 and its provider-neutral conformance
extension, and nothing else. Execution routes and payloads are fixed by
`EndpointOracleClient`; conformance adds only the read-only route and client
method. `protocol_version` remains the literal `bfcl-oracle-http-v1`, which names the
execution contract and is unrelated to the MCP version.

| BFCL route | Gateway obligation |
| --- | --- |
| `GET /v1/metadata` | Return `{protocol_version, oracle_id, oracle_version, content_digest}` with exactly those four keys, all non-empty strings, `content_digest` matching `sha256:<64 hex>`. The digest is the effective digest derived per §9. |
| `GET /v1/conformance` | Return the strict attestation document pinned by the endpoint config. This route is read only, bounded by the endpoint response limit, and may be called before any session exists. |
| `GET /v1/tools` | Return `{"tools": [<name>, ...]}` — the published business tool names in catalog order (§7). Control tools never appear. The client re-reads metadata before every list, so identity must be stable within a run. |
| `POST /v1/sessions` | Body carries `context{clock, seed, timeout_s, task_id}` and `fixtures`. Open one isolated episode, apply the reset strategy, propagate the frozen clock and seed, and return `{session_id, oracle{...identity...}}`. The identity must equal the metadata identity or the client aborts. |
| `POST /v1/sessions/{id}/calls` | Body carries `{name, arguments, turn_index}`. Invoke `tools/call` on the bound episode and return the mapped result object (§8). |
| `GET /v1/sessions/{id}/state` | Return the episode's state as a JSON object, per the mode's state strategy. This is what assertions, `expect.state_unchanged`, and check `D1` read. |
| `DELETE /v1/sessions/{id}` | End the episode and release its resources. |

Robustness rules that are not visible in the route table but decide whether a run can
be trusted:

- **An unknown or expired `session_id` is an error, never a new session.** Silently
  seeding a fresh episode would let a task that lost its session pass against empty
  state, which is the one failure mode that produces a plausible wrong benchmark
  instead of a visible defect. Answer with a named failure (`mcp_session_unknown`)
  and let the run fail.
- **`DELETE` is idempotent.** `reset()` closes the previous session before opening a
  new one, and `close(suppress_errors=True)` can arrive after a failure, so a delete
  for an already-gone session must succeed rather than raise.
- **A call is never retried.** `EndpointOracleClient` does not retry mutating
  requests, and the gateway must not retry `tools/call` either: a transport failure
  after the server applied a mutation is indistinguishable from one before, so a
  retry can double-apply. Report `mcp_call_failed` and let the task drop. A timeout
  or transport failure with unknown commit status **poisons the episode**: the
  gateway attempts teardown and refuses later calls or state reads for that session.
  Closing an HTTP response stream is not evidence that the server rolled back.
- **Infrastructure failures use non-2xx HTTP responses.** `mcp_session_unknown`,
  `mcp_call_failed`, `mcp_call_timeout`, and other `mcp_*` codes are gateway
  diagnostics, not tool results. They are written to the gateway/conformance
  artifact and returned in a bounded error body, but the current
  `EndpointOracleClient` intentionally turns the response into a raised endpoint
  error. A gateway must never return HTTP 200 with `{"error": {"code":
  "mcp_call_failed"}}`: BFCL would classify that as a business error and a negative
  validation case could pass for the wrong reason.
- **Concurrency is supported, not relied upon.** BFCL generation drives one session
  at a time, so correctness must not depend on parallelism; but the gateway must
  serve at least two concurrent episodes, otherwise the isolation probe `P6` cannot
  be written. A session ceiling and an idle TTL bound orphaned episodes, and an
  expiry surfaces as `mcp_session_unknown`.
- **`fixtures` may be `null`.** A pack without `fixtures.json` sends no fixtures;
  that means "no seeding requested", not "empty state".
- **In mode `C` the fixtures BFCL sends must equal the pinned snapshot.** Nothing is
  pushed, but the pack still transmits its `fixtures.json`, and that document is the
  snapshot the identity covers. A mismatch means the pack's slot values describe data
  the server does not have, so it is answered with `mcp_fixtures_not_snapshot` rather
  than ignored — ignoring it would bind templates to values that only exist in the
  pack.
- **`context.clock` arrives as `datetime.isoformat()` output** and may or may not
  carry a timezone offset. Parse it, do not reinterpret it, and never substitute the
  host clock.
- **`turn_index` is advisory.** It records the conversation position for logs. Tool
  results must not depend on it, or the oracle's behavior would depend on how the
  surface was rendered.
- **Nothing is invented.** A missing capability is an error at the route, not a
  default value. BFCL treats every returned object as oracle truth and would certify
  a fabricated one.

## 4. Episode Control Plane

MCP standardizes discovery and invocation only, so the episode operations are carried
as **reserved MCP tools** where the server can provide them. Tools are the one
extension point every MCP SDK and both transports support; a custom JSON-RPC method
is not portable and would not survive Streamable HTTP method routing. Names come from
`mcp_oracle.yaml`, never hardcoded, the same way a pack chooses its own confirmation
vocabulary.

| Operation | Input | Required output | Strategy alternatives |
| --- | --- | --- | --- |
| `describe_oracle` | none | `{oracle_id, oracle_version, content_digest}`, required but not exhaustive | Absent, §9's fallback applies and the level is capped at `L1`. |
| `reset_episode` | `{fixtures, context{clock, seed, timeout_s, task_id}}` | `{episode_id}` | `control_tool`; or `process_restart` / `namespace` (mode `B`); or `no_op_verified` (mode `C`). |
| `get_episode_state` | `{episode_id}` | a JSON object | `control_tool`; or `read_only_projection` — a declared, ordered list of non-mutating calls whose canonicalized results form the state document. |
| `end_episode` | `{episode_id}` | any object | `control_tool`; or process termination / namespace teardown. Must be idempotent either way. |

Whichever strategy is used, these hold:

- **Control results are read from `structuredContent` only.** A control operation that
  answers with `content` text alone is refused, even when that text happens to be
  parseable JSON. Inferring a shape from prose would let an untrusted server steer the
  identity checks that decide publication eligibility, and the fallback the profile
  already defines is a reviewed Mode `B` shim that declares the structure (§6). This is
  the same refusal §8 applies to business results, applied earlier.
- **Reset replaces, never merges.** A second episode must not inherit the first's
  mutations. Under `no_op_verified` the strategy is only valid if no exposed tool can
  mutate observed state, and probe `P10` must confirm exactly that.
- **A `read_only_projection` is deterministic, pure, and complete for the exposed
  effects.** The declared probe calls
  run in declared order, their results are serialized with the repo's
  `canonical_json`, and they must not mutate. Reading state must be repeatable,
  because check `D1` compares state around a call. For `L2`, every state change an
  exposed tool can cause must change this document. A projection that only samples
  visible fields while hidden state may mutate is diagnostic state and caps the
  endpoint at `L1`.
- **Episode binding is verified, not assumed.** Allowed channels, in order of
  preference: `transport` (one process or namespace per episode, so no token is
  needed), `meta` (`_meta.bfcl.episode_id` on the `tools/call` request, the
  specification's designated extension slot), or `argument` (an injected argument
  named in config). A server that accepts `_meta` and ignores it routes every call to
  one shared episode while looking healthy, so probe `P6` must prove binding works
  rather than trusting that it does. With `argument`, the **upstream** MCP input
  schema must declare the episode property; normalization removes that property and
  its `required` entry from the model-facing schema, and the gateway injects it only
  after validating the model arguments. If the upstream schema does not declare it,
  argument binding is invalid at `L2` even when `additionalProperties` is true. The
  evaluated model never sees or chooses an episode id.
- **Isolation.** Two concurrent episodes must not observe each other's state, and a
  new episode must not inherit a previous one's. BFCL check `I1` separately requires
  `oracle_runtime.worker: process`; the gateway's own isolation mode is declared in
  config and probed by `P6`.
- **Determinism.** Given the same fixtures and the same `context`, two fresh episodes
  must produce identical results and identical state. Wall-clock reads, unseeded
  randomness, unpinned upstream dependencies, and counters that survive a reset all
  violate this, and check `D1` is what detects them.
- **Frozen time and seed are conditional inputs.** The externally testable
  requirement is observational determinism: identical snapshot, arguments, and
  context yield identical results and state. A tool that generates timestamps,
  random identifiers, or sampled values must consume `context.clock` and
  `context.seed`; a static read-only server need not accept inputs it never uses.
- **Bounded calls.** A call must return or fail within `limits.tool_timeout_s`. Check
  `T1` requires that a hung tool is killed on the same path pack tools run through,
  so the gateway must cancel in flight — `notifications/cancelled` on stdio, closing
  the response stream on HTTP — and answer the route with a failure rather than
  hanging.
- **Structured errors.** Every failure the pack is expected to certify must arrive as
  a machine-readable code (§8). Check `D2` requires `error.code` on every observed
  structured error, and validation cases compare `expect.error_code`.
- **Mutation honesty.** A tool declared mutating must change state in at least one
  successful probe, and a tool that changes observed state must be declared
  (check `M1`). Declarations come from reviewed config, never from MCP `annotations`,
  which the specification requires clients to treat as untrusted.
- **Confirmation protocol.** A gated tool must return the pending status without
  mutating state when the confirmation argument is false, matching the pack
  manifest's `confirmation.{parameter, status_field, pending_status}` — which default
  to `confirm`, `status`, and `awaiting_confirmation`.

## 5. `mcp_oracle.yaml` Schema

The file lives in the oracle pack tree, so `pack_fingerprint` covers it and its
content is part of what a Gold verdict describes. It holds env var **names** only,
never secret values, mirroring `_reject_model_secrets`. Unknown keys are rejected
rather than ignored, mirroring `_reject_unknown`: a misspelled control name that
silently defaulted would be certified as a working oracle.

```yaml
profile_version: bfcl-mcp-oracle-v1
mode: A                              # A | B | C, per §2
mcp_protocol_versions: ["2026-07-28", "2025-06-18"]   # accepted at initialize

transport:
  kind: stdio                        # stdio | streamable_http
  # kind: stdio
  command: ["python", "-m", "acme_banking_mcp"]
  cwd: /opt/acme-banking-mcp         # must resolve under an allowed root
  env_passthrough: [ACME_MCP_TOKEN, ACME_MCP_TENANT]
  executable_policy: acme-mcp-runtimes
  # kind: streamable_http
  # url: https://mcp.internal.example.com/mcp
  # auth: {bearer_token_env: ACME_MCP_TOKEN, headers: {X-Tenant: ACME_MCP_TENANT}}
  # tls: {ca_bundle_path: ./ca/internal-root.pem}

expected:
  server_name: acme-banking-mcp
  server_version: "3.2.0"
  tool_catalog_digest: "sha256:<64 hex>"
  oracle_id: acme-banking
  oracle_version: "3.2.0"
  server_content_digest: "sha256:<64 hex>"  # omit only when describe_oracle is absent

control:
  reset_strategy: control_tool       # control_tool | process_restart | namespace | no_op_verified
  state_strategy: control_tool       # control_tool | read_only_projection
  describe_oracle: bfcl.describe_oracle
  reset_episode: bfcl.reset_episode
  get_episode_state: bfcl.get_episode_state
  end_episode: bfcl.end_episode
  episode_binding: meta              # transport | meta | argument
  # episode_argument: episode_id     # required only when episode_binding=argument
  # state_projection:                # required only when state_strategy=read_only_projection
  #   - {tool: get_account, arguments: {account_id: acc_1}}
  #   - {tool: list_recent_transactions, arguments: {account_id: acc_1, limit: 10}}

fixtures:
  direction: pushed                  # pushed | snapshot (snapshot is mode C)
  # snapshot_calls:                  # required only when direction=snapshot
  #   - {tool: list_accounts, arguments: {}, collection: accounts}

tools:
  include: [get_account, list_recent_transactions, create_transfer]
  aliases: {"admin.audit.read": admin_audit_read}
  mutates: [create_transfer]
  requires_confirmation: [create_transfer]
  trust_annotations: false

results:
  error_path: error                  # where the server puts it; republished as "error"
  status_field: status               # must equal the pack manifest confirmation vocabulary
  pending_status: awaiting_confirmation
  confirmation_parameter: confirm    # must equal manifest confirmation.parameter

isolation: process_per_episode        # process_per_episode | namespace_per_episode

limits:
  connect_timeout_s: 5
  handshake_timeout_s: 5
  tool_timeout_s: 5
  reset_timeout_s: 10
  episode_timeout_s: 60
  max_response_bytes: 10485760
  max_tools: 64
  max_catalog_pages: 20
  max_concurrent_episodes: 4
  session_idle_ttl_s: 300
```

Validation rules the loader enforces:

- YAML duplicate keys are refused in both `mcp_oracle.yaml` and the host-owned
  `trusted_executables` policy. PyYAML otherwise keeps the last value silently, which
  would let a file display one reviewed identity, mode, or digest while the runtime
  uses another. Malformed YAML is reported as an `McpConfigError`, not leaked as an
  uncaught parser exception.
- `profile_version` must be a profile this build implements; an unknown value is
  refused instead of being treated as the current one.
- `mode` must be consistent with the strategies it implies: `no_op_verified` requires
  an empty `tools.mutates`, `snapshot` requires `snapshot_calls`, and
  `read_only_projection` requires `state_projection`. A mode that contradicts its
  strategies is refused rather than reconciled.
- Exactly one transport shape: `command` with `stdio`, `url` with
  `streamable_http`. Mixing them is refused.
- `url` must be `https` with no credentials, query, or fragment. `http` is allowed
  only for `localhost` and only under a non-publication lineage policy.
- Custom HTTP header names are compared case-insensitively. A profile cannot declare
  both `X-Tenant` and `x-tenant`, because HTTP treats them as the same field while a
  Python mapping does not, leaving client behavior dependent on serialization order.
- `command` must be an argument vector, never a shell string, and its executable must
  resolve through a separately configured `trusted_executables` policy. That policy
  pins the resolved executable path and artifact/package digest; it is not
  `oracle_runtime.allowed_roots`. Pack roots govern what pack data may be read,
  whereas executable policy governs what host code may run. `env_passthrough` names
  variables to forward; every other variable is withheld from the child.
- `ca_bundle_path` must resolve under an allowed root and inside the pack tree, so it
  participates in the fingerprint the way the endpoint CA bundle does.
- `expected.tool_catalog_digest` is required. `expected.server_content_digest` is
  required when `control.describe_oracle` is present and must equal its response.
  When the operation is absent, §9's fallback applies and a live endpoint is capped
  at `L1`. The three identity fields must be present in that response; additional
  descriptive fields such as a build id are allowed, are recorded verbatim in the
  discovery report as `oracle_declaration`, and therefore drift the report digest
  rather than being silently dropped or hard-failing a conformant server.
- The generated `endpoint_config.yaml`, not this intake file, pins the conformance
  attestation as shown below. Omitting that block caps the endpoint at `L1`; a pack
  file that merely says `level: L2` without a matching live endpoint attestation has
  no authority.
- `tools.include` is an explicit ordered allowlist. A discovered tool that is not
  listed is excluded; a listed tool that is not discovered is an error. There is no
  implicit "expose everything".
- `aliases` maps a discovered MCP name to the published BFCL name (§7). Values must
  be unique, must not collide with an unaliased name, and must not name a control
  tool.
- `mutates` and `requires_confirmation` must reference published names, and
  `requires_confirmation` requires the tool to declare a boolean input named by
  `results.confirmation_parameter`. That name is configuration, not a constant, so a
  pack whose manifest gates on `confirmed` is expressible without editing BFCL. Its
  schema must admit both `false` and `true`; `const: true`, `enum: [true]`, and their
  false-only counterparts are refused because probe `P8` must exercise both decisions.
- `trust_annotations` must be `false` at `L2`, and mode `C` refuses it outright:
  deriving `x-mutates` from `readOnlyHint` would let the server contradict the
  read-only surface the operator declared and the loader already validated.
- `episode_argument` is required when and only when `episode_binding: argument`, and
  the named argument must be declared by every bound upstream tool, then removed from
  each published model-facing parameter schema. Removing it must leave a valid schema;
  the gateway injects it after model-argument validation.
- `results.error_path` names where the server places its error object inside
  `structuredContent`, written as a dotted path such as `error` or `result.error`;
  the gateway republishes it under the key `error`, because
  `_classify_result` classifies on exactly that key and nothing else.
  `results.status_field`, `results.pending_status`, and
  `results.confirmation_parameter` must equal the pack manifest's
  `confirmation.{status_field, pending_status, parameter}`; disagreement is
  refused rather than resolved in favor of one file, since BFCL classifies with the
  manifest's vocabulary while the gateway would be mapping with another.
  `results.confirmation_parameter` must also differ from `control.episode_argument`,
  because the episode argument is stripped from every published schema and would
  silently remove the gate the confirmation probe depends on.
- Every value in `limits` must be a positive finite number; `NaN` and infinities are
  refused before they reach deadline arithmetic, as elsewhere in BFCL config. Each also
  has a sanity ceiling — one hour for any duration, 256 MiB for `max_response_bytes`,
  4096 tools, 1000 pages, 256 concurrent episodes. These are not policy: they exist so
  that `tool_timeout_s: 5000` written in milliseconds, or a bound large enough to
  disable itself while still looking configured, fails at load time rather than at the
  first hung call.
- No key may hold a secret-looking literal. Credentials are named, never inlined.

The host-owned executable policy is a separate file, never part of the Oracle Pack:

```yaml
schema_version: bfcl-trusted-executables-v1
policies:
  acme-banking-server:
    executable: /opt/acme/bin/python
    sha256: "sha256:<64 hex>"
    allowed_argv:
      - ["-m", "acme_banking.mcp_server"]
    allowed_cwd_roots:
      - /srv/bfcl/mcp-packs
```

The launcher replaces the pack-provided executable token with the pinned absolute
path, verifies the binary digest, requires an exact allowed argument vector, and
requires `cwd` to remain under one of the policy roots. This prevents a reviewed
interpreter policy from becoming permission to run arbitrary `-c` code or an
unreviewed module.

The implementation uses MCP Python SDK v2 for the 2026 protocol. Data Designer
currently requires SDK v1, so the repository exposes a mutually exclusive
`bfcl-mcp` runtime extra instead of installing v2 into the `byob` authoring
environment. The two environments meet at the gateway process boundary defined by
the ADR. Discovery is available through:

```bash
uv run --extra bfcl-mcp python -m \
  nemotron.steps.byob.scripts.discover_mcp_oracle \
  --config path/to/mcp_oracle.yaml \
  --output path/to/mcp_discovery_report.json
```

For first-time pinning, `--bootstrap-catalog-digest` writes a non-conformant
`needs_catalog_pin` report containing the observed digest and exits nonzero. The
operator reviews that report, copies the digest into `expected`, and reruns without
the bootstrap flag. Only the second, matching run attains `L0`.

The report's `source_config_digest` covers the reviewed `mcp_oracle.yaml` document as
written, not the loaded model. Loading resolves `cwd` and `ca_bundle_path` against the
host filesystem, so digesting the model would make the same reviewed file hash
differently on a developer laptop and in CI, and the surrounding `report_digest` would
stop being reproducible evidence. `implementation.sdk_version` records the exact MCP
SDK distribution observed at runtime in addition to the accepted range
`sdk_requirement`; two minor SDK releases can negotiate the same protocol while
changing serialization or transport behavior, so the range alone is not provenance.
Before opening a connection, discovery reparses `raw_document`, resolves only those two
host-relative paths, and proves it equals the effective validated model. The public API
therefore cannot be given reviewed document A and runtime model B and produce a report
that hashes A while executing B.

`report_digest` covers the complete report before that field is added. Both
`to_dict()` and the atomic writer recompute it and refuse export if the mutable report
document changed after discovery. This prevents a caller from editing `status`,
evidence, or checks while leaving a stale digest that appears to attest the edited
content.

For MCP-backed endpoints, conformance extends the provider-neutral endpoint config with
one optional block. Existing Python packs and endpoints that do not claim conformance
remain valid:

```yaml
protocol_version: bfcl-oracle-http-v1
base_url: https://127.0.0.1:9443
expected:
  oracle_id: acme-banking
  oracle_version: "3.2.0"
  content_digest: "sha256:<effective digest>"
attestation:
  kind: bfcl-endpoint-conformance-v1
  expected_digest: "sha256:<digest of GET /v1/conformance>"
```

`load_endpoint_config` rejects unknown keys today, so this block requires an explicit
schema change; documentation alone does not activate it. The loader rejects unknown
attestation fields and requires the block whenever publication is requested for an
MCP-backed endpoint.

## 6. MCP Version Support

`mcp_protocol_versions` lists the revisions the operator accepts. The version
negotiated at `initialize` must be one of them, and the negotiated value — not the
list — enters the catalog digest, so a server that silently downgrades changes the
pack's identity instead of changing its behavior unnoticed.

The profile depends on three features that arrived at different revisions, so support
is stated per feature rather than pinned to one specification date:

| Feature | Available from | If the negotiated version lacks it |
| --- | --- | --- |
| `structuredContent` on a tool result | `2025-06-18` | Mode `B` with a shim that projects `content` into a declared structured shape, from reviewed config. The gateway never parses free text on its own. |
| `outputSchema` on a tool definition | `2025-06-18` | Authoring evidence is weaker; assertions must be drafted from probes instead. Not a blocker. |
| `annotations` on a tool definition | `2025-03-26` | No effect: annotations are untrusted and unused for declarations either way. |

`ttlMs` and `cacheScope` on `tools/list` exist only in the newest revisions and are
ignored in every version: a cache directive is not part of a tool contract, and
honoring one would let a stale catalog back a fresh certification.

## 7. Tool Normalization

`tools.json` is an OpenAI-compatible function catalog whose parameter schema is
checked by `validate_tool_definition`, which enforces a deliberately small JSON Schema
subset. Normalization maps an MCP tool definition into that subset, or excludes the
tool with a recorded reason. It never widens the subset and never guesses.

| MCP field | BFCL destination |
| --- | --- |
| `name` | `function.name`, after the name rules below |
| `description` | `function.description`, after the text rules below |
| `inputSchema` | `function.parameters` |
| `outputSchema` | not published; recorded as authoring evidence and as input for assertion drafting |
| `annotations` | not published; recorded as unverified metadata, plus a `mutation_source` showing whether it influenced `x-mutates` |
| `title`, `icons`, `_meta`, `x-mcp-header` | dropped |

`x-mutates` and `x-requires-confirmation` are siblings of `function` on the tool
entry, written from `tools.mutates` and `tools.requires_confirmation`. They are
internal: `project_model_facing_tools` strips them before a row is published.

### Names

MCP allows dots and up to 128 characters. Function-calling endpoints commonly accept
only `[A-Za-z0-9_-]` up to 64 characters, so a name discovery accepts can fail at
evaluation time instead. A discovered name is published unchanged when it matches
`^[A-Za-z0-9_-]{1,64}$`; otherwise it needs an explicit `tools.aliases` entry that
does, and the gateway keeps the alias-to-real mapping for its own `tools/call`. An
unmatched, unaliased name is excluded as `name_not_publishable`. Names are
case-sensitive and must be unique after aliasing. A discovered name with leading or
trailing whitespace is refused rather than trimmed: trimming would make discovery pin
`inventory.lookup` while the server's callable route remains `inventory.lookup `, so
the catalog would pass and every later call would address a different name.

### Parameter schemas

Allowed keywords are exactly the ones `validate_function_schema` accepts: `type`,
`description`, `title`, `default`, `examples`, `deprecated`, `readOnly`, `writeOnly`,
`format`, `enum`, `const`, `properties`, `required`, `additionalProperties`, `items`,
`pattern`, `minLength`, `maxLength`, `minimum`, `maximum`, `minItems`, `maxItems`.
Allowed types are `string`, `integer`, `number`, `boolean`, `array`, `object`, `null`.

- `inputSchema` must explicitly say `type: object`, or the tool is excluded as
  `parameters_not_object`. Missing `type` is not repaired: under JSON Schema it means
  “any type”, and synthesizing `object` would narrow the upstream contract while
  claiming normalization never guesses.
- `$ref`, `allOf`, `anyOf`, `oneOf`, and `not` are excluded as
  `unsupported_schema_keyword`. MCP permits them and BFCL refuses them, so the tool is
  reported rather than silently flattened — an incorrectly inlined `$ref` would ship a
  schema the server does not actually enforce.
- Any other unrecognized keyword is excluded as `unsupported_schema_keyword`. Dropping
  it would publish a looser contract than the server implements.
- `additionalProperties` must be a boolean and is preserved as declared. It is never
  synthesized: adding `false` would make BFCL reject arguments the server accepts, and
  adding `true` would claim the opposite.
- `required` must list declared properties only and must not contain duplicates.
  Duplicates are invalid JSON Schema and are refused rather than silently deduplicated.
- An unsatisfiable constraint is refused, not published: `minimum` above `maximum`, an
  empty `enum`, a duplicated `enum` value, or an `enum` or `const` value the declared
  `type` rejects. Check 3 would fail the pack anyway; failing at intake names the tool.
- Order is deterministic: tools sorted by published name, object keys canonicalized
  through `canonical_json`, `required` sorted. Two discoveries of an unchanged server
  must produce byte-identical `tools.json`.
- With `episode_binding: argument`, normalization performs one additional,
  deterministic projection: remove the configured episode property from
  `properties` and `required` after verifying it is declared upstream. The unmodified
  upstream schema remains in evidence and is what the gateway uses after injection.
  This is the only parameter removed for transport purposes; arbitrary backend-only
  parameter stripping is not allowed. The removed property must explicitly declare
  `type: string`, matching the identifier returned by `reset_episode`; otherwise the
  gateway could inject a string into a schema that accepts only numbers and every
  business call would fail after model-argument validation had already passed.

### Descriptions

A description is model-facing text from an untrusted source, so it is treated as data.
It is length-capped, and the primary control is that BFCL never executes it as an
instruction. See `bfcl-mcp-threat-model.md` §2. Two further checks apply, and the
difference between them matters:

- **Invisible and direction-overriding characters are refused, not flagged.** C0/C1
  controls other than tab, newline, and carriage return; the bidirectional overrides,
  embeddings, and isolates `U+202A`–`U+202E` and `U+2066`–`U+2069`; and the zero-width
  space and byte-order mark are rejected for a selected tool. A warning would be
  incoherent here, because the reviewer it asks cannot see the characters in question,
  and a bidi override can make one string read differently to a human and to a parser.
  The directional marks `U+200E`/`U+200F` and the joiners `U+200C`/`U+200D` are
  deliberately allowed: Arabic, Hebrew, Persian, Indic, and emoji text needs them, and
  refusing them would make the profile usable only for Latin-script servers.
- **Instruction-like phrasing is a warning, and the lexicon is English only.** Matching
  words such as `ignore` or `system prompt` catches the most common phrasing and
  nothing else; the same instruction in Vietnamese or Japanese passes. It is recorded
  as `suspicious_description` for review and must not be read as a control. The
  language-independent companions are `description_embeds_block` and
  `description_embeds_url`, since prose in any script does not smuggle a fenced block,
  an HTML comment, or a URL by accident.

### Mutation and confirmation

`annotations.readOnlyHint`, `destructiveHint`, `idempotentHint`, and `openWorldHint`
are unverified server claims. By default they are shown to a reviewer and recorded as
evidence but do not become the declaration: check `M1` compares the declaration
against observed state change, and an untrusted hint could certify a false claim.
Modes `A` and `B` may explicitly opt into `trust_annotations`; mode `C` and eventual
`L2` certification refuse that shortcut.

Every published tool records a `mutation_source` of `config`, `server_annotation`, or
null, so a reviewer can see whether a mutation flag came from the reviewed pack or from
the server's claim about itself. Disagreement in either direction is reported rather
than silently resolved in favor of the config:

| Reviewed config | `readOnlyHint` | Result | Warning |
| --- | --- | --- | --- |
| lists the tool in `mutates` | `true` | mutating, `mutation_source: config` | `mutation_disagreement` |
| omits the tool | `false`, `trust_annotations: true` | mutating, `mutation_source: server_annotation` | none |
| omits the tool | `false`, `trust_annotations: false` | not mutating | `undeclared_mutation_hint` |

The last row is the dangerous one: BFCL publishes a non-mutating surface while the
server says otherwise, which is exactly the state that would make check `M1` certify a
false claim, so it is surfaced at intake instead of being discovered later.

### Catalog paging

`tools/list` may return `nextCursor`. All pages are followed up to
`limits.max_catalog_pages`; a truncated catalog is an error, not a smaller catalog.
`listChanged` notifications are not subscribed to during a run: the catalog is pinned
by digest, so a change is drift to report rather than an update to apply.

Because MCP result models mirror the wire schema, the continuation cursor arrives as
camelCase `nextCursor` while Python callers habitually reach for `next_cursor`. Both
spellings are accepted and a result exposing neither is a hard failure, since treating
an unreadable cursor as "no more pages" would end pagination after the first page and
pin a digest over a catalog BFCL never finished reading — a silent partial catalog that
still passes every other check as long as the selected tools happen to appear early.
The same rule governs the rest of the SDK surface: a missing `serverInfo`,
capabilities, or protocol version fails loudly rather than defaulting.

## 8. Result And Error Mapping

`call_tool` must return an object BFCL can classify into exactly one of `success`,
`structured_error`, or `awaiting_confirmation` — the three classes
`validation_cases.yaml` declares through `expect.result_class`. MCP's result shape is
looser than that, so the mapping is total and explicit.

| MCP `tools/call` result | Mapped BFCL result | Class |
| --- | --- | --- |
| `isError` absent or false, no value exists at `results.error_path`, and `status_field` equals `pending_status` | `structuredContent` verbatim | `awaiting_confirmation` |
| `isError` absent or false, no value exists at `results.error_path`, and `status_field` is absent or differs from `pending_status` | `structuredContent` verbatim | `success` |
| `isError: true` **and** the object at `results.error_path` carries `code` as a non-empty string | that object under the key `error` | `structured_error` |
| `isError` absent or false but any value exists at `results.error_path` | route failure `mcp_error_flag_inconsistent` | none |
| `isError: true` without a machine-readable code | route failure `mcp_unstructured_error` | none |
| `structuredContent` absent, or not a JSON object | route failure `mcp_result_not_object` | none |
| `InputRequiredResult` | route failure `mcp_input_required_unsupported` | none |
| A task handle (Tasks extension), recognized by a `task` or `taskId` member rather than by a `resultType` value | route failure `mcp_async_task_unsupported` | none |
| Any other `resultType`, which the specification leaves as an open string | route failure `mcp_unsupported_result_type` | none |
| JSON-RPC protocol error | route failure `mcp_protocol_error` | none |
| Timeout or cancellation | route failure `mcp_call_timeout` | none |
| Transport failure mid-call | route failure `mcp_call_failed`, never a retry | none |

The precedence is fixed: reject unsupported result shapes, then protocol and shape
errors; then, only when `isError: true`, map a valid structured error; then map pending
confirmation; finally map success. `resultType` is allowlisted rather than
denylisted — `complete` and an absent value are the only mappable ones — because an
unrecognized extension that happened to carry `structuredContent` would otherwise be
published as a business outcome the server never asserted. The same allowlist guards
control results, so a reset or state read cannot be satisfied by an extension shape. An object cannot be both a pending confirmation and an error. A legacy
server that returns an error envelope without `isError` needs a reviewed mode-`B`
shim that supplies the missing protocol signal; the gateway does not reinterpret a
domain field by itself.

The `content` array is never the source of truth. It is human-facing text a server may
format freely, and MCP only *recommends* mirroring `structuredContent` into it. It is
retained in the diagnostic artifacts and never reaches a published row.

When the tool declares `outputSchema`, the gateway validates `structuredContent`
against that schema before applying the table above. A mismatch is the infrastructure
failure `mcp_output_schema_mismatch`, not a business error. The original output schema
and validator implementation digest are retained in evidence; silently dropping an
unsupported output constraint would certify data the server's own contract rejects.

`structuredContent` may legally be any JSON value, including an array or a scalar.
BFCL reads results as mappings — `_classify_result` looks up the `error` key and a
status field, and assertions index fields — so a non-object result is refused rather
than wrapped. Wrapping would invent a key the server never emitted, and every
assertion and validation case would then be written against gateway-invented
structure. A **declared** projection is a different matter and is the deferred
extension in §11: the distinction that matters is whether a human reviewed and pinned
the shape, not whether the gateway could guess it.

The error object passes through as-is beyond requiring `code`, so a server may carry
`entity`, `id`, `field`, and `message` exactly as `bfcl-oracle-pack.md` describes for
a Python backend. A route failure is a defect in the oracle or the transport, not a
business outcome: it surfaces as a failing probe or a dropped task and can never
satisfy a negative validation case.

Worked example of a confirmation-gated mutation:

```json
// tools/call → create_transfer {"to_account": "acc_2", "amount_vnd": 200000, "confirm": false}
{
  "content": [{"type": "text", "text": "Confirm transfer of 200,000 VND to acc_2?"}],
  "structuredContent": {"status": "awaiting_confirmation", "amount_vnd": 200000}
}
// mapped to {"status": "awaiting_confirmation", "amount_vnd": 200000} → awaiting_confirmation
// check 6 then requires the state read to be unchanged
```

```json
// tools/call → create_transfer {"to_account": "acc_9", "amount_vnd": 200000, "confirm": true}
{
  "content": [{"type": "text", "text": "No such account: acc_9"}],
  "structuredContent": {"error": {"code": "account_not_found", "entity": "account", "id": "acc_9"}},
  "isError": true
}
// mapped to {"error": {"code": "account_not_found", ...}} → structured_error, satisfies expect.error_code
```

## 9. Identity And Digests

BFCL compares oracle identity at four moments — prepare, session creation,
generation, and publication — so the gateway must publish an identity that changes
when the oracle's behavior changes and stays stable when nothing does. Every digest
below is `sha256` over `row_schema.canonical_json` of the stated document, so key
order, separators, and non-ASCII handling are decided once, in code the pipeline
already uses.

- **`tool_catalog_digest`** covers
  `{profile_version, mode, negotiated_mcp_version, server_name, server_version, tools: [normalized definitions in published order], control: {resolved names and strategies}}`.
  Volatile fields — `nextCursor`, `ttlMs`, `cacheScope`, request ids — are excluded,
  because a cache directive or a pagination token is not part of the contract.
- **`server_content_digest`** is the digest the server reports through
  `describe_oracle`. It is the server's statement about domain content — fixture
  handling, business rules, and data version — which a catalog digest cannot see.
- **Artifact digests, not version labels.** `gateway_artifact_digest` and an optional
  `shim_artifact_digest` hash the exact executable/package artifacts used. Version
  strings remain descriptive metadata and never substitute for these hashes.
- **`snapshot_digest`** is present in mode `C` and covers the exact canonical fixture
  snapshot BFCL sends back on session creation.
- **`effective_content_digest`**, exposed as `/v1/metadata.content_digest`, is
  `sha256(canonical_json({server_content_digest, tool_catalog_digest,
  gateway_artifact_digest, shim_artifact_digest, snapshot_digest,
  profile_version, negotiated_mcp_version}))`. Null optional components are retained
  so two modes cannot hash the same document accidentally.
- **Fallback.** Without `describe_oracle`, `server_content_digest` is null. The
  effective digest still detects catalog, adapter, shim, and snapshot drift, but not a
  live server changing business logic behind the same catalog. A live endpoint is
  therefore capped at `L1`. Mode `C` may reach `L2` without a server digest only when
  it executes against an immutable local snapshot sandbox rather than the live
  third-party service; the sandbox artifact and snapshot are then the oracle being
  certified.
- **Handshake pinning.** `expected.server_name`, `expected.server_version`, and the
  negotiated MCP version are checked before any probe runs, because a probe against
  an unexpected server produces confident results about the wrong thing.

### 9.1 Conformance attestation

`GET /v1/conformance` returns exactly this provider-neutral shape:

```json
{
  "schema_version": "bfcl-endpoint-conformance-v1",
  "provider_kind": "mcp",
  "profile_version": "bfcl-mcp-oracle-v1",
  "level": "L2",
  "effective_content_digest": "sha256:<64 hex>",
  "gateway_artifact_digest": "sha256:<64 hex>",
  "shim_artifact_digest": null,
  "tool_catalog_digest": "sha256:<64 hex>",
  "server_content_digest": "sha256:<64 hex>",
  "snapshot_digest": null,
  "probe_report_digest": "sha256:<64 hex>",
  "gateway_conformance_report_digest": "sha256:<64 hex>",
  "gateway_evidence_kind": "locally_verified",
  "gateway_evidence_issuer": "bfcl-mcp-conformance-v1",
  "state_observability": "complete",
  "read_only_boundary": null,
  "checks": [
    {"id": "P1", "requirement": "required", "status": "pass", "reason": null},
    {"id": "P8", "requirement": "conditional", "status": "not_applicable", "reason": "no gated tool"}
  ]
}
```

Unknown or missing fields are rejected. Optional digests are explicit JSON `null`,
not omitted. `state_observability` is `complete` or `diagnostic`; only `complete`
can reach `L2`. A mode-C attestation additionally names `read_only_boundary` as
`upstream_authorization`, `immutable_snapshot_sandbox`, or another profile value
implemented by the verifier; free-form claims are not accepted.

The Gold Gate treats `checks` generically: every `required` check must be `pass`; a
`conditional` check may be `not_applicable` only with a non-empty reason; `skipped`
never satisfies either. It does not need to know what `P1` or `P8` means. The
provider-specific report explains each check and is pinned by
`probe_report_digest`.

The gateway does not get to certify itself. `gateway_evidence_kind` is either
`locally_verified`, meaning prepare ran the BFCL-owned conformance suite against the
exact artifact digest, or `signed_release`, meaning the report signature chains to a
trust root configured outside the pack. A self-reported or self-signed report caps the
endpoint at `L1`, regardless of its `level` field.

The attestation digest is `sha256(canonical_json(document))`. The probe and gateway
conformance reports are content-addressed artifacts retained with the prepare output;
their digests are not evidence when the corresponding artifact is absent. The Gold
Gate verifies the attestation and reports rather than trusting `level` as a summary
flag, following the same rule by which generation re-derives Gold eligibility from
individual checks.

## 10. Conformance Probes

The attestation separates **gateway conformance tests**, run against a controlled MCP
fixture server when the gateway artifact is built, from **target probes**, run against
the selected server and its declared validation cases. This avoids requiring an
arbitrary third-party server to expose a deliberately slow or invalid tool merely to
test gateway mechanics.

| Probe | Passes when | Protects | Needed for |
| --- | --- | --- | --- |
| `P1 handshake` | negotiated MCP version, server name, and server version match `expected` | identity comparison | `L0` |
| `P2 catalog` | all pages retrieved, digest matches `expected.tool_catalog_digest`, every `tools.include` entry found, published set is exactly `tools.include` | check 3, `TM-13` | `L0` |
| `P3 normalization` | every included tool maps into the supported schema subset | check 3 | `L0` |
| `P4 executable smoke` | the mode's reset, state, and teardown strategies work, and at least one declared success case per included tool maps under §8 | reset, call, state, close | `L1` |
| `P5 reset` | two fresh episodes with identical fixtures and context yield identical state | check `D1` | `L2` |
| `P6 isolation` | mutating modes prove cross-episode state separation; read-only mode proves its authorization/sandbox boundary and that bound calls reach the intended episode | check `I1`, binding | `L2` |
| `P7 error shape` | every validation case expected to produce a structured error returns `error.code`; not applicable when the pack declares no structured-error case | check `D2` | `L2` when applicable |
| `P8 confirmation` | a gated tool called with confirmation false returns the pending status and leaves state unchanged | check 6 | `L2` when any tool is gated |
| `P9 gateway timeout conformance` | the pinned gateway artifact cancels a deliberately hung call on a controlled fixture MCP server, never retries it, poisons the episode, and does not mistake stream closure for rollback | check `T1`, unknown commit state | `L2`; trusted build-time or prepare-time conformance test |
| `P10 mutation` | every `tools.mutates` entry changes state in at least one successful probe, and no undeclared tool does | check `M1` | `L2` |
| `P11 observed result shapes` | every result path exercised by validation cases maps under §8; coverage is recorded per tool and result class rather than claimed for unobserved outputs | §8 | `L2` |

A required probe whose precondition failed is recorded as skipped, never as passed,
following the rule `run_oracle_validation` already applies. A conditionally
applicable probe is recorded as `not_applicable` with the fact that made it
inapplicable — for example, `P8` when no tool is gated or `P7` when no structured
error case is declared. `not_applicable` does not claim coverage. The attestation
contains the per-tool paths actually observed, so `P11` cannot be read as a universal
claim about outputs no test exercised.

### 10.1 Executable gateway boundary

The gateway implementation is **mode-A and L1-capable**. It lives
under `runtime/mcp/gateway/` and exposes the six execution routes in §3 through a
Starlette adapter while keeping the session lifecycle in a transport-neutral service.
`scripts/run_mcp_gateway.py` is the operator entry point and must run in the isolated
`bfcl-mcp` dependency environment.

This implementation:

- reruns identity and complete-catalog discovery on each fresh MCP connection before
  reset, so startup success cannot authorize a later drifted connection;
- assigns one opaque BFCL session to one MCP episode; a dedicated episode worker owns
  the MCP context from enter through exit so reset, calls, state, cancellation, expiry,
  and teardown cannot violate the transport's task-affinity rules;
- supports `argument`, `_meta`, and transport-scoped episode binding for business
  calls without publishing the gateway-owned episode argument;
- validates inbound BFCL payloads, rejects duplicate JSON keys, enforces request,
  session, idle, episode, call, and reset bounds, and permits at least two isolated
  episodes up to the configured ceiling;
- uses the SDK's low-level one-round call surface, exposes rather than auto-drives
  `InputRequiredResult`, and never retries `tools/call`; a timeout, transport failure,
  or ambiguous result poisons the session and closes its upstream connection;
- lets a poisoned worker close its own transport within a bounded grace period derived
  from `limits.tool_timeout_s` before cancelling it, because cancelling a connection
  that is already terminating is what orphans an upstream episode;
- maps only object `structuredContent` from an allowlisted result shape, validates
  declared output schemas, and returns infrastructure failures as bounded non-2xx
  `mcp_*` error envelopes;
- publishes the §9 effective digest and requires the operator to pin the reviewed
  `gateway_artifact_digest`; how a build system derives that artifact digest remains
  a packaging decision, not a runtime guess; and
- supports optional constant-time bearer authentication and requires TLS unless the
  CLI is explicitly placed in loopback-only debug mode.

Mode B remains blocked on a fingerprinted shim interface and namespace contract. Mode
C remains blocked on a pinned snapshot artifact and an enforceable read-only boundary.
The gateway fails startup for either mode instead of silently approximating reset or
state behavior. Mode A now supports the P4–P11 evidence handoff described below; Modes B
and C remain non-executable rather than inheriting Mode A's result.

### 10.2 Authoring intake boundary

Intake lives under `runtime/mcp/authoring/` and runs in the same isolated `bfcl-mcp`
environment as discovery, for a reason worth stating plainly: Data Designer requires MCP
SDK v1 and this integration requires v2, and the two extras are mutually exclusive. The
drafting phase therefore **cannot hold an MCP connection**. Everything it will ever know
about the server has to be written down first, which makes the evidence bundle a process
boundary rather than a convenience. A file can be diffed, digested, and approved; a live
connection cannot.

`scripts/build_mcp_intake.py` turns a reviewed `mcp_intake.yaml` into five things: a
sanitized evidence bundle, the three pack files intake can derive without a model
(`tools.json`, `manifest.yaml`, `endpoint_config.yaml`), the discovery report they came
from, the live gateway attestation whose digest the endpoint config pins, and a provenance
record. The declaration itself is deliberately small — pack
identity, which MCP profile to discover, and which gateway will serve the result — because
everything else is derived, and a derived value that was also declared is a value that can
disagree with itself.

Four properties carry the trust:

- `tools.json` is the normalized catalog copied through, not re-derived. Re-deriving it
  would risk publishing something the pinned `tool_catalog_digest` does not cover.
- `endpoint_config.yaml` pins its `expected.content_digest` from the same §9 identity
  function the gateway serves at `GET /v1/metadata`, so the pack and the gateway cannot
  drift into two separate calculations of the same digest.
- The attestation pin covers the document fetched from the live gateway after every identity
  component is compared with discovery. It is preserved beside the evidence bundle and
  digested in intake provenance; no L0 prediction stands in for a later L2 document.
- Every prose string reaching the bundle is scanned, not just the `function.description`
  that §7 already covers. `outputSchema` descriptions and `annotations` are injection
  surface for the drafting model even though BFCL never publishes them; text a reviewer
  cannot see blocks the whole draft, and merely suspicious text is flagged for a human and
  kept verbatim for them. Bundle text is tagged as data so the drafting phase cannot embed
  it without going through the quoting fence.

The bundle carries `L0` evidence and states the rest as explicit unknowns, each naming the
authoring decision it blocks: observed result shapes, observed error codes, state deltas,
confirmation behavior, fixture samples, and tool dependencies. A drafting model that needs
one of these must find an unknown and refuse rather than invent a plausible value. For the
same reason the bundle's status is never `approved`; approval is a human act recorded in
provenance, and the intake record sets `model` to null because nothing was inferred in this
phase. Fixtures, task templates, validation cases, and assertions are listed as pending
rather than emitted as stubs, since a stub turns a missing input into a file that looks
authored.

### 10.3 Drafting boundary

Drafting lives under `runtime/pack_authoring/` and reads one file: the evidence bundle. Two
gates stand in front of the model. The bundle digest is recomputed from the bytes on disk, so
a bundle edited after review is refused rather than drafted from. And an approval document
must name that exact digest and acknowledge every advisory finding by
`location:code` — a blanket approval would let a newly appearing flag ride along on an older
decision, which would make the review in §7 decorative.

Four calls run in dependency order: coverage plan, then validation cases, task templates, and
assertion specifications, each given the coverage plan as input rather than re-deriving it.
Every call goes through the same content-addressed cache the generation stages use, keyed on
model identity, prompt version, input, output schema, and seed, so a rerun reproduces the
approved draft and provenance records which model produced it.

What keeps the drafts honest is that the model is not allowed to answer questions the
evidence cannot support, and `grounding.py` enforces this after every call rather than
trusting the prompt:

- Tool and parameter names must exist in the bundle. A drafted call against a tool the server
  never advertised is rejected, not renamed.
- A literal argument value is legal only where the tool's own input schema pins the value set
  with an `enum` or a boolean type. Every other value names its source and waits for the
  probe that will supply it, because a plausible-looking identifier is the failure mode that
  looks most like progress: the pack would generate, pass, and test nothing.
- Each draft declares `blocked_on` against the bundle's open unknowns, and the check runs both
  ways. A success probe must admit it has not observed a result shape; a probe blocked on an
  unknown the bundle already resolved is equally wrong, because it would park work behind a
  gate that will never close.
- Model prose is scanned with the same §7 rules. The model is not the untrusted party, but it
  read untrusted text, and a bidi override copied out of a tool description into a policy
  string defeats review just as well as it did upstream.

A refused draft raises with the full list of violations and nothing retries. A retry loop
would reward whichever attempt happened to pass, which is precisely how an ungrounded claim
reaches a pack.

Compilation to `assertions.py` covers trace predicates only, and is all-or-nothing. BFCL
records which tools an episode called, so `tool_called`, `tool_not_called`, and
`tool_called_after` are checkable against the benchmark's own evidence. Whether a result field
exists or a collection grew is a claim about a server nobody probed, so it stays a
specification until L1 probes resolve it. A partially compiled `assertions.py` would be worse
than none: the pack would load, the suite would pass, and the coverage a reviewer believes
they have would not exist. Drafts are therefore written to a `drafts/` directory beside the
pack rather than into it, because a `task_templates.yaml` inside a pack directory is loaded by
the pipeline as though a human had authored it.

### 10.4 Trust spine

`GET /v1/conformance` exists, and so does the verification in front of it. The producer lives
in `runtime/mcp/gateway/conformance.py`, the verifier in
`runtime/benchmark_families/bfcl/conformance.py`, and both read one schema so a document
cannot be valid to write and invalid to read. The route serves canonical bytes rather than
re-encoded JSON, because the pinned digest covers exactly those bytes and a re-ordering
encoder on either side would break verification while changing nothing semantically.

Three digests must agree before anything publishes: the one the pack pinned at intake, the one
the live endpoint reports at `GET /v1/metadata`, and the one inside the attestation. They are
produced by different parties at different times, so agreement is the only available evidence
that the build being certified is the build answering calls. Intake fetches and preserves the
document the live gateway actually serves, verifies every identity component against discovery,
and pins that document's digest. It does not predict a discovery-only document: adding P4–P11
would necessarily change such a prediction and make the pack unable to move from L0 to L2.

The verifier never reads `level` as a verdict. It re-derives what the document has earned and
reports both, so `attested_level` and `effective_level` can disagree, and only `L2` with no
findings publishes. Two outcomes are distinguished on purpose. A **finding** is a
contradiction — a digest mismatch, a failed or skipped probe, a required check declared
inapplicable — and drops the document to `L0`. A **cap** is missing evidence rather than a
defect: no `server_content_digest` outside an immutable snapshot sandbox, or incomplete state
observability.
Caps lower `L2` to `L1`, which still blocks publication but says something different to a
reviewer, and both appear in the report as named reasons rather than as a silent downgrade.

The rule that matters most is that a gateway cannot certify itself. `locally_verified` asserts
that BFCL ran the conformance suite against that exact artifact digest, so the verifier requires
the exact probe and gateway-conformance report documents and independently hashes both. Repeating
the digests from the endpoint is not evidence. `signed_release` remains capped until a
cryptographic verifier backed by a configured trust root exists; matching an issuer name is not
a signature. Consequently a gateway serving a technically perfect `L2` document, with every
identity digest agreeing, still cannot publish on its own word.

Gold eligibility is enforced through the machinery that already exists rather than beside it.
Check `A1` `endpoint_conformance` joins `extra_checks`, and `derive_pack_tier` refuses gold for
any non-passing check, so there is no second gate to keep in step with the first. `A1` is
emitted for every endpoint pack. An endpoint with no attestation may be exercised in a smoke
run, but `A1` records `endpoint_attestation_missing` and caps it below publication; removing the
block from a generated pack is therefore not a Gold bypass. Local Python oracles do not receive
an endpoint check.

The evidence bootstrap is explicit. A discovery-only gateway starts at `L0`; BFCL validates a
provisional pack and writes `mcp_probe_report` from the live target observations. Separately,
`run_gateway_timeout_conformance` exercises the pinned gateway core against a controlled hanging
fixture and writes the P9 suite. The gateway is then restarted with both documents using
`--probe-report` and `--gateway-suite`. It refuses incomplete, unordered, or non-passing evidence.
The exact bound documents are available at `/v1/conformance/probe-report` and
`/v1/conformance/gateway-report`; the latter is sealed in the pack at
`.bfcl/conformance/gateway_conformance_report.json`.

Final validation does not trust the served probe report. It reruns P1–P8 and P10–P11 against
the live endpoint, reruns interleaved episode isolation, derives a new canonical report, loads
the sealed P9 gateway report, and supplies both to `A1`. The verifier hashes both, compares the
fresh probe list with the attestation, validates every P9 timeout/poisoning/cleanup property,
and only then permits effective `L2`.

### 10.5 Review boundary

`runtime/mcp/release/review.py` implements the first handoff boundary. The review packet is not
an informal summary: it is a canonical, content-addressed packet that pins the evidence
bundle, intake and draft provenance, live gateway attestation, validation report, reviewed MCP
profile, complete canonical pack fingerprint, and optional held-out policy. The canonical
manifest, tools, fixtures, templates, validation cases, assertion source, endpoint config, and
held-out document are embedded in the packet so accepting “semantics” refers to visible bytes,
not only to draft artifact names.
The packet exposes the exact tool descriptions and schemas, exclusions, mutation and
confirmation declarations, control mapping, fixture and held-out policy, model calls,
assumptions, validation failures, endpoint conformance verdict, observed oracle calls, and
before/after state deltas. Stable risk IDs make a newly appearing warning require a new
decision rather than inheriting an older blanket acceptance.

A packet is still written when evidence is incomplete, because failed review evidence is useful
to a reviewer. Its status is then `blocked`, with named reasons. Missing complete call or state
delta logs, unresolved drafting unknowns, uncompiled assertions, a non-Gold validation report,
or an endpoint below independently verified `L2` are blockers. This is intentionally stricter
than treating a review meeting as proof: a failed, stale, or incomplete P4–P11 report can generate
a packet explaining the gap but cannot acquire a freeze approval.

The call and state-delta evidence is read from an `mcp_observations` object in the oracle
validation report, carrying `calls`, `calls_complete`, `state_deltas`, and `state_deltas_complete`.
Oracle validation now writes this object for attested endpoint packs. Completeness is false if
any declared executable case raises, returns a non-object, or lacks its before/after state pair;
review therefore cannot silently treat a partial log as complete evidence.

Approval records a second, domain-level decision distinct from the earlier approval to let a
model read an intake bundle. The reviewer must name themselves, provide a timezone-qualified
timestamp, pin one exact review packet digest, acknowledge every risk ID, and explicitly accept
semantics, control mapping, descriptions and snapshots, held-out policy, assumptions, and
validation evidence. The approval has its own digest. A changed packet, omitted checklist item,
unknown acknowledgement, or blocked packet is refused. `scripts/build_mcp_review.py` and
`scripts/approve_mcp_review.py` expose these gates without adding another publication path.

The packet also refuses a canonical pack whose `endpoint_config.yaml` pins an effective content
digest other than the one discovery observed. Freeze asserts only the one property it owns, that
an MCP release is endpoint-backed rather than `backend.py`; every other pack requirement stays the
Gold Gate's decision, because a freeze that re-litigates those rules becomes a second gate free to
drift from the first.

`runtime/mcp/release/freeze.py` implements sealing and fingerprinting. Freeze accepts only a canonical
endpoint pack whose fingerprint is in an approved packet and whose approval covers exactly that
packet. It refuses symbolic links, special files, external declared pack paths, reserved release
paths, source drift during copying, and an existing destination. Files are opened with
`O_NOFOLLOW`, copied into a private staging directory, and renamed into place only after all
provenance has been written and the final fingerprint is stable. The resulting `pack/` includes
the reviewed MCP profile and canonical provenance records, is made read-only, and is accompanied
by `freeze_manifest.json`. The manifest pins the final pack fingerprint, effective-content,
conformance, catalog, review, approval, and lineage digests without introducing a self-referential
digest inside the pack. Because the manifest sits outside the fingerprinted tree, reopening a
release recomputes the lineage, review-packet, and approval digests from the sealed files instead
of trusting the manifest's own word for them.

Handoff and its re-validation live in `runtime/mcp/release/handoff.py`. The supplied BFCL config must
resolve to the frozen tree and exact final fingerprint. Handoff forces a new oracle validation
rather than accepting the same-process memoized verdict, requires freshly derived Gold plus an
independently verified publishable `L2`, verifies the freeze again, and then calls the existing
BFCL prepare/generate implementation. There is no MCP generator. Publication must produce
`benchmark.parquet`, `benchmark_raw.parquet`, and `run_manifest.json`, and all three are checked
before success is returned.

For origin lineage, freeze writes `provenance/mcp_lineage.json` inside the fingerprinted pack.
`origin_provenance.py` validates it against `endpoint_config.yaml` and projects only non-secret
origin fields into `run_manifest.json`: provider/profile/mode, frozen and pre-freeze pack
fingerprints, effective content, conformance, catalog, lineage, review-packet, and approval
digests. Reviewer identity, endpoint URL, headers, and credential values are not published. Eval
source verification re-derives that projection from the frozen pack and rejects a manifest whose
MCP provenance was removed or changed.

A lineage file is one copyable JSON document, so it is never believed on its own. Publication
accepts it only when it names this pack's own `pack_id` and version, and when the review packet
and approval sealed beside it recompute to the digests the lineage cites, the approval covers that
same packet, and the packet's approved pre-freeze fingerprint is the one the lineage records.
Lifting the record into an unrelated endpoint pack therefore fails rather than inheriting someone
else's approval. The claim is provenance, not a gate: a pack still cannot publish unless the Gold
Gate independently verifies the live endpoint at `L2`. To keep this layering intact,
`origin_provenance.py` reads these records structurally and imports nothing from the MCP release
path.

## 11. Deferred Extension Points

Each of these is a decision, not an omission. They are listed so a later reader can
tell the difference.

| Deferred | Why, and what would unblock it |
| --- | --- |
| Declared projection of non-object `structuredContent` | The refusal in §8 is about unreviewed guessing, not about the shape itself. A per-tool projection pinned in config and fingerprinted would be reviewable; it needs its own probe proving the projection is total over observed results. |
| MCP Tasks extension and `InputRequiredResult` | Both turn one BFCL call into a multi-round-trip exchange with server-side lifecycle, while `expected_trace` and `executable_replay` assume one call, one result. Needs a conversation-level contract first. |
| `listChanged`-driven catalog refresh | A catalog that can change mid-run cannot be pinned by digest. Would need a rule for what a mid-run change means for rows already generated. |
| Multi-server tool aggregation | MCP scopes name uniqueness per server, so aggregation needs a disambiguation and provenance scheme per tool, and `content_digest` would have to compose. |
| MCP resources and prompts as oracle truth | Neither is executable state, and a prompt from an untrusted server entering a benchmark is threat `TM-01` with fewer controls. |
| Sampling requests from the server | Would let the oracle call a model during certification, making the certified result model-dependent. |
| Multilingual detection of instruction-like descriptions | The lexicon in §7 is English only and is a review aid, not a control. Extending it language by language would grow a list that is always incomplete while implying coverage it does not have. A real version needs a classifier plus a statement of what it does and does not catch; until then the load-bearing controls are that descriptions are inert data and that invisible characters are refused outright. |
| Pre-deserialization enforcement of `max_response_bytes` | The limit is currently checked after the SDK has parsed a response, so it bounds what enters BFCL rather than what the transport will buffer. Enforcing it earlier means a byte-counting transport wrapper, which the SDK does not expose today; until then the child process or HTTP client remains the first line of defense. |
