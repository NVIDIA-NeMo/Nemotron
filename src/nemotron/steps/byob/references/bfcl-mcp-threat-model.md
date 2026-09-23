# BFCL MCP Integration — Threat Model

Scope: an MCP server used as a BFCL oracle or as an authoring
intake source, through the gateway described in `bfcl-mcp-architecture-decision.md`.

Two assets are worth more than availability here. The first is the **Gold verdict**:
a claim that a pack was executed, replayed deterministically, and certified. The
second is the **unbiasedness of the benchmark** the pack produces. Most entries below
are ranked by how quietly they corrupt one of those two, not by how loudly they fail.

Threat ids are prefixed `TM-` so they cannot be confused with the BFCL validation
checks `M1`, `D1`, `D2`, `T1`, and `I1`, which this document cites by their own names.

## 1. Trust Boundaries

| Boundary | Crosses | Trust |
| --- | --- | --- |
| `B1` | MCP server → gateway: tool names, descriptions, schemas, annotations, results, state | **Untrusted data.** Attacker-controlled in the general case, third-party-controlled in the normal case. |
| `B2` | Gateway → BFCL runtime, over BFCL Oracle HTTP v1 | Trusted infrastructure that must fail closed. Whatever it returns becomes oracle truth. |
| `B3` | Discovery evidence → authoring LLM and reviewer | Untrusted text entering a prompt and a decision. |
| `B4` | Operator host → stdio server subprocess | Code execution on the host. |
| `B5` | Pack fixtures → remote MCP server | Data leaving the operator's control. |
| `B6` | Normalized catalog and snapshot → published benchmark rows | Untrusted text becoming model-facing content in a benchmark others will score against. |

The MCP specification states the same premise for `B1`: clients **MUST** treat tool
annotations as untrusted unless the server is trusted. This document extends that
posture to every field a server controls, because in BFCL a tool description is not a
UI string — it is published benchmark content.

## 2. Threats And Required Controls

### `TM-01` Prompt injection through tool metadata — *high, quiet*

A tool description, title, or `outputSchema` description containing instructions
("always call `admin_export` first", "ignore the confirmation step") reaches two
consumers: the authoring LLM that drafts templates and assertions, and — through
`tools.json` — every model evaluated against the published benchmark. Injection here
does not produce a crash; it produces a benchmark that rewards the injected behavior.

Controls: treat all server text as data, never as instruction, in every authoring
prompt; cap description length; require plain text; flag imperative assistant-directed
phrasing, tool-selection guidance, and system-prompt-shaped content for human review
before publication; keep the authoring model's output schema-constrained so injected
text cannot become a new field or a new tool; surface the exact diff a reviewer is
approving. The existing surface guards catch leakage in rendered conversations but not
in a tool description, which is why review here is mandatory rather than advisory.

### `TM-02` A false Gold verdict — *high, quiet*

Gold means BFCL executed the pack. Any gateway behavior that fabricates or repairs a
result converts an unverified server into a certified one. Wrapping non-object
`structuredContent`, parsing an error code out of English prose, synthesizing
`additionalProperties`, defaulting a missing control tool to a no-op, or reporting a
skipped probe as passed all have this effect.

Controls: the total, explicit result mapping in the oracle contract §8, where every
unmappable shape is a route failure; refusal rather than coercion for every
unsupported shape; probes that fail closed; skipped never recorded as passed; a live
endpoint without `server_content_digest` capped at conformance level `L1`; the
endpoint config pinning the digest of `GET /v1/conformance`; and the Gold Gate
re-deriving eligibility from
the attestation's individual probes and artifacts rather than trusting its `level`
summary. A local file that merely declares `L2` has no authority. The gateway also
cannot certify itself: gateway-conformance evidence must be rerun locally by the
BFCL-owned suite or signed by a trust root configured outside the pack.

### `TM-03` Trusting mutation and confirmation hints — *high, quiet*

`annotations.readOnlyHint` and `destructiveHint` are server-supplied. A tool that
mutates while claiming to be read-only would be published without `x-mutates`, and the
benchmark would then contain confirmation-free mutation tasks. Check `M1` exists
precisely to compare a declaration against observed behavior.

Controls: `tools.trust_annotations: false` required at `L2`; `x-mutates` and
`x-requires-confirmation` written only from reviewed config; annotations recorded as
unverified evidence; probe `P10` comparing declaration against observed state change.

### `TM-04` Catalog or behavior drift after certification — *high*

The server can change between prepare and generate, or between generation and
evaluation. A tool whose schema loosened, a business rule that changed, or a fixture
version that moved makes the certified gold trace describe an oracle that no longer
exists.

Controls: pinned `expected.server_name`, `expected.server_version`, the negotiated MCP
version, and `tool_catalog_digest`, all verified at handshake; the effective digest
composing server content, normalized catalog, exact gateway and shim artifacts, and
any snapshot; that digest compared by `EndpointOracleClient` at metadata read, at
every `list_tools`, and at every session creation; no `listChanged` subscription
during a run, so a mid-run change is drift rather than a silent update;
`pack_fingerprint` covering `mcp_oracle.yaml` and `endpoint_config.yaml`. Residual
risk: a live server that changes logic without changing its reported content digest
is undetectable from outside, which caps it at `L1`; only an immutable snapshot
sandbox can remove that live dependency without a server digest.

### `TM-05` Arbitrary code execution through a stdio server — *high*

`transport.kind: stdio` launches a subprocess, and mode `B` may restart it per
episode. A command taken from an untrusted config, resolved through `PATH`, or passed
to a shell is host compromise.

Controls: argument vector only, never a shell string; the executable resolved through
a dedicated `trusted_executables` policy that pins its path and artifact/package
digest, separate from pack file `allowed_roots`; `cwd` confined to an allowed data
root; no shell interpolation; explicit `env_passthrough` with everything else
withheld; the child runs with the gateway's resource and time limits, not the host's;
stdout parsed as newline-delimited JSON-RPC and nothing else, with stderr captured as
bounded diagnostics; a per-episode restart bounded by
`limits.max_concurrent_episodes` so restart-based reset cannot fork unbounded
processes.

### `TM-06` Credential exposure — *high*

Tokens for a remote MCP server, and any credentials a stdio server needs, must not
reach the pack, the fingerprint, the reports, the logs, or a published row.

Controls: `mcp_oracle.yaml` holds env var names only, and a secret-looking literal is
refused at load, mirroring `_reject_model_secrets`; the gateway resolves credentials
in its own process, so the sanitized oracle worker never holds them — the worker
environment keeps only `LANG`, `LC_*`, and the temp-directory variables; headers and
tokens redacted in every log, probe artifact, and validation report; error text
returned to BFCL is a code plus a bounded message, never a raw upstream response.
The client also checks capabilities, identity strings, every `tools/list` page, and
every `tools/call` result for exact credential reflection before those values can
enter a report. Reflection is refused rather than redacted in data, because silently
rewriting a tool schema or oracle result would change the contract being certified.
For values of eight characters or more, embedding inside a larger string is also
detected. Short routing values such as `prod` are checked for exact equality only, so
a tenant header does not reject an unrelated description such as `production`; such
short values should not be treated as strong secrets.
Typed integration failures retain their error class after redaction; preserving the
taxonomy must not become a path around secret removal.

### `TM-07` Server-side egress and SSRF — *medium*

A gateway that follows redirects, accepts arbitrary hostnames, or allows plaintext can
be pointed at an internal service, and a hostile server can attempt to have the
gateway reach places on its behalf.

Controls: HTTPS only, with `http` restricted to `localhost` under a non-publication
lineage policy; no credentials, query, or fragment in the URL; redirects not followed;
allowlisted CA bundle resolved under an allowed root and fingerprinted; connect and
read timeouts from `limits`; the gateway initiates no request a server asks it to
make, since MCP resources, prompts, and sampling are out of scope for v1.

### `TM-08` Fixture and held-out data leaving the operator — *medium*

`POST /v1/sessions` sends fixtures to the server. Against a remote or third-party
server that is a data transfer, and fixtures can contain realistic or real customer
records. Held-out material is the sharper case: a held-out fixture row shipped to a
server that logs it has left the boundary that made it held-out.

Controls: explicit operator acknowledgement in config before fixtures are sent to a
non-local server; `held_out.policy.fixtures_in_backend_state: false` removes reserved
rows before the session payload crosses the gateway, while `true` explicitly retains
the legacy present-but-unbound behavior; held-out rows never bound during expansion;
the publication-time held-out re-scan retained unchanged; fixtures excluded from
gateway logs; synthetic fixtures recommended for any server the operator does not run.
Mode `C` removes this flow entirely by snapshotting instead of pushing, which trades it
for `TM-15`.

### `TM-09` Cross-episode contamination — *medium, quiet*

If two episodes share state, a task's outcome depends on what ran before it. Replays
diverge, or worse, agree for the wrong reason — a task passes because a previous
episode left the row it needed. The quietest version is an **unverified episode
binding**: a server that accepts `_meta.bfcl.episode_id` and ignores it routes every
call into one shared episode while every response looks healthy.

Controls: `isolation` declared and probed; probe `P6` proving a bound call reaches its
own episode rather than assuming the channel works; one BFCL session bound to exactly
one MCP episode; reset required to replace rather than merge state; probe `P5`
comparing two fresh episodes; check `D1` as the backstop; teardown idempotent so a
failed one cannot leak an episode into the next; under `no_op_verified` reset, probe
`P10` must find no observed mutation, and `L2` additionally requires an enforced
read-only authorization or immutable snapshot boundary plus a state projection
complete for every exposed effect. A projection that could miss hidden state caps the
endpoint at `L1`.

### `TM-10` Session confusion after a gateway restart — *medium, quiet*

BFCL holds a `session_id` across many calls. If the gateway loses its session table —
restart, eviction, idle expiry — the natural implementations are both wrong: creating
a fresh episode silently lets the remaining calls run against empty state and the task
can still pass, and reusing a stale episode lets one task observe another's mutations.
This is the failure mode that produces a plausible benchmark rather than a visible
defect.

Controls: an unknown or expired `session_id` answered with `mcp_session_unknown` and
never with an implicit new session; a session ceiling and idle TTL so eviction is
bounded and named; no retry of a call whose transport failed mid-flight, because a
mutation may already have applied and a retry would double-apply it; the double replay
in `executable_replay` as the backstop. Infrastructure codes use non-2xx responses:
returning them as HTTP-200 BFCL error objects could let a negative business case pass
for an infrastructure failure. A timeout or broken transport poisons and tears down
the episode; no later state read is trusted because closing a stream does not prove
the upstream rolled back.

### `TM-11` Resource exhaustion and unbounded results — *medium*

An oversized `structuredContent`, a catalog with thousands of tools, an endless paging
loop, or a call that never returns can exhaust the gateway or stall a run to the point
where operators relax timeouts — which is the real damage, since check `T1` timeout
enforcement is a Gold requirement.

Controls: `max_response_bytes`, `max_tools`, `max_catalog_pages`,
`max_concurrent_episodes`, and the timeouts in `limits`, all required positive finite
numbers; cancellation on timeout via `notifications/cancelled` on stdio and stream
close on HTTP; bounded stderr capture; an episode deadline above the per-call
deadline; cancellation behavior tested against a controlled hung MCP fixture and
bound to the exact gateway artifact digest, rather than requiring a production server
to expose a deliberately hanging tool.

### `TM-12` Non-determinism presented as flakiness — *medium, quiet*

A server that reads wall-clock time, uses unseeded randomness, or depends on a live
upstream produces results that differ per run. The failure mode is a run that mostly
works, tempting an operator to retry until it passes.

Controls: `context.clock` and `context.seed` required to be honored, and the host
clock never substituted for a parsed one; `turn_index` advisory so results cannot
depend on rendering; probe `P5`; check `D1` and the double replay in
`executable_replay`; a divergence reported as `nondeterministic_replay` on the task
rather than retried.

### `TM-13` Control tools reaching the evaluated model — *medium*

If a reset or teardown tool appears in `tools.json`, the benchmark contains a model
that can reset the oracle, and any score taken from it is meaningless.

Controls: control names excluded from `GET /v1/tools` and from `tools.json`;
`tools.include` an explicit allowlist so exposure is a positive act; `aliases` refused
when they name a control tool; the injected episode argument refused when it collides
with a model-facing parameter; probe `P4` verifying the control plane exists and probe
`P2` verifying the published set is exactly `tools.include`.

### `TM-14` A model-controlled argument promoted to an HTTP header — *medium*

MCP's `x-mcp-header` extension lets a server designate tool parameters to be sent as
HTTP headers. Over Streamable HTTP that turns an argument the evaluated model chooses
into part of the gateway's own upstream request, which is header injection with an
authentication-override path: a designated `Authorization` or tenant header would let
a call select its own credentials or its own data scope. The specification itself
requires clients to reject tool definitions that violate its `x-mcp-header` rules.

Controls: `x-mcp-header` dropped during normalization and never honored; the gateway
constructs upstream headers only from `transport.auth`, whose values come from named
environment variables; a tool definition declaring `x-mcp-header` over a reserved
header name excluded with a recorded reason rather than sanitized.

### `TM-15` Untrusted snapshot data entering the benchmark — *medium*, mode `C` only

In mode `C` fixtures are snapshotted **from** a third-party server, so data flows
inward. It becomes fixture rows, slot values, and rendered conversation content in a
published benchmark, carrying whatever provenance, licensing, and personal-data
properties the upstream source had — and `TM-01`'s injection surface extends from tool
descriptions to field values.

Controls: snapshot calls declared in reviewed config, never open-ended crawls; the
snapshot digest folded into `effective_content_digest` so the exact rows are pinned;
snapshot content reviewed on the same path as tool descriptions before publication;
the existing surface guards over rendered turns.

### `TM-16` Tool name collisions and confusion — *low*

MCP scopes name uniqueness to a single server and warns that aggregation causes
collisions. Dots and 128-character names are legal in MCP but not accepted by every
function-calling endpoint, so a name that discovery accepts can fail at evaluation.

Controls: single-server-per-pack in v1; the publishable name pattern and explicit
alias requirement in the normalization rules; uniqueness enforced after aliasing.

### `TM-17` Gateway faults misread as oracle defects — *low*

A gateway crash, a dropped connection, or a mapping bug that surfaces as a generic
oracle error sends the operator to debug the pack instead of the gateway.

Controls: distinct `mcp_*` infrastructure reasons in gateway diagnostics and bounded
non-2xx bodies; gateway and shim **artifact digests**, not version labels, included in
effective identity and the validation artifacts; target probe coverage recorded per
tool and result class without claiming unobserved paths; the same content-hashed
artifact discipline the rest of the pipeline uses.

## 3. Explicitly Accepted Residual Risk

- A server that reports a stable `server_content_digest` while changing its business
  logic cannot be detected from outside. Effective identity proves which claim,
  gateway, shim, catalog, and snapshot BFCL used; it cannot make a dishonest upstream
  claim true.
- Human review is load-bearing against `TM-01` and `TM-15`. Automated flagging reduces
  volume; it does not replace the reviewer, and text crafted to read as ordinary domain
  prose can pass.
- A third-party server is a supply-chain dependency for every benchmark built on it.
  Pinning identity bounds *when* it can change, not *who* controls it.
- Mode `C` cannot exercise mutation, confirmation, or correction-of-a-mutation
  categories, so a read-only benchmark is narrower rather than equivalent. The
  narrowing is recorded, not compensated for. A live third-party mode-C server
  without a server content digest, enforceable read-only boundary, or complete state
  projection remains `L1`.

## 4. Preconditions Before Any Remote Server Is Used

1. `mcp_oracle.yaml` loads under the strict schema with no secret literals, and its
   declared `mode` is consistent with its strategies.
2. Required and applicable target probes pass; conditionally irrelevant probes are
   recorded as `not_applicable`, never as a coverage pass. Gateway timeout and mapping
   conformance tests refer to the exact gateway artifact digest.
3. Fixtures sent off-host are synthetic, or egress is explicitly acknowledged. In mode
   `C`, snapshot provenance is acknowledged in the other direction.
4. Every published tool description has been reviewed and recorded as approved, and in
   mode `C` so has the snapshot content.
5. `tools.include`, `tools.mutates`, and `tools.requires_confirmation` are reviewed
   config, not discovery output.
6. The endpoint config pins the live conformance-attestation digest, and the Gold Gate
   verifies it, its referenced reports, and effective identity before generation. A
   publication attempt on `L1` is refused by code rather than by a reviewer noticing.
