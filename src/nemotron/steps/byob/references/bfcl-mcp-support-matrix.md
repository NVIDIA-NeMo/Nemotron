# BFCL MCP support matrix

This page states what the MCP onboarding path can prove today. It is intentionally stricter
than a feature list: an implemented transport does not make an endpoint Gold-eligible.

## Rollout status

The MCP path is **experimental and feature-flagged**. Discovery, deterministic catalog
normalization, the MCP-to-BFCL gateway, evidence-bound authoring, P4–P11 evidence, review, freeze,
and publication handoff exist for Mode A. A discovery-only gateway still attests `L0`; only a
gateway restarted with strict BFCL-produced target evidence and a passing controlled P9 suite can
attest `L2`, and the existing BFCL Gold Gate independently reruns and verifies that evidence.

Live discovery, gateway startup, and intake are disabled by default. Set
`BFCL_ENABLE_MCP_MODE_A=1`. `BFCL_ENABLE_EXPERIMENTAL_MCP` remains an alias for one
deprecation window; if both variables are present they must agree. Omitted, false,
misspelled, unknown-kind, and conflicting settings fail closed. Offline review and
verification of already-produced artifacts do not need a rollout flag.

Do not describe an MCP pack as Gold merely because it has:

- an endpoint conformance document;
- a completed human review;
- an immutable frozen release; or
- a gateway that can execute tools.

Those are necessary evidence, not substitutes for fresh BFCL validation.

## Capability matrix

Every implemented/refused claim names its executable evidence. Broader adapter status is
in [bfcl-authoring-support-matrix.md](bfcl-authoring-support-matrix.md).

| Capability | Current status | Gold requirement or boundary | Evidence |
| --- | --- | --- | --- |
| Strict `mcp_oracle.yaml` | implemented, experimental | Unknown keys and literal credentials are rejected. | [`test_bfcl_mcp_config.py`](../../../../../tests/steps/byob/test_bfcl_mcp_config.py) |
| stdio transport | implemented, experimental | Host-owned executable path, digest, argv, and cwd policy are mandatory. | [`test_bfcl_mcp_transport_integration.py`](../../../../../tests/steps/byob/test_bfcl_mcp_transport_integration.py) |
| Streamable HTTP transport | implemented, experimental | HTTPS is mandatory except explicit loopback debugging; redirects and ambient proxy trust are disabled. | [`test_bfcl_mcp_transport_integration.py`](../../../../../tests/steps/byob/test_bfcl_mcp_transport_integration.py) |
| MCP protocol | implemented, experimental | SDK v2 runtime and profile `2026-07-28`; other SDK majors are refused. | [`test_bfcl_mcp_discovery.py`](../../../../../tests/steps/byob/test_bfcl_mcp_discovery.py) |
| Tool discovery | implemented, experimental | All pages must be consumed within configured limits. | [`test_bfcl_mcp_discovery.py`](../../../../../tests/steps/byob/test_bfcl_mcp_discovery.py) |
| Catalog normalization | implemented, experimental | Unsupported or ambiguous schemas are excluded; no guessing. | [`test_bfcl_mcp_discovery.py`](../../../../../tests/steps/byob/test_bfcl_mcp_discovery.py) |
| Business/control separation | implemented, experimental | Control tools are never model-facing. | [`test_bfcl_mcp_gateway.py`](../../../../../tests/steps/byob/test_bfcl_mcp_gateway.py) |
| Synchronous tool calls | implemented, experimental | One request produces one bounded result. | [`test_bfcl_mcp_gateway.py`](../../../../../tests/steps/byob/test_bfcl_mcp_gateway.py) |
| JSON-object `structuredContent` | implemented, experimental | Non-object projection is unimplemented. | [`test_bfcl_mcp_gateway.py`](../../../../../tests/steps/byob/test_bfcl_mcp_gateway.py) |
| MCP Tasks / claimed results | refused | A conversation-level contract is required first. | [`test_bfcl_mcp_gateway.py`](../../../../../tests/steps/byob/test_bfcl_mcp_gateway.py) |
| `InputRequiredResult` | refused | BFCL v1 cannot authorize hidden extra MCP exchanges. | [`test_bfcl_mcp_gateway.py`](../../../../../tests/steps/byob/test_bfcl_mcp_gateway.py) |
| Reset/state/end control mapping | implemented, experimental | P4/P5 prove lifecycle and deterministic fresh replay. | [`test_bfcl_mcp_target_probes.py`](../../../../../tests/steps/byob/test_bfcl_mcp_target_probes.py) |
| Process/namespace isolation | implemented, experimental | P6 interleaves live sessions and checks state separation. | [`test_bfcl_mcp_gateway.py`](../../../../../tests/steps/byob/test_bfcl_mcp_gateway.py) |
| Confirmation boundary | implemented, experimental | P8 proves unconfirmed mutation leaves state unchanged. | [`test_bfcl_mcp_gateway.py`](../../../../../tests/steps/byob/test_bfcl_mcp_gateway.py) |
| Resources and prompts as truth | refused | They are not Oracle Pack truth sources in this profile. | [`test_bfcl_mcp_discovery.py`](../../../../../tests/steps/byob/test_bfcl_mcp_discovery.py) |
| Dynamic `listChanged` catalog | refused | Catalog identity remains pinned through publication. | [`test_bfcl_mcp_discovery.py`](../../../../../tests/steps/byob/test_bfcl_mcp_discovery.py) |
| Evidence-bound model drafting | implemented, experimental | Compilation, validation, and review remain authoritative. | [`test_bfcl_mcp_authoring.py`](../../../../../tests/steps/byob/test_bfcl_mcp_authoring.py) |
| Review and immutable freeze | implemented, experimental | Approval cannot raise conformance level. | [`test_bfcl_authoring_release.py`](../../../../../tests/steps/byob/test_bfcl_authoring_release.py) |
| Mode A publication handoff | implemented, experimental | Fresh Gold plus independently verified publishable L2. | [`test_bfcl_authoring_e2e.py`](../../../../../tests/steps/byob/test_bfcl_authoring_e2e.py) |
| Mode B executable shim | unimplemented | Discovery shape does not authorize execution. | unimplemented |
| Mode C executable snapshot | unimplemented | Snapshot shape does not authorize execution. | unimplemented |

## Supported operating modes

- **Mode A — cooperative server:** implemented and experimental; server exposes reviewed
  describe/reset/state/end controls
  ([`test_bfcl_mcp_gateway.py`](../../../../../tests/steps/byob/test_bfcl_mcp_gateway.py)).
- **Mode B — shimmed server:** **unimplemented** executable target.
- **Mode C — immutable read-only snapshot:** **unimplemented** executable target.

Mode is not a quality tier. `L0`, `L1`, and `L2` describe independently attained conformance:

- `L0`: discovery identity is reproducible.
- `L1`: executable behavior has passed the P4 boundary, but is not publishable.
- `L2`: reset, state, isolation, mutation, confirmation, replay, and failure handling have all
  passed the applicable independent probes and the endpoint may proceed to the Gold Gate.

## Deferred extensions

The following are explicit deferrals, not silently supported behavior:

- non-object `structuredContent` projection;
- MCP Tasks and elicitation/input-required exchanges;
- mid-run catalog refresh;
- trust in server annotations without an explicit profile decision;
- signed-release `L2` without a cryptographic verifier; and
- publication from a server that cannot expose deterministic state and episode isolation.

The normative contracts are
[bfcl-mcp-oracle-contract.md](bfcl-mcp-oracle-contract.md),
[bfcl-mcp-architecture-decision.md](bfcl-mcp-architecture-decision.md), and
[bfcl-mcp-threat-model.md](bfcl-mcp-threat-model.md).
