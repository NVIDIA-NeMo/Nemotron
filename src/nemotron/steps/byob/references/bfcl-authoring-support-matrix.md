# BFCL assisted-authoring support matrix

This matrix is the support source for assisted authoring. “Supported” means the behavior has a
named executable test. “Experimental” is implemented and tested but remains behind a
fail-closed rollout flag. “Unimplemented” is a target, not usable behavior.

| Surface | Status | Boundary | Evidence |
| --- | --- | --- | --- |
| Manual Oracle Pack generation | supported | Unchanged by assisted authoring | [`test_bfcl_stages.py`](../../../../../tests/steps/byob/test_bfcl_stages.py) |
| `local_python` static identity | supported | Reviewed package and import closure | [`test_bfcl_local_authoring_adapter.py`](../../../../../tests/steps/byob/test_bfcl_local_authoring_adapter.py) |
| `local_python` A1/A2 probes | supported | Least-privilege process worker | [`test_bfcl_source_intake.py`](../../../../../tests/steps/byob/test_bfcl_source_intake.py) |
| `http_package` reviewed intake | supported | Reviewed schema plus live identity and attestation | [`test_bfcl_http_authoring_adapter.py`](../../../../../tests/steps/byob/test_bfcl_http_authoring_adapter.py) |
| `http_package` A1/A2 probes | supported | One endpoint session per episode; deadlines delete the session | [`test_bfcl_http_probe_certification.py`](../../../../../tests/steps/byob/test_bfcl_http_probe_certification.py) |
| MCP Mode A intake and gateway | experimental | `BFCL_ENABLE_MCP_MODE_A=1`; independent probes still required | [`test_bfcl_mcp_gateway.py`](../../../../../tests/steps/byob/test_bfcl_mcp_gateway.py) |
| MCP Mode A A1/A2 probes | experimental | `BFCL_ENABLE_MCP_MODE_A=1`; a reviewed probe plan is required, and Mode A is the only mode whose reset can be probed | [`test_bfcl_mcp_probe_certification.py`](../../../../../tests/steps/byob/test_bfcl_mcp_probe_certification.py) |
| MCP Mode B executable shim | unimplemented | Discovery records do not authorize execution | unimplemented |
| MCP Mode C executable snapshot | unimplemented | Snapshot records do not authorize execution | unimplemented |
| Candidate pack assembly from a local source | supported | The certified source tree is copied into the pack; every binding is proved against evidence | [`test_bfcl_authoring_pack_assembly.py`](../../../../../tests/steps/byob/test_bfcl_authoring_pack_assembly.py) |
| Candidate pack assembly from a session-backed source | supported | The pack points at the certified endpoint instead of carrying it; fixtures come from the reviewed probe plan | [`test_bfcl_mcp_pack_assembly.py`](../../../../../tests/steps/byob/test_bfcl_mcp_pack_assembly.py) |
| Shared evidence schema v2 | supported | Same trust envelope for all built-in adapters | [`test_bfcl_source_intake.py`](../../../../../tests/steps/byob/test_bfcl_source_intake.py) |
| Guided two-boundary CLI | supported | Model exposure and release approval remain distinct | [`test_bfcl_authoring_cli.py`](../../../../../tests/steps/byob/test_bfcl_authoring_cli.py) |
| A2 Gold freeze boundary | supported | A0/A1 may draft but cannot freeze as Gold | [`test_bfcl_authoring_release.py`](../../../../../tests/steps/byob/test_bfcl_authoring_release.py) |
| Local Python publication | supported | Fresh `stage=all` Gold verification | [`test_bfcl_authoring_e2e.py`](../../../../../tests/steps/byob/test_bfcl_authoring_e2e.py) |
| MCP Mode A publication | experimental | Publishable L2 plus fresh Gold | [`test_bfcl_authoring_e2e.py`](../../../../../tests/steps/byob/test_bfcl_authoring_e2e.py) |
| HTTP-package publication | refused | Freeze is supported; publication adapter is intentionally absent | [`test_bfcl_authoring_e2e.py`](../../../../../tests/steps/byob/test_bfcl_authoring_e2e.py) |
| Structured operational events | supported | Allowlisted digest/code payloads only | [`test_bfcl_authoring_events.py`](../../../../../tests/steps/byob/test_bfcl_authoring_events.py) |
| Environment and secret-manager credentials | supported | Values remain memory-only; authorization context is digest-bound | [`test_bfcl_authoring_credentials.py`](../../../../../tests/steps/byob/test_bfcl_authoring_credentials.py) |
| Authoring cache retention | supported | Reference-aware dry-run and reviewed execute | [`test_bfcl_authoring_retention.py`](../../../../../tests/steps/byob/test_bfcl_authoring_retention.py) |
| Release revoke and supersede | supported | Signed registry required at enforcement boundary | [`test_bfcl_release_revocation.py`](../../../../../tests/steps/byob/test_bfcl_release_revocation.py) |
| Dynamically installed third-party adapters | unimplemented | Built-in registry remains static | unimplemented |

Detailed MCP transport behavior is in
[bfcl-mcp-support-matrix.md](bfcl-mcp-support-matrix.md). Contract and operator
references are indexed in [bfcl-authoring-user-guide.md](bfcl-authoring-user-guide.md).
