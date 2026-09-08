# BFCL MCP onboarding guide

This guide is for operators onboarding an MCP server and for server authors implementing the
BFCL MCP Oracle Profile. The path is feature-flagged; Mode A can publish only after fresh,
independent `P4`–`P11` evidence passes. Those probes and the conformance levels they
establish are defined in
[bfcl-mcp-oracle-contract.md](bfcl-mcp-oracle-contract.md), and which of them are supported
today is recorded in [bfcl-mcp-support-matrix.md](bfcl-mcp-support-matrix.md).
For the shared local/HTTP/MCP workflow, use
[bfcl-authoring-user-guide.md](bfcl-authoring-user-guide.md).

This file is the command-level reference, with every invocation below executed as a smoke
case by the test suite. For the narrative walkthrough, read
[the MCP onboarding task guide](../../../../../docs/build-benchmarks/function-calling/how-to/mcp-server.md)
first and return here for exact arguments and refusal codes.

## 1. Choose an operating mode

- Use **Mode A** when the server can implement the reviewed describe/reset/state/end controls.
- Use **Mode B** only with a separately reviewed and fingerprinted shim.
- Use **Mode C** only for an immutable, read-only snapshot.

Mode A is the only executable gateway mode currently implemented
([`test_bfcl_mcp_gateway.py`](../../../../../tests/steps/byob/test_bfcl_mcp_gateway.py)).
Mode B and C execution is **unimplemented**; their declarations are inert discovery records.

## 2. Install the isolated transport runtime

The MCP 2026 protocol requires SDK v2, while model-authoring dependencies may require SDK v1.
Do not resolve them into one environment.

This environment-setup template is not an executable documentation smoke:

```text
uv sync --extra bfcl-mcp
export BFCL_ENABLE_MCP_MODE_A=1
```

`BFCL_ENABLE_EXPERIMENTAL_MCP` is accepted as a compatibility alias for one
deprecation window. If both variables are set, they must resolve to the same boolean.

For stdio, create a host-owned trusted-executable policy that pins the executable's absolute
path, SHA-256, exact allowed argument vectors, and allowed working-directory roots. The
server-supplied config cannot weaken this policy.

For Streamable HTTP:

- use HTTPS outside explicit loopback debugging;
- reference credentials by environment-variable name;
- do not put bearer tokens or header values in YAML;
- expect redirects, ambient proxy variables, and credential reflection to be refused.

## 3. Author `mcp_oracle.yaml`

Start from the normative schema in
[bfcl-mcp-oracle-contract.md](bfcl-mcp-oracle-contract.md). Select business tools explicitly,
alias them to stable BFCL names where needed, and declare mutation and confirmation behavior in
the reviewed profile. Control tools must not appear in `tools.include`.

Inspect the discovery CLI:

<!-- doc-smoke: mcp-discovery-help -->
```shell
python -m nemotron.steps.byob.scripts.discover_mcp_oracle --help
```

The first discovery run may bootstrap the catalog digest. This is an operator template:

```text
python -m nemotron.steps.byob.scripts.discover_mcp_oracle \
  --config mcp_oracle.yaml \
  --output mcp_discovery_report.json \
  --bootstrap-catalog-digest
```

Review the complete paginated catalog and exclusions, copy the observed digest into the expected
identity, then run discovery again without bootstrap. Bootstrap output is pre-L0 evidence and is
not approval.

## 4. Run the gateway

The gateway is the only MCP execution boundary. It exposes BFCL Oracle HTTP v1 and keeps
`generate_bfcl` unaware of MCP. Inspect its arguments:

<!-- doc-smoke: mcp-gateway-help -->
```shell
python -m nemotron.steps.byob.scripts.run_mcp_gateway --help
```

The live-server command below is an operator template:

```text
python -m nemotron.steps.byob.scripts.run_mcp_gateway \
  --config mcp_oracle.yaml \
  --gateway-artifact-digest sha256:<digest> \
  --host 127.0.0.1 \
  --port 8765 \
  --allow-insecure-loopback
```

Production bindings require TLS. Client bearer authentication is configured by environment
reference. A gateway process must be treated as part of the fingerprinted execution environment.

The first gateway starts at L0. Use it to validate the provisional pack and retain
`mcp_probe_report` from `oracle_validation_report.json`. Run the BFCL-owned controlled hanging
fixture through `run_gateway_timeout_conformance`
(`../runtime/mcp/gateway/conformance.py`) and write its returned suite to
`gateway_suite.json`. No CLI wraps that helper yet, so drive it through the Python API; see
[`test_bfcl_mcp_gateway.py`](../../../../../tests/steps/byob/test_bfcl_mcp_gateway.py) for a
worked invocation. Restart the same pinned gateway artifact with this operator template:

```text
python -m nemotron.steps.byob.scripts.run_mcp_gateway \
  --config mcp_oracle.yaml \
  --gateway-artifact-digest sha256:<digest> \
  --probe-report mcp_probe_report.json \
  --gateway-suite gateway_suite.json \
  --host 127.0.0.1 \
  --port 8765 \
  --allow-insecure-loopback
```

Fetch `/v1/conformance`, `/v1/conformance/probe-report`, and
`/v1/conformance/gateway-report`. Pin the attestation digest in `endpoint_config.yaml` and seal
the gateway report as `.bfcl/conformance/gateway_conformance_report.json`. Final validation
reruns the target probes; it does not trust the served probe report.

## 5. Author, review, and freeze

The artifact sequence is:

```text
discovery report
  -> sanitized evidence bundle + intake provenance
  -> evidence-bound model draft + draft provenance
  -> canonical Oracle Pack
  -> fresh validation evidence
  -> deterministic review packet
  -> named checklist approval
  -> immutable frozen release
  -> fresh BFCL prepare
  -> existing BFCL generation
```

Each transition pins exact digests. Editing any upstream artifact requires rebuilding the
downstream packet and approval. Reviewer identity remains in the frozen audit record but is not
published in `run_manifest.json`.

## 6. Publication status

A discovery-only gateway truthfully attains `L0`. A gateway whose control plane works and whose
result mapping is total attains `L1`, which is executable but restricted to
`lineage.policy: smoke_no_publication`. A Mode-A gateway with a complete ordered probe
report and a passing P9 build suite may attest `L2`, but publication still requires final BFCL
validation to reproduce the target report, verify both evidence digests, produce complete call
and state-delta logs, pass review, and freeze the exact pack. Do not bypass this by editing an
attestation, validation report, review packet, or freeze manifest.

## Server-author checklist

A cooperative server must:

1. Return a stable implementation identity and complete paginated tool catalog.
2. Return JSON-object `structuredContent` for selected tools.
3. Implement deterministic describe/reset/state/end controls.
4. Isolate episodes so one cannot observe or mutate another.
5. Accept a frozen clock, seed, task ID, timeout, and fixtures at reset.
6. Return stable structured error codes.
7. Leave state unchanged when a confirmation-gated mutation is not confirmed.
8. Bound cancellation, timeout, crash, malformed input, and oversized output behavior.
9. Keep the catalog and effective content identity stable for the complete run.
10. Never interpret benchmark fixtures, descriptions, or model text as host instructions.

## Troubleshooting

### “MCP onboarding is experimental”

Set `BFCL_ENABLE_MCP_MODE_A=1` for live discovery, gateway startup, or MCP intake.
Offline review and verification do not require the flag. `BFCL_ENABLE_EXPERIMENTAL_MCP` is
accepted as a deprecated alias; see
[the rollout policy](bfcl-authoring-rollout.md) for how the two interact.

### SDK major mismatch

Run transport operations in the `bfcl-mcp` environment. The client intentionally refuses SDK
majors other than 2.

### Catalog digest mismatch

Do not overwrite the expected digest automatically. Compare the complete normalized catalog,
aliases, exclusions, schemas, adapter version, and server identity. A legitimate change requires
new evidence and review.

### Endpoint remains L0

Check that the provisional validation produced P1–P11 in order, the controlled gateway suite
contains a passing P9 observation, and both files were supplied when the gateway restarted.
A working tool call alone is not proof of reset, isolation, state observability, confirmation
safety, deterministic replay, or bounded failures.

### Stdio executable rejected

The configured executable token, exact argv, cwd, executable digest, and host policy must all
agree. PATH lookup is not an authorization mechanism.

### HTTP authentication or reflected credential failure

Check that referenced environment variables exist and contain no line breaks. If the server
echoes a credential in metadata, catalog text, errors, or results, fix the server; BFCL will not
persist the response.

## Migrating an existing endpoint pack

Do not convert a working Oracle HTTP pack merely to label it MCP-backed. MCP provenance is valid
only when produced from MCP discovery, review, and freeze records. To migrate:

1. expose or shim the BFCL MCP controls;
2. create and review `mcp_oracle.yaml`;
3. discover and pin the complete catalog;
4. regenerate the canonical endpoint pack from MCP evidence;
5. re-run validation and domain review;
6. freeze a new release; and
7. retain the old endpoint pack as a separate lineage.

See [bfcl-mcp-support-matrix.md](bfcl-mcp-support-matrix.md) for supported and deferred behavior.
