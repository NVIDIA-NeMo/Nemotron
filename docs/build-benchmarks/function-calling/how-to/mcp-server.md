<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Onboard an MCP Server as the Oracle

Use this guide when the domain you want to benchmark is already exposed as a Model Context Protocol (MCP) server. The server becomes the executable oracle: BFCL discovers its tool catalog, exposes it through a gateway that speaks the BFCL Oracle HTTP v1 contract, certifies it from observed probes, and then carries a reviewed frozen pack into the same generation pipeline every other flow uses.

:::{warning}
This transport is experimental and disabled by default. Only Mode A, in which the server itself implements the reviewed describe, reset, state, and end controls, is implemented. Mode B and Mode C declarations are inert discovery records with no execution path. Read `src/nemotron/steps/byob/references/bfcl-mcp-threat-model.md` before you point BFCL at a server you do not control; it states the trust boundaries this flow assumes.
:::

This page is the walkthrough. `src/nemotron/steps/byob/references/bfcl-mcp-user-guide.md` is the matching command-level reference, with every invocation executed as a smoke case by the test suite; use it when you need exact arguments and refusal codes.

## Before You Start

- Confirm the server can implement the Mode A control tools. If it cannot, this flow has nothing to onboard.
- Install the isolated transport runtime below. The MCP protocol requires SDK major version 2, while the model-authoring dependencies may require version 1, so the two must not be resolved into one environment.
- Prepare a domain brief, a probe plan, and a certification key pair, exactly as in {doc}`assisted-authoring`.

```bash
uv sync --extra bfcl-mcp
export BFCL_ENABLE_MCP_MODE_A=1
```

`BFCL_ENABLE_EXPERIMENTAL_MCP` is accepted as a compatibility alias for one deprecation window. If both variables are set they must resolve to the same boolean, or startup fails with `rollout_settings_conflict`. The flag is required for live discovery, gateway startup, and MCP intake; offline review, verification, approval, and freeze do not need it.

## Step 1: Write `mcp_oracle.yaml`

Start from the normative schema in `src/nemotron/steps/byob/references/bfcl-mcp-oracle-contract.md`. In that file you select business tools explicitly under `tools.include` and alias them to stable BFCL names where the server's own names are unsuitable, declare mutation and confirmation behavior in the reviewed profile because BFCL never infers either from a tool name or a live result, and keep the control tools out of `tools.include`.

For a stdio transport, also create a host-owned trusted-executable policy that pins the executable's absolute path, its SHA-256, the exact allowed argument vectors, and the allowed working-directory roots. A server-supplied configuration cannot weaken that policy, and `PATH` lookup is not an authorization mechanism.

For a Streamable HTTP transport, use HTTPS outside explicit loopback debugging, reference credentials by environment-variable name only, and expect redirects, ambient proxy variables, and credential reflection to be refused.

## Step 2: Discover and Pin the Catalog

Discovery reads the server's identity and its complete paginated tool catalog and writes a deterministic report.

```bash
python -m nemotron.steps.byob.scripts.discover_mcp_oracle \
  --config mcp_oracle.yaml \
  --output mcp_discovery_report.json \
  --bootstrap-catalog-digest
```

`--config` and `--output` are required. `--bootstrap-catalog-digest` writes a report containing the observed digest instead of failing on the placeholder in your configuration; use it only for the first run. Add `--trusted-executables <POLICY_FILE>` for stdio, and `--allow-insecure-localhost` only for debug-only cleartext loopback, which is never publication-eligible.

Review the complete catalog and its exclusions, copy the observed digest into the expected identity in `mcp_oracle.yaml`, then run discovery again *without* `--bootstrap-catalog-digest`. Bootstrap output is pre-attestation evidence, not approval.

:::{important}
Never overwrite an expected catalog digest automatically. Compare the complete normalized catalog, the aliases, the exclusions, the schemas, the adapter version, and the server identity. A legitimate catalog change requires new evidence and a new review.
:::

## Step 3: Start the Gateway

The gateway is the only MCP execution boundary. It maps the MCP server onto the BFCL Oracle HTTP v1 contract, so that the generation pipeline drives sessions, calls, and state through the same routes it uses for any endpoint-backed pack and stays entirely unaware of MCP. A running gateway process is part of the fingerprinted execution environment.

```bash
python -m nemotron.steps.byob.scripts.run_mcp_gateway \
  --config mcp_oracle.yaml \
  --gateway-artifact-digest sha256:<digest> \
  --host 127.0.0.1 \
  --port 8765 \
  --allow-insecure-loopback
```

`--config` and `--gateway-artifact-digest` are required. `--host` defaults to `127.0.0.1` and `--port` to `8765`. Production bindings require TLS through `--tls-certfile` and `--tls-keyfile`; `--allow-insecure-loopback` is a debugging affordance for explicit loopback only. Client bearer authentication is configured by reference with `--client-bearer-token-env`, and `--max-request-bytes` bounds the accepted request size.

### Attestation levels

A gateway attests only what it has evidence for, and human approval never raises the attained level. `L0` is discovery only: the identity and catalog were verified and nothing was proven about execution behavior. `L1` adds a working control plane and a total result mapping, so episodes can be reset, called, and read; it is executable but not certifiable, which means it runs under `lineage.policy: smoke_no_publication` and cannot publish. `L2` requires a Mode A gateway with a complete ordered probe report and a passing build suite, and publication on top of that still requires the final BFCL validation to reproduce the target report.

The first gateway starts at `L0`. Use it to validate the provisional pack and retain `mcp_probe_report` from the resulting `oracle_validation_report.json`. The controlled-timeout suite is a separate artifact: it is produced by running the `P9` conformance helper, `run_gateway_timeout_conformance` in `src/nemotron/steps/byob/runtime/mcp/gateway/conformance.py`, against a controlled hanging fixture, and its returned document is what you write to `gateway_suite.json`. No CLI wraps that helper yet, so this step is driven through the Python API. Then restart the same pinned gateway artifact with both files supplied:

```bash
python -m nemotron.steps.byob.scripts.run_mcp_gateway \
  --config mcp_oracle.yaml \
  --gateway-artifact-digest sha256:<digest> \
  --probe-report mcp_probe_report.json \
  --gateway-suite gateway_suite.json \
  --host 127.0.0.1 \
  --port 8765 \
  --allow-insecure-loopback
```

Fetch `/v1/conformance`, `/v1/conformance/probe-report`, and `/v1/conformance/gateway-report`, pin the attestation digest in `endpoint_config.yaml`, and seal the gateway report. Final validation reruns the target probes rather than trusting the report the gateway served.

## Step 4: Run Intake and Certification

Intake turns the discovery evidence into a sanitized evidence bundle, a pack draft, and a signed certification report.

```bash
python -m nemotron.steps.byob.scripts.build_mcp_intake \
  --intake mcp_intake.yaml \
  --domain-brief /srv/sources/domain-brief.txt \
  --held-out-not-applicable-reason "The catalog is public reference data." \
  --held-out-reviewed-by reviewer@example.test \
  --certification-private-key /srv/bfcl/keys/certification-private.pem \
  --certification-key-id warehouse-authoring \
  --output /srv/bfcl/authoring/warehouse/intake \
  --probe-plan /srv/sources/probe-plan.json
```

`--intake`, `--domain-brief`, `--certification-private-key`, `--certification-key-id`, and `--output` are required, as is exactly one of `--held-out-policy` or `--held-out-not-applicable-reason` together with `--held-out-reviewed-by`. `--domain-brief-language` defaults to `en`. Add `--trusted-executables` for stdio and `--resolved-authoring-config` when a reviewed rollout policy supplies the adapter decision.

`--probe-plan` is optional, and omitting it certifies A0 only, exactly as for the other transports. Supply it to reach A1 or A2. The plan is accepted for Mode A alone, because Mode A is the only mode whose reset and state are control tools. A session-based plan must carry `fixtures`, since a session is handed its world when it opens rather than reading it from a reviewed file. The certification tiers are the same A0, A1, and A2 tiers described in {doc}`assisted-authoring`, and a Gold freeze requires A2.

Intake can also be delegated through the guided CLI, which binds its output into a
session for you. For the one-approval path, use the reviewed policy described in
{doc}`assisted-authoring`:

```bash
python -m nemotron.steps.byob.scripts.bfcl_author \
  --ci author \
  --workspace /srv/bfcl/authoring/warehouse \
  --source <REVIEWED_MCP_INTAKE> \
  --brief /srv/sources/domain-brief.txt \
  --adapter mcp_mode_a \
  --policy /srv/bfcl/policies/warehouse-authoring.yaml \
  --required-tier A2
```

Adapter-specific flags supplied after the guided flags are delegated to the intake command.

## Step 5: Continue Through the Guided CLI

From here the flow is identical to the conventional-source flow, and {doc}`assisted-authoring` documents each command in detail:

1. `answer` applies any digest-bound open questions.
2. For clean evidence, `apply-policy` derives model-exposure and evidence records from
   the reviewed policy; otherwise use the granular `authorize` and
   `approve --boundary evidence` fallback.
3. `draft` runs bounded, cached structured model calls, and `assemble` binds those drafts and the reviewed supplement into a candidate pack.
4. `review` builds the deterministic review packet.
5. `release` asks for the normal flow's one per-run semantic/risk confirmation, then
   records approval, seals the reviewed bytes, and reruns fresh Gold validation plus
   the generation pipeline.

`review` and `publish` accept `--adapter-kind`, which already defaults to `mcp_mode_a`. For an MCP source, the assembled pack names the certified endpoint, pinned to the identity and TLS bundle intake verified, and takes its fixtures from the reviewed probe plan its sessions were opened with. There is no backend file to copy, because a session-based source has no tree.

## Verify Success

- The second discovery run passes with the expected catalog digest pinned, not bootstrapped.
- The gateway serves a conformance report whose probes ran in order and whose build suite passed.
- The certification report records the tier you required, and fresh validation of the candidate pack reports `gold_eligible: true`.
- Publication wrote `run_manifest.json` as the commit marker.

## Common Failures

| Symptom | What it means |
| --- | --- |
| "MCP onboarding is experimental" | Set `BFCL_ENABLE_MCP_MODE_A=1` for live discovery, gateway startup, or intake. |
| SDK major mismatch | Run transport operations in the `bfcl-mcp` environment. The client refuses SDK majors other than 2. |
| Catalog digest mismatch | The normalized catalog changed. Compare it in full and collect new evidence; do not silently repin. |
| The endpoint stays at `L0` | The provisional validation did not produce the ordered probes, the controlled suite has no passing build observation, or both files were not supplied when the gateway restarted. A working tool call alone proves none of reset, isolation, state observability, confirmation safety, deterministic replay, or bounded failure. |
| Stdio executable rejected | The configured token, exact argument vector, working directory, executable digest, and host policy must all agree. |
| Reflected credential failure | The server echoed a credential in metadata, catalog text, an error, or a result. Fix the server, because BFCL will not persist the response. |

:::{note}
Do not convert a working Oracle HTTP pack merely to label it MCP-backed. MCP provenance is valid only when it is produced from MCP discovery, review, and freeze records, so a migration means exposing the controls, reviewing a new configuration, regenerating the pack from MCP evidence, revalidating, and freezing a new release while retaining the old pack as a separate lineage.
:::

## Next Steps

- Take the frozen pack to publication scale: {doc}`publish-a-release`.
- Compare the three authoring flows: {doc}`../explanation/authoring-flows`.
