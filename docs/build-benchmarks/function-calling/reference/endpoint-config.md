<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Endpoint Configuration

`endpoint_config.yaml` pins the reported identity and conformance bytes of a BFCL
Oracle HTTP v1 service used as the executable oracle. It replaces `backend.py`; a pack
must never declare both.

The configuration stores identity and credential references, not secret values.

## Create An Endpoint Configuration

Create the complete endpoint-backed pack shape with:

```bash
python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain my_domain \
  --target /srv/bfcl/packs/my_domain \
  --transport endpoint \
  --language en \
  --version 0.1.0
```

The generated URL and digests are placeholders. Replace them with metadata from a
deployed, versioned service before validation. A missing conformance attestation may be
useful for smoke diagnostics but cannot reach Gold.

For MCP-backed authoring, the gateway artifact emitter can produce endpoint identity,
attestation, and an optional CA bundle. Do not manually invent those values.

## Top-Level Fields

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `protocol_version` | Yes | None | Must be `bfcl-oracle-http-v1`. |
| `base_url` | Yes | None | HTTPS service origin and optional base path. |
| `expected` | Yes | None | Pinned oracle and authorization identity. |
| `auth` | Optional | `{}` | Environment or secret-manager references. |
| `attestation` | Required for Gold | None | Pinned endpoint-conformance evidence. |
| `tls` | Optional | `{}` | Optional allowlisted CA bundle. |
| `max_request_bytes` | Optional | 10 MiB | Positive request-size limit. |
| `max_response_bytes` | Optional | 10 MiB | Positive response-size limit. |

Unknown fields are refused. `base_url` cannot contain user information, a query, or a
fragment. Redirects are not followed, and mutating requests are not retried.

## Expected Identity

```yaml
expected:
  oracle_id: my_oracle
  oracle_version: "1.0.0"
  content_digest: sha256:<64-hex-characters>
```

`oracle_id`, `oracle_version`, and `content_digest` are required. The digest must use
the `sha256:` prefix.

Authenticated endpoints can additionally pin:

```yaml
expected:
  principal_digest: sha256:<64-hex-characters>
  permission_digest: sha256:<64-hex-characters>
  authorization_context_digest: sha256:<64-hex-characters>
```

When an authorization-context digest is configured with credentials, the principal
and permission digests are also required. Assisted HTTP source intake requires all
three authorization commitments when auth is present.

Live metadata and every newly created session must report the pinned identity exactly.
A changed id, version, content digest, or authorization context stops the run.

## Authentication And TLS

Reference a bearer token by environment-variable name:

```yaml
auth:
  bearer_token_env: BFCL_ORACLE_TOKEN
  headers:
    X-Tenant-ID: BFCL_ORACLE_TENANT
```

Header values are credential references, not literal secrets. `Authorization`, `Host`,
and `Content-Length` cannot be configured as custom headers. A structured
`bearer_token_ref` can select an environment or secret-manager resolver; do not declare
it together with `bearer_token_env`.

The pack does not authorize access to those credentials. Before ambient process
credentials can be resolved, the operator must set `BFCL_ENDPOINT_CREDENTIAL_POLICY`
to a JSON policy file that grants exact credential references to an exact HTTPS
destination:

```json
{
  "schema_version": "bfcl-endpoint-credential-policy-v1",
  "grants": [
    {
      "base_url": "https://oracle.example/v1",
      "credential_references": ["BFCL_ORACLE_TOKEN", "BFCL_ORACLE_TENANT"]
    }
  ]
}
```

The policy is operator-owned and is never read from the pack. A missing policy,
different destination, or unlisted reference fails before any secret is resolved.
Code that passes an explicit credential mapping or resolver creates the equivalent
one-call scope for that exact loaded endpoint; it should contain only credentials
the caller intends to expose to that invocation.

An optional CA bundle is pack-relative or absolute and must resolve under an allowed
root:

```yaml
tls:
  ca_bundle_path: certificates/oracle-ca.pem
```

The CA file participates in the pack fingerprint.

## Conformance Attestation

Gold requires:

```yaml
attestation:
  kind: bfcl-endpoint-conformance-v1
  expected_digest: sha256:<64-hex-characters>
```

Pin the digest derived from the reviewed `/v1/conformance` response. Preparation
cross-checks that report and its digest against the configuration. Because the service
reports its own identity and conformance document, pinning detects later byte changes
but is not by itself an independent trust source or proof that the service is
immutable. The attestation digest is distinct from the oracle content digest.

## Minimal Configuration Shape

```yaml
protocol_version: bfcl-oracle-http-v1
base_url: https://oracle.example.invalid
expected:
  oracle_id: my_oracle
  oracle_version: "1.0.0"
  content_digest: sha256:<64-hex-characters>
attestation:
  kind: bfcl-endpoint-conformance-v1
  expected_digest: sha256:<64-hex-characters>
max_request_bytes: 10485760
max_response_bytes: 10485760
```

This shows the required shape only. It is not runnable until the host and digests come
from the reviewed service deployment.

## HTTP V1 Routes

The service implements:

- `GET /v1/metadata`
- `GET /v1/tools`
- `GET /v1/conformance`
- `POST /v1/sessions`
- `POST /v1/sessions/{id}/calls`
- `GET /v1/sessions/{id}/state`
- `DELETE /v1/sessions/{id}`

Creating a session resets an isolated episode with frozen context and fixtures. Each
replay uses a new opaque session id.

## Validate The Endpoint

There is no standalone endpoint-config validator. Run whole-pack preparation:

```bash
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config /srv/bfcl/packs/my_domain/validate.yaml \
  --output-dir /tmp/bfcl-my-domain-validation
```

Preparation parses the strict config, resolves credentials and CA policy, checks live
metadata and tool names, creates isolated sessions, exercises reset/call/state/delete,
and verifies the pinned conformance attestation.

## Common Failures

- **`endpoint_contract_invalid`:** invalid config shape or HTTP route behavior.
- **HTTPS error:** use an HTTPS base URL without credentials, query, or fragment.
- **Unknown key:** remove fields outside the strict contract.
- **`endpoint_identity_changed`:** repin only after reviewing and versioning the changed
  service.
- **`endpoint_attestation_missing`:** produce and pin conformance evidence.
- **Authorization digest mismatch:** regenerate commitments from the reviewed credential
  references, principal, and permissions.
- **Missing credential environment variable:** configure it in the runtime environment,
  never in the pack.
- **Missing CA bundle:** restore the fingerprinted file under an allowed root.

## Assisted-Authoring Limitation

An `http_package` source can currently reach intake, drafting, review, and freeze.
Publication is deliberately refused until an independently verified publication
adapter exists. This limitation does not apply automatically to every MCP gateway;
consult the relevant transport support matrix.

## Complete Example

See `src/nemotron/steps/byob/references/bfcl-endpoint-config.example.yaml`. Replace
every digest placeholder and add a valid conformance attestation for Gold.

## Related Information

- {doc}`manifest` for selecting the endpoint transport.
- {doc}`tools-and-fixtures` for the catalog and reset records.
- {doc}`../how-to/mcp-server` for MCP gateway onboarding.
- {doc}`../how-to/assisted-authoring` for HTTP source support limits.
