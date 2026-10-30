# BFCL release revocation and supersession

BFCL revokes the immutable `frozen_pack_fingerprint`; a revocation never edits frozen
release bytes or an existing `run_manifest.json`. Each `bfcl-release-revocation-v1`
record is signed with an operations Ed25519 key and also binds the freeze-manifest
digest, adapter kind, reason code, effective time, sequence, and previous record.

`supersede` means the revoked release has a named replacement fingerprint. The old
release remains revoked, and the replacement is verified independently; the pointer
does not transfer trust or Gold eligibility.

Records are distributed in a signed `bfcl-release-revocation-registry-v1` snapshot.
The snapshot has a monotonic generation and a bounded freshness window. Consumers must
pin the expected issuer, trusted public key, and minimum accepted generation. Missing or
invalid signatures, untrusted issuers or keys, expired/rolled-back snapshots, and
forked/incomplete supersession chains fail closed
([`test_bfcl_release_revocation.py`](../../../../../tests/steps/byob/test_bfcl_release_revocation.py)).

<!-- doc-smoke: bfcl-revoke-help -->
```shell
python -m nemotron.steps.byob.scripts.revoke_authoring_release --help
```

Issue a record with:

The following is an operator template, not an executable smoke command:

```text
python -m nemotron.steps.byob.scripts.revoke_authoring_release \
  --release FROZEN_RELEASE --registry revocations.json \
  --issuer bfcl-release-operations --key-id release-key-1 \
  --private-key release-key.pem --action revoke \
  --reason-code invalid_oracle_behavior
```

For `--action supersede`, also pass `--replacement-release`. Registry updates are
serialized by a private sidecar lock and replace the signed snapshot atomically.

Publication can load the registry using `--revocation-registry`,
`--revocation-issuer`, `--revocation-public-key`, and `--revocation-key-id`.
The handoff checks the registry before validation, before generation, and before
accepting the published manifest. Consumer verification uses policy `reject` by
default; explicit `warn` policy returns a stable `release_revoked` warning and records
that the authenticated policy was applied.

Revocation prevents new use only where publication or consumer verification consults
the authenticated registry. Copies already downloaded to other systems cannot be
physically recalled or deleted by BFCL. Offline consumers that skip registry
verification may continue to use revoked bytes; operators must distribute fresh
registry snapshots and enforce the verifier at those boundaries.
