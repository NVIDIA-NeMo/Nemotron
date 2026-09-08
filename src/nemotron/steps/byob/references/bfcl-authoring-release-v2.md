# BFCL authoring release v2

The v2 release kernel is transport-neutral. It owns deterministic review records,
explicit approval, atomic freeze, byte sealing, fresh validation, and publication
handoff. A registered adapter owns only the typed operations in
`runtime/authoring_release/contracts.py`:

- canonical pack validation and semantic fingerprinting;
- adapter review facts, blockers, and risks;
- reviewed, digest-bound freeze sidecars;
- fresh-Gold interpretation and publication-origin verification.

## Versioned records

- `bfcl-authoring-review-packet-v2`
- `bfcl-authoring-review-approval-v2`
- `bfcl-authoring-frozen-release-v2`

The packet binds the adapter kind, independently verified certification tier, source
identity, candidate pack fingerprint, source digests, adapter review data, blockers, and
stable risks. Approval binds one exact packet and exactly its risk IDs. Freeze requires
A2, rejects blockers, copies without following symbolic links, and seals every pack file,
source record, review record, approval, and adapter sidecar.

The v2 approval checklist separately acknowledges independently verified certification,
pre-model authorization, and answered questions. Review assembly verifies the signed
certification and exact exposure authorization rather than accepting release approval as
a substitute. Revised evidence must replay against its parent, open-question artifact,
and answer set; all three files enter `source_digests`.

Adapters may add sidecars only when `adapter_review.freeze_sidecars` contains the exact
relative path and SHA-256 digest. Sidecars may change the final pack fingerprint, but
cannot appear after approval without invalidating the packet.

## Compatibility

Existing `bfcl-mcp-*-v1` records remain loadable through the version-dispatching v2
loaders and through their original `runtime/mcp/release/` imports. New v2 records are not
byte-identical to v1 records. Compatibility means equivalent candidate-pack semantics,
stable legacy API behavior, and continued verification of sealed v1 releases.

`build_authoring_review.py`, `approve_authoring_review.py`, and
`freeze_authoring_pack.py` expose this contract for local Python, HTTP-package, and MCP
sources. `publish_authoring_release.py` dispatches v2 local Python and MCP releases
through fresh Gold validation and the existing `stage=all` pipeline. HTTP packages may
be frozen but remain fail-closed at publication until HTTP origin verification is
available.

Review requires both `--validation-config` and `--validation-report`. The review command
reruns BFCL prepare from that config and accepts the report only when it is the exact
fresh output path. Freeze installs a transport-neutral, digest-bound authoring lineage
record plus the approved packet and approval inside the fingerprinted pack. Publication
requires that lineage, a Gold tier, `gold_eligible: true`, and no ineligibility reasons.
Guided commands advance an immutable workspace session; artifact-only operation should
use the lower-level scripts directly.
