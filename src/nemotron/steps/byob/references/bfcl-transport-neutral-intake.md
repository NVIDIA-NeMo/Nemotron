# BFCL transport-neutral intake

`local_python`, `http_package`, and `mcp_mode_a` share one trust spine.
Transport code may collect identity and raw observations, but it cannot issue a tier,
sign certification, construct the final evidence digest, or authorize model exposure.

## Shared sequence

1. Resolve `bfcl-source-declaration-v1` through the fixed adapter registry.
2. Validate the reviewed held-out decision and load runtime-only sensitive terms.
3. Collect a source identity, reviewed catalog, descriptor, and BFCL-measured probe
   records. Local paths remain constrained to reviewed roots; HTTP uses the pinned
   endpoint contract; MCP performs its existing discovery through a compatibility
   collector.
4. Bind every outcome to the descriptor, source identity, certification profile, and
   optional execution-plan digest.
5. Let the BFCL certification authority derive the attained tier and sign the report.
   A source below the requested tier is refused before publication.
6. Sanitize the domain brief, construct `bfcl-source-evidence-v2`, prove held-out
   redaction, and derive the exact `bfcl-model-exposure-authorization-v1` subject.
   Intake does not grant authorization: the subject is the digest-bound object a human
   or organizational policy must authorize before a model request.
7. Write evidence and all trust sidecars to a sibling staging directory, write
   provenance last, and rename the staging directory into place. Any failure removes
   staging and leaves no authoritative output directory.

## Conventional entry point

`run_conventional_intake` accepts an inert source declaration, an explicit source base
directory and allowed roots, pack identity, domain brief, certification authority, and
held-out decision. An optional probe plan can raise evidence above A0 only when the
published profile derives that tier from what the probes observed. The plan is the same
document for every transport, including the MCP intake entry point, because the questions
on the ladder do not change when the source is reached over a socket: a local package is
probed through a child process, an HTTP package through one endpoint session per episode,
an MCP Mode A source through one gateway session per episode, and a deadline that expires
deletes that session before the observation is recorded. A session-based plan without
`fixtures` fails closed, since a session cannot read a reviewed fixture file.

The published envelope contains:

- `evidence_bundle.json`
- `adapter_certification.json`
- `source_observations.json`
- `domain_brief.source.txt`
- `domain_brief_redaction.json`
- `held_out_redaction.json`
- `model_exposure_subject.json`
- `intake_provenance.json`

`intake_provenance.json` binds the declaration, profile, identity, certification,
evidence, reports, observations, and exposure subject. Paths are not evidence and are
therefore not part of the content identity.

## MCP compatibility

The legacy MCP runner still owns MCP connection discovery and pack compatibility
artifacts. Its v2 path delegates certification, brief sanitization, evidence
construction, held-out proof, and exposure-subject construction to
`finalize_source_intake`. Legacy evidence is transformed only inside the evidence
factory supplied to that shared orchestrator. Consequently, all three transports feed
the same strict `SourceEvidenceDocument` loader and downstream drafting receives no
adapter-specific branch.

## Tier behavior

- A0 exposes catalog and pinned identity only. Result shapes, error codes,
  confirmation behavior, and reset isolation remain explicit unresolved gaps.
- A1 may mark observation capability as observed, but reset and confirmation behavior
  remain unresolved.
- A2 may mark state/reset capabilities observed only when the profile verifies reset,
  isolation, timeout cleanup, mutation truthfulness, and result coverage.

The intake phase emits an exposure subject, not an approval. Drafting must still verify
a separately supplied authorization against that exact subject.

## Shared contract suite

The suite runs one parameterized trust-contract matrix against `local_python`,
`http_package`, and the MCP compatibility path. Every fixture is reloaded through
`SourceEvidenceDocument`; the suite then independently verifies certification input
binding and signature, domain-brief source/report binding, held-out proof, exposure
subject and authorization binding, observation/profile equality, and stale source
identity refusal.

The matrix requires A0 for all three transports and proves that the same A0 report
cannot satisfy an A2 boundary. Local process fixtures additionally exercise successful
A1 and A2 intake and prove that an A1 run requested as A2 publishes no output. A0 and
A1 unresolved gaps are asserted explicitly. Conventional probe records must carry
BFCL-measured cleanup evidence; MCP's discovery-only projection is the one exception and
is tied to the versioned `mcp-mode-a-v1` profile. That exception applies only when no
probe plan was supplied: a probed Mode A gateway runs the shared choreography and carries
the same BFCL-measured cleanup evidence as the conventional transports.

The downstream evidence loader resolves certification by the persisted profile ID
through the fixed published registry, rather than special-casing MCP. Local, HTTP, and
MCP A0 outputs therefore enter the same drafting loader. For A1/A2 evidence the loader
also accepts `source_observations.json`, verifies its own digest and exact equality with
the signed report/profile, and uses its execution-plan digest when rebinding
certification. This prevents a valid higher-tier report from becoming unverifiable—or
from being verified against a different probe plan—during drafting.
The Gold freeze boundary invokes this loader with A2 required; A0 or A1 evidence can
be drafted and reviewed but cannot be frozen as Gold.

Authenticated HTTP packages and MCP authoring sources must pin `principal_digest`,
`permission_digest`, and `authorization_context_digest`. The final digest binds the
first two to canonical environment or secret-manager references, never secret bytes.
Live metadata/control output must report the same values. Rotating a token behind the
same reference is harmless; changing the reference, principal, or effective permissions
changes source identity and makes prior observations and certification stale.

The refusal-taxonomy guard requires every `CertificationRefusalCode` to remain listed
in the published profile reference, and every profile failure reason to resolve to a
member of that enum. Focused certification, local, HTTP, and MCP suites exercise the
corresponding budget, timeout, cleanup, schema, identity, catalog, attestation, import,
credential-reference, and isolation producers. Adding an undocumented code or a
free-form profile failure therefore fails the contract suite.
