# BFCL authoring credential lifecycle

Authoring configuration stores credential references, never credential values.
`bfcl-credential-reference-v1` supports environment references and injectable
secret-manager providers. Environment names, provider names, logical secret names, and
optional versions may be persisted. Resolved values remain in `ResolvedCredential`
objects only; their string representation is always redacted.

Authenticated HTTP packages and MCP authoring intake bind three non-secret values:

- `principal_digest` identifies the authenticated principal;
- `permission_digest` identifies its effective authorization scope;
- `authorization_context_digest` binds those digests to the canonical credential
  references.

The context digest does not hash credential bytes. Rotating a token behind the same
reference leaves the context stable. Changing a reference, principal, or effective
permission set changes the context and invalidates prior observations, requiring intake
and certification to run again. HTTP metadata and authenticated MCP `describe_oracle`
responses must report the pinned context fields.

Secret-manager integration is dependency-injected through `SecretManagerBackend`; BFCL
does not embed a provider SDK or global credential registry. Backend failures are
converted to stable `credential_provider_unavailable` or `credential_unresolved` errors
without preserving provider exception text.

The authorization-context digest is included in source identity artifacts and the
allowlisted adapter-identity event. Config files, model caches, provenance, structured
events, exceptions, and release files may contain references and non-secret digests,
but never resolved values.

Existing non-authoring endpoint and MCP configurations remain readable. The stricter
three-digest context is mandatory when authenticated sources enter the authoring intake
or when a context is explicitly declared.
