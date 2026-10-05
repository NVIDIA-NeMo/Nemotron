# Conventional assisted-authoring source packages

These package formats are inputs to source inspection. They are not Oracle Packs and
cannot carry certification, approval, Gold, or publication fields.

## Local Python

Required files:

- `backend.py`: the only import-closure root.
- `tools.json`: reviewed OpenAI function-tool definitions.
- `dependency-lock.json`: canonical `bfcl-python-dependency-lock-v1`.

Optional:

- `fixtures.json`: deterministic fixture metadata represented as one JSON object, whose
  every value is a list of objects. That much is required because it is what every reader
  of a collection assumes: rows are indexed and fields are read off them. Nothing further
  is required of the rows, which are free to disagree about which fields they carry. A
  collection that is not a list, or a row that is not an object, fails
  `fixture_metadata_invalid` at inspection rather than at whichever reader reaches it
  first.

The dependency lock has exactly `schema_version` and `dependencies`. Dependencies are
sorted by `import_name`; each entry has `import_name`, `distribution`, `version`, and a
lowercase SHA-256 `artifact_digest`.

### Backend interface

`backend.py` must define four module-level callables, because the episode runner reaches for
them by name: `list_tools()` returning the published names, `reset(*, ctx, fixtures=None)`
which is handed the fixtures on a worker's first call, `get_state()` whose result is digested
for the mutation and isolation probes, and `call_tool(name, arguments, *, ctx)` which
dispatches. There is no per-tool function convention: a source may dispatch however it likes,
provided `list_tools()` matches `tools.json` exactly. The catalogue probe establishes both at
A1, from a child process; `check_source_package.py` establishes the statically decidable part
of the same thing first, which is the difference between a line number and a failed probe.

### Generated sources

`scaffold_source_package.py` writes `backend.py` and `fixtures.json` from a reviewed
`tools.json`: the four calls above, one raising handler per published tool, and a fixture row
per placeholder value a published call has to be given. With `--draft-with-model` a model
additionally chooses the fixture data and, in a fixed declarative vocabulary, what each tool
does to it; the command compiles that into the same generated shape. A model never emits the
Python, because this file is executed inside the least-privilege boundary and a model that
authored the oracle would be certifying its own output.

Every generated file carries the literal `BFCL-TODO` on each decision no catalogue could have
made. Intake refuses a source containing it — `source_package_invalid` for a file in the
import closure, `fixture_metadata_invalid` for `fixtures.json` — so a scaffold cannot be
certified, and removing the marks is the record that a person read them.

BFCL parses Python source without importing or executing it. The identity closure
contains `backend.py`, every statically resolved in-package module, and package
`__init__.py` files that Python would execute. Standard-library imports are permitted.
Third-party imports must match the reviewed lock. Dynamic imports/code execution,
namespace packages, module/package ambiguity, undeclared imports, unknown encodings,
syntax errors, and symlinks fail closed.

The effective identity binds the closure paths and bytes, reviewed catalog, canonical
fixtures, dependency lock, interpreter implementation/version/cache tag, platform
ABI/SOABI, and machine architecture. Unreferenced files do not affect it.

The static-inspection descriptor declares only `describe_tools` and `pin_identity`, uses
`identity_only` probe safety, and requires no process cleanup. Live probing must issue a
new, digest-distinct descriptor before any observe/reset capability or A1/A2 result can
be certified.

### Local probe plan

`bfcl-local-probe-plan-v1` supplies a timezone-qualified clock, integer seed, canonical
runtime fixtures, and sorted cases. Cases are bounded to 16 successful observations, 8
structured errors, and 1 timeout. Every tool needs a successful case; every declared
mutating tool needs a reviewed state-changing case. Error cases name an expected code,
while timeout cases cannot claim an outcome or state transition.

[`bfcl-probe-plan.example.json`](bfcl-probe-plan.example.json) is a worked plan of that
shape, taken from a published release: ten successes covering every published tool, two
of them state-changing for the two mutating tools, four structured errors naming their
codes, and the one timeout case A2 requires. Its `fixtures` block is abridged to the
records its own cases reach, because a real plan inlines the source's complete canonical
fixtures and is dominated by them.

Before execution, BFCL scans the complete plan with the held-out detector and rejects
source imports or calls outside `bfcl-local-least-privilege-v1`. The initial policy
permits only a closed data-processing subset of the standard library and no locked
third-party dependency. It rejects host file access, dynamic code, networking, and
process APIs. This restriction is intentional: a dependency digest establishes identity,
not safe execution authority.

The process runner independently verifies backend symbols/catalog, successful and
structured-error observations, reset replay, fresh-process isolation, confirmation
non-mutation, timeout termination and recovery, mutation declarations, and result-shape
coverage. It persists digests and shapes rather than raw result/state values, then
recomputes A0 identity after execution. Only BFCL projects these records and derives A1
or A2.

## HTTP package

Required files:

- `endpoint_config.yaml`: strict, secret-free Oracle HTTP v1 declaration.
- `tools.json`: reviewed companion schemas.

The endpoint config contains credential environment-variable names, never values. The
adapter applies the existing HTTPS, TLS, no-redirect, bounded request/response, identity,
and auth-reference rules. A conformance attestation digest is mandatory for assisted
authoring.

`GET /v1/tools` supplies names only. Its unique name set must exactly equal the reviewed
catalog. Parameter schemas, descriptions, mutation declarations, and confirmation
declarations always come from `tools.json`; BFCL never infers them from names or live
results.

The A0 identity binds the normalized endpoint declaration, live metadata identity,
reviewed catalog digest, pinned and parsed `bfcl-http-oracle-v1` attestation, and optional
CA-bundle bytes. Credential values are never persisted.

## Shared reviewed catalog

Both formats use the same strict loader. The catalog must be a non-empty JSON array with
unique function names, supported BFCL JSON Schemas, boolean mutation/confirmation
annotations, no duplicate JSON keys, no unknown envelope fields, and a bounded byte
size. Canonical identity sorts tools by function name; file ordering is not treated as a
semantic change.
