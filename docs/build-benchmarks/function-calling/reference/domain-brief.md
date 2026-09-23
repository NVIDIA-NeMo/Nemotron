<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Domain Brief

A domain brief is the human-reviewed description passed to assisted authoring with
`--brief`. It tells the authoring model what the source is for and which domain
behaviors should shape benchmark coverage. Intake sanitizes it, binds its digest into
the evidence, and exposes it to a model only after the pre-model authorization and
approval gates.

The brief is context, not executable truth. `tools.json` defines the public call
interface, and measured probes establish source behavior and certification tier. A
claim in the brief cannot add a tool, change a schema, or make an unobserved behavior
Gold-eligible.

## Create a Brief

Copy the domain-neutral skeleton rather than the banking example:

```bash
cp src/nemotron/steps/byob/references/bfcl-domain-brief.skeleton.txt \
  /srv/sources/my-domain-brief.txt
```

Replace every bracketed `BFCL-SKELETON` block, including the instruction block at the
top. Intake refuses a file while any marker remains.

The shipped files serve different purposes:

- [`src/nemotron/steps/byob/references/bfcl-domain-brief.skeleton.txt`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/byob/references/bfcl-domain-brief.skeleton.txt) is the form to
  complete for a new domain.
- [`src/nemotron/steps/byob/references/bfcl-domain-brief.example.txt`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/byob/references/bfcl-domain-brief.example.txt) is a completed
  Vietnamese banking example. Use it to understand the expected level of detail, not
  as a template for another domain.

## Required Content

Write concise prose that covers all seven subjects named by the skeleton:

1. What the domain is, who the assistant serves, and what the oracle represents.
2. Which read-only questions the assistant answers.
3. Which operations change state, or that the domain has none.
4. Which mutations require confirmation and how confirmation is supplied.
5. When a schema-valid request is still refused because of business rules or state.
6. How identifiers are shaped and how unknown identifiers are handled.
7. Which language the benchmark uses.

Describe capabilities in domain language rather than repeating tool names. The
authoring pipeline reads the reviewed catalog separately; the brief explains how those
capabilities relate to user intent.

## Complete Example

This compact library brief matches the source used throughout the domain-data guide:

```text
Benchmark an English library assistant over a deterministic catalog and loan oracle.

The assistant answers whether a known book is available and checks out an available
book for a known patron. Checkout changes loan state only after the caller confirms.
It is refused when the book or patron is unknown, or when the book is unavailable.

Book identifiers use the BK- prefix and patron identifiers use the P- prefix.
Unknown identifiers return a structured not-found error and are never approximated.

The benchmark is authored and answered in English.
```

The brief does not need to enumerate fixtures, JSON fields, expected responses, or
probe cases. Those belong in the source package, `tools.json`, and the probe plan.

## Safety and Validation

A finished brief must:

- be non-empty plain UTF-8 and smaller than 16 KiB;
- contain no `BFCL-SKELETON` marker;
- contain no credential, customer data, private hostname, or other secret;
- describe only reviewed behavior the source is intended to provide.

There is no standalone domain-brief validator. Intake checks encoding, size,
placeholder removal, and review-blocking content, then writes the sanitized brief and
redaction report into the evidence bundle. Editing the source brief after intake
invalidates downstream digest-bound approvals.

Use the completed file when intake starts:

```bash
uv run python -m nemotron.steps.byob.scripts.bfcl_author \
  --ci author \
  --workspace /srv/bfcl/authoring/my-domain \
  --source /srv/sources/my-domain \
  --brief /srv/sources/my-domain-brief.txt \
  <remaining intake arguments>
```

## Common Failures

| Symptom | What it means |
| --- | --- |
| Still contains `BFCL-SKELETON` | Replace every bracketed block, including the instruction header, then retry intake. |
| Exceeds the 16 KiB limit | Shorten the brief. Intake persists at most 16384 bytes. |
| Not valid UTF-8 | Save the file as plain UTF-8 without a conflicting encoding. |
| Empty or whitespace only | Write the seven required subjects; a blank file is refused. |
| `cannot be reviewed safely` | The sanitizer found blocking text such as a credential, private hostname, or similar secret. Remove it; do not rely on redaction to hide operator-owned secrets after the fact. |
| Cannot read the path | Point `--brief` at a file that exists and is readable by the intake process. |

## Related Information

- {doc}`../how-to/start-from-domain-data` for preparing the source, catalog, brief,
  and probe plan in dependency order.
- {doc}`../how-to/assisted-authoring` for the complete intake, drafting, review,
  freeze, and publication workflow.
- {doc}`probe-plan` for the intake cases that accompany the brief.
- {doc}`oracle-pack-inputs` for the distinction between authoring inputs and Oracle
  Pack files.
- {doc}`tools-and-fixtures` for the public tool interface and deterministic source
  records.
