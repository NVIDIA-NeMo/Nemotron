<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Held-Out Policy

`held_out.yaml` reserves existing fixture primary ids and template ids from ordinary
generation. It is optional and is referenced by top-level `manifest.held_out`, not by
`manifest.paths`.

Held-out ids are real records withheld from generated rows. They are different from
`manifest.absent_ids`, which are guaranteed not to exist.

## Create A Held-Out Policy

Generate a complete starter shape with:

```bash
python -m nemotron.steps.byob.scripts.scaffold_oracle_pack \
  --domain my_domain \
  --target /srv/bfcl/packs/my_domain \
  --transport python \
  --language en \
  --version 0.1.0 \
  --include-held-out
```

Replace every example id and policy value after fixtures and templates are stable.
Then wire the file from the manifest:

```yaml
held_out: held_out.yaml
```

Do not put `held_out` under `paths`. In assisted source intake, held-out applicability
is a reviewed human decision; model drafting does not invent the policy.

## Field Contract

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `version` | Yes | None | Non-empty policy version string or number; Boolean values are refused. |
| `fixtures` | Optional | `{}` | Collection names mapped to reserved scalar primary ids. |
| `templates` | Optional | `[]` | Existing template ids reserved from ordinary generation. |
| `policy.fixtures_in_backend_state` | Optional | `true` | Whether reserved fixture rows remain visible to backend reset state. |
| `policy.seed` | Optional | `0` | Deterministic seed used by held-out evaluation and sampling. Exact reservation exclusion is set membership and does not depend on it. |

Only `version`, `fixtures`, `templates`, and `policy` are accepted at the file root.
Only `fixtures_in_backend_state` and `seed` are accepted under `policy`.

The authoring evidence format may record a source for review and digest binding. Do not
copy that authoring-only `source` key into the pack's `held_out.yaml`; the pack loader
derives source identity from the manifest path.

## Minimal Example

```yaml
version: "1"
fixtures:
  records:
    - REC-HELD-OUT-1
templates:
  - record_sensitive_workflow
policy:
  fixtures_in_backend_state: false
  seed: 7
```

The referenced fixture id must exist in `fixtures.json`, and the template id must exist
in `task_templates.yaml`. An empty reservation set is valid when policy provenance must
still be recorded:

```yaml
version: "1"
fixtures: {}
templates: []
```

## Backend-State Choice

With `fixtures_in_backend_state: true`, reserved rows remain part of reset state but
cannot be selected for generated tasks. Use this only when the oracle needs the full
state to behave correctly.

With `false`, reserved rows are removed before fixture projection reaches the oracle.
This provides stronger isolation but can change behavior if other records depend on
them. Validate the choice against domain invariants.

:::{warning}
Fixture projection is not an operating-system sandbox. It protects normal pack-relative
fixture access, but trusted `backend.py` or `assertions.py` code can still access host
paths available to its process. Do not use projection as the only control protecting
secrets from hostile pack code; review the pack and run it in an appropriately isolated
environment.
:::

## Validate The Policy

There is no standalone held-out validator. Run whole-pack preparation:

```bash
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config /srv/bfcl/packs/my_domain/validate.yaml \
  --output-dir /tmp/bfcl-my-domain-validation
```

Load-time validation checks:

- referenced collections and ids exist;
- ids are unique after scalar normalization;
- reserved ids do not overlap `manifest.absent_ids`;
- primary keys are unambiguous;
- referenced template ids exist.

Generation excludes reserved bindings and writes
`stage_cache/held_out_bindings.json`. Before publication, the final stage rescans every
row and aborts if a reserved id or template leaked.

## Common Failures

- **`held_out_contract_invalid`:** correct key names, types, ids, or policy values.
- **Unknown primary id:** reserve an id that actually exists in the named collection.
- **Overlap with `absent_ids`:** decide whether the id is existing-and-reserved or
  deliberately absent; it cannot be both.
- **Unknown template id:** correct the id after template renames.
- **Ambiguous primary key:** declare `manifest.primary_keys.<collection>`.
- **Duplicate after scalar normalization:** do not mix equivalent ids such as `1` and
  `"1"`.
- **`held_out_binding_starved`:** reservations leave too few eligible records or
  templates for the requested generation budget.
- **`held_out_leak_detected`:** stop publication and inspect binding and final-output
  artifacts.

## Complete Example

Run `scaffold_oracle_pack --include-held-out` and inspect the generated
`held_out.yaml`. The runtime-normalized `contract_version: "1.0"` is internal and does
not belong in the authored file.

## Related Information

- {doc}`manifest` for the top-level `held_out` pointer and primary keys.
- {doc}`tools-and-fixtures` for existing and deliberately absent ids.
- {doc}`task-templates` for template identities.
- {doc}`output-files` for held-out diagnostics written during generation.
