# BFCL authoring rollout policy

Live source intake is disabled by default for every built-in adapter. Enablement may come
from a reviewed `adapter_rollout` policy block or these strict environment overrides:

- `BFCL_ENABLE_LOCAL_PYTHON`
- `BFCL_ENABLE_HTTP_PACKAGE`
- `BFCL_ENABLE_MCP_MODE_A`

Accepted environment values are `1`, `true`, `yes`, `0`, `false`, and `no`
(case-insensitive, surrounding whitespace ignored). Omitted and explicit false values
disable the adapter. Malformed values and unknown policy adapter kinds fail closed.
Environment values override reviewed policy for the selected adapter and the effective
decision is recorded in `bfcl-resolved-authoring-config-v2`.

`BFCL_ENABLE_EXPERIMENTAL_MCP` remains an alias for MCP Mode A for one deprecation
window. If the legacy and current MCP variables are both present, they must resolve to
the same boolean; disagreement fails with `rollout_settings_conflict`.

The gate applies to operations that inspect or execute a source: local Python intake,
HTTP-package intake, MCP discovery/intake, and MCP gateway startup. Offline artifact
verification, review, approval, and freeze do not require rollout enablement.
