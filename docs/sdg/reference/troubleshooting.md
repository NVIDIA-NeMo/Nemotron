<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

(sdg-troubleshooting)=
# Troubleshooting

Failure modes for local runs and cluster dispatch. For cluster-specific setup, see {doc}`../how-to/dispatch-to-cluster`.

## Local Run Failures

::::{dropdown} `Unknown column type: 'person'` or similar ValueError

**Cause**: The YAML declares a column `type` that `step.py`'s `build_columns()` does not recognise. Currently supported types: `category`, `seed`, `llm_text`, `llm_structured`, `llm_judge`.

**Solution**: Check the spelling. For `person` and `datetime` sampler support, `step.py` must be extended — see the extension reference in {doc}`config-schema`.
::::

::::{dropdown} `config must declare a non-empty columns: list`

**Cause**: The YAML has an empty or missing `columns:` block.

**Solution**: Add at least one column spec. A minimal config must include at least one `llm_text` or `llm_structured` column that produces output content.
::::

::::{dropdown} `Jinja2` template references an undefined variable

**Cause**: A prompt uses `{{ column_name }}` but `column_name` is neither a declared column, a seed field in `seed_dataset.fields`, nor an earlier column in the list.

**Solution**: Add the column or seed field, or fix the typo. Run `preview=true num_records=2` to catch this cheaply before a full generation job.
::::

::::{dropdown} Model health check fails at startup

**Cause**: Data Designer probes the model endpoint at startup. If the model is not available from the configured provider, or if `NVIDIA_API_KEY` is not set, the probe fails and the step exits before generating any records.

**Solution**:
- Confirm `export NVIDIA_API_KEY="..."` is set.
- Add `skip_health_check: true` to the model spec to bypass the probe (useful for local or vLLM endpoints that aren't in the provider catalog).
::::

::::{dropdown} Output JSONL is empty or has fewer records than `num_records`

**Cause**: Data Designer skips or drops records where the structured output doesn't validate against `output_format`, or where the LLM returns a refusal.

**Solution**:
- Run `preview=true` and inspect a sample for refusals or schema mismatches.
- Simplify the `output_format` if the model consistently fails to match a complex schema.
- Raise `max_tokens` if responses are being cut off mid-JSON.
::::

## Cluster Dispatch Failures

::::{dropdown} Job exits immediately with `No such file or directory` (launch script)

**Cause**: `nemo_run_dir` is not on shared storage. The data-mover sidecar writes the launch script to `nemo_run_dir`, but the main container cannot see it if the path is local to a different node or not mounted.

**Solution**: Set `nemo_run_dir` to a path on the shared NFS mount and add the corresponding `mounts` entry to the env.toml profile. See {doc}`../how-to/dispatch-to-cluster`.
::::

::::{dropdown} `data-designer` import error inside the container

**Cause**: The NeMo container image does not pre-install `data-designer`.

**Solution**: Add to `startup_commands`:

```toml
startup_commands = [
    "python -m pip install --quiet --break-system-packages 'data-designer==0.5.5'"
]
```
::::

::::{dropdown} Job rejected or OOM-killed immediately on a CPU node

**Cause**: The default `shared_memory_size` (65536 MB) exceeds the available RAM on the CPU node type.

**Solution**: Set `shared_memory_size = 1024` in the env.toml profile. The SDG step makes no use of shared memory.
::::

::::{dropdown} `NVIDIA_API_KEY` not available inside the container

**Cause**: `NVIDIA_API_KEY` is not automatically forwarded to the job environment the way `HF_TOKEN` and `WANDB_API_KEY` are.

**Solution**: Declare it explicitly in the env.toml profile:

```toml
[lepton_sdg_data_designer.env_vars]
NVIDIA_API_KEY = "${oc.env:NVIDIA_API_KEY}"
```

And set it in your shell before submitting: `export NVIDIA_API_KEY="..."`.
::::

## Related

- {doc}`../how-to/dispatch-to-cluster` — Full cluster setup walkthrough.
- {doc}`cli-reference` — Flags and hydra overrides.
- {doc}`config-schema` — YAML field reference.
