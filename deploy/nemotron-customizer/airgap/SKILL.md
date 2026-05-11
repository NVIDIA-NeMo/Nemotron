---
name: nemotron-customizer-airgap
description: Prepare, validate, build, and use Nemotron Customizer airgap image bundles for offline clusters. Use when planning airgapped deployments, editing deploy/nemotron-customizer/airgap/airgap.yaml, selecting workflow targets, grouping step execution images, baking repo overlays or wheel additions, resuming airgap runner builds, or submitting `nemotron steps run` jobs inside an airgapped environment.
---

# Nemotron Customizer Airgap

Use this skill to help an agent produce a connected-machine airgap bundle and
then submit Nemotron Customizer steps from the airgapped side. Keep it grounded
in the checked-in runner and manifests; do not invent a parallel packaging flow.

## Read First

- `deploy/nemotron-customizer/airgap/README.md` for the operator flow.
- `deploy/nemotron-customizer/airgap/airgap.yaml` for the current image map.
- `deploy/nemotron-customizer/airgap/runner.py` when changing behavior.
- `tests/deploy/test_airgap_runner.py` before editing runner logic.
- `deploy/nemotron-customizer/airgap/configs/` for runtime overlay configs.

For selected steps, inspect the catalog through the CLI:

```bash
uv run nemotron steps show <step_id> --json
```

## Workflow

1. Establish the side of the workflow:
   - Connected machine: validate, build, save image tarballs.
   - Airgapped side: load images, set env profiles, run selected steps.

2. Gather the minimum inputs:
   - Target steps and config names, for example `sft/megatron_bridge:tiny`.
   - Target architecture or Docker platform, for example `linux/amd64`.
   - Available base images and whether the connected machine can pull them.
   - Airgapped env profile name, mounts, model/data/checkpoint locations.
   - Whether destructive or expensive actions such as `--execute`, Docker build,
     Docker volume cleanup, or state-file removal are explicitly allowed.

3. Plan with the runner first:

```bash
uv run python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml
```

Use `--target <step_id>:<config>` for one-off selections without editing YAML.
The runner expands dependencies from `dependencies`, validates selected step
files/configs, groups execution images, and prints selected execution images.

4. Edit `airgap.yaml` only where the runner expects configuration:
   - `workflow.stages` or CLI `--target` for selected customer steps.
   - `dependencies` for explicit upstream Nemotron Customizer step outputs.
   - `step_execution_images` for step-to-image mapping.
   - `execution_images` for base image, tag, tar, platform, and import probes.
   - `launcher_image` for the launcher container.

5. Execute only when the user asks for a real build:

```bash
uv run python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml \
  --execute
```

If a build fails midway, keep `airgap-build-state.yaml` and rerun the same
command. Remove or move that state only when intentionally changing the plan.

6. On the airgapped side, use images from `out/airgap-manifest.yaml` under
`step_execution_images`. Submit with the plural CLI:

```bash
uv run nemotron steps run <step_id> \
  -c <config-or-airgap-overlay> \
  -b <airgap-profile> \
  run.env.container_image=<image-from-manifest>
```

For `sft/megatron_bridge`, prefer the airgap overlay configs under
`deploy/nemotron-customizer/airgap/configs/`; they clear runtime git auto-mounts
because the runner bakes those repos into the execution image.

## Guardrails

- Keep models, datasets, checkpoints, secrets, and customer files out of images.
  Put them on persistent storage and reference them through config overrides and
  `run.env.mounts`.
- Treat `${auto_mount:git+...}` as a connected-machine build input. The runner
  bakes pinned repo overlays into execution images so airgapped jobs do not clone
  from GitHub.
- Do not add missing packages blindly. Let `discover-execution-deps` and
  import probes determine small additions; keep heavyweight framework deps in
  the base image choice.
- Preserve offline defaults unless the user has an internal mirror:
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`,
  and `WANDB_MODE=offline`.
- Use `nemotron steps ...`; do not reintroduce `nemotron step ...`.

## Validation

After edits to runner logic, YAML structure, or airgap docs, run:

```bash
uv run pytest tests/deploy/test_airgap_runner.py -q
```

For CLI-facing examples, also smoke the command shape:

```bash
uv run nemotron steps --help
uv run nemotron steps show prep/sft_packing --json
```

Do not run Docker build/save stages during validation unless the user explicitly
asked for a real connected-machine bundle build.
