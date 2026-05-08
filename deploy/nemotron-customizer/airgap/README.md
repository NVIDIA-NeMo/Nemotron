# Nemotron Customizer Airgap

This folder is scoped only to Nemotron Customizer steps under
`src/nemotron/steps/`.

The flow is intentionally small:

1. Build one **submitter image** with this repo and `uv.lock`.
2. Build one or more **task images** by grouping selected workflow stages by base image.
3. Save those images as tarballs for the airgapped side.
4. Keep models, datasets, checkpoints, and customer files on persistent storage.

Edit `airgap.yaml` first:

- `workflow.stages`: the Nemotron Customizer steps the customer wants to run
- `dependencies`: central step dependency map, for example SFT training needs SFT packing
- `step_images`: which task image each step should use
- `task_images`: the base image, output tag, and known/import-probed Python requirements

Only steps reached from `workflow.stages` are built. Steps are grouped by
`base_image + repo_overlays`; each group gets one derivative image with the
union of its small missing packages. If two selected step families share the
same base image and repo overlays, the runner emits one combined task image for
both.

Run from the repo root:

```bash
uv run python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml
```

That prints the plan. To actually pull/build/save images on the connected
machine:

```bash
uv run python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml \
  --execute
```

To run only a few stages:

```bash
uv run python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml \
  --stage validate \
  --stage discover-task-deps
```

To override the workflow without editing YAML, pass one or more selected
Nemotron step targets. Dependencies are still expanded from `dependencies`.
For example, SDG plus SFT also adds `prep/sft_packing` because SFT needs packed
data:

```bash
uv run python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml \
  --target sdg/data_designer:tiny \
  --target sft/megatron_bridge:tiny
```

Outputs are written under `deploy/nemotron-customizer/airgap/out/` by default:

- `airgap-manifest.yaml`: what was validated and built
- `airgap-progress.yaml`: incomplete execute run state used for resume
- `airgap-complete.yaml`: final execute run state after success
- `requirements-<task-group>.txt`: small missing packages per task image
- `repo-overlays-<task-group>.json`: git auto-mounts discovered from selected step configs
- `submitter-image.tar`
- `task-*.tar`
- SHA256 checksums for saved image tarballs in `airgap-manifest.yaml`

If an execute run fails midway, leave `airgap-progress.yaml` in place and rerun
the same command. Completed expensive actions are reused when their artifacts
still exist. If you intentionally change the workflow or image plan before
finishing, move or remove `airgap-progress.yaml` first; the runner will not
silently overwrite incomplete state from a different plan.

Runtime dependency probes use Docker volumes named
`nemotron-airgap-pip-cache-<platform>` to avoid downloading the same wheels on
every probe loop. To reset them, run `docker volume ls | grep
nemotron-airgap-pip-cache` and remove the relevant volume with
`docker volume rm`.

Large assets are not baked into images. The customer should stage them on
executor-visible persistent storage and reference them through config overrides
and `run.env.mounts`.

During dependency discovery, the runner mounts the connected-machine checkout
into each task image only to probe imports. The final task image deliberately
does not bake this repo; the submitter image and the normal nemo-run/nemo-runspec
code transport provide the repo to the remote job at submission time.

Repo logistics stay outside `airgap.yaml`. If a selected step config contains
`${auto_mount:git+...}`, the runner treats it as a connected-machine build input:
it fetches that pinned repo and bakes it into the derivative task image at the
requested target path. Runtime jobs then use the baked image and do not clone
from GitHub. Site-specific data/model mounts remain in env profiles or step
overrides.

If the connected machine is not the same architecture as the target cluster,
set `platform: linux/amd64` on the submitter or task image entry in
`airgap.yaml`. If you need to minimize transfer size for several images that
share layers, `docker save -o all-images.tar tag1 tag2 ...` can be used after
the runner builds the images; a single tar deduplicates shared layers better
than one tar per image.

The Dockerfiles expect the chosen base images to have Python and `pip` available
for bootstrapping small offline additions. The runtime defaults bake
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and
`WANDB_MODE=offline`; customers with an internal mirror can override those at
submission time through their env profile or `run.env.env_vars`.

For SFT Megatron-Bridge, build with the normal config so the runner can discover
the pinned Megatron-LM and Megatron-Bridge auto-mounts:

```yaml
workflow:
  stages:
    - sft/megatron_bridge:tiny
```

When submitting inside the airgap, use the deploy overlay config so those git
auto-mounts are cleared at runtime while persistent storage mounts from the env
profile still apply. Use the image printed by the runner under
`selected step images`, or read it from `out/airgap-manifest.yaml` under
`step_images`.

```bash
uv run nemotron step run sft/megatron_bridge \
  -c deploy/nemotron-customizer/airgap/configs/sft_megatron_bridge_tiny.yaml \
  -b <your-airgap-profile> \
  run.env.container_image=<image-printed-for-sft/megatron_bridge>
```
