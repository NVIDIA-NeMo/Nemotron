# Draft Local Embedding Runtimes

This maintainer setup selects dependencies for local embedding prep, finetuning,
and evaluation. It does not change the recipe implementations or introduce a
second synthetic data generation pipeline.

This is a draft source-build workflow, not a released installation. The required
modified packages are not fetched from a public index. Container execution does
not yet use this selector. Local installation and GPU inference have been checked,
but the complete generation-to-finetuning workflow remains unvalidated. Do not use
these instructions as evidence of end-to-end support.

## Runtime Selection

The local CLI validates the resolved stage configuration, including command-line
overrides, before selecting its dependency project:

| Configuration | Project |
|---|---|
| `model_family=text`, prep or evaluation | Existing stage-local project |
| `model_family=text`, finetuning | `text_finetune/`, preserving the pre-VL source and Transformers 5.12.1 |
| `model_family=mistral3_vl`, prep, finetuning, or evaluation | Shared `native/` project with Transformers 5.15.1 |

The native project scopes the CUDA 12.9 package index to Torch and torchvision.
The pinned AutoModel dependency cannot resolve with the Torch 2.10 CUDA 13 wheel
because their CUDA Python binding requirements conflict. Dependency resolution
alone does not establish hardware compatibility.

The native VL profiles leave query and document prompts and image resizing to
the checkpoint. Prompts are read from `config_sentence_transformers.json`;
image settings come from the saved processor. A `null` profile value inherits
these settings, while an explicit empty prefix disables that prompt. Query and
passage token limits remain explicit runtime budgets. Mining selects the
retrieval processor through the loaded backbone rather than a separate
processor path or model-specific mining configuration.

## Prepare Reviewed Wheels

Use reviewed checkouts at the following revisions. The separate DataDesigner
parser patch is included as a source-build input. It is never applied to an
installed engine at runtime.

| Checkout | Revision |
|---|---|
| AutoModel | `aa6245acfcf129b22d83e815817e1560b10064c7` |
| DataDesignerPlugins | `aa6c9652e6e66ee0a6d721a10d66d58bf9e2c7ab` |
| DataDesigner 0.9.1 | `27acf141170eceb1e8242c132d56b49107462fce` plus the included patch |

To reproduce the patched core without the local patch commit, create a NEW
DataDesigner checkout from the exact public release base. Set `designer_source`
to a directory that does not exist and run these commands from the Nemotron root:

```bash
designer_source=<new-datadesigner-checkout>
recipe_root="$PWD"
git clone https://github.com/NVIDIA-NeMo/DataDesigner.git "$designer_source"
git -C "$designer_source" checkout --detach 27acf141170eceb1e8242c132d56b49107462fce
git -C "$designer_source" apply --check \
  "$recipe_root/src/nemotron/recipes/embed/runtimes/patches/data-designer-0.9.1-bare-json.patch"
git -C "$designer_source" apply \
  "$recipe_root/src/nemotron/recipes/embed/runtimes/patches/data-designer-0.9.1-bare-json.patch"
```

The `0.9.1+barejson.1` local version identifies this exact release base and
included patch without changing release tags. The patch includes its parser regression tests. Schema
validation remains required; accepting a complete bare JSON object is not a
quality-gate override.

From the Nemotron checkout root, set the checkout paths and build the wheels.
Replace each angle-bracket placeholder with an absolute path. Do not overwrite
previous experiment artifacts; use this recipe's dedicated wheel directory.

```bash
automodel_source=<automodel-checkout>
plugin_source=<plugin-checkout>
designer_source=<patched-datadesigner-checkout>
wheel_dir="$PWD/src/nemotron/recipes/embed/runtimes/wheels"
plugin_wheel_dir="$wheel_dir/plugin-aa6c9652"

git clone https://github.com/NVIDIA-NeMo/Automodel.git "$automodel_source"
git -C "$automodel_source" checkout --detach aa6245acfcf129b22d83e815817e1560b10064c7
git clone https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git "$plugin_source"
git -C "$plugin_source" checkout --detach aa6c9652e6e66ee0a6d721a10d66d58bf9e2c7ab
uv build "$automodel_source" --wheel --out-dir "$wheel_dir"
uv build "$plugin_source/plugins/data-designer-retrieval-sdg" --wheel --out-dir "$plugin_wheel_dir"
UV_DYNAMIC_VERSIONING_BYPASS=0.9.1+barejson.1 uv build \
  --project "$designer_source" --all-packages --wheel --out-dir "$wheel_dir"
```

The explicit core version distinguishes the patched package from unmodified
0.9.1 and avoids depending on which Git tags are available locally. All three
core namespace packages must use that version. The AutoModel build adds its
source revision to the package version. The native project names these exact
wheel files and does not fall back to released packages if they are missing.

Verify that the five wheels are present before resolving dependencies:

```bash
ls "$wheel_dir"/*.whl "$plugin_wheel_dir"/*.whl
uv lock --project src/nemotron/recipes/embed/runtimes/native
uv lock --project src/nemotron/recipes/embed/stage0_sdg
```

Keep the reviewed source revisions and wheel hashes with the run record. A
package version alone is not provenance for the unreleased plugin changes.
Stage 0 also uses the same plugin and patched-core wheels for both text and
image-bearing sources. Local wheels are ignored by Git and are not packaged for
remote execution. Stage 0 and multimodal Stages 1-3 therefore fail before remote
submission; use the reviewed local route. Do not commit or publish the wheels as
part of the recipe source changes.

The legacy container wrapper still uses one fixed `/opt/nemotron-venv` ready
marker without a dependency-project identity. Use a fresh container for each
legacy stage; cross-stage environment reuse has not been validated.

## Validation Status

Earlier integration-source checks covered:

- CPU runtime-selection and data-contract tests, plus imports of the mining,
  training, and optimizer implementations from the installed packages.
- Local GPU query and image-text document encoding through the recipe evaluation
  adapter, producing finite, normalized embeddings from the unchanged checkpoint.
- Portable export from hosted generation, artifact-integrity checks, and recipe
  training-input resolution for the text, image, and image-and-text views.

For the public revisions above, source-wheel builds and lockfile checks have been
repeated. The resulting locked set selects Torch 2.10 CUDA 12.9, Transformers
5.15.1, and FlashOptim 0.1.4. That exact public-source set has not been installed,
import-tested, or exercised on a GPU.

For image-bearing sources, the generation and judging models must accept image
requests. Successful query-only judging does not establish that the judge accepts
images. Keep model identities and provider failures in the run record. Using the
same model for generation and judging is recorded as a warning, not evidence of
independent model agreement.

The native dependency set does not include optional FlashAttention or Transformer
Engine kernels. The preview profile therefore selects PyTorch SDPA and
FlashAdamW explicitly. The inference check does not validate FlashAdamW kernel
execution, full hard-negative mining, finetuning, checkpoint save and reload, or
fresh post-training retrieval evaluation. Those checks, container execution, and
an independently verified end-to-end run remain prerequisites for release.
