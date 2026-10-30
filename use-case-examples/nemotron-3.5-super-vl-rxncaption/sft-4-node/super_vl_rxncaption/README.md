# Chemical reaction parsing: support code and data formats

These utilities support the
[four-node full-SFT script](../super_vl_rxncaption_sft/README.md).
Keep this directory beside the script directory when copying them.
The [LoRA notebook](../../lora-1-node/chemical_reaction_parsing_lora.ipynb)
uses its own adjacent `rxncaption_support.py`; copy that file and
`reaction_example.png` with the notebook. Neither workflow requires installing
a custom Python package.
The pipeline downloads inputs, prepares data, trains with `torchrun`, exports the
checkpoint, starts a local vLLM server, and scores its predictions.

## Download sources

The examples default to [NVIDIA Nemotron 3.5 Super VL (BF16)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16).
Configure data archives and the software environment
in `settings.yaml`. Set the environment variables below or replace their
references with values directly. The downloader reports missing settings before
starting large downloads. Use a vision-language checkpoint compatible with the
included recipe, and choose AutoModel/vLLM versions that support it.

The model is pinned to a public release. Supply the prepared-data archive locations
and compatible AutoModel/vLLM versions. To use a different model, override both
model environment variables below. Authenticate gated alternatives with `hf auth login`.

| Variable | Value to provide |
|---|---|
| `SUPER_VL_MODEL_ID` | Optional override of the configured BF16 HF model repository |
| `SUPER_VL_MODEL_REVISION` | Optional override of the pinned 40-character HF commit |
| `RXN_DATA_URL` | Downloadable HTTPS URL for the prepared BIVP training/test ZIP |
| `RXN_DATA_SHA256` | SHA-256 digest of that ZIP |
| `RXN_BENCHMARK_URL` | Downloadable HTTPS URL for the held-out benchmark ZIP |
| `RXN_BENCHMARK_SHA256` | SHA-256 digest of that ZIP |
| `AUTOMODEL_REVISION` | Immutable 40-character commit from the configured AutoModel repository |
| `AUTOMODEL_IMAGE` | [NeMo AutoModel NGC image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel): `nvcr.io/nvidia/nemo-automodel:<tag>` compatible with that revision |
| `VLLM_IMAGE` | Serving container image supporting the checkpoint architecture |

`download` performs the Hugging Face snapshot download, checksum-checked ZIP
downloads/extraction, AutoModel source checkout, and Docker image pulls. Cached
downloads can be reused. It does not require pre-staged model/data files or a
separate evaluation repository. Signed archive URLs are not written to run reports.
Pin source revisions and container digests for repeatable runs. `train_python`
and `eval_python` select Python inside the respective images; their default is
`python3` on the image's PATH. Set them to your image's executable when it uses
a separate environment. These are container settings, not host Python paths.

### Paths on your machine and inside Docker

You configure one host path: `workspace`. A relative path resolves beside
`settings.yaml`, so launching the script from another directory does not change
where artifacts are written. For four nodes, use a shared absolute path.

| Path | How it is created |
|---|---|
| `<workspace>/assets/` | Model, data, benchmark, and AutoModel downloads |
| `<workspace>/runs/<run_id>/` | Your run's configuration, checkpoints, logs, and results |
| `/workspace` inside Docker | Mount of your configured host workspace |
| `/tutorial` inside Docker | Read-only mount of these tutorial directories |

The launcher creates these Docker mounts. You do not need matching `/workspace`
or `/tutorial` directories on your host. Recipe templates use environment
placeholders; the launcher fills in the mounted model, data, and checkpoint
paths when it generates `train.yaml`.

### Training archive layout

The ZIP may contain these paths directly or under one wrapping folder:

```text
train/dataset.jsonl
train/images/...
test/dataset.jsonl
test/images/...
```

Each JSONL row has one image and one user/assistant exchange:

```json
{"messages":[{"role":"user","content":"<image> Extract the reactions as JSON."},{"role":"assistant","content":"[{\"reactants\":[{\"type\":\"mol\",\"index\":1}],\"conditions\":[],\"products\":[{\"type\":\"mol\",\"index\":2}]}]"}],"images":["images/example.png"]}
```

Paths are relative to the split directory. Images must already show numbered
molecular boxes matching the targets. The tutorial's
[public data conversion guide](../../README.md#public-dataset-and-license)
downloads U-RxnDiagram-15k and runs `prepare_rxncaption_data.py` to create this
exact schema and the benchmark archive.

The source and converted data are CC BY-NC 4.0. Keep the generated
`DATA_LICENSE.md` in both archives, provide attribution, and obtain separate
permission before any commercial use.

Preparation filters OCR/reasoning-contaminated targets, removes empty text items,
reserves 400 validation examples by a seed-42 hash of image IDs, and checks that
training/validation do not overlap the benchmark. Training repeats examples with
5–8 reactions twice and those with 9+ reactions three times. The number of updates
is computed from the weighted dataset and global batch size 16. Validation is not
oversampled. Source files remain unchanged.

### Benchmark archive layout

```text
images/gt_bivp/                 # 400 numbered, ground-truth-box PNG images
ground_truth/ground_truth.json # COCO-style images, bboxes, and reaction roles
mapdicts/mapdict_gt.json       # visual box indices -> pixel bounding boxes
```

The scorer and converter are included beside this README; the archive contains
only evaluation data. Ground truth uses per-image `file_name`, `bboxes` (integer
`id`, `category_id`, `[x,y,width,height]`, optional `text`), and `reactions` with
`reactants`, `conditions`, and `products` lists of bbox IDs. Each mapping entry
has `file_name` and `map_dict`, keyed by 1-based visual index, with `yolo_bbox` in
the same pixel format. The 400 test IDs must match the training package's test
split. The evaluation images are never filtered by training-label cleanup.

## Evaluation and artifacts

Evaluation uses BF16, tensor parallelism 2, temperature 0, thinking disabled,
2,048 output tokens, one sequence at a time, and no prefix caching. Every image
is scored. Malformed responses become empty predictions and are reported;
failed inference requests cause a nonzero exit after saving diagnostics.

The scorer reports the two metrics used in the
[RxnCaption paper](https://arxiv.org/abs/2511.02384). SoftMatch ignores text,
merges molecular conditions into reactants, and preserves product direction.
HybridMatch compares molecules by role, requires exact reactant/product text,
and allows normalized edit distance up to 0.2 for condition text. Results are
computed from your checkpoint's predictions on the downloaded benchmark.

Each `workspace/runs/<run_id>/` contains `train.yaml`, `prepared.json`,
`split_ids.json`, `example.png`, `logs/`, `audit/`, `checkpoints/`, `export.json`,
and `evaluation/{raw,normalized,converted,metrics}.json`. Training never silently
restarts in an existing run. Use a new run ID after a failed/partial training run.
If a stage fails, inspect its log before continuing with a later stage.

Full SFT uses AutoModel's consolidated export, with an offline consolidation
fallback. Every rank's trainability audit and the exported shards are checked
before evaluation.

## Checking your setup

Start with `--smoke-steps 5` to check downloads, preparation, training, export,
and evaluation with your selected environment. The smoke run uses the full
model and benchmark, so startup and export still take time.

CPU checks can be run from this directory:

```bash
python test_workflow.py
```

They cover download validation, data splits, launch commands, export, and
scoring. The tensor-export test additionally requires PyTorch and safetensors;
it reports a skip if those packages are absent. These checks do not validate
GPU training or compatibility of a newly selected model/container combination.
