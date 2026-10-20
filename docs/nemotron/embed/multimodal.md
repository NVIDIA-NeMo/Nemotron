# Fine-Tune Multimodal Embeddings

Use `nemotron-3.5` to fine-tune the public
[Nemotron 3.5 Embed 1B BF16 checkpoint](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16)
for text queries over text, images, or paired documents. This guide takes you
from a directory of text and image files to a mined training set, a consolidated
checkpoint, and a base-versus-fine-tuned retrieval comparison.

The profile uses `output/embed/nemotron-3.5` by default. To continue an existing
run, pass its original `artifact_root` to every stage.

## Prerequisites

Use Linux, Python 3.12, [uv](https://docs.astral.sh/uv/), and NVIDIA GPUs with
enough memory for your batch, sequence lengths, and image sizes. Install the CLI
from a source checkout:

```bash
git clone https://github.com/NVIDIA-NeMo/Nemotron.git
cd Nemotron
uv sync --python 3.12
uv run nemotron embed info
```

Run the remaining commands from the repository root. Each stage installs its
isolated dependencies; do not combine the `text` and `vl` extras. Preparation,
training, and checkpoint evaluation require local execution.
Keep scaled dot product attention (`sdpa`) enabled: the pinned Transformers
runtime has a known Flash Attention 2 issue with image inputs, documented in
the model card.

Prepare a directory of UTF-8 text, Markdown, or static images. Extract PDF text,
render page images, and perform optical character recognition (OCR) before this
recipe. For explicit text/image pairs or page relationships, use the advanced
[source-record input](data-reference.md#source-records).

You also need approved image-capable generation and judge models that support
structured responses. These hosted models generate the training queries; they
are separate from the local embedding model being fine-tuned.

## 1. Prepare Your Directory

Place your documents under `./documents`. The recipe scans subdirectories and
accepts `.txt`, `.md`, `.markdown`, `.png`, `.jpg`, `.jpeg`, and `.webp` files.
For example, your directory can contain:

```text
documents/
  installation.md
  maintenance.txt
  charts/
    pump-limits.png
```

Each text file becomes a document with retrieval units of at most 1,024 tokens
by default. Each image becomes a separate document. Matching filenames do not
pair text with images or establish page relationships. Use a source JSONL file
when you need those relationships or custom unit boundaries.

Use enough distinct content to produce training and evaluation query groups and
eligible negatives. Keep the output root outside `./documents`. Refer to the
[directory input reference](data-reference.md#directory-input) for chunking, file
filters, token limits, and generated source IDs.

## 2. Configure Model Roles

Configure `NVIDIA_API_KEY` through your credential manager or local environment.
Select an available local GPU; the example uses device `0`:

```bash
export CUDA_VISIBLE_DEVICES=0
export EMBED_RUN=./output/embed/multimodal-first-run
```

Generation and summary/query judging both default to
[`nvidia/nemotron-3.5-super-vl-120b-a12b`](https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b).
Optionally set `NEMOTRON_3_5_SDG_GENERATOR_MODEL` or
`NEMOTRON_3_5_SDG_JUDGE_MODEL` to use other approved models.
The generator produces visual descriptions, summaries, and retrieval queries.

Generation and judging default to `https://integrate.api.nvidia.com/v1`.
Set `NVIDIA_API_BASE_URL` for another compatible service. Both roles must accept
images and structured responses. The default section workflow also calls a
separate text-embedding endpoint for summary clustering. Refer to
[Configure Multimodal SDG](sdg-reference.md) to configure each role independently.
That reference also shows how to enable JSON output mode when your endpoint
supports it and returns malformed structured responses.

No embedding checkpoint override is required for the 1B model.
`NEMOTRON_3_5_EMBED_MODEL` changes the checkpoint selected for mining, training,
and base evaluation. It does not update other profile settings.

For an alternate checkpoint, match `nim_embedding_dimension` to its output size
when evaluating a served endpoint. Also set `mining_embedding_dimension` when
using `mining_backend=vllm`. For example, the
[8B model card](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16#output)
specifies 4,096 dimensions; the profile's served evaluation check defaults to
2,048. Validate the alternate checkpoint's runtime and training settings, and
keep its processor, prompts, pooling, normalization, and sequence limits aligned
across stages. For directory input, also match `source_directory.tokenizer` and
its token budget to the selected embedding model.

## 3. Generate and Prepare Training Data

Preview the configuration, then run Stage 0 with your document directory:

```bash
uv run nemotron embed sdg -c nemotron-3.5 -d \
  source_directory.path=./documents artifact_root="$EMBED_RUN"
uv run nemotron embed sdg -c nemotron-3.5 \
  source_directory.path=./documents artifact_root="$EMBED_RUN"
test -f "$EMBED_RUN/stage0_sdg/generation_result.json"
```

Stage 0 prepares source records, enriches visual sources, plans sections and
summary combinations, generates queries, judges their support, and exports a
portable bundle. After the producer finishes successfully, Stage 0 publishes
`generation_result.json` to identify that export.
Inspect the prepared records at
`$EMBED_RUN/stage0_sdg/multimodal/inputs/sources.jsonl`, retained query counts, and
rejected-query diagnostics before proceeding.

For an initial run without semantic summary combinations, use a separate
artifact root and add `sdg_options.context_strategy=unit` and
`sdg_options.combination_iterations=0` to the Stage 0 command. This changes the
generation strategy; it does not relax quality gates. `preview=true` is not
supported for this workflow. A CLI dry run uses `-d`.

Prepare the generated data with the default `mixed` view, which retains
text-only, image-only, and paired units:

```bash
uv run nemotron embed prep -c nemotron-3.5 \
  artifact_root="$EMBED_RUN"
test -f "$EMBED_RUN/stage1_data_prep/train_mined.automodel_unrolled.json"
```

Stage 1 checks the selected view and required inputs, preserves the producer's
query split and positive sets, then mines negatives with the local model.
Use the same view during evaluation. For a full export audit, refer to
[Validate an Export](data-reference.md#validate-an-export).
Select another [retrieval view](data-reference.md#retrieval-views) only when you
intend to restrict the document content. In particular, `image_and_text` excludes
text-only units.

## 4. Run a Short Training Check

Keep the profile's pooling, normalization, prompts, image processor settings,
and query/passage limits aligned across mining, training, and evaluation.
In-batch negatives remain disabled because unrolled queries can share positives.

Run two optimizer steps to check the data-to-checkpoint path:

```bash
uv run nemotron embed finetune -c nemotron-3.5 artifact_root="$EMBED_RUN" \
  num_epochs=null max_steps=2 global_batch_size=2 local_batch_size=1 \
  train_n_passages=2 lr_warmup_steps=0
test -f "$EMBED_RUN/stage2_finetune/checkpoints/LATEST/model/consolidated/config.json"
```

This command uses one mined negative per positive. It retains the profile's
SDPA backend and FlashAdamW optimizer. Choose a larger, measured training budget
for an actual adaptation experiment; two steps only check execution.

Use a new artifact root for a fresh experiment. To resume, keep the original
visible GPU count and checkpoint directory. `max_steps` is the total target,
and the saved optimizer and learning-rate schedule are restored; extending the
target does not restart warmup or decay. Confirm the restored step and learning
rate in the log. To reuse prepared data with a new root, set `train_data_path`
to the existing Stage 1 JSON and `sdg_input_path` to the original Stage 0 handoff
for evaluation. Keep the complete source bundle available.

Choose a full training budget by setting `num_epochs` or by clearing it with
`num_epochs=null` and setting `max_steps`. The profile defaults to one epoch,
`learning_rate=2e-6`, a global batch of eight, and four passages per query.
Mine at least `train_n_passages - 1` negatives, and adjust warmup and checkpoint
intervals to the selected budget. A lower training loss does not establish
better retrieval quality.

`checkpoint.max_recent_checkpoints=1` retains the newest recent checkpoint plus
any directories protected by pointers such as `LATEST` or `LOWEST_VAL`. This
limits checkpoint history; it does not guarantee that only one directory remains.
The consolidated model stays inside the retained checkpoint.

Before loading model weights, training reports available checkpoint disk space
and host RAM. When model metadata is available, it also estimates checkpoint
size and the space needed while the next checkpoint is written. Review warnings
before scaling. These estimates exclude model caches, data, and concurrent jobs;
they do not guarantee that a run fits.

## 5. Evaluate Both Checkpoints

Keep the default `mixed` view used for preparation and enable the base model
explicitly. The multimodal profile evaluates only the fine-tuned checkpoint by
default.

```bash
uv run nemotron embed eval -c nemotron-3.5 \
  artifact_root="$EMBED_RUN" \
  eval_base=true eval_finetuned=true eval_nim=false
python -m json.tool "$EMBED_RUN/stage3_eval/eval_results.json"
```

Check for both `base` and `finetuned` results. Each model encodes the queries and
corpus independently. Evaluation uses held-out query groups over the shared
collection, including distractors. Documents can appear in both training and
evaluation positives; this is not an unseen-document evaluation.

These relevance labels come from the same synthetic generation process. Use an
[independent retrieval evaluation](#evaluate-independent-data) to
measure quality beyond that process before deployment.

## Evaluate Independent Data

Use independently authored queries and relevance labels to assess quality beyond
synthetic-query style. Provide a BEIR directory with `corpus.jsonl`,
`queries.jsonl`, and `qrels/test.tsv`. Corpus rows use `_id` and `text` and can
include `title` and `image_path`; query rows use `_id` and `text`. The TSV columns
are `query-id`, `corpus-id`, and `score`. Preserve all known relevant documents.

Clear the inherited portable input and view, then select the dataset and the
root for relative image paths:

```bash
uv run nemotron embed eval -c nemotron-3.5 artifact_root="$EMBED_RUN" \
  sdg_input_path=null retrieval_view=null \
  eval_data_path=/absolute/path/to/independent-beir \
  image_root=/absolute/path/to/independent-bundle \
  output_dir="$EMBED_RUN/stage3_eval_independent" \
  eval_base=true eval_finetuned=true eval_nim=false
```

Compare nDCG, Recall, Precision, and MAP on a fixed corpus and query set. Report
query counts and results by modality. Keep final test data separate from
hyperparameter selection; a high score on a small synthetic set is not evidence
of generalization. Rebuild the retrieval index after changing checkpoints.

## Serve the Consolidated Checkpoint

Skip Stage 4: `nemotron-3.5` has no export profile. Install Docker with NVIDIA
GPU support, then preview and launch vLLM using the same artifact root:

```bash
uv run nemotron embed deploy -c nemotron-3.5 -d \
  artifact_root="$EMBED_RUN" gpus='device=0'
uv run nemotron embed deploy -c nemotron-3.5 \
  artifact_root="$EMBED_RUN" gpus='device=0' detach=true
curl --fail-with-body http://localhost:8000/v2/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts":["What is the operating temperature?"],"model":"nvidia/Nemotron-3.5-Embed-1B-BF16","input_type":"query"}'
```

Select the Docker GPU explicitly with `gpus`; `CUDA_VISIBLE_DEVICES` in your
shell does not select the container's GPU. The example uses host GPU `0`.

The profile mounts `stage2_finetune/checkpoints/LATEST/model/consolidated` at
`/model` in `vllm/vllm-openai:v0.31.0`. Detached launch waits for the health check.
Expect a finite 2,048-dimensional vector. The checkpoint supplies query/document
prompts; do not add prefixes yourself. For image or paired requests, use the
[model card's client](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16#online-serving).

Compare local and served retrieval on the same data and view:

```bash
uv run nemotron embed eval -c nemotron-3.5 artifact_root="$EMBED_RUN" \
  eval_base=false eval_finetuned=true eval_nim=true \
  embedding_api_backend=vllm output_dir="$EMBED_RUN/stage3_eval_served"
```

Despite its name, `eval_nim=true` enables either API backend. Match `nim_url` and
`nim_model` to the server if you change the default port or alias. The profile's
`vllm_max_model_len=8192` is separate from the 512-token query and 4096-token
passage limits. `embedding_api_truncate=NONE` rejects overlength endpoint input;
include image tokens in the serving budget. Keep the saved processor settings
aligned with local inference.

Inspect retry warnings and representative text/image responses. Metric agreement
alone does not prove checkpoint identity or vector parity. Verify the mounted
checkpoint, dimensions, normalization, and preprocessing before integration.

Stop and remove the service when finished:

```bash
docker logs --tail 100 nemotron-3.5-embed
docker stop nemotron-3.5-embed
docker rm nemotron-3.5-embed
```

Detached containers remain for log inspection. With `detach=false`, the command
stays attached and reports server failures through its exit status;
`remove_on_exit=true` removes only attached containers.

## Run a Basic Verification

The repository includes a small, fictional text-and-image fixture for checking
the installed workflow. From the repository root, select a new output directory
on a filesystem with enough space for model caches and checkpoints:

```bash
python scripts/retrieval/smoke.py \
  --output /absolute/path/to/new-verification-run --gpu 0
```

The script uses the stage lockfiles, mines negatives, trains for two steps,
resumes to four steps, and evaluates the base and saved models. It retains logs,
checkpoint checks, metrics, and dependency lockfile hashes in the output directory.
Add `--serve` to check vLLM deployment and endpoint evaluation with Docker.
The script removes only the serving container it creates.

Add `--rerank --rerank-gpus 0,1` to exercise full 8B reranker checkpoint saving
and resume with the production writer. Provision GPU and host memory as described
in the [reranker guide](../rerank/multimodal.md#checkpoint-resources).
Reserve approximately 160 GiB for two full reranker checkpoints, plus model
caches, embedding outputs, and other data. The script does not delete checkpoints.

This fixture checks execution. It does not call hosted synthetic data generation
models or measure adaptation quality.

The focused CPU contract tests also run in GitHub Actions. To reproduce that
check without GPU workloads or hosted model calls, run:

```bash
uv sync --frozen --project tests/recipes/retrieval_ci
uv pip install --python tests/recipes/retrieval_ci/.venv/bin/python \
  --no-deps --no-build-isolation -e .
tests/recipes/retrieval_ci/.venv/bin/python tests/recipes/retrieval_ci/run.py
```

The CPU runner fails if a selected test is skipped because of a missing dependency.

## Troubleshoot a Run

Inspect the failing stage's configuration with `-d` and preserve its diagnostics
before retrying. A dry run does not check inputs, endpoint access, or GPU memory.

| Symptom | Check and Recovery |
| --- | --- |
| Source input fails | Set exactly one of `source_directory.path` or `sources_file`. Keep output outside the source directory and check [directory limits](data-reference.md#directory-input). |
| No accepted queries or too few negatives | Inspect source coverage, query counts, and quality diagnostics. Add representative sources and eligible distractors; preserve the generation quality gates. |
| Generation run already exists | Use the original settings with `resume=always` only for compatible state. Use a new root after input, configuration, or producer changes. |
| Mining runs out of GPU memory | Reduce `query_embedding_batch_size` and `document_embedding_batch_size`; inspect sequence lengths and image sizes. |
| Training runs out of GPU memory | Reduce `local_batch_size` while retaining compatible global batch and device counts. Record preprocessing changes. |
| Image attention fails | Keep `attn_implementation=sdpa`. |
| Evaluation has no base result | Set `eval_base=true`. |
| Served results differ | Check checkpoint identity, image paths, processor settings, dimensions, and truncation; retain retry diagnostics. |

For agent-assisted work, use the `nemotron-retrieval-recipes` skill and provide
the directory, model profile, artifact root, available GPUs, approved endpoints,
and intended stopping point. Ask for resolved configuration and a bounded run
before scaling; retain commands, artifact paths, and evaluation results. For
Nemotron 3.5 reranking, use the separate [Stage 2 reranker guide](../rerank/multimodal.md#checkpoint-resources).
