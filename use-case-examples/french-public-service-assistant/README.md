# French Public-Service Assistant: Post-Training Nemotron 3.5 Lightning with Grounded Synthetic Data

Companion code for the blog post on post-training Nemotron for French-language applications. It adapts
[Nemotron 3.5 Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) into a
French public-service assistant that answers questions about identity-document and vehicle-registration
procedures, using [Nemotron-Personas-France](https://huggingface.co/datasets/nvidia/Nemotron-Personas-France)
to vary the simulated users and official procedure text to keep the answers grounded.

The blog explains the workflow. This directory holds the pieces that are too long for the blog: the seed
data, Data Designer config, audit script, LoRA config, and test prompts.

> **What post-training changes here.** The adapter changes *how the model responds*: voice, length, and
> staying inside the supplied facts. It does not add facts the model lacks. Fees, documents and deadlines
> change, so a production assistant should read them from your own data or retrieval at answer time.

## What is in this directory

| Path | What it is |
|---|---|
| `sources/procedures.json` | Six procedures with facts taken from published pages, each with its source URL and verification date |
| `scripts/sample_personas.py` | Samples a balanced slice of Nemotron-Personas-France into persona seeds |
| `scripts/build_seeds.py` | Joins personas with procedure facts (and a region note) into the Data Designer seed file; strips persona names |
| `config/sdg_data_designer.yaml` | Data Designer config: scenario, grounded answer, and judge columns |
| `scripts/audit_and_split.py` | Filters generated records and splits 90/10 by persona |
| `config/peft_automodel.yaml` | LoRA config, the DGX Station recipe with only checkpoint/validation cadence and paths changed |
| `eval/test_prompts.jsonl` | Ten test prompts with the expected behavior for each |
| `scripts/compare_models.py` | Runs the test prompts against the base and adapted models (together or one at a time) and writes a side-by-side file |
| `tests/test_scripts.py` | Regression tests for name redaction and the audit filters (`uv run pytest use-case-examples/french-public-service-assistant/tests`) |

## The workflow

| Step | What happens | Where |
|---|---|---|
| 1. Define the behavior | Write the behavior contract (audience, task, language, grounding, safety, output) | Blog, Table 1 |
| 2. Seeds | Sample personas, then join each with a procedure and its published facts | `sample_personas.py`, `build_seeds.py` |
| 3. Serve the generator | Serve the base model locally with vLLM (or use a hosted endpoint) | Commands below |
| 4. Generate | Data Designer writes a user message and a grounded answer per seed, then a judge checks grounding | `sdg_data_designer.yaml` |
| 5. Audit and split | Drop failing records, split by persona | `audit_and_split.py` |
| 6. Fine-tune | LoRA on the filtered conversations | `peft_automodel.yaml` |
| 7. Merge, serve, compare | Merge the adapter, serve base and adapted models, compare answers | Commands below, `compare_models.py` |

## Requirements

- A DGX Station GB300 or a DGX Spark (the workflow has been run end to end on a DGX Spark). The committed `peft_automodel.yaml` uses DGX Station values, with the DGX Spark values (`local_batch_size: 1`, `seq_length: 8192`) in comments. Other NVIDIA GPU systems need adjusted settings.
- Python 3.11 to 3.13 (the `data-sdg` extra installs Data Designer only on these versions), `uv`, Git, Docker, and the NVIDIA Container Toolkit.
- This repo checked out and synced: `uv sync --extra data-sdg --extra evaluator`.
- About 130 GB of free disk for the base model, merged model and checkpoints.
- A second, different model to act as judge, reachable through an OpenAI-compatible endpoint (see Step 4).

All commands run from the repository root.

## Run it

### Step 2: seeds

```bash
uv run python use-case-examples/french-public-service-assistant/scripts/sample_personas.py
```

This writes `data/persona_seeds.jsonl` and then runs `build_seeds.py` for you, producing `data/seeds.jsonl`
(persona + procedure + official facts), the file the Data Designer config reads. To rerun only the join, for
example after changing `sources/procedures.json`, run `scripts/build_seeds.py` on its own.

### Step 3: serve the generator (DGX Station)

```bash
docker run -d --name lightning-gen --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.26.0 \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --served-model-name lightning \
  --trust-remote-code --max-model-len 8192 --max-num-seqs 64 \
  --gpu-memory-utilization 0.80 --enable-prefix-caching --reasoning-parser nemotron_v3
```

On a DGX Spark, use `config/sdg_data_designer.hosted.yaml` instead: it points the `local_vllm` provider at the
NVIDIA hosted API (`NVIDIA_API_KEY`) and lowers `max_parallel_requests`, since a hosted API rate-limits aggressively.

### Step 4: generate

The judge uses the same `NVIDIA_API_KEY` as the rest of this repo, read from the environment; never put the key in the YAML.

```bash
export NVIDIA_API_KEY=...           # key for build.nvidia.com / your judge endpoint
# optional: JUDGE_BASE_URL, JUDGE_MODEL (defaults: NVIDIA hosted API, nvidia/llama-3.3-nemotron-super-49b-v1.5)

CFG=use-case-examples/french-public-service-assistant/config/sdg_data_designer.yaml
uv run nemotron steps run sdg/data_designer -c $CFG preview=true num_records=8   # read all 8 records
uv run nemotron steps run sdg/data_designer -c $CFG                              # full run
```

Always read the preview. Check that the answers use only the supplied facts and contain no reasoning text.

### Step 5: audit and split

```bash
uv run python use-case-examples/french-public-service-assistant/scripts/audit_and_split.py
```

The route and number checks are cheap filters (a route the answer says is *not* available is kept, and "un an"
and "1 an" count as the same number); the judge carries the real grounding check. Then read a sample of
`data/train.jsonl` yourself. The judge can be wrong in both directions.

### Step 6: fine-tune

Stop the generator first so the GPU is free, then train inside the AutoModel container:

```bash
docker rm -f lightning-gen
docker run -it --rm --name lightning-lora --gpus all --network host --ipc=host \
  --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HOME=/hf -v ~/.cache/huggingface:/hf \
  -v "$PWD":/workspace/Nemotron -w /workspace/Nemotron \
  nvcr.io/nvidia/nemo-automodel:26.08 \
  automodel use-case-examples/french-public-service-assistant/config/peft_automodel.yaml --nproc-per-node 1
```

Use `--shm-size=128g` on a DGX Station. The container writes as root; fix ownership with
`docker run --rm -v "$PWD/use-case-examples/french-public-service-assistant/checkpoints":/ckpt nvcr.io/nvidia/nemo-automodel:26.08 chown -R $(id -u):$(id -g) /ckpt`.

### Step 7: merge, serve, compare

```bash
docker run --rm --gpus all --ipc=host \
  -u $(id -u):$(id -g) -e HOME=/tmp -e USER=$USER -e LOGNAME=$USER -e HF_HOME=/hf \
  -v ~/.cache/huggingface:/hf -v "$PWD":/workspace/Nemotron -w /workspace/Nemotron \
  -e PYTHONPATH=/workspace/Nemotron/src \
  nvcr.io/nvidia/nemo-automodel:26.08 \
  python3 -m nemotron.steps.convert.merge_lora.step \
    lora_checkpoint=use-case-examples/french-public-service-assistant/checkpoints/lightning_fr_public_service_lora/epoch_1_step_99/model \
    base_hf_path=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
    output_hf_path=use-case-examples/french-public-service-assistant/checkpoints/lightning_fr_merged
```

The checkpoint folder name depends on how many epochs your data spans (about 650 training rows at global batch size 8
crosses into a second epoch within 100 steps, giving `epoch_1_step_99`). Check
`ls use-case-examples/french-public-service-assistant/checkpoints/lightning_fr_public_service_lora/` and use the
latest folder.

Serve the merged model with the same vLLM command as Step 3 (`--model` pointing at the merged folder,
`--served-model-name lightning-fr`). If GPU memory only fits one model, serve one at a time and
run the comparison in two passes; each pass saves its answers and the second writes the side-by-side file:

```bash
CMP=use-case-examples/french-public-service-assistant/scripts/compare_models.py
# with the base model served as `lightning` on port 8000
uv run python $CMP --base-url http://localhost:8000/v1 --base-model lightning
# then switch to the merged model, served as `lightning-fr`
uv run python $CMP --tuned-url http://localhost:8000/v1 --tuned-model lightning-fr
```

If both models are served at once, pass all four options in one command. If an endpoint serves a single model
under a different name than the one you passed, the script uses the served name and prints a note.

## Adapting it to your own application

1. Replace `sources/procedures.json` with your own approved source text. Keep the source URL and the date you
   checked it for every entry.
2. Edit the scenario and answer prompts in the config, and the `local_constraint` values, to match your users.
3. Replace `eval/test_prompts.jsonl` with prompts from your own domain, and write the expected behavior for each.
4. Re-read the generated records and the audit rules. Add a check for any failure you find.

## Data sources and limits

- Procedure facts are summarized from published service-public.gouv.fr pages (source URL and verification date
  per entry in `procedures.json`). Check each page's reuse terms before redistributing the text, and refresh
  the facts before relying on them: fees and requirements change.
- One page on proof of address appeared to contradict itself about whether a tax notice is accepted, so that
  document type is deliberately left out of the facts. A person should resolve it from the live page.
- The official pages say nothing about help for people without internet access, so the data does not claim
  any. Answers can only say what the sources say.
- This is an example of a pattern, not a production assistant. Review generated data with native French
  speakers before using it for anything user-facing.
- Judge-based filtering is noisy and can be wrong in both directions. Use a separate judge model from the generator,
  and read a sample of the kept records by hand before training.
