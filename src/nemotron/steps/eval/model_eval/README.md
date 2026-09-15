# Model Evaluation (NeMo Evaluator)

Use `eval/model_eval` to score a model on benchmark suites. The model is always
reached over HTTP, so the first question is who serves it:

- **`direct`** (`-c direct`) — you host the endpoint; the harness runs inside
  this step's own job. Scheduling is whatever `--batch` profile you pass, so
  every Nemotron backend works, including Run:ai.
- **`launcher`** (`-c default`) — hand the run to NeMo Evaluator Launcher,
  which can deploy a Megatron checkpoint for you on local or Slurm.

> **Is your benchmark in the launcher registry?** Check with
> `nemo-evaluator-launcher ls tasks`. If it is, use this step — including for
> instruct models (see the `instruct_en` suite). If it exists only in NeMo Gym,
> use [`gym_eval`](gym_eval/README.md), which also handles agentic and tool-use
> benchmarks this step cannot: it parses tool calls and reasoning traces, and
> serves an HF checkpoint with vLLM in-job.

Use this README to run an eval; use [REFERENCE.md](REFERENCE.md) for mechanics,
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) for failures, and `step.toml` for the
full parameter, strategy and error list.

## Inputs And Outputs

- Consume a hosted OpenAI-compatible endpoint, or `checkpoint_megatron`
  (launcher mode only).
- Produce `eval_results` in `output_dir`: one directory per task, plus
  `summary.json`, `run_manifest.json`, and `failures.txt` if anything failed.

## CLI And Overlay Knobs

Direct mode is configured entirely by environment; you should not need to write
a config file.

- `EVAL_ENDPOINT_URL`: OpenAI-compatible URL, e.g. `https://host/v1/completions`.
- `EVAL_MODEL_HANDLE`: must equal the server's `--served-model-name`, or every
  request 404s.
> **Backend support differs by mode.** Launcher-managed Lepton deployment is
> experimental and there is no launcher Run:ai executor; direct mode works on
> every backend because you host the endpoint. See
> [REFERENCE.md](REFERENCE.md) before choosing a backend.

- `EVAL_TOKENIZER`: HF repo id or local path. **Required** — the harness
  tokenizes client-side for chat and completions alike, and otherwise tries to
  load your served-model-name as an HF repo and 404s.
- `EVAL_RESULTS_DIR`: durable storage, not container disk.
- `EVAL_API_KEY_NAME` / the variable it names: the endpoint token.
- `EVAL_HARNESS_IMAGE`: the image whose tasks you want. Pin by digest for
  anything you publish.
- `EVAL_LIMIT_SAMPLES`: deterministic first-N. Smoke with `5`.

Task selection is `-t <task>`, repeatable. Anything else is a Hydra override,
e.g. `evaluation.nemo_evaluator_config.config.params.task=mmlu_prox_hi`.

## Shipped Suites

Each is a few lines on top of `direct.yaml`; pass one with `-c <name>` and
override anything with `-t` or a dotted key.

| Config | Endpoint | Tasks |
|---|---|---|
| `base_en` | completions | `adlr_arc_challenge_llama_25_shot`, `hellaswag`, `gsm8k` |
| `mmlu_prox` | completions | `mmlu_prox_completions`, one language per run |
| `milu` | completions | `milu_Hindi`, `milu_English` — sovereign image |
| `instruct_en` | chat | `ifeval`, `mmlu_instruct`, `gsm8k_cot_instruct` (+ opt-in code benchmark) |
| `mmlu_prox_chat` | chat | `mmlu_prox_chat`, one language per run |
| `tiny_chat` | chat | one task, one sample — plumbing smoke test |

The `mmlu_prox*` suites read `MMLU_PROX_LANG` (default `en`):

```bash
MMLU_PROX_LANG=hi uv run nemotron steps run eval/model_eval \
  -c mmlu_prox --batch <profile>
```

Base models want the completions suites, instruct models the chat suites.
**Both need `EVAL_TOKENIZER`** — lm-evaluation-harness loads a tokenizer
client-side either way. Reach for [`gym_eval`](gym_eval/README.md) when a benchmark is
Gym-only, or when it needs tool-call or reasoning-trace parsing.

## Finding A Benchmark

The suites above are a starting point, not the catalogue. This step runs
whatever the harness image ships, so there are three places to look, in
increasing order of authority:

1. **Upstream docs** — the human-readable catalogue:
   [Built-in Benchmarks](https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks.html)
   and the [NeMo Evaluator repo](https://github.com/NVIDIA-NeMo/Evaluator).
   Good for "does something like X exist".
2. **The launcher registry** — searchable, and the single most useful command
   in this step:

   ```bash
   uv run nemo-evaluator-launcher ls tasks                  # every routable task
   uv run nemo-evaluator-launcher ls task <name> --json     # ← run this before every new task
   ```

3. **The image itself** — authoritative for direct mode, because the registry
   is whatever that image ships:

   ```bash
   docker run --rm <harness-image> nemo-evaluator ls
   ```

Prefer (3) before a real run. Task sets differ between tags of the same image
family, and some benchmarks are in an image but **not** in the launcher
registry at all — MILU is the worked example, which is why `-c milu` overrides
the harness image.

### Read `ls task --json` before you run anything

It answers, in one call, every question that otherwise costs a failed job:

```jsonc
{
  "name": "mmlu_prox_completions",
  "container": "nvcr.io/nvidia/eval-factory/lm-evaluation-harness:26.03",
  "container_digest": "sha256:9593456f…",        // pin THIS for published numbers
  "defaults": {
    "config": {
      "params": {
        "task": "mmlu_prox",                     // ⚠ the FAMILY, all 29 languages
        "parallelism": 10,
        "request_timeout": 30,                   // seconds, often too low
        "extra": { "tokenizer": null, "tokenizer_backend": "None" }
      },
      "supported_endpoint_types": ["completions"] // chat vs completions
    }
  }
}
```

| Field | Why it matters |
|---|---|
| `supported_endpoint_types` | `chat` and `completions` are different tasks. Using the wrong one fails, or silently scores something else. |
| `params.task` | If it differs from the task name, it is a **family** — pin the subset or the run never finishes. |
| `container` / `container_digest` | The image the task really needs, and the digest to pin for reproducibility. |
| `extra.tokenizer` | `null` here means *you* must supply one for completions. |
| `request_timeout`, `parallelism` | Registry defaults are tuned for hosted APIs, not your endpoint. |
| `extra.num_fewshot` | The shot count a published number assumes. |

Anything you find here maps onto `evaluation.nemo_evaluator_config.config.params`
— as a suite config, or a dotted override on the command line.

## Config Nuances

- **One job per endpoint, all tasks inside it.** Loop over checkpoints, not
  benchmarks.
- **Multi-subset tasks must be pinned.** `mmlu_prox_completions` defaults to
  all 29 languages (~350k requests); it does not error, it never finishes.
  `config.params.task` is global to a run, so give such a task its own
  invocation and `EVAL_RESULTS_DIR`.
- **Different benchmarks live in different images.** Swap
  `EVAL_HARNESS_IMAGE`; `nemo-evaluator ls` inside an image lists what it ships.
- **Adapter caching is on by default** and should stay on — without it the
  harness buffers every request in memory and dies client-side on large tasks.
- `--batch` needs a `*_eval_direct` profile. If yours predates it, copy the
  profile from `steps/env/env_toml/config/<backend>.yaml` into your `env.toml`.

## Run It

Host the model first — direct mode never creates or destroys an endpoint. See
[REFERENCE.md](REFERENCE.md#standing-the-endpoint-up-yourself) for the Lepton
and Run:ai commands.

```bash
export EVAL_ENDPOINT_URL=https://<host>/v1/completions
export EVAL_MODEL_HANDLE=<served-model-name>
export EVAL_TOKENIZER=<hf-repo-id-or-path>
export EVAL_API_KEY_NAME=ENDPOINT_TOKEN
export ENDPOINT_TOKEN=<token>
export EVAL_RESULTS_DIR=<durable path>/smoke
```

Smoke it. One cheap job proves endpoint, token, handle, tokenizer and mount:

```bash
EVAL_LIMIT_SAMPLES=5 uv run nemotron steps run eval/model_eval \
  -c direct --batch <profile> -t hellaswag
```

Then the real run:

```bash
EVAL_RESULTS_DIR=<durable path>/en uv run nemotron steps run eval/model_eval \
  -c direct --batch <profile> \
  -t adlr_arc_challenge_llama_25_shot -t hellaswag -t gsm8k
```

A multi-subset benchmark takes its own invocation:

```bash
EVAL_RESULTS_DIR=<durable path>/prox_hi uv run nemotron steps run eval/model_eval \
  -c direct --batch <profile> -t mmlu_prox_completions \
  evaluation.nemo_evaluator_config.config.params.task=mmlu_prox_hi
```

For a Megatron checkpoint you want the tool to deploy, use launcher mode:

```bash
uv run nemotron steps run eval/model_eval -c default \
  deployment.checkpoint_path=<run>/iter_<n> \
  run.env.launcher_executor=slurm
```

## Repository Layout

- Manifest: `step.toml`
- Runner: `step.py`
- Runtime helpers: `runtime.py`
- Configs: `config/direct.yaml` (direct-mode base), `config/default.yaml`
  (launcher + Megatron deploy), and the shipped suites above
- Reference: `REFERENCE.md` · Failures: `TROUBLESHOOTING.md`

## Guardrails

- Always smoke with `EVAL_LIMIT_SAMPLES` before a full split.
- Don't compare scores across different endpoint types, tokenizers, or
  generation settings.
- Pin `EVAL_HARNESS_IMAGE` by digest for any number you publish;
  `run_manifest.json` records whether you did.
- A tokenizer-extended checkpoint must use its own tokenizer, not the base one.
- `limit_samples` results are a subset, not a full-split score. Say so.
- Don't put raw tokens in YAML, profiles, or command output.
- Inspect a handful of generations before trusting aggregate metrics.
