# Model Evaluation — Reference

Detail behind [README.md](README.md). Read the README first; come here for the
mechanics of a specific piece. Symptom-to-fix tables live in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Who owns the endpoint

This step evaluates a model over HTTP. Which layer *creates* that endpoint
depends on the mode and backend, and the honest current picture is:

| Backend | Launcher mode (`run.env.launcher_executor=…`) | Direct mode (`--batch lepton_eval_direct`) |
|---|---|---|
| Local | `local` — launcher can start a configured model server | runs the harness against a URL you provide |
| Slurm | `slurm` — launcher can start the server, health-check it, evaluate, clean up | no server lifecycle — supply a URL |
| Lepton | `lepton` — **experimental**, see below | no server lifecycle — supply a URL |
| Run:ai / DGX Cloud | no launcher executor exists | no server lifecycle — supply a URL |

**Direct mode never creates or tears down an endpoint.** You host the model
yourself — see *Standing the endpoint up yourself* below — and pass
`EVAL_ENDPOINT_URL`, `EVAL_MODEL_HANDLE` and the credential.

> **Launcher-managed Lepton deployment is experimental.** The launcher's Lepton
> executor accepts only specific deployment types (vLLM, SGLang, NIM, or none) —
> not the `generic` type this step's `default.yaml` uses — no complete Lepton
> launcher config ships here, created endpoints are **not** torn down after a
> successful run, and a multi-task run can create one GPU endpoint per task.
> Treat Lepton as: host the endpoint yourself, evaluate it with direct mode.

> **Two schedulers, two fields.** Launcher mode has an outer scheduler (the
> runspec executor that runs this step) and an inner one (the backend NeMo
> Evaluator Launcher submits to). They are separate config fields on purpose:
>
> - `run.env.executor` — outer; overridden by `--run` / `--batch`.
> - `run.env.launcher_executor` — inner; `local`, `slurm` or `lepton`.
>
> Pointing both at one field made "outer Slurm" silently mean "launcher submits
> again from inside the allocation", and left "outer local, launcher Slurm" —
> the normal way to use launcher mode on a cluster — inexpressible:
>
> ```bash
> # submit from the login node; the launcher does the sbatch
> uv run nemotron steps run eval/model_eval -c default \
>   run.env.launcher_executor=slurm run.env.account=<acct> run.env.partition=<part>
> ```
>
> Use direct mode when you want the outer `--batch` profile to be the only
> scheduler.

> **Auto-squash does not run for launcher mode.** Squashing Docker images to
> `.sqsh` on a remote cluster is keyed to `run.env`, and the generic runner
> strips `run.env` before invoking the step — so `_maybe_auto_squash` returns
> immediately in a submitted run, whatever the executor. On a Slurm cluster
> that cannot pull from your registry, pre-squash the deployment and evaluation
> images yourself and reference the `.sqsh` paths in the config. Direct mode is
> unaffected: the harness image is the step's own container, which the `--batch`
> profile handles like any other step's.

## Standing the endpoint up yourself

Direct mode needs a URL. Any OpenAI-compatible server will do, hosted however
your platform hosts things — that part is yours. What the endpoint has to
satisfy for this step:

| Requirement | Why |
|---|---|
| OpenAI-compatible `/v1/completions` or `/v1/chat/completions` | the harness speaks nothing else |
| `--served-model-name` **equals** `EVAL_MODEL_HANDLE` | otherwise every request 404s and `summary.json` shows only `failed(N)` |
| a reasoning parser, for a reasoning model | without it the reasoning trace is returned as content and generative benchmarks score the model thinking out loud — IFEval and HumanEval come back *plausibly* bad, so nothing looks wrong |
| a tool-call parser, for tool-use benchmarks | same failure mode, for tool calls |
| context length ≥ the longest prompt a task builds | few-shot prompts are longer than they look |
| reachable from wherever the eval job runs | direct mode only sends HTTP |

A minimal vLLM server, whatever you wrap it in:

```bash
vllm serve <model-path-or-repo-id> \
  --served-model-name my-ckpt --port 8000 \
  --tensor-parallel-size <n> --max-model-len 8192 \
  --trust-remote-code \
  --reasoning-parser <parser>        # reasoning models only
```

Then `EVAL_ENDPOINT_URL=<url>/v1/completions` and `EVAL_MODEL_HANDLE=my-ckpt`.

**Verify before you evaluate.** A 200 is not evidence the response is
scoreable — read one raw completion:

```bash
curl -sS -H "Authorization: Bearer $ENDPOINT_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-ckpt","prompt":"The capital of France is","max_tokens":16}' \
  <url>/v1/completions
```

Coherent continuation, no reasoning trace leaking into `text`, model id as
expected. That check costs seconds and catches the parser mistake above, which
otherwise surfaces as a bad score rather than an error.

## Several checkpoints, several benchmarks

The common case. Direct mode runs **one job per endpoint, all tasks inside it**,
so the loop is over checkpoints, not over benchmarks:

```bash
# fixed for the whole sweep
export EVAL_HARNESS_IMAGE=nvcr.io/nvidia/eval-factory/lm-evaluation-harness@sha256:<digest>
export EVAL_ENDPOINT_TYPE=completions      # base models; `chat` for instruct
export EVAL_API_KEY_NAME=ENDPOINT_TOKEN
export ENDPOINT_TOKEN=<token>

for CKPT in run-a run-b run-c; do
  export EVAL_MODEL_HANDLE=$CKPT           # == the server's --served-model-name
  export EVAL_ENDPOINT_URL=https://$CKPT.example.com/v1/completions
  export EVAL_RESULTS_DIR=/mnt/shared/eval/$CKPT
  uv run nemotron steps run eval/model_eval -c direct --batch lepton_eval_direct \
    -t adlr_arc_challenge_llama_25_shot -t hellaswag -t mmlu_prox_completions
done
```

Each checkpoint gets its own `output_dir` with one subdirectory per task, plus
`summary.json` and `run_manifest.json`. Comparing checkpoints then means
comparing manifests: same harness image, same merged `params`, different model.

Two things this step does **not** do:

- **It does not serve your checkpoints.** Direct mode needs a URL. Stand each
  endpoint up first — see *Standing the endpoint up yourself* above — or use
  launcher mode, which can deploy a Megatron checkpoint itself.
- **It does not sweep checkpoints for you.** The loop above is the interface.

Smoke the first checkpoint with `EVAL_LIMIT_SAMPLES=5` before looping: one cheap
job proves endpoint, token, model handle, tokenizer and mount are all wired.

## Multi-subset tasks: pin the subset, always

Some tasks are a whole family behind one name. `mmlu_prox_completions` defaults
to **all 29 languages** — roughly 350k requests. It does not error and it does
not warn; it simply never finishes. `global_mmlu_full` and `global_mmlu` behave
the same way.

Pin the subset on the command line — no config file needed:

```bash
uv run nemotron steps run eval/model_eval -c direct --batch lepton_eval_direct \
  -t mmlu_prox_completions \
  evaluation.nemo_evaluator_config.config.params.task=mmlu_prox_hi
```

`config.params.task` is global to the run, so give a multi-subset task its own
invocation and its own `EVAL_RESULTS_DIR`. To pin per task instead, in a config
that mixes it with other benchmarks:

```yaml
evaluation:
  tasks:
    - name: mmlu_prox_completions
      nemo_evaluator_config: {config: {params: {task: mmlu_prox_hi}}}
```

Check the task's real default before you run it — `ls task <name> --json` shows
it (`"task": "mmlu_prox"`, i.e. everything). See *Discovering a task's real
defaults*.

## The adapter proxy, and why caching is not optional

Between the harness and your endpoint sits an adapter proxy, configured under
`evaluation.nemo_evaluator_config.target.api_endpoint.adapter_config`. Direct
mode forwards this namespace whole, alongside `config.params.*`.

`use_caching: true` is on by default in `direct.yaml` and should stay on.
Without it the harness holds every request and response for a task in memory:
around 30k requests it dies client-side, on a large CPU shape, **while the
endpoint is still returning 200 to every call** — so the symptom points at the
model server when the problem is the eval job. MILU-Hindi at full split is past
that line; it completed at `limit_samples: 2000`.

Adapter keys are deliberately **not** allowlisted (unlike `config.params.*`,
where an unknown key is rejected). The interceptor set is open-ended and version
dependent, and an unknown adapter key is inert rather than a silent change to
how the model is scored. Leaving `output_dir: null` gives each task its own
`<output_dir>/<task>/adapter`, so two tasks in one run cannot collide.

`run_manifest.json` records the merged `adapter_config` per task next to
`params`, so a cached run and an uncached one are distinguishable after the
fact. The adapter's own `endpoint_type` defaults to `chat`; direct mode fills it
in from `target.api_endpoint.type` unless you set it explicitly, so it cannot
silently disagree with the endpoint being called.

> Caching removes the memory *ceiling*; it has not been shown to make an
> unbounded full split finish. The largest verified run is MILU-Hindi at
> `limit_samples: 2000` with caching on (peak 4.55 GB). `limit_samples` is a
> deterministic first-N, so the subset is identical across checkpoints and
> scores stay comparable within an ablation — but it is not a full-split score,
> and should not be reported as one.

## Discovering a task's real defaults

> **Start with `nemo-evaluator-launcher ls task <name> --json`.** It reports the
> endpoint type, the container and its digest, the real few-shot count, and
> whether the name is one task or a family of subsets — see *Finding A
> Benchmark* in [README.md](README.md) for an annotated example.

Task defaults differ substantially — few-shot counts, endpoint type, whether
log-probabilities are required. Inspect the exact image you intend to pin,
because the registry is per-image:

```bash
nemo-evaluator-launcher ls tasks --from <pinned-image>
nemo-evaluator-launcher ls task <task-name> --from <pinned-image> --json
# direct mode / harness-only tasks:
docker run --rm <pinned-image> nemo-evaluator ls
```

Map what you find onto `evaluation.nemo_evaluator_config`. A useful split:

- **global** — `request_timeout`, `max_retries`, `parallelism`, `limit_samples`
- **per task** — `task` (subset/language), `temperature`, `top_p`,
  `max_new_tokens`, and anything harness-specific under `extra:`
  (for example `extra.num_fewshot`, `extra.tokenizer`)

Direct mode **rejects** an unrecognised top-level param rather than dropping it,
so a typo fails loudly instead of silently changing generation settings.

## `run_manifest.json`

Every direct-mode run writes `run_manifest.json` into `output_dir` **before the
first task starts**, so a preempted job still records what it was doing:

```json
{
  "schema_version": 1,
  "generated_at": "2026-09-04T11:02:31+00:00",
  "mode": "direct",
  "model": {
    "id": "my-model",
    "endpoint_url": "https://host/v1/completions",
    "endpoint_type": "completions",
    "api_key_name": "ENDPOINT_TOKEN"
  },
  "harness": {
    "image": "nvcr.io/nvidia/eval-factory/lm-evaluation-harness@sha256:...",
    "image_pinned_by_digest": true,
    "packages": {"nemo-evaluator": "..."}
  },
  "code": {"nemotron_version": "...", "git_sha": "...", "git_dirty": false},
  "tasks": {"hellaswag": {"output_dir": "...", "params": {...}, "command": [...]}},
  "output_dir": "...",
  "summary_path": ".../summary.json"
}
```

`params` is the **merged** global + per-task set, i.e. what the task actually
ran with. `harness` says what computed the score; `code` says what decided its
inputs — its three fields are always present, `null` when unavailable, so "not a
git checkout" is distinguishable from provenance quietly dropped.

`image_pinned_by_digest` requires a real digest — `sha256:` followed by exactly
64 hex characters. If it is `false`, the run is not reproducible; re-run against
a digest-pinned image before citing the number.

The manifest is safe to attach to a report. The endpoint URL is stored with
userinfo and query string stripped, only the credential's *name* is recorded,
and credential-valued parameters (`api_key`, `*_token`, `password`, …) are
replaced with `***` in both `params` and `command`. The command echoed to the
job log is scrubbed identically. `tokenizer` is deliberately not treated as a
credential, and `api_key_name` is kept because it is a variable name.

Per-task outcomes stay in `summary.json`; a dry run writes
`run_manifest.dry-run.json` so it cannot overwrite a real one. If any task
fails, `failures.txt` collects the tail of each failed task's `harness.log` in
one place — on a cluster whose job logs vanish after termination, that is the
difference between one `cat` and a job per task.

**Direct mode does not log metrics to W&B**, by design: it writes files and does
not require credentials. Nothing in the eval path requests W&B, so a `wandb`
login on the submitting host is neither read nor forwarded. To publish results,
upload `run_manifest.json` and `summary.json` yourself, or use launcher mode
with `execution.auto_export` and `export.wandb` enabled.

## What a run writes

Everything lands under `EVAL_RESULTS_DIR`. Reading it back is a property of your
platform, not of this step — point it at durable storage and retrieve it however
you normally would.

```
<output_dir>/
├── summary.json          per task: "ok" or "failed(<exit code>)"
├── run_manifest.json     what ran: model, endpoint (redacted), harness image
│                         and whether it was digest-pinned, merged per-task
│                         params and adapter config, Nemotron version + git SHA
├── failures.txt          written only on failure: the tail of each failed
│                         task's harness.log, so one file explains the run
└── <task>/
    ├── harness.log       the harness's own stdout and stderr
    ├── results_*.json    scores
    ├── samples_*.jsonl   per-sample requests and responses
    ├── run_config.yml    the config the harness resolved
    └── adapter/          proxy cache and request/response logs
```

A dry run writes `summary.dry-run.json` and `run_manifest.dry-run.json` so it
cannot overwrite a real one.

**Start with the three top-level files.** `summary.json` says what failed,
`failures.txt` says why, `run_manifest.json` says what ran. They are small and
answer most questions; reach into a task directory only when they do not.

## Reproducibility: pin the harness by digest

A harness image is the source of truth for task definitions, prompt formatting
and dependencies, so a mutable tag makes a run unreproducible. Two tags of the
same image family have been observed to expose different task sets while
reporting the same internal version, so the tag alone is not a reliable record.

Use `:latest` only to discover what exists, then record the digest and pin it:

```bash
# discover
docker run --rm <image>:latest nemo-evaluator ls
# resolve the immutable digest
docker buildx imagetools inspect <image>:latest --format '{{.Manifest.Digest}}'
# pin it for the run you intend to cite
export EVAL_HARNESS_IMAGE=<image>@sha256:<digest>
```

Report the digest alongside any number you publish.

## The `--batch` profile

A direct-mode profile ships for each backend, so there is nothing to author:

| Backend | Profile |
|---|---|
| Lepton | `lepton_eval_direct` |
| Slurm | `slurm_eval_direct` |
| Run:ai / DGX Cloud | `dgxcloud_eval_direct` |

All three are **CPU-only** — the harness drives an endpoint you host yourself,
so the GPUs are wherever the model is served.

> **A profile you already have an `env.toml` for will not gain these.**
> `env.toml` is generated once by `steps/env/env_toml` and then gitignored, so
> profiles added to the template do not reach a file already on disk. Check with
> `grep '^\[lepton_eval_direct\]' env.toml`; if it is missing, copy the profile
> across from `steps/env/env_toml/config/<backend>.yaml`. Regenerating with
> `force=true` rewrites the whole file and discards site values.

If you write your own, the one rule that is ours: direct mode resolves its
config **inside the job**, so the profile must forward every `${oc.env:...}` the
config reads. Exporting a variable in your shell only affects submission.

```toml
[my_direct_profile]
executor = "<your backend>"
# No container_image: direct.yaml owns the harness image (see below).
# The harness image is not a Nemotron image and ships none of its deps.
pip_extras = ["typer", "rich", "pydantic-settings", "omegaconf", "tomli", "wandb"]
env_vars = { EVAL_ENDPOINT_URL = "...", EVAL_MODEL_HANDLE = "...", EVAL_RESULTS_DIR = "...", EVAL_ENDPOINT_TYPE = "...", EVAL_API_KEY_NAME = "...", EVAL_LIMIT_SAMPLES = "...", EVAL_TOKENIZER = "...", HF_HOME = "..." }
```

Resource sizing is yours, with one trap worth knowing: a CPU-only profile that
inherits a GPU base may also inherit a shared-memory request no CPU shape can
satisfy, and the scheduler rejects the submission before a job exists. Neither
config generation nor `--dry-run` catches that, because neither talks to the
scheduler.

## What `--dry-run` does and does not prove

`nemotron steps run -d` compiles the config, prints it, and **returns before the
step executes**. Nothing in this step runs, so it cannot tell you whether the
launcher will accept the config — and the launcher validates at submission time.
A clean `-d` is not evidence that a launcher-mode run will schedule.

This bites hardest in launcher mode, where `inject_wandb_env_mappings()` mutates
the config *after* compilation: a config that is valid in the file and valid in
the `-d` output can still be rejected at submission. The step now runs the
launcher's own schema checks after that injection, so `dry_run=true` fails
identically to a real run.

| | compiles config | launcher schema | scheduler accepts |
|---|---|---|---|
| `-d` | yes | **no** | no |
| `dry_run=true` | yes | yes | no |
| real run | yes | yes | yes |

Nothing short of a real submission proves the scheduler will accept the job —
resource shapes in particular are only checked there.
