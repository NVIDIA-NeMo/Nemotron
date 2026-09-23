# `eval/model_eval` — troubleshooting

Symptoms hit in practice and what caused them. Most are defects in the layers
below this step (the launcher's executors, harness containers, checkpoint
exports) rather than in the step itself, but they surface here.

See [README.md](README.md) for installation and normal use.

## Lepton executor: field notes

Where this step's assumptions meet the launcher's Lepton executor. Backend
specific, and it assumes you already know your cluster — on Slurm or Run:ai
skip to *Harness containers*, which applies to any backend.

Everything here was hit in practice while evaluating CPT checkpoints. The
symptoms are mostly silent — jobs report success, or hang with no logs — so
check this list before debugging from first principles.

### Two upstream launcher bugs (nemo-evaluator-launcher 0.2.6)

Both break `execution.type: lepton`. They are upstream defects, not carried as
patches in this repository; direct mode does not use that code path.

1. **Every job dies before evaluating.** The generated launch script does
   `echo "Command: {eval_command}"`, and the command embeds
   `trap "code=$?; source post_cmd.sh; exit $code" EXIT`. The inner quote ends
   the echo string, bash executes `source post_cmd.sh` before that file exists,
   and `set -e` aborts. Fix: `shlex.quote` the interpolation (or drop the echo).
2. **`dataset_dir` cannot mount.** The executor builds
   `{"path": <container>, "mount_from": {"path": <host>}}`, but Lepton's `Mount`
   model is `{path, from, mount_path}` with `mount_path` **required** and no
   `mount_from` field — so it always fails validation with
   `Invalid mount configuration`. Fix: emit `{"path": <host>, "mount_path":
   <container>, "from": <storage>}`.

### Executor behaviours that look like bugs

| Symptom | Cause | Fix |
|---|---|---|
| Job "succeeds", no results anywhere | Executor rewrites `--output_dir /results`, but the lm-eval template emits `--output_path /results`; results land on ephemeral disk and vanish when Lepton reaps the job | ship a `post_cmd` that copies `/results` to the mounted `${output_dir}` |
| Mounted dir is empty inside the container | Task mounts get `/<invocation_id>` appended to the **host** path for isolation | use `dataset_dir` for a verbatim mount (checkpoints, tokenizers) |
| Endpoint returns 401 for every request | The executor ignores `evaluation.env_vars`; the job gets no environment | put env in `execution.lepton_platform.tasks.env_vars` |
| `container image is required` | Lepton executor treats task-key **presence** as an override, so `container: null` overrides the registry with nothing | omit the key, or set a real image string |
| All requests 404 / model not found | `target.api_endpoint.model_id` must equal the endpoint's `--served-model-name` | keep them in one variable |
| `container.command.2 not a string` | The Lepton executor ignores `deployment.command`/`base_command` and builds argv from `deployment.checkpoint_path` | put the model in `checkpoint_path`; put `--trust-remote-code` etc. in `extra_args` |

## Harness containers: quirks that are not the step's fault

| Symptom | Cause | Fix |
|---|---|---|
| Every request 401s; `summary.json` says `failed(1)` | The **sovereign** container's MILU harness runs lm-eval `local-completions`, which reads its bearer token from `OPENAI_API_KEY` and ignores `--api_key_name` | export the key as `OPENAI_API_KEY` too — direct mode does this automatically from the configured `api_key_name` |
| `nemo-evaluator ls` shows no MILU tasks | `sovereign:26.05` has 105 tasks and no MILU; `sovereign:latest` has 127 including `milu_*` for 11 languages. **Both report version `26.5`** | use `:latest` only to discover coverage, then pin the resolved digest; the version string is not a reliable record |
| Long generative task dies with `TimeoutError` after doing most of the work | `request_timeout` is a deadline from task **creation**: lm-eval builds every request as an asyncio task upfront behind a concurrency semaphore, so queued tasks burn the clock while waiting | raise `request_timeout` **and** `parallelism`. Lowering concurrency makes it worse |
| A task is inexplicably ~10x slower per item than its sibling | MILU defaults to `max_gen_toks=1024`; a base model with no stop token generates to the cap (`avg_latency_ms` 15,197 vs 1,766 for the same checkpoint) | set `config.params.max_new_tokens` to something small for MCQA tasks |

**Known upstream limit — long generative runs can abort.** Large MILU splits do
not reliably complete through `nemo-evaluator`. Every request authenticates and
returns 200, then the run aborts with `TimeoutError`. It is not a fixed size
ceiling — the abort point varies between runs, and the fastest run was not the
one that got furthest — so the pattern looks like tail latency: a few requests
never return, retries exhaust, the run dies. Smaller splits complete normally.

None of the settings this step exposes avoid it: raising `request_timeout`,
changing `parallelism`, disabling adapter caching, lowering `max_new_tokens`,
and switching `adapter_config.mode` to `client` were all tried without effect.
Requests travel `lm-eval -> nemo-evaluator's adapter (localhost) -> your
endpoint`, and the timeout is raised in lm-eval's client against that adapter,
which this step does not control.

**Workaround for large splits:** drive the harness directly and bypass the
adapter, e.g. inside the sovereign container `milu-lm-eval --tasks milu_<Lang>
--model local-completions --model_args "base_url=<url>,model=<handle>,
tokenizer=<path>,num_concurrent=32" --num_fewshot 5`. Results obtained that way
do not come from this step and should be labelled accordingly.

**Diagnosing a failed task.** `summary.json` records only `ok` /
`failed(<code>)`, which cannot distinguish an auth failure from a timeout. Two
places carry the detail:

- `<output_dir>/<task>/harness.log` — the harness's own stdout/stderr, written
  by direct mode so it survives the job being reaped.
- `<output_dir>/<task>/eval_factory_metrics.json` →
  `response_stats.status_codes` and `successful_count`. Note this file
  **accumulates across runs** if a directory is reused; direct mode refuses a
  non-empty task directory for that reason (`overwrite=true` replaces it).

## Checkpoint tokenizers

HF exports produced by the NeMo tokenizer-extension pipeline declare
`"tokenizer_class": "TokenizersBackend"`, which stock `transformers` cannot
import — so lm-eval fails with *"Tokenizer class TokenizersBackend does not
exist"*. Copy `tokenizer.json` / `tokenizer_config.json` /
`special_tokens_map.json` to a side directory, rewrite `tokenizer_class` to
`PreTrainedTokenizerFast`, drop `auto_map`, and mount that via `dataset_dir`.
The vocabulary is unchanged; only the loader class differs. Fixing this in the
export step would make the checkpoints portable to any harness container.

## Coverage

`nemo-evaluator-launcher ls tasks` is the source of truth for **launcher** mode
(421 tasks, 23 harness containers). It does not list NeMo **Gym** benchmarks —
no task routes to the `nemo-gym` container in 0.2.6.

In **direct** mode the registry is whatever the harness image ships, so the
launcher's task list does not bound you. MILU is the worked example: absent from
the launcher registry entirely, but present as `milu_<Language>` (11 languages,
5-shot and `_0_shot`) in `nvcr.io/nvidia/eval-factory/sovereign:latest`. Run
`nemo-evaluator ls` inside an image to see its real coverage.
