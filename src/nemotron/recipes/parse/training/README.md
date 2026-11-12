# Reusable Parse training

The SFT and RLVR stages use the modules in this directory for model preparation,
manifest loading, token alignment, framework configuration and training. These
components support the Nemotron Parse architecture and do not interpret dental
labels or define a domain's correctness criteria.

Data preparation, monitoring, rewards and evaluation require task-specific
implementations. The [dental task](../dental/task.py) is a reference example;
develop these components for your own documents and business requirements.

## Select a task

Both training stages accept `task`, a fully qualified Python class or factory
that takes no arguments. The factory's module must be importable in the Parse
runtime, including its worker processes. The bundled profiles select:

```yaml
task: nemotron.recipes.parse.dental.task.DentalTask
```

After implementing your own task, select it in a stage config or CLI override:

```bash
nemotron parse sft task=my_domain.task.DocumentTask data_dir=/path/to/data
nemotron parse rlvr task=my_domain.task.DocumentTask data_dir=/path/to/data
```

Pass the appropriate model/output paths as described in the
[recipe guide](../README.md#run-the-dental-worked-example). Changing `data_dir`
alone retains the configured task. SFT can use `task=null` for the native
validation loss without additional task monitoring. RLVR requires a task with
an explicit reward environment.

## Task contract

The [Python protocols](task.py) define the hooks. An SFT-only task needs only
`monitor`; an RLVR-only task needs the remaining hooks and attributes.

| Hook or attribute | Responsibility |
| --- | --- |
| `monitor(model, processor, rows)` | Return `(predictions, metrics)`: a list of per-page records and a dictionary of aggregate numeric metrics. The SFT trainer adds these to its native validation loss and saves the predictions. [Greedy Parse generation](sft.py) is available as `predict`; define scoring in your task. |
| `create_candidates(output, seed=..., contexts=...)` | Create or load candidate pages, placing their images beneath `output`. Return rows with absolute image paths and stable IDs. |
| `select_candidates(candidates, answers, pages=..., seed=..., excluded=...)` | Select the requested training pages using your task's scores. Reject duplicate contexts and overlap with the excluded train/monitor/test pages. |
| `evaluate(model, rows, output, samples=..., processor_source=...)` | Generate, score and save predictions. Return ordered groups with each page's `id` and its `answers`. Use the same protocol before and after training, and release GPU resources before returning. |
| `summarize(answers)` | Return a JSON-serializable summary using your correctness criteria and component metrics. |
| `prompt_file` | Path to the extraction prompt used by the RL data processor. Each row's `prompt` must match it. |
| `environment_class` | Fully qualified Ray actor implementing NeMo RL's `EnvironmentInterface`, including your scalar reward. |
| `reward_metric` | Optional component metric name to expose as `reward` when the native logger has not supplied one; set to `None` to disable this fallback. |

The common page fields are `id`, `image`, `prompt` and `context_id`; SFT also
requires a complete `target`. Add whatever scoring references your task needs.
The RL processor passes the complete row in `extra_env_info` to the reward
environment, keeping those references out of the prompt. There is no required
dental `record` schema in the training modules.

The bundled GRPO config uses the `parse_page` processor and the `parse-page`
and `parse-page-validation` environment names. The driver binds both names to
the task's `environment_class`. Configure its training and validation settings
under the corresponding `env` entries. Environments may append component metrics
as JSONL records of the form `{"samples": [{"metric": 0.5}]}` to `metrics_file`;
the logger aggregates them for the matching `phase`. This component log is
optional; reward computation uses NeMo RL's environment return contract.

## Reference data and evaluation stages

`nemotron parse data` and `nemotron parse eval` remain dental reference stages.
For another domain, adapt those producers and evaluators alongside your task.
In particular, implement consistent correctness criteria in SFT monitoring,
RLVR mining/evaluation, the reward environment and standalone evaluation.
The [optional document scorers](../document_rewards/README.md) can help implement
these functions, but their thresholds, matching rules and weights need to be
chosen for your use case.
