# Adapt Parse to Your Domain

Start by defining the document types, complete extraction targets and success
criteria for your task. The shared SFT and RLVR stages handle model preparation,
token alignment and training; their task-specific behavior comes from the
configured `task`.
The bundled profiles select `nemotron.recipes.parse.dental.task.DentalTask`.
Changing `data_dir` alone retains that dental task.

Implement your own importable task class or factory and select it with
`task=my_domain.task.DocumentTask`. The [task contract below](#task-contract)
describes the hooks, input/output contracts and reward environment wiring.
SFT can use `task=null` for native validation loss without additional task
monitoring. RLVR requires a task and its reward environment.

| Component | What to implement for your domain |
| --- | --- |
| Data preparation | Develop or load images, prompts, full targets and scoring references. Define context groups and independent splits. Follow the [page manifest contract](workflow.md#data-preparation-example). |
| Dataset handoff | Write `train.jsonl`, `monitor.jsonl` and `test.jsonl`; publish the dataset artifact and retain manifest/image identity checks. See [publish a dataset](#publish-a-dataset). |
| SFT monitoring | Reuse Parse image/prompt/target training. Implement `monitor` with your metrics, or use `task=null` for native validation loss only. |
| RL candidates and curriculum | Implement `create_candidates` and `select_candidates` using your scoring criteria; exclude all fixed dataset contexts. |
| RLVR reward | Implement your reward environment and select it through `environment_class`. The shared processor passes reference metadata without interpreting it. |
| Evaluation | Develop correctness criteria, component metrics and reports. Use them in the task's `evaluate` and `summarize` hooks, and adapt the standalone evaluation example consistently. |

`nemotron parse data` and `nemotron parse eval` remain dental reference stages;
they do not accept the training stages' `task` override. Adapt these stages for
your domain alongside the task's monitoring, reward and evaluation hooks.

Prepare the three fixed manifests and dataset artifact through your adapted data
stage. The identity checks expect unique page IDs, disjoint context IDs across
fixed splits, and unchanged manifest/image bytes. Keep variants from the same
source family in one split. The shared training modules do not require a dental
`record`: define the labels consumed by your task and reward environment.
The dental curriculum recomputes context IDs from dental record fields; your
selection hook must enforce the corresponding exclusions for your own schema.

The SFT collator trains on `image`, `prompt` and `target`. Its token alignment
and masking are specific to this Parse architecture. The shared greedy generation
helper requires one prompt per monitor batch, and the RL processor checks each
row's prompt against `task.prompt_file`. Update prompts and scoring references
together when defining a new task.

Choose evaluation metrics before defining the RLVR reward. The optional
[document scorers](rewards.md) cover transcription, boxes,
tables, charts and completion. They are reference building blocks: customers
must choose extraction/matching rules, tolerances, thresholds and reward weights
for their use cases. Connect these decisions to both saved-answer evaluation
and the reward environment, and report individual metrics alongside any scalar
reward. The dental reward does not establish correctness for other domains.

After implementing the data and task components, use the shared SFT/RLVR stages
and artifact handoffs. Check complete target lengths, image dimensions and
generation limits for your documents. Start with a small run and saved-checkpoint
reload, then compare SFT and RLVR on the same independent held-out pages using
your domain's evaluation. The [dental workflow and results](workflow.md) illustrate the reference example.

## Publish a dataset

Prepare a new dataset directory with `train.jsonl`, `monitor.jsonl`, `test.jsonl`
and the referenced images. Each manifest contains one JSON object per line with
the [page fields](workflow.md#data-preparation-example). Keep image paths relative
to the manifest and inside the dataset directory. Choose stable, unique page IDs
and disjoint context groups across the three splits.

After writing all manifests and images, save this snippet as
`publish_dataset.py` in the Nemotron checkout:

```python
from pathlib import Path

from nemotron.recipes.parse.common import artifact, dataset_identity

data_dir = Path("/path/to/prepared-data").resolve()
identity = dataset_identity(data_dir)
artifact(data_dir, "parse/dataset", data_dir, identity=identity)
```

Run it with the stage dependencies:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  uv run --locked --project src/nemotron/recipes/parse/stage1_sft python publish_dataset.py
```

This records manifest and image hashes and split counts in `artifact.json`.
SFT and RLVR verify that identity before consuming the dataset. Keep a completed
dataset unchanged and use a new directory for another dataset version.

## Select a task

Both training stages accept `task`, a fully qualified Python class or factory
that takes no arguments. The factory's module must be importable in the Parse
stage environment, including its worker processes. The bundled profiles select:

```yaml
task: nemotron.recipes.parse.dental.task.DentalTask
```

After implementing your own task, select it in a stage config or CLI override:

```bash
nemotron parse sft task=my_domain.task.DocumentTask data_dir=/path/to/data
nemotron parse rlvr task=my_domain.task.DocumentTask data_dir=/path/to/data
```

Put external task modules on `PYTHONPATH` before launching the CLI. Local
stage execution and Ray workers preserve this import path. For containers,
mount the task code and set its container path in the profile's `env_vars`.
The reward environment must be importable there as well. Add any additional
task dependencies to the relevant stage projects and regenerate their locks.

Pass the appropriate model/output paths as described in the
[recipe guide](workflow.md#run-the-dental-worked-example). Changing `data_dir`
alone retains the configured task. SFT can use `task=null` for the native
validation loss without additional task monitoring. RLVR requires a task with
an explicit reward environment.

## Task contract

Implement the hooks below on the object returned by your task factory. An
SFT-only task needs only `monitor`; an RLVR-only task needs the remaining hooks
and attributes. Paths are `pathlib.Path` objects, page rows and answers are
dictionaries, and numeric metric values must be JSON serializable.

| Hook or attribute | Responsibility |
| --- | --- |
| `monitor(model, processor, rows)` | Return `(predictions, metrics)`: a list of per-page records and a dictionary of aggregate numeric metrics. The SFT trainer adds these to its native validation loss and saves the predictions. The helper `nemotron.recipes.parse.training.sft.predict` provides greedy Parse generation; define scoring in your task. |
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
Default evaluation follows the recorded base/SFT/RLVR lineage and compares all
three checkpoints with the same held-out data and generation/scoring protocol.
The shared lineage helpers record and verify checkpoint and processor file
hashes without interpreting domain labels. SFT records the configured base;
RLVR requires that SFT artifact and propagates its lineage. Preserve the referenced
model directories. Older artifacts without explicit lineage are not reconstructed.

For another domain, adapt those producers and evaluators alongside your task.
In particular, implement consistent correctness criteria in SFT monitoring,
RLVR mining/evaluation, the reward environment and standalone evaluation.
The [optional document scorers](rewards.md) can help implement
these functions, but their thresholds, matching rules and weights need to be
chosen for your use case.
