# Multimodal Embedding Fine-Tuning EA Guide

The `mistral3-vl` preview profile connects one canonical JSONL source to a
portable `image_and_text` training view, native hard-negative mining, training,
checkpoint reload, and a fresh retrieval evaluation.

## Prepare your input

Stage 0 expects a **UTF-8 JSONL file plus local page images**, not a PDF directory
or pre-generated questions. Write one JSON object per line for each independently
retrievable unit, usually one page. Render PDFs and extract/OCR their text before
running the recipe; use your preferred parser. There is no dataset-specific
ingestion or automatic PDF parsing in this profile.

For example, prepare this directory:

```text
my-corpus/
  sources.jsonl
  pages/
    manual-a-001.png
    manual-a-002.png
  contexts.jsonl       # optional explicit generation contexts
```

`sources.jsonl` follows the public `RetrievalSource` contract:

```json
{"unit_id":"manual-a-p1","document_id":"manual-a","text":"The pump operates between 10 and 30 degrees Celsius.","images":["pages/manual-a-001.png"],"page_number":1,"language":"en"}
{"unit_id":"manual-a-p2","document_id":"manual-a","text":"The pressure chart shows the operating range.","images":["pages/manual-a-002.png"],"page_number":2,"language":"en"}
```

| Field | Required | Format and meaning |
|---|---|---|
| `unit_id` | Yes | Nonempty string, unique across the entire input file. Use a stable ID for each page or other retrievable unit; do not include tabs or line breaks. |
| `document_id` | Yes | Nonempty string identifying the source document. Pages from the same document share this ID; use distinct IDs for different documents. This is not a train/test split assignment. |
| `text` | Conditional | String containing the unit's extracted text or OCR, not a generated summary. Defaults to `""`; may be empty when an image is supplied. |
| `images` | Conditional | List containing zero or one local image path. Defaults to `[]`. Use PNG, JPEG, or WebP. Relative paths resolve from the directory containing `sources.jsonl`, not the working directory; absolute paths also work. |
| `page_number` | No | Integer starting at 1 for paginated documents. It records provenance; it does not reorder the input. |
| `language` | No | Nonempty language string, such as `"en"`. Defaults to `"source"` to preserve the source language. |
| `source_uri` | No | String recording the original source location, or `null`. This is provenance only: the recipe does not download this URI. |

At least one of nonblank `text` or a local image is required. The source contract
also accepts image-only and text-only units, for example:

```json
{"unit_id":"chart-b-p1","document_id":"chart-b","images":["pages/chart-b-001.png"]}
{"unit_id":"notes-c-s1","document_id":"notes-c","text":"Store replacement parts in a dry location."}
```

These are alternative examples; create the referenced chart image if you use
that row. For the recipe's `image_and_text` path, supply each page image with its
matching extracted text where available. Image-only/text-only inputs are valid
sources, but usable training records depend on the selected export view and
the evidence supporting each generated query.

Keep units in document reading order. The profile's automatic section
planner preserves input order within each document/language; it does not sort by
`page_number`. Escape embedded newlines in JSON strings as `\n` rather than
splitting a record across lines. Do not wrap the file in a JSON array.

The input loader rejects empty files, malformed records, unknown fields,
duplicate `unit_id` values, units without content, more than one image per unit,
and missing image files. Images must be accessible locally to the process running
Stage 0; HTTP image URLs are not downloaded. Do not add benchmark labels,
questions, answers, negatives, or split assignments to source rows: SDG and the
subsequent preparation stages produce the training artifacts.

Optionally, `contexts.jsonl` can specify which units should be considered together
when generating questions:

```json
{"context_id":"manual-a-operating-limits","unit_ids":["manual-a-p1","manual-a-p2"],"language":"en"}
```

Each context needs a unique nonempty `context_id` and a nonempty `unit_ids` list
referencing existing sources; `language` is optional and defaults to `"source"`.
Explicit contexts replace automatic section grouping and can span documents.
They still undergo the unit/character bounds described below. The profile also
proposes semantic summary combinations by default, including when an explicit
contexts file is supplied. To generate only from your supplied memberships, keep
each context within both bounds and set
`sdg_options.combination_iterations=0`.
They control generation evidence, **not query split groups**. Units not selected
by a context remain in the exported eligible corpus.

After configuring the model endpoints and credentials in the following section,
run with your source file:

```bash
nemotron embed sdg -c mistral3-vl sources_file=/absolute/path/to/my-corpus/sources.jsonl
# Optional explicit contexts:
nemotron embed sdg -c mistral3-vl \
  sources_file=/absolute/path/to/my-corpus/sources.jsonl \
  contexts_file=/absolute/path/to/my-corpus/contexts.jsonl
```

Choose one invocation, not both for the same output directory. No example corpus
is downloaded automatically by this profile.

## Configure and run the preview

Each stage declares dependencies in its own `pyproject.toml`. The CLI selects
the `vl` extra for this profile and the `text` extra for existing text profiles.
These extras preserve their different Transformers requirements. AutoModel uses
immutable commit archives from the companion
[AutoModel PR](https://github.com/NVIDIA-NeMo/Automodel/pull/3915); archive URLs
prevent AutoModel's repository-local uv index policy from replacing the recipe's
CUDA 12.9 Torch source. The retrieval-SDG plugin uses a commit-pinned Git source
from the consolidated public [DataDesignerPlugins multimodal SDG EA candidate](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/pull/92)
(the exact reviewed commit is pinned in the Stage 0 project).
No manually built wheels, private scripts, or private package index are required.
The checked-in locks resolve both mutually exclusive extras from those public
sources.

Stage 0 and Stage 1 use released Data Designer 0.9.1, without the experimental
core patch. The plugin's `retrieval-structured` column accepts one complete bare
or JSON-fenced object, rejects malformed or ambiguous JSON, and delegates schema
validation and bounded correction retries to Data Designer's native structured
generation path. It does not extract JSON from prose, repair values, prune
fields, or turn negative judge decisions into passes. Do not bypass schema
checks or quality gates to compensate for parsing failures. Query generation
also validates the exact requested slot membership within native correction,
before caching; missing, duplicate or extra slots cannot become successful cache entries.

For direct stage invocation, select the same extra, for example:

```bash
uv run --project src/nemotron/recipes/embed/stage1_data_prep --extra vl \
  python src/nemotron/recipes/embed/stage1_data_prep/data_prep.py \
  --config src/nemotron/recipes/embed/stage1_data_prep/config/mistral3-vl.yaml
```

Set `MISTRAL3_VL_EMBED_MODEL` to a checkpoint available to your Hugging Face
credentials, provide sources producing enough independent query groups for both
the 80% training and 20% evaluation splits, and run locally. Documents intentionally
remain shared across query partitions. The `nemotron`
executable must be installed from this same reviewed checkout; use
`uv run --no-sync nemotron` in place of `nemotron` below if needed:

The EA recipe supports embedding fine-tuning; multimodal reranking is deferred.
In-batch negatives are disabled because unrolled queries can share positives.
Keep `do_distributed_inbatch_negative=false` until positive-ID masking is supported;
training uses each query's mined negatives. The positive-ID unrolling changes are
deferred. Mining, training, and local evaluation use the same 512-token query
and 4096-token passage limits. Remote model code is disabled by default.

```bash
export NVIDIA_API_KEY=your_endpoint_credential
# Optional: override the public https://integrate.api.nvidia.com/v1 endpoint.
# export NVIDIA_API_BASE_URL=https://your-provider.example/v1
export MISTRAL3_SDG_QA_MODEL=your-image-capable-generation-model
export MISTRAL3_SDG_JUDGE_MODEL=your-image-capable-judge-model
export MISTRAL3_VL_EMBED_MODEL=your-org/your-multimodal-embedding-checkpoint

nemotron embed sdg -c mistral3-vl sources_file=/absolute/path/to/sources.jsonl
nemotron embed prep -c mistral3-vl
nemotron embed finetune -c mistral3-vl \
  num_epochs=null max_steps=2 global_batch_size=2 local_batch_size=1 \
  train_n_passages=2 lr_warmup_steps=0 \
  attn_implementation=sdpa optimizer_backend=flash_adamw
test -f output/embed/mistral3-vl-preview/stage2_finetune/checkpoints/LATEST/model/consolidated/config.json
nemotron embed eval -c mistral3-vl eval_base=true eval_finetuned=true eval_nim=false
python -c 'import json; p="output/embed/mistral3-vl-preview/stage3_eval/eval_results.json"; r=json.load(open(p)); assert {"base","finetuned"} <= r.keys()'
```

The model roles are independent:

| Role | Default or required configuration |
|---|---|
| Visual enrichment, descriptions, summaries and query generation | User-supplied `MISTRAL3_SDG_QA_MODEL`; must accept images and structured responses. |
| Summary/query judging and support localization | User-supplied `MISTRAL3_SDG_JUDGE_MODEL`; must accept images and structured responses. |
| Summary embeddings for semantic clustering | [`nvidia/nemotron-3-embed-1b`](https://build.nvidia.com/nvidia/nemotron-3-embed-1b) at the public NVIDIA API, using passage mode. |
| Hard-negative mining, fine-tuning and base evaluation | User-supplied `MISTRAL3_VL_EMBED_MODEL`, a compatible Mistral3 multimodal embedding checkpoint or local path. |
| Fine-tuned evaluation | The local Stage 2 checkpoint. Optional endpoint evaluation requires a compatible user-configured service. |

Hosted generation and judging default to `https://integrate.api.nvidia.com/v1`
and use `NVIDIA_API_KEY`. Choose image-capable models available on
[build.nvidia.com](https://build.nvidia.com), or configure your own compatible
endpoint and model IDs. `NVIDIA_API_BASE_URL` changes the generator/judge endpoint;
per-role `sdg_generator_options` and `sdg_judge_options` can override it independently.
No private inference service or internal model ID is required.

Summary embedding settings are independent of those chat settings. The public
embedding endpoint receives section-summary text, not images, using the plugin's
`multimodal` extra. Override `sdg_options.summary_embedding_model`,
`summary_embedding_endpoint`, `summary_embedding_credential_env` and
`summary_embedding_extra_body` for another service. `input_type: passage` is
required by the default model; `truncate: NONE` rejects oversized inputs.
Embedding responses are validated and cached for resume. For local Sentence
Transformers inference, set `summary_embedding_endpoint: null` and
`summary_embedding_extra_body: null`, supply a Hugging Face model ID or local path,
and optionally set `summary_embedding_revision` and `summary_embedding_device`
(CPU by default). The default summary model is text-only; it is not the
multimodal checkpoint being fine-tuned.

Use `null` to clear inherited request options. An empty mapping (`{}`) is merged
with the profile defaults and does not remove them. For example, with the public CLI:

```bash
nemotron embed sdg -c mistral3-vl sources_file=/absolute/path/to/sources.jsonl \
  sdg_options.summary_embedding_endpoint=null \
  sdg_options.summary_embedding_extra_body=null \
  sdg_options.summary_embedding_model=your-org/your-local-summary-embedder
```

For direct `data_prep.py` invocation, pass dictionary overrides as a quoted JSON
mapping, for example
`'sdg_options={"summary_embedding_endpoint":null,"summary_embedding_extra_body":null,"summary_embedding_model":"your-org/your-local-summary-embedder"}'`.

The `mistral3-vl` profile selects `sdg_workflow=retrieval_first`: separate visual
enrichment, document descriptions, five-unit section summaries, 20 seeded
semantic clustering iterations, summary grading/selection, then bounded queries.
Sections preserve whole generic units in input order; no page identifiers or
benchmark delimiters are parsed. Language groups with fewer than twelve summaries
skip semantic combinations. Detailed standalone-query templates and weighted
text/figure/table instructions follow the adapted reference behavior.
Self-sufficiency sees the original source text. Answer leakage and observed
query labels are judged separately using only the query. Relevance and positive
localization see the original source text/images. Requested style is
diagnostic. No generated answers, dataset loaders, benchmark-specific repairs,
or preconverted handoff bypasses are involved. Existing text profiles retain
`legacy_qa` and their prior behavior.

For explicit multi-page or cross-document questions pass
`contexts_file=/absolute/path/to/contexts.jsonl`, whose rows have `context_id`,
`unit_ids`, and optional `language`. Memberships are partitioned into at most
`sdg_max_units_per_context` units (default 8), never truncated. A single unit
over the character bound must be preprocessed by the caller. Context membership
does not imply query grouping.
The complete eligible corpus, including unselected distractors, is exported.

The producer writes immutable state under `<output_dir>/multimodal`, which is
`stage0_sdg/multimodal` in the profile. `corpus_id` supplies the dataset identity
and `sdg_batch_size` controls request batching; legacy QA settings `artifact_path`,
`dataset_name`, and `buffer_size` do not control this workflow. Stage 0 owns the
exported train/evaluation split. Stage 1 validates and mines that existing split;
its legacy conversion ratios and quality filter do not apply to portable input.
The profile uses local native AutoModel mining, so endpoint-only mining settings
are omitted.

Inspect failure/attempt evidence before explicitly choosing `resume=always`. Changed
sources, settings or code cannot reuse the run. Stage 0 validates the full
portable bundle before publishing its top-level `generation_result.json`;
Stage 1 performs the same full validation before native AutoModel mining.
No HNM ranking/negative selection or training optimizer behavior changes here.

`sdg_options` forwards the public producer's generic controls, with validation by
the same public config model. For example:

```yaml
sdg_options:
  context_strategy: sections
  section_size: 5
  combination_iterations: 20
  summary_embedding_model: nvidia/nemotron-3-embed-1b
  summary_embedding_endpoint: https://integrate.api.nvidia.com/v1
  summary_embedding_credential_env: NVIDIA_API_KEY
  summary_embedding_extra_body:
    input_type: passage
    truncate: NONE
  max_context_chars: 100000
  judge_summaries: true
  summary_count: 400  # or clear this and set summary_fraction; never both
  group_near_duplicates: true
  relevance_threshold: 4
  self_sufficiency_threshold: 4
  require_verbatim_quotes: false  # quote fidelity remains diagnostic
  missing_response_attempts: 3  # only missing rows, never raised runtime failures
sdg_generator_options:
  temperature: 0.6
  max_tokens: 8192
sdg_judge_options:
  temperature: 0.6
  max_tokens: 8192
```

Model option mappings accept the public `ModelSettings` fields, including
independent endpoint, credential **environment-variable name**, timeout and
`extra_body`. Both credentials are checked before generation; never put secrets
in model option dictionaries. Unknown fields fail. Recipe-owned paths, corpus ID,
execution bounds, seed, split ratios, resume and model-role objects cannot be
overridden in `sdg_options`; use their named recipe settings. No producer settings
are silently discarded. These option mappings require `sdg_workflow=retrieval_first`;
nonempty mappings are rejected in `legacy_qa`.

All four summary grades (information richness, persona relevance,
query-generation potential and conceptual clarity) must reach 4 by default.
Exact membership/language deduplication is always enabled. Optional
`summary_near_duplicate_threshold` requires at least 90% source overlap, matching
document sets and protected numeric/negation tokens. Distinct eight-unit contexts
cannot meet the overlap guard in the simpler `unit`/`document` modes. The new
`sections` mode deduplicates before generation bounding, allowing larger combined
summaries to qualify; character bounds still apply. The example omits this
optional setting. `group_near_duplicates` independently groups equivalent
queries before splitting.

The templates/sampling policy are adapted from the MIT-licensed ViDoRe v3
implementation, with attribution packaged by the plugin. All inputs, IDs and
outputs remain generic; no dataset loader or benchmark-specific schema is used.
Context-aware self-sufficiency reproduces the reference judging policy rather
than measuring query-only standalone quality. Do not interpret matching thresholds
as proof of matching retention. For a controlled Qwen run, explicitly set both
roles to temperature 0.6, max_tokens 8192, and the provider-supported
`extra_body: {chat_template_kwargs: {enable_thinking: false}}`. Keep these
provider-specific flags out of configurations for endpoints that do not support them.

This is an unreleased EA candidate. A bounded corpus run using the public
Nemotron summary embedding default and operator-configured Qwen 3.6 generator/
judge completed SDG, locked stage-local dependency installation, native mining,
two-A100 BF16 training, checkpoint resume and fresh base/final evaluation. It
tested recipe commit `1d1864c8eacb0e58ec450a649ea5d9498afb617e`, producer commit
`9984b28bc89a523a07b0afcb65c9b4198bd64c69` and the pinned AutoModel dependency.

The run accepted 141 of 157 candidates (89.8%), with five abstentions and
generation coverage across all 71 input units. The combined view contained 112
training and 29 held-out queries; mining produced 169 unrolled training examples.
Training completed two steps, then resumed to step four. Base and fine-tuned
NDCG@10 were both 0.95987; Recall@10 was 0.95862. Original text/image content and
shared query-group split integrity were verified. These few-step results
establish pipeline execution, not retrieval uplift or convergence.

This run required one explicit SDG resume after inspecting a quoted-null query
response rejected by native schema validation. Valid cached responses were
preserved, missing responses recovered, and all quality gates remained enabled.
The optimizer configuration was unchanged: BF16 weights, quantized FlashAdamW
states, 32-bit master-weight correction and FP32 saved checkpoint moments.

The default evaluation above uses the held-out split from the same synthetic
generation run. It is a pipeline smoke test, not independent model-quality
evidence. For the latter, point the evaluator at a separately sourced BEIR
dataset and its portable image root:

```bash
nemotron embed eval -c mistral3-vl \
  sdg_input_path=null retrieval_view=null \
  eval_data_path=/absolute/path/to/independent-beir \
  image_root=/absolute/path/to/independent-bundle \
  output_dir=./output/embed/mistral3-vl-preview/stage3_eval_independent \
  eval_base=true eval_finetuned=true eval_nim=false
```

Multimodal Stages 1-3 currently require local execution. Container dependency
selection remains unvalidated, so Docker and Slurm invocations fail before
submission. Text containers select the `text` extra; use a fresh container for
each stage because the shared wrapper does not track dependency changes in its
environment-ready marker. CPU configuration tests do not establish GPU or
container execution compatibility. The local-stage qualification above ran inside
a manually prepared GPU container; it did not exercise the recipe's Docker/Slurm
launcher.


## Optional deployment overrides

The `mistral3-vl` deployment profile uses vLLM and preserves the checkpoint's
saved model and processor settings by default (`vllm_hf_overrides: null`).
Deployment is outside the Stage 0–3 qualification described above.
Only supply a config override when you have verified that the selected
checkpoint requires it. For a checkpoint with a confirmed image-size mismatch,
for example:

```bash
nemotron embed deploy -c mistral3-vl \
  'vllm_hf_overrides={"vision_config":{"image_size":1120}}'
```

The value `1120` is an example for that specific mismatch, not a requirement for
all compatible checkpoints. Omit the override for checkpoints whose settings
already agree; verify serving separately against the local evaluation path.
