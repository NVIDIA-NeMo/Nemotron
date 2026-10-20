# Configure Multimodal SDG

Use these settings to control synthetic data generation (SDG) with the
multimodal producer selected by
`nemotron-3.5`. Start with the [multimodal walkthrough](multimodal.md), then adjust
the roles, generation contexts, or execution bounds for your corpus.

## Workflow Selection

The `nemotron-3.5` profile selects `sdg_workflow: multimodal`. The text-only
`default` and `llama` profiles select `sdg_workflow: text_qa` and continue to accept
`corpus_dir`. The two producers have different input and output contracts.
Refer to [Choose the Input Format](data-reference.md#choose-the-input-format).

The multimodal workflow always exports a portable retrieval bundle. The text
workflow uses its existing QA output contract.

## Source Input

Start with `source_directory.path=./documents` for a directory of independent
text and image files. Stage 0 prepares source records before generation. Use
`sources_file` for explicit text/image pairing, page relationships, or custom
unit boundaries. Set only one input mode.

Configure file discovery and chunking under `source_directory`, including
`source_directory.num_files` for a bounded first run. These settings are separate
from generation contexts and `sdg_options`. Refer to the
[directory input reference](data-reference.md#directory-input) for supported
formats, token limits, and the prepared-input location.

## Model Roles and Credentials

The embedding checkpoint you fine-tune does not generate or judge queries.
Configure the following roles separately.

| Role | Configuration | Input |
| --- | --- | --- |
| Visual enrichment, descriptions, summaries, and query generation | `sdg_generator_model` and `sdg_generator_options` | Source text and images |
| Summary/query judging and support localization | `sdg_judge_model` and `sdg_judge_options` | Queries, summaries, and supporting sources |
| Summary clustering embeddings | `sdg_options.summary_embedding_*` | Summary text |
| Mining, training, and base evaluation | Public Nemotron 3.5 checkpoint; optional `NEMOTRON_3_5_EMBED_MODEL` override | Text queries and view-selected documents |

For `sdg_workflow: multimodal`, both roles default to
`nvidia/nemotron-3.5-super-vl-120b-a12b`. The model accepts images
and supports structured output through its
[chat-completions API](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b-infer).

Override either model with `NEMOTRON_3_5_SDG_GENERATOR_MODEL` or
`NEMOTRON_3_5_SDG_JUDGE_MODEL`. Explicit YAML or CLI model settings take
precedence over these environment variables. The resolved model identities
appear in CLI previews and saved run configurations.

Use `sdg_generator_model` for multimodal descriptions, summaries, and retrieval
queries, and `sdg_judge_model` for the shared multimodal judge. The `text_qa`
workflow retains `qa_generation_model`, `quality_judge_model`, and its existing
model defaults; the multimodal model settings and environment variables do not
change those defaults.

Generation and judging default to `https://integrate.api.nvidia.com/v1` with
`NVIDIA_API_KEY`. `NVIDIA_API_BASE_URL` changes both chat endpoints. Per-role
options can independently set `endpoint`, `credential_env`, `timeout`,
`temperature`, `max_tokens`, and provider-specific `extra_body`.
`credential_env` names an environment variable; it never contains a secret.

For example, the following YAML fragment configures independent model roles.
Merge it into a copy of the Stage 0 profile and select that file with `-c`.
Replace the model and endpoint placeholders with values for your service.

```yaml
sdg_generator_model: <image-capable-generator-model>
sdg_judge_model: <image-capable-judge-model>
sdg_generator_options:
  endpoint: https://generator.example.com/v1
  credential_env: GENERATOR_API_KEY
  temperature: 0.6
  max_tokens: 8192
sdg_judge_options:
  endpoint: https://judge.example.com/v1
  credential_env: JUDGE_API_KEY
  temperature: 0.6
  max_tokens: 8192
```

The producer validates unknown fields and checks both role credentials before
generation. Use provider-specific request fields only when your endpoint
supports them.

If your chat endpoint supports JSON object output, you can enable JSON mode for
both roles. The producer continues to validate every response against its
requested schema and quality thresholds.

```yaml
sdg_generator_options:
  extra_body:
    response_format:
      type: json_object
sdg_judge_options:
  extra_body:
    response_format:
      type: json_object
```

Use a new `artifact_root` when changing model request settings for an existing
run, because the response cache is bound to those settings.

## Summary Embeddings

The default section workflow uses the public text model
`nvidia/nemotron-3-embed-1b` at `https://integrate.api.nvidia.com/v1` for semantic
clustering. This service receives summary text. Its endpoint and credential are
independent of the generator and judge settings.

The profile sets these options:

```yaml
sdg_options:
  summary_embedding_model: nvidia/nemotron-3-embed-1b
  summary_embedding_endpoint: https://integrate.api.nvidia.com/v1
  summary_embedding_credential_env: NVIDIA_API_KEY
  summary_embedding_extra_body:
    input_type: passage
    truncate: NONE
```

For local Sentence Transformers inference, provide a compatible model ID or
local path and clear the endpoint and request options:

```bash
uv run nemotron embed sdg -c nemotron-3.5 \
  source_directory.path=./documents \
  artifact_root=./output/embed/local-summary-run \
  sdg_options.summary_embedding_endpoint=null \
  sdg_options.summary_embedding_extra_body=null \
  sdg_options.summary_embedding_model='<local-summary-embedding-model>'
```

Optionally set `sdg_options.summary_embedding_revision` and
`sdg_options.summary_embedding_device`. Local summary encoding defaults to CPU.
Use `null` to clear inherited request options. An empty mapping merges with defaults and does not remove them.
Summary embedding responses are validated and cached for resume.

## Context Planning and Selection

By default, the producer enriches images, describes documents, plans sections,
generates summaries, combines summaries semantically, grades and selects them,
and generates bounded queries.

The most relevant controls are listed below.

| Setting | Profile Value | Effect |
| --- | --- | --- |
| `sdg_options.context_strategy` | `sections` | Plan sections from ordered source units; `unit` and `document` are alternatives |
| `sdg_options.section_size` | `5` | Target units per automatic section |
| `sdg_options.combination_iterations` | `20` | Seeded semantic-clustering iterations; `0` disables combinations |
| `sdg_options.summary_count` | `400` | Cap selected summaries; clear before using `summary_fraction` |
| `sdg_max_units_per_context` | `8` | Bound units used together for query generation |
| `sdg_options.max_context_chars` | `null` | No additional source-text character cap |
| `sdg_batch_size` | `30` | Producer request-batch size |
| `max_parallel_requests_for_gen` | Unset | Adapter uses concurrency `8` |
| `export_seed` | `42` | Generation/split seed |

Automatic sections preserve input order within document/language groups.
Markdown headings and confirmed table-of-contents boundaries guide grouping.
`page_number` can help align a table of contents; it does not reorder input.
When structure is unusable, the planner falls back to fixed-size grouping.
`planning/section_boundaries.json` records section decisions and coverage.

Explicit `contexts_file` memberships replace automatic grouping. They do not
disable semantic combinations. To use only bounded supplied memberships, set
`sdg_options.combination_iterations=0` and keep each context within the configured
unit and character bounds. Language groups with fewer than 12 summaries skip
semantic combinations.

For a smaller first run, select a representative source directory and set
`source_directory.num_files` to bound the selected files. With explicit JSONL,
prepare a smaller source file. Use `sdg_options.context_strategy=unit` with
`sdg_options.combination_iterations=0` to disable semantic combinations. Preserve
enough content for both query partitions and negative mining. The top-level
`num_files` setting controls only the legacy text loader.

## Quality Gates and Request Bounds

Summary grades, query self-sufficiency, answer leakage, and source relevance
are separate checks. Relevance and support localization use original source
text and images. Self-sufficiency uses source context, so its score is not a
query-only standalone-quality measure.

The producer defaults to thresholds of `4` for all four summary grades,
self-sufficiency, and relevance. `require_verbatim_quotes=false` treats missing
or nonmatching quotes as diagnostics. Setting it to `true` requires matching
quotes for text-supported positives; image-only support uses visual evidence.

`group_near_duplicates=true` groups equivalent queries before splitting.
Optional `summary_near_duplicate_threshold` instead controls summary selection;
its overlap and content guards are separate. Neither setting guarantees that
all semantic equivalents can be detected.

Complete rendered prompts remain subject to Data Designer's secure-renderer
limit and the provider's context window. Provider size errors stop the run;
source evidence is not silently truncated.

Optional local token preflight is configured independently for each model role.
Provide `context_window_tokens`, a matching local `tokenizer_file`,
`request_overhead_tokens`, and `image_tokens_per_image` for image requests.
The check includes schema, output reserve, and image overhead. Incomplete
settings fail. Oversized contexts split without dropping units; an indivisible
oversized request must be preprocessed before retrying.

The plugin accepts a complete bare or JSON-fenced structured response. Parsing
or provider failures stop the run and preserve successful responses. Inspect
failure diagnostics before resuming. Do not lower quality gates or alter accepted
records to conceal a parsing failure. `missing_response_attempts=3` retries
missing rows, not raised runtime failures.

## Resume Generation

Retrieval-first state lives under `<artifact_root>/stage0_sdg/multimodal`.
Inspect failed attempts, then rerun the original command with `resume=always`:

```bash
uv run nemotron embed sdg -c nemotron-3.5 \
  source_directory.path=./documents \
  artifact_root=./output/embed/multimodal-first-run resume=always
```

Repeat every override from the original run. Sources, model roles, settings,
and code must remain compatible with the stored run fingerprint. Directory
resume also verifies selected file contents, preparation settings, and tokenizer
identity. Use a new artifact root after upgrading the producer dependency or
changing input or configuration. Existing runs from the earlier producer pin
cannot be resumed under this version. `resume=never` does not overwrite an
existing multimodal run. Both `always` and `if_possible` map
to the producer's explicit resume flag; they do not provide the legacy workflow's
fresh-run fallback for incompatible state.

The text `text_qa` workflow has a different resume contract. Keep its corpus,
`artifact_path`, `dataset_name`, and `buffer_size` unchanged. Its configuration
fingerprint does not hash corpus contents. In that workflow, `never` starts a
fresh dataset, `always` requires compatible prior state, and `if_possible`
resumes when possible. `buffer_size` controls checkpoint writes, not independent
corpus partitions.

## Reference and Ownership

Recipe-owned paths, `source_directory`, corpus ID, split ratios, seed, concurrency,
batching, resume, and model-role objects use named recipe settings. Do not place
them inside `sdg_options`. Unknown or conflicting options fail validation.

The recipe pins the producer and Data Designer in each stage project. Refer to
the [pinned producer guide](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/blob/a4c57f8013a6190a2a101b744ac82c90f67da937/plugins/data-designer-retrieval-sdg/docs/multimodal.md)
for the complete generation workflow and producer configuration contract.
