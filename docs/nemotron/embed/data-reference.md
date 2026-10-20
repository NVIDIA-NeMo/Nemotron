# Data and Artifact Reference

Use this reference to configure directory input or explicit source records,
select the documents a model sees, and identify the artifact passed between
stages. For a complete procedure,
follow the [multimodal guide](multimodal.md).

## Choose the Input Format

Choose the input contract for your model profile. Text question-and-answer (QA)
generation and multimodal generation use different source formats.

| Model Profile | Workflow | Stage 0 Input | Output Used by Stage 1 |
| --- | --- | --- | --- |
| `nemotron-3` or `llama` | `text_qa` | `corpus_dir`: local directory or supported `hf://` text corpus | Generated QA JSONL, then text QA conversion |
| `nemotron-3.5` | `multimodal` | `source_directory.path`: local text/image directory, or `sources_file`: explicit source JSONL | Portable retrieval bundle with queries, relevance labels, and text/image views |

Text QA still scans and sentence-chunks text files through the producer's
package-root API. Follow the [Nemotron 3 and Llama workflows](text.md) for that path.

For multimodal generation, set exactly one of `source_directory.path` or
`sources_file`. Use directory input for independent text and image files. Use
source records for custom unit boundaries, paired text and images, or page
relationships. Both routes produce the same portable retrieval-bundle contract.
PDF extraction, optical character recognition (OCR), and page rendering happen
before the recipe.

## Directory Input

Set `source_directory.path` to your document directory. For example, after
configuring model roles and credentials, preview the default directory workflow:

```bash
uv run nemotron embed sdg -c nemotron-3.5 -d source_directory.path=./documents
```

Place `artifact_root` outside the source directory. Stage 0 scans supported files
recursively, prepares retrieval units, and records them under
`<artifact_root>/stage0_sdg/multimodal/inputs/sources.jsonl` before model calls.
No hand-written manifest is required.

Directory settings use the `source_directory` prefix, as shown below.

| Setting | Default | Effect |
| --- | --- | --- |
| `source_directory.path` | Required | Directory of local source files |
| `source_directory.recursive` | `true` | Search nested directories |
| `source_directory.file_pattern` | `"*"` | Case-sensitive basename glob |
| `source_directory.file_extensions` | `.txt`, `.md`, `.markdown`, `.png`, `.jpg`, `.jpeg`, `.webp` | Restrict discovery to a subset of supported extensions; matching ignores extension case |
| `source_directory.num_files` | `null` | Limit files after filtering and sorting |
| `source_directory.text_chunking` | `sentences` | Pack complete sentences into units; `none` preserves each text file as one unit |
| `source_directory.max_tokens_per_chunk` | `1024` | Token cap for text units; `null` disables it |
| `source_directory.tokenizer` | `nvidia/Nemotron-3.5-Embed-1B-BF16` | Hugging Face model ID or saved local fast-tokenizer directory for token counting |
| `source_directory.sentences_per_chunk` | `null` | Optional sentence-count cap in sentence mode |
| `source_directory.reserved_tokens` | `0` | Allowance for prefixes or other tokens absent from the tokenizer count |
| `source_directory.split_oversized_sentences` | `false` | Explicitly allow splitting a sentence that exceeds the token cap |
| `source_directory.language` | `source` | Label prepared units; does not select a sentence tokenizer |

Sentence mode uses the shared Text QA sentence splitter and joins consecutive
sentences with spaces, without overlap. Counts include configured special tokens
and `reserved_tokens`. If both caps are set, a chunk closes before the next
sentence would exceed either limit. A sentence that alone exceeds the token cap
fails by default. Whole-file mode preserves internal formatting, but still
applies the token cap. Generator request limits are separate.

Tokenizer assets and missing NLTK sentence-tokenizer resources download on first
use. Model weights are not loaded for directory preparation. For offline setup,
custom token budgets, or section-aware preparation, refer to the
[pinned directory reference](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/blob/a4c57f8013a6190a2a101b744ac82c90f67da937/plugins/data-designer-retrieval-sdg/docs/multimodal-reference.md#directory-input).

Files are sorted by relative path. Each text file is one document with numbered
chunk IDs; each image is a separate document with one image unit. Matching
filenames do not pair text with images. The loader does not infer page numbers,
related documents, or Markdown image links. Use `sources_file` for those
relationships. Inspect the generated source IDs before supplying `contexts_file`.

Hidden files, directory symlinks, and unsupported extensions are ignored.
Selected symlink files, unreadable files, invalid UTF-8, empty text, and damaged
or animated images fail explicitly. An empty selection also fails. Use the
`mixed` view, the profile default, to retain text-only and image-only documents.

## Source Records

Use `sources_file` instead of `source_directory` when you need explicit
text/image pairs, page relationships, or custom units. Clear an inherited
directory configuration with `source_directory=null`. For example, preview
explicit source input after configuring model roles and credentials:

```bash
uv run nemotron embed sdg -c nemotron-3.5 -d \
  sources_file=/absolute/path/to/sources.jsonl source_directory=null
```

The file uses UTF-8 JSONL: one JSON object per line, without an enclosing array.
Escape newlines inside strings as `\n`. For example:

```json
{"unit_id":"pump-a-p1","document_id":"pump-a","text":"The pump operates between 10 and 30 degrees Celsius.","images":["pages/pump-a-001.png"],"page_number":1,"language":"en"}
{"unit_id":"pump-b-p1","document_id":"pump-b","images":["pages/pump-b-001.png"],"page_number":1,"language":"en"}
{"unit_id":"parts-c-s1","document_id":"parts-c","text":"Store replacement parts in a dry location.","language":"en"}
```

Create the referenced image files before generation. These records illustrate
the schema; they are not a sufficient training corpus.

The source schema has the following fields.

| Field | Required | Meaning |
| --- | --- | --- |
| `unit_id` | Yes | Stable, nonempty ID unique across the file. Do not include tabs or line breaks. |
| `document_id` | Yes | Nonempty ID of the source document. Pages of one document share this ID. |
| `text` | Conditional | Original extracted text or optical character recognition (OCR). Defaults to `""`. |
| `images` | Conditional | Zero or one local image path. Defaults to `[]`. PNG, JPEG, and WebP are supported. |
| `page_number` | No | Page number starting at 1. Records provenance; does not sort the input. |
| `language` | No | Nonempty language string, such as `"en"`. Defaults to `"source"`. |
| `source_uri` | No | Original source location or `null`. Records provenance; does not trigger a download. |

Every unit requires nonblank text, an image, or both. Relative image paths
resolve from the source file's directory. Absolute paths are also accepted.
Images must be locally accessible; HTTP image URLs are not downloaded.

The loader rejects empty input, malformed records, unknown fields, duplicate
unit IDs, missing content, missing images, and multiple images per unit.
Keep original source content in these records. Do not insert generated
questions, answers, negatives, or train/test assignments.

## Generation Contexts

A context names units to consider together during query generation. For example,
a row in `contexts.jsonl` can group two pages:

```json
{"context_id":"pump-operating-limits","unit_ids":["pump-a-p1","pump-b-p1"],"language":"en"}
```

Use a unique, nonempty `context_id` and a nonempty list of existing unit IDs.
`language` is optional and defaults to `"source"`. Pass the file through
`contexts_file` in Stage 0.

Explicit contexts replace automatic grouping and can span documents. They are
still bounded by `sdg_max_units_per_context` and any configured character or
token limits. The default summary-combination step can propose additional
memberships even with explicit contexts. Set
`sdg_options.combination_iterations=0` to disable those combinations.

A context controls generation evidence. It does not assign a query group or a
train/evaluation split. Units outside selected contexts remain in the eligible
retrieval corpus.

## Retrieval Views

Stage 0 exports views of the same sources. Stage 1 and Stage 3 default to
`retrieval_view=mixed` for `nemotron-3.5`. If you change the view, set it in both
stages. The view defines eligible documents and available query support.

| View | Eligible Units | Document Content |
| --- | --- | --- |
| `text` | Units with nonblank text | Text |
| `image` | Units with one image | Image |
| `image_and_text` | Units with one image, with or without text | Image and available text |
| `mixed` | Units with text, one image, or both | Available text and image content |

`mixed` requires a schema-version-2 grouped-query bundle that exports that view.
An unavailable view produces an error; the recipe does not silently substitute
another view.

A query is exported into a view only when all its positive supports are eligible
and their evidence modalities are retained. Different views can therefore have
different query sets. Keep the view fixed when comparing checkpoints.

To preserve a heterogeneous corpus, use `mixed` for preparation and evaluation:

```bash
uv run nemotron embed prep -c nemotron-3.5 retrieval_view=mixed
uv run nemotron embed eval -c nemotron-3.5 retrieval_view=mixed eval_base=true
```

Add your run's `artifact_root` to both commands when it differs from the
profile default. Training consumes the mined Stage 1 file; it does not take a
`retrieval_view` setting. Keep `use_text_in_document=true` to retain available
text in mixed or paired documents.

Mixed Parquet rows use `docid`, `text`, and binary `image` columns. Text-only
rows have a null image and omit `image_path` in evaluation. The recipe adapter
handles these absent images. Do not bypass it with a raw Automodel dataset
loader that expects an image for every row.

## Query Splits and Relevance Labels

The Data Designer plugin assigns query groups and splits before positive
unrolling. The profile requests 80% training, no validation partition, and 20%
evaluation.
Small or uneven groups can change the realized counts. Stage 1 preserves the
exported split; its text-conversion ratios and quality filter do not apply.

All partitions search the full view-eligible collection. A query can have
multiple graded positive documents. The plugin keeps each query group in one
partition and groups normalized duplicate queries before splitting. Downstream
stages preserve these assignments and all positive documents; they do not
resplit queries.

Preparation retains query-group and source annotations through mining and
positive unrolling. Mined `score` values describe encoder similarity;
`source_score` preserves the original positive relevance grade.

Documents can occur in both training and evaluation positives. This is
adaptation to a fixed collection, not evaluation on unseen documents. Synthetic
qrels are generated labels, not independent gold judgments. Sharing a document,
page, or generation context does not automatically place queries in one group;
unknown paraphrases cannot be inferred from opaque group IDs.

The bundle declares `split_protocol: grouped_query_disjoint` and
`corpus_scope: full_collection`. View corpora are shared across query partitions;
query assignments must match the exported queries. Portable SDG handoffs
require this grouped-query protocol.

Other producers must implement the same checksummed, grouped-query bundle
contract. Refer to the [pinned producer reference](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/blob/a4c57f8013a6190a2a101b744ac82c90f67da937/plugins/data-designer-retrieval-sdg/docs/multimodal-reference.md)
for its schema and export layout.

## Validate an Export

Normal stage handoffs require schema version 2, `grouped_query_disjoint` splits,
and `full_collection` corpora. They check the selected view, required files, and
basic data loadability. They reject paths outside the bundle and verify that
`generation_result.json` identifies the expected bundle manifest. They do not
repeat a full audit of file hashes, source content, or split consistency at every
stage.

For a full audit after copying an export or when investigating data problems,
run the following command from the repository root. Replace the manifest path
with the exported `run_manifest.json` and select the view used by your run:

```bash
PYTHONPATH="$PWD/src" uv run --project src/nemotron/recipes/embed/stage0_sdg --locked \
  python -m nemotron.recipes.embed.stage0_sdg.validate \
  '<path/to/run_manifest.json>' --view mixed
```

The command uses the Stage 0 dependencies. It checks inventory hashes, the
selected view's corpus and source consistency, relevance references, and query
assignments. It also checks that query groups and normalized duplicates do not
cross partitions. A failed check exits with an error; inspect the reported
artifact before using the export. This audit does not establish that generated
relevance labels are correct or independently judged.

## Artifact Handoffs

Every profile derives paths from `artifact_root`. The checked-in defaults are:

| Profile | Default Root |
| --- | --- |
| `nemotron-3` | `./output/embed/nemotron-3-1b` |
| `nemotron-3.5` | `./output/embed/nemotron-3.5` |
| `llama` | `./output/embed` |

Use a descriptive experiment directory with `artifact_root`.

The principal handoffs below are relative to the selected root.

| Artifact | Purpose |
| --- | --- |
| `stage0_sdg/generation_result.json` | Identifies the exact successful generation output and any portable bundle |
| `stage0_sdg/multimodal/` | Retrieval-first run state, caches, diagnostics, and exported data |
| `stage0_sdg/multimodal/inputs/sources.jsonl` | Prepared records from directory input |
| `stage1_data_prep/train_mined.automodel_unrolled.json` | Stage 2 training input, with mined negatives and unrolled positives |
| `stage1_data_prep/eval_beir/` | Evaluation data created by legacy text conversion |
| `stage2_finetune/checkpoints/` | Resumable model, optimizer, and scheduler state |
| `stage2_finetune/checkpoints/LATEST/model/consolidated/` | Complete model for evaluation and direct deployment |
| `stage3_eval/eval_results.json` | Metrics for the selected evaluation modes |

For portable input, Stage 3 resolves evaluation data from the Stage 0 manifest.
Do not assume a bundle name or use the text-only `eval_beir` path. Keep the
complete source bundle and its assets alongside prepared training data. Moving
only the training JSON breaks its corpus references.

## Reuse Existing Data

Choose the entry point that matches your artifact.

| Available Input | Entry Point |
| --- | --- |
| Compatible legacy SDG JSON, JSONL, or Parquet | `embed prep -c nemotron-3 sdg_input_path=<path>` |
| Portable `generation_result.json` | `embed prep -c nemotron-3.5 sdg_input_path=<path> retrieval_view=<view>` |
| Existing Automodel training JSON to mine | `embed prep -c nemotron-3 sdg_input_path=null train_input_file=<path>` |
| Mined and unrolled training JSON | `embed finetune -c <profile> train_data_path=<path>` |
| Independent BEIR dataset and checkpoint | Follow [Evaluate Independent Data](multimodal.md#evaluate-independent-data) |

Replace angle-bracket placeholders before execution. Legacy QA input must match
the retrieval-SDG conversion schema; an arbitrary list of query/text pairs is
not automatically compatible. The public
[Retrieval-Synthetic-NVDocs-v1 dataset](https://huggingface.co/datasets/nvidia/Retrieval-Synthetic-NVDocs-v1)
is an existing text SDG source. Export its records to a file and pass that file
explicitly to avoid ambiguous directory selection.

`source_directory` or `sources_file` input and portable export require
`sdg_workflow=multimodal`. Existing `nemotron-3` and `llama` generation use
`text_qa` with `corpus_dir`. Do not use `corpus_dir` for multimodal directory input.
