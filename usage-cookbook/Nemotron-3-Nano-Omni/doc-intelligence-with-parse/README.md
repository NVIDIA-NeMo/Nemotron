# Document Intelligence with Nemotron Parse + Nemotron 3 Nano Omni

An end-to-end, **all-modality** document-AI cookbook that pairs two
NVIDIA models on a single hosted endpoint:

- **[Nemotron Parse](https://build.nvidia.com/nvidia/nemotron-parse)** -
  the structural Architect that gives every PDF page a clean, typed
  layout (titles, sections, tables, figures, picture bounding-boxes).
- **[Nemotron 3 Nano Omni](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning)** -
  a 30B-A3B vision-language MoE that wears two hats in this notebook:
  the per-picture **Specialist** (classify + transcribe each cropped
  figure) *and* the cross-page **Reasoning Engine** (answer multi-page
  questions with `enable_thinking=True`).

One credential, one URL, one model family - no two-model drift, no
embedding-alignment surgery.

## What you build

A four-stage pipeline that runs end-to-end against the hosted
[`build.nvidia.com`](https://build.nvidia.com) catalog (no local GPU
required):

1. **Layout extraction** - Parse turns each PDF page into a typed
   block tree.
2. **Picture transcription** - the cropped pictures (charts, tables,
   infographics, screenshots, diagrams) are routed to Nano Omni in
   *Instruct* mode for content-aware transcription.
3. **Reading-order document** - text + picture transcriptions are
   stitched back into a single Markdown document per PDF.
4. **Multi-page reasoning** - the same Nano Omni model answers free-
   form questions over the assembled document in *Thinking* mode.

The notebook walks through all four stages on four real public PDFs
(Pew Research, two LinkedIn social-media analytics pages, and a
Graduate-Studies brochure) so you can see the model card's "Instruct
mode" vs. "Thinking mode" sampling defaults applied to a realistic
multi-modal workload.

## Models

| Role                          | Model                                                       | Mode                              |
| ----------------------------- | ----------------------------------------------------------- | --------------------------------- |
| Architect (layout)            | `nvidia/nemotron-parse`                                     | tool-use, deterministic           |
| Specialist (picture content)  | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`             | Instruct (`temperature=0.2, top_k=1`) |
| Reasoning engine (final QA)   | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`             | Thinking (`temperature=0.6, top_p=0.95`) |

Both models ship as NIMs on NGC -- see the **Deploy** tab on each
model card for copy-paste vLLM / SGLang / TensorRT-LLM launch
commands
([Parse · Deploy](https://build.nvidia.com/nvidia/nemotron-parse/deploy)
· [Nano Omni · Deploy](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning/deploy))
-- and as open weights (BF16 / FP8 / NVFP4) on Hugging Face. Tested
hardware: H100, B200, DGX Spark, RTX Pro 6000, Jetson Thor.

## Requirements

- Python 3.10+
- A free NVIDIA Developer account + API key from
  [build.nvidia.com](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning)
  (the same key unlocks both models).
- [`uv`](https://docs.astral.sh/uv/) for fast dependency installs
  (the install cell falls back to `pip` if `uv` is not on your
  `PATH`).

That's it - **no GPU, no Docker, no local model weights** are needed
to run this notebook end-to-end.

## Quick start

```bash
# 1. install uv (one-time; skip if you already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. create an isolated venv for the notebook + Jupyter
uv venv .venv && source .venv/bin/activate
uv pip install jupyter

# 3. set your API key (any one of these works)
export NVIDIA_API_KEY=nvapi-...
#   - or -
echo "NVIDIA_API_KEY=nvapi-..." > .env

# 4. launch
jupyter lab doc_intelligence_cookbook.ipynb
```

The notebook installs its five runtime deps (`pymupdf`, `pillow`,
`requests`, `pandas`, `python-dotenv`) on first execution via
`uv pip install --python {sys.executable} ...`, and auto-downloads the
four demo PDFs from the public
[`yubo2333/MMLongBench-Doc`](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc)
Hugging Face dataset on first run.

## What gets written to disk

Running the notebook end-to-end produces two local sub-folders next
to it (both `.gitignore`d):

- `data/documents/` - the auto-downloaded source PDFs
  (~2 MB total, fetched once and cached).
- `output_results/` - per-document `*.parse_omni.json` artefacts
  containing the typed layout + Specialist transcriptions, used by
  the final QA cells.

Delete either folder to force a fresh run from scratch.

## Troubleshooting

- **`NVIDIA_API_KEY is not set`** - get a free key from the model card
  (top-right "Get API Key") and either drop it in a `.env` file in
  this directory or `export NVIDIA_API_KEY=...` before launching
  Jupyter.
- **HTTP 403 / 404 from the hosted endpoint** - confirm your account
  has access to both `nvidia/nemotron-parse` and
  `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` on
  [build.nvidia.com](https://build.nvidia.com); both are in the
  public catalog at GA.
- **Cells time out** - the notebook makes ~30 hosted calls; on a
  slow link bump `--ExecutePreprocessor.timeout` if you re-run with
  `nbconvert`.
