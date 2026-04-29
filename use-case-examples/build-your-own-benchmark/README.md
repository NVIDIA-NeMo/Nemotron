# Build your own MCQ benchmark (BYOB)

This folder is a **hands-on example** of the Nemotron **Build Your Own Benchmark (BYOB)** flow for **multiple-choice questions (MCQ)**. You start from **domain text** (plain `.txt` files), borrow **few-shot style** from a public benchmark on Hugging Face (for example MMLU), and run a **multi-stage pipeline** that drafts questions, judges them, deduplicates, checks coverage and distractors, and exports a **Parquet** file in an MMLU-Pro–style layout for evaluation or review.

For the product-level pattern (when to use BYOB, quality bar, exceptions), see  
`src/nemotron/steps/patterns/custom-mcq-benchmark-byob.md`. The catalog step is `benchmark/byob` (see `src/nemotron/steps/types.toml`).

---

## What you get

| Goal | How this example helps |
|------|-------------------------|
| A **custom** MCQ suite aligned with **public** MCQ format and difficulty | Few-shots are drawn from a supported HF dataset; questions are grounded in **your** corpus. |
| **Traceable** artifacts | Intermediate Parquet under a run directory; final columns match a standard MCQ layout. |
| **Quality controls** | Optional semantic dedup, coverage vs source text, distractor checks, hallucination / easiness filters (all configurable). |

---

## What’s in this directory

| Item | Role |
|------|------|
| **`default.yaml`** | Single place to set corpus paths, HF benchmark name, LLM / embedding models, and which optional stages run. Paths are relative to **this folder** as Jupyter’s working directory. |
| **`build_mcq_benchmark.ipynb`** | Walkthrough: paths → API keys → seed preparation → **stage-by-stage** MCQ pipeline (NeMo Curator / Data Designer where applicable). Run **top to bottom** with cwd = this directory. |
| **`download_wikipedia_data.ipynb`** | Optional helper to download Wikipedia articles into a **folder of `.txt` files** you can use as the domain corpus. |
| **`assets/`** | Example layout: put corpora under something like `assets/<your_corpus_name>/` and point `input_dir` / `target_source_mapping` at it. Generated outputs land under `assets/output/` by default (see `output_dir` in YAML). |

---

## Quick start

1. **Working directory**  
   Open Jupyter or a shell with **current working directory** set to this folder  
   `Nemotron/use-case-examples/build-your-own-benchmark`  
   so `./default.yaml` and `./assets/...` resolve correctly.

2. **Corpus**  
   - **Your own text:** create a subfolder with one or more `.txt` files.  
   - **Or** run `download_wikipedia_data.ipynb` to fill a folder with `.txt` articles.

3. **Configure `default.yaml`**  
   - **`input_dir`**: parent of your corpus folders (e.g. `./assets`).  
   - **`target_source_mapping`**: keys must be **directory names** under `input_dir`, each containing your `.txt` files.  
   - **`hf_dataset`**: must be one of the supported IDs listed in `nemotron/steps/byob/constants.py` (`ALLOWED_HF_DATASETS`).  
   - **`expt_name`** / **`output_dir`**: names the run folder under which `seed.parquet`, `stage_cache/`, `artifacts/`, and `benchmark.parquet` are written.

4. **Credentials**  
   - **`NVIDIA_API_KEY`**: required for the NVIDIA provider / NIM-style models in the sample config.  
   - **`HF_TOKEN`**: optional; useful if Hugging Face models or datasets need authentication.

5. **Run**  
   Open `build_mcq_benchmark.ipynb` and run all cells in order.

6. **Install / imports**  
   From the Nemotron repo root, an editable install is typical:  
   `pip install -e .`  
   The notebook can also prepend `Nemotron/src` to `sys.path` for a one-off session.

---

## Pipeline stages (high level)

Stage order follows `build_mcq_benchmark.ipynb`:

1. **Generation** — LLM drafts MCQs from seed rows.  
2. **Judgement** — LLM refines / scores drafts.  
3. **Semantic deduplication** — embeddings + clustering to thin near-duplicates.  
4. **Distractor expansion** (optional, `do_distractor_expansion`) — e.g. expand to more choices.  
5. **Coverage check** (optional, `do_coverage_check`) — embedding-based check vs source windows (**GPU-friendly Ray stack**: see note below).  
6. **Distractor validity** — LLM checks plausibility of wrong answers.  
7. **Semantic outlier detection** — flags odd answer choices via embeddings.  
8. **Hallucination / easiness filtering** — LLM panel filters; thresholds in YAML.  
9. **Final export** — `benchmark.parquet` (and raw variant); **human review** is still recommended before treating output as production benchmark data.

---

## Outputs (typical layout)

Under `<output_dir>/<expt_name>/` (defaults in `default.yaml`):

- **`seed.parquet`** — prepared seed rows for generation.  
- **`stage_cache/*.parquet`** — one file per major stage for debugging and resume.  
- **`artifacts/`** — Data Designer runs, embedding caches, etc.  
- **`benchmark.parquet`** — final MCQ table for downstream eval (after optional drops for hallucination / easy items).

---

## Customization tips

- **More or fewer questions:** adjust `few_shot_samples_per_query`, `queries_per_target_subject_document`, `num_questions_per_query`, and corpus size.  
- **Prompts:** `prompt_config: null` uses packaged defaults; point to a YAML file to override system/user prompts per stage (see comments in `default.yaml`).  
- **Models:** `generation_model_config`, `judge_model_config`, and filtering blocks use provider-specific fields—keep them consistent with your deployment (NVIDIA sample uses `provider: nvidia`).