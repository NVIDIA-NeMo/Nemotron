# Nemotron-CC Data Curation

The Nemotron-CC recipe curates high-quality pretraining data from [Common Crawl](https://commoncrawl.org/), producing datasets similar to [nvidia/Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2). It serves as a reference for running these curation steps on your own data.

Built on [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) and [Ray](https://ray.io/), the pipeline scales from a single machine to large GPU clusters.

## Pipeline

The recipe is a four-step pipeline that progressively refines raw web data into curated text and synthetic training data:

```
Common Crawl → Extract & Clean → Deduplicate → Quality Classify → Synthetic Data Generation
```

| Step | Script | Description | Resources |
|------|--------|-------------|-----------|
| 1 | `step_1-download_extract.py` | Download, extract, language ID, Unicode cleanup | CPU-only |
| 2a | `step_2a-exact_dedup.py` | GPU-accelerated exact deduplication | GPU (identify), CPU (remove) |
| 2b | `step_2b-fuzzy_dedup.py` | MinHash + LSH fuzzy deduplication | GPU (identify), CPU (remove) |
| 2c | `step_2c-substring_dedup/` | Exact substring deduplication using suffix arrays | CPU-only |
| 3 | `step_3-quality_classification.py` | Ensemble quality scoring into 20 buckets | GPU (classify), CPU (ensemble) |
| 4 | `step_4-sdg.py` | LLM-based synthetic data generation on top-quality data | CPU + LLM endpoint |

Steps 1–3 progressively filter and annotate the data. Step 4 generates synthetic training data (diverse QA, distillation, knowledge extraction, knowledge lists) from the highest-quality documents (buckets 18–19).

## Getting Started

The recipe scripts live in:

```
src/nemotron/recipes/data_curation/nemotron-cc/
```

See the recipe README at `src/nemotron/recipes/data_curation/nemotron-cc/README.md` for detailed per-step documentation, resource recommendations, and usage examples.

## Prerequisites

- [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) installed with Ray support
- GPU(s) for steps 2a, 2b, and 3 (deduplication and classification)
- Access to an OpenAI-compatible LLM endpoint for step 4 (NVIDIA NIM, vLLM, or cloud API)

## After Curation

Once curated, the output can be tokenized and used for downstream model training.

## Further Reading

- [Nemotron-CC paper](https://arxiv.org/abs/2412.02595) — methodology and evaluation
- [nvidia/Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) — released dataset on Hugging Face
- [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) — the underlying data curation library
- [Data Preparation](./data-prep.md) — last-mile processing for training
