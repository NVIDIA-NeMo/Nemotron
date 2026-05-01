# Available Steps

## convert — Conversion

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [convert/hf_to_megatron](convert/hf_to_megatron/) | Convert a HuggingFace safetensors checkpoint to Megatron distributed format. | checkpoint_hf | checkpoint_megatron |
| [convert/megatron_to_hf](convert/megatron_to_hf/) | Convert a Megatron distributed checkpoint to HuggingFace safetensors format. | checkpoint_megatron | checkpoint_hf |
| [convert/merge_lora](convert/merge_lora/) | Merge a LoRA adapter into the base model to produce a standalone HuggingFace checkpoint. | checkpoint_lora, checkpoint_hf | checkpoint_hf |

## curate — Data Curation

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [curate/nemo_curator](curate/nemo_curator/) | Acquire public or custom text corpora with NeMo Curator, then annotate and filter them by language, domain, and quality to produce downstream-ready JSONL. | - | filtered_jsonl |

## eval — Evaluation

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [eval/model_eval](eval/model_eval/) | Deploy a trained checkpoint behind an OpenAI-compatible endpoint and run benchmark suites with NeMo Evaluator, producing consolidated evaluation results. | checkpoint_megatron (optional), checkpoint_hf (optional) | eval_results |

## prep — Data Preparation

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [prep/sft_packing](prep/sft_packing/) | Apply the chat template, tokenize training JSONL, and pack examples into Megatron-Bridge-compatible Parquet shards for SFT. | training_jsonl | packed_parquet |

## rl — Reinforcement Learning

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [rl/nemo_rl_grpo](rl/nemo_rl_grpo/) | Planned: align an SFT-trained Megatron checkpoint with GRPO using NeMo-RL. | training_jsonl, checkpoint_megatron | checkpoint_megatron |

## sft — Supervised Fine-Tuning

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [sft/automodel](sft/automodel/) | Supervised fine-tuning with the AutoModel stack. Best for smaller GPU counts, rapid iteration, and LoRA-style adapter tuning on JSONL datasets that already use OpenAI chat-format messages. | training_jsonl | checkpoint_hf |
| [sft/megatron_bridge](sft/megatron_bridge/) | Supervised fine-tuning using NVIDIA Megatron-Bridge. Best for large-scale distributed training with tensor/pipeline/context parallelism. Requires packed Parquet data from prep/sft_packing. | packed_parquet, checkpoint_megatron (optional) | checkpoint_megatron |

## synth — Synthetic Data Generation

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [synth/data_designer](synth/data_designer/) | Planned: generate synthetic conversation JSONL with Data Designer for downstream SFT. | training_jsonl (optional) | synthetic_jsonl |

## translate — Translation

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [translate/translation](translate/translation/) | Translate JSONL or Parquet training corpora with NeMo Curator's TranslationStage, preserving structured fields and optionally attaching FAITH quality scores. | filtered_jsonl | translated_jsonl |
