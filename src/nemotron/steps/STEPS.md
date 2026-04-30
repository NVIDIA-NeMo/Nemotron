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

## optimize — Model Optimization

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [optimize/modelopt/distill](optimize/modelopt/distill/) | Distill a student model from a teacher model with NVIDIA Model Optimizer and Megatron-Bridge. Can run standalone or recover quality after pruning or quantization; real-data runs consume Megatron bin/idx data. | checkpoint_hf, binidx (optional) | checkpoint_megatron |
| [optimize/modelopt/prune](optimize/modelopt/prune/) | Prune a HuggingFace model with NVIDIA Model Optimizer's Minitron flow in the Megatron-Bridge framework. Supports target-parameter search or manual architecture pruning via prune_export_config. | checkpoint_hf | checkpoint_hf |
| [optimize/modelopt/quantize](optimize/modelopt/quantize/) | Post-training quantization with NVIDIA Model Optimizer through Megatron-Bridge. Supports Super3 FP8 recipes for Hopper/H100 and NVFP4 recipes for Blackwell/B200, producing Megatron distributed checkpoints ready for export/evaluation. | checkpoint_hf | checkpoint_megatron |

## peft — Parameter-Efficient Fine-Tuning

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [peft/automodel](peft/automodel/) | Parameter-efficient fine-tuning (LoRA) with the AutoModel stack. Same training loop as sft/automodel but with a LoRA adapter wired in by default — best for small GPU counts where a full SFT would not fit. | training_jsonl | checkpoint_lora |
| [peft/megatron_bridge](peft/megatron_bridge/) | Parameter-efficient fine-tuning (LoRA) on top of Megatron-Bridge. Useful when a full SFT exceeds memory but you still want TP/PP/CP scaling. Consumes packed Parquet from prep/sft_packing. | packed_parquet, checkpoint_megatron | checkpoint_lora |

## prep — Data Preparation

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [prep/pretrain_prep](prep/pretrain_prep/) | Tokenise raw text (HF datasets or local parquet/jsonl) into Megatron bin/idx shards and emit a blend.json that pretrain/megatron_bridge and pretrain/automodel can ingest directly. | filtered_jsonl | binidx |
| [prep/rl_prep](prep/rl_prep/) | Resolve HuggingFace dataset references in an RL data blend and shard the output JSONL into the prompt / preference layout expected by rl/nemo_rl/*. | training_jsonl | training_jsonl |
| [prep/sft_packing](prep/sft_packing/) | Apply the chat template, tokenize training JSONL, and pack examples into Megatron-Bridge-compatible Parquet shards for SFT. | training_jsonl | packed_parquet |

## pretrain — Pretraining

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [pretrain/automodel](pretrain/automodel/) | Causal-LM pretraining or continued pretraining (CPT) using the NeMo-AutoModel stack. Reads tokenized text data and trains from scratch or from an HF base. | binidx | checkpoint_hf |
| [pretrain/megatron_bridge](pretrain/megatron_bridge/) | Pretraining or continued pretraining with NVIDIA Megatron-Bridge. Best for large-scale runs that need TP/PP/CP/EP parallelism on bin/idx data. | binidx, checkpoint_megatron (optional) | checkpoint_megatron |

## rl — Reinforcement Learning

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [rl/nemo_rl/dpo](rl/nemo_rl/dpo/) | Direct Preference Optimisation alignment with NeMo-RL. Consumes a preference dataset (chosen / rejected pairs) and an SFT-trained checkpoint. | training_jsonl, checkpoint_megatron | checkpoint_megatron |
| [rl/nemo_rl/rlhf](rl/nemo_rl/rlhf/) | RLHF with a learned judge / generative reward model on top of NeMo-RL's GRPO loop. Uses NeMo-Gym for GenRM-style comparison rewards by default. | training_jsonl, checkpoint_megatron, checkpoint_hf | checkpoint_megatron |
| [rl/nemo_rl/rlvr](rl/nemo_rl/rlvr/) | RL with Verifiable Rewards via GRPO (NeMo-RL). Designed for tasks with programmatic reward signals such as math problem solving or unit-tested code. Use config/nemo_gym.yaml for NeMo-Gym resource-server rewards. | training_jsonl, checkpoint_megatron | checkpoint_megatron |

## sft — Supervised Fine-Tuning

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [sft/automodel](sft/automodel/) | Supervised fine-tuning with the AutoModel stack. Best for smaller GPU counts, rapid iteration, and LoRA-style adapter tuning on JSONL datasets that already use OpenAI chat-format messages. | training_jsonl | checkpoint_hf |
| [sft/megatron_bridge](sft/megatron_bridge/) | Supervised fine-tuning using NVIDIA Megatron-Bridge. Best for large-scale distributed training with tensor/pipeline/context parallelism. Requires packed Parquet data from prep/sft_packing. | packed_parquet, checkpoint_megatron (optional) | checkpoint_megatron |

## synth — Synthetic Data Generation

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [synth/data_designer](synth/data_designer/) | Build a NeMo Data Designer pipeline declaratively and generate synthetic data. Two recipes ship in config/: 'default' produces SFT chat data, 'rl_pref' produces preference pairs (chosen / rejected) for DPO.  Customisation lives in YAML — step.py just translates declarative column specs into the upstream DataDesignerConfigBuilder API. | training_jsonl (optional) | synthetic_jsonl |

## translate — Translation

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| [translate/nemo_skills](translate/nemo_skills/) | Translate filtered JSONL into a target language with NeMo Skills and attach FAITH-based quality signals so downstream steps can keep high-faith training data. | filtered_jsonl | translated_jsonl |
