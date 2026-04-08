# SKILL: Stage 5 -- Quantization

## Purpose

Compress the trained model for efficient deployment by reducing numerical precision (INT4, INT8, FP8). Quantization reduces model size and inference latency while maintaining acceptable accuracy, making deployment feasible on smaller GPU configurations.

## When to Use

Run this stage when:
- Deploying to production (inference cost optimization)
- Deploying to edge devices or smaller GPU configurations
- Latency is a critical requirement
- Serving multiple models on shared infrastructure

Skip this stage if:
- Running research experiments (use full-precision checkpoint)
- Accuracy is the sole metric (quantization introduces small accuracy loss)
- Deployment infrastructure has ample GPU memory

## Inputs Required

Before running this stage, confirm these with the user:

| Input | Required? | Default | Notes |
|-------|-----------|---------|-------|
| Model path to quantize | Yes | None | Ask: "Path to the model checkpoint to quantize? (from RL, SFT, or CPT stage)" |
| Quantization method | Yes | `fp8` | Ask: "Which quantization method? FP8 (<0.5% accuracy loss), INT4-AWQ (~4x smaller), or INT8-SQ (balanced)?" |
| Calibration dataset | No | `cnn_dailymail` | Ask: "Calibration dataset? (domain-representative data recommended; provide a JSONL path or use default)" |
| Calibration sample count | No | 512 | Ask: "How many calibration samples? (256-4096, more = better calibration)" |
| Export format | No | `huggingface` | Ask: "Export format? (huggingface or trt_llm)" |
| Output directory | Yes | None | Ask: "Where should the quantized model be saved?" |
| Max accuracy drop tolerance | No | 2% | Ask: "Maximum acceptable accuracy drop from quantization? (used for validation)" |

If any required input is missing, ask the user before proceeding.

## Quantization Methods

| Method | Precision | Size Reduction | Accuracy Impact | Speed | Best For |
|--------|-----------|---------------|-----------------|-------|----------|
| INT4 AWQ | 4-bit weights | ~4x | 1-3% drop | Very fast inference | Production deployment |
| INT8 SmoothQuant | 8-bit weights | ~2x | <1% drop | Fast inference | Balanced quality/speed |
| FP8 | 8-bit float | ~2x | <0.5% drop | Fast inference | High-accuracy deployment |

### Decision Tree

```
Is accuracy loss tolerance < 1%?
  YES --> Use FP8 or INT8 SmoothQuant (int8_sq)
  NO  --> Is model size the primary constraint?
           YES --> Use INT4 AWQ (int4_awq)
           NO  --> Use INT8 SmoothQuant (int8_sq, best balance)
```

## Prerequisites

| Prerequisite | Description |
|-------------|-------------|
| Model checkpoint | From stage2_rl (or stage1_sft if RL was skipped) |
| Calibration data | Representative text samples for quantization calibration |
| TensorRT-LLM | `pip install tensorrt-llm` (requires CUDA) |
| TensorRT Model Optimizer | For advanced quantization (AWQ, SmoothQuant) |
| 1 node x 8 GPUs | Quantization is single-node; inference testing may need less |

## Calibration Data

Quantization requires calibration data to determine optimal scaling factors. This data should be representative of the model's expected inputs.

**Best practices:**
- Use 512-2048 samples from the target domain
- Include both short and long sequences
- Include the languages the model will serve
- Do NOT use training data exclusively -- include held-out samples

Calibration data is a JSONL file where each line has a `"text"` field. You can prepare it from any domain corpus:

```bash
# Example: extract calibration samples from a domain corpus using standard tools
python -c "
import json, random
with open('/data/domain_corpus/train.jsonl') as f:
    records = [json.loads(line) for line in f if line.strip()]
random.seed(42)
samples = random.sample(records, min(1024, len(records)))
with open('/data/calibration_data.jsonl', 'w') as out:
    for r in samples:
        out.write(json.dumps({'text': r.get('text', '')}) + '\n')
print(f'Wrote {len(samples)} calibration samples')
"
```

## Config Reference (`config/default.yaml`)

```yaml
# Input model
model:
  name_or_path: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  trust_remote_code: true

# Quantization method: fp8, int4_awq, int8_sq
quantization:
  method: fp8
  output_dir: ./output/quantized_model

  # Calibration dataset (for PTQ methods)
  calibration:
    dataset: cnn_dailymail
    split: train
    num_samples: 512
    max_length: 2048

  # FP8-specific settings
  fp8:
    kv_cache_quant: true

  # INT4-AWQ-specific settings
  int4_awq:
    group_size: 128

  # INT8 SmoothQuant settings
  int8_sq:
    alpha: 0.5

# Export format: trt_llm, huggingface
export:
  format: huggingface
  dtype: float16
```

The `QuantizeConfig` dataclass (in `data_prep/quantize.py`) uses flat fields: `model_path`, `output_dir`, `method`, `calibration_data_path`, `calibration_num_samples`, `calibration_max_length`, `calibration_batch_size`, `awq_group_size`, `awq_zero_point`, `build_trt_engine`, `trt_tp_size`, `trt_max_batch_size`. Supported methods in `_METHOD_MAP`: `fp8`, `int4_awq`, `int8_sq`, `int8` (alias for `int8_sq`).

### Key Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `quantization.method` | fp8 | fp8/int4_awq/int8_sq | Primary quantization method |
| `quantization.calibration.num_samples` | 512 | 256-4096 | More = better calibration, slower |
| `quantization.calibration.max_length` | 2048 | 2048-8192 | Match model context length |
| `quantization.int4_awq.group_size` | 128 | 64/128/-1 | Smaller = more accurate, larger model |

## Execution

### INT4 AWQ Quantization

```bash
python src/nemotron/customization_recipes/nemotron/stage5_quantization/run_quantize.py \
  --config src/nemotron/customization_recipes/nemotron/stage5_quantization/config/default.yaml \
  model.name_or_path=/results/rl_checkpoint \
  quantization.output_dir=/deploy/hindi_medical_int4 \
  quantization.method=int4_awq
```

### FP8 Quantization

```bash
python src/nemotron/customization_recipes/nemotron/stage5_quantization/run_quantize.py \
  --config src/nemotron/customization_recipes/nemotron/stage5_quantization/config/default.yaml \
  model.name_or_path=/results/rl_checkpoint \
  quantization.output_dir=/deploy/hindi_medical_fp8 \
  quantization.method=fp8
```

### INT8 SmoothQuant Quantization

```bash
python src/nemotron/customization_recipes/nemotron/stage5_quantization/run_quantize.py \
  --config src/nemotron/customization_recipes/nemotron/stage5_quantization/config/default.yaml \
  model.name_or_path=/results/rl_checkpoint \
  quantization.output_dir=/deploy/hindi_medical_int8 \
  quantization.method=int8_sq
```

### Validation After Quantization

Validation is handled through stage4_eval. Run a benchmark evaluation comparing the quantized model against the full-precision baseline:

```bash
# Evaluate the quantized model using the standard eval pipeline
nemotron customize eval --run MY-CLUSTER \
  -t adlr_mmlu \
  -t byob_hindi_medical_mcq.hindi-medical-mcq \
  deployment.checkpoint_path=/deploy/hindi_medical_int4

# Compare results against the full-precision evaluation
# to verify accuracy drop is within acceptable thresholds
```

## How to Verify Success

1. **Accuracy validation passes**: Accuracy drop is within `max_accuracy_drop` threshold.
   ```
   Full-precision MMLU:     67.2%
   INT4 AWQ MMLU:           65.8%  (drop: 1.4% -- PASS, threshold 2%)
   ```

2. **Perplexity ratio**: Quantized perplexity / full precision perplexity < `perplexity_threshold`.
   ```
   Full-precision perplexity:  8.42
   INT4 AWQ perplexity:        8.71  (ratio: 1.034 -- PASS, threshold 1.05)
   ```

3. **Model loads successfully**: Quantized model can be loaded and generates coherent text.
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("/deploy/hindi_medical_int4")
   tokenizer = AutoTokenizer.from_pretrained("/deploy/hindi_medical_int4")
   output = model.generate(tokenizer.encode("Hello", return_tensors="pt"), max_length=50)
   print(tokenizer.decode(output[0]))
   ```

4. **File size reduction**: Quantized model should be ~2-4x smaller than original.
   ```bash
   du -sh /results/rl_checkpoint          # e.g., 60GB
   du -sh /deploy/hindi_medical_int4      # e.g., 16GB (INT4)
   ```

5. **Inference speed**: Measure tokens/second on representative prompts.
   - INT4 should be 1.5-3x faster than full precision
   - FP8 should be 1.3-2x faster

## Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| OOM during calibration | Calibration batch too large | Reduce `calibration.batch_size` to 1-2 |
| Accuracy drop >5% | Poor calibration data or aggressive quantization | Use more representative calibration data, switch to FP8 or INT8 |
| Model outputs garbage after quantization | Quantization failed silently | Check for NaN in quantized weights, try different `group_size` |
| TRT-LLM engine build fails | Version mismatch or unsupported architecture | Verify TRT-LLM version compatibility with model architecture |
| Quantized model loads but is slow | Not using optimized kernels | Verify AWQ kernel is installed, check quantization backend |
| FP8 not available | GPU does not support FP8 | FP8 requires Hopper (H100) or later; use INT8 instead |
| Calibration data loading error | Wrong format or tokenizer mismatch | Ensure JSONL format with "text" field, verify tokenizer matches model |

## Artifacts Produced

| Artifact | Type | Path | Consumed By |
|----------|------|------|-------------|
| Quantized model | HF checkpoint | `output_dir/` | Deployment (NIM, TRT-LLM, vLLM) |
| TRT-LLM engine (optional) | Engine files | `output_dir/trt_engine/` | TRT-LLM inference server |
| Calibration metadata | JSON | `output_dir/calibration_meta.json` | Reproducibility |
| Validation report | JSON | `output_dir/validation_report.json` | Quality gate |
