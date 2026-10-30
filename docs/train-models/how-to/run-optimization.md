# Run Post-Training Optimization

Optimization steps apply NVIDIA Model Optimizer flows after you have a trained model that is worth compressing. Run them only on checkpoints that already passed your quality bar.

## Steps

| Step id | Purpose | Typical input | Output |
|---------|---------|---------------|--------|
| `optimize/modelopt/quantize` | Post-training quantization | `checkpoint_hf` | `checkpoint_megatron` |
| `optimize/modelopt/prune` | Structured pruning | `checkpoint_hf` | `checkpoint_hf` |
| `optimize/modelopt/distill` | Teacher-student recovery | Teacher and student `checkpoint_hf`, optional `binidx` | `checkpoint_megatron` |

## Decision Flow

1. If you only need a smaller numeric type for inference, start with `optimize/modelopt/quantize`. Pick a quantization recipe that matches your hardware. FP8 suits Hopper. NVFP4 suits Blackwell. Read `step.toml` for parameter names and allowed values.
2. If you need a smaller architecture, use `optimize/modelopt/prune`.
3. If quality drops after compression, use `optimize/modelopt/distill` with a full-precision teacher while the student matches the compressed artifact.
4. Do not run optimization on unmerged adapters. When the source was parameter-efficient fine tuning (PEFT) with low-rank adaptation (LoRA), merge LoRA into a base checkpoint first.

## Order of Operations

Distillation after pruning is the usual recovery order when both apply. Quantization is largely independent. Quantization still needs a benchmark before and after you run it.

## Sample Commands

```console
$ uv run nemotron steps run optimize/modelopt/quantize -c tiny
$ uv run nemotron steps run optimize/modelopt/prune -c tiny
$ uv run nemotron steps run optimize/modelopt/distill -c tiny
```

Tiny runs and mock-data runs validate end-to-end execution only. Judge final quality on full calibration or distillation data.

## Success Criteria

- You keep the original high-precision checkpoint and its evaluation baseline.
- Post-optimization evaluation uses the same benchmark suite as pre-optimization evaluation.

## Related Reading

- [Optimization Steps Reference](../reference/optimize/index.md)
- [Artifact Graph](../explanation/artifact-graph.md)
