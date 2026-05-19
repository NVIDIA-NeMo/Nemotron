#!/usr/bin/env python3
"""Smoke tests for optimizer, parameter, and autocast dtype behavior.

Run from the repository root:
    uv run --project src/nemotron/recipes/rerank/stage2_finetune \
        python tests/smoke_optimizer_dtype.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, BertConfig, BertForSequenceClassification


def _first_float_param_dtype(model: nn.Module) -> torch.dtype:
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    raise AssertionError("model has no floating-point parameters")


def _adamw_exp_avg_dtype(optimizer: torch.optim.Optimizer) -> torch.dtype:
    for state in optimizer.state.values():
        exp_avg = state.get("exp_avg")
        if isinstance(exp_avg, torch.Tensor):
            return exp_avg.dtype
    raise AssertionError("AdamW did not create exp_avg state")


def smoke_transformers_dtype_kwarg_controls_loaded_weight_dtype() -> None:
    """Validate the dtype kwarg that our raw Automodel config now passes through."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        config = BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=2,
        )
        BertForSequenceClassification(config).save_pretrained(model_dir)

        fp32_model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype="float32")
        bf16_model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype="bfloat16")

        assert _first_float_param_dtype(fp32_model) == torch.float32
        assert _first_float_param_dtype(bf16_model) == torch.bfloat16

    print("transformers dtype load: ok")


def _run_autocast_adamw_case(param_dtype: torch.dtype, device_type: str) -> dict[str, torch.dtype]:
    device = torch.device(device_type)
    torch.manual_seed(1234)

    model = nn.Sequential(nn.Linear(8, 4), nn.GELU(), nn.Linear(4, 2)).to(device=device, dtype=param_dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    inputs = torch.randn(3, 8, device=device, dtype=torch.float32)

    param_before = _first_float_param_dtype(model)
    with torch.amp.autocast(device_type, dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = outputs.float().square().mean()

    loss.backward()
    grad_dtype = next(model.parameters()).grad.dtype
    optimizer.step()
    state_dtype = _adamw_exp_avg_dtype(optimizer)
    param_after = _first_float_param_dtype(model)

    return {
        "param_before": param_before,
        "output": outputs.dtype,
        "grad": grad_dtype,
        "adamw_state": state_dtype,
        "param_after": param_after,
    }


def _run_autocast_adamw_pair(device_type: str) -> tuple[dict[str, torch.dtype], dict[str, torch.dtype]]:
    fp32 = _run_autocast_adamw_case(torch.float32, device_type)
    bf16 = _run_autocast_adamw_case(torch.bfloat16, device_type)
    return fp32, bf16


def _select_device_type() -> str:
    requested = os.environ.get("SMOKE_DTYPE_DEVICE", "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise AssertionError("SMOKE_DTYPE_DEVICE must be one of: auto, cpu, cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise AssertionError("SMOKE_DTYPE_DEVICE=cuda requested, but CUDA is unavailable")
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def smoke_autocast_changes_compute_not_parameter_or_optimizer_state_dtype() -> None:
    """Autocast makes the matmul output bf16, but params and AdamW state keep param dtype."""
    device_type = _select_device_type()
    if device_type == "cpu":
        print("CUDA unavailable; using CPU bf16 autocast as a local approximation")

    try:
        fp32, bf16 = _run_autocast_adamw_pair(device_type)
    except torch.AcceleratorError as e:
        if device_type != "cuda" or "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        print("CUDA OOM during smoke test; falling back to CPU bf16 autocast approximation")
        device_type = "cpu"
        fp32, bf16 = _run_autocast_adamw_pair(device_type)

    assert fp32["output"] == torch.bfloat16
    assert bf16["output"] == torch.bfloat16

    assert fp32["param_before"] == torch.float32
    assert fp32["param_after"] == torch.float32
    assert fp32["grad"] == torch.float32
    assert fp32["adamw_state"] == torch.float32

    assert bf16["param_before"] == torch.bfloat16
    assert bf16["param_after"] == torch.bfloat16
    assert bf16["grad"] == torch.bfloat16
    assert bf16["adamw_state"] == torch.bfloat16

    print("autocast + AdamW dtype smoke: ok")
    print(f"  fp32 params -> output={fp32['output']}, grad={fp32['grad']}, adamw_state={fp32['adamw_state']}")
    print(f"  bf16 params -> output={bf16['output']}, grad={bf16['grad']}, adamw_state={bf16['adamw_state']}")


def smoke_small_adamw_update_is_lost_with_bf16_parameters() -> None:
    """A tiny AdamW update can disappear when the parameter itself is stored in bf16."""

    def run_case(param_dtype: torch.dtype) -> dict[str, torch.dtype | float]:
        model = nn.Linear(1, 1, bias=False).to(dtype=param_dtype)
        with torch.no_grad():
            model.weight.fill_(1.0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=0.0)
        inputs = torch.ones(1, 1, dtype=torch.float32)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = outputs.float().sum()

        loss.backward()
        optimizer.step()
        state = next(iter(optimizer.state.values()))
        return {
            "output_dtype": outputs.dtype,
            "grad_dtype": model.weight.grad.dtype,
            "state_dtype": state["exp_avg"].dtype,
            "weight": float(model.weight.detach().float()),
        }

    fp32 = run_case(torch.float32)
    bf16 = run_case(torch.bfloat16)

    assert fp32["output_dtype"] == torch.bfloat16
    assert bf16["output_dtype"] == torch.bfloat16
    assert fp32["weight"] < 1.0
    assert bf16["weight"] == 1.0

    print("small AdamW update precision smoke: ok")
    print(f"  fp32 params -> weight_after={fp32['weight']}, grad={fp32['grad_dtype']}, state={fp32['state_dtype']}")
    print(f"  bf16 params -> weight_after={bf16['weight']}, grad={bf16['grad_dtype']}, state={bf16['state_dtype']}")


def smoke_flash_adamw_preserves_small_update_in_master_weight_state() -> None:
    """FlashAdamW keeps bf16 visible params but preserves tiny updates in 32-bit effective master weights."""
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping FlashAdamW CUDA smoke")
        return

    try:
        from flashoptim import FlashAdamW
    except ImportError:
        print("flashoptim unavailable; skipping FlashAdamW CUDA smoke")
        return

    model = nn.Linear(1, 1, bias=False, device="cuda").to(dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1.0)

    optimizer = FlashAdamW(
        model.parameters(),
        lr=1.0e-3,
        weight_decay=0.0,
        master_weight_bits=32,
        quantize=False,
        compress_state_dict=False,
        fused=True,
    )
    inputs = torch.ones(1, 1, device="cuda", dtype=torch.float32)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = outputs.float().sum()

    loss.backward()
    optimizer.step()

    visible_weight = float(model.weight.detach().float().cpu())
    fp32_state = optimizer.get_fp32_model_state_dict(model)
    reconstructed_weight = float(next(iter(fp32_state.values())).detach().float().cpu())
    state = next(iter(optimizer.state.values()))

    assert outputs.dtype == torch.bfloat16
    assert model.weight.grad.dtype == torch.bfloat16
    assert visible_weight == 1.0
    assert reconstructed_weight < visible_weight
    assert state["error_bits"].dtype == torch.int16

    print("FlashAdamW bf16-param master-weight smoke: ok")
    print(
        "  visible_bf16_weight="
        f"{visible_weight}, reconstructed_master_weight={reconstructed_weight}, "
        f"error_bits={state['error_bits'].dtype}"
    )


def main() -> None:
    smoke_transformers_dtype_kwarg_controls_loaded_weight_dtype()
    smoke_autocast_changes_compute_not_parameter_or_optimizer_state_dtype()
    smoke_small_adamw_update_is_lost_with_bf16_parameters()
    smoke_flash_adamw_preserves_small_update_in_master_weight_state()


if __name__ == "__main__":
    main()
