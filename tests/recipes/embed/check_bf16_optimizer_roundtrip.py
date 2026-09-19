# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bounded CUDA check of the existing BF16/FusedAdam checkpoint contract.

Run inside the supported training container. This tests the optimizer and
checkpoint serialization, not full distributed recipe training or model quality.
"""

import io
import json

import torch
from transformer_engine.pytorch.optimizers import FusedAdam


def advance(model, optimizer, inputs):
    """Perform one deterministic update with finite gradients."""
    optimizer.zero_grad()
    loss = model(inputs).float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()


def check_state(optimizer):
    """Require full-precision moments and master parameters for every model parameter."""
    counts = {}
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq", "master_param"):
            assert state[key].dtype == torch.float32, (key, state[key].dtype)
            counts[key] = counts.get(key, 0) + 1
    assert all(counts[key] == 2 for key in ("exp_avg", "exp_avg_sq", "master_param"))
    return counts


def main():
    """Verify saved model dtype, optimizer dtype, and exact continuation after reload."""
    torch.manual_seed(42)
    model = torch.nn.Linear(8, 4, device="cuda", dtype=torch.bfloat16)
    optimizer = FusedAdam(model.parameters(), lr=2e-6, adam_w_mode=True, master_weights=True)
    inputs = torch.randn(3, 8, device="cuda", dtype=torch.bfloat16)
    advance(model, optimizer, inputs)
    counts = check_state(optimizer)
    stream = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    assert all(value.dtype == torch.bfloat16 for value in saved["model"].values())
    restored = torch.nn.Linear(8, 4, device="cuda", dtype=torch.bfloat16)
    restored.load_state_dict(saved["model"])
    resumed = FusedAdam(restored.parameters(), lr=2e-6, adam_w_mode=True, master_weights=True)
    resumed.load_state_dict(saved["optimizer"])
    check_state(resumed)
    advance(model, optimizer, inputs)
    advance(restored, resumed, inputs)
    for actual, expected in zip(restored.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    check_state(resumed)
    print(
        json.dumps(
            {
                "passed": True,
                "model_checkpoint_dtype": "bfloat16",
                "optimizer_fp32": counts,
                "resume_matches_uninterrupted": True,
                "torch": torch.__version__,
            }
        )
    )


if __name__ == "__main__":
    main()
