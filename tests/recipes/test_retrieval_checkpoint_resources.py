# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemotron.recipes._retrieval import checkpoint

GIB = 1024**3
FLASH = {
    "_target_": "flashoptim.FlashAdamW",
    "compress_state_dict": False,
    "quantize": False,
    "master_weight_bits": 32,
}


def _local_model(tmp_path: Path, *, dtype: str = "bfloat16") -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"dtype": dtype}))
    (model / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 2 * GIB}}))
    return model


@pytest.mark.parametrize("has_previous,additional", [(False, "20.0"), (True, "10.0")])
def test_checkpoint_headroom_accounts_for_previous_save(
    tmp_path: Path, monkeypatch, capsys, has_previous: bool, additional: str
) -> None:
    model = _local_model(tmp_path)
    destination = tmp_path / "output" / "checkpoints"
    if has_previous:
        previous = destination / "epoch_0_step_0"
        previous.mkdir(parents=True)
        (destination / "LATEST").symlink_to(previous.name)
    monkeypatch.setattr(checkpoint.shutil, "disk_usage", lambda path: SimpleNamespace(free=12 * GIB))
    monkeypatch.setattr(checkpoint, "_available_host_memory", lambda: 4 * GIB)

    checkpoint.report_checkpoint_resources(
        destination, base_model=str(model), model_revision=None, optimizer_config=FLASH
    )

    output = capsys.readouterr().out
    assert "full checkpoint with consolidated export: 10.0 GiB" in output
    assert f"through the next save/retention cycle: {additional} GiB" in output
    assert ("available disk is below" in output) is (not has_previous)
    assert "available host RAM is below" in output
    assert "6.0 GiB of optimizer tensors" in output
    assert "max_recent_checkpoints=1" in output


@pytest.mark.parametrize("emit", [False, True])
def test_full_filesystem_fails_on_every_rank(tmp_path: Path, monkeypatch, emit: bool) -> None:
    monkeypatch.setattr(checkpoint.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="no available disk space"):
        checkpoint.report_checkpoint_resources(
            tmp_path / "not-created",
            base_model="custom/model",
            model_revision=None,
            optimizer_config=FLASH,
            emit=emit,
        )


@pytest.mark.parametrize("emit", [False, True])
def test_exact_validated_revision_requires_space_for_model_payload(tmp_path: Path, monkeypatch, emit: bool) -> None:
    monkeypatch.setattr(checkpoint.shutil, "disk_usage", lambda path: SimpleNamespace(free=2 * GIB))
    with pytest.raises(ValueError, match=r"less than the validated model payload alone \(15\.6 GiB\)"):
        checkpoint.report_checkpoint_resources(
            tmp_path,
            base_model="nvidia/Nemotron-3.5-Rerank-8B-BF16",
            model_revision="39aa01525864f736e63f1d5742fc4cf59e754abd",
            optimizer_config=FLASH,
            emit=emit,
        )


@pytest.mark.parametrize("revision", [None, "custom-revision"])
def test_other_revisions_do_not_inherit_strict_payload_requirement(tmp_path: Path, monkeypatch, revision) -> None:
    monkeypatch.setattr(checkpoint.shutil, "disk_usage", lambda path: SimpleNamespace(free=2 * GIB))
    checkpoint.report_checkpoint_resources(
        tmp_path,
        base_model="nvidia/Nemotron-3.5-Rerank-8B-BF16",
        model_revision=revision,
        optimizer_config=FLASH,
        emit=False,
    )


def test_unknown_model_does_not_inherit_default_model_estimates(tmp_path: Path, monkeypatch, capsys) -> None:
    model = _local_model(tmp_path, dtype="float32")
    monkeypatch.setattr(checkpoint.shutil, "disk_usage", lambda path: SimpleNamespace(free=GIB))
    monkeypatch.setattr(checkpoint, "_available_host_memory", lambda: None)
    checkpoint.report_checkpoint_resources(
        tmp_path / "checkpoints", base_model=str(model), model_revision=None, optimizer_config=FLASH
    )
    output = capsys.readouterr().out
    assert "Available disk: 1.0 GiB" in output
    assert "Checkpoint estimate unavailable" in output
    assert "Estimated full checkpoint" not in output


def test_estimate_accepts_automodel_serialized_optimizer_target(tmp_path: Path, monkeypatch, capsys) -> None:
    # Captured from the pinned ConfigNode.to_yaml_dict(); the native integration
    # test verifies this representation against the optional model runtime.
    optimizer = {**FLASH, "_target_": "flashoptim.optimizers.FlashAdamW"}
    monkeypatch.setattr(checkpoint.shutil, "disk_usage", lambda path: SimpleNamespace(free=200 * GIB))
    monkeypatch.setattr(checkpoint, "_available_host_memory", lambda: 128 * GIB)

    checkpoint.report_checkpoint_resources(
        tmp_path,
        base_model="nvidia/Nemotron-3.5-Rerank-8B-BF16",
        model_revision="39aa01525864f736e63f1d5742fc4cf59e754abd",
        optimizer_config=optimizer,
    )

    output = capsys.readouterr().out
    assert "full checkpoint with consolidated export: 78.1 GiB" in output
    assert "46.8 GiB of optimizer tensors" in output
    assert "Checkpoint estimate unavailable" not in output


def test_cgroup_limit_bounds_available_host_memory(monkeypatch) -> None:
    files = {
        "/proc/meminfo": "MemTotal: 262144000 kB\nMemAvailable: 104857600 kB\n",
        "/proc/self/cgroup": "0::/job/worker\n",
        "/sys/fs/cgroup/memory.max": "max\n",
        "/sys/fs/cgroup/job/memory.max": str(32 * GIB),
        "/sys/fs/cgroup/job/memory.current": str(24 * GIB),
        "/sys/fs/cgroup/job/worker/memory.max": str(16 * GIB),
        "/sys/fs/cgroup/job/worker/memory.current": str(2 * GIB),
    }

    def read_text(path: Path, *args, **kwargs) -> str:
        try:
            return files[str(path)]
        except KeyError:
            raise FileNotFoundError(path) from None

    monkeypatch.setattr(Path, "read_text", read_text)
    assert checkpoint._available_host_memory() == 8 * GIB


@pytest.mark.parametrize(
    "optimizer,expected",
    [
        (FLASH, 6 * GIB),
        ({**FLASH, "quantize": True}, 10 * GIB),
        ({**FLASH, "master_weight_bits": 24}, 6 * GIB),
        ({**FLASH, "master_weight_bits": None}, 4 * GIB),
        (
            {
                "_target_": "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam",
                "master_weights": True,
            },
            12 * GIB,
        ),
        ({**FLASH, "compress_state_dict": True}, None),
        ({**FLASH, "_target_": "custom.optimizers.FlashAdamW"}, None),
    ],
)
def test_optimizer_estimate_tracks_actual_serialization(optimizer: dict, expected: int | None) -> None:
    assert checkpoint._optimizer_bytes(2 * GIB, optimizer) == expected
