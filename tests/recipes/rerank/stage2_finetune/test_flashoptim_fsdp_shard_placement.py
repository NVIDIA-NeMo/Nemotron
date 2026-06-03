from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nemotron.recipes.rerank.stage2_finetune.train import (  # noqa: E402
    _get_fsdp_shard_mesh_size,
    _patch_flashoptim_fsdp2_shard_placement,
)


def _torch_and_shard():
    torch = pytest.importorskip("torch")
    shard = pytest.importorskip("torch.distributed.tensor").Shard
    return torch, shard


class _Mesh:
    ndim = 1

    def __init__(self, size: int) -> None:
        self._size = size

    def size(self, mesh_dim: int) -> int:
        assert mesh_dim == 0
        return self._size


def test_get_fsdp_shard_mesh_size_uses_last_dim_for_hsdp() -> None:
    class Mesh2D:
        ndim = 2

        def size(self, mesh_dim: int) -> int:
            return (4, 8)[mesh_dim]

    assert _get_fsdp_shard_mesh_size(_Mesh(2)) == 2
    assert _get_fsdp_shard_mesh_size(Mesh2D()) == 8


def test_flashoptim_patch_shards_uneven_2d_params_on_dim1(monkeypatch: pytest.MonkeyPatch) -> None:
    torch, shard = _torch_and_shard()
    parallelizer = pytest.importorskip("nemo_automodel.components.distributed.parallelizer")
    captured: dict[str, object] = {}

    def fake_fully_shard(module: object, *args: object, **kwargs: object) -> object:
        captured["shard_placement_fn"] = kwargs["shard_placement_fn"]
        return module

    monkeypatch.setattr(parallelizer, "fully_shard", fake_fully_shard)
    _patch_flashoptim_fsdp2_shard_placement()

    module = object()
    assert parallelizer.fully_shard(module, mesh=_Mesh(2)) is module

    shard_placement_fn = captured["shard_placement_fn"]
    assert callable(shard_placement_fn)

    placement = shard_placement_fn(torch.nn.Parameter(torch.empty(1, 32)))
    assert isinstance(placement, shard)
    assert placement.dim == 1

    assert shard_placement_fn(torch.nn.Parameter(torch.empty(32, 32))) is None
    assert shard_placement_fn(torch.nn.Parameter(torch.empty(1, 31))) is None


def test_flashoptim_patch_preserves_existing_shard_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    torch, shard = _torch_and_shard()
    parallelizer = pytest.importorskip("nemo_automodel.components.distributed.parallelizer")
    captured: dict[str, object] = {}

    def fake_fully_shard(module: object, *args: object, **kwargs: object) -> object:
        captured["shard_placement_fn"] = kwargs["shard_placement_fn"]
        return module

    monkeypatch.setattr(parallelizer, "fully_shard", fake_fully_shard)
    _patch_flashoptim_fsdp2_shard_placement()

    parallelizer.fully_shard(
        object(),
        mesh=_Mesh(2),
        shard_placement_fn=lambda param: shard(0),
    )

    shard_placement_fn = captured["shard_placement_fn"]
    assert callable(shard_placement_fn)

    placement = shard_placement_fn(torch.nn.Parameter(torch.empty(1, 32)))
    assert isinstance(placement, shard)
    assert placement.dim == 0
