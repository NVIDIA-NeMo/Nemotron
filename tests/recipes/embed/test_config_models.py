"""Unit tests for embed recipe Pydantic config models.

Tests validation constraints, defaults, and model validators without any I/O.
"""

from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

from .conftest import STAGES, _import_config_class


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _all_config_classes():
    """Return (id_string, ConfigClass) for each stage."""
    return [(s["name"], _import_config_class(s)) for s in STAGES]


ALL_CONFIGS = _all_config_classes()


# ---------------------------------------------------------------------------
# TestExtraForbid — extra="forbid" catches unknown fields
# ---------------------------------------------------------------------------
class TestExtraForbid:
    @pytest.mark.parametrize("name,cls", ALL_CONFIGS, ids=[c[0] for c in ALL_CONFIGS])
    def test_unknown_field_raises(self, name, cls):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            cls(bogus_field="x")


# ---------------------------------------------------------------------------
# TestDefaults — all config classes constructible with no args
# ---------------------------------------------------------------------------
class TestDefaults:
    @pytest.mark.parametrize("name,cls", ALL_CONFIGS, ids=[c[0] for c in ALL_CONFIGS])
    def test_construct_with_defaults(self, name, cls):
        instance = cls()
        assert instance is not None


# ---------------------------------------------------------------------------
# TestSDGConfigValidation
# ---------------------------------------------------------------------------
class TestSDGConfigValidation:
    @pytest.fixture()
    def SDGConfig(self):
        mod = importlib.import_module("nemotron.recipes.embed.stage0_sdg.data_prep")
        return mod.SDGConfig

    def test_min_hops_exceeds_max_hops(self, SDGConfig):
        with pytest.raises(ValidationError, match="min_hops"):
            SDGConfig(min_hops=5, max_hops=2)

    def test_min_complexity_out_of_range_low(self, SDGConfig):
        with pytest.raises(ValidationError):
            SDGConfig(min_complexity=0)

    def test_min_complexity_out_of_range_high(self, SDGConfig):
        with pytest.raises(ValidationError):
            SDGConfig(min_complexity=6)

    def test_zero_batch_size(self, SDGConfig):
        with pytest.raises(ValidationError):
            SDGConfig(batch_size=0)


# ---------------------------------------------------------------------------
# TestDataPrepConfigValidation
# ---------------------------------------------------------------------------
class TestDataPrepConfigValidation:
    @pytest.fixture()
    def DataPrepConfig(self):
        mod = importlib.import_module("nemotron.recipes.embed.stage1_data_prep.data_prep")
        return mod.DataPrepConfig

    def test_ratios_not_summing_to_one(self, DataPrepConfig):
        with pytest.raises(ValidationError, match="1.0"):
            DataPrepConfig(train_ratio=0.5, val_ratio=0.1, test_ratio=0.1)

    def test_valid_ratios(self, DataPrepConfig):
        cfg = DataPrepConfig(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        assert abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) < 1e-6

    def test_quality_threshold_below_range(self, DataPrepConfig):
        with pytest.raises(ValidationError):
            DataPrepConfig(quality_threshold=-1)

    def test_quality_threshold_above_range(self, DataPrepConfig):
        with pytest.raises(ValidationError):
            DataPrepConfig(quality_threshold=11)


# ---------------------------------------------------------------------------
# TestFinetuneConfigValidation
# ---------------------------------------------------------------------------
class TestFinetuneConfigValidation:
    @pytest.fixture()
    def FinetuneConfig(self):
        mod = importlib.import_module("nemotron.recipes.embed.stage2_finetune.train")
        return mod.FinetuneConfig

    def test_zero_epochs(self, FinetuneConfig):
        with pytest.raises(ValidationError):
            FinetuneConfig(num_epochs=0)

    def test_negative_learning_rate(self, FinetuneConfig):
        with pytest.raises(ValidationError):
            FinetuneConfig(learning_rate=-1e-5)

    def test_zero_temperature(self, FinetuneConfig):
        with pytest.raises(ValidationError):
            FinetuneConfig(temperature=0)


# ---------------------------------------------------------------------------
# TestExportConfigValidation
# ---------------------------------------------------------------------------
class TestExportConfigValidation:
    @pytest.fixture()
    def ExportConfig(self):
        mod = importlib.import_module("nemotron.recipes.embed.stage4_export.export")
        return mod.ExportConfig

    def test_trt_batch_profile_order(self, ExportConfig):
        with pytest.raises(ValidationError, match="trt_min_batch"):
            ExportConfig(trt_min_batch=32, trt_opt_batch=16)

    def test_trt_seq_len_profile_order(self, ExportConfig):
        with pytest.raises(ValidationError, match="trt_opt_seq_len"):
            ExportConfig(trt_opt_seq_len=512, trt_max_seq_len=256)

    def test_invalid_quant_cfg(self, ExportConfig):
        with pytest.raises(ValidationError):
            ExportConfig(quant_cfg="bad")

    def test_invalid_export_dtype(self, ExportConfig):
        with pytest.raises(ValidationError):
            ExportConfig(export_dtype="bf16")


# ---------------------------------------------------------------------------
# TestDeployConfigValidation
# ---------------------------------------------------------------------------
class TestDeployConfigValidation:
    @pytest.fixture()
    def DeployConfig(self):
        mod = importlib.import_module("nemotron.recipes.embed.stage5_deploy.deploy")
        return mod.DeployConfig

    def test_port_zero(self, DeployConfig):
        with pytest.raises(ValidationError):
            DeployConfig(host_port=0)

    def test_port_too_high(self, DeployConfig):
        with pytest.raises(ValidationError):
            DeployConfig(host_port=70000)
