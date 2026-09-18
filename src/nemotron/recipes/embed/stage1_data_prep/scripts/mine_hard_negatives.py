# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To run this script, use the following command:
# torchrun --nproc_per_node=8 --master_port=29500 ./examples/biencoder/mine_hard_negatives.py \
#     --config examples/biencoder/mining_config.yaml \
#     --mining.model_name_or_path /path/to/biencoder/checkpoint \
#     --mining.train_qa_file_path /path/to/input.json \
#     --mining.train_file_output_path /path/to/output.json \
#     --mining.cache_embeddings_dir /path/to/cache \
#     --mining.hard_neg_margin 0.95
#
# The model is loaded directly from the checkpoint path (--mining.model_name_or_path),
# so no model architecture config is needed. This allows mining with any saved
# biencoder checkpoint without requiring the original training config.
#
# The mining_config.yaml contains only mining parameters and dist_env settings,
# not the model architecture. All mining parameters can also be overridden via
# command line arguments.

from __future__ import annotations

from pathlib import Path

from nemo_automodel._transformers.auto_model import NeMoAutoModelBiEncoder
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.retrieval import mine_hard_negatives as automodel_mining

from nemotron.recipes.embed.stage1_data_prep.mining_provenance import restore_mining_provenance


class MineHardNegativesRecipe(automodel_mining.MineHardNegativesRecipe):
    """Automodel miner that explicitly supports trusted custom HF architectures."""

    def setup(self):
        """Build the miner, forwarding ``trust_remote_code`` to AutoModel."""
        if self.cfg.get("mining.multimodal_encoder", None) is not None:
            if "multimodal_encoder" not in automodel_mining.MINING_DEFAULTS:
                raise RuntimeError(
                    "Native multimodal mining requires an AutoModel revision with multimodal_encoder support. "
                    "Install the matching AutoModel revision; the released text-only miner cannot process images."
                )
            # The native encoder owns multimodal preprocessing. Keep the legacy
            # setup below for text-only recipes running released AutoModel 0.4.
            super().setup()
            return
        self.dist_env = automodel_mining.build_distributed(self.cfg.get("dist_env", {}))

        self.mining_cfg = self.cfg.get("mining", None)
        if self.mining_cfg is None:
            raise ValueError("Missing mining configuration")

        self._extract_mining_params()
        self._validate_mining_params()

        model_kwargs = {
            "use_liger_kernel": False,
            "use_sdpa_patching": True,
            "trust_remote_code": self._get_mining_param("trust_remote_code", False),
        }
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        automodel_mining.logger.info(f"Loading encoder model from {self.model_name_or_path}...")
        self.model = NeMoAutoModelBiEncoder.from_pretrained(
            self.model_name_or_path,
            **model_kwargs,
        )
        self.model = self.model.to(self.dist_env.device)
        self.model.eval()

        self._configure_tokenizer()
        self._load_data()
        self._build_document_mappings()
        self._prepare_data()

    def _configure_tokenizer(self) -> None:
        """Load the tokenizer, optionally bypassing model-type registry dispatch."""
        if not self._get_mining_param("tokenizer_force_default", False):
            super()._configure_tokenizer()
            return

        tokenizer_kwargs = {"force_default": True}
        if self.add_bos_token is not None:
            tokenizer_kwargs["add_bos_token"] = self.add_bos_token
        if self.add_eos_token is not None:
            tokenizer_kwargs["add_eos_token"] = self.add_eos_token
        self.tokenizer = NeMoAutoTokenizer.from_pretrained(self.tokenizer_name_or_path, **tokenizer_kwargs)
        automodel_mining.logger.info("Using AutoModel's Hugging Face tokenizer wrapper")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"


def main(default_config_path="examples/biencoder/mining_config.yaml"):
    """Main entry point for hard negative mining.

    Loads the configuration, sets up the recipe, and runs the mining pipeline.

    The model is loaded directly from --mining.model_name_or_path, so a full
    model architecture config is not required. The default config file only
    contains mining parameters and dist_env settings.

    Args:
        default_config_path: Path to config file with mining parameters.
                            The model architecture is not specified here -
                            it's loaded directly from model_name_or_path.
    """
    cfg = parse_args_and_load_config(default_config_path)
    recipe = MineHardNegativesRecipe(cfg)
    recipe.setup()
    recipe.run()
    if recipe.dist_env.is_main:
        restore_mining_provenance(Path(recipe.train_qa_file_path), Path(recipe.train_file_output_path))


if __name__ == "__main__":
    main()
