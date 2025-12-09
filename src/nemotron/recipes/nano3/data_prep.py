"""Data preparation for Nemotron Nano3 training stages.

Tokenizes raw text data into Megatron-Bridge format for training.

CLI Usage:
    # Pretrain stage (default)
    uv run python -m nemotron.recipes.nano3.data_prep --stage pretrain

    # SFT stage
    uv run python -m nemotron.recipes.nano3.data_prep -s sft

    # RL stage with sampling
    uv run python -m nemotron.recipes.nano3.data_prep -s rl --sample 1000

    # With config file (YAML, TOML, or JSON)
    uv run python -m nemotron.recipes.nano3.data_prep -s pretrain --config-file config.yaml

    # Custom settings via CLI
    uv run python -m nemotron.recipes.nano3.data_prep -s sft \\
        --output-dir /data/tokenized \\
        --num-shards 256 \\
        --tokenizer-model nvidia/Llama-3.1-Nemotron-Nano-8B-Instruct

Example config.yaml:
    ```yaml
    output_dir: /data/tokenized
    num_shards: 256
    tokenizer_model: nvidia/Llama-3.1-Nemotron-Nano-8B-Instruct
    sample: 1000  # Limit for fast iteration
    ```

Programmatic Usage:
    from nemotron.recipes.nano3.data_prep import Nano3DataPrepConfig, Stage
    from nemotron.data_prep import run_data_prep

    # Pretrain stage (default)
    config = Nano3DataPrepConfig(stage=Stage.pretrain)

    # SFT stage with sampling
    config = Nano3DataPrepConfig(stage=Stage.sft, sample=1000)

    # Execute
    artifact = run_data_prep(config)
    print(f"Blend path: {artifact.path}")
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated

import tyro

from nemotron.kit import cli, print_step_complete, DataBlendsArtifact
from nemotron.data_prep import DataPrepConfig, run_data_prep


# Base path for nano3 recipe
NANO3_RECIPE_PATH = Path(__file__).parent


class Stage(str, Enum):
    """Training stage for data preparation."""

    pretrain = "pretrain"
    sft = "sft"
    rl = "rl"


# Stage directory mapping
STAGE_DIRS = {
    Stage.pretrain: "stage0_pretrain",
    Stage.sft: "stage1_sft",
    Stage.rl: "stage2_rl",
}


@dataclass
class Nano3DataPrepConfig:
    """Data preparation config with Nano3-specific defaults.

    Defaults:
    - Llama-3.2-1B tokenizer (Nano3 base model)
    - 128 shards (suitable for multi-node training)
    - 99990:8:2 split ratio (~99.99% train)
    - EOS tokens appended (standard for pretraining)

    Use --stage/-s to select pretrain, sft, or rl data blend.
    """

    # Stage selection - displayed first
    stage: Annotated[
        Stage,
        tyro.conf.arg(aliases=["-s"]),
    ] = Stage.pretrain
    """Training stage: pretrain, sft, or rl"""

    blend_path: Path = field(
        default_factory=lambda: NANO3_RECIPE_PATH / STAGE_DIRS[Stage.pretrain] / "data_blend_raw.json"
    )
    """Path to data blend JSON file"""

    output_dir: Path = field(
        default_factory=lambda: Path(f"./output/nano3/{STAGE_DIRS[Stage.pretrain]}")
    )
    """Output directory for tokenized data"""

    num_shards: int = 128
    """Number of output shards for parallel loading"""

    split: str | None = "99990,8,2"
    """Train:valid:test ratio (e.g., '99990,8,2') or None to disable"""

    tokenizer_model: str = "meta-llama/Llama-3.2-1B"
    """HuggingFace tokenizer model name"""

    add_bos: bool = False
    """Prepend BOS token to documents"""

    add_eos: bool = True
    """Append EOS token to documents"""

    text_field: str = "text"
    """Default text field name in datasets"""

    min_doc_chars: int | None = None
    """Skip documents shorter than this"""

    max_doc_tokens: int | None = None
    """Truncate documents longer than this"""

    sample: int | None = None
    """Limit rows per dataset (for quick tests)"""

    num_actors: int | None = None
    """Ray actors for parallel processing (None = auto)"""

    force: bool = False
    """Force new run, ignoring cache"""

    def __post_init__(self) -> None:
        """Update blend_path and output_dir based on stage and sample."""
        selected_stage = self.stage

        # Check if blend_path is still the default pretrain path
        default_pretrain_blend = NANO3_RECIPE_PATH / STAGE_DIRS[Stage.pretrain] / "data_blend_raw.json"
        if self.blend_path == default_pretrain_blend and selected_stage != Stage.pretrain:
            self.blend_path = NANO3_RECIPE_PATH / STAGE_DIRS[selected_stage] / "data_blend_raw.json"

        # Check if output_dir is still the default pretrain path
        default_pretrain_output = Path(f"./output/nano3/{STAGE_DIRS[Stage.pretrain]}")
        if self.output_dir == default_pretrain_output:
            # Update for stage if not pretrain
            if selected_stage != Stage.pretrain:
                self.output_dir = Path(f"./output/nano3/{STAGE_DIRS[selected_stage]}")
            # Append sample size to path if sampling
            if self.sample is not None:
                self.output_dir = self.output_dir / f"sample-{self.sample}"


def main(cfg: Nano3DataPrepConfig) -> DataBlendsArtifact:
    """CLI entry point for data preparation.

    Runs the pipeline and prints completion summary.
    """
    # Convert to DataPrepConfig for run_data_prep
    data_prep_config = DataPrepConfig(
        blend_path=cfg.blend_path,
        output_dir=cfg.output_dir,
        num_shards=cfg.num_shards,
        split=cfg.split,
        tokenizer_model=cfg.tokenizer_model,
        add_bos=cfg.add_bos,
        add_eos=cfg.add_eos,
        text_field=cfg.text_field,
        min_doc_chars=cfg.min_doc_chars,
        max_doc_tokens=cfg.max_doc_tokens,
        sample=cfg.sample,
        num_actors=cfg.num_actors,
        force=cfg.force,
    )
    artifact = run_data_prep(data_prep_config)

    # Set semantic artifact name: nano3/{stage}/data[?sample=N]
    name_parts = [f"nano3/{cfg.stage.value}/data"]
    if cfg.sample is not None:
        name_parts.append(f"?sample={cfg.sample}")
    artifact.name = "".join(name_parts)

    print_step_complete(data_prep=artifact)
    return artifact


if __name__ == "__main__":
    cli(main)
