"""Data preparation for Nano3 pretraining stage.

Tokenizes raw text data into Megatron bin/idx format.

Usage:
    python -m nemotron.recipes.nano3.stage0_pretrain.data_prep [options]
"""

from dataclasses import dataclass, field
from pathlib import Path

from nemotron.data_prep import DataPrepConfig, run_data_prep
from nemotron.kit import DataBlendsArtifact, cli, print_step_complete
from nemotron.kit.wandb import add_wandb_tags

STAGE_PATH = Path(__file__).parent


@dataclass
class PreTrainDataPrepConfig:
    """Pretrain data preparation config.

    Tokenizes text into Megatron bin/idx format for pretraining.
    """

    blend_path: Path = field(default_factory=lambda: STAGE_PATH / "data_blend_raw.json")
    """Path to data blend JSON file"""

    output_dir: Path = field(default_factory=lambda: Path("./output/nano3/stage0_pretrain"))
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
        if self.sample is not None:
            self.output_dir = self.output_dir / f"sample-{self.sample}"


def main(cfg: PreTrainDataPrepConfig) -> DataBlendsArtifact:
    """Run pretrain data preparation."""
    # Add stage-specific tags to wandb run
    add_wandb_tags(["data-prep", "pretrain"])

    # Build artifact name (e.g., "nano3/pretrain/data" or "nano3/pretrain/data?sample=100")
    artifact_name = f"nano3/pretrain/data{'?sample=' + str(cfg.sample) if cfg.sample else ''}"

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
        artifact_name=artifact_name,
    )
    artifact = run_data_prep(data_prep_config)
    print_step_complete(data_prep=artifact)
    return artifact


if __name__ == "__main__":
    cli(main)
