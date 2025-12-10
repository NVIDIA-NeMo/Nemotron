# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""Weights & Biases configuration for experiment tracking and artifact storage.

This module provides a WandbConfig dataclass that can be passed via CLI to enable
W&B artifact tracking. When configured, it automatically initializes the kit
wandb backend.

Example:
    >>> from nemotron.kit.wandb import WandbConfig, init_wandb_if_configured
    >>>
    >>> # In your config dataclass
    >>> @dataclass
    ... class MyConfig:
    ...     wandb: WandbConfig | None = None
    >>>
    >>> # In your main function
    >>> def main(cfg: MyConfig):
    ...     init_wandb_if_configured(cfg.wandb)
    ...     # Now kit.init() has been called with wandb backend
    ...     artifact.save(name="my-artifact")  # Will track in W&B
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WandbConfig:
    """Weights & Biases configuration for experiment tracking and artifact storage.

    When project is set, enables W&B artifact tracking. All fields are optional
    to support both tracked and untracked runs.

    Example CLI usage:
        nemotron nano3 data prep pretrain --wandb.project my-project --wandb.entity my-team
    """

    project: str | None = None
    """W&B project name (required to enable tracking)"""

    entity: str | None = None
    """W&B entity/team name"""

    run_name: str | None = None
    """W&B run name (auto-generated if not specified)"""

    tags: tuple[str, ...] = ()
    """Tags for filtering runs"""

    notes: str | None = None
    """Notes/description for the run"""

    @property
    def enabled(self) -> bool:
        """Returns True if wandb is configured (project is set)."""
        return self.project is not None


def init_wandb_if_configured(
    wandb_config: WandbConfig | None,
    job_type: str = "data-prep",
) -> None:
    """Initialize kit with wandb backend if WandbConfig is provided and enabled.

    This should be called at the start of command handlers to enable artifact tracking.
    If wandb_config is None or project is not set, this is a no-op.

    Args:
        wandb_config: WandbConfig instance or None
        job_type: W&B job type for categorizing runs (default: "data-prep")

    Example:
        >>> def main(cfg: MyConfig):
        ...     init_wandb_if_configured(cfg.wandb, job_type="training")
        ...     # Artifacts will now be tracked in W&B
    """
    if wandb_config is None or not wandb_config.enabled:
        return

    import nemotron.kit as kit

    # Initialize kit with wandb backend (enables artifact tracking)
    kit.init(
        backend="wandb",
        wandb_project=wandb_config.project,
        wandb_entity=wandb_config.entity,
    )

    # Initialize wandb run if not already active
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "wandb is required for W&B tracking. Install with: pip install wandb"
        )

    if wandb.run is None:
        wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=wandb_config.run_name,
            tags=list(wandb_config.tags) if wandb_config.tags else None,
            notes=wandb_config.notes,
            job_type=job_type,
        )
