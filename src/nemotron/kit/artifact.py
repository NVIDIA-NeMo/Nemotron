# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Artifact module for nemotron.kit.

Re-exports artifact classes from nemotron.kit.artifacts and provides utilities.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from nemotron.kit.artifacts import (
    Artifact,
    DataBlendsArtifact,
    ModelArtifact,
    PretrainBlendsArtifact,
    PretrainDataArtifact,
    SFTDataArtifact,
    SplitJsonlDataArtifact,
    TrackingInfo,
)

__all__ = [
    # Artifact classes
    "Artifact",
    "TrackingInfo",
    "DataBlendsArtifact",
    "PretrainBlendsArtifact",
    "PretrainDataArtifact",
    "SFTDataArtifact",
    "SplitJsonlDataArtifact",
    "ModelArtifact",
    # Utilities
    "ArtifactInput",
    "apply_scale",
    "print_step_complete",
]


# =============================================================================
# Artifact Input for CLI Commands
# =============================================================================


@dataclass
class ArtifactInput:
    """Defines an artifact input slot for a CLI command.

    Used with App.command() to specify named artifact inputs that can be
    provided via --art.<name> CLI arguments or stdin piping.

    Example:
        >>> app.command(
        ...     "pretrain",
        ...     TrainingConfig,
        ...     training_main,
        ...     artifacts={
        ...         "data": ArtifactInput(
        ...             default_name="DataBlendsArtifact-pretrain",
        ...             mappings={"path": "dataset.data_path"},
        ...         ),
        ...     },
        ... )

    Then users can run:
        nemotron nano3 pretrain --art.data v10
        nemotron nano3 pretrain --art.data DataBlendsArtifact-pretrain:latest
        nemotron nano3 pretrain --art.data romeyn/nemotron/DataBlendsArtifact-pretrain:v10
    """

    default_name: str
    """Default W&B artifact name (e.g., 'DataBlendsArtifact-pretrain').

    Used when only a version is provided (e.g., --art.data v10 or --art.data latest).
    """

    mappings: dict[str, str]
    """Mapping from artifact metadata fields to config field paths.

    Keys are field names from the artifact's metadata.json (e.g., 'path').
    Values are dot-separated config field paths (e.g., 'dataset.data_path').

    Example: {"path": "dataset.data_path"} means:
    - Load artifact metadata
    - Get metadata["path"] value
    - Set config.dataset.data_path = that value
    """


# =============================================================================
# Utilities
# =============================================================================


def apply_scale(count: int, scale: str) -> int:
    """Apply scale factor for fast iteration.

    Scale factors:
    - tiny: 1% (minimum 1, maximum 10,000)
    - small: 10%
    - medium: 30%
    - full: 100%

    Example:
        >>> apply_scale(100_000, "tiny")
        1000
        >>> apply_scale(2_000_000, "tiny")  # Capped at 10k
        10000
        >>> apply_scale(100_000, "full")
        100000
    """
    scale_factors = {
        "tiny": 0.01,
        "small": 0.10,
        "medium": 0.30,
        "full": 1.0,
    }

    if scale not in scale_factors:
        raise ValueError(f"Invalid scale: {scale}. Must be one of: {list(scale_factors.keys())}")

    scaled = int(count * scale_factors[scale])
    result = max(1, scaled)  # Ensure at least 1

    # Cap tiny at 10k for reasonable testing time
    if scale == "tiny":
        result = min(result, 10_000)

    return result


def print_step_complete(
    *args: dict[str, Artifact],
    title: str = "Complete",
    **artifacts: Artifact,
) -> None:
    """Print completion message with named artifacts.

    - Rich table to stderr (for humans)
    - JSON to stdout automatically when stdout is piped (for pipeline composition)

    Args:
        *args: Legacy dict syntax for backward compatibility
        title: Title for the completion message
        **artifacts: Named artifacts (e.g., data=artifact, model=checkpoint)

    Example:
        >>> print_step_complete(data=data_artifact)
        >>> print_step_complete(data=data_artifact, model=model_artifact)
    """
    # Support legacy dict syntax for backward compatibility
    if args and isinstance(args[0], dict):
        artifacts = args[0]

    # Auto-enable JSON output when stdout is piped
    output_json = not sys.stdout.isatty()

    # Output JSON to stdout when piped
    if output_json:
        # Output format: {"name": {"path": "...", "type": "..."}, ...}
        output = {name: json.loads(art.to_json()) for name, art in artifacts.items()}
        print(json.dumps(output), flush=True)

    # Output human-readable panel to stderr
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console(file=sys.stderr)

        panels = []
        for name, artifact in artifacts.items():
            # Build content lines - URI first for easy copy/paste
            lines = Text()
            lines.append(f"{artifact.art_path}\n\n", style="bold yellow")
            lines.append("Path:    ", style="dim")
            lines.append(f"{artifact.path.resolve()}\n", style="blue")

            # Add metrics if present
            if artifact.metrics:
                lines.append("Metrics: ", style="dim")
                metrics_parts = [
                    f"{k}={v:,.0f}" if v > 100 else f"{k}={v:.2f}"
                    for k, v in artifact.metrics.items()
                ]
                lines.append(", ".join(metrics_parts), style="green")

            panel = Panel(
                lines,
                title=f"[bold cyan]{name}[/bold cyan] [dim]({artifact.type})[/dim]",
                title_align="left",
                border_style="green",
            )
            panels.append(panel)

        # Print all panels
        console.print()
        for panel in panels:
            console.print(panel)

    except ImportError:
        # Fallback without rich
        sys.stderr.write(f"\nComplete {title}\n")
        sys.stderr.write("=" * 70 + "\n")
        for name, artifact in artifacts.items():
            sys.stderr.write(f"{name} ({artifact.type}):\n")
            sys.stderr.write(f"  {artifact.art_path}\n\n")
            sys.stderr.write(f"  Path: {artifact.path.resolve()}\n")
            if artifact.metrics:
                metrics_parts = [
                    f"{k}={v:,.0f}" if v > 100 else f"{k}={v:.2f}"
                    for k, v in artifact.metrics.items()
                ]
                sys.stderr.write(f"  Metrics: {', '.join(metrics_parts)}\n")
        sys.stderr.write("=" * 70 + "\n")
        sys.stderr.flush()
