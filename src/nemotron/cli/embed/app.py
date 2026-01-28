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

"""Embed Typer group.

Contains the embed command group with subcommands for embedding model
fine-tuning workflow:
- sdg: Generate synthetic Q&A pairs from documents
- prep: Prepare training data (convert, mine, unroll)
- finetune: Fine-tune the embedding model
- eval: Evaluate models on retrieval metrics
- export: Export model to ONNX/TensorRT for optimized inference
- deploy: Deploy NIM container with custom model
"""

from __future__ import annotations

import typer
from rich.console import Console

from nemotron.cli.embed import sdg as sdg_module
from nemotron.cli.embed import prep as prep_module
from nemotron.cli.embed import deploy as deploy_module
from nemotron.cli.embed import eval as eval_module
from nemotron.cli.embed import export as export_module
from nemotron.cli.embed import finetune as finetune_module
from nemotron.cli.embed import run as run_module
from nemotron.kit.cli.recipe_typer import RecipeTyper

console = Console()

# Create embed app with RecipeTyper
embed_app = RecipeTyper(
    name="embed",
    help="Embedding model fine-tuning recipe",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@embed_app.command(name="info")
def info() -> None:
    """Display embed workspace information."""
    console.print("[bold green]Embed Workspace[/bold green]")
    console.print("  Fine-tune embedding models for domain-adapted retrieval.")
    console.print()
    console.print("[bold]Workflow Stages:[/bold]")
    console.print("  1. [cyan]sdg[/]      - Generate synthetic Q&A pairs from documents")
    console.print("  2. [cyan]prep[/]     - Prepare training data (convert, mine, unroll)")
    console.print("  3. [cyan]finetune[/] - Fine-tune the embedding model")
    console.print("  4. [cyan]eval[/]     - Evaluate base vs fine-tuned models")
    console.print("  5. [cyan]export[/]   - Export model to ONNX/TensorRT")
    console.print("  6. [cyan]deploy[/]   - Deploy NIM with custom model")
    console.print()
    console.print("[bold]Key Components:[/bold]")
    console.print("  - retriever-sdg (synthetic data generation)")
    console.print("  - Automodel (embedding model training)")
    console.print("  - BEIR (evaluation framework)")
    console.print()
    console.print("[bold]Base Model:[/bold]")
    console.print("  - nvidia/llama-nemotron-embed-1b-v2")


# Register sdg command
embed_app.recipe_command(
    config_dir=str(sdg_module.CONFIG_DIR),
    run_fn=sdg_module._sdg_nemo_run,
    config_model=sdg_module.CONFIG_MODEL,
    rich_help_panel="Data",
)(sdg_module.sdg)

# Register prep command
embed_app.recipe_command(
    config_dir=str(prep_module.CONFIG_DIR),
    run_fn=prep_module._prep_nemo_run,
    config_model=prep_module.CONFIG_MODEL,
    rich_help_panel="Data",
)(prep_module.prep)

# Register finetune command
embed_app.recipe_command(
    config_dir=str(finetune_module.CONFIG_DIR),
    artifact_overrides={"data": "Training data artifact (mined + unrolled)"},
    run_fn=finetune_module._finetune_nemo_run,
    config_model=finetune_module.CONFIG_MODEL,
    rich_help_panel="Training",
)(finetune_module.finetune)

# Register eval command
embed_app.recipe_command(
    config_dir=str(eval_module.CONFIG_DIR),
    artifact_overrides={
        "model": "Fine-tuned model checkpoint",
        "data": "Evaluation data artifact (BEIR format)",
    },
    run_fn=eval_module._eval_nemo_run,
    config_model=eval_module.CONFIG_MODEL,
    rich_help_panel="Evaluation",
)(eval_module.eval)

# Register export command
embed_app.recipe_command(
    config_dir=str(export_module.CONFIG_DIR),
    artifact_overrides={
        "model": "Fine-tuned model checkpoint to export",
    },
    run_fn=export_module._export_nemo_run,
    config_model=export_module.CONFIG_MODEL,
    rich_help_panel="Deployment",
)(export_module.export)

# Register run (pipeline) command
embed_app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    rich_help_panel="Pipeline",
)(run_module.run)

# Register deploy command
# Deploy is a simple Docker wrapper - no nemo-run composition needed
embed_app.recipe_command(
    config_dir=str(deploy_module.CONFIG_DIR),
    artifact_overrides={
        "model": "Exported model directory (ONNX/TensorRT)",
    },
    run_fn=None,  # Deploy doesn't support remote execution or composition
    config_model=deploy_module.CONFIG_MODEL,
    rich_help_panel="Deployment",
)(deploy_module.deploy)
