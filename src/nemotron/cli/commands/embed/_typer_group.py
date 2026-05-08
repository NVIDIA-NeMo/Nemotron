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

from rich.console import Console

from nemotron.cli.commands.embed.sdg import META as SDG_META, sdg
from nemotron.cli.commands.embed.prep import META as PREP_META, prep
from nemotron.cli.commands.embed.finetune import META as FINETUNE_META, finetune
from nemotron.cli.commands.embed.eval import META as EVAL_META, eval as eval_cmd
from nemotron.cli.commands.embed.export import META as EXPORT_META, export
from nemotron.cli.commands.embed.deploy import META as DEPLOY_META, deploy
from nemotron.cli.commands.embed.run import run as run_cmd
from nemo_runspec.recipe_typer import RecipeTyper

console = Console()

# Create embed app using RecipeTyper
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


# Register stage commands
embed_app.add_recipe_command(sdg, meta=SDG_META, rich_help_panel="Data")
embed_app.add_recipe_command(prep, meta=PREP_META, rich_help_panel="Data")
embed_app.add_recipe_command(finetune, meta=FINETUNE_META, rich_help_panel="Training")
embed_app.add_recipe_command(eval_cmd, meta=EVAL_META, rich_help_panel="Evaluation")
embed_app.add_recipe_command(export, meta=EXPORT_META, rich_help_panel="Deployment")
embed_app.add_recipe_command(deploy, meta=DEPLOY_META, rich_help_panel="Deployment")

# Register run (pipeline) command
embed_app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    rich_help_panel="Pipeline",
)(run_cmd)
