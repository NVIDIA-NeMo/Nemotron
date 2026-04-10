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

"""Rerank Typer group.

Contains the rerank command group with subcommands for cross-encoder
reranking model fine-tuning workflow:
- sdg: Generate synthetic Q&A pairs from documents
- prep: Prepare training data (convert, mine, unroll)
- finetune: Fine-tune the cross-encoder reranking model
- eval: Evaluate models on reranking metrics
- export: Export model to ONNX/TensorRT for optimized inference
- deploy: Deploy NIM container with custom model
"""

from __future__ import annotations

from rich.console import Console

from nemotron.cli.commands.rerank.sdg import META as SDG_META, sdg
from nemotron.cli.commands.rerank.prep import META as PREP_META, prep
from nemotron.cli.commands.rerank.finetune import META as FINETUNE_META, finetune
from nemotron.cli.commands.rerank.eval import META as EVAL_META, eval as eval_cmd
from nemotron.cli.commands.rerank.export import META as EXPORT_META, export
from nemotron.cli.commands.rerank.deploy import META as DEPLOY_META, deploy
from nemotron.cli.commands.rerank.run import run as run_cmd
from nemo_runspec.recipe_typer import RecipeTyper

console = Console()

# Create rerank app using RecipeTyper
rerank_app = RecipeTyper(
    name="rerank",
    help="Cross-encoder reranking model fine-tuning recipe",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@rerank_app.command(name="info")
def info() -> None:
    """Display rerank workspace information."""
    console.print("[bold green]Rerank Workspace[/bold green]")
    console.print("  Fine-tune cross-encoder reranking models for domain-adapted re-ranking.")
    console.print()
    console.print("[bold]Workflow Stages:[/bold]")
    console.print("  1. [cyan]sdg[/]      - Generate synthetic Q&A pairs from documents")
    console.print("  2. [cyan]prep[/]     - Prepare training data (convert, mine, unroll)")
    console.print("  3. [cyan]finetune[/] - Fine-tune the cross-encoder reranking model")
    console.print("  4. [cyan]eval[/]     - Evaluate base vs fine-tuned rerankers")
    console.print("  5. [cyan]export[/]   - Export model to ONNX/TensorRT")
    console.print("  6. [cyan]deploy[/]   - Deploy NIM with custom model")
    console.print()
    console.print("[bold]Key Components:[/bold]")
    console.print("  - retriever-sdg (synthetic data generation)")
    console.print("  - Automodel (cross-encoder model training)")
    console.print("  - BEIR (reranking evaluation framework)")
    console.print()
    console.print("[bold]Base Model:[/bold]")
    console.print("  - nvidia/llama-nemotron-rerank-1b-v2")


# Register stage commands
rerank_app.add_recipe_command(sdg, meta=SDG_META, rich_help_panel="Data")
rerank_app.add_recipe_command(prep, meta=PREP_META, rich_help_panel="Data")
rerank_app.add_recipe_command(finetune, meta=FINETUNE_META, rich_help_panel="Training")
rerank_app.add_recipe_command(eval_cmd, meta=EVAL_META, rich_help_panel="Evaluation")
rerank_app.add_recipe_command(export, meta=EXPORT_META, rich_help_panel="Deployment")
rerank_app.add_recipe_command(deploy, meta=DEPLOY_META, rich_help_panel="Deployment")

# Register run (pipeline) command
rerank_app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    rich_help_panel="Pipeline",
)(run_cmd)
