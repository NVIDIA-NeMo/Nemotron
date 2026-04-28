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

"""Typer group for ``nemotron data sdg long-document``."""

from __future__ import annotations

from rich.console import Console

from nemo_runspec.recipe_typer import RecipeTyper

from nemotron.cli.commands.data.sdg.long_document.commands import (
    META_JUDGE,
    META_OCR,
    META_PAGE_CLASSIFICATION,
    META_SEED,
    META_SINGLE_PAGE_QA,
    META_TEXT_QA,
    META_VISUAL_QA,
    META_WHOLE_DOCUMENT_QA,
    META_WINDOWED_QA,
    judge,
    ocr,
    page_classification,
    seed,
    single_page_qa,
    text_qa,
    visual_qa,
    whole_document_qa,
    windowed_qa,
)

console = Console()

long_document_app = RecipeTyper(
    name="long-document",
    help="Long-document SDG recipes (FinePDFs → judged QA pipeline).",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register the 9 stages.  Names must match the PEP 723 [tool.runspec] `name`
# tail so users can map between docs and CLI without ambiguity.
long_document_app.add_recipe_command(seed, meta=META_SEED, rich_help_panel="Seed")
long_document_app.add_recipe_command(ocr, meta=META_OCR, rich_help_panel="Producers")
long_document_app.add_recipe_command(
    text_qa, meta=META_TEXT_QA, name="text-qa", rich_help_panel="Producers"
)
long_document_app.add_recipe_command(
    page_classification,
    meta=META_PAGE_CLASSIFICATION,
    name="page-classification",
    rich_help_panel="Producers",
)
long_document_app.add_recipe_command(
    visual_qa, meta=META_VISUAL_QA, name="visual-qa", rich_help_panel="Producers"
)
long_document_app.add_recipe_command(
    single_page_qa,
    meta=META_SINGLE_PAGE_QA,
    name="single-page-qa",
    rich_help_panel="Producers",
)
long_document_app.add_recipe_command(
    windowed_qa, meta=META_WINDOWED_QA, name="windowed-qa", rich_help_panel="Producers"
)
long_document_app.add_recipe_command(
    whole_document_qa,
    meta=META_WHOLE_DOCUMENT_QA,
    name="whole-document-qa",
    rich_help_panel="Producers",
)
long_document_app.add_recipe_command(judge, meta=META_JUDGE, rich_help_panel="Judge")


@long_document_app.command()
def info() -> None:
    """Show the long-document SDG pipeline overview."""
    console.print("[bold]Long-document SDG pipeline[/]\n")
    console.print(
        "[dim]"
        "01 seed ──┬─ 02 ocr ──── 03 text-qa ─────┐\n"
        "          ├─ 04 classify ── 05 visual-qa ┤\n"
        "          ├─ 06 single-page-qa ──────────┤── 09 judge\n"
        "          ├─ 07 windowed-qa ─────────────┤\n"
        "          └─ 08 whole-doc-qa ────────────┘"
        "[/]\n"
    )
