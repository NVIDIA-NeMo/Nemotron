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

"""``nemotron data sdg long-document ...`` — long-document SDG recipes.

Wires the upstream FinePDFs → judged QA pipeline (9 stages) into the Nemotron
CLI so each stage can be dispatched via nemo-run (``--run dlw`` / ``--batch dlw``)
or run locally via ``uv``.
"""

from nemotron.cli.commands.data.sdg.long_document._typer_group import long_document_app

__all__ = ["long_document_app"]
