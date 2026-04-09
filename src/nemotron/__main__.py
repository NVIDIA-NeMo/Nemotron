#!/usr/bin/env python3

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

"""Nemotron CLI entry point.

Usage:
    nemotron nano3 pretrain -c test                       # local execution
    nemotron nano3 pretrain --config test --run dlw       # nemo-run attached
    nemotron nano3 pretrain -c test -r dlw train.train_iters=5000
    nemotron nano3 pretrain -c test --dry-run             # preview config
"""

from __future__ import annotations


def main() -> None:
    """Main CLI entry point."""
    from nemotron.cli.bin.nemotron import main as typer_main

    typer_main()


if __name__ == "__main__":
    main()
