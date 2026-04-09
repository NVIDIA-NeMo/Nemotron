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

"""CLI commands - execution layer.

This package contains all CLI command implementations with visible
execution logic. Structure mirrors the CLI hierarchy:

    commands/
    └── nano3/
        ├── _typer_group.py     # Registers commands
        ├── pretrain.py         # Pretrain execution
        ├── sft.py              # SFT execution
        ├── rl.py               # RL execution (Ray)
        └── data/
            └── prep/
                ├── _typer_group.py
                ├── pretrain.py
                ├── sft.py
                └── rl.py

To change execution backend (e.g., swap nemo-run for SkyPilot),
modify the command files in this package.
"""
