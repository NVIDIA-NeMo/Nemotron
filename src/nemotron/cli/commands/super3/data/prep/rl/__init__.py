# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""RL data prep subgroup for super3 (rlvr/swe1/swe2/rlhf sub-stages)."""

from nemotron.cli.commands.super3.data.prep.rl._typer_group import rl_app

__all__ = ["rl_app"]
