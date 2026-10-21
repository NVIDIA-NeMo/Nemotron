#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/curate/decontamination"
#
# [tool.runspec.run]
# launch = "python"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "yaml"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 1
# ///
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Remove training documents that near-duplicate a held-out split."""

from nemotron.steps.curate.nemo_curator.scripts import run_decontamination


def main() -> None:
    run_decontamination.main()


if __name__ == "__main__":
    main()
