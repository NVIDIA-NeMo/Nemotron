#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/byob"
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
# gpus_per_node = 0
# ///
"""Run the BYOB benchmark step."""

from nemotron.steps.byob.scripts.run import main

if __name__ == "__main__":
    main()
