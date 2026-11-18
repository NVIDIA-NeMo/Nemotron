# Parse runtime setup

This package installs the dedicated Parse dependency environment and launches
recipe stages in it. [setup.py](setup.py) prepares the environment and framework
patches; [launch.py](launch.py) runs stages using that runtime and this source
checkout.

The [runtime setup guide](../../../../../docs/nemotron/parse/runtime.md) in `docs/`
covers prerequisites, installation, stage invocation, and environment reuse.
Pass the same `runtime_dir` to each stage.
