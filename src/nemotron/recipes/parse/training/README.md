# Reusable Parse training

This package contains the shared Parse model, data, SFT, RLVR, and checkpoint
lineage implementation. [task.py](task.py) defines the configurable task hooks
for monitoring, candidate preparation and selection, rewards, and evaluation.

The [domain adaptation guide](../../../../../docs/nemotron/parse/adaptation.md)
in `docs/` covers data contracts, task selection, and hook inputs and outputs.
Bundled profiles select the dental reference task; changing `data_dir` alone
does not change that task. Supply your own domain logic through `task` and adapt
the reference data preparation and standalone evaluation stages as well.
