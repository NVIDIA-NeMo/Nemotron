# Nemotron 3.5 Super VL RxnCaption Full SFT

Four-node, 16-GPU full-parameter fine-tuning example for parsing chemical
reaction diagrams into structured JSON.

The example intentionally preserves two sibling source directories because the
launcher imports and mounts the shared support code by directory name:

```text
super_vl_rxncaption_sft/   # four-node launcher, recipe, settings, and guide
super_vl_rxncaption/       # shared download, data, export, and evaluation code
```

Start with:

[`super_vl_rxncaption_sft/README.md`](super_vl_rxncaption_sft/README.md)

Run the documented smoke workflow before attempting a full epoch. The CPU test
suite covers configuration, data preparation, orchestration, export logic, and
scoring, but full four-node GPU execution remains unverified.

Use the parent [public data conversion guide](../README.md#public-dataset-and-license)
to create the training and benchmark archives from U-RxnDiagram-15k. The source
and converted data are CC BY-NC 4.0, with attribution required and commercial
use prohibited without separate permission.
