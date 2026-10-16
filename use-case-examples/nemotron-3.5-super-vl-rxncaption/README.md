# Nemotron 3.5 Super VL RxnCaption Fine-Tuning

Fine-tuning examples for converting numbered chemical-reaction diagrams into
structured JSON with NVIDIA Nemotron 3.5 Super VL.

Choose the workflow that matches your hardware and adaptation needs:

- [`lora-1-node/`](lora-1-node/README.md): single-node LoRA fine-tuning, adapter merge, and evaluation.
- [`sft-4-node/`](sft-4-node/README.md): four-node, 16-GPU full-parameter SFT, export, and evaluation.

Both workflows use the same task definition and benchmark conventions. Start
with each workflow's smoke test before launching a complete training run.

## Public dataset and license

Download the original
[U-RxnDiagram-15k dataset](https://huggingface.co/datasets/opendatalab/U-RxnDiagram-15k).
It contains 15,000 training images and 400 test images with reaction-graph
annotations.

> **Data license:** U-RxnDiagram-15k is licensed
> [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Attribution
> is required, and commercial use is prohibited without separate permission
> from the dataset rights holder. This restriction applies to the converted
> numbered images and training records as adaptations of the source data. The
> Nemotron repository's software license does not replace the dataset license.
> See [DATA_LICENSE.md](DATA_LICENSE.md) for the attribution notice covering
> the included example image and generated data.

Download and extract the two public archives:

```bash
python -m pip install "huggingface_hub>=0.34,<2" "Pillow>=10.1,<13"
hf download opendatalab/U-RxnDiagram-15k --repo-type dataset \
  --revision 274be957df104c4fde2104c2e02578b643aad3d1 \
  --include train_set.zip test_set.zip --local-dir ./u-rxndiagram-15k
unzip ./u-rxndiagram-15k/train_set.zip -d ./u-rxndiagram-15k/extracted
unzip ./u-rxndiagram-15k/test_set.zip -d ./u-rxndiagram-15k/extracted
```

Convert the original images/annotations into the exact BIVP schema used by
both tutorials:

```bash
python lora-1-node/prepare_rxncaption_data.py \
  --source-dir ./u-rxndiagram-15k/extracted \
  --output-dir ./rxncaption-prepared
```

The LoRA notebook runs this pinned download/conversion automatically. The
commands above remain useful for preparing shared archives for the four-node
SFT workflow.

The converter draws and numbers the ground-truth molecular boxes, translates
reaction members to `mol`/`txt`/`idt` targets, and creates:

```text
rxncaption-prepared/
├── finetuning-data/
│   ├── train/{dataset.jsonl,images/}
│   ├── test/{dataset.jsonl,images/}
│   └── DATA_LICENSE.md
├── benchmark/
│   ├── images/gt_bivp/
│   ├── ground_truth/ground_truth.json
│   ├── mapdicts/mapdict_gt.json
│   └── DATA_LICENSE.md
└── manifest.json
```

The reaction targets and 1-based box-index mapping are deterministic. PNG
encoding and font rasterization can vary between Pillow releases, so record the
checksums of the archives you actually train and evaluate with.

Each output includes a license/attribution notice. The LoRA notebook handles
these outputs directly. For the four-node SFT workflow, ZIP
`finetuning-data/` and `benchmark/` separately, publish them at access-controlled
HTTPS locations, and record each archive's `sha256sum`.

```bash
(cd ./rxncaption-prepared && \
  zip -r ../rxncaption-finetuning-data.zip finetuning-data && \
  zip -r ../rxncaption-benchmark.zip benchmark)
sha256sum ./rxncaption-finetuning-data.zip ./rxncaption-benchmark.zip
```

When publishing results or adaptations, cite:

> Jiahe Song et al., “RxnCaption: Reformulating Reaction Diagram Parsing as
> Visual Prompt Guided Captioning,” arXiv:2511.02384.
