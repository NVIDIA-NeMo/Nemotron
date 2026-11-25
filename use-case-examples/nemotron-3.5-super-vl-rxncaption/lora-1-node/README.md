# Nemotron 3.5 Super VL RxnCaption LoRA

Single-node LoRA fine-tuning example for parsing chemical reaction diagrams
into structured JSON with NVIDIA Nemotron 3.5 Super VL.

Keep these files together:

- [`chemical_reaction_parsing_lora.ipynb`](chemical_reaction_parsing_lora.ipynb)
- `rxncaption_support.py`
- `prepare_rxncaption_data.py`
- `reaction_example.png`
- `DATA_LICENSE.md`

Open the notebook for the complete setup, data preparation, training, adapter
merge, and evaluation workflow. The public model, dataset, AutoModel source,
and ARM64 container images are pinned to the versions used by the completed
run. Environment variables in Step 2 can override them deliberately.

The notebook downloads and converts the public U-RxnDiagram-15k release
directly. The dataset and converted artifacts are CC BY-NC 4.0: retain
attribution and do not use them commercially without separate permission. The
included `reaction_example.png` is also an adaptation of that dataset; its
attribution is in [DATA_LICENSE.md](DATA_LICENSE.md).
