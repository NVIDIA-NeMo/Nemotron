# Fine-tune Nemotron 3.5 Super VL finance adapters with NeMo AutoModel

Train separate **visual OCR, reasoning, and code-generation** LoRA adapters, then serve and evaluate them with NVIDIA Dynamo using task-tag routing.

## Prerequisites

- **Linux GPU machine**: Git, Python 3.10+, Docker access, and NVIDIA Container Toolkit. `docker info` and `nvidia-smi` must work.
- **Profiles**: 8x H100/4x GB200 GPUs.
- **Writable storage**: approximately 232 GiB for model weights, at least 80 GiB free after downloads, plus Docker image/build-cache space.
- GitHub, Hugging Face, and NVIDIA registry access; an authorized Hugging Face read token. Review model/dataset licenses before use.

## 1. Clone and set up

Clone Nemotron’s main branch. Once done, start with the setup as follows:

```bash
cd Nemotron/usage-cookbook/Nemotron-3.5-Super-VL/lora-finance/nemo-automodel
bash setup.sh
```

When prompted, edit the private `.env` file created in the repository:

```dotenv
HF_TOKEN=your_hugging_face_read_token
```

Save it and press Enter in the setup terminal.

Setup downloads the pinned **BF16** model, TAT-DQA, and BizBench; installs official NeMo AutoModel `main`, recording its exact commit; creates the Jupyter kernel; and starts Jupyter. Cached downloads are reused. Setup does **not** start training.

**Revisions**: [environment](manifests/environment.json), [datasets](manifests/datasets/sources.json). Quantized checkpoints are unsupported; MTP-head training is disabled.

## 2. Run the notebook

Keep the setup terminal open. Open its private Jupyter URL in a browser on the GPU machine and select **NeMo AutoModel**.

Open the repository's single notebook:

```text
notebooks/nemotron_3_5_super_vl_finance_three_adapters.ipynb
```

If Jupyter opens the launcher instead of the notebook, use the file browser on the left to open `notebooks/` and double-click the notebook above, then select the **NeMo AutoModel** kernel.

Run each cell individually, in order:

1. **Settings:** detect GPUs and verify the model/environment.
2. **Data:** prepare task datasets and audit split leakage. Test rows contain no reference reasoning traces or programs.
3. **YAML:** generate task configs from the upstream recipe and the checked-in H100/GB200 profile.
4. **Training:** train OCR, reasoning, and code generation sequentially, each with a one-epoch step budget calculated from its dataset size and global batch size.
5. **Dynamo:** start serving and load all three adapters.
6. **Inference:** evaluate the shuffled test set with task-based routing.
7. **Examples:** display one generated response per adapter.
8. **Plots:** display evaluation metrics and diagnostics.

Wait for each cell to finish; outputs appear beneath it. Stop on errors and inspect `logs/`.

Dynamo is handled by two files in `dynamo/`: `start_dynamo_after_training.sh` starts the serving container and processes, while `sitecustomize.py` loads adapters and applies the required vLLM compatibility patches. The notebook calls these automatically.

## Outputs

- `checkpoints/`: adapters and training checkpoints; `LOWEST_VAL/model/` identifies each best adapter.
- `evaluation/`: predictions and metrics.
- `configs/`: generated task-specific AutoModel YAML files.
- `logs/`: setup, training, serving, and revision records.
- `notebooks/nemotron_3_5_super_vl_finance_three_adapters.ipynb`: the notebook and its saved cell outputs.
