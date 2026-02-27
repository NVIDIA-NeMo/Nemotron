# mamba-ssm / PyTorch version mismatch (Nemotron-3-Nano)

## Repo steps to reproduce the error

From the `finetuning_and_deployment` directory (where `finetune.py` and `bird_peft_nemotron_nano.yaml` live):

1. **Create env and install deps (PyTorch 2.9 + mamba-ssm):**
   ```bash
   uv venv
   uv pip install nemo_automodel datasets transformers mamba-ssm
   ```
   (This pulls whatever `torch` nemo_automodel/mamba-ssm resolve to; on many systems that is PyTorch 2.9+cu128.)

2. **Optional – prepare BIRD data** (only needed if you want a full run; the error happens at model load, not at data load):
   ```bash
   uv run python -c "
   from datasets import load_dataset
   import os, json
   os.makedirs('dataset', exist_ok=True)
   ds = load_dataset('birdsql/bird23-train-filtered', split='train')
   def row(ex):
       q, ev, sql = ex.get('question',''), ex.get('evidence') or '', ex.get('SQL','')
       return {'input': f\"%s\n%s\" % (q, ev).strip(), 'output': sql.strip()}
   ds = ds.map(row, remove_columns=ds.column_names)
   ds.to_json('dataset/training.jsonl', orient='records', lines=True, force_ascii=False)
   "
   ```

3. **Run fine-tuning (2 GPUs); error occurs when the recipe loads the model and imports mamba_ssm:**
   ```bash
   uv run torchrun --nproc-per-node=2 finetune.py
   ```
   The process will fail with:
   `ImportError: ... selective_scan_cuda....so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb`
   when the Nemotron-3-Nano model (and thus `mamba_ssm`) is first loaded.

So the minimal repro is: **steps 1 and 3** (install with current PyTorch + mamba-ssm, then run `finetune.py` with the YAML that uses Nemotron-3-Nano).

---

## Minimal repro (no repo context – dependency-only)

These steps reproduce the error using only the conflicting dependencies. No NeMo, no YAML, no dataset.

1. **Fresh environment** (any directory):
   ```bash
   uv venv
   source .venv/bin/activate   # or on Windows: .venv\Scripts\activate
   ```

2. **Install PyTorch (with CUDA) and mamba-ssm** (versions that often produce the mismatch):
   ```bash
   uv pip install torch mamba-ssm
   ```
   If your environment already has a newer PyTorch (e.g. 2.9+cu128), this will keep it; otherwise you get whatever PyPI resolves. The error appears when the **mamba-ssm** wheel or source build was compiled against a different PyTorch/c10 ABI than the one you run with.

3. **Trigger the failure** (import the mamba-ssm CUDA extension):
   ```bash
  python -c "import mamba_ssm; print('OK')"
   ```
   With the ABI mismatch you get:
   ```text
   ImportError: .../selective_scan_cuda....so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb
   ```

So the minimal repro is: **uv venv → uv pip install torch mamba-ssm → python -c "import mamba_ssm"**. No repo, no NeMo, no training—just those two dependencies and the import that loads the extension.

---

## Exact error from the logs

When loading Nemotron-3-Nano (which uses Mamba layers), you see:

```
ImportError: .../selective_scan_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb
```

### Where it happens

1. **Import chain:** `import mamba_ssm` → `mamba_ssm.ops.selective_scan_interface` → `import selective_scan_cuda` (the compiled CUDA extension).
2. **Failure:** The dynamic linker loads `selective_scan_cuda.so` and then cannot resolve the symbol above when resolving the extension’s dependencies.

So the crash is in the **mamba-ssm** CUDA extension, not in NeMo AutoModel or your YAML.

---

## What the symbol is

The mangled name decodes (Itanium C++ ABI) to:

- **Namespace:** `c10::cuda`
- **Function:** `c10_cuda_check_implementation`
- **Signature (simplified):** `(int, char const*, char const*, unsigned long, bool)`

So it’s a **PyTorch internal** CUDA error-checking helper from the **c10** library (PyTorch’s core C++/CUDA layer). The mamba-ssm extension was compiled against a PyTorch build that exposed this symbol; at runtime you’re using a different PyTorch build where that symbol is missing or has a different name/signature.

---

## What the version mismatch is

- **Your environment:** **PyTorch 2.9.0+cu128** (CUDA 12.8).
- **mamba-ssm (PyPI):** Declares dependency **`torch`** with **no version upper bound**; docs say “PyTorch 1.12+”. So:
  - Any wheel or build of mamba-ssm is tied to the **exact** PyTorch (and c10 ABI) it was built with.
  - There are no official mamba-ssm wheels for PyTorch 2.9; the package is source-only on PyPI and often built against 2.4–2.6 in practice.
- **What “undefined symbol” means:** The `.so` was built against one version of PyTorch’s C++/CUDA ABI (e.g. 2.5 or 2.6). Your process is using **2.9**, where:
  - `c10_cuda_check_implementation` may have been moved, renamed, or inlined, or
  - The c10/CUDA ABI changed so the symbol no longer exists in the form the extension expects.

So the mismatch is **exactly**: the **PyTorch (and c10) version/ABI** that `selective_scan_cuda.so` was built with **vs** the **PyTorch 2.9.0+cu128** you run with. It’s not “just” a PyTorch patch version; it’s that the **binary interface** of the c10 CUDA layer is different.

---

## Summary table

| Item | Value |
|------|--------|
| **Failing library** | `selective_scan_cuda.so` (from mamba-ssm) |
| **Missing symbol** | `c10::cuda::c10_cuda_check_implementation(...)` (PyTorch c10) |
| **Your PyTorch** | 2.9.0+cu128 |
| **mamba-ssm PyPI** | Requires `torch` (no upper bound); 2.3.0 source-only |
| **Likely build target** | Older PyTorch (e.g. 2.4–2.6) in wheel or in your earlier build |
| **Mismatch** | ABI of PyTorch c10/CUDA at **build time** vs **run time** |

---

## Ways to fix it (for Nemotron-3-Nano 30B)

1. **Use the official NeMo/Nemotron container** (e.g. `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`) so PyTorch and Mamba-related bits are built and tested together.
2. **Pin PyTorch to a version mamba-ssm is known to work with** (e.g. 2.5.x or 2.6.x), then reinstall mamba-ssm and nemo_automodel in a fresh venv.
3. **Build mamba-ssm from source in the same env as PyTorch 2.9** and ensure the build uses **only** that venv’s torch (no system or other torch). If the mamba-ssm source still uses the old symbol name, it may need a patch or a newer mamba-ssm release that targets PyTorch 2.9.

This file is for reference when debugging the mamba-ssm / PyTorch mismatch; the cookbook itself does not change the model (Nemotron-3-Nano 30B is required).
