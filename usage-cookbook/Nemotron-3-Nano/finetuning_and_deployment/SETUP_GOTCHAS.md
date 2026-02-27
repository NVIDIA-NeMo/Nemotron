# Setup gotchas: NeMo AutoModel + Nemotron-3-Nano (BIRD SQL)

Notes from getting the cookbook running in the **nvcr.io/nvidia/nemo-automodel:25.11.00** container on 2 GPUs (Brev / H200).

---

## 1. Mamba / PyTorch ABI (host install)

**Issue:** On the host, `uv pip install nemo_automodel mamba-ssm` can lead to `ImportError: ... undefined symbol: c10_cuda_check_implementation` when loading the model. The `mamba-ssm` CUDA extension was built against a different PyTorch than the one you run with.

**Fix:** Use the **NeMo AutoModel container** so PyTorch and Mamba are built together. Do not rely on a host venv for this cookbook. See `MAMBA_SSM_VERSION_NOTES.md` for details. For the exact container image and dependency versions that worked, see **`WORKING_SETUP_VERSIONS.md`**.

---

## 2. Jupyter kernel vs container Python

**Issue:** The notebook at http://localhost:8889 reported `No module named 'nemo_automodel'` even though you were in the container’s Jupyter. The **kernel** was system Python (`/usr/bin/python`), not the venv that has the editable `nemo_automodel` install.

**Fix:** Register the venv as a Jupyter kernel and select it in the notebook:

```bash
source /opt/venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name automodel-venv --display-name "Python (AutoModel venv)"
```

Restart Jupyter from the same venv, then in the notebook: **Kernel → Change Kernel → "Python (AutoModel venv)"**. The container’s `nemo_automodel` lives at `/opt/Automodel` and is visible to `/opt/venv/bin/python`.

---

## 3. Dataset config: `start_of_turn_token` not supported

**Issue:** Training failed with `ColumnMappedTextInstructionDataset.__init__() got an unexpected keyword argument 'start_of_turn_token'`. The container’s AutoModel version doesn’t accept that parameter.

**Fix:** Remove `start_of_turn_token` from the `dataset` section in `bird_peft_nemotron_nano.yaml` (and from any notebook cell that generates that YAML).

---

## 4. Docker: `REPO_PATH` empty

**Issue:** `docker run ... -v "${REPO_PATH}:/workspace" ...` fails with `invalid spec: :/workspace: empty section between colons` because `REPO_PATH` was not set in that shell.

**Fix:** Set it in the same shell before running, or use the path explicitly:

```bash
export REPO_PATH=/home/shadeform/30b-bird
docker run -it --gpus all -v "${REPO_PATH}:/workspace" ...
# or
docker run -it --gpus all -v /home/shadeform/30b-bird:/workspace ...
```

---

## 5. Docker: port already allocated

**Issue:** `Bind for 0.0.0.0:8889 failed: port is already allocated` when starting a new container with `-p 8889:8888`.

**Fix:** Either stop whatever is using 8889, or **omit the port** if you only need a shell (e.g. for `nvidia-smi` or running training):

```bash
docker run -it --gpus all -v "${REPO_PATH}:/workspace" nvcr.io/nvidia/nemo-automodel:25.11.00 bash
```

Use `-p 8889:8888` only when you need to access Jupyter from the host.

---

## 6. First training step looks “stuck”

**Issue:** After “Max train steps: 125” and the model summary, there are no new log lines for several minutes and it looks frozen.

**Fix:** The **first step** is often slow (JIT/compile, FSDP setup, CUDA warmup). Wait at least **5–10 minutes** before assuming it’s stuck. Check GPU use with `nvidia-smi` in another terminal; if utilization is high, it’s working. Do not Ctrl+C unless you intend to stop the run.

---

## 7. “Buffer is not writable” warning

**Issue:** When loading the model you see: `UserWarning: The given buffer is not writable ... tensor = torch.frombuffer(...)` from `hf_storage.py`.

**Fix:** Safe to ignore. It’s a known PyTorch/safetensors behavior; the message is suppressed for the rest of the run and does not affect training or correctness.

---

## 8. SHMEM / ulimit recommendation

**Issue:** At container start, NVIDIA prints that the default SHMEM limit may be insufficient for PyTorch and recommends extra flags.

**Fix:** Next time you start the container, add:

```bash
docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v "${REPO_PATH}:/workspace" nvcr.io/nvidia/nemo-automodel:25.11.00 bash
```

Useful if you hit CUDA or shared-memory errors during training.

---

## 9. OOM (out of memory)

**Issue:** Training runs out of GPU memory, especially on smaller GPUs or with large batch sizes.

**Fix:** In `bird_peft_nemotron_nano.yaml`, reduce **local_batch_size** (and **global_batch_size** so `global_batch_size = num_gpus * local_batch_size`). For 2 GPUs, e.g. `local_batch_size: 4`, `global_batch_size: 8`. For 2× H200 (141 GiB), `local_batch_size: 12`, `global_batch_size: 24` is a reasonable starting point; decrease if OOM.

---

## 10. Training time and dataset size

**Issue:** Full BIRD train (6601 samples) at global_batch_size 24 is ~275 steps per epoch and may exceed your time budget.

**Fix:** The config uses the **full BIRD train set** (6601 samples); with global_batch_size 24 that’s ~275 steps per epoch. To cap runtime, add `limit_dataset_samples: 2000` (or another number) to the dataset section. To **measure training time**, run: `time torchrun --nproc-per-node=2 finetune.py --config bird_peft_nemotron_nano.yaml`.

---

## 11. Resume fails: "Missing key in checkpoint state_dict: optim.state...step"

**Issue:** Training fails immediately with `RuntimeError: Missing key in checkpoint state_dict: optim.state.backbone.layers.0.mixer.out_proj.lora_A.weight.step` when loading a previous checkpoint. The recipe auto-resumes if it finds a checkpoint in `checkpoint_dir`; the saved optimizer state can be missing keys expected by the loader (format/version mismatch).

**Fix:** Start from scratch by moving or removing the old checkpoint directory so the recipe does not resume:

```bash
mv checkpoints checkpoints_epoch0_step83
# or: rm -rf checkpoints
```

Then run `torchrun ...` again. To **resume** later you’d need a checkpoint format that matches the loader (or a recipe option to skip loading the optimizer).

---

## 12. Inference speed: ~1.5–2 tok/s in notebook (transformers/NeMo)

**Issue:** Step 5 (base-only inference timing) reports **~1.5–2 tokens per second**. Total time per query is 50–75+ seconds. That feels way too slow.

**Why it happens:**

1. **Eager attention** — The model is a Mamba+transformer hybrid. When loaded via NeMo/transformers you see "Falling back to sdpa attention" then "Falling back to eager attention" and "Retrying without SDPA patching." So the **fast attention kernels (SDPA/FlashAttention) are not used**; the stack uses the slow, generic path. That's expected for this architecture in this stack.

2. **Autoregressive decode** — Each new token needs a full forward pass. With 30B parameters and no fused attention, that's many slow steps per token.

3. **Single-query timing** — Step 5 runs one prompt at a time to get per-query stats. Batching would improve throughput (tokens per second across the batch) but not the per-token latency that dominates your experience.

**What you can do:**

| Option | Effect |
|--------|--------|
| **Use SGLang** | SGLang has day-0 support for Nemotron 3 Nano and optimized paths for Mamba+transformer. Expect **much higher** tok/s (order of 10+). See the repo's `sglang_cookbook.ipynb` and run the model there for fast inference; call it from this notebook via API if you want. |
| **Use NVIDIA NIM** | Serve the model with NIM for production-style throughput. |
| **Shorten responses** | Prompt the model to "output only SQL, no explanation" and use a smaller `MAX_NEW_TOKENS` (e.g. 64). You won't get more tok/s, but **wall-clock time per query** drops because the model stops sooner. |
| **Batched generate** | In Step 5, run all 5 prompts in one `generate()` call instead of a loop. Throughput (total tokens / total time) can improve; per-query latency is still limited by the slow decode path. |
| **Confirm device** | Ensure the model is on GPU: e.g. `print(next(base_model.parameters()).device)`. If any part is on CPU, that would make things even slower. |

**Bottom line:** ~1.7 tok/s is **expected** in this notebook with the transformers/NeMo path. To go faster, use an engine built for this model (SGLang or NIM), or reduce how many tokens you need per query (shorter max, prompt for SQL-only).

---

## 13. NIM: "Free memory on device ... is less than desired GPU memory utilization"

**Issue:** NIM (vLLM) fails at startup with e.g. `ValueError: Free memory on device (111.76/139.8 GiB) on startup is less than desired GPU memory utilization (0.9, 125.82 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.` vLLM defaults to using 90% of total GPU memory; that check runs before NIM’s relaxed KV-cache logic.

**Fix:** Free enough GPU memory so that **at least ~126 GiB** (0.9 × 139.8) is free on the target GPU before starting NIM. The **`VLLM_GPU_MEMORY_UTILIZATION`** env var is **not** honored by this NIM image—NIM builds vLLM engine args itself (default 0.9).

1. **Check usage:** `nvidia-smi` — note "Memory-Usage" and which processes use the GPU.
2. **Stop other workloads:** **Stop any other Docker containers** (e.g. the NeMo AutoModel container used for training or Jupyter) so they release GPU memory—you need to stop them before starting NIM. Also stop Jupyter kernels that loaded the model, or run NIM on a different machine/GPU with nothing else running.
3. **Re-check:** Run `nvidia-smi` again; free memory should be ≥ ~126 GiB for a 140 GiB GPU.
4. Keep **`NIM_RELAX_MEM_CONSTRAINTS=1`** for KV-cache; it does not change the 0.9 check but helps once the engine starts.

If you cannot free that much memory on one GPU, use **2 GPUs** with a tp:2 profile and `--gpus all` (no `device=0`) so the model is split across GPUs. See §12b for the exact profile ID and command. See **§12a** for how to find what is using GPU memory and how to free it.

---

## 12a. Finding and freeing GPU memory (why one GPU has only ~110 GiB free)

**Goal:** Get to ≥ ~126 GiB free on each GPU so NIM (tp:1 or tp:2) can start. The missing ~15–30 GiB is usually held by another process.

### 1. See what’s using each GPU

On the **host** (outside any container):

```bash
nvidia-smi
```

Check the table: **Memory-Usage** per GPU (e.g. "28000MiB / 141GiB") and the **Processes** section at the bottom (PID, Process name, GPU memory). If you don’t see processes, use:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

That lists every process using GPU memory and how much.

### 2. Typical culprits and what to do

| Cause | What you see | What to do |
|-------|----------------|------------|
| **NeMo AutoModel container** (training or Jupyter) | Docker process using one or both GPUs | **Stop the container before starting NIM:** `docker ps`, then `docker stop <container_id>`. Or exit the shell inside that container and then `docker stop` it. Other containers must be stopped so NIM has enough GPU memory. |
| **Jupyter kernel** (notebook ran merge or loaded model) | A `python` process (often from Jupyter) holding 20–60+ GiB | In Jupyter: **Kernel → Restart** (or Restart & Clear Output). That frees GPU memory held by that kernel. |
| **Leftover training/inference** | `python` or `torchrun` with large memory | `kill <PID>` (or `kill -9 <PID>` if it doesn’t exit). Get PID from the Processes section of `nvidia-smi`. |
| **Another NIM or vLLM run** | Another `python`/container using the GPU | Stop the other container or process first. |
| **Driver/cache** | No (or few) processes but free memory still low | Reboot the machine if nothing else is running; rare. |

### 3. Checklist before starting NIM

1. **No other Docker containers** using the GPUs:  
   `docker ps` — stop any that are running training, Jupyter, or another NIM.
2. **No Jupyter kernels** that loaded the model or did the merge: restart those kernels or close the notebook.
3. **Re-check:** Run `nvidia-smi` again. For **tp:1** you need one GPU with ≥ ~126 GiB free; for **tp:2** you need **both** GPUs with ≥ ~126 GiB free each.

If one GPU has ~110 GiB free and the other is nearly full, something is using that second GPU (often the AutoModel container or a Jupyter kernel that was using both GPUs during training). Stop that workload, then try NIM again.

---

## 12b. NIM: tp:2 profile for 2 GPUs (Nemotron-3-Nano 1.7.0-variant)

**When to use:** You have 2 GPUs and want to split the model across them. **Important:** vLLM still requires **each** GPU to have ≥ 0.9× its total memory free (e.g. ~126 GiB per 140 GiB GPU). If you see "Free memory on device (110.36/139.8 GiB) ... less than desired (0.9, 125.82 GiB)" with tp:2, one or both GPUs don't have enough free memory—stop other workloads on **both** GPUs and ensure `nvidia-smi` shows ≥ ~126 GiB free on each before starting NIM.

**Profile choice (from `docker run ... list-model-profiles`):**

| Profile name           | Profile ID | Use case                          |
|------------------------|------------|------------------------------------|
| vllm-bf16-tp2-pp1      | `7cbe1181600064c6e10ebaf843497acae35aacff2ab96fe8247ae541ae0ac28a` | **Recommended:** matches merged BF16 model |
| vllm-fp8-tp2-pp1      | `3f553fb62de4dd3bee2f38c25a8fc5af69d0947ab79b78226993424038c815bd` | FP8; smaller/faster if you use an FP8 model |

**Example (2 GPUs, BF16 tp:2):**

```bash
export REPO_PATH=/home/shadeform/30b-bird
export MERGED_HOST="${REPO_PATH}/usage-cookbook/Nemotron-3-Nano/finetuning_and_deployment/merged_model"

docker run -it --rm --gpus all \
  -e "NGC_API_KEY=$NGC_API_KEY" \
  -e NIM_DISABLE_MODEL_DOWNLOAD=1 \
  -e NIM_RELAX_MEM_CONSTRAINTS=1 \
  -e NIM_MODEL_PROFILE=7cbe1181600064c6e10ebaf843497acae35aacff2ab96fe8247ae541ae0ac28a \
  -v "${MERGED_HOST}:/opt/nim/workspace" \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/nemotron-3-nano:1.7.0-variant
```

Use `--gpus all` (not `--gpus '"device=0"'`) so both GPUs are visible. **No trailing space** after the `\` on the `NIM_MODEL_PROFILE` line (a space there can break the env var). NIM may require `--shm-size=16GB` for multi-GPU; add it if you see NCCL/shm errors.

---

## 14. NIM: "To support LoRA for MoE model, 'get_expert_mapping' must be implemented"

**Issue:** NIM starts but then fails with `AttributeError: To support LoRA for MoE model, 'get_expert_mapping' must be implemented`. This happens when NIM has LoRA enabled (e.g. default `peft_source=/loras`) and discovers a LoRA adapter; the Nemotron-3-Nano MoE model does not support LoRA in this NIM/vLLM build.

**Fix:** Deploy the **merged** model only: do not set `NIM_PEFT_SOURCE`, do not mount a LoRA directory into the container, and use the merged model directory (from Step 6) as the only model source at `/opt/nim/workspace`. If you previously had a LoRA mount, remove it and restart with only the merged model mount. For a clean two-container setup (notebook + NIM/vLLM) and networking advice, see **TWO_CONTAINER_NOTEBOOK_AND_INFERENCE.md**.

---

## 15. Step 8 eval: same dev set for both NIMs

**Goal:** Compare base vs merged NIM on a **fixed** eval set so results are reproducible and comparable.

**Fix:** Generate **dataset/eval_mini_dev.jsonl** once (500 examples from [birdsql/bird_mini_dev](https://huggingface.co/datasets/birdsql/bird_mini_dev), same `input`/`output` format as training). Run **inside the AutoModel container** (host Python often has NumPy/datasets conflicts). From the container shell:

```bash
cd /workspace/usage-cookbook/Nemotron-3-Nano/finetuning_and_deployment
source /opt/venv/bin/activate
pip install -q datasets
python prepare_eval_mini_dev.py
```

The file is written under the mounted repo, so the notebook sees it when you run Step 8.

Step 8 in the notebook will then load this file first when present; both NIMs (e.g. base on 8001, merged on 8000) are evaluated on the same rows. If the file is missing, the notebook falls back to loading bird_mini_dev from Hugging Face or the last rows of `dataset/training.jsonl`.

---

## 16. Step 8: Connection refused when calling NIM

**Issue:** Step 8 fails with `ConnectionRefusedError` or `URLError: Connection refused` when calling `NIM_BASE_URL` (default `http://localhost:8000`).

**Causes:**

1. **NIM is not running.** Start the NIM container first (Step 7), with the merged model mounted and `-p 8000:8000`, and wait until it is ready (e.g. `curl http://localhost:8000/v1/models` from the host).

2. **Notebook runs inside a container, NIM in another container.** From inside the AutoModel container, `localhost:8000` is the container’s own loopback, not the host. The NIM is listening on the host’s port 8000, so the notebook cannot reach it via `localhost`. Use the **host’s address** in the notebook instead:
   - **Linux (Docker bridge):** set `NIM_BASE_URL = "http://172.17.0.1:8000"` in the Step 8 cell (or use your host’s IP if 172.17.0.1 doesn’t work).
   - **Docker Desktop / host.docker.internal:** set `NIM_BASE_URL = "http://host.docker.internal:8000"`.

Then re-run the Step 8 eval cell. The notebook now **auto-resolves** the NIM URL by trying the container’s **default gateway** (host) first; see §16.

---

## 17. Progress, what we learned, and where we got stuck

**How far we got**

- **Training and merge:** Fine-tuned Nemotron-3-Nano (BIRD SQL) with NeMo AutoModel, merged LoRA into base, produced `merged_model/`.
- **NIM:** Deployed the merged model in a separate container with the NIM image; NIM starts successfully, serves on `0.0.0.0:8000` inside its container, host publishes `-p 8000:8000`. From the **host**, `curl http://localhost:8000/v1/models` works when NIM is up.
- **Eval dataset:** Built `dataset/eval_mini_dev.jsonl` from `birdsql/bird_mini_dev` (500 examples) via `prepare_eval_mini_dev.py` so both NIMs can be evaluated on the same set.
- **Notebook (Step 8):** Eval cell loads the dev set, resolves NIM URL by trying: localhost, then the container’s **default gateway** (from `ip route show default`), then 172.17.0.1, then host.docker.internal. If no candidate works, it raises a clear error instead of running 50 failing requests.
- **Jupyter kernel:** Documented registering and selecting the **Python (AutoModel venv)** kernel (Step 1 in the notebook); optional volume `automodel-jupyter-kernels:/root/.local/share/jupyter` so the kernel persists across new containers.

**What we learned**

1. **Two containers:** Notebook runs in the AutoModel container; NIM runs in another. From the notebook, `localhost` is the AutoModel container’s loopback, not the host—so the notebook must use the **host’s** IP (e.g. the container’s default gateway) to reach NIM.
2. **Gateway = host:** On Linux, from inside a container, the default gateway (e.g. `ip route show default` → `via 172.18.0.1`) is the host. NIM is bound on the host at port 8000 via `-p 8000:8000`, so the URL to use from the notebook is `http://<gateway>:8000`. The Step 8 cell now discovers this automatically.
3. **172.17.0.1 is not universal:** On some setups (Brev, custom networks), the host is not at 172.17.0.1; using the gateway from `ip route` is more reliable.
4. **Kernel persistence:** Each new AutoModel container starts from a clean image; the registered Jupyter kernel is lost. Either reuse the same container (`docker start` + `docker exec`) or mount a volume for `~/.local/share/jupyter` so the kernel is stored on the host/volume.
5. **Port 8000 in use:** If NIM fails to start with “port already allocated”, another container or process is using 8000; stop it (`docker stop <id>`) or run NIM on another port and set `NIM_PORT` in the notebook.

**Where we got stuck**

- **Container-to-host connectivity:** With NIM confirmed running on the host (logs show “Application is ready to receive API requests”), the Step 8 probe from inside the AutoModel container sometimes still failed on all candidates (localhost, gateway, 172.17.0.1, host.docker.internal). So either: (a) the gateway discovery or a different candidate works after the latest notebook change and we just need to re-run Step 8 with NIM up, or (b) the environment (e.g. Brev / cloud networking or firewall) blocks container→host access to port 8000. If it’s (b), next steps could be: run both containers on the same Docker network and use the NIM container’s name as hostname, or run the notebook on the host (not in a container) so localhost:8000 works.

---

## SGLang in the same container (without changing PyTorch/CUDA)

**Goal:** Run SGLang for faster inference or serving in the same NeMo AutoModel container, without upgrading or replacing the existing PyTorch/CUDA (to avoid breaking mamba-ssm or training).

**Steps (inside the container):**

```bash
source /opt/venv/bin/activate
cd /workspace/usage-cookbook/Nemotron-3-Nano/finetuning_and_deployment
bash install_sglang_keep_torch.sh
```

The script:

1. Detects the current `torch` and `torchaudio` versions.
2. Installs SGLang with a **constraint file** so pip does not upgrade PyTorch or torchaudio.
3. On **CUDA 13** (e.g. nemo-automodel:25.11): installs the `sgl_kernel` wheel from the SGLang cu130 index (PyPI does not ship cu130 wheels).
4. On CUDA 12.x: keeps the `sgl-kernel` that SGLang pulled from PyPI.

**If you see resolver errors:** SGLang’s dependencies pin specific versions (e.g. `torch==2.9.1`). The script constrains `torch` to your current version (e.g. `2.9.0a0`). If pip reports a conflict, try installing with a slightly relaxed constraint (e.g. `torch>=2.9.0,<2.10`) by editing the generated constraint line in the script, or install SGLang in a separate venv and use that only for serving.

**Run the server (example):**

```bash
python -m sglang.launch_server --model-path /path/to/merged_model --host 0.0.0.0 --port 30000
```

**Note:** Nemotron-3-Nano may or may not be officially supported by SGLang’s model list; you may need to try with a compatible chat template or open an issue on the SGLang repo for Mamba/Nemotron support.

---

## Quick reference: recommended docker run (with Jupyter port)

```bash
export REPO_PATH=/home/shadeform/30b-bird
docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8889:8888 \
  -v "${REPO_PATH}:/workspace" \
  nvcr.io/nvidia/nemo-automodel:25.11.00 bash
```

Then inside the container:

```bash
cd /workspace/usage-cookbook/Nemotron-3-Nano/finetuning_and_deployment
source /opt/venv/bin/activate
# Jupyter (if needed): jupyter notebook --ip=0.0.0.0 --allow-root
# Training: torchrun --nproc-per-node=2 finetune.py --config bird_peft_nemotron_nano.yaml
```
