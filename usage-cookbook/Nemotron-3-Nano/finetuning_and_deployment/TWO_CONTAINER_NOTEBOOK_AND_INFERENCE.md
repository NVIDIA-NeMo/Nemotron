# Two-Container Setup: Notebook + Inference (NIM or vLLM)

This guide walks through running **two separate containers**: one for your **notebook and eval**, one for **inference** (NIM or vLLM). The notebook stays in the NeMo AutoModel container (training, merge, eval); inference runs in its own container to avoid dependency clashes. The notebook’s eval step calls the inference server over HTTP.

**Tunneling from your local machine:** You can drive the notebook from your laptop/desktop by tunneling to the remote server. Jupyter runs in the notebook container on the server; you open it in your local browser via SSH port forwarding. The two-container layout does not change—only how you connect to Jupyter (see [Tunneling from your local machine](#tunneling-from-your-local-machine) below). **Shortest path:** use the [Idiot-proof checklist](#idiot-proof-checklist-copy-paste-order). To do everything inside **Cursor** (Remote-SSH + port forward + Jupyter in the IDE), see [Using Cursor IDE](#using-cursor-ide-all-in-one-remote--tunnel). For a **single linear walkthrough** (network → notebook container → NIM container → eval), follow **[Tutorial: Network + NIM + Eval](#tutorial-network--nim--eval-follow-in-order)**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Host machine (your GPU server)                                  │
│                                                                  │
│  ┌──────────────────────────┐    HTTP (e.g. :8000)              │
│  │  Container 1: Notebook    │ ──────────────────────────────►  │
│  │  (NeMo AutoModel)         │                                   │
│  │  - Jupyter                │    ┌──────────────────────────┐ │
│  │  - Training / merge       │    │  Container 2: Inference   │ │
│  │  - Eval (sends requests)  │    │  (NIM or vLLM)            │ │
│  │  - No model loaded for    │    │  - Serves merged model    │ │
│  │    inference              │    │  - OpenAI-compatible API  │ │
│  └──────────────────────────┘    │  - Uses GPU(s)             │ │
│            │                      └──────────────────────────┘ │
│            │  same Docker network         │                    │
│            └──────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

- **Notebook container:** Training, LoRA merge, and **eval**. For eval it only sends HTTP requests to the inference container; it does **not** load the model for inference. So it can stay running while the inference container uses the GPU(s).
- **Inference container:** Loads the **merged** Nemotron-3-Nano model and serves an OpenAI-compatible API. This is where NIM or vLLM runs.

---

## Why two containers?

- **Dependency isolation:** NeMo AutoModel (PyTorch, mamba-ssm, etc.) and NIM/vLLM have different stacks. Keeping them in separate containers avoids the conflicts you’ve seen.
- **GPU usage:** The notebook can stay up for editing and running eval; the inference container owns the GPU(s) for serving. You don’t need to load the model inside the notebook for eval.
- **Tutorial clarity:** “One container for training/notebook, one for serving” is easy to explain and replicate.

---

## NIM vs vLLM: which to use?

| | **NIM (NVIDIA Inference Microservices)** | **vLLM** |
|---|------------------------------------------|----------|
| **What it is** | NVIDIA’s inference stack; Nemotron images on NGC | Open-source inference server; run your own container and load the model |
| **Nemotron-3-Nano** | First-class: use `nvcr.io/nim/nvidia/nemotron-3-nano:1.7.0-variant` with a known model profile | Possible if the model is supported; may need to confirm Mamba/Nemotron support and chat template |
| **Setup** | Pull one image, set env vars, mount merged model, run | Pull vLLM image or build from source; pass model path and any custom options |
| **API** | OpenAI-compatible (`/v1/chat/completions`, etc.) | OpenAI-compatible |
| **PEFT** | For this NIM Nemotron image: **merged model only** (no LoRA adapters in this build) | Can often serve adapters or merged; depends on vLLM version and model |
| **Best for a tutorial** | **Simpler:** one NGC image, documented profile, fewer moving parts | Good if you want to avoid NGC or need a model NIM doesn’t ship |

**Recommendation for this Nemotron-3-Nano + BIRD eval tutorial:** Use **NIM** first. You already have the image name, model profile, and merge-only workflow in SETUP_GOTCHAS. If you hit limits (e.g. no NGC, or need LoRA serving), try vLLM and confirm Nemotron-3-Nano support for your use case.

---

## Tutorial: Network + NIM + Eval (follow in order)

This is the linear walkthrough: create the network, start the notebook container, start NIM on the same network, then run eval in the notebook. Do the steps in order.

### Prerequisites

- The **merged model** exists (you ran the merge step in the notebook and have `merged_model/` in `finetuning_and_deployment/`).
- **NGC_API_KEY** is set on the host (e.g. `export NGC_API_KEY=your-key`).
- You have **two terminals** available: one for the notebook container (stays attached), one for the host to start NIM.

### Step A: Create the network and start the notebook container

**Terminal 1** (on the GPU server):

```bash
# Set your repo path (where 30b-bird is on the server)
export REPO_PATH=/path/to/30b-bird

# Create the shared network (safe to run again if it already exists)
docker network create evalnet 2>/dev/null || true

# Start the notebook container; use --name so we can reach it later
docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --network evalnet --name notebook \
  -p 8889:8888 \
  -v "${REPO_PATH}:/workspace" \
  nvcr.io/nvidia/nemo-automodel:25.11.00 bash
```

You are now **inside** the notebook container. Leave this terminal attached.

### Step B: Start Jupyter inside the notebook container

Still in **Terminal 1** (inside the container):

```bash
cd /workspace/usage-cookbook/Nemotron-3-Nano/finetuning_and_deployment
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```

Copy the URL Jupyter prints (e.g. `http://0.0.0.0:8888/?token=...`). You will use it from your browser or Cursor after tunneling (Step C). Leave Jupyter running.

### Step C (optional): Tunnel from your local machine

If you want to open the notebook in your **local** browser or Cursor:

- **Browser:** On your laptop, open a **second terminal** and run:  
  `ssh -L 8889:localhost:8889 your-user@your-gpu-server`  
  Then in the browser go to `http://localhost:8889` and paste the token when prompted.
- **Cursor:** Use Remote-SSH to connect to the server, then in the Ports view forward port `8889`, and add “Existing Jupyter Server” with that URL and the token.

If you are already on the server (e.g. in a desktop session), you can open `http://localhost:8889` directly.

### Step D: Start the NIM container (second terminal)

Open a **new terminal on the GPU server** (Terminal 2). Leave Terminal 1 attached to the notebook container with Jupyter running.

```bash
# Same REPO_PATH as in Step A (or set it again)
export REPO_PATH=/path/to/30b-bird
export MERGED="${REPO_PATH}/usage-cookbook/Nemotron-3-Nano/finetuning_and_deployment/merged_model"

# Ensure NGC_API_KEY is set
echo "NGC_API_KEY is set: $(if [ -n \"$NGC_API_KEY\" ]; then echo yes; else echo no; fi)"

# Start NIM on the same network as the notebook container
docker run -it --rm --gpus all --network evalnet --name nim \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_DISABLE_MODEL_DOWNLOAD=1 \
  -e NIM_RELAX_MEM_CONSTRAINTS=1 \
  -e NIM_MODEL_PROFILE=7cbe1181600064c6e10ebaf843497acae35aacff2ab96fe8247ae541ae0ac28a \
  -v "${MERGED}:/opt/nim/workspace" \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/nemotron-3-nano:1.7.0-variant
```

- **Wait** until the NIM logs show the server is ready (e.g. “Application is ready to receive API requests”). This can take a few minutes while the model loads.
- **Optional check from the host:** In a third terminal, `curl -s http://localhost:8000/v1/models` should return JSON listing the model.

### Step E: Run eval in the notebook

1. Open the deployment notebook (e.g. `finetuning_deployment_guide.ipynb`) in Jupyter (via the tunnel or local browser).
2. Find the cell that sends a request to NIM (it may look like `url = "http://localhost:8007/v1/chat/completions"`).
3. **Use the NIM container by name** so the notebook (running inside the notebook container) can reach NIM over the Docker network. Set the base URL and request URL to use the container name `nim` and port `8000`:
   - Set **base URL** (if the notebook has a variable):  
     `NIM_BASE_URL = "http://nim:8000"`
   - Set the **request URL** to:  
     `url = "http://nim:8000/v1/chat/completions"`  
     (or `url = f"{NIM_BASE_URL}/v1/chat/completions"` if you defined `NIM_BASE_URL`).
4. Run the eval / chat cell. The notebook kernel runs inside the notebook container, so it will resolve `nim` to the NIM container on `evalnet` and send requests to it. You should get a response with the model’s output.

**If you see “Connection refused” or “Name or service not known”:**

- Confirm both containers are on `evalnet`:  
  `docker network inspect evalnet` should show both `notebook` and `nim`.
- Confirm NIM is ready: check the NIM terminal for “Application is ready” and, from the host, `curl -s http://localhost:8000/v1/models`.

That’s the full flow: network → notebook container + Jupyter → (optional tunnel) → NIM container → eval in the notebook using `http://nim:8000`.

---

Two options:

### Option A: Shared Docker network (recommended for tutorials)

Create a network and attach **both** containers to it. The notebook then uses the **inference container’s name** as hostname. No gateway or host IP needed.

```bash
# Create a network (once)
docker network create evalnet

# Run notebook container (NeMo AutoModel) on that network
docker run -it --gpus all --ipc=host \
  --network evalnet --name notebook \
  -p 8889:8888 \
  -v /path/to/30b-bird:/workspace \
  nvcr.io/nvidia/nemo-automodel:25.11.00 bash

# Run inference container (NIM) on the same network
docker run -it --rm --gpus all --network evalnet --name nim \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_DISABLE_MODEL_DOWNLOAD=1 \
  -e NIM_RELAX_MEM_CONSTRAINTS=1 \
  -e NIM_MODEL_PROFILE=7cbe1181600064c6e10ebaf843497acae35aacff2ab96fe8247ae541ae0ac28a \
  -v /path/to/merged_model:/opt/nim/workspace \
  nvcr.io/nim/nvidia/nemotron-3-nano:1.7.0-variant
```

In the notebook’s eval cell, set the base URL to:

- **NIM:** `NIM_BASE_URL = "http://nim:8000"` (container name `nim`, port 8000 inside the container; no `-p` needed for notebook→NIM traffic).

If you still want to call NIM from the **host** (e.g. `curl`), publish the port:

```bash
docker run ... --network evalnet --name nim -p 8000:8000 ...
```

Then from the host: `http://localhost:8000`. From the notebook container: `http://nim:8000`.

### Option B: Host port publish + gateway IP (what you had before)

- Run the inference container with `-p 8000:8000` (no shared network).
- Run the notebook container as usual.
- From inside the notebook container, `localhost` is **not** the host, so use the host’s IP. Your notebook already tries: default gateway, `172.17.0.1`, `host.docker.internal`. That works but is more fragile (gateway can differ per environment).

For a clean, repeatable tutorial, **Option A (shared network + container name)** is simpler.

---

## Tunneling from your local machine

So the **user runs the notebook from their laptop/desktop** while everything (Jupyter, training, eval, inference) actually runs on the remote GPU server. Only the Jupyter UI is shown in the local browser.

### How it works

1. **On the remote server:** The notebook container runs Jupyter and publishes it to the host, e.g. `-p 8889:8888`. Jupyter inside the container should listen on `0.0.0.0` so the host can forward (e.g. `jupyter notebook --ip=0.0.0.0 --allow-root`).
2. **On the user’s local machine:** They open an SSH tunnel that forwards a local port to the remote host’s published port.
3. **In the local browser:** They open `http://localhost:8889` (or whatever local port they used). Traffic goes through the tunnel to the remote server and into the notebook container; the UI appears locally but **all execution happens on the server**.

Eval works without any extra tunnel: when the user runs the eval cell, that code runs in the notebook container on the server, which then calls the inference container (e.g. `http://nim:8000`) over the Docker network. The user does **not** need to tunnel port 8000 unless they want to call the inference API from a script on their local machine.

### Steps for the user (local machine)

1. **SSH into the remote server with port forwarding** (in a terminal on their laptop/desktop):

   ```bash
   ssh -L 8889:localhost:8889 your-user@remote-gpu-server
   ```

   - `8889` (first): port on the **local** machine (browser will use `http://localhost:8889`).
   - `localhost:8889` (second): on the **remote** host, i.e. the host’s published Jupyter port (which maps to the container’s 8888).

   If the remote host publishes Jupyter on a different port (e.g. 8890), use that in the right-hand side: `-L 8889:localhost:8890`.

2. **Start the notebook container on the server** (if not already running) with Jupyter published, e.g.:

   ```bash
   docker run -it --gpus all --ipc=host --network evalnet --name notebook \
     -p 8889:8888 \
     -v /path/to/30b-bird:/workspace \
     nvcr.io/nvidia/nemo-automodel:25.11.00 bash
   ```

   Inside the container, start Jupyter bound to all interfaces:

   ```bash
   jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
   ```

3. **In the local browser:** Open `http://localhost:8889`. Log in with the token Jupyter prints in the container (or the token file path shown there). The user can now run cells, train, merge, and run eval; everything executes on the server.

4. **(Optional)** To call the inference API from the **local** machine (e.g. `curl` or a local script), add a second forward when opening the SSH session:

   ```bash
   ssh -L 8889:localhost:8889 -L 8000:localhost:8000 your-user@remote-gpu-server
   ```

   Then from the local machine, `http://localhost:8000` reaches the inference server (assuming NIM is published with `-p 8000:8000` on the host).

### What to watch out for

| Issue | What to do |
|-------|------------|
| **“Connection refused” in browser** | Ensure the notebook container is running, Jupyter is started inside it with `--ip=0.0.0.0`, and the host publishes the port (`-p 8889:8888`). On the remote host, `curl -s http://localhost:8889` should respond if Jupyter is up. |
| **Token / password** | Jupyter prints the token in the container stdout; or read it from the path Jupyter shows (e.g. `~/.local/share/jupyter/runtime/jupyter_server_config.json` or the URL in the log). Pass it in the browser when prompted. |
| **Eval works without tunneling 8000** | Eval runs in the notebook on the server; the notebook container talks to NIM over the Docker network. The user only needs the Jupyter tunnel to drive the notebook. |
| **Firewall** | The tunnel uses SSH (port 22). No need to open 8889 or 8000 on the server’s firewall for the user; only SSH is required. |

---

## Using Cursor IDE (all-in-one remote + tunnel)

Cursor can keep everything in one window: connect to the GPU server via **Remote-SSH**, run Docker and Jupyter from Cursor’s terminal (on the remote), and use **port forwarding** so the Jupyter server is available as `localhost` for Cursor’s notebook UI. No separate browser or manual SSH tunnel needed if you use this flow.

### What Cursor gives you

- **Remote-SSH:** You “attach” to the remote host; the workspace is the server’s filesystem and the integrated terminal runs on the server. You can start the notebook and NIM containers from that terminal.
- **Port forwarding:** When something on the remote listens on a port (e.g. 8889), Cursor can forward it to your local machine. You then add that forwarded address as an “Existing Jupyter Server” in Cursor and open/edit/run notebooks inside Cursor.
- **Single window:** Edit code, run terminal, and run notebooks in one IDE; no switching to a browser for Jupyter.

### Steps (Cursor-first, more idiot-proof)

1. **Configure SSH (once)**  
   Ensure you can SSH to the GPU server (e.g. `ssh your-user@remote-gpu-server`). In Cursor, use **Remote-SSH** (or the built-in remote flow): connect to that host (Command Palette → “Remote-SSH: Connect to Host…”).

2. **Open the repo on the remote**  
   After connecting, open the folder that contains the cookbook (e.g. `/path/to/30b-bird` or `/workspace` if that’s where the repo is mounted on the server).

3. **Start containers from Cursor’s terminal**  
   The terminal is on the remote. Run the same commands as in [Step-by-step: NIM](#step-by-step-nim-as-the-inference-container): create `evalnet`, start the notebook container with `-p 8889:8888`, then inside the container start Jupyter with `--ip=0.0.0.0 --port=8888 --allow-root`. In a second terminal (or split), start the NIM container on `evalnet`.

4. **Forward port 8889 in Cursor**  
   When Jupyter is running on the remote on port 8889 (host) / 8888 (container), Cursor may auto-detect it. If not: **Ports** view (bottom panel) → “Forward a Port” → enter `8889`. Note the **local** address Cursor shows (e.g. `localhost:8889` or a different local port if Cursor remaps it).

5. **Add Jupyter server in Cursor**  
   Open a notebook (e.g. `finetuning_deployment_guide.ipynb`). When prompted to select a kernel/server, choose **“Existing Jupyter Server”** (or “Jupyter Server”) and enter:
   - URL: `http://localhost:8889` (or the forwarded port Cursor shows in the Ports view).
   - If Jupyter requires a token: `http://localhost:8889/?token=YOUR_TOKEN` (get the token from the container terminal where Jupyter was started).

6. **Run and eval**  
   Run cells as usual. Eval runs on the server; the notebook container talks to NIM over the Docker network. Set `NIM_BASE_URL = "http://nim:8000"` in the eval cell.

### Cursor-specific gotchas

| Issue | What to do |
|-------|------------|
| **Port 8889 not listed** | In the Ports view, click “Forward a Port” and add `8889` manually. Use the **local** URL Cursor shows when adding the Jupyter server. |
| **Cursor forwards to a different local port** | Cursor sometimes maps remote 8889 to something like `localhost:12345`. In the Ports view, find the row for 8889 and use that local URL (e.g. `http://localhost:12345`) when adding the Jupyter server. |
| **“Cannot open resource with notebook editor” / Jupyter over SSH flaky** | Some Cursor versions have had trouble with remote Jupyter. If notebooks don’t open or kernels don’t attach, use the **browser method**: keep the SSH tunnel in a local terminal (`ssh -L 8889:localhost:8889 user@server`) and open `http://localhost:8889` in your browser. |
| **Token not obvious** | In the container terminal, Jupyter prints a URL with `?token=...`. Copy that full URL and use it as the server URL in Cursor, or read the token from the path Jupyter prints (e.g. `~/.local/share/jupyter/runtime/jupyter_server_config.json` or the server root). |

### Optional: `cursor tunnel` instead of SSH

If your setup uses **Cursor’s tunnel** (e.g. `cursor tunnel` on the server) instead of plain SSH, connect to the host through that tunnel. Port forwarding and “Existing Jupyter Server” work the same: forward 8889 (or the port Cursor assigns) and point the notebook UI at the forwarded URL with token if required.

---

## Idiot-proof checklist (copy-paste order)

Use this order so you don’t miss a step:

- [ ] **On the GPU server:** Create network: `docker network create evalnet`
- [ ] **Notebook container:** Run with `--network evalnet --name notebook -p 8889:8888` and volume mount to repo. Enter the container.
- [ ] **Inside notebook container:** Start Jupyter: `jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root`. Copy the `http://...?token=...` URL or the token.
- [ ] **From your local machine (pick one):**  
  - **Cursor:** Remote-SSH to the server → Ports → Forward 8889 → In notebook, add “Existing Jupyter Server” with the forwarded URL + token.  
  - **Browser:** In a local terminal run `ssh -L 8889:localhost:8889 user@server`, then open `http://localhost:8889` in the browser and paste the token.
- [ ] **On the server (second terminal):** Start NIM with `--network evalnet --name nim`, mount merged model, wait until “Application is ready”.
- [ ] **In the notebook:** Set `NIM_BASE_URL = "http://nim:8000"` in the eval cell and run eval.

If something fails: see [What to watch out for](#what-to-watch-out-for) (tunneling) and [Cursor-specific gotchas](#cursor-specific-gotchas).

---

## Step-by-step: NIM as the inference container

### 1. Create network and start the notebook container

```bash
export REPO_PATH=/path/to/30b-bird   # or your repo path
docker network create evalnet 2>/dev/null || true

docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --network evalnet --name notebook \
  -p 8889:8888 \
  -v "${REPO_PATH}:/workspace" \
  nvcr.io/nvidia/nemo-automodel:25.11.00 bash
```

Inside the notebook container: start Jupyter so it listens on all interfaces (needed for tunneling from a local machine):

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```

**To use the notebook from your local machine:** From your laptop/desktop, open an SSH tunnel (e.g. `ssh -L 8889:localhost:8889 your-user@remote-server`) and in your browser go to `http://localhost:8889`. See [Tunneling from your local machine](#tunneling-from-your-local-machine) for details.

### 2. Start NIM (inference) on the same network

From a **second terminal on the host** (so the first terminal stays in the notebook container):

```bash
export REPO_PATH=/path/to/30b-bird
export MERGED="${REPO_PATH}/usage-cookbook/Nemotron-3-Nano/finetuning_and_deployment/merged_model"

docker run -it --rm --gpus all --network evalnet --name nim \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_DISABLE_MODEL_DOWNLOAD=1 \
  -e NIM_RELAX_MEM_CONSTRAINTS=1 \
  -e NIM_MODEL_PROFILE=7cbe1181600064c6e10ebaf843497acae35aacff2ab96fe8247ae541ae0ac28a \
  -v "${MERGED}:/opt/nim/workspace" \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/nemotron-3-nano:1.7.0-variant
```

Wait until NIM logs show the server is ready (e.g. “Application is ready to receive API requests”).

### 3. Point the notebook eval at NIM

In the eval step (Step 8 or wherever you call the API), set:

- **If using shared network:**  
  `NIM_BASE_URL = "http://nim:8000"`  
  (So the notebook talks to the container named `nim` on port 8000.)

- **If using host port only (Option B):**  
  Keep using your existing auto-resolution (gateway / 172.17.0.1 / host.docker.internal) or set `NIM_BASE_URL` to `http://<host-ip>:8000`.

Then run the eval cell; it will send requests to NIM and compute metrics.

### 4. Optional: two NIMs (base vs merged)

To compare base and merged in the same notebook:

- Run first NIM (base) with `--name nim-base -p 8001:8000` and mount the base model.
- Run second NIM (merged) with `--name nim-merged -p 8000:8000` and mount the merged model.
- Both on `evalnet`. In the notebook: base URL `http://nim-base:8000`, merged URL `http://nim-merged:8000` (or use ports 8001/8000 if you prefer).

---

## Step-by-step: vLLM as the inference container (alternative)

If you prefer vLLM:

1. Use an official vLLM image (e.g. from Docker Hub or build from source) that supports your CUDA/driver.
2. Run it on the **same** `evalnet` network with a name, e.g. `--name vllm`.
3. Mount the **merged** model and start the server (see vLLM docs for Nemotron/Mamba if needed).
4. In the notebook, set the eval base URL to `http://vllm:8000` (or whatever port vLLM uses).

Eval code can stay the same as long as the server exposes an OpenAI-compatible `/v1/chat/completions` (or whatever your eval uses).

---

## What to watch out for

| Issue | What to do |
|-------|------------|
| **Notebook can’t reach NIM/vLLM** | Use **Option A**: put both containers on the same Docker network and use `http://<container-name>:8000`. If you use Option B, ensure the notebook’s gateway/host resolution matches your environment (see SETUP_GOTCHAS §16). |
| **NIM “Free memory … less than desired”** | NIM (and vLLM) want most of the GPU. **Don’t** load the model in the notebook (e.g. restart kernel so no big tensors). Stop any other containers/processes using the GPU before starting NIM. See SETUP_GOTCHAS §13. |
| **NIM “get_expert_mapping” / LoRA** | This NIM Nemotron image doesn’t support serving LoRA for this model. Serve the **merged** model only (no `NIM_PEFT_SOURCE`, no LoRA mount). See SETUP_GOTCHAS §14. |
| **Port 8000 in use** | Change the host port, e.g. `-p 8001:8000`, and set `NIM_BASE_URL = "http://nim:8000"` (inside the container it’s still 8000; only the host port changes). |
| **Notebook container exits** | Use `docker start notebook && docker exec -it notebook bash` to reattach. If you didn’t use `--name notebook`, use the container ID from `docker ps -a`. |
| **Eval dataset** | Generate `dataset/eval_mini_dev.jsonl` once in the notebook container (e.g. `python prepare_eval_mini_dev.py`) so eval uses a fixed dev set. See SETUP_GOTCHAS §15. |

---

## Minimal “tutorial” flow (no extra complexity)

1. **One network:** `docker network create evalnet`
2. **Notebook container:** NeMo AutoModel, `--network evalnet --name notebook`, `-p 8889:8888`, mount repo. Start Jupyter with `--ip=0.0.0.0` inside the container.
3. **Tunnel from local (optional):** User runs `ssh -L 8889:localhost:8889 user@server` and opens `http://localhost:8889` in their browser to drive the notebook.
4. **Inference container:** NIM with merged model, `--network evalnet --name nim`, mount `merged_model` at `/opt/nim/workspace`.
5. **Eval:** In the notebook set `NIM_BASE_URL = "http://nim:8000"` and run the eval cell. No gateway or host IP needed; eval runs on the server and talks to NIM over the Docker network.

That keeps the story simple: “Notebook container for training and eval; NIM container for inference; they talk over a Docker network by name.”

For more detail on NIM commands, profiles, and gotchas, see **SETUP_GOTCHAS.md** (§12b, §13, §14, §16).
