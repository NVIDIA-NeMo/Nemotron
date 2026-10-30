#!/usr/bin/env bash
# Internal subcommands; end users run the root setup.sh.
set -euo pipefail

download_assets() (
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=${NEMO_LORA_WORKSPACE_HOST:-$(cd "${script_dir}/.." && pwd)}
container=${NEMO_TRAINING_CONTAINER:-nemotron-finance-automodel}
image=${NEMO_TRAINING_IMAGE:-nvcr.io/nvidia/pytorch:26.08-py3}
workspace=$(realpath "${workspace}")
# Load only HF_TOKEN as data, never `source .env`. Exported values take precedence.
if [[ "${NEMO_HF_AUTH_LOADED:-}" != 1 ]]; then
    exec python3 "${script_dir}/environment.py" credentials --workspace "${workspace}" -- bash "${script_dir}/bootstrap.sh" download "$@"
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is missing; configure the private .env file" >&2; exit 2; }
# Read the verified BF16 model pin; never substitute another checkpoint.
mapfile -t model_pin < <(python3 - "${workspace}/manifests/environment.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))['model']
assert m['precision'] == 'BF16' and m['quant_method'] is None
print(m['repo_id']); print(m['revision']); print(m['local_dir'])
PY
)
[[ ${#model_pin[@]} == 3 ]] || { echo "Invalid BF16 model manifest" >&2; exit 2; }
model_dir=${model_pin[2]}
[[ "${model_dir}" == artifacts/model-bf16 ]] || { echo "Unexpected model directory; review manifest" >&2; exit 2; }
# Refuse an existing different model before invoking a downloader that overwrites files.
python3 "${script_dir}/model.py" --model-path "${workspace}/${model_dir}" --check-download-destination

mkdir -p "${workspace}/artifacts" "${workspace}/data/sources" \
    "${workspace}/logs" "${workspace}/.container-home"
exec 9>"${workspace}/logs/huggingface-download.lock"
flock -n 9 || { echo "Asset download already active for this workspace" >&2; exit 2; }
status_file=${workspace}/logs/huggingface-download.status
trap 'code=$?; printf "exit_%s\n" "${code}" > "${status_file}"' EXIT
printf 'running\n' > "${status_file}"

if docker container inspect "${container}" >/dev/null 2>&1; then
    mounted=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/mount"}}{{.Source}}{{end}}{{end}}' "${container}")
    [[ "$(realpath "${mounted}")" == "${workspace}" ]] || {
        echo "Container ${container} mounts ${mounted}, not ${workspace}" >&2
        exit 2
    }
    [[ "$(docker inspect -f '{{.State.Running}}' "${container}")" == true ]] || docker start "${container}" >/dev/null
else
    mapfile -t gpu_flags < <(python3 "${script_dir}/environment.py" hardware --docker-gpu-flags)
    (( ${#gpu_flags[@]} >= 2 )) || { echo "GPU injection selection failed" >&2; exit 2; }
    docker run "${gpu_flags[@]}" --network host --ipc=host \
        --user "$(id -u):$(id -g)" --ulimit memlock=-1 --ulimit stack=67108864 \
        --env HOME=/mount/.container-home --volume "${workspace}:/mount" --workdir /mount \
        --env USER="$(id -un)" --env LOGNAME="$(id -un)" \
        --name "${container}" -d "${image}" sleep infinity >/dev/null
fi

hf_command=$(docker exec -e HOME=/mount/.container-home "${container}" bash -lc \
    'if command -v hf >/dev/null; then command -v hf; elif command -v huggingface-cli >/dev/null; then command -v huggingface-cli; fi')
if [[ -z "${hf_command}" ]]; then
    echo "The pinned training image does not provide a Hugging Face CLI" >&2
    exit 2
fi
docker_env=(-e HOME=/mount/.container-home -e HF_TOKEN)
{
    echo "Downloading pinned BF16 checkpoint metadata"
    docker exec "${docker_env[@]}" "${container}" "${hf_command}" download \
        "${model_pin[0]}" config.json --revision "${model_pin[1]}" \
        --local-dir "/mount/${model_dir}"
    python3 "${script_dir}/model.py" --model-path "${workspace}/${model_dir}" --metadata-only
    echo "Downloading the original BF16 weight shards; no conversion"
    docker exec "${docker_env[@]}" "${container}" "${hf_command}" download \
        "${model_pin[0]}" --revision "${model_pin[1]}" --local-dir "/mount/${model_dir}"
    echo "Verifying every weight shard against its Hugging Face LFS SHA-256"
    python3 "${script_dir}/model.py" --model-path "${workspace}/${model_dir}" --verify-hashes \
        | tee "${workspace}/logs/bf16-download-integrity.json"

    echo "Downloading pinned TAT-DQA source"
    docker exec "${docker_env[@]}" "${container}" "${hf_command}" download \
        next-tat/TAT-DQA --repo-type dataset \
        --revision bc46208b22f0bfe47152c98fd0938d238ce868c2 \
        --local-dir /mount/data/sources/tat-dqa

    echo "Downloading pinned BizBench source"
    docker exec "${docker_env[@]}" "${container}" "${hf_command}" download \
        kensho/bizbench --repo-type dataset \
        --revision 0a793f2f886156902c72b4a22cab82bb9dceaecf \
        --local-dir /mount/data/sources/bizbench
} 2>&1 | tee -a "${workspace}/logs/huggingface-download.log"

echo "Hugging Face assets are ready under ${workspace}/artifacts and ${workspace}/data/sources"
)

install_environment() (
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=${NEMO_LORA_WORKSPACE:-$(cd "${script_dir}/.." && pwd)}
automodel_root=${AUTOMODEL_ROOT:-${workspace}/Automodel}
status_file=${workspace}/logs/automodel-environment.status
manifest_file=${workspace}/logs/automodel-environment.json
kernel_prefix=${workspace}/.jupyter

mkdir -p "${workspace}/logs" "${workspace}/uv-cache" "${kernel_prefix}"
trap 'code=$?; if (( code != 0 )); then printf "exit_%s\n" "${code}" > "${status_file}"; fi' EXIT
printf 'running\n' > "${status_file}"

if [[ -n "${AUTOMODEL_COMMIT:-}" || -n "${AUTOMODEL_REPO:-}" ]]; then
    echo "This workflow resolves official AutoModel main; remove legacy AUTOMODEL_COMMIT/REPO overrides" >&2
    exit 2
fi
python3 "${workspace}/scripts/environment.py" source --workspace "${workspace}" --root "${automodel_root}"
automodel_commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "${workspace}/logs/automodel-source.json")
python3 "${workspace}/scripts/environment.py" hardware --source torch --output "${workspace}/logs/hardware-container.json"
detected_arch=$(python3 "${workspace}/scripts/environment.py" hardware --source torch --field torch_cuda_arch_list)
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" && "${TORCH_CUDA_ARCH_LIST}" != "${detected_arch}" ]]; then
    echo "TORCH_CUDA_ARCH_LIST conflicts with the detected GPU profile" >&2
    exit 2
fi
export TORCH_CUDA_ARCH_LIST="${detected_arch}"

environment_healthy=false
if [[ -x "${automodel_root}/.venv/bin/python" ]] && \
   grep -qi '^include-system-site-packages = true' "${automodel_root}/.venv/pyvenv.cfg" 2>/dev/null && \
   "${automodel_root}/.venv/bin/python" -c 'import torch, datasets, transformers, transformer_engine, nemo_automodel, ipykernel, fitz, jupyterlab' >/dev/null 2>&1; then
    environment_healthy=true
fi

if [[ "${environment_healthy}" != true ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh -o /tmp/nemo-automodel-uv-install.sh
        sh /tmp/nemo-automodel-uv-install.sh
        export PATH="${HOME}/.local/bin:${PATH}"
    fi
    cd "${automodel_root}"
    if [[ ! -f .venv/pyvenv.cfg ]]; then
        uv venv --system-site-packages .venv
    fi
    if ! grep -qi '^include-system-site-packages = true' .venv/pyvenv.cfg; then
        echo "AutoModel venv must expose the NGC container's system packages" >&2
        exit 3
    fi
    bash docker/common/update_pyproject_pytorch.sh "${automodel_root}"
    export UV_CACHE_DIR="${workspace}/uv-cache"
    export MAX_JOBS=${MAX_JOBS:-16}
    uv sync --locked --all-groups --no-group magi --extra all --extra vlm-media
    uv pip install --python .venv/bin/python ipykernel PyMuPDF jupyterlab
fi

# HybridEP is installed by AutoModel's Dockerfile, not its Python dependency
# groups. This workflow uses NGC PyTorch directly and must supply it explicitly.
# Only single-host training is supported here; do not build the RDMA backend.
if ! "${automodel_root}/.venv/bin/python" -c 'from deep_ep import HybridEPBuffer; import hybrid_ep_cpp' >/dev/null 2>&1; then
    export PATH="${HOME}/.local/bin:${PATH}"
    export UV_CACHE_DIR="${workspace}/uv-cache"
    export MAX_JOBS=${MAX_JOBS:-16}
    deepep_commit=$(sed -n 's/^ARG DEEPEP_COMMIT=//p' "${automodel_root}/docker/Dockerfile")
    if [[ ! "${deepep_commit}" =~ ^[0-9a-f]{40}$ ]]; then
        echo "Cannot resolve the upstream Dockerfile's pinned HybridEP revision" >&2
        exit 4
    fi
    deepep_root="${workspace}/dependencies/DeepEP"
    if [[ ! -d "${deepep_root}" ]]; then
        git -c core.hooksPath=/dev/null clone --no-checkout https://github.com/deepseek-ai/DeepEP.git "${deepep_root}"
        git -C "${deepep_root}" -c core.hooksPath=/dev/null checkout "${deepep_commit}"
    fi
    [[ $(git -C "${deepep_root}" rev-parse HEAD) == "${deepep_commit}" ]] || {
        echo "Existing HybridEP checkout does not match AutoModel's pin" >&2; exit 4;
    }
    if git -C "${deepep_root}" apply --check "${automodel_root}/docker/common/deepep.patch"; then
        git -C "${deepep_root}" apply "${automodel_root}/docker/common/deepep.patch"
    else
        git -C "${deepep_root}" apply --reverse --check "${automodel_root}/docker/common/deepep.patch"
    fi
    # This standalone build uses the already-installed NGC Torch. Do not inherit
    # AutoModel's uv match-runtime rules: --no-deps has no Torch resolution.
    HYBRID_EP_MULTINODE=0 CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl \
        uv pip install --no-config --python "${automodel_root}/.venv/bin/python" \
        --no-build-isolation --no-deps -v "${deepep_root}"
fi
"${automodel_root}/.venv/bin/python" -c \
    'from deep_ep import HybridEPBuffer; from nemo_automodel.components.moe.megatron.token_dispatcher import hybrid_ep_dispatch; assert hybrid_ep_dispatch is not None, "AutoModel cannot import HybridEP"'

"${automodel_root}/.venv/bin/python" -m ipykernel install \
    --prefix "${kernel_prefix}" --name nemo-automodel --display-name 'NeMo AutoModel' >/dev/null

AUTOMODEL_ROOT="${automodel_root}" AUTOMODEL_COMMIT="${automodel_commit}" \
KERNEL_PREFIX="${kernel_prefix}" MANIFEST_FILE="${manifest_file}" \
"${automodel_root}/.venv/bin/python" - <<'PY'
import json, os, platform, importlib.metadata
from pathlib import Path
import datasets, torch, transformers

payload = {
    "automodel_root": os.environ["AUTOMODEL_ROOT"],
    "automodel_commit": os.environ["AUTOMODEL_COMMIT"],
    "automodel_ref": "main",
    "hardware": json.loads((Path(os.environ["MANIFEST_FILE"]).parent / "hardware-container.json").read_text()),
    "cpu_arch": platform.machine(),
    "torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "deep_ep": importlib.metadata.version("deep_ep"),
    "hybrid_ep_scope": "single-host",
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "kernel": "nemo-automodel",
    "kernel_prefix": os.environ["KERNEL_PREFIX"],
}
if not payload["cuda_available"] or payload["cuda_device_count"] == 0:
    raise RuntimeError("No CUDA GPU is visible inside the AutoModel environment")
Path(os.environ["MANIFEST_FILE"]).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

printf 'complete\n' > "${status_file}"
)

start_jupyter() (
# Internal Jupyter launcher, invoked by the root setup.sh.
set -euo pipefail
workspace=${NEMO_LORA_WORKSPACE:?Set workspace}
mkdir -p "${workspace}/logs"
exec 9>"${workspace}/logs/jupyter-server.lock"
flock -n 9 || { echo "Jupyter already runs for this workspace; reuse its original private URL."; exit 0; }
cd "${workspace}"
"${AUTOMODEL_ROOT}/.venv/bin/python" -m jupyterlab --no-browser --ip=127.0.0.1 \
    --port="${NEMO_JUPYTER_PORT:-8888}" --ServerApp.port_retries=0 \
    --ServerApp.root_dir="${workspace}" \
    --ServerApp.default_url=/lab/tree/notebooks/nemotron_3_5_super_vl_finance_three_adapters.ipynb
)

case "${1:-}" in
    download) shift; download_assets "$@" ;;
    environment) shift; install_environment "$@" ;;
    jupyter) shift; start_jupyter "$@" ;;
    -h|--help) echo 'Internal commands: download, environment, jupyter. Run bash setup.sh to begin.' ;;
    *) echo 'Use bash setup.sh from the project root.' >&2; exit 2 ;;
esac
