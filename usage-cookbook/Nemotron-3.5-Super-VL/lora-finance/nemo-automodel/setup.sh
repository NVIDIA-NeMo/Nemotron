#!/usr/bin/env bash
# Public entry point. Notebook stages remain explicitly user-executed.
set -euo pipefail
download=1
case "${1:-}" in
    '') ;;
    --skip-download) download=0; shift ;;
    -h|--help)
        echo 'Usage: bash setup.sh [--skip-download]'
        echo 'Checks this GPU host, downloads pinned assets, sets up AutoModel, and starts local Jupyter.'
        echo 'Add your Hugging Face token to the private .env when prompted.'
        echo '--skip-download: use already downloaded assets; readiness checks still apply.'
        echo 'No notebook cells, training, or inference are started. Keep the terminal open.'
        exit 0 ;;
    *) echo 'Unknown option. Use bash setup.sh --help.' >&2; exit 2 ;;
esac
(( $# == 0 )) || { echo 'Unexpected arguments. Use bash setup.sh --help.' >&2; exit 2; }
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for command in python3 docker nvidia-smi git flock realpath; do
    command -v "${command}" >/dev/null || { echo "Missing prerequisite: ${command}" >&2; exit 2; }
done
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10 or newer is required")'
workspace=$(realpath "${NEMO_LORA_WORKSPACE_HOST:-${project_dir}}")
[[ "${workspace}" == "${project_dir}" ]] || {
    echo 'Run setup.sh from the complete repository in your chosen workspace.' >&2; exit 2;
}
export NEMO_LORA_WORKSPACE_HOST="${workspace}"
cd "${workspace}"
[[ -w . ]] || { echo 'The repository must be writable.' >&2; exit 2; }
mkdir -p .jupyter logs
exec 7>.jupyter/launcher.lock
flock -n 7 || { echo 'Setup or Jupyter already runs; reconnect to the existing local URL.' >&2; exit 2; }
# Also detect a server started through the lower-level helper before downloading.
flock -n .jupyter/setup.lock true || { echo 'Jupyter/setup already runs; reconnect to it.' >&2; exit 2; }
echo '[1/3] Checking GPU host and project'
docker info >/dev/null
python3 scripts/environment.py hardware --source smi --output logs/hardware-host.json
python3 scripts/environment.py preflight
if (( download )); then
    echo '[2/3] Downloading/verifying pinned model and datasets (cached files are reused)'
    python3 scripts/environment.py credentials --workspace "${workspace}" --init
    if ! python3 scripts/environment.py credentials --workspace "${workspace}" --check; then
        echo "Edit ${workspace}/.env in your editor and set HF_TOKEN to your authorized read token."
        if [[ -t 0 ]]; then
            read -r -p 'After saving .env, press Enter to continue (Ctrl+C to stop): ' reply
            python3 scripts/environment.py credentials --workspace "${workspace}" --check
        else
            echo 'Then rerun bash setup.sh. Never paste the token into chat or notebook cells.' >&2
            exit 2
        fi
    fi
    bash scripts/bootstrap.sh download
else
    echo '[2/3] Using existing downloads; setup will validate local assets'
fi
echo '[3/3] Preparing NeMo AutoModel and starting local Jupyter'
echo 'Open the private local URL printed below; select NeMo AutoModel and run cells individually.'
echo 'Keep this terminal open. No training or inference is started by setup.'

# Environment setup and local Jupyter startup
# Setup only. No data preparation, training, serving, or evaluation is launched.
set -euo pipefail
script_dir="${project_dir}/scripts"
workspace=$(realpath "${NEMO_LORA_WORKSPACE_HOST:-${script_dir}/..}")
container=${NEMO_TRAINING_CONTAINER:-nemotron-finance-automodel}
port=${NEMO_JUPYTER_PORT:-8888}
[[ "${port}" =~ ^[0-9]+$ ]] && (( port > 1024 && port < 65536 )) || { echo "Invalid Jupyter port" >&2; exit 2; }
mkdir -p "${workspace}/.jupyter"
exec 8>"${workspace}/.jupyter/setup.lock"
flock -n 8 || { echo "Setup or Jupyter already runs for this workspace; reconnect to it." >&2; exit 2; }
python3 "${script_dir}/environment.py" hardware --source smi --output "${workspace}/logs/hardware-host.json"
docker info >/dev/null
python3 "${script_dir}/environment.py" preflight --repo "${workspace}" --assets
mkdir -p "${workspace}/logs" "${workspace}/.container-home"
if docker container inspect "${container}" >/dev/null 2>&1; then
    mounted=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/mount"}}{{.Source}}{{end}}{{end}}' "${container}")
    [[ "$(realpath "${mounted}")" == "${workspace}" ]] || { echo "Container mount mismatch" >&2; exit 2; }
    [[ "$(docker inspect -f '{{.State.Running}}' "${container}")" == true ]] || docker start "${container}" >/dev/null
else
    mapfile -t gpu_flags < <(python3 "${script_dir}/environment.py" hardware --docker-gpu-flags)
    (( ${#gpu_flags[@]} >= 2 )) || { echo "GPU injection selection failed" >&2; exit 2; }
    docker run "${gpu_flags[@]}" --network host --ipc=host --user "$(id -u):$(id -g)" \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --env HOME=/mount/.container-home --env USER="$(id -un)" --env LOGNAME="$(id -un)" \
        --volume "${workspace}:/mount" --workdir /mount --name "${container}" \
        -d nvcr.io/nvidia/pytorch:26.08-py3 sleep infinity >/dev/null
fi
common_env=(-e NEMO_LORA_WORKSPACE=/mount -e AUTOMODEL_ROOT=/mount/Automodel
    -e HOME=/mount/.container-home -e USER="$(id -un)" -e LOGNAME="$(id -un)"
    -e JUPYTER_PATH=/mount/.jupyter/share/jupyter -e HF_HUB_OFFLINE=1
    -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}")
# Clear the image's generic multi-GPU build list unless the user supplied an
# explicit host override. Bootstrap selects and validates the detected profile.
for mapping in MODEL_PATH TAT_SOURCE_DIR BIZ_SOURCE_DIR; do
    variable=NEMO_LORA_${mapping}_HOST
    if [[ -n "${!variable:-}" ]]; then
        path=$(realpath "${!variable}")
        [[ "${path}" == "${workspace}/"* ]] || { echo "Assets must be inside workspace" >&2; exit 2; }
        common_env+=(-e "NEMO_LORA_${mapping}=/mount${path#${workspace}}")
        if [[ "${mapping}" == MODEL_PATH ]]; then
            export NEMO_LORA_MODEL_PATH_CONTAINER="/mount${path#${workspace}}"
        fi
    fi
done
docker exec "${common_env[@]}" "${container}" \
    bash /mount/scripts/bootstrap.sh environment
# Open the repository notebook directly; setup never copies or rewrites it.
nohup python3 "${script_dir}/notebook.py" --workspace "${workspace}" \
    >>"${workspace}/logs/notebook-serving-bridge.log" 2>&1 </dev/null 7>&- 8>&- &
echo "Setup complete. Select NeMo AutoModel and run the notebook cell by cell."
echo "Open the local URL in a browser on this GPU machine (port ${port}); no SSH is needed."
echo "Jupyter prints its token-protected URL below; keep that URL private."
exec docker exec "${common_env[@]}" -e NEMO_JUPYTER_PORT="${port}" "${container}" \
    bash /mount/scripts/bootstrap.sh jupyter
