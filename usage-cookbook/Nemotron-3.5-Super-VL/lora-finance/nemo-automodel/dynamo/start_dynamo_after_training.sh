#!/usr/bin/env bash
# Host launch by default; --worker runs the serving processes inside the container.
set -euo pipefail
case "${1:-}" in
    --help|-h) echo 'Usage: bash dynamo/start_dynamo_after_training.sh [--worker]'; exit 0 ;;
    ''|--worker) ;;
    *) echo 'Unknown option; use --help.' >&2; exit 2 ;;
esac
(( $# <= 1 )) || { echo 'Unexpected arguments.' >&2; exit 2; }

if [[ "${1:-}" == --worker ]]; then
    workspace=${NEMO_LORA_WORKSPACE:-/mount}
    mkdir -p "${workspace}/logs"
    exec 9>"${workspace}/logs/dynamo-serving.lock"
    if ! flock -n 9; then
        echo "Dynamo already has an active launcher for this workspace; reusing it."
        exit 0
    fi
    python3 "${workspace}/scripts/model.py" --require-runtime serving
    config=${DYNAMO_ROUTE_CONFIG:-${workspace}/configs/finance_three_adapter_dynamo.yaml}
    python=${DYNAMO_CONFIG_PYTHON:-python3}

    eval "$(${python} - "${config}" "${workspace}" <<'PY'
import os, shlex, sys, yaml
from pathlib import Path
c=yaml.safe_load(Path(sys.argv[1]).read_text())
w=Path(sys.argv[2]).resolve()
r=c['runtime']; b=c['base_model']
values={
 'DYN_HTTP_PORT':r['frontend_port'], 'DYN_SYSTEM_PORT':r['system_port'],
 'DYN_TP':os.environ.get('DYNAMO_TENSOR_PARALLEL_SIZE', r['tensor_parallel_size']), 'DYN_MAX_MODEL_LEN':r['max_model_len'],
 'DYN_GPU_MEMORY':r['gpu_memory_utilization'], 'DYN_MAX_LORA_RANK':r['max_lora_rank'],
 'DYN_MAX_LORAS':r['max_loras'],
 'DYN_MODEL_PATH':Path(os.environ.get('NEMO_LORA_MODEL_PATH', w/b['snapshot'])),
 'DYN_DTYPE':b['dtype'],
}
for key,value in values.items(): print(f"{key}={shlex.quote(str(value))}")
PY
    )"

    for capability in visual_ocr reasoning code_generation; do
        adapter=$("${python}" - "${config}" "${workspace}" "${capability}" <<'PY'
import sys, yaml
from pathlib import Path
c=yaml.safe_load(Path(sys.argv[1]).read_text())
print((Path(sys.argv[2]).resolve()/c['routes'][sys.argv[3]]['adapter']).resolve())
PY
    )
        test -f "${adapter}/adapter_model.safetensors"
        test -f "${adapter}/adapter_config.json"
    done
    test -f "${DYN_MODEL_PATH}/config.json"
    "${python}" "${workspace}/scripts/model.py" --model-path "${DYN_MODEL_PATH}"

    export DYN_REQUEST_PLANE=tcp
    export DYN_LORA_ENABLED=true
    export DYN_LORA_PATH=${DYN_LORA_PATH:-${workspace}/dynamo-lora-cache}
    export DYN_SYSTEM_ENABLED=true
    export NEMOTRON_SUPER_LORA_BACKPORT=1
    # ARM64 GCC 13 can crash while consuming Inductor precompiled CPU headers
    # during multimodal processor profiling. Keep compilation enabled, but bypass
    # this cache optimization; an explicit operator override remains supported.
    if [[ $(uname -m) == aarch64 ]]; then
        export TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=${TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS:-0}
    fi
    export PYTHONPATH="${workspace}/dynamo${PYTHONPATH:+:${PYTHONPATH}}"
    mkdir -p "${DYN_LORA_PATH}" "${workspace}/logs"
    python3 "${workspace}/dynamo/sitecustomize.py" --patch

    frontend_log=${workspace}/logs/dynamo-frontend.log
    worker_log=${workspace}/logs/dynamo-vllm-worker.log
    status_file=${workspace}/logs/dynamo-serving.status
    trap 'code=$?; printf "exit_%s\n" "${code}" > "${status_file}"; exit "${code}"' ERR
    printf 'running\n' > "${status_file}"

    python3 -m dynamo.frontend \
        --http-port "${DYN_HTTP_PORT}" --discovery-backend file \
        >"${frontend_log}" 2>&1 &
    frontend_pid=$!

    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:${DYN_HTTP_PORT}/v1/models" >/dev/null; then break; fi
        kill -0 "${frontend_pid}" 2>/dev/null || { tail -n 100 "${frontend_log}"; exit 3; }
        sleep 1
    done

    DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT}" python3 -m dynamo.vllm \
        --model "${DYN_MODEL_PATH}" \
        --dtype "${DYN_DTYPE}" \
        --hf-overrides '{"architectures":["NemotronH_Super_Omni_Reasoning_V3"]}' \
        --discovery-backend file \
        --enable-multimodal \
        --enable-lora --max-lora-rank "${DYN_MAX_LORA_RANK}" --max-loras "${DYN_MAX_LORAS}" \
        --tensor-parallel-size "${DYN_TP}" \
        --trust-remote-code \
        --gpu-memory-utilization "${DYN_GPU_MEMORY}" \
        --max-model-len "${DYN_MAX_MODEL_LEN}" \
        --reasoning-parser nemotron_v3 \
        --kv-events-config '{"enable_kv_cache_events": false}' \
        >"${worker_log}" 2>&1 &
    worker_pid=$!

    wait -n "${frontend_pid}" "${worker_pid}"
    exit 4
else
    script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    workspace=${NEMO_LORA_WORKSPACE_HOST:-$(cd "${script_dir}/.." && pwd)}
    python3 "${workspace}/scripts/model.py" --require-runtime serving
    config=${workspace}/configs/finance_three_adapter_dynamo.yaml
    image=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0
    name=nemotron-finance-dynamo
    model_path_container=${NEMO_LORA_MODEL_PATH_CONTAINER:-/mount/artifacts/model-bf16}
    mkdir -p "${workspace}/logs"
    exec 8>"${workspace}/logs/notebook-training.lock"
    flock -n 8 || { echo "A notebook training cell is still active" >&2; exit 2; }

    for capability in visual_ocr reasoning code_generation; do
        status=${workspace}/logs/finance_${capability}_lora_bf16_20261001_training.status
        adapter=${workspace}/checkpoints/finance_${capability}_lora_bf16_20261001/LOWEST_VAL/model/adapter_model.safetensors
        if [[ ! -f "${status}" || ! -f "${adapter}" ]] || [[ "$(tr -d '\r\n' < "${status}")" != exit_0 ]]; then
            echo "Complete training for ${capability} before starting Dynamo" >&2
            exit 2
        fi
    done
    gpu_count=$(python3 "${workspace}/scripts/environment.py" hardware --source smi | python3 -c 'import json,sys; print(json.load(sys.stdin)["num_gpus"])')
    if pgrep -af '[a]utomodel.*finance_(visual_ocr|reasoning|code_generation)_lora' >/dev/null; then
        echo "Training still owns the GPUs" >&2
        exit 2
    fi

    if docker container inspect "${name}" >/dev/null 2>&1; then
        mounted=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/mount"}}{{.Source}}{{end}}{{end}}' "${name}")
        test "$(realpath "${mounted}")" = "$(realpath "${workspace}")" || {
            echo "Container ${name} mounts ${mounted}, not ${workspace}" >&2
            exit 2
        }
        test "$(docker inspect -f '{{.State.Running}}' "${name}")" = true || docker start "${name}" >/dev/null
    else
        mapfile -t gpu_flags < <(python3 "${workspace}/scripts/environment.py" hardware --docker-gpu-flags)
        (( ${#gpu_flags[@]} >= 2 )) || { echo "GPU injection selection failed" >&2; exit 2; }
        docker run "${gpu_flags[@]}" --network host --ipc=host --user 0:0 \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            --volume "${workspace}:/mount" --workdir /mount \
            --name "${name}" -d "${image}" sleep infinity >/dev/null
    fi

    docker exec -d \
        -e NEMO_LORA_WORKSPACE=/mount \
        -e NEMO_LORA_MODEL_PATH="${model_path_container}" \
        -e DYNAMO_TENSOR_PARALLEL_SIZE="${gpu_count}" \
        -e HF_HUB_OFFLINE=1 \
        -e DYNAMO_ROUTE_CONFIG=/mount/configs/finance_three_adapter_dynamo.yaml \
        "${name}" bash /mount/dynamo/start_dynamo_after_training.sh --worker
    echo "Dynamo launch requested in persistent container ${name}; load adapters after the worker system API is ready."
fi
