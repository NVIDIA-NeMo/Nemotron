#!/usr/bin/env python3
"""Dynamo adapter loading and opt-in vLLM 0.28 Nemotron LoRA compatibility.

Run this file with --workspace to load adapters, or --patch to update vLLM's
source inside the serving container. Python also imports it as sitecustomize
in child workers: the metadata backport runs only when explicitly enabled.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path



def request(url: str, *, method: str = "GET", payload: dict | None = None) -> object:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method=method), timeout=120) as response:
        body = response.read()
        return json.loads(body) if body else {}


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--container-workspace",
        type=Path,
        default=Path(os.environ.get("DYNAMO_CONTAINER_WORKSPACE", "/mount")),
        help="Workspace mount as seen by the Dynamo worker container",
    )
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    config_path = (args.config or workspace / "configs" / "finance_three_adapter_dynamo.yaml").resolve()
    config = yaml.safe_load(config_path.read_text())
    system_url = os.environ.get(
        "DYNAMO_SYSTEM_URL", f"http://127.0.0.1:{config['runtime']['system_port']}",
    ).rstrip("/")
    frontend_url = os.environ.get(
        "DYNAMO_FRONTEND_URL", f"http://127.0.0.1:{config['runtime']['frontend_port']}",
    ).rstrip("/")

    deadline = time.monotonic() + args.timeout
    next_notice = 0.0
    while True:
        try:
            request(frontend_url + "/v1/models")
            request(system_url + "/v1/loras")
            break
        except (OSError, urllib.error.HTTPError):
            if time.monotonic() >= deadline:
                raise TimeoutError("Dynamo frontend/worker did not become ready")
            if time.monotonic() >= next_notice:
                print("Waiting for Dynamo frontend and worker readiness...", flush=True)
                next_notice = time.monotonic() + 30
            time.sleep(5)

    loaded = {}
    for capability, route in config["routes"].items():
        print(f"Loading adapter: {capability}", flush=True)
        adapter = (workspace / route["adapter"]).resolve()
        for required in ("adapter_model.safetensors", "adapter_config.json"):
            if not (adapter / required).is_file():
                raise FileNotFoundError(adapter / required)
        adapter_relative = adapter.relative_to(workspace)
        worker_adapter = (args.container_workspace / adapter_relative).resolve()
        payload = {
            "lora_name": route["model"],
            "source": {"uri": worker_adapter.as_uri()},
        }
        try:
            result = request(system_url + "/v1/loras", method="POST", payload=payload)
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            if error.code == 409 and route["model"] in body:
                result = {"already_loaded": True, "detail": body}
            else:
                raise RuntimeError(f"Failed loading {capability}: HTTP {error.code}: {body}") from error
        loaded[capability] = {
            "model": route["model"],
            "adapter": str(adapter),
            "worker_adapter": str(worker_adapter),
            "response": result,
        }

    expected = {route["model"] for route in config["routes"].values()}
    registration_deadline = time.monotonic() + min(args.timeout, 120)
    while True:
        models = request(frontend_url + "/v1/models")
        available = {item["id"] for item in models.get("data", [])}
        missing = expected - available
        if not missing:
            break
        if time.monotonic() >= registration_deadline:
            raise RuntimeError(f"Loaded adapters absent from /v1/models: {sorted(missing)}")
        print(f"Waiting for frontend adapter registration: {sorted(missing)}", flush=True)
        time.sleep(2)
    output = {"loaded": loaded, "available_models": sorted(available)}
    manifest = workspace / "logs" / "dynamo-loaded-adapters.json"
    manifest.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


def patch_source() -> None:
    import py_compile
    import shutil
    from pathlib import Path

    import vllm


    source = (
        Path(vllm.__file__).resolve().parent
        / "model_executor"
        / "models"
        / "nano_nemotron_vl.py"
    )
    marker = "# Backported from upstream vLLM: LoRA covers the language model only"
    text = source.read_text()

    if marker not in text:
        replacements = [
            (
                "from typing import Annotated, Literal, TypeAlias\n",
                "from typing import Annotated, Literal, TypeAlias, cast\n",
            ),
            (
                "from vllm.logger import init_logger\n",
                "from vllm.logger import init_logger\n"
                "from vllm.lora.layers.base import BaseLayerWithLoRA\n",
            ),
            (
                "    MultiModalEmbeddings,\n    SupportsMultiModal,\n",
                "    MultiModalEmbeddings,\n    SupportsLoRA,\n    SupportsMultiModal,\n",
            ),
            (
                "class NemotronH_Nano_VL_V2(\n"
                "    nn.Module, HasInnerState, IsHybrid, SupportsMultiModal, SupportsMultiModalPruning\n"
                "):\n",
                "class NemotronH_Nano_VL_V2(\n"
                "    nn.Module,\n"
                "    HasInnerState,\n"
                "    IsHybrid,\n"
                "    SupportsMultiModal,\n"
                "    SupportsMultiModalPruning,\n"
                "    SupportsLoRA,\n"
                "):\n",
            ),
            (
                "    requires_sequential_video_encoding = True\n"
                "    \"\"\"Temporarily needed for dynamic res video w/ conv3d, doesn't support bs>1 yet\"\"\"\n\n"
                "    hf_to_vllm_mapper = WeightsMapper(\n",
                "    requires_sequential_video_encoding = True\n"
                "    \"\"\"Temporarily needed for dynamic res video w/ conv3d, doesn't support bs>1 yet\"\"\"\n\n"
                f"    {marker}\n"
                "    is_non_gated_moe = NemotronHForCausalLM.is_non_gated_moe\n"
                "    packed_modules_mapping = NemotronHForCausalLM.packed_modules_mapping\n"
                "    embedding_modules = NemotronHForCausalLM.embedding_modules\n"
                "    lora_skip_prefixes = NemotronHForCausalLM.lora_skip_prefixes\n\n"
                "    hf_to_vllm_mapper = WeightsMapper(\n",
            ),
            (
                "        text_embeddings = self.get_language_model().embed_input_ids(repl_token_ids)\n",
                "        # Replacement tokens are synthesized inside the video encoder and have\n"
                "        # no request-token LoRA index mapping; use the base embedding weights.\n"
                "        language_model = cast(NemotronHForCausalLM, self.get_language_model())\n"
                "        embed_tokens = language_model.model.embed_tokens\n"
                "        if isinstance(embed_tokens, BaseLayerWithLoRA):\n"
                "            embed_tokens = embed_tokens.base_layer\n"
                "        text_embeddings = embed_tokens(repl_token_ids)\n",
            ),
        ]
        for old, new in replacements:
            count = text.count(old)
            if count != 1:
                raise RuntimeError(
                    f"Expected one vLLM 0.28 source match, found {count}: {old[:80]!r}"
                )
            text = text.replace(old, new, 1)

        backup = source.with_suffix(source.suffix + ".pre-nemotron-lora-backport")
        if not backup.exists():
            shutil.copy2(source, backup)
        source.write_text(text)

    py_compile.compile(str(source), doraise=True)
    print(f"Nemotron VLM LoRA source backport active: {source}")


if os.environ.get("NEMOTRON_SUPER_LORA_BACKPORT") == "1":
    from vllm.model_executor.models.interfaces import supports_lora
    from vllm.model_executor.models.nano_nemotron_vl import NemotronH_Nano_VL_V2
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    NemotronH_Nano_VL_V2.supports_lora = True
    NemotronH_Nano_VL_V2.is_non_gated_moe = NemotronHForCausalLM.is_non_gated_moe
    NemotronH_Nano_VL_V2.packed_modules_mapping = (
        NemotronHForCausalLM.packed_modules_mapping
    )
    NemotronH_Nano_VL_V2.embedding_modules = NemotronHForCausalLM.embedding_modules
    NemotronH_Nano_VL_V2.lora_skip_prefixes = NemotronHForCausalLM.lora_skip_prefixes

    if not supports_lora(NemotronH_Nano_VL_V2):
        raise RuntimeError("Nemotron Super VLM LoRA backport did not activate")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--patch"]:
        patch_source()
    else:
        main()
