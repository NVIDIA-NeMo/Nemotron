"""Temporary native-vLLM worker for NeMo RL 5a21699af759.

Set policy.generation.worker_extension_cls_fqn to this module's
NativeParseGenerationWorker in an environment using that pinned NeMo RL code.
The pinned dependency includes the upstream packaging fix.

The generation-worker extension is imported by Ray in its own process. Only
this worker's configuration initializer bypasses the older library's vLLM source
patches; installed files and the original worker remain unchanged.
"""

import ast
import hashlib
import inspect
import logging
import os
import textwrap
from importlib.metadata import version
from types import FunctionType

import ray
from nemo_rl.models.generation.vllm.vllm_worker import (
    BaseVllmGenerationWorker,
    VllmGenerationWorkerImpl,
)
from transformers import AutoConfig

_INITIALIZER_SHA256 = "62f1908fafe8612e6413f21568647c4b1ab97bbe3cfdd96bf147e0c1812c37c9"


def _forward_environment(*, extra_env_vars=None, **features):
    if any(features.values()):
        raise ValueError("Native Parse worker does not support model-specific vLLM patch features")
    names = os.environ.get("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", "").split(",")
    names.extend(["RAY_ENABLE_UV_RUN_RUNTIME_ENV", *(extra_env_vars or [])])
    os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] = ",".join(sorted({name.strip() for name in names if name.strip()}))


def _native_initializer(original):
    source = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fingerprint = hashlib.sha256(ast.dump(source.body[0], include_attributes=False).encode()).hexdigest()
    if fingerprint != _INITIALIZER_SHA256:
        raise RuntimeError("Native Parse worker expects NeMo RL 5a21699af759; configuration initializer changed")
    # Give this copy its own global lookup for the patch function. The original
    # method and its module globals remain unchanged, including for other models.
    namespace = dict(original.__globals__, _apply_vllm_patches=_forward_environment)
    return FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)


class NativeParseGenerationWorkerImpl(VllmGenerationWorkerImpl):
    def shutdown(self) -> bool:
        """Restore sleeping allocation handles before native vLLM frees them."""
        try:
            if self.llm is not None and self.cfg.get("colocated", {}).get("enabled", False):
                # vLLM 0.33 frees sleeping CUDA handles twice during teardown.
                # Wake every tag, including KV cache, through the public API.
                self.llm.wake_up()
        finally:
            result = super().shutdown()
        if result is False:
            raise RuntimeError("Native vLLM worker cleanup failed")
        return result

    def _init_config(self, config, bundle_indices, fraction_of_gpus, seed, extra_env_vars):
        expected = {"vllm": "0.33.0", "transformers": "5.16.1"}
        for package, required in expected.items():
            if version(package) != required:
                raise RuntimeError(
                    f"Native Parse worker requires {package}=={required}; revalidate before changing it"
                )
        architectures = AutoConfig.from_pretrained(config["model_name"], trust_remote_code=True).architectures
        if architectures != ["RadioMBartPrefixForCausalLM"]:
            raise ValueError("Native Parse worker supports only RadioMBartPrefixForCausalLM")
        cfg = config["vllm_cfg"]
        if cfg["async_engine"] or any(
            cfg[name] != 1 for name in ("tensor_parallel_size", "pipeline_parallel_size", "expert_parallel_size")
        ):
            raise ValueError("Native Parse worker is validated for synchronous single-GPU generation only")
        if config.get("quant_cfg") is not None:
            raise ValueError("Native Parse worker does not support quantized generation")
        logging.getLogger(__name__).info("Using native vLLM 0.33 Parse worker; installed sources remain unchanged")
        initialize = _native_initializer(BaseVllmGenerationWorker._init_config)
        return initialize(self, config, bundle_indices, fraction_of_gpus, seed, extra_env_vars)


NativeParseGenerationWorker = ray.remote(NativeParseGenerationWorkerImpl)
