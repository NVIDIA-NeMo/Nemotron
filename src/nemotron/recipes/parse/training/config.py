"""Load the SFT and RLVR configurations with local artifact paths."""

import os
from pathlib import Path


def recipe_root():
    """Return the installed Parse recipe directory, independent of the caller's CWD."""
    return Path(__file__).resolve().parents[1]


def worker_env():
    """Keep the installed recipe and native sampler available in Ray workers."""
    package_parent = str(Path(__file__).resolve().parents[4])
    paths = [package_parent, *os.environ.get("PYTHONPATH", "").split(os.pathsep)]
    return {
        "PYTHONPATH": os.pathsep.join(dict.fromkeys(path for path in paths if path)),
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }


def sft_config(path, model, data, output):
    from omegaconf import OmegaConf

    config = OmegaConf.load(path)
    config.model_path = str(Path(model).absolute())
    config.data_dir = str(Path(data).absolute())
    config.output_dir = str(Path(output).absolute())
    resolved = OmegaConf.to_container(config, resolve=True)
    for name in ("model_path", "data_dir", "output_dir"):
        resolved.pop(name)
    return resolved


def rlvr_config(path, model, data, output, *, prompt_file):
    import nemo_rl
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    from omegaconf import OmegaConf

    register_omegaconf_resolvers()
    # Inherit native algorithm defaults from the pinned framework checkout.
    native = Path(nemo_rl.__file__).parent.parent / "examples/configs/vlm_grpo_3B.yaml"
    if not native.is_file():
        raise FileNotFoundError("Install the pinned NeMo RL source checkout, including examples/configs")
    overrides = OmegaConf.load(path)
    config = OmegaConf.merge(load_config(native), overrides)
    # The upstream example defines unrelated tasks; this recipe has two envs.
    config.env = overrides.env
    for name, value in dict(
        model_path=model,
        data_dir=data,
        output_dir=output,
        prompt_file=prompt_file,
    ).items():
        config[name] = str(Path(value).absolute())
    resolved = OmegaConf.to_container(config, resolve=True)
    for name in ("model_path", "data_dir", "output_dir", "prompt_file"):
        resolved.pop(name)
    return resolved
