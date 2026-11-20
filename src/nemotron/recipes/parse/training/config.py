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


def rlvr_config(path, model, data, output, *, prompt_file, overrides=None):
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    from omegaconf import OmegaConf

    register_omegaconf_resolvers()
    selected = OmegaConf.load(path)
    if "defaults" in selected:
        config = load_config(path)
    else:
        # Preserve custom override files that omit an explicit parent.
        defaults = load_config(recipe_root() / "stage2_rlvr/config/vlm_grpo_3B.yaml")
        defaults.env = {}
        config = OmegaConf.merge(defaults, selected)
    # Resolve only after user overrides and artifact paths have been supplied.
    config = OmegaConf.merge(config, overrides or {})
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
