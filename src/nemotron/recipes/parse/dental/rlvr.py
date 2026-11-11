"""Run NeMo RL GRPO with progress reporting and component metrics."""

import json
from pathlib import Path
import sys


def train(config, observer):
    import ray
    from nemo_rl.algorithms.grpo import MasterConfig, setup
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.utils import setup_response_data
    from nemo_rl.data.processors import PROCESSOR_REGISTRY, register_processor
    from nemo_rl.data_plane.factory import make_policy_factory, select_sync_trainer
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
    from nemo_rl.environments.utils import ENV_REGISTRY, register_env, create_env
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.utils.logger import LoggerInterface
    from nemo_rl.utils.venvs import add_hf_modules_cache_to_pythonpath
    from .config import worker_env
    from .rl_data import process_page

    # All native workers use the lab's shared environment.
    ACTOR_ENVIRONMENT_REGISTRY.update(
        {name: PY_EXECUTABLES.SYSTEM for name in ACTOR_ENVIRONMENT_REGISTRY}
    )
    master = MasterConfig(**config)
    if "dental_page" not in PROCESSOR_REGISTRY:
        register_processor("dental_page", process_page)
    actor = "nemotron.recipes.parse.dental.environment.DentalEnvironment"
    ACTOR_ENVIRONMENT_REGISTRY[actor] = PY_EXECUTABLES.SYSTEM
    for name in ("dental-page", "dental-page-validation"):
        if name not in ENV_REGISTRY:
            register_env(name, actor)
    for actor in ("VllmGenerationWorker", "VllmAsyncGenerationWorker"):
        module = "vllm_worker_async" if actor.startswith("VllmAsync") else "vllm_worker"
        ACTOR_ENVIRONMENT_REGISTRY[f"nemo_rl.models.generation.vllm.{module}.{actor}"] = sys.executable

    class LocalLogger(LoggerInterface):
        def __init__(self):
            self.offsets = {}
            self.pending = {}

        def log_hyperparams(self, params):
            pass

        def log_metrics(self, metrics, step, prefix="", step_metric=None, step_finished=False):
            if prefix == "train":
                self.pending.setdefault(int(step), {}).update(metrics)
                return
            if prefix == "timing/train" and step_finished:
                metrics = self.pending.pop(int(step), {})
                prefix = "train"
            elif prefix != "validation":
                return
            values = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            # One actor writes each file; consume only complete appended records.
            for cfg in config["env"].values():
                if cfg["phase"] != prefix:
                    continue
                path = Path(cfg["metrics_file"])
                if not path.exists():
                    continue
                samples = []
                with path.open() as stream:
                    stream.seek(self.offsets.get(path, 0))
                    while True:
                        offset = stream.tell()
                        line = stream.readline()
                        if not line.endswith("\n"):
                            stream.seek(offset)
                            break
                        samples.extend(json.loads(line)["samples"])
                    self.offsets[path] = stream.tell()
                if samples:
                    values.update(
                        {k: sum(s[k] for s in samples) / len(samples) for k in samples[0]}
                    )
            if "page_exact" in values:
                values.setdefault("reward", values["page_exact"])
            observer.update(int(step), prefix, values)

        def log_histogram(self, histogram, step, name):
            pass

        def log_plot(self, figure, step, name):
            pass

    ray.init(
        num_cpus=8,
        num_gpus=1,
        include_dashboard=False,
        runtime_env={
            "py_executable": sys.executable,
            "env_vars": add_hf_modules_cache_to_pythonpath(worker_env()),
        },
    )
    processor = get_tokenizer(master.policy["tokenizer"], get_processor=True)
    tokenizer = processor.tokenizer
    master.policy["generation"] = configure_generation_config(
        master.policy["generation"], tokenizer
    )
    dataset, validation = setup_response_data(processor, master.data, None, is_vlm=True)
    environment = create_env("dental-page", master.env["dental-page"])
    validation_environment = create_env(
        "dental-page-validation", master.env["dental-page-validation"]
    )

    def mapping(data, env):
        entries = data.values() if isinstance(data, dict) else [data]
        return {name: env for entry in entries for name in entry.task_data_processors}

    (
        policy,
        generation,
        _,
        cluster,
        loader,
        val_loader,
        loss,
        logger,
        checkpointer,
        state,
        master,
        teachers,
        aliases,
    ) = setup(
        master,
        tokenizer,
        dataset,
        validation,
        processor=processor,
        policy_factory=make_policy_factory(master.data_plane),
    )
    observer.configure(master.grpo.max_num_steps, len(loader))
    local = LocalLogger()
    logger.loggers.append(local)
    try:
        with checkpointer:
            select_sync_trainer(master, label="Dental RLVR")(
                policy,
                generation,
                loader,
                val_loader,
                tokenizer,
                loss,
                mapping(dataset, environment),
                mapping(validation, validation_environment),
                logger,
                checkpointer,
                state,
                master,
                processor=processor,
            )
    finally:
        logger.finish()
        generation.shutdown()
        policy.shutdown()
        ray.shutdown()
    checkpoints = list(
        Path(config["checkpointing"]["checkpoint_dir"]).glob(
            "step_*/policy/weights/model/consolidated"
        )
    )
    if not checkpoints:
        raise RuntimeError("Native GRPO did not publish a consolidated checkpoint")
    return max(checkpoints, key=lambda p: int(p.parents[3].name.removeprefix("step_")))
