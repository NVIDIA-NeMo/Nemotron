import os
import ray
from transformers import AutoTokenizer
from datasets import load_dataset
# Import NeMo-RL modules (assuming Nemo RL is installed and in PYTHONPATH)
from nemo_rl.environments.langgraph_cli_environment import LangGraphCLIEnv
from nemo_rl.algorithms import grpo
from nemo_rl.algorithms.loss_functions import ClippedPGLossFn

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="examples/configs", config_name="grpo_langgraph_cli.yaml")
def main(cfg: DictConfig):
    # Initialize tokenizer from the model name
    tokenizer = AutoTokenizer.from_pretrained(cfg.policy.tokenizer.name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load the training and validation datasets from JSONL files
    data_files = {"train": cfg.dataset.train_file}
    if cfg.dataset.val_file:
        data_files["val"] = cfg.dataset.val_file
    raw_datasets = load_dataset("json", data_files=data_files)
    train_data = raw_datasets["train"]
    val_data = raw_datasets.get("val", None)

    # Preprocess dataset: convert each example to NeMo-RL's message log format
    train_samples = []
    for idx, ex in enumerate(train_data):
        prompt = ex["input"]
        expected = ex["output"]
        # Build a single-turn conversation with user prompt
        message_log = [{"role": "user", "content": prompt}]
        # Store expected JSON output in metadata for the environment to use in reward calculation
        metadata = {"expected_output": expected}
        train_samples.append({
            "message_log": message_log,
            "metadata": metadata,
            "idx": idx,
            "task_name": "LangGraphCLI"
        })
    val_samples = []
    if val_data:
        for idx, ex in enumerate(val_data):
            message_log = [{"role": "user", "content": ex["input"]}]
            metadata = {"expected_output": ex["output"]}
            val_samples.append({
                "message_log": message_log,
                "metadata": metadata,
                "idx": idx,
                "task_name": "LangGraphCLI"
            })

    # Initialize Ray and instantiate the environment as a remote actor
    ray.init(ignore_reinit_error=True)
    env_config = cfg.env.lang_graph_cli.get("cfg", {})  # environment configuration dictionary
    env = LangGraphCLIEnv.remote(env_config)

    # Ensure logging and checkpoint directories exist
    os.makedirs(cfg.logger.log_dir, exist_ok=True)
    os.makedirs(cfg.checkpointing.checkpoint_dir, exist_ok=True)

    # Set up the GRPO training components (policy model, dataloader, loss, etc.)
    # The GRPO setup will create the policy (with model and generation backend), 
    # and prepare the training loop with our dataset and environment.
    policy, _, dataloader, _, loss_fn, _, logger_obj, master_cfg, val_dataloader = \
        grpo.setup(cfg, train_dataset=train_samples, val_dataset=val_samples, env=env, tokenizer=tokenizer)

    # Use the Clipped Policy Gradient loss for GRPO (PPO-style clipped objective)
    if loss_fn is None:
        loss_fn = ClippedPGLossFn()  # GRPO uses PPO-style clipped loss:contentReference[oaicite:7]{index=7}

    # Train the model using GRPO for max_num_steps iterations
    grpo.grpo_train(
        policy=policy,
        policy_generation=policy.generation,        # Generation interface for the policy
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        logger=logger_obj,
        master_config=master_cfg
    )

    # (The training loop internally handles model updates, rollouts generation via vLLM, 
    # reward computation via env, and periodic validation. Checkpoints will be saved to the specified directory.)
    # After training, you may optionally shut down Ray and save the final model.
    ray.shutdown()

if __name__ == "__main__":
    main()
