"""Test-only dense actor for the manual GPU smoke; not a production reward."""

import json
from pathlib import Path

import ray
import torch
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from smoke_task import serialization_similarity


@ray.remote(num_cpus=0.25)
class SmokeEnvironment(EnvironmentInterface):
    def __init__(self, cfg):
        self.path = Path(cfg["metrics_file"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.phase = cfg["phase"]

    def shutdown(self):
        pass

    def step(self, message_log_batch, metadata, return_extracted_answer=False):
        texts, values = [], []
        for log, row in zip(message_log_batch, metadata, strict=True):
            text = "".join(message["content"] for message in log if message["role"] == "assistant")
            texts.append(text)
            values.append({"serialization_similarity": serialization_similarity(text, row["target"])})
        with self.path.open("a") as stream:
            stream.write(json.dumps(dict(phase=self.phase, samples=values)) + "\n")
        rewards = torch.tensor([value["serialization_similarity"] for value in values], dtype=torch.float32)
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": f"Reward: {reward:g}"} for reward in rewards],
            metadata=metadata,
            next_stop_strings=[None] * len(rewards),
            rewards=rewards,
            terminateds=torch.ones_like(rewards),
            answers=texts if return_extracted_answer else None,
        )

    def global_post_process_and_metrics(self, batch):
        return batch, {"reward": batch["rewards"].mean().item()}
