"""Exact-page reward, with independently logged component measurements."""

import json
from pathlib import Path
import ray
import torch
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from .metrics import components, exact_page_reward


@ray.remote(num_cpus=0.25)
class DentalEnvironment(EnvironmentInterface):
    def __init__(self, cfg):
        self.path = Path(cfg["metrics_file"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.phase = cfg["phase"]

    def shutdown(self):
        pass

    def step(self, message_log_batch, metadata, return_extracted_answer=False):
        texts, values = [], []
        for log, row in zip(message_log_batch, metadata, strict=True):
            text = "".join(m["content"] for m in log if m["role"] == "assistant")
            texts.append(text)
            values.append(components(text, row))
        with self.path.open("a") as stream:
            stream.write(json.dumps(dict(phase=self.phase, samples=values)) + "\n")
        rewards = torch.tensor(
            [exact_page_reward(text, row) for text, row in zip(texts, metadata, strict=True)],
            dtype=torch.float32,
        )
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": f"Reward: {r:g}"} for r in rewards],
            metadata=metadata,
            next_stop_strings=[None] * len(rewards),
            rewards=rewards,
            terminateds=torch.ones_like(rewards),
            answers=texts if return_extracted_answer else None,
        )

    def global_post_process_and_metrics(self, batch):
        return batch, {"reward": batch["rewards"].mean().item()}
