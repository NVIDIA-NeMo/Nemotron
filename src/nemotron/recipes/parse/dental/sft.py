"""Small observation hooks around Automodel's native VLM training recipe."""

import json
from pathlib import Path
import time
from .data import load_pages
from .metrics import components


def predict(model, processor, rows):
    import torch
    from PIL import Image
    from torch.nn.attention import SDPBackend, sdpa_kernel

    model.eval()
    results = []
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]),
    ):
        for start in range(0, len(rows), 4):
            batch = rows[start : start + 4]
            if len({r["prompt"] for r in batch}) != 1:
                raise ValueError("Monitor batches must share a prompt")
            images = []
            for row in batch:
                with Image.open(row["image"]) as image:
                    images.append(image.convert("RGB"))
            inputs = {k: v.cuda() for k, v in processor(images=images, return_tensors="pt").items()}
            inputs["pixel_values"] = inputs["pixel_values"].bfloat16()
            prompt = processor.tokenizer.encode(batch[0]["prompt"], add_special_tokens=False)
            ids = torch.tensor([prompt] * len(batch), device="cuda")
            outputs = model.generate(
                **inputs,
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=1536,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                forced_eos_token_id=None,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                use_cache=True,
            )
            for row, tokens in zip(batch, outputs[:, len(prompt) :].tolist(), strict=True):
                ended = processor.tokenizer.eos_token_id in tokens
                if ended:
                    tokens = tokens[: tokens.index(processor.tokenizer.eos_token_id) + 1]
                text = processor.tokenizer.decode(tokens, skip_special_tokens=False)
                results.append(
                    dict(id=row["id"], prediction=text, ended=ended, **components(text, row))
                )
    return results


def make_trainer(config, observer):
    import nemo_rl  # noqa: F401 -- pinned Transformers remote-code compatibility fixes
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

    class DentalSFT(FinetuneRecipeForVLM):
        def _make_progress_bar(self, total=None, initial=0):
            return None  # The stage observer reports training progress.

        def _publish(self, sample, phase):
            step = 0 if self.baseline else int(sample.step) + 1
            observer.update(
                step,
                phase,
                {k: float(v) for k, v in sample.metrics.items()},
                seconds=time.monotonic() - self.started,
            )

        def log_train_metrics(self, sample):
            super().log_train_metrics(sample)
            self._publish(sample, "train")

        def log_val_metrics(self, sample):
            super().log_val_metrics(sample)
            self._publish(sample, "validation")

        def _run_validation_epoch(self, loader):
            result = super()._run_validation_epoch(loader)
            rows = load_pages(config["validation_dataset"]["path"])
            predictions = predict(self.model_parts[0], self.processor, rows)
            for key in ("chart_exact", "text_cer"):
                result.metrics[key] = sum(r[key] for r in predictions) / len(predictions)
            step = 0 if self.baseline else int(result.step) + 1
            path = Path(config["checkpoint"]["checkpoint_dir"]).parent / f"predictions-{step}.json"
            path.write_text(json.dumps(predictions, indent=2))
            return result

    import torch

    torch.set_num_threads(4)
    trainer = DentalSFT(ConfigNode(config))
    trainer.baseline = True
    trainer.started = time.monotonic()
    trainer.setup()
    observer.configure(config["step_scheduler"]["max_steps"], trainer.step_scheduler.epoch_len)
    return trainer
