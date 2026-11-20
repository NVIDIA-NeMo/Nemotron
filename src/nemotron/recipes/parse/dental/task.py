"""Dental reference hooks for monitoring, candidate mining, rewards and evaluation."""

from pathlib import Path

from nemotron.recipes.parse.training.data import load_pages

from . import curriculum, data, evaluate, metrics


class DentalTask:
    """Example task; replace these hooks and the reward environment for your domain."""

    prompt_file = Path(__file__).with_name("prompt.txt")
    environment_class = "nemotron.recipes.parse.dental.environment.DentalEnvironment"
    reward_metric = "page_exact"

    def monitor(self, model, processor, rows):
        from nemotron.recipes.parse.training.sft import predict

        predictions = predict(model, processor, rows)
        for prediction, row in zip(predictions, rows, strict=True):
            prediction.update(metrics.components(prediction["prediction"], row))
        summary = {key: sum(row[key] for row in predictions) / len(predictions) for key in ("chart_exact", "text_cer")}
        return predictions, summary

    def create_candidates(self, output, *, seed, contexts):
        manifest = data.save_pages(data.make_records(seed=seed, contexts=contexts), output, "rl_candidates")
        return load_pages(manifest)

    def select_candidates(self, candidates, answers, *, pages, seed, excluded):
        return curriculum.select_training_pages(candidates, answers, pages=pages, seed=seed, excluded=excluded)

    def evaluate(self, model, rows, output, *, samples, processor_source):
        return evaluate.evaluate_isolated(model, rows, output, samples=samples, processor_source=processor_source)

    def summarize(self, answers):
        return evaluate.summarize(answers)
