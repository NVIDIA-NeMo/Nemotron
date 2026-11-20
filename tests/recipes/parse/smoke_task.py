"""Test-only dense serialization reward; never a document quality metric."""

from difflib import SequenceMatcher


def serialization_similarity(prediction: str, target: str) -> float:
    """Retain all text/markup except surrounding EOS and padding markers."""

    def strip_edges(value):
        while True:
            before = value
            for token in ("</s>", "<pad>"):
                if value.startswith(token):
                    value = value[len(token) :]
                if value.endswith(token):
                    value = value[: -len(token)]
            if before == value:
                return value

    return SequenceMatcher(None, strip_edges(prediction), strip_edges(target), autojunk=False).ratio()


def make_smoke_task():
    """Reuse dental mining/evaluation, exercising the external reward hook only."""
    from nemotron.recipes.parse.dental.task import DentalTask

    task = DentalTask()
    task.environment_class = "smoke_environment.SmokeEnvironment"
    task.reward_metric = "serialization_similarity"
    return task
