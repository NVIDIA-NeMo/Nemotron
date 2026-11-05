"""Train-only reaction-count sampling; original data and validation stay unchanged."""

import json

from rxncaption_dataset_adapter import make_rxncaption_dataset


def make_tuning_dataset(path_or_dataset, split="train", medium_repeats=1, dense_repeats=1, **kwargs):
    assert isinstance(medium_repeats, int) and 1 <= medium_repeats <= 4
    assert isinstance(dense_repeats, int) and 1 <= dense_repeats <= 6
    dataset = make_rxncaption_dataset(path_or_dataset, split=split, **kwargs)
    if dataset.split != "train":
        return dataset
    original = dataset.records
    dataset.records = [
        record
        for record in original
        for _ in range(
            dense_repeats
            if len(json.loads(record["target"])) >= 9
            else medium_repeats
            if len(json.loads(record["target"])) >= 5
            else 1
        )
    ]
    dataset.stats.update(
        unique_train_rows=len(original),
        sampled_train_rows=len(dataset.records),
        medium_repeats=medium_repeats,
        dense_repeats=dense_repeats,
    )
    return dataset
