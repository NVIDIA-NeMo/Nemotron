# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCQ BYOB dataset: load HF benchmark rows, sample few-shots, pair with target text (ported from Speaker)."""

from __future__ import annotations

import glob
import logging
import os
import random
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import pandas as pd

from nemotron.customization_recipes.data_prep.byob.config import ByobConfig
from nemotron.customization_recipes.data_prep.byob.hf_utils import load_dataset

logger = logging.getLogger(__name__)


class ByobDataset(ABC):
    """Abstract base for BYOB dataset implementations (Speaker-compatible interface)."""

    @abstractmethod
    def load_source_dataset(self):
        pass

    @abstractmethod
    def parse_dataset(self, dataset):
        pass

    @abstractmethod
    def make_samples(self, dataset_parsed):
        pass

    @abstractmethod
    def sample_and_dump(self):
        pass


class McqByobDataset(ByobDataset):
    """MCQ benchmark few-shot + target text seed construction."""

    def __init__(self, config: ByobConfig):
        self.config = config
        self.dataset_parsed = self.load_source_dataset()

    def load_source_dataset(self):
        subset = self.config.subset
        logger.info(
            "Loading source dataset %s (subset=%s, split=%s)",
            self.config.hf_dataset,
            subset,
            self.config.split,
        )
        dataset = load_dataset(self.config.hf_dataset, subset, split=self.config.split)
        dataset_parsed = self.parse_dataset(dataset)
        if self.config.metadata_file is not None:
            metadata = pd.read_csv(self.config.metadata_file)
            for subject in dataset_parsed:
                merged = pd.merge(
                    dataset_parsed[subject],
                    metadata,
                    on="id",
                    how="left",
                ).dropna()
                if len(merged) != len(dataset_parsed[subject]):
                    msg = f"Metadata and dataset IDs do not match for subject: {subject}"
                    raise ValueError(msg)
                dataset_parsed[subject] = merged
        else:
            for subject in dataset_parsed:
                dataset_parsed[subject]["tags"] = "-"
        return dataset_parsed

    def parse_dataset(self, dataset) -> dict:
        """Normalize HF rows to ``id, question, subject, choices, answer, ...`` (Speaker logic)."""
        df = dataset.to_pandas()
        df["id"] = df.index
        prefix = f"{self.config.hf_dataset}/{self.config.subset}/{self.config.split}"
        df["id"] = df["id"].apply(lambda x: f"{prefix}#{x}")
        hfd = self.config.hf_dataset

        if hfd == "cais/mmlu":
            out: dict = {}
            df = df[["id", "question", "subject", "choices", "answer"]]
            for subj in self.config.source_subjects:
                out[subj] = df[df["subject"] == subj].reset_index(drop=True)
            return out
        if hfd == "TIGER-Lab/MMLU-Pro":
            out = {}
            df = df.rename(columns={"category": "subject", "options": "choices"})
            df["answer"] = df["answer_index"]
            df = df[["id", "question", "subject", "choices", "answer"]]
            for subj in self.config.source_subjects:
                out[subj] = df[df["subject"] == subj].reset_index(drop=True)
            return out
        if hfd == "ai4bharat/MILU":
            out = {}
            df["choices"] = df[["option1", "option2", "option3", "option4"]].apply(
                lambda x: [x["option1"], x["option2"], x["option3"], x["option4"]],
                axis=1,
            )
            df["answer"] = df["target"].map(
                {"option1": 0, "option2": 1, "option3": 2, "option4": 3}
            )
            df = df[["id", "question", "subject", "language", "choices", "answer"]]
            for subj in self.config.source_subjects:
                out[subj] = df[df["subject"] == subj].reset_index(drop=True)
            return out
        if hfd in ("CohereLabs/Global-MMLU", "CohereLabs/Global-MMLU-Lite"):
            out = {}
            df["choices"] = df[["option_a", "option_b", "option_c", "option_d"]].apply(
                lambda x: [x["option_a"], x["option_b"], x["option_c"], x["option_d"]],
                axis=1,
            )
            df["answer"] = df["answer"].map({"A": 0, "B": 1, "C": 2, "D": 3})
            df = df[["id", "question", "subject", "choices", "answer"]]
            for subj in self.config.source_subjects:
                out[subj] = df[df["subject"] == subj].reset_index(drop=True)
            return out
        if hfd == "LinguaLift/IndicMMLU-Pro":
            out = {}
            df["subject"] = df["category"]
            df["answer"] = df["answer_index"]
            df["choices"] = df["options"]
            df = df[["id", "question", "subject", "choices", "answer"]]
            for subj in self.config.source_subjects:
                out[subj] = df[df["subject"] == subj].reset_index(drop=True)
            return out
        if hfd == "openai/MMMLU":
            out = {}
            df = df.rename(
                columns={"Question": "question", "Answer": "answer", "Subject": "subject"}
            )
            df["answer"] = df["answer"].map({"A": 0, "B": 1, "C": 2, "D": 3})
            df["choices"] = df[["A", "B", "C", "D"]].apply(
                lambda x: [x["A"], x["B"], x["C"], x["D"]],
                axis=1,
            )
            df = df[["id", "question", "subject", "choices", "answer"]]
            for subj in self.config.source_subjects:
                out[subj] = df[df["subject"] == subj].reset_index(drop=True)
            return out
        if hfd == "sarvamai/mmlu-indic":
            out = {}
            df["subject"] = "all"
            df = df[["id", "question", "subject", "choices", "answer"]]
            for subj in self.config.source_subjects:
                out[subj] = df[df["subject"] == subj].reset_index(drop=True)
            return out
        if hfd == "Idavidrein/gpqa":
            out = {}
            df = df.rename(columns={"Subdomain": "subject", "Question": "question"})
            df["choices"] = df.apply(
                lambda x: random.sample(
                    [
                        x["Correct Answer"],
                        x["Incorrect Answer 1"],
                        x["Incorrect Answer 2"],
                        x["Incorrect Answer 3"],
                    ],
                    k=4,
                ),
                axis=1,
            )
            df["answer"] = df[["choices", "Correct Answer"]].apply(
                lambda x: x["choices"].index(x["Correct Answer"]),
                axis=1,
            )
            df = df[["id", "question", "subject", "choices", "answer"]]
            for subj in self.config.source_subjects:
                out[subj] = df[df["subject"] == subj].reset_index(drop=True)
            return out
        msg = f"Unsupported dataset: {hfd}"
        raise ValueError(msg)

    @staticmethod
    def extract_text_from_path(path: str):
        if ".parquet:" in path:
            parquet_path, file_name = path.rsplit(":", 1)
            pqf = pd.read_parquet(parquet_path)
            return pqf[pqf["file_name"] == file_name]["text"].values
        with open(path, encoding="utf-8") as f:
            return [f.read()]

    def chunk_text(self, text: str) -> pd.Series:
        window_size = self.config.chunking_config["window_size"]
        if window_size is None:
            return pd.Series({"text": text, "segment_start": 0, "segment_end": len(text)})
        max_start = max(0, len(text) - window_size)
        start_idx = int(np.random.randint(0, max_start + 1))
        end_idx = start_idx + window_size
        return pd.Series(
            {"text": text[start_idx:end_idx], "segment_start": start_idx, "segment_end": end_idx}
        )

    def make_samples(self, queries_per_target_subject_document: int | None = None):
        if queries_per_target_subject_document is None:
            queries_per_target_subject_document = self.config.queries_per_target_subject_document

        dataframe_list = []
        for target_subject in self.config.target_source_mapping:
            tpath = os.path.join(self.config.input_dir, target_subject)
            if os.path.isdir(tpath):
                target_files = glob.glob(os.path.join(tpath, "*.txt"))
            else:
                pq = tpath + ".parquet"
                if not os.path.exists(pq):
                    msg = f"Target subject path {pq} does not exist"
                    raise FileNotFoundError(msg)
                tdf = pd.read_parquet(pq)
                if "file_name" not in tdf.columns or "text" not in tdf.columns:
                    msg = f"Parquet {pq} must have file_name and text columns"
                    raise ValueError(msg)
                target_files = [f"{pq}:{fn}" for fn in tdf["file_name"].tolist()]

            for target_subject_file in target_files:
                document_id = target_subject_file
                tmap = self.config.target_source_mapping[target_subject]
                source_subjects = tmap["source_subjects"]
                source_weights = tmap["source_weights"]
                source_tags = tmap["source_tags"]
                source_tag_weights = tmap["source_tag_weights"]
                pairs = [(s, t) for s in source_subjects for t in source_tags]
                pair_weights = np.array(
                    [w_s * w_t for w_s in source_weights for w_t in source_tag_weights],
                    dtype=float,
                )
                pair_weights = pair_weights / np.sum(pair_weights)
                idxs = np.random.choice(
                    len(pairs),
                    size=queries_per_target_subject_document,
                    replace=True,
                    p=pair_weights,
                )
                sampled = [pairs[i] for i in idxs]
                counts = Counter(sampled)

                for (source_subject, source_tag) in counts:
                    n = counts[(source_subject, source_tag)]
                    num_samples = n * self.config.few_shot_samples_per_query
                    numberline = list(range(n)) * self.config.few_shot_samples_per_query
                    sdf = self.dataset_parsed[source_subject].copy()
                    if source_tag != ("",):
                        sdf = sdf[
                            sdf["tags"].apply(
                                lambda x, st=source_tag: set(st).issubset(set(x.split(",")))
                            )
                        ]

                    if len(sdf) == 0:
                        logger.warning(
                            "No rows for source subject %r tags %r; dropping %d planned samples",
                            source_subject,
                            ",".join(source_tag),
                            num_samples,
                        )
                        continue

                    sampled_df = sdf.sample(n=num_samples, replace=True)
                    sampled_df["numberline"] = numberline[: len(sampled_df)]
                    sampled_df["tags"] = [",".join(source_tag)] * len(sampled_df)
                    grouped = sampled_df.groupby("numberline").agg(list).reset_index()
                    grouped = grouped.drop(columns=["numberline"])
                    grouped["target_subject"] = target_subject
                    grouped["text"] = target_subject_file
                    grouped["document_id"] = document_id
                    dataframe_list.append(grouped)
                    logger.info(
                        "Added %d samples for target %r from source %r tags %r (%s)",
                        len(grouped),
                        target_subject,
                        source_subject,
                        ",".join(source_tag),
                        target_subject_file,
                    )

        if not dataframe_list:
            msg = (
                "No samples were generated. Check target_source_mapping, input files, and tag filters."
            )
            raise ValueError(msg)
        out = pd.concat(dataframe_list).reset_index(drop=True)
        out["text_path"] = out["text"]
        out["text"] = out["text"].apply(self.extract_text_from_path)
        out = out.explode("text", ignore_index=True)
        out[["text", "segment_start", "segment_end"]] = out["text"].apply(self.chunk_text)
        out = out.rename(columns={"id": "id_source"})
        out["id_target"] = out.index.astype(str)

        return out

    def sample_and_dump(self, queries_per_target_subject_document: int | None = None) -> pd.DataFrame:
        seed_df = self.make_samples(queries_per_target_subject_document)
        out_path = os.path.join(self.config.output_dir, self.config.expt_name, "seed.parquet")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        seed_df.to_parquet(out_path)
        logger.info("Wrote %d seed rows to %s", len(seed_df), out_path)
        return seed_df
