# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import glob
import logging
import os
import shutil
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.embedders import EmbeddingCreatorStage
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.stages.text.models.tokenizer import TokenizerStage
from nemo_curator.stages.text.models.utils import (
    ATTENTION_MASK_FIELD,
    INPUT_ID_FIELD,
    SEQ_ORDER_FIELD,
    TOKEN_LENGTH_FIELD,
)
from nemo_curator.tasks import DocumentBatch

from nemotron.steps.byob.runtime.config import ByobConfig

logger = logging.getLogger(__name__)


class CallableTokenizerStage(TokenizerStage):
    """Curator tokenizer stage using the stable callable tokenizer interface.

    NeMo Curator 1.3 calls ``batch_encode_plus``, which is absent from some
    Transformers 5 tokenizers. Calling the tokenizer with the same arguments
    works on both Transformers 4 and 5.
    """

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        if self.max_chars is not None and self.max_chars > 0:
            df[self.text_field] = df[self.text_field].str.slice(0, self.max_chars)

        with torch.no_grad():
            tokens = self.tokenizer(
                df[self.text_field].tolist(),
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="np",
                truncation=True,
                add_special_tokens=True,
                return_token_type_ids=False,
            )

        output = df.copy()
        output[INPUT_ID_FIELD] = tokens.input_ids.tolist()
        output[ATTENTION_MASK_FIELD] = tokens.attention_mask.tolist()
        if self.sort_by_length:
            output[SEQ_ORDER_FIELD] = np.arange(len(df))
            output[TOKEN_LENGTH_FIELD] = tokens.attention_mask.sum(axis=1)
            output = output.sort_values(by=TOKEN_LENGTH_FIELD, kind="stable", ignore_index=True).drop(
                columns=[TOKEN_LENGTH_FIELD]
            )
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=output,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class TextSemanticDeduplication(ABC):
    def __init__(self, config: ByobConfig):
        # Importing Curator's semantic workflow loads its GPU-only dependencies.
        # Keep that import on the execution path so CPU-only tooling can import
        # and test the tokenizer adapter.
        from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow

        self.config = config
        self.input_path = os.path.join(
            self.config.output_dir, self.config.expt_name, "artifacts", "semantic_deduplication", "input_data"
        )
        self.output_path = os.path.join(
            self.config.output_dir, self.config.expt_name, "artifacts", "semantic_deduplication", "results"
        )
        self.cache_path = os.path.join(
            self.config.output_dir, self.config.expt_name, "artifacts", "semantic_deduplication", "cache"
        )
        self.embeddings_path = os.path.join(self.cache_path, "embeddings")
        self.semantic_cache_path = os.path.join(self.cache_path, "semantic_dedup")

        self.workflow = SemanticDeduplicationWorkflow(
            input_path=self.embeddings_path,
            output_path=self.output_path,
            cache_path=self.semantic_cache_path,
            n_clusters=self.config.semantic_deduplication_config["n_clusters"],
            id_field="id",
            embedding_field="embeddings",
            eps=self.config.semantic_deduplication_config["eps"],
        )
        self.executor = RayDataExecutor()
        self.kmeans_executor = RayActorPoolExecutor()
        if os.path.exists(self.output_path):
            logger.warning(f"Output path {self.output_path} already exists. Removing it.")
            shutil.rmtree(self.output_path)
        if os.path.exists(self.cache_path):
            logger.warning(f"Cache path {self.cache_path} already exists. Removing it.")
            shutil.rmtree(self.cache_path)

        os.makedirs(self.input_path, exist_ok=True)

    @abstractmethod
    def prepare_input_data(self, dataset: pd.DataFrame):
        pass

    def _compute_embeddings(self, input_file: str) -> None:
        model_identifier = self.config.semantic_deduplication_config["model_identifier"]
        embedding_stage = EmbeddingCreatorStage(
            model_identifier=model_identifier,
            text_field="text",
            embedding_field="embeddings",
        )
        # Retain Curator's download/setup and embedding-model stages, replacing
        # only the tokenizer call that is incompatible with newer Transformers.
        original = embedding_stage.stages[0]
        embedding_stage.stages[0] = CallableTokenizerStage(
            model_identifier=original.model_identifier,
            cache_dir=original.cache_dir,
            hf_token=original.hf_token,
            text_field=original.text_field,
            max_chars=original.max_chars,
            max_seq_length=original.max_seq_length,
            padding_side=original.padding_side,
            sort_by_length=original.sort_by_length,
            unk_token=original.unk_token,
            transformers_init_kwargs=original.transformers_init_kwargs,
        )
        embedding_pipeline = Pipeline(
            stages=[
                ParquetReader(file_paths=input_file, _generate_ids=False),
                embedding_stage,
                ParquetWriter(path=self.embeddings_path, fields=["id", "embeddings"], mode="overwrite"),
            ],
            name="byob_semantic_dedup_embedding_pipeline",
        )
        embedding_pipeline.run(self.executor)

    def _mark_duplicates(self, dataset: pd.DataFrame):
        paths = glob.glob(os.path.join(self.output_path, "duplicates/*.parquet"))
        if not paths:
            dataset["is_duplicate"] = False
            return dataset

        duplicates = pd.concat([pd.read_parquet(path) for path in paths])
        dataset["is_duplicate"] = dataset["id_question"].isin(duplicates["id"])
        return dataset

    def run(self, dataset: pd.DataFrame):
        dataset_temp = self.prepare_input_data(dataset)
        input_file = os.path.join(self.input_path, "questions.parquet")
        dataset_temp.to_parquet(input_file)
        self._compute_embeddings(input_file)
        self.workflow.run(kmeans_executor=self.kmeans_executor, pairwise_executor=self.executor)
        dataset_dedup = self._mark_duplicates(dataset)
        num_duplicates = dataset_dedup["is_duplicate"].sum()
        logger.info(f"Found {num_duplicates}/{len(dataset_dedup)} duplicate questions")
        if self.config.semantic_deduplication_config["remove_duplicates"]:
            dataset_dedup = dataset_dedup[~dataset_dedup["is_duplicate"]]
        return dataset_dedup
