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

"""
Data quality assessment: filter registry, scoring, and aggregation.

Wraps NeMo Curator's filter/scorer pipeline with a declarative YAML recipe
interface and provides the ``AssessmentTool`` convenience class.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)

_CURATOR_MSG = (
    "nemo-curator is required for quality assessment. "
    "Install with: pip install nemo-curator"
)

# ---------------------------------------------------------------------------
# Filter registry
# ---------------------------------------------------------------------------

# Registry is populated lazily on first access to avoid hard imports.
FILTER_REGISTRY: Dict[str, type] = {}
_REGISTRY_LOADED = False


def _load_registry() -> None:
    """Populate FILTER_REGISTRY from nemo-curator filter classes."""
    global _REGISTRY_LOADED
    if _REGISTRY_LOADED:
        return
    try:
        from nemo_curator.stages.text.filters.doc_filter import DocumentFilter
        from nemo_curator.stages.text import filters as nc_filters

        # Discover all DocumentFilter subclasses exposed by nemo-curator
        for name in dir(nc_filters):
            obj = getattr(nc_filters, name)
            if isinstance(obj, type) and issubclass(obj, DocumentFilter) and obj is not DocumentFilter:
                FILTER_REGISTRY[name] = obj
    except ImportError:
        pass

    _REGISTRY_LOADED = True


def create_filter(filter_name: str, parameters: Dict[str, Any]):
    """Instantiate a NeMo Curator filter by name.

    Args:
        filter_name: Class name of the filter (e.g. ``DomainFilter``).
        parameters: Keyword arguments forwarded to the filter constructor.

    Returns:
        A filter instance.

    Raises:
        ValueError: If the filter name is not found.
    """
    _load_registry()

    # Handle nested ConversationFilterWrapper
    if filter_name == "ConversationFilterWrapper":
        base_spec = parameters.get("base_filter", {})
        if isinstance(base_spec, dict):
            parameters = dict(parameters)
            parameters["base_filter"] = create_filter(
                base_spec["name"], base_spec.get("parameters", {})
            )

    if filter_name in FILTER_REGISTRY:
        return FILTER_REGISTRY[filter_name](**parameters)

    # Fallback: attempt to import from nemo_curator.stages.text.filters
    try:
        from nemo_curator.stages.text import filters as nc_filters
        from nemo_curator.stages.text.filters.doc_filter import DocumentFilter

        cls = getattr(nc_filters, filter_name, None)
        if cls is not None and issubclass(cls, DocumentFilter):
            return cls(**parameters)
    except ImportError:
        pass

    raise ValueError(
        f"Unknown filter: '{filter_name}'. "
        f"Available: {sorted(FILTER_REGISTRY.keys()) if FILTER_REGISTRY else '(nemo-curator not installed)'}"
    )


# ---------------------------------------------------------------------------
# Scorer list builder
# ---------------------------------------------------------------------------


def create_scorer_list(
    recipe: List[dict],
    text_field: str = "messages",
) -> list:
    """Build a list of NeMo Curator ``BatchScore`` stages from a recipe.

    Args:
        recipe: List of step dicts. Each step has keys ``name``, ``alias``,
            optional ``parameters``, ``enabled``, ``n_gpu``, and ``filter``.
        text_field: Name of the text column in the dataset.

    Returns:
        List of BatchScore stages ready for a Pipeline.
    """
    try:
        from nemo_curator.stages.resources import Resources
    except ImportError as exc:
        raise ImportError(_CURATOR_MSG) from exc

    # Lazy import of BatchScore -- the actual class lives in the curator
    # filter utilities.  We import it here to keep top-level imports light.
    from nemo_curator.stages.text.filters.doc_filter import DocumentFilter  # noqa: F401

    # We replicate the Speaker logic: separate CPU vs GPU filters, then
    # build BatchScore stages.  The ``BatchScore`` wrapper is expected to
    # be available via nemo-curator.
    try:
        from nemo_curator.stages.text.scoring.batch_score import BatchScore
    except ImportError:
        # Fallback for older curator versions
        from nemo_curator.stages.text.filters.batch_score import BatchScore  # type: ignore[no-redef]

    used_aliases: set[str] = set()
    cpu_filters: dict[str, Any] = {}
    gpu_filters: dict[str, Any] = {}
    scorer2ngpu: dict[str, float] = {}

    for step in recipe:
        if not step.get("enabled", True):
            continue

        alias = step["alias"]
        if alias in used_aliases:
            raise ValueError(f"Duplicate alias: {alias}")
        used_aliases.add(alias)

        filter_obj = step.get("filter") or create_filter(
            step["name"], step.get("parameters", {})
        )
        n_gpu = step.get("n_gpu", 0.0)
        scorer2ngpu[alias] = n_gpu

        if n_gpu > 0:
            gpu_filters[alias] = filter_obj
        else:
            cpu_filters[alias] = filter_obj

    scorers_cpu = [
        BatchScore(
            score_fn=filt,
            text_field=text_field,
            score_field=f"score_{alias}",
        )
        for alias, filt in cpu_filters.items()
    ]

    scorers_gpu: list = []
    if gpu_filters:
        score_fn = list(gpu_filters.values())
        score_field = [f"score_{a}" for a in gpu_filters]
        bs = BatchScore(
            score_fn=score_fn if len(score_fn) > 1 else score_fn[0],
            text_field=text_field,
            score_field=score_field if len(score_field) > 1 else score_field[0],
        ).with_(
            resources=Resources(
                gpus=max(scorer2ngpu[a] for a in gpu_filters)
            )
        )
        scorers_gpu.append(bs)

    return scorers_gpu + scorers_cpu


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def aggregate_dicts(
    dicts: list[dict],
    reduce_fn_numeric=np.mean,
    reduce_fn_string=lambda x: dict(Counter(x)),
    ignore_keys: list[str] | None = None,
) -> dict:
    """Recursively aggregate a list of dicts with identical keys."""
    ignore_keys = ignore_keys or []
    if not dicts:
        return {}

    result: dict = {}
    all_keys: set[str] = set()
    for d in dicts:
        all_keys.update(d.keys())

    for key in sorted(all_keys):
        if key in ignore_keys:
            continue

        values = [d[key] for d in dicts if key in d]
        if not values:
            result[key] = "N/A"
            continue

        first = values[0]
        if isinstance(first, dict):
            result[key] = aggregate_dicts(
                [v for v in values if isinstance(v, dict)],
                reduce_fn_numeric,
                reduce_fn_string,
                ignore_keys,
            )
        elif isinstance(first, (int, float, np.integer, np.floating)):
            result[key] = reduce_fn_numeric(values)
        elif isinstance(first, str):
            result[key] = reduce_fn_string(values)
        elif isinstance(first, list) and (not first or isinstance(first[0], str)):
            flat = [v for sub in values for v in sub]
            result[key] = reduce_fn_string(flat)
        else:
            result[key] = "N/A"

    return result


def calculate_aggregates(
    df: "pd.DataFrame",
    ignore_keys: list[str] | None = None,
) -> Dict[str, Any]:
    """Aggregate ``score_*`` columns in a DataFrame.

    Args:
        df: DataFrame with score columns.
        ignore_keys: Keys to skip during nested aggregation.

    Returns:
        Dict mapping score column names to their aggregated values.
    """
    import pandas as pd

    ignore_keys = ignore_keys or []
    score_cols = [c for c in df.columns if c.startswith("score_")]
    result: dict = {}

    for col in score_cols:
        values = list(df[col].dropna())
        if not values:
            result[col] = "N/A"
            continue

        first = values[0]
        if isinstance(first, (int, float, np.integer, np.floating)):
            result[col] = float(np.mean(values))
        elif isinstance(first, str):
            result[col] = dict(Counter(values))
        elif isinstance(first, dict):
            result[col] = aggregate_dicts(values, ignore_keys=ignore_keys)
        else:
            result[col] = "N/A"

    return result


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AssessmentConfig:
    """Configuration for the quality assessment pipeline."""

    recipe: str = ""
    """Path to YAML recipe file, or empty to pass recipe list at runtime."""

    input_file: str = ""
    """Path to input JSONL file."""

    output_dir: str = "data/quality"
    """Directory for output details and aggregates."""

    output_prefix: Optional[str] = None
    """Optional prefix for output filenames."""

    num_workers: Optional[int] = None
    """Number of workers for distributed scoring (default: cpu_count)."""

    aggregate_keys_to_ignore: str = "reasoning,turns,speakers"
    """Comma-separated keys to skip when aggregating."""

    lines_per_split: int = 1000
    """Lines per file when splitting input for parallel processing."""

    splits_per_worker: Optional[int] = None
    """Splits assigned per worker (None = use block_size)."""

    allow_llm_failures: bool = False
    """If True, LLM-based filter failures are ignored."""

    block_size: str = "1kb"
    """Block size for file reading."""

    fields: str = "messages"
    """Comma-separated list of input fields (must include 'messages')."""

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> "AssessmentConfig":
        from omegaconf import OmegaConf

        schema = OmegaConf.structured(AssessmentConfig)
        merged = OmegaConf.merge(schema, cfg)
        return AssessmentConfig(**OmegaConf.to_container(merged, resolve=True))


# ---------------------------------------------------------------------------
# AssessmentTool
# ---------------------------------------------------------------------------


class AssessmentTool:
    """Run a quality-assessment pipeline over a JSONL dataset.

    Wraps NeMo Curator's Pipeline / XennaExecutor to score conversations
    using a declarative recipe, then aggregates results.

    Usage::

        tool = AssessmentTool(cfg)
        results = tool.run()  # -> {"details": ..., "aggregates": ...}
    """

    def __init__(self, cfg: AssessmentConfig, recipe: list[dict] | None = None):
        self.cfg = cfg
        self._recipe_raw = recipe
        self.ray_client = None

    # -- lifecycle ---------------------------------------------------------

    def setup(self) -> None:
        """Initialize Ray, load recipe, split input files."""
        try:
            from nemo_curator.core.client import RayClient
            from nemo_curator.stages.text.io.reader import JsonlReader  # noqa: F401
        except ImportError as exc:
            raise ImportError(_CURATOR_MSG) from exc

        os.environ["ALLOW_LLM_FAILURES"] = "1" if self.cfg.allow_llm_failures else "0"
        self._temp_out = os.path.join(self.cfg.output_dir, "temp")
        os.makedirs(self._temp_out, exist_ok=True)

        self.ray_client = RayClient()
        self.ray_client.start()

        # Load recipe
        if self._recipe_raw is not None:
            recipe = self._recipe_raw
        elif self.cfg.recipe:
            import yaml

            with open(self.cfg.recipe) as f:
                recipe = yaml.safe_load(f)
        else:
            raise ValueError("Either pass recipe= or set cfg.recipe path.")

        self.scorers = create_scorer_list(
            recipe, text_field="messages"
        )

        # Split input
        self._fields = self.cfg.fields.split(",")
        assert "messages" in self._fields, "'messages' field is required"

        self.files = [self.cfg.input_file]
        self._num_workers = self.cfg.num_workers or os.cpu_count() or 1

    def shutdown(self) -> None:
        if self.ray_client:
            self.ray_client.stop()
            self.ray_client = None

    # -- main entry --------------------------------------------------------

    def run(self) -> Dict[str, str]:
        """Execute the full assessment pipeline.

        Returns:
            Dict with ``details`` and ``aggregates`` file paths.
        """
        import pandas as pd

        try:
            from nemo_curator.pipeline import Pipeline
            from nemo_curator.backends.xenna import XennaExecutor
            from nemo_curator.stages.text.io.reader import JsonlReader
        except ImportError as exc:
            raise ImportError(_CURATOR_MSG) from exc

        self.setup()

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None  # type: ignore[assignment]

        all_output_files: list[str] = []
        files = self.files
        spw = self.cfg.splits_per_worker or 1

        iterable = range(0, len(files), self._num_workers * spw)
        if tqdm:
            iterable = tqdm(iterable, desc="Evaluating")

        for idx in iterable:
            batch = files[idx : idx + self._num_workers * spw]
            pipeline = Pipeline(name="assessment_tool")
            pipeline.add_stage(JsonlReader(file_paths=batch, fields=self._fields))

            for scorer in self.scorers:
                pipeline.add_stage(scorer)

            executor = XennaExecutor(
                config={
                    "execution_mode": "streaming",
                    "cpu_allocation_percentage": 0.8,
                    "max_workers_per_stage": self._num_workers,
                }
            )
            results = pipeline.run(executor=executor)
            df = pd.concat([r.to_pandas() for r in results])

            fname = "_".join(
                os.path.basename(f).split(".")[0] for f in batch
            )
            out_path = os.path.join(self._temp_out, f"{fname}.jsonl")
            df.to_json(out_path, orient="records", lines=True)
            all_output_files.append(out_path)

        # Aggregate
        pipeline = Pipeline(name="aggregation")
        pipeline.add_stage(
            JsonlReader(file_paths=all_output_files, fields=list(df.columns))
        )
        results = pipeline.run()
        df = pd.concat([r.to_pandas() for r in results])

        agg = calculate_aggregates(
            df,
            ignore_keys=self.cfg.aggregate_keys_to_ignore.split(","),
        )

        prefix = self.cfg.output_prefix or Path(self.cfg.input_file).stem
        details_path = os.path.join(self.cfg.output_dir, f"{prefix}_details.jsonl")
        agg_path = os.path.join(self.cfg.output_dir, f"{prefix}_aggregates.json")

        df.to_json(details_path, orient="records", lines=True)
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)

        # Cleanup temp
        if os.path.isdir(self._temp_out):
            shutil.rmtree(self._temp_out, ignore_errors=True)

        self.shutdown()

        log.info("Details  -> %s", details_path)
        log.info("Aggregates -> %s", agg_path)
        return {"details": details_path, "aggregates": agg_path}


# ---------------------------------------------------------------------------
# High-level entry points (called from CLI / run scripts)
# ---------------------------------------------------------------------------


def evaluate_data_quality(cfg: DictConfig) -> Dict[str, str]:
    """Run data quality assessment using NeMo Curator filters/scorers.

    This is the main entry point for ``--mode data`` evaluation.
    It loads the assessment config, builds the AssessmentTool pipeline,
    and returns output file paths.

    Args:
        cfg: OmegaConf config dict with keys matching :class:`AssessmentConfig`
            fields (recipe, input_file, output_dir, etc.).

    Returns:
        Dict with ``details`` and ``aggregates`` file paths.

    Raises:
        ValueError: If required config fields (input_file, recipe) are missing.
    """
    assessment_cfg = AssessmentConfig.from_omegaconf(cfg)

    if not assessment_cfg.input_file:
        raise ValueError(
            "data_eval.input_file is required for data quality evaluation. "
            "Set it via CLI: data_eval.input_file=/path/to/data.jsonl"
        )

    # Resolve num_workers=-1 to auto-detect
    if assessment_cfg.num_workers is not None and assessment_cfg.num_workers < 0:
        assessment_cfg.num_workers = None  # AssessmentTool defaults to cpu_count()

    tool = AssessmentTool(assessment_cfg)
    return tool.run()


def evaluate_model(cfg: DictConfig) -> Dict[str, Any]:
    """Run model benchmark evaluation using nemo-evaluator-launcher.

    This is a thin wrapper for use from run scripts. The CLI command
    (customize/eval) calls ``run_eval()`` directly (same as nano3/super3),
    so this function is primarily for programmatic / run-script usage.

    Args:
        cfg: OmegaConf config dict with evaluation, deployment, execution
            sections (same structure as the stage5_eval default.yaml).

    Returns:
        Dict with ``invocation_id`` if evaluation was submitted.

    Raises:
        ImportError: If nemo-evaluator-launcher is not installed.
    """
    from omegaconf import OmegaConf

    try:
        from nemo_evaluator_launcher.api.functional import run_eval
    except ImportError as exc:
        raise ImportError(
            "nemo-evaluator-launcher is required for model evaluation. "
            'Install with: pip install "nemotron[evaluator]"'
        ) from exc

    # Extract evaluator config (strip 'run' section if present)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    eval_config = {k: v for k, v in cfg_dict.items() if k != "run"}
    eval_config = OmegaConf.create(eval_config)

    invocation_id = run_eval(eval_config, dry_run=False)

    result: Dict[str, Any] = {"invocation_id": invocation_id}
    if invocation_id:
        log.info("Evaluation submitted: %s", invocation_id)
    return result
