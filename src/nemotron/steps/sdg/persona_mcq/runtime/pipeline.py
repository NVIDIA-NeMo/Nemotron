# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate the config-driven Persona MCQ pipeline."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from nemotron.steps.sdg.persona_mcq.runtime.answers import generate_answers
from nemotron.steps.sdg.persona_mcq.runtime.io import (
    canonical_hash,
    read_jsonl,
    redact_config,
    require_file,
    write_json,
    write_jsonl,
)
from nemotron.steps.sdg.persona_mcq.runtime.languages import compile_script_pattern, validate_fraction
from nemotron.steps.sdg.persona_mcq.runtime.lexical import deduplicate as lexical_deduplicate
from nemotron.steps.sdg.persona_mcq.runtime.semantic import deduplicate_embeddings, embed_questions
from nemotron.steps.sdg.persona_mcq.runtime.sft import build_sft_records, prepare_answer_seed, sample_aligned_datasets

STAGES = (
    "personas",
    "questions",
    "lexical_dedup",
    "semantic_dedup",
    "answer_seed",
    "answers",
    "build_sft",
    "sample",
)


def _stage_list(value: Any) -> list[str]:
    """Accept YAML lists and the bracket-list strings produced by CLI overrides."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return list(value or [])


def _identity_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove execution controls that may legitimately change between resumes."""
    identity = redact_config(config)
    pipeline = identity.get("pipeline") or {}
    for key in ("stages", "resume", "overwrite"):
        pipeline.pop(key, None)
    return identity


def validate_config(config: dict[str, Any]) -> None:
    pipeline = config.get("pipeline") or {}
    if not pipeline.get("experiment_name"):
        raise ValueError("pipeline.experiment_name is required")
    stages = _stage_list(pipeline.get("stages"))
    if not stages:
        raise ValueError("pipeline.stages must select 'all' or at least one named stage")
    unknown = set(stages) - set(STAGES) - {"all"}
    if unknown:
        raise ValueError(f"Unknown Persona MCQ stages: {sorted(unknown)}")
    if "all" in stages and len(stages) != 1:
        raise ValueError("pipeline.stages may contain 'all' or named stages, not both")
    languages = config.get("languages") or {}
    if not languages:
        raise ValueError("at least one language must be configured")
    for name, language in languages.items():
        required = ("locale", "display_name", "answer_label", "script_pattern")
        missing = [field for field in required if not language.get(field)]
        if missing:
            raise ValueError(f"language {name!r} requires: {', '.join(missing)}")
        aliases = language.get("answer_label_aliases") or []
        if not isinstance(aliases, list) or not all(isinstance(alias, str) and alias for alias in aliases):
            raise ValueError(f"languages.{name}.answer_label_aliases must be a list of non-empty strings")
        compile_script_pattern(str(language["script_pattern"]))
        for field in ("question_script_fraction", "answer_script_fraction"):
            validate_fraction(language.get(field) or {}, f"languages.{name}.{field}")
    models = config.get("models") or {}
    for group in ("question_models", "answer_models"):
        selected = config.get(group) or []
        if not selected:
            raise ValueError(f"{group} must select at least one model")
        missing = set(selected) - set(models)
        if missing:
            raise ValueError(f"{group} references unknown models: {sorted(missing)}")
    if len(config["answer_models"]) < 3:
        raise ValueError("Persona MCQ requires at least three answer models for voting")
    response_teachers = (config.get("sft") or {}).get("response_teachers") or []
    if not response_teachers:
        raise ValueError("sft.response_teachers must select at least one answer model")
    missing_teachers = set(response_teachers) - set(config["answer_models"])
    if missing_teachers:
        raise ValueError(
            f"sft.response_teachers references models not selected in answer_models: {sorted(missing_teachers)}"
        )
    for key, model in models.items():
        for field in ("model", "endpoint", "api_key_env"):
            if not model.get(field):
                raise ValueError(f"model {key!r} requires {field}")
        if not str(model["api_key_env"]).replace("_", "").isalnum():
            raise ValueError(f"model {key!r} api_key_env must be an environment-variable name")
    semantic = config.get("semantic_dedup") or {}
    if not 0 <= float(semantic.get("threshold", 0.965)) <= 1:
        raise ValueError("semantic_dedup.threshold must be between zero and one")
    agreement = (config.get("sft") or {}).get("agreement")
    if agreement not in {"unanimous", "majority"}:
        raise ValueError("sft.agreement must be 'unanimous' or 'majority'")
    reasoning = (config.get("sft") or {}).get("reasoning") or {}
    if not reasoning.get("display_name") or not reasoning.get("script_pattern"):
        raise ValueError("sft.reasoning requires display_name and script_pattern")
    compile_script_pattern(str(reasoning["script_pattern"]))
    validate_fraction(reasoning.get("script_fraction") or {}, "sft.reasoning.script_fraction")
    sampling = config.get("sampling") or {}
    if int(sampling.get("per_language", 0)) <= 0:
        raise ValueError("sampling.per_language must be greater than zero")
    if not 0 <= float(sampling.get("reasoning_off_fraction", -1)) <= 1:
        raise ValueError("sampling.reasoning_off_fraction must be between zero and one")
    if sampling.get("answer_variant") not in {"full", "stripped"}:
        raise ValueError("sampling.answer_variant must be 'full' or 'stripped'")


def _records_from_result(result: Any) -> list[dict[str, Any]]:
    dataset = result.load_dataset() if hasattr(result, "load_dataset") else getattr(result, "dataset", None)
    if dataset is None:
        raise ValueError("Data Designer returned no dataset")
    if isinstance(dataset, list):
        return dataset
    if hasattr(dataset, "to_pandas"):
        dataset = dataset.to_pandas()
    if hasattr(dataset, "to_dict"):
        return dataset.to_dict(orient="records")
    raise TypeError(f"Unsupported Data Designer dataset type: {type(dataset).__name__}")


def generate_questions(
    *,
    model_key: str,
    model: dict[str, Any],
    language_key: str,
    language: dict[str, Any],
    count: int,
    artifact_root: Path,
    dataset_name: str,
    resume: bool,
    random_seed: int,
    max_parallel: int,
    buffer_size: int,
) -> list[dict[str, Any]]:
    from nemotron.steps.sdg.plugins.persona_mcq.plugin import ensure_registered

    # Source-staged remote jobs do not have this checkout's package entry-point
    # metadata. Register before importing Data Designer's column-type union.
    ensure_registered()

    import data_designer.config as dd
    from data_designer.config.models import ModelProvider
    from data_designer.interface import DataDesigner

    from nemotron.steps.sdg.plugins.persona_mcq.config import PersonaMCQConfig

    inference = dict((model.get("question") or {}).get("inference") or {})
    provider_name = f"persona-mcq-{model_key}"
    provider = ModelProvider(
        name=provider_name,
        endpoint=model["endpoint"],
        provider_type=model.get("provider_type", "openai"),
        api_key=model["api_key_env"],
        extra_headers=model.get("extra_headers"),
    )
    model_config = dd.ModelConfig(
        alias="question_model",
        model=model["model"],
        provider=provider_name,
        skip_health_check=bool(model.get("skip_health_check", False)),
        inference_parameters=dd.ChatCompletionInferenceParams(**inference),
    )
    builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
    builder.add_column(
        dd.SamplerColumnConfig(
            name="persona",
            drop=False,
            sampler_type=dd.SamplerType.PERSON,
            params=dd.PersonSamplerParams(locale=language["locale"], with_synthetic_personas=True),
        )
    )
    builder.add_column(
        PersonaMCQConfig(
            name="conversation",
            drop=False,
            model_alias="question_model",
            language=language["display_name"],
            random_seed=random_seed,
        )
    )
    client = DataDesigner(artifact_path=artifact_root, model_providers=[provider])
    client.set_run_config(
        dd.RunConfig(
            non_inference_max_parallel_workers=max_parallel,
            buffer_size=buffer_size,
            disable_early_shutdown=True,
        )
    )
    kwargs: dict[str, Any] = {"config_builder": builder, "num_records": count, "dataset_name": dataset_name}
    if "resume" in inspect.signature(client.create).parameters:
        kwargs["resume"] = dd.ResumeMode.IF_POSSIBLE if resume else dd.ResumeMode.NEVER
    result = client.create(**kwargs)
    records = _records_from_result(result)
    for record in records:
        record["source_model"] = model_key
        record["language"] = language_key
    return records


class PersonaMCQPipeline:
    """Run selected Persona MCQ stages against one immutable experiment configuration."""

    def __init__(self, config: dict[str, Any]) -> None:
        config.setdefault("pipeline", {})["stages"] = _stage_list((config.get("pipeline") or {}).get("stages"))
        validate_config(config)
        self.config = config
        pipeline = config["pipeline"]
        self.root = Path(pipeline["output_root"]).expanduser().resolve() / pipeline["experiment_name"]
        self.resume = bool(pipeline.get("resume", True))
        self.overwrite = bool(pipeline.get("overwrite", False))
        self.summary: dict[str, Any] = {"stages": {}}

    def run(self) -> None:
        stages = list(self.config["pipeline"].get("stages") or ["all"])
        selected = list(STAGES) if stages == ["all"] else stages
        self._prepare_experiment()
        functions: dict[str, Callable[[], None]] = {
            "personas": self._personas,
            "questions": self._questions,
            "lexical_dedup": self._lexical,
            "semantic_dedup": self._semantic,
            "answer_seed": self._answer_seed,
            "answers": self._answers,
            "build_sft": self._build_sft,
            "sample": self._sample,
        }
        for stage in STAGES:
            if stage in selected:
                print(f"[persona_mcq] starting stage: {stage}", flush=True)
                functions[stage]()
                self._write_summary()

    def _prepare_experiment(self) -> None:
        state_path = self.root / "run.json"
        config_hash = canonical_hash(_identity_config(self.config))
        if self.overwrite and self.root.exists():
            shutil.rmtree(self.root)
            self.summary = {"stages": {}}
        self.root.mkdir(parents=True, exist_ok=True)
        if state_path.exists():
            existing = json.loads(state_path.read_text(encoding="utf-8"))
            if existing.get("config_hash") != config_hash and not self.overwrite:
                raise ValueError(
                    f"Experiment {self.root} was created with a different config; choose a new experiment_name "
                    "or set pipeline.overwrite=true"
                )
            summary_path = self.root / "summary.json"
            if self.resume and summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if not isinstance(summary, dict) or not isinstance(summary.get("stages"), dict):
                    raise ValueError(f"Invalid Persona MCQ summary: {summary_path}")
                self.summary = summary
        try:
            dd_version = version("data-designer")
        except PackageNotFoundError:
            dd_version = "not-installed"
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        state = {
            "config_hash": config_hash,
            "config": redact_config(self.config),
            "data_designer_version": dd_version,
            "nemotron_commit": commit,
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_summary(self) -> None:
        (self.root / "summary.json").write_text(
            json.dumps(self.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _personas(self) -> None:
        locales = list(dict.fromkeys(str(language["locale"]) for language in self.config["languages"].values()))
        data_designer_home = Path(os.environ.get("DATA_DESIGNER_HOME", "~/.data-designer")).expanduser().resolve()
        assets_root = (
            Path(os.environ.get("DATA_DESIGNER_MANAGED_ASSETS_PATH", data_designer_home / "managed-assets"))
            .expanduser()
            .resolve()
        )
        datasets_dir = assets_root / "datasets"
        cached = [locale for locale in locales if (datasets_dir / f"{locale}.parquet").is_file()]
        missing = [locale for locale in locales if locale not in cached]

        if missing:
            if not os.environ.get("NGC_API_KEY"):
                raise RuntimeError(
                    "Missing persona assets for locales "
                    f"{', '.join(missing)}. Export NGC_API_KEY and rerun the pipeline."
                )
            if shutil.which("ngc") is None:
                raise RuntimeError(
                    "NGC CLI is required to download missing persona assets. Install it and ensure 'ngc' is on PATH."
                )

            from data_designer.cli.repositories.persona_repository import PersonaRepository
            from data_designer.cli.services.download_service import DownloadService

            datasets_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="persona-mcq-ngc-") as temporary_dir:
                # NGC CLI authenticates from NGC_API_KEY, but authenticated
                # public-catalog downloads also require an org context. Keep
                # that non-secret context in an ephemeral HOME so no API key is
                # ever written to disk.
                ngc_home = Path(temporary_dir) / "home"
                ngc_config = ngc_home / ".ngc" / "config"
                ngc_config.parent.mkdir(parents=True)
                ngc_config.write_text(
                    "[CURRENT]\nformat_type = ascii\norg = no-org\nteam = no-team\nace = no-ace\n",
                    encoding="utf-8",
                )
                ngc_config.chmod(0o600)
                service_options = {}
                if "ngc_config_path" in inspect.signature(DownloadService).parameters:
                    service_options["ngc_config_path"] = ngc_config
                service = DownloadService(data_designer_home, PersonaRepository(), **service_options)
                service.managed_assets_dir = datasets_dir
                previous_home = os.environ.get("HOME")
                os.environ["HOME"] = str(ngc_home)
                try:
                    for locale in missing:
                        print(f"[persona_mcq] downloading persona asset: {locale}", flush=True)
                        service.download_persona_dataset(locale)
                finally:
                    if previous_home is None:
                        os.environ.pop("HOME", None)
                    else:
                        os.environ["HOME"] = previous_home

            incomplete = [locale for locale in missing if not (datasets_dir / f"{locale}.parquet").is_file()]
            if incomplete:
                raise RuntimeError("Persona downloads did not produce assets for: " + ", ".join(incomplete))

        self.summary["stages"]["personas"] = {
            "assets_path": str(assets_root),
            "required_locales": locales,
            "cached_locales": cached,
            "downloaded_locales": missing,
        }

    def _questions(self) -> None:
        cfg = self.config
        stats: dict[str, int] = {}
        for model_key in cfg["question_models"]:
            for language_key, language in cfg["languages"].items():
                output = self.root / "questions" / model_key / language_key / "records.jsonl"
                if self.resume and not self.overwrite and output.exists():
                    stats[f"{model_key}/{language_key}"] = len(read_jsonl(output))
                    continue
                records = generate_questions(
                    model_key=model_key,
                    model=cfg["models"][model_key],
                    language_key=language_key,
                    language=language,
                    count=int(cfg["question_generation"]["num_records"]),
                    artifact_root=output.parent / "artifacts",
                    dataset_name=f"{cfg['pipeline']['experiment_name']}_{model_key}_{language_key}",
                    resume=self.resume and not self.overwrite,
                    random_seed=int(cfg["pipeline"]["seed"]),
                    max_parallel=int(cfg["question_generation"]["max_parallel_requests"]),
                    buffer_size=int(cfg["question_generation"]["buffer_size"]),
                )
                stats[f"{model_key}/{language_key}"] = write_jsonl(output, records)
        self.summary["stages"]["questions"] = stats

    def _lexical(self) -> None:
        cfg = self.config["lexical_dedup"]
        stage_stats: dict[str, Any] = {}
        for language, language_cfg in self.config["languages"].items():
            question_fraction = language_cfg["question_script_fraction"]
            accepted: list[dict[str, Any]] = []
            per_model: dict[str, Any] = {}
            for model in self.config["question_models"]:
                path = require_file(self.root / "questions" / model / language / "records.jsonl", "lexical_dedup")
                records, stats = lexical_deduplicate(
                    read_jsonl(path),
                    language=language,
                    script_pattern=language_cfg["script_pattern"],
                    min_script_fraction=float(question_fraction["min"]),
                    max_script_fraction=float(question_fraction["max"]),
                    strip_latin_glosses=bool(language_cfg.get("strip_latin_glosses", False)),
                    source_model=model,
                    threshold=float(cfg["threshold"]),
                    shingle_size=int(cfg["shingle_size"]),
                    permutations=int(cfg["permutations"]),
                    bands=int(cfg["bands"]),
                    seed=int(cfg["seed"]),
                )
                accepted.extend(records)
                per_model[model] = stats
            pooled_input = [
                {
                    "conversation": json.dumps(
                        {
                            "metadata": {
                                "parsed_question": {"question": row["question"], "choices": row["choices"]},
                                **row["metadata"],
                            }
                        }
                    )
                }
                for row in accepted
            ]
            pooled, pooled_stats = lexical_deduplicate(
                pooled_input,
                language=language,
                script_pattern=language_cfg["script_pattern"],
                min_script_fraction=float(question_fraction["min"]),
                max_script_fraction=float(question_fraction["max"]),
                strip_latin_glosses=bool(language_cfg.get("strip_latin_glosses", False)),
                source_model="pooled",
                threshold=float(cfg["threshold"]),
                shingle_size=int(cfg["shingle_size"]),
                permutations=int(cfg["permutations"]),
                bands=int(cfg["bands"]),
                seed=int(cfg["seed"]),
            )
            # Restore source provenance by query identity from the first pass.
            source_by_identity = {(row["question"], tuple(row["choices"])): row["metadata"] for row in accepted}
            for row in pooled:
                row["metadata"] = source_by_identity[(row["question"], tuple(row["choices"]))]
            write_jsonl(self.root / "lexical" / f"{language}.jsonl", pooled)
            stage_stats[language] = {"models": per_model, "pooled": pooled_stats}
        self.summary["stages"]["lexical_dedup"] = stage_stats

    def _semantic(self) -> None:
        cfg = self.config["semantic_dedup"]
        stage_stats: dict[str, Any] = {}
        for language in self.config["languages"]:
            records = read_jsonl(require_file(self.root / "lexical" / f"{language}.jsonl", "semantic_dedup"))
            embeddings = embed_questions(
                records,
                model_name=cfg["model"],
                device=cfg["device"],
                batch_size=int(cfg["batch_size"]),
            )
            records, stats = deduplicate_embeddings(
                records,
                embeddings,
                threshold=float(cfg["threshold"]),
                method=cfg["method"],
                seed=int(cfg["seed"]),
                chunk_size=int(cfg["chunk_size"]),
            )
            write_jsonl(self.root / "semantic" / f"{language}.jsonl", records)
            stage_stats[language] = stats
        self.summary["stages"]["semantic_dedup"] = stage_stats

    def _answer_seed(self) -> None:
        stats: dict[str, int] = {}
        seed = int(self.config["pipeline"]["seed"])
        for language in self.config["languages"]:
            records = read_jsonl(require_file(self.root / "semantic" / f"{language}.jsonl", "answer_seed"))
            stats[language] = write_jsonl(
                self.root / "answer_seed" / f"{language}.jsonl", prepare_answer_seed(records, seed)
            )
        self.summary["stages"]["answer_seed"] = stats

    def _answers(self) -> None:
        cfg = self.config["answer_generation"]
        stats: dict[str, Any] = {}
        for model_key in self.config["answer_models"]:
            model = {**self.config["models"][model_key], **(self.config["models"][model_key].get("answer") or {})}
            for language, language_cfg in self.config["languages"].items():
                records = read_jsonl(require_file(self.root / "answer_seed" / f"{language}.jsonl", "answers"))
                key = f"{model_key}/{language}"
                stats[key] = asyncio.run(
                    generate_answers(
                        records,
                        language_config=language_cfg,
                        reasoning_config=self.config["sft"]["reasoning"],
                        model=model,
                        output_path=self.root / "answers" / model_key / language / "answers.jsonl",
                        failure_path=self.root / "answers" / model_key / language / "failures.jsonl",
                        resume=self.resume and not self.overwrite,
                        max_parallel=int(cfg["max_parallel_requests"]),
                        timeout=float(cfg["timeout"]),
                        max_retries=int(cfg["max_retries"]),
                    )
                )
        self.summary["stages"]["answers"] = stats

    def _build_sft(self) -> None:
        cfg = self.config["sft"]
        stats: dict[str, Any] = {}
        for language, language_cfg in self.config["languages"].items():
            answers = {
                model: read_jsonl(
                    require_file(self.root / "answers" / model / language / "answers.jsonl", "build_sft")
                )
                for model in self.config["answer_models"]
            }
            for teacher in cfg["response_teachers"]:
                records, result = build_sft_records(
                    answers,
                    response_model=teacher,
                    language=language,
                    language_config=language_cfg,
                    reasoning_config=cfg["reasoning"],
                    agreement=cfg["agreement"],
                )
                write_jsonl(self.root / "sft" / teacher / f"{language}.jsonl", records)
                stats[f"{teacher}/{language}"] = result
        self.summary["stages"]["build_sft"] = stats

    def _sample(self) -> None:
        cfg = self.config["sampling"]
        datasets = {
            teacher: {
                language: read_jsonl(require_file(self.root / "sft" / teacher / f"{language}.jsonl", "sample"))
                for language in self.config["languages"]
            }
            for teacher in self.config["sft"]["response_teachers"]
        }
        outputs, stats = sample_aligned_datasets(
            datasets,
            sample_per_language=int(cfg["per_language"]),
            seed=int(cfg["seed"]),
            reasoning_off_fraction=float(cfg["reasoning_off_fraction"]),
            answer_variant=cfg["answer_variant"],
        )

        def write_handoff(root: Path, name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
            training_jsonl = root / "train.jsonl"
            blend_path = root / "blend.json"
            count = write_jsonl(training_jsonl, records)
            write_json(
                blend_path,
                {
                    "datasets": [
                        {
                            "name": name,
                            "path": str(training_jsonl.resolve()),
                            "weight": 1.0,
                        }
                    ]
                },
            )
            return {
                "records": count,
                "training_jsonl": str(training_jsonl),
                "blend": str(blend_path),
            }

        artifacts: dict[str, dict[str, Any]] = {}
        for teacher, records in outputs.items():
            teacher_root = self.root / "training" / teacher
            views = {
                "all": write_handoff(teacher_root, f"persona-mcq-{teacher}", records),
            }
            if "english" in self.config["languages"]:
                for target_language in self.config["languages"]:
                    if target_language == "english":
                        continue
                    view_name = f"english_{target_language}"
                    paired = [
                        record for record in records if record["metadata"]["language"] in {"english", target_language}
                    ]
                    views[view_name] = write_handoff(
                        teacher_root / view_name,
                        f"persona-mcq-{teacher}-{view_name.replace('_', '-')}",
                        paired,
                    )
            artifacts[teacher] = {"views": views}
        self.summary["stages"]["sample"] = {**stats, "teachers": artifacts}
