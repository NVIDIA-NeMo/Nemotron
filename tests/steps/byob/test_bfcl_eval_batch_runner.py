from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

import pytest

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import eval_artifacts, eval_runner
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_errors import (
    CandidateClientError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_errors import (
    ContaminationError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_contract import (
    EpisodeStatus,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_errors import (
    ConversationDriverError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.error_taxonomy import (
    ERROR_TAXONOMY_HASH,
    EXECUTABLE_EPISODE_ATTRIBUTION,
    EXECUTABLE_NON_CANDIDATE_STOPS,
    FATAL_EVAL_ERROR_CODES,
    TRACE_EPISODE_ATTRIBUTION,
    TRACE_NON_CANDIDATE_STOPS,
    episode_attribution,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.errors import (
    EvalConfigError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.eval_artifacts import (
    EvalArtifactError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_contract import (
    ExecutableEpisodeStatus,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_errors import (
    ExecutableEvalError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_scoring_errors import (
    ExecutableScoringError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_errors import (
    SourceVerificationError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_scoring_errors import (
    TraceScoringError,
)


class _Plan:
    candidate_aliases = ("candidate_a",)

    def __init__(self, task_ids: tuple[str, ...]) -> None:
        self._task_ids = task_ids

    def evaluation_task_ids(self, alias: str) -> tuple[str, ...]:
        assert alias == "candidate_a"
        return self._task_ids


class _Client:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events

    async def complete(self, request: Any, *, deadline: float | None = None) -> Any:
        raise AssertionError("the driver is replaced in these orchestration tests")

    async def aclose(self) -> None:
        if self.events is not None:
            self.events.append("client.close")


class _Oracle:
    def __init__(self, task_id: str, closed: list[str]) -> None:
        self.task_id = task_id
        self._closed = closed

    async def close(self) -> None:
        self._closed.append(self.task_id)


def _batch_config(tmp_path: Path, parallelism: int) -> Any:
    return SimpleNamespace(
        limits=SimpleNamespace(max_parallel_tasks=parallelism),
        scoring=SimpleNamespace(),
        outputs=SimpleNamespace(output_dir=tmp_path),
    )


def test_published_candidate_cache_exists_before_any_completion(tmp_path: Path) -> None:
    config = SimpleNamespace(
        outputs=SimpleNamespace(
            output_dir=tmp_path / "eval-output",
            cache_candidate_responses=True,
        )
    )

    _cache, path, temporary = eval_runner._candidate_cache(config)

    assert temporary is None
    assert path.is_file()
    assert path.read_bytes() == b""


def test_batch_runner_bounds_tasks_preserves_order_and_isolates_oracles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    peak = 0
    closed: list[str] = []
    oracles: list[_Oracle] = []

    def build(_projection: Any, task_id: str, **_kwargs: Any) -> Any:
        return SimpleNamespace(task_id=task_id)

    def oracle_factory(_source: Any, task: Any, _limits: Any) -> _Oracle:
        oracle = _Oracle(task.task_id, closed)
        oracles.append(oracle)
        return oracle

    async def drive(**kwargs: Any) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.02)
            return kwargs["task"].task_id
        finally:
            active -= 1
            await kwargs["oracle"].close()

    monkeypatch.setattr(eval_runner, "build_executable_task_spec", build)
    monkeypatch.setattr(eval_runner, "run_executable_episode", drive)
    monkeypatch.setattr(
        eval_runner,
        "score_executable_episode",
        lambda **kwargs: f"score:{kwargs['episode']}",
    )
    task_ids = tuple(f"task-{index}" for index in range(6))

    scores = asyncio.run(
        eval_runner._run_candidate_tasks(
            config=_batch_config(tmp_path, 2),
            source=object(),
            plan=_Plan(task_ids),
            projection=object(),
            candidate=SimpleNamespace(alias="candidate_a"),
            client=_Client(),
            tool_trace_cache=None,
            oracle_factory=oracle_factory,
        )
    )

    assert scores == tuple(f"score:{task_id}" for task_id in task_ids)
    assert peak == 2
    assert len({id(oracle) for oracle in oracles}) == len(task_ids)
    assert sorted(closed) == sorted(task_ids)


def test_batch_runner_cancels_siblings_and_closes_every_open_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    closed: list[str] = []

    monkeypatch.setattr(
        eval_runner,
        "build_executable_task_spec",
        lambda _projection, task_id, **_kwargs: SimpleNamespace(task_id=task_id),
    )

    def oracle_factory(_source: Any, task: Any, _limits: Any) -> _Oracle:
        opened.append(task.task_id)
        return _Oracle(task.task_id, closed)

    async def drive(**kwargs: Any) -> str:
        try:
            if kwargs["task"].task_id == "task-1":
                raise RuntimeError("infrastructure failed")
            await asyncio.sleep(10)
            return kwargs["task"].task_id
        finally:
            await kwargs["oracle"].close()

    monkeypatch.setattr(eval_runner, "run_executable_episode", drive)

    with pytest.raises(RuntimeError, match="infrastructure failed"):
        asyncio.run(
            eval_runner._run_candidate_tasks(
                config=_batch_config(tmp_path, 2),
                source=object(),
                plan=_Plan(("task-0", "task-1", "task-2")),
                projection=object(),
                candidate=SimpleNamespace(alias="candidate_a"),
                client=_Client(),
                tool_trace_cache=None,
                oracle_factory=oracle_factory,
            )
        )

    assert sorted(closed) == sorted(opened)
    assert "task-2" not in opened


def test_run_pipeline_orders_gates_candidates_aggregation_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    source = SimpleNamespace(
        evaluation_benchmark=SimpleNamespace(path=tmp_path / "benchmark.parquet", content_hash="h"),
        task_ids=("task-0",),
    )

    class Plan:
        candidate_aliases = ("candidate_a", "candidate_b")

    plan = Plan()
    candidates = {
        alias: SimpleNamespace(alias=alias) for alias in plan.candidate_aliases
    }
    config = SimpleNamespace(
        settings=SimpleNamespace(executable=True, held_out_eval=False),
        eval_config_hash="sha256:" + "1" * 64,
        outputs=SimpleNamespace(
            output_dir=tmp_path,
            cache_tool_results=False,
            cache_candidate_responses=False,
        ),
        limits=SimpleNamespace(max_parallel_tasks=1),
        scoring=SimpleNamespace(),
        candidate=lambda alias: candidates[alias],
    )
    monkeypatch.setattr(
        eval_runner,
        "write_resolved_eval_config",
        lambda *_args: events.append("config") or "hash",
    )
    monkeypatch.setattr(
        eval_runner,
        "verify_eval_source",
        lambda *_args, **_kwargs: events.append("verify") or source,
    )
    monkeypatch.setattr(
        eval_runner,
        "write_source_verification_report",
        lambda *_args: events.append("source.report"),
    )
    monkeypatch.setattr(
        eval_runner,
        "evaluate_contamination",
        lambda *_args: events.append("contamination") or plan,
    )
    monkeypatch.setattr(
        eval_runner,
        "write_contamination_report",
        lambda *_args: events.append("contamination.report"),
    )
    monkeypatch.setattr(
        eval_runner,
        "project_published_benchmark",
        lambda *_args, **_kwargs: events.append("projection") or object(),
    )
    monkeypatch.setattr(
        eval_runner,
        "assert_plan_unchanged",
        lambda *_args: events.append("plan.recheck"),
    )

    async def run_tasks(**kwargs: Any) -> tuple[str, ...]:
        alias = kwargs["candidate"].alias
        events.append(f"tasks:{alias}")
        return (f"score:{alias}",)

    monkeypatch.setattr(eval_runner, "_run_candidate_tasks", run_tasks)
    monkeypatch.setattr(
        eval_runner,
        "aggregate_executable_scores",
        lambda **kwargs: events.append(f"aggregate:{kwargs['candidate_alias']}")
        or f"aggregate:{kwargs['candidate_alias']}",
    )
    artifact = object()
    monkeypatch.setattr(
        eval_runner,
        "write_executable_eval_artifacts",
        lambda **_kwargs: events.append("artifacts") or artifact,
    )

    result = asyncio.run(
        eval_runner.run_bfcl_eval(
            config,
            eval_run_id="eval-run",
            client_factory=lambda *_args: _Client(events),
        )
    )

    assert events == [
        "config",
        "verify",
        "source.report",
        "contamination",
        "contamination.report",
        "projection",
        "plan.recheck",
        "tasks:candidate_a",
        "client.close",
        "aggregate:candidate_a",
        "tasks:candidate_b",
        "client.close",
        "aggregate:candidate_b",
        "artifacts",
    ]
    assert result.task_scores == ("score:candidate_a", "score:candidate_b")
    assert result.artifacts is artifact


def _codes(base: type[Exception]) -> set[str]:
    pending = [base]
    found: set[str] = set()
    while pending:
        parent = pending.pop()
        pending.extend(parent.__subclasses__())
        code = getattr(parent, "code", None)
        if isinstance(code, str):
            found.add(code)
    return found


def test_error_taxonomy_covers_statuses_and_structured_exception_codes() -> None:
    assert set(get_args(EpisodeStatus)) == set(TRACE_EPISODE_ATTRIBUTION)
    assert set(get_args(ExecutableEpisodeStatus)) == set(
        EXECUTABLE_EPISODE_ATTRIBUTION
    )
    assert TRACE_NON_CANDIDATE_STOPS == {
        status
        for status in get_args(EpisodeStatus)
        if episode_attribution(status, executable=False) == "infrastructure"
    }
    assert EXECUTABLE_NON_CANDIDATE_STOPS == {
        status
        for status in get_args(ExecutableEpisodeStatus)
        if episode_attribution(status, executable=True) == "infrastructure"
    }
    structured_codes = set().union(
        *(
            _codes(base)
            for base in (
                EvalConfigError,
                SourceVerificationError,
                ContaminationError,
                CandidateClientError,
                ConversationDriverError,
                ExecutableEvalError,
                ExecutableScoringError,
                TraceScoringError,
                EvalArtifactError,
                eval_runner.EvalRunnerError,
            )
        )
    )
    assert structured_codes <= FATAL_EVAL_ERROR_CODES
    assert ERROR_TAXONOMY_HASH.startswith("sha256:")
    assert len(ERROR_TAXONOMY_HASH) == 71


def test_batch_runner_rejects_trace_only_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        eval_runner,
        "write_resolved_eval_config",
        lambda *_args: pytest.fail("trace-only mode must fail before writing"),
    )
    config = SimpleNamespace(
        settings=SimpleNamespace(executable=False, held_out_eval=False),
        outputs=SimpleNamespace(output_dir=tmp_path),
    )
    with pytest.raises(eval_runner.UnsupportedRunnerModeError) as caught:
        asyncio.run(eval_runner.run_bfcl_eval(config))
    assert caught.value.code == "eval_runner_mode_unsupported"
    assert inspect.isclass(eval_runner.BfclEvalRunResult)


def test_default_run_ids_are_unique_and_completed_run_identity_is_reused(
    tmp_path: Path,
) -> None:
    first_config = SimpleNamespace(
        eval_config_hash="sha256:" + "1" * 64,
        outputs=SimpleNamespace(output_dir=tmp_path / "first"),
    )
    second_config = SimpleNamespace(
        eval_config_hash=first_config.eval_config_hash,
        outputs=SimpleNamespace(output_dir=tmp_path / "second"),
    )
    first_config.outputs.output_dir.mkdir()
    second_config.outputs.output_dir.mkdir()

    first = eval_runner._resolve_eval_run_id(first_config, None)
    second = eval_runner._resolve_eval_run_id(second_config, None)
    assert first != second
    report = {
        "eval_run_id": first,
        "eval_config_hash": first_config.eval_config_hash,
    }
    (first_config.outputs.output_dir / "eval_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    assert eval_runner._resolve_eval_run_id(first_config, None) == first
    with pytest.raises(eval_runner.EvalRunnerError, match="conflicts"):
        eval_runner._resolve_eval_run_id(first_config, second)


def test_contamination_violations_count_only_colliding_tasks_that_were_scored() -> None:
    collision = SimpleNamespace(task_ids=("exposed-task",))
    candidate = SimpleNamespace(
        alias="candidate_a",
        definite_collisions=(collision,),
    )

    class Plan:
        candidates = (candidate,)

        def __init__(self, scored: tuple[str, ...]) -> None:
            self.scored = scored

        def evaluation_task_ids(self, alias: str) -> tuple[str, ...]:
            assert alias == candidate.alias
            return self.scored

    assert eval_artifacts._contamination_violations(Plan(("safe-task",))) == 0
    assert (
        eval_artifacts._contamination_violations(
            Plan(("safe-task", "exposed-task"))
        )
        == 1
    )
