"""Tests for `eval/model_eval` direct mode.

Each test pins a behaviour that was a real defect found while validating the
step against live endpoints; the docstrings state what each one prevents.
"""

from __future__ import annotations

import io
import json

import pytest
from omegaconf import OmegaConf

from nemotron.steps.eval.model_eval.runtime import (
    _direct_overrides,
    _ensure_visible_chat_content,
    _is_digest_pinned,
    _is_set,
    _override_value,
    _redact_params,
    _redact_url,
    run_direct,
)


def _cfg(**over):
    base = {
        "dry_run": True,
        "output_dir": over.pop("output_dir", "/tmp/unused"),
        "target": {
            "api_endpoint": {
                "url": "https://host/v1/completions",
                "model_id": "my-model",
                "type": "completions",
                "api_key_name": "ENDPOINT_TOKEN",
            }
        },
        "evaluation": {
            "tasks": [{"name": "adlr_arc_challenge_llama_25_shot"}, {"name": "hellaswag"}],
            "nemo_evaluator_config": {"config": {"params": {}}},
        },
    }
    for k, v in over.items():
        base[k] = v
    return OmegaConf.create(base)


# --- _is_set -----------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "null", "None", "NONE", "", "   "])
def test_is_set_treats_sentinels_as_unset(value):
    """`${oc.env:VAR,null}` resolves to the literal string 'null', which reached
    nemo-evaluator as a model name and produced 'Unrecognized model in None'."""
    assert _is_set(value) is False


@pytest.mark.parametrize("value", ["0", "false", 0, False, "mmlu_prox_en"])
def test_is_set_keeps_real_values(value):
    assert _is_set(value) is True


# --- _direct_overrides -------------------------------------------------------


def test_task_is_forwarded():
    """Dropping config.params.task turned `mmlu_prox_completions` from one
    language into the full 29-language run, which never finishes."""
    out = _direct_overrides({"task": "mmlu_prox_en"})
    assert "config.params.task=mmlu_prox_en" in out


def test_extra_params_are_namespaced():
    out = _direct_overrides({"extra": {"tokenizer": "/t", "tokenizer_backend": "huggingface"}})
    assert "config.params.extra.tokenizer=/t" in out
    assert "config.params.extra.tokenizer_backend=huggingface" in out


def test_nested_adapter_params_are_forwarded_as_top_level_payload_overrides():
    out = _direct_overrides(
        {},
        {"params_to_add": {"chat_template_kwargs": {"enable_thinking": False}}},
    )
    assert (
        "target.api_endpoint.adapter_config.params_to_add."
        "chat_template_kwargs.enable_thinking=false"
    ) in out


def test_chat_defaults_to_visible_content_via_payload_modifier():
    config = {
        "target": {"api_endpoint": {"type": "chat"}},
        "evaluation": {"nemo_evaluator_config": {}},
    }
    _ensure_visible_chat_content(config)
    adapter = config["evaluation"]["nemo_evaluator_config"]["target"]["api_endpoint"]["adapter_config"]
    assert adapter["params_to_add"]["chat_template_kwargs"]["enable_thinking"] is False
    assert "extra_body" not in adapter["params_to_add"]


def test_unset_params_are_omitted():
    out = _direct_overrides({"task": None, "limit_samples": "null", "parallelism": 16})
    assert "task" not in out
    assert "limit_samples" not in out
    assert out == "config.params.parallelism=16"


def test_booleans_render_lowercase():
    assert _override_value(True) == "true"
    assert _override_value(False) == "false"


def test_comma_value_is_rejected():
    """--overrides is comma-joined; a comma inside a value would be parsed as
    another key=value pair and corrupt every following override."""
    with pytest.raises(SystemExit, match="comma"):
        _override_value("a,b")


def test_collection_value_is_rejected():
    with pytest.raises(SystemExit, match="Unsupported|cannot pass"):
        _override_value(["a", "b"])


# --- task selection ----------------------------------------------------------


def test_cli_tasks_are_used_exactly_as_supplied(tmp_path, capsys):
    """Regression: the old intersection silently dropped a requested task that
    was not already in evaluation.tasks, so `-t configured -t new` ran only the
    configured one -- a silent partial run."""
    run_direct(_cfg(output_dir=str(tmp_path)), task_filters=["adlr_arc_challenge_llama_25_shot", "milu_Hindi"])
    summary = json.loads((tmp_path / "summary.dry-run.json").read_text())
    assert set(summary) == {"adlr_arc_challenge_llama_25_shot", "milu_Hindi"}


def test_tasks_default_to_config_when_no_filter(tmp_path):
    run_direct(_cfg(output_dir=str(tmp_path)), task_filters=None)
    summary = json.loads((tmp_path / "summary.dry-run.json").read_text())
    assert set(summary) == {"adlr_arc_challenge_llama_25_shot", "hellaswag"}


def test_task_name_that_escapes_the_output_dir_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="directory names"):
        run_direct(_cfg(output_dir=str(tmp_path)), task_filters=["../escape"])


# --- endpoint validation -----------------------------------------------------


def test_missing_endpoint_fails_fast(tmp_path):
    cfg = _cfg(output_dir=str(tmp_path))
    cfg.target.api_endpoint.url = "null"  # the resolved-null sentinel
    with pytest.raises(SystemExit, match="direct mode needs"):
        run_direct(cfg, task_filters=["hellaswag"])


def test_bad_endpoint_type_is_rejected(tmp_path):
    cfg = _cfg(output_dir=str(tmp_path))
    cfg.target.api_endpoint.type = "completion"  # note: missing trailing 's'
    with pytest.raises(SystemExit, match="chat.*completions"):
        run_direct(cfg, task_filters=["hellaswag"])


# --- result contamination ----------------------------------------------------


def test_non_empty_task_dir_is_refused(tmp_path):
    """eval_factory_metrics.json accumulates across runs in a reused directory,
    so a stale failure's status codes blend into a fresh run's."""
    stale = tmp_path / "hellaswag"
    stale.mkdir(parents=True)
    (stale / "eval_factory_metrics.json").write_text("{}")
    with pytest.raises(SystemExit, match="not empty"):
        run_direct(_cfg(output_dir=str(tmp_path)), task_filters=["hellaswag"])


def test_overwrite_replaces_rather_than_merges(tmp_path):
    """`overwrite` must CLEAR the directory. Merely bypassing the guard would
    leave the stale eval_factory_metrics.json the guard exists to prevent."""
    stale = tmp_path / "hellaswag"
    stale.mkdir(parents=True)
    (stale / "eval_factory_metrics.json").write_text('{"stale": true}')
    cfg = _cfg(output_dir=str(tmp_path))
    cfg.overwrite = True
    cfg.dry_run = False
    monkey = pytest.MonkeyPatch()
    monkey.setattr("nemotron.steps.eval.model_eval.runtime.subprocess.Popen", lambda cmd, env=None, **kw: _fake_proc())
    try:
        run_direct(cfg, task_filters=["hellaswag"])
    finally:
        monkey.undo()
    assert not (stale / "eval_factory_metrics.json").exists(), "stale results survived overwrite"


def test_dry_run_never_deletes_existing_results(tmp_path):
    """A dry run must be safe to point at a directory holding real results."""
    stale = tmp_path / "hellaswag"
    stale.mkdir(parents=True)
    (stale / "eval_factory_metrics.json").write_text('{"precious": true}')
    cfg = _cfg(output_dir=str(tmp_path))
    cfg.overwrite = True  # dry_run stays True
    run_direct(cfg, task_filters=["hellaswag"])
    assert (stale / "eval_factory_metrics.json").exists(), "dry run deleted real results"


# --- subprocess path: credential forwarding and log persistence --------------


class _FakeProc:
    """Stand-in for subprocess.Popen: output is streamed, not buffered."""

    def __init__(self, returncode=0, stdout=b"harness said hello\n"):
        self._returncode = returncode
        self.stdout = io.BytesIO(stdout)  # BytesIO provides read1()

    def wait(self):
        return self._returncode


def _fake_proc(returncode=0, stdout=b"harness said hello\n"):
    return _FakeProc(returncode, stdout)


def _live_cfg(tmp_path):
    cfg = _cfg(output_dir=str(tmp_path))
    cfg.dry_run = False
    return cfg


def test_endpoint_credential_is_also_exported_as_openai_api_key(tmp_path, monkeypatch):
    """The sovereign container's MILU tasks run lm-eval `local-completions`,
    which reads the bearer token from OPENAI_API_KEY and ignores
    --api_key_name. Without this every request 401s."""
    monkeypatch.setenv("ENDPOINT_TOKEN", "s3cret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = {}

    def fake_run(cmd, env=None, **kw):
        seen["env"] = env
        return _fake_proc()

    monkeypatch.setattr("nemotron.steps.eval.model_eval.runtime.subprocess.Popen", fake_run)
    run_direct(_live_cfg(tmp_path), task_filters=["hellaswag"])
    assert seen["env"]["OPENAI_API_KEY"] == "s3cret"
    assert seen["env"]["ENDPOINT_TOKEN"] == "s3cret"


def test_existing_openai_api_key_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setenv("ENDPOINT_TOKEN", "s3cret")
    monkeypatch.setenv("OPENAI_API_KEY", "preexisting")
    seen = {}
    monkeypatch.setattr(
        "nemotron.steps.eval.model_eval.runtime.subprocess.Popen",
        lambda cmd, env=None, **kw: (seen.update(env=env), _fake_proc())[1],
    )
    run_direct(_live_cfg(tmp_path), task_filters=["hellaswag"])
    assert seen["env"]["OPENAI_API_KEY"] == "preexisting"


def test_harness_output_is_persisted_next_to_results(tmp_path, monkeypatch):
    """Pod logs are reaped and summary.json records only failed(N); without a
    persisted log a 401 is indistinguishable from a timeout."""
    monkeypatch.setattr(
        "nemotron.steps.eval.model_eval.runtime.subprocess.Popen",
        lambda cmd, env=None, **kw: _fake_proc(stdout=b"TimeoutError: boom\n"),
    )
    run_direct(_live_cfg(tmp_path), task_filters=["hellaswag"])
    log = tmp_path / "hellaswag" / "harness.log"
    assert log.exists()
    assert b"TimeoutError" in log.read_bytes()


def test_failure_is_recorded_in_summary_with_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nemotron.steps.eval.model_eval.runtime.subprocess.Popen",
        lambda cmd, env=None, **kw: _fake_proc(returncode=1, stdout=b"nope\n"),
    )
    # a failed task must fail the job, not exit 0 with a bad summary
    with pytest.raises(SystemExit):
        run_direct(_live_cfg(tmp_path), task_filters=["hellaswag"])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["hellaswag"] == "failed(1)"
    assert (tmp_path / "hellaswag" / "harness.log").exists()


# --- destructive-path guards ------------------------------------------------


@pytest.mark.parametrize("name", ["", "   ", ".", "..", "a/b"])
def test_unsafe_task_names_are_rejected(tmp_path, name):
    """An empty name resolves to output_dir itself; with overwrite=true the
    rmtree would then delete the entire output root."""
    with pytest.raises(SystemExit, match="directory names"):
        run_direct(_cfg(output_dir=str(tmp_path)), task_filters=[name])


def test_dry_run_does_not_clobber_a_real_summary(tmp_path):
    real = tmp_path / "summary.json"
    real.write_text('{"hellaswag": "ok"}')
    run_direct(_cfg(output_dir=str(tmp_path)), task_filters=["hellaswag"])
    assert json.loads(real.read_text())["hellaswag"] == "ok", "dry run overwrote real summary"
    assert (tmp_path / "summary.dry-run.json").exists()


# --- task-level configuration ----------------------------------------------


def test_task_level_params_override_globals(tmp_path):
    """A per-task nemo_evaluator_config must win over the suite-wide default;
    reducing tasks to names silently dropped settings like top_p."""
    cfg = _cfg(output_dir=str(tmp_path))
    cfg.evaluation.nemo_evaluator_config.config.params = {"top_p": 1.0, "parallelism": 4}
    cfg.evaluation.tasks = [{"name": "adlr_mmlu", "nemo_evaluator_config": {"config": {"params": {"top_p": 0.0}}}}]
    from nemotron.steps.eval.model_eval.runtime import _merged_params

    merged = _merged_params(
        {"top_p": 1.0, "parallelism": 4},
        {"name": "adlr_mmlu", "nemo_evaluator_config": {"config": {"params": {"top_p": 0.0}}}},
    )
    assert merged["top_p"] == 0.0, "task-level top_p was discarded"
    assert merged["parallelism"] == 4, "global param lost"


def test_task_level_extra_is_merged_not_replaced(tmp_path):
    from nemotron.steps.eval.model_eval.runtime import _merged_params

    merged = _merged_params(
        {"extra": {"tokenizer": "/global", "tokenizer_backend": "huggingface"}},
        {"nemo_evaluator_config": {"config": {"params": {"extra": {"tokenizer": "/task"}}}}},
    )
    assert merged["extra"]["tokenizer"] == "/task"
    assert merged["extra"]["tokenizer_backend"] == "huggingface"


# --- preflight: no side effects before the whole plan validates -------------


def test_a_later_bad_task_aborts_before_any_task_runs(tmp_path, monkeypatch):
    """Validation used to happen inside the execution loop, so a bad task N
    aborted only after task 1 had already consumed GPU time."""
    calls = []
    monkeypatch.setattr(
        "nemotron.steps.eval.model_eval.runtime.subprocess.Popen",
        lambda cmd, env=None, **kw: (calls.append(cmd), _fake_proc())[1],
    )
    cfg = _cfg(output_dir=str(tmp_path))
    cfg.dry_run = False
    # second task's directory is occupied -> the whole run must abort up front
    stale = tmp_path / "hellaswag"
    stale.mkdir(parents=True)
    (stale / "eval_factory_metrics.json").write_text("{}")
    with pytest.raises(SystemExit, match="not empty"):
        run_direct(cfg, task_filters=["adlr_arc_challenge_llama_25_shot", "hellaswag"])
    assert calls == [], "a task ran before the plan was fully validated"
    assert not (tmp_path / "adlr_arc_challenge_llama_25_shot").exists(), "directory created before validation"


def test_bad_override_on_a_later_task_aborts_before_deleting(tmp_path, monkeypatch):
    """An unserialisable override on task 2 must not leave task 1's directory
    already deleted."""
    monkeypatch.setattr(
        "nemotron.steps.eval.model_eval.runtime.subprocess.Popen",
        lambda cmd, env=None, **kw: _fake_proc(),
    )
    first = tmp_path / "adlr_arc_challenge_llama_25_shot"
    first.mkdir(parents=True)
    (first / "old.json").write_text("{}")
    cfg = _cfg(output_dir=str(tmp_path))
    cfg.dry_run = False
    cfg.overwrite = True
    # comma in a value is rejected by the override serialiser
    cfg.evaluation.nemo_evaluator_config.config.params = {"task": "a,b"}
    with pytest.raises(SystemExit, match="comma"):
        run_direct(cfg, task_filters=["adlr_arc_challenge_llama_25_shot", "hellaswag"])
    assert (first / "old.json").exists(), "deleted results before the plan validated"


# --- alias and containment ---------------------------------------------------


def test_symlinked_task_directory_is_rejected(tmp_path):
    """A symlinked task dir can point anywhere; with overwrite=true the rmtree
    would follow it out of the output root."""
    victim = tmp_path / "real_results"
    victim.mkdir()
    (victim / "precious.json").write_text("{}")
    out = tmp_path / "out"
    out.mkdir()
    (out / "hellaswag").symlink_to(victim, target_is_directory=True)
    cfg = _cfg(output_dir=str(out))
    cfg.dry_run = False
    cfg.overwrite = True
    with pytest.raises(SystemExit, match="symlink"):
        run_direct(cfg, task_filters=["hellaswag"])
    assert (victim / "precious.json").exists(), "followed a symlink out of the output root"


def test_two_tasks_resolving_to_one_directory_are_rejected(tmp_path):
    """Two names resolving to the same directory would interleave results while
    summary.json reported both as separate successes."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "alias").symlink_to(out / "hellaswag", target_is_directory=True)
    (out / "hellaswag").mkdir()
    cfg = _cfg(output_dir=str(out))
    cfg.dry_run = False
    with pytest.raises(SystemExit, match="symlink|resolve"):
        run_direct(cfg, task_filters=["hellaswag", "alias"])


def test_failed_preflight_does_not_create_the_output_root(tmp_path):
    """A plan that never runs must not leave a new output root behind."""
    missing = tmp_path / "does_not_exist_yet"
    cfg = _cfg(output_dir=str(missing))
    cfg.dry_run = False
    cfg.evaluation.nemo_evaluator_config.config.params = {"task": "a,b"}  # unserialisable
    with pytest.raises(SystemExit, match="comma"):
        run_direct(cfg, task_filters=["hellaswag"])
    assert not missing.exists(), "output root created despite failed preflight"


def test_output_dir_under_a_symlinked_parent_is_accepted(tmp_path):
    """Legitimate path, and it must not be mistaken for an escape: the root and
    the task path have to be normalised the same way."""
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_parent, target_is_directory=True)
    out = link / "results_not_yet_created"  # nonexistent, under a symlink
    run_direct(_cfg(output_dir=str(out)), task_filters=["hellaswag"])
    assert json.loads((out / "summary.dry-run.json").read_text())["hellaswag"] == "dry-run"


def test_output_dir_containing_dotdot_is_accepted(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    out = nested / ".." / "results"
    run_direct(_cfg(output_dir=str(out)), task_filters=["hellaswag"])
    assert (tmp_path / "a" / "results" / "summary.dry-run.json").exists()


def test_unknown_top_level_params_are_rejected():
    """Silently dropping an unsupported key let a task run with different
    generation settings than the config asked for."""
    with pytest.raises(SystemExit, match="unsupported.*params"):
        _direct_overrides({"top_k": 5})


def test_harness_specific_args_go_under_extra():
    out = _direct_overrides({"extra": {"num_fewshot": 5}})
    assert "config.params.extra.num_fewshot=5" in out


# --- run_manifest.json -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://host/v1/completions", "https://host/v1/completions"),
        ("https://host:8000/v1/completions?x=1", "https://host:8000/v1/completions"),
        ("https://host/v1?api_key=sekrit", "https://host/v1"),
    ],
)
def test_redact_url_strips_query(raw, expected):
    """The manifest is meant to be published alongside results, and tokens turn
    up in query strings."""
    assert _redact_url(raw) == expected


def test_redact_url_removes_userinfo():
    out = _redact_url("https://user:sekrit@host/v1/completions")
    assert "sekrit" not in out and "user" not in out
    assert out.startswith("https://host/v1/completions")
    # Say that something was removed rather than silently reporting a different URL.
    assert "redacted" in out


def test_manifest_is_written_with_merged_params(tmp_path):
    out = tmp_path / "results"
    cfg = _cfg(output_dir=str(out))
    cfg.run = {"env": {"container_image": f"example/harness@sha256:{'a' * 64}"}}
    cfg.evaluation.nemo_evaluator_config.config.params = {"parallelism": 16, "request_timeout": 3600}
    cfg.evaluation.tasks = [{"name": "hellaswag", "nemo_evaluator_config": {"config": {"params": {"parallelism": 1}}}}]
    run_direct(cfg)

    # A dry run must not overwrite a real manifest either.
    manifest = json.loads((out / "run_manifest.dry-run.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["model"]["id"] == "my-model"
    assert manifest["model"]["endpoint_type"] == "completions"
    # The NAME of the credential variable, never a value.
    assert manifest["model"]["api_key_name"] == "ENDPOINT_TOKEN"
    assert manifest["harness"]["image_pinned_by_digest"] is True
    # Per-task settings must win over the suite-wide default in the record too,
    # otherwise the manifest reports parameters the task did not actually use.
    assert manifest["tasks"]["hellaswag"]["params"] == {"parallelism": 1, "request_timeout": 3600}
    assert manifest["summary_path"].endswith("summary.dry-run.json")


def test_manifest_flags_an_unpinned_image(tmp_path):
    """A mutable tag makes the run unreproducible; the record has to say so."""
    out = tmp_path / "results"
    cfg = _cfg(output_dir=str(out))
    cfg.run = {"env": {"container_image": "example/harness:latest"}}
    run_direct(cfg, task_filters=["hellaswag"])
    manifest = json.loads((out / "run_manifest.dry-run.json").read_text())
    assert manifest["harness"]["image_pinned_by_digest"] is False


def test_manifest_never_contains_the_token(tmp_path, monkeypatch):
    monkeypatch.setenv("ENDPOINT_TOKEN", "super-secret-value")
    out = tmp_path / "results"
    cfg = _cfg(output_dir=str(out))
    cfg.target.api_endpoint.url = "https://u:super-secret-value@host/v1/completions"
    run_direct(cfg, task_filters=["hellaswag"])
    assert "super-secret-value" not in (out / "run_manifest.dry-run.json").read_text()


def test_echoed_command_is_redacted(tmp_path, capsys):
    """The command line is printed to the job log; a token in the URL's
    userinfo would land there verbatim."""
    out = tmp_path / "results"
    cfg = _cfg(output_dir=str(out))
    cfg.target.api_endpoint.url = "https://u:super-secret-value@host/v1/completions"
    run_direct(cfg, task_filters=["hellaswag"])
    assert "super-secret-value" not in capsys.readouterr().out


# --- digest pinning ----------------------------------------------------------


@pytest.mark.parametrize(
    "image",
    [
        # A real digest is sha256: + exactly 64 hex characters.
        "img@sha256:" + "a" * 64,
        "reg.example.com/ns/img@sha256:" + "0123456789abcdef" * 4,
    ],
)
def test_real_digests_count_as_pinned(image):
    assert _is_digest_pinned(image) is True


@pytest.mark.parametrize(
    "image",
    [
        None,
        "",
        "img:latest",
        # Substring-matching '@sha256:' certified these as pinned, which is
        # worse than no check: the manifest would vouch for an unpinned run.
        "img@sha256:abc",
        "img@sha256:" + "a" * 63,
        "img@sha256:" + "a" * 65,
        "img@sha256:" + "g" * 64,
        "img@sha256:" + "A" * 64,
        "img@sha256:" + "a" * 64 + "-suffix",
        "img@md5:" + "a" * 64,
    ],
)
def test_fake_or_absent_digests_are_not_pinned(image):
    assert _is_digest_pinned(image) is False


# --- credential-valued params ------------------------------------------------


def test_credential_params_are_redacted():
    out = _redact_params({"extra": {"api_key": "sekrit", "auth_token": "t", "password": "p"}})
    assert out["extra"] == {"api_key": "***", "auth_token": "***", "password": "***"}


def test_tokenizer_is_not_mistaken_for_a_credential():
    """`tokenizer` contains 'token'; redacting it would destroy the single most
    important parameter for a tokenizer-extended checkpoint."""
    params = {"extra": {"tokenizer": "/mnt/tok", "tokenizer_backend": "huggingface"}}
    assert _redact_params(params) == params


def test_api_key_name_is_a_name_not_a_secret():
    """It holds the NAME of the variable; redacting it would hide which
    credential a run used."""
    assert _redact_params({"extra": {"api_key_name": "ENDPOINT_TOKEN"}})["extra"]["api_key_name"] == "ENDPOINT_TOKEN"


def test_credential_param_absent_from_manifest_command_and_log(tmp_path, capsys):
    """A credential passed via `extra:` lands inside one `--overrides` argv
    element, so element-wise comparison would miss it."""
    out = tmp_path / "results"
    cfg = _cfg(output_dir=str(out))
    cfg.evaluation.nemo_evaluator_config.config.params = {"extra": {"api_key": "sekrit-value"}}
    run_direct(cfg, task_filters=["hellaswag"])
    assert "sekrit-value" not in capsys.readouterr().out
    text = (out / "run_manifest.dry-run.json").read_text()
    assert "sekrit-value" not in text
    assert "config.params.extra.api_key=***" in text


# --- code provenance ---------------------------------------------------------


def test_manifest_records_the_code_that_ran(tmp_path):
    """Harness versions say what computed the score; this says what decided
    the score's inputs."""
    out = tmp_path / "results"
    run_direct(_cfg(output_dir=str(out)), task_filters=["hellaswag"])
    code = json.loads((out / "run_manifest.dry-run.json").read_text())["code"]
    # Always present, null when unavailable -- so a reader can tell "not a git
    # checkout" from "provenance silently omitted".
    assert set(code) == {"nemotron_version", "git_sha", "git_dirty"}
    if code["git_sha"] is not None:
        assert len(code["git_sha"]) == 40
        assert isinstance(code["git_dirty"], bool)


# --- adapter_config passthrough ----------------------------------------------


def test_adapter_config_is_forwarded():
    """Dropping this namespace put a hard ceiling on direct mode: without
    `use_caching` the harness buffers every request and response in memory, and
    a task past ~30k requests dies client-side while the endpoint is still
    answering 200 to everything."""
    out = _direct_overrides({}, {"use_caching": True, "output_dir": "/results/adapter"})
    assert "target.api_endpoint.adapter_config.use_caching=true" in out
    assert "target.api_endpoint.adapter_config.output_dir=/results/adapter" in out


def test_adapter_keys_are_not_allowlisted():
    """The interceptor set is open-ended and version dependent, and an unknown
    key here is inert rather than silently changing how the model is scored."""
    out = _direct_overrides({}, {"some_future_interceptor": "on"})
    assert "target.api_endpoint.adapter_config.some_future_interceptor=on" in out


def test_unset_adapter_values_are_omitted():
    assert _direct_overrides({}, {"output_dir": None, "use_caching": False}) == (
        "target.api_endpoint.adapter_config.use_caching=false"
    )


def test_adapter_values_get_the_same_comma_guard():
    with pytest.raises(SystemExit, match="comma"):
        _direct_overrides({}, {"output_dir": "/a,/b"})


def test_per_task_adapter_overrides_the_suite_default(tmp_path):
    out = tmp_path / "results"
    cfg = _cfg(output_dir=str(out))
    cfg.evaluation.nemo_evaluator_config.target = {
        "api_endpoint": {"adapter_config": {"use_caching": True, "max_logged_requests": 10}}
    }
    cfg.evaluation.tasks = [
        {
            "name": "hellaswag",
            "nemo_evaluator_config": {"target": {"api_endpoint": {"adapter_config": {"max_logged_requests": 99}}}},
        }
    ]
    run_direct(cfg)
    manifest = json.loads((out / "run_manifest.dry-run.json").read_text())
    adapter = manifest["tasks"]["hellaswag"]["adapter_config"]
    assert adapter["use_caching"] is True
    assert adapter["max_logged_requests"] == 99


def test_adapter_output_dir_defaults_per_task(tmp_path):
    """Two tasks in one run must not share an adapter cache directory."""
    out = tmp_path / "results"
    cfg = _cfg(output_dir=str(out))
    cfg.evaluation.nemo_evaluator_config.target = {
        "api_endpoint": {"adapter_config": {"use_caching": True, "output_dir": None}}
    }
    run_direct(cfg, task_filters=["hellaswag", "adlr_arc_challenge_llama_25_shot"])
    manifest = json.loads((out / "run_manifest.dry-run.json").read_text())
    dirs = {t: v["adapter_config"]["output_dir"] for t, v in manifest["tasks"].items()}
    assert len(set(dirs.values())) == 2
    assert dirs["hellaswag"] == str(out / "hellaswag" / "adapter")


def test_adapter_endpoint_type_follows_the_endpoint(tmp_path):
    """The adapter's own endpoint_type defaults to `chat`. Disagreeing with
    target.api_endpoint.type is inert today, but it is a trap the moment an
    interceptor branches on it."""
    out = tmp_path / "results"
    cfg = _cfg(output_dir=str(out))
    cfg.target.api_endpoint.type = "completions"
    cfg.evaluation.nemo_evaluator_config.target = {"api_endpoint": {"adapter_config": {"use_caching": True}}}
    run_direct(cfg, task_filters=["hellaswag"])
    manifest = json.loads((out / "run_manifest.dry-run.json").read_text())
    assert manifest["tasks"]["hellaswag"]["adapter_config"]["endpoint_type"] == "completions"


def test_explicit_adapter_endpoint_type_is_kept(tmp_path):
    out = tmp_path / "results"
    cfg = _cfg(output_dir=str(out))
    cfg.target.api_endpoint.type = "completions"
    cfg.evaluation.nemo_evaluator_config.target = {
        "api_endpoint": {"adapter_config": {"use_caching": True, "endpoint_type": "chat"}}
    }
    run_direct(cfg, task_filters=["hellaswag"])
    manifest = json.loads((out / "run_manifest.dry-run.json").read_text())
    assert manifest["tasks"]["hellaswag"]["adapter_config"]["endpoint_type"] == "chat"


# --- concurrent runs on one output_dir ---------------------------------------


def _real_run(cfg):
    """A non-dry run whose subprocess is stubbed out."""
    cfg.dry_run = False
    return cfg


def test_a_second_run_cannot_claim_a_live_output_dir(tmp_path, monkeypatch):
    """The not-empty check only catches a directory that ALREADY has results.
    Two submissions racing on the same fresh directory both passed preflight,
    interleaved their writes, and the summary reported whichever finished last."""
    out = tmp_path / "results"
    out.mkdir()
    (out / ".nemotron-eval.lock").write_text('{"pid": 999, "host": "other"}')

    cfg = _real_run(_cfg(output_dir=str(out)))
    with pytest.raises(SystemExit, match="claimed by another run"):
        run_direct(cfg, task_filters=["hellaswag"])


def test_overwrite_does_not_override_a_live_claim(tmp_path):
    """`overwrite` is about replacing stale results, not about racing a live
    job -- silently proceeding would be the exact corruption it prevents."""
    out = tmp_path / "results"
    out.mkdir()
    (out / ".nemotron-eval.lock").write_text("{}")

    cfg = _real_run(_cfg(output_dir=str(out)))
    cfg.overwrite = True
    with pytest.raises(SystemExit, match="does not override a live claim"):
        run_direct(cfg, task_filters=["hellaswag"])


def test_a_dry_run_takes_no_claim_and_is_not_blocked(tmp_path):
    """A dry run writes nothing that could collide, so it must neither take a
    claim nor be blocked by one."""
    out = tmp_path / "results"
    out.mkdir()
    (out / ".nemotron-eval.lock").write_text("{}")

    run_direct(_cfg(output_dir=str(out)), task_filters=["hellaswag"])
    assert (out / "summary.dry-run.json").exists()


def test_the_claim_is_released_when_the_run_finishes(tmp_path, monkeypatch):
    """Otherwise every re-run of a completed job needs a manual cleanup."""
    import subprocess

    class _Done:
        stdout = io.BytesIO(b"")

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _Done())
    out = tmp_path / "results"
    cfg = _real_run(_cfg(output_dir=str(out)))
    run_direct(cfg, task_filters=["hellaswag"])
    assert not (out / ".nemotron-eval.lock").exists()
    # ... and the directory is immediately reusable with overwrite.
    cfg2 = _real_run(_cfg(output_dir=str(out)))
    cfg2.overwrite = True
    run_direct(cfg2, task_filters=["hellaswag"])


# --- missing tokenizer -------------------------------------------------------


def test_warns_when_a_completions_run_has_no_tokenizer(tmp_path, capsys):
    """lm-eval falls back to loading the served-model-name as an HF repo and
    dies with RepositoryNotFoundError deep in a Hub traceback, with nothing
    pointing back at the unset tokenizer."""
    cfg = _cfg(output_dir=str(tmp_path / "r"))
    cfg.target.api_endpoint.type = "completions"
    run_direct(cfg, task_filters=["hellaswag"])
    err = capsys.readouterr().err
    assert "no tokenizer configured" in err
    assert "EVAL_TOKENIZER" in err
    # Name the thing it will wrongly try to load.
    assert "my-model" in err


def test_no_warning_when_a_tokenizer_is_set(tmp_path, capsys):
    cfg = _cfg(output_dir=str(tmp_path / "r"))
    cfg.evaluation.nemo_evaluator_config.config.params = {"extra": {"tokenizer": "/mnt/tok"}}
    run_direct(cfg, task_filters=["hellaswag"])
    assert "no tokenizer configured" not in capsys.readouterr().err


def test_the_null_sentinel_counts_as_unset(tmp_path, capsys):
    """`${oc.env:EVAL_TOKENIZER,null}` resolves to the STRING 'null'."""
    cfg = _cfg(output_dir=str(tmp_path / "r"))
    cfg.evaluation.nemo_evaluator_config.config.params = {"extra": {"tokenizer": "null"}}
    run_direct(cfg, task_filters=["hellaswag"])
    assert "no tokenizer configured" in capsys.readouterr().err


def test_chat_endpoints_are_warned_too(tmp_path, capsys):
    """`local-chat-completions` loads a tokenizer as well -- ifeval,
    mmlu_instruct and gsm8k_cot_instruct all ship `extra.tokenizer`. Gating
    this on `type == completions` meant it could never fire on an instruct
    run, which is exactly where it was needed."""
    cfg = _cfg(output_dir=str(tmp_path / "r"))
    cfg.target.api_endpoint.type = "chat"
    run_direct(cfg, task_filters=["ifeval"])
    err = capsys.readouterr().err
    assert "no tokenizer configured" in err
    assert "chat endpoint" in err


# --- failures.txt ------------------------------------------------------------


def test_failure_tails_are_collected_into_one_file(tmp_path, monkeypatch):
    """On a cluster whose job logs are unreadable after termination, a cause
    buried in a per-task harness.log costs an extra job to reach."""
    import subprocess

    class _Failing:
        def __init__(self):
            self.stdout = io.BytesIO(b"Tasks were not found: mmlu_prox_vi\nTraceback...\n")

        def wait(self):
            return 1

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _Failing())
    out = tmp_path / "results"
    cfg = _cfg(output_dir=str(out))
    cfg.dry_run = False
    with pytest.raises(SystemExit):
        run_direct(cfg, task_filters=["mmlu_prox_completions"])

    text = (out / "failures.txt").read_text()
    assert "===== mmlu_prox_completions =====" in text
    assert "Tasks were not found: mmlu_prox_vi" in text


def test_no_failures_file_when_everything_passes(tmp_path, monkeypatch):
    import subprocess

    class _Ok:
        stdout = io.BytesIO(b"fine\n")

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _Ok())
    out = tmp_path / "results"
    cfg = _cfg(output_dir=str(out))
    cfg.dry_run = False
    run_direct(cfg, task_filters=["hellaswag"])
    assert not (out / "failures.txt").exists()


# --- claim before destroy ----------------------------------------------------


def test_a_live_claim_blocks_before_any_deletion(tmp_path):
    """A resubmit with overwrite=true used to rmtree the first run's
    IN-PROGRESS results and only then discover that run held the lock -- the
    destruction happened before the check meant to prevent it."""
    out = tmp_path / "results"
    task_dir = out / "hellaswag"
    task_dir.mkdir(parents=True)
    (task_dir / "partial.jsonl").write_text("run A is still writing this")
    (out / ".nemotron-eval.lock").write_text('{"pid": 999, "host": "other"}')

    cfg = _cfg(output_dir=str(out))
    cfg.dry_run = False
    cfg.overwrite = True
    with pytest.raises(SystemExit, match="does not override a live claim"):
        run_direct(cfg, task_filters=["hellaswag"])

    assert (task_dir / "partial.jsonl").read_text() == "run A is still writing this"


def test_overwrite_still_clears_when_no_one_holds_the_claim(tmp_path, monkeypatch):
    import subprocess

    class _Ok:
        stdout = io.BytesIO(b"")

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _Ok())
    out = tmp_path / "results"
    task_dir = out / "hellaswag"
    task_dir.mkdir(parents=True)
    (task_dir / "stale.jsonl").write_text("old")
    cfg = _cfg(output_dir=str(out))
    cfg.dry_run = False
    cfg.overwrite = True
    run_direct(cfg, task_filters=["hellaswag"])
    assert not (task_dir / "stale.jsonl").exists()


# --- adapter credentials -----------------------------------------------------


def test_a_credential_in_adapter_config_is_scrubbed(tmp_path, capsys):
    """`_secret_values` scanned params only, so a token in adapter_config
    reached the recorded command and the job log intact."""
    out = tmp_path / "results"
    cfg = _cfg(output_dir=str(out))
    cfg.evaluation.nemo_evaluator_config.target = {
        "api_endpoint": {"adapter_config": {"use_caching": True, "api_key": "adapter-secret"}}
    }
    run_direct(cfg, task_filters=["hellaswag"])
    assert "adapter-secret" not in capsys.readouterr().out
    assert "adapter-secret" not in (out / "run_manifest.dry-run.json").read_text()


# --- a file where a directory belongs ----------------------------------------


def test_a_file_occupying_a_task_path_exits_cleanly(tmp_path):
    """iterdir() would raise NotADirectoryError and bypass the clean SystemExit
    every other preflight failure uses."""
    out = tmp_path / "results"
    out.mkdir()
    (out / "hellaswag").write_text("not a directory")
    with pytest.raises(SystemExit, match="not a directory"):
        run_direct(_cfg(output_dir=str(out)), task_filters=["hellaswag"])


# --- lock and root-level artifact lifecycle ----------------------------------


def test_the_claim_is_released_when_a_task_crashes(tmp_path, monkeypatch):
    """Releasing only on the normal path meant a Popen failure left the claim
    behind and blocked every retry until someone deleted it by hand."""
    import subprocess

    def _explode(*a, **k):
        raise OSError("cannot spawn")

    monkeypatch.setattr(subprocess, "Popen", _explode)
    out = tmp_path / "results"
    cfg = _cfg(output_dir=str(out))
    cfg.dry_run = False
    with pytest.raises(OSError, match="cannot spawn"):
        run_direct(cfg, task_filters=["hellaswag"])
    assert not (out / ".nemotron-eval.lock").exists()


def test_a_stale_failures_file_is_cleared_by_a_clean_rerun(tmp_path, monkeypatch):
    """A previous run's failures.txt beside a clean summary.json reads as a
    failure that did not happen."""
    import subprocess

    class _Ok:
        stdout = io.BytesIO(b"")

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _Ok())
    out = tmp_path / "results"
    out.mkdir()
    (out / "failures.txt").write_text("===== hellaswag =====\nolder run blew up")

    cfg = _cfg(output_dir=str(out))
    cfg.dry_run = False
    run_direct(cfg, task_filters=["hellaswag"])
    assert not (out / "failures.txt").exists()
    assert json.loads((out / "summary.json").read_text())["hellaswag"] == "ok"
