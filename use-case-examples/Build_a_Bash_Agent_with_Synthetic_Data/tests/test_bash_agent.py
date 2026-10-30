"""CPU-only tests for the tutorial's LangGraph agent runtime."""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))

from bash_agent.bash import Bash  # noqa: E402
from bash_agent.commands import (  # noqa: E402
    CommandValidationError,
    LangGraphInvocation,
    TEMPLATE_IDS,
    invocation_from_argv,
    invocation_from_payload,
)
from bash_agent.config import Config  # noqa: E402
from bash_agent.helpers import (  # noqa: E402
    Messages,
    OpenAILLM,
    _model_load_settings,
    parse_model_tool_calls,
)
from bash_agent.main_hf import (  # noqa: E402
    config_from_args,
    execute_with_confirmation,
    main,
    parse_args,
)


class CommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_current_templates_are_explicit(self) -> None:
        self.assertEqual(
            TEMPLATE_IDS,
            {
                "agent-python",
                "deep-agent-js",
                "deep-agent-python",
                "new-langgraph-project-js",
                "new-langgraph-project-python",
            },
        )

    def test_structured_payloads_become_immutable_canonical_argv(self) -> None:
        cases = [
            (
                {"command": "new", "template": "agent-python", "path": "my-agent"},
                ("langgraph", "new", "my-agent", "--template", "agent-python"),
            ),
            (
                {"command": "dev", "port": 2024, "no_browser": True},
                ("langgraph", "dev", "--port", "2024", "--no-browser"),
            ),
            (
                {"command": "up", "port": 8123, "watch": True, "wait": True},
                (
                    "langgraph",
                    "up",
                    "--port",
                    "8123",
                    "--watch",
                    "--wait",
                ),
            ),
            (
                {"command": "build", "tag": "registry.local/my-app:v1"},
                ("langgraph", "build", "-t", "registry.local/my-app:v1"),
            ),
            (
                {"command": "dockerfile", "output_path": "docker/Dockerfile"},
                ("langgraph", "dockerfile", "docker/Dockerfile"),
            ),
        ]
        for payload, expected in cases:
            with self.subTest(payload=payload):
                invocation = invocation_from_payload(payload, self.root)
                self.assertEqual(invocation.argv, expected)
                self.assertIsInstance(invocation.argv, tuple)
                with self.assertRaises(FrozenInstanceError):
                    invocation.argv = ("langgraph", "dev")

    def test_null_irrelevant_training_fields_are_allowed(self) -> None:
        invocation = invocation_from_payload(
            {
                "command": "build",
                "template": None,
                "path": None,
                "port": None,
                "no_browser": None,
                "watch": None,
                "wait": None,
                "tag": "my-app:v1",
                "output_path": None,
            },
            self.root,
        )
        self.assertEqual(invocation.argv, ("langgraph", "build", "-t", "my-app:v1"))

    def test_dockerfile_save_path_is_positional(self) -> None:
        invocation = invocation_from_argv(
            ["langgraph", "dockerfile", "Dockerfile.custom"], self.root
        )
        self.assertEqual(invocation.argv, ("langgraph", "dockerfile", "Dockerfile.custom"))
        with self.assertRaises(CommandValidationError):
            invocation_from_argv(["langgraph", "dockerfile", "-o", "Dockerfile.custom"], self.root)

    def test_argv_is_normalized_after_validation(self) -> None:
        invocation = invocation_from_argv(
            ["langgraph", "new", "--template", "deep-agent-python", "project"],
            self.root,
        )
        self.assertEqual(
            invocation.argv,
            ("langgraph", "new", "project", "--template", "deep-agent-python"),
        )

    def test_rejects_unknown_commands_fields_and_old_templates(self) -> None:
        bad_payloads = [
            {"command": "deploy"},
            {"command": "build", "tag": "v1", "port": 8123},
            {"command": "build", "tag": "v1", "surprise": None},
            {"command": "new", "template": "react-agent", "path": "app"},
            {"command": "new", "template": "agent-python", "path": None},
            {"command": "build", "tag": "ok; touch pwned"},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload), self.assertRaises(CommandValidationError):
                invocation_from_payload(payload, self.root)

    def test_rejects_bad_ports_and_switches(self) -> None:
        for port in (True, "8123", 0, 65536):
            with self.subTest(port=port), self.assertRaises(CommandValidationError):
                invocation_from_payload({"command": "dev", "port": port}, self.root)
        with self.assertRaises(CommandValidationError):
            invocation_from_argv(
                ["langgraph", "dev", "--port", "2024", "--port", "2025"],
                self.root,
            )
        with self.assertRaises(CommandValidationError):
            invocation_from_argv(["langgraph", "up", "--verbose"], self.root)

    def test_up_accepts_wait_and_watch_in_either_order(self) -> None:
        invocation = invocation_from_argv(
            ["langgraph", "up", "--wait", "--port", "8123", "--watch"],
            self.root,
        )
        self.assertEqual(
            invocation.argv,
            ("langgraph", "up", "--port", "8123", "--watch", "--wait"),
        )
        with self.assertRaises(CommandValidationError):
            invocation_from_argv(["langgraph", "up", "--wait", "--wait"], self.root)

    def test_rejects_paths_outside_root_including_symlinks(self) -> None:
        for path in ("../escape", str(self.root / "absolute"), ".", "--help"):
            with self.subTest(path=path), self.assertRaises(CommandValidationError):
                invocation_from_payload(
                    {"command": "new", "template": "agent-python", "path": path},
                    self.root,
                )

        outside = self.root.parent / "outside"
        link = self.root / "outside-link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks are unavailable on this platform")
        with self.assertRaises(CommandValidationError):
            invocation_from_payload(
                {"command": "dockerfile", "output_path": "outside-link/Dockerfile"},
                self.root,
            )

    def test_rejects_control_character_injection(self) -> None:
        with self.assertRaises(CommandValidationError):
            invocation_from_argv(["langgraph", "dockerfile", "Dockerfile\nuname"], self.root)


class ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = Config(root_dir=self.temp_dir.name, command_timeout_seconds=7)
        self.executor = Bash(self.config)

    def tearDown(self) -> None:
        self.executor.close()
        self.temp_dir.cleanup()

    @staticmethod
    def _binary_stream(contents: bytes = b"") -> io.BufferedRandom:
        stream = tempfile.TemporaryFile(mode="w+b")
        stream.write(contents)
        stream.seek(0)
        return stream

    @patch("bash_agent.bash.subprocess.Popen")
    def test_finite_executor_uses_popen_with_exact_argv(self, popen: Mock) -> None:
        invocation = self.executor.prepare_payload(
            {"command": "dockerfile", "output_path": "Dockerfile;touch-pwned"}
        )
        process = Mock(pid=1234, returncode=0)
        process.stdout = self._binary_stream(b"created\n")
        process.stderr = self._binary_stream()
        process.wait.return_value = 0
        popen.return_value = process

        result = self.executor.exec_langgraph(invocation)

        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["stdout"], "created")
        called_argv = popen.call_args.args[0]
        self.assertEqual(called_argv, list(invocation.argv))
        self.assertEqual(called_argv[-1], "Dockerfile;touch-pwned")
        self.assertIs(popen.call_args.kwargs["shell"], False)
        self.assertIs(popen.call_args.kwargs["start_new_session"], True)
        self.assertEqual(popen.call_args.kwargs["cwd"], self.executor.cwd)
        self.assertEqual(popen.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.PIPE)
        self.assertNotIn("text", popen.call_args.kwargs)
        process.wait.assert_called_once_with(timeout=7)

    @patch("bash_agent.bash.os.killpg")
    @patch("bash_agent.bash.subprocess.Popen")
    def test_finite_timeout_terminates_then_kills_process_group(
        self, popen: Mock, killpg: Mock
    ) -> None:
        invocation = self.executor.prepare_payload({"command": "build", "tag": "app:v1"})
        process = Mock(pid=4321, returncode=-signal.SIGKILL)
        process.stdout = self._binary_stream(b"partial output")
        process.stderr = self._binary_stream(b"partial error")
        process.wait.side_effect = [
            subprocess.TimeoutExpired(list(invocation.argv), 7),
            -signal.SIGKILL,
        ]
        popen.return_value = process

        with patch.object(
            self.executor,
            "_wait_for_process_group_exit",
            side_effect=[False, True],
        ):
            result = self.executor.exec_langgraph(invocation)

        self.assertIn("process group was terminated", result["stderr"])
        self.assertEqual(result["stdout"], "partial output")
        self.assertEqual(
            killpg.call_args_list,
            [
                call(4321, signal.SIGTERM),
                call(4321, signal.SIGKILL),
            ],
        )
        self.assertEqual(
            process.wait.call_args_list,
            [call(timeout=7), call(timeout=0)],
        )

    @patch("bash_agent.bash.os.killpg")
    @patch("bash_agent.bash.subprocess.Popen")
    def test_keyboard_interrupt_terminates_finite_process_group(
        self, popen: Mock, killpg: Mock
    ) -> None:
        invocation = self.executor.prepare_payload({"command": "build", "tag": "app:v1"})
        process = Mock(pid=9876, returncode=-signal.SIGTERM)
        process.stdout = self._binary_stream()
        process.stderr = self._binary_stream()
        process.wait.side_effect = [KeyboardInterrupt, -signal.SIGTERM]
        popen.return_value = process

        with (
            patch.object(
                self.executor,
                "_wait_for_process_group_exit",
                return_value=True,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.executor.exec_langgraph(invocation)

        killpg.assert_called_once_with(9876, signal.SIGTERM)
        self.assertEqual(
            process.wait.call_args_list,
            [call(timeout=7), call(timeout=0)],
        )

    @patch("bash_agent.bash.os.killpg")
    def test_cleanup_signals_descendants_after_group_leader_exits(self, killpg: Mock) -> None:
        process = Mock(pid=7654, returncode=0)
        process.poll.return_value = 0
        process.wait.return_value = 0

        with patch.object(
            self.executor,
            "_wait_for_process_group_exit",
            side_effect=[False, True],
        ):
            self.executor._terminate_process_group(process)

        self.assertEqual(
            killpg.call_args_list,
            [
                call(7654, signal.SIGTERM),
                call(7654, signal.SIGKILL),
            ],
        )

    @patch("bash_agent.bash.subprocess.Popen")
    def test_finite_output_is_capped_while_the_pipe_is_drained(self, popen: Mock) -> None:
        self.config.output_limit_bytes = 64
        invocation = self.executor.prepare_payload({"command": "build", "tag": "app:v1"})
        process = Mock(pid=1111, returncode=0)
        process.stdout = self._binary_stream(b"x" * 4096)
        process.stderr = self._binary_stream(b"y" * 4096)
        process.wait.return_value = 0
        popen.return_value = process

        result = self.executor.exec_langgraph(invocation)

        self.assertLessEqual(len(result["stdout"].encode()), 64)
        self.assertLessEqual(len(result["stderr"].encode()), 64)
        self.assertIn("output truncated", result["stdout"])
        self.assertIn("output truncated", result["stderr"])
        self.assertEqual(result["output_limit_bytes_per_stream"], 64)

    @patch("bash_agent.bash.os.killpg")
    @patch("bash_agent.bash.subprocess.Popen")
    def test_background_process_is_logged_tracked_and_closed(
        self, popen: Mock, killpg: Mock
    ) -> None:
        invocation = self.executor.prepare_payload(
            {"command": "dev", "port": 2024, "no_browser": True}
        )
        process = Mock(pid=2468, returncode=None)
        process.stdout = self._binary_stream()
        process.wait.side_effect = [
            subprocess.TimeoutExpired(list(invocation.argv), 1),
            0,
        ]
        popen.return_value = process

        result = self.executor.exec_langgraph(invocation)

        self.assertIs(result["background"], True)
        self.assertEqual(result["pid"], 2468)
        log_path = Path(result["log_path"])
        self.assertTrue(log_path.is_file())
        managed = self.executor._background_processes[2468]
        log_file = managed.log_file
        self.assertEqual(popen.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.STDOUT)
        self.assertIs(popen.call_args.kwargs["start_new_session"], True)
        self.assertFalse(log_file.closed)
        self.assertEqual(result["log_retention"], "deleted_on_close")

        with patch.object(
            self.executor,
            "_wait_for_process_group_exit",
            return_value=True,
        ):
            self.executor.close()

        killpg.assert_called_once_with(2468, signal.SIGTERM)
        self.assertTrue(log_file.closed)
        self.assertFalse(log_path.exists())

    @patch("bash_agent.bash.os.killpg")
    @patch("bash_agent.bash.subprocess.Popen")
    def test_background_startup_interrupt_cleans_process_and_log(
        self, popen: Mock, killpg: Mock
    ) -> None:
        invocation = self.executor.prepare_payload({"command": "dev"})
        process = Mock(pid=1357, returncode=-signal.SIGTERM)
        process.stdout = self._binary_stream()
        process.wait.side_effect = [KeyboardInterrupt, -signal.SIGTERM]
        popen.return_value = process
        log_file = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
        log_path = Path(log_file.name)

        with (
            patch(
                "bash_agent.bash.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch.object(
                self.executor,
                "_wait_for_process_group_exit",
                return_value=True,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.executor.exec_langgraph(invocation)

        killpg.assert_called_once_with(1357, signal.SIGTERM)
        self.assertTrue(log_file.closed)
        self.assertFalse(log_path.exists())
        self.assertEqual(self.executor._background_processes, {})
        self.assertEqual(self.executor._temporary_logs, set())

    @patch("bash_agent.bash.subprocess.Popen")
    def test_background_log_is_capped_and_deleted_on_close(self, popen: Mock) -> None:
        self.config.output_limit_bytes = 64
        invocation = self.executor.prepare_payload({"command": "dev"})
        process = Mock(pid=8642, returncode=2)
        process.stdout = self._binary_stream(b"z" * 4096)
        process.wait.return_value = 2
        popen.return_value = process

        result = self.executor.exec_langgraph(invocation)

        log_path = Path(result["log_path"])
        self.assertTrue(log_path.is_file())
        self.assertLessEqual(log_path.stat().st_size, 64)
        self.assertIn("output truncated", log_path.read_text(encoding="utf-8"))
        self.assertIn("output truncated", result["stdout"])
        self.assertEqual(result["log_limit_bytes"], 64)

        self.executor.close()
        self.assertFalse(log_path.exists())

    def test_only_dev_and_up_without_wait_are_background(self) -> None:
        dev = self.executor.prepare_payload({"command": "dev"})
        up = self.executor.prepare_payload({"command": "up"})
        up_wait = self.executor.prepare_payload({"command": "up", "wait": True})
        build = self.executor.prepare_payload({"command": "build", "tag": "app:v1"})
        self.assertTrue(self.executor._is_background(dev))
        self.assertTrue(self.executor._is_background(up))
        self.assertFalse(self.executor._is_background(up_wait))
        self.assertFalse(self.executor._is_background(build))

    def test_child_environment_scrubs_secrets_unless_explicitly_allowed(self) -> None:
        environment = {
            "PATH": "/usr/bin",
            "ORDINARY_SETTING": "kept",
            "NVIDIA_API_KEY": "drop",
            "OPENROUTER_API_KEY": "drop",
            "OPENAI_API_KEY": "explicitly-kept",
            "LANGSMITH_API_KEY": "drop",
            "LANGGRAPH_CLOUD_LICENSE_KEY": "drop",
            "SERVICE_API_KEY": "drop",
            "HF_TOKEN": "explicitly-kept",
            "DATABASE_PASSWORD": "drop",
            "SERVICE_SECRET": "drop",
            "GOOGLE_CREDENTIALS": "drop",
            "AWS_ACCESS_KEY_ID": "drop",
            "AWS_SECRET_ACCESS_KEY": "drop",
        }
        self.config.pass_environment.update({"OPENAI_API_KEY", "HF_TOKEN"})
        with patch.dict(os.environ, environment, clear=True):
            child_environment = self.executor.child_environment()

        self.assertEqual(child_environment["PATH"], "/usr/bin")
        self.assertEqual(child_environment["ORDINARY_SETTING"], "kept")
        self.assertEqual(child_environment["OPENAI_API_KEY"], "explicitly-kept")
        self.assertEqual(child_environment["HF_TOKEN"], "explicitly-kept")
        for secret in {
            "NVIDIA_API_KEY",
            "OPENROUTER_API_KEY",
            "LANGSMITH_API_KEY",
            "LANGGRAPH_CLOUD_LICENSE_KEY",
            "SERVICE_API_KEY",
            "DATABASE_PASSWORD",
            "SERVICE_SECRET",
            "GOOGLE_CREDENTIALS",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        }:
            self.assertNotIn(secret, child_environment)

    def test_output_limit_must_leave_room_for_a_truncation_marker(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_limit_bytes"):
            Bash(Config(root_dir=self.temp_dir.name, output_limit_bytes=1))

    def test_executor_requires_prepared_invocation(self) -> None:
        with self.assertRaises(TypeError):
            self.executor.exec_langgraph(["langgraph", "dev"])  # type: ignore[arg-type]
        with self.assertRaises(CommandValidationError):
            self.executor.exec_langgraph(LangGraphInvocation(("uname", "-a")))

    def test_human_decline_never_executes(self) -> None:
        invocation = self.executor.prepare_payload({"command": "build", "tag": "app:v1"})
        self.executor.exec_langgraph = Mock()  # type: ignore[method-assign]
        confirmed: list[tuple[str, ...]] = []

        def decline(argv: tuple[str, ...]) -> bool:
            confirmed.append(argv)
            return False

        result = execute_with_confirmation(self.executor, invocation, decline)
        self.assertIn("declined", result["error"])
        self.assertEqual(confirmed, [invocation.argv])
        self.executor.exec_langgraph.assert_not_called()

    def test_human_approval_executes_same_immutable_invocation(self) -> None:
        invocation = self.executor.prepare_payload({"command": "build", "tag": "app:v1"})
        expected = {"stdout": "ok"}
        self.executor.exec_langgraph = Mock(return_value=expected)  # type: ignore[method-assign]
        result = execute_with_confirmation(self.executor, invocation, lambda argv: True)
        self.assertIs(result, expected)
        self.executor.exec_langgraph.assert_called_once_with(invocation)

    def test_root_dir_cli_resolves_an_existing_project_directory(self) -> None:
        project = Path(self.temp_dir.name) / "project"
        project.mkdir()
        args = parse_args(["--root-dir", str(project / ".." / "project")])
        self.assertEqual(args.root_dir, str(project.resolve()))
        self.assertEqual(config_from_args(args).root_dir, str(project.resolve()))

    def test_root_dir_cli_rejects_missing_paths_and_files(self) -> None:
        file_path = Path(self.temp_dir.name) / "langgraph.json"
        file_path.write_text("{}", encoding="utf-8")
        for candidate in (Path(self.temp_dir.name) / "missing", file_path):
            with (
                self.subTest(candidate=candidate),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(["--root-dir", str(candidate)])

    def test_runtime_cli_options_update_config(self) -> None:
        args = parse_args(
            [
                "--command-timeout",
                "12.5",
                "--pass-env",
                "OPENAI_API_KEY",
                "--pass-env",
                "CUSTOM_TOKEN",
                "--omit-api-thinking-override",
            ]
        )

        config = config_from_args(args)

        self.assertEqual(config.command_timeout_seconds, 12.5)
        self.assertEqual(
            config.pass_environment,
            {"OPENAI_API_KEY", "CUSTOM_TOKEN"},
        )
        self.assertIs(config.api_send_thinking_override, False)

    def test_command_timeout_cli_requires_finite_positive_seconds(self) -> None:
        for value in ("0", "-1", "nan", "inf", "not-a-number"):
            with (
                self.subTest(value=value),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(["--command-timeout", value])

    def test_pass_env_cli_requires_an_environment_variable_name(self) -> None:
        for value in ("BAD-NAME", "1STARTS_WITH_NUMBER", "NAME=value"):
            with (
                self.subTest(value=value),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(["--pass-env", value])

    def test_main_closes_executor_when_agent_raises(self) -> None:
        executor = Mock()
        with (
            patch("bash_agent.main_hf.Bash", return_value=executor),
            patch(
                "bash_agent.main_hf._run_agent",
                side_effect=RuntimeError("agent failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "agent failed"),
        ):
            main(self.config)

        executor.close.assert_called_once_with()

    def test_cli_help_documents_runtime_safety_options(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--root-dir", output.getvalue())
        self.assertIn("trusted directory", output.getvalue())
        self.assertIn("--command-timeout", output.getvalue())
        self.assertIn("--pass-env", output.getvalue())
        self.assertIn("--omit-api-thinking-override", output.getvalue())


class MessageAndInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = Config(root_dir=self.temp_dir.name, device="cpu")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_default_temperature_matches_nemotron_rollout_setting(self) -> None:
        self.assertEqual(self.config.temperature, 0.6)

    def test_assistant_tool_call_precedes_matching_tool_result(self) -> None:
        class SDKToolCall:
            def model_dump(self, exclude_none: bool = False) -> dict[str, object]:
                self.exclude_none = exclude_none
                return {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "run_langgraph",
                        "arguments": '{"argv":["langgraph","dev"]}',
                    },
                }

        call = SDKToolCall()
        messages = Messages("system")
        messages.add_user_message("start")
        messages.add_assistant_tool_calls(None, [call])
        messages.add_tool_message('{"stdout":"ok"}', "call_123")

        serialized = messages.to_list()
        self.assertEqual(
            [item["role"] for item in serialized],
            ["system", "user", "assistant", "tool"],
        )
        self.assertEqual(serialized[2]["tool_calls"][0]["id"], "call_123")
        self.assertEqual(serialized[3]["tool_call_id"], "call_123")
        self.assertTrue(call.exclude_none)

    def test_local_parser_converts_structured_output_to_argv(self) -> None:
        calls = parse_model_tool_calls(
            '<think>reason</think>{"command":"build","tag":"app:v2"}',
            self.temp_dir.name,
        )
        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(arguments["argv"], ["langgraph", "build", "-t", "app:v2"])

    def test_local_parser_rejects_legacy_shell_string(self) -> None:
        calls = parse_model_tool_calls(
            '{"tool":"exec_bash_command","cmd":"echo ok\\nuname -a"}',
            self.temp_dir.name,
        )
        self.assertEqual(calls, [])

    def test_openai_query_preserves_native_calls_and_current_parameters(self) -> None:
        native_call = {
            "id": "call_native",
            "type": "function",
            "function": {
                "name": "run_langgraph",
                "arguments": '{"argv":["langgraph","dev"]}',
            },
        }
        create = Mock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[native_call]))
                ]
            )
        )
        llm = object.__new__(OpenAILLM)
        llm.config = self.config
        llm.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        content, calls = llm.query(Messages("system"), [{"type": "function"}])

        self.assertEqual(content, "")
        self.assertEqual(calls, [native_call])
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["max_completion_tokens"], self.config.max_new_tokens)
        self.assertIs(kwargs["parallel_tool_calls"], False)
        self.assertEqual(
            kwargs["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertNotIn("max_tokens", kwargs)

    def test_openai_query_falls_back_to_trained_json_content(self) -> None:
        create = Mock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"command":"dockerfile","output_path":"Dockerfile"}',
                            tool_calls=None,
                        )
                    )
                ]
            )
        )
        llm = object.__new__(OpenAILLM)
        llm.config = self.config
        llm.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        _, calls = llm.query(Messages("system"))
        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(arguments["argv"], ["langgraph", "dockerfile", "Dockerfile"])

    def test_openai_query_can_omit_vllm_thinking_extension(self) -> None:
        create = Mock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None))]
            )
        )
        self.config.api_send_thinking_override = False
        llm = object.__new__(OpenAILLM)
        llm.config = self.config
        llm.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        llm.query(Messages("system"))

        self.assertNotIn("extra_body", create.call_args.kwargs)

    def test_cpu_model_settings_use_float32_and_explicit_device_map(self) -> None:
        class FakeDevice:
            type = "cpu"

            def __str__(self) -> str:
                return "cpu"

        fake_torch = SimpleNamespace(
            device=lambda value: FakeDevice(),
            float32=object(),
            float16=object(),
            bfloat16=object(),
        )
        dtype, device_map = _model_load_settings(self.config, fake_torch)
        self.assertIs(dtype, fake_torch.float32)
        self.assertEqual(device_map, {"": "cpu"})


if __name__ == "__main__":
    unittest.main()
