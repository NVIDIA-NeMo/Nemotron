import unittest
import asyncio
import json

import httpx

from server.agent import NemotronAgent, parse_response
from server.vllm_inference import VllmInferenceAgent


class AgentParserTests(unittest.TestCase):
    def test_thinking_extra_body_matches_nemotron_omni_contract(self):
        agent = NemotronAgent(
            api_key="test",
            reasoning_budget=16384,
            reasoning_grace_tokens=1024,
            thinking=True,
        )

        self.assertEqual(
            agent.build_extra_body(),
            {
                "thinking_token_budget": 17408,
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_budget": 16384,
                    "truncate_history_thinking": False,
                },
            },
        )

    def test_non_thinking_extra_body_disables_thinking_explicitly(self):
        agent = NemotronAgent(api_key="test", thinking=False)

        self.assertEqual(
            agent.build_extra_body(),
            {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "truncate_history_thinking": False,
                }
            },
        )

    def test_parses_same_line_action_and_code_fence(self):
        parsed = parse_response(
            "## Action: Click Chrome.\n## Code: ```python\npyautogui.click(0.5, 0.5)\n```",
            "click chrome",
            1920,
            1080,
            thinking=True,
        )

        self.assertEqual(parsed.status, "continue")
        self.assertEqual(parsed.action, "Click Chrome.")
        self.assertEqual(parsed.code, "pyautogui.click(960, 540)")

    def test_falls_back_to_reasoning_when_action_is_in_reasoning(self):
        reasoning = (
            "I should click the browser.\n"
            "## Action: Click Chrome.\n"
            "## Code:\n"
            "```python\npyautogui.click(0.25, 0.75)\n```"
        )

        parsed = parse_response("", reasoning, 1600, 1200, thinking=True)

        self.assertEqual(parsed.status, "continue")
        self.assertEqual(parsed.action, "Click Chrome.")
        self.assertEqual(parsed.code, "pyautogui.click(400, 900)")

    def test_accepts_json_fenced_computer_function_inside_code_section(self):
        parsed = parse_response(
            (
                "## Action: Done.\n"
                "## Code:\n"
                "```json\n"
                '{"name":"computer.terminate","parameters":{"status":"success"}}\n'
                "```"
            ),
            "done",
            1920,
            1080,
            thinking=True,
        )

        self.assertEqual(parsed.status, "done")
        self.assertEqual(parsed.code, "DONE")

    def test_strips_unmatched_trailing_code_fence_from_code_section(self):
        parsed = parse_response(
            (
                "## Action: Type query.\n"
                "## Code: pyautogui.click(0.5, 0.5)\n"
                'pyautogui.typewrite("NVDA stock price")\n'
                "```"
            ),
            "",
            1000,
            1000,
            thinking=True,
        )

        self.assertEqual(parsed.status, "continue")
        self.assertEqual(
            parsed.code,
            'pyautogui.click(500, 500)\npyautogui.typewrite("NVDA stock price")',
        )

    def test_does_not_execute_reasoning_json_without_action_heading(self):
        parsed = parse_response(
            "",
            'Example only:\n```json\n{"type":"text","text":"hello"}\n```',
            1920,
            1080,
            thinking=True,
        )

        self.assertEqual(parsed.status, "error")
        self.assertIn("action", parsed.error or "")

    def test_uses_first_executable_code_and_ignores_trailing_terminate_spam(self):
        content = (
            "The previous on the image.\n"
            "## Action: Press the Enter key to execute the search.\n"
            "## Code: pyautogui.keyDown('return')\n"
            "## Code: pyautogui.keyUp('return')\n"
            "</think>\n"
            "## Action: Press the Enter key to execute the search.\n"
            "## Code: pyautogui.keyDown('return')\n"
            "## Code: pyautogui.keyUp('return')\n"
            "## Code: computer.terminate(status='success')\n"
            "## Code: computer.terminate(status='success')\n"
        )

        parsed = parse_response(content, "", 1920, 1080, thinking=True)

        self.assertEqual(parsed.status, "continue")
        self.assertIn("Press the Enter key", parsed.action)
        self.assertEqual(
            parsed.code,
            "pyautogui.keyDown('return')\npyautogui.keyUp('return')",
        )

    def test_vllm_path_parses_stubbed_openai_response(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        agent = VllmInferenceAgent(
            api_key="EMPTY",
            api_base="http://stub/v1",
            model="vllm_local",
            max_retry=1,
            thinking=True,
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.assertEqual(body["model"], "vllm_local")
            self.assertEqual(body["thinking_token_budget"], 17408)
            self.assertEqual(
                body["chat_template_kwargs"],
                {
                    "enable_thinking": True,
                    "reasoning_budget": 16384,
                    "truncate_history_thinking": False,
                },
            )
            self.assertTrue(
                any(
                    isinstance(message["content"], list)
                    and any(part["type"] == "image_url" for part in message["content"])
                    for message in body["messages"]
                )
            )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "reasoning_content": "click around",
                                "content": (
                                    "## Action:\nclick the icon\n## Code:\n"
                                    "```python\npyautogui.click(0.25, 0.75)\n```"
                                ),
                            },
                        }
                    ]
                },
            )

        async def run_step():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                return await agent.step("do a thing", png, (1600, 1200), client=client)

        parsed = asyncio.run(run_step())
        self.assertEqual(parsed.status, "continue")
        self.assertEqual(parsed.code, "pyautogui.click(400, 900)")


if __name__ == "__main__":
    unittest.main()
