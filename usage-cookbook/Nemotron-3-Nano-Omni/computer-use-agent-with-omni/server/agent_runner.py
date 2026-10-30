"""Agent loop that drives a NemotronAgent against the desktop container.

Publishes events to an asyncio queue so the FastAPI SSE endpoint can
stream them to the browser in real-time.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from server.agent import NemotronAgent, ParsedStep
from server.desktop_client import DesktopClient


@dataclass
class AgentJob:
    job_id: str
    instruction: str
    status: str = "pending"   # pending | running | done | failed | stopped | error
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    _task: Optional[asyncio.Task] = None
    _stop: asyncio.Event = field(default_factory=asyncio.Event)


class AgentRunner:
    def __init__(
        self,
        agent: NemotronAgent,
        desktop: DesktopClient,
        *,
        max_steps: int = 40,
        wait_after_action: float = 0.0,
        computer_wait_seconds: float = 3.0,
    ) -> None:
        self.agent = agent
        self.desktop = desktop
        self.max_steps = max_steps
        self.wait_after_action = wait_after_action
        self.computer_wait_seconds = max(0.0, computer_wait_seconds)

    def start(self, instruction: str) -> AgentJob:
        job = AgentJob(
            job_id=f"job-{uuid.uuid4().hex[:8]}",
            instruction=instruction,
            started_at=time.time(),
        )
        self.agent.reset()
        job._task = asyncio.create_task(self._run(job))
        return job

    async def _emit(self, job: AgentJob, kind: str, **payload) -> None:
        await job.events.put({"kind": kind, "ts": time.time(), **payload})

    async def _run(self, job: AgentJob) -> None:
        job.status = "running"
        await self._emit(job, "started", instruction=job.instruction)

        try:
            screen_w, screen_h = await self.desktop.screen_size()
            await self._emit(job, "screen_size", width=screen_w, height=screen_h)

            for step in range(1, self.max_steps + 1):
                if job._stop.is_set():
                    job.status = "stopped"
                    await self._emit(job, "stopped")
                    return

                await self._emit(job, "step_started", step=step)

                # Take screenshot
                try:
                    png = await self.desktop.screenshot()
                except Exception as exc:
                    job.status = "error"
                    job.error = f"screenshot failed: {exc}"
                    await self._emit(job, "error", message=job.error)
                    return

                # Stream reasoning tokens to frontend
                async def _on_delta(r_delta: str, c_delta: str, _step=step) -> None:
                    await self._emit(
                        job, "thought_delta",
                        step=_step,
                        reasoning=r_delta,
                        content=c_delta,
                    )

                # Call the model
                parsed: ParsedStep = await self.agent.step(
                    job.instruction, png, (screen_w, screen_h),
                    delta_callback=_on_delta,
                )

                if job._stop.is_set():
                    job.status = "stopped"
                    await self._emit(job, "stopped")
                    return

                await self._emit(
                    job, "thought",
                    step=step,
                    thought=parsed.thought[:500] if parsed.thought else "",
                    action=parsed.action,
                    code=parsed.original_code,
                    status=parsed.status,
                )

                if parsed.status == "error":
                    job.status = "error"
                    job.error = parsed.error or "parse error"
                    await self._emit(job, "error", message=job.error)
                    return
                if parsed.status == "done":
                    job.status = "done"
                    await self._emit(job, "done", step=step, action=parsed.action)
                    return
                if parsed.status == "fail":
                    job.status = "failed"
                    await self._emit(job, "failed", step=step, action=parsed.action)
                    return
                if parsed.status == "wait":
                    await self._emit(job, "wait", step=step, seconds=self.computer_wait_seconds)
                    try:
                        await asyncio.wait_for(
                            job._stop.wait(),
                            timeout=self.computer_wait_seconds,
                        )
                    except asyncio.TimeoutError:
                        pass
                    if job._stop.is_set():
                        job.status = "stopped"
                        await self._emit(job, "stopped")
                        return
                    continue

                # Execute pyautogui code
                if job._stop.is_set():
                    job.status = "stopped"
                    await self._emit(job, "stopped")
                    return
                try:
                    output = await self.desktop.run_pyautogui(parsed.code)
                    await self._emit(
                        job, "executed", step=step, code=parsed.code, output=output[:500]
                    )
                    self.agent.record_tool_result(parsed.tool_name or "pyautogui", output)
                except Exception as exc:
                    err = str(exc)
                    await self._emit(
                        job, "execute_error", step=step, code=parsed.code, message=err
                    )
                    self.agent.record_tool_result(
                        parsed.tool_name or "pyautogui", f"error: {err}"
                    )

                try:
                    await asyncio.wait_for(
                        job._stop.wait(),
                        timeout=self.wait_after_action,
                    )
                except asyncio.TimeoutError:
                    pass
                if job._stop.is_set():
                    job.status = "stopped"
                    await self._emit(job, "stopped")
                    return

            # Hit step ceiling
            job.status = "failed"
            await self._emit(job, "failed", reason="max_steps")

        except asyncio.CancelledError:
            job.status = "stopped"
            await self._emit(job, "stopped")
            raise
        except Exception as exc:
            logger.exception("agent loop crashed")
            job.status = "error"
            job.error = str(exc)
            await self._emit(job, "error", message=str(exc))
        finally:
            job.finished_at = time.time()
            await self._emit(job, "finished", status=job.status)
