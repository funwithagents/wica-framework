"""A deterministic, network-free, key-less LangChain chat model for tests.

`FakeChatModel` replays a **scripted** sequence of responses: each model call consumes the next
step of `script` and returns the corresponding `AIMessage` (text, native tool calls, or both).
It is selected like any other provider via `provider: "fake"` (see `build_chat_model` in
[agent.py](agent.py)), so a test can drive the Agent's whole step loop over a model whose output
it fully controls — no real LLM, no credentials. See specs/fake-provider.md.

This is **test tooling**, deliberately reached via its own submodule
(`from wica.fake_model import FakeChatModel`) rather than the runtime `wica` namespace.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr


def _tool_name(tool: Any) -> str | None:
    """Best-effort tool name for `bind_tools` validation. The Agent binds `BaseTool`s, so that
    branch covers real use; dicts/callables are handled leniently for direct-construction tests."""
    if isinstance(tool, BaseTool):
        return tool.name
    if isinstance(tool, dict):
        fn = tool.get("function")
        if isinstance(fn, dict) and "name" in fn:
            return str(fn["name"])
        if "name" in tool:
            return str(tool["name"])
        return None
    return getattr(tool, "__name__", None)


class FakeChatModel(BaseChatModel):
    """A scripted chat model. Configure it through `model_kwargs` (so it flows through the ordinary
    JSON config path) or construct it directly:

        FakeChatModel(script=[{"text": "hi"}, {"tool_calls": [{"name": "add", "args": {...}}]}])

    Each `ainvoke`/`invoke` consumes the next `script` step. Once the script is spent, `default` is
    returned (unless `loop` is set, which cycles the script). `bind_tools` records the bound tool
    names and, on the first call, validates that every scripted tool call names one of them.
    """

    # Scripted config — JSON-expressible, so it rides through `model_kwargs`.
    script: list[dict[str, Any]] = Field(default_factory=list)
    default: dict[str, Any] = Field(default_factory=lambda: {"text": ""})
    loop: bool = False
    # Simulated latency awaited before each async response (200 ms). A non-zero default keeps the
    # Agent loop's timing realistic (coalescing window, barge-in, mid-step cancellation) instead of
    # collapsing it to zero; set 0 for instant responses. See specs/fake-provider.md.
    delay_s: float = 0.2

    # Mutable run state — private so pydantic doesn't treat it as config.
    _cursor: int = PrivateAttr(default=0)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _tool_names: set[str] | None = PrivateAttr(default=None)
    _validated: bool = PrivateAttr(default=False)

    @property
    def _llm_type(self) -> str:
        return "wica-fake"

    @property
    def calls(self) -> list[list[BaseMessage]]:
        """The message lists this model was handed, one per call, in order — for prompt assertions."""
        return list(self._calls)

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # The canned output is unaffected by binding; we only record the tool names so the scripted
        # tool-call names can be validated (lazily — see _validate_once). Returning self keeps the
        # bound runnable's `ainvoke` pointing at this model.
        self._tool_names = {
            name for tool in tools if (name := _tool_name(tool)) is not None
        }
        return cast("Runnable[LanguageModelInput, AIMessage]", self)

    def _validate_once(self) -> None:
        # Validate lazily, on the first generate rather than inside bind_tools: the Agent rebinds on
        # every register_command (each time with the full current tool list) and auto-registers
        # cancel_command in start(), so bind-time validation would fire against a partial set and
        # wrongly reject a script naming a not-yet-registered command. By the first model call all
        # binds are done, so _tool_names is the final set.
        if self._validated:
            return
        self._validated = True
        if self._tool_names is None:
            return  # bind_tools never called (e.g. a script with no tool calls) — nothing to check
        unknown = sorted(
            {
                call["name"]
                for step in self.script
                for call in step.get("tool_calls", []) or []
                if call["name"] not in self._tool_names
            }
        )
        if unknown:
            raise ValueError(
                f"FakeChatModel script references tool(s) {unknown} not among the bound tools "
                f"{sorted(self._tool_names)}. Check the command name(s) in the script."
            )

    def _next_step(self) -> tuple[dict[str, Any], int]:
        """Consume and return the next (step, step_index). Past the end: `default` (index -1),
        unless `loop` cycles the script."""
        n = len(self.script)
        if self._cursor < n:
            idx = self._cursor
            step = self.script[idx]
        elif self.loop and n:
            idx = self._cursor % n
            step = self.script[idx]
        else:
            idx = -1
            step = self.default
        self._cursor += 1
        return step, idx

    def _to_result(self, step: dict[str, Any], idx: int) -> ChatResult:
        text = step.get("text", "") or ""
        tool_calls = [
            {
                "name": call["name"],
                "args": call.get("args", {}) or {},
                # A stable, deterministic id when none is supplied, so assertions are reproducible.
                "id": call.get("id") or f"fake_call_{idx}_{i}",
                "type": "tool_call",
            }
            for i, call in enumerate(step.get("tool_calls", []) or [])
        ]
        message = AIMessage(content=text, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._calls.append(list(messages))
        self._validate_once()
        step, idx = self._next_step()
        return self._to_result(step, idx)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # A genuine coroutine that yields at the sleep: it runs on the Agent's loop like any
        # provider, and a task.cancel() landing during the sleep unwinds cleanly.
        self._calls.append(list(messages))
        self._validate_once()
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        step, idx = self._next_step()
        return self._to_result(step, idx)
