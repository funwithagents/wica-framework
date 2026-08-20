from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from wica.fake_model import FakeChatModel


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def greet(name: str) -> str:
    """Greet someone."""
    return f"hi {name}"


def test_script_consumed_in_order():
    model = FakeChatModel(
        script=[{"text": "first"}, {"text": "second"}, {"text": "third"}], delay_s=0
    )
    assert [model.invoke("q").content for _ in range(3)] == ["first", "second", "third"]


def test_text_and_tool_calls_mapping():
    model = FakeChatModel(
        script=[
            {
                "text": "doing it",
                "tool_calls": [
                    {"name": "add", "args": {"a": 1, "b": 2}},  # id omitted -> generated
                    {"name": "greet", "args": {"name": "sam"}, "id": "supplied-id"},
                ],
            }
        ],
        delay_s=0,
    )
    message = model.invoke("q")

    assert message.content == "doing it"
    assert [(c["name"], c["args"]) for c in message.tool_calls] == [
        ("add", {"a": 1, "b": 2}),
        ("greet", {"name": "sam"}),
    ]
    # Generated id is deterministic (step 0, call 0); a supplied id is preserved verbatim.
    assert message.tool_calls[0]["id"] == "fake_call_0_0"
    assert message.tool_calls[1]["id"] == "supplied-id"


def test_exhaustion_returns_default():
    model = FakeChatModel(
        script=[{"text": "only step"}], default={"text": "DEFAULT"}, delay_s=0
    )
    assert model.invoke("q").content == "only step"
    assert model.invoke("q").content == "DEFAULT"
    assert model.invoke("q").content == "DEFAULT"


def test_default_defaults_to_empty_text():
    model = FakeChatModel(script=[{"text": "one"}], delay_s=0)
    spent = model.invoke("q")  # consume the only step
    assert spent.content == "one"
    exhausted = model.invoke("q")
    assert exhausted.content == ""
    assert exhausted.tool_calls == []


def test_loop_cycles_the_script():
    model = FakeChatModel(script=[{"text": "a"}, {"text": "b"}], loop=True, delay_s=0)
    assert [model.invoke("q").content for _ in range(5)] == ["a", "b", "a", "b", "a"]


def test_calls_records_messages_per_call():
    model = FakeChatModel(script=[{"text": "x"}, {"text": "y"}], delay_s=0)
    model.invoke([HumanMessage("one")])
    model.invoke([HumanMessage("two")])

    assert len(model.calls) == 2
    assert [m.content for call in model.calls for m in call] == ["one", "two"]


def test_delay_defaults_to_200ms():
    # A non-zero default is load-bearing (keeps the Agent loop's timing realistic); pin it.
    assert FakeChatModel().delay_s == 0.2


def test_ainvoke_is_cancellable_during_delay():
    async def scenario() -> None:
        model = FakeChatModel(script=[{"text": "slow"}], delay_s=5)
        task = asyncio.create_task(model.ainvoke("q"))
        await asyncio.sleep(0.02)  # let it reach the sleep
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_bind_tools_accepts_a_valid_script():
    model = FakeChatModel(
        script=[{"tool_calls": [{"name": "add", "args": {"a": 1, "b": 2}}]}], delay_s=0
    )
    bound = model.bind_tools([add])
    assert bound is model  # canned output unaffected; the same model drives ainvoke
    assert bound.invoke("q").tool_calls[0]["name"] == "add"


def test_bind_tools_rejects_unknown_tool_name():
    model = FakeChatModel(
        script=[{"tool_calls": [{"name": "subtract", "args": {}}]}], delay_s=0
    )
    model.bind_tools([add, greet])
    with pytest.raises(ValueError, match="subtract"):
        model.invoke("q")


def test_validation_is_lazy_against_the_latest_binding():
    # The Agent rebinds on every register_command; a script naming a tool bound only later must not
    # be rejected by an earlier, partial binding. Validation runs on first invoke, against the last.
    model = FakeChatModel(
        script=[{"tool_calls": [{"name": "greet", "args": {"name": "sam"}}]}], delay_s=0
    )
    model.bind_tools([add])  # partial: greet not yet present — must NOT reject here
    model.bind_tools([add, greet])  # full set
    assert model.invoke("q").tool_calls[0]["name"] == "greet"


def test_no_bind_no_validation():
    # A tool-call script with bind_tools never called (unusual, but shouldn't crash on a missing set).
    model = FakeChatModel(
        script=[{"tool_calls": [{"name": "whatever", "args": {}}]}], delay_s=0
    )
    assert model.invoke("q").tool_calls[0]["name"] == "whatever"
