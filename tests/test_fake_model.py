from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
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
                    {
                        "name": "add",
                        "args": {"a": 1, "b": 2},
                    },  # id omitted -> generated
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


def test_looped_script_generates_unique_ids():
    # Generated ids key on the call ordinal, not the script position, so a looped script never
    # repeats a tool-call id (a repeat would collide in the Agent's rendered tool_call/tool_result
    # stream). The first call's id is unchanged for non-looped scripts (`fake_call_0_0`).
    model = FakeChatModel(
        script=[{"tool_calls": [{"name": "add", "args": {}}]}], loop=True, delay_s=0
    )
    ids = [model.invoke("q").tool_calls[0]["id"] for _ in range(3)]
    assert ids == ["fake_call_0_0", "fake_call_1_0", "fake_call_2_0"]


def test_usage_is_estimated_from_the_prompt_and_reply_by_default():
    model = FakeChatModel(script=[{"text": "x" * 40}], delay_s=0)
    short = model.invoke([HumanMessage(content="a" * 40)])
    assert short.usage_metadata is not None
    assert short.usage_metadata["input_tokens"] == 10
    assert short.usage_metadata["output_tokens"] == 10
    # Exhausted → default (empty text); a longer prompt reports more input tokens.
    long = model.invoke([HumanMessage(content="a" * 400)])
    assert long.usage_metadata is not None
    assert long.usage_metadata["input_tokens"] == 100
    assert long.usage_metadata["output_tokens"] == 0


def test_scripted_usage_overrides_the_estimate():
    model = FakeChatModel(
        script=[{"text": "hi", "usage": {"input_tokens": 90000, "output_tokens": 7}}],
        delay_s=0,
    )
    message = model.invoke([HumanMessage(content="short")])
    assert message.usage_metadata is not None
    assert message.usage_metadata["input_tokens"] == 90000
    assert message.usage_metadata["output_tokens"] == 7
    assert message.usage_metadata["total_tokens"] == 90007


# --- Streaming (specs/fake-provider.md "Streaming") ----------------------------------------------


async def _collect(model: FakeChatModel, prompt: str) -> list[AIMessageChunk]:
    chunks: list[AIMessageChunk] = []
    async for chunk in model.astream(prompt):
        assert isinstance(chunk, AIMessageChunk)
        chunks.append(chunk)
    return chunks


STEP = {
    "text": "Adding them now.",
    "tool_calls": [
        {"name": "add", "args": {"a": 1, "b": 2}},
        {"name": "greet", "args": {"name": "Ada"}},
    ],
}


def test_astream_splits_a_step_into_chunks_that_sum_to_the_ainvoke_message():
    async def scenario() -> None:
        streamed = FakeChatModel(script=[STEP], delay_s=0)
        chunks = await _collect(streamed, "q")
        whole = await FakeChatModel(script=[STEP], delay_s=0).ainvoke("q")

        # The text arrives in pieces, each tool call across two chunks, the usage on one chunk
        # after them (LangChain may close the stream with an empty chunk of its own).
        assert [c.text for c in chunks if c.text] == ["Adding ", "them ", "now."]
        assert sum(1 for c in chunks if c.tool_call_chunks) == 4
        with_usage = [i for i, c in enumerate(chunks) if c.usage_metadata]
        assert len(with_usage) == 1
        assert chunks[with_usage[0]].usage_metadata == whole.usage_metadata
        assert with_usage[0] > max(
            i for i, c in enumerate(chunks) if c.tool_call_chunks
        )
        summed = chunks[0]
        for chunk in chunks[1:]:
            summed = summed + chunk
        assert summed.text == whole.text
        assert summed.tool_calls == whole.tool_calls
        assert summed.usage_metadata == whole.usage_metadata
        # One call recorded, one script step consumed.
        assert len(streamed.calls) == 1
        assert (await streamed.ainvoke("q")).text == ""

    asyncio.run(scenario())


def test_astream_of_an_empty_step_yields_only_the_usage():
    async def scenario() -> None:
        chunks = await _collect(FakeChatModel(script=[{}], delay_s=0), "four")
        assert all(c.text == "" and c.tool_call_chunks == [] for c in chunks)
        usages = [c.usage_metadata for c in chunks if c.usage_metadata]
        assert len(usages) == 1
        assert usages[0]["input_tokens"] == 1

    asyncio.run(scenario())


def test_astream_is_cancellable_during_the_delay_before_its_first_chunk():
    async def scenario() -> None:
        model = FakeChatModel(script=[{"text": "slow"}], delay_s=5)
        task = asyncio.create_task(_collect(model, "q"))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_astream_validates_scripted_tool_names_like_ainvoke():
    model = FakeChatModel(
        script=[{"tool_calls": [{"name": "typo", "args": {}}]}], delay_s=0
    )
    model.bind_tools([add])
    with pytest.raises(ValueError, match="typo"):
        asyncio.run(_collect(model, "q"))
