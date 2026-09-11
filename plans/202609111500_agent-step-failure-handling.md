# Agent step failure handling: log model-call and output-sink errors, keep history well-formed

**Status:** Done

Fixes a gap found in the 2026-09-11 codebase review: today a model call that raises inside a
reasoning step is invisible to the framework's own logging (it surfaces only as asyncio's "Task
exception was never retrieved" at garbage-collection time) and leaves the Agent's history in a
shape that renders as two consecutive user-role messages. An output sink that raises has the
same problems and additionally drops the Commands issued in the same response.

This plan is written to be executed step by step, in order, by an agent with no other context.
Every step says **which file**, **what to change**, **what the result must look like**,
and **which test proves it**. Line numbers refer to the tree right after the 2026-09-11 analysis
fixes; re-locate by the quoted code if they have drifted. Read [AGENTS.md](../AGENTS.md) first
(verification and status rules), then [specs/agent.md](../specs/agent.md) sections "History record
shape", "Rendering to messages", "Instrumentation: observability Events", and "Open questions".

## Goal

After this plan:

1. Any exception escaping a reasoning step's **model call** is caught, logged at **ERROR** under
   the `wica.agent` logger (with traceback), and the Agent is immediately ready for the next
   trigger. It never surfaces as asyncio's "Task exception was never retrieved".
2. Any exception raised by the **output sink** is caught and logged at **ERROR**; the Commands in
   the same model response are **still dispatched**.
3. History stays **well-formed for every provider**: two Observation records with no assistant
   record between them (a failed call, a call cancelled by `stop()`, or an empty model response)
   render as **one** user message, never two consecutive ones. Nothing recorded in the orphaned
   observation is lost (it can hold the only copy of a retired Command's outcome).
4. Every Agent-owned task has a done-callback safety net that logs an unexpected exception at
   ERROR, so no future code path can fail silently either.
5. [specs/agent.md](../specs/agent.md) describes all of the above and is back to `Implemented`.

**Out of scope** (still deferred — do not build): representing a failed step as a World entry or
history record the model can see, retry/backoff policy, and a new instrumentation Event for
errors. The spec's open question 4 stays open but narrower (see Step 1).

## What the code looks like today (read this before editing)

In [src/wica/agent.py](../src/wica/agent.py):

- `_flush_window` (≈line 644) sets `self._busy = True` and creates the step task via
  `self._track_task(self._run_batch(batch))`.
- `_run_batch` (≈line 659) is just `try: await self._run_step(batch) finally: self._busy = False`.
- `_run_step` (≈line 665) does, in order: emit `on_trigger` per batched entry; `self._append_observation()` (appends an `ObservationRecord` to `self._history` **and retires terminal command entries from the World**); `self._render_messages()`; emit `on_prompt`; `response = await self._bound_model.ainvoke(messages)` (≈line 687); if `response.text`, append `AssistantTextRecord` then `await self._output_sink(text)` (≈line 697); then loop over `response.tool_calls` dispatching each.
- `_track_task` (≈line 557) creates the task and adds only `self._owned_tasks.discard` as a done-callback.
- `_render_messages` (≈line 865) walks `self._history`; for an `ObservationRecord` it calls `flush_assistant()` then builds `blocks` and does `messages.append(HumanMessage(content=blocks))` (≈line 924).

The problem: nothing catches the `ainvoke` or sink exception, so it escapes `_run_batch` into the task; and because `_append_observation()` already ran, the next step's observation lands right after this one.

## Ground rules for the implementer

- Work the steps **in order**. Step 1 (spec) comes first so the status is honest while code lags.
- After **every** code step run `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`,
  `uv run pytest`. Do not move on with a failure.
- Do not change public names or signatures. Do not add dependencies. Do not add a new `Event`.
- Never catch `BaseException` or `asyncio.CancelledError` in the new handlers: cancellation (from
  `stop()` or `cancel_command`) must keep propagating exactly as it does today.
- Test style (from AGENTS.md): drive the public API, assert on observable results (log records via
  `caplog`, the messages the model received via `model.calls`, World entry state), never assert
  that a mock was called. The existing tests in `tests/test_agent.py` show the fixtures to use:
  `loop`, `world`, `sink`, `make_agent`, `ProgrammableChatModel`, `sequence`, `text_response`,
  `tool_call_response`, `wait_until`, and `caplog` (see
  `test_single_in_flight_trigger_dropped_and_logged` for the `caplog.at_level(..., logger="wica.agent")` pattern).

## Step 1 — Spec first: describe the behavior, mark the spec `Updated`

File: [specs/agent.md](../specs/agent.md). Also [specs/_index.md](../specs/_index.md).

1. Change the `**Status:** Implemented` line (line 10) to `**Status:** Updated`, and the
   `agent.md` row's Status cell in `specs/_index.md` (line 24) to `Updated`.
2. In the paragraph starting `**Rendering to messages.**` (line 88), append this sentence at the
   end:

   > **Consecutive Observations merge.** If two Observation records follow each other with no assistant-side record between them — the step's model call raised, was cancelled by `stop()`, or returned neither text nor tool calls — the renderer emits them as **one** user message (the blocks of the older observation followed by the newer one's) rather than two consecutive user-role messages, which some providers' chat templates reject. Each observation keeps its own fresh/archival rendering (only the newest is fresh), so the merge changes nothing about prefix stability, and the older observation's content (which may hold the only record of a retired Command's outcome) is never dropped.

3. In the logging paragraph of "Instrumentation: observability Events" (line 167), replace the
   last sentence

   > Listener and Event-subscriber exceptions are caught, logged, and isolated; model-call and output-sink failures still need a first-class state/history story (see Open questions).

   with

   > **ERROR** for a reasoning step whose model call raised, and for an output sink that raised. Listener and Event-subscriber exceptions are caught, logged, and isolated. A model-call failure ends the step (nothing is dispatched, the Agent is free for the next trigger, and the already-recorded Observation renders merged with the next one — see "Rendering to messages"); an output-sink failure is logged and the step **continues** to dispatch the response's Commands. Neither is yet represented in World state or history as something the model can see (see Open questions). Every Agent-owned task also carries a done-callback that logs any unexpected exception at ERROR, so no step or Command can fail silently.

4. Rewrite open question 4 (line 176) to:

   > 4. **Agent-level errors as model-visible state.** Model-call and output-sink failures are now caught and logged at ERROR, the step ends (or continues, for the sink) cleanly, and history stays well-formed. What is still open is whether such a failure should be **represented to the model** — a World entry or history record it can observe, and a retry/backoff policy — the way Command failures already are via terminal `CommandExecution` state.

5. In "Future improvements", replace the bullet `**Agent-level (non-Command) error handling.** …`
   (line 189) with:

   > - **Agent-level errors as model-visible state.** Failures are logged and contained (see Instrumentation) but not yet represented in World state/history; Command failures are.

Verification: `uv run pytest tests/test_project_map.py` still passes (it checks frontmatter and
statuses, not prose).

## Step 2 — Safety net on every Agent-owned task

File: [src/wica/agent.py](../src/wica/agent.py), `_track_task` (≈line 557).

Add a second done-callback that logs an unexpected exception. Result must look like:

```python
    def _track_task[T](self, coroutine: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Create an Agent-owned loop task and forget it only after it has settled. Every owned
        task also gets a logging done-callback: an exception that escapes a task is otherwise only
        reported by asyncio at garbage-collection time ("Task exception was never retrieved"),
        never under a wica.* logger. See specs/agent.md ("Instrumentation")."""
        task = self._loop.create_task(coroutine)
        self._owned_tasks.add(task)
        task.add_done_callback(self._owned_tasks.discard)
        task.add_done_callback(self._log_task_failure)
        return task

    def _log_task_failure(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            _logger.error("agent task %s raised", task.get_name(), exc_info=exc)
```

Note: `task.exception()` on a task that raised marks the exception as retrieved, which is what
silences the asyncio warning.

Test (add to `tests/test_agent.py`): none needed on its own — Step 3's test asserts the ERROR
record; this net is exercised by any future regression. Run the suite anyway.

## Step 3 — Catch the model-call failure in `_run_step`

File: [src/wica/agent.py](../src/wica/agent.py), `_run_step` (≈line 665).

Wrap only the `ainvoke` line. Result must look like:

```python
        self.on_prompt.emit(copy.deepcopy(messages))
        try:
            response = await self._bound_model.ainvoke(messages)
        except Exception:
            # The observation is already in history (and terminal command entries are retired),
            # so the model never saw it this step; the renderer merges it into the next
            # observation's user message (see _render_messages). Nothing is dispatched. Ending the
            # step here frees the single-in-flight loop for the next trigger. CancelledError is a
            # BaseException and keeps propagating. See specs/agent.md ("Instrumentation").
            _logger.exception(
                "model call failed; step abandoned (trigger: %s)",
                _describe_entry(representative),
            )
            return
```

Everything after (text handling, tool-call loop, the final "step complete" debug log) stays as
it is, now only reached on success.

Test — add to `tests/test_agent.py`:

```python
def test_model_call_failure_is_logged_and_the_next_step_runs(loop, world, sink, caplog):
    world.register("input", str, serialize_fn=identity_serialize, triggers_llm_call=True)

    async def explode(messages):
        raise ConnectionError("provider down")

    model = ProgrammableChatModel(respond=sequence(explode, text_response("recovered")))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()

    with caplog.at_level(logging.ERROR, logger="wica.agent"):
        world.update("input", "first")
        wait_until(lambda: len(model.calls) == 1 and not agent._busy)
        world.update("input", "second")
        assert sink.event.wait(timeout=WAIT_TIMEOUT)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("model call failed" in r.message for r in errors)
    assert any(r.exc_info and isinstance(r.exc_info[1], ConnectionError) for r in errors)
    assert sink.texts == ["recovered"]
```

(`sequence` pops responders in order; `explode` is a plain async callable with the same signature
as the ones `text_response` builds.) The message-shape assertion is Step 5's test.

## Step 4 — Contain an output-sink failure; keep dispatching

File: [src/wica/agent.py](../src/wica/agent.py), `_run_step`, the block starting `if text:`
(≈line 695). Result must look like:

```python
        if text:
            self._history.append(AssistantTextRecord(text))
            try:
                await self._output_sink(text)
            except Exception:
                # The sink is application code; its failure must not drop the Commands the model
                # issued in the same response, nor escape the step. The text stays in history —
                # the model did say it. See specs/agent.md ("Instrumentation").
                _logger.exception("output sink raised; continuing with the step's commands")
```

Test — add to `tests/test_agent.py`:

```python
def test_output_sink_failure_is_logged_and_commands_still_dispatch(loop, world, caplog):
    world.register("input", str, serialize_fn=identity_serialize, triggers_llm_call=True)

    async def bad_sink(text: str) -> None:
        raise RuntimeError("sink broke")

    async def respond(messages):
        return AIMessage(
            content="doing it",
            tool_calls=[{"name": "wave", "args": {}, "id": "w1"}],
        )

    model = ProgrammableChatModel(respond=respond)
    agent = make_agent(model, world=world, loop=loop, output_sink=bad_sink)
    waved = threading.Event()

    async def wave() -> str:
        """Wave."""
        waved.set()
        return "waved"

    agent.register_command(wave)
    agent.start()

    with caplog.at_level(logging.ERROR, logger="wica.agent"):
        world.update("input", "hello")
        assert waved.wait(timeout=WAIT_TIMEOUT)

    assert any("output sink raised" in r.message for r in caplog.records)
    # The utterance is still in history: the next prompt carries it as assistant text.
    wait_until(lambda: len(model.calls) >= 2)  # wave's completion re-triggers a step
    assert any(isinstance(m, AIMessage) and "doing it" in str(m.content) for m in model.calls[1])
```

## Step 5 — Merge consecutive Observations at render time

File: [src/wica/agent.py](../src/wica/agent.py), `_render_messages` (≈line 865), the
`if isinstance(record, ObservationRecord):` branch (≈line 905).

Today the branch ends with `messages.append(HumanMessage(content=blocks))`. Change it so that,
when the previous message in `messages` is a `HumanMessage` **and** nothing assistant-side was
pending (i.e. `flush_assistant()` appended nothing), the new blocks are appended to that existing
message instead of creating a new one. Concretely, replace the branch body's ending with:

```python
                if messages and isinstance(messages[-1], HumanMessage):
                    # Consecutive observations (a failed/cancelled/empty step in between) merge
                    # into one user message — some providers reject back-to-back user turns, and
                    # the older observation may hold the only record of a retired Command's
                    # outcome. See specs/agent.md ("Rendering to messages").
                    previous = messages[-1]
                    merged = list(previous.content) if isinstance(previous.content, list) else [previous.content]
                    messages[-1] = HumanMessage(content=[*merged, *blocks])
                else:
                    messages.append(HumanMessage(content=blocks))
```

`flush_assistant()` is still called first in the branch; when it appended an `AIMessage`/
`ToolMessage`s, `messages[-1]` is not a `HumanMessage` and the normal path runs. The
`archival = i != newest_observation_index` logic is untouched: the older observation in a merged
message renders archival, the newest fresh.

Tests — add to `tests/test_agent.py` (one per cause):

```python
def _no_consecutive_human_messages(messages: list[BaseMessage]) -> bool:
    return not any(
        isinstance(a, HumanMessage) and isinstance(b, HumanMessage)
        for a, b in zip(messages, messages[1:])
    )


def test_failed_step_observation_merges_into_the_next_prompt(loop, world, sink):
    world.register("input", str, serialize_fn=identity_serialize, triggers_llm_call=True)

    async def explode(messages):
        raise ConnectionError("provider down")

    model = ProgrammableChatModel(respond=sequence(explode, text_response("ok")))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()
    world.update("input", "first")
    wait_until(lambda: len(model.calls) == 1 and not agent._busy)
    world.update("input", "second")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)

    second = model.calls[1]
    assert _no_consecutive_human_messages(second)
    humans = [m for m in second if isinstance(m, HumanMessage)]
    assert len(humans) == 1
    text = str(humans[0].content)
    assert "first" in text and "second" in text  # both observations, one message


def test_empty_model_response_does_not_split_the_next_prompt(loop, world, sink):
    world.register("input", str, serialize_fn=identity_serialize, triggers_llm_call=True)
    model = ProgrammableChatModel(respond=sequence(text_response(""), text_response("ok")))
    agent = make_agent(model, world=world, loop=loop, output_sink=sink)
    agent.start()
    world.update("input", "first")
    wait_until(lambda: len(model.calls) == 1 and not agent._busy)
    world.update("input", "second")
    assert sink.event.wait(timeout=WAIT_TIMEOUT)
    assert _no_consecutive_human_messages(model.calls[1])
```

Also extend the existing `tests/test_wica.py::test_stop_cancels_reasoning_before_a_restart`: after
the restart's step completes, assert `_no_consecutive_human_messages(...)` over the messages the
second cycle's `on_agent_prompt` delivered (subscribe a list-appender to `wica.on_agent_prompt`
before `start()`). This pins the cancelled-mid-call cause.

## Step 6 — Flip statuses and index

1. `specs/agent.md` line 10: `**Status:** Updated` → `**Status:** Implemented`; the `agent.md`
   row in `specs/_index.md` back to `Implemented`.
2. This plan: `**Status:** Todo` → `**Status:** Done`, and its row in
   [plans/_index.md](_index.md) to `Done`.

Final verification: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`,
`uv run pytest`, and `uv run pytest tests-e2e -k "fake or example"` all pass.
