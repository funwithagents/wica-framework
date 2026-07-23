# Agent v1 implementation

**Status:** Done

Implements a first working slice of the Agent reasoning loop ([specs/agent.md](../specs/agent.md)), on top of the shipped `Content` model (`src/wica/content.py`) and World registry (`src/wica/world.py`). `agent.md` is Draft with several open questions still unresolved; this plan deliberately scopes v1 to a subset and records the concrete calls made to fill the gaps that block writing code at all. Everything else stays deferred (see "Out of scope / deferred").

Builds on [plans/202607231754_e2e-test-framework.md](202607231754_e2e-test-framework.md) for the e2e tier (see "E2E tests" below) — that plan should land first.

## Decisions made for v1 (confirm before implementing)

Resolved with the user before drafting:

1. **Output = free-text-as-speech** (agent.md Q2). Visible assistant text is the utterance; no `speak()` tool in v1.
2. **Concurrency = single in-flight call** (agent.md "Concurrency & interruption", Q3). v1 does not implement `agent:activity` entries, `cancel_activity`, or genuine concurrent calls. A trigger arriving while a call is in flight is **dropped and logged**: no step is started for it, no history record is created for it, and a log line records the key/id that was dropped so it's visible rather than silently lost.
3. **Parallel tool calls: N independent World entries.** A response requesting multiple tool calls registers one tool-status entry per call; whichever finishes first re-triggers a step while others keep running (still bounded by "single in-flight" above — only one step runs at a time, but multiple tools can be mid-flight).
4. **A "step" is exactly one LLM call** (clarified in discussion, not a separate open question). The event-driven World-entry tool lifecycle is the spec's stated *default* for all tools, not an optimization — so v1 never loops within a step. Dispatching a tool call always ends the step; the tool's own completion is what starts the next one. This makes agent.md open question #6 ("step termination") moot for v1: there is no multi-call loop to bound yet. The deferred sync/async optimization (agent.md "Sync vs. async") is what would eventually reintroduce a within-step loop, and stays deferred here.
5. **Async tool execution is in scope, not deferred.** Per the "Tool lifecycle as a World entry" section, this is core v1 work: dispatch → World tool-status entry (`running`) + history description → step ends → tool runs as a cancellable asyncio task on the Agent's loop → completion updates the entry to terminal → World's trigger fires a new step.

New calls made to fill spec gaps not covered by the discussion above (flagged for review, not blocking — reasonable defaults, easy to revise):

6. **History captures the full `include_in_prompt` World state per trigger, not just the entry that fired** — now written into the specs as the resolution of world.md Q2 / agent.md Q4/Q9 (not just a plan-local choice). Rationale: an entry registered `include_in_prompt=True, triggers_llm_call=False` (passive context, e.g. set once and meant to just sit in the prompt) would otherwise never reach the model at all under a strictly-incremental, entry-that-fired-only scheme — which defeats the purpose of `include_in_prompt`. So each `ObservationRecord` wraps the **full bundle** of currently-included entries (via the new `World.get_prompt_entries()`, added by this plan — see Scope), captured at trigger time. The newest bundle in history renders every entry in it fresh; every older bundle renders every entry in it archival — freshness flips at the bundle boundary. This still uniformly covers a tool-status entry going terminal (it's simply part of the next bundle, alongside everything else) — matching agent.md's "it sees the result in world state" framing for tool completion.
7. **Freshness policy: the single newest `ObservationRecord` in history is fresh; every earlier one is archival.** Simplest instance of the Agent-owned policy agent.md leaves open (Q5).
8. **Tool-call history has two records, both already anticipated by agent.md open question #9's closing note** ("captured both as the history description and, as a snapshot"): a `ToolCallRecord(name, args)` written at dispatch time (no result yet — matches agent.md's "calling tool xxx…" phrasing), and, separately, the tool-status entry's own terminal update arrives later as an `ObservationRecord` whose rendered text carries the result (`Called x(args) → result`). No record is ever mutated after creation.
9. **Tool-status World keys are unregistered once their terminal update has been folded into history.** Not spec'd either way; done to avoid the World's registry growing one permanent key per tool call ever made. The immutable history record already retains the snapshot, so nothing is lost.
10. **Streaming and live-progress-to-World during speech are deferred**, since agent.md ties both to the concurrency model (a sink "updates a World entry as it speaks" so a *concurrent* call has something to judge against) — moot with a single in-flight call. v1's sink receives the complete text once a step finishes producing it.
11. **The running tool's `asyncio.Task` handle lives in Agent-private state (`self._running_tasks`), never in the World.** "Tool execution is async and cancellable" is a *Settled* requirement (agent.md "Tools"), not deferred — but the task object itself isn't World-value shaped (not serializable, not safe to touch off its own loop thread), and the spec's "all task manipulation happens on the loop thread" principle already says this is Agent-loop state, not World state. `create_task()` followed immediately by the dict assignment, with no `await` between them, is sufficient to make the task cancellable before it has run any of its body — asyncio has no separate "create but don't start" step; a created task's body cannot run until the current synchronous stretch of code yields.

## Scope

- `pyproject.toml` — add `langchain-core` and `langchain` to `dependencies`. Provider packages (`langchain-anthropic`, `langchain-openai`, …) are **not** added — they're deployment-time extras per whichever provider is configured, consistent with "switching providers is a config change."
- `src/wica/world.py` — **small addition**: `World.get_prompt_entries() -> list[WorldEntry]`, the same `include_in_prompt` selection `render_full_prompt()` already computes internally, exposed as raw entries so the Agent can apply its own fresh/archival choice at render time (see decision #6; now documented in [specs/world.md](../specs/world.md) World API).
- `src/wica/agent.py` — **new**: `Agent`, `AgentConfig`, history record types, the `Content ↔ LangChain messages` adapter, the tool-status value model, the event-loop bridge. This is the sole module importing LangChain (per "LangChain quarantined to the Agent's I/O boundary").
- `src/wica/__init__.py` — export `Agent`, `AgentConfig`.
- `tests/test_world.py` — add coverage for `get_prompt_entries()`.
- `tests/test_agent.py` — **new**, functional tests against a hand-written fake chat model (full control over responses/tool-calls, no network).
- `tests-e2e/test_agent.py` — **new**, a small live tier on top of [plans/202607231754_e2e-test-framework.md](202607231754_e2e-test-framework.md)'s `real_chat_model()`/`require_env` (see "E2E tests" below). Lives outside `tests/`, so the default `uv run pytest` never collects it; run via `uv run pytest tests-e2e`.

## Data model (`src/wica/agent.py`)

```python
@dataclass
class AgentConfig:
    provider: str                 # passed to init_chat_model's model_provider=
    model: str
    system_prompt: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ToolCallStatus:
    name: str
    args: dict[str, Any]
    state: Literal["running", "complete", "failed", "cancelled"]
    result: str | None = None
    error: str | None = None

    def is_terminal(self) -> bool:
        return self.state != "running"

# History records — see decision #6/#8 above
@dataclass(frozen=True)
class ObservationRecord:
    entries: list[WorldEntry]     # the full include_in_prompt bundle at trigger time, not just
                                   # the entry that fired (decision #6)

@dataclass(frozen=True)
class AssistantTextRecord:
    text: str

@dataclass(frozen=True)
class ToolCallRecord:
    call_id: str
    name: str
    args: dict[str, Any]

HistoryRecord = ObservationRecord | AssistantTextRecord | ToolCallRecord
```

## `Agent` API

```python
class Agent:
    def __init__(
        self,
        model: BaseChatModel,
        *,
        system_prompt: str,
        world: World | None = None,          # defaults to get_world()
        loop: AbstractEventLoop | None = None,  # None -> owns a daemon-thread loop
        output_sink: Callable[[str], Awaitable[None]] | None = None,
    ) -> None: ...

    @classmethod
    def from_config(cls, config: AgentConfig, **kwargs) -> Agent:
        """Builds the model via init_chat_model(config.model, model_provider=config.provider,
        **config.model_kwargs) and forwards system_prompt + kwargs to __init__."""

    def register_tool(self, fn: Callable[..., Any] | BaseTool, *, name: str | None = None,
                       description: str | None = None) -> None: ...

    def start(self) -> None:
        """world.set_trigger_handler(self._on_world_trigger); starts the owned loop thread if any."""

    def stop(self) -> None:
        """world.set_trigger_handler(None) FIRST (so shutdown cleanup below can't itself kick off
        a new step), then cancels every still-running tool task via cancel_tool_call, then stops
        the owned loop thread if any."""

    def cancel_tool_call(self, call_id: str) -> None:
        """Cancels a running tool task by call_id, if still running; no-op otherwise (id-guarded,
        same pattern as the World's TTL expected_id guard). Must run on the loop thread — schedules
        itself via call_soon_threadsafe if called from elsewhere."""
```

- `register_tool` accepts either a plain callable or an already-built `BaseTool` (off-the-shelf LangChain tools work unmodified, per spec's "Tools" section); wraps plain callables with LangChain's `tool()` helper. Stored in `self._tools: dict[str, BaseTool]`; `self._bound_model = self.model.bind_tools(list(self._tools.values()))` is recomputed on each `register_tool` call (rare, not hot-path).
- `output_sink` defaults to a no-op (or a `print`-based stub) if `None`, so `Agent` is usable standalone in tests without wiring real TTS.

## Event loop bridge

Per agent.md "The Agent owns the event loop":

```python
def _on_world_trigger(self, entry: WorldEntry) -> None:      # runs on World's executor thread
    asyncio.run_coroutine_threadsafe(self._handle_trigger(entry), self._loop)
```

`_handle_trigger` and everything downstream of it runs on the Agent's loop thread — this is where the single-in-flight guard (`self._busy`) lives, safely, with **no lock needed** (single-threaded event loop, matches the spec's rationale for wanting the loop at all):

```python
_logger = logging.getLogger(__name__)

async def _handle_trigger(self, entry: WorldEntry) -> None:
    if self._busy:
        _logger.info("dropping trigger for %r (id=%d) — a call is already in flight",
                      entry.key, entry.current.id)
        return
    self._busy = True
    try:
        await self._run_step(entry)
    finally:
        self._busy = False
```

## `_run_step` — the one-call step

```python
async def _run_step(self, entry: WorldEntry) -> None:
    self._append_observation(entry)          # captures world.get_prompt_entries() as one
                                               # ObservationRecord bundle (decision #6); also drops
                                               # entry's own tool-status key from the World if
                                               # terminal, once captured (decision #9)
    messages = self._render_messages()
    response = await self._bound_model.ainvoke(messages)

    if response.content:
        text = _flatten_text(response.content)
        self._history.append(AssistantTextRecord(text))
        await self._output_sink(text)

    for call in response.tool_calls:          # [] if none
        self._history.append(ToolCallRecord(call["id"], call["name"], call["args"]))
        self._dispatch_tool_call(call["id"], call["name"], call["args"])
```

`_dispatch_tool_call` registers the World entry and starts the background task:

```python
def _dispatch_tool_call(self, call_id: str, name: str, args: dict[str, Any]) -> None:
    key = f"agent:tool_call:{call_id}"
    self._tool_call_keys.add(key)
    world.register(
        key, ToolCallStatus,
        serialize_fn=_serialize_tool_call_status,
        include_in_prompt=True,
        triggers_llm_call=True,
        trigger_condition_fn=lambda old, new: new is not None and new.is_terminal(),
    )
    world.update(key, ToolCallStatus(name=name, args=args, state="running"))
    task = self._loop.create_task(self._run_tool(key, call_id, name, args))
    self._running_tasks[key] = task           # no `await` since create_task() above — the task
                                               # cannot have run any of its body yet, so this is
                                               # already race-free (decision #11)
```

```python
async def _run_tool(self, key: str, call_id: str, name: str, args: dict[str, Any]) -> None:
    tool = self._tools[name]
    try:
        result = await tool.ainvoke(args)
        status = ToolCallStatus(name=name, args=args, state="complete", result=str(result))
    except asyncio.CancelledError:
        world.update(key, ToolCallStatus(name=name, args=args, state="cancelled"))
        raise                                  # convention: let the task actually finish cancelled
    except Exception as exc:                   # tool failure surfaces into World + history (partial
        status = ToolCallStatus(name=name, args=args, state="failed", error=str(exc))  # answer to Q7
    else:
        world.update(key, status)               # terminal -> triggers the next step
    finally:
        self._running_tasks.pop(key, None)
```

```python
def cancel_tool_call(self, call_id: str) -> None:
    key = f"agent:tool_call:{call_id}"
    task = self._running_tasks.get(key)
    if task is not None:
        task.cancel()                           # id-guarded implicitly: a finished task is already
                                                 # popped from _running_tasks by _run_tool's finally,
                                                 # so cancelling a stale call_id is just a dict miss
```

`world.update(key, status)` runs on the loop thread (we're inside a task on `self._loop`); the World dispatches the trigger handler on *its own* executor thread regardless of caller thread, which calls `_on_world_trigger` → bridges back onto `self._loop` via `run_coroutine_threadsafe` — the same uniform path as any externally-sourced trigger, no special-casing. The one exception is during `stop()`: the trigger handler is cleared *before* in-flight tasks are cancelled, so the resulting `"cancelled"` update can't kick off a step nobody will service.

`_append_observation(entry)`:

```python
def _append_observation(self, entry: WorldEntry) -> None:
    self._history.append(ObservationRecord(world.get_prompt_entries()))
    if entry.key in self._tool_call_keys and entry.current.value.is_terminal():
        world.unregister(entry.key)
        self._tool_call_keys.discard(entry.key)
```

The bundle is captured *before* the cleanup unregister, so the terminal tool-status entry is still present in `get_prompt_entries()`'s result for this bundle — the snapshot is safely retained in history first, then the live World key is dropped (decision #9).

## `Content ↔ LangChain` adapter

- `_content_to_message_blocks(content: Content) -> list[dict]` — maps `TextPart` → a text block, `ImagePart` → a multimodal image block. **To verify against the installed `langchain-core` version's standard content-block schema at implementation time** (the block shape has changed across langchain-core releases — v0.3+ uses `{"type": "image", "source_type": "base64", "data": ..., "mime_type": ...}`-style blocks; confirm against the pinned version before writing this).
- `_render_messages(self) -> list[BaseMessage]`:
  - `SystemMessage(self.system_prompt)` first.
  - Walk `self._history`, grouping consecutive `AssistantTextRecord`/`ToolCallRecord` entries (same step) into one `AIMessage` (text + rendered `"Calling {name}({args})…"` lines), and rendering each `ObservationRecord` as its own `HumanMessage`. For the single newest `ObservationRecord` in the whole list, every entry in its bundle renders fresh (`world.render_entry(e)` for each `e` in `record.entries`); for every earlier `ObservationRecord`, every entry in *its* bundle renders archival (`world.render_entry(e, archival=True)`) — freshness flips at the bundle boundary, not per entry (decision #6/#7).
  - Each `ObservationRecord`'s `HumanMessage.content` is the concatenation of `_content_to_message_blocks(world.render_entry(e, archival=...))` across all `e` in `record.entries`, in the entries' timestamp order (same ordering `render_full_prompt()` uses).

## Implementation steps

1. `pyproject.toml`: add `langchain-core`, `langchain` deps; `uv sync --dev`.
2. `src/wica/world.py`: add `get_prompt_entries()`; `tests/test_world.py` coverage for it.
3. `src/wica/agent.py`: data model (`AgentConfig`, `ToolCallStatus`, history records), `_serialize_tool_call_status`, `_content_to_message_blocks`.
4. `Agent.__init__` / `from_config` / `register_tool` / `start` / `stop`.
5. Event loop bridge (`_on_world_trigger`, `_handle_trigger` with the busy-drop guard + log line).
6. `_render_messages` (history → `BaseMessage` list, bundle-level freshness policy).
7. `_run_step`, `_dispatch_tool_call`, `_run_tool` (incl. `CancelledError` handling), `cancel_tool_call`, `_append_observation` (full-bundle capture + tool-key cleanup).
8. Exports in `__init__.py`.

## Tests (`tests/test_agent.py`)

All functional, driving `Agent` through the real `World` singleton (reset via the existing `conftest.py` fixture) with a hand-written fake chat model — a minimal `BaseChatModel` subclass whose `ainvoke`/`bind_tools` are scripted per-test to return canned `AIMessage`s (text-only, tool-call-only, or both), giving deterministic control over multi-step sequences without network calls or relying on `langchain_core`'s built-in fakes (which don't give enough control over `.tool_calls`).

- Text-only response: a World trigger → `Agent` calls the model once → output sink receives the response text; history gets one `ObservationRecord` + one `AssistantTextRecord`.
- Tool-call response: trigger → model returns a tool call → a World entry `agent:tool_call:<id>` appears with `state="running"` and is `include_in_prompt` → the step ends *without* a second model call → the registered tool (a fake async tool) resolves → its entry updates to `state="complete"` → this automatically triggers a second model call (assert the fake model was called twice, and the second call's rendered messages include the "Called …" result text) → the tool-status key is gone from the World afterward (`get_entry` raises `KeyError`).
- Tool failure: fake tool raises → entry ends up `state="failed"` with the error, and the next step's rendered messages reflect it.
- Parallel tool calls: a response with two tool calls → two independent World keys appear; the faster one completing triggers a step while the second is still `running` (assert the rendered messages for that step show one `Called …` and one `Calling …(running)`-style line, per decision #3).
- Single-in-flight drop: hold the fake model's first call open (an `asyncio.Event` it awaits before returning) and fire two more triggers while it's pending — assert the fake model is called only once for those three triggers combined (the two dropped ones never start a step or add a history record), and that each drop produced a log line (via `caplog`).
- Full-bundle capture: register two entries (`a`, `b`) with `include_in_prompt=True`, only one of which (`a`) has `triggers_llm_call=True`; update `a` to fire a trigger — assert the resulting `ObservationRecord` bundle contains *both* `a` and `b` (not just `a`), proving passive `include_in_prompt` entries reach history even though they never fire (decision #6's core rationale).
- Freshness: after two sequential triggers, render messages and assert every entry in the second (newest) `ObservationRecord`'s bundle used the fresh serializer, while every entry in the first (older) bundle's message used archival (distinguish via a registered entry whose fresh/archival serializers return visibly different text) — freshness flips at the bundle boundary, not per entry.
- `stop()` clears the World's trigger handler (`get_world()` no longer calls into the agent on a subsequent `update()`).
- Cancellation: dispatch a tool call whose fake implementation blocks on an `asyncio.Event`; call `cancel_tool_call(call_id)` → the task raises `CancelledError`, the World entry ends up `state="cancelled"`, and (with the trigger handler still attached) this fires a new step same as any other terminal state. A `cancel_tool_call` with an unknown/already-finished `call_id` is a no-op (no exception). Separately: `stop()` on an Agent with a still-running tool task cancels it without triggering a new step (trigger handler already cleared first).

## E2E tests (`tests-e2e/test_agent.py`)

Per [specs/project.md](../specs/project.md) "Live/e2e tests" and [plans/202607231754_e2e-test-framework.md](202607231754_e2e-test-framework.md): lives under `tests-e2e/`, so it's excluded from the default `uv run pytest` run by directory alone (no marker needed), opt-in via `uv run pytest tests-e2e`, and each test skips (not fails) without `WICA_ANTHROPIC_API_KEY` set (via `real_chat_model()`'s `require_env`). The fake-model tests above already cover every control-flow branch of the loop; these two exist only to catch what a fake model can't — a wrong message shape or a tool schema the real provider actually rejects. Assertions are behavioral, never exact-text, since real output varies run to run.

- **Plain text round-trip:** build a real `Agent` via `Agent(real_chat_model(), system_prompt="You are a terse test assistant.")`, `start()` it, register no tools, fire a trigger with a simple text entry (e.g. `include_in_prompt=True` entry saying "Say hello in one short sentence."). Assert the output sink was called with non-empty text within a reasonable timeout (a few seconds) — proves the `Content → LangChain message` direction and a bare `ainvoke()` actually round-trip against the real API.
- **Real tool-calling round-trip:** register one trivial real tool (e.g. `add(a: int, b: int) -> int`) via `register_tool`, fire a trigger whose observation text asks the model to use it (e.g. "What is 2 + 2? Use the add tool."). Assert, within a timeout: (a) a `agent:tool_call:<id>` World entry actually appears with `state="running"` — proving the model produced a real LangChain tool call our binding correctly exposed and the provider's response shape parses; (b) it reaches `state="complete"` with a `result` containing `"4"`; (c) the output sink eventually receives text (the model's final answer after seeing the tool result) — proving the full async dispatch → World-entry → re-trigger → second-call loop works against a live provider end to end, not just the fake.

Both tests poll/await with a generous timeout (e.g. `asyncio.wait_for(..., timeout=15)` on an `asyncio.Event` set by the output sink / a listener on the tool-call key) rather than a fixed `sleep`, since real API latency varies.

## Out of scope / deferred

- Full concurrency/interruption model — `agent:activity` entries, `cancel_activity`, bounded concurrency, atomic take-over (agent.md "Concurrency & interruption", Q3). v1 is single-in-flight; a trigger arriving while busy is dropped and logged, not queued or coalesced.
- Sync/async tool optimization — inline fast-path handling for quick tools (agent.md "Sync vs. async (deferred optimization)", part of Q1).
- Streaming output and live-progress World updates during speech (tied to concurrency, per decision #10 above).
- `speak()` tool alternative (Q2) — free-text-as-speech only, per decision #1.
- Config loading from env/file — `AgentConfig` is constructed directly by the caller; no file/env source (agent.md "Provider-agnostic model, from config" leaves source TBD).
- Context-window growth / truncation of `self._history` (Q8) — unbounded for v1.
- Agent-level (not tool-level) error handling — a model call raising is not yet given a World-state/history story (remaining edge of Q7); v1 lets it propagate/log, to revisit with the project's broader logging story.
- Multiple agents / partitioning (Q10) — one `Agent`, one World, as today.
