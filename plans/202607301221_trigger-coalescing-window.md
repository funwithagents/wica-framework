# Trigger coalescing window

**Status:** Done

Implements the [agent.md](../specs/agent.md) "Trigger coalescing" section and the resolution of [world.md](../specs/world.md) open question #1: the Agent batches a burst of triggers arriving close together into a **single** step, via a fixed leading-edge **coalescing window**, with a per-entry `bypass_coalescing` escape hatch declared at `register()`.

Flips [world.md](../specs/world.md) and [inputs.md](../specs/inputs.md) back from `Updated` to `Stable` once done.

## Background

Today `World.update()` → trigger handler → `Agent._handle_trigger` starts a step **immediately**, and v1's single-in-flight loop **drops** any trigger arriving while `_busy`. So a burst of near-simultaneous Inputs is actively harmful: the first trigger's step snapshots the World (`get_prompt_entries()`) *before* the rest of the burst lands, and those later triggers are dropped — their changes lost until an unrelated later trigger. A short window that lets the burst settle before the single step snapshots the World fixes this "for free," because a step already observes the *whole* included World, not just the entry that fired.

## Design decisions (settled with the user)

- **Fixed leading-edge window, not reset-on-each-trigger debounce.** The first trigger opens the window; later triggers *join* it but do **not** extend it → latency bounded by the window, never starved by a fast stream.
- **Global window only**, `coalesce_window` (seconds) on the Agent, **default `0.2`**. `coalesce_window=0` disables coalescing — each trigger fires immediately, exactly the pre-coalescing behavior (a zero window is a window that closes at once; no separate code path).
- **Per-entry `bypass_coalescing`** (World `register()` field, default `False`): an update to such an entry flushes the current window **early**, carrying along whatever's already batched. Skips the *wait*, not the *in-flight drop*.
- **Still drop-when-busy.** Coalescing only batches the arrival burst *before a step starts*. A trigger arriving *while a step is in flight* is still dropped, exactly as v1 does now. Collecting during-flight triggers is the deferred concurrency/queue question — out of scope.
- **`on_trigger` fires once per collected trigger**, but the Observation record, `on_prompt`, and the model call happen **once** for the coalesced step.
- **All window manipulation on the loop thread** (open/join/flush/cancel), same invariant that makes cancellation race-free — the World→loop bridge still just schedules `_handle_trigger` onto the loop.

## Steps

### World — carry the declaration (`src/wica/world.py`)

1. **`WorldEntryConfig`**: add `bypass_coalescing: bool = False`.
2. **`WorldEntry`**: add `bypass_coalescing: bool` (like `type`, it's config copied onto the snapshot so the trigger handler is self-describing; never changes across updates).
3. **`register()`**: add a keyword-only `bypass_coalescing: bool = False` param; store it in the `WorldEntryConfig`; pass it to the initial `WorldEntry`. Optionally add it to the registration debug log line.
4. **`_update()`**: carry `bypass_coalescing` onto the new `WorldEntry` from `old_entry.bypass_coalescing` (mirrors how `type=old_entry.type` is carried).

The World does **not** act on the flag — it only stores/carries it. No change to trigger dispatch.

### Agent — own the window (`src/wica/agent.py`)

5. **Constructor**: add `coalesce_window: float = 0.2`; store `self._coalesce_window`. Add window state: `self._window_batch: list[WorldEntry] = []` and `self._window_timer: asyncio.TimerHandle | None = None`. `self._busy` stays.

6. **`_handle_trigger(entry)`** — rewrite around the window (still async; runs on the loop thread):
   - If `self._busy`: log + drop, return (unchanged in-flight-drop behavior; keep the existing comment about not retiring a dropped command completion).
   - Append `entry` to `self._window_batch`.
   - If `entry.bypass_coalescing or self._coalesce_window <= 0`: call `self._flush_window()` (fire now).
   - Elif `self._window_timer is None`: `self._window_timer = self._loop.call_later(self._coalesce_window, self._flush_window)` (open the window).
   - Else: window already open — the entry just joined the batch; do nothing.

7. **`_flush_window(self) -> None`** (sync, loop-thread): cancel + clear `self._window_timer` if set; if `self._window_batch` is empty, return; take and clear the batch, set `self._busy = True`, and `self._loop.create_task(self._run_batch(batch))`. Setting `_busy` here (before the task runs) is what makes triggers arriving between flush and step-start get dropped.

8. **`_run_batch(self, batch) -> None`** (async): `try: await self._run_step(batch)` / `finally: self._busy = False`. (Replaces the busy-management that used to live in `_handle_trigger`.)

9. **`_run_step`**: change signature to take `batch: list[WorldEntry]`. Fire `on_trigger` **once per entry** in `batch` (loop over `self._fire_hook(self._on_trigger, e)`), use `batch[-1]` as the representative entry for the step-start/complete debug logs (log the batch size when `len(batch) > 1`). The rest is unchanged — one `_append_observation()`, one `_render_messages()`, one `on_prompt`, one `ainvoke`.

10. **`_append_observation`**: drop its unused `entry` parameter (it only ever calls `get_prompt_entries()`); update its call site.

11. **`stop()`**: if `self._window_timer is not None`, cancel it and clear it (hygiene — the loop is stopping anyway, but don't leave a dangling handle).

No change to `_on_world_trigger` (still `run_coroutine_threadsafe(self._handle_trigger(entry), self._loop)`).

## Tests

### World (`tests/test_world.py`)

- **`bypass_coalescing` defaults to `False`** on a plain `register()`, and is `True` on the `WorldEntry` when registered `bypass_coalescing=True` — both on the initial entry and after an `update()` (carried across versions).

### Agent (`tests/test_agent.py`, using `FakeChatModel` + `wait_until`)

- **A burst coalesces into one step.** With a modest window (e.g. `coalesce_window=0.2`), register two `triggers_llm_call` Inputs, `update()` both back-to-back, then wait for the step. Assert: exactly **one** model call (`len(model.calls) == 1`, and no second appears after waiting past the window), the single Observation/prompt contains **both** entries' values, and `on_trigger` fired **twice** (once per input).
- **`coalesce_window=0` preserves today's behavior.** Each `update()` starts its own immediate step (two spaced updates → two calls), and a trigger fired while a step is in flight is dropped (use a gate — e.g. an `asyncio.Event` the fake model awaits — to hold step 1 open, fire a second update, release, assert only one call ran for the burst).
- **`bypass_coalescing` flushes early.** With a deliberately long window (e.g. `1.0`), an update to a `bypass_coalescing=True` entry runs the step **promptly** (well under the window) rather than waiting it out; a non-bypass update to a normal entry made just before it is carried into the same single step's observation (early flush pulls the batch forward).
- **Drop-when-busy still holds** for a trigger arriving during an in-flight step (bypass or not) — no second step, matching the existing single-in-flight guarantee.

## Verification

`uv run ruff check .`, `uv run pyright`, `uv run pytest` all green. Then:
- Flip this plan to `Done` (here and in [plans/_index.md](_index.md)).
- Flip [world.md](../specs/world.md) and [inputs.md](../specs/inputs.md) back to `Stable` (status line + [specs/_index.md](../specs/_index.md) rows), since design and code are back in sync.
