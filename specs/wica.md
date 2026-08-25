---
code:
  - src/wica/wica.py
tests:
  - tests/test_wica.py
  - tests-e2e/test_wica.py
  - tests-e2e/test_fake_flows.py
---

# Wica (the facade)

**Status:** Implemented

## Purpose

`Wica` is the framework's **single entry point** — one object that stands up the whole system from a `WicaConfig` and exposes everything a consumer needs behind one surface. Without it, a consumer wires the internals by hand: hold the event loop, own a `World` on it, construct and configure an `Agent` against that World and loop, apply logging, register commands, and juggle three separate instrumentation callbacks. `Wica` folds all of that into one class whose job is to own the loop and the `World` + `Agent` pair, wire them together, run their shared lifecycle, and surface the World's and Agent's `Event`s as one public, multi-subscriber instrumentation surface.

This is the "wrap all the APIs in WICA so it is the only interface" item — the consumer talks to `Wica` (and, for World-schema work, to `wica.world`), not to the framework's internals.

## Settled

### What `Wica` owns and exposes

A `Wica` instance owns exactly one `World` and one `Agent`, constructed together and wired to each other. Its surface:

| Member | Kind | Role |
|---|---|---|
| `Wica.init(config, *, output_sink=…, coalesce_window=…, loop=…)` | classmethod | Build a `World` + `Agent` from a `WicaConfig`, wire them, apply logging, return the `Wica`. The one construction path. |
| `wica.world` | attribute (`World`) | The owned World — the home for **all** World-schema work (`register`/`update`/`get`/listeners). Not duplicated onto `Wica`. |
| `wica.agent` | attribute (`Agent`) | The owned Agent. Directly reachable, but the common paths (command registration, lifecycle, instrumentation) are surfaced on `Wica` so a consumer rarely needs it. |
| `wica.start()` / `wica.stop()` | methods | The shared lifecycle — start/stop the World and Agent together (see "Lifecycle"). |
| `wica.register_command(fn, *, name=…, description=…)` | method | Delegates to `agent.register_command` — the one convenience method that *is* mirrored onto `Wica`, since it's part of the everyday setup flow. |
| `wica.on_world_trigger` | `Event[WorldEntry]` | The World's **raw** trigger — fires on every qualifying update, pre-coalescing (= `world.on_trigger`). |
| `wica.on_agent_trigger` | `Event[WorldEntry]` | The Agent's **filtered** trigger — fires once per trigger a run-to-completion step actually observes (= `agent.on_trigger`). |
| `wica.on_agent_prompt` | `Event[list[BaseMessage]]` | Fires with the exact rendered messages before each model call (= `agent.on_prompt`). |
| `wica.on_agent_command` | `Event[CommandIssued]` | Fires with each Command the model issues, at dispatch time (= `agent.on_command`). |

The **asymmetry is deliberate**: command registration is mirrored onto `Wica` because it's part of everyday setup, but the World's own API (registering and updating entries) stays on `wica.world`. Duplicating the entire World surface onto `Wica` would be churn with no payoff — `wica.world.register(...)` / `wica.world.update(...)` reads clearly and keeps `World` the single home for its own concept.

### Construction: `Wica.init`

`Wica.init(config: WicaConfig, …)` is the sole constructor. It:

1. Creates the single asyncio event **loop** the whole system runs on (or adopts one passed in — see "The event loop, and symmetric `start()`/`stop()`").
2. Builds a fresh `World(loop)` (no global — see [world.md](world.md)).
3. Builds the `Agent` from `config.agent`, injecting the same `loop` and the owned World: `Agent(config.agent, world=self.world, loop=self._loop, output_sink=…, coalesce_window=…)`. It then **surfaces the Events** rather than adapting hooks (below): `self.on_world_trigger = self.world.on_trigger`, `self.on_agent_trigger = self.agent.on_trigger`, `self.on_agent_prompt = self.agent.on_prompt`, `self.on_agent_command = self.agent.on_command`.
4. Calls `apply_logging(config.logging)` **once**, so the caller no longer does it by hand. Logging is set on the process-global `wica` logger tree, so it survives for the life of the process regardless of later object churn (see [config.md](config.md), "Flow into the Agent").

The code-only wiring a JSON file can't express — `output_sink`, `coalesce_window`, and optionally a `loop` — are keyword arguments to `init`. The file carries provider/model/key/prompt/logging; `init`'s kwargs carry the callables and runtime objects.

`init` takes an **already-loaded `WicaConfig`**, not a path. There is deliberately **no `Wica.from_json`**: loading is one line (`WicaConfig.from_json(path)`, which already exists — see [config.md](config.md)) and `init` needs several code-only kwargs besides the config, so a path-taking convenience would save nothing and hide the config object the caller often wants. The startup shape stays two honest calls:

```python
config = WicaConfig.from_json(path)
wica = Wica.init(config, output_sink=my_sink, coalesce_window=0.2)
wica.world.register("speech_input", str, serialize_fn=…, triggers_llm_call=True)
wica.register_command(dance)
wica.start()
```

### The event loop, and symmetric `start()`/`stop()`

**`Wica` owns the one event loop and the daemon thread that runs it**, and injects that loop into both the `World` and the `Agent` (see [world.md](world.md), "The shared event loop"; [agent.md](agent.md), "The shared event loop"). This is the symmetry that makes the system easy to reason about: a **single background execution context with a single owner** — not a World-owned thread pool plus a separate Agent-owned loop bridged together. `World` and `Agent` are pure consumers of the injected loop; neither creates or tears down a thread of its own.

`World` and `Agent` share a symmetric `start()`/`stop()` lifecycle, and `Wica` drives both plus the loop thread:

- **`wica.start()`** → start the loop thread (if `Wica` owns it), then `world.start()`, then `agent.start()`.
- **`wica.stop()`** → `agent.stop()`, then `world.stop()`, then stop + join the loop thread (if `Wica` owns it).

**Order matters and teardown is the reverse of startup.** The loop must be running before the World dispatches or the Agent attaches its trigger handler; and the Agent must stop *while the World is still running*, because `agent.stop()` cancels in-flight Commands and cancellation writes `CommandExecution(state="cancelled")` back into the World via `world.update` (see [agent.md](agent.md)). Stopping the World first would make those final updates raise. So it's loop→World→Agent to start, Agent→World→loop to stop.

**Injected loop.** `Wica.init(config, …, loop=…)` accepts an existing loop for an app that already runs one; then `Wica` does not own the thread, and `start()`/`stop()` leave it alone (only attaching/detaching World and Agent). The batteries-included default — no `loop` passed — has `Wica` create the loop and run it in its own daemon thread.

**Lifecycle is once per instance, not a reusable pause/resume.** `wica.start()` is called once and `wica.stop()` tears the instance down; the instance is not designed to be restarted. The hard constraint is that `Wica`-owned loop thread: stopping it joins a daemon `threading.Thread`, which cannot be restarted. **To "reset" the whole system, discard the `Wica` and `init` a new one** (re-running the setup) — see "Reset is recreation". This keeps the model simple and matches the reset-by-recreation decision below.

### `wica.world` / `wica.agent` are borrowed references

`Wica` **owns** its World and Agent; the public `wica.world`/`wica.agent` attributes are **borrowed references, valid only for the life of the `Wica`**. A consumer uses them while the `Wica` is alive and discards them with it — the same contract as a file object obtained from a context manager. After `wica.stop()`, reactive mutation through `world.update()` raises `RuntimeError("World is not running")` (see [world.md](world.md), "Lifecycle"). Schema operations and reads remain deliberately unguarded for inspection, but the stopped objects should not be reused as a live system. The stated rule: *`wica.world` is borrowed from Wica; after `wica.stop()` inspect if needed, then get a fresh one from a new Wica.*

Because the objects never change identity within one `Wica`'s life (there is no in-place reset that swaps them — see below), plain attributes are correct: a cached `wica.world` reference never goes stale *while the Wica is running*. Accessor methods would only earn their keep if the objects were swapped under the caller, which the recreation model avoids.

### Command registration is delegated

`wica.register_command(fn, *, name=…, description=…)` forwards verbatim to `agent.register_command` ([agent.md](agent.md)). It's the single command-side convenience mirrored onto `Wica` because registering the agent's capabilities is part of every setup. Everything else command-related (the auto-registered `cancel_command`, execution-as-World-entry) stays entirely inside the Agent.

### `Event`s are surfaced, not adapted

The World and the Agent each own their instrumentation as `Event`s (see [world.md](world.md), "The trigger — the `on_trigger` Event"; [agent.md](agent.md), "Instrumentation"). `Wica` doesn't wrap or adapt them — it **surfaces the same objects**, so subscribing on `Wica` is subscribing on the underlying World/Agent:

| `Wica` attribute | is | Signal |
|---|---|---|
| `wica.on_world_trigger` | `world.on_trigger` | `Event[WorldEntry]` — **raw** trigger, every qualifying update (pre-coalescing) |
| `wica.on_agent_trigger` | `agent.on_trigger` | `Event[WorldEntry]` — **filtered** trigger, per trigger a step observed |
| `wica.on_agent_prompt` | `agent.on_prompt` | `Event[list[BaseMessage]]` — the rendered messages before each model call |
| `wica.on_agent_command` | `agent.on_command` | `Event[CommandIssued]` — each Command issued, at dispatch |

**Two triggers, on purpose.** `on_world_trigger` and `on_agent_trigger` are kept distinct because they answer different questions: the World's is "an input qualified to wake the agent" (fires even for triggers the busy single-in-flight loop later drops); the Agent's is "a step actually processed this trigger." A consumer showing every incoming input uses the former; one showing only what the robot reacted to uses the latter. The demo uses `on_agent_trigger` for its input panel (matching today's behavior).

**`CommandIssued`** is a small frozen dataclass (`name: str`, `args: dict[str, Any]`) — a named, evolvable payload preferred over a bare `(name, args)` tuple. It is defined in `agent.py` (the Agent emits it) and re-exported from the package.

**No `Wica`-level adapters or guards.** Because each signal is already an `Event`, `Wica` holds a reference and nothing more — no `emit` shims, no payload adapters. Subscriber isolation is provided by `Event.emit` itself (catch-log-continue — see [events.md](events.md)), uniformly for every `Event` in the system, so neither `Wica` nor the Agent needs a guard of its own. The prompt Event carrying `list[BaseMessage]` (a LangChain type) lives naturally on the Agent, the framework's LangChain I/O boundary, and does not leak into the `Event` primitive, which stays a pure project-agnostic leaf.

### Reset is recreation, not a method

There is **no `wica.reset()` and no `reset_world()`**. "Reset everything" is expressed by disposing the `Wica` and building a new one:

```python
wica.stop()
wica = Wica.init(config, …)   # re-run the setup: register entries, commands, start
```

This was a deliberate simplification. An in-place `reset()` had to answer "what state survives?" (registered entries? commands? history?) and either swap the owned objects (forcing accessor methods so held references don't go stale) or clear them in place (adding a `World.reset()` with subtle timer and loop re-wiring). Recreation sidesteps all of it: the consumer's own setup code is the single source of truth for reconstruction, nothing needs to be "remembered" inside `Wica`, and the objects never change identity mid-life so `wica.world`/`wica.agent` can stay plain attributes. The cost — re-running setup after a reset — is exactly the code the consumer already wrote to stand the system up the first time.

## Open questions

These are deferrals, not blockers to what's specified above.

1. **Multiple `Wica` instances.** Because World is not a singleton (see [world.md](world.md)), several `Wica` instances are *structurally* possible — each owns its own World/Agent, with no shared global. Whether multi-instance is a *supported, tested* configuration (e.g. two independent agents in one process) is not yet exercised; today's one-Agent-per-World design still frames the implementation. The facade doesn't prevent multiple instances; it just isn't validated for them yet.
2. **Restartable lifecycle.** `start()`/`stop()` are once-per-instance because the `Wica`-owned loop thread can't be restarted (a joined `threading.Thread` is spent). Making a `Wica` genuinely restartable (recreate the loop + its thread on a second `start()`) is possible but unbuilt — recreation covers the need for now.
3. **Raw-model injection — available at the `Agent` layer.** `Agent.__init__` takes an optional `model=` override: config builds the model unless a bespoke `BaseChatModel` is passed (see [agent.md](agent.md), "Provider-agnostic model, from config"). `Wica.init` deliberately does **not** surface it — the facade stays config-only (provider/model/key/prompt from the file); a caller needing a raw model constructs the `Agent` directly. Whether the facade should ever expose it is left open, but no facade-level case has appeared.
4. **Context-manager sugar.** `Wica` could implement `__enter__`/`__exit__` (calling `start()`/`stop()`) so `with Wica.init(config) as wica:` reads naturally. Deferred as pure ergonomics on top of the explicit `start()`/`stop()` the lifecycle already provides.
5. **Config-driven World/Command wiring.** World-entry schema and Command registration remain **code**, run against `wica.world`/`wica.register_command` after `init` — a JSON file can't express the callables involved (see [config.md](config.md), open question #3). If a future config ever seeds part of the World, `Wica.init` is where that would be applied.
