from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from wica.content import Content, TextPart
from wica.events import Event

_logger = logging.getLogger(__name__)

# A World listener may be a plain sync function or an async coroutine function; the World
# detects which and dispatches accordingly on the shared loop (see World.add_listener).
Listener = Callable[["WorldEntry"], None] | Callable[["WorldEntry"], Awaitable[None]]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _describe_value(value: Any) -> str:
    """A short, log-safe rendering of a stored value — never dumps large blobs (e.g. image
    bytes) in full."""
    if value is None:
        return "None"
    if isinstance(value, bytes | bytearray):
        return f"<{len(value)} bytes>"
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


@dataclass
class WorldEntryConfig:
    type: type
    serialize_fn: Callable[[Any, Any], Content]
    archival_serialize_fn: Callable[[Any, Any], Content] | None = None
    include_in_prompt: bool = True
    triggers_llm_call: bool = False
    trigger_condition_fn: Callable[[Any, Any], bool] | None = None
    ttl: timedelta | None = None
    bypass_coalescing: bool = False


@dataclass(frozen=True)
class WorldEntryVersion:
    id: int
    value: Any
    timestamp: datetime


@dataclass(frozen=True)
class WorldEntry:
    key: str
    type: type
    # Copied from WorldEntryConfig at register() (like `type`) so the snapshot handed to an
    # on_trigger subscriber is self-describing: the Agent reads this to decide whether an update
    # skips its coalescing window. The World only carries the bit — it never acts on it. See
    # specs/world.md and specs/agent.md ("Trigger coalescing").
    bypass_coalescing: bool
    current: WorldEntryVersion
    previous: WorldEntryVersion | None


class World:
    """The World state registry.

    Constructed with the asyncio event loop it dispatches reactive callbacks on (``World(loop)``)
    — the *same* loop the Agent runs on, owned and run by the ``Wica`` facade. The World does not
    own a thread pool of its own; it schedules listeners and the ``on_trigger`` Event onto that
    shared loop. ``update()`` remains callable from any thread — it reaches the loop via
    ``call_soon_threadsafe``. See specs/world.md ("The shared event loop", "Lifecycle").
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._configs: dict[str, WorldEntryConfig] = {}
        self._entries: dict[str, WorldEntry] = {}
        self._id_counters: dict[str, int] = {}
        self._listeners: dict[str, list[Listener]] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.RLock()
        self._running = False
        # The World's single trigger signal, carrying the entry that qualified to wake the agent.
        # The Agent subscribes to it (via an async-scheduling shim); other observers may subscribe
        # alongside. Emitted on the loop thread. See specs/world.md ("The trigger — the on_trigger
        # Event") and specs/events.md.
        self.on_trigger: Event[WorldEntry] = Event()

    # --- Lifecycle -----------------------------------------------------------
    #
    # start()/stop() gate the *reactive dispatch*, not the data model: only update()/_update are
    # guarded (they dispatch onto the loop). register/get/render_* stay unguarded so setup-before-
    # start and reads-after-stop keep working. The loop itself is owned by Wica, not the World, so
    # these are light — stop() just cancels TTL timers. See specs/world.md ("Lifecycle").

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def start(self) -> None:
        expired: list[tuple[str, int]] = []
        with self._lock:
            if self._running:
                return
            self._running = True
            now = _now()
            # stop() cancels timers but retains state. Restore each retained value against its
            # original update deadline so wall-clock TTL continues across the pause.
            for key, config in self._configs.items():
                entry = self._entries[key]
                if config.ttl is None or entry.current.value is None:
                    continue
                remaining = (entry.current.timestamp + config.ttl - now).total_seconds()
                if remaining <= 0:
                    expired.append((key, entry.current.id))
                else:
                    timer = self._make_ttl_timer(key, entry.current.id, remaining)
                    self._timers[key] = timer
                    timer.start()
        for key, expected_id in expired:
            try:
                self._update(key, None, ttl_reset=True, expected_id=expected_id)
            except RuntimeError:
                # A concurrent stop won the race after the lifecycle lock was released.
                break
        _logger.debug("world started")

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            # Change the guard and detach the timers atomically: an update can happen wholly before
            # this transition or fail wholly after it, but cannot arm a timer into a stopped World.
            self._running = False
            timers = list(self._timers.values())
            self._timers.clear()
        for timer in timers:
            timer.cancel()
        _logger.debug("world stopped")

    def register[T](
        self,
        key: str,
        type: type[T],
        *,
        serialize_fn: Callable[[T | None, T | None], Content],
        archival_serialize_fn: Callable[[T | None, T | None], Content] | None = None,
        include_in_prompt: bool = True,
        triggers_llm_call: bool = False,
        trigger_condition_fn: Callable[[T | None, T | None], bool] | None = None,
        ttl: timedelta | None = None,
        bypass_coalescing: bool = False,
    ) -> None:
        with self._lock:
            if key in self._configs:
                raise ValueError(f"key {key!r} is already registered")
            self._configs[key] = WorldEntryConfig(
                type=type,
                serialize_fn=serialize_fn,
                archival_serialize_fn=archival_serialize_fn,
                include_in_prompt=include_in_prompt,
                triggers_llm_call=triggers_llm_call,
                trigger_condition_fn=trigger_condition_fn,
                ttl=ttl,
                bypass_coalescing=bypass_coalescing,
            )
            new_id = self._id_counters.get(key, 0) + 1
            self._id_counters[key] = new_id
            self._entries[key] = WorldEntry(
                key=key,
                type=type,
                bypass_coalescing=bypass_coalescing,
                current=WorldEntryVersion(id=new_id, value=None, timestamp=_now()),
                previous=None,
            )
        _logger.debug(
            "registered %r (type=%s, include_in_prompt=%s, triggers_llm_call=%s, ttl=%s, "
            "bypass_coalescing=%s)",
            key,
            type.__name__,
            include_in_prompt,
            triggers_llm_call,
            ttl,
            bypass_coalescing,
        )

    def unregister(self, key: str) -> None:
        with self._lock:
            if key not in self._configs:
                raise KeyError(key)
            del self._configs[key]
            del self._entries[key]
            self._listeners.pop(key, None)
            timer = self._timers.pop(key, None)
            if timer is not None:
                timer.cancel()
        _logger.debug("unregistered %r", key)

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            raise KeyError(key)
        return copy.deepcopy(entry.current.value)

    def get_entry(self, key: str) -> WorldEntry:
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            raise KeyError(key)
        return copy.deepcopy(entry)

    def update(self, key: str, value: Any) -> None:
        self._update(key, value, ttl_reset=False)

    def _update(
        self, key: str, value: Any, *, ttl_reset: bool, expected_id: int | None = None
    ) -> None:
        listeners: list[Listener] = []
        should_trigger = False
        new_entry: WorldEntry

        with self._lock:
            # The reactive path is the sole guarded surface: it dispatches onto the loop, so it must
            # fail fast rather than touch a torn-down World. register/get/render_* stay unguarded.
            if not self._running:
                raise RuntimeError("World is not running")
            if key not in self._configs:
                raise KeyError(key)
            old_entry = self._entries[key]
            # A TTL timer is armed for a specific version id. If a newer update has since
            # advanced the entry (its current id no longer matches), this expiry is stale —
            # skip it so it can't clobber the fresher value or cancel the fresher timer.
            if expected_id is not None and old_entry.current.id != expected_id:
                return
            config = self._configs[key]
            if value is not None and not isinstance(value, config.type):
                raise TypeError(
                    f"value for key {key!r} must be an instance of {config.type!r}, "
                    f"got {type(value).__name__!r}"
                )

            # The World owns the value once it is updated. Copy on ingress so a producer retaining
            # and mutating its original object cannot change versioned state without another
            # update() (and therefore without a new id/timestamp/trigger).
            stored_value = copy.deepcopy(value)
            new_id = self._id_counters[key] + 1
            self._id_counters[key] = new_id
            new_entry = WorldEntry(
                key=key,
                type=old_entry.type,
                bypass_coalescing=old_entry.bypass_coalescing,
                current=WorldEntryVersion(id=new_id, value=stored_value, timestamp=_now()),
                previous=old_entry.current,
            )
            self._entries[key] = new_entry

            old_timer = self._timers.pop(key, None)
            if old_timer is not None:
                old_timer.cancel()
            if config.ttl is not None and stored_value is not None:
                timer = self._make_ttl_timer(key, new_id, config.ttl.total_seconds())
                self._timers[key] = timer
                timer.start()

            listeners = list(self._listeners.get(key, ()))
            triggers_configured = config.triggers_llm_call and not ttl_reset
            should_trigger = triggers_configured and (
                config.trigger_condition_fn is None
                or config.trigger_condition_fn(
                    copy.deepcopy(old_entry.current.value), copy.deepcopy(stored_value)
                )
            )

        new_id = new_entry.current.id
        _logger.debug(
            "updated %r → id=%d, value=%s%s",
            key,
            new_id,
            _describe_value(value),
            " [ttl reset]" if ttl_reset else "",
        )
        if listeners:
            _logger.debug("dispatching %d listener(s) for %r (id=%d)", len(listeners), key, new_id)
        if should_trigger:
            _logger.debug("update to %r (id=%d) triggers an LLM call", key, new_id)
        elif triggers_configured:
            _logger.debug(
                "update to %r (id=%d) did not trigger an LLM call (condition unmet)", key, new_id
            )

        # Both reactive outputs are fire-and-forget onto the shared loop, so a slow callback never
        # blocks update()'s (possibly cross-thread) caller. Listeners run sync-on-the-pool or
        # async-on-the-loop and are individually guarded; the trigger emits on_trigger on the loop
        # thread so a subscriber may safely create_task. See specs/world.md ("The shared event loop").
        for listener in listeners:
            self._loop.call_soon_threadsafe(self._dispatch, listener, copy.deepcopy(new_entry))
        if should_trigger:
            self._loop.call_soon_threadsafe(self.on_trigger.emit, copy.deepcopy(new_entry))

    def _dispatch(self, callback: Listener, entry: WorldEntry) -> None:
        """Schedule one listener on the loop (runs on the loop thread, via call_soon_threadsafe).

        An async listener becomes a loop task; a sync listener is offloaded to the loop's default
        thread-pool executor so a blocking callback never stalls the loop. Each is individually
        guarded so a raising listener is caught-and-logged, isolated from siblings and the loop."""
        if inspect.iscoroutinefunction(callback):
            self._loop.create_task(self._run_guarded_async(callback, entry))
        else:
            fut = self._loop.run_in_executor(None, callback, entry)  # type: ignore[arg-type]
            fut.add_done_callback(self._log_if_failed)

    async def _run_guarded_async(
        self, callback: Callable[[WorldEntry], Awaitable[None]], entry: WorldEntry
    ) -> None:
        try:
            await callback(entry)
        except Exception:
            _logger.exception("world listener raised; ignoring")

    def _log_if_failed(self, fut: Future[Any] | asyncio.Future[Any]) -> None:
        if fut.cancelled():
            return
        exc = fut.exception()
        if exc is not None:
            _logger.error("world listener raised; ignoring", exc_info=exc)

    def _ttl_expire(self, key: str, expected_id: int) -> None:
        _logger.debug("TTL fired for %r (expected id=%d)", key, expected_id)
        try:
            self._update(key, None, ttl_reset=True, expected_id=expected_id)
        except KeyError:
            pass
        except RuntimeError:
            # World stopped between the timer firing and this reset acquiring the lock — the timer
            # was about to be cancelled anyway, so the stale reset is a harmless no-op.
            pass

    def _make_ttl_timer(self, key: str, expected_id: int, delay: float) -> threading.Timer:
        timer = threading.Timer(delay, self._ttl_expire, args=(key, expected_id))
        timer.daemon = True
        return timer

    def add_listener(self, key: str, callback: Listener) -> None:
        with self._lock:
            if key not in self._configs:
                raise KeyError(key)
            self._listeners.setdefault(key, []).append(callback)
        _logger.debug("added listener for %r", key)

    def remove_listener(self, key: str, callback: Listener) -> None:
        with self._lock:
            listeners = self._listeners.get(key)
            if listeners is not None and callback in listeners:
                listeners.remove(callback)

    def get_prompt_entries(self) -> list[WorldEntry]:
        with self._lock:
            entries = [
                entry
                for key, entry in self._entries.items()
                if self._configs[key].include_in_prompt
            ]
            entries.sort(key=lambda e: e.current.timestamp)
        return copy.deepcopy(entries)

    def render_entry(
        self,
        entry: WorldEntry,
        *,
        archival: bool = False,
        serialize_fn: Callable[[Any, Any], Content] | None = None,
    ) -> Content:
        # An explicit serialize_fn overrides the registered one and skips the config lookup
        # entirely, so a caller can still render an entry whose key has since been unregistered
        # (e.g. the Agent re-rendering a retired command entry from a history snapshot — see
        # specs/agent.md). The <entry …> envelope below stays owned by the World either way.
        if serialize_fn is None:
            with self._lock:
                config = self._configs.get(entry.key)
            if config is None:
                raise KeyError(entry.key)
            serialize_fn = config.serialize_fn
            if archival and config.archival_serialize_fn is not None:
                serialize_fn = config.archival_serialize_fn

        previous_value = entry.previous.value if entry.previous is not None else None
        body = serialize_fn(
            copy.deepcopy(entry.current.value),
            copy.deepcopy(previous_value),
        )
        opening = TextPart(f'<entry key="{entry.key}" id="{entry.current.id}">\n')
        # The closing part ends with a trailing newline so that, when entries are concatenated
        # (the way adjacent parts merge — see content.md), each `</entry>` sits on its own line
        # and the next `<entry ...>` starts on the following one, rather than gluing together.
        closing = TextPart(f"\nUpdated: {entry.current.timestamp.isoformat()}\n</entry>\n")
        return [opening, *body, closing]

    def render_full_prompt(self) -> Content:
        entries = self.get_prompt_entries()

        content: Content = []
        for entry in entries:
            content.extend(self.render_entry(entry))
        return content
