from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from wica.content import Content, TextPart

_logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class WorldEntryVersion:
    id: int
    value: Any
    timestamp: datetime


@dataclass(frozen=True)
class WorldEntry:
    key: str
    type: type
    current: WorldEntryVersion
    previous: WorldEntryVersion | None


class World:
    def __init__(self) -> None:
        self._configs: dict[str, WorldEntryConfig] = {}
        self._entries: dict[str, WorldEntry] = {}
        self._id_counters: dict[str, int] = {}
        self._listeners: dict[str, list[Callable[[WorldEntry], None]]] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._trigger_handler: Callable[[WorldEntry], None] | None = None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor()

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
            )
            new_id = self._id_counters.get(key, 0) + 1
            self._id_counters[key] = new_id
            self._entries[key] = WorldEntry(
                key=key,
                type=type,
                current=WorldEntryVersion(id=new_id, value=None, timestamp=_now()),
                previous=None,
            )
        _logger.debug(
            "registered %r (type=%s, include_in_prompt=%s, triggers_llm_call=%s, ttl=%s)",
            key,
            type.__name__,
            include_in_prompt,
            triggers_llm_call,
            ttl,
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
        return entry.current.value

    def get_entry(self, key: str) -> WorldEntry:
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            raise KeyError(key)
        return entry

    def update(self, key: str, value: Any) -> None:
        self._update(key, value, ttl_reset=False)

    def _update(
        self, key: str, value: Any, *, ttl_reset: bool, expected_id: int | None = None
    ) -> None:
        listeners: list[Callable[[WorldEntry], None]] = []
        trigger_handler: Callable[[WorldEntry], None] | None = None
        new_entry: WorldEntry

        with self._lock:
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

            new_id = self._id_counters[key] + 1
            self._id_counters[key] = new_id

            new_version = WorldEntryVersion(id=new_id, value=value, timestamp=_now())
            new_entry = WorldEntry(
                key=key, type=old_entry.type, current=new_version, previous=old_entry.current
            )
            self._entries[key] = new_entry

            old_timer = self._timers.pop(key, None)
            if old_timer is not None:
                old_timer.cancel()
            if config.ttl is not None and value is not None:
                timer = threading.Timer(
                    config.ttl.total_seconds(), self._ttl_expire, args=(key, new_id)
                )
                timer.daemon = True
                self._timers[key] = timer
                timer.start()

            listeners = list(self._listeners.get(key, ()))
            has_handler = self._trigger_handler is not None
            triggers_configured = config.triggers_llm_call and not ttl_reset
            if (
                triggers_configured
                and has_handler
                and (
                    config.trigger_condition_fn is None
                    or config.trigger_condition_fn(old_entry.current.value, value)
                )
            ):
                trigger_handler = self._trigger_handler

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
        if trigger_handler is not None:
            _logger.debug("update to %r (id=%d) triggers an LLM call", key, new_id)
        elif triggers_configured and has_handler:
            _logger.debug(
                "update to %r (id=%d) did not trigger an LLM call (condition unmet)", key, new_id
            )

        for listener in listeners:
            self._executor.submit(listener, new_entry)
        if trigger_handler is not None:
            self._executor.submit(trigger_handler, new_entry)

    def _ttl_expire(self, key: str, expected_id: int) -> None:
        _logger.debug("TTL fired for %r (expected id=%d)", key, expected_id)
        try:
            self._update(key, None, ttl_reset=True, expected_id=expected_id)
        except KeyError:
            pass

    def add_listener(self, key: str, callback: Callable[[WorldEntry], None]) -> None:
        with self._lock:
            if key not in self._configs:
                raise KeyError(key)
            self._listeners.setdefault(key, []).append(callback)
        _logger.debug("added listener for %r", key)

    def remove_listener(self, key: str, callback: Callable[[WorldEntry], None]) -> None:
        with self._lock:
            listeners = self._listeners.get(key)
            if listeners is not None and callback in listeners:
                listeners.remove(callback)

    def set_trigger_handler(self, handler: Callable[[WorldEntry], None] | None) -> None:
        with self._lock:
            self._trigger_handler = handler
        _logger.debug("trigger handler %s", "set" if handler is not None else "cleared")

    def get_prompt_entries(self) -> list[WorldEntry]:
        with self._lock:
            entries = [
                entry
                for key, entry in self._entries.items()
                if self._configs[key].include_in_prompt
            ]
            entries.sort(key=lambda e: e.current.timestamp)
        return entries

    def render_entry(self, entry: WorldEntry, *, archival: bool = False) -> Content:
        with self._lock:
            config = self._configs.get(entry.key)
        if config is None:
            raise KeyError(entry.key)

        serialize_fn = config.serialize_fn
        if archival and config.archival_serialize_fn is not None:
            serialize_fn = config.archival_serialize_fn

        previous_value = entry.previous.value if entry.previous is not None else None
        body = serialize_fn(entry.current.value, previous_value)
        opening = TextPart(f'<entry key="{entry.key}" id="{entry.current.id}">\n')
        closing = TextPart(f"\nUpdated: {entry.current.timestamp.isoformat()}\n</entry>")
        return [opening, *body, closing]

    def render_full_prompt(self) -> Content:
        entries = self.get_prompt_entries()

        content: Content = []
        for i, entry in enumerate(entries):
            if i > 0:
                content.append(TextPart("\n"))
            content.extend(self.render_entry(entry))
        return content

    def _cancel_all_timers(self) -> None:
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()


_world: World | None = None


def get_world() -> World:
    global _world
    if _world is None:
        _world = World()
    return _world


def reset_world() -> None:
    global _world
    if _world is not None:
        _world._cancel_all_timers()
        _world._executor.shutdown(wait=True)
    _world = None
