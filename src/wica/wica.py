from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import BaseMessage

from wica.agent import Agent, CommandIssued
from wica.command import Command
from wica.config import WicaConfig
from wica.events import Event
from wica.world import World, WorldEntry

__all__ = ["Wica"]


class Wica:
    """The framework's single entry point.

    A ``Wica`` owns exactly one ``World`` and one ``Agent`` — constructed together from a
    ``WicaConfig`` and wired to each other — plus the single asyncio event loop they both run on.
    It drives their restartable ``start()``/``stop()`` lifecycle, owns terminal ``close()``, and
    surfaces the World's and Agent's instrumentation ``Event``s as one multi-subscriber surface.
    See specs/wica.md.

    Build one with :meth:`init` (the sole constructor); ``__init__`` is internal wiring.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        owns_loop: bool,
        world: World,
        agent: Agent,
    ) -> None:
        self._loop = loop
        self._owns_loop = owns_loop
        # A Thread is one-shot, so an owned Wica creates a fresh one for every running cycle while
        # retaining this one shared loop. Injected-loop Wicas never create a thread.
        self._loop_thread: threading.Thread | None = None
        self._running = False
        self._closed = False
        self._lifecycle_lock = threading.RLock()
        # Borrowed references, valid only for the life of this Wica (see specs/wica.md).
        self.world = world
        self.agent = agent
        # Surface the World's and Agent's Events directly — the *same* objects, no adapters or emit
        # shims. Subscribing on Wica is subscribing on the underlying World/Agent; Event.emit's
        # per-subscriber isolation guards each. See specs/wica.md ("Events are surfaced, not
        # adapted").
        self.on_world_trigger: Event[WorldEntry] = world.on_trigger
        self.on_agent_trigger: Event[WorldEntry] = agent.on_trigger
        self.on_agent_prompt: Event[list[BaseMessage]] = agent.on_prompt
        self.on_agent_command: Event[CommandIssued] = agent.on_command

    @classmethod
    def init(
        cls,
        config: WicaConfig,
        *,
        output_sink: Callable[[str], Awaitable[None]] | None = None,
        output_command: Callable[..., Any] | Command | None = None,
        coalesce_window: float = 0.2,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> Wica:
        """Build a ``World`` + ``Agent`` from a ``WicaConfig``, wire them, return the ``Wica``. The
        one construction path.

        Owns the single event loop the whole system runs on: if ``loop`` is None (the batteries-
        included default), Wica creates one and will run it in its own daemon thread on ``start()``;
        otherwise it adopts the injected loop and leaves the thread to the caller.

        The code-only wiring a JSON file can't express — ``output_sink``, ``output_command``,
        ``coalesce_window``, and optionally ``loop`` — are keyword arguments here; the config
        carries provider/model/key/prompt. ``output_command`` (a callable or a ``Command``) is the
        user-facing output channel; when set, free text becomes the agent's private reasoning
        stream (see specs/agent.md, "Output"). Resolution (env key, prompt file) happens inside
        ``Agent.__init__``, so a
        ``MissingEnvError`` or unreadable prompt surfaces here, at ``init``. Logging is not
        configured here: WICA is a library, so it only emits under the ``wica.*`` loggers and
        leaves handlers/levels to the embedding application. See specs/config.md, specs/wica.md.
        """
        owns_loop = loop is None
        if loop is None:
            loop = asyncio.new_event_loop()
        try:
            world = World(loop)
            agent = Agent(
                config.agent,
                world=world,
                loop=loop,
                output_sink=output_sink,
                output_command=output_command,
                coalesce_window=coalesce_window,
            )
        except BaseException:
            # Construction failed before a Wica could be returned to own this resource.
            if owns_loop:
                loop.close()
            raise
        return cls(loop=loop, owns_loop=owns_loop, world=world, agent=agent)

    @property
    def is_running(self) -> bool:
        """Whether this Wica is between a successful ``start()`` and ``stop()``."""
        with self._lifecycle_lock:
            return self._running

    def start(self) -> None:
        """Start or restart the system: owned loop thread, then World, then Agent.

        Order matters — the loop must be running before the World dispatches or the Agent attaches
        its trigger handler. Idempotent while already running; raises after terminal ``close()``.
        See specs/wica.md."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Wica is closed")
            if self._running:
                return
            if self._loop.is_closed():
                raise RuntimeError("Wica event loop is closed")

            if self._owns_loop:
                self._loop_thread = threading.Thread(
                    target=self._loop.run_forever,
                    name="wica-event-loop",
                    daemon=True,
                )
                self._loop_thread.start()
            try:
                self.world.start()
                self.agent.start()
            except BaseException:
                self.agent.stop()
                self.world.stop()
                self._stop_owned_loop_thread()
                raise
            self._running = True

    def stop(self) -> None:
        """Pause the system: the Agent, then World, then the current owned loop thread.

        Teardown is the reverse of startup: the Agent must stop *while the World is still running*,
        because cancelling in-flight Commands writes their terminal state back into the World.
        Stopping the World first would make those final updates raise. The loop stays open for a
        later ``start()``. Idempotent while already stopped. See specs/wica.md."""
        with self._lifecycle_lock:
            if not self._running:
                return
            self.agent.stop()
            self.world.stop()
            self._stop_owned_loop_thread()
            self._running = False

    def close(self) -> None:
        """Permanently stop this Wica and close its owned loop.

        An injected loop remains caller-owned and is never closed. Idempotent; ``start()`` after
        this terminal operation raises ``RuntimeError``.
        """
        with self._lifecycle_lock:
            if self._closed:
                return
            self.stop()
            if self._owns_loop and not self._loop.is_closed():
                self._loop.close()
            self._closed = True

    def _stop_owned_loop_thread(self) -> None:
        if not self._owns_loop or self._loop_thread is None:
            return
        if self._loop_thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join()
        self._loop_thread = None

    def register_command(self, fn: Callable[..., Any] | Command) -> None:
        """Register a Command — delegates verbatim to ``agent.register_command``. One argument, a
        callable or a ``Command`` (override name/description via ``Command(fn, name=…, …)``). The one
        command-side convenience mirrored onto ``Wica`` (see specs/wica.md, "Command registration is
        delegated")."""
        self.agent.register_command(fn)
