from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from wica.agent import Agent, CommandIssued
from wica.config import WicaConfig, apply_logging
from wica.events import Event
from wica.world import World, WorldEntry

__all__ = ["Wica"]


class Wica:
    """The framework's single entry point.

    A ``Wica`` owns exactly one ``World`` and one ``Agent`` — constructed together from a
    ``WicaConfig`` and wired to each other — plus the single asyncio event loop they both run on.
    It drives their shared ``start()``/``stop()`` lifecycle and surfaces the World's and Agent's
    instrumentation ``Event``s as one multi-subscriber surface. See specs/wica.md.

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
        # Only created when Wica owns the loop; None when an existing loop was injected.
        self._loop_thread: threading.Thread | None = (
            threading.Thread(target=loop.run_forever, daemon=True) if owns_loop else None
        )
        self._stopped = False
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
        coalesce_window: float = 0.2,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> Wica:
        """Build a ``World`` + ``Agent`` from a ``WicaConfig``, wire them, apply logging, return the
        ``Wica``. The one construction path.

        Owns the single event loop the whole system runs on: if ``loop`` is None (the batteries-
        included default), Wica creates one and will run it in its own daemon thread on ``start()``;
        otherwise it adopts the injected loop and leaves the thread to the caller.

        The code-only wiring a JSON file can't express — ``output_sink``, ``coalesce_window``, and
        optionally ``loop`` — are keyword arguments here; the config carries provider/model/key/
        prompt/logging. Resolution (env key, prompt file) happens inside ``Agent.__init__``, so a
        ``MissingEnvError`` or unreadable prompt surfaces here, at ``init``. See specs/config.md,
        specs/wica.md.
        """
        owns_loop = loop is None
        if loop is None:
            loop = asyncio.new_event_loop()
        world = World(loop)
        agent = Agent(
            config.agent,
            world=world,
            loop=loop,
            output_sink=output_sink,
            coalesce_window=coalesce_window,
        )
        apply_logging(config.logging)
        return cls(loop=loop, owns_loop=owns_loop, world=world, agent=agent)

    def start(self) -> None:
        """Start the whole system: the loop thread (if owned), then the World, then the Agent.

        Order matters — the loop must be running before the World dispatches or the Agent attaches
        its trigger handler. Called once per instance; the instance is not designed to be restarted
        (see specs/wica.md, "Reset is recreation")."""
        if self._owns_loop and self._loop_thread is not None and not self._loop_thread.is_alive():
            self._loop_thread.start()
        self.world.start()
        self.agent.start()

    def stop(self) -> None:
        """Tear the system down: the Agent, then the World, then (if owned) the loop thread.

        Teardown is the reverse of startup: the Agent must stop *while the World is still running*,
        because cancelling in-flight Commands writes their terminal state back into the World.
        Stopping the World first would make those final updates raise. Idempotent — a second call is
        a no-op (the loop is already closed). See specs/wica.md."""
        if self._stopped:
            return
        self._stopped = True
        self.agent.stop()
        self.world.stop()
        if self._owns_loop and self._loop_thread is not None and self._loop_thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join()
            self._loop.close()

    def register_command(
        self,
        fn: Callable[..., Any] | BaseTool,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Register a Command — delegates verbatim to ``agent.register_command``. The one command-
        side convenience mirrored onto ``Wica`` (see specs/wica.md, "Command registration is
        delegated")."""
        self.agent.register_command(fn, name=name, description=description)
