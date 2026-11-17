"""The `Command` definition object — WICA's unit of agent action on the World, at build time.

`Command` wraps either a plain callable or an off-the-shelf LangChain `BaseTool` and holds the
backing `BaseTool` internally. It is the concrete "registration/wrapper layer" the specs name, and
the one place `BaseTool` appears at the command-definition surface: `register_command` and
`set_output_command` speak `Command | Callable` only, so the command-definition surface
stays LangChain-free even though this module imports LangChain to build the tool. See
specs/commands.md ("The `Command` object").
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, tool


class Command:
    """A WICA Command definition.

    Construct one from a callable (name from ``__name__``, description from the docstring unless
    overridden) or by wrapping an existing tool:

        Command(walk_to)                       # plain callable
        Command(add, name="sum", description=…)  # overrides
        Command(existing_tool)                  # off-the-shelf BaseTool, unmodified
        Command(set_led, triggers_on_completion=False)  # completion wakes no step
    """

    def __init__(
        self,
        fn: Callable[..., Any] | BaseTool,
        *,
        name: str | None = None,
        description: str | None = None,
        triggers_on_completion: bool = True,
    ) -> None:
        # A WICA registration option, not tool metadata: accepted for a wrapped BaseTool too.
        self._triggers_on_completion = triggers_on_completion
        # The callable as given, or None when wrapping an off-the-shelf tool. The output Command
        # is called through it, not through the tool (see specs/commands.md, "The output Command").
        self._callable: Callable[..., Any] | None = (
            None if isinstance(fn, BaseTool) else fn
        )
        if isinstance(fn, BaseTool):
            # An already-built tool carries its own name/description; overriding them here would be
            # silently ineffective, so reject it explicitly rather than mislead the caller.
            if name is not None or description is not None:
                raise ValueError(
                    "name/description cannot be overridden when wrapping an existing tool; "
                    "set them on the tool itself, or pass a plain callable"
                )
            self._tool: BaseTool = fn
        elif name is not None:
            self._tool = tool(name, description=description)(fn)
        else:
            # description=None makes LangChain derive it from the docstring; an undocumented
            # callable with no description raises a clear ValueError (LangChain's own).
            self._tool = tool(fn, description=description)

    @property
    def tool(self) -> BaseTool:
        """The backing LangChain tool — the Agent binds this to the model."""
        return self._tool

    @property
    def callable(self) -> Callable[..., Any] | None:
        """The callable this Command was built from, or None when it wraps an existing tool."""
        return self._callable

    @property
    def description(self) -> str:
        """The Command's description (the backing tool's)."""
        return self._tool.description

    @property
    def name(self) -> str:
        """The Command's name (the backing tool's name)."""
        return self._tool.name

    @property
    def triggers_on_completion(self) -> bool:
        """Whether this Command's terminal completion wakes a reasoning step (default True).
        False removes only the trigger — the execution entry still exists and is observed by the
        next step. Not read for the output Command, whose wake is the model's `<next>` note. See
        specs/commands.md ("The `Command` object")."""
        return self._triggers_on_completion
