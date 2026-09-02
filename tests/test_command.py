from __future__ import annotations

import pytest
from langchain_core.tools import BaseTool, tool

from wica.command import Command


def test_from_callable_derives_name_and_description_from_signature_and_docstring():
    def walk_to(place: str) -> str:
        """Walk to a named place."""
        return f"walking to {place}"

    command = Command(walk_to)

    assert command.name == "walk_to"
    assert isinstance(command.tool, BaseTool)
    assert command.tool.description == "Walk to a named place."
    # The backing tool exposes the callable's schema, so the model can be bound to it.
    assert "place" in command.tool.args


def test_name_and_description_override_the_callable_defaults():
    def raw(x: int) -> int:
        """Original doc."""
        return x

    command = Command(raw, name="doubler", description="Double the input.")

    assert command.name == "doubler"
    assert command.tool.description == "Double the input."


def test_wraps_an_off_the_shelf_tool_unmodified():
    @tool
    def existing(a: int) -> int:
        """An existing tool."""
        return a

    command = Command(existing)

    # Wrapped, not rebuilt: it is the very same tool object.
    assert command.tool is existing
    assert command.name == "existing"


def test_overriding_name_or_description_on_a_wrapped_tool_raises():
    @tool
    def existing(a: int) -> int:
        """An existing tool."""
        return a

    with pytest.raises(ValueError, match="cannot be overridden"):
        Command(existing, name="other")

    with pytest.raises(ValueError, match="cannot be overridden"):
        Command(existing, description="other")


def test_undocumented_callable_without_description_raises():
    def bare(a: int) -> int:
        return a

    # LangChain requires a docstring when no description is provided; Command preserves that.
    with pytest.raises(ValueError, match="docstring"):
        Command(bare)

    # ...but an explicit description lets an undocumented callable through.
    command = Command(bare, description="Bare tool.")
    assert command.tool.description == "Bare tool."
