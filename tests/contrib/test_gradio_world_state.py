"""Fast tests of the contrib's World-state table over a real `World`: the rows and the HTML the
panel re-renders each tick, without a browser. See specs/gradio-contrib.md ("Component 1").
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Iterator

import pytest

from wica import CommandExecution, Content, TextPart, World
from wica.contrib.gradio import world_html, world_rows


@pytest.fixture
def world() -> Iterator[World]:
    """A running World on its own loop (updates need a running World), stopped on teardown."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    w = World(loop)
    w.start()
    yield w
    w.stop()
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


def populate(world: World) -> None:
    def plain(value: object, previous: object) -> Content:
        return [TextPart(str(value))]

    world.register("nobody", str, serialize_fn=plain)  # value None
    world.register("tags", list, serialize_fn=plain)
    world.register("html", str, serialize_fn=plain)
    world.register("hidden", str, serialize_fn=plain, include_in_prompt=False)
    world.register("agent:command:c1", CommandExecution, serialize_fn=plain)
    world.update("tags", ["a", "b"])
    world.update("html", "<b>bold</b>")
    world.update("hidden", "not shown")
    world.update(
        "agent:command:c1",
        CommandExecution(name="walk_to", args={"room": "kitchen"}, state="running"),
    )


def test_world_rows_formats_values_for_a_person(world: World):
    populate(world)
    rows = {key: (value, stamp) for key, value, stamp in world_rows(world)}
    assert rows["nobody"][0] == "—"
    assert rows["tags"][0] == "[a, b]"
    assert rows["html"][0] == "<b>bold</b>"
    assert rows["agent:command:c1"][0] == "walk_to(room='kitchen') [running]"
    assert "hidden" not in rows  # prompt entries only
    assert all(re.fullmatch(r"\d\d:\d\d:\d\d", stamp) for _, stamp in rows.values())


def test_world_html_marks_command_rows_and_escapes_values(world: World):
    populate(world)
    html = world_html(world)
    assert "<b>bold</b>" not in html and "&lt;b&gt;bold&lt;/b&gt;" in html
    command_rows = re.findall(r'<tr class="cmd">(.*?)</tr>', html)
    assert len(command_rows) == 1
    assert "walk_to(room=&#x27;kitchen&#x27;) [running]" in command_rows[0]
    # A plain entry is not highlighted, even though it is rendered.
    assert re.search(r"<tr><td>tags</td>", html)


def test_world_html_of_an_empty_world_is_a_table_with_headers_only():
    loop = asyncio.new_event_loop()
    html = world_html(World(loop))
    loop.close()
    assert "<th>Key</th><th>Value</th><th>Updated</th>" in html
    assert "<tbody></tbody>" in html
