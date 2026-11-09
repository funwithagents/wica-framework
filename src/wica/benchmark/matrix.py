"""The matrix file: every configuration under test in one JSON document, selected by id glob and
tag. Implements specs/benchmark.md, "The matrix".

    {"agent_defaults": {...}, "entries": [{"id": ..., "agent": {...}, "tags": [...], "enabled": true,
                                           "requires_env": [...], "notes": "..."}]}

An entry's `agent` block is the configuration (provider, model, key variable, history settings),
merged over `agent_defaults` (the shared persona, typically; an either-or pair such as
`system_prompt` / `system_prompt_file` is merged as alternatives, see `merge_agent`); a relative
`system_prompt_file`
resolves against the matrix file's directory when the target builds its `WicaConfig`
(`Entry.base_dir`). `enabled: false` marks an opt-in entry; `skip_reason` gives the live tier's
gate — the key variable, `requires_env` — as a reason string or None.
"""

from __future__ import annotations

import fnmatch
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ENTRY_KEYS = {"id", "agent", "tags", "enabled", "requires_env", "notes"}
# The agent block's either-or pairs (specs/config.md): an entry that sets one member of a pair
# drops the other from the defaults, so a fixture with an inline `system_prompt` can sit in a
# matrix whose defaults carry the shared `system_prompt_file`.
_EXCLUSIVE = (("system_prompt", "system_prompt_file"), ("api_key", "api_key_env"))


class MatrixError(ValueError):
    """A matrix file that does not load: the message names the file and the entry."""


@dataclass(frozen=True)
class Entry:
    """One configuration of the matrix. `extra` holds the keys the format does not define — an
    application's own (an embodiment, an initial state), read by its target factory."""

    id: str
    agent: dict[str, Any]
    base_dir: Path
    tags: tuple[str, ...] = ()
    enabled: bool = True
    requires_env: tuple[str, ...] = ()
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def skip_reason(self) -> str | None:
        """Why this entry cannot run here: its `api_key_env` or one of `requires_env` unset.
        None when every variable is set (an entry without any always runs)."""
        names = list(self.requires_env)
        key_env = self.agent.get("api_key_env")
        if key_env:
            names.insert(0, key_env)
        for name in names:
            if not os.environ.get(name):
                return f"environment variable {name!r} is not set"
        return None


@dataclass(frozen=True)
class Matrix:
    path: Path
    entries: tuple[Entry, ...]
    comment: str = ""

    def entry(self, entry_id: str) -> Entry:
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        raise KeyError(entry_id)

    @property
    def ids(self) -> list[str]:
        return [e.id for e in self.entries]


def load_matrix(path: str | Path) -> Matrix:
    path = Path(path)

    def fail(what: str, where: str | None = None) -> MatrixError:
        prefix = f"{path}: " + (f"entry {where}: " if where else "")
        return MatrixError(prefix + what)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise fail(f"could not read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise fail(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise fail("the document must be a JSON object")
    unknown = set(data) - {"_comment", "agent_defaults", "entries"}
    if unknown:
        raise fail(f"unknown key(s) {sorted(unknown)}")
    defaults = data.get("agent_defaults", {})
    if not isinstance(defaults, dict):
        raise fail("'agent_defaults' must be an object")
    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise fail("'entries' must be a non-empty list")

    entries: list[Entry] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_entries):
        where = str(index)
        if not isinstance(raw, dict):
            raise fail("an entry must be an object", where)
        entry_id = raw.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            raise fail("'id' must be a non-empty string", where)
        where = repr(entry_id)
        if entry_id in seen:
            raise fail("duplicate id", where)
        seen.add(entry_id)
        agent = raw.get("agent")
        if not isinstance(agent, dict):
            raise fail("'agent' must be an object (the agent block)", where)
        tags = raw.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise fail("'tags' must be a list of strings", where)
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise fail("'enabled' must be a boolean", where)
        requires_env = raw.get("requires_env", [])
        if not isinstance(requires_env, list) or not all(
            isinstance(v, str) for v in requires_env
        ):
            raise fail("'requires_env' must be a list of strings", where)
        notes = raw.get("notes", "")
        if not isinstance(notes, str):
            raise fail("'notes' must be a string", where)
        entries.append(
            Entry(
                id=entry_id,
                agent=merge_agent(defaults, agent),
                base_dir=path.parent,
                tags=tuple(tags),
                enabled=enabled,
                requires_env=tuple(requires_env),
                notes=notes,
                extra={k: v for k, v in raw.items() if k not in _ENTRY_KEYS},
            )
        )
    return Matrix(
        path=path, entries=tuple(entries), comment=str(data.get("_comment", ""))
    )


def merge_agent(defaults: dict[str, Any], agent: dict[str, Any]) -> dict[str, Any]:
    """*agent* over *defaults*, the entry's keys winning; a default that is the other member of an
    either-or pair the entry sets (`system_prompt` / `system_prompt_file`, `api_key` /
    `api_key_env`) is left out."""
    merged = dict(defaults)
    for first, second in _EXCLUSIVE:
        if first in agent:
            merged.pop(second, None)
        if second in agent:
            merged.pop(first, None)
    merged.update(agent)
    return merged


def select_entries(
    entries: Iterable[Entry],
    *,
    only: Iterable[str] = (),
    exclude: Iterable[str] = (),
    tags: Iterable[str] = (),
    exclude_tags: Iterable[str] = (),
) -> list[Entry]:
    """The entries matching the selection: any `only` id glob (all when none), no `exclude`
    glob, any of `tags` (all when none), none of `exclude_tags`. Order kept; `enabled` is not a
    selection criterion — a disabled entry selected here is skipped, with the reason, by the
    runner's caller unless opted in."""
    only = list(only)
    exclude = list(exclude)
    tags_set = set(tags)
    excluded_tags = set(exclude_tags)
    selected: list[Entry] = []
    for entry in entries:
        if only and not any(fnmatch.fnmatchcase(entry.id, p) for p in only):
            continue
        if any(fnmatch.fnmatchcase(entry.id, p) for p in exclude):
            continue
        if tags_set and not tags_set & set(entry.tags):
            continue
        if excluded_tags & set(entry.tags):
            continue
        selected.append(entry)
    return selected
