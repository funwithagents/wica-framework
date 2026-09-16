"""Latency and reactivity instrumentation: the reaction/trigger/Command records, the shared
wall-clock, the parent-trigger rule and the per-reaction measures. See specs/instrumentation.md.

Deviation from the spec's field sketch: `prompt_ready_at` and `model_started_at` on ReactionTrace
are `datetime | None` (not a bare `datetime`) because a reaction cancelled by `stop()` while
rendering never reaches them — the spec's intent (a record for every reaction) wins over its
field sketch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from opentelemetry import trace

TRACER_NAME = "wica"
COMMAND_KEY_PREFIX = "agent:command:"  # duplicated from agent.py on purpose: this module must not import it


def now() -> datetime:
    """The one clock every stamp uses: wall-clock UTC, the World's own. See "One clock"."""
    return datetime.now(timezone.utc)


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


@dataclass(frozen=True)
class TriggerTrace:
    key: str
    version_id: int
    written_at: datetime
    arrived_at: datetime
    is_command_completion: bool

    @property
    def hop(self) -> float:
        """Seconds from the World write to the trigger reaching the loop."""
        return (self.arrived_at - self.written_at).total_seconds()


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None


ReactionOutcome = Literal["ok", "empty", "model_error", "cancelled"]


@dataclass(frozen=True)
class ReactionTrace:
    reaction_id: int
    triggers: tuple[TriggerTrace, ...]
    window_opened_at: datetime
    window_closed_at: datetime
    prompt_ready_at: datetime | None
    model_started_at: datetime | None
    model_ended_at: datetime | None
    outcome: ReactionOutcome
    error: str | None
    text_length: int
    sink_duration: float | None
    command_call_ids: tuple[str, ...]
    noop: bool
    usage: TokenUsage | None
    ended_at: datetime
    trace_id: str | None
    span_id: str | None

    @property
    def coalescing_wait(self) -> float:
        return (self.window_closed_at - self.window_opened_at).total_seconds()

    @property
    def render_time(self) -> float | None:
        if self.prompt_ready_at is None:
            return None
        return (self.prompt_ready_at - self.window_closed_at).total_seconds()

    @property
    def model_latency(self) -> float | None:
        if self.model_started_at is None or self.model_ended_at is None:
            return None
        return (self.model_ended_at - self.model_started_at).total_seconds()

    @property
    def busy_time(self) -> float:
        return (self.ended_at - self.window_closed_at).total_seconds()


@dataclass(frozen=True)
class CommandTrace:
    call_id: str
    name: str
    reaction_id: int
    started_at: datetime
    ended_at: datetime
    state: Literal["complete", "failed", "cancelled"]

    @property
    def duration(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()


def parent_trigger(triggers: Sequence[TriggerTrace]) -> TriggerTrace:
    """The trigger a reaction is parented to: the earliest-arrived trigger that is not a Command
    completion; if all are completions, the earliest one. `triggers` must be non-empty."""
    externals = [t for t in triggers if not t.is_command_completion]
    pool = externals if externals else list(triggers)
    return min(pool, key=lambda t: t.arrived_at)


def reaction_latency(reaction: ReactionTrace) -> float | None:
    """written_at of the earliest external trigger -> ended_at, for a reaction that issued at least
    one Command or delivered text (command_call_ids non-empty or text_length > 0); else None.
    A reaction whose every trigger is a Command completion has no external trigger -> None."""
    if not reaction.command_call_ids and reaction.text_length <= 0:
        return None
    externals = [t for t in reaction.triggers if not t.is_command_completion]
    if not externals:
        return None
    earliest = min(t.written_at for t in externals)
    return (reaction.ended_at - earliest).total_seconds()


def reactions_per_input(reactions: Sequence[ReactionTrace]) -> list[int]:
    """For each reaction whose parent trigger is external, the number of reactions from it (inclusive)
    up to the next such reaction (exclusive) — the re-trigger chain's cost. Reactions before the
    first external one are ignored."""
    starts = [
        i
        for i, reaction in enumerate(reactions)
        if not parent_trigger(reaction.triggers).is_command_completion
    ]
    counts = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(reactions)
        counts.append(end - start)
    return counts
