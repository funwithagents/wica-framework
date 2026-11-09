"""Primer-level checks and per-run measures over a finished timeline. Implements
specs/benchmark.md, "Primer-level checks".

A check is a pure function `(timeline, context) -> Finding`: it never waits, it runs after
`run_script` returns, and it enforces one clause of the **runtime primer** every WICA agent is
under (specs/agent.md, "System prompt") — nothing an application's persona adds. The registry is
open: an application registers its own checks through `build_registry(extra_checks)` and selects
them in its scenario files by name like the built-ins. Measures are numbers, never thresholded.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from wica.instrumentation import ReactionTrace, TriggerTrace
from wica.testing.report import Finding, Report, SutEvent
from wica.testing.scenario import Requires, Scenario

NOOP = "noop"
CANCEL = "cancel_command"


@dataclass(frozen=True)
class CheckContext:
    """What a check may need beyond the timeline: the scenario, the output Command's name and the
    target's surface (its Inputs — those that trigger among them — and its Commands)."""

    scenario: Scenario
    output_command_name: str | None
    inputs: tuple[str, ...] = ()
    triggering_inputs: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()

    @property
    def turn_inputs(self) -> tuple[str, ...]:
        """The scenario's `requires.inputs` that trigger a step — the writes that open a turn."""
        return tuple(
            k for k in self.scenario.requires.inputs if k in self.triggering_inputs
        )


CheckFn = Callable[[list[SutEvent], CheckContext], Finding]


@dataclass(frozen=True)
class Check:
    name: str
    fn: CheckFn
    requires: Requires = Requires()

    def missing(self, context: CheckContext) -> list[str]:
        return self.requires.missing(
            inputs=set(context.inputs),
            commands=set(context.commands),
            output_command_name=context.output_command_name,
        )


# --- timeline helpers ----------------------------------------------------------------------------


def reactions(timeline: list[SutEvent]) -> list[ReactionTrace]:
    return [e.reaction for e in timeline if e.reaction is not None]


def external_triggers(reaction: ReactionTrace) -> list[TriggerTrace]:
    """The reaction's triggers that are World writes, not Command completions."""
    return [t for t in reaction.triggers if not t.is_command_completion]


def commands_by_reaction(timeline: list[SutEvent]) -> dict[int, list[SutEvent]]:
    """`reaction_id → its command_issued events` (joined through `command_call_ids`; `noop` has no
    call id there, see `ReactionTrace.noop`)."""
    issued = {
        e.call_id: e
        for e in timeline
        if e.kind == "command_issued" and e.call_id is not None
    }
    return {
        r.reaction_id: [issued[c] for c in r.command_call_ids if c in issued]
        for r in reactions(timeline)
    }


def reaction_of_call(timeline: list[SutEvent], call_id: str) -> ReactionTrace | None:
    for r in reactions(timeline):
        if call_id in r.command_call_ids:
            return r
    return None


@dataclass(frozen=True)
class Turn:
    """From the write that opened it (index `start` in the timeline) to the next opening write
    (`end`, exclusive) or the end of the timeline."""

    start: int
    end: int
    key: str
    t: float


def turns(timeline: list[SutEvent], keys: Iterable[str]) -> list[Turn]:
    """The turns opened by observed writes to *keys*: one per reaction that saw at least one such
    write among its triggers, opened at the earliest of those writes. A write whose trigger was
    dropped, or coalesced into a step another write opened, opens no turn of its own."""
    wanted = set(keys)
    written = {
        (e.key, e.version_id): i
        for i, e in enumerate(timeline)
        if e.kind == "input_written" and e.key in wanted
    }
    openings: dict[int, SutEvent] = {}
    for reaction in reactions(timeline):
        indexes = [
            written[(t.key, t.version_id)]
            for t in external_triggers(reaction)
            if (t.key, t.version_id) in written
        ]
        if indexes:
            start = min(indexes)
            openings[start] = timeline[start]
    starts = sorted(openings)
    result: list[Turn] = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(timeline)
        event = openings[start]
        result.append(Turn(start=start, end=end, key=event.key or "", t=event.t))
    return result


def output_issued(timeline: list[SutEvent], output_name: str | None) -> list[SutEvent]:
    if output_name is None:
        return []
    return [e for e in timeline if e.kind == "command_issued" and e.name == output_name]


def _end_t(timeline: list[SutEvent]) -> float:
    return timeline[-1].t if timeline else 0.0


def _origin(timeline: list[SutEvent]) -> float:
    return timeline[0].t if timeline else 0.0


def _finding(
    name: str,
    timeline: list[SutEvent],
    ok: bool,
    detail: str,
    event: SutEvent | None = None,
) -> Finding:
    return Finding(
        expectation=name, ok=ok, detail=detail, t=_end_t(timeline), event=event
    )


# --- the built-in checks -------------------------------------------------------------------------


def answers_each_input(timeline: list[SutEvent], context: CheckContext) -> Finding:
    """Primer: "anything addressed to you should be answered" / "must be said by calling
    `<output>`" — every turn opened by one of the scenario's triggering inputs issues the output
    Command at least once."""
    name = "answers_each_input"
    output = context.output_command_name
    all_turns = turns(timeline, context.turn_inputs)
    origin = _origin(timeline)
    for turn in all_turns:
        window = timeline[turn.start : turn.end]
        if not any(e.kind == "command_issued" and e.name == output for e in window):
            return _finding(
                name,
                timeline,
                False,
                f"the turn opened at t={turn.t - origin:.3f} by {turn.key!r} never issued "
                f"{output!r}",
                timeline[turn.start],
            )
    return _finding(
        name,
        timeline,
        True,
        f"{len(all_turns)} turn(s), each answered through {output!r}",
    )


def no_noop_on_input(timeline: list[SutEvent], context: CheckContext) -> Finding:
    """Primer: "never use `noop` to skip an expected response" — a reaction triggered by a write to
    one of the scenario's `requires.inputs` does not end in `noop` alone."""
    name = "no_noop_on_input"
    wanted = set(context.scenario.requires.inputs)
    origin = _origin(timeline)
    for event in timeline:
        reaction = event.reaction
        if reaction is None or not reaction.noop or reaction.command_call_ids:
            continue
        keys = [t.key for t in external_triggers(reaction) if t.key in wanted]
        if keys:
            return _finding(
                name,
                timeline,
                False,
                f"reaction {reaction.reaction_id} (ended t={event.t - origin:.3f}) answered the "
                f"write to {keys[0]!r} with noop only",
                event,
            )
    return _finding(name, timeline, True, "no input answered with noop")


def no_failed_reactions(timeline: list[SutEvent], context: CheckContext) -> Finding:
    """Every reaction ended `ok` — no model error, no empty response."""
    name = "no_failed_reactions"
    origin = _origin(timeline)
    for event in timeline:
        reaction = event.reaction
        if reaction is not None and reaction.outcome != "ok":
            detail = f"reaction {reaction.reaction_id} ended {reaction.outcome!r} at t={event.t - origin:.3f}"
            if reaction.error:
                detail += f": {reaction.error}"
            return _finding(name, timeline, False, detail, event)
    return _finding(
        name, timeline, True, f"{len(reactions(timeline))} reaction(s), all ok"
    )


def no_colliding_cancel(timeline: list[SutEvent], context: CheckContext) -> Finding:
    """Primer: "when one action must finish before another starts, do not issue both together" —
    no reaction issues `cancel_command` together with another Command."""
    name = "no_colliding_cancel"
    origin = _origin(timeline)
    for reaction_id, issued in commands_by_reaction(timeline).items():
        names = [e.name for e in issued if e.name != NOOP]
        if CANCEL in names and len(names) > 1:
            others = [n for n in names if n != CANCEL]
            first = next(e for e in issued if e.name == CANCEL)
            return _finding(
                name,
                timeline,
                False,
                f"reaction {reaction_id} issued {CANCEL} together with {others} at "
                f"t={first.t - origin:.3f}",
                first,
            )
    return _finding(
        name, timeline, True, f"no {CANCEL} issued alongside another Command"
    )


CHECKS: dict[str, Check] = {
    check.name: check
    for check in (
        Check("answers_each_input", answers_each_input, Requires(output_command=True)),
        Check("no_noop_on_input", no_noop_on_input),
        Check("no_failed_reactions", no_failed_reactions),
        Check("no_colliding_cancel", no_colliding_cancel),
    )
}


def build_registry(extra_checks: Iterable[Check] | None = None) -> dict[str, Check]:
    """The built-ins plus an application's *extra_checks*; a name clash with a built-in (or between
    extras) is an error."""
    registry = dict(CHECKS)
    for check in extra_checks or ():
        if check.name in registry:
            raise ValueError(f"check {check.name!r} is already registered")
        registry[check.name] = check
    return registry


def resolve_checks(
    names: Iterable[str] | None,
    registry: Mapping[str, Check],
    context: CheckContext,
) -> list[Check]:
    """The checks to run: the named ones (an unknown name raises), or — when *names* is None —
    every registered check whose own `requires` the target satisfies."""
    if names is None:
        return [c for c in registry.values() if not c.missing(context)]
    resolved: list[Check] = []
    for name in names:
        if name not in registry:
            raise ValueError(f"unknown check {name!r}; known: {sorted(registry)}")
        resolved.append(registry[name])
    return resolved


def apply_checks(
    report: Report,
    names: Iterable[str] | None,
    context: CheckContext,
    registry: Mapping[str, Check] | None = None,
) -> Report:
    """*report* with one finding per resolved check appended to its findings."""
    checks = resolve_checks(
        names, registry if registry is not None else CHECKS, context
    )
    new = [check.fn(report.timeline, context) for check in checks]
    return dataclasses.replace(report, findings=[*report.findings, *new])


# --- measures ------------------------------------------------------------------------------------


def measures(timeline: list[SutEvent], context: CheckContext) -> dict[str, float]:
    """The per-run numbers next to the framework's metrics: `output_calls`, `unprompted_output`
    (output issued by a reaction with no external trigger — the chained reply), `noop_steps`,
    `failed_reactions`, `output_words` (words in the output Command's string args), and the turn
    numbers a session is read by — `turns` (opened by the scenario's triggering inputs, see
    `turns`), `turns_answered` (the output Command issued within the turn), `noops_on_input`
    (reactions that answered a write to a scenario input with `noop` alone)."""
    outputs = output_issued(timeline, context.output_command_name)
    output = context.output_command_name
    all_turns = turns(timeline, context.turn_inputs)
    answered = sum(
        1
        for turn in all_turns
        if any(
            e.kind == "command_issued" and e.name == output
            for e in timeline[turn.start : turn.end]
        )
    )
    wanted = set(context.scenario.requires.inputs)
    noops_on_input = sum(
        1
        for r in reactions(timeline)
        if r.noop
        and not r.command_call_ids
        and any(t.key in wanted for t in external_triggers(r))
    )
    unprompted = 0
    for event in outputs:
        reaction = reaction_of_call(timeline, event.call_id or "")
        if reaction is not None and not external_triggers(reaction):
            unprompted += 1
    words = sum(
        len(value.split())
        for e in outputs
        for value in (e.args or {}).values()
        if isinstance(value, str)
    )
    all_reactions = reactions(timeline)
    return {
        "output_calls": float(len(outputs)),
        "unprompted_output": float(unprompted),
        "noop_steps": float(sum(1 for r in all_reactions if r.noop)),
        "failed_reactions": float(sum(1 for r in all_reactions if r.outcome != "ok")),
        "output_words": float(words),
        "turns": float(len(all_turns)),
        "turns_answered": float(answered),
        "noops_on_input": float(noops_on_input),
    }
