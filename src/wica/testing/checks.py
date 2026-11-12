"""Checks: rules judged after a run, over a window of its timeline. Implements specs/wica-tester.md,
"Scenarios" (the `check` step and the top-level `checks`).

A check is a pure function `(window, context) -> Finding`. It never waits and never stops a run: it
is evaluated after a **completed** run, over the **window** it concerns — from the input its step
follows to the next input (a step check), or the whole run (a top-level check). The built-ins are
generic: they hold for any WICA agent because the runtime primer states them (specs/agent.md,
"System prompt") or because any assistant persona promises them. The registry is open: an
application adds its own through `build_registry(extra_checks)`.

The per-run `measures` live here too: numbers taken from a whole run, never thresholded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from wica.instrumentation import ReactionTrace, TriggerTrace
from wica.testing.report import Finding, SutEvent

NOOP = "noop"
CANCEL = "cancel_command"
_COMMAND_KEY_PREFIX = "agent:command:"

Placement = Literal["anywhere", "step"]
ArgKind = Literal["none", "command"]


@dataclass(frozen=True)
class CheckSpec:
    """One check a scenario applies: its name and its argument, written as a bare name or as
    `{<name>: <arg>}` (e.g. `{"acts_when_asked": "dance"}`)."""

    name: str
    arg: Any = None


@dataclass(frozen=True)
class CheckContext:
    """What a check may need beyond its window: the output Command's name, the SUT's surface (its
    Inputs, which of them trigger, its Commands), the Inputs the scenario writes, the check's
    argument, and the run's start (so details cite run-relative times)."""

    output_command_name: str | None
    inputs: tuple[str, ...] = ()
    triggering_inputs: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    scenario_inputs: tuple[str, ...] = ()
    arg: Any = None
    origin: float = 0.0

    @property
    def turn_inputs(self) -> tuple[str, ...]:
        """The scenario's Inputs that trigger a step — the writes that open a turn."""
        return tuple(k for k in self.scenario_inputs if k in self.triggering_inputs)


CheckFn = Callable[[list[SutEvent], CheckContext], Finding]


@dataclass(frozen=True)
class Check:
    """A rule. `placement="step"` marks one that only means something for a given input (it is
    written on a step, never top-level); `arg="command"` one that takes a Command name, or
    `{<Command>: {<arg>: <value>}}`."""

    name: str
    fn: CheckFn
    placement: Placement = "anywhere"
    arg: ArgKind = "none"


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


def input_windows(timeline: list[SutEvent]) -> list[tuple[int, int]]:
    """One `(start, end)` per tester input, in order: from its `set` action to the next one (or the
    end of the timeline). The *n*-th input step of a scenario owns the *n*-th window."""
    sets = [i for i, e in enumerate(timeline) if e.kind == "action" and e.name == "set"]
    return [
        (start, sets[n + 1] if n + 1 < len(sets) else len(timeline))
        for n, start in enumerate(sets)
    ]


def _finding(
    name: str,
    window: list[SutEvent],
    ok: bool,
    detail: str,
    event: SutEvent | None = None,
) -> Finding:
    t = window[-1].t if window else 0.0
    return Finding(
        expectation=name, ok=ok, detail=detail, t=t, event=event, kind="check"
    )


def _at(event: SutEvent, context: CheckContext) -> str:
    return f"t={event.t - context.origin:.3f}"


# --- the built-in checks -------------------------------------------------------------------------


def answers_each_input(window: list[SutEvent], context: CheckContext) -> Finding:
    """Primer: "anything addressed to you should be answered" / "must be said by calling
    `<output>`" — every turn opened in the window by one of the scenario's triggering Inputs issues
    the output Command before the next such write."""
    name = "answers_each_input"
    output = context.output_command_name
    all_turns = turns(window, context.turn_inputs)
    for turn in all_turns:
        span = window[turn.start : turn.end]
        if not any(e.kind == "command_issued" and e.name == output for e in span):
            opening = window[turn.start]
            return _finding(
                name,
                window,
                False,
                f"the turn opened at {_at(opening, context)} by {turn.key!r} never issued "
                f"{output!r}",
                opening,
            )
    return _finding(
        name,
        window,
        True,
        f"{len(all_turns)} turn(s), each answered through {output!r}",
    )


def no_noop_on_input(window: list[SutEvent], context: CheckContext) -> Finding:
    """Primer: "never use `noop` to skip an expected response" — no reaction triggered by a write to
    one of the scenario's Inputs ends in `noop` alone."""
    name = "no_noop_on_input"
    wanted = set(context.scenario_inputs)
    for event in window:
        reaction = event.reaction
        if reaction is None or not reaction.noop or reaction.command_call_ids:
            continue
        keys = [t.key for t in external_triggers(reaction) if t.key in wanted]
        if keys:
            return _finding(
                name,
                window,
                False,
                f"reaction {reaction.reaction_id} (ended {_at(event, context)}) answered the "
                f"write to {keys[0]!r} with noop only",
                event,
            )
    return _finding(name, window, True, "no input answered with noop")


def no_failed_reactions(window: list[SutEvent], context: CheckContext) -> Finding:
    """Every reaction ended `ok` — no model error, no empty response."""
    name = "no_failed_reactions"
    for event in window:
        reaction = event.reaction
        if reaction is not None and reaction.outcome != "ok":
            detail = (
                f"reaction {reaction.reaction_id} ended {reaction.outcome!r} at "
                f"{_at(event, context)}"
            )
            if reaction.error:
                detail += f": {reaction.error}"
            return _finding(name, window, False, detail, event)
    return _finding(name, window, True, f"{len(reactions(window))} reaction(s), all ok")


def no_colliding_cancel(window: list[SutEvent], context: CheckContext) -> Finding:
    """Primer: "when one action must finish before another starts, do not issue both together" —
    no reaction cancels a Command and issues a new call of that same Command (cancel a running
    `dance` + `dance`): a response's Commands start at once, so the two race. Any other action
    beside the cancel is not a collision (a `set_emotion`), and neither is a new call of the
    output Command: cutting off a speech and saying "okay, stopping" is what a person does. The
    target is resolved from the window by the call id the cancel names; a target the window does
    not name is not judged."""
    name = "no_colliding_cancel"
    names_by_call = {
        e.call_id: e.name
        for e in window
        if e.kind in ("command_issued", "command_state") and e.call_id and e.name
    }
    for reaction_id, issued in commands_by_reaction(window).items():
        for cancel in (e for e in issued if e.name == CANCEL):
            target_id = str((cancel.args or {}).get("call_id", ""))
            target = names_by_call.get(target_id.removeprefix(_COMMAND_KEY_PREFIX))
            if target is None or target in (CANCEL, context.output_command_name):
                continue
            if any(e.name == target and e is not cancel for e in issued):
                return _finding(
                    name,
                    window,
                    False,
                    f"reaction {reaction_id} cancelled {target} and issued {target} again "
                    f"at {_at(cancel, context)}",
                    cancel,
                )
    return _finding(
        name, window, True, f"no {CANCEL} issued with a new call of its target"
    )


def cancels_when_asked(window: list[SutEvent], context: CheckContext) -> Finding:
    """The input it follows asks the SUT to stop: `cancel_command` is issued in its window."""
    name = "cancels_when_asked"
    for e in window:
        if e.kind == "command_issued" and e.name == CANCEL:
            return _finding(name, window, True, f"{CANCEL} at {_at(e, context)}", e)
    return _finding(name, window, False, f"{CANCEL} was never issued")


def command_arg(arg: Any) -> tuple[str, dict[str, Any]]:
    """`"dance"` → `("dance", {})`; `{"switch_user_tracking": {"user_id": "u2"}}` →
    `("switch_user_tracking", {"user_id": "u2"})`."""
    if isinstance(arg, str):
        return arg, {}
    if isinstance(arg, dict) and len(arg) == 1:
        ((command, where),) = arg.items()
        if isinstance(command, str) and isinstance(where, dict):
            return command, dict(where)
    raise ValueError(
        "expected a Command name or {<Command>: {<arg>: <value>}}, got " + repr(arg)
    )


def acts_when_asked(window: list[SutEvent], context: CheckContext) -> Finding:
    """The input it follows asks for a Command: that Command (with the given args) is issued in
    its window."""
    name = "acts_when_asked"
    command, where = command_arg(context.arg)
    for e in window:
        if e.kind != "command_issued" or e.name != command:
            continue
        args = e.args or {}
        if all(k in args and args[k] == v for k, v in where.items()):
            return _finding(
                name, window, True, f"{command}({args}) at {_at(e, context)}", e
            )
    return _finding(name, window, False, f"{command}({where}) was never issued")


def quiet_on_background_change(
    window: list[SutEvent], context: CheckContext
) -> Finding:
    """Primer: "use `noop` when nothing new needs a response" — the input it follows is a change
    the SUT was not asked about: the output Command is not issued in its window."""
    name = "quiet_on_background_change"
    output = context.output_command_name
    for e in window:
        if e.kind == "command_issued" and e.name == output:
            return _finding(
                name, window, False, f"{output} issued at {_at(e, context)}", e
            )
    return _finding(name, window, True, f"no {output} after the change")


CHECKS: dict[str, Check] = {
    check.name: check
    for check in (
        Check("answers_each_input", answers_each_input),
        Check("no_noop_on_input", no_noop_on_input),
        Check("no_failed_reactions", no_failed_reactions),
        Check("no_colliding_cancel", no_colliding_cancel),
        Check("cancels_when_asked", cancels_when_asked, placement="step"),
        Check("acts_when_asked", acts_when_asked, placement="step", arg="command"),
        Check(
            "quiet_on_background_change", quiet_on_background_change, placement="step"
        ),
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


def run_check(
    check: Check,
    spec: CheckSpec,
    window: list[SutEvent],
    context: CheckContext,
    *,
    label: str = "",
) -> Finding:
    """*check* over *window* with *spec*'s argument; *label* (e.g. "on input 3") is prefixed to the
    detail so a reader knows which window it judged."""
    finding = check.fn(window, replace(context, arg=spec.arg))
    if label:
        finding = replace(finding, detail=f"{label}: {finding.detail}")
    return finding


# --- measures ------------------------------------------------------------------------------------


def measures(timeline: list[SutEvent], context: CheckContext) -> dict[str, float]:
    """The per-run numbers next to the framework's metrics: `output_calls`, `extra_output` (output
    issued beyond a turn's first one — a second utterance in the same turn, however many steps the
    first took), `noop_steps`, `failed_reactions`, `output_words` (words in the output Command's
    string args), and the turn numbers a session is read by — `turns` (opened by the scenario's
    triggering Inputs, see `turns`), `turns_answered` (the output Command issued within the turn),
    `noops_on_input` (reactions that answered a write to a scenario Input with `noop` alone)."""
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
    wanted = set(context.scenario_inputs)
    noops_on_input = sum(
        1
        for r in reactions(timeline)
        if r.noop
        and not r.command_call_ids
        and any(t.key in wanted for t in external_triggers(r))
    )
    words = sum(
        len(value.split())
        for e in outputs
        for value in (e.args or {}).values()
        if isinstance(value, str)
    )
    all_reactions = reactions(timeline)
    return {
        "output_calls": float(len(outputs)),
        "extra_output": float(len(outputs) - answered),
        "noop_steps": float(sum(1 for r in all_reactions if r.noop)),
        "failed_reactions": float(sum(1 for r in all_reactions if r.outcome != "ok")),
        "output_words": float(words),
        "turns": float(len(all_turns)),
        "turns_answered": float(answered),
        "noops_on_input": float(noops_on_input),
    }


def answer_latencies(timeline: list[SutEvent], context: CheckContext) -> list[float]:
    """One sample per answered turn: from the write that opened it to its first output Command —
    the time to the first word, whether the model spoke on its first step or after a side Command
    (an emotion change) whose completion woke the step that spoke. Unanswered turns give none."""
    output = context.output_command_name
    samples: list[float] = []
    for turn in turns(timeline, context.turn_inputs):
        first = next(
            (
                e
                for e in timeline[turn.start : turn.end]
                if e.kind == "command_issued" and e.name == output
            ),
            None,
        )
        if first is not None:
            samples.append(first.t - turn.t)
    return samples


def registry_or_default(registry: Mapping[str, Check] | None) -> Mapping[str, Check]:
    return registry if registry is not None else CHECKS
