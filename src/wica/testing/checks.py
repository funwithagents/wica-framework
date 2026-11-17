"""Checks: rules judged after a run, over a window of its timeline. Implements specs/wica-tester.md,
"Scenarios" (the `check` step and the top-level `checks`).

A check is a pure function `(window, context) -> Finding`. It never waits and never stops a run: it
is evaluated after a **completed** run, over the **window** it concerns — from the input its step
follows to the next input (a step check), or the whole run (a top-level check). The built-ins are
generic: they hold for any WICA agent because the runtime primer states them (specs/agent.md,
"System prompt") or because any assistant persona promises them. The registry is open: an
application adds its own through `build_registry(extra_checks)`.

An **output** is a `command_issued` record whose `is_output` is set (one per tag of the model's
text, reported once its body is complete), and the outputs of one reaction are listed in its
record's `command_call_ids`; a reaction that **declined** is one whose record has the `noop` flag
set (it delivered no output and issued no tool call). These are fields the framework reports
(specs/instrumentation.md): the checks tell speech from action and from silence by reading them,
never by a Command's name or a position in a list.

The per-run `measures` live here too: numbers taken from a whole run, never thresholded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from wica.instrumentation import ReactionTrace, TriggerTrace
from wica.testing.report import Finding, SutEvent

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
    """`reaction_id → its command_issued events`, outputs included (joined through
    `command_call_ids`)."""
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


def output_issued(timeline: list[SutEvent]) -> list[SutEvent]:
    """The outputs reported in *timeline*, in order: the `command_issued` records whose
    `is_output` is set."""
    return [e for e in timeline if e.kind == "command_issued" and e.is_output]


def reply_of(timeline: list[SutEvent]) -> dict[str, int]:
    """`call_id → reaction_id` for every Command a reaction record in *timeline* lists in its
    `command_call_ids` (outputs included): the outputs of one reaction are one reply."""
    return {c: r.reaction_id for r in reactions(timeline) for c in r.command_call_ids}


def replies(outputs: list[SutEvent], reply: Mapping[str, int]) -> int:
    """How many replies *outputs* make: one per reaction that issued some of them (through
    *reply*, see `reply_of`); an output with no reaction record in the timeline counts alone."""
    groups: set[tuple[str, object]] = set()
    for e in outputs:
        reaction_id = reply.get(e.call_id or "")
        groups.add(
            ("call", e.call_id) if reaction_id is None else ("reaction", reaction_id)
        )
    return len(groups)


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
    """Primer: "anything addressed to you should be answered" — every turn opened in the window by
    one of the scenario's triggering Inputs is answered by an output before the next such write,
    whichever step issued it. Private text is not an answer."""
    name = "answers_each_input"
    output = context.output_command_name
    all_turns = turns(window, context.turn_inputs)
    for turn in all_turns:
        if not output_issued(window[turn.start : turn.end]):
            opening = window[turn.start]
            return _finding(
                name,
                window,
                False,
                f"the turn opened at {_at(opening, context)} by {turn.key!r} was never "
                f"answered by an output ({output!r})",
                opening,
            )
    return _finding(
        name,
        window,
        True,
        f"{len(all_turns)} turn(s), each answered through {output!r}",
    )


def no_noop_on_input(window: list[SutEvent], context: CheckContext) -> Finding:
    """Primer: "never use `<noop/>` to skip an expected response" — no reaction triggered by a write
    to one of the scenario's Inputs declines: its record's `noop` flag is set — it delivered no
    output and issued no tool call (`<noop/>`, private text then `<noop/>`, or an empty
    response)."""
    name = "no_noop_on_input"
    wanted = set(context.scenario_inputs)
    for event in window:
        reaction = event.reaction
        if reaction is None or not reaction.noop:
            continue
        keys = [t.key for t in external_triggers(reaction) if t.key in wanted]
        if keys:
            return _finding(
                name,
                window,
                False,
                f"reaction {reaction.reaction_id} (ended {_at(event, context)}) declined the "
                f"write to {keys[0]!r} (noop)",
                event,
            )
    return _finding(name, window, True, "no input declined (noop)")


_NOT_FAILED = ("ok", "empty")


def failed(reaction: ReactionTrace) -> bool:
    """The reaction failed: a model error, a deadline, a response still rejected, a cancellation.
    An empty response is not a failure: it is a decline (`noop`), judged by the checks on
    declining."""
    return reaction.outcome not in _NOT_FAILED


def no_failed_reactions(window: list[SutEvent], context: CheckContext) -> Finding:
    """Every reaction ended `ok` or `empty` — no model error, no timeout, no rejected response, no
    cancellation. An empty response is the model declining, not failing."""
    name = "no_failed_reactions"
    for event in window:
        reaction = event.reaction
        if reaction is not None and failed(reaction):
            detail = (
                f"reaction {reaction.reaction_id} ended {reaction.outcome!r} at "
                f"{_at(event, context)}"
            )
            if reaction.error:
                detail += f": {reaction.error}"
            return _finding(name, window, False, detail, event)
    return _finding(
        name, window, True, f"{len(reactions(window))} reaction(s), none failed"
    )


def no_colliding_cancel(window: list[SutEvent], context: CheckContext) -> Finding:
    """Primer: "when one action must finish before another starts, do not issue both together" —
    no reaction cancels a Command and issues a new call of that same Command (cancel a running
    `dance` + `dance`): a response's Commands start at once, so the two race. Any other action
    beside the cancel is not a collision (a `set_emotion`), and neither is an output: cutting off a
    speech and saying "okay, stopping" is what a person does, and a response that speaks replaces
    the running output anyway. The check reads tool calls only, so a cancel whose target is an
    output never collides. The target is resolved from the window by the call id the cancel
    names; a target the window does not name is not judged."""
    name = "no_colliding_cancel"
    names_by_call = {
        e.call_id: e.name
        for e in window
        if e.kind in ("command_issued", "command_state")
        and e.call_id
        and e.name
        and not e.is_output
    }
    for reaction_id, calls in commands_by_reaction(window).items():
        issued = [e for e in calls if not e.is_output]  # tool calls only
        for cancel in (e for e in issued if e.name == CANCEL):
            target_id = str((cancel.args or {}).get("call_id", ""))
            target = names_by_call.get(target_id.removeprefix(_COMMAND_KEY_PREFIX))
            if target is None or target == CANCEL:
                continue  # unnamed in the window, a cancel, or an output
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
    """The input it follows asks the SUT to stop: something is stopped in its window —
    `cancel_command` is issued, or an output that was running when the window opened reaches
    `cancelled` in it (speaking again cuts off the running output with no `cancel_command`, and
    that is a correct way to stop talking). For any other Command only `cancel_command` counts.
    Under the one-voice rule any new output cuts the running one off, so for speech this
    confirms the speech stopped: it holds as soon as the model says anything, and breaks only
    when it stays silent or only acts."""
    name = "cancels_when_asked"
    for e in window:
        if e.kind == "command_issued" and e.name == CANCEL and not e.is_output:
            return _finding(name, window, True, f"{CANCEL} at {_at(e, context)}", e)
    # An output whose opening (its first `running` state) is inside the window started in it: it
    # was not running when the window opened.
    opened_in_window = {
        e.call_id
        for e in window
        if e.kind == "command_state" and e.is_output and e.state == "running"
    }
    for e in window:
        if (
            e.kind == "command_state"
            and e.is_output
            and e.state == "cancelled"
            and e.call_id not in opened_in_window
        ):
            return _finding(
                name,
                window,
                True,
                f"the running {e.name} ({e.call_id}) cut off at {_at(e, context)}",
                e,
            )
    return _finding(
        name,
        window,
        False,
        f"{CANCEL} was never issued and no running output was cut off",
    )


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
    """Primer: "use `<noop/>` when nothing new needs a response" — the input it follows is a change
    the SUT was not asked about: no output is reported in its window. Private text and Commands
    issued as tool calls do not break it."""
    name = "quiet_on_background_change"
    output = context.output_command_name
    said = output_issued(window)
    if said:
        e = said[0]
        return _finding(name, window, False, f"{e.name} output at {_at(e, context)}", e)
    return _finding(name, window, True, f"no {output} output after the change")


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
    """The per-run numbers next to the framework's metrics: `output_calls` (outputs reported, one
    per tag), `extra_output` (replies beyond a turn's first one — a second reply in the same turn,
    however many steps the first took; the outputs one reaction delivered, several tags
    included, are one reply), `noop_steps` (reactions that declined: the record's `noop` flag),
    `failed_reactions` (an empty response is a decline, not a failure), `output_words` (words in
    the outputs' `text`), and the turn numbers a session is read by — `turns` (opened by the
    scenario's triggering Inputs, see `turns`), `turns_answered` (an output reported within the
    turn), `noops_on_input` (reactions that declined a write to a scenario Input)."""
    outputs = output_issued(timeline)
    reply = reply_of(timeline)
    all_turns = turns(timeline, context.turn_inputs)
    answered = sum(
        1 for turn in all_turns if output_issued(timeline[turn.start : turn.end])
    )
    wanted = set(context.scenario_inputs)
    all_reactions = reactions(timeline)
    noops_on_input = sum(
        1
        for r in all_reactions
        if r.noop and any(t.key in wanted for t in external_triggers(r))
    )
    words = sum(len(str((e.args or {}).get("text", "")).split()) for e in outputs)
    return {
        "output_calls": float(len(outputs)),
        "extra_output": float(max(replies(outputs, reply) - answered, 0)),
        "noop_steps": float(sum(1 for r in all_reactions if r.noop)),
        "failed_reactions": float(sum(1 for r in all_reactions if failed(r))),
        "output_words": float(words),
        "turns": float(len(all_turns)),
        "turns_answered": float(answered),
        "noops_on_input": float(noops_on_input),
    }


def answer_latencies(timeline: list[SutEvent], context: CheckContext) -> list[float]:
    """One sample per answered turn: from the write that opened it to the first words of its first
    output — the output's own start (`CommandTrace.started_at`, when it opened), whether the model
    spoke on its first step or after a side Command (an emotion change) whose completion woke the
    step that spoke. When the output's end record is not in the timeline, the time the output was
    reported stands in. Unanswered turns give none."""
    first_of = {
        e.command.call_id: e.command.started_at
        for e in timeline
        if e.kind == "command_ended" and e.command is not None and e.command.is_output
    }
    samples: list[float] = []
    for turn in turns(timeline, context.turn_inputs):
        answers = output_issued(timeline[turn.start : turn.end])
        if not answers:
            continue
        first = answers[0]
        opened = first_of.get(first.call_id or "")
        reached = opened.timestamp() if opened is not None else first.t
        samples.append(reached - turn.t)
    return samples


def registry_or_default(registry: Mapping[str, Check] | None) -> Mapping[str, Check]:
    return registry if registry is not None else CHECKS
