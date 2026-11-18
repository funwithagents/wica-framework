"""The model's text output, LangChain-free: the model addresses its user by writing a tag in its
response text. See specs/agent.md ("Output: tags in the model's text") and specs/commands.md ("The
output Command — what the model says to its user").

    <say emotion="happy">Hello!</say>   an output: the output Command's tag, body and attributes
    <noop/>                             do nothing (only as the whole response)
    <next>what remains to do</next>     a private note asking for one more step after the output

Here: the `OutputStream` an output Command receives, the incremental parser the Agent feeds with
the model's text as it streams, and the reading of an output Command's signature.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import typing
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

NOOP_TAG = "noop"
NEXT_TAG = "next"
_TAG_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")


class OutputStream:
    """The text of one output, as the model writes it: an async iterator of text pieces that ends
    when the output's body ends. `text` is everything received so far; `read()` waits for the end
    and returns the whole body. An output Command cancelled part-way may set `heard` to what was
    actually delivered: history then keeps that."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._closed = False
        self.text = ""
        self.heard: str | None = None

    def push(self, piece: str) -> None:
        if piece and not self._closed:
            self.text += piece
            self._queue.put_nowait(piece)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(None)

    @property
    def closed(self) -> bool:
        return self._closed

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        piece = await self._queue.get()
        if piece is None:
            self._queue.put_nowait(None)  # stay ended for a later iteration
            raise StopAsyncIteration
        return piece

    async def read(self) -> str:
        async for _ in self:
            pass
        return self.text


def is_tag_name(name: str) -> bool:
    """Whether `name` can be an output tag's name."""
    return bool(_TAG_NAME.match(name)) and name.lower() not in (NOOP_TAG, NEXT_TAG)


@dataclass(frozen=True)
class OutputSignature:
    """How an output Command takes an output: its text parameter (a stream or the whole text) and
    its attributes, each with the values it allows (None: any) and its default."""

    text_parameter: str
    streams: bool
    attributes: dict[str, tuple[str, ...] | None]
    defaults: dict[str, str]


def output_signature(fn: Callable[..., Any]) -> OutputSignature:
    """Read `fn`'s signature as an output Command's. The first parameter receives the text —
    annotated `OutputStream` it gets the stream, `str` (or nothing) the whole text. Every other
    parameter is a tag attribute: `str` or a `Literal` of strings, with a default. Anything else
    raises ValueError. See specs/commands.md ("The output Command")."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception as exc:
        raise ValueError(f"output Command: cannot read the type hints ({exc})") from exc
    parameters = [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if any(
        p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        for p in inspect.signature(fn).parameters.values()
    ):
        raise ValueError("output Command: *args / **kwargs are not supported")
    if not parameters:
        raise ValueError("output Command: needs a first parameter for the text")
    first, rest = parameters[0], parameters[1:]
    annotation = hints.get(first.name, str)
    if annotation is OutputStream:
        streams = True
    elif annotation is str:
        streams = False
    else:
        raise ValueError(
            f"output Command: the text parameter {first.name!r} must be annotated "
            f"OutputStream or str, not {annotation!r}"
        )
    attributes: dict[str, tuple[str, ...] | None] = {}
    defaults: dict[str, str] = {}
    for parameter in rest:
        hint = hints.get(parameter.name, str)
        if hint is str:
            allowed = None
        elif typing.get_origin(hint) is Literal and all(
            isinstance(value, str) for value in typing.get_args(hint)
        ):
            allowed = tuple(typing.get_args(hint))
        else:
            raise ValueError(
                f"output Command: attribute {parameter.name!r} must be annotated str or a "
                f"Literal of strings, not {hint!r}"
            )
        if parameter.default is parameter.empty or not isinstance(
            parameter.default, str
        ):
            raise ValueError(
                f"output Command: attribute {parameter.name!r} needs a string default"
            )
        attributes[parameter.name] = allowed
        defaults[parameter.name] = parameter.default
    return OutputSignature(first.name, streams, attributes, defaults)


_ATTRIBUTE = re.compile(r"""([\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>/]+))""")


def parse_attributes(raw: str) -> dict[str, str]:
    """The attributes inside an opening tag (`emotion="happy" speed=fast`)."""
    return {
        m.group(1): next(g for g in m.groups()[1:] if g is not None)
        for m in _ATTRIBUTE.finditer(raw)
    }


@dataclass
class ParsedOutput:
    """One output of a response: what the model wrote for it."""

    attributes: dict[str, str]
    text: str = ""
    tagged: bool = (
        True  # False: the lenient whole-text output of a response with no tag
    )
    raw_attributes: str = ""


@dataclass
class ParseResult:
    """What one response's text came to."""

    raw: str  # the text exactly as the model wrote it
    outputs: list[ParsedOutput]
    private: str  # the text outside the tags
    note: str | None
    noop: bool  # nothing was output: a noop tag, or no text at all
    # The response in order, to rebuild it after a cut: ("text", str) | ("output", index) |
    # ("note", str) | ("noop", "").
    segments: list[tuple[str, Any]] = field(default_factory=list)

    def rebuilt(self, name: str, kept: dict[int, str], *, private: bool = True) -> str:
        """The response's text with each output in `kept` cut to what it maps to (an output
        mapped to "" is removed), and without its private text when `private` is False. See
        specs/agent.md ("History keeps what the model wrote", "Garbled text")."""
        pieces: list[str] = []
        for kind, value in self.segments:
            if kind == "text":
                if private:
                    pieces.append(value)
            elif kind == "note":
                pieces.append(f"<{NEXT_TAG}>{value}</{NEXT_TAG}>")
            elif kind == "noop":
                pieces.append(f"<{NOOP_TAG}/>")
            else:
                output = self.outputs[value]
                text = kept.get(value, output.text)
                if not text:
                    continue
                if output.tagged:
                    pieces.append(f"<{name}{output.raw_attributes}>{text}</{name}>")
                else:
                    pieces.append(text)
        return "".join(pieces).strip()


class TextOutputParser:
    """Incremental parser for one response: `feed(piece)` as text arrives, `finish()` at the end.

    `on_open(attributes)` fires at an output's first words, `on_delta(text)` for each piece of
    its body, `on_close()` when it ends, so the caller delivers before the response is complete.
    Each output tag is its own output. With `name=None` (no output Command) nothing is an output:
    all text is private and only the noop tag is recognised.

    Lenient: text with no output tag is one output delivered at the end; a tag left open is
    closed by the end of the response; a noop tag beside an output is ignored; any other `<…>` is
    ordinary text."""

    def __init__(
        self,
        name: str | None,
        *,
        on_open: Callable[[dict[str, str]], None] | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._name = name
        self._on_open = on_open or (lambda attributes: None)
        self._on_delta = on_delta or (lambda text: None)
        self._on_close = on_close or (lambda: None)
        alternatives = [r"(?P<noop><noop\s*/?>)", r"(?P<noopclose></noop>)"]
        self._prefixes = ["<noop", "</noop>"]
        self._close = ""
        if name is not None:
            n = re.escape(name)
            alternatives = [
                rf"<{n}(?P<attrs>\s[^>]*)?>",
                rf"(?P<outclose></{n}>)",
                r"(?P<next><next>)",
                r"(?P<nextclose></next>)",
                *alternatives,
            ]
            self._prefixes += [
                f"<{name}".lower(),
                f"</{name}>".lower(),
                "<next>",
                "</next>",
            ]
            self._close = f"</{name}>"
        self._tags = re.compile("|".join(alternatives), re.I)
        self._buf = ""
        self._state: Literal["outside", "output", "next"] = "outside"
        self._raw: list[str] = []
        self._outputs: list[ParsedOutput] = []
        self._segments: list[tuple[str, Any]] = []
        self._notes: list[str] = []
        self._note_buf: list[str] = []
        self._pending: tuple[dict[str, str], str] | None = (
            None  # an opened tag with no text yet
        )
        self._open = False
        self._noop = False

    @property
    def text(self) -> str:
        """The text fed so far, exactly as the model wrote it."""
        return "".join(self._raw)

    @property
    def started(self) -> bool:
        """Whether an output has started: from then on the response cannot be taken back."""
        return bool(self._outputs)

    def _emit(self, text: str) -> None:
        if not text:
            return
        if not self._open:
            text = text.lstrip()  # an output starts at its first word
            if not text:
                return
            attributes, raw_attributes = self._pending or ({}, "")
            self._pending = None
            self._open = True
            self._outputs.append(
                ParsedOutput(attributes, raw_attributes=raw_attributes)
            )
            self._segments.append(("output", len(self._outputs) - 1))
            self._on_open(dict(attributes))
        self._outputs[-1].text += text
        self._on_delta(text)

    def _end_output(self) -> None:
        self._pending = None
        if self._open:
            self._open = False
            self._on_close()

    def _private(self, text: str) -> None:
        if text:
            if self._segments and self._segments[-1][0] == "text":
                self._segments[-1] = ("text", self._segments[-1][1] + text)
            else:
                self._segments.append(("text", text))

    def feed(self, piece: str) -> None:
        if not piece:
            return
        self._raw.append(piece)
        self._buf += piece
        self._drain(final=False)

    @staticmethod
    def _tail_overlap(buf: str, token: str) -> int:
        """How many trailing characters of `buf` could be the start of `token`."""
        low, tok = buf.lower(), token.lower()
        for size in range(min(len(tok) - 1, len(low)), 0, -1):
            if tok.startswith(low[-size:]):
                return size
        return 0

    def _could_be_tag(self, rest: str) -> bool:
        low = rest.lower()
        return ">" not in low and any(
            p.startswith(low) or low.startswith(p) for p in self._prefixes
        )

    def _until(self, closing: str, sink: Callable[[str], None], *, final: bool) -> bool:
        """Consume the buffer up to `closing`; True when the closing tag was found."""
        end = self._buf.lower().find(closing.lower())
        if end >= 0:
            sink(self._buf[:end])
            self._buf = self._buf[end + len(closing) :]
            return True
        keep = 0 if final else self._tail_overlap(self._buf, closing)
        sink(self._buf[: len(self._buf) - keep])
        self._buf = self._buf[len(self._buf) - keep :]
        return False

    def _drain(self, *, final: bool) -> None:
        while self._buf:
            if self._state == "output":
                if not self._until(self._close, self._emit, final=final):
                    return
                self._end_output()
                self._state = "outside"
                continue
            if self._state == "next":
                if not self._until("</next>", self._note_buf.append, final=final):
                    return
                self._end_note()
                continue
            lt = self._buf.find("<")
            if lt < 0:
                self._private(self._buf)
                self._buf = ""
                return
            self._private(self._buf[:lt])
            self._buf = self._buf[lt:]
            match = self._tags.match(self._buf)
            if match is None:
                if not final and self._could_be_tag(self._buf):
                    return  # wait for the rest of the tag
                self._private("<")
                self._buf = self._buf[1:]
                continue
            self._buf = self._buf[match.end() :]
            groups = match.groupdict()
            if groups.get("next"):
                self._state = "next"
            elif groups.get("noop"):
                self._noop = True
                self._segments.append(("noop", ""))
            elif (
                groups.get("outclose")
                or groups.get("nextclose")
                or groups.get("noopclose")
            ):
                pass  # a stray closing tag: dropped
            else:
                raw_attributes = groups.get("attrs") or ""
                self._pending = (
                    parse_attributes(raw_attributes),
                    raw_attributes.rstrip(),
                )
                self._state = "output"

    def _end_note(self) -> None:
        note = "".join(self._note_buf).strip()
        self._note_buf = []
        if note:
            self._notes.append(note)
            self._segments.append(("note", note))
        self._state = "outside"

    def finish(self) -> ParseResult:
        self._drain(final=True)
        if self._state == "next":
            self._end_note()
        private = "".join(v for k, v in self._segments if k == "text").strip()
        if self._name is not None and not self._outputs and private and not self._noop:
            # The lenient fallback: a response with no output tag is delivered whole.
            self._segments = [s for s in self._segments if s[0] != "text"]
            self._pending = None
            self._open = False
            self._outputs.append(ParsedOutput({}, tagged=False))
            self._segments.insert(0, ("output", 0))
            self._on_open({})
            self._outputs[0].text = private
            self._on_delta(private)
            self._open = True
            private = ""
        self._end_output()
        for output in self._outputs:
            output.text = output.text.strip()
        return ParseResult(
            raw="".join(self._raw),
            outputs=self._outputs,
            private=private,
            note=" / ".join(self._notes) or None,
            noop=not self._outputs,
            segments=self._segments,
        )
