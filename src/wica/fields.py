"""Config fields declared as data (specs/setup.md, "Fields are declared once").

A module lists its fields once, as `Field`s. `read_fields` validates a config
block against them and returns the values, raising on the first invalid one;
`field_problems` runs the same rules and returns every failure; `fields_schema`
emits the JSON Schema a form is drawn from. One declaration feeds all three, so
what a form offers and what a constructor accepts cannot drift apart.

Standard library only, and nothing here knows about speech: the error type is
the caller's.

A verbatim copy of tts-engine's `tts_engine/modules/fields.py`, the helper the
family's libraries share: it is to move to a shared core library, which will
replace the copies. Change it there, not here.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "Field",
    "describe_bounds",
    "field_problems",
    "fields_schema",
    "is_finite",
    "read_fields",
]

# "objects" is a list of objects: only its being a list is checked here, its
# content is the module's to validate.
FieldKind = Literal["string", "number", "integer", "boolean", "objects"]


@dataclass(frozen=True)
class Field:
    """One field of a config block.

    Absent, a field takes `default`. A present value is validated; an explicit
    `null` counts as absent when there is no `default`, and is rejected like
    any other wrong value when there is one. `required` makes absence itself
    invalid.
    """

    name: str
    # One kind, or several for a field that accepts more than one (a speaker
    # given by name or by index): a value is valid when it is for one of them.
    kind: FieldKind | tuple[FieldKind, ...]
    description: str
    default: Any = None
    # The value the module ends up using when the field is absent and `default`
    # is None (a fallback applied in code, or the provider's own default): shown
    # by a form, never used to validate.
    implied: Any = None
    required: bool = False
    # False refuses an explicit `null` even without a `default` (the field may
    # still be absent).
    nullable: bool = True
    choices: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: float | None = None
    # A string field is non-empty unless this is set.
    allow_empty: bool = False
    advanced: bool = False
    # "env": the value names an environment variable holding a secret;
    # "literal": the value is the secret itself.
    secret: Literal["env", "literal"] | None = None
    # The named list a form can fill this field from.
    options: str | None = None
    # The whole message of an invalid value, when the default wording does not fit.
    message: str | None = None


def is_finite(value: float) -> bool:
    """`math.isfinite`, with an integer too large for a float (valid JSON,
    `1e400` written out) counted as not finite instead of raising."""
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _bounds(
    minimum: float | None, maximum: float | None, exclusive_minimum: float | None
) -> str:
    bounds = []
    if exclusive_minimum is not None:
        bounds.append(f"> {exclusive_minimum:g}")
    if minimum is not None:
        bounds.append(f">= {minimum:g}")
    if maximum is not None:
        bounds.append(f"<= {maximum:g}")
    return " and ".join(bounds)


def describe_bounds(
    minimum: float | None, maximum: float | None, exclusive_minimum: float | None
) -> str:
    """What a number must be, in words: "a finite number between 0 and 1"."""
    if minimum is not None and maximum is not None:
        return f"a finite number between {minimum:g} and {maximum:g}"
    bounds = _bounds(minimum, maximum, exclusive_minimum)
    return "a finite number" + (" " + bounds if bounds else "")


def _kinds(field: Field) -> tuple[FieldKind, ...]:
    return (field.kind,) if isinstance(field.kind, str) else field.kind


def _expected_kind(field: Field, kind: FieldKind) -> str:
    if kind == "string":
        return "a string" if field.allow_empty else "a non-empty string"
    if kind == "number":
        return describe_bounds(field.minimum, field.maximum, field.exclusive_minimum)
    if kind == "integer":
        bounds = _bounds(field.minimum, field.maximum, field.exclusive_minimum)
        return "an integer" + (" " + bounds if bounds else "")
    if kind == "objects":
        return "a list"
    return "a boolean"


def _expected(field: Field) -> str:
    if field.choices is not None:
        return "one of " + ", ".join(repr(choice) for choice in field.choices)
    return " or ".join(_expected_kind(field, kind) for kind in _kinds(field))


def _in_bounds(value: float, field: Field) -> bool:
    return not (
        (field.minimum is not None and value < field.minimum)
        or (field.maximum is not None and value > field.maximum)
        or (field.exclusive_minimum is not None and value <= field.exclusive_minimum)
    )


def _valid_kind(value: Any, field: Field, kind: FieldKind) -> bool:
    if kind == "string":
        return isinstance(value, str) and (field.allow_empty or bool(value))
    if kind == "number":
        return (
            not isinstance(value, bool)
            and isinstance(value, int | float)
            and is_finite(value)
            and _in_bounds(value, field)
        )
    if kind == "integer":
        return (
            not isinstance(value, bool)
            and isinstance(value, int)
            and _in_bounds(value, field)
        )
    if kind == "objects":
        return isinstance(value, list)
    return isinstance(value, bool)


def _valid(value: Any, field: Field) -> bool:
    return any(_valid_kind(value, field, kind) for kind in _kinds(field)) and (
        field.choices is None or value in field.choices
    )


def _read(config: dict, field: Field, label: str) -> tuple[Any, str | None]:
    """The field's value and, when it is invalid, the message saying so."""
    value = config.get(field.name, field.default)
    absent = field.nullable or field.name not in config
    if value is None and field.default is None and not field.required and absent:
        return None, None
    if not _valid(value, field):
        message = field.message or (
            f"{label} '{field.name}' must be {_expected(field)}"
        )
        return None, message
    if field.kind == "number":
        return float(value), None  # pyright: ignore[reportArgumentType] - validated above
    return value, None


def read_fields(
    config: dict,
    fields: Sequence[Field],
    *,
    label: str,
    error: type[Exception] = ValueError,
) -> dict[str, Any]:
    """Each field's value (its default when absent), raising `error` on the
    first invalid one. `label` ("Tone module") prefixes the message. A number
    comes back as a float."""
    values: dict[str, Any] = {}
    for field in fields:
        value, message = _read(config, field, label)
        if message is not None:
            raise error(message)
        values[field.name] = value
    return values


def field_problems(
    config: dict, fields: Sequence[Field], *, label: str
) -> list[tuple[str, str]]:
    """Every invalid field as `(name, message)`, with `read_fields`' rules and
    wording; empty when all are valid."""
    problems = []
    for field in fields:
        _, message = _read(config, field, label)
        if message is not None:
            problems.append((field.name, message))
    return problems


def fields_schema(fields: Sequence[Field]) -> dict[str, Any]:
    """The fields as the `properties` and `required` of a JSON Schema object."""
    properties: dict[str, Any] = {}
    for field in fields:
        kinds: list[str] = [
            "array" if kind == "objects" else kind for kind in _kinds(field)
        ]
        if field.default is None and not field.required and field.nullable:
            kinds.append("null")
        schema: dict[str, Any] = {
            "type": kinds[0] if len(kinds) == 1 else kinds,
            **({"items": {"type": "object"}} if "array" in kinds else {}),
            "description": field.description,
        }
        shown = field.default if field.default is not None else field.implied
        if shown is not None:
            schema["default"] = shown
        if field.choices is not None:
            schema["enum"] = list(field.choices)
        if field.minimum is not None:
            schema["minimum"] = field.minimum
        if field.maximum is not None:
            schema["maximum"] = field.maximum
        if field.exclusive_minimum is not None:
            schema["exclusiveMinimum"] = field.exclusive_minimum
        if field.advanced:
            schema["x-advanced"] = True
        if field.secret == "env":
            schema["x-secret-env"] = True
        elif field.secret == "literal":
            schema["x-secret"] = True
        if field.options is not None:
            schema["x-options"] = field.options
        properties[field.name] = schema
    return {
        "properties": properties,
        "required": [field.name for field in fields if field.required],
    }
