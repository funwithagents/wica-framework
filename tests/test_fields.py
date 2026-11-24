"""Fields declared as data: one declaration validates, reports and describes.

See specs/setup.md, "Fields are declared once".
"""

import math

import pytest

from wica.fields import (
    Field,
    field_problems,
    fields_schema,
    read_fields,
)

_LABEL = "Demo module"

NAME = Field("name", "string", "A name.")
MODEL = Field("model", "string", "A model.", default="small")
SPEED = Field("speed", "number", "A rate.", minimum=0.5, maximum=2.0)
TIMEOUT = Field("timeout", "number", "Seconds.", default=30.0, exclusive_minimum=0)
THREADS = Field("threads", "integer", "Threads.", minimum=1)
RATE = Field("rate", "integer", "Hz.", default=24000, exclusive_minimum=0)
FAST = Field("fast", "boolean", "Go fast.", default=False)
MODE = Field("mode", "string", "A mode.", default="a", choices=("a", "b"))
VOICE = Field(
    "voice", "string", "A voice.", required=True, message="Demo needs a 'voice'"
)
KEY = Field("key", "string", "A key.", allow_empty=True)

# (field, config value, the message of the refusal)
INVALID = [
    (NAME, "", "Demo module 'name' must be a non-empty string"),
    (NAME, 3, "Demo module 'name' must be a non-empty string"),
    (MODEL, None, "Demo module 'model' must be a non-empty string"),
    (SPEED, 3, "Demo module 'speed' must be a finite number between 0.5 and 2"),
    (SPEED, True, "Demo module 'speed' must be a finite number between 0.5 and 2"),
    (SPEED, "1", "Demo module 'speed' must be a finite number between 0.5 and 2"),
    (TIMEOUT, 0, "Demo module 'timeout' must be a finite number > 0"),
    (TIMEOUT, math.nan, "Demo module 'timeout' must be a finite number > 0"),
    (TIMEOUT, math.inf, "Demo module 'timeout' must be a finite number > 0"),
    (TIMEOUT, 10**400, "Demo module 'timeout' must be a finite number > 0"),
    (TIMEOUT, None, "Demo module 'timeout' must be a finite number > 0"),
    (THREADS, 0, "Demo module 'threads' must be an integer >= 1"),
    (THREADS, 2.0, "Demo module 'threads' must be an integer >= 1"),
    (THREADS, True, "Demo module 'threads' must be an integer >= 1"),
    (RATE, 0, "Demo module 'rate' must be an integer > 0"),
    (FAST, 1, "Demo module 'fast' must be a boolean"),
    (MODE, "c", "Demo module 'mode' must be one of 'a', 'b'"),
    (VOICE, "", "Demo needs a 'voice'"),
    (VOICE, None, "Demo needs a 'voice'"),
    (KEY, 5, "Demo module 'key' must be a string"),
]


@pytest.mark.parametrize(("field", "value", "message"), INVALID)
def test_an_invalid_value_is_raised_and_reported_with_the_same_message(
    field: Field, value: object, message: str
) -> None:
    config = {field.name: value}

    with pytest.raises(ValueError, match="^" + message.replace("(", r"\(") + "$"):
        read_fields(config, [field], label=_LABEL)
    assert field_problems(config, [field], label=_LABEL) == [(field.name, message)]


def test_a_required_field_that_is_absent_is_invalid() -> None:
    assert field_problems({}, [VOICE], label=_LABEL) == [
        ("voice", "Demo needs a 'voice'")
    ]


def test_absent_fields_take_their_default_and_null_counts_as_absent_without_one() -> (
    None
):
    fields = [NAME, MODEL, SPEED, TIMEOUT, THREADS, RATE, FAST, MODE, KEY]

    assert read_fields({"name": None, "speed": None}, fields, label=_LABEL) == {
        "name": None,
        "model": "small",
        "speed": None,
        "timeout": 30.0,
        "threads": None,
        "rate": 24000,
        "fast": False,
        "mode": "a",
        "key": None,
    }
    assert field_problems({}, fields, label=_LABEL) == []


def test_values_come_back_typed() -> None:
    values = read_fields(
        {"speed": 1, "threads": 4, "fast": True, "key": "", "voice": "v"},
        [SPEED, THREADS, FAST, KEY, VOICE],
        label=_LABEL,
    )

    assert values == {"speed": 1.0, "threads": 4, "fast": True, "key": "", "voice": "v"}
    assert isinstance(values["speed"], float)
    assert isinstance(values["threads"], int)


def test_every_invalid_field_is_reported_but_only_the_first_is_raised() -> None:
    config = {"name": "", "threads": 0, "fast": True}
    fields = [NAME, THREADS, FAST]

    assert [name for name, _ in field_problems(config, fields, label=_LABEL)] == [
        "name",
        "threads",
    ]
    with pytest.raises(KeyError, match="'name'"):
        read_fields(config, fields, label=_LABEL, error=KeyError)


def test_the_schema_carries_what_a_form_needs() -> None:
    fields = [
        VOICE,
        MODEL,
        SPEED,
        TIMEOUT,
        MODE,
        Field("path", "string", "A file.", implied="default.bin", advanced=True),
        Field("key_env", "string", "A variable.", secret="env"),
        Field("secret", "string", "A key.", secret="literal"),
        Field("device", "string", "A device.", options="devices"),
    ]

    schema = fields_schema(fields)

    assert schema["required"] == ["voice"]
    properties = schema["properties"]
    assert list(properties) == [field.name for field in fields]
    assert properties["voice"] == {"type": "string", "description": "A voice."}
    assert properties["model"] == {
        "type": "string",
        "description": "A model.",
        "default": "small",
    }
    assert properties["speed"] == {
        "type": ["number", "null"],
        "description": "A rate.",
        "minimum": 0.5,
        "maximum": 2.0,
    }
    assert properties["timeout"]["exclusiveMinimum"] == 0
    assert properties["mode"]["enum"] == ["a", "b"]
    assert properties["path"] == {
        "type": ["string", "null"],
        "description": "A file.",
        "default": "default.bin",
        "x-advanced": True,
    }
    assert properties["key_env"]["x-secret-env"] is True
    assert properties["secret"]["x-secret"] is True
    assert properties["device"]["x-options"] == "devices"
