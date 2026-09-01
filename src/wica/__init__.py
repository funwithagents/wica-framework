import logging

from wica.agent import Agent, CommandIssued
from wica.config import (
    AgentConfig,
    ConfigError,
    MissingEnvError,
    WicaConfig,
)
from wica.content import Content, ContentPart, ImagePart, TextPart
from wica.events import Event
from wica.wica import Wica
from wica.world import World, WorldEntry, WorldEntryConfig, WorldEntryVersion

# WICA is a library: it only emits records under the `wica.*` loggers and never configures
# handlers or levels — that policy belongs to the embedding application. The one thing a library
# should do is attach a NullHandler to its top-level logger, so records don't hit the stdlib
# last-resort handler when the application hasn't configured logging.
logging.getLogger("wica").addHandler(logging.NullHandler())

__all__ = [
    "Agent",
    "AgentConfig",
    "CommandIssued",
    "ConfigError",
    "Content",
    "ContentPart",
    "Event",
    "ImagePart",
    "MissingEnvError",
    "TextPart",
    "Wica",
    "WicaConfig",
    "World",
    "WorldEntry",
    "WorldEntryConfig",
    "WorldEntryVersion",
]
