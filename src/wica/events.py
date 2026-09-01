"""A minimal, generic publish/subscribe primitive: ``Event[T]``.

A project-agnostic building block — it depends on nothing but the standard
library, so it can be lifted into any project unchanged. An ``Event[T]`` holds
a list of handlers; ``subscribe``/``unsubscribe`` manage them and ``emit`` calls
each one **inline, in subscription order**, with the published value. It is
synchronous and holds no state beyond its handler list — it *carries* a value to
subscribers and stores nothing.

Design in specs/events.md.
"""

import logging
from collections.abc import Callable

_logger = logging.getLogger(__name__)


class Event[T]:
    """A synchronous, generic pub/sub signal carrying a single value of type ``T``.

    Handlers subscribe with :meth:`subscribe` and are invoked, in subscription
    order, each time :meth:`emit` is called. ``emit`` iterates a *snapshot* of the
    handler list, so a handler may safely ``subscribe``/``unsubscribe`` (itself or
    another) during dispatch without perturbing the in-progress round.
    """

    def __init__(self) -> None:
        self._handlers: list[Callable[[T], None]] = []

    def subscribe(self, handler: Callable[[T], None]) -> None:
        """Register ``handler`` to be called on every future :meth:`emit`."""
        self._handlers.append(handler)

    def unsubscribe(self, handler: Callable[[T], None]) -> None:
        """Remove a previously subscribed ``handler``.

        Raises ``ValueError`` if it was never subscribed.
        """
        self._handlers.remove(handler)

    def emit(self, value: T) -> None:
        """Call every subscribed handler with ``value``, in subscription order.

        Each handler call is isolated: an exception it raises is caught and logged,
        and dispatch continues to the remaining handlers, so one bad subscriber can
        neither abort the ``emit`` nor starve its siblings. ``emit`` itself never
        propagates a handler's exception.
        """
        for handler in list(self._handlers):
            try:
                handler(value)
            except Exception:
                _logger.exception(
                    "Event subscriber raised; continuing to next subscriber"
                )
