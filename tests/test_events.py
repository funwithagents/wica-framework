"""Functional tests for the generic pub/sub primitive (specs/events.md)."""

from wica.events import Event


def test_subscribed_handler_receives_each_emitted_value():
    received: list[int] = []
    event: Event[int] = Event()
    event.subscribe(received.append)

    event.emit(7)
    event.emit(42)

    assert received == [7, 42]


def test_handlers_fire_in_subscription_order():
    order: list[str] = []
    event: Event[str] = Event()
    event.subscribe(lambda v: order.append(f"a:{v}"))
    event.subscribe(lambda v: order.append(f"b:{v}"))

    event.emit("x")

    assert order == ["a:x", "b:x"]


def test_unsubscribe_stops_delivery():
    received: list[int] = []
    event: Event[int] = Event()

    def handler(value: int) -> None:
        received.append(value)

    event.subscribe(handler)
    event.emit(1)
    event.unsubscribe(handler)
    event.emit(2)

    assert received == [1]


def test_emit_with_no_subscribers_is_a_noop():
    event: Event[int] = Event()
    event.emit(1)  # must not raise


def test_raising_handler_is_isolated_and_siblings_still_fire():
    # A subscriber that raises is caught-and-logged; the next subscriber still runs,
    # and emit does not propagate the exception.
    event: Event[int] = Event()
    seen: list[str] = []

    def boom(value: int) -> None:
        seen.append(f"boom:{value}")
        raise RuntimeError("subscriber failure")

    event.subscribe(boom)
    event.subscribe(lambda value: seen.append(f"other:{value}"))

    event.emit(1)  # must not raise

    assert seen == ["boom:1", "other:1"]


def test_handler_may_unsubscribe_itself_during_emit():
    # emit iterates a snapshot, so a handler that removes itself mid-dispatch does
    # not perturb the in-progress round: both handlers still fire this round, and
    # only the survivor fires next round.
    event: Event[int] = Event()
    seen: list[str] = []

    def once(value: int) -> None:
        seen.append(f"once:{value}")
        event.unsubscribe(once)

    event.subscribe(once)
    event.subscribe(lambda value: seen.append(f"other:{value}"))

    event.emit(1)
    event.emit(2)

    assert seen == ["once:1", "other:1", "other:2"]
