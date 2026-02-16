"""Tests for batch and suppress context managers and singledispatch aggregation."""

from unittest.mock import Mock

from mesa.experimental.mesa_signals import (
    HasObservables,
    ListSignals,
    Observable,
    ObservableList,
    ObservableSignals,
    SignalType,
    aggregate,
    computed_property,
)
from mesa.experimental.mesa_signals.signals_util import Message


class MyObj(HasObservables):  # noqa: D101
    value = Observable()

    def __init__(self, val=0):  # noqa: D107
        super().__init__()
        self.value = val


class MultiObj(HasObservables):  # noqa: D101
    x = Observable()
    y = Observable()

    def __init__(self, x=0, y=0):  # noqa: D107
        super().__init__()
        self.x = x
        self.y = y


class ListObj(HasObservables):  # noqa: D101
    items = ObservableList()

    def __init__(self):  # noqa: D107
        super().__init__()
        self.items = []


class ComputedObj(HasObservables):  # noqa: D101
    base = Observable()

    def __init__(self, val=0):  # noqa: D107
        super().__init__()
        self.base = val

    @computed_property
    def doubled(self):
        """Test."""
        return self.base * 2


def test_batch_observable_basic():
    """Batch two CHANGED signals on one Observable, verify single aggregated signal."""
    obj = MyObj(10)
    handler = Mock()
    obj.observe("value", ObservableSignals.CHANGED, handler)

    with obj.batch():
        obj.value = 20
        obj.value = 30

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == 10
    assert signal.additional_kwargs["new"] == 30
    assert signal.signal_type == ObservableSignals.CHANGED


def test_batch_observable_no_net_change():
    """Change value and change back, verify no signal emitted."""
    obj = MyObj(10)
    handler = Mock()
    obj.observe("value", ObservableSignals.CHANGED, handler)

    with obj.batch():
        obj.value = 20
        obj.value = 10  # back to original

    handler.assert_not_called()


def test_batch_suppress_basic():
    """Suppress signals, verify handler never called."""
    obj = MyObj(10)
    handler = Mock()
    obj.observe("value", ObservableSignals.CHANGED, handler)

    with obj.suppress():
        obj.value = 20
        obj.value = 30

    handler.assert_not_called()
    assert obj.value == 30  # value still changed


def test_batch_list_aggregation():
    """Append/remove/insert during batch, verify single SET signal."""
    obj = ListObj()
    obj.items = [1, 2, 3]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items.append(4)
        obj.items.append(5)
        obj.items.remove(2)

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.signal_type == ListSignals.SET
    assert signal.additional_kwargs["old"] == [1, 2, 3]
    assert signal.additional_kwargs["new"] == [1, 3, 4, 5]


def test_batch_nesting():
    """Nested batch merges into outer, only outermost dispatches."""
    obj = MyObj(10)
    handler = Mock()
    obj.observe("value", ObservableSignals.CHANGED, handler)

    with obj.batch():
        obj.value = 20
        handler.assert_not_called()

        with obj.batch():
            obj.value = 30
            handler.assert_not_called()

        # Inner batch exited, but outer still active
        handler.assert_not_called()

    # Outer batch exited, now signal should fire
    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == 10
    assert signal.additional_kwargs["new"] == 30

    # test if only signal in inner loop is properly
    # transferred to outer loop
    handler.reset_mock()
    with obj.batch():
        with obj.batch():
            # only change inside inner batch
            obj.value = 10
            handler.assert_not_called()

        # Inner batch exited, but outer still active
        handler.assert_not_called()

    # Outer batch exited, now signal should fire
    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == 30
    assert signal.additional_kwargs["new"] == 10


def test_suppress_nesting():
    """Nested suppress, outer exit restores."""
    obj = MyObj(10)
    handler = Mock()
    obj.observe("value", ObservableSignals.CHANGED, handler)

    with obj.suppress():
        obj.value = 20
        with obj.suppress():
            obj.value = 30
        # Inner suppress exited, but outer still active
        obj.value = 40
        handler.assert_not_called()
    handler.assert_not_called()

    # Outer suppress exited, signals should work again
    obj.value = 50
    handler.assert_called_once()


def test_batch_error_discards():
    """Exception inside batch, verify no signals dispatched."""
    obj = MyObj(10)
    handler = Mock()
    obj.observe("value", ObservableSignals.CHANGED, handler)

    try:
        with obj.batch():
            obj.value = 20
            raise ValueError("test error")
    except ValueError:
        pass

    handler.assert_not_called()
    assert obj.value == 20  # value was still set


def test_suppress_inside_batch():
    """Suppress inside batch, signals during suppress are dropped."""
    obj = MyObj(10)
    handler = Mock()
    obj.observe("value", ObservableSignals.CHANGED, handler)

    with obj.batch():
        obj.value = 20
        with obj.suppress():
            obj.value = 30  # dropped by suppress
        obj.value = 40

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == 10
    assert signal.additional_kwargs["new"] == 40


def test_batch_inside_suppress():
    """Batch inside suppress, all signals dropped."""
    obj = MyObj(10)
    handler = Mock()
    obj.observe("value", ObservableSignals.CHANGED, handler)

    with obj.suppress(), obj.batch():
        obj.value = 20
        obj.value = 30

    handler.assert_not_called()


def test_batch_computed_staleness():
    """Computed returns stale value during batch, correct value after flush."""
    obj = ComputedObj(10)

    # Access to initialize the computed
    assert obj.doubled == 20

    handler = Mock()
    obj.observe("doubled", ObservableSignals.CHANGED, handler)

    with obj.batch():
        obj.base = 20
        # During batch, computed may be stale
        # (dirty notification hasn't fired yet)

    # After batch, computed should be updated
    handler.assert_called_once()
    assert obj.doubled == 40


def test_suppress_computed_staleness():
    """Computed returns stale value during and after suppress."""
    obj = ComputedObj(10)

    # Access to initialize the computed
    assert obj.doubled == 20

    handler = Mock()
    obj.observe("doubled", ObservableSignals.CHANGED, handler)

    with obj.suppress():
        obj.base = 20
        # During batch, computed may be stale
        # (dirty notification hasn't fired yet)

    # After suppress, computed should still be stale
    handler.assert_not_called()
    assert obj.doubled == 20


def test_aggregate_register_custom():
    """User registers custom aggregator via @aggregate.register, verify it's used."""

    class CustomSignals(SignalType):
        TICK = "tick"

    class CustomObj(HasObservables):
        def __init__(self):
            super().__init__()
            self.observables["counter"] = frozenset([CustomSignals.TICK])

    @aggregate.register(CustomSignals)
    def _aggregate_custom(signal_type, signals, value=None):
        """Aggregate custom signals: count them."""
        if not signals:
            return []
        return [
            Message(
                name=signals[0].name,
                owner=signals[0].owner,
                signal_type=CustomSignals.TICK,
                additional_kwargs={"count": len(signals)},
            )
        ]

    obj = CustomObj()
    handler = Mock()
    obj.observe("counter", CustomSignals.TICK, handler)

    with obj.batch():
        obj.notify("counter", CustomSignals.TICK)
        obj.notify("counter", CustomSignals.TICK)
        obj.notify("counter", CustomSignals.TICK)

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["count"] == 3


def test_aggregate_default_keep_all():
    """Signals with unregistered SignalType pass through unchanged."""

    class UnknownSignals(SignalType):
        FOO = "foo"

    signals = [
        Message(
            name="test",
            owner=None,
            signal_type=UnknownSignals.FOO,
            additional_kwargs={"a": 1},
        ),
        Message(
            name="test",
            owner=None,
            signal_type=UnknownSignals.FOO,
            additional_kwargs={"a": 2},
        ),
    ]

    result = aggregate(UnknownSignals.FOO, signals)
    assert result == signals
    assert len(result) == 2


def test_batch_multiple_observables():
    """Batch with multiple observables on same instance, each aggregated independently."""
    obj = MultiObj(x=1, y=10)
    handler_x = Mock()
    handler_y = Mock()
    obj.observe("x", ObservableSignals.CHANGED, handler_x)
    obj.observe("y", ObservableSignals.CHANGED, handler_y)

    with obj.batch():
        obj.x = 2
        obj.y = 20
        obj.x = 3
        obj.y = 30

    handler_x.assert_called_once()
    handler_y.assert_called_once()

    signal_x = handler_x.call_args[0][0]
    assert signal_x.additional_kwargs["old"] == 1
    assert signal_x.additional_kwargs["new"] == 3

    signal_y = handler_y.call_args[0][0]
    assert signal_y.additional_kwargs["old"] == 10
    assert signal_y.additional_kwargs["new"] == 30


def test_aggregate_original_list_reconstruction():
    """Verify pre-batch list state is correctly reconstructed."""
    obj = ListObj()
    obj.items = [1, 2, 3]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items.append(4)

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1, 2, 3]
    assert signal.additional_kwargs["new"] == [1, 2, 3, 4]

    obj = ListObj()
    obj.items = [1, 2, 3]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items.insert(1, 99)

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1, 2, 3]
    assert signal.additional_kwargs["new"] == [1, 99, 2, 3]

    obj = ListObj()
    obj.items = [1, 2, 3]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items.remove(2)

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1, 2, 3]
    assert signal.additional_kwargs["new"] == [1, 3]

    obj = ListObj()
    obj.items = [1, 2, 3]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items[1] = 99

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1, 2, 3]
    assert signal.additional_kwargs["new"] == [1, 99, 3]

    obj = ListObj()
    obj.items = [1, 2, 3]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items = [4, 5, 6]

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1, 2, 3]
    assert signal.additional_kwargs["new"] == [4, 5, 6]

    obj = ListObj()
    obj.items = [1, 2, 3]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items.append(4)
        obj.items.remove(4)

    handler.assert_not_called()

    obj = ListObj()
    obj.items = [1, 2, 3]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items[1::] = [1, 1]

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1, 2, 3]
    assert signal.additional_kwargs["new"] == [1, 1, 1]

    obj = ListObj()
    obj.items = [1, 2, 3]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items[:] = obj.items[::-1]

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1, 2, 3]
    assert signal.additional_kwargs["new"] == [3, 2, 1]


def test_batch_list_negative_indices():
    """Batch with negative index operations produces correct aggregated SET signal."""
    obj = ListObj()
    obj.items = [1, 2, 3, 4, 5]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items[-1] = 99  # replace last: [1, 2, 3, 4, 99]
        del obj.items[-2]  # delete index 3: [1, 2, 3, 99]
        obj.items.insert(-1, 77)  # insert before last: [1, 2, 3, 77, 99]

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.signal_type == ListSignals.SET
    assert signal.additional_kwargs["old"] == [1, 2, 3, 4, 5]
    assert signal.additional_kwargs["new"] == [1, 2, 3, 77, 99]


def test_aggregate_custom_uses_value():
    """Custom aggregator receives captured value and uses it for aggregation."""

    class AccumSignals(SignalType):
        ADDED = "added"

    class AccumObj(HasObservables):
        def __init__(self):
            super().__init__()
            self._total = 0
            self.observables["total"] = frozenset([AccumSignals.ADDED])

    @aggregate.register(AccumSignals)
    def _aggregate_accum(signal_type, signals, value=None):
        """Aggregate AccumSignals: sum all deltas, use first signal's old as baseline."""
        if not signals:
            return []
        total_delta = sum(s.additional_kwargs.get("delta", 0) for s in signals)
        if total_delta == 0:
            return []
        old = signals[0].additional_kwargs.get("old", value)
        return [
            Message(
                name=signals[0].name,
                owner=signals[0].owner,
                signal_type=AccumSignals.ADDED,
                additional_kwargs={
                    "old": old,
                    "delta": total_delta,
                },
            )
        ]

    obj = AccumObj()
    handler = Mock()
    obj.observe("total", AccumSignals.ADDED, handler)

    with obj.batch():
        obj.notify("total", AccumSignals.ADDED, old=0, delta=5)
        obj.notify("total", AccumSignals.ADDED, old=5, delta=3)
        obj.notify("total", AccumSignals.ADDED, old=8, delta=2)

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == 0
    assert signal.additional_kwargs["delta"] == 10


def test_batch_list_slice_replacement():
    """Batch with length-changing slice replacement reconstructs old correctly."""
    # Shrink: replace 3 elements with 1
    obj = ListObj()
    obj.items = [1, 2, 3, 4, 5]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items[1:4] = [99]

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1, 2, 3, 4, 5]
    assert signal.additional_kwargs["new"] == [1, 99, 5]

    # Grow: replace 1 element with 3
    obj = ListObj()
    obj.items = [1, 2, 3]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items[1:2] = [10, 20, 30]

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1, 2, 3]
    assert signal.additional_kwargs["new"] == [1, 10, 20, 30, 3]

    # Insert via empty slice
    obj = ListObj()
    obj.items = [1, 2, 3]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items[1:1] = [10, 20]

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1, 2, 3]
    assert signal.additional_kwargs["new"] == [1, 10, 20, 2, 3]


def test_batch_list_inherited_methods():
    """MutableSequence inherited methods (extend, pop, clear, +=) emit correct signals in batch."""
    # extend
    obj = ListObj()
    obj.items = [1, 2]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items.extend([3, 4, 5])

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1, 2]
    assert signal.additional_kwargs["new"] == [1, 2, 3, 4, 5]

    # pop
    obj = ListObj()
    obj.items = [1, 2, 3]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        val = obj.items.pop()

    assert val == 3
    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1, 2, 3]
    assert signal.additional_kwargs["new"] == [1, 2]

    # clear
    obj = ListObj()
    obj.items = [1, 2, 3]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items.clear()

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1, 2, 3]
    assert signal.additional_kwargs["new"] == []

    # +=
    obj = ListObj()
    obj.items = [1]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        obj.items += [2, 3]

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1]
    assert signal.additional_kwargs["new"] == [1, 2, 3]


def test_suppress_inside_nested_batch_snapshot():
    """Suppressed mutation snapshots into outer batch; nested inner batch must not overwrite it."""
    obj = ListObj()
    obj.items = [1, 2, 3]

    handler = Mock()
    obj.observe("items", ListSignals.SET, handler)

    with obj.batch():
        with obj.suppress():
            obj.items.append(4)  # suppressed, but snapshots [1,2,3] into outer
        with obj.batch():
            obj.items.append(
                5
            )  # inner snapshot [1,2,3,4] must NOT overwrite outer's [1,2,3]

    handler.assert_called_once()
    signal = handler.call_args[0][0]
    assert signal.additional_kwargs["old"] == [1, 2, 3]
    assert signal.additional_kwargs["new"] == [1, 2, 3, 4, 5]
