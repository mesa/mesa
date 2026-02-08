"""Tests for mesa_signals."""

from unittest.mock import Mock, patch

import pytest

from mesa import Agent, Model
from mesa.experimental.mesa_signals import (
    ALL,
    HasObservables,
    ListSignals,
    Observable,
    ObservableList,
    ObservableSignals,
    SignalType,
    computed_property,
    emit,
)
from mesa.experimental.mesa_signals.signals_util import Message, _AllSentinel


def test_observables():
    """Test Observable."""

    class MyAgent(Agent, HasObservables):
        some_attribute = Observable()

        def __init__(self, model, value):
            super().__init__(model)
            self.some_attribute = value

    handler = Mock()

    model = Model(rng=42)
    agent = MyAgent(model, 10)
    agent.observe("some_attribute", ObservableSignals.CHANGED, handler)

    agent.some_attribute = 10
    handler.assert_called_once()


def test_HasObservables():
    """Test Observable."""

    class MyAgent(Agent, HasObservables):
        some_attribute = Observable()
        some_other_attribute = Observable()

        def __init__(self, model, value):
            super().__init__(model)
            self.some_attribute = value
            self.some_other_attribute = 5

    handler = Mock()

    model = Model(rng=42)
    agent = MyAgent(model, 10)
    agent.observe("some_attribute", ObservableSignals.CHANGED, handler)

    subscribers = {
        entry()
        for entry in agent.subscribers[("some_attribute", ObservableSignals.CHANGED)]
    }
    assert handler in subscribers

    agent.unobserve("some_attribute", ObservableSignals.CHANGED, handler)
    subscribers = {
        entry()
        for entry in agent.subscribers[("some_attribute", ObservableSignals.CHANGED)]
    }
    assert handler not in subscribers

    subscribers = {
        entry()
        for entry in agent.subscribers[
            ("some_other_attribute", ObservableSignals.CHANGED)
        ]
    }
    assert len(subscribers) == 0

    agent.observe(ALL, ObservableSignals.CHANGED, handler)

    for attr in ["some_attribute", "some_other_attribute"]:
        subscribers = {
            entry() for entry in agent.subscribers[(attr, ObservableSignals.CHANGED)]
        }
        assert handler in subscribers

    agent.unobserve(ALL, ObservableSignals.CHANGED, handler)
    for attr in ["some_attribute", "some_other_attribute"]:
        subscribers = {
            entry() for entry in agent.subscribers[(attr, ObservableSignals.CHANGED)]
        }
        assert handler not in subscribers
        assert len(subscribers) == 0

    # testing for clear_all_subscriptions
    ## test single string
    nr_observers = 3
    handlers = [Mock() for _ in range(nr_observers)]
    for handler in handlers:
        agent.observe("some_attribute", ObservableSignals.CHANGED, handler)
        agent.observe("some_other_attribute", ObservableSignals.CHANGED, handler)

    subscribers = {
        entry()
        for entry in agent.subscribers[("some_attribute", ObservableSignals.CHANGED)]
    }
    assert len(subscribers) == nr_observers

    agent.clear_all_subscriptions("some_attribute")
    subscribers = {
        entry()
        for entry in agent.subscribers[("some_attribute", ObservableSignals.CHANGED)]
    }
    assert len(subscribers) == 0

    ## test All
    subscribers = {
        entry()
        for entry in agent.subscribers[
            ("some_other_attribute", ObservableSignals.CHANGED)
        ]
    }
    assert len(subscribers) == 3

    agent.clear_all_subscriptions(ALL)
    subscribers = {
        entry()
        for entry in agent.subscribers[("some_attribute", ObservableSignals.CHANGED)]
    }
    assert len(subscribers) == 0

    subscribers = {
        entry()
        for entry in agent.subscribers[
            ("some_other_attribute", ObservableSignals.CHANGED)
        ]
    }
    assert len(subscribers) == 0

    ## test list of strings
    nr_observers = 3
    handlers = [Mock() for _ in range(nr_observers)]
    for handler in handlers:
        agent.observe("some_attribute", ObservableSignals.CHANGED, handler)
        agent.observe("some_other_attribute", ObservableSignals.CHANGED, handler)

    subscribers = {
        entry()
        for entry in agent.subscribers[
            ("some_other_attribute", ObservableSignals.CHANGED)
        ]
    }
    assert len(subscribers) == 3

    agent.clear_all_subscriptions(["some_attribute", "some_other_attribute"])
    assert len(agent.subscribers) == 0

    # test raises
    with pytest.raises(ValueError):
        agent.observe("some_attribute", "unknonw_signal", handler)

    with pytest.raises(ValueError):
        agent.observe("unknonw_attribute", ObservableSignals.CHANGED, handler)


def test_ObservableList():
    """Test ObservableList."""

    class MyAgent(Agent, HasObservables):
        my_list = ObservableList()

        def __init__(
            self,
            model,
        ):
            super().__init__(model)
            self.my_list = []

    model = Model(rng=42)
    agent = MyAgent(model)

    assert len(agent.my_list) == 0

    # add
    handler = Mock()
    agent.observe("my_list", ListSignals.APPENDED, handler)

    agent.my_list.append(1)
    assert len(agent.my_list) == 1
    handler.assert_called_once()
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.APPENDED,
            owner=agent,
            additional_kwargs={"index": 0, "new": 1},
        )
    )
    agent.unobserve("my_list", ListSignals.APPENDED, handler)

    # remove
    handler = Mock()
    agent.observe("my_list", ListSignals.REMOVED, handler)

    agent.my_list.remove(1)
    assert len(agent.my_list) == 0
    handler.assert_called_once()

    agent.unobserve("my_list", ListSignals.REMOVED, handler)

    # overwrite the existing list
    a_list = [1, 2, 3, 4, 5]
    handler = Mock()
    agent.observe("my_list", ListSignals.SET, handler)
    agent.my_list = a_list
    assert len(agent.my_list) == len(a_list)
    handler.assert_called_once()

    agent.my_list = a_list
    assert len(agent.my_list) == len(a_list)
    handler.assert_called()
    agent.unobserve("my_list", ListSignals.SET, handler)

    # pop
    handler = Mock()
    agent.observe("my_list", ListSignals.REMOVED, handler)

    index = 4
    entry = agent.my_list.pop(index)
    assert entry == a_list.pop(index)
    assert len(agent.my_list) == len(a_list)
    handler.assert_called_once()
    agent.unobserve("my_list", ListSignals.REMOVED, handler)

    # insert
    handler = Mock()
    agent.observe("my_list", ListSignals.INSERTED, handler)
    agent.my_list.insert(0, 5)
    handler.assert_called()
    agent.unobserve("my_list", ListSignals.INSERTED, handler)

    # overwrite
    handler = Mock()
    agent.observe("my_list", ListSignals.REPLACED, handler)
    agent.my_list[0] = 10
    assert agent.my_list[0] == 10
    handler.assert_called_once()
    agent.unobserve("my_list", ListSignals.REPLACED, handler)

    # combine two lists
    handler = Mock()
    agent.observe("my_list", ListSignals.APPENDED, handler)
    a_list = [1, 2, 3, 4, 5]
    agent.my_list = a_list
    assert len(agent.my_list) == len(a_list)
    agent.my_list += a_list
    assert len(agent.my_list) == 2 * len(a_list)
    handler.assert_called()

    # some more non signalling functionality tests
    assert 5 in agent.my_list
    assert agent.my_list.index(5) == 4


def test_Message():
    """Test Message."""

    class MyAgent(Agent, HasObservables):
        some_attribute = Observable()

        def __init__(self, model, value):
            super().__init__(model)
            self.some_attribute = value

    def on_change(signal: Message):
        assert signal.name == "some_attribute"
        assert signal.signal_type == ObservableSignals.CHANGED
        assert signal.additional_kwargs["old"] == 10
        assert signal.additional_kwargs["new"] == 5
        assert signal.owner == agent
        assert signal.additional_kwargs == {
            "old": 10,
            "new": 5,
        }

        items = dir(signal)
        for entry in ["name", "signal_type", "owner", "additional_kwargs"]:
            assert entry in items

    model = Model(rng=42)
    agent = MyAgent(model, 10)
    agent.observe("some_attribute", ObservableSignals.CHANGED, on_change)
    agent.some_attribute = 5


def test_computed_property():
    """Test @computed_property."""

    class MyAgent(Agent, HasObservables):
        some_other_attribute = Observable()

        def __init__(self, model, value):
            super().__init__(model)
            self.some_other_attribute = value

        @computed_property
        def some_attribute(self):
            return self.some_other_attribute * 2

    model = Model(rng=42)
    agent = MyAgent(model, 10)
    # Initial Access (Calculates 10 * 2)
    assert agent.some_attribute == 20

    # Dependency Tracking
    handler = Mock()
    agent.observe("some_attribute", ObservableSignals.CHANGED, handler)

    agent.some_other_attribute = 9  # Update Observable dependency

    # ComputedState._set_dirty triggers owner.notify immediately
    handler.assert_called_once()

    agent.unobserve("some_attribute", ObservableSignals.CHANGED, handler)

    # Value Update
    handler = Mock()
    agent.observe("some_attribute", ObservableSignals.CHANGED, handler)

    assert agent.some_attribute == 18  # Re-calculation happens here

    # Note: Accessing the value does NOT trigger 'change' again,
    # it was triggered when the dirty flag was set by the parent.
    handler.assert_not_called()

    agent.unobserve("some_attribute", ObservableSignals.CHANGED, handler)

    # Cyclical dependencies
    # Scenario: A computed property tries to modify an observable
    # that it also reads (or that is currently locked).
    class CyclicalAgent(Agent, HasObservables):
        o1 = Observable()

        def __init__(self, model, value):
            super().__init__(model)
            self.o1 = value

        @computed_property
        def c1(self):
            # c1 depends on o1 (read) but also tries to write to it.
            # Writing to o1 triggers notify -> sets c1 dirty -> checks cycles.
            # But here we are *inside* c1 evaluation.
            self.o1 = self.o1 - 1
            return self.o1

    agent = CyclicalAgent(model, 10)

    # Error should be raised when we try to evaluate the property
    with pytest.raises(ValueError, match="cyclical dependency"):
        _ = agent.c1


def test_computed_dynamic_dependencies():
    """Test that dependencies are correctly pruned (cleared) when code paths change.

    This ensures that if a computed property stops using a dependency (e.g. via an if/else),
    it stops listening to that dependency (Zombie Dependencies).
    """

    class DynamicAgent(Agent, HasObservables):
        use_a = Observable()
        val_a = Observable()
        val_b = Observable()

        def __init__(self, model):
            super().__init__(model)
            self.use_a = True
            self.val_a = 10
            self.val_b = 20

        @computed_property
        def result(self):
            if self.use_a:
                return self.val_a
            else:
                return self.val_b

    model = Model(rng=42)
    agent = DynamicAgent(model)

    # Use Path A (depends on val_a)
    assert agent.result == 10

    # Switch to Path B (should now depend ONLY on val_b)
    agent.use_a = False
    assert agent.result == 20

    # Modify 'val_a'
    # Since we are on Path B, changes to val_a should be ignored.
    handler = Mock()
    agent.observe("result", ObservableSignals.CHANGED, handler)

    agent.val_a = 999  # Should NOT trigger 'result' change
    handler.assert_not_called()

    # Modify 'val_b'
    # This SHOULD trigger a notification
    agent.val_b = 30
    handler.assert_called_once()
    assert agent.result == 30


def test_chained_computations():
    """Test that a computed property can depend on another computed property."""

    class ChainedAgent(Agent, HasObservables):
        base = Observable()

        def __init__(self, model, val):
            super().__init__(model)
            self.base = val

        @computed_property
        def intermediate(self):
            # When this runs, CURRENT_COMPUTED should be 'final'
            return self.base * 2

        @computed_property
        def final(self):
            # This sets CURRENT_COMPUTED = final_state
            # Then it accesses self.intermediate
            return self.intermediate + 1

    model = Model(rng=42)
    agent = ChainedAgent(model, 10)

    # Trigger the chain
    # Access final -> Sets CURRENT_COMPUTED = final -> Access intermediate
    # intermediate sees CURRENT_COMPUTED is final -> registers dependency
    assert agent.final == 21

    # Verify dependency flows through the chain
    # Changing 'base' should invalidate 'intermediate', which invalidates 'final'
    agent.base = 20
    assert agent.final == 41


def test_dead_parent_fallback():
    """Test defensive check for garbage collected parents."""

    class SimpleAgent(Agent, HasObservables):
        @computed_property
        def prop(self):
            return 1

    model = Model(rng=42)
    agent = SimpleAgent(model)

    _ = agent.prop

    # Get the internal state object (name is _computed_{func_name})
    state = agent._computed_prop

    # Mark it dirty so it enters the re-evaluation check loop
    state.is_dirty = True

    # Mock parents.items() to simulate a dead parent (None key).
    # WeakKeyDictionary usually prevents this, so we must mock it to hit the defensive line.
    with patch.object(state.parents, "items", return_value=[(None, {})]):
        # Accessing the property calls the wrapper.
        # It sees is_dirty=True -> iterates parents -> finds None -> sets changed=True
        val = agent.prop

    # parents disappearing
    # Ensure it re-calculated
    assert val == 1


def test_list_support():
    """Test using list of strings for name and signal_type in observe/unobserve."""

    class MyAgent(Agent, HasObservables):
        attr1 = Observable()
        attr2 = Observable()
        attr3 = Observable()

        def __init__(self, model):
            super().__init__(model)
            self.attr1 = 1
            self.attr2 = 2
            self.attr3 = 3

    model = Model(rng=42)
    agent = MyAgent(model)
    handler = Mock()

    # Test observe with list of names
    agent.observe(["attr1", "attr2"], ObservableSignals.CHANGED, handler)

    # Check subscriptions
    assert handler in [
        ref() for ref in agent.subscribers[("attr1", ObservableSignals.CHANGED)]
    ]
    assert handler in [
        ref() for ref in agent.subscribers[("attr2", ObservableSignals.CHANGED)]
    ]
    assert handler not in [
        ref() for ref in agent.subscribers[("attr3", ObservableSignals.CHANGED)]
    ]

    # Test unobserve with list of names
    agent.unobserve(["attr1", "attr2"], ObservableSignals.CHANGED, handler)
    assert handler not in [
        ref() for ref in agent.subscribers[("attr1", ObservableSignals.CHANGED)]
    ]
    assert handler not in [
        ref() for ref in agent.subscribers[("attr2", ObservableSignals.CHANGED)]
    ]


def test_emit():
    """Test emit decorator."""

    class TestSignals(SignalType):
        BEFORE = "before"
        AFTER = "after"

    class MyModel(Model):
        def __init__(self, rng=42):
            super().__init__(rng=rng)

        @emit("test", TestSignals.BEFORE, when="before")
        def test_before(self, value):
            pass

        @emit("test", TestSignals.AFTER, when="after")
        def test_after(self, some_value=None):
            pass

    model = MyModel()

    handler_before = Mock()
    model.observe("test", signal_type=TestSignals.BEFORE, handler=handler_before)

    handler_after = Mock()
    model.observe("test", signal_type=TestSignals.AFTER, handler=handler_after)

    model.test_before(10)
    handler_before.assert_called_once_with(
        Message(
            name="test",
            signal_type=TestSignals.BEFORE,
            owner=model,
            additional_kwargs={"args": (10,)},
        )
    )
    handler_after.assert_not_called()

    model.test_after(some_value=10)
    handler_after.assert_called_once_with(
        Message(
            name="test",
            signal_type=TestSignals.AFTER,
            owner=model,
            additional_kwargs={"args": (), "some_value": 10},
        )
    )


def test_all_sentinel():
    """Test the ALL sentinel."""
    import pickle  # noqa: PLC0415

    sentinel = _AllSentinel()

    assert sentinel == ALL
    assert sentinel is ALL
    assert str(sentinel) == str(ALL)
    assert repr(sentinel) == repr(ALL)
    assert hash(sentinel) == hash(ALL)

    a = pickle.loads(pickle.dumps(sentinel))  # noqa: S301
    assert a is ALL


def test_batch_signals_deduplicates_changed():
    """Test CHANGED signals are deduplicated during batch."""

    class MyAgent(Agent, HasObservables):
        value = Observable()

        def __init__(self, model):
            super().__init__(model)
            self.value = 0

    model = Model(rng=42)
    agent = MyAgent(model)
    handler = Mock()
    agent.observe("value", ObservableSignals.CHANGED, handler)

    with agent.batch_signals():
        agent.value = 1
        agent.value = 2
        agent.value = 3

    handler.assert_called_once()
    signal = handler.call_args.args[0]
    assert signal.additional_kwargs["new"] == 3


def test_batch_signals_keeps_non_changed_order():
    """Test non-CHANGED signals are preserved in order."""

    class MyAgent(Agent, HasObservables):
        values = ObservableList()

        def __init__(self, model):
            super().__init__(model)
            self.values = []

    model = Model(rng=42)
    agent = MyAgent(model)
    handler = Mock()
    agent.observe("values", ALL, handler)

    with agent.batch_signals():
        agent.values.append(1)
        agent.values.insert(0, 2)
        agent.values.remove(1)

    assert handler.call_count == 3
    assert [call.args[0].signal_type for call in handler.call_args_list] == [
        ListSignals.APPENDED,
        ListSignals.INSERTED,
        ListSignals.REMOVED,
    ]


def test_batch_signals_nested_context():
    """Test nested batch context only flushes on outer exit."""

    class MyAgent(Agent, HasObservables):
        value = Observable()

        def __init__(self, model):
            super().__init__(model)
            self.value = 0

    model = Model(rng=42)
    agent = MyAgent(model)
    handler = Mock()
    agent.observe("value", ObservableSignals.CHANGED, handler)

    with agent.batch_signals():
        agent.value = 1
        with agent.batch_signals():
            agent.value = 2
            handler.assert_not_called()
        handler.assert_not_called()
    handler.assert_called_once()


def test_suppress_signals_discards_all():
    """Test suppression discards observable, list, and @emit signals."""

    class TestSignals(SignalType):
        PING = "ping"

    class MyModel(Model):
        value = Observable()
        values = ObservableList()

        def __init__(self, rng=42):
            super().__init__(rng=rng)
            self.value = 0
            self.values = []

        @emit("test", TestSignals.PING, when="after")
        def ping(self):
            pass

    model = MyModel()
    value_handler = Mock()
    list_handler = Mock()
    emit_handler = Mock()
    model.observe("value", ObservableSignals.CHANGED, value_handler)
    model.observe("values", ALL, list_handler)
    model.observe("test", TestSignals.PING, emit_handler)

    with model.suppress_signals():
        model.value = 1
        model.values.append(1)
        model.ping()

    value_handler.assert_not_called()
    list_handler.assert_not_called()
    emit_handler.assert_not_called()


def test_suppress_signals_nested_context():
    """Test nested suppression contexts."""

    class MyAgent(Agent, HasObservables):
        value = Observable()

        def __init__(self, model):
            super().__init__(model)
            self.value = 0

    model = Model(rng=42)
    agent = MyAgent(model)
    handler = Mock()
    agent.observe("value", ObservableSignals.CHANGED, handler)

    with agent.suppress_signals():
        with agent.suppress_signals():
            agent.value = 1
        agent.value = 2
    handler.assert_not_called()


def test_batch_and_suppress_interaction():
    """Test suppress takes precedence over batch and only non-suppressed batched signals flush."""

    class MyAgent(Agent, HasObservables):
        value = Observable()

        def __init__(self, model):
            super().__init__(model)
            self.value = 0

    model = Model(rng=42)
    agent = MyAgent(model)
    handler = Mock()
    agent.observe("value", ObservableSignals.CHANGED, handler)

    with agent.batch_signals():
        agent.value = 1
        with agent.suppress_signals():
            agent.value = 2
            with agent.batch_signals():
                agent.value = 3
        agent.value = 4

    handler.assert_called_once()
    signal = handler.call_args.args[0]
    assert signal.additional_kwargs["new"] == 4


def test_batch_signal_buffer_cleared_after_flush():
    """Test batch buffer is cleared after flush."""

    class MyAgent(Agent, HasObservables):
        value = Observable()

        def __init__(self, model):
            super().__init__(model)
            self.value = 0

    model = Model(rng=42)
    agent = MyAgent(model)
    handler = Mock()
    agent.observe("value", ObservableSignals.CHANGED, handler)

    with agent.batch_signals():
        agent.value = 1

    agent.value = 2

    assert handler.call_count == 2
    assert handler.call_args_list[0].args[0].additional_kwargs["new"] == 1
    assert handler.call_args_list[1].args[0].additional_kwargs["new"] == 2


def test_custom_batch_aggregator_for_user_defined_signal():
    """Test user-defined signal aggregation policy can be plugged in."""

    class TestSignals(SignalType):
        PULSE = "pulse"

    class MyModel(Model):
        @emit("events", TestSignals.PULSE, when="after")
        def pulse(self, value):
            return value

    def keep_last_pulse(signals):
        last_index = None
        for i, signal in enumerate(signals):
            if signal.signal_type == TestSignals.PULSE:
                last_index = i

        if last_index is None:
            return signals

        return [
            signal
            for i, signal in enumerate(signals)
            if signal.signal_type != TestSignals.PULSE or i == last_index
        ]

    model = MyModel(rng=42)
    handler = Mock()
    model.observe("events", TestSignals.PULSE, handler)
    model.clear_batch_aggregators()
    model.add_batch_aggregator(keep_last_pulse)

    with model.batch_signals():
        model.pulse(1)
        model.pulse(2)
        model.pulse(3)

    handler.assert_called_once()
    assert handler.call_args.args[0].additional_kwargs["args"] == (3,)
