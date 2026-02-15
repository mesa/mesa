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
    handler.assert_not_called()  # we change it to the same value so no signal

    agent.some_attribute = 20
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


def test_ObservableList_negative_index_normalization():
    """Test that __setitem__ with negative index emits normalized positive index."""

    class MyAgent(Agent, HasObservables):
        my_list = ObservableList()

        def __init__(self, model):
            super().__init__(model)
            self.my_list = [1, 2, 3]

    # replaced
    model = Model(rng=42)
    agent = MyAgent(model)
    handler = Mock()
    agent.observe("my_list", ListSignals.REPLACED, handler)

    # Replace last element with -1
    agent.my_list[-1] = 99
    assert agent.my_list[2] == 99
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.REPLACED,
            owner=agent,
            additional_kwargs={"index": 2, "old": 3, "new": 99},
        )
    )

    # Replace first element with -len
    handler.reset_mock()
    agent.my_list[-3] = 50
    assert agent.my_list[0] == 50
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.REPLACED,
            owner=agent,
            additional_kwargs={"index": 0, "old": 1, "new": 50},
        )
    )

    # removed
    model = Model(rng=42)
    agent = MyAgent(model)
    handler = Mock()
    agent.observe("my_list", ListSignals.REMOVED, handler)

    # Delete last element with -1
    del agent.my_list[-1]
    assert list(agent.my_list) == [1, 2]
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.REMOVED,
            owner=agent,
            additional_kwargs={"index": 2, "old": 3},
        )
    )

    # Delete first element with -2 (on now 2-element list)
    handler.reset_mock()
    del agent.my_list[-2]
    assert list(agent.my_list) == [2]
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.REMOVED,
            owner=agent,
            additional_kwargs={"index": 0, "old": 1},
        )
    )

    # inserted
    model = Model(rng=42)
    agent = MyAgent(model)
    handler = Mock()
    agent.observe("my_list", ListSignals.INSERTED, handler)

    # Insert with -1 (before last element, normalized to index 2)
    agent.my_list.insert(-1, 99)
    assert list(agent.my_list) == [1, 2, 99, 3]
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.INSERTED,
            owner=agent,
            additional_kwargs={"index": 2, "new": 99},
        )
    )

    # Insert with large negative (clamped to 0)
    handler.reset_mock()
    agent.my_list.insert(-100, 50)
    assert agent.my_list[0] == 50
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.INSERTED,
            owner=agent,
            additional_kwargs={"index": 0, "new": 50},
        )
    )

    # Insert beyond length (clamped to len)
    handler.reset_mock()
    current_len = len(agent.my_list)
    agent.my_list.insert(100, 77)
    assert agent.my_list[-1] == 77
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.INSERTED,
            owner=agent,
            additional_kwargs={"index": current_len, "new": 77},
        )
    )


def test_ObservableList_slice_setitem():
    """Test that __setitem__ with a slice emits a REPLACED signal with normalized index."""

    class MyAgent(Agent, HasObservables):
        my_list = ObservableList()

        def __init__(self, model):
            super().__init__(model)
            self.my_list = [1, 2, 3, 4, 5]

    model = Model(rng=42)
    agent = MyAgent(model)
    handler = Mock()
    agent.observe("my_list", ListSignals.REPLACED, handler)

    # Replace a slice
    agent.my_list[1:3] = [20, 30]
    assert list(agent.my_list) == [1, 20, 30, 4, 5]
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.REPLACED,
            owner=agent,
            additional_kwargs={
                "index": slice(1, 3, 1),
                "old": [2, 3],
                "new": [20, 30],
            },
        )
    )

    # Replace with different length (shrink)
    handler.reset_mock()
    agent.my_list[1:4] = [99]
    assert list(agent.my_list) == [1, 99, 5]
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.REPLACED,
            owner=agent,
            additional_kwargs={
                "index": slice(1, 4, 1),
                "old": [20, 30, 4],
                "new": [99],
            },
        )
    )

    # Replace with a generator
    handler.reset_mock()
    agent.my_list[0:2] = (x * 10 for x in range(3))
    assert list(agent.my_list) == [0, 10, 20, 5]
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.REPLACED,
            owner=agent,
            additional_kwargs={
                "index": slice(0, 2, 1),
                "old": [1, 99],
                "new": [0, 10, 20],
            },
        )
    )

    # Replace with negative slice
    handler.reset_mock()
    agent.my_list[-2:] = [10, 20, 30]
    assert list(agent.my_list) == [0, 10, 10, 20, 30]
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.REPLACED,
            owner=agent,
            additional_kwargs={
                "index": slice(2, 4, 1),
                "old": [20, 5],
                "new": [10, 20, 30],
            },
        )
    )


def test_ObservableList_slice_delitem():
    """Test that __delitem__ with a slice emits a REMOVED signal with normalized index."""

    class MyAgent(Agent, HasObservables):
        my_list = ObservableList()

        def __init__(self, model):
            super().__init__(model)
            self.my_list = [1, 2, 3, 4, 5]

    model = Model(rng=42)
    agent = MyAgent(model)
    handler = Mock()
    agent.observe("my_list", ListSignals.REMOVED, handler)

    del agent.my_list[1:3]
    assert list(agent.my_list) == [1, 4, 5]
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.REMOVED,
            owner=agent,
            additional_kwargs={"index": slice(1, 3, 1), "old": [2, 3]},
        )
    )

    # Step slice
    handler.reset_mock()
    agent.my_list = [1, 2, 3, 4, 5]
    handler.reset_mock()
    del agent.my_list[::2]
    assert list(agent.my_list) == [2, 4]
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.REMOVED,
            owner=agent,
            additional_kwargs={"index": slice(0, 5, 2), "old": [1, 3, 5]},
        )
    )

    # Negative slice
    handler.reset_mock()
    agent.my_list = [1, 2, 3, 4, 5]
    handler.reset_mock()
    del agent.my_list[-3:-1]
    assert list(agent.my_list) == [1, 2, 5]
    handler.assert_called_once_with(
        Message(
            name="my_list",
            signal_type=ListSignals.REMOVED,
            owner=agent,
            additional_kwargs={"index": slice(2, 4, 1), "old": [3, 4]},
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
