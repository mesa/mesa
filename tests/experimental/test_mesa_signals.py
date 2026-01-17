"""Tests for mesa_signals."""

from unittest.mock import Mock, patch

import pytest

from mesa import Agent, Model
from mesa.experimental.mesa_signals import (
    All,
    HasObservables,
    Observable,
    ObservableList,
    computed,
)
from mesa.experimental.mesa_signals.signals_util import AttributeDict


def test_observables():
    """Test Observable."""

    class MyAgent(Agent, HasObservables):
        some_attribute = Observable()

        def __init__(self, model, value):
            super().__init__(model)
            self.some_attribute = value

    handler = Mock()

    model = Model(seed=42)
    agent = MyAgent(model, 10)
    agent.observe("some_attribute", "change", handler)

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

    model = Model(seed=42)
    agent = MyAgent(model, 10)
    agent.observe("some_attribute", "change", handler)

    subscribers = {entry() for entry in agent.subscribers["some_attribute"]["change"]}
    assert handler in subscribers

    agent.unobserve("some_attribute", "change", handler)
    subscribers = {entry() for entry in agent.subscribers["some_attribute"]["change"]}
    assert handler not in subscribers

    subscribers = {
        entry() for entry in agent.subscribers["some_other_attribute"]["change"]
    }
    assert len(subscribers) == 0

    # testing All()
    agent.observe(All(), "change", handler)

    for attr in ["some_attribute", "some_other_attribute"]:
        subscribers = {entry() for entry in agent.subscribers[attr]["change"]}
        assert handler in subscribers

    agent.unobserve(All(), "change", handler)
    for attr in ["some_attribute", "some_other_attribute"]:
        subscribers = {entry() for entry in agent.subscribers[attr]["change"]}
        assert handler not in subscribers
        assert len(subscribers) == 0

    # testing for clear_all_subscriptions
    nr_observers = 3
    handlers = [Mock() for _ in range(nr_observers)]
    for handler in handlers:
        agent.observe("some_attribute", "change", handler)
        agent.observe("some_other_attribute", "change", handler)

    subscribers = {entry() for entry in agent.subscribers["some_attribute"]["change"]}
    assert len(subscribers) == nr_observers

    agent.clear_all_subscriptions("some_attribute")
    subscribers = {entry() for entry in agent.subscribers["some_attribute"]["change"]}
    assert len(subscribers) == 0

    subscribers = {
        entry() for entry in agent.subscribers["some_other_attribute"]["change"]
    }
    assert len(subscribers) == 3

    agent.clear_all_subscriptions(All())
    subscribers = {entry() for entry in agent.subscribers["some_attribute"]["change"]}
    assert len(subscribers) == 0

    subscribers = {
        entry() for entry in agent.subscribers["some_other_attribute"]["change"]
    }
    assert len(subscribers) == 0

    # test raises
    with pytest.raises(ValueError):
        agent.observe("some_attribute", "unknonw_signal", handler)

    with pytest.raises(ValueError):
        agent.observe("unknonw_attribute", "change", handler)


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

    model = Model(seed=42)
    agent = MyAgent(model)

    assert len(agent.my_list) == 0

    # add
    handler = Mock()
    agent.observe("my_list", "append", handler)

    agent.my_list.append(1)
    assert len(agent.my_list) == 1
    handler.assert_called_once()
    handler.assert_called_once_with(
        AttributeDict(
            name="my_list", new=1, old=None, type="append", index=0, owner=agent
        )
    )
    agent.unobserve("my_list", "append", handler)

    # remove
    handler = Mock()
    agent.observe("my_list", "remove", handler)

    agent.my_list.remove(1)
    assert len(agent.my_list) == 0
    handler.assert_called_once()

    agent.unobserve("my_list", "remove", handler)

    # overwrite the existing list
    a_list = [1, 2, 3, 4, 5]
    handler = Mock()
    agent.observe("my_list", "change", handler)
    agent.my_list = a_list
    assert len(agent.my_list) == len(a_list)
    handler.assert_called_once()

    agent.my_list = a_list
    assert len(agent.my_list) == len(a_list)
    handler.assert_called()
    agent.unobserve("my_list", "change", handler)

    # pop
    handler = Mock()
    agent.observe("my_list", "remove", handler)

    index = 4
    entry = agent.my_list.pop(index)
    assert entry == a_list.pop(index)
    assert len(agent.my_list) == len(a_list)
    handler.assert_called_once()
    agent.unobserve("my_list", "remove", handler)

    # insert
    handler = Mock()
    agent.observe("my_list", "insert", handler)
    agent.my_list.insert(0, 5)
    handler.assert_called()
    agent.unobserve("my_list", "insert", handler)

    # overwrite
    handler = Mock()
    agent.observe("my_list", "replace", handler)
    agent.my_list[0] = 10
    assert agent.my_list[0] == 10
    handler.assert_called_once()
    agent.unobserve("my_list", "replace", handler)

    # combine two lists
    handler = Mock()
    agent.observe("my_list", "append", handler)
    a_list = [1, 2, 3, 4, 5]
    agent.my_list = a_list
    assert len(agent.my_list) == len(a_list)
    agent.my_list += a_list
    assert len(agent.my_list) == 2 * len(a_list)
    handler.assert_called()

    # some more non signalling functionality tests
    assert 5 in agent.my_list
    assert agent.my_list.index(5) == 4


def test_AttributeDict():
    """Test AttributeDict."""

    class MyAgent(Agent, HasObservables):
        some_attribute = Observable()

        def __init__(self, model, value):
            super().__init__(model)
            self.some_attribute = value

    def on_change(signal):
        assert signal.name == "some_attribute"
        assert signal.type == "change"
        assert signal.old == 10
        assert signal.new == 5
        assert signal.owner == agent

        items = dir(signal)
        for entry in ["name", "type", "old", "new", "owner"]:
            assert entry in items

    model = Model(seed=42)
    agent = MyAgent(model, 10)
    agent.observe("some_attribute", "change", on_change)
    agent.some_attribute = 5


def test_computed():
    """Test @computed."""

    class MyAgent(Agent, HasObservables):
        some_other_attribute = Observable()

        def __init__(self, model, value):
            super().__init__(model)
            self.some_other_attribute = value

        @computed
        def some_attribute(self):
            return self.some_other_attribute * 2

    model = Model(seed=42)
    agent = MyAgent(model, 10)
    # Initial Access (Calculates 10 * 2)
    assert agent.some_attribute == 20

    # Dependency Tracking
    handler = Mock()
    agent.observe("some_attribute", "change", handler)

    agent.some_other_attribute = 9  # Update Observable dependency

    # ComputedState._set_dirty triggers owner.notify immediately
    handler.assert_called_once()

    agent.unobserve("some_attribute", "change", handler)

    # Value Update
    handler = Mock()
    agent.observe("some_attribute", "change", handler)

    assert agent.some_attribute == 18  # Re-calculation happens here

    # Note: Accessing the value does NOT trigger 'change' again,
    # it was triggered when the dirty flag was set by the parent.
    handler.assert_not_called()

    agent.unobserve("some_attribute", "change", handler)

    # Cyclical dependencies
    # Scenario: A computed property tries to modify an observable
    # that it also reads (or that is currently locked).
    class CyclicalAgent(Agent, HasObservables):
        o1 = Observable()

        def __init__(self, model, value):
            super().__init__(model)
            self.o1 = value

        @computed
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

        @computed
        def result(self):
            if self.use_a:
                return self.val_a
            else:
                return self.val_b

    model = Model(seed=42)
    agent = DynamicAgent(model)

    # Use Path A (depends on val_a)
    assert agent.result == 10

    # Switch to Path B (should now depend ONLY on val_b)
    agent.use_a = False
    assert agent.result == 20

    # Modify 'val_a'
    # Since we are on Path B, changes to val_a should be ignored.
    handler = Mock()
    agent.observe("result", "change", handler)

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

        @computed
        def intermediate(self):
            # When this runs, CURRENT_COMPUTED should be 'final'
            return self.base * 2

        @computed
        def final(self):
            # This sets CURRENT_COMPUTED = final_state
            # Then it accesses self.intermediate
            return self.intermediate + 1

    model = Model(seed=42)
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
        @computed
        def prop(self):
            return 1

    model = Model(seed=42)
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

        # Ensure it re-calculated
        assert val == 1
