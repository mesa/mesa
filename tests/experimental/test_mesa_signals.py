"""Tests for mesa_signals."""

from unittest.mock import Mock

import pytest

from mesa import Agent, Model
from mesa.experimental.mesa_signals import (
    All,
    Computable,
    Computed,
    HasObservables,
    Observable,
    ObservableList,
)
from mesa.experimental.mesa_signals.signals_util import AttributeDict


def test_observables():
    """Test Observable."""

    class MyAgent(Agent, HasObservables):
        some_attribute = Observable()

        def __init__(self, model, value):
            super().__init__(model)
            some_attribute = value  # noqa: F841

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
            some_attribute = value  # noqa: F841
            some_other_attribute = 5  # noqa: F841

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


def test_Computable():
    """Test Computable and Computed."""

    class MyAgent(Agent, HasObservables):
        some_attribute = Computable()
        some_other_attribute = Observable()

        def __init__(self, model, value):
            super().__init__(model)
            self.some_other_attribute = value
            self.some_attribute = Computed(lambda x: x.some_other_attribute * 2, self)

    model = Model(seed=42)
    agent = MyAgent(model, 10)
    assert agent.some_attribute == 20

    handler = Mock()
    agent.observe("some_attribute", "change", handler)
    agent.some_other_attribute = 9  # we change the dependency of computed
    handler.assert_called_once()
    agent.unobserve("some_attribute", "change", handler)

    handler = Mock()
    agent.observe("some_attribute", "change", handler)
    assert (
        agent.some_attribute == 18
    )  # this forces a re-evaluation of the value of computed
    handler.assert_called_once()  # and so, our change handler should be called
    agent.unobserve("some_attribute", "change", handler)

    # cyclical dependencies
    def computed_func(agent):
        # this creates a cyclical dependency
        # our computed is dependent on o1, but also modifies o1
        agent.o1 = agent.o1 - 1

    class MyAgent(Agent, HasObservables):
        c1 = Computable()
        o1 = Observable()

        def __init__(self, model, value):
            super().__init__(model)
            self.o1 = value
            self.c1 = Computed(computed_func, self)

    model = Model(seed=42)
    with pytest.raises(ValueError):
        MyAgent(model, 10)

    # parents disappearing


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

    model = Model(seed=42)
    agent = MyAgent(model)
    handler = Mock()

    # Test observe with list of names
    agent.observe(["attr1", "attr2"], "change", handler)

    # Check subscriptions
    assert handler in [ref() for ref in agent.subscribers["attr1"]["change"]]
    assert handler in [ref() for ref in agent.subscribers["attr2"]["change"]]
    assert handler not in [ref() for ref in agent.subscribers["attr3"]["change"]]

    # Test unobserve with list of names
    agent.unobserve(["attr1", "attr2"], "change", handler)
    assert handler not in [ref() for ref in agent.subscribers["attr1"]["change"]]
    assert handler not in [ref() for ref in agent.subscribers["attr2"]["change"]]

    # Test observe with list of signal types (though Observable only has 'change' by default,
    # we can register checking error handling or adding custom signals if needed,
    # but for now Observable only emits 'change', so we can't easily test list of signal types
    # without a custom observable that emits multiple types. Let's create one.)

    class MultiSignalAgent(Agent, HasObservables):
        def __init__(self, model):
            super().__init__(model)
            # Register a custom observable that emits multiple signal types
            self._register_signal_emitter("custom_attr", {"type1", "type2", "type3"})

    agent2 = MultiSignalAgent(model)
    handler2 = Mock()

    # Test observe with list of signal types
    agent2.observe("custom_attr", ["type1", "type3"], handler2)

    assert handler2 in [ref() for ref in agent2.subscribers["custom_attr"]["type1"]]
    assert handler2 not in [ref() for ref in agent2.subscribers["custom_attr"]["type2"]]
    assert handler2 in [ref() for ref in agent2.subscribers["custom_attr"]["type3"]]

    # Test unobserve with list of signal types
    agent2.unobserve("custom_attr", ["type1", "type3"], handler2)
    assert handler2 not in [ref() for ref in agent2.subscribers["custom_attr"]["type1"]]
    assert handler2 not in [ref() for ref in agent2.subscribers["custom_attr"]["type3"]]

    # Test clear_all_subscriptions with list of names
    agent.observe(["attr1", "attr2", "attr3"], "change", handler)
    agent.clear_all_subscriptions(["attr1", "attr3"])

    # helper to check emptiness
    def is_empty(attr):
        return len([ref() for ref in agent.subscribers[attr]["change"] if ref()]) == 0

    assert is_empty("attr1")
    assert not is_empty("attr2")
    assert is_empty("attr3")
