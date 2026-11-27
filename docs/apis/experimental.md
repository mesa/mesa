# Experimental
This namespace contains experimental features. These are under development, and their API is not necessarily stable.

## Continuous Space

```{eval-rst}
.. automodule:: experimental.continuous_space.continuous_space
   :members:
```

```{eval-rst}
.. automodule:: experimental.continuous_space.continuous_space_agents
   :members:
```

## Locations

```{eval-rst}
.. automodule:: experimental.locations
   :members:
```

## Scenarios

```{eval-rst}
.. automodule:: experimental.scenarios.scenario
   :members:
```

```{eval-rst}
.. automodule:: experimental.scenarios.runner
   :members:
```

```{eval-rst}
.. automodule:: experimental.scenarios.disk_store
   :members:
```

## States

```{eval-rst}
.. automodule:: experimental.states.state
   :members:
```

## Signals

Signals let objects react to changes in other objects. Declare an `Observable` as a class attribute of a `HasEmitters` subclass; assigning a new value to it emits a signal to every subscribed handler. `ObservableList` does the same for a list and also signals changes made in place, and `computed_property` derives a cached value from observables.

```{eval-rst}
.. autoclass:: experimental.mesa_signals.Observable

.. autoclass:: experimental.mesa_signals.HasEmitters
   :members: observe, unobserve, observe_class, unobserve_class, clear_all_subscriptions, clear_all_class_subscriptions, notify, batch, suppress, peek

.. autofunction:: experimental.mesa_signals.computed_property

.. autoclass:: experimental.mesa_signals.ObservableList

.. autofunction:: experimental.mesa_signals.emit

.. autoclass:: experimental.mesa_signals.Message

.. autoclass:: experimental.mesa_signals.ObservableSignals

.. autoclass:: experimental.mesa_signals.ListSignals

.. autoclass:: experimental.mesa_signals.ModelSignals
   :members:
   :undoc-members:

.. autoclass:: experimental.mesa_signals.SignalType

.. autofunction:: experimental.mesa_signals.aggregate
```
