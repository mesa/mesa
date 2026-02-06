"""Helper classes for collecting statistics."""

from __future__ import annotations

import abc
import operator
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from mesa.agent import AbstractAgentSet, Agent

if TYPE_CHECKING:
    from mesa.model import Model

__all__ = [
    "AgentDataSet",
    "DataRegistry",
    "DataSet",
    "ModelDataSet",
    "NumpyAgentDataSet",
    "TableDataSet",
]


@runtime_checkable
class DataSet(Protocol):
    """Protocol for all data collection classes."""

    name: str

    @property
    def data(self) -> Any:
        """Return collected data."""
        ...

    def close(self):
        """Close the dataset."""
        ...


class BaseDataSet(abc.ABC):
    """Abstract base class for data sets.

    Args:
       name: the name of the data set
       args: the fields to collect

    """

    def __init__(self, name, *args):
        """Init."""
        self.name = name
        self._attributes = args
        self._collector = operator.attrgetter(*self._attributes)
        self._closed = False

    @property
    @abc.abstractmethod
    def data(self):
        """Return the data of the table."""
        ...

    def _check_closed(self):
        """Check to see if the data set has been closed."""
        if self._closed:
            raise RuntimeError(f"DataSet '{self.name}' has been closed")

    def close(self):
        """Cleanup and close the data set."""
        if self._closed:
            return
        self._collector = None
        self._closed = True


class AgentDataSet[A: Agent](BaseDataSet):
    """Data set for agents data.

    Args:
        name: the name of the data set
        agents: the agents to collect data from
        args: fields to collect
        select_kwargs : dict of valid keyword arguments for agentset.select
                        is passed, the agentset will be filtered before gathering
                        the data


    """

    def __init__(
        self,
        name: str,
        agents: AbstractAgentSet[A],
        *args,
        select_kwargs: dict | None = None,
    ):
        """Init."""
        super().__init__(name, "unique_id", *args)
        self.agents = agents
        self.select_kwargs = select_kwargs

    @property
    def data(self) -> list[dict[str, Any]]:
        """Return the data of the table."""
        self._check_closed()

        agents = (
            self.agents
            if self.select_kwargs is None
            else self.agents.select(**self.select_kwargs)
        )

        return [dict(zip(self._attributes, self._collector(agent))) for agent in agents]

    def close(self):
        """Close the data set."""
        super().close()
        self.agents = None


class ModelDataSet[M: Model](BaseDataSet):
    """Data set for model data.

    Args:
        name: the name of the data set
        model: the model to collect data from
        args: the fields to collect.

    """

    def __init__(self, name, model: M, *args):
        """Init."""
        super().__init__(name, *args)
        self.model = model

    @property
    def data(self) -> dict[str, Any]:
        """Return the data of the table."""
        self._check_closed()
        values = self._collector(self.model)
        if len(self._attributes) == 1:
            return {self._attributes[0]: values}
        else:
            return dict(zip(self._attributes, values))

    def close(self):
        """Close the data set."""
        super().close()
        self.model = None


class TableDataSet:
    """A Table DataSet.

    Args:
        name: the name of the data set
        fields: string or list of strings specifying the columns

    fixme: this needs a closer look
        it now follows the datacollector, so you just add
        a row.

    """

    def __init__(self, name, fields: str | list[str]):
        """Init."""
        self.name = name
        self.fields = fields if isinstance(fields, list) else [fields]
        self.rows = []

    def add_row(self, row: dict[str, Any]):
        """Add a row to the table."""
        if self.rows is None:
            raise RuntimeError(f"DataSet '{self.name}' has been closed")

        row_to_add = {k: row.pop(k) for k in self.fields}
        if len(row) > 0:
            raise ValueError(f"Row contains unexpected fields: {row.keys()}")
        self.rows.append(row_to_add)

    @property
    def data(self) -> list[dict[str, Any]]:
        """Return the data of the table."""
        if self.rows is None:
            raise RuntimeError(f"DataSet '{self.name}' has been closed")
        return self.rows

    def close(self):
        """Close the data set."""
        self.rows = None


class NumpyAgentDataSet[A: Agent]:
    """A NumPy array data set for storing agent data.

    Uses swap-with-last removal to keep data contiguous, allowing views.

    Note:
        This class does not inherit from BaseDataSet because it doesn't
        use the attrgetter-based collection mechanism.

    """

    _GROWTH_FACTOR = 2.0
    _MIN_GROWTH = 100

    def __init__(
        self,
        name: str,
        agent_type: type[A],
        *args: str,
        n: int = 100,
        dtype: np.dtype | type = np.float64,
    ):
        """Initialize the dataset.

        Args:
            name: Name of the dataset
            agent_type: The agent class to install properties on
            args: Attribute names to track
            n: Initial capacity
            dtype: NumPy dtype for the array

        """
        if not args:
            raise ValueError("At least one attribute must be specified")

        self.name = name
        self._attributes = args
        self._closed = False
        self.dtype = dtype
        self._index_in_table = f"_index_datatable_{name}"

        # Core data storage - always contiguous from 0 to _n_active-1
        self._agent_data: np.ndarray = np.empty((n, len(self._attributes)), dtype=dtype)
        self._n_active = 0

        # Mappings - index is always position in _agent_data
        self._index_to_agent: dict[int, A] = {}
        self._agent_to_index: dict[A, int] = {}
        self._attribute_to_index: dict[str, int] = {
            attr: i for i, attr in enumerate(args)
        }

        # Install properties on the agent class
        self.agent_type = agent_type
        if not hasattr(agent_type, "_datasets"):
            agent_type._datasets = set()
        agent_type._datasets.add(self.name)

        self._install_properties()

    def _make_getter_setter(self, attribute_name: str):
        """Generate getter and setter for the specified attribute."""
        j = self._attribute_to_index[attribute_name]
        index_attr = self._index_in_table
        data = self._agent_data

        def getter(agent: A):
            return data[agent.__dict__[index_attr], j]

        def setter(agent: A, value):
            data[agent.__dict__[index_attr], j] = value

        return getter, setter

    def _install_properties(self) -> None:
        """Install properties on the agent class for all attributes."""
        for attr in self._attributes:
            setattr(self.agent_type, attr, property(*self._make_getter_setter(attr)))

    def _expand_storage(self) -> None:
        """Expand the internal array when out of space."""
        current_size = self._agent_data.shape[0]
        growth = max(int(current_size * (self._GROWTH_FACTOR - 1)), self._MIN_GROWTH)
        new_size = current_size + growth

        new_data = np.empty((new_size, len(self._attributes)), dtype=self.dtype)
        new_data[:current_size] = self._agent_data
        self._agent_data = new_data

        # Reinstall properties to capture new array reference
        self._install_properties()

    def _check_closed(self) -> None:
        """Raise if dataset has been closed."""
        if self._closed:
            raise RuntimeError(f"DataSet '{self.name}' has been closed")

    def add_agent(self, agent: A) -> int:
        """Add an agent to the dataset.

        Args:
            agent: The agent to add

        Returns:
            The index assigned to the agent

        Complexity: O(1) amortized

        """
        self._check_closed()
        index = self._n_active

        # Expand if necessary
        if index >= self._agent_data.shape[0]:
            self._expand_storage()

        # Store index on agent
        agent.__dict__[self._index_in_table] = index

        # Update mappings
        self._agent_to_index[agent] = index
        self._index_to_agent[index] = agent
        self._n_active += 1

        return index

    def remove_agent(self, agent: A) -> None:
        """Remove an agent from the dataset using swap-with-last.

        Args:
            agent: The agent to remove

        Complexity: O(1)

        """
        self._check_closed()
        index = agent.__dict__.get(self._index_in_table)
        if index is None:
            return  # Not in dataset

        last_index = self._n_active - 1

        if index != last_index:
            # Swap data row with last active row
            self._agent_data[index] = self._agent_data[last_index]

            # Update the swapped agent's index
            swapped_agent = self._index_to_agent[last_index]
            swapped_agent.__dict__[self._index_in_table] = index
            self._agent_to_index[swapped_agent] = index
            self._index_to_agent[index] = swapped_agent

        # Remove the agent
        del self._agent_to_index[agent]
        del self._index_to_agent[last_index]
        agent.__dict__.pop(self._index_in_table, None)
        self._n_active -= 1

    @property
    def data(self) -> np.ndarray:
        """Return active agent data as a VIEW (no copy).

        Warning:
            Modifying the returned array modifies the underlying data.

        """
        self._check_closed()
        return self._agent_data[: self._n_active]

    @property
    def data_copy(self) -> np.ndarray:
        """Return a copy of active agent data."""
        self._check_closed()
        return self._agent_data[: self._n_active].copy()

    @property
    def active_agents(self) -> list[A]:
        """Return list of all active agents (order matches data rows)."""
        self._check_closed()
        return [self._index_to_agent[i] for i in range(self._n_active)]

    def _reset(self) -> None:
        """Reset the dataset to an empty state."""
        self._agent_to_index.clear()
        self._index_to_agent.clear()
        self._n_active = 0

    def __len__(self) -> int:
        """Return the number of active agents."""
        return self._n_active

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"NumpyAgentDataSet(name={self.name!r}, "
            f"active={self._n_active}, "
            f"capacity={self._agent_data.shape[0]})"
        )

    def close(self) -> None:
        """Close the dataset and remove properties from agent class."""
        if self._closed:
            return

        self._reset()
        # Remove properties from agent class
        for attr in self._attributes:
            with suppress(AttributeError):
                delattr(self.agent_type, attr)
        self._closed = True


class DataRegistry:
    """A registry for data sets."""

    def __init__(self):
        """Init."""
        self.datasets = {}

    def add_dataset(self, dataset: DataSet):
        """Add a dataset to the registry."""
        self.datasets[dataset.name] = dataset

    def create_dataset(self, dataset_type, name, *args, **kwargs) -> DataSet:
        """Create a dataset of the specified type and add it to the registry."""
        dataset = dataset_type(name, *args, **kwargs)
        self.datasets[name] = dataset
        return dataset

    def track_agents(
        self,
        agents: AbstractAgentSet,
        name: str,
        *args,
        select_kwargs: dict | None = None,
    ):
        """Track the specified fields for the agents in the AgentSet."""
        return self.create_dataset(
            AgentDataSet, name, agents, *args, select_kwargs=select_kwargs
        )

    def track_model(self, model: Model, name: str, *args):
        """Track the specified fields in the model."""
        return self.create_dataset(ModelDataSet, name, model, *args)

    def track_agents_numpy(
        self,
        agent_type: type[Agent],
        name: str,
        *args,
        n: int = 100,
        dtype: np.dtype | type = np.float64,
    ) -> NumpyAgentDataSet:
        """Track agent fields using NumPy storage.

        Args:
            agent_type: The agent class to install properties on
            name: Name of the dataset
            args: Attribute names to track
            n: Initial capacity
            dtype: NumPy dtype for the array

        Returns:
            The created NumpyAgentDataSet

        """
        dataset = NumpyAgentDataSet(name, agent_type, *args, n=n, dtype=dtype)
        self.datasets[name] = dataset
        return dataset

    def close(self):
        """Close all datasets."""
        for dataset in self.datasets.values():
            dataset.close()

    def __getitem__(self, name: str) -> DataSet:
        """Get a dataset by name."""
        return self.datasets[name]

    def __contains__(self, name: str) -> bool:
        """Check if a dataset exists."""
        return name in self.datasets

    def get(self, name: str) -> DataSet:
        """Get a dataset by name.

        raises KeyError if not found.

        """
        return self.datasets[name]

    def __iter__(self):
        """Iterate over datasets."""
        return iter(self.datasets.values())


if __name__ == "__main__":
    from mesa.examples import BoltzmannWealth

    model = BoltzmannWealth()
    model.test = 5
    agent_data = AgentDataSet("wealth", model.agents, "wealth")
    # model_data = ModelDataSet("gini", model, "test", gini=model.compute_gini)
    data = []
    for _ in range(5):
        model.step()
        data.append(agent_data.data)
        # data.append(model_data.data)
    print("blaat")
