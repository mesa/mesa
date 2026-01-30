"""Helper classes for collecting statistics."""
from __future__ import annotations

import abc
import operator
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

import numpy as np

from mesa.agent import Agent, AgentSet, AbstractAgentSet

if TYPE_CHECKING:
    from mesa.model import Model

__all__ = [
    "AgentDataSet",
    "DataRegistry",
    "DataSet",
    "ModelDataSet",
    "TableDataSet",
    "NumpyAgentDataSet",
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
        # gets the data for the fields from the agents
        data: list[dict[str, Any]] = []

        agents = (
            self.agents
            if self.select_kwargs is None
            else self.agents.select(**self.select_kwargs)
        )

        for agent in agents:
            data.append(dict(zip(self._attributes, self._collector(agent))))
        return data

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
        # gets the data for the fields from the agents
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
        self.rows.append({k: row[k] for k in self.fields})

    @property
    def data(self) -> list[dict[str, Any]]:
        """Return the data of the table."""
        if self.rows is None:
            raise RuntimeError(f"DataSet '{self.name}' has been closed")
        # gets the data for the fields from the agents
        return self.rows

    def close(self):
        """Close the data set."""
        self.rows = None


class NumpyAgentDataSet[A: Agent](BaseDataSet):
    """A NumPy array data set for storing agent data.

    Uses swap-with-last removal to keep data contiguous, allowing views.
    """

    _GROWTH_FACTOR = 2.0
    _MIN_GROWTH = 100

    def __init__(
            self,
            name: str,
            agent_type: type[A],
            *args,
            n=100,
            dtype=np.float64,
    ):
        """Init."""
        super().__init__(name, *args)
        self.dtype = dtype
        self._index_in_table = f"_index_datatable_{name}"

        # Core data storage - always contiguous from 0 to _n_active-1
        self._agent_data: np.ndarray = np.empty((n, len(self._attributes)), dtype=dtype)
        self._n_active = 0

        # Mappings - index is always position in _agent_data
        self._index_to_agent: dict[int, A] = {}
        self._agent_to_index: dict[A, int] = {}
        self.attribute_to_index: dict[str, int] = {
            attr: i for i, attr in enumerate(args)
        }

        # Install properties on the agent class
        self.agent_type = agent_type
        if not hasattr(agent_type, "_datasets"):
            agent_type._datasets = set()
        agent_type._datasets.add(self.name)
        for attr in args:
            setattr(
                agent_type, attr, property(*generate_getter_and_setter(self, attr))
            )

    def _expand_storage(self) -> None:
        """Expand the internal array when out of space."""
        current_size = self._agent_data.shape[0]
        growth = max(int(current_size * (self._GROWTH_FACTOR - 1)), self._MIN_GROWTH)
        new_size = current_size + growth

        new_data = np.empty((new_size, len(self._attributes)), dtype=self.dtype)
        new_data[:current_size] = self._agent_data
        self._agent_data = new_data

    def add_agent(self, agent: A) -> int:
        """Add an agent to the dataset. O(1)."""
        index = self._n_active

        # Expand if necessary
        if index >= self._agent_data.shape[0]:
            self._expand_storage()

            # Store index on agent
        setattr(agent, self._index_in_table, index)

        # Update mappings
        self._agent_to_index[agent] = index
        self._index_to_agent[index] = agent
        self._n_active += 1

        return index

    def remove_agent(self, agent: A) -> None:
        """Remove an agent from the dataset. O(1) using swap-with-last."""
        index = getattr(agent, self._index_in_table, None)
        if index is None:
            return  # Not in dataset

        last_index = self._n_active - 1

        if index != last_index:
            # Swap data row with last active row
            self._agent_data[index] = self._agent_data[last_index]

            # Update the swapped agent's index
            swapped_agent = self._index_to_agent[last_index]
            setattr(swapped_agent, self._index_in_table, index)
            self._agent_to_index[swapped_agent] = index
            self._index_to_agent[index] = swapped_agent

            # Remove the agent
        del self._agent_to_index[agent]
        del self._index_to_agent[last_index]
        delattr(agent, self._index_in_table)
        self._n_active -= 1

    @property
    def data(self) -> np.ndarray:
        """Return active agent data as a VIEW (no copy).

        WARNING: Modifying the returned array modifies the underlying data.
        """
        self._check_closed()
        return self._agent_data[:self._n_active]

    @property
    def data_copy(self) -> np.ndarray:
        """Return a copy of active agent data."""
        self._check_closed()
        return self._agent_data[:self._n_active].copy()

    @property
    def active_agents(self) -> list[A]:
        """Return list of all active agents (order matches data rows)."""
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
        return (
            f"NumpyAgentDataSet(name={self.name!r}, "
            f"active={self._n_active}, "
            f"capacity={self._agent_data.shape[0]})"
        )

    def close(self):
        """Close the dataset."""
        super().close()
        self._reset()
        for attr in self._attributes:
            delattr(self.agent_type, attr)
        self.agent_type._datasets.discard(self.name)

def generate_getter_and_setter(table: NumpyAgentDataSet, attribute_name: str):
    """Generate getter and setter for the specified attribute."""
    # fixme: what if we make this  a method on the data set instead?
    #     or just pass it along when generating the getter and setter?
    j = table.attribute_to_index[attribute_name]
    index_attr = table._index_in_table

    def getter(obj: Agent):
        i = getattr(
            obj, index_attr
        )  # should just exist because of registration in Agent.__init__
        return table._agent_data[i, j]

    def setter(obj: Agent, value):
        i = getattr(obj, index_attr)
        table._agent_data[i, j] = value

    return getter, setter


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
        self, agents: AbstractAgentSet, name: str, *args, select_kwargs: dict | None = None
    ):
        """Track the specified fields for the agents in the AgentSet."""
        return self.create_dataset(
            AgentDataSet, name, agents, *args, select_kwargs=select_kwargs
        )

    def track_model(self, model: Model, name: str, *args):
        """Track the specified fields in the model."""
        return self.create_dataset(ModelDataSet, name, model, *args)

    def close_all(self):
        """Close all datasets."""
        for dataset in self.datasets.values():
            dataset.close()

    def __getitem__(self, name: str):
        """Get a dataset by name."""
        return self.datasets[name]


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
