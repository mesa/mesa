"""Helper classes for collecting statistics."""

import abc
import operator
from typing import Any

import numpy as np

from mesa.agent import Agent, AgentSet
from mesa.model import Model

__all__ = ["AgentDataSet", "DataRegistry", "DataSet", "ModelDataSet", "TableDataSet"]


class DataSet(abc.ABC):
    """Abstract base class for data sets.

    Args:
       name: the name of the data set
       kwargs: key value pairs specifying the fields to collect
               a value must be either a string (for attributes) or a
               callable.


    """

    def __init__(self, name, *args):
        """Init."""
        self.name = name
        self._attributes = args
        self._collector = operator.attrgetter(*self._attributes)

    @property
    @abc.abstractmethod
    def data(self):
        """Return the data of the table."""
        ...


class AgentDataSet[A: Agent](DataSet):
    """Data set for agents data.

    Args:
        name: the name of the data set
        agents: the agents to collect data from
        select_kwargs : dict of valid keyword arguments for agentset.select
                        is passed, the agentset will be filtered before gathering
                        the data
        kwargs: key value pairs specifying the fields to collect.
                a value must be either a string (for attributes) or a
                callable.


    """

    def __init__(
        self,
        name,
        agents: AgentSet[A],
        *args,
        select_kwargs: dict | None = None,
    ):
        """Init."""
        super().__init__(name, *["unique_id", *args])
        self.agents = agents
        self.select_kwargs = select_kwargs

    @property
    def data(self) -> list[dict[str, Any]]:
        """Return the data of the table."""
        # gets the data for the fields from the agents
        data: list[dict[str, Any]] = []

        agents = (
            self.agents
            if self.select_kwargs is None
            else self.agents.select(*self.select_kwargs)
        )

        for agent in agents:
            data.append(dict(zip(self._attributes, self._collector(agent))))
        return data


class ModelDataSet[M: Model](DataSet):
    """Data set for model data.

    Args:
        name: the name of the data set
        model: the model to collect data from
        kwargs: key value pairs specifying the fields to collect.
                a value must be either a string (for attributes) or a
                callable.

    """

    def __init__(self, name, model: M, *args):
        """Init."""
        super().__init__(name, *args)
        self.model = model

    @property
    def data(self) -> dict[str, Any]:
        """Return the data of the table."""
        # gets the data for the fields from the agents
        values = self._collector(self.model)
        try:
            return dict(zip(self._attributes, values))
        except TypeError:
            return {self._attributes[0]: values}


class TableDataSet(DataSet):
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
        super().__init__(name)
        self.fields = fields
        self.rows = []

    def add_row(self, row: dict[str, Any]):
        """Add a row to the table."""
        self.rows.append({k: row[k] for k in self.fields})

    @property
    def data(self) -> list[dict[str, Any]]:
        """Return the data of the table."""
        # gets the data for the fields from the agents
        return self.rows


class DataRegistry[M: Model]:
    """A registry for data sets."""

    def __init__(self):
        """Init."""
        self.datasets = {}

    def add_dataset(self, dataset: DataSet):
        """Add a dataset to the registry."""
        self.datasets[dataset.name] = dataset

    def create_dataset(self, dataset_type, name, *args, **kwargs):
        """Create a dataset of the specified type and add it to the registry."""
        self.datasets[name] = dataset_type(name, *args, **kwargs)

    def track_agents(
        self, agents: AgentSet, name: str, *args, select_kwargs: dict | None = None
    ):
        """Track the specified fields for the agents in the AgentSet."""
        self.create_dataset(
            AgentDataSet, name, agents, *args, select_kwargs=select_kwargs
        )

    def track_model(self, model: M, name: str, *args):
        """Track the specified fields in the model."""
        self.create_dataset(ModelDataSet, name, model, *args)

    def __getitem__(self, name: str):
        """Get a dataset by name."""
        return self.datasets[name]


class NumpyAgentDataSet[A: Agent](DataSet):
    """A NumPy array data set for storing agent data.

    Args:
       name: the name of the data set
       agent_type: the type of agents to collect data from
       *args: the attributes to collect
       n: the initial size of the internal numpy array

    Notes:
        The internal numpy array is automatically made larger if needed.
        However, it can be beneficial to prespecify its exact size via `n` if known to avoid resizing.

    """

    # Constants
    _GROWTH_FACTOR = 2.0  # Double size when full
    _MIN_GROWTH = 100  # Minimum rows to add
    _COMPACT_THRESHOLD = 0.3  # when to reindex everything for memory reasons

    def __init__(
        self,
        name: str,
        agent_type: type[A],
        *args,
        n=100,  # just for initial sizing of the inner numpy array
        dtype=np.float64,
    ):
        """Init."""
        super().__init__(
            name,
            *args,  # fixme: what about unique_id?
        )
        self.dtype = dtype
        self._index_in_table = f"_index_datatable_{name}"

        # Core data storage
        self._agent_data: np.ndarray = np.empty((n, len(self._attributes)), dtype=dtype)
        self._is_active: np.ndarray = np.zeros(n, dtype=bool)

        # Agent tracking
        self._n_slots = 0  # Total slots used (including tombstones)
        self._n_active = 0  # Number of active agents
        self._free_indices: list[int] = []  # Recycled tombstone indices

        # Mappings
        self._index_to_agent: dict[int, A] = {}
        self._agent_to_index: dict[A, int] = {}
        self.attribute_to_index: dict[str, int] = {
            attr: i for i, attr in enumerate(args)
        }

        # Install properties on the agent class
        agent_type._datasets.add(self.name)
        for attr in args:
            setattr(agent_type, attr, property(*generate_getter_and_setter(self, attr)))

    def _expand_storage(self) -> None:
        """Expand the internal arrays when out of space."""
        current_size = self._agent_data.shape[0]
        growth = max(int(current_size * (self._GROWTH_FACTOR - 1)), self._MIN_GROWTH)
        new_size = current_size + growth

        # Expand data array
        new_data = np.empty((new_size, len(self._attributes)), dtype=self.dtype)
        new_data[:current_size] = self._agent_data
        self._agent_data = new_data

        # Expand active mask
        new_mask = np.zeros(new_size, dtype=bool)
        new_mask[:current_size] = self._is_active
        self._is_active = new_mask

    def add_agent(self, agent: A) -> int:
        # New agent - try to reuse a freed slot first
        if self._free_indices:
            index = self._free_indices.pop()
        else:
            # Need a new slot
            index = self._n_slots
            self._n_slots += 1

            # Expand array if necessary
            if index >= self._agent_data.shape[0]:
                self._expand_storage()

        # Activate the slot
        setattr(agent, self._index_in_table, index)  # set row index on agent
        self._is_active[index] = True
        self._agent_to_index[agent] = index
        self._index_to_agent[index] = agent
        self._n_active += 1

        return index

    def remove_agent(self, agent: A) -> None:
        """Remove an agent from the dataset in O(1) time.

        Uses tombstone marking - the slot is marked inactive but not deleted.
        The slot can be reused when new agents are added.

        Automatically compacts if fragmentation exceeds threshold.
        """
        index = getattr(agent, self._index_in_table, None)
        if index is None:
            return  # Not in dataset

        # Mark as tombstone
        self._is_active[index] = False

        # Clean up mappings
        del self._agent_to_index[agent]
        del self._index_to_agent[index]
        delattr(agent, self._index_in_table)

        # Add to list of available indices for reuse
        self._free_indices.append(index)
        self._n_active -= 1

        # Auto-compact if too fragmented
        fragmentation = len(self._free_indices) / max(self._n_slots, 1)
        if fragmentation > self._COMPACT_THRESHOLD:
            self.compact()

    def _reset(self) -> None:
        """Reset the dataset to an empty state."""
        self._agent_to_index.clear()
        self._index_to_agent.clear()
        self._is_active[:] = False
        self._n_slots = 0
        self._n_active = 0
        self._free_indices.clear()

    def compact(self) -> None:
        """Remove all tombstones and reindex the dataset.

        This is an O(n) operation that eliminates fragmentation by removing
        dead slots and reassigning indices to active agents sequentially.

        Call this manually or let it happen automatically when fragmentation
        exceeds the threshold.
        """
        if not self._free_indices:
            return  # Already compact

        # Get all active indices in order
        active_indices = np.where(self._is_active[: self._n_slots])[0]
        n_active = len(active_indices)

        if n_active == 0:
            # No active agents - reset everything
            self._reset()
            return

        # Create a new compacted data array
        new_data = self._agent_data[active_indices]

        # Rebuild mappings with new sequential indices
        new_agent_to_index = {}
        new_index_to_agent = {}

        for new_index, old_index in enumerate(active_indices):
            agent = self._index_to_agent[old_index]
            new_agent_to_index[agent] = new_index
            new_index_to_agent[new_index] = agent
            setattr(agent, self._index_in_table, new_index)

        # Update state
        self._agent_data[:n_active] = new_data
        self._agent_to_index = new_agent_to_index
        self._index_to_agent = new_index_to_agent
        self._is_active[:n_active] = True
        self._is_active[n_active:] = False
        self._n_slots = n_active
        self._n_active = n_active
        self._free_indices.clear()

    @property
    def data(self) -> np.ndarray:
        """Return only active agent data as a numpy array.

        Returns a view (not a copy) when possible for efficiency.
        """
        if self._n_active == 0:
            return np.empty((0, len(self._attributes)), dtype=self.dtype)

        # Boolean indexing - returns only active rows
        # self._n_slots is the number of slots that have been handed out
        # some might be inactive now, but there can never be more
        # the alternative is to just do self._agent_data[self._is_active]
        return self._agent_data[: self._n_slots][self._is_active[: self._n_slots]]

    @property
    def active_agents(self) -> list[A]:
        """Return list of all active agents."""
        active_indices = np.where(self._is_active[: self._n_slots])[0]
        return [self._index_to_agent[i] for i in active_indices]

    @property
    def fragmentation(self) -> float:
        """Return the fragmentation ratio (0.0 to 1.0).

        0.0 means no fragmentation, 1.0 means all slots are tombstones.
        """
        if self._n_slots == 0:
            return 0.0
        return len(self._free_indices) / self._n_slots

    def __len__(self) -> int:
        """Return the number of active agents."""
        return self._n_active

    def __repr__(self) -> str:
        return (
            f"NumpyAgentDataSet(name={self.name!r}, "
            f"active={self._n_active}, "
            f"slots={self._n_slots}, "
            f"fragmentation={self.fragmentation:.1%})"
        )


def generate_getter_and_setter(table: NumpyAgentDataSet, attribute_name: str):
    """Generate getter and setter for the specified attribute."""
    # fixme: what if we make this  a method on the data set instead?
    #     or just pass it along when generating the getter and setter?
    j = table.attribute_to_index[attribute_name]

    def getter(obj: Agent):
        i = getattr(
            obj, table._index_in_table
        )  # should just exist because of registration in Agent.__init__
        return table._agent_data[i, j]

    def setter(obj: Agent, value):
        i = getattr(obj, table._index_in_table)
        table._agent_data[i, j] = value

    return getter, setter


if __name__ == "__main__":
    from mesa.examples import BoltzmannWealth

    model = BoltzmannWealth()
    model.test = 5
    agent_data = AgentDataSet("wealth", model.agents, "wealth")
    model_data = ModelDataSet("gini", model, "test", gini=model.compute_gini)
    data = []
    for _ in range(5):
        model.step()
        data.append(agent_data.data)
        data.append(model_data.data)
    print("blaat")
