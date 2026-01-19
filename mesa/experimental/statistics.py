"""Helper classes for collecting statistics."""

import abc
import operator
from collections.abc import Callable
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

    def __init__(self, name, *args, **kwargs):
        """Init."""
        self.name = name
        self._args = args
        self._collectors: dict[str, Callable] = {}
        self._kwargs = kwargs

        if args:
            self._collectors["attributes"] = operator.attrgetter(*args)

        for name, reporter in kwargs.items():
            if not isinstance(reporter, Callable):
                raise ValueError(f"Invalid reporter type for {name}")
            self._collectors[name] = reporter

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
        **kwargs,
    ):
        """Init."""
        super().__init__(name, *["unique_id", *args], **kwargs)
        self.agents = agents
        self.select_kwargs = select_kwargs

    @property
    def data(self) -> list[dict[str, Any]]:
        """Return the data of the table."""
        # gets the data for the fields from the agents
        data: list[dict[str, Any]] = []

        agents = (
            self._agents
            if self.select_kwargs is None
            else self.agents.select(*self.select_kwargs)
        )

        for agent in agents:
            attribute_data = dict(
                zip(self._args, self._collectors["attributes"](agent))
            )
            callable_data = {
                k: func(agent)
                for k, func in self._collectors.items()
                if k != "attributes"
            }
            data.append(attribute_data | callable_data)
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

    def __init__(self, name, model: M, *args, **kwargs):
        """Init."""
        super().__init__(name, *args, **kwargs)
        self.model = model

    @property
    def data(self) -> dict[str, Any]:
        """Return the data of the table."""
        # gets the data for the fields from the agents
        try:
            attribute_data = self._collectors["attributes"]
        except KeyError:
            attribute_data = {}
        else:
            if len(self._args) == 1:
                attribute_data = {self._args[0]: attribute_data(model)}
            else:
                attribute_data = dict(zip(self._args, attribute_data(model)))

        callable_data = {
            k: func() for k, func in self._collectors.items() if k != "attributes"
        }

        return attribute_data | callable_data


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
        self.datasets = []

    def add_row(self, row: dict[str, Any]):
        """Add a row to the table."""
        self.datasets.append({k: row[k] for k in self.fields})

    @property
    def data(self) -> list[dict[str, Any]]:
        """Return the data of the table."""
        # gets the data for the fields from the agents
        return self.datasets


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

    def track_agents(self, agents: AgentSet, name: str, **kwargs):
        """Track the specified fields for the agents in the AgentSet."""
        self.create_dataset(AgentDataSet, name, agents, **kwargs)

    def track_model(self, model: M, name: str, **kwargs):
        """Track the specified fields in the model."""
        self.create_dataset(ModelDataSet, name, model, **kwargs)

    def __getitem__(self, name: str):
        """Get a dataset by name."""
        return self.datasets[name]


class NumpyAgentDataSet[A: Agent](DataSet):
    """A numpy array based data set for agents, used in conjunction with DataField"""

    def __init__(
        self,
        name: str,
        agent_type: type[A],
        *args,
        n=100,  # just for initial sizing of the inner numpy array
    ):
        """Init."""
        super().__init__(
            name,
            *args, # fixme: what about unique_id?
        )

        self._agent_data: np.array = np.empty((n, len(self._args)), dtype=float)
        self._data: (
            np.array
        )  # a view on _agent_positions containing all active positions

        # the list of agents in the space
        self.active_agents = []
        self._n_agents = 0  # the number of active agents in the space

        #  a mapping from agents to index and vice versa
        self._index_to_agent: dict[int, A] = {}
        self._agent_to_index: dict[A, int | None] = {}
        self.attribute_to_index: dict[str, int] = {
            arg: i for i, arg in enumerate(self._args)
        }

        for arg in self._args:
            setattr(agent_type, arg, property(*generate_getter_and_setter(self, arg)))

    def agent_to_index(self, agent: A):
        """Helper method to map an agent to its index in the table"""
        try:
            return self._agent_to_index[agent]
        except KeyError:
            # agent is new
            # we need to somehow keep track if agents are removed
            # or we need a weakkey trick but then how to do the reindexing
            # or we can update remove to cleanup all datafields
            index = self._n_agents
            self._n_agents += 1

            if self._agent_data.shape[0] <= index:
                # we are out of space
                fraction = 0.2  # we add 20%  Fixme
                n = round(fraction * self._n_agents, None)
                self._agent_data = np.vstack(
                    [
                        self._agent_data,
                        np.empty(
                            (n, len(self._args)),
                        ),
                    ]
                )

            self._agent_to_index[agent] = index
            self._index_to_agent[index] = agent

            # we want to maintain a view rather than a copy on the active agents and positions
            # this is essential for the performance of the rest of this code
            self.active_agents.append(agent)
            self._data = self._agent_data[0 : self._n_agents]

            return index

    @property
    def data(self):
        return self._data

    def _remove_agent(self, agent: A) -> None:
        """Remove an agent from the table.

        fixme when to call this

        """
        index = self._agent_to_index[agent]
        self._agent_to_index.pop(agent, None)
        self._index_to_agent.pop(index, None)
        del self.active_agents[index]

        # Shift all subsequent agents up by 1
        for agent in self.active_agents[index::]:
            old_index = self._agent_to_index[agent]
            self._agent_to_index[agent] = old_index - 1
            self._index_to_agent[old_index - 1] = agent

        # Clean up the stale entry from the last shifted agent
        if len(self.active_agents) > index:
            self._index_to_agent.pop(len(self.active_agents), None)

        # we move all data below the removed agent one row up
        self._agent_data[index : self._n_agents - 1] = self._agent_data[
            index + 1 : self._n_agents
        ]
        self._n_agents -= 1
        self._data = self._agent_data[0 : self._n_agents]


def generate_getter_and_setter(table: NumpyAgentDataSet, attribute_name: str):
    """Generate getter and setter for a DataField"""
    data = table._agent_data
    j = table.attribute_to_index[attribute_name]

    def getter(obj: Agent):
        i = table.agent_to_index(obj)
        return data[i, j]

    def setter(obj: Agent, value):
        i = table.agent_to_index(obj)
        data[i, j] = value

    return getter, setter





class DataField(property):
    """A property that tracks a field of an object."""

    def __init__(self, table: str, attribute_name: str):
        # fixme, here a full descriptor might work nicer
        # because of __set_name__
        super().__init__(self.getter, self.setter)
        self.table_name = table
        self.attribute_name = attribute_name

    def getter(self, obj: Agent):
        table = obj.model.data_registry[self.table_name]
        i = table.agent_to_index(obj)
        j = table.attribute_to_index[self.attribute_name]
        return table._data[i, j]

    def setter(self, obj: Agent, value):
        table = obj.model.data_registry[self.table_name]
        i = table.agent_to_index(obj)
        j = table.attribute_to_index[self.attribute_name]
        table._data[i, j] = value


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
