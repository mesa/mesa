"""Helper classes for collecting statistics."""

import abc
import operator
from collections.abc import Callable
from typing import Any

from mesa.agent import Agent, AgentSet
from mesa.examples import BoltzmannWealth
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


if __name__ == "__main__":
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
