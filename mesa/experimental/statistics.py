import abc
import operator
from collections import defaultdict

from collections.abc import Callable


from typing import Any

from mesa.agent import AgentSet
from mesa.examples import BoltzmannWealth
from mesa.model import Model



class DataSet(abc.ABC):
    """Abstract base class for data sets.

    Args:
       name: the name of the data set
       kwargs: key value pairs specifying the fields to collect
               a value must be either a string (for attributes) or a
               callable.


    """

    def __init__(self, name, **kwargs):
        """Init."""
        self.name = name
        self.fields = kwargs

        self._collectors = {}
        # fixme: the operator.itemgetter optimization
        #    for all attributes in one go can also be used here.
        for name, reporter in kwargs.items():
            match reporter:
                case str():
                    self._collectors[name] = operator.attrgetter(reporter)
                case Callable():
                    self._collectors[name] = reporter
                case _:
                    raise ValueError("Invalid reporter type for {name}")

        # internal datastructure


    @property
    @abc.abstractmethod
    def data(self):
        """Return the data of the table"""
        ...


class AgentDataSet(DataSet):
    """Data set for agents data.

    Args:
        name: the name of the data set
        agents: the agents to collect data from
        kwargs: key value pairs specifying the fields to collect.
                a value must be either a string (for attributes) or a
                callable.


    """

    def __init__(self, name, agents: AgentSet, **kwargs):
        """Init."""
        super().__init__(name)
        self.agents = agents

        # you can basically always do this
        # but just for the strings.
        self.string_fields = []
        self._collectors = {}
        for k, v in kwargs.items():
            match v:
                case str():
                    self.string_fields.append(v)
                case Callable():
                    self._collectors[k] = v
                case _:
                    raise ValueError("Invalid reporter type for {k}")
        self.string_fields.append("unique_id")

        # Create a single getter for [unique_id, attr1, attr2, ...]
        self._agent_getter = operator.attrgetter(*self.string_fields)

    @property
    def data(self) -> list[dict[str, Any]]:
        # gets the data for the fields from the agents
        data: list[dict[str, Any]] = []
        for agent in self.agents:
            agent_data = {
                k: v for k, v in zip(self.string_fields, self._agent_getter(agent))
            } | {k: func(agent) for k, func in self._collectors.items()}
            data.append(agent_data)
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

    def __init__(self, name, model: M, **kwargs):
        """Init."""
        super().__init__(name, **kwargs)
        self.model = model

    @property
    def data(self) -> dict[str, Any]:
        # gets the data for the fields from the agents
        data = {}
        for k, v in self._collectors.items():
            data[k] = v()

        return data


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
        self.datasets.append({k:row[k] for k in self.fields})

    @property
    def data(self) -> list[dict[str, Any]]:
        # gets the data for the fields from the agents
        return self.datasets


class DataRegistry:
    """A registry for data sets."""

    def __init__(self):
        self.datasets = {}

    def add_dataset(self, dataset: DataSet):
        self.datasets[dataset.name] = dataset

    def create_dataset(self, dataset_type, name, *args, **kwargs):
        self.datasets[name] = dataset_type(name, *args, **kwargs)

    def track_agents(self, agents, **kwargs):
        self.create_dataset(AgentDataSet, agents.name, agents, **kwargs)

    def track_model(self, model, **kwargs):
        self.create_dataset(ModelDataSet, model.name, model, **kwargs)

    def __getitem__(self, name: str):
        return self.datasets[name]



if __name__ == '__main__':
    model = BoltzmannWealth()
    agent_data = AgentDataSet("wealth", model.agents, wealth="wealth")
    model_data = ModelDataSet("gini", model, gini=model.compute_gini)
    data = []
    for _ in range(5):
        model.step()
        data.append(agent_data.data)
        data.append(model_data.data)
    print("blaat")