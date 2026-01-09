import networkx as nx
import numpy as np

import mesa
from mesa import Agent
from mesa.examples.advanced.alliance_formation.agents import AllianceAgent
from mesa.experimental.meta_agents.meta_agent import (
    create_meta_agent,
    find_combinations,
)


class MultiLevelAllianceModel(mesa.Model):
    """
    Model for simulating multi-level alliances among agents.
    """

    def __init__(self, n=50, mean=0.5, std_dev=0.1, seed=42):
        """
        Initialize the model.

        Args:
            n (int): Number of agents.
            mean (float): Mean value for normal distribution.
            std_dev (float): Standard deviation for normal distribution.
            seed (int): Random seed.
        """
        super().__init__(seed=seed)
        self.population = n
        self.network = nx.Graph()  # Initialize the network
        self.datacollector = mesa.DataCollector(model_reporters={"Network": "network"})

        # Create Agents
        power = self.rng.normal(mean, std_dev, n)
        power = np.clip(power, 0, 1)
        position = self.rng.normal(mean, std_dev, n)
        position = np.clip(position, 0, 1)
        AllianceAgent.create_agents(self, n, power, position)
        agent_ids = [
            (agent.unique_id, {"size": 300, "level": 0}) for agent in self.agents
        ]
        self.network.add_nodes_from(agent_ids)

    def add_link(self, meta_agent, agents):
        """
        Add links between a meta agent and its constituent agents in the network.

        Args:
            meta_agent (MetaAgent): The meta agent.
            agents (list): List of agents.
        """
        for agent in agents:
            self.network.add_edge(meta_agent.unique_id, agent.unique_id)

    def calculate_shapley_value(self, agents):
        """
        Calculate the Shapley value of the two agents.

        Args:
            agents (list): List of agents.

        Returns:
            tuple: Potential utility, new position, and level.
        """
        agent_0 = agents[0]
        agent_1 = agents[1]

        positions = agents.get("position")
        new_position = 1 - (max(positions) - min(positions))
        potential_utility = agents.agg("power", sum) * 1.2 * new_position

        value_0 = 0.5 * agent_0.power + 0.5 * (potential_utility - agent_1.power)
        value_1 = 0.5 * agent_1.power + 0.5 * (potential_utility - agent_0.power)

        if value_0 > agent_0.power and value_1 > agent_1.power:
            if agent_0.level > agent_1.level:
                level = agent_0.level
            elif agent_0.level == agent_1.level:
                level = agent_0.level + 1
            else:
                level = agent_1.level

            return potential_utility, new_position, level

    def only_best_combination(self, combinations):
        """
        Filter to keep only the best combination for each agent.

        Args:
            combinations (list): List of combinations.

        Returns:
            dict: Unique combinations.
        """
        best = {}
        # Determine best option for EACH agent
        for group, value in combinations:
            agent_ids = sorted(group.get("unique_id"))  # by default is bilateral
            # Deal with all possibilities
            if (
                agent_ids[0] not in best and agent_ids[1] not in best
            ):  # if neither in add both
                best[agent_ids[0]] = [group, value, agent_ids]
                best[agent_ids[1]] = [group, value, agent_ids]
            elif (
                agent_ids[0] in best and agent_ids[1] in best
            ):  # if both in, see if both would be trading up
                if (
                    value[0] > best[agent_ids[0]][1][0]
                    and value[0] > best[agent_ids[1]][1][0]
                ):
                    # Remove the old alliances
                    del best[best[agent_ids[0]][2][1]]
                    del best[best[agent_ids[1]][2][0]]
                    # Add the new alliance
                    best[agent_ids[0]] = [group, value, agent_ids]
                    best[agent_ids[1]] = [group, value, agent_ids]
            elif (
                agent_ids[0] in best
            ):  # if only agent_ids[0] in, see if it would be trading up
                if value[0] > best[agent_ids[0]][1][0]:
                    # Remove the old alliance for agent_ids[0]
                    del best[best[agent_ids[0]][2][1]]
                    # Add the new alliance
                    best[agent_ids[0]] = [group, value, agent_ids]
                    best[agent_ids[1]] = [group, value, agent_ids]
            elif (
                agent_ids[1] in best
            ):  # if only agent_ids[1] in, see if it would be trading up
                if value[0] > best[agent_ids[1]][1][0]:
                    # Remove the old alliance for agent_ids[1]
                    del best[best[agent_ids[1]][2][0]]
                    # Add the new alliance
                    best[agent_ids[0]] = [group, value, agent_ids]
                    best[agent_ids[1]] = [group, value, agent_ids]

        # Create a unique dictionary of the best combinations
        unique_combinations = {}
        for group, value, agents_nums in best.values():
            unique_combinations[tuple(agents_nums)] = [group, value]

        return unique_combinations.values()

    def step(self):
        """
        Execute one step of the model.
        """
        # Get all other agents of the same type
        agent_types = list(self.agents_by_type.keys())

        for agent_type in agent_types:
            similar_agents = self.agents_by_type[agent_type]

            # Find the best combinations using find_combinations
            if (
                len(similar_agents) > 1
            ):  # only form alliances if there are more than 1 agent
                combinations = find_combinations(
                    self,
                    similar_agents,
                    size=2,
                    evaluation_func=self.calculate_shapley_value,
                    filter_func=self.only_best_combination,
                )

                for alliance, attributes in combinations:
                    class_name = f"MetaAgentLevel{attributes[2]}"
                    meta = create_meta_agent(
                        self,
                        class_name,
                        alliance,
                        Agent,
                        meta_attributes={
                            "level": attributes[2],
                            "power": attributes[0],
                            "position": attributes[1],
                        },
                    )

                    # Update the network if a new meta agent instance created
                    if meta:
                        self.network.add_node(
                            meta.unique_id,
                            size=(meta.level + 1) * 300,
                            level=meta.level,
                        )
                        self.add_link(meta, meta.agents)
