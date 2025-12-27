from collections import defaultdict

import numpy as np
from agents import CultureAgent

import mesa
from mesa.space_adaptive import AdaptiveVoronoiSpace


class CulturalSegregation(mesa.Model):
    """A model demonstrating AdaptiveVoronoiSpace."""

    def __init__(
        self,
        n_agents=200,
        n_clusters=10,
        width=100,
        height=100,
        tolerance=0.15,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.n_agents = n_agents
        self.n_clusters = n_clusters

        self.space = AdaptiveVoronoiSpace(
            x_max=width, y_max=height, torus=False, n_clusters=n_clusters
        )

        self.region_opinions = {}

        for _ in range(n_agents):
            if self.random.random() < 0.5:
                opinion = self.random.gauss(0.2, 0.1)
            else:
                opinion = self.random.gauss(0.8, 0.1)

            opinion = np.clip(opinion, 0.0, 1.0)
            agent = CultureAgent(self, opinion, threshold=tolerance)

            pos = (self.random.random() * width, self.random.random() * height)
            self.space.place_agent(agent, pos)

        self.space.rebuild_voronoi()
        self.calculate_region_aggregates()

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Unhappy Agents": lambda m: sum(1 for a in m.agents if not a.happy),
                "Inertia": lambda m: getattr(m.space, "clustering_kwargs", {}).get(
                    "inertia_", 0
                ),
            }
        )

    def calculate_region_aggregates(self):
        region_opinions_list = defaultdict(list)

        if self.space.agent_labels is not None:
            for agent_idx, label in enumerate(self.space.agent_labels):
                agent = self.space._index_to_agent[agent_idx]
                region_opinions_list[label].append(agent.opinion)

        self.region_opinions = {}
        for region_id in range(self.n_clusters):
            opinions = region_opinions_list[region_id]
            if opinions:
                self.region_opinions[region_id] = np.mean(opinions)
            else:
                self.region_opinions[region_id] = 0.5

    def step(self):
        self.space.rebuild_voronoi()
        self.calculate_region_aggregates()
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
