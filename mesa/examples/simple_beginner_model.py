import mesa


class SimpleAgent(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.energy = 1

    def step(self):
        self.energy += 1


class SimpleModel(mesa.Model):
    def __init__(self, n_agents=5):
        super().__init__()

        # create agents
        SimpleAgent.create_agents(self, n_agents)

        self.datacollector = mesa.DataCollector(
            model_reporters={"total_energy": lambda m: m.agents.agg("energy", sum)},
            agent_reporters={"energy": "energy"},
        )

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


if __name__ == "__main__":
    model = SimpleModel(n_agents=5)
    model.run_for(10)

    print(model.datacollector.get_model_vars_dataframe())
