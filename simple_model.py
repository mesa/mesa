"""A minimal Mesa example for beginners."""

from mesa import Agent, Model


class SimpleAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id

    def step(self):
        print(f"Agent {self.unique_id} is acting")


class SimpleModel(Model):
    def __init__(self, n):
        super().__init__()
        self.num_agents = n
        self.my_agents = [SimpleAgent(i, self) for i in range(n)]

    def step(self):
        for agent in self.my_agents:
            agent.step()


if __name__ == "__main__":
    model = SimpleModel(3)
    model.step()
