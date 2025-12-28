from mesa import Agent


class BasicAgent(Agent):
    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

    def step(self):
        if self.random.random() > 0.5:
            print(f"[Basic {self.unique_id}] Collecting nearby resource.")
        else:
            print(f"[Basic {self.unique_id}] Waiting...")


class CognitiveAgent(Agent):
    def __init__(self, model, orchestrator):
        super().__init__(model)
        self.orchestrator = orchestrator
        self.memory = []

    def step(self):
        context = {"goal": "collect", "memory": self.memory}
        action = self.orchestrator.execute_graph("plan", self, context)
        self.memory.append(action)
        print(f"[Cognitive {self.unique_id}] {action}")
