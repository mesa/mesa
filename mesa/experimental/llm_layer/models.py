from agents import BasicAgent, CognitiveAgent

from mesa import Model


class HybridModel(Model):
    def __init__(self, num_agents, orchestrator):
        super().__init__()
        self.num_agents = num_agents
        self.orchestrator = orchestrator

        # Create 3 basic agents and 2 cognitive agents
        BasicAgent.create_agents(model=self, n=3)
        CognitiveAgent.create_agents(model=self, n=2, orchestrator=orchestrator)

    def step(self):
        self.agents.shuffle_do("step")


def llm_collect(agent, state):
    return f"[LLM] Reasoned to collect based on memory length {len(state['memory'])}"


def wait_tool(agent, state):
    return "[RULE] Wait due to uncertainty or cooldown"
