from models import HybridModel, llm_collect, wait_tool
from orchestrator import Orchestrator

# Set up the orchestrator and graph
orchestrator = Orchestrator()
orchestrator.add_node("plan", llm_collect)
orchestrator.add_node("wait", wait_tool)

# Add conditional edge: alternate between collecting and waiting
orchestrator.add_conditional_edges(
    "plan", lambda state: "wait" if len(state["memory"]) % 2 == 0 else None
)

# Run the model
model = HybridModel(num_agents=5, orchestrator=orchestrator)
for step in range(3):
    print(f"\n--- Step {step + 1} ---")
    model.step()
