"""Basic usage examples for Mesa LLM Assistant."""

import asyncio

from mesa_llm import MesaCodeGenerator, MesaDebugger, MesaExplainer, MesaOptimizer
from mesa_llm.utils import LLMProvider


async def example_generate_simulation():
    """Example: Generate a simulation from natural language."""
    print("=== Generating Simulation ===")

    generator = MesaCodeGenerator(LLMProvider.OPENAI)

    prompt = "Create a predator-prey model with 50 wolves and 100 sheep on a 20x20 grid"

    result = await generator.generate_simulation(prompt)

    if result["validation"]["is_valid"]:
        print("✅ Generated valid Mesa simulation!")
        print(f"Code length: {len(result['code'])} characters")
        print(f"Classes found: {result['metadata']['classes']}")
    else:
        print("❌ Generated code has issues:")
        for error in result["validation"]["errors"]:
            print(f"  - {error}")


async def example_debug_code():
    """Example: Debug Mesa simulation code."""
    print("\n=== Debugging Code ===")

    # Example code with issues
    buggy_code = """
import mesa

class MyAgent(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.energy = 10

    # Missing step method!

class MyModel(mesa.Model):
    def __init__(self):
        super().__init__()
        # Missing agent creation and grid setup

    def step(self):
        pass  # Empty step method
"""

    debugger = MesaDebugger(LLMProvider.OPENAI)

    result = await debugger.debug_code(buggy_code)

    print(f"Found {result['summary']['total_issues']} issues")
    print(f"Critical issues: {result['summary']['critical_issues']}")

    for issue in result["static_analysis"]["mesa_issues"]:
        print(f"  - {issue['message']}")


async def example_explain_simulation():
    """Example: Explain a Mesa simulation."""
    print("\n=== Explaining Simulation ===")

    # Example Schelling model code
    schelling_code = """
import mesa
from mesa.discrete_space import OrthogonalMooreGrid

class SchellingAgent(mesa.discrete_space.CellAgent):
    def __init__(self, model, cell, agent_type, homophily=0.3):
        super().__init__(model)
        self.cell = cell
        self.type = agent_type
        self.homophily = homophily
        self.happy = False

    def step(self):
        neighbors = list(self.cell.get_neighborhood().agents)
        if neighbors:
            similar = len([n for n in neighbors if n.type == self.type])
            similarity = similar / len(neighbors)
            self.happy = similarity >= self.homophily

        if not self.happy:
            empty_cells = [c for c in self.model.grid.all_cells if c.is_empty]
            if empty_cells:
                new_cell = self.random.choice(empty_cells)
                self.move_to(new_cell)

class SchellingModel(mesa.Model):
    def __init__(self, width=20, height=20, density=0.8, minority_pc=0.5):
        super().__init__()
        self.grid = OrthogonalMooreGrid((width, height), capacity=1)

        for cell in self.grid.all_cells:
            if self.random.random() < density:
                agent_type = 1 if self.random.random() < minority_pc else 0
                SchellingAgent(self, cell, agent_type)

    def step(self):
        self.agents.shuffle_do("step")
"""

    explainer = MesaExplainer(LLMProvider.OPENAI)

    result = await explainer.explain_simulation(
        schelling_code, audience_level="beginner"
    )

    print("📖 Simulation Explanation:")
    print(result["explanation"]["overview"])


async def example_optimize_code():
    """Example: Optimize Mesa simulation code."""
    print("\n=== Optimizing Code ===")

    # Example inefficient code
    inefficient_code = """
import mesa

class SlowAgent(mesa.Agent):
    def step(self):
        # Inefficient: nested loops
        for agent in self.model.agents:
            for other in self.model.agents:
                if agent != other:
                    # Expensive calculation in nested loop
                    distance = ((agent.pos[0] - other.pos[0])**2 +
                               (agent.pos[1] - other.pos[1])**2)**0.5

class SlowModel(mesa.Model):
    def __init__(self, n_agents=100):
        super().__init__()
        for i in range(n_agents):
            SlowAgent(self)

    def step(self):
        # Inefficient: manual iteration
        for agent in self.agents:
            agent.step()
"""

    optimizer = MesaOptimizer(LLMProvider.OPENAI)

    result = await optimizer.optimize_simulation(inefficient_code)

    print(
        f"Optimization Score: {result['optimization_report']['optimization_score']}/100"
    )
    print(
        f"Found {result['optimization_report']['total_opportunities']} optimization opportunities"
    )

    for opp in result["optimization_report"]["top_recommendations"]:
        print(f"  - {opp['issue']} (Priority: {opp['priority']})")


async def main():
    """Run all examples."""
    try:
        await example_generate_simulation()
        await example_debug_code()
        await example_explain_simulation()
        await example_optimize_code()

        print("\n✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Example failed: {e!s}")


if __name__ == "__main__":
    asyncio.run(main())
