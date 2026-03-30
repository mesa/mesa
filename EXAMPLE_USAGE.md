# Example Usage Guide

This guide demonstrates how to use the Mesa LLM Assistant to create, debug, and optimize agent-based simulations.

## Quick Start

### 1. Setup

```bash
# Clone and install
git clone <repository-url>
cd mesa-llm-assistant
make dev-setup

# Configure API keys in .env file
cp mesa_llm/.env.example .env
# Edit .env with your OpenAI/Gemini API keys

# Start the server
make run-dev
```

### 2. Basic API Usage

```python
import requests

# Generate a simulation
response = requests.post("http://localhost:8000/api/v1/generate", json={
    "prompt": "Create a predator-prey model with wolves and sheep",
    "llm_provider": "openai"
})

if response.json()["success"]:
    generated_code = response.json()["code"]
    print("Generated simulation code!")
```

## Complete Workflow Example

### Step 1: Generate Simulation

**Prompt:** "Create a social segregation model like Schelling's model with 200 agents on a 20x20 grid"

**Generated Code:**
```python
import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.datacollection import DataCollector
from mesa.experimental.scenarios import Scenario

class SchellingScenario(Scenario):
    width: int = 20
    height: int = 20
    density: float = 0.8
    minority_pc: float = 0.5
    homophily: float = 0.3

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
    def __init__(self, scenario: SchellingScenario = SchellingScenario):
        super().__init__(scenario=scenario)
        self.grid = OrthogonalMooreGrid(
            (scenario.width, scenario.height),
            capacity=1,
            random=self.random
        )

        self.datacollector = DataCollector(
            model_reporters={
                "Happy": lambda m: len([a for a in m.agents if a.happy]),
                "Unhappy": lambda m: len([a for a in m.agents if not a.happy])
            }
        )

        for cell in self.grid.all_cells:
            if self.random.random() < scenario.density:
                agent_type = 1 if self.random.random() < scenario.minority_pc else 0
                SchellingAgent(self, cell, agent_type, scenario.homophily)

        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

# Example usage
if __name__ == "__main__":
    model = SchellingModel()
    model.run_for(100)
    print(f"Final happiness: {len([a for a in model.agents if a.happy])}/{len(model.agents)}")
```

### Step 2: Explain the Simulation

**API Call:**
```python
response = requests.post("http://localhost:8000/api/v1/explain", json={
    "code": generated_code,
    "audience_level": "beginner"
})
```

**Generated Explanation:**
> **Overview:** This simulation models residential segregation patterns, inspired by Thomas Schelling's famous model. It shows how individual preferences for similar neighbors can lead to large-scale segregation patterns, even when individuals have only mild preferences.
>
> **Agents:** Each agent represents a household that can be one of two types (think of different ethnic groups, income levels, or any social category). Agents have a "homophily" preference - they want at least 30% of their neighbors to be similar to them.
>
> **Environment:** The world is a 20x20 grid where each cell can hold one household. Initially, households are randomly distributed across the grid.
>
> **Behavior:** In each time step, unhappy agents (those with too few similar neighbors) move to random empty locations. This continues until most agents find satisfactory neighborhoods.
>
> **Emergent Pattern:** Even with mild individual preferences, the simulation typically produces highly segregated neighborhoods, demonstrating how individual choices can create unexpected collective outcomes.

### Step 3: Debug Issues

If there were issues with the code:

```python
response = requests.post("http://localhost:8000/api/v1/debug", json={
    "code": generated_code,
    "error_message": "AttributeError: 'NoneType' object has no attribute 'agents'"
})
```

**Debug Analysis:**
- **Static Analysis:** Found 0 syntax errors, 1 Mesa pattern issue
- **Issue:** Potential null reference when accessing cell neighborhoods
- **Suggestion:** Add null checks before accessing cell.agents
- **Fixed Code:** [Provides corrected version with proper error handling]

### Step 4: Optimize Performance

```python
response = requests.post("http://localhost:8000/api/v1/optimize", json={
    "code": generated_code,
    "focus_areas": ["performance", "memory"]
})
```

**Optimization Report:**
- **Score:** 78/100
- **Key Issues:**
  1. Inefficient neighbor counting in agent.step()
  2. Repeated empty cell searches
  3. Missing AgentSet optimizations
- **Optimized Code:** [Provides version using AgentSet operations and cached spatial queries]

### Step 5: Execute and Test

```python
response = requests.post("http://localhost:8000/api/v1/execute", json={
    "code": optimized_code,
    "steps": 50,
    "collect_data": True
})
```

**Execution Results:**
- **Success:** True
- **Steps Completed:** 50
- **Final State:** 89% of agents happy
- **Data:** Time series of happiness levels showing convergence

## Advanced Examples

### 1. Predator-Prey Ecosystem

**Prompt:** "Create a wolf-sheep ecosystem with energy-based survival, reproduction, and grass regrowth"

**Key Features Generated:**
- Energy-based agent lifecycle
- Predator-prey interactions
- Environmental resource dynamics
- Population tracking and analysis

### 2. Economic Market Simulation

**Prompt:** "Model a simple market with buyers, sellers, and price discovery mechanism"

**Key Features Generated:**
- Economic agents with different strategies
- Supply and demand dynamics
- Price formation mechanisms
- Market efficiency metrics

### 3. Disease Spread Model

**Prompt:** "Create an epidemiological model with susceptible, infected, and recovered states"

**Key Features Generated:**
- SEIR disease dynamics
- Contact networks
- Intervention strategies
- Public health metrics

## Integration Examples

### 1. Jupyter Notebook Integration

```python
# In Jupyter notebook
from mesa_llm import MesaCodeGenerator
import asyncio

async def generate_and_run():
    generator = MesaCodeGenerator()
    result = await generator.generate_simulation(
        "Create a flocking model with boids"
    )

    if result["validation"]["is_valid"]:
        # Execute the generated code
        exec(result["code"])

        # Run simulation
        model = BoidsModel()  # Generated class
        model.run_for(100)

        # Visualize results
        import matplotlib.pyplot as plt
        data = model.datacollector.get_model_vars_dataframe()
        data.plot()
        plt.show()

# Run in notebook
await generate_and_run()
```

### 2. Educational Platform Integration

```python
class MesaLearningPlatform:
    def __init__(self):
        self.generator = MesaCodeGenerator()
        self.explainer = MesaExplainer()

    async def create_lesson(self, topic, difficulty):
        # Generate example simulation
        prompt = f"Create a {difficulty} level {topic} simulation for education"
        code_result = await self.generator.generate_simulation(prompt)

        # Generate explanation
        explanation_result = await self.explainer.explain_simulation(
            code_result["code"],
            audience_level=difficulty
        )

        return {
            "code": code_result["code"],
            "explanation": explanation_result["explanation"],
            "exercises": self.generate_exercises(code_result["code"])
        }
```

### 3. Research Workflow Integration

```python
class ResearchWorkflow:
    def __init__(self):
        self.generator = MesaCodeGenerator()
        self.optimizer = MesaOptimizer()
        self.debugger = MesaDebugger()

    async def prototype_model(self, research_question):
        # Generate initial prototype
        code_result = await self.generator.generate_simulation(research_question)

        # Optimize for research use
        opt_result = await self.optimizer.optimize_simulation(
            code_result["code"],
            focus_areas=["performance", "scalability"]
        )

        # Validate and debug
        debug_result = await self.debugger.debug_code(
            opt_result["optimized_code"]
        )

        return {
            "prototype_code": code_result["code"],
            "optimized_code": opt_result["optimized_code"],
            "validation_report": debug_result["summary"],
            "research_notes": self.extract_research_insights(opt_result)
        }
```

## Best Practices

### 1. Prompt Engineering
- Be specific about agent behaviors and interactions
- Specify grid size, population, and key parameters
- Mention data collection requirements
- Include any special constraints or rules

### 2. Code Validation
- Always validate generated code before execution
- Review Mesa patterns and best practices
- Test with small populations first
- Check for proper error handling

### 3. Performance Optimization
- Use AgentSet operations instead of manual loops
- Implement efficient spatial queries
- Consider memory usage for large simulations
- Profile performance for bottlenecks

### 4. Educational Use
- Start with simple models and add complexity
- Use explanations to understand generated code
- Modify parameters to explore behavior
- Compare different modeling approaches

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure API keys are set in .env file
   - Check key validity and quotas

2. **Code Generation Issues**
   - Try more specific prompts
   - Specify simulation type explicitly
   - Review validation errors

3. **Execution Failures**
   - Check for missing imports
   - Verify Mesa version compatibility
   - Review resource limits

4. **Performance Problems**
   - Use optimization features
   - Reduce population size for testing
   - Check for infinite loops

### Getting Help

- Check the API documentation at `/docs`
- Review example code in `examples/`
- Use the debug endpoint for code issues
- Check logs for detailed error information