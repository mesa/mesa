"""Prompt templates for code generation tasks."""

BASIC_SIMULATION_TEMPLATE = """Generate a complete Mesa simulation based on this description:

USER REQUEST: {user_prompt}

REQUIREMENTS:
1. Create a complete, runnable Python script
2. Include all necessary imports
3. Follow this structure:
   - Imports
   - Scenario class (if needed for parameters)
   - Agent class(es) with step() method
   - Model class with __init__() and step() methods
   - Example usage code

4. Use appropriate Mesa components:
   - mesa.Agent or mesa.discrete_space.CellAgent
   - mesa.Model
   - Appropriate grid type (OrthogonalMooreGrid, OrthogonalVonNeumannGrid, etc.)
   - mesa.DataCollector for data collection

5. Include comprehensive docstrings and comments
6. Add type hints throughout
7. Handle edge cases and errors appropriately

EXAMPLE STRUCTURE:
```python
import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.datacollection import DataCollector
from mesa.experimental.scenarios import Scenario

class MyScenario(Scenario):
    \"\"\"Scenario parameters.\"\"\"
    # Define parameters here

class MyAgent(mesa.discrete_space.CellAgent):
    \"\"\"Agent description.\"\"\"
    
    def __init__(self, model, cell=None):
        super().__init__(model)
        self.cell = cell
        # Initialize agent attributes
    
    def step(self):
        \"\"\"Agent behavior per step.\"\"\"
        # Implement agent logic

class MyModel(mesa.Model):
    \"\"\"Model description.\"\"\"
    
    def __init__(self, scenario: MyScenario = MyScenario):
        super().__init__(scenario=scenario)
        # Initialize model components
        
    def step(self):
        \"\"\"Model step.\"\"\"
        # Activate agents and collect data

# Example usage
if __name__ == "__main__":
    model = MyModel()
    model.run_for(100)
```

Generate the complete simulation code now:"""

PREDATOR_PREY_TEMPLATE = """Generate a predator-prey simulation with these specifications:

USER REQUEST: {user_prompt}

SPECIFIC REQUIREMENTS:
1. Two agent types: Predator and Prey
2. Spatial movement on a grid
3. Predators hunt prey for energy
4. Prey reproduce when conditions are met
5. Energy-based survival system
6. Data collection for population tracking

AGENT BEHAVIORS:
- Prey: Move randomly, reproduce if energy sufficient, get eaten by predators
- Predators: Hunt prey, gain energy from eating, reproduce if energy sufficient, die if energy depleted

PARAMETERS TO INCLUDE:
- Grid size
- Initial populations
- Reproduction rates
- Energy values
- Movement patterns

Generate the complete simulation following Mesa best practices."""

SOCIAL_DYNAMICS_TEMPLATE = """Generate a social dynamics simulation with these specifications:

USER REQUEST: {user_prompt}

SPECIFIC REQUIREMENTS:
1. Agents with social attributes (opinions, influence, etc.)
2. Network-based or spatial interactions
3. Opinion dynamics or social influence mechanisms
4. Data collection for tracking social metrics

COMMON PATTERNS:
- Opinion formation and polarization
- Social influence and conformity
- Network effects on behavior
- Cultural transmission
- Segregation dynamics

Generate the complete simulation with appropriate social mechanisms."""

ECONOMIC_MODEL_TEMPLATE = """Generate an economic simulation with these specifications:

USER REQUEST: {user_prompt}

SPECIFIC REQUIREMENTS:
1. Economic agents (consumers, producers, traders, etc.)
2. Economic mechanisms (markets, trading, resource allocation)
3. Wealth/resource tracking
4. Economic indicators and data collection

COMMON PATTERNS:
- Market dynamics
- Wealth distribution
- Trading mechanisms
- Resource competition
- Economic cycles

Generate the complete simulation with realistic economic behaviors."""

CUSTOM_SIMULATION_TEMPLATE = """Generate a custom Mesa simulation based on this description:

USER REQUEST: {user_prompt}

ANALYSIS:
Based on the request, I need to determine:
1. What type of agents are needed?
2. What environment/space is appropriate?
3. What behaviors and interactions should be modeled?
4. What data should be collected?
5. What parameters should be configurable?

APPROACH:
1. Identify the core simulation elements from the description
2. Choose appropriate Mesa components
3. Design agent behaviors and interactions
4. Implement data collection for relevant metrics
5. Create a complete, runnable simulation

Generate the complete simulation code following Mesa best practices."""