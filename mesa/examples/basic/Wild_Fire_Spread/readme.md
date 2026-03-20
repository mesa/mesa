# Forest Fire Spread Model

## Overview
This example presents a wildfire spread simulation developed with Mesa using an agent-based modeling framework. Each grid cell represents a fuel unit, and wildfire propagation emerges from local interaction among neighboring cells under environmental influence.

The model is designed as a compact academic study of how fire behavior can be represented through agents, state transitions, and probabilistic spread dynamics. It provides a clear conceptual foundation for wildfire simulation in computational modeling contexts.

## Objectives
- Simulate wildfire propagation on a two-dimensional grid.
- Represent fuel behavior through agent-level attributes.
- Include environmental effects such as humidity, wind, and slope.
- Track the changing condition of the forest during the simulation.
- Provide a foundation for further wildfire modeling work.

## Model Components
### ForestFireModel
The model defines the simulation environment, initializes the grid, assigns environmental conditions, ignites an initial set of burning cells, and advances the system through discrete time steps.

### FuelAgent
Each grid cell is modeled as a fuel agent. A fuel agent can exist in one of three states:
- HEALTHY
- BURNING
- BURNED

Each agent stores local properties related to fuel amount, flammability, moisture, slope, and burn duration.

## Agent Attributes
- State: HEALTHY, BURNING, or BURNED
- Fuel type: tree or grass
- Flammability: depends on fuel type
- Fuel: available combustible material
- Moisture: derived from humidity
- Slope: local terrain factor
- Burn time: number of steps the agent remains burning
- Firebreak status: whether ignition is blocked

## Fire Spread Logic
At each simulation step, burning agents attempt to ignite neighboring healthy agents. The probability of spread depends on fuel condition, moisture, wind influence, slope, and directional effect. Once ignited, a neighboring agent enters the burning state for a limited duration. Burning agents consume fuel over time and eventually transition to the burned state.

## Mathematical Formulation
The moisture level is defined as:

```bash
moisture = humidity / 100
```

The fire intensity is defined as:

```bash
intensity = fuel * flammability * (1 - moisture)
```

The effective wind term is defined as:

```bash
effective_wind = wind_speed + 0.2 * burning_neighbors_ratio
```

The directional factor is represented as:

```bash
same direction = 2.0
side direction = 1.0
opposite direction = 0.5
```

The spread probability is computed as:

```bash
P_spread = P_BASE * (1 + effective_wind * direction_factor) * (1 + slope) * intensity
```

The spread probability is bounded by:

```bash
P_spread <= 1
```

The fuel consumption rule is:

```bash
fuel(t + 1) = fuel(t) - 0.1
```

The burn-time update rule is:

```bash
burn_time(t + 1) = burn_time(t) - 1
```

The state-transition condition is:

```bash
if fuel <= 0 or burn_time <= 0, then state = BURNED
```

## Data Collection
The model records only the number of agents in each state over time:
- Healthy agents
- Burning agents
- Burned agents

## Visualization
The simulation is visualized on a grid in which color indicates the condition of each cell:
- Green for healthy fuel
- Yellow or orange for burning fuel
- Red for burned fuel

## Project Structure
The example is organized into separate files for the model definition, agent behavior, application interface, and documentation.

## Installation
Install the required dependencies:

```bash
pip install mesa solara
```

## Run The Model
Run the interactive application with:

```bash
solara run mesa/examples/basic/Wild_Fire_Spread/app.py
```

Then open the local address provided by Solara in the browser.

