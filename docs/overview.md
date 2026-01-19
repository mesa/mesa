## Overview of the MESA library

Mesa is modular, meaning that its modeling, analysis and visualization components are kept separate but intended to work together. The modules are grouped into three categories:

1. **Modeling:** Classes used to build the models themselves: a model and agent classes, space for them to move around in, and built-in functionality for managing agents.
2. **Analysis:** Tools to collect data generated from your model, or to run it multiple times with different parameter values.
3. **Visualization:** Classes to create and launch an interactive model visualization, using a browser-based interface.

### Modeling modules

Most models consist of one class to represent the model itself and one or more classes for agents. Mesa provides built-in functionality for managing agents and their interactions. These are implemented in Mesa's modeling modules:

- [mesa.model](apis/model)
- [mesa.agent](apis/agent)
- [mesa.space](apis/space)

The skeleton of a model might look like this:

```python
import mesa

class MyAgent(mesa.Agent):
    def __init__(self, model, age):
        super().__init__(model)
        self.age = age

    def step(self):
        self.age += 1
        print(f"Agent {self.unique_id} now is {self.age} years old")
        # Whatever else the agent does when activated

class MyModel(mesa.Model):
    def __init__(self, n_agents):
        super().__init__()
        self.grid = mesa.space.MultiGrid(10, 10, torus=True)
        for _ in range(n_agents):
            initial_age = self.random.randint(0, 80)
            a = MyAgent(self, initial_age)
            coords = (self.random.randrange(0, 10), self.random.randrange(0, 10))
            self.grid.place_agent(a, coords)

    def step(self):
        self.agents.shuffle_do("step")
```

### Spaces in Mesa

Mesa provides several types of spaces where agents can exist and interact:

#### Discrete Spaces
Mesa implements discrete spaces using a doubly-linked structure where each cell maintains connections to its neighbors. Available variants include:

1. **Grid-based Spaces:**
   ```python
   # Create a Von Neumann grid (4 neighbors per cell)
   grid = mesa.space.OrthogonalVonNeumannGrid((width, height), torus=False)

   # Create a Moore grid (8 neighbors per cell)
   grid = mesa.space.OrthogonalMooreGrid((width, height), torus=True)

   # Create a hexagonal grid
   grid = mesa.space.HexGrid((width, height), torus=False)
   ```

2. **Network Space:**
   ```python
   # Create a network-based space
   network = mesa.space.NetworkGrid(network)
   ```

3. **Voronoi Space:**
   ```python
   # Create an irregular tessellation
   mesh = mesa.space.VoronoiMesh(points)
   ```

#### Property Layers
Discrete spaces support PropertyLayers - efficient numpy-based arrays for storing cell-level properties:

```python
# Create and use a property layer
grid.create_property_layer("elevation", default_value=10)
high_ground = grid.elevation.select_cells(lambda x: x > 50)
```

#### Continuous Space
For models requiring continuous movement:

```python
# Create a continuous space
space = mesa.space.ContinuousSpace(x_max, y_max, torus=True)

# Move an agent to specific coordinates
space.move_agent(agent, (new_x, new_y))
```

### Time Advancement and Agent Activation

Mesa supports multiple approaches to advancing time and activating agents:

#### Basic Time Steps
The simplest approach runs the model for a specified number of steps:

```python
model = MyModel(seed=42)
for _ in range(100):
    model.step()
```

#### Agent Activation Patterns
Mesa 3.0 provides flexible agent activation through the AgentSet API:

```python
# Sequential activation
model.agents.do("step")

# Random activation
model.agents.shuffle_do("step")

# Multi-stage activation
for stage in ["move", "eat", "reproduce"]:
    model.agents.do(stage)

# Activation by agent type
for klass in model.agent_types:
    model.agents_by_type[klass].do("step")
```

#### Event-Based Scheduling
Mesa also supports event-based time progression (experimental):

```python
# Pure event-based
simulator = mesa.experimental.DiscreteEventSimulator()
model = MyModel(seed=42, simulator=simulator)
simulator.schedule_event_relative(some_function, 3.1415)

# Hybrid time-step and event scheduling
model = MyModel(seed=42, simulator=mesa.experimental.ABMSimulator())
model.simulator.schedule_event_next_tick(some_function)
```

### AgentSet and model.agents
Mesa 3.0 makes `model.agents` and the AgentSet class central in managing and activating agents.

#### model.agents
`model.agents` is an AgentSet containing all agents in the model. It's automatically updated when agents are added or removed:

```python
# Get total number of agents
num_agents = len(model.agents)

# Iterate over all agents
for agent in model.agents:
    print(agent.unique_id)
```

#### AgentSet Functionality
AgentSet offers several methods for efficient agent management:

1. **Selecting**: Filter agents based on criteria.
   ```python
   high_energy_agents = model.agents.select(lambda a: a.energy > 50)
   ```
2. **Shuffling and Sorting**: Randomize or order agents.
   ```python
   shuffled_agents = model.agents.shuffle()
   sorted_agents = model.agents.sort(key="energy", ascending=False)
   ```
3. **Applying methods**: Execute methods on all agents.
   ```python
   model.agents.do("step")
   model.agents.shuffle_do("move")  # Shuffle then apply method
   ```
4. **Aggregating**: Compute aggregate values across agents.
   ```python
   avg_energy = model.agents.agg("energy", func=np.mean)
   ```
5. **Grouping**: Group agents by attributes.
   ```python
   grouped_agents = model.agents.groupby("species")

   for _, agent_group in grouped_agents:
      agent_group.shuffle_do()
   species_counts = grouped_agents.count()
   mean_age_by_group = grouped_agents.agg("age", np.mean)
   ```
`model.agents` can also be accessed within a model instance using `self.agents`.

These are just some examples of using the AgentSet, there are many more possibilities, see the [AgentSet API docs](apis/agent).

### Analysis modules

If you're using modeling for research, you'll want a way to collect the data each model run generates. You'll probably also want to run the model multiple times, to see how some output changes with different parameters. Data collection and batch running are implemented in the appropriately-named analysis modules:

- [mesa.datacollection](apis/datacollection)
- [mesa.batchrunner](apis/batchrunner)

You'd add a data collector to the model like this:

```python
import mesa
import numpy as np

# ...

class MyModel(mesa.Model):
    def __init__(self, n_agents):
        super().__init__()
        # ... (model initialization code)
        self.datacollector = mesa.DataCollector(
            model_reporters={"mean_age": lambda m: m.agents.agg("age", np.mean)},
            agent_reporters={"age": "age"}
        )

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
```

The data collector will collect the specified model- and agent-level data at each step of the model. After you're done running it, you can extract the data as a [pandas](http://pandas.pydata.org/) DataFrame:

```python
model = MyModel(5)
for t in range(10):
    model.step()
model_df = model.datacollector.get_model_vars_dataframe()
agent_df = model.datacollector.get_agent_vars_dataframe()
```

To batch-run the model while varying, for example, the n_agents parameter, you'd use the [`batch_run`](apis/batchrunner) function:

```python
import mesa

parameters = {"n_agents": range(1, 6)}
results = mesa.batch_run(
    MyModel,
    parameters,
    iterations=5,
    max_steps=100,
    data_collection_period=1,
    number_processes=1  # Change to use multiple CPU cores for parallel execution
)
```

The results are returned as a list of dictionaries, which can be easily converted to a pandas DataFrame for further analysis.

### Visualization
Mesa now uses a new browser-based visualization system called SolaraViz. This allows for interactive, customizable visualizations of your models.

Note: SolaraViz is experimental and still in active development in Mesa 3.x. While we attempt to minimize them, there might be API breaking changes in minor releases.
> **Note:** SolaraViz instantiates new models using `**model_parameters.value`, so all model inputs must be keyword arguments.

Ensure your model's `__init__` method accepts keyword arguments matching the `model_params` keys.

```python
class MyModel(Model):
    def __init__(self, n_agents=10, seed=None):
        super().__init__(seed=seed)
        # Initialize the model with N agents
```
The core functionality for building your own visualizations resides in the [`mesa.visualization`](apis/visualization) namespace.

Here's a basic example of how to set up a visualization:

```python
from mesa.visualization import SolaraViz, make_space_component, make_plot_component


def agent_portrayal(agent):
    return {"color": "blue", "size": 50}


model_params = {
    "n_agents": Slider(
        label="Number of agents:",
        value=50,
        min=1,
        max=100,
        step=1
    )
}

page = SolaraViz(
    MyModel(n_agents=42),
    [
        make_space_component(agent_portrayal),
        make_plot_component("mean_age")
    ],
    model_params=model_params
)
page
```
This will create an interactive visualization of your model, including:

- A grid visualization of agents
- A plot of a model metric over time
- A slider to adjust the number of agents

```{toctree}
:hidden: true
:maxdepth: 7

Overview <overview>
Creating Your First Model <tutorials/0_first_model>
Adding Space <tutorials/1_adding_space>
Collecting Data <tutorials/2_collecting_data>
AgentSet <tutorials/3_agentset>
Basic Visualization <tutorials/4_visualization_basic>
Dynamic Agent Visualization <tutorials/5_visualization_dynamic_agents>
Visualisation using SpaceRenderer <tutorials/6_visualization_rendering_with_space_renderer>
Property Layer Visualization <tutorials/7_visualization_propertylayer_visualization>
Custom Visualization Components <tutorials/8_visualization_custom>
Parameter Sweeps <tutorials/9_batch_run>
Comparing Scenarios <tutorials/10_comparing_scenarios>
Discrete Event Simulation <tutorials/11_discrete_event_simulation>
Hybrid ABM-DEVS Models <tutorials/12_hybrid_abm_devs>
Best Practices <best-practices>


```