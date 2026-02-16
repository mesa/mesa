## Overview of the MESA library

Mesa is modular, meaning that its modeling, analysis and visualization components are kept separate but intended to work together. The modules are grouped into three categories:

1. **Modeling:** Classes used to build the models themselves: a model and agent classes, space for them to move around in, and built-in functionality for managing agents.
2. **Analysis:** Tools to collect data generated from your model, or to run it multiple times with different parameter values.
3. **Visualization:** Classes to create and launch an interactive model visualization, using a browser-based interface.

### Modeling modules

Most models consist of one class to represent the model itself and one or more classes for agents. Mesa provides built-in functionality for managing agents and their interactions. These are implemented in Mesa's modeling modules:

- [mesa.model](apis/model)
- [mesa.agent](apis/agent)
- [mesa.agentset](apis/agentset)
- [mesa.discrete_space](apis/discrete_space)

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
        self.grid = mesa.discrete_space.OrthogonalMooreGrid((10, 10), torus=True)
        initial_ages = self.rng.integers(0, 80, size=n_agents)
        agents = MyAgent.create_agents(self, n_agents, initial_ages)
        for agent in agents:
            agent.cell = self.grid.all_cells.select_random_cell()

    def step(self):
        self.agents.shuffle_do("step")
```

You can instantiate a model and run it for one step with:

```python
model = MyModel(n_agents=5)
model.run_for(1)
```

### Spaces in Mesa

Mesa provides several types of spaces where agents can exist and interact:

#### Discrete Spaces
Mesa implements discrete spaces through the `mesa.discrete_space` module, using a doubly-linked structure where each cell maintains connections to its neighbors. Available variants include:

1. **Grid-based Spaces:**
   ```python
   # Create a Von Neumann grid (4 neighbors per cell)
   grid = mesa.discrete_space.OrthogonalVonNeumannGrid((width, height), torus=False)

   # Create a Moore grid (8 neighbors per cell)
   grid = mesa.discrete_space.OrthogonalMooreGrid((width, height), torus=True)

   # Create a hexagonal grid
   grid = mesa.discrete_space.HexGrid((width, height), torus=False)
   ```

2. **Network Space:**
   ```python
   # Create a network-based space
   network = mesa.discrete_space.Network(graph)
   ```

3. **Voronoi Space:**
   ```python
   # Create an irregular tessellation
   mesh = mesa.discrete_space.VoronoiMesh(points)
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

> **Note:** The legacy `mesa.space` module (including `MultiGrid`, `SingleGrid`, etc.) is in maintenance-only mode. For new projects, use `mesa.discrete_space` and `mesa.experimental.continuous_space` instead.

### Time Advancement and Agent Activation

Mesa supports multiple approaches to advancing time and activating agents.

#### Running the model
Use the time advancement methods on `Model` to run your simulation:

```python
model = MyModel()
model.run_for(100)     # Run for 100 time units
model.run_until(50.0)  # Run until absolute time 50
```
By default, the model's `step()` method is scheduled to run every 1.0 time units, so `model.run_for(10)` executes 10 steps.

#### Agent Activation Patterns
Mesa provides flexible agent activation through the AgentSet API:

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

#### Event Scheduling
Mesa supports event-based time progression directly on the `Model`:

```python
# Schedule one-off events
model.schedule_event(some_function, at=25.0)     # At absolute time
model.schedule_event(some_function, after=5.0)   # Relative to now

# Cancel a scheduled event
event = model.schedule_event(callback, at=100.0)
event.cancel()

# Schedule recurring events
from mesa.time import Schedule

model.schedule_recurring(func, Schedule(interval=10))            # Every 10 time units
model.schedule_recurring(func, Schedule(interval=10, start=0))   # Starting immediately
model.schedule_recurring(func, Schedule(interval=1.0, count=10)) # Limited to 10 executions

# Stop a recurring event
gen = model.schedule_recurring(func, Schedule(interval=5.0))
gen.stop()

# Advance time, processing all scheduled events along the way
model.run_for(50)
model.run_until(100.0)
```

This enables pure event-driven models, hybrid approaches combining agent-based steps with scheduled events, and everything in between.

### AgentSet and model.agents
`model.agents` and the AgentSet class are central in managing and activating agents.

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

These are just some examples of using the AgentSet, there are many more possibilities, see the [AgentSet API docs](apis/agentset).

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
model.run_for(10)
model_df = model.datacollector.get_model_vars_dataframe()
agent_df = model.datacollector.get_agent_vars_dataframe()
```

To batch-run the model while varying, for example, the n_agents parameter, you'd use the [`batch_run`](apis/batchrunner) function:

```python
import mesa
import numpy as np
import sys

rng = np.random.default_rng(42)
rng_values = rng.integers(0, sys.maxsize, size=(5,))

parameters = {"n_agents": range(1, 6)}
results = mesa.batch_run(
    MyModel,
    parameters,
    rng=rng_values.tolist(),
    max_steps=100,
    data_collection_period=1,
    number_processes=1  # Change to use multiple CPU cores for parallel execution
)
```

The results are returned as a list of dictionaries, which can be easily converted to a pandas DataFrame for further analysis.

### Visualization
Mesa uses a browser-based visualization system called SolaraViz. This allows for interactive, customizable visualizations of your models.

Note: SolaraViz is in active development in Mesa 3.x. While we attempt to minimize them, there might be API breaking changes in minor releases.
> **Note:** SolaraViz instantiates new models using `**model_parameters.value`, so all model inputs must be keyword arguments.

Ensure your model's `__init__` method accepts keyword arguments matching the `model_params` keys.

```python
class MyModel(Model):
    def __init__(self, n_agents=10):
        super().__init__()
        # Initialize the model with N agents
```
The core functionality for building your own visualizations resides in the [`mesa.visualization`](apis/visualization) namespace.

Here's a basic example of how to set up a visualization:

```python
from mesa.visualization import SolaraViz, make_space_component, make_plot_component


def agent_portrayal(agent):
    return AgentPortrayalStyle(color="blue", size=50)


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
AgentSet <tutorials/1_agentset>
Agent Activation <tutorials/2_agent_activation>
Event Scheduling <tutorials/3_event_scheduling>
Adding Space <tutorials/4_adding_space>
Collecting Data <tutorials/5_collecting_data>
Basic Visualization <tutorials/6_visualization_basic>
Dynamic Agent Visualization <tutorials/7_visualization_dynamic_agents>
Visualisation using SpaceRenderer <tutorials/8_visualization_rendering_with_space_renderer>
Property Layer Visualization <tutorials/9_visualization_propertylayer_visualization>
Custom Visualization Components <tutorials/10_visualization_custom>
Parameter Sweeps <tutorials/11_batch_run>
Comparing Scenarios <tutorials/12_comparing_scenarios>
Best Practices <best-practices>


```