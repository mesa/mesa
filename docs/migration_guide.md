# Mesa Migration guide
This guide contains breaking changes between major Mesa versions and how to resolve them.

Non-breaking changes aren't included, for those see our [Release history](https://github.com/mesa/mesa/releases).

## Mesa 3.4.0

### batch run
`batch_run` has been updated to offer explicit control over the random seeds that are used to run multiple replications of a given experiment. For this a new keyword argument, `rng` has been added and `iterations` will issue a `DeprecationWarning`. The new `rng` keyword argument takes a valid value for seeding or a list of valid values. If you want to run multiple iterations/replications of a given experiment, you need to pass the required seeds explicitly.

Below is a simple example of the new recommended usage of `batch_run`. Note how we first create 5 random integers which we then use as seed values for the new `rng` keyword argument.

```python
import numpy as np
import sys

# let's create 5 random integers
rng = np.random.default_rng(42)
rng_values = rng.integers(0, sys.maxsize, size=(5,))

results = mesa.batch_run(
    MoneyModel,
    parameters=params,
    rng=rng_values.tolist(), # we pass the 5 seed values to rng
    max_steps=100,
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
)
```

## Mesa 3.3.0

Mesa 3.3.0 is a visualization upgrade introducing a new and improved API, full support for both `altair` and `matplotlib` backends, and resolving several recurring issues from previous versions.
For full details on how to visualize your model, refer to the [Mesa Documentation](https://mesa.readthedocs.io/latest/tutorials/4_visualization_basic.html).


_This guide is a work in progress. The development of it is tracked in [Issue #2233](https://github.com/mesa/mesa/issues/2233)._


### Defining Portrayal Components
Previously, `agent_portrayal` returned a dictionary. Now, it returns an instance of a dedicated portrayal component called `AgentPortrayalStyle`.

```python
# Old
def agent_portrayal(agent):
    return {
        "color": "white" if agent.state == 0 else "black",
        "marker": "s",
        "size": "30"
    }

# New
def agent_portrayal(agent):
    return AgentPortrayalStyle(
        color="white" if agent.state == 0 else "black",
        marker="s",
        size=30,
    )
```

Similarly, `propertylayer_portrayal` has moved from a dictionary-based interface to a function-based one, following the same pattern as `agent_portrayal`. It now returns a `PropertyLayerStyle` instance instead of a dictionary.

```python
# Old
propertylayer_portrayal = {
    "sugar": {
        "colormap": "pastel1",
        "alpha": 0.75,
        "colorbar": True,
        "vmin": 0,
        "vmax": 10,
    }
}

# New
def propertylayer_portrayal(layer):
    if layer.name == "sugar":
        return PropertyLayerStyle(
            color="pastel1", alpha=0.75, colorbar=True, vmin=0, vmax=10
        )
```

* Ref: [PR #2786](https://github.com/mesa/mesa/pull/2786)

### Passing portrayal arguments to draw methods
Passing portrayal arguments directly to `draw_agents()` and `draw_propertylayer()` is deprecated. Use the `setup_agents()` and `setup_propertylayer()` methods before calling the draw methods.

```python
# Old
renderer.draw_agents(agent_portrayal=agent_portrayal)
renderer.draw_propertylayer(propertylayer_portrayal)

# New
renderer.setup_agents(agent_portrayal).draw_agents()
renderer.setup_propertylayer(propertylayer_portrayal).draw_propertylayer()
```

This change allows for better method chaining and separates the configuration phase from the rendering phase.

* Ref: [PR #2893](https://github.com/mesa/mesa/pull/2893)

### Default Space Visualization
While the visualization methods from Mesa versions before 3.3.0 still work, version 3.3.0 introduces `SpaceRenderer`, which changes how space visualizations are rendered. Check out the updated [Mesa documentation](https://mesa.readthedocs.io/latest/tutorials/4_visualization_basic.html) for guidance on upgrading your model’s visualization using `SpaceRenderer`.

A basic example of how `SpaceRenderer` works:

```python
# Old
from mesa.visualization import SolaraViz, make_space_component

SolaraViz(model, components=[make_space_component(agent_portrayal)])

# New
from mesa.visualization import SolaraViz, SpaceRenderer

renderer = SpaceRenderer(model, backend="matplotlib").render(
    agent_portrayal=agent_portrayal,
    ...
)

SolaraViz(
    model,
    renderer,
    components=[],
    ...
)
```

* Ref: [PR #2803](https://github.com/mesa/mesa/pull/2803), [PR #2810](https://github.com/mesa/mesa/pull/2810)

### Page Tab View

Version 3.3.0 adds support for defining pages for different plot components. Learn more in the [Mesa documentation](https://mesa.readthedocs.io/latest/tutorials/6_visualization_rendering_with_space_renderer.html).

In short, you can define multiple pages using the following syntax:

```python
from mesa.visualization import SolaraViz, make_plot_component

SolaraViz(
    model,
    components=[
        make_plot_component("foo", page=1),
        make_plot_component("bar", "baz", page=2),
    ],
)
```

* Ref: [PR #2827](https://github.com/mesa/mesa/pull/2827)


## Mesa 3.0
Mesa 3.0 introduces significant changes to core functionalities, including agent and model initialization, scheduling, and visualization. The guide below outlines these changes and provides instructions for migrating your existing Mesa projects to version 3.0.


### Upgrade strategy
We recommend the following upgrade strategy:
- Update to the latest Mesa 2.x release (`mesa<3`).
- Update to the latest Mesa 3.0.x release (`mesa<3.1`).
- Update to the latest Mesa 3.x release (`mesa<4`).

With each update, resolve all errors and warnings, before updating to the next one.


### Reserved and private variables
<!-- TODO: Update this section based on https://github.com/mesa/mesa/discussions/2230 -->

#### Reserved variables
Currently, we have reserved the following variables:
  - Model: `agents`, `current_id`, `random`, `running`, `steps`, `time`.
  - Agent: `unique_id`, `model`.

You can use (read) any reserved variable, but Mesa may update them automatically and rely on them, so modify/update at your own risk.

#### Private variables
Any variables starting with an underscore (`_`) are considered private and for Mesa's internal use. We might use any of those. Modifying or overwriting any private variable is at your own risk.

- Ref: [Discussion #2230](https://github.com/mesa/mesa/discussions/2230), [PR #2225](https://github.com/mesa/mesa/pull/2225)


### Removal of `mesa.flat` namespace
The `mesa.flat` namespace is removed. Use the full namespace for your imports.

- Ref: [PR #2091](https://github.com/mesa/mesa/pull/2091)


### Mandatory Model initialization with `super().__init__()`
In Mesa 3.0, it is now mandatory to call `super().__init__()` when initializing your model class. This ensures that all necessary Mesa model variables are correctly set up and agents are properly added to the model. If you want to control the seed of the random number generator, you have to pass this as a keyword argument to super as shown below.

Make sure all your model classes explicitly call `super().__init__()` in their `__init__` method:

```python
class MyModel(mesa.Model):
    def __init__(self, some_arg_I_need, seed=None, some_kwarg_I_need=True):
        super().__init__(seed=seed)  # Calling super is now required, passing seed is highly recommended
        # Your model initialization code here
        # this code uses some_arg_I_need and my_init_kwarg
```

This change ensures that all Mesa models are properly initialized, which is crucial for:
- Correctly adding agents to the model
- Setting up other essential Mesa model variables
- Maintaining consistency across all models

If you forget to call `super().__init__()`, you'll now see this error:

```
RuntimeError: The Mesa Model class was not initialized. You must explicitly initialize the Model by calling super().__init__() on initialization.
```

- Ref: [PR #2218](https://github.com/mesa/mesa/pull/2218), [PR #1928](https://github.com/mesa/mesa/pull/1928), Mesa-examples [PR #83](https://github.com/mesa/mesa-examples/pull/83)


### Automatic assignment of `unique_id` to Agents
In Mesa 3.0, `unique_id` for agents is now automatically assigned, simplifying agent creation and ensuring unique IDs across all agents in a model.

1. Remove `unique_id` from agent initialization:
   ```python
   # Old
   agent = MyAgent(unique_id=unique_id, model=self, ...)
   agent = MyAgent(unique_id, self, ...)
   agent = MyAgent(self.next_id(), self, ...)

   # New
   agent = MyAgent(model=self, ...)
   agent = MyAgent(self, ...)
   ```

2. Remove `unique_id` from Agent super() call:
   ```python
   # Old
   class MyAgent(Agent):
       def __init__(self, unique_id, model, ...):
           super().__init__(unique_id, model)

   # New
   class MyAgent(Agent):
       def __init__(self, model, ...):
           super().__init__(model)
   ```

3. Important notes:
   - `unique_id` is now automatically assigned relative to a Model instance and starts from 1
   - `Model.next_id()` is removed
   - If you previously used custom `unique_id` values, store that information in a separate attribute

- Ref: [PR #2226](https://github.com/mesa/mesa/pull/2226), [PR #2260](https://github.com/mesa/mesa/pull/2260), Mesa-examples [PR #194](https://github.com/mesa/mesa-examples/pull/194), [Issue #2213](https://github.com/mesa/mesa/issues/2213)


### AgentSet and `Model.agents`
In Mesa 3.0, the Model class internally manages agents using several data structures:

- `self._agents`: A dictionary containing hard references to all agents, indexed by their `unique_id`.
- `self._agents_by_type`: A dictionary of AgentSets, organizing agents by their type.
- `self._all_agents`: An AgentSet containing all agents in the model.

These internal structures are used to efficiently manage and access agents. Users should interact with agents through the public `model.agents` property, which returns the `self._all_agents` AgentSet.

#### `Model.agents`
- Attempting to set `model.agents` now raises an `AttributeError` instead of a warning. This attribute is reserved for internal use by Mesa.
- If you were previously setting `model.agents` in your code, you must update it to use a different attribute name for custom agent storage.

For example, replace:
```python
model.agents = my_custom_agents
```

With:
```python
model.custom_agents = my_custom_agents
```


### Time and schedulers
<!-- TODO general explanation-->

#### Automatic increase of the `steps` counter
The `steps` counter is now automatically increased. With each call to `Model.steps()` it's increased by 1, at the beginning of the step.

You can access it by `Model.steps`, and it's internally in the datacollector, batchrunner and the visualisation.

- Ref: [PR #2223](https://github.com/mesa/mesa/pull/2223), Mesa-examples [PR #161](https://github.com/mesa/mesa-examples/pull/161)

#### Removal of `Model._time` and rename `._steps`
- `Model._time` is removed. You can define your own time variable if needed.
- `Model._steps` steps is renamed to `Model.steps`.

#### Removal of `Model._advance_time()`
- The `Model._advance_time()` method is removed. This now happens automatically.

#### Replacing Schedulers with AgentSet functionality
The whole Time module in Mesa is deprecated and will be removed in Mesa 3.1. All schedulers should be replaced with AgentSet functionality and the internal `Model.steps` counter. This allows much more flexibility in how to activate Agents and makes it explicit what's done exactly.

Here's how to replace each scheduler:

##### BaseScheduler
Replace:
```python
self.schedule = BaseScheduler(self)
self.schedule.step()
```
With:
```python
self.agents.do("step")
```

##### RandomActivation
Replace:
```python
self.schedule = RandomActivation(self)
self.schedule.step()
```
With:
```python
self.agents.shuffle_do("step")
```

##### SimultaneousActivation
Replace:
```python
self.schedule = SimultaneousActivation(self)
self.schedule.step()
```
With:
```python
self.agents.do("step")
self.agents.do("advance")
```

##### StagedActivation
Replace:
```python
self.schedule = StagedActivation(self, ["stage1", "stage2", "stage3"])
self.schedule.step()
```
With:
```python
for stage in ["stage1", "stage2", "stage3"]:
    self.agents.do(stage)
```

If you were using the `shuffle` and/or `shuffle_between_stages` options:
```python
stages = ["stage1", "stage2", "stage3"]
if shuffle:
    self.random.shuffle(stages)
for stage in stages:
    if shuffle_between_stages:
        self.agents.shuffle_do(stage)
    else:
        self.agents.do(stage)
```

##### RandomActivationByType
Replace:
```python
self.schedule = RandomActivationByType(self)
self.schedule.step()
```
With:
```python
for agent_class in self.agent_types:
    self.agents_by_type[agent_class].shuffle_do("step")
```

###### Replacing `step_type`
The `RandomActivationByType` scheduler had a `step_type` method that allowed stepping only agents of a specific type. To replicate this functionality using AgentSet:

Replace:
```python
self.schedule.step_type(AgentType)
```

With:
```python
self.agents_by_type[AgentType].shuffle_do("step")
```

##### General Notes

1. The `Model.steps` counter is now automatically incremented. You don't need to manage it manually.
2. If you were using `self.schedule.agents`, replace it with `self.agents`.
3. If you were using `self.schedule.get_agent_count()`, replace it with `len(self.agents)`.
4. If you were using `self.schedule.agents_by_type`, replace it with `self.agents_by_type`.
5. Agents are now automatically added to or removed from the model's `AgentSet` (`model.agents`) when they are created or deleted, eliminating the need to manually call `self.schedule.add()` or `self.schedule.remove()`.
   - However, you still need to explicitly remove the Agent itself by using `Agent.remove()`. Typically, this means:
     - Replace `self.schedule.remove(agent)` with `agent.remove()` in the Model.
     - Replace `self.model.schedule.remove(self)` with `self.remove()` within the Agent.

From now on you're now not bound by 5 distinct schedulers, but can mix and match any combination of AgentSet methods (`do`, `shuffle`, `select`, etc.) to get the desired Agent activation.

Ref: Original discussion [#1912](https://github.com/mesa/mesa/discussions/1912), decision discussion [#2231](https://github.com/mesa/mesa/discussions/2231), example updates [#183](https://github.com/mesa/mesa-examples/pull/183) and [#201](https://github.com/mesa/mesa-examples/pull/201), PR [#2306](https://github.com/mesa/mesa/pull/2306)

### Visualisation

Mesa has adopted a new API for our frontend. If you already migrated to the experimental new SolaraViz you can still use
the import from mesa.experimental. Otherwise here is a list of things you need to change.

> **Note:** SolaraViz is experimental and still in active development for Mesa 3.0. While we attempt to minimize them, there might be API breaking changes between Mesa 3.0 and 3.1. There won't be breaking changes between Mesa 3.0.x patch releases.

#### Model Initialization

Previously SolaraViz was initialized by providing a `model_cls` and a `model_params`. This has changed to expect a model instance `model`. You can still provide (user-settable) `model_params`, but only if users should be able to change them. It is now also possible to pass in a "reactive model" by first calling `model = solara.reactive(model)`. This is useful for notebook environments. It allows you to pass the model to the SolaraViz Module, but continue to use the model. For example calling `model.value.step()` (notice the extra .value) will automatically update the plots. This currently only automatically works for the step method, you can force visualization updates by calling `model.value.force_update()`.

### Model Initialization with Keyword Arguments

With the introduction of SolaraViz in Mesa 3.0, models are now instantiated using `**model_parameters.value`. This means all inputs for initializing a new model must be keyword arguments. Ensure your model's `__init__` method accepts keyword arguments matching the keys in `model_params`.

```python
class MyModel(mesa.Model):
    def __init__(self, n_agents=10, seed=None):
        super().__init__(seed=seed)
        # Initialize the model with N agents
```

#### Default space visualization

Previously we included a default space drawer that you could configure with an `agent_portrayal` function. You now have to explicitly create a space drawer with the `agent_portrayal` function

```python
# old
from mesa.experimental import SolaraViz

SolaraViz(model_cls, model_params, agent_portrayal=agent_portrayal)

# new
from mesa.visualization import SolaraViz, make_space_component

SolaraViz(model, components=[make_space_component(agent_portrayal)])
```

#### Plotting "measures"

"Measure" plots also need to be made explicit here. Previously, measure could either be 1) A function that receives a model and returns a solara component or 2) A string or list of string of variables that are collected by the datacollector and are to be plotted as a line plot. 1) still works, but you can pass that function to "components" directly. 2) needs to explicitly call the `make_plot_measure()`function.

```python
# old
from mesa.experimental import SolaraViz


def make_plot(model):
    ...


SolaraViz(model_cls, model_params, measures=[make_plot, "foo", ["bar", "baz"]])

# new
from mesa.visualization import SolaraViz, make_plot_component

SolaraViz(model, components=[make_plot, make_plot_component("foo"), make_plot_component("bar", "baz")])
```

#### Plotting text

To plot model-dependent text the experimental SolaraViz provided a `make_text` function that wraps another functions that receives the model and turns its string return value into a solara text component. Again, this other function can now be passed directly to the new SolaraViz components array. It is okay if your function just returns a string.

```python
# old
from mesa.experimental import SolaraViz, make_text

def show_steps(model):
    return f"Steps: {model.steps}"

SolaraViz(model_cls, model_params, measures=make_text(show_steps))

# new
from mesa.visualisation import SolaraViz

def show_steps(model):
    return f"Steps: {model.steps}"

SolaraViz(model, components=[show_steps])
```

### Other changes
#### Removal of Model.initialize_data_collector
The `initialize_data_collector` in the Model class is removed. In the Model class, replace:

Replace:
```python
self.initialize_data_collector(...)
```

With:
```python
self.datacollector = DataCollector(...)
```

- Ref: [PR #2327](https://github.com/mesa/mesa/pull/2327), Mesa-examples [PR #208](https://github.com/mesa/mesa-examples/pull/208))
