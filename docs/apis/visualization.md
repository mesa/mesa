# Visualization


⚠️ **Important note for SolaraViz users**

When using **SolaraViz**, Mesa models must be instantiated **using keyword arguments only**.
SolaraViz creates model instances internally via keyword-based parameters, and positional arguments are **not supported**.

**Not supported:**

```python
MyModel(10, 10)
```

**Supported:**

```python
MyModel(width=10, height=10)
```

To avoid errors, it is recommended to define your model constructor with keyword-only arguments, for example:

```python
class MyModel(Model):
    def __init__(self, *, width, height, seed=None):
        ...
```


For detailed tutorials, please refer to:

- [Basic Visualization](../tutorials/4_visualization_basic)
- [Dynamic Agent Visualization](../tutorials/5_visualization_dynamic_agents)
- [Custom Agent Visualization](../tutorials/6_visualization_custom)


## Jupyter Visualization

```{eval-rst}
.. automodule:: mesa.visualization.solara_viz
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: mesa.visualization.components.__init__
   :members:
   :undoc-members:
   :show-inheritance:
```

## User Parameters

```{eval-rst}
.. automodule:: mesa.visualization.user_param
   :members:
   :undoc-members:
   :show-inheritance:
```

## Custom Parameter Types

Mesa's visualization system supports custom parameter types through an extractor registry. This allows you to define complex parameter structures beyond the built-in types (Slider, Checkbox, etc.).

### Basic Usage

Register a custom parameter extractor before creating your `SolaraViz` instance:

```python
from mesa.visualization.solara_viz import register_param_extractor

def extract_dict_param(spec):
    """Extract default values from a nested dictionary parameter."""
    entries = spec.get("entries", {})
    result = {}
    for key, value in entries.items():
        if isinstance(value, dict) and "value" in value:
            result[key] = value["value"]
        elif isinstance(value, dict):
            result[key] = extract_dict_param({"entries": value})
        else:
            result[key] = value
    return result

# Register the extractor for "Dict" type parameters
register_param_extractor("Dict", extract_dict_param)

# Now you can use Dict type in model_params
model_params = {
    "config": {
        "type": "Dict",
        "entries": {
            "learning_rate": {"value": 0.01},
            "batch_size": {"value": 32},
            "optimizer": {
                "name": {"value": "adam"},
                "momentum": {"value": 0.9}
            }
        }
    }
}

# The model will receive config as:
# {"learning_rate": 0.01, "batch_size": 32,
#  "optimizer": {"name": "adam", "momentum": 0.9}}
```

### How It Works

When SolaraViz initializes model parameters:
1. It checks if a parameter has a registered extractor for its "type"
1. If found, it calls the extractor function with the parameter specification
1. If not found, it falls back to extracting the "value" key (default behavior)
1. This allows you to create arbitrarily complex parameter types while keeping Mesa's core simple and extensible.

For a complete working example, see `mesa/examples/basic/boltzmann_wealth_model/app_with_dict_params.py` and note the way we overwrite Mesa's UserInputs with our own CustomUserInputs:

```python
# Import custom parameter support
from custom_params import CustomUserInputs
import mesa.visualization.solara_viz as solara_viz

# Override UserInputs with CustomUserInputs
solara_viz.UserInputs = CustomUserInputs
```

Common Use Cases
* Nested configurations: Complex model settings organized hierarchically
* Matrix/array parameters: Initial states for grid-based models
* Graph structures: Network topologies as parameters
* Custom data structures: Any domain-specific parameter format
* Data file uploads (GeoJSON, for example) in a way that keeps the parsing logic separate from your model code


## Matplotlib-based visualizations

```{eval-rst}
.. automodule:: mesa.visualization.components.matplotlib_components
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: mesa.visualization.mpl_space_drawing
   :members:
   :undoc-members:
   :show-inheritance:
```


## Altair-based visualizations

```{eval-rst}
.. automodule:: mesa.visualization.components.altair_components
   :members:
   :undoc-members:
   :show-inheritance:
```


## Command Console

```{eval-rst}
.. automodule:: mesa.visualization.command_console
   :members:
   :undoc-members:
   :show-inheritance:
```


## Portrayal Components
```{eval-rst}
.. automodule:: mesa.visualization.components.portrayal_components
   :members:
   :undoc-members:
   :show-inheritance:
```


## Backends

```{eval-rst}
.. automodule:: mesa.visualization.backends.__init__
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: mesa.visualization.backends.abstract_renderer
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: mesa.visualization.backends.altair_backend
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: mesa.visualization.backends.matplotlib_backend
   :members:
   :undoc-members:
   :show-inheritance:
```


## Space Renderer

```{eval-rst}
.. automodule:: mesa.visualization.space_renderer
   :members:
   :undoc-members:
   :show-inheritance:
```


## Space Drawers

```{eval-rst}
.. automodule:: mesa.visualization.space_drawers
   :members:
   :undoc-members:
   :show-inheritance:
```
