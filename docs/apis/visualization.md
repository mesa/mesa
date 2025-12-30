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
