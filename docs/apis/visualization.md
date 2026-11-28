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
    def __init__(self, *, width, height, rng=None):
        ...
```


For detailed tutorials, please refer to:

- [Basic Visualization](../tutorials/6_visualization_basic)
- [Dynamic Agent Visualization](../tutorials/7_visualization_dynamic_agents)
- [Custom Visualization Components](../tutorials/10_visualization_custom)


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


## Altair-based visualizations

```{eval-rst}
.. automodule:: mesa.visualization.components.altair_components
   :members:
   :undoc-members:
   :show-inheritance:
```


### Agent tooltip columns and post-processing

In the Altair backend, agent tooltip values are stored in columns named
`tooltip_{key}`, using `str(key)` for the label. For example, `tooltip={"wealth": 5}`
adds a `tooltip_wealth` column to the agent chart's dataframe. The displayed hover
label is still `wealth`. Tooltip values do not overwrite rendering columns such
as `x`, `y` or `size`.

Post-processing code that previously used the original tooltip key as a column
name must use the prefixed name instead. For example, on the agent chart:

```python
def post_process(agent_chart):
    return agent_chart.transform_filter("datum.tooltip_wealth > 0")
```

If the renderer combines several charts into layers, apply the change to the
agent layer. Dataframe column names preserve punctuation in keys. When using
these names in an Altair encoding, escape dots, brackets and backslashes so
Vega-Lite treats the name as a literal field rather than a nested path. Tooltip
encodings created by Mesa already do this. Different tooltip keys must have
distinct string labels: keys such as `1` and `"1"` in the same chart raise
`ValueError` rather than silently replacing one another.


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
