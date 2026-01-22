import matplotlib.pyplot as plt
import networkx as nx
import solara
from matplotlib.figure import Figure

from mesa.examples.advanced.alliance_formation.model import MultiLevelAllianceModel
from mesa.visualization import SolaraViz
from mesa.visualization.utils import update_counter

model_params = {
    "rng": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "n": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of agents:",
        "min": 10,
        "max": 100,
        "step": 1,
    },
}

# Create visualization elements. The visualization elements are solara components
# that receive the model instance as a "prop" and display it in a certain way.
# Under the hood these are just classes that receive the model instance.
# You can also author your own visualization elements, which can also be functions
# that receive the model instance and return a valid solara component.


@solara.component
def plot_network(model):
    update_counter.get()
    g = model.network
    pos = nx.fruchterman_reingold_layout(g)
    fig = Figure()
    ax = fig.subplots()
    labels = {agent.unique_id: agent.unique_id for agent in model.agents}
    node_sizes = [g.nodes[node]["size"] for node in g.nodes]
    node_colors = [g.nodes[node]["size"] for node in g.nodes()]

    nx.draw(
        g,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.coolwarm,
        labels=labels,
        ax=ax,
    )

    solara.FigureMatplotlib(fig)


# Create initial model instance
model = MultiLevelAllianceModel(50)

# Create the SolaraViz page. This will automatically create a server and display the
# visualization elements in a web browser.
# Display it using the following command in the example directory:
# solara run app.py
# It will automatically update and display any changes made to this file
page = SolaraViz(
    model,
    components=[plot_network],
    model_params=model_params,
    name="Alliance Formation Model",
)
page  # noqa
