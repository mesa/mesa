import matplotlib.cm as cm
import matplotlib.colors as colors
from agents import CultureAgent
from model import CulturalSegregation
from scipy.spatial import Voronoi, voronoi_plot_2d

from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle


def agent_portrayal(agent):
    """Visualizes agents."""
    if not isinstance(agent, CultureAgent):
        return

    cmap = cm.get_cmap("coolwarm")
    rgba = cmap(agent.opinion)
    color = colors.to_hex(rgba)

    marker = "o" if agent.happy else "x"
    size = 50 if agent.happy else 80

    return AgentPortrayalStyle(color=color, marker=marker, size=size, alpha=0.8)


def make_post_process(model):
    """Create a post-process function that has access to the model instance."""

    def post_process(ax):
        ax.set_aspect("equal")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_yticks([])

        if hasattr(model.space, "centroids") and model.space.centroids is not None:
            try:
                vor_data = Voronoi(model.space.centroids)

                voronoi_plot_2d(
                    vor_data,
                    ax=ax,
                    show_vertices=False,
                    show_points=False,
                    line_colors="orange",
                    line_width=2,
                    line_alpha=0.6,
                )
            except Exception as e:
                print(f"Voronoi plot error: {e}")

    return post_process


model_params = {
    "n_agents": {
        "type": "SliderInt",
        "value": 200,
        "label": "Number of Agents",
        "min": 50,
        "max": 500,
        "step": 10,
    },
    "n_clusters": {
        "type": "SliderInt",
        "value": 8,
        "label": "Voronoi Clusters",
        "min": 4,
        "max": 20,
        "step": 1,
    },
    "tolerance": {
        "type": "SliderFloat",
        "value": 0.15,
        "label": "Tolerance Threshold",
        "min": 0.05,
        "max": 0.5,
        "step": 0.05,
    },
}

model = CulturalSegregation()
renderer = (
    SpaceRenderer(model, backend="matplotlib").setup_agents(agent_portrayal).render()
)

renderer.post_process = make_post_process(model)

page = SolaraViz(
    model,
    renderer,
    components=[make_plot_component("Unhappy Agents")],
    model_params=model_params,
    name="Adaptive Voronoi: Cultural Segregation",
)
