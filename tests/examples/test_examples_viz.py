# noqa: D100
import base64

import playwright.sync_api
import pytest
from IPython.display import display

from mesa.examples import (
    BoidFlockers,
    BoltzmannWealth,
    ConwaysGameOfLife,
    EpsteinCivilViolence,
    PdGrid,
    Schelling,
    SugarscapeG1mt,
    VirusOnNetwork,
    WolfSheep,
)
from mesa.visualization.components import AgentPortrayalStyle
from mesa.visualization.components.matplotlib_components import (
    PlotMatplotlib,
    SpaceMatplotlib,
)


def run_model_test(
    model,
    agent_portrayal,
    solara_test,
    page_session: playwright.sync_api.Page,
    measure_config=None,
    steps=5,
):
    """Generic test for agent-based models to verify visual changes after steps.

    For more details, see the documentation:
        https://solara.dev/documentation/advanced/howto/testing#testing-widgets-using-solara-server
    """
    try:
        # Create visualizations for the initial model state
        space_viz = SpaceMatplotlib(
            model=model, agent_portrayal=agent_portrayal, propertylayer_portrayal=None
        )
        initial_graph = None

        if measure_config:
            graph_viz = PlotMatplotlib(model=model, measure=measure_config)

        # Display and capture the initial visualizations
        display(space_viz)
        page_session.wait_for_selector("img")  # buffer for rendering
        initial_space = page_session.locator("img").screenshot()

        if measure_config:
            display(graph_viz)
            page_session.wait_for_selector("img")
            initial_graph = page_session.locator("img").screenshot()

        # Run the model for specified number of steps
        for _ in range(steps):
            model.step()

        # Create new visualizations for the updated model state
        space_viz = SpaceMatplotlib(
            model=model, agent_portrayal=agent_portrayal, propertylayer_portrayal=None
        )
        changed_graph = None

        if measure_config:
            graph_viz = PlotMatplotlib(model=model, measure=measure_config)

        # Display and capture the updated visualizations
        display(space_viz)
        page_session.wait_for_selector("img")
        changed_space = page_session.locator("img").first.screenshot()

        if measure_config:
            display(graph_viz)
            page_session.wait_for_selector("img")
            changed_graph = page_session.locator("img").last.screenshot()

        # Convert screenshots to base64 for comparison
        initial_space_encoding = base64.b64encode(initial_space).decode()
        changed_space_encoding = base64.b64encode(changed_space).decode()

        if measure_config and initial_graph is not None and changed_graph is not None:
            initial_graph_encoding = base64.b64encode(initial_graph).decode()
            changed_graph_encoding = base64.b64encode(changed_graph).decode()

        # Assert that visualizations changed after running steps
        assert initial_space_encoding != changed_space_encoding, (
            "The space visualization did not change after steps."
        )

        if measure_config and initial_graph is not None and changed_graph is not None:
            assert initial_graph_encoding != changed_graph_encoding, (
                "The graph visualization did not change after steps."
            )
    except MemoryError:
        pytest.skip("Skipping test due to memory shortage.")
    except Exception:
        raise


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_schelling_model(solara_test, page_session: playwright.sync_api.Page):
    """Test schelling model behavior and visualization."""
    model = Schelling(rng=42)

    def agent_portrayal(agent):
        return AgentPortrayalStyle(
            color="tab:orange" if agent.type == 0 else "tab:blue"
        )

    measure_config = {"happy": "tab:green"}

    run_model_test(
        model=model,
        agent_portrayal=agent_portrayal,
        measure_config=measure_config,
        solara_test=solara_test,
        page_session=page_session,
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_wolf_sheep_model(solara_test, page_session: playwright.sync_api.Page):
    """Test wolf-sheep model behavior and visualization."""
    from mesa.examples.advanced.wolf_sheep.agents import (  # noqa: PLC0415
        GrassPatch,
        Sheep,
        Wolf,
    )
    from mesa.experimental.devs import ABMSimulator  # noqa: PLC0415

    model = WolfSheep(simulator=ABMSimulator(), rng=42)

    def agent_portrayal(agent):
        if agent is None:
            return

        if isinstance(agent, Wolf):
            return AgentPortrayalStyle(
                size=25,
                color="tab:red",
                marker="o",
                zorder=2,
            )
        elif isinstance(agent, Sheep):
            return AgentPortrayalStyle(
                size=25,
                color="tab:cyan",
                marker="o",
                zorder=2,
            )
        elif isinstance(agent, GrassPatch):
            return AgentPortrayalStyle(
                size=75,
                color="tab:green" if agent.fully_grown else "tab:brown",
                marker="s",
            )

    measure_config = {"Wolves": "tab:orange", "Sheep": "tab:cyan", "Grass": "tab:green"}

    run_model_test(
        model=model,
        agent_portrayal=agent_portrayal,
        measure_config=measure_config,
        solara_test=solara_test,
        page_session=page_session,
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_boid_flockers_model(solara_test, page_session: playwright.sync_api.Page):
    """Test boid flockers model behavior and visualization."""
    model = BoidFlockers(rng=42)

    def agent_portrayal(agent):
        return AgentPortrayalStyle(color="tab:blue")

    run_model_test(
        model=model,
        agent_portrayal=agent_portrayal,
        measure_config=None,
        solara_test=solara_test,
        page_session=page_session,
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_boltzmann_wealth_model(solara_test, page_session: playwright.sync_api.Page):
    """Test Boltzmann wealth model behavior and visualization."""
    model = BoltzmannWealth(rng=42)

    def agent_portrayal(agent):
        return AgentPortrayalStyle(color=agent.wealth)

    measure_config = "Gini"

    run_model_test(
        model=model,
        agent_portrayal=agent_portrayal,
        measure_config=measure_config,
        solara_test=solara_test,
        page_session=page_session,
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_virus_on_network_model(solara_test, page_session: playwright.sync_api.Page):
    """Test virus on network model behavior and visualization."""
    from mesa.examples.basic.virus_on_network.model import State  # noqa: PLC0415

    model = VirusOnNetwork(rng=42)

    def agent_portrayal(agent):
        node_color_dict = {
            State.INFECTED: "tab:red",
            State.SUSCEPTIBLE: "tab:green",
            State.RESISTANT: "tab:gray",
        }
        return AgentPortrayalStyle(color=node_color_dict[agent.state], size=10)

    measure_config = {
        "Infected": "tab:red",
        "Susceptible": "tab:green",
        "Resistant": "tab:gray",
    }

    run_model_test(
        model=model,
        agent_portrayal=agent_portrayal,
        measure_config=measure_config,
        solara_test=solara_test,
        page_session=page_session,
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_conways_game_of_life_model(
    solara_test, page_session: playwright.sync_api.Page
):
    """Test Conway's Game of Life model behavior and visualization."""
    model = ConwaysGameOfLife(rng=42)

    def agent_portrayal(agent):
        return AgentPortrayalStyle(
            color="white" if agent.state == 0 else "black",
            marker="s",
            size=25,
        )

    measure_config = None

    run_model_test(
        model=model,
        agent_portrayal=agent_portrayal,
        measure_config=measure_config,
        solara_test=solara_test,
        page_session=page_session,
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_epstein_civil_violence_model(
    solara_test, page_session: playwright.sync_api.Page
):
    """Test Epstein civil violence model behavior and visualization."""
    from mesa.examples.advanced.epstein_civil_violence.agents import (  # noqa: PLC0415
        Citizen,
        CitizenState,
        Cop,
    )
    from mesa.examples.advanced.epstein_civil_violence.app import (  # noqa: PLC0415
        COP_COLOR,
        agent_colors,
    )

    model = EpsteinCivilViolence(rng=42)

    def agent_portrayal(agent):
        if agent is None:
            return

        if isinstance(agent, Citizen):
            return AgentPortrayalStyle(
                size=50,
                color=agent_colors[agent.state],
            )
        elif isinstance(agent, Cop):
            return AgentPortrayalStyle(
                size=50,
                color=COP_COLOR,
            )

    measure_config = {state.name.lower(): agent_colors[state] for state in CitizenState}

    run_model_test(
        model=model,
        agent_portrayal=agent_portrayal,
        measure_config=measure_config,
        solara_test=solara_test,
        page_session=page_session,
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_sugarscape_g1mt_model(solara_test, page_session: playwright.sync_api.Page):
    """Test Sugarscape G1mt model behavior and visualization."""
    model = SugarscapeG1mt(rng=42)

    def agent_portrayal(agent):
        return AgentPortrayalStyle(marker="o", color="red", size=10)

    measure_config = "Price"

    run_model_test(
        model=model,
        agent_portrayal=agent_portrayal,
        measure_config=measure_config,
        solara_test=solara_test,
        page_session=page_session,
        steps=50,
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_pd_grid_model(solara_test, page_session: playwright.sync_api.Page):
    """Test Prisoner's Dilemma model behavior and visualization."""
    model = PdGrid(rng=42)

    def agent_portrayal(agent):
        return AgentPortrayalStyle(
            color="blue" if agent.move == "C" else "red",
            marker="s",  # square marker
            size=25,
        )

    measure_config = "Cooperating_Agents"

    run_model_test(
        model=model,
        agent_portrayal=agent_portrayal,
        measure_config=measure_config,
        solara_test=solara_test,
        page_session=page_session,
    )
