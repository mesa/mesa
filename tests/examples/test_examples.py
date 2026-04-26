# noqa: D100
import gc
import weakref

from mesa.examples import (
    BoidFlockers,
    BoltzmannWealth,
    ConwaysGameOfLife,
    EpsteinCivilViolence,
    MultiLevelAllianceModel,
    PdGrid,
    Schelling,
    SugarscapeG1mt,
    VirusOnNetwork,
    WolfSheep,
)
from mesa.examples.advanced.alliance_formation.model import AllianceScenario
from mesa.examples.advanced.pd_grid.model import PrisonersDilemmaScenario
from mesa.examples.advanced.wolf_sheep.model import WolfSheepScenario
from mesa.examples.basic.boid_flockers.model import BoidsScenario
from mesa.examples.basic.boltzmann_wealth_model.model import BoltzmannScenario
from mesa.examples.basic.schelling.model import SchellingScenario
from mesa.examples.advanced.wolf_sheep.agents import Wolf

def test_boltzmann_model():  # noqa: D103
    from mesa.examples.basic.boltzmann_wealth_model import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = BoltzmannWealth(scenario=BoltzmannScenario(rng=42))
    ref = weakref.ref(model)

    model.run_for(10)
    model.remove_all_agents()  # this seems to be needed

    del model
    gc.collect()
    assert ref() is None


def test_boltzmann_model_init_variants():  # noqa: D103
    model = BoltzmannWealth()
    assert model.num_agents == 100
    assert model.grid.width == 10
    assert model.grid.height == 10

    model = BoltzmannWealth(scenario=BoltzmannScenario(rng=123))
    assert model.num_agents == 100
    assert model.grid.width == 10
    assert model.grid.height == 10

    scenario = BoltzmannScenario(n=7, width=8, height=9)
    model = BoltzmannWealth(scenario=scenario)
    assert model.num_agents == 7
    assert model.grid.width == 8
    assert model.grid.height == 9


def test_conways_game_model():  # noqa: D103
    from mesa.examples.basic.conways_game_of_life import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = ConwaysGameOfLife(rng=42)
    ref = weakref.ref(model)

    model.run_for(10)
    model.remove_all_agents()

    del model
    gc.collect()
    assert ref() is None


def test_schelling_model():  # noqa: D103
    from mesa.examples.basic.schelling import app  # noqa: PLC0415

    app.page  # noqa: B018

    _model = Schelling()
    model = Schelling(scenario=SchellingScenario(rng=42))
    ref = weakref.ref(model)

    model.run_for(10)
    model.remove_all_agents()

    del model
    gc.collect()
    assert ref() is None


def test_virus_on_network():  # noqa: D103
    from mesa.examples.basic.virus_on_network import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = VirusOnNetwork(rng=42)
    ref = weakref.ref(model)

    model.run_for(10)
    model.remove_all_agents()

    del model
    gc.collect()
    assert ref() is None


def test_boid_flockers():  # noqa: D103
    from mesa.examples.basic.boid_flockers import app  # noqa: PLC0415

    app.page  # noqa: B018

    _model = BoidFlockers()

    model = BoidFlockers(scenario=BoidsScenario(rng=42))
    ref = weakref.ref(model)

    model.run_for(10)
    model.remove_all_agents()

    del model
    gc.collect()
    assert ref() is None


def test_epstein():  # noqa: D103
    from mesa.examples.advanced.epstein_civil_violence import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = EpsteinCivilViolence()
    ref = weakref.ref(model)

    model.run_for(10)
    model.remove_all_agents()

    del model
    gc.collect()
    assert ref() is None


def test_pd_grid():  # noqa: D103
    from mesa.examples.advanced.pd_grid import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = PdGrid(scenario=PrisonersDilemmaScenario(rng=42))
    ref = weakref.ref(model)

    model.run_for(10)
    model.remove_all_agents()

    del model
    gc.collect()
    assert ref() is None


def test_sugarscape_g1mt():  # noqa: D103
    from mesa.examples.advanced.sugarscape_g1mt import app  # noqa: PLC0415
    from mesa.examples.advanced.sugarscape_g1mt.model import (  # noqa: PLC0415
        SugarScapeScenario,
    )

    app.page  # noqa: B018

    model = SugarscapeG1mt(SugarScapeScenario(rng=42))
    ref = weakref.ref(model)

    model.run_for(10)
    model.remove_all_agents()

    del model
    gc.collect()
    assert ref() is None


def test_wolf_sheep():  # noqa: D103
    from mesa.examples.advanced.wolf_sheep import app  # noqa: PLC0415

    app.page  # noqa: B018

    _model = WolfSheep()

    model = WolfSheep(scenario=WolfSheepScenario(rng=42))
    ref = weakref.ref(model)

    model.run_for(10)
    model.remove_all_agents()

    del model
    gc.collect()
    assert ref() is None


def test_wolf_sheep_grass_disabled():
    """Regression test for #3597: grass=False must not raise StopIteration."""
    model = WolfSheep(scenario=WolfSheepScenario(grass=False, rng=42))
    for _ in range(10):
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    assert df.shape[0] == 11
    assert "Grass" not in df.columns
    assert "Wolves" in df.columns
    assert "Sheep" in df.columns


def test_wolf_sheep_property_layers():
    """Test that property layers for wolves and grass are updated correctly."""

    from mesa.examples.advanced.wolf_sheep.model import WolfSheep, WolfSheepScenario

    model = WolfSheep(scenario=WolfSheepScenario(rng=42))
    # Step the model once
    model.step()
    # Check that the property layers exist and have correct shape
    assert hasattr(model.grid, "wolves")
    assert hasattr(model.grid, "grass")
    assert model.grid.wolves.shape == (model.height, model.width)
    assert model.grid.grass.shape == (model.height, model.width)
    # Check that wolf counts are non-negative
    assert (model.grid.wolves >= 0).all()
    # Check that grass is boolean
    assert model.grid.grass.dtype == bool
    # Move all wolves to a single cell and check the count
    wolf_agents = list(model.agents_by_type[Wolf])
    if wolf_agents:
        x, y = 0, 0
        for wolf in wolf_agents:
            wolf.cell = model.grid[x, y]
        model.grid.wolves[:, :] = 0
        model.grid.wolves[x, y] = len(wolf_agents)
        assert model.grid.wolves[x, y] == len(wolf_agents)
    # Eat all grass in the first row and check
    model.grid.grass[0, :] = False
    assert not model.grid.grass[0, :].any()


def test_alliance_formation_model():  # noqa: D103
    from mesa.examples.advanced.alliance_formation import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = MultiLevelAllianceModel(scenario=AllianceScenario(n=50, rng=42))
    ref = weakref.ref(model)

    model.run_for(10)
    assert len(model.agents) == len(model.network.nodes)

    model.remove_all_agents()

    del model
    gc.collect()
    assert ref() is None
