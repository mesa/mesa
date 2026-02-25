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
from mesa.examples.advanced.wolf_sheep.model import WolfSheepScenario
from mesa.examples.basic.boid_flockers.model import BoidsScenario
from mesa.examples.basic.boltzmann_wealth_model.model import BoltzmannScenario
from mesa.examples.basic.schelling.model import SchellingScenario


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

    _model = Schelling(scenario=None)
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

    _model = BoidFlockers(scenario=None)

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

    model = PdGrid(rng=42)
    ref = weakref.ref(model)

    model.run_for(10)
    model.remove_all_agents()

    del model
    gc.collect()
    assert ref() is None


def test_sugarscape_g1mt():  # noqa: D103
    from mesa.examples.advanced.sugarscape_g1mt import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = SugarscapeG1mt(rng=42)
    ref = weakref.ref(model)

    model.run_for(10)
    model.remove_all_agents()

    del model
    gc.collect()
    assert ref() is None


def test_wolf_sheep():  # noqa: D103
    from mesa.examples.advanced.wolf_sheep import app  # noqa: PLC0415

    app.page  # noqa: B018

    _model = WolfSheep(scenario=None)

    model = WolfSheep(scenario=WolfSheepScenario(rng=42))
    ref = weakref.ref(model)

    model.run_for(10)
    model.remove_all_agents()

    del model
    gc.collect()
    assert ref() is None


def test_alliance_formation_model():  # noqa: D103
    from mesa.examples.advanced.alliance_formation import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = MultiLevelAllianceModel(50, rng=42)
    ref = weakref.ref(model)

    model.run_for(10)
    assert len(model.agents) == len(model.network.nodes)

    model.remove_all_agents()

    del model
    gc.collect()
    assert ref() is None
