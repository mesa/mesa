# noqa: D100
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


def test_boltzmann_model():  # noqa: D103
    from mesa.examples.basic.boltzmann_wealth_model import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = BoltzmannWealth(rng=42)

    for _i in range(10):
        model.step()


def test_conways_game_model():  # noqa: D103
    from mesa.examples.basic.conways_game_of_life import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = ConwaysGameOfLife(rng=42)
    for _i in range(10):
        model.step()


def test_schelling_model():  # noqa: D103
    from mesa.examples.basic.schelling import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = Schelling(rng=42)
    for _i in range(10):
        model.step()


def test_virus_on_network():  # noqa: D103
    from mesa.examples.basic.virus_on_network import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = VirusOnNetwork(rng=42)
    for _i in range(10):
        model.step()


def test_boid_flockers():  # noqa: D103
    from mesa.examples.basic.boid_flockers import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = BoidFlockers(rng=42)

    for _i in range(10):
        model.step()


def test_epstein():  # noqa: D103
    from mesa.examples.advanced.epstein_civil_violence import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = EpsteinCivilViolence(rng=42)

    for _i in range(10):
        model.step()


def test_pd_grid():  # noqa: D103
    from mesa.examples.advanced.pd_grid import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = PdGrid(rng=42)

    for _i in range(10):
        model.step()


def test_sugarscape_g1mt():  # noqa: D103
    from mesa.examples.advanced.sugarscape_g1mt import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = SugarscapeG1mt(rng=42)

    for _i in range(10):
        model.step()


def test_wolf_sheep():  # noqa: D103
    from mesa.examples.advanced.wolf_sheep import app  # noqa: PLC0415
    from mesa.experimental.devs import ABMSimulator  # noqa: PLC0415

    app.page  # noqa: B018

    simulator = ABMSimulator()
    WolfSheep(rng=42, simulator=simulator)
    simulator.run_for(10)


def test_alliance_formation_model():  # noqa: D103
    from mesa.examples.advanced.alliance_formation import app  # noqa: PLC0415

    app.page  # noqa: B018

    model = MultiLevelAllianceModel(50, rng=42)

    for _i in range(10):
        model.step()

    assert len(model.agents) == len(model.network.nodes)
