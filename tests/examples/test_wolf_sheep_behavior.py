"""Behavioral tests for the Wolf-Sheep example model."""

from mesa.examples.advanced.wolf_sheep.agents import Sheep, Wolf
from mesa.examples.advanced.wolf_sheep.model import WolfSheep, WolfSheepScenario


def test_sheep_risk_aware_move_prefers_lower_wolf_pressure():
    """Risk-aware sheep should move only among lowest-pressure safe cells."""
    scenario = WolfSheepScenario(
        width=5,
        height=5,
        initial_sheep=1,
        initial_wolves=1,
        grass=False,
        sheep_risk_aware_move=True,
        rng=42,
    )
    model = WolfSheep(scenario=scenario)
    model.remove_all_agents()

    center = model.grid[(2, 2)]
    sheep = Sheep(model, energy=10, p_reproduce=0.0, energy_from_food=0, cell=center)

    # Wolves are placed to create unequal "wolf pressure" around candidate cells.
    Wolf(model, energy=10, p_reproduce=0.0, energy_from_food=0, cell=model.grid[(1, 1)])
    Wolf(model, energy=10, p_reproduce=0.0, energy_from_food=0, cell=model.grid[(1, 3)])

    safe_cells = [
        cell
        for cell in sheep.cell.neighborhood
        if not any(isinstance(obj, Wolf) for obj in cell.agents)
    ]
    min_pressure = min(Sheep._wolf_pressure(cell) for cell in safe_cells)

    sheep.move()

    assert sheep.cell in safe_cells
    assert Sheep._wolf_pressure(sheep.cell) == min_pressure
