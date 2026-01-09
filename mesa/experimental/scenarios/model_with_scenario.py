"""Small extension to the base model class to add support for the experimental Scenario."""

from mesa.model import Model

from .scenario import Scenario


class ModelWithScenario(Model):
    """Base model class with support for the experimental Scenario class."""

    @property
    def scenario(self) -> Scenario:
        """Return scenario instance."""
        return self._scenario

    @scenario.setter
    def scenario(self, scenario: Scenario) -> None:
        """Set scenario instance."""
        self._scenario = scenario
        scenario.model = self

    def __init__(self, *args, scenario:Scenario|None=None, **kwargs):
        """"Init of ModelWithScenario.

        Args:
            args: all positional args # fixme we might want to completely disable this because of solara
            scenario: a scenario instance, optional
            kwargs: all additional keyword args


        """
        if scenario is None:
            scenario = Scenario(rng=None)
        super().__init__(*args, rng=scenario["rng"], **kwargs)
        self.scenario = scenario
