from mesa.experimental.data_collection.dataset import DataRegistry
from mesa.experimental.runspec import RunSpec
from mesa.experimental.scenarios import Scenario
from mesa.model import Model


def test_runspec_basic_execution():
    """RunSpec executes a model and returns expected tuple."""

    class TestModel(Model):
        def __init__(self, scenario):
            super().__init__(scenario=scenario)

        def run_for(self, steps):
            pass

    scenario = Scenario(rng=1)

    spec = RunSpec(TestModel, steps=5)
    sid, rid, result = spec(scenario)

    assert sid == scenario.scenario_id
    assert rid == scenario.replication_id
    assert isinstance(result, DataRegistry)


def test_runspec_override_execute():
    """Custom execute method overrides default behavior."""

    class TestModel(Model):
        def __init__(self, scenario):
            super().__init__(scenario=scenario)
            self.value = 0

    class CustomRunSpec(RunSpec):
        def execute(self, model):
            model.value = 42

        def extract(self, model):
            return model.value

    scenario = Scenario(rng=1)

    spec = CustomRunSpec(TestModel)
    _, _, result = spec(scenario)

    assert result == 42


def test_runspec_override_extract():
    """Custom extract method returns custom result."""

    class TestModel(Model):
        def __init__(self, scenario):
            super().__init__(scenario=scenario)
            self.value = 99

        def run_for(self, steps):
            pass

    class CustomRunSpec(RunSpec):
        def extract(self, model):
            return model.value

    scenario = Scenario(rng=1)

    spec = CustomRunSpec(TestModel)
    _, _, result = spec(scenario)

    assert result == 99


def test_runspec_format_output_override():
    """Custom output formatting is respected."""

    class TestModel(Model):
        def __init__(self, scenario):
            super().__init__(scenario=scenario)

        def run_for(self, steps):
            pass

    class CustomRunSpec(RunSpec):
        def format_output(self, scenario, result):
            return {"id": scenario.scenario_id}

    scenario = Scenario(rng=1)

    spec = CustomRunSpec(TestModel)
    output = spec(scenario)

    assert isinstance(output, dict)
    assert output["id"] == scenario.scenario_id


def test_runspec_override_build_model():
    """Custom build_model method is used."""

    class TestModel(Model):
        def __init__(self, scenario):
            super().__init__(scenario=scenario)
            self.flag = False

    class CustomRunSpec(RunSpec):
        def build_model(self, scenario):
            model = super().build_model(scenario)
            model.flag = True
            return model

        def extract(self, model):
            return model.flag

    scenario = Scenario(rng=1)

    spec = CustomRunSpec(TestModel)
    _, _, result = spec(scenario)

    assert result is True


def test_runspec_execute_affects_model():
    """Default execute runs model logic."""

    class TestModel(Model):
        def __init__(self, scenario):
            super().__init__(scenario=scenario)
            self.counter = 0

        def run_for(self, steps):
            self.counter = steps

    class CustomRunSpec(RunSpec):
        def extract(self, model):
            return model.counter

    scenario = Scenario(rng=1)

    spec = CustomRunSpec(TestModel, steps=7)
    _, _, result = spec(scenario)

    assert result == 7
