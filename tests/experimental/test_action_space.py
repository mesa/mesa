"""Tests for the experimental ActionSpace module."""

import pytest

from mesa import Model
from mesa.agent import Agent
from mesa.experimental.action_space import Action, ActionSpace, Constraint


# Test Fixtures
class DummyAgent(Agent):
    """Agent with energy for testing soft constraints."""

    def __init__(self, model, energy=100.0):
        """Initialize a test agent with energy."""
        super().__init__(model)
        self.energy = energy


class MaxSpeed(Constraint):
    """Hard constraint: clips speed to a maximum."""

    def __init__(self, limit):
        """Initialize with a maximum speed limit."""
        super().__init__(name=f"MaxSpeed({limit})")
        self.limit = limit

    def check(self, action, agent):
        """Return True if the action's speed is within the limit."""
        if action.action_type != "move":
            return True
        return action.params.get("speed", 0) <= self.limit

    def project(self, action, agent):
        """Clip speed to the maximum allowed limit."""
        params = dict(action.params)
        params["speed"] = min(params.get("speed", 0), self.limit)
        return Action(action.action_type, params)

    def describe(self, agent):
        """Return a human-readable description of this constraint."""
        return f"Max speed: {self.limit}"


class EnergyCost(Constraint):
    """Soft constraint: movement costs energy proportional to speed."""

    def __init__(self, rate=1.0):
        """Initialize with an energy cost rate per unit speed."""
        super().__init__(name=f"EnergyCost({rate})")
        self.rate = rate

    def cost(self, action, agent):
        """Return the energy cost of a move action."""
        if action.action_type != "move":
            return {}
        speed = action.params.get("speed", 0)
        return {"energy": speed * self.rate}

    def describe(self, agent):
        """Return a human-readable description of this constraint."""
        energy = getattr(agent, "energy", "?")
        return f"Energy cost: {self.rate}/unit (current: {energy})"


class GridBounds(Constraint):
    """Hard constraint: clips position to grid boundaries."""

    def __init__(self, width, height):
        """Initialize with grid width and height."""
        super().__init__(name=f"GridBounds({width}x{height})")
        self.width = width
        self.height = height

    def check(self, action, agent):
        """Return True if the position is within grid bounds."""
        if action.action_type != "move":
            return True
        x = action.params.get("x", 0)
        y = action.params.get("y", 0)
        return 0 <= x < self.width and 0 <= y < self.height

    def project(self, action, agent):
        """Clip position coordinates to valid grid bounds."""
        params = dict(action.params)
        params["x"] = max(0, min(params.get("x", 0), self.width - 1))
        params["y"] = max(0, min(params.get("y", 0), self.height - 1))
        return Action(action.action_type, params)


class AlwaysFails(Constraint):
    """Hard constraint that always fails — for testing projection."""

    def __init__(self):
        """Initialize the always-failing constraint."""
        super().__init__(name="AlwaysFails")

    def check(self, action, agent):
        """Always return False to trigger projection."""
        return False

    def project(self, action, agent):
        """Return a fixed action for testing."""
        return Action(action.action_type, {"fixed": True})


# Action Tests


def test_action_creation():
    """Test basic Action creation and attributes."""
    action = Action("move", {"speed": 10, "direction": "north"})
    assert action.action_type == "move"
    assert action.params == {"speed": 10, "direction": "north"}


def test_action_default_params():
    """Test Action with no params defaults to empty dict."""
    action = Action("rest")
    assert action.params == {}


def test_action_repr():
    """Test Action string representation."""
    action = Action("move", {"speed": 5})
    assert "move" in repr(action)
    assert "speed" in repr(action)


def test_action_copy():
    """Test Action copy creates independent instance."""
    original = Action("move", {"speed": 10})
    copied = original.copy()
    copied.params["speed"] = 5
    assert original.params["speed"] == 10  # Original unchanged


# Constraint Tests


def test_constraint_defaults():
    """Test that the base Constraint is fully permissive by default."""
    model = Model()
    agent = DummyAgent(model)
    constraint = Constraint()
    action = Action("move", {"speed": 100})

    assert constraint.check(action, agent) is True
    assert constraint.project(action, agent) is action
    assert constraint.cost(action, agent) == {}
    assert constraint.describe(agent) == "Constraint"


def test_constraint_custom_name():
    """Test custom constraint naming."""
    c = Constraint(name="MyRule")
    assert c.name == "MyRule"


def test_constraint_default_name():
    """Test that default name uses class name."""
    c = MaxSpeed(10)
    assert c.name == "MaxSpeed(10)"


def test_constraint_repr():
    """Test Constraint repr."""
    c = MaxSpeed(10)
    assert "MaxSpeed" in repr(c)


# Hard Constraint Tests


def test_hard_constraint_passes():
    """Test hard constraint allows valid action."""
    model = Model()
    agent = DummyAgent(model)
    c = MaxSpeed(10)

    action = Action("move", {"speed": 5})
    assert c.check(action, agent) is True


def test_hard_constraint_fails():
    """Test hard constraint rejects invalid action."""
    model = Model()
    agent = DummyAgent(model)
    c = MaxSpeed(10)

    action = Action("move", {"speed": 15})
    assert c.check(action, agent) is False


def test_hard_constraint_projects():
    """Test hard constraint clips to valid range."""
    model = Model()
    agent = DummyAgent(model)
    c = MaxSpeed(10)

    action = Action("move", {"speed": 15})
    projected = c.project(action, agent)
    assert projected.params["speed"] == 10


def test_hard_constraint_ignores_irrelevant_actions():
    """Test that MaxSpeed doesn't constrain non-move actions."""
    model = Model()
    agent = DummyAgent(model)
    c = MaxSpeed(10)

    action = Action("eat", {"amount": 100})
    assert c.check(action, agent) is True


def test_grid_bounds_check():
    """Test GridBounds constraint validates position."""
    model = Model()
    agent = DummyAgent(model)
    c = GridBounds(50, 50)

    assert c.check(Action("move", {"x": 25, "y": 25}), agent) is True
    assert c.check(Action("move", {"x": -1, "y": 25}), agent) is False
    assert c.check(Action("move", {"x": 25, "y": 50}), agent) is False
    assert c.check(Action("move", {"x": 50, "y": 0}), agent) is False


def test_grid_bounds_project():
    """Test GridBounds clips position to boundaries."""
    model = Model()
    agent = DummyAgent(model)
    c = GridBounds(50, 50)

    projected = c.project(Action("move", {"x": -5, "y": 100}), agent)
    assert projected.params["x"] == 0
    assert projected.params["y"] == 49


# Soft Constraint Tests


def test_soft_constraint_cost():
    """Test EnergyCost returns correct cost."""
    model = Model()
    agent = DummyAgent(model)
    c = EnergyCost(rate=0.5)

    action = Action("move", {"speed": 10})
    cost = c.cost(action, agent)
    assert cost == {"energy": 5.0}


def test_soft_constraint_no_cost_for_irrelevant():
    """Test EnergyCost returns empty for non-move actions."""
    model = Model()
    agent = DummyAgent(model)
    c = EnergyCost(rate=0.5)

    action = Action("speak", {"target": "wolf_2"})
    assert c.cost(action, agent) == {}


def test_soft_constraint_describe():
    """Test dynamic description includes agent state."""
    model = Model()
    agent = DummyAgent(model, energy=42.0)
    c = EnergyCost(rate=0.5)

    desc = c.describe(agent)
    assert "42" in desc


# ActionSpace Tests


def test_actionspace_empty():
    """Test empty ActionSpace passes everything."""
    model = Model()
    agent = DummyAgent(model)
    space = ActionSpace()

    action = Action("move", {"speed": 1000})
    result, report = space.validate(action, agent)
    assert not report.was_modified
    assert result.params["speed"] == 1000


def test_actionspace_add_and_len():
    """Test adding constraints and checking length."""
    space = ActionSpace()
    assert len(space) == 0

    space.add(MaxSpeed(10))
    assert len(space) == 1

    space.add(EnergyCost(0.5))
    assert len(space) == 2


def test_actionspace_add_type_check():
    """Test that add() rejects non-Constraint objects."""
    space = ActionSpace()
    with pytest.raises(TypeError, match="Expected Constraint"):
        space.add("not a constraint")


def test_actionspace_remove():
    """Test removing a specific constraint."""
    space = ActionSpace()
    c = MaxSpeed(10)
    space.add(c)
    assert len(space) == 1

    space.remove(c)
    assert len(space) == 0


def test_actionspace_remove_nonexistent():
    """Test removing a constraint that doesn't exist raises ValueError."""
    space = ActionSpace()
    with pytest.raises(ValueError):
        space.remove(MaxSpeed(10))


def test_actionspace_remove_by_type():
    """Test removing all constraints of a given type."""
    space = ActionSpace()
    space.add(MaxSpeed(10))
    space.add(MaxSpeed(20))
    space.add(EnergyCost(0.5))
    assert len(space) == 3

    removed = space.remove_by_type(MaxSpeed)
    assert removed == 2
    assert len(space) == 1
    assert isinstance(space.constraints[0], EnergyCost)


def test_actionspace_clear():
    """Test clearing all constraints."""
    space = ActionSpace()
    space.add(MaxSpeed(10))
    space.add(EnergyCost(0.5))
    space.clear()
    assert len(space) == 0


def test_actionspace_constraints_property_returns_copy():
    """Test that constraints property returns a copy, not the internal list."""
    space = ActionSpace()
    c = MaxSpeed(10)
    space.add(c)

    constraints = space.constraints
    constraints.clear()  # Modifying the copy
    assert len(space) == 1  # Internal list unaffected


def test_actionspace_repr():
    """Test ActionSpace repr."""
    space = ActionSpace()
    space.add(MaxSpeed(10))
    assert "1" in repr(space)


# Validation Pipeline Tests


def test_validate_hard_constraint_clips():
    """Test that validation clips an action that violates a hard constraint."""
    model = Model()
    agent = DummyAgent(model)
    space = ActionSpace()
    space.add(MaxSpeed(10))

    action = Action("move", {"speed": 20})
    result, report = space.validate(action, agent)

    assert report.was_modified
    assert result.params["speed"] == 10
    assert len(report.reasons) == 1
    assert "MaxSpeed" in report.reasons[0]


def test_validate_hard_constraint_passes():
    """Test that valid actions pass through unchanged."""
    model = Model()
    agent = DummyAgent(model)
    space = ActionSpace()
    space.add(MaxSpeed(10))

    action = Action("move", {"speed": 5})
    result, report = space.validate(action, agent)

    assert not report.was_modified
    assert result.params["speed"] == 5


def test_validate_soft_constraint_scales():
    """Test that soft constraints scale action when agent can't afford it."""
    model = Model()
    agent = DummyAgent(model, energy=5.0)  # Only 5 energy
    space = ActionSpace()
    space.add(EnergyCost(rate=1.0))  # Cost = speed * 1.0

    action = Action("move", {"speed": 10})  # Would cost 10 energy
    result, report = space.validate(action, agent)

    assert report.was_modified
    assert result.params["speed"] == pytest.approx(5.0)  # Scaled to 50%
    assert "Scaled to 50%" in report.reasons[0]


def test_validate_soft_constraint_affordable():
    """Test that affordable actions pass soft constraints unchanged."""
    model = Model()
    agent = DummyAgent(model, energy=100.0)
    space = ActionSpace()
    space.add(EnergyCost(rate=1.0))

    action = Action("move", {"speed": 10})  # Cost 10, have 100
    _result, report = space.validate(action, agent)

    assert not report.was_modified


def test_validate_hard_then_soft():
    """Test the full pipeline: hard clip first, then soft scale."""
    model = Model()
    agent = DummyAgent(model, energy=3.0)
    space = ActionSpace()
    space.add(MaxSpeed(10))  # Hard: clip to 10
    space.add(EnergyCost(rate=1.0))  # Soft: cost = speed * 1.0

    # Agent wants speed 20 → hard clips to 10 → costs 10 → has 3 → scale to 30%
    action = Action("move", {"speed": 20})
    result, report = space.validate(action, agent)

    assert report.was_modified
    assert result.params["speed"] == pytest.approx(3.0)
    assert len(report.reasons) == 2  # One for hard, one for soft


def test_validate_multiple_hard_constraints():
    """Test multiple hard constraints applied in order."""
    model = Model()
    agent = DummyAgent(model)
    space = ActionSpace()
    space.add(MaxSpeed(10))
    space.add(GridBounds(50, 50))

    action = Action("move", {"speed": 20, "x": -5, "y": 100})
    result, report = space.validate(action, agent)

    assert report.was_modified
    assert result.params["speed"] == 10
    assert result.params["x"] == 0
    assert result.params["y"] == 49


def test_validate_preserves_original():
    """Test that the original action in the report is unchanged."""
    model = Model()
    agent = DummyAgent(model)
    space = ActionSpace()
    space.add(MaxSpeed(10))

    action = Action("move", {"speed": 20})
    _, report = space.validate(action, agent)

    assert report.original.params["speed"] == 20  # Original preserved
    assert report.result.params["speed"] == 10  # Result modified


def test_validate_report_costs():
    """Test that the report includes computed costs."""
    model = Model()
    agent = DummyAgent(model, energy=100.0)
    space = ActionSpace()
    space.add(EnergyCost(rate=0.5))

    action = Action("move", {"speed": 10})
    _, report = space.validate(action, agent)

    assert report.costs == {"energy": 5.0}


# ─── Query Interface Tests ─────────────────────────────────────


def test_is_feasible_true():
    """Test is_feasible returns True for valid actions."""
    model = Model()
    agent = DummyAgent(model)
    space = ActionSpace()
    space.add(MaxSpeed(10))

    assert space.is_feasible(Action("move", {"speed": 5}), agent)


def test_is_feasible_false():
    """Test is_feasible returns False for invalid actions."""
    model = Model()
    agent = DummyAgent(model)
    space = ActionSpace()
    space.add(MaxSpeed(10))

    assert not space.is_feasible(Action("move", {"speed": 15}), agent)


def test_get_cost():
    """Test get_cost returns total costs across all constraints."""
    model = Model()
    agent = DummyAgent(model)
    space = ActionSpace()
    space.add(EnergyCost(rate=0.5))
    space.add(EnergyCost(rate=0.3))  # Two energy costs stack

    cost = space.get_cost(Action("move", {"speed": 10}), agent)
    assert cost == {"energy": 8.0}  # (0.5 + 0.3) * 10


def test_describe_all():
    """Test describe_all returns descriptions from all constraints."""
    model = Model()
    agent = DummyAgent(model, energy=42.0)
    space = ActionSpace()
    space.add(MaxSpeed(10))
    space.add(EnergyCost(rate=0.5))

    descriptions = space.describe_all(agent)
    assert len(descriptions) == 2
    assert "10" in descriptions[0]
    assert "42" in descriptions[1]


# ─── Agent Integration Tests ────────────────────────────────────


def test_agent_default_no_action_space():
    """Test that agents have no ActionSpace by default."""
    model = Model()
    agent = Agent(model)
    assert agent.action_space is None


def test_agent_validate_action_no_space():
    """Test validate_action works when no ActionSpace is set."""
    model = Model()
    agent = Agent(model)
    action = Action("move", {"speed": 100})

    result, report = agent.validate_action(action)
    assert not report.was_modified
    assert result is action  # Same object, no modification


def test_agent_validate_action_with_space():
    """Test validate_action delegates to ActionSpace when set."""
    model = Model()
    agent = DummyAgent(model)
    agent.action_space = ActionSpace()
    agent.action_space.add(MaxSpeed(10))

    action = Action("move", {"speed": 20})
    result, report = agent.validate_action(action)

    assert report.was_modified
    assert result.params["speed"] == 10


def test_agent_action_space_per_instance():
    """Test that each agent gets its own independent ActionSpace."""
    model = Model()
    agent1 = DummyAgent(model)
    agent2 = DummyAgent(model)

    agent1.action_space = ActionSpace()
    agent1.action_space.add(MaxSpeed(5))

    agent2.action_space = ActionSpace()
    agent2.action_space.add(MaxSpeed(20))

    action = Action("move", {"speed": 15})

    r1, _ = agent1.validate_action(action.copy())
    r2, _ = agent2.validate_action(action.copy())

    assert r1.params["speed"] == 5  # Agent 1's limit
    assert r2.params["speed"] == 15  # Within agent 2's limit


# Edge Cases


def test_zero_energy_scales_to_zero():
    """Test that zero energy scales action to zero."""
    model = Model()
    agent = DummyAgent(model, energy=0.0)
    space = ActionSpace()
    space.add(EnergyCost(rate=1.0))

    action = Action("move", {"speed": 10})
    result, report = space.validate(action, agent)

    assert report.was_modified
    assert result.params["speed"] == 0.0


def test_non_numeric_params_not_scaled():
    """Test that non-numeric params are not affected by soft scaling."""
    model = Model()
    agent = DummyAgent(model, energy=5.0)
    space = ActionSpace()
    space.add(EnergyCost(rate=1.0))

    action = Action("move", {"speed": 10, "direction": "north"})
    result, _report = space.validate(action, agent)

    assert result.params["direction"] == "north"  # Unchanged string param


def test_agent_without_resource_attribute():
    """Test soft constraint when agent lacks the resource attribute."""
    model = Model()
    agent = Agent(model)  # No 'energy' attribute
    space = ActionSpace()
    space.add(EnergyCost(rate=1.0))

    action = Action("move", {"speed": 10})
    _result, report = space.validate(action, agent)

    # Should pass — no 'energy' attr means no budget to check against
    assert not report.was_modified


def test_constraint_order_matters():
    """Test that constraint order affects projection results."""
    model = Model()
    agent = DummyAgent(model)

    # AlwaysFails replaces all params with {"fixed": True}
    # MaxSpeed clips speed to 10
    space = ActionSpace()
    space.add(AlwaysFails())
    space.add(MaxSpeed(10))

    action = Action("move", {"speed": 20})
    result, _ = space.validate(action, agent)

    # AlwaysFails fires first, replaces params
    assert result.params.get("fixed") is True


def test_empty_params_action():
    """Test validating an action with no params."""
    model = Model()
    agent = DummyAgent(model)
    space = ActionSpace()
    space.add(MaxSpeed(10))

    action = Action("rest")
    _result, report = space.validate(action, agent)
    assert not report.was_modified
