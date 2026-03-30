"""Tests for Mesa code generator."""

from unittest.mock import AsyncMock, Mock

import pytest

from mesa_llm.llm import LLMResponse
from mesa_llm.simulation import MesaCodeGenerator
from mesa_llm.utils import LLMProvider


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    provider = Mock()
    provider.generate = AsyncMock()
    return provider


@pytest.fixture
def code_generator(mock_llm_provider):
    """Code generator with mocked LLM provider."""
    generator = MesaCodeGenerator(LLMProvider.OPENAI)
    generator.llm_provider = mock_llm_provider
    return generator


class TestMesaCodeGenerator:
    """Test cases for MesaCodeGenerator."""

    @pytest.mark.asyncio
    async def test_generate_simulation_success(self, code_generator, mock_llm_provider):
        """Test successful simulation generation."""
        # Mock LLM response with valid Mesa code
        mock_code = """
import mesa
from mesa.discrete_space import OrthogonalMooreGrid

class TestAgent(mesa.discrete_space.CellAgent):
    def __init__(self, model):
        super().__init__(model)
        self.energy = 10

    def step(self):
        self.energy -= 1

class TestModel(mesa.Model):
    def __init__(self):
        super().__init__()
        self.grid = OrthogonalMooreGrid((10, 10))
        TestAgent(self)

    def step(self):
        self.agents.do("step")
"""

        mock_llm_provider.generate.return_value = LLMResponse(
            content=f"```python\n{mock_code}\n```",
            model="gpt-4",
            usage={"total_tokens": 500},
        )

        result = await code_generator.generate_simulation(
            "Create a simple agent model", validate_code=True
        )

        assert result["code"] == mock_code.strip()
        assert result["validation"]["is_valid"] is True
        assert "TestAgent" in result["metadata"]["classes"]
        assert "TestModel" in result["metadata"]["classes"]

    @pytest.mark.asyncio
    async def test_generate_simulation_invalid_code(
        self, code_generator, mock_llm_provider
    ):
        """Test generation with invalid code."""
        # Mock LLM response with invalid code
        invalid_code = """
import mesa

class BadAgent(mesa.Agent):
    # Missing __init__ and step methods
    pass

# Missing Model class
"""

        mock_llm_provider.generate.return_value = LLMResponse(
            content=f"```python\n{invalid_code}\n```", model="gpt-4"
        )

        result = await code_generator.generate_simulation(
            "Create a bad model", validate_code=True
        )

        assert result["validation"]["is_valid"] is False
        assert len(result["validation"]["errors"]) > 0

    def test_extract_code_from_response(self, code_generator):
        """Test code extraction from LLM response."""
        response_with_code_block = """
Here's your Mesa simulation:

```python
import mesa

class MyAgent(mesa.Agent):
    def step(self):
        pass
```

This creates a basic agent.
"""

        extracted = code_generator._extract_code_from_response(response_with_code_block)
        expected = "import mesa\n\nclass MyAgent(mesa.Agent):\n    def step(self):\n        pass"
        assert extracted == expected

    def test_validate_code_syntax_error(self, code_generator):
        """Test code validation with syntax error."""
        invalid_code = "import mesa\nclass BadClass\n    pass"  # Missing colon

        validation = code_generator._validate_code(invalid_code)

        assert validation["is_valid"] is False
        assert any("syntax error" in error.lower() for error in validation["errors"])

    def test_validate_code_mesa_patterns(self, code_generator):
        """Test Mesa pattern validation."""
        code_without_step = """
import mesa

class MyAgent(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
    # Missing step method

class MyModel(mesa.Model):
    def __init__(self):
        super().__init__()
    # Missing step method
"""

        validation = code_generator._validate_code(code_without_step)

        # Should have warnings about missing step methods
        assert len(validation["warnings"]) >= 2
        assert any("step" in warning for warning in validation["warnings"])


if __name__ == "__main__":
    pytest.main([__file__])
