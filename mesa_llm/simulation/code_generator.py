"""Mesa simulation code generation."""

import ast
import re
from typing import Any

from ..llm import LLMMessage, LLMProviderFactory
from ..prompts import PromptManager, SimulationType, TaskType
from ..utils import LLMProvider, logger


class CodeValidationError(Exception):
    """Raised when generated code fails validation."""


class MesaCodeGenerator:
    """Generates Mesa simulation code using LLMs."""

    def __init__(self, llm_provider: LLMProvider = LLMProvider.OPENAI):
        self.llm_provider = LLMProviderFactory.create_provider(llm_provider)
        self.prompt_manager = PromptManager()

    async def generate_simulation(
        self,
        user_prompt: str,
        simulation_type: SimulationType | None = None,
        validate_code: bool = True,
    ) -> dict[str, Any]:
        """Generate a complete Mesa simulation from natural language description.

        Args:
            user_prompt: Natural language description of the simulation
            simulation_type: Optional specific simulation type
            validate_code: Whether to validate the generated code

        Returns:
            Dictionary containing generated code and metadata
        """
        try:
            logger.info(f"Generating simulation for prompt: {user_prompt[:100]}...")

            # Create conversation messages
            messages = [
                LLMMessage(
                    role="system",
                    content=self.prompt_manager.get_system_prompt(TaskType.GENERATE),
                ),
                LLMMessage(
                    role="user",
                    content=self.prompt_manager.get_generation_prompt(
                        user_prompt, simulation_type
                    ),
                ),
            ]

            # Generate code using LLM
            response = await self.llm_provider.generate(messages)
            generated_code = self._extract_code_from_response(response.content)

            # Validate code if requested
            validation_results = None
            if validate_code:
                validation_results = self._validate_code(generated_code)
                if not validation_results["is_valid"]:
                    logger.warning(
                        f"Generated code validation failed: {validation_results['errors']}"
                    )

            # Extract metadata from code
            metadata = self._extract_code_metadata(generated_code)

            result = {
                "code": generated_code,
                "metadata": metadata,
                "validation": validation_results,
                "llm_response": {
                    "model": response.model,
                    "usage": response.usage,
                    "finish_reason": response.finish_reason,
                },
            }

            logger.info("Simulation generation completed successfully")
            return result

        except Exception as e:
            logger.error(f"Simulation generation failed: {e!s}")
            raise

    async def generate_agent_class(
        self, agent_description: str, base_class: str = "mesa.discrete_space.CellAgent"
    ) -> str:
        """Generate a specific agent class."""
        prompt = f"""Generate a Mesa agent class with these specifications:

DESCRIPTION: {agent_description}
BASE CLASS: {base_class}

Requirements:
1. Inherit from {base_class}
2. Include proper __init__ method
3. Implement step() method with agent behavior
4. Add comprehensive docstrings
5. Use type hints
6. Handle edge cases appropriately

Generate only the agent class code:"""

        messages = [
            LLMMessage(
                role="system",
                content=self.prompt_manager.get_system_prompt(TaskType.GENERATE),
            ),
            LLMMessage(role="user", content=prompt),
        ]

        response = await self.llm_provider.generate(messages)
        return self._extract_code_from_response(response.content)

    async def generate_model_class(
        self, model_description: str, agent_classes: list[str] = None
    ) -> str:
        """Generate a Mesa model class."""
        agent_info = ""
        if agent_classes:
            agent_info = f"\nAGENT CLASSES TO USE: {', '.join(agent_classes)}"

        prompt = f"""Generate a Mesa model class with these specifications:

DESCRIPTION: {model_description}{agent_info}

Requirements:
1. Inherit from mesa.Model
2. Include proper __init__ method with scenario parameter
3. Initialize grid, agents, and data collection
4. Implement step() method
5. Add comprehensive docstrings
6. Use type hints
7. Follow Mesa best practices

Generate only the model class code:"""

        messages = [
            LLMMessage(
                role="system",
                content=self.prompt_manager.get_system_prompt(TaskType.GENERATE),
            ),
            LLMMessage(role="user", content=prompt),
        ]

        response = await self.llm_provider.generate(messages)
        return self._extract_code_from_response(response.content)

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        code_block_pattern = r"```python\n(.*?)\n```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # If no code blocks, look for code-like content
        lines = response.split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            # Start of code (import statements, class definitions, etc.)
            if line.strip().startswith(("import ", "from ", "class ", "def ", "@")) or (
                line.strip() and not line.strip().startswith(("#", "//", "/*"))
            ):
                in_code = True

            if in_code:
                code_lines.append(line)

        return "\n".join(code_lines).strip()

    def _validate_code(self, code: str) -> dict[str, Any]:
        """Validate generated Python code."""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "mesa_components": [],
        }

        try:
            # Parse AST to check syntax
            tree = ast.parse(code)

            # Check for required Mesa components
            mesa_imports = self._check_mesa_imports(tree)
            validation_result["mesa_components"] = mesa_imports

            # Check for required classes and methods
            classes = self._extract_classes(tree)
            validation_result["classes"] = classes

            # Validate Mesa patterns
            mesa_validation = self._validate_mesa_patterns(tree, code)
            validation_result.update(mesa_validation)

        except SyntaxError as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Syntax error: {e!s}")
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Validation error: {e!s}")

        return validation_result

    def _check_mesa_imports(self, tree: ast.AST) -> list[str]:
        """Check for Mesa-related imports."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "mesa" in alias.name:
                        imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and "mesa" in node.module:
                    imports.append(f"from {node.module}")

        return imports

    def _extract_classes(self, tree: ast.AST) -> list[dict[str, Any]]:
        """Extract class information from AST."""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "bases": [
                        base.id if isinstance(base, ast.Name) else str(base)
                        for base in node.bases
                    ],
                    "methods": [
                        method.name
                        for method in node.body
                        if isinstance(method, ast.FunctionDef)
                    ],
                }
                classes.append(class_info)

        return classes

    def _validate_mesa_patterns(self, tree: ast.AST, code: str) -> dict[str, Any]:
        """Validate Mesa-specific patterns."""
        validation = {"errors": [], "warnings": []}

        classes = self._extract_classes(tree)

        # Check for Agent classes
        agent_classes = [
            cls for cls in classes if any("Agent" in base for base in cls["bases"])
        ]
        for agent_class in agent_classes:
            if "step" not in agent_class["methods"]:
                validation["warnings"].append(
                    f"Agent class {agent_class['name']} missing step() method"
                )

        # Check for Model classes
        model_classes = [
            cls for cls in classes if any("Model" in base for base in cls["bases"])
        ]
        for model_class in model_classes:
            if "step" not in model_class["methods"]:
                validation["warnings"].append(
                    f"Model class {model_class['name']} missing step() method"
                )

        # Check for common Mesa patterns in code
        if "super().__init__" not in code:
            validation["warnings"].append("Missing super().__init__() calls")

        return validation

    def _extract_code_metadata(self, code: str) -> dict[str, Any]:
        """Extract metadata from generated code."""
        metadata = {
            "classes": [],
            "imports": [],
            "functions": [],
            "lines_of_code": len([line for line in code.split("\n") if line.strip()]),
            "has_docstrings": '"""' in code or "'''" in code,
            "has_type_hints": "->" in code or ": " in code,
        }

        try:
            tree = ast.parse(code)

            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    metadata["classes"].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    metadata["functions"].append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        metadata["imports"].extend([alias.name for alias in node.names])
                    else:
                        metadata["imports"].append(node.module or "")

        except:
            pass  # Ignore parsing errors for metadata extraction

        return metadata
