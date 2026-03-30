"""Prompt management and template system."""

from enum import Enum

from .templates.generation_prompts import (
    BASIC_SIMULATION_TEMPLATE,
    CUSTOM_SIMULATION_TEMPLATE,
    ECONOMIC_MODEL_TEMPLATE,
    PREDATOR_PREY_TEMPLATE,
    SOCIAL_DYNAMICS_TEMPLATE,
)
from .templates.system_prompts import (
    CODE_GENERATION_SYSTEM_PROMPT,
    DEBUGGING_SYSTEM_PROMPT,
    EXPLANATION_SYSTEM_PROMPT,
    MESA_EXPERT_SYSTEM_PROMPT,
    OPTIMIZATION_SYSTEM_PROMPT,
)


class TaskType(str, Enum):
    """Types of tasks the system can perform."""

    GENERATE = "generate"
    DEBUG = "debug"
    EXPLAIN = "explain"
    OPTIMIZE = "optimize"


class SimulationType(str, Enum):
    """Types of simulations that can be generated."""

    BASIC = "basic"
    PREDATOR_PREY = "predator_prey"
    SOCIAL_DYNAMICS = "social_dynamics"
    ECONOMIC = "economic"
    CUSTOM = "custom"


class PromptManager:
    """Manages prompt templates and generation."""

    def __init__(self):
        self.system_prompts = {
            TaskType.GENERATE: CODE_GENERATION_SYSTEM_PROMPT,
            TaskType.DEBUG: DEBUGGING_SYSTEM_PROMPT,
            TaskType.EXPLAIN: EXPLANATION_SYSTEM_PROMPT,
            TaskType.OPTIMIZE: OPTIMIZATION_SYSTEM_PROMPT,
        }

        self.generation_templates = {
            SimulationType.BASIC: BASIC_SIMULATION_TEMPLATE,
            SimulationType.PREDATOR_PREY: PREDATOR_PREY_TEMPLATE,
            SimulationType.SOCIAL_DYNAMICS: SOCIAL_DYNAMICS_TEMPLATE,
            SimulationType.ECONOMIC: ECONOMIC_MODEL_TEMPLATE,
            SimulationType.CUSTOM: CUSTOM_SIMULATION_TEMPLATE,
        }

    def get_system_prompt(self, task_type: TaskType) -> str:
        """Get system prompt for a specific task type."""
        base_prompt = MESA_EXPERT_SYSTEM_PROMPT
        task_prompt = self.system_prompts.get(task_type, "")
        return f"{base_prompt}\n\n{task_prompt}"

    def get_generation_prompt(
        self, user_prompt: str, simulation_type: SimulationType | None = None
    ) -> str:
        """Get generation prompt for creating simulations."""
        # Determine simulation type if not provided
        if simulation_type is None:
            simulation_type = self._classify_simulation_type(user_prompt)

        template = self.generation_templates[simulation_type]
        return template.format(user_prompt=user_prompt)

    def get_debug_prompt(self, code: str, error_message: str | None = None) -> str:
        """Get debugging prompt for analyzing code."""
        prompt = f"Analyze this Mesa simulation code for errors and improvements:\n\n```python\n{code}\n```"

        if error_message:
            prompt += f"\n\nERROR MESSAGE:\n{error_message}"

        prompt += "\n\nProvide a detailed analysis and corrected code."
        return prompt

    def get_explanation_prompt(self, code: str, focus_area: str | None = None) -> str:
        """Get explanation prompt for describing simulations."""
        prompt = (
            f"Explain this Mesa simulation in simple terms:\n\n```python\n{code}\n```"
        )

        if focus_area:
            prompt += f"\n\nFocus particularly on: {focus_area}"

        return prompt

    def get_optimization_prompt(
        self, code: str, performance_issues: str | None = None
    ) -> str:
        """Get optimization prompt for improving code."""
        prompt = f"Analyze and optimize this Mesa simulation:\n\n```python\n{code}\n```"

        if performance_issues:
            prompt += f"\n\nKNOWN PERFORMANCE ISSUES:\n{performance_issues}"

        prompt += "\n\nProvide optimized code with explanations of improvements."
        return prompt

    def _classify_simulation_type(self, user_prompt: str) -> SimulationType:
        """Classify the type of simulation based on user prompt."""
        prompt_lower = user_prompt.lower()

        # Keywords for different simulation types
        predator_prey_keywords = [
            "predator",
            "prey",
            "wolf",
            "sheep",
            "hunt",
            "eat",
            "food chain",
            "ecosystem",
            "population dynamics",
            "species",
        ]

        social_keywords = [
            "social",
            "opinion",
            "influence",
            "network",
            "community",
            "culture",
            "segregation",
            "schelling",
            "voter",
            "consensus",
            "polarization",
        ]

        economic_keywords = [
            "economic",
            "market",
            "trade",
            "wealth",
            "money",
            "price",
            "supply",
            "demand",
            "auction",
            "exchange",
            "financial",
            "economy",
        ]

        # Check for specific simulation types
        if any(keyword in prompt_lower for keyword in predator_prey_keywords):
            return SimulationType.PREDATOR_PREY
        elif any(keyword in prompt_lower for keyword in social_keywords):
            return SimulationType.SOCIAL_DYNAMICS
        elif any(keyword in prompt_lower for keyword in economic_keywords):
            return SimulationType.ECONOMIC
        else:
            return SimulationType.CUSTOM

    def create_conversation_messages(
        self, task_type: TaskType, user_prompt: str, **kwargs
    ) -> list[dict[str, str]]:
        """Create a complete conversation for LLM interaction."""
        messages = []

        # Add system prompt
        system_prompt = self.get_system_prompt(task_type)
        messages.append({"role": "system", "content": system_prompt})

        # Add user prompt based on task type
        if task_type == TaskType.GENERATE:
            simulation_type = kwargs.get("simulation_type")
            user_content = self.get_generation_prompt(user_prompt, simulation_type)
        elif task_type == TaskType.DEBUG:
            code = kwargs.get("code", "")
            error_message = kwargs.get("error_message")
            user_content = self.get_debug_prompt(code, error_message)
        elif task_type == TaskType.EXPLAIN:
            code = kwargs.get("code", "")
            focus_area = kwargs.get("focus_area")
            user_content = self.get_explanation_prompt(code, focus_area)
        elif task_type == TaskType.OPTIMIZE:
            code = kwargs.get("code", "")
            performance_issues = kwargs.get("performance_issues")
            user_content = self.get_optimization_prompt(code, performance_issues)
        else:
            user_content = user_prompt

        messages.append({"role": "user", "content": user_content})

        return messages
