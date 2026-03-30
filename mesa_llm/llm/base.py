"""Base classes for LLM integration."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class LLMMessage(BaseModel):
    """Represents a message in LLM conversation."""

    role: str  # "system", "user", "assistant"
    content: str


class LLMResponse(BaseModel):
    """Represents an LLM response."""

    content: str
    usage: dict[str, Any] | None = None
    model: str | None = None
    finish_reason: str | None = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the LLM provider with configuration.

        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the provider configuration."""

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLM response
        """

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ):
        """Generate a streaming response from the LLM.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Yields:
            Partial LLM responses
        """

    def create_system_message(self, content: str) -> LLMMessage:
        """Create a system message."""
        return LLMMessage(role="system", content=content)

    def create_user_message(self, content: str) -> LLMMessage:
        """Create a user message."""
        return LLMMessage(role="user", content=content)

    def create_assistant_message(self, content: str) -> LLMMessage:
        """Create an assistant message."""
        return LLMMessage(role="assistant", content=content)
