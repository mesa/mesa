"""LLM provider factory."""

from typing import Any

from ..utils import LLMProvider, get_llm_config
from .base import BaseLLMProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    _providers = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.GEMINI: GeminiProvider,
    }

    @classmethod
    def create_provider(
        self, provider_type: LLMProvider, config: dict[str, Any] = None
    ) -> BaseLLMProvider:
        """Create an LLM provider instance.

        Args:
            provider_type: Type of LLM provider to create
            config: Optional custom configuration (uses default if not provided)

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider type is not supported
        """
        if provider_type not in self._providers:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")

        # Use provided config or get default config
        if config is None:
            config = get_llm_config(provider_type)

        provider_class = self._providers[provider_type]
        return provider_class(config)

    @classmethod
    def get_available_providers(cls) -> list[LLMProvider]:
        """Get list of available LLM providers."""
        return list(cls._providers.keys())

    @classmethod
    def register_provider(
        cls, provider_type: LLMProvider, provider_class: type[BaseLLMProvider]
    ) -> None:
        """Register a new LLM provider.

        Args:
            provider_type: Provider type identifier
            provider_class: Provider class that extends BaseLLMProvider
        """
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError("Provider class must extend BaseLLMProvider")

        cls._providers[provider_type] = provider_class
