"""OpenAI LLM provider implementation."""

from typing import Any

import openai

from ..utils import logger
from .base import BaseLLMProvider, LLMMessage, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.client = openai.AsyncOpenAI(api_key=self.config["api_key"])

    def _validate_config(self) -> None:
        """Validate OpenAI configuration."""
        required_keys = ["api_key", "model"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required OpenAI config key: {key}")

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            # Prepare parameters
            params = {
                "model": self.config["model"],
                "messages": openai_messages,
                "temperature": temperature or self.config.get("temperature", 0.1),
                "max_tokens": max_tokens or self.config.get("max_tokens", 4000),
                **kwargs,
            }

            logger.info(f"Calling OpenAI API with model: {params['model']}")

            # Make API call
            response = await self.client.chat.completions.create(**params)

            # Extract response data
            content = response.choices[0].message.content
            usage = (
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None
            )

            logger.info(f"OpenAI API call successful. Tokens used: {usage}")

            return LLMResponse(
                content=content,
                usage=usage,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
            )

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e!s}")
            raise

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ):
        """Generate streaming response using OpenAI API."""
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            # Prepare parameters
            params = {
                "model": self.config["model"],
                "messages": openai_messages,
                "temperature": temperature or self.config.get("temperature", 0.1),
                "max_tokens": max_tokens or self.config.get("max_tokens", 4000),
                "stream": True,
                **kwargs,
            }

            logger.info(f"Starting OpenAI streaming with model: {params['model']}")

            # Make streaming API call
            stream = await self.client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield LLMResponse(
                        content=chunk.choices[0].delta.content,
                        model=chunk.model,
                        finish_reason=chunk.choices[0].finish_reason,
                    )

        except Exception as e:
            logger.error(f"OpenAI streaming API call failed: {e!s}")
            raise
