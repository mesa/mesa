"""Google Gemini LLM provider implementation."""

from typing import Any

import google.generativeai as genai

from ..utils import logger
from .base import BaseLLMProvider, LLMMessage, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        genai.configure(api_key=self.config["api_key"])
        self.model = genai.GenerativeModel(self.config["model"])

    def _validate_config(self) -> None:
        """Validate Gemini configuration."""
        required_keys = ["api_key", "model"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required Gemini config key: {key}")

    def _convert_messages_to_gemini_format(self, messages: list[LLMMessage]) -> str:
        """Convert messages to Gemini format.

        Gemini uses a simpler format where we combine system and user messages.
        """
        system_messages = [msg.content for msg in messages if msg.role == "system"]
        user_messages = [msg.content for msg in messages if msg.role == "user"]
        assistant_messages = [
            msg.content for msg in messages if msg.role == "assistant"
        ]

        # Combine system messages as context
        context = "\n\n".join(system_messages) if system_messages else ""

        # Build conversation history
        conversation = []
        if context:
            conversation.append(f"Context: {context}")

        # Add conversation history (simplified for Gemini)
        for i, (user_msg, assistant_msg) in enumerate(
            zip(user_messages[:-1], assistant_messages)
        ):
            conversation.append(f"User: {user_msg}")
            conversation.append(f"Assistant: {assistant_msg}")

        # Add current user message
        if user_messages:
            conversation.append(f"User: {user_messages[-1]}")

        return "\n\n".join(conversation)

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate response using Gemini API."""
        try:
            # Convert messages to Gemini format
            prompt = self._convert_messages_to_gemini_format(messages)

            # Prepare generation config
            generation_config = genai.types.GenerationConfig(
                temperature=temperature or self.config.get("temperature", 0.1),
                max_output_tokens=max_tokens or self.config.get("max_tokens", 4000),
            )

            logger.info(f"Calling Gemini API with model: {self.config['model']}")

            # Make API call
            response = await self.model.generate_content_async(
                prompt, generation_config=generation_config
            )

            logger.info("Gemini API call successful")

            return LLMResponse(
                content=response.text,
                model=self.config["model"],
                finish_reason="stop",  # Gemini doesn't provide detailed finish reasons
            )

        except Exception as e:
            logger.error(f"Gemini API call failed: {e!s}")
            raise

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ):
        """Generate streaming response using Gemini API."""
        try:
            # Convert messages to Gemini format
            prompt = self._convert_messages_to_gemini_format(messages)

            # Prepare generation config
            generation_config = genai.types.GenerationConfig(
                temperature=temperature or self.config.get("temperature", 0.1),
                max_output_tokens=max_tokens or self.config.get("max_tokens", 4000),
            )

            logger.info(f"Starting Gemini streaming with model: {self.config['model']}")

            # Make streaming API call
            response = await self.model.generate_content_async(
                prompt, generation_config=generation_config, stream=True
            )

            async for chunk in response:
                if chunk.text:
                    yield LLMResponse(
                        content=chunk.text,
                        model=self.config["model"],
                        finish_reason=None,
                    )

        except Exception as e:
            logger.error(f"Gemini streaming API call failed: {e!s}")
            raise
