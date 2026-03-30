"""Configuration management for Mesa LLM Assistant."""

from enum import Enum
from typing import Any

from pydantic import BaseSettings, Field


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    GEMINI = "gemini"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Config(BaseSettings):
    """Application configuration."""

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=False, env="API_RELOAD")

    # LLM Configuration
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    gemini_api_key: str | None = Field(default=None, env="GEMINI_API_KEY")
    default_llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI, env="DEFAULT_LLM_PROVIDER"
    )

    # OpenAI specific
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=4000, env="OPENAI_MAX_TOKENS")

    # Gemini specific
    gemini_model: str = Field(default="gemini-pro", env="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.1, env="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=4000, env="GEMINI_MAX_TOKENS")

    # Code Execution Safety
    max_execution_time: int = Field(default=30, env="MAX_EXECUTION_TIME")  # seconds
    max_memory_mb: int = Field(default=512, env="MAX_MEMORY_MB")
    allowed_imports: list[str] = Field(
        default=[
            "mesa",
            "numpy",
            "pandas",
            "matplotlib",
            "random",
            "math",
            "collections",
            "itertools",
            "enum",
            "typing",
            "dataclasses",
        ]
    )

    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # seconds

    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_file: str | None = Field(default=None, env="LOG_FILE")

    # File paths
    examples_dir: str = Field(default="examples", env="EXAMPLES_DIR")
    templates_dir: str = Field(default="prompts/templates", env="TEMPLATES_DIR")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global configuration instance
config = Config()


def get_llm_config(provider: LLMProvider) -> dict[str, Any]:
    """Get configuration for specific LLM provider."""
    if provider == LLMProvider.OPENAI:
        return {
            "api_key": config.openai_api_key,
            "model": config.openai_model,
            "temperature": config.openai_temperature,
            "max_tokens": config.openai_max_tokens,
        }
    elif provider == LLMProvider.GEMINI:
        return {
            "api_key": config.gemini_api_key,
            "model": config.gemini_model,
            "temperature": config.gemini_temperature,
            "max_tokens": config.gemini_max_tokens,
        }
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def validate_config() -> None:
    """Validate configuration and raise errors for missing required settings."""
    errors = []

    if not config.openai_api_key and not config.gemini_api_key:
        errors.append(
            "At least one LLM API key must be provided (OPENAI_API_KEY or GEMINI_API_KEY)"
        )

    if config.default_llm_provider == LLMProvider.OPENAI and not config.openai_api_key:
        errors.append(
            "OPENAI_API_KEY is required when using OpenAI as default provider"
        )

    if config.default_llm_provider == LLMProvider.GEMINI and not config.gemini_api_key:
        errors.append(
            "GEMINI_API_KEY is required when using Gemini as default provider"
        )

    if errors:
        raise ValueError(
            "Configuration errors:\n" + "\n".join(f"- {error}" for error in errors)
        )
