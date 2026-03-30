"""Pydantic models for API requests and responses."""

from typing import Any

from pydantic import BaseModel, Field

from ..prompts import SimulationType
from ..utils import LLMProvider


class GenerateSimulationRequest(BaseModel):
    """Request model for simulation generation."""

    prompt: str = Field(
        ..., description="Natural language description of the simulation"
    )
    llm_provider: LLMProvider | None = Field(
        default=None, description="LLM provider to use"
    )
    simulation_type: SimulationType | None = Field(
        default=None, description="Type of simulation"
    )
    validate_code: bool = Field(
        default=True, description="Whether to validate generated code"
    )


class GenerateSimulationResponse(BaseModel):
    """Response model for simulation generation."""

    success: bool
    code: str | None = None
    metadata: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    error: str | None = None


class DebugCodeRequest(BaseModel):
    """Request model for code debugging."""

    code: str = Field(..., description="Mesa simulation code to debug")
    error_message: str | None = Field(
        default=None, description="Optional error message"
    )
    llm_provider: LLMProvider | None = Field(
        default=None, description="LLM provider to use"
    )
    run_tests: bool = Field(default=True, description="Whether to run execution tests")


class DebugCodeResponse(BaseModel):
    """Response model for code debugging."""

    success: bool
    static_analysis: dict[str, Any] | None = None
    execution_analysis: dict[str, Any] | None = None
    llm_analysis: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None


class ExplainCodeRequest(BaseModel):
    """Request model for code explanation."""

    code: str = Field(..., description="Mesa simulation code to explain")
    focus_area: str | None = Field(
        default=None, description="Specific area to focus on"
    )
    audience_level: str = Field(default="beginner", description="Target audience level")
    llm_provider: LLMProvider | None = Field(
        default=None, description="LLM provider to use"
    )


class ExplainCodeResponse(BaseModel):
    """Response model for code explanation."""

    success: bool
    explanation: dict[str, Any] | None = None
    code_analysis: dict[str, Any] | None = None
    error: str | None = None


class OptimizeCodeRequest(BaseModel):
    """Request model for code optimization."""

    code: str = Field(..., description="Mesa simulation code to optimize")
    focus_areas: list[str] | None = Field(
        default=None, description="Specific optimization areas"
    )
    performance_profile: dict[str, Any] | None = Field(
        default=None, description="Performance profiling data"
    )
    llm_provider: LLMProvider | None = Field(
        default=None, description="LLM provider to use"
    )


class OptimizeCodeResponse(BaseModel):
    """Response model for code optimization."""

    success: bool
    optimization_report: dict[str, Any] | None = None
    optimization_opportunities: list[dict[str, Any]] | None = None
    llm_optimization: dict[str, Any] | None = None
    error: str | None = None


class ExecuteCodeRequest(BaseModel):
    """Request model for code execution."""

    code: str = Field(..., description="Mesa simulation code to execute")
    steps: int = Field(default=10, description="Number of simulation steps to run")
    timeout: int | None = Field(
        default=None, description="Execution timeout in seconds"
    )
    collect_data: bool = Field(
        default=True, description="Whether to collect simulation data"
    )


class ExecuteCodeResponse(BaseModel):
    """Response model for code execution."""

    success: bool
    output: str | None = None
    simulation_data: dict[str, Any] | None = None
    steps_completed: int | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    available_providers: list[str]
    configuration: dict[str, Any]
