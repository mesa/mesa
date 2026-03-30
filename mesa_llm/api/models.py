"""Pydantic models for API requests and responses."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from ..utils import LLMProvider
from ..prompts import SimulationType


class GenerateSimulationRequest(BaseModel):
    """Request model for simulation generation."""
    prompt: str = Field(..., description="Natural language description of the simulation")
    llm_provider: Optional[LLMProvider] = Field(default=None, description="LLM provider to use")
    simulation_type: Optional[SimulationType] = Field(default=None, description="Type of simulation")
    validate_code: bool = Field(default=True, description="Whether to validate generated code")


class GenerateSimulationResponse(BaseModel):
    """Response model for simulation generation."""
    success: bool
    code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DebugCodeRequest(BaseModel):
    """Request model for code debugging."""
    code: str = Field(..., description="Mesa simulation code to debug")
    error_message: Optional[str] = Field(default=None, description="Optional error message")
    llm_provider: Optional[LLMProvider] = Field(default=None, description="LLM provider to use")
    run_tests: bool = Field(default=True, description="Whether to run execution tests")


class DebugCodeResponse(BaseModel):
    """Response model for code debugging."""
    success: bool
    static_analysis: Optional[Dict[str, Any]] = None
    execution_analysis: Optional[Dict[str, Any]] = None
    llm_analysis: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ExplainCodeRequest(BaseModel):
    """Request model for code explanation."""
    code: str = Field(..., description="Mesa simulation code to explain")
    focus_area: Optional[str] = Field(default=None, description="Specific area to focus on")
    audience_level: str = Field(default="beginner", description="Target audience level")
    llm_provider: Optional[LLMProvider] = Field(default=None, description="LLM provider to use")


class ExplainCodeResponse(BaseModel):
    """Response model for code explanation."""
    success: bool
    explanation: Optional[Dict[str, Any]] = None
    code_analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class OptimizeCodeRequest(BaseModel):
    """Request model for code optimization."""
    code: str = Field(..., description="Mesa simulation code to optimize")
    focus_areas: Optional[List[str]] = Field(default=None, description="Specific optimization areas")
    performance_profile: Optional[Dict[str, Any]] = Field(default=None, description="Performance profiling data")
    llm_provider: Optional[LLMProvider] = Field(default=None, description="LLM provider to use")


class OptimizeCodeResponse(BaseModel):
    """Response model for code optimization."""
    success: bool
    optimization_report: Optional[Dict[str, Any]] = None
    optimization_opportunities: Optional[List[Dict[str, Any]]] = None
    llm_optimization: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ExecuteCodeRequest(BaseModel):
    """Request model for code execution."""
    code: str = Field(..., description="Mesa simulation code to execute")
    steps: int = Field(default=10, description="Number of simulation steps to run")
    timeout: Optional[int] = Field(default=None, description="Execution timeout in seconds")
    collect_data: bool = Field(default=True, description="Whether to collect simulation data")


class ExecuteCodeResponse(BaseModel):
    """Response model for code execution."""
    success: bool
    output: Optional[str] = None
    simulation_data: Optional[Dict[str, Any]] = None
    steps_completed: Optional[int] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    available_providers: List[str]
    configuration: Dict[str, Any]