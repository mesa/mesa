"""API routes for Mesa LLM Assistant."""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..analysis import MesaDebugger, MesaExplainer, MesaOptimizer
from ..simulation import MesaCodeGenerator, SafeExecutor
from ..utils import LLMProvider, config, logger, validate_config
from .models import (
    DebugCodeRequest,
    DebugCodeResponse,
    ExecuteCodeRequest,
    ExecuteCodeResponse,
    ExplainCodeRequest,
    ExplainCodeResponse,
    GenerateSimulationRequest,
    GenerateSimulationResponse,
    HealthResponse,
    OptimizeCodeRequest,
    OptimizeCodeResponse,
)

# Create router
router = APIRouter()


# Dependency to get default LLM provider
def get_llm_provider(provider: LLMProvider | None = None) -> LLMProvider:
    """Get LLM provider with fallback to default."""
    return provider or config.default_llm_provider


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        validate_config()

        available_providers = []
        if config.openai_api_key:
            available_providers.append("openai")
        if config.gemini_api_key:
            available_providers.append("gemini")

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            available_providers=available_providers,
            configuration={
                "default_provider": config.default_llm_provider,
                "max_execution_time": config.max_execution_time,
                "max_memory_mb": config.max_memory_mb,
            },
        )
    except Exception as e:
        logger.error(f"Health check failed: {e!s}")
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {e!s}")


@router.post("/generate", response_model=GenerateSimulationResponse)
async def generate_simulation(
    request: GenerateSimulationRequest,
    llm_provider: LLMProvider = Depends(get_llm_provider),
):
    """Generate a Mesa simulation from natural language description."""
    try:
        logger.info(
            f"Generating simulation with provider: {request.llm_provider or llm_provider}"
        )

        generator = MesaCodeGenerator(request.llm_provider or llm_provider)

        result = await generator.generate_simulation(
            user_prompt=request.prompt,
            simulation_type=request.simulation_type,
            validate_code=request.validate_code,
        )

        return GenerateSimulationResponse(
            success=True,
            code=result["code"],
            metadata=result["metadata"],
            validation=result["validation"],
        )

    except Exception as e:
        logger.error(f"Simulation generation failed: {e!s}")
        return GenerateSimulationResponse(success=False, error=str(e))


@router.post("/debug", response_model=DebugCodeResponse)
async def debug_code(
    request: DebugCodeRequest, llm_provider: LLMProvider = Depends(get_llm_provider)
):
    """Debug Mesa simulation code."""
    try:
        logger.info("Starting code debugging")

        debugger = MesaDebugger(request.llm_provider or llm_provider)

        result = await debugger.debug_code(
            code=request.code,
            error_message=request.error_message,
            run_tests=request.run_tests,
        )

        return DebugCodeResponse(
            success=True,
            static_analysis=result["static_analysis"],
            execution_analysis=result["execution_analysis"],
            llm_analysis=result["llm_analysis"],
            summary=result["summary"],
        )

    except Exception as e:
        logger.error(f"Code debugging failed: {e!s}")
        return DebugCodeResponse(success=False, error=str(e))


@router.post("/explain", response_model=ExplainCodeResponse)
async def explain_code(
    request: ExplainCodeRequest, llm_provider: LLMProvider = Depends(get_llm_provider)
):
    """Explain Mesa simulation code in simple terms."""
    try:
        logger.info(f"Explaining code for audience: {request.audience_level}")

        explainer = MesaExplainer(request.llm_provider or llm_provider)

        result = await explainer.explain_simulation(
            code=request.code,
            focus_area=request.focus_area,
            audience_level=request.audience_level,
        )

        return ExplainCodeResponse(
            success=True,
            explanation=result["explanation"],
            code_analysis=result["code_analysis"],
        )

    except Exception as e:
        logger.error(f"Code explanation failed: {e!s}")
        return ExplainCodeResponse(success=False, error=str(e))


@router.post("/optimize", response_model=OptimizeCodeResponse)
async def optimize_code(
    request: OptimizeCodeRequest, llm_provider: LLMProvider = Depends(get_llm_provider)
):
    """Optimize Mesa simulation code for performance and best practices."""
    try:
        logger.info("Starting code optimization")

        optimizer = MesaOptimizer(request.llm_provider or llm_provider)

        result = await optimizer.optimize_simulation(
            code=request.code,
            focus_areas=request.focus_areas,
            performance_profile=request.performance_profile,
        )

        return OptimizeCodeResponse(
            success=True,
            optimization_report=result["optimization_report"],
            optimization_opportunities=result["optimization_opportunities"],
            llm_optimization=result["llm_optimization"],
        )

    except Exception as e:
        logger.error(f"Code optimization failed: {e!s}")
        return OptimizeCodeResponse(success=False, error=str(e))


@router.post("/execute", response_model=ExecuteCodeResponse)
async def execute_code(request: ExecuteCodeRequest):
    """Execute Mesa simulation code safely."""
    try:
        logger.info(f"Executing code for {request.steps} steps")

        executor = SafeExecutor()

        result = executor.run_simulation_steps(
            code=request.code, steps=request.steps, collect_data=request.collect_data
        )

        return ExecuteCodeResponse(
            success=result["success"],
            output=result.get("output"),
            simulation_data=result.get("simulation_data"),
            steps_completed=result.get("steps_completed"),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Code execution failed: {e!s}")
        return ExecuteCodeResponse(success=False, error=str(e))


@router.post("/generate/stream")
async def generate_simulation_stream(
    request: GenerateSimulationRequest,
    llm_provider: LLMProvider = Depends(get_llm_provider),
):
    """Generate a Mesa simulation with streaming response."""
    try:
        from ..llm import LLMProviderFactory
        from ..prompts import PromptManager, TaskType

        provider = LLMProviderFactory.create_provider(
            request.llm_provider or llm_provider
        )
        prompt_manager = PromptManager()

        messages = [
            provider.create_system_message(
                prompt_manager.get_system_prompt(TaskType.GENERATE)
            ),
            provider.create_user_message(
                prompt_manager.get_generation_prompt(
                    request.prompt, request.simulation_type
                )
            ),
        ]

        async def generate_stream():
            try:
                async for chunk in provider.generate_stream(messages):
                    yield f"data: {json.dumps({'content': chunk.content, 'done': False})}\n\n"
                yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    except Exception as e:
        logger.error(f"Streaming generation failed: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))
