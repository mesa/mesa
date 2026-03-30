"""Main application entry point."""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv

from .api import router
from .utils import config, logger, validate_config

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Mesa LLM Assistant",
    description="LLM-powered simulation assistant for Mesa agent-based modeling framework",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    try:
        logger.info("Starting Mesa LLM Assistant")
        validate_config()
        logger.info("Configuration validated successfully")
        logger.info(f"Available LLM providers: {[p for p in ['openai', 'gemini'] if getattr(config, f'{p}_api_key')]}")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Mesa LLM Assistant")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "Mesa LLM Assistant",
        "version": "1.0.0",
        "description": "LLM-powered simulation assistant for Mesa agent-based modeling",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

def main():
    """Main entry point for running the application."""
    uvicorn.run(
        "mesa_llm.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.api_reload,
        log_level=config.log_level.lower()
    )

if __name__ == "__main__":
    main()