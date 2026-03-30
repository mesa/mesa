"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch

from mesa_llm.main import app


@pytest.fixture
def client():
    """Test client for API."""
    return TestClient(app)


@pytest.fixture
def mock_generator():
    """Mock code generator."""
    generator = Mock()
    generator.generate_simulation = AsyncMock()
    return generator


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Mesa LLM Assistant"
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        with patch('mesa_llm.utils.config') as mock_config:
            mock_config.openai_api_key = "test_key"
            mock_config.gemini_api_key = None
            mock_config.default_llm_provider = "openai"
            mock_config.max_execution_time = 30
            mock_config.max_memory_mb = 512
            
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "openai" in data["available_providers"]
    
    @patch('mesa_llm.api.routes.MesaCodeGenerator')
    def test_generate_simulation_success(self, mock_generator_class, client):
        """Test successful simulation generation."""
        # Mock the generator
        mock_generator = Mock()
        mock_generator.generate_simulation = AsyncMock(return_value={
            "code": "import mesa\n\nclass TestAgent(mesa.Agent):\n    pass",
            "metadata": {"classes": ["TestAgent"], "functions": []},
            "validation": {"is_valid": True, "errors": []}
        })
        mock_generator_class.return_value = mock_generator
        
        response = client.post("/api/v1/generate", json={
            "prompt": "Create a simple agent model",
            "llm_provider": "openai"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "import mesa" in data["code"]
        assert "TestAgent" in data["metadata"]["classes"]
    
    @patch('mesa_llm.api.routes.MesaDebugger')
    def test_debug_code_success(self, mock_debugger_class, client):
        """Test successful code debugging."""
        # Mock the debugger
        mock_debugger = Mock()
        mock_debugger.debug_code = AsyncMock(return_value={
            "static_analysis": {"syntax_errors": [], "mesa_issues": []},
            "execution_analysis": {"can_execute": True},
            "llm_analysis": {"analysis": "Code looks good"},
            "summary": {"total_issues": 0, "critical_issues": 0}
        })
        mock_debugger_class.return_value = mock_debugger
        
        response = client.post("/api/v1/debug", json={
            "code": "import mesa\n\nclass TestAgent(mesa.Agent):\n    pass",
            "llm_provider": "openai"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["summary"]["total_issues"] == 0
    
    @patch('mesa_llm.api.routes.MesaExplainer')
    def test_explain_code_success(self, mock_explainer_class, client):
        """Test successful code explanation."""
        # Mock the explainer
        mock_explainer = Mock()
        mock_explainer.explain_simulation = AsyncMock(return_value={
            "explanation": {
                "overview": "This is a simple agent-based model",
                "agents": [],
                "model": None
            },
            "code_analysis": {"agents": [], "model": None}
        })
        mock_explainer_class.return_value = mock_explainer
        
        response = client.post("/api/v1/explain", json={
            "code": "import mesa\n\nclass TestAgent(mesa.Agent):\n    pass",
            "audience_level": "beginner",
            "llm_provider": "openai"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "simple agent-based model" in data["explanation"]["overview"]
    
    @patch('mesa_llm.api.routes.MesaOptimizer')
    def test_optimize_code_success(self, mock_optimizer_class, client):
        """Test successful code optimization."""
        # Mock the optimizer
        mock_optimizer = Mock()
        mock_optimizer.optimize_simulation = AsyncMock(return_value={
            "optimization_report": {
                "optimization_score": 85,
                "total_opportunities": 2,
                "top_recommendations": []
            },
            "optimization_opportunities": [],
            "llm_optimization": {"optimization_suggestions": "Code is well optimized"}
        })
        mock_optimizer_class.return_value = mock_optimizer
        
        response = client.post("/api/v1/optimize", json={
            "code": "import mesa\n\nclass TestAgent(mesa.Agent):\n    pass",
            "llm_provider": "openai"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["optimization_report"]["optimization_score"] == 85
    
    @patch('mesa_llm.api.routes.SafeExecutor')
    def test_execute_code_success(self, mock_executor_class, client):
        """Test successful code execution."""
        # Mock the executor
        mock_executor = Mock()
        mock_executor.run_simulation_steps = Mock(return_value={
            "success": True,
            "output": "Simulation completed successfully",
            "simulation_data": {"model_data": {}, "agent_data": {}},
            "steps_completed": 10
        })
        mock_executor_class.return_value = mock_executor
        
        response = client.post("/api/v1/execute", json={
            "code": "import mesa\n\nclass TestModel(mesa.Model):\n    pass\n\nmodel = TestModel()",
            "steps": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["steps_completed"] == 10
    
    def test_invalid_request_data(self, client):
        """Test API with invalid request data."""
        response = client.post("/api/v1/generate", json={
            "invalid_field": "test"
        })
        
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__])