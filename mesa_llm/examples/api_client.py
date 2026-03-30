"""Example API client for Mesa LLM Assistant."""

import requests
import json
import time
from typing import Dict, Any


class MesaLLMClient:
    """Client for interacting with Mesa LLM Assistant API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api/v1"
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = requests.get(f"{self.api_base}/health")
        response.raise_for_status()
        return response.json()
    
    def generate_simulation(
        self,
        prompt: str,
        llm_provider: str = "openai",
        simulation_type: str = None,
        validate_code: bool = True
    ) -> Dict[str, Any]:
        """Generate a Mesa simulation from natural language."""
        data = {
            "prompt": prompt,
            "llm_provider": llm_provider,
            "validate_code": validate_code
        }
        if simulation_type:
            data["simulation_type"] = simulation_type
        
        response = requests.post(f"{self.api_base}/generate", json=data)
        response.raise_for_status()
        return response.json()
    
    def debug_code(
        self,
        code: str,
        error_message: str = None,
        llm_provider: str = "openai",
        run_tests: bool = True
    ) -> Dict[str, Any]:
        """Debug Mesa simulation code."""
        data = {
            "code": code,
            "llm_provider": llm_provider,
            "run_tests": run_tests
        }
        if error_message:
            data["error_message"] = error_message
        
        response = requests.post(f"{self.api_base}/debug", json=data)
        response.raise_for_status()
        return response.json()
    
    def explain_code(
        self,
        code: str,
        focus_area: str = None,
        audience_level: str = "beginner",
        llm_provider: str = "openai"
    ) -> Dict[str, Any]:
        """Explain Mesa simulation code."""
        data = {
            "code": code,
            "audience_level": audience_level,
            "llm_provider": llm_provider
        }
        if focus_area:
            data["focus_area"] = focus_area
        
        response = requests.post(f"{self.api_base}/explain", json=data)
        response.raise_for_status()
        return response.json()
    
    def optimize_code(
        self,
        code: str,
        focus_areas: list = None,
        llm_provider: str = "openai"
    ) -> Dict[str, Any]:
        """Optimize Mesa simulation code."""
        data = {
            "code": code,
            "llm_provider": llm_provider
        }
        if focus_areas:
            data["focus_areas"] = focus_areas
        
        response = requests.post(f"{self.api_base}/optimize", json=data)
        response.raise_for_status()
        return response.json()
    
    def execute_code(
        self,
        code: str,
        steps: int = 10,
        collect_data: bool = True
    ) -> Dict[str, Any]:
        """Execute Mesa simulation code."""
        data = {
            "code": code,
            "steps": steps,
            "collect_data": collect_data
        }
        
        response = requests.post(f"{self.api_base}/execute", json=data)
        response.raise_for_status()
        return response.json()


def example_workflow():
    """Example workflow using the API client."""
    client = MesaLLMClient()
    
    print("🔍 Checking API health...")
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Available providers: {health['available_providers']}")
    
    print("\n🤖 Generating simulation...")
    generation_result = client.generate_simulation(
        prompt="Create a simple wealth distribution model with 100 agents",
        llm_provider="openai"
    )
    
    if generation_result["success"]:
        print("✅ Simulation generated successfully!")
        generated_code = generation_result["code"]
        print(f"Generated {len(generated_code)} characters of code")
        
        print("\n📖 Explaining the simulation...")
        explanation_result = client.explain_code(
            code=generated_code,
            audience_level="beginner"
        )
        
        if explanation_result["success"]:
            print("✅ Explanation generated!")
            print("Overview:", explanation_result["explanation"]["overview"][:200] + "...")
        
        print("\n🔧 Optimizing the code...")
        optimization_result = client.optimize_code(
            code=generated_code,
            focus_areas=["performance", "readability"]
        )
        
        if optimization_result["success"]:
            score = optimization_result["optimization_report"]["optimization_score"]
            print(f"✅ Optimization complete! Score: {score}/100")
            
            opportunities = optimization_result["optimization_report"]["total_opportunities"]
            print(f"Found {opportunities} optimization opportunities")
        
        print("\n▶️ Executing the simulation...")
        execution_result = client.execute_code(
            code=generated_code,
            steps=5
        )
        
        if execution_result["success"]:
            print("✅ Simulation executed successfully!")
            print(f"Completed {execution_result['steps_completed']} steps")
        else:
            print(f"❌ Execution failed: {execution_result['error']}")
            
            print("\n🐛 Debugging the code...")
            debug_result = client.debug_code(
                code=generated_code,
                error_message=execution_result['error']
            )
            
            if debug_result["success"]:
                issues = debug_result["summary"]["total_issues"]
                print(f"Found {issues} issues in the code")
    
    else:
        print(f"❌ Generation failed: {generation_result['error']}")


if __name__ == "__main__":
    try:
        example_workflow()
        print("\n🎉 Workflow completed!")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")