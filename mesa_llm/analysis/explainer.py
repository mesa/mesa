"""Mesa simulation explanation and documentation generator."""

import ast
import re
from typing import Dict, Any, List, Optional
from ..llm import LLMProviderFactory, LLMMessage
from ..prompts import PromptManager, TaskType
from ..utils import LLMProvider, logger


class MesaExplainer:
    """Explains Mesa simulations in simple terms."""
    
    def __init__(self, llm_provider: LLMProvider = LLMProvider.OPENAI):
        self.llm_provider = LLMProviderFactory.create_provider(llm_provider)
        self.prompt_manager = PromptManager()
    
    async def explain_simulation(
        self,
        code: str,
        focus_area: Optional[str] = None,
        audience_level: str = "beginner"
    ) -> Dict[str, Any]:
        """Generate a comprehensive explanation of a Mesa simulation.
        
        Args:
            code: Mesa simulation code to explain
            focus_area: Optional specific area to focus on
            audience_level: Target audience (beginner, intermediate, advanced)
            
        Returns:
            Detailed explanation of the simulation
        """
        logger.info(f"Generating explanation for simulation (audience: {audience_level})")
        
        # Analyze code structure
        code_analysis = self._analyze_code_structure(code)
        
        # Generate LLM explanation
        llm_explanation = await self._generate_llm_explanation(
            code, focus_area, audience_level, code_analysis
        )
        
        # Create structured explanation
        structured_explanation = self._create_structured_explanation(
            code_analysis, llm_explanation
        )
        
        return {
            "code_analysis": code_analysis,
            "explanation": structured_explanation,
            "llm_response": llm_explanation
        }
    
    def _analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyze the structure of Mesa simulation code."""
        analysis = {
            "agents": [],
            "model": None,
            "scenario": None,
            "imports": [],
            "key_concepts": [],
            "data_collection": False,
            "spatial_components": [],
            "parameters": []
        }
        
        try:
            tree = ast.parse(code)
            
            # Analyze classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node, code)
                    
                    if self._is_agent_class(class_info):
                        analysis["agents"].append(class_info)
                    elif self._is_model_class(class_info):
                        analysis["model"] = class_info
                    elif self._is_scenario_class(class_info):
                        analysis["scenario"] = class_info
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports = self._extract_imports(node)
                    analysis["imports"].extend(imports)
            
            # Identify key concepts
            analysis["key_concepts"] = self._identify_key_concepts(code, analysis)
            
            # Check for data collection
            analysis["data_collection"] = "DataCollector" in code or "datacollector" in code
            
            # Identify spatial components
            analysis["spatial_components"] = self._identify_spatial_components(code)
            
        except SyntaxError:
            logger.warning("Could not parse code for structure analysis")
        
        return analysis
    
    def _analyze_class(self, node: ast.ClassDef, code: str) -> Dict[str, Any]:
        """Analyze a class definition."""
        class_info = {
            "name": node.name,
            "bases": [self._get_node_name(base) for base in node.bases],
            "methods": [],
            "attributes": [],
            "docstring": self._extract_docstring(node)
        }
        
        # Analyze methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = {
                    "name": item.name,
                    "args": [arg.arg for arg in item.args.args if arg.arg != 'self'],
                    "docstring": self._extract_docstring(item)
                }
                class_info["methods"].append(method_info)
            elif isinstance(item, ast.Assign):
                # Extract class attributes
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info["attributes"].append(target.id)
        
        return class_info
    
    def _is_agent_class(self, class_info: Dict[str, Any]) -> bool:
        """Check if a class is an agent class."""
        return any("Agent" in base for base in class_info["bases"])
    
    def _is_model_class(self, class_info: Dict[str, Any]) -> bool:
        """Check if a class is a model class."""
        return any("Model" in base for base in class_info["bases"])
    
    def _is_scenario_class(self, class_info: Dict[str, Any]) -> bool:
        """Check if a class is a scenario class."""
        return any("Scenario" in base for base in class_info["bases"])
    
    def _extract_imports(self, node: ast.AST) -> List[str]:
        """Extract import information."""
        imports = []
        if isinstance(node, ast.Import):
            imports.extend([alias.name for alias in node.names])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(f"from {module}")
        return imports
    
    def _identify_key_concepts(self, code: str, analysis: Dict[str, Any]) -> List[str]:
        """Identify key Mesa concepts used in the code."""
        concepts = []
        
        # Spatial concepts
        spatial_keywords = {
            "Grid": "Spatial grid for agent movement",
            "Cell": "Individual grid cells",
            "neighborhood": "Agent neighborhoods",
            "move": "Agent movement"
        }
        
        # Agent concepts
        agent_keywords = {
            "step": "Agent behavior per time step",
            "AgentSet": "Collections of agents",
            "activate": "Agent activation patterns"
        }
        
        # Model concepts
        model_keywords = {
            "DataCollector": "Data collection and analysis",
            "schedule": "Event scheduling",
            "run_for": "Running simulations"
        }
        
        all_keywords = {**spatial_keywords, **agent_keywords, **model_keywords}
        
        for keyword, description in all_keywords.items():
            if keyword.lower() in code.lower():
                concepts.append({"concept": keyword, "description": description})
        
        return concepts
    
    def _identify_spatial_components(self, code: str) -> List[str]:
        """Identify spatial components used."""
        spatial_components = []
        
        spatial_patterns = {
            "OrthogonalMooreGrid": "Moore neighborhood (8 neighbors)",
            "OrthogonalVonNeumannGrid": "Von Neumann neighborhood (4 neighbors)",
            "HexGrid": "Hexagonal grid",
            "Network": "Network/graph structure",
            "ContinuousSpace": "Continuous space"
        }
        
        for pattern, description in spatial_patterns.items():
            if pattern in code:
                spatial_components.append({"type": pattern, "description": description})
        
        return spatial_components
    
    async def _generate_llm_explanation(
        self,
        code: str,
        focus_area: Optional[str],
        audience_level: str,
        code_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate explanation using LLM."""
        
        # Customize prompt based on audience level
        audience_instructions = {
            "beginner": "Explain in very simple terms, avoiding technical jargon. Use analogies and real-world examples.",
            "intermediate": "Provide a balanced explanation with some technical details but keep it accessible.",
            "advanced": "Include technical details, design patterns, and implementation considerations."
        }
        
        instruction = audience_instructions.get(audience_level, audience_instructions["beginner"])
        
        # Add code analysis context
        context = f"""
CODE ANALYSIS CONTEXT:
- Agents found: {[agent['name'] for agent in code_analysis['agents']]}
- Model class: {code_analysis['model']['name'] if code_analysis['model'] else 'None'}
- Spatial components: {[comp['type'] for comp in code_analysis['spatial_components']]}
- Key concepts: {[concept['concept'] for concept in code_analysis['key_concepts']]}

AUDIENCE LEVEL: {audience_level}
INSTRUCTION: {instruction}
"""
        
        if focus_area:
            context += f"\nFOCUS AREA: {focus_area}"
        
        prompt = f"{context}\n\n{self.prompt_manager.get_explanation_prompt(code, focus_area)}"
        
        messages = [
            LLMMessage(role="system", content=self.prompt_manager.get_system_prompt(TaskType.EXPLAIN)),
            LLMMessage(role="user", content=prompt)
        ]
        
        response = await self.llm_provider.generate(messages)
        
        return {
            "explanation": response.content,
            "model_used": response.model,
            "usage": response.usage
        }
    
    def _create_structured_explanation(
        self,
        code_analysis: Dict[str, Any],
        llm_explanation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a structured explanation combining analysis and LLM output."""
        
        structured = {
            "overview": self._extract_overview(llm_explanation["explanation"]),
            "agents": self._create_agent_explanations(code_analysis["agents"]),
            "model": self._create_model_explanation(code_analysis["model"]),
            "environment": self._create_environment_explanation(code_analysis["spatial_components"]),
            "key_concepts": code_analysis["key_concepts"],
            "full_explanation": llm_explanation["explanation"]
        }
        
        return structured
    
    def _extract_overview(self, explanation: str) -> str:
        """Extract overview section from LLM explanation."""
        # Look for overview patterns
        overview_patterns = [
            r"## Overview\n(.*?)(?=\n##|\n\n[A-Z]|\Z)",
            r"\*\*Overview\*\*:?\s*(.*?)(?=\n\*\*|\n\n[A-Z]|\Z)",
            r"Overview:?\s*(.*?)(?=\n[A-Z]|\Z)"
        ]
        
        for pattern in overview_patterns:
            match = re.search(pattern, explanation, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no specific overview found, return first paragraph
        paragraphs = explanation.split('\n\n')
        return paragraphs[0] if paragraphs else ""
    
    def _create_agent_explanations(self, agents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create explanations for agent classes."""
        explanations = []
        
        for agent in agents:
            explanation = {
                "name": agent["name"],
                "purpose": agent.get("docstring", "Agent class"),
                "behaviors": [],
                "key_methods": []
            }
            
            # Analyze methods for behaviors
            for method in agent["methods"]:
                if method["name"] == "step":
                    explanation["behaviors"].append("Executes behavior each time step")
                elif method["name"] == "move":
                    explanation["behaviors"].append("Can move through space")
                elif method["name"] == "__init__":
                    explanation["behaviors"].append("Initializes agent properties")
                
                if method["name"] in ["step", "move", "interact", "decide"]:
                    explanation["key_methods"].append({
                        "name": method["name"],
                        "description": method.get("docstring", f"Handles {method['name']} behavior")
                    })
            
            explanations.append(explanation)
        
        return explanations
    
    def _create_model_explanation(self, model: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Create explanation for model class."""
        if not model:
            return None
        
        explanation = {
            "name": model["name"],
            "purpose": model.get("docstring", "Main simulation model"),
            "responsibilities": [],
            "key_methods": []
        }
        
        # Analyze methods for responsibilities
        method_responsibilities = {
            "__init__": "Sets up the simulation environment",
            "step": "Advances the simulation by one time step",
            "run_model": "Runs the complete simulation",
            "collect_data": "Gathers data from the simulation"
        }
        
        for method in model["methods"]:
            if method["name"] in method_responsibilities:
                explanation["responsibilities"].append(method_responsibilities[method["name"]])
                explanation["key_methods"].append({
                    "name": method["name"],
                    "description": method.get("docstring", method_responsibilities[method["name"]])
                })
        
        return explanation
    
    def _create_environment_explanation(self, spatial_components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create explanation for environment/space."""
        if not spatial_components:
            return {"type": "No spatial environment", "description": "Agents exist without spatial relationships"}
        
        primary_component = spatial_components[0]
        
        return {
            "type": primary_component["type"],
            "description": primary_component["description"],
            "features": [comp["description"] for comp in spatial_components]
        }
    
    def _get_node_name(self, node: ast.AST) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_name(node.value)}.{node.attr}"
        else:
            return str(node)
    
    def _extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from class or function."""
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            return node.body[0].value.value
        return None