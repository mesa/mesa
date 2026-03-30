"""Mesa simulation debugging and analysis."""

import ast
import re
from typing import Dict, Any, List, Optional, Tuple
from ..llm import LLMProviderFactory, LLMMessage
from ..prompts import PromptManager, TaskType
from ..simulation import SafeExecutor
from ..utils import LLMProvider, logger


class MesaDebugger:
    """Analyzes and debugs Mesa simulation code."""
    
    def __init__(self, llm_provider: LLMProvider = LLMProvider.OPENAI):
        self.llm_provider = LLMProviderFactory.create_provider(llm_provider)
        self.prompt_manager = PromptManager()
        self.executor = SafeExecutor()
    
    async def debug_code(
        self,
        code: str,
        error_message: Optional[str] = None,
        run_tests: bool = True
    ) -> Dict[str, Any]:
        """Debug Mesa simulation code and provide fixes.
        
        Args:
            code: Mesa simulation code to debug
            error_message: Optional error message from execution
            run_tests: Whether to run execution tests
            
        Returns:
            Debug analysis and suggested fixes
        """
        logger.info("Starting code debugging analysis")
        
        # Perform static analysis
        static_analysis = self._static_code_analysis(code)
        
        # Run execution tests if requested
        execution_analysis = None
        if run_tests:
            execution_analysis = self._execution_analysis(code)
        
        # Get LLM analysis and suggestions
        llm_analysis = await self._llm_debug_analysis(code, error_message, static_analysis, execution_analysis)
        
        return {
            "static_analysis": static_analysis,
            "execution_analysis": execution_analysis,
            "llm_analysis": llm_analysis,
            "summary": self._create_debug_summary(static_analysis, execution_analysis, llm_analysis)
        }
    
    def _static_code_analysis(self, code: str) -> Dict[str, Any]:
        """Perform static analysis of Mesa code."""
        analysis = {
            "syntax_errors": [],
            "mesa_issues": [],
            "best_practice_violations": [],
            "warnings": [],
            "code_metrics": {}
        }
        
        try:
            tree = ast.parse(code)
            
            # Analyze code structure
            analysis["code_metrics"] = self._analyze_code_metrics(code, tree)
            
            # Check Mesa-specific patterns
            analysis["mesa_issues"] = self._check_mesa_patterns(tree, code)
            
            # Check best practices
            analysis["best_practice_violations"] = self._check_best_practices(tree, code)
            
        except SyntaxError as e:
            analysis["syntax_errors"].append({
                "line": e.lineno,
                "message": str(e),
                "text": e.text
            })
        
        return analysis
    
    def _execution_analysis(self, code: str) -> Dict[str, Any]:
        """Analyze code by attempting execution."""
        analysis = {
            "can_execute": False,
            "execution_error": None,
            "runtime_issues": [],
            "performance_metrics": {}
        }
        
        # Try to execute the code
        execution_result = self.executor.execute_code(code, timeout=10)
        
        if execution_result["success"]:
            analysis["can_execute"] = True
            
            # Try to validate as Mesa simulation
            validation_result = self.executor.validate_mesa_simulation(code)
            analysis["mesa_validation"] = validation_result
            
            # Try to run a few simulation steps
            if validation_result.get("is_valid", False):
                sim_result = self.executor.run_simulation_steps(code, steps=5)
                analysis["simulation_test"] = sim_result
        else:
            analysis["execution_error"] = execution_result["error"]
        
        return analysis
    
    async def _llm_debug_analysis(
        self,
        code: str,
        error_message: Optional[str],
        static_analysis: Dict[str, Any],
        execution_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get LLM analysis and suggestions for debugging."""
        
        # Prepare context for LLM
        context = f"STATIC ANALYSIS RESULTS:\n{self._format_analysis_for_llm(static_analysis)}\n\n"
        
        if execution_analysis:
            context += f"EXECUTION ANALYSIS RESULTS:\n{self._format_analysis_for_llm(execution_analysis)}\n\n"
        
        # Create debug prompt
        debug_prompt = self.prompt_manager.get_debug_prompt(code, error_message)
        full_prompt = f"{context}{debug_prompt}"
        
        messages = [
            LLMMessage(role="system", content=self.prompt_manager.get_system_prompt(TaskType.DEBUG)),
            LLMMessage(role="user", content=full_prompt)
        ]
        
        response = await self.llm_provider.generate(messages)
        
        return {
            "analysis": response.content,
            "model_used": response.model,
            "usage": response.usage
        }
    
    def _analyze_code_metrics(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code metrics."""
        lines = code.split('\n')
        
        metrics = {
            "total_lines": len(lines),
            "code_lines": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
            "blank_lines": len([line for line in lines if not line.strip()]),
            "classes": 0,
            "functions": 0,
            "imports": 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                metrics["classes"] += 1
            elif isinstance(node, ast.FunctionDef):
                metrics["functions"] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics["imports"] += 1
        
        return metrics
    
    def _check_mesa_patterns(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Check for Mesa-specific issues."""
        issues = []
        
        # Extract classes
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "bases": [self._get_base_name(base) for base in node.bases],
                    "methods": [method.name for method in node.body if isinstance(method, ast.FunctionDef)]
                }
                classes.append(class_info)
        
        # Check Agent classes
        agent_classes = [cls for cls in classes if any('Agent' in base for base in cls['bases'])]
        for agent_class in agent_classes:
            if 'step' not in agent_class['methods']:
                issues.append({
                    "type": "missing_method",
                    "severity": "error",
                    "message": f"Agent class '{agent_class['name']}' is missing step() method",
                    "suggestion": "Add a step() method to define agent behavior"
                })
            
            if '__init__' not in agent_class['methods']:
                issues.append({
                    "type": "missing_method",
                    "severity": "warning",
                    "message": f"Agent class '{agent_class['name']}' is missing __init__() method",
                    "suggestion": "Add __init__() method for proper initialization"
                })
        
        # Check Model classes
        model_classes = [cls for cls in classes if any('Model' in base for base in cls['bases'])]
        for model_class in model_classes:
            if 'step' not in model_class['methods']:
                issues.append({
                    "type": "missing_method",
                    "severity": "error",
                    "message": f"Model class '{model_class['name']}' is missing step() method",
                    "suggestion": "Add a step() method to advance the simulation"
                })
        
        # Check for common Mesa imports
        if 'import mesa' not in code and 'from mesa' not in code:
            issues.append({
                "type": "missing_import",
                "severity": "error",
                "message": "No Mesa imports found",
                "suggestion": "Add 'import mesa' or specific Mesa imports"
            })
        
        return issues
    
    def _check_best_practices(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Check for best practice violations."""
        violations = []
        
        # Check for docstrings
        classes_without_docstrings = []
        functions_without_docstrings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not self._has_docstring(node):
                    classes_without_docstrings.append(node.name)
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                if not self._has_docstring(node):
                    functions_without_docstrings.append(node.name)
        
        if classes_without_docstrings:
            violations.append({
                "type": "missing_docstring",
                "severity": "warning",
                "message": f"Classes without docstrings: {', '.join(classes_without_docstrings)}",
                "suggestion": "Add docstrings to document class purpose and usage"
            })
        
        if functions_without_docstrings:
            violations.append({
                "type": "missing_docstring",
                "severity": "warning",
                "message": f"Functions without docstrings: {', '.join(functions_without_docstrings)}",
                "suggestion": "Add docstrings to document function purpose and parameters"
            })
        
        # Check for type hints
        if '->' not in code and ': ' not in code:
            violations.append({
                "type": "missing_type_hints",
                "severity": "info",
                "message": "No type hints found",
                "suggestion": "Consider adding type hints for better code documentation"
            })
        
        return violations
    
    def _get_base_name(self, base_node: ast.AST) -> str:
        """Extract base class name from AST node."""
        if isinstance(base_node, ast.Name):
            return base_node.id
        elif isinstance(base_node, ast.Attribute):
            return f"{self._get_base_name(base_node.value)}.{base_node.attr}"
        else:
            return str(base_node)
    
    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if a class or function has a docstring."""
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            return True
        return False
    
    def _format_analysis_for_llm(self, analysis: Dict[str, Any]) -> str:
        """Format analysis results for LLM consumption."""
        formatted = []
        
        for key, value in analysis.items():
            if isinstance(value, list) and value:
                formatted.append(f"{key.upper()}:")
                for item in value:
                    if isinstance(item, dict):
                        formatted.append(f"  - {item.get('message', str(item))}")
                    else:
                        formatted.append(f"  - {item}")
            elif isinstance(value, dict) and value:
                formatted.append(f"{key.upper()}: {value}")
            elif value:
                formatted.append(f"{key.upper()}: {value}")
        
        return '\n'.join(formatted)
    
    def _create_debug_summary(
        self,
        static_analysis: Dict[str, Any],
        execution_analysis: Optional[Dict[str, Any]],
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a summary of debugging results."""
        summary = {
            "total_issues": 0,
            "critical_issues": 0,
            "can_execute": False,
            "is_valid_mesa": False,
            "recommendations": []
        }
        
        # Count issues from static analysis
        for issue_type in ["syntax_errors", "mesa_issues", "best_practice_violations"]:
            issues = static_analysis.get(issue_type, [])
            summary["total_issues"] += len(issues)
            
            # Count critical issues
            critical_issues = [issue for issue in issues 
                             if isinstance(issue, dict) and issue.get("severity") == "error"]
            summary["critical_issues"] += len(critical_issues)
        
        # Check execution status
        if execution_analysis:
            summary["can_execute"] = execution_analysis.get("can_execute", False)
            mesa_validation = execution_analysis.get("mesa_validation", {})
            summary["is_valid_mesa"] = mesa_validation.get("is_valid", False)
        
        # Generate recommendations
        if summary["critical_issues"] > 0:
            summary["recommendations"].append("Fix critical errors before running simulation")
        
        if not summary["can_execute"]:
            summary["recommendations"].append("Resolve execution errors")
        
        if not summary["is_valid_mesa"]:
            summary["recommendations"].append("Ensure code creates a valid Mesa simulation")
        
        if summary["total_issues"] == 0:
            summary["recommendations"].append("Code looks good! Consider running performance optimization")
        
        return summary