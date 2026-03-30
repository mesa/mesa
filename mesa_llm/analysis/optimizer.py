"""Mesa simulation optimization and performance analysis."""

import ast
import re
from typing import Dict, Any, List, Optional, Tuple
from ..llm import LLMProviderFactory, LLMMessage
from ..prompts import PromptManager, TaskType
from ..simulation import SafeExecutor
from ..utils import LLMProvider, logger


class MesaOptimizer:
    """Optimizes Mesa simulation code for performance and best practices."""
    
    def __init__(self, llm_provider: LLMProvider = LLMProvider.OPENAI):
        self.llm_provider = LLMProviderFactory.create_provider(llm_provider)
        self.prompt_manager = PromptManager()
        self.executor = SafeExecutor()
    
    async def optimize_simulation(
        self,
        code: str,
        focus_areas: Optional[List[str]] = None,
        performance_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize Mesa simulation code.
        
        Args:
            code: Mesa simulation code to optimize
            focus_areas: Specific areas to focus on (performance, memory, readability)
            performance_profile: Optional performance profiling data
            
        Returns:
            Optimization analysis and improved code
        """
        logger.info("Starting simulation optimization analysis")
        
        # Analyze current code
        code_analysis = self._analyze_code_for_optimization(code)
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(code, code_analysis)
        
        # Get LLM optimization suggestions
        llm_optimization = await self._get_llm_optimization(
            code, optimization_opportunities, focus_areas, performance_profile
        )
        
        # Create optimization report
        optimization_report = self._create_optimization_report(
            code_analysis, optimization_opportunities, llm_optimization
        )
        
        return {
            "code_analysis": code_analysis,
            "optimization_opportunities": optimization_opportunities,
            "llm_optimization": llm_optimization,
            "optimization_report": optimization_report
        }
    
    def _analyze_code_for_optimization(self, code: str) -> Dict[str, Any]:
        """Analyze code structure for optimization opportunities."""
        analysis = {
            "complexity_metrics": {},
            "mesa_patterns": {},
            "performance_indicators": {},
            "memory_usage_patterns": {},
            "algorithmic_complexity": {}
        }
        
        try:
            tree = ast.parse(code)
            
            # Analyze complexity
            analysis["complexity_metrics"] = self._calculate_complexity_metrics(tree, code)
            
            # Analyze Mesa-specific patterns
            analysis["mesa_patterns"] = self._analyze_mesa_patterns(tree, code)
            
            # Identify performance indicators
            analysis["performance_indicators"] = self._identify_performance_indicators(code)
            
            # Analyze memory usage patterns
            analysis["memory_usage_patterns"] = self._analyze_memory_patterns(code)
            
        except SyntaxError:
            logger.warning("Could not parse code for optimization analysis")
        
        return analysis
    
    def _calculate_complexity_metrics(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """Calculate code complexity metrics."""
        metrics = {
            "cyclomatic_complexity": 0,
            "nested_loops": 0,
            "function_lengths": [],
            "class_sizes": [],
            "import_count": 0
        }
        
        for node in ast.walk(tree):
            # Count decision points for cyclomatic complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                metrics["cyclomatic_complexity"] += 1
            elif isinstance(node, ast.BoolOp):
                metrics["cyclomatic_complexity"] += len(node.values) - 1
            
            # Count nested loops
            elif isinstance(node, (ast.For, ast.While)):
                depth = self._calculate_nesting_depth(node)
                if depth > 1:
                    metrics["nested_loops"] += 1
            
            # Measure function lengths
            elif isinstance(node, ast.FunctionDef):
                func_length = len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
                metrics["function_lengths"].append(func_length)
            
            # Measure class sizes
            elif isinstance(node, ast.ClassDef):
                class_size = len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
                metrics["class_sizes"].append(class_size)
            
            # Count imports
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics["import_count"] += 1
        
        return metrics
    
    def _analyze_mesa_patterns(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """Analyze Mesa-specific patterns for optimization."""
        patterns = {
            "agent_activation_patterns": [],
            "grid_operations": [],
            "data_collection_efficiency": {},
            "agentset_usage": [],
            "spatial_queries": []
        }
        
        # Look for agent activation patterns
        if "agents.do(" in code:
            patterns["agent_activation_patterns"].append("Sequential activation")
        if "agents.shuffle_do(" in code:
            patterns["agent_activation_patterns"].append("Random activation")
        if "agents_by_type" in code:
            patterns["agent_activation_patterns"].append("Type-based activation")
        
        # Analyze grid operations
        grid_operations = [
            "get_neighborhood", "move_to", "get_neighbors", 
            "get_cell_list_contents", "iter_neighborhood"
        ]
        for operation in grid_operations:
            if operation in code:
                patterns["grid_operations"].append(operation)
        
        # Check AgentSet usage efficiency
        agentset_operations = ["select", "agg", "map", "groupby", "get", "set"]
        for operation in agentset_operations:
            if f"agents.{operation}(" in code:
                patterns["agentset_usage"].append(operation)
        
        return patterns
    
    def _identify_performance_indicators(self, code: str) -> Dict[str, Any]:
        """Identify potential performance issues."""
        indicators = {
            "potential_bottlenecks": [],
            "inefficient_patterns": [],
            "optimization_opportunities": []
        }
        
        # Check for potential bottlenecks
        bottleneck_patterns = {
            "nested loops in step()": r"def step\(.*?\):.*?for.*?for",
            "expensive operations in loops": r"for.*?(sort|sorted|max|min|sum)\(",
            "repeated calculations": r"(\w+\.\w+\(\).*?){3,}",
            "large list comprehensions": r"\[.*?for.*?for.*?\]"
        }
        
        for pattern_name, pattern in bottleneck_patterns.items():
            if re.search(pattern, code, re.DOTALL):
                indicators["potential_bottlenecks"].append(pattern_name)
        
        # Check for inefficient patterns
        inefficient_patterns = {
            "iterating over all agents repeatedly": "for agent in self.agents:",
            "creating new lists unnecessarily": "list(",
            "using append in loops": ".append(",
            "string concatenation in loops": "+="
        }
        
        for pattern_name, pattern in inefficient_patterns.items():
            if pattern in code:
                indicators["inefficient_patterns"].append(pattern_name)
        
        return indicators
    
    def _analyze_memory_patterns(self, code: str) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        patterns = {
            "large_data_structures": [],
            "memory_leaks_potential": [],
            "efficient_patterns": []
        }
        
        # Check for large data structures
        if "pandas.DataFrame" in code:
            patterns["large_data_structures"].append("DataFrame usage")
        if "numpy.array" in code:
            patterns["large_data_structures"].append("NumPy arrays")
        
        # Check for potential memory leaks
        if "global " in code:
            patterns["memory_leaks_potential"].append("Global variables")
        if "cache" in code.lower():
            patterns["memory_leaks_potential"].append("Caching without cleanup")
        
        # Check for efficient patterns
        if "AgentSet" in code:
            patterns["efficient_patterns"].append("Using AgentSet for collections")
        if "property_layers" in code:
            patterns["efficient_patterns"].append("Using property layers for spatial data")
        
        return patterns
    
    def _identify_optimization_opportunities(
        self, 
        code: str, 
        code_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities."""
        opportunities = []
        
        complexity_metrics = code_analysis.get("complexity_metrics", {})
        performance_indicators = code_analysis.get("performance_indicators", {})
        mesa_patterns = code_analysis.get("mesa_patterns", {})
        
        # High complexity functions
        function_lengths = complexity_metrics.get("function_lengths", [])
        if function_lengths and max(function_lengths) > 50:
            opportunities.append({
                "type": "complexity",
                "priority": "high",
                "issue": "Large function detected",
                "description": f"Function with {max(function_lengths)} statements should be refactored",
                "suggestion": "Break down large functions into smaller, focused methods"
            })
        
        # Nested loops
        if complexity_metrics.get("nested_loops", 0) > 0:
            opportunities.append({
                "type": "performance",
                "priority": "high",
                "issue": "Nested loops detected",
                "description": "Nested loops can cause performance issues with large agent populations",
                "suggestion": "Consider using AgentSet operations or vectorized operations"
            })
        
        # Inefficient agent activation
        activation_patterns = mesa_patterns.get("agent_activation_patterns", [])
        if "Sequential activation" in activation_patterns and "Random activation" not in activation_patterns:
            opportunities.append({
                "type": "mesa_optimization",
                "priority": "medium",
                "issue": "Only sequential activation used",
                "description": "Sequential activation can create unrealistic patterns",
                "suggestion": "Consider using shuffle_do() for more realistic agent activation"
            })
        
        # Missing AgentSet optimizations
        agentset_usage = mesa_patterns.get("agentset_usage", [])
        if not agentset_usage and "for agent in" in code:
            opportunities.append({
                "type": "mesa_optimization",
                "priority": "high",
                "issue": "Not using AgentSet operations",
                "description": "Manual iteration over agents is less efficient than AgentSet methods",
                "suggestion": "Use AgentSet.select(), .agg(), .map() methods for better performance"
            })
        
        # Potential bottlenecks
        bottlenecks = performance_indicators.get("potential_bottlenecks", [])
        for bottleneck in bottlenecks:
            opportunities.append({
                "type": "performance",
                "priority": "high",
                "issue": bottleneck,
                "description": f"Performance bottleneck: {bottleneck}",
                "suggestion": "Optimize or refactor this pattern for better performance"
            })
        
        return opportunities
    
    async def _get_llm_optimization(
        self,
        code: str,
        opportunities: List[Dict[str, Any]],
        focus_areas: Optional[List[str]],
        performance_profile: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get LLM-based optimization suggestions."""
        
        # Prepare context
        context = "OPTIMIZATION ANALYSIS:\n"
        context += f"Found {len(opportunities)} optimization opportunities:\n"
        
        for i, opp in enumerate(opportunities, 1):
            context += f"{i}. {opp['issue']} (Priority: {opp['priority']})\n"
            context += f"   Description: {opp['description']}\n"
            context += f"   Suggestion: {opp['suggestion']}\n\n"
        
        if focus_areas:
            context += f"FOCUS AREAS: {', '.join(focus_areas)}\n\n"
        
        if performance_profile:
            context += f"PERFORMANCE PROFILE: {performance_profile}\n\n"
        
        prompt = f"{context}{self.prompt_manager.get_optimization_prompt(code)}"
        
        messages = [
            LLMMessage(role="system", content=self.prompt_manager.get_system_prompt(TaskType.OPTIMIZE)),
            LLMMessage(role="user", content=prompt)
        ]
        
        response = await self.llm_provider.generate(messages)
        
        return {
            "optimization_suggestions": response.content,
            "model_used": response.model,
            "usage": response.usage
        }
    
    def _create_optimization_report(
        self,
        code_analysis: Dict[str, Any],
        opportunities: List[Dict[str, Any]],
        llm_optimization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a comprehensive optimization report."""
        
        # Categorize opportunities by priority
        high_priority = [opp for opp in opportunities if opp["priority"] == "high"]
        medium_priority = [opp for opp in opportunities if opp["priority"] == "medium"]
        low_priority = [opp for opp in opportunities if opp["priority"] == "low"]
        
        # Calculate optimization score
        complexity_metrics = code_analysis.get("complexity_metrics", {})
        optimization_score = self._calculate_optimization_score(complexity_metrics, opportunities)
        
        report = {
            "optimization_score": optimization_score,
            "total_opportunities": len(opportunities),
            "priority_breakdown": {
                "high": len(high_priority),
                "medium": len(medium_priority),
                "low": len(low_priority)
            },
            "top_recommendations": high_priority[:5],  # Top 5 high-priority items
            "complexity_summary": {
                "cyclomatic_complexity": complexity_metrics.get("cyclomatic_complexity", 0),
                "average_function_length": (
                    sum(complexity_metrics.get("function_lengths", [])) / 
                    len(complexity_metrics.get("function_lengths", [1]))
                ),
                "nested_loops": complexity_metrics.get("nested_loops", 0)
            },
            "optimization_categories": self._categorize_optimizations(opportunities),
            "estimated_impact": self._estimate_optimization_impact(opportunities),
            "llm_suggestions": llm_optimization["optimization_suggestions"]
        }
        
        return report
    
    def _calculate_optimization_score(
        self, 
        complexity_metrics: Dict[str, Any], 
        opportunities: List[Dict[str, Any]]
    ) -> float:
        """Calculate an optimization score (0-100, higher is better)."""
        base_score = 100
        
        # Deduct points for complexity
        cyclomatic_complexity = complexity_metrics.get("cyclomatic_complexity", 0)
        base_score -= min(cyclomatic_complexity * 2, 30)
        
        # Deduct points for opportunities
        high_priority_count = len([opp for opp in opportunities if opp["priority"] == "high"])
        medium_priority_count = len([opp for opp in opportunities if opp["priority"] == "medium"])
        
        base_score -= high_priority_count * 10
        base_score -= medium_priority_count * 5
        
        return max(base_score, 0)
    
    def _categorize_optimizations(self, opportunities: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize optimization opportunities."""
        categories = {}
        for opp in opportunities:
            category = opp["type"]
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _estimate_optimization_impact(self, opportunities: List[Dict[str, Any]]) -> Dict[str, str]:
        """Estimate the impact of optimizations."""
        high_count = len([opp for opp in opportunities if opp["priority"] == "high"])
        
        if high_count >= 5:
            return {"performance": "High", "maintainability": "High", "scalability": "High"}
        elif high_count >= 3:
            return {"performance": "Medium", "maintainability": "Medium", "scalability": "Medium"}
        else:
            return {"performance": "Low", "maintainability": "Low", "scalability": "Low"}
    
    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """Calculate the nesting depth of loops."""
        depth = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)) and child != node:
                depth = max(depth, 1 + self._calculate_nesting_depth(child))
        return depth