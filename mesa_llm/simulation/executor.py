"""Safe execution environment for Mesa simulations."""

import ast
import sys
import io
import contextlib
import traceback
import signal
import resource
from typing import Dict, Any, Optional, List
from contextlib import redirect_stdout, redirect_stderr
from ..utils import config, logger


class ExecutionTimeoutError(Exception):
    """Raised when code execution times out."""
    pass


class ExecutionMemoryError(Exception):
    """Raised when code execution exceeds memory limits."""
    pass


class SafeExecutor:
    """Safe execution environment for generated Mesa code."""
    
    def __init__(self):
        self.allowed_imports = set(config.allowed_imports)
        self.max_execution_time = config.max_execution_time
        self.max_memory_mb = config.max_memory_mb
    
    def execute_code(
        self,
        code: str,
        capture_output: bool = True,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute Python code safely with resource limits.
        
        Args:
            code: Python code to execute
            capture_output: Whether to capture stdout/stderr
            timeout: Optional timeout override
            
        Returns:
            Dictionary with execution results
        """
        timeout = timeout or self.max_execution_time
        
        # Validate code before execution
        validation_result = self._validate_code_safety(code)
        if not validation_result["is_safe"]:
            return {
                "success": False,
                "error": f"Code safety validation failed: {validation_result['reason']}",
                "output": "",
                "stderr": ""
            }
        
        # Set up execution environment
        execution_globals = self._create_safe_globals()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Set resource limits
            self._set_resource_limits()
            
            # Set timeout alarm
            if timeout > 0:
                signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(timeout)
            
            # Execute code with output capture
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, execution_globals)
            
            # Cancel timeout
            if timeout > 0:
                signal.alarm(0)
            
            result = {
                "success": True,
                "output": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "globals": {k: v for k, v in execution_globals.items() 
                          if not k.startswith('__') and k not in ['__builtins__']},
                "error": None
            }
            
            logger.info("Code execution completed successfully")
            return result
            
        except ExecutionTimeoutError:
            return {
                "success": False,
                "error": f"Execution timed out after {timeout} seconds",
                "output": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue()
            }
        except MemoryError:
            return {
                "success": False,
                "error": f"Execution exceeded memory limit of {self.max_memory_mb}MB",
                "output": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue()
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {str(e)}\n{traceback.format_exc()}",
                "output": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue()
            }
        finally:
            # Clean up
            if timeout > 0:
                signal.alarm(0)
    
    def validate_mesa_simulation(self, code: str) -> Dict[str, Any]:
        """Validate that code creates a proper Mesa simulation.
        
        Args:
            code: Python code to validate
            
        Returns:
            Validation results
        """
        try:
            # Execute code to check if it runs
            execution_result = self.execute_code(code, timeout=10)
            
            if not execution_result["success"]:
                return {
                    "is_valid": False,
                    "error": execution_result["error"],
                    "suggestions": ["Fix execution errors before running simulation"]
                }
            
            # Check if Mesa model was created
            globals_dict = execution_result["globals"]
            model_instances = []
            model_classes = []
            
            for name, obj in globals_dict.items():
                if hasattr(obj, '__class__'):
                    class_name = obj.__class__.__name__
                    if 'Model' in class_name:
                        model_instances.append(name)
                elif isinstance(obj, type) and hasattr(obj, '__bases__'):
                    if any('Model' in str(base) for base in obj.__bases__):
                        model_classes.append(name)
            
            validation_result = {
                "is_valid": True,
                "model_classes": model_classes,
                "model_instances": model_instances,
                "suggestions": []
            }
            
            if not model_classes and not model_instances:
                validation_result["is_valid"] = False
                validation_result["suggestions"].append("No Mesa Model class or instance found")
            
            return validation_result
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": str(e),
                "suggestions": ["Fix code errors before validation"]
            }
    
    def run_simulation_steps(
        self,
        code: str,
        steps: int = 10,
        collect_data: bool = True
    ) -> Dict[str, Any]:
        """Run a Mesa simulation for specified steps and collect data.
        
        Args:
            code: Mesa simulation code
            steps: Number of steps to run
            collect_data: Whether to collect and return simulation data
            
        Returns:
            Simulation results and data
        """
        # Add simulation execution code
        execution_code = f"""
{code}

# Find and run the model
import mesa
model_instance = None
for name, obj in locals().items():
    if isinstance(obj, mesa.Model):
        model_instance = obj
        break
    elif isinstance(obj, type) and issubclass(obj, mesa.Model):
        try:
            model_instance = obj()
            break
        except:
            continue

if model_instance is None:
    raise ValueError("No Mesa Model found or could not be instantiated")

# Run simulation
print(f"Running simulation for {steps} steps...")
model_instance.run_for({steps})

# Collect data if available
simulation_data = {{}}
if hasattr(model_instance, 'datacollector') and model_instance.datacollector:
    try:
        simulation_data['model_data'] = model_instance.datacollector.get_model_vars_dataframe().to_dict()
        simulation_data['agent_data'] = model_instance.datacollector.get_agent_vars_dataframe().to_dict()
    except:
        simulation_data['error'] = 'Failed to collect data'

print(f"Simulation completed. Final time: {{model_instance.time}}")
print(f"Number of agents: {{len(model_instance.agents)}}")
"""
        
        result = self.execute_code(execution_code, timeout=self.max_execution_time * 2)
        
        if result["success"]:
            # Extract simulation data from globals
            globals_dict = result["globals"]
            simulation_data = globals_dict.get("simulation_data", {})
            
            result["simulation_data"] = simulation_data
            result["steps_completed"] = steps
        
        return result
    
    def _validate_code_safety(self, code: str) -> Dict[str, Any]:
        """Validate code for safety before execution."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"is_safe": False, "reason": f"Syntax error: {str(e)}"}
        
        # Check for dangerous operations
        dangerous_nodes = []
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_names = []
                if isinstance(node, ast.Import):
                    module_names = [alias.name for alias in node.names]
                else:
                    module_names = [node.module] if node.module else []
                
                for module_name in module_names:
                    if module_name and not any(allowed in module_name for allowed in self.allowed_imports):
                        dangerous_nodes.append(f"Disallowed import: {module_name}")
            
            # Check for dangerous function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    dangerous_funcs = ['exec', 'eval', 'compile', '__import__', 'open', 'input']
                    if node.func.id in dangerous_funcs:
                        dangerous_nodes.append(f"Dangerous function call: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    dangerous_attrs = ['system', 'popen', 'spawn']
                    if node.func.attr in dangerous_attrs:
                        dangerous_nodes.append(f"Dangerous method call: {node.func.attr}")
        
        if dangerous_nodes:
            return {
                "is_safe": False,
                "reason": f"Dangerous operations detected: {', '.join(dangerous_nodes)}"
            }
        
        return {"is_safe": True, "reason": "Code passed safety validation"}
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a safe globals dictionary for code execution."""
        safe_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes', 'chr',
            'classmethod', 'complex', 'dict', 'dir', 'divmod', 'enumerate',
            'filter', 'float', 'format', 'frozenset', 'getattr', 'hasattr',
            'hash', 'hex', 'id', 'int', 'isinstance', 'issubclass', 'iter',
            'len', 'list', 'map', 'max', 'min', 'next', 'object', 'oct',
            'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed',
            'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod',
            'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip'
        }
        
        # Create restricted builtins
        restricted_builtins = {name: __builtins__[name] for name in safe_builtins if name in __builtins__}
        
        return {
            '__builtins__': restricted_builtins,
            '__name__': '__main__'
        }
    
    def _set_resource_limits(self):
        """Set resource limits for code execution."""
        try:
            # Set memory limit (in bytes)
            memory_limit = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            # Set CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (self.max_execution_time, self.max_execution_time))
        except (ImportError, OSError):
            # Resource limits not available on all platforms
            logger.warning("Could not set resource limits")
    
    def _timeout_handler(self, signum, frame):
        """Handle execution timeout."""
        raise ExecutionTimeoutError("Code execution timed out")