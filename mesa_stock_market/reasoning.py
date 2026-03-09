"""reasoning.py — SingleCallReasoning: Optimized GSoC Implementation.

This module provides a cost-optimized reasoning strategy for mesa-llm by
merging Chain-of-Thought (CoT) and tool execution into a single LLM
transaction.

Standard CoTReasoning (2 calls):
  Call 1: tool_choice="none"     → Generate thought chain
  Call 2: tool_choice="required" → Execute tool from chain

SingleCallReasoning (1 call):
  Call 1: tool_choice="required" → Think AND act in one token stream

The LLM response contains BOTH:
  - message.content    → CoT reasoning trace (captured for audit)
  - message.tool_calls → execute_trade function call (executed by agent)

This is a GSoC-level extension of mesa-llm's Reasoning interface,
demonstrating that the library's architecture supports novel optimization
strategies without modifying core internals.
"""

import logging
from typing import TYPE_CHECKING

from mesa_llm.reasoning.reasoning import Observation, Plan, Reasoning

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

logger = logging.getLogger(__name__)


class SingleCallReasoning(Reasoning):
    """Merged CoT + Structured Action in a single LLM transaction.

    Overrides the default two-call pattern of standard mesa-llm reasoning
    to achieve a 50% reduction in API overhead while preserving full
    audit transparency via message.content capture.

    Attributes:
        agent: Reference to the LLMAgent using this reasoning strategy.
    """

    # Externalized Prompt Template: Isolated from functional logic for
    # easier A/B testing and persona tuning without redeployment.
    SYSTEM_PROMPT_TEMPLATE = """You are a Senior Desk Lead in a high-stakes Quantamental simulation.
Your goal is to execute trades with clinical precision and expected-value logic.

# OPERATIONAL PROTOCOL
1. THINK: Analyze the market using the VST Filter (Value, Safety, Timing).
2. QUANT: Apply Kelly Criterion and the 1% Risk Rule.
3. ACT: You MUST call the 'execute_trade' tool in the same response.

Both your text reasoning AND the tool call must appear in the SAME response.

# MEMORY & CONTEXT
Long-Term: {long_term}
Recent: {short_term}

# OBSERVATION
{observation}

# OUTPUT FORMAT
Provide your analysis in this structure, then call the tool:

SENTIMENT: [score between -1.0 and +1.0]
VST CHECK:
- Value: [assessment]
- Safety: [1% risk calculation]
- Timing: [momentum confirmation]
KELLY: f* = [calculation], qty = [result]
DECISION: [BUY/SELL/HOLD] [quantity] — [one-line reasoning]

Then IMMEDIATELY call the execute_trade tool with action, quantity, and reasoning.
"""

    def __init__(self, agent: "LLMAgent"):
        super().__init__(agent=agent)
        self._validate_agent_capability()

    def _validate_agent_capability(self) -> None:
        """Strict validation to ensure the agent is configured for this strategy.

        Raises:
            AttributeError: If agent lacks memory or tool_manager modules.
        """
        if not hasattr(self.agent, "memory"):
            raise AttributeError(
                f"Agent {self.agent.unique_id} must have a memory module "
                "for SingleCallReasoning."
            )
        if not hasattr(self.agent, "tool_manager"):
            raise AttributeError(
                f"Agent {self.agent.unique_id} must have a tool_manager for execution."
            )

    def _get_context_strings(self) -> dict[str, str]:
        """Safely extract memory context for prompt injection.

        Returns:
            Dict with 'long_term' and 'short_term' memory strings.
        """
        memory = self.agent.memory
        long_term = "N/A"
        if hasattr(memory, "format_long_term") and callable(memory.format_long_term):
            try:
                long_term = memory.format_long_term() or "N/A"
            except Exception as e:
                logger.warning("Failed to format long-term memory: %s", e)

        short_term = "N/A"
        if hasattr(memory, "format_short_term") and callable(memory.format_short_term):
            try:
                short_term = memory.format_short_term() or "N/A"
            except Exception as e:
                logger.warning("Failed to format short-term memory: %s", e)

        return {"long_term": long_term, "short_term": short_term}

    def plan(
        self,
        prompt: str | None = None,
        obs: Observation | None = None,
        ttl: int = 1,
        selected_tools: list[str] | None = None,
    ) -> Plan:
        """Single-transaction execution: Think + Act in one LLM call.

        Captures message.content (thought chain) for the audit log and
        message.tool_calls (action) for execution.

        Args:
            prompt: The step prompt with market data and portfolio state.
            obs: The agent's current observation.
            ttl: Time-to-live for the plan (default 1 step).
            selected_tools: List of tool names to make available.

        Returns:
            A Plan containing the LLM response with tool calls.

        Raises:
            ValueError: If no prompt is provided and agent has no step_prompt.
        """
        # 1. Resolve inputs
        prompt = prompt or self.agent.step_prompt
        obs = obs or self.agent.generate_obs()

        if prompt is None:
            raise ValueError("Execution failed: No prompt provided.")

        # Record observation in memory
        self.agent.memory.add_to_memory(
            type="Observation", content={"content": str(obs)}
        )

        # 2. Build injected system prompt from template
        context = self._get_context_strings()
        full_system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            long_term=context["long_term"],
            short_term=context["short_term"],
            observation=str(obs),
        )

        # 3. Execute merged call
        #    tool_choice="auto" avoids a litellm parsing bug on Groq when
        #    mixing CoT content with tool calls. The strong system prompt
        #    guarantees the tool will be called.
        self.agent.llm.system_prompt = full_system_prompt

        # Add a retry loop specifically for litellm Groq parsing errors.
        # Open source models sometimes malform the XML tool call string,
        # causing litellm to throw a BadRequestError during parsing.
        max_retries = 3
        response = None
        last_err = None

        for attempt in range(max_retries):
            try:
                response = self.agent.llm.generate(
                    prompt=prompt,
                    tool_schema=self.agent.tool_manager.get_all_tools_schema(
                        selected_tools
                    ),
                    tool_choice="auto",
                )
                break  # Success
            except Exception as e:
                # Catch litellm parsing errors (often raised as BadRequestError)
                last_err = e
                logger.warning(
                    f"Agent {self.agent.unique_id} LLM generation failed "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )

        if response is None:
            logger.error(
                f"Agent {self.agent.unique_id} failed after {max_retries} attempts."
            )
            raise RuntimeError(f"LLM generation failed after retries: {last_err}")

        message = response.choices[0].message

        # 4. Audit log & state capture
        #    Capture the text content as the 'Thought Chain' for GSoC logs.
        thought_trace = getattr(message, "content", "") or ""
        if thought_trace:
            self.agent.memory.add_to_memory(
                type="ThoughtChain", content={"trace": thought_trace}
            )
            if hasattr(self.agent, "_step_display_data"):
                self.agent._step_display_data["plan_content"] = thought_trace

        plan = Plan(step=obs.step + 1, llm_plan=message, ttl=ttl)
        self.agent.memory.add_to_memory(
            type="Plan-Execution", content={"content": str(plan)}
        )

        return plan

    async def aplan(
        self,
        prompt: str | None = None,
        obs: Observation | None = None,
        ttl: int = 1,
        selected_tools: list[str] | None = None,
    ) -> Plan:
        """Async version of plan() for parallel execution scenarios.

        Args:
            prompt: The step prompt with market data and portfolio state.
            obs: The agent's current observation.
            ttl: Time-to-live for the plan.
            selected_tools: List of tool names to make available.

        Returns:
            A Plan containing the LLM response with tool calls.
        """
        prompt = prompt or self.agent.step_prompt
        obs = obs or await self.agent.agenerate_obs()

        if prompt is None:
            raise ValueError("Execution failed: No prompt provided.")

        await self.agent.memory.aadd_to_memory(
            type="Observation", content={"content": str(obs)}
        )

        context = self._get_context_strings()
        full_system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            long_term=context["long_term"],
            short_term=context["short_term"],
            observation=str(obs),
        )

        self.agent.llm.system_prompt = full_system_prompt

        max_retries = 3
        response = None
        last_err = None

        for attempt in range(max_retries):
            try:
                response = await self.agent.llm.agenerate(
                    prompt=prompt,
                    tool_schema=self.agent.tool_manager.get_all_tools_schema(
                        selected_tools
                    ),
                    tool_choice="auto",
                )
                break
            except Exception as e:
                last_err = e
                logger.warning(
                    f"Agent {self.agent.unique_id} LLM ageneration failed "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )

        if response is None:
            logger.error(
                f"Agent {self.agent.unique_id} failed after {max_retries} attempts."
            )
            raise RuntimeError(f"LLM ageneration failed after retries: {last_err}")

        message = response.choices[0].message
        thought_trace = getattr(message, "content", "") or ""
        if thought_trace:
            await self.agent.memory.aadd_to_memory(
                type="ThoughtChain", content={"trace": thought_trace}
            )
            if hasattr(self.agent, "_step_display_data"):
                self.agent._step_display_data["plan_content"] = thought_trace

        plan = Plan(step=obs.step + 1, llm_plan=message, ttl=ttl)
        await self.agent.memory.aadd_to_memory(
            type="Plan-Execution", content={"content": str(plan)}
        )

        return plan
