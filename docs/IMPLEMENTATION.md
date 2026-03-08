# Implementation Notes: Reflexivity Engine

## Overview
The MarketModel introduces a reflexivity engine to project Mesa. This engine connects LLM agent psychological decisions (Sentiment + Order Action) directly into environmental feedback loops (Price Discovery).

## How it works (The Math)

1. **Calculate Net Flow**:
   We determine the buy/sell imbalance during a single agent execution step.
   ```
   net_flow = total_bought - total_sold
   ```

2. **Calculate Price Impact**:
   A fractional modifier determines how sensitive the price is to the volume traded compared to the total float (liquidity) of the asset.
   ```
   price_impact = (net_flow / total_liquidity) * impact_multiplier
   ```

3. **Reflexive Feedback Loop**:
   The new price is set. In the very next step, agents observe this new price and experience "FOMO" or "Panic", further accelerating the flow.
   ```
   new_price = max(1.0, current_price * (1 + price_impact))
   ```

## Why this matters for Generative AI
Traditional behavioral models use arbitrary thresholds (e.g., "If price drops 10%, sell"). Our LLM engine requires no heuristic coding. Agents respond dynamically to the resulting price action and the natural language `Market_Pulse`.

### 2026 Architectural Alignment
Designed for Mesa 4.0, the Reflexivity Engine uses an observable-driven pattern. Price changes act as 'Signals' that agents subscribe to, ensuring the simulation remains performant even as agent counts scale.

---

## Key Innovation: SingleCallReasoning

```diff:reasoning.py
===
"""
reasoning.py — SingleCallReasoning: 50% API Cost Reduction via Merged CoT.

This is a custom Reasoning strategy that extends mesa-llm's base Reasoning
class to merge Chain-of-Thought reasoning and tool execution into a single
LLM API call.

The standard CoTReasoning makes 2 calls per agent step:
  Call 1: tool_choice="none"     → Generate thought chain
  Call 2: tool_choice="required" → Execute tool based on thoughts

SingleCallReasoning makes 1 call:
  Call 1: tool_choice="required" → Think AND call tool in one response

The LLM's response contains BOTH:
  - message.content  → The CoT reasoning trace (captured for audit)
  - message.tool_calls → The execute_trade function call

This is the key GSoC innovation: demonstrating that mesa-llm's Reasoning
interface is extensible enough to support novel optimization strategies.
"""
```

**Architecture difference vs CoTReasoning:**

| | CoTReasoning | SingleCallReasoning |
|---|---|---|
| API calls/agent/step | 2 | **1** |
| Call 1 | `tool_choice="none"` (think) | `tool_choice="required"` (think + act) |
| Call 2 | `tool_choice="required"` (act) | — |
| Token cost | 2× prompt | **1× prompt** |


## Development History & Challenges Overcome

1. **Tool registration** — `ToolManager` has no `.tool` attribute; correct API is `@tool(tool_manager=...)` from `mesa_llm.tools.tool_decorator`
2. **.env format** — Initial prototype had raw key values without `GEMINI_API_KEY=` prefix
3. **dotenv path** — `load_dotenv()` defaults to `cwd`, but `.env` was buried inside a sub-package; added explicit script path resolution for resilience logic
