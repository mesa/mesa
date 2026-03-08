# Mesa 4.0: Quantamental AI Stock Market 📈

[![Tests: 29/29 PASSED](https://img.shields.io/badge/Tests-29%2F29%20PASSED-success.svg)](#)

> A reflexive stock market simulation where LLM agents don't just trade; they move the price.

This is a GSoC 2026 Proof-of-Concept for `mesa-llm` demonstrating **SingleCallReasoning** (50% API cost saving), deep Pydantic-enforced trading, and **Offline Resilience** compliant with Mesa 4.0 standards.

## Innovation Stack

### 1. Pydantic-Enforced Trading (First in Mesa)
LLM agents return **structured `TradeDecision` objects** — not raw strings. The schema merges Chain-of-Thought reasoning with the trade action in a **single API call**, enforced via `litellm`'s `response_format` parameter.

```python
class TradeDecision(BaseModel):
    thinking_process: str     # Full CoT reasoning trace
    sentiment_score: float    # -1.0 (fear) to +1.0 (greed)
    action: TradeAction       # BUY | SELL | HOLD
    quantity: int             # Kelly-Criterion scaled
    reasoning: str            # Standard financial summary
```

### 2. Kelly-Criterion Agents
Agents don't guess position sizes — they **calculate** them using the Kelly Criterion (`f* = (bp - q) / b`), scaling quantity based on conviction strength and win probability derived from sentiment analysis.

### 3. Reflexive Market Loop
Price discovery is **endogenous**: agent trades create order flow → order flow moves the price → changed price affects the next agent's observation. This closed feedback loop produces emergent phenomena like **Irrational Exuberance** and **Panic Selling**.

### 4. Single-Call CoT (50% API Cost Reduction)
Traditional approaches require 2 LLM calls per agent per step (think + act). Our merged Pydantic schema forces thinking *and* acting into a single token stream, cutting API costs by half while maintaining full audit transparency.

## Structure

| Component | Path | Purpose |
|---|---|---|
| **Core Model** | `mesa_stock_market/model.py` | MarketModel simulation + Reflexivity Engine |
| **Agents** | `mesa_stock_market/agents.py` | TraderAgent LLM wrapper |
| **Guardrails** | `mesa_stock_market/schemas.py` | Pydantic TradeDecision models |
| **Logic** | `mesa_stock_market/reasoning.py` | SingleCallReasoning (The "Soul") |
| **Config** | `mesa_stock_market/config/personas.yaml` | Traits & Prompts (The "Personality") |
| **Tests** | `tests/test_model.py` | 29 mock-driven green unit tests |
| **Docs** | `docs/` | Arch/Implementation specs |
| **CLI Runner** | `run.py` | Shock Event runner with rich output |

## Architecture

```
MarketModel.step()
  ├── get_global_market_context()    → Market Pulse (unified for all agents)
  ├── agents.shuffle_do("step")     → Mesa 4.0 activation
  │     └── Trader.step()
  │           ├── generate_obs()     → Self + other agents' state
  │           ├── build_step_prompt  → Price + Pulse + Portfolio
  │           ├── reasoning.plan()   → Single LLM call (CoT + action)
  │           └── apply_plan()       → execute_trade tool
  ├── _apply_reflexivity()           → Price discovery from order flow
  ├── datacollector.collect()        → Metrics snapshot
  └── log_data()                     → In-memory step logger
```

## Agent Personas

| Persona | Style | Risk | Behavior |
|---|---|---|---|
| **Raging Bull** | Aggressive momentum | High | Loads up on rallies |
| **The Hedgehog** | Cautious value | Low | Waits for extreme mispricing |
| **The Algorithm** | Systematic quant | Medium | Pure Kelly, zero emotion |
| **The Contrarian** | Contrarian | Medium | Buys panic, sells euphoria |
| **The Momentum Speculator** | Sentiment-driven | High | Chases and panics |

## Quick Start

### 1. The Offline Smoke Test
Run the simulation locally immediately without any API keys or internet connection.
```bash
python run.py --offline
```

### 2. Install Full Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

```bash
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

### 3. Run the Simulation

```bash
python run.py
```

### 4. Review Audit Logs

```bash
ls logs/
# step_000_report.json, step_001_report.json, ...
```

## The Shock Event

At **Step 15**, a "CEO arrested for fraud" news event is injected. The simulation tests whether agents — each with unique risk profiles — respond appropriately:

- **Raging Bull**: Should reduce exposure but may resist selling
- **The Hedgehog**: Should have been cautious already; minimal impact
- **The Algorithm**: Should calculate negative EV and go to HOLD
- **The Contrarian**: Should start accumulating after the crash
- **The Momentum Speculator**: Should panic sell immediately

## Risk Management

Dual-enforced 1% Risk Rule:
1. **Prompt-level**: System prompt tells the LLM to self-limit
2. **Server-side**: `execute_trade` tool rejects trades exceeding 1% of portfolio value, returning a structured `TradeError` that syncs to the agent's internal state

## Tech Stack

- **Python 3.12** | **Mesa 4.0** | **Mesa-LLM** | **LiteLLM** | **Pydantic v2**
- Scheduling: `agents.shuffle_do()` (unified Mesa 4.0 API)
- LLM: `gemini/gemini-2.0-flash` (configurable via litellm)
- Reporting: `mesa.DataCollector` + custom JSON audit logger

## License

MIT — Part of the [Project Mesa](https://github.com/mesa/mesa) ecosystem.
