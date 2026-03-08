# Contributing to Mesa Stock Market PoC
Thank you for checking out this GSoC 2026 contribution! This guide ensures you can set up, test, and audit the simulation in under 5 minutes.

## 1. Rapid Onboarding
To get started without any API costs or keys:

```bash
git clone https://github.com/shubham/mesa-stock-market
cd mesa-stock-market
pip install -r requirements.txt
python run.py --offline
```
**What to expect:** A terminal dashboard showing "Mock" agents trading. This verifies the Reflexivity Math and UI logic work on your machine.

## 2. Testing Protocols
We maintain a strict 2026 Mesa 4.0 Standard for reliability.

- **Unit Tests:** Run `pytest tests/test_model.py`. This validates the MarketModel price discovery logic.
- **Schema Audit:** Run `pytest tests/test_schemas.py`. This ensures the LLM's Pydantic output matches the expected trade structure.
- **Mock Verification:** All agents have a `use_mock=True` flag for deterministic testing.

## 3. Working with LLMs (The "Live" Mode)
If you wish to test with live generative intelligence:

1. Add your `GROQ_API_KEY` or `GEMINI_API_KEY` to `.env`.
2. The system uses **SingleCallReasoning** to save you 50% on token costs.

**Note on 429 Errors:** If you hit a rate limit, the Quota Sentinel will trigger. It will flush all current agent "Thoughts" and "Actions" to `logs/full_simulation_audit.json` before exiting. No data is lost.

## 4. Technical Architecture (For Maintainers)
- **Data Flow:** We follow the Mesa 4.0 Push Architecture. Agent decisions are pushed to the DataRegistry, which then calculates the reflexive price impact.
- **Extending:** To add a new persona (e.g., "High-Frequency Bot"), add a new entry to `config/personas.yaml`. The Trader agent will automatically pick up the new natural language prompt.
