# Walkthrough: Running the Simulation

This guide explains how to run the Mesa 4.0 Quantamental Stock Market PoC locally.

## 1. Environment Setup

Copy `.env.example` to `.env` and add your API token. We recommend using Groq's high-speed API for the best experience.

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY
```

## 2. Install Dependencies

You must install the exact requirements via `pip`:

```bash
pip install -r requirements.txt
```

## 3. Run the Live Simulation

Run the CLI tool from the project root. The `run.py` script includes a rich UI dashboard with live progress spinners and final summary tables.

```bash
python run.py
```

## 4. Understanding the Output

1. **Initialization:** You will see the `MarketModel` spin up 5 agents driven by `config/personas.yaml`.
2. **Pulse Phase:** Starting at step 15, the "Shock Event" occurs (fraud arrest). Look at the agent table to see 'Momentum Speculator' panic selling and the 'Contrarian' attempting to buy the dip.
3. **Persisted Logs:** After the run completes, check the `logs/full_simulation_audit.json` file for the exact Pydantic reasoning strings generated during execution.

## 5. The Quota Sentinel Tooling

While designed for high-fidelity agentic reasoning, the system includes a **Quota Sentinel** to handle the Free-Tier API minute-limits gracefully.

If the prompt generation + observation context exceeds the free-tier per-minute ceiling (a common scenario when running 5 agents simultaneously on Gemini Free Tier), the simulation does *not* hang or crash with a python stack trace.

Instead, it triggers a **Stress Test Victory**, which leverages the DataRegistry to flush all currently cached agent observations and price signals to logs/ before a clean shutdown. This provides a clean exit reporting the exact metrics discovered before the wall was hit.

## 6. Expected Exceptions

If something goes wrong, consult this Troubleshooting Table so you don't panic:

| Scenario | What you see | What it means |
|---|---|---|
| **No API Key** | `EnvironmentError: Key not found` | The `.env` is missing or the key isn't exported. |
| **Rate Limit** | `Stress Test Victory` Panel | You've hit the API tier limit. Data is safely saved. |
| **Validation Error** | `Pydantic ValidationError` | The LLM sent a weird response (e.g., negative quantity). The system caught it. |
