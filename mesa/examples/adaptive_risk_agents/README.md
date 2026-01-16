# Adaptive Risk Agents

This example demonstrates agents that **adapt their risk-taking behavior
based on past experiences**, implemented using only core Mesa primitives.

The model is intentionally simple in structure but rich in behavior, making it
useful as a diagnostic example for understanding how adaptive decision-making
is currently modeled in Mesa.



## Motivation

Many real-world agents do not follow fixed rules.
Instead, they:

- make decisions under uncertainty,
- remember past outcomes,
- adapt future behavior based on experience.

In Mesa today, modeling this kind of adaptive behavior often results in
a large amount of logic being concentrated inside `agent.step()`, combining
multiple concerns in a single execution phase.

This example exists to **make that structure explicit**, not to abstract it away.



## Model Overview

- Each agent chooses between:
  - a **safe action** (low or zero payoff, no risk),
  - a **risky action** (stochastic payoff).
- Agents track recent outcomes of risky actions in a short memory window.
- If recent outcomes are negative, agents become more risk-averse.
- If outcomes are positive, agents increase their risk preference.

All behavior is implemented using plain Python and Mesa’s public APIs.



## Observations From This Example

This model intentionally does **not** introduce new abstractions
(tasks, goals, states, schedulers, etc.).

Instead, it highlights several patterns that commonly arise when modeling
adaptive behavior in Mesa today:

- Decision-making, action execution, memory updates, and learning logic
  are handled within a single `step()` method.
- There is no explicit separation between decision phases.
- Actions are instantaneous, with no notion of duration or interruption.
- As behaviors grow richer, agent logic can become deeply nested and harder
  to maintain.

These observations may be useful input for ongoing discussions around:

- Behavioral frameworks
- Tasks and continuous states
- Richer agent decision abstractions



## Mesa Version & API Alignment

This example is written to align with the **Mesa 4 design direction**:

- Uses `AgentSet` and `shuffle_do`
- Avoids deprecated schedulers
- Avoids `DataCollector`
- Uses keyword-only arguments for public APIs
- Relies on `model.random` for reproducibility

No experimental or private APIs are used.



## Running the Example

From the Mesa repository root:

python -m mesa.examples.adaptive_risk_agents.run
