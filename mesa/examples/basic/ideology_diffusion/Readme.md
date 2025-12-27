# Ideological Diffusion Under Economic Stress

---

## Overview

Historical analysis reveals that authoritarian regimes often gain momentum by exploiting acute **socio-economic crises**. These movements leverage public dissatisfaction to shift political paradigms and consolidate power by convincing the population that radical measures are the only solution to systemic failures.

This **Agent-Based Model (ABM)** simulates how a population, when faced with adverse conditions such as high unemployment and economic instability, becomes vulnerable to extremist influence. The model explores the transition from neutral or moderate positions to radical ideologies, highlighting how external stressors and propaganda can drive political alignment—often as a desperate response to environmental pressure.

---

## Key Features

* **Ideological Evolution:** Agents move through three stages (Neutral, Moderate, and Radical) based on cumulative pressure.
* **Socio-Economic Stressors:** Global variables like economic crises and unemployment rates act as catalysts for dissatisfaction.
* **Media Influence:** Simulates the impact of propaganda based on individual susceptibility.
* **State Dynamics:** Includes government repression logic, which can either suppress or inadvertently accelerate radicalization (the "backfire effect").

---

## How to Run

To run the model interactively, ensure you have `mesa` and `solara` installed, then run the following command in the project root:

```bash
solara run app.py