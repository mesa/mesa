# Forager Model

## Summary

The Forager model demonstrates Mesa's experimental **Action API** — timed, interruptible, and resumable agent tasks that integrate with the event scheduler.

Each agent spends time foraging for food. At any point a predator threat can interrupt the forager mid-task. When interrupted, the foraging action is **re-queued with its progress preserved** (`reschedule_on_interrupt="remainder"`), so the agent resumes from where it left off after fleeing.

This shows three key Action API features:

- **`reschedule_on_interrupt="remainder"`** — resume a paused task without losing progress.
- **`request_action`** — priority-aware preemption: a high-priority "flee" action interrupts a low-priority "forage" action.
- **Automatic queue draining** — when fleeing ends, the forager automatically resumes the interrupted foraging action.

## How to Run

```
solara run app.py
```

Or run headlessly:

```python
from mesa.examples.basic.forager.model import ForagerModel

model = ForagerModel()
model.run_for(50)
df = model.datacollector.get_model_vars_dataframe()
print(df.tail())
```

## Files

- `agents.py` — `ForagerAgent` class (extends `ActionAgent`)
- `model.py` — `ForagerModel` and `ForagerScenario`
- `app.py` — interactive Solara visualization

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_agents` | 20 | Number of forager agents |
| `threat_prob` | 0.1 | Per-agent, per-step probability of a predator threat |
| `forage_duration` | 5.0 | Sim-time units needed to complete one forage |
| `flee_duration` | 1.0 | Sim-time units needed to flee a threat |
| `food_per_forage` | 1.0 | Max food earned per completed forage (scales with `linear` reward curve) |

## Further Reading

- [Mesa Action API documentation](../../../docs/actions.md)
- Mesa experimental actions issue [#3393](https://github.com/projectmesa/mesa/issues/3393)
