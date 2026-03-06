# Agent Actions

`mesa.experimental.actions` lets agents perform **timed, interruptible, and resumable tasks** that integrate directly with Mesa's event scheduler.

```{note}
This module is experimental. Its API may change between releases.
```

---

## Why Actions?

In a classic Mesa model every agent runs `step()` once per tick and its work is instantaneous. Actions let you express that some work *takes time*:

- A forager needs 5 time units to collect food.
- A builder needs 10 time units to finish a house — and can be interrupted.
- A service agent finishes a call after a variable duration, then picks up the next one from a queue.

Mesa's event scheduler fires the completion callback at exactly the right moment, so Actions compose cleanly with everything else in the model.

---

## Core Concepts

### `Action`

An `Action` is a named task with a duration and an optional effect callback.

```python
from mesa.experimental.actions import Action, linear, step

forage = Action(
    name="forage",
    duration=5.0,           # sim-time units to complete
    priority=1.0,           # used for preemption (higher = more important)
    reward_curve=linear,    # how partial completion maps to reward
    on_effect=apply_food,   # callback(agent, completion_ratio) on finish or interrupt
    interruptible=True,     # can this action be cut short?
    reschedule_on_interrupt=False,  # queuing policy on interruption (see below)
)
```

**Key properties:**

| Property | Description |
|---|---|
| `progress` | Current progress, 0.0–1.0 (updated at interruption time) |
| `effective_completion` | `reward_curve(progress)` — reward earned so far |
| `remaining_time` | `duration * (1 - progress)` |

### Reward Curves

A reward curve maps progress `[0, 1]` → earned reward `[0, 1]`.

| Curve | Behaviour |
|---|---|
| `linear` (default) | 50 % through → 50 % reward |
| `step` | All-or-nothing: reward is 0 until fully done, then 1 |
| Any callable | e.g. `lambda p: p**2` for diminishing early returns |

### `ActionAgent`

`ActionAgent` extends `Agent` with the ability to execute one action at a time.

```python
from mesa.experimental.actions import ActionAgent

class MyAgent(ActionAgent):
    def step(self):
        if not self.is_busy:
            self.start_action(Action("work", duration=3.0, on_effect=self.apply_work))

    def apply_work(self, agent, completion):
        self.output += 10 * completion
```

**Key attributes and methods:**

| | Description |
|---|---|
| `is_busy` | `True` while an action is running |
| `current_action` | The running `Action`, or `None` |
| `action_queue` | Pending actions (list), drained automatically |
| `start_action(action)` | Start action from zero; raises if already busy |
| `interrupt_for(action)` | Interrupt current action, start a new one |
| `cancel_action(clear_queue=False)` | Cancel current action; optionally wipe the queue |
| `request_action(action)` | Priority-aware entry point (see below) |

---

## Getting Started

### Simple example: a working agent

```python
from mesa import Model
from mesa.experimental.actions import Action, ActionAgent

def on_work_done(agent, completion):
    agent.widgets_made += completion  # linear: proportional to time spent

class Worker(ActionAgent):
    def __init__(self, model):
        super().__init__(model)
        self.widgets_made = 0.0

    def step(self):
        if not self.is_busy:
            self.start_action(Action("build", duration=3.0, on_effect=on_work_done))

class Factory(Model):
    def __init__(self):
        super().__init__()
        self.workers = Worker.create_agents(self, 5)

    def step(self):
        self.workers.do("step")

model = Factory()
model.run_for(15)
total = sum(w.widgets_made for w in model.workers)
print(f"Total widgets: {total:.1f}")   # 25.0  (5 workers × 5 completions × 1.0)
```

---

## Interrupt / Resume Semantics

### `reschedule_on_interrupt`

Controls what happens to an action when it is interrupted.

| Value | Behaviour |
|---|---|
| `False` (default) | Action is discarded. `on_effect` fires with partial reward. |
| `"remainder"` | Action is pushed to the **front of the queue** with its progress preserved. It resumes from where it left off once the interrupting action finishes. `on_effect` fires at interruption and again at eventual completion. |
| `"full"` | Action is pushed to the **front of the queue** with progress reset to 0. It restarts from the beginning next time. `on_effect` fires at interruption with partial reward, then fires again with full reward on eventual completion. |

### Example: forager interrupted by a predator (resumes with remainder)

```python
def collect_food(agent, completion):
    agent.food += 10 * completion

class Forager(ActionAgent):
    def __init__(self, model):
        super().__init__(model)
        self.food = 0.0

    def step(self):
        if self.model.predator_nearby and self.is_busy:
            flee = Action("flee", duration=1.0, priority=10.0)
            self.interrupt_for(flee)        # foraging paused, flee starts
        elif not self.is_busy:
            forage = Action(
                "forage",
                duration=10.0,
                priority=1.0,
                on_effect=collect_food,
                reschedule_on_interrupt="remainder",  # resume where we left off
            )
            self.start_action(forage)
```

Timeline for a forager that gets scared at t=3 during a 10-unit forage:

```
t=0   start_action(forage)         → event scheduled at t=10
t=3   interrupt_for(flee)
         → forage progress = 0.3, on_effect(agent, 0.3) fires
         → forage (30% done) pushed to front of queue
         → flee starts, event scheduled at t=4
t=4   flee completes               → queue drains: forage resumes with 7 units remaining
t=11  forage completes             → on_effect(agent, 1.0) fires
```

---

## Action Queue

`action_queue` is a plain list. It is drained automatically whenever the agent becomes free (after completion or `cancel_action()`).

### Manually enqueuing actions

```python
agent.start_action(Action("first", duration=3.0))
agent.action_queue.append(Action("second", duration=2.0))
agent.action_queue.append(Action("third", duration=1.0))
# After 6 time units: first → second → third execute in order
```

### `cancel_action(clear_queue=True)`

Cancels the current action **and** discards everything in the queue:

```python
agent.cancel_action(clear_queue=True)   # agent is completely idle, queue is empty
```

---

## Priority-Based Preemption: `request_action`

`request_action` is the recommended entry point when multiple sources can trigger actions on the same agent.

```
if not is_busy:
    start immediately
elif new.priority > current.priority AND current is interruptible:
    interrupt_for(new)          ← preempt
else:
    action_queue.append(new)    ← wait in line
```

```python
agent.request_action(Action("low_priority_work", duration=5.0, priority=1.0))
agent.request_action(Action("urgent",            duration=1.0, priority=8.0))
# "urgent" preempts "low_priority_work" (assuming default interruptible=True)
```

---

## Integration with Mesa's Event Scheduler

Actions use `Model.schedule_event()` under the hood. When `start_action` is called at time *t*:

```
model.schedule_event(agent._complete_action, after=action.remaining_time)
```

This means:
- Sub-step precision: completion fires at the exact right moment, not just at the next integer step.
- Cancellation is automatic: if the agent is removed via `agent.remove()`, the scheduled event is cancelled, so no ghost callbacks fire.
- Zero-duration actions complete synchronously (no event is scheduled).

---

## API Reference

See [Experimental API](apis/experimental.md) for the full auto-generated reference.
