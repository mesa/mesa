# Emergency Room Simulation

An example model built on Mesa's new `Action` system ([PR #3461](https://github.com/mesa/mesa/pull/3461)).

Simulates an emergency room where patients arrive at random intervals, get triaged, and are treated by doctors. Uses the Action API's interruption and resumption mechanics to handle priority scheduling — a critical patient can bump a moderate one mid-treatment.

## What it demonstrates

- **Triage action** — assigns a random severity (`critical`, `moderate`, `minor`) on completion
- **Treatment action** — duration depends on severity, interruptible for non-critical cases
- **Priority scheduling** — critical patients get seen first, can interrupt ongoing moderate treatments
- **Resumable treatments** — interrupted treatments resume from where they left off
- **Stochastic arrivals** — patients arrive via exponential distribution (configurable rate)

## Running it

```bash
pip install git+https://github.com/mesa/mesa.git@main
python er_model.py
```

## Sample output

```
--- ER Simulation Results (50 time units) ---
total patients:   73
discharged:       58
in treatment:     3
still waiting:    12
avg wait time:    4.2
avg critical wait: 2.1
doctor 1: treated 21 patients
doctor 2: treated 19 patients
doctor 3: treated 18 patients
```
