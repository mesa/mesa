# Tram Route Model (Continuous State Example)

## Summary

This model demonstrates Mesa's continuous state capability.

**Overview of continuous states:** Most agent-based models advance in discrete ticks, and anything that changes smoothly -- a position, a temperature, a stock level -- has to be re-integrated by hand on every tick, with the model polling on each tick to check whether some threshold has been crossed yet. That is both wasteful (nothing interesting happens on most ticks) and imprecise (the crossing is only ever detected at the end of the tick in which it happened).

A `ContinuousState` instead stores a value as a *trajectory*: a baseline value, a timestamp, and a rate of change. Reading the attribute extrapolates from that baseline to the current model time, so the value is always exact and is never wrong between ticks. Rates can be chained -- here `position' = speed` and `speed' = acceleration` -- in which case the extrapolation picks up the second-order term and becomes piecewise-quadratic.

A `Threshold` builds on this. Because the trajectory is known analytically, the exact time at which a state will reach a limit can be *solved for* rather than watched for, and a single event scheduled at that moment. When the trajectory changes -- a new acceleration, a new limit -- the threshold re-solves and re-schedules itself automatically.

To provide a simple demonstration of this capability is a tram route model.

A single tram runs an ordered route of stations. Departing a station it accelerates at a fixed rate; on reaching cruise speed it coasts; at a brake point computed from `v^2 / 2a` it applies the brakes; on reaching zero speed it has arrived, dwells for a fixed time, and departs for the next station. Every one of those transitions is a `Threshold` crossing:

```python
_cruise_threshold = Threshold(state=speed, limit=cruise_speed, callback="start_coasting", direction="rising")
_braking_threshold = Threshold(state=position, limit=brake_point, callback="brake", direction="rising")
_stop_threshold = Threshold(state=speed, limit=0.0, callback="arrive_at_station", direction="falling")
```

Note that the first two limits are not constants but the `cruise_speed` and `brake_point` Observables. `brake_point` is rewritten at the start of every segment and `cruise_speed` comes from the scenario, and in both cases the threshold observes the attribute and re-solves for the new crossing time on its own; nothing has to re-arm it.

The brake point is computed from the segment's *peak* speed rather than the cruise speed. On a short segment, or with weak acceleration, the tram never reaches cruise speed, so it accelerates to `sqrt(2d / (1/a + 1/b))` and brakes straight from there. Assuming cruise speed instead puts the brake point outside the segment, and a rising threshold that starts out already past its limit never fires at all.

The result is a model with an empty agent step. `TransitSystem.step` only samples the tram's states for the data collector -- if you deleted it, the tram would still run its route with identical timings, driven entirely by the event queue.

## How to Run

Install Mesa with recommended dependencies:

```
pip install "mesa[rec]"
```

Then run the example:

```
solara run app.py
```

Open the displayed local URL in your browser.

To run the model without visualization and print the collected trace:

```
python model.py
```

## Files

* ``model.py``: Contains the `TransitSystem` model class and its `TramScenario`.
* ``agents.py``: Contains the `Tram` agent class, its continuous states and its thresholds.
* ``app.py``: Code for the interactive visualization.
* ``utils.py``: Palette and matplotlib drawing helpers used by ``app.py``.

## Further Reading

The continuous state and threshold API used here lives in `mesa.experimental.states`, and builds on the reactive observables in `mesa.experimental.mesa_signals`. Both are experimental and carry no semver guarantees.

