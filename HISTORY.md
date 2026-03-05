<!-- vale off -->

---
title: Release History
---
# 3.5.0 (2026-02-15)
## Highlights
Mesa 3.5.0 is a major feature release that introduces a public event scheduling API, stabilizes the event system, and lays the groundwork for Mesa 4.0 by deprecating several legacy patterns.

### Public event scheduling and time advancement
The headline feature of 3.5.0 is a new public API for event scheduling and time advancement directly on `Model` (#3266). Instead of interacting with experimental `Simulator` classes, users can now schedule events and advance time with simple, expressive methods:

```python
# Run the model for a duration
model.run_for(10)       # Advance 10 time units
model.run_until(50.0)   # Run until absolute time 50

# Schedule one-off events
model.schedule_event(callback, at=25.0)    # At absolute time
model.schedule_event(callback, after=5.0)  # Relative to now

# Schedule recurring events
from mesa.time import Schedule
model.schedule_recurring(func, Schedule(interval=10, start=0))
```

For traditional ABMs, `model.run_for(1)` is functionally equivalent to `model.step()`, but this generalizes naturally to event-driven and hybrid models. The mental model shifts from "execute step N" to "advance time, and whatever is scheduled will run."

Our new [Agent activation](https://mesa.readthedocs.io/latest/tutorials/2_agent_activation.html) and [event scheduling](https://mesa.readthedocs.io/latest/tutorials/3_event_scheduling.html) tutorials cover this extensively (#3280).

### Stabilized event scheduling system
The event scheduling system (EventList, Event, EventGenerator, Schedule, Priority) has been moved from `mesa.experimental.devs` to the new stable `mesa.time` module (#3276), making event-based simulation a first-class feature of Mesa. The new `Schedule` dataclass provides a clean way to define recurring timing patterns with interval, start, end, and count parameters (#3250). See our [migration guide](https://mesa.readthedocs.io/latest/migration_guide.html#event-scheduling-and-time-advancement).

### Deprecations toward Mesa 4.0
This release deprecates several legacy patterns to prepare for Mesa 4.0:
- **Simulator classes deprecated** (#3277): `ABMSimulator` and `DEVSimulator` are replaced by the new `Model` methods above.
- **`seed` parameter deprecated** (#3147): Use the `rng` parameter in `Model.__init__()` instead.
- **AgentSet sequence behavior deprecated** (#3208): Indexing/slicing on `AgentSet` is deprecated in favor of the new `to_list()` method.
- **Portrayal dictionaries deprecated** (#3144): Use `AgentPortrayalStyle` and `PropertyLayerStyle` instead.

See our [migration guide](https://mesa.readthedocs.io/latest/migration_guide.html#mesa-3-5-0) for more detail.

### Internal architecture improvements
Under the hood, `Model` now uses an `EventGenerator` internally for step scheduling (#3260), unifying the step mechanism with the broader event system. A new `_HardKeyAgentSet` (#3219, #3224) replaces `WeakKeyDictionary`-based storage for model-managed agent collections, eliminating weak reference overhead while preventing memory leaks through automatic downgrading when creating views. An `AbstractAgentSet` base class (#3210) formalizes the AgentSet interface. Discrete spaces now distinguish between logical cell indices and physical spatial positions (#3268), laying the foundation for stacked spaces by adding a `Cell.position` property.

### New agent creation from DataFrames
Agents can now be created directly from a pandas DataFrame (#3199), mapping columns to constructor arguments:

```python
agents = MyAgent.from_dataframe(model, df)
```

### Experimental features
Several experimental features see significant progress in this release.

#### Scenarios
Explicit `Scenario` support (#3103, #3168) provides a structured way to define and manage computational experiments separately from model logic. Scenarios encapsulate parameter sets and can be used with SolaraViz (#3178), which automatically routes slider parameters to the correct Scenario or Model and reconstructs scenarios on reset.

```python
class MyScenario(Scenario):
    density: float = 0.7

class MyModel(Model):
    def __init__(self, width=40, scenario=None):
        super().__init__(scenario=scenario)

# 'density' is auto-detected as a Scenario parameter
page = SolaraViz(MyModel(), model_params={"width": 40, "density": Slider("Density", 0.7, 0, 1, 0.1)})
```

#### Reactive model and data collection
The Model class is now reactive (#3212): `model.time` is observable, and signals are emitted when agents are registered or deregistered, enabling event-driven architectures. Building on this, a new `DataRecorder` (#3145, #3295) provides a decoupled, event-driven data collection system that separates what to collect (`DataRegistry`) from when and how to store it, with memory, SQLite, Parquet, and JSON backends.

```python
self.recorder = DataRecorder(self)
self.data_registry.track_agents(self.agents, "agent_data", "wealth").record(self.recorder)
self.data_registry.track_model(self, "model_data", "gini").record(
    self.recorder, configuration=DatasetConfig(start_time=4, interval=2)
)
```

Other experimental progress includes improved meta-agent support with overlapping memberships (#3172).

Note that experimental features are in active development and can have (breaking) changes in every release.

## What's Changed
### ⏳ Deprecations
* Deprecate the agent and propertylayer portrayal dicts by @EwoutH in #3144
* Deprecate `seed` parameter in favor of `rng` in Model by @quaquel in #3147
* Deprecate AgentSet sequence behavior and add `to_list()` method by @codebyNJ in #3208
* Deprecate Simulator classes, add migration guide entry by @EwoutH in #3277
### 🎉 New features added
* Add DataFrame support to Agent creation by @falloficarus22 in #3199
* Add Schedule dataclass and refactor EventGenerator by @EwoutH in #3250
* Add public event scheduling and time advancement methods by @EwoutH in #3266
### 🛠 Enhancements made
* Fix: Cell.get_neighborhood() RecursionError for large radius (#3105) by @Nithin9585 in #3106
* Replace PropertyDescriptor with properties by @quaquel in #3125
* Ensure Cell only uses __slots__ by @quaquel in #3121
* Allow list inputs for `MesaSignal` observable names and signal types by @Sonu0305 in #3139
* Optimise create_agents by replacing 'ListLike' approach with itertools by @codebreaker32 in #3163
* Micro optimization of grid.select_random_empty_cell by @quaquel in #3214
* Introduce AbstractAgentSet to agent.py and refactor AgentSet to inherit from it by @codebreaker32 in #3210
* Introduce `_HardKeyAgentSet` in agents.py by @codebreaker32 in #3219
* Refactor step scheduling to use `EventGenerator` internally by @EwoutH in #3260
* fix: handle deprecated space_kwargs gracefully in SpaceRenderer by @DipayanDasgupta in #3269
* Distinguish Logical Index from Physical Position by @codebreaker32 in #3268
### 🧪 Experimental features
* Add explicit support for Scenarios by @quaquel in #3103
* feat: Add SignalType enum for type-safe signal definitions. by @codebyNJ in #3056
* Replace Computable Descriptor with @computed in mesa_signals by @codebreaker32 in #3153
* Add scenario property to Agent class by @EwoutH in #3164
* replace AttributeDict with a dataclass by @quaquel in #3138
* Enable type-hinted Scenario subclassing and fix Model generic typing by @EwoutH in #3168
* Optimise mesa_signals by skipping signals for empty subscribers to reduce subsequent overheads by @codebreaker32 in #3198
* Add EventGenerator class for recurring event patterns by @EwoutH in #3201
* Support multiple and overlapping meta-agent memberships by @falloficarus22 in #3172
* fix: update meta_agents extract_class to use to_list() by @Jayantparashar10 in #3241
* Making Model reactive by @quaquel in #3212
* Add data registry by @quaquel in #3156
* Add `DataRecorder` for reactive Data Storage and `DatasetConfig` for Configuration by @codebreaker32 in #3145
* Support Scenarios in SolaraViz visualization by @falloficarus22 in #3178
* Add `DataSet.record()` by @quaquel in #3295
* Resolve `DataRecorder` off-by-one timestamp error by @codebreaker32 in #3299
* Adding batch and suppress to mesa_signals by @quaquel in #3261
* Only emit signal of new value of observable is different from old value by @quaquel in #3312
### 🐛 Bugs fixed
* Fix batch_run Data Collection to Ensure Accuracy and Capture All Steps by @codebreaker32 in #3109
* Fix network visualization bug: Replace array indexing with dictionary lookup by @codebyNJ in #3045
* Fix TypeError in find_combinations when evaluation_func is None by @codebyNJ in #3112
* Make create_meta_agent deterministic by @quaquel in #3183
* Fix seed logic to ensure reproducibility by @codebreaker32 in #3192
* Bugfix for pickling dynamically modified grids by @quaquel in #3217
* check whether model reporter is a partial function by @wang-boyu in #3220
* Prevent RecursionError and data loss in Cell/Grid deepcopy by @falloficarus22 in #3222
* fix: update alliance_formation example to use AgentSet.to_list() by @souro26 in #3235
* fix: preserve falsy evaluation values in find_combinations by @souro26 in #3237
* Naming collision in meta-agents `add_atrributes` by @tpike3 in #3239
* fix: update Altair tooltip type inference for compatibility by @souro26 in #3234
* Fix Solara Altair dependency updates by @falloficarus22 in #3244
### 🔍 Examples updated
* Update wolf-sheep example to use new event scheduling API by @EwoutH in #3278
* Update Epstein Civil Violence model to use the new experimental `Scenario` class by @EwoutH in #3167
### 📜 Documentation improvements
* Add deprecation of passing portrayal arguments to draw() methods to migration guide by @EwoutH in #3202
* docs: clarify SolaraViz keyword-only model arguments by @Arjun-Sasi in #3044
* docs: Add Mesa migration guide for AgentSet sequence behavior by @codebyNJ in #3218
* docs: improve Boltzmann Wealth Model README structure and clarity by @souro26 in #3229
* Update SpaceRenderer API in Tutorials 4, 5, and 6 by @falloficarus22 in #3231
* Fix docstring in `mesa_signals/core.py` by @codebreaker32 in #3255
* cleanup by @quaquel in #3258
* Migrate tutorials and examples to model.run_for() by @falloficarus22 in #3270
* Add agent activation, event scheduling and time progression tutorials by @EwoutH in #3280
* Update overview.md for Mesa 3.5 by @EwoutH in #3310
### 🔧 Maintenance
* Fix pickling for scheduled events by serializing weak-referenced callbacks safely by @EwoutH in #3205
* Unify event list between Model and Simulator by @EwoutH in #3204
* Fix: Replace 'assert ValueError' with proper return None by @codebyNJ in #3189
* Update `model.py` to replace `AgentSet` with `_HardKeyAgentSet` by @codebreaker32 in #3224
* Use type(model.value).__init__ in ModelCreator param check by @falloficarus22 in #3264
* Stabilize event scheduling system from experimental to mesa.time by @EwoutH in #3276
* Prevent infinite loop from non-positive Schedule intervals by @souro26 in #3289

## New Contributors
* @champ-byte made their first contribution in #3098
* @Arjun-Sasi made their first contribution in #3044
* @souro26 made their first contribution in #3235
* @Jayantparashar10 made their first contribution in #3241
* @ApurvaPatil2401 made their first contribution in #3249

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.4.2...v3.5.0

# 3.4.2 (2026-01-23)
## Highlights
Mesa 3.4.2 is a bugfix release that addresses a critical memory leak affecting all Mesa models.

This release fixes a significant memory leak where model instances could never be garbage collected after agents were created (#3180). The root cause was the `Agent._ids` class attribute: a `defaultdict` that stored references to model instances to ensure `unique_id` values were unique per model. Because this was a class-level attribute persisting across the Python process, any model used as a key maintained a hard reference indefinitely, preventing cleanup of the model and all its associated objects even after going out of scope.

This bug had serious implications for users running multiple simulations or batch experiments, as each model instance would accumulate in RAM rather than being cleaned up, eventually exhausting available memory. The fix moves `unique_id` assignment from the `Agent` class into `Model.register_agent()`, with each model now maintaining its own `agent_id_counter` instance attribute. This eliminates persistent class-level references and allows proper garbage collection of model objects.

This release also includes a fix for `PropertyLayer.from_data()` to prevent unintended side effects (#3122), along with several small documentation improvements (#3104, #3124, #3127), and expanded contribution guidelines detailing Mesa's development philosophy and workflow (#3135).

## What's Changed
### 🐛 Bugs fixed
* Ensure PropertyLayer.from_data() does not have side effects by @quaquel in #3122
* Fix for Memory leak by @quaquel in #3180
### 📜 Documentation improvements
* Document `Agent` hashability requirement by @Sonu0305 in #3104
* Fix `chart_property_layers` docstring by @Sonu0305 in #3124
* Fix `remove_property_layer` docstring by @Sonu0305 in #3127
* Add Mesa development process guidelines by @EwoutH in #3135

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.4.1...v3.4.2

# 3.4.1 (2026-01-10)
## Highlights
Mesa 3.4.1 is a patch release with bug fixes, performance improvements, and documentation enhancements. This release addresses issues affecting data collection, memory management, and grid operations while introducing performance optimizations.

This release resolves several bugs affecting simulation accuracy: fixed multiple `batch_run` and `DataCollector` issues including `agenttype_reporters` support (#3095), sparse data collection (#2988), and proper handling of multiple `collect()` calls per step (#3058); resolved `MultiGrid._empty_mask` not updating correctly (#3019) and infinite loops in `select_random_empty_cell()` (#3014); fixed a memory leak in `ContinuousSpace` agent removal (#3031); corrected `EventList.peek_ahead()` chronological ordering (#3010); and ensured `FixedAgent` state consistency after removal (#3100).

Several optimizations improve Mesa's performance: `PropertyLayer` was refactored to implement the NumPy interface directly, enabling standard NumPy syntax (#3074); `select_random_empty_cell()` now uses vectorized NumPy operations instead of slow O(N) Python iteration (#3087); and `Cell.is_empty`/`is_full` checks were optimized to remove unnecessary O(n) copies (#3069).

The documentation received several improvements including version warnings for visualization tutorials (#2949), updated contribution guidelines (#3028), and implementation of Vale for consistent documentation style (#3022). The entire `mesa.space` module is now marked as maintenance-only (#3082), with users encouraged to use `mesa.discrete_space` for new projects. Generic type parameters were added to `Agent`, `AgentSet`, and `Model` classes to improve static type checking (#2885).

We're excited to welcome 9 new contributors to Mesa in this release! Thank you to everyone who contributed bug fixes, performance improvements, and documentation enhancements.

## What's Changed
### 🛠 Enhancements made
* Optimize Cell is_empty/is_full by @codebyNJ in #3069
* Refactor PropertyLayer to implement NumPy interface and deprecate wrappers by @codebreaker32 in #3074
* Optimise select_random_empty_cell() in grid.py by @codebreaker32 in #3087
* Add generic type parameters to Agent, AgentSet, and Model by @SiddharthBansal007 in #2885
### 🐛 Bugs fixed
* Fix: peak_ahead returns events in correct chronological order by @Nithin9585 in #3010
* fix: prevent infinite loop in select_random_empty_cell via heuristic fallback by @DipayanDasgupta in #3014
* Fix: datacollector missing attribute error by @codebyNJ in #3041
* Fix: Add deepcopy to agent reporters to prevent mutable reference lea… by @Nithin9585 in #3038
* Add initialization check in Simulator.run_for() by @codebreaker32 in #3036
* Fix: Method Reporter Validation in DataCollector by @vedantvakharia in #3002
* Fix MultiGrid._empty_mask not updated correctly by @Nithin9585 in #3019
* Fix: Correct data retrieval in batch_run when DataCollector.collect() called multiple times per step by @Nithin9585 in #3058
* Fix: IndexError in batch_run with sparse data collection by @Nithin9585 in #2988
* Fix: Add agenttype_reporters support to batch_run by @BhoomiAgrawal12 in #3095
* Fix memory leak in ContinuousSpace agent removal by @Nithin9585 in #3031
* Minor Refactoring in solara_viz by @codebreaker32 in #3059
* Fix: Correct `FixedAgent` state after removal by @Sonu0305 in #3100
### 🔍 Examples updated
* Resolve FIXME in `sugarscape_g1mt/agents.py` by @Sonu0305 in #3062
* Update alliance formation mode by @quaquel in #3075
### 📜 Documentation improvements
* document Model.time in Model API by @Gee307 in #3020
* docs: Add example structure and policy to contributing guide by @EwoutH in #3028
* Changed broken documentation link in docs/tutorials/1_adding_space.ipynb by @ShashwatAwate in #2973
* Updated CONTRIBUTING.md to replace old black documentation with new ruff format. by @falloficarus22 in #3071
* docs: Add Mesa 3.3+ version warnings to visualization tutorials by @Srinath0916 in #2949
* Implementation of Vale by @vedantvakharia in #3022
* Mark whole `mesa.space` module maintenance-only by @quaquel in #3082
### 🔧 Maintenance
* CI: Restore coverage collection after test reorganization by @falloficarus22 in #3006
* Fix flaky Playwright test in `test_examples_viz.py` by @vedantvakharia in #3039
* tests: consolidate Solara viz tests and restore Altair coverage #2993 by @DipayanDasgupta in #3011
* test: add coverage for ignore_missing=True in DataCollector by @disgruntled-penguin in #3054
* test: ensure DataCollector raises ValueError when collecting data for… by @disgruntled-penguin in #3053
* Lint: Enable RUF012 and annotate mutable class attributes with ClassVar by @falloficarus22 in #3033

## New Contributors
* @Gee307 made their first contribution in #3020
* @DipayanDasgupta made their first contribution in #3014
* @ShashwatAwate made their first contribution in #2973
* @vedantvakharia made their first contribution in #3039
* @codebyNJ made their first contribution in #3041
* @disgruntled-penguin made their first contribution in #3054
* @Sonu0305 made their first contribution in #3062
* @Srinath0916 made their first contribution in #2949
* @BhoomiAgrawal12 made their first contribution in #3095

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.4.0...v3.4.1

# 3.4.0 (2025-12-24)
## Highlights
The Mesa 3.4.0 feature release introduces universal time tracking, improves batch run reproducibility, and strengthens our deprecation policy. This release also requires Python 3.12+ and includes numerous bug fixes and quality-of-life improvements.

### Universal simulation time with `model.time`
Mesa now provides a single source of truth for simulation time through the `model.time` attribute (#2903). Previously, time was fragmented across different components - simple models used `model.steps` as a proxy, while discrete event simulations stored time in `simulator.time`. This created complexity for features needing consistent time access.

Now all models have a `model.time` attribute that:
- Automatically increments with each step (by 1.0)
- Works seamlessly with discrete event simulators
- Provides a consistent interface for data collection, visualization, and user code

```python
# Basic usage - time auto-increments
model = Model()
model.step()
print(model.time)  # 1.0

# With discrete event simulation
simulator = DEVSimulator()
simulator.setup(model)
simulator.schedule_event_absolute(some_function, 2.5)
simulator.run_until(3.0)
print(model.time)  # 3.0
```

The old `simulator.time` still works but emits a deprecation warning.

### Improved batch run reproducibility
The `batch_run` function has been updated to provide explicit control over random seeds across replications (#2841). Previously, using `iterations` with a fixed seed caused all iterations to use the same seed, producing identical results instead of independent replications.

The new `rng` parameter accepts either a single seed value or an iterable of seed values, giving you complete control over reproducibility:

```python
rng = np.random.default_rng(42)
seed_values = rng.integers(0, sys.maxsize, size=(5,))

results = mesa.batch_run(
    MoneyModel,
    parameters=params,
    rng=seed_values.tolist(),  # Each iteration uses a different seed
)
```

The old `iterations` parameter is now deprecated and will be removed in Mesa 4.0. See the [migration guide](https://mesa.readthedocs.io/latest/migration_guide.html#batch-run) for details on updating your code.

### Strengthened deprecation policy
Mesa now has a formal [deprecation policy](https://github.com/mesa/mesa/blob/main/CONTRIBUTING.md#deprecation-policy) that ensures users have adequate time to migrate while allowing Mesa to evolve (#2900). A related change is that all deprecation warnings now use `FutureWarning` instead of `DeprecationWarning` (#2905), making them visible by default since `DeprecationWarning` is hidden for imported modules.

The policy guarantees:
- New features must be available for at least one minor release before deprecating old ones
- Documentation, migration guides, and examples must be updated before active deprecation
- Breaking changes only occur in major version releases
- Experimental features have more flexibility but still communicate changes

### Organization name change
Mesa has migrated from the `projectmesa` to `mesa` organization on GitHub (#2880), (#2887), thanks to Daniel Langemann ([@dlangemann](https://github.com/dlangemann)) who generously transferred the `mesa` name to us. Our repositories are now accessible at `github.com/mesa/mesa` instead of `github.com/projectmesa/mesa`, signaling authority and maturity in a way that better reflects Mesa's position as *the* agent-based modeling framework in Python. GitHub automatically redirects old links, so existing URLs and git remotes continue to work seamlessly.

### Python 3.12+ required
Mesa 3.4.0 drops support for Python 3.11 and now requires Python 3.12 or higher (#2842). This allows Mesa to use modern Python type parameter syntax and prepares the codebase for future Python features. Python 3.14 is now also fully tested in CI.

### Other improvements
This release includes significant enhancements to the visualization system, including support for `AgentPortrayalStyle` in Altair components, improved property layer styling, and better error handling. The experimental cell space module has been removed in favor of the stable `mesa.discrete_space` module (#2969).

Numerous bugs were fixed, including issues with scalar color handling in matplotlib, empty cell collection indexing, and cell capacity handling. The test suite was reorganized to mirror the source module structure and now treats warnings as errors to prevent accumulation.

We welcome 10 new contributors to the Mesa project in this release! Thank you to everyone who contributed bug fixes, documentation improvements, and feature enhancements.

## What's Changed
### ⏳ Deprecations
* Replace DeprecationWarnings with FutureWarnings by @EwoutH in #2905
* Fix: Auto-increment seed across batch_run iterations by @EwoutH in #2841
### 🎉 New features added
* Add `model.time` as universal source of truth for simulation time by @EwoutH in #2903
### 🛠 Enhancements made
* Refactor SpaceRenderer API to separate setup and draw methods by @Sahil-Chhoker in #2893
* Validate HexGrid torus dimensions by @ShreyasN707 in #2951
* Allow PropertyLayerStyle instance in draw_propertylayer by @ShreyasN707 in #2936
* Support AgentPortrayalStyle in Altair visualization components by @EwoutH in #2985
### 🐛 Bugs fixed
* Update visualization tabs to be compatible with ipyvuetify 3.0 by @EwoutH in #2919
* Use pandas type checking for numeric dtype detection in Altair backend by @EwoutH in #2917
* Fix for Model.reset_rng by @quaquel in #2946
* Fix scalar color handling in legacy `matplotlib` backend. by @falloficarus22 in #2959
* Fix: IndexError in select_random_agent when cell collection is empty by @Nithurshen in #2983
* Fix: Handle capacity=None in Cell.is_full property by @Nithin9585 in #2981
* Fix/cell capacity zero by @ahmednabiled in #2990
* Fix SolaraViz multipage rendering when renderer is absent by @falloficarus22 in #2966
### 🔍 Examples updated
* examples: Add Activation Order and Grid Type selectors to EpsteinCivilViolence by @Nithurshen in #2955
### 📜 Documentation improvements
* Docs: Link to online example model demo in a few places by @EwoutH in #2882
* Update overview.md by @quaquel in #2883
* docs: Changed colab notebook url for tutorial 3 - agentset by @aten2005 in #2890
* Add AI/LLM use guidelines to Code of Conduct by @EwoutH in #2898
* docs: Add deprecation policy to contributor guide by @EwoutH in #2900
* [Docs] Fix typos and inconsistent comments in tutorials by @Mannan-15 in #2948
* Remove capacity constraint on cells from tutorial by @quaquel in #2964
* Fix Navigation Issue by @codebreaker32 in #2938
### 🔧 Maintenance
* Drop Python 3.11, require Python 3.12+ and update to modern type parameter syntax by @EwoutH in #2842
* Pin Readthedocs build to Python 3.13 by @EwoutH in #2899
* tests: Resolved "Missing Random Number Generator" warnings by @ShreyasN707 in #2911
* tests: Update agent portrayals to use AgentPortrayalStyle by @Tejasv-Singh in #2909
* Add Python 3.14 support to workflows and metadata by @EwoutH in #2843
* Fix dtype/default_value mismatches in PropertyLayer tests to resolve UserWarnings by @ShreyasN707 in #2913
* Replace deprecated np.bmat with np.block in Voronoi Delaunay triangulation by @EwoutH in #2915
* Speed up CI by installing only Playwright chromium-headless-shell by @EwoutH in #2916
* Add deprecation category to release notes by @EwoutH in #2914
* test: Update simulator.time deprecation test to expect FutureWarning by @EwoutH in #2922
* tests: Migrate PropertyLayer portrayals from `dict` to `PropertyLayerStyle` by @Tejasv-Singh in #2920
* Fix matplotlib figure memory leak warning in tests by @EwoutH in #2924
* Update organization name from `projectmesa` to `mesa` by @EwoutH in #2880
* Update dict-based agent portrayals to use `AgentPortrayalStyle` in tests by @falloficarus22 in #2934
* Remove experimental cell_space module and update references by @EwoutH in #2969
* Fix PropertyLayer default value type mismatch warnings by @falloficarus22 in #2963
* test: Replace deprecated `iterations` with `rng` in batch_run tests by @EwoutH in #2984
* End of year cleanup by @EwoutH in #2971
* Reorganize tests to mirror source module structure by @EwoutH in #2994
* SolaraViz: move hooks before early return in ComponentsView by @EwoutH in #2925
* Fix reproducibility warnings by adding explicit random parameters by @codebreaker32 in #2978
* ci: treat warnings as errors to prevent accumulation by @EwoutH in #2926
* Reject negative time_delta in schedule_event_relative by @Nithin9585 in #2999
* CI: Add job that runs tests with pip pre-release dependencies by @EwoutH in #1852
* CI: Split off separate coverage job by @EwoutH in #3005
* Model: remove unreleased `step_duration` parameter by @EwoutH in #3007

## New Contributors
* @GlyphicGuy made their first contribution in #2892
* @aten2005 made their first contribution in #2890
* @Tejasv-Singh made their first contribution in #2908
* @ShreyasN707 made their first contribution in #2911
* @Mannan-15 made their first contribution in #2948
* @falloficarus22 made their first contribution in #2959
* @codebreaker32 made their first contribution in #2938
* @Nithurshen made their first contribution in #2955
* @Nithin9585 made their first contribution in #2981
* @ahmednabiled made their first contribution in #2990

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.3.1...v3.4.0

# 3.3.1 (2025-11-07)
## Highlights
Mesa 3.3.1 is a maintenance release focused on bug fixes and documentation improvements following the major 3.3.0 visualization update.

This release addresses two critical visualization bugs affecting PropertyLayers on HexGrids and property layer data mapping across both Altair and Matplotlib backends.

The documentation received several important updates, including fixes to tutorial code examples, a new guide for Google Summer of Code contributors, and improved organization of the documentation structure. The migration guide has been updated to reflect the deprecation of the old `agent_portrayal` parameter in favor of the new `AgentPortrayalStyle` introduced in Mesa 3.3.0.

We're excited to welcome six new contributors to the Mesa project in this release! Thank you to everyone who contributed bug fixes, documentation improvements, and test coverage enhancements.

## What's Changed
### 🐛 Bugs fixed
* Fix visualization error for PropertyLayers on HexGrids. Add transpose … by @flucco in https://github.com/mesa/mesa/pull/2868
* Fix: Property layer data mapping for both Altair and Matplotlib backends by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2869
* Fix: AgentSet initialization should not require explicit random number generator by @verisimilidude2 in https://github.com/mesa/mesa/pull/2789
### 🔍 Examples updated
* Fix missing variable g in the tutorial by @pazmiller in https://github.com/mesa/mesa/pull/2849
### 📜 Documentation improvements
* fix broken link in example by @BigTurtle8 in https://github.com/mesa/mesa/pull/2847
* Docs: Fix ValueError in tutorial by adding default value to MoneyModel by @pragam-m25 in https://github.com/mesa/mesa/pull/2871
* Add guide for GSoC contributors by @colinfrisch in https://github.com/mesa/mesa/pull/2873
* Deprecate `agent_portrayal` and update Migration guide by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2872
* Docs: List overview separate from tutorials by @dhiraj-143r in https://github.com/mesa/mesa/pull/2878
### 🔧 Maintenance
* ruff fixes in visualization by @quaquel in https://github.com/mesa/mesa/pull/2867
* [pre-commit.ci] pre-commit autoupdate: Update to ruff v0.13 by @pre-commit-ci[bot] in https://github.com/mesa/mesa/pull/2840
* Tests: Add test for InputText in UserInputs by @pragam-m25 in https://github.com/mesa/mesa/pull/2870

## New Contributors
* @BigTurtle8 made their first contribution in https://github.com/mesa/mesa/pull/2847
* @pazmiller made their first contribution in https://github.com/mesa/mesa/pull/2849
* @flucco made their first contribution in https://github.com/mesa/mesa/pull/2868
* @pragam-m25 made their first contribution in https://github.com/mesa/mesa/pull/2870
* @dhiraj-143r made their first contribution in https://github.com/mesa/mesa/pull/2878
* @verisimilidude2 made their first contribution in https://github.com/mesa/mesa/pull/2789

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.3.0...v3.3.1

# 3.3.0 (2025-09-06)
## Highlights

The major highlight of release 3.3.0 is the introduction of a new and improved visualization module. This effort was @Sahil-Chhoker's Google Summer of Code project . The new module is backwards compatible and continues to use Solara. It has several new and improved features to include:
- `AgentPortrayalStyle`: a more user-friendly way to specify agent portrayal
- `PropertyLayerStyle`:  Makes property_layer and agent portrayal consistent.
- `SpaceRender`: a new component for drawing spaces, agents and property layers with extensive customization
- Improved support for Altair and Matplotlib
- Multipage support (e.g., users can display simulation  on one page and charts of the model on another)
- Updated tutorials for visualization

[You can read more about the update here](https://gist.github.com/Sahil-Chhoker/470d78683664dffc05725bd858beb494)

In addition, there were many other improvements to mesa, from bug fixes to improved CI/CD pipelines. Thanks to the PyCON sprints and the developers who supported Mesa!

## What's Changed
### 🧪 Experimental features
* Introduction of Space Renderer by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2803
### 🎉 New features added
* docs/nix flake by @hussainsultan in https://github.com/mesa/mesa/pull/2788
### 🛠 Enhancements made
* Added `AgentPortrayalStyle` and `PropertyLayerStyle` by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2786
* Allow image markers to plot agents by @Holzhauer in https://github.com/mesa/mesa/pull/2799
* Renderer null check fix in solaraviz by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2809
* Add Altair plotting functionality by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2810
* Fix: Grid not showing in altair if both draw_structure and draw_agents are called. by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2817
* Allowing color mapping to integers in AgentPortrayalStyle by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2818
* Added multipage functionality by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2827
### 🐛 Bugs fixed
* Bugfix ModelCreator for required model parameters and user adjusted model parameters by @Holzhauer in https://github.com/mesa/mesa/pull/2780
* Remove unneeded int cast by @derkweijers in https://github.com/mesa/mesa/pull/2791
* Add quotes to readme to insure multi-terminal compliance. by @jackiekazil in https://github.com/mesa/mesa/pull/2787
* Fix broken docs link - Fix issue 2792 by @catherinedevlin in https://github.com/mesa/mesa/pull/2793
* Bug fixes for portrayal components by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2798
* fix : prevent breakdown when citizen_density (Initial Agent Density s… by @colinfrisch in https://github.com/mesa/mesa/pull/2806
* Fix issue #2812 by @Tosiekdev in https://github.com/mesa/mesa/pull/2821
### 📜 Documentation improvements
* Benchmarks documentation by @colinfrisch in https://github.com/mesa/mesa/pull/2764
* minor corrections in tutorials by @Holzhauer in https://github.com/mesa/mesa/pull/2778
* Correct example code in "Overview of the MESA library" by @Holzhauer in https://github.com/mesa/mesa/pull/2781
* Updating Examples and Docs inline with the new API. by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2819
### 🔧 Maintenance
* Add `.coderabbit.yaml` file to allow reviewing and updating our CodeRabbit configuration by @EwoutH in https://github.com/mesa/mesa/pull/2761
* add nix flake by @hussainsultan in https://github.com/mesa/mesa/pull/2785
* Removing Deprecations Warnings from `agent_portrayal` and `propertylayer_portrayal` until next release. by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2797
* Added Basic Visualization tests by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2767
### Other changes
* Update __init__.py by @tpike3 in https://github.com/mesa/mesa/pull/2770
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/mesa/mesa/pull/2751
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/mesa/mesa/pull/2802
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/mesa/mesa/pull/2808
* Bump actions/download-artifact from 4 to 5 by @dependabot[bot] in https://github.com/mesa/mesa/pull/2826
* Bump actions/checkout from 4 to 5 by @dependabot[bot] in https://github.com/mesa/mesa/pull/2825
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/mesa/mesa/pull/2823

## New Contributors
* @derkweijers made their first contribution in https://github.com/mesa/mesa/pull/2791
* @hussainsultan made their first contribution in https://github.com/mesa/mesa/pull/2785
* @catherinedevlin made their first contribution in https://github.com/mesa/mesa/pull/2793
* @Tosiekdev made their first contribution in https://github.com/mesa/mesa/pull/2821

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.2.0...v3.3.0

# 3.2.0 (2025-05-04)
## Highlights
Mesa 3.2.0 is a feature-packed release, which stabilizes our discrete space, improves many of our visualisation capabilities, improves our tutorial organization, adds the experimental meta-agents, and includes other quality of life enhancements.

We also celebrate the publication of our peer-reviewed [Mesa 3 paper](https://doi.org/10.21105/joss.07668) in JOSS. See the updated [Citing Mesa](https://github.com/mesa/mesa?tab=readme-ov-file#citing-mesa) section on how to cite us.

### Stabilization of Discrete Space
The experimental Cell Space system has been stabilized and is now available as `mesa.discrete_space` ([#2610](https://github.com/mesa/mesa/pull/2610)). This powerful spatial modeling framework enables cell-centric simulations with integrated PropertyLayers and improved agent movement capabilities. Key improvements include:

- Support for dynamic modifications to discrete spaces during simulation ([#2755](https://github.com/mesa/mesa/pull/2755))
- Methods to add/remove cells and connections in real-time
- Full integration with PropertyLayers ([#2440](https://github.com/mesa/mesa/pull/2440)) for representing environmental variables
- Compatible with all examples and existing visualizations

The PropertyLayer itself has also been stabilized, allowing for efficient management of spatial environmental properties like terrain, resources, or any grid-based variables. Core examples including [Schelling](https://github.com/mesa/mesa/tree/main/mesa/examples/basic/schelling), [Game of Life](https://github.com/mesa/mesa/tree/main/mesa/examples/basic/conways_game_of_life), [Boltzmann Wealth](https://github.com/mesa/mesa/tree/main/mesa/examples/basic/boltzmann_wealth_model), and [Virus on Network](https://github.com/mesa/mesa/tree/main/mesa/examples/basic/virus_on_network) have been updated to use the new discrete space system.

### Enhanced Visualization Experience
The SolaraViz visualization system has received substantial upgrades:

- **Command Console** ([#2697](https://github.com/mesa/mesa/pull/2697)): An interactive Python console embedded directly in the visualization, allowing real-time model inspection and manipulation
- **Asynchronous Updates** ([#2656](https://github.com/mesa/mesa/pull/2656)): Visualization now runs in a separate thread, dramatically improving performance for complex models
- **Dark Mode** ([#2689](https://github.com/mesa/mesa/pull/2689)): Support for Solara Dark theme, automatically matching system preferences
- **Improved Error Handling** ([#2747](https://github.com/mesa/mesa/pull/2747), [#2753](https://github.com/mesa/mesa/pull/2753)): Better visualization of errors with options to view detailed traceback information
- **Enhanced UI**: Arrow key navigation ([#2725](https://github.com/mesa/mesa/pull/2725)), movable input field with auto-scroll ([#2710](https://github.com/mesa/mesa/pull/2710)), and other quality-of-life improvements
- **PropertyLayer visualisation in Altair** ([#2643](https://github.com/mesa/mesa/pull/2643)): Support for visualising PropertyLayers using the Altair frontend in Solara.

### Restructured and updated tutorial
Our [introduction tutorial](https://mesa.readthedocs.io/latest/overview.html) has been completely restructured ([#2731](https://github.com/mesa/mesa/pull/2731)). Instead of one huge notebook, the tutorial is now divided into smaller, focused modules that build progressively:

- **Creating Your First Model**: Essential basics to get started
- **Adding Space**: Learn how to use the new discrete space system
- **Collecting Data**: Effectively gather model statistics
- **AgentSet**: Master agent management techniques
- **Visualization**: Series of modules covering basic to advanced visualization


### Introducing Meta-Agents
Complex systems often have multiple levels of components. An organization is not one entity, but is made of departments, sub-departments, and people. A person is not a single entity, but it is made of micro biomes, organs and cells. A city is not a single entity, but it is made of districts, neighborhoods, buildings, and people. A forest comprises an ecosystem of trees, plants, animals, and microorganisms.

This reality is the motivation for meta-agents. It allows users to represent these multiple levels, where each level can have meta-agents with constituting agents.
- **Meta agents** [#2748](https://github.com/mesa/mesa/pull/2748)

To demonstrate meta-agents capability there are two examples:

1.  Dynamic meta-agent creation - [Alliance formation](https://mesa.readthedocs.io/latest/examples/advanced/alliance_formation.html), which shows emergent meta-agent formation through a game theoretic based alliance formation.
2. Deliberate meta-agent creation - [Warehouse model](https://github.com/mesa/mesa-examples/tree/main/examples/warehouse), which provides a pseudo warehouse model to demonstrate meta-agent functionality.

Let us know what you think on our [Matrix Channel](https://matrix.to/#/#project-mesa:matrix.org)

### Other improvements
- AgentSet's `agg` method now supports multiple aggregation functions in a single call ([#2743](https://github.com/mesa/mesa/pull/2743))
- Many documentation improvements have been made based on feedback by our JOSS reviewers

## What's Changed
### 🎉 New features added
* Stabilize experimental Cell Space as `mesa.discrete_space` by @quaquel in https://github.com/mesa/mesa/pull/2610
* Feat: Added Command Console by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2697
* Stabilize PropertyLayer by @EwoutH in https://github.com/mesa/mesa/pull/2440
### 🛠 Enhancements made
* Visualisation: Add dark mode by @sanika-n in https://github.com/mesa/mesa/pull/2689
* Creating threads to update visualization asynchronously  by @HMNS19 in https://github.com/mesa/mesa/pull/2656
* Movable Input Field with auto-scroll by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2710
* Feat: Added arrow key navigation by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2725
* added property_layer with altair by @sanika-n in https://github.com/mesa/mesa/pull/2643
* Debug option for errors in visualisation by @colinfrisch in https://github.com/mesa/mesa/pull/2747
* Debug option for errors in visualisation front end by @colinfrisch in https://github.com/mesa/mesa/pull/2753
* Add support for dynamic discrete spaces by @quaquel in https://github.com/mesa/mesa/pull/2755
* Support multiple functions in AgentSet `agg` method by @EwoutH in https://github.com/mesa/mesa/pull/2743
### 🔍 Examples updated
* Move boltzmann to cell spaces by @quaquel in https://github.com/mesa/mesa/pull/2680
* Move Game of life to cell space by @quaquel in https://github.com/mesa/mesa/pull/2681
* Move Schelling to discrete space by @quaquel in https://github.com/mesa/mesa/pull/2684
* Move virus_on_a_network to discrete_space by @quaquel in https://github.com/mesa/mesa/pull/2688
* Update boid_flockers example include flight direction by @sanika-n in https://github.com/mesa/mesa/pull/2696
### 📜 Documentation improvements
* docs: Split off the overview page, extend with spaces/activation by @EwoutH in https://github.com/mesa/mesa/pull/2673
* docs: Add Roles section to CONTRIBUTING.md by @EwoutH in https://github.com/mesa/mesa/pull/2694
* [JOSS] Docs: Clarify difference between core and user examples by @EwoutH in https://github.com/mesa/mesa/pull/2706
* [JOSS] docs: Improve batch_run documentation for parallel processing by @EwoutH in https://github.com/mesa/mesa/pull/2707
* JOSS Tutorial Fixes by @tpike3 in https://github.com/mesa/mesa/pull/2708
* Clean-up old images, compress wolf-sheep image, remove version switch artifact by @EwoutH in https://github.com/mesa/mesa/pull/2717
* docs: Update citation to JOSS article by @EwoutH in https://github.com/mesa/mesa/pull/2740
* update intro tutorial discrete space by @tpike3 in https://github.com/mesa/mesa/pull/2731
* docs: fix broken links to Readthedocs in example descriptions by @reyan-singh in https://github.com/mesa/mesa/pull/2757
### 🧪 Experimental features
* Meta agents by @tpike3 in https://github.com/mesa/mesa/pull/2748
### 🐛 Bugs fixed
* Minor (bug) fixes to discrete_space by @quaquel in https://github.com/mesa/mesa/pull/2687
* Fixed Marker in mpl by @sanika-n in https://github.com/mesa/mesa/pull/2670

## New Contributors
* @colinfrisch made their first contribution in https://github.com/mesa/mesa/pull/2747
* @reyan-singh made their first contribution in https://github.com/mesa/mesa/pull/2757

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.1.4...v3.2.0

# 3.1.5 (2025-03-26)
## Highlights
Mesa 3.1.5 is a maintenance release focused on improving documentation and cleaning up our resources. This update includes no breaking changes and is compatible with previous 3.1.x releases.

The documentation improvements include enhanced explanations for batch run parallel processing, a significantly expanded overview section covering spaces, property layers and time advancement methods with practical code examples, and a clearer installation guide detailing what the optional `[rec]` dependencies provide. We've also improved our examples documentation to better distinguish between core and user-contributed examples, and added a new "Roles" section to the contributing guide that outlines the project's community structure and progression paths.

On the maintenance side, we've cleaned up unused images and compressed existing ones, reducing the repository size by over 2MB and distribution size from 3.2MB to 1.2MB.

## What's Changed
### 📜 Documentation improvements
* docs: Split off the overview page, extend with spaces/activation by @EwoutH in https://github.com/mesa/mesa/pull/2673
* docs: Add Roles section to CONTRIBUTING.md by @EwoutH in https://github.com/mesa/mesa/pull/2694
* [JOSS] Docs: Clarify difference between core and user examples by @EwoutH in https://github.com/mesa/mesa/pull/2706
* [JOSS] docs: Improve batch_run documentation for parallel processing by @EwoutH in https://github.com/mesa/mesa/pull/2707
* Clean-up old images, compress wolf-sheep image, remove version switch artifact by @EwoutH in https://github.com/mesa/mesa/pull/2717

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.1.4...v3.1.5

# 3.1.4 (2025-02-09)
## Highlights
This release contains various improvements and bugfixes to the matplotlib-based visualization of spaces. Hexgrids are now fully supported, including property layers. In making this possible, various minor bugs were encountered and also fixed. In addition to the visualization improvements, there are various minor convenience improvements to the docs.

## What's Changed
### 🛠 Enhancements made
* Fixed hex-space draw function to avoid overlaps by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2609
* Fix: Property layer visualization for HexGrid by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2646
* Enhance DataCollector to validate model_reporters functions by @peter-kinger in https://github.com/mesa/mesa/pull/2605
* Implemented post_process in Altair based components by @sanika-n in https://github.com/mesa/mesa/pull/2641
### 🐛 Bugs fixed
* bugfix for draw_property_layer by @quaquel in https://github.com/mesa/mesa/pull/2639
* Uses array for hex grid property layer fix by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2651
* Update Binder environment to use latest Mesa version (#2652) by @aarav-shukla07 in https://github.com/mesa/mesa/pull/2655
* Change Hexgrid._connect_cells_2d to use x,y coordinates by @quaquel in https://github.com/mesa/mesa/pull/2632
### 🔍 Examples updated
* Added property layer viz to sugarscape by @sanika-n in https://github.com/mesa/mesa/pull/2653
* added color-bar for spice by @sanika-n in https://github.com/mesa/mesa/pull/2622
### 📜 Documentation improvements
* remove any reference to using --pre by @quaquel in https://github.com/mesa/mesa/pull/2618
* Updated Docs by @sanika-n in https://github.com/mesa/mesa/pull/2624
* Fixed 404 error for Examples Tab in Introductory Tutorial (#2662) by @aarav-shukla07 in https://github.com/mesa/mesa/pull/2664
* Documentation by @Spartan-71 in https://github.com/mesa/mesa/pull/2630
* Adding a copy option at the top of the code written in the docs by @PrashantChoudhary13579 in https://github.com/mesa/mesa/pull/2628
* Adding Mesa Extension page by @PrashantChoudhary13579 in https://github.com/mesa/mesa/pull/2627
### 🔧 Maintenance
* remove remnants of mesa cli by @quaquel in https://github.com/mesa/mesa/pull/2617
* benchmarks.yml: Install SciPy and use uv for pip install by @EwoutH in https://github.com/mesa/mesa/pull/2633
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/mesa/mesa/pull/2659

## New Contributors
* @PrashantChoudhary13579 made their first contribution in https://github.com/mesa/mesa/pull/2628
* @aarav-shukla07 made their first contribution in https://github.com/mesa/mesa/pull/2655
* @peter-kinger made their first contribution in https://github.com/mesa/mesa/pull/2605

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.1.3...v3.1.4

# 3.1.3 (2025-01-11)
## Highlights
Mesa 3.1.3 introduces a major experimental reimplementation of Mesa's continuous space, providing an intuitive agent-centric API and significant performance improvements. The new implementation supports n-dimensional spaces and offers streamlined methods for agent movement and neighbor calculations.

### New Continuous Space Features
- Agent-centric movement API similar to cell spaces
- Efficient neighbor calculations and position updates
- Support for n-dimensional spaces
- Improved memory management with dynamic array resizing

Here's a quick look at the new API:

```python
# Create a 2D continuous space
space = ContinuousSpace(
    dimensions=[[0, 1], [0, 1]],
    torus=True,
    random=model.random
)

# Create and position an agent
agent = ContinuousSpaceAgent(space, model)
agent.position = [0.5, 0.5]

# Move agent using vector arithmetic
agent.position += [0.1, 0.1]

# Get neighbors within radius
neighbors, distances = agent.get_neighbors_in_radius(radius=0.2)

# Find k nearest neighbors
nearest, distances = agent.get_nearest_neighbors(k=5)
```

The new implementation particularly benefits models requiring frequent position updates and neighbor queries, such as flocking simulations or particle systems. See [#2584](https://github.com/mesa/mesa/pull/2584) for more details. We would love to get feedback on the new Continuous Space in [#2611](https://github.com/mesa/mesa/discussions/2611).

Other improvements in this release include consistent visualization behavior across space types with the reimplementation of `draw_voronoi` [#2608](https://github.com/mesa/mesa/pull/2608), and a new render interval slider for controlling visualization update frequency in SolaraViz, which helps improve performance when working with complex visualizations [#2596](https://github.com/mesa/mesa/pull/2596). We've also fixed a bug affecting random number generation determinism when using `Model(seed=something)`, ensuring both `model.random` and `model.rng` now behave consistently when seeded with the same initial value [#2598](https://github.com/mesa/mesa/pull/2598).

## What's Changed
### 🧪 Experimental features
* Reimplementation of Continuous Space by @quaquel in https://github.com/mesa/mesa/pull/2584
### 🛠 Enhancements made
* reimplementation of draw_voroinoi by @quaquel in https://github.com/mesa/mesa/pull/2608
* Add render interval slider to control visualization update frequency by @HMNS19 in https://github.com/mesa/mesa/pull/2596
### 🐛 Bugs fixed
* Bugfix for non deterministic rng behavior by @quaquel in https://github.com/mesa/mesa/pull/2598
### 🔍 Examples updated
* Clarify ContinuousSpace.get_neighbors behavior with multiple agents at same position by @quaquel in https://github.com/mesa/mesa/pull/2599

## New Contributors
* @HMNS19 made their first contribution in https://github.com/mesa/mesa/pull/2596

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.1.2...v3.1.3

# 3.1.2 (2025-01-04)
## Highlights
Mesa v3.1.2 is a patch release containing updates to our wolf-sheep, shelling and prisoner's dilemma example models and improving documentation in the tutorials and visualisation docstring. No functional changes to the core library were made.

## What's Changed
### 🔍 Examples updated
* examples/wolf_sheep: Don't allow dumb moves by @EwoutH in https://github.com/mesa/mesa/pull/2503
* Added homophily ratio in basic schelling example by @vbv-shm in https://github.com/mesa/mesa/pull/2520
* examples: Update pd_grid analysis.ipynb to use new spaces by @quaquel in https://github.com/mesa/mesa/pull/2553
### 📜 Documentation improvements
* Corrected a few errors in Intro tutorial by @sanika-n in https://github.com/mesa/mesa/pull/2583
* Small draw_space docstring fix by @quaquel in https://github.com/mesa/mesa/pull/2554
* fix: model name in visualization tutorial by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2591

## New Contributors
* @vbv-shm made their first contribution in https://github.com/mesa/mesa/pull/2520
* @sanika-n made their first contribution in https://github.com/mesa/mesa/pull/2583

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.1.1...v3.1.2

# 3.1.1 (2024-12-14)
## Highlights
Mesa 3.1.1 is a maintenance release that includes visualization improvements and documentation updates. The key enhancement is the addition of an interactive play interval control to the visualization interface, allowing users to dynamically adjust simulation speed between 1ms and 500ms through a slider in the Controls panel.

Several example models were updated to use Mesa 3.1's recommended practices, particularly the `create_agents()` method for more efficient agent creation and NumPy's `rng.integers()` for random number generation. The Sugarscape example was modernized to use PropertyLayers.

Bug fixes include improvements to PropertyLayer visualization and a correction to the Schelling model's neighbor similarity calculation. The tutorials were also updated to reflect current best practices in Mesa 3.1.

## What's Changed
### 🎉 New features added
* Add Interactive Play Interval Control to Mesa Visualization by @AdamZh0u in https://github.com/mesa/mesa/pull/2540
### 🐛 Bugs fixed
* bug fixes for draw_property_layers by @quaquel in https://github.com/mesa/mesa/pull/2548
### 🔍 Examples updated
* Wolf-sheep to use `create_agent` by @quaquel in https://github.com/mesa/mesa/pull/2543
* Shift sugarscape example to using create_agent by @quaquel in https://github.com/mesa/mesa/pull/2544
* Fix: Schelling Model Neighbor Similarity Calculation by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2518
* Change pd_grid example to use create_agents by @quaquel in https://github.com/mesa/mesa/pull/2545
* Switch sugarscape to using property layers by @quaquel in https://github.com/mesa/mesa/pull/2546
### 📜 Documentation improvements
* Updated docs and check_model param by @nissu99 in https://github.com/mesa/mesa/pull/2510
* Update tutorials to use `create_agents` and `rng.integers` by @DarshPareek in https://github.com/mesa/mesa/pull/2541

## New Contributors
* @nissu99 made their first contribution in https://github.com/mesa/mesa/pull/2510
* @DarshPareek made their first contribution in https://github.com/mesa/mesa/pull/2541

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.1.0...3.1.1

# 3.1.0 (2024-12-04)
## Highlights
With Mesa 3.1.0 we're back on our regular release schedule after the big Mesa 3.0 release, with some exciting new features.

This release adds experimental support for Observables and Computed, enabling a more reactive and responsive programming model for agent-based simulations. The new `Observable` and `Computable` classes allow developers to declaratively define attributes that automatically emit signals when their values change, and compute derived values that update dynamically. This lays the groundwork for more advanced event handling and data visualization features in future releases (#2291).

The experimental cell space module has been updated with full support for n-dimensional property layers. These allow agents to easily interact with and modify spatial properties of the environment, such as terrain, resources, or environmental conditions. The new implementation provides a more intuitive attribute-based API and ensures tight integration with the cell space architecture (#2512).

Mesa now includes built-in support for logging using the standard Python `logging` module. This provides developers with a flexible and powerful way to add structured diagnostic and debug output to their simulations, without the need for custom logging solutions. The logging system is integrated throughout the library, including the new SolaraViz visualization system (#2506).

Creating multiple agents with varying initialization parameters is now significantly easier with the new `Agent.create_agents` class method. This factory function supports both uniform and per-agent parameters, simplifying the code required to set up a simulation with a large number of heterogeneous agents (#2351).

In addition to the major new features, this release includes a number of smaller enhancements and bug fixes that improve the overall developer experience. These include removing deprecated functionality, cleaning up examples, and addressing various edge cases reported by the community. Mesa 3.1 requires Python 3.11 or higher.

## What's Changed
### 🧪 Experimental features
* Add support for Observables to MESA by @quaquel in https://github.com/mesa/mesa/pull/2291
* Add full support for property layers to cell spaces by @quaquel in https://github.com/mesa/mesa/pull/2512
### 🎉 New features added
* Add logging to MESA by @quaquel in https://github.com/mesa/mesa/pull/2506
* Add `create_agents` factory method to Agent by @quaquel in https://github.com/mesa/mesa/pull/2351
### 🔍 Examples updated
* Add seed control to all examples by @quaquel in https://github.com/mesa/mesa/pull/2496
### 📜 Documentation improvements
* doc fix for pip install error on mac by @quaquel in https://github.com/mesa/mesa/pull/2508
* Refactored docs for Introductory Tutorial by @Spartan-71 in https://github.com/mesa/mesa/pull/2511
* Add module-level docstring to experimental features by @EwoutH in https://github.com/mesa/mesa/pull/2532
### 🔧 Maintenance
* Remove deprecated time module by @EwoutH in https://github.com/mesa/mesa/pull/2476
* Drop support for Python 3.10, require Python >= 3.11 by @EwoutH in https://github.com/mesa/mesa/pull/2474
* Remove deprecated functionality by @EwoutH in https://github.com/mesa/mesa/pull/2483
* Remove visualization modules from `mesa.experimental` by @quaquel in https://github.com/mesa/mesa/pull/2495
* Cleanup two occurrences of removed scheduler by @EwoutH in https://github.com/mesa/mesa/pull/2499
* move _setup_agent_registration into `Model.__init__` by @quaquel in https://github.com/mesa/mesa/pull/2501
* remove devs related examples from devs/examples by @quaquel in https://github.com/mesa/mesa/pull/2507
* added empty iterable checks and updated tests by @Sahil-Chhoker in https://github.com/mesa/mesa/pull/2523
* Fix: running Mesa in Docker with Schelling model by @AdamZh0u in https://github.com/mesa/mesa/pull/2524

## New Contributors
* @Spartan-71 made their first contribution in https://github.com/mesa/mesa/pull/2511
* @Sahil-Chhoker made their first contribution in https://github.com/mesa/mesa/pull/2523
* @AdamZh0u made their first contribution in https://github.com/mesa/mesa/pull/2524

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.0.3...v3.1.0

# 3.0.3 (2024-11-14)
## Highlights
A small bugfix release that fixes two bugs.

## What's Changed
### 🧪 Experimental features
* cell_space: Allow CellCollection to be empty by @EwoutH in https://github.com/mesa/mesa/pull/2502
### 🐛 Bugs fixed
* Only set model_parameters once by @Corvince in https://github.com/mesa/mesa/pull/2505

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.0.2...v3.0.3

# 3.0.2 (2024-11-11)
## Highlights
Mesa 3.0.2 is a small follow-up patch release, in which we fixed a lot of small bugs in the example models their visualisation, and improved their testing.

## What's Changed
### 🐛 Bugs fixed
* allow components as a positional argument again by @Corvince in https://github.com/mesa/mesa/pull/2488
### 🔍 Examples updated
* examples: Add required components keyword by @EwoutH in https://github.com/mesa/mesa/pull/2485
* examples: Fix boid_flockers viz by @EwoutH in https://github.com/mesa/mesa/pull/2492
* examples: Fix schelling viz by @EwoutH in https://github.com/mesa/mesa/pull/2490
* example: Add input sliders to Sugarscape viz by @EwoutH in https://github.com/mesa/mesa/pull/2487
* examples/gol: Add initial fraction alive, add sliders to viz by @EwoutH in https://github.com/mesa/mesa/pull/2489
### 🔧 Maintenance
* test app init of examples by @Corvince in https://github.com/mesa/mesa/pull/2491

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.0.1...v3.0.2

# 3.0.1 (2024-11-11)
## Highlights
After our huge [3.0.0 release](https://github.com/mesa/mesa/releases/tag/v3.0.0), Mesa 3.0.1 follows up with two improvements to experimental features, examples and docs.

## What's Changed
### 🧪 Experimental features
* Bugfixes to DEVS by @quaquel in https://github.com/mesa/mesa/pull/2478
* Support simulators in SolaraViz by @quaquel in https://github.com/mesa/mesa/pull/2470
### 🛠 Enhancements made
* issue a user warning if random is None by @quaquel in https://github.com/mesa/mesa/pull/2479
### 🔍 Examples updated
* Integrate benchmarks and example models by @EwoutH in https://github.com/mesa/mesa/pull/2473
### 📜 Documentation improvements
* docs/tutorial: Replace scheduler in MoneyModel by @EwoutH in https://github.com/mesa/mesa/pull/2475
* docs: update migration_guide.md by @eltociear in https://github.com/mesa/mesa/pull/2480
* Update some DeprecationWarnings to note they are removed in Mesa 3.1 by @EwoutH in https://github.com/mesa/mesa/pull/2481

## New Contributors
* @eltociear made their first contribution in https://github.com/mesa/mesa/pull/2480

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.0.0...v3.0.1

# 3.0.0 (2024-11-09)
## Highlights
Mesa 3.0 brings major improvements to agent-based modeling, making it more intuitive and powerful while reducing complexity. This release modernizes core functionalities and introduces new capabilities for both beginners and advanced users.

### Streamlined agent management
The centerpiece of Mesa 3.0 is its new agent management system. Agents are now automatically tracked and assigned unique IDs, eliminating common boilerplate code. The new AgentSet functionality provides an elegant and flexible way to work with agents, for example:

```python
# Find agents meeting specific criteria
wealthy_agents = model.agents.select(lambda a: a.wealth > 1000)

# Group and analyze agents
grouped = model.agents.groupby("state")
state_stats = grouped.agg({
     "count": len,
     "avg_age": ("age", np.mean),
     "total_wealth": ("wealth", sum)
 })

# Activate agents in different patterns
model.agents.shuffle_do("step")  # Random activation
model.agents.select(lambda a: a.energy > 0).do("move")  # Conditional activation
```

The AgentSet provides powerful methods for filtering, grouping, and analyzing agents, making it easier to express complex model logic. Each model automatically maintains an AgentSet containing all agents (`model.agents`) and separate AgentSets for each agent type (`model.agents_by_type`). See the full [AgentSet docs](https://mesa.readthedocs.io/latest/apis/agent.html#mesa.agent.AgentSet) here.

### Modern Visualization with SolaraViz
Mesa 3.0's new experimental visualization system, SolaraViz, provides a modern, interactive interface for model exploration:

```python
from mesa.visualization import SolaraViz, make_space_component, make_plot_component

visualization = SolaraViz(
    model,
    [
        make_space_component(agent_portrayal),
        make_plot_component(["population", "average_wealth"]),
        lambda m: f"Step {m.steps}: {len(m.agents)} agents"  # Custom text component
    ],
    model_params=parameter_controls
)
```

Key visualization features:
- Interactive browser-based interface with real-time updates
- Support for both grid-based and network models
- Visualization of PropertyLayers and hexagonal grids
- Custom components using Matplotlib or text
- Improved performance and responsiveness

Check out the [Visualization Tutorial](https://mesa.readthedocs.io/latest/tutorials/visualization_tutorial.html) to get started.

*Note: SolaraViz is in active development. We might make API breaking changes between Mesa 3.0 and 3.1.*

### Enhanced data collection
The DataCollector now supports collecting different metrics for different agent types, using  `agenttype_reporters`:

```python
self.datacollector = DataCollector(
    model_reporters={"total_wealth": lambda m: m.agents.agg("wealth", sum)},
    agent_reporters={"age": "age", "wealth": "wealth"},
    agenttype_reporters={
        Predator: {"kills": "kills_count"},
        Prey: {"distance_fled": "total_flight_distance"}
    }
)
```

### Experimental features
Mesa 3.0 introduces several experimental features for advanced modeling:
- [Cell Space](https://mesa.readthedocs.io/latest/apis/experimental.html#module-experimental.cell_space.cell) with integrated PropertyLayers and improved agent movement capabilities
- Voronoi grid implementation
- [Event-scheduling simulation](https://mesa.readthedocs.io/latest/apis/experimental.html#module-experimental.devs.eventlist) capabilities

These experimental features are in active development and might break API between releases.

## Breaking changes
_See our [Mesa 3.0 migration guide](https://mesa.readthedocs.io/latest/migration_guide.html#mesa-3-0) for a full overview._

If you want to move existing models from Mesa 2.x to 3.0, there are a few things you have to change.

1. Models must explicitly initialize the Mesa base class:
```python
class MyModel(mesa.Model):
    def __init__(self, n_agents, seed=None):
        super().__init__(seed=seed)  # Required in Mesa 3.0
```

2. Agents are created without manual ID assignment:
```python
# Old
agent = MyAgent(unique_id=1, model=self)
# New
agent = MyAgent(model=self)
```

3. Scheduler replacement with AgentSet operations:
```python
# Old (RandomActivation)
self.schedule = RandomActivation(self)
self.schedule.step()

# New
self.agents.shuffle_do("step")

# Old (SimultaneousActivation)
self.schedule = SimultaneousActivation(self)
self.schedule.step()

# New
self.agents.do("step")
self.agents.do("advance")
```

Furthermore:
- Steps counter automatically increments
- `mesa.flat` namespace removed
- Python 3.10+ required
- Reserved model variables (`agents`, `steps`, etc.) protected
- Simplified DataCollector initialization
- Old visualization system replaced by SolaraViz

## Getting Started
Install Mesa 3.0:
```bash
pip install --upgrade mesa
```

If building a new model, we recommend checking out the updated [Mesa Overview](https://mesa.readthedocs.io/latest/overview.html) and [Introductory Tutorial](https://mesa.readthedocs.io/latest/tutorials/0_first_model.html).

For updating existing models, we recommend upgrading in steps:
1. Update to latest Mesa 2.x
2. Address deprecation warnings
3. Upgrade to Mesa 3.0
4. Replace schedulers with AgentSet functionality

A detailed [migration guide](https://mesa.readthedocs.io/latest/migration_guide.html#mesa-3-0) is available to help moving to Mesa 3.0. For questions or support, join our [GitHub Discussions](https://github.com/mesa/mesa/discussions) or [Matrix Chat](https://matrix.to/#/#project-mesa:matrix.org).

We would love to hear what you think about Mesa 3.0! [Say hello here](https://github.com/mesa/mesa/discussions/2465) and leave any [feedback on 3.0 here](https://github.com/mesa/mesa/discussions/2338).

# 3.0.0rc0 (2024-11-06)
## Highlights
We're releasing the Mesa 3.0 Release Candidate, ready for final testing before we release Mesa 3.0 later this week!

In this last 3.0 pre-release, the visualisation has been thoroughly updated, with a brand new API. Visualizing the experimental Cell Space, including PropertyLayers and hexagonal grids, is now also supported.

We're still working very active on the visualisation, so we have marked that experimental for Mesa 3.0. We will stabilize SolaraViz in Mesa 3.1.

Any feedback and last-minute bug reports are welcome [here](https://github.com/mesa/mesa/discussions/2338).

## What's Changed
### ⚠️ Breaking changes
* Viz: Refactor Matplotlib plotting by @quaquel in https://github.com/mesa/mesa/pull/2430
* api reorganization by @quaquel in https://github.com/mesa/mesa/pull/2447
### 🧪 Experimental features
* Mark SolaraViz as experimental for Mesa 3.0 by @EwoutH in https://github.com/mesa/mesa/pull/2459
### 🛠 Enhancements made
* expand ax.scatter kwargs that can be used by @quaquel in https://github.com/mesa/mesa/pull/2445
### 🐛 Bugs fixed
* Fix #2452 - handle solara viz model params better by @Corvince in https://github.com/mesa/mesa/pull/2454
* Update MoneyModel.py by @quaquel in https://github.com/mesa/mesa/pull/2458
### 🔍 Examples updated
* Updates to Epstein example by @quaquel in https://github.com/mesa/mesa/pull/2429
* Update examples to use updated space drawing by @quaquel in https://github.com/mesa/mesa/pull/2442
### 📜 Documentation improvements
* Update wolf-sheep png and fix typo in file name by @quaquel in https://github.com/mesa/mesa/pull/2444
* Include main examples readme in docs by @quaquel in https://github.com/mesa/mesa/pull/2448
* remove how-to guide and update docs in places by @quaquel in https://github.com/mesa/mesa/pull/2449
### 🔧 Maintenance
* remove deprecated HexGrid class by @quaquel in https://github.com/mesa/mesa/pull/2441
* rename make_plot_measure to make_plot_component and add some kwargs by @quaquel in https://github.com/mesa/mesa/pull/2446

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.0.0b2...v3.0.0rc0

# 3.0.0b2 (2024-10-26)
## Highlights
Mesa 3.0 beta 2 includes major work on the example models, docs, a new tutorial and visualisation.

The included example models are now part of the Mesa package itself and directly importable, using:
```Python
from mesa.examples import BoidFlockers, BoltzmannWealthModel, ConwaysGameOfLife, ...
```
The advanced examples were also restructured and cleaned up.

The tutorial was completely rewritten for Mesa 3.0, including it's latest features and practices. Many of our other docs also received some love, and almost everything is now ready for Mesa 3.0.

A new feature to remove all agents from the model was added, and the visualisation now supports drawing the experimental discrete spaces in both matplotlib and altair. All agents which are in a space can now conveniently be accessed with `.agents`.

The rarely used `mesa startproject` cookiecutter feature was removed. We updated our best-practice guide to include how to structure a modern Mesa project, which is now very straightforward.

## What's Changed
### ⚠️ Breaking changes
* remove cookiecutter by @quaquel in https://github.com/mesa/mesa/pull/2421
### 🧪 Experimental features
* Add support for drawing discrete grids by @quaquel in https://github.com/mesa/mesa/pull/2386
* Altair spaces by @quaquel in https://github.com/mesa/mesa/pull/2397
### 🎉 New features added
* remove_all_agents method added to model by @quaquel in https://github.com/mesa/mesa/pull/2394
* Pass through model.rgn in agent analogous to model.random by @quaquel in https://github.com/mesa/mesa/pull/2400
* add agents property to all spaces by @quaquel in https://github.com/mesa/mesa/pull/2418
### 🛠 Enhancements made
* update_tutorial environment by @tpike3 in https://github.com/mesa/mesa/pull/2411
### 🐛 Bugs fixed
* Fix for mistaken removal of _draw_grid by @quaquel in https://github.com/mesa/mesa/pull/2398
* fixes weakref bug in shuffle_do by @quaquel in https://github.com/mesa/mesa/pull/2399
### 🔍 Examples updated
* refactor: Simplify Schelling code by @rht in https://github.com/mesa/mesa/pull/2353
* Move examples into mesa by @Corvince in https://github.com/mesa/mesa/pull/2387
* Explicitly test basic examples by @quaquel in https://github.com/mesa/mesa/pull/2390
* Make example import absolute by @quaquel in https://github.com/mesa/mesa/pull/2402
* Cleanup and restructure EpsteinCivilViolence and PdGrid examples by @EwoutH in https://github.com/mesa/mesa/pull/2408
* Reorganize advanced examples: wolf_sheep and sugarscape_g1mt by @quaquel in https://github.com/mesa/mesa/pull/2410
* reactivate ruff for advanced examples and include them in tests by @quaquel in https://github.com/mesa/mesa/pull/2414
### 📜 Documentation improvements
* Include examples in readthedocs (port) by @EwoutH in https://github.com/mesa/mesa/pull/2392
* Update into_tutorial by @tpike3 in https://github.com/mesa/mesa/pull/2372
* Update Schelling Readme.md by @quaquel in https://github.com/mesa/mesa/pull/2406
* Update Conway example by @quaquel in https://github.com/mesa/mesa/pull/2403
* Boltzmann readme by @quaquel in https://github.com/mesa/mesa/pull/2405
* Update Readme.md of Boid flockers by @quaquel in https://github.com/mesa/mesa/pull/2404
* add advanced examples to rtd by @quaquel in https://github.com/mesa/mesa/pull/2413
* Tutorial Improvements by @tpike3 in https://github.com/mesa/mesa/pull/2415
* space: Add note that Grids are maintenance only by @EwoutH in https://github.com/mesa/mesa/pull/2420
* Migration guide: Update automatic unique_id assignment examples by @EwoutH in https://github.com/mesa/mesa/pull/2419
* Update docstring of SimEvent by @quaquel in https://github.com/mesa/mesa/pull/2417
* best-practices: Update Model Layout section by @EwoutH in https://github.com/mesa/mesa/pull/2424
* docs: Clean-up index.md by @EwoutH in https://github.com/mesa/mesa/pull/2422
### 🔧 Maintenance
* Add empty `pull_request_template.md` to enable PR template chooser by @EwoutH in https://github.com/mesa/mesa/pull/2409

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.0.0b1...v3.0.0b2

# 3.0.0b1 (2024-10-17)
## Highlights
Mesa 3.0 beta 1 is our last beta release before the Mesa 3.0 stable release. We are restructuring our examples and have move 9 core examples from [mesa-examples](https://github.com/mesa/mesa-examples) to mesa itself ([#2358](https://github.com/mesa/mesa/pull/2358)). The 5 basic examples are now directly importable ([#2381](https://github.com/mesa/mesa/pull/2381)):
```Python
from examples.basic import BoidFlockers, BoltzmannWealthModel, ConwaysGameOfLife, Schelling, VirusOnNetwork
```
The 5 basic examples will always use stable Mesa features, we are also working on 4 more advanced example which can also include experimental features.

All our core examples can now be viewed in the [`examples`](https://github.com/mesa/mesa/tree/main/examples) folder. [mesa-examples](https://github.com/mesa/mesa-examples) will continue to exists for user showcases. We're also working on making the examples visible in the ReadtheDocs ([#2382](https://github.com/mesa/mesa/pull/2382)) and on an website ([mesa-examples#139](https://github.com/mesa/mesa-examples/issues/139)). Follow all our work on the examples in this tracking issue [#2364](https://github.com/mesa/mesa/issues/2364).

Furthermore, the visualizations are improved by making visualization elements scalable and more clearly labeling the plots, and the Model now has an `rng` argument for an [SPEC 7](https://scientific-python.org/specs/spec-0007/) compliant NumPy random number generator ([#2352](https://github.com/mesa/mesa/pull/2352)). Following SPEC 7, you have to pass either `seed` or `rng`. Whichever one you pass will be used to seed both `random.Random`, and `numpy.random.Generator.`

## What's Changed
### ⚠️ Breaking changes
* replace model with random in AgentSet init by @quaquel in https://github.com/mesa/mesa/pull/2350
### 🧪 Experimental features
* cell space: Add convenience properties for grid width and height by @quaquel in https://github.com/mesa/mesa/pull/2348
* Bugfix for deepcopy / pickling discrete spaces by @quaquel in https://github.com/mesa/mesa/pull/2378
### 🎉 New features added
* Move core example models back (v2) by @EwoutH in https://github.com/mesa/mesa/pull/2358
* Add Model.rng for SPEC-7 compliant numpy random number generation by @quaquel in https://github.com/mesa/mesa/pull/2352
### 🛠 Enhancements made
* use GridDraggable instead of Column in SolaraViz by @wang-boyu in https://github.com/mesa/mesa/pull/2344
* update legend, xlabel & format of matplotlib plots by @wang-boyu in https://github.com/mesa/mesa/pull/2346
* __init__.py: Import mesa.experimental by @EwoutH in https://github.com/mesa/mesa/pull/2374
* Importable examples by @Corvince in https://github.com/mesa/mesa/pull/2381
### 🐛 Bugs fixed
* experimental init: Fix Solara import by making it lazy by @EwoutH in https://github.com/mesa/mesa/pull/2357
* fix: pass `model.random` to schedulers by @quaquel in https://github.com/mesa/mesa/pull/2359
* fix: register agent after creating unique_id and pos attributes by @wang-boyu in https://github.com/mesa/mesa/pull/2368
* solara: viz tutorial: fix histogram code by @Corvince in https://github.com/mesa/mesa/pull/2379
### 🔍 Examples updated
* Cleanup and restructure basic example models by @EwoutH in https://github.com/mesa/mesa/pull/2365
* Ruff basic examples by @EwoutH in https://github.com/mesa/mesa/pull/2370
### 📜 Documentation improvements
* Update migration_guide.md by @quaquel in https://github.com/mesa/mesa/pull/2347
### 🔧 Maintenance
* Code coverage: ignore experimental and visualization by @Corvince in https://github.com/mesa/mesa/pull/2361
* add codecov token, fixes #2363 by @Corvince in https://github.com/mesa/mesa/pull/2366
* add test_time back by @quaquel in https://github.com/mesa/mesa/pull/2367
* Release notes: Add example category by @EwoutH in https://github.com/mesa/mesa/pull/2369

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.0.0b0...v3.0.0b1

# 3.0.0b0 (2024-10-04)
## Highlights
We're proud to release the first Mesa 3.0 beta! This pre-release announces that we're ready for Mesa 3.0 to be tested by all our regular users. We try to not making breaking changes anymore, but focus on resolving bugs and imperfections.

In this beta, not so much has changed as in the alphas (we're stabilizing, that's a good sign), but there are still a few notable things:
- Agents now have to be initialized without their `unique_id`. See [#2328](https://github.com/mesa/mesa/pull/2328) and the [Migration guide](https://mesa.readthedocs.io/latest/migration_guide.html#automatic-assignment-of-unique-id-to-agents).
- PropertyLayers can now be visualized! See [#2336](https://github.com/mesa/mesa/pull/2336) for details and some examples, and [mesa-examples#214](https://github.com/mesa/mesa-examples/pull/214) as a simple example model.
- We reduced the core dependencies of Mesa, so that's a lighter and simpler install. You can now use extras to install the dependencies, for example add `[viz]` to install all visualisation dependencies: `pip install -U --pre mesa[viz]`. See [#2265](https://github.com/mesa/mesa/pull/2265) for details.
- The [Mesa Overview](https://mesa.readthedocs.io/latest/overview.html) as fully updated for 3.0. We highly recommend reading though it!
- We made some more progress on the experimental Cell Space, adding movement and integrating the PropertyLayer. Among others, Agents have nu initial movement capabilities for grids. Development continues during the betas and

We plan to release one or two more beta's in the coming weeks, and tag a release candidate and Mesa 3.0 late October. In the [v3.0 milestone](https://github.com/mesa/mesa/milestone/43) are the critical items on our todo-list.

You can install this pre-release as usual with:

```bash
pip install --upgrade --pre mesa
```
We're very curious what you think, try it out and ask any questions or share any feedback [here](https://github.com/mesa/mesa/discussions/2338)!

## What's Changed
### ⚠️ Breaking changes
* update `Agent.__init__` to remove deprecation warning by @quaquel in https://github.com/mesa/mesa/pull/2328
### 🎉 New features added
* Visualize PropertyLayers by @EwoutH in https://github.com/mesa/mesa/pull/2336
### 🧪 Experimental features
* Encapsulate cell movement in properties by @quaquel in https://github.com/mesa/mesa/pull/2333
* experimental: Integrate PropertyLayers into cell space by @EwoutH in https://github.com/mesa/mesa/pull/2319
* Generalize CellAgent by @Corvince in https://github.com/mesa/mesa/pull/2292
### 🛠 Enhancements made
* Reduce core dependencies, split in optional dependencies by @EwoutH in https://github.com/mesa/mesa/pull/2265
### 🐛 Bugs fixed
* viz: stop running and disable buttons when model.running is False by @wang-boyu in https://github.com/mesa/mesa/pull/2332
### 📜 Documentation improvements
* docs: Update overview for Mesa 3.0 by @EwoutH in https://github.com/mesa/mesa/pull/2317
* Readthedocs: Add version switch and update URL by @EwoutH in https://github.com/mesa/mesa/pull/2324
### 🔧 Maintenance
* tests: Resolve warnings by removing scheduler and updating arguments by @EwoutH in https://github.com/mesa/mesa/pull/2329
* add super call to Model and remove self.schedule by @quaquel in https://github.com/mesa/mesa/pull/2334
### Other changes
* Deprecate `initialize_data_collector` by @EwoutH in https://github.com/mesa/mesa/pull/2327

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.0.0a5...v3.0.0b0

# 3.0.0a5 (2024-09-21)
## Highlights
Mesa v3.0 alpha 5 release contains many quality of life updates, a big new feature for the DataCollector and a major deprecation.

The entire `mesa.time` module, including all schedulers, has been deprecated ([#2306](https://github.com/mesa/mesa/pull/2306)). Users are encouraged to transition to AgentSet functionality for more flexible and explicit agent activation patterns. Check the [migration guide](https://mesa.readthedocs.io/latest/migration_guide.html#time-and-schedulers) on how to upgrade.

The DataCollector now supports collecting data from specific Agent subclasses using the new `agenttype_reporters` parameter ([#2300](https://github.com/mesa/mesa/pull/2300)). This allows collecting different metrics for different agent types. For example:

```python
self.datacollector = DataCollector(
    agenttype_reporters={
        Wolf: {"sheep_eaten": "sheep_eaten"},
        Sheep: {"wool": "wool_amount"}
    }
)
```

Furthermore, a new `shuffle_do()` method for AgentSets provides a faster way to perform `shuffle().do()` ([#2283](https://github.com/mesa/mesa/pull/2283)). The GroupBy class gained `count()` and `agg()` methods to count the number of agents in groups and aggregate variables of them ([#2290](https://github.com/mesa/mesa/pull/2290)).

In the experimental Cell Space, the `CellCollection.select` method was updated to use `at_most` instead of `n`, aligning with the AgentSet API ([#2307](https://github.com/mesa/mesa/pull/2307)). Additionally, the Cell class now features a dedicated `neighborhood` property for direct neighbors (default radius=1) and a `get_neighborhood` method for larger radii ([#2309](https://github.com/mesa/mesa/pull/2309)).

Finally, SolaraViz received updates improving its interface and performance ([#2299](https://github.com/mesa/mesa/pull/2299), [#2304](https://github.com/mesa/mesa/pull/2304)). Cell connections in grids and networks are now public and named for more intuitive agent movements ([#2296](https://github.com/mesa/mesa/pull/2296)). The Model class initialization process was simplified by moving random seed and random object creation to `__init__` ([#1940](https://github.com/mesa/mesa/pull/1940)). Documentation has been extensively updated, including enforcing Google docstrings ([#2294](https://github.com/mesa/mesa/pull/2294)) and reorganizing the API documentation ([#2298](https://github.com/mesa/mesa/pull/2298)) for better clarity and navigation.

While the Mesa 3.0 timeline is still being discussed, we're aiming at the first Mesa 3.0 beta in October followed by a stable release in November. Testing new features and sharing feedback is appreciated!

## What's Changed
### 🎉 New features added
* GroupBy: Add `count` and `agg` methods by @EwoutH in https://github.com/mesa/mesa/pull/2290
* datacollector: Allow collecting data from Agent (sub)classes by @EwoutH in https://github.com/mesa/mesa/pull/2300
* Add optimized shuffle_do() method to AgentSet by @EwoutH in https://github.com/mesa/mesa/pull/2283
### 🛠 Enhancements made
* Make cell connections public and named by @Corvince in https://github.com/mesa/mesa/pull/2296
* SolaraViz Updates by @Corvince in https://github.com/mesa/mesa/pull/2299
* Solara viz: use_task for non-threaded continuous play by @Corvince in https://github.com/mesa/mesa/pull/2304
### 🧪 Experimental features
* Update to CellCollection.select by @quaquel in https://github.com/mesa/mesa/pull/2307
* Have a dedicated neighborhood property and a get_neighborhood method on Cell by @quaquel in https://github.com/mesa/mesa/pull/2309
### 📜 Documentation improvements
* Enforce google docstrings by @quaquel in https://github.com/mesa/mesa/pull/2294
* Api docs by @quaquel in https://github.com/mesa/mesa/pull/2298
* update migration guide to describe solaraviz updates by @Corvince in https://github.com/mesa/mesa/pull/2297
* Migration Guide: Add Model initialization requirement and automatic Agent.unique_id assignment by @EwoutH in https://github.com/mesa/mesa/pull/2302
* Deprecate Time module and all its Schedulers by @EwoutH in https://github.com/mesa/mesa/pull/2306
* intro_tutorial: Don't initialize agents with an unique_id by @EwoutH in https://github.com/mesa/mesa/pull/2315
* Migration guide: Intro, upgrade strategy, model.agents, headers by @EwoutH in https://github.com/mesa/mesa/pull/2314
### 🔧 Maintenance
* make typing behavior of AgentSet.get explicit by @quaquel in https://github.com/mesa/mesa/pull/2293
* model: Move random seed and random to __init__ by @rht in https://github.com/mesa/mesa/pull/1940
* Remove schedulers from benchmark models. by @quaquel in https://github.com/mesa/mesa/pull/2308

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.0.0a4...v3.0.0a5

# 3.0.0a4 (2024-09-09)
## Highlights
Mesa 3.0.0a4 contains two major breaking changes:
1. The Agent's `unique_id` is now automatically assigned, so doesn't need to be passed to the Agent class anymore. In a subclassed custom Agent, like normally used, this now looks like this:
    ```diff
    class Wolf(Agent):
    -    def __init__(self, unique_id, model, energy=None):
    +    def __init__(self, model, energy=None):
            # When initializing the super class (Agent), passing unique_id isn't needed anymore
    -        super().__init__(unique_id, model)
    +        super().__init__(model)

    - wolf = Wolf(unique_id, model)
    + wolf = Wolf(model)
    ```
    Example models were updated in [mesa-examples#194](https://github.com/mesa/mesa-examples/pull/194), which shows more examples on how to update existing models.

2. Our visualisation API is being overhauled, to be more flexible and powerful. For more details, see [#2278](https://github.com/mesa/mesa/pull/2278).
    - An initial update to the tutorial was made in [#2289](https://github.com/mesa/mesa/pull/2289) and is [available here](https://mesa.readthedocs.io/latest/tutorials/visualization_tutorial.html).
    - An initial example model was updated in [mesa-examples#195](https://github.com/mesa/mesa-examples/pull/195), and more examples will be updated in [mesa-examples#195](https://github.com/mesa/mesa-examples/pull/193).
    - The old SolaraViz API is still available at `mesa.experimental`, but might be removed in future releases.

Furthermore, the AgentSet has a new `agg` method to quickly get an aggregate value (for example `min_energy = model.agents.agg("energy", min)`) ([#2266](https://github.com/mesa/mesa/pull/2266)), The Model `get_agents_of_type` function is replaced by directly exposing the `agents_by_type` property (which can be accessed as a dict) ([#2267](https://github.com/mesa/mesa/pull/2267), [mesa-examples#190](https://github.com/mesa/mesa-examples/pull/190)) and the AgentSet get() methods can now handle missing values by replacing it with a default value ([#2279](https://github.com/mesa/mesa/pull/2279)).

Finally, it fixes a bug in which the Grid's `move_agent_to_one_of` method with `selection="closest"` selected a location deterministically, instead of randomly ([#2118](https://github.com/mesa/mesa/pull/2118)).

## What's Changed
### ⚠️ Breaking changes
* move solara_viz back to experimental by @Corvince in https://github.com/mesa/mesa/pull/2278
* track unique_id automatically by @quaquel in https://github.com/mesa/mesa/pull/2260
### 🎉 New features added
* AgentSet: Add `agg` method by @EwoutH in https://github.com/mesa/mesa/pull/2266
* Implement new SolaraViz API by @Corvince in https://github.com/mesa/mesa/pull/2263
### 🛠 Enhancements made
* Model: Replace `get_agents_of_type` method with `agents_by_type` property by @EwoutH in https://github.com/mesa/mesa/pull/2267
* add default SolaraViz by @Corvince in https://github.com/mesa/mesa/pull/2280
* Simplify ModelController by @Corvince in https://github.com/mesa/mesa/pull/2282
* Add default values and missing value handling to `agentset.get` by @quaquel in https://github.com/mesa/mesa/pull/2279
### 🐛 Bugs fixed
* Fix deterministic behavior in `move_agent_to_one_of` with `selection="closest"` by @OrenBochman in https://github.com/mesa/mesa/pull/2118
### 📜 Documentation improvements
* docs: Fix Visualization Tutorial (main branch) by @EwoutH in https://github.com/mesa/mesa/pull/2271
* Docs: Fix broken relative links by removing `.html` suffix by @EwoutH in https://github.com/mesa/mesa/pull/2274
* Readthedocs: Don't let notebook failures pass silently by @EwoutH in https://github.com/mesa/mesa/pull/2276
* Update viz tutorial to the new API by @Corvince in https://github.com/mesa/mesa/pull/2289
### 🔧 Maintenance
* Resolve multiprocessing warning, state Python 3.13 support by @rht in https://github.com/mesa/mesa/pull/2246

## New Contributors
* @OrenBochman made their first contribution in https://github.com/mesa/mesa/pull/2118

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.0.0a3...v3.0.0a4

# 3.0.0a3 (2024-08-30)
## Highlights
Developments toward Mesa 3.0 are steaming ahead, and our fourth alpha release is packed with features and updates - only 8 days after our third.

Mesa 3.0.0a3 contains one breaking change: We now automatically increase the `steps` counter by one at the beginning of each `Model.steps()` call. That means increasing `steps` by hand isn't necessary anymore.

The big new features is the experimental Voronoi grid that @vitorfrois implemented in #2084. It allows creating cells in a [Voronoi](https://en.wikipedia.org/wiki/Voronoi_diagram) layout as part of the experimental cell space. An example using it to model Cholera spread can be [found here](https://github.com/mesa/mesa-examples/pull/118).

The AgentSet got a lot of love with two brand new methods: `.groupby()` to split in groups (#2220) and `.set()` to easily assign variables to all agents in that set (#2254). The `select()` method is improved by allowing to select at most a fraction of the agents (#2253), and we split the `do()` method in `do()` and `map()` to make a distinction between the return types (#2237).

Furthermore, we improved the performance of accessing `Model.agents`, squashed a bug in SolaraViz, started testing on Python 3.13 and added a new benchmark model.

Our example models also got more love: We removed the `RandomActivation` scheduler in 14 models and removed SimultaneousActivation in 3 models ([examples#183](https://github.com/mesa/mesa-examples/pull/183)). They now use the automatic step increase and AgentSet functionality. We started testing our GIS model in CI ([examples#171](https://github.com/mesa/mesa-examples/pull/171)) and resolved a lot of bugs in them ([examples#172](https://github.com/mesa/mesa-examples/issues/172), help appreciated!).

Finally, we have two brand new examples: An Ant Colony Optimization model using an Ant System approach to the Traveling Salesman problem, a Mesa NetworkGrid, and a custom visualisation with SolaraViz ([examples#157](https://github.com/mesa/mesa-examples/pull/157) by @zjost). The first example using the `PropertyLayer` was added, a very fast implementation of Conway's Game of Life ([examples#182](https://github.com/mesa/mesa-examples/pull/182)).

To help the transition to Mesa 3.0, we started writing a [migration guide](https://mesa.readthedocs.io/latest/migration_guide.html). Progress is tracked in #2233, feedback and help is appreciated! Finally, we also added a new section to our [contributor guide](https://github.com/mesa/mesa/blob/main/CONTRIBUTING.md#i-have-no-idea-where-to-start) to get new contributors up to speed.

This pre-release can be installed as always with `pip install --pre mesa`

## What's Changed
### ⚠️ Breaking changes
* model: Automatically increase `steps` counter by @EwoutH in https://github.com/mesa/mesa/pull/2223
### 🧪 Experimental features
* Voronoi Tessellation based Discrete Space by @vitorfrois in https://github.com/mesa/mesa/pull/2084
### 🎉 New features added
* Add AgentSet.groupby by @quaquel in https://github.com/mesa/mesa/pull/2220
* AgentSet: Add `set` method by @EwoutH in https://github.com/mesa/mesa/pull/2254
### 🛠 Enhancements made
* Split AgentSet into map and do to separate return types by @quaquel in https://github.com/mesa/mesa/pull/2237
* Performance enhancements for Model.agents by @quaquel in https://github.com/mesa/mesa/pull/2251
* AgentSet: Allow selecting a fraction of agents in the AgentSet by @EwoutH in https://github.com/mesa/mesa/pull/2253
### 🐛 Bugs fixed
* SolaraViz: Reset components when params are changed by @rht in https://github.com/mesa/mesa/pull/2240
### 📜 Documentation improvements
* Contribution: Add "I have no idea where to start" section by @EwoutH in https://github.com/mesa/mesa/pull/2258
* Write initial Mesa Migration guide by @EwoutH in https://github.com/mesa/mesa/pull/2257
### 🔧 Maintenance
* CI: Add test job for Python 3.13 by @EwoutH in https://github.com/mesa/mesa/pull/2173
* Add pull request templates by @EwoutH in https://github.com/mesa/mesa/pull/2217
* benchmarks: Add BoltzmannWealth model by @EwoutH in https://github.com/mesa/mesa/pull/2252
* CI: Add optional dependency for examples by @EwoutH in https://github.com/mesa/mesa/pull/2261

## New Contributors
* @vitorfrois made their first contribution in https://github.com/mesa/mesa/pull/2084

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.0.0a2...v3.0.0a3

# 3.0.0a2 (2024-08-21)
## Highlights
In Mesa 3.0 alpha 2 (`v3.0.0a2`) we've done more clean-up in preparation for Mesa 3.0. We now [require](https://github.com/mesa/mesa/pull/2218) `super().__init__()`  to run on initializing a Mesa model subclass, `Model.agents` is now fully reserved for the Model's internal AgentSet and we fixed a bug in our Solara space_drawer.

A new feature was added in [#2219](https://github.com/mesa/mesa/pull/2219), which now also allows `AgentSet.do()` to take any callable function, instead of only a string referencing to an Agent method. The argument name was changed from `method_name` to `method`.

The new Solara visualisation now allows portraying agents in different shapes, checkout some examples in [#2214](https://github.com/mesa/mesa/pull/2214).

We're also working hard on our [example models](https://github.com/mesa/mesa-examples). All model warnings were [resolved](https://github.com/mesa/mesa-examples/pull/153) and we've [replaced](https://github.com/mesa/mesa-examples/pull/161) a lot of schedulers with simpler and more flexible AgentSet functionality. Checkout our [open issues](https://github.com/mesa/mesa-examples/issues) if you want to help improve our example models further!

## What's Changed
### ⚠️ Breaking changes
* breaking: Add dependencies argument to custom space_drawer by @rht in https://github.com/mesa/mesa/pull/2209
* Require Mesa models to be initialized with `super().__init__()` by @EwoutH in https://github.com/mesa/mesa/pull/2218
* Allow AgentSet.do() to take Callable function by @quaquel in https://github.com/mesa/mesa/pull/2219
* Change warning when setting model.agents to error by @EwoutH in https://github.com/mesa/mesa/pull/2225
### 🧪 Experimental features
* devs/eventlist: Add repr method to print EventList pretty by @EwoutH in https://github.com/mesa/mesa/pull/2195
### 🛠 Enhancements made
* Visualisation: Allow specifying Agent shapes in agent_portrayal by @rmhopkins4 in https://github.com/mesa/mesa/pull/2214
### 📜 Documentation improvements
* docs/conf.py: Use modern `intersphinx_mapping` format by @EwoutH in https://github.com/mesa/mesa/pull/2206
* docs: Update Readme and tutorials to mention Mesa 3.0 pre-releases by @EwoutH in https://github.com/mesa/mesa/pull/2203
### 🔧 Maintenance
* CI: Let pytest treat warnings as errors for examples by @EwoutH in https://github.com/mesa/mesa/pull/2204

## New Contributors
* @rmhopkins4 made their first contribution in https://github.com/mesa/mesa/pull/2214

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.0.0a1...v3.0.0a2

# 3.0.0a1 (2024-08-01)
## Highlights
Mesa 3.0 alpha 1 (`v3.0.0a1`) is another step towards our next major version. This release introduces a name change from JupyterViz (jupyter_viz) to SolaraViz (solara_viz), to better represent the tech stack being used. It also includes two bugfixes also present in 2.3.2.

## What's Changed
### ⚠️ Breaking changes
* viz: Combine code for rendering in browser and Jupyter by @rht in https://github.com/mesa/mesa/pull/2180
### 🛠 Enhancements made
* Rename JupyterViz to SolaraViz by @rht in https://github.com/mesa/mesa/pull/2187
* refactor: Rename jupyter_viz namespace to solara_viz by @rht in https://github.com/mesa/mesa/pull/2188
### 🐛 Bugs fixed
* fix: Render agent marker radius correctly by @rht in https://github.com/mesa/mesa/pull/2181
* fix: Use model.schedule.steps -> mode._steps for batch_run by @rht in https://github.com/mesa/mesa/pull/2183
### 📜 Documentation improvements
* Add original conference paper link to docs by @ENUMERA8OR in https://github.com/mesa/mesa/pull/2160

## New Contributors
* @ENUMERA8OR made their first contribution in https://github.com/mesa/mesa/pull/2160

**Full Changelog**: https://github.com/mesa/mesa/compare/v3.0.0a0...v3.0.0a1

# 3.0.0a0 (2024-07-04)
## Highlights
This is the first pre-release in the Mesa 3.0 series, which is still in active development. The `v3.0.0a0` pre-release can help active Mesa developers help starting to test the latest features in their models.

Since it's in active development, more breaking changes may follow and it's not recommended for general usage.

There are two major breaking changes at this point:
- The old visualisation is removed, in favor of the new, Solara based, Jupyter Viz. This was already available in the 2.3.x release series, but is now stabilized. Checkout out our new [Visualization Tutorial](https://mesa.readthedocs.io/latest/tutorials/visualization_tutorial.html). More examples and a migration guide will follow later in the Mesa 3.0 development.
- The `mesa.flat` namespace is removed, since was not used very often.

Mesa 3.0 will require Python 3.10+.

This pre-release can be installed with `pip install mesa --upgrade --pre`.

## What's Changed
### ⚠️ Breaking changes
* Remove mesa.flat namespace by @rht in https://github.com/mesa/mesa/pull/2091
* breaking: Remove visualization_old (mesa-viz-tornado) by @rht in https://github.com/mesa/mesa/pull/2133
### 🎉 New features added
* Set JupyterViz as stable by @rht in https://github.com/mesa/mesa/pull/2090
### 🐛 Bugs fixed
* Jupyter_viz: Allow measures to be None by @EwoutH in https://github.com/mesa/mesa/pull/2163
* Jupyter Viz: Don't avoid interactive backend by @EwoutH in https://github.com/mesa/mesa/pull/2165
### 📜 Documentation improvements
* Fix image on landing page of docs. by @jackiekazil in https://github.com/mesa/mesa/pull/2146
* Replace links in docs - google group to matrix. by @jackiekazil in https://github.com/mesa/mesa/pull/2148
* Update visualisation docs by @EwoutH in https://github.com/mesa/mesa/pull/2162
### 🔧 Maintenance
* CI: Add weekly scheduled run to all CI workflows by @EwoutH in https://github.com/mesa/mesa/pull/2130
* Drop support for Python 3.9, require Python >= 3.10  by @EwoutH in https://github.com/mesa/mesa/pull/2132
* Add script to list unlabeled PR's since latest release by @rht in https://github.com/mesa/mesa/pull/2047

## New Contributors
* @stephenfmann made their first contribution in https://github.com/mesa/mesa/pull/2154

**Full Changelog**: https://github.com/mesa/mesa/compare/v2.3.1...v3.0.0a0

# 2.3.2 (2024-07-22)
## Highlights
Mesa 2.3.2 is a small patch release which fixes two bugs, one to the batch_run function still depending on `schedule.steps`, and one in the agent marker visualisation.

## What's Changed
### 🐛 Bugs fixed
* fix: Render agent marker radius correctly by @rht in https://github.com/mesa/mesa/pull/2181
* fix: Use model.schedule.steps -> mode._steps for batch_run by @rht in https://github.com/mesa/mesa/pull/2183

**Full Changelog**: https://github.com/mesa/mesa/compare/v2.3.1...v2.3.2

# 2.3.1 (2024-07-03)
## Highlights
Mesa 2.3.1 is a small patch release with a datacollector bug fixed and improved documentation.

## What's Changed
### 🐛 Bugs fixed
* datacollector: store separate snapshots of model data per step by @EwoutH in https://github.com/mesa/mesa/pull/2129
### 📜 Documentation improvements
* Add experimental features to documentation as per #2122 by @stephenfmann in https://github.com/mesa/mesa/pull/2154

**Full Changelog**: https://github.com/mesa/mesa/compare/v2.3.0...v2.3.1

# 2.3.0 (2024-04-23)
## Highlights
Mesa 2.3.0 is a big feature release and the last feature release before 3.0.

There are two main new features:
- The experimental cell-centric discrete spaces, as added in #1994. It allows having cells with not only properties but also active behaviors: the `CellAgent`. Its inspired by NetLogo's [patches](https://ccl.northwestern.edu/netlogo/bind/primitive/patches.html) but extend and generalize this concept further.
- Full support for discrete event scheduling, as added in #2066. It allows scheduling events (like Agent actions) at any time, including non-integer timesteps.

There are a lot of other features: The Jupyter visualisation now supports easier way to specify sliders, `NetworkGrid.get_neighbors()` supports a radius, `AgentSet.get()` can retrieve multiple attributes and there are now benchmarks to track Mesa performance during development.

Finally, 2.3.0 stabilizes the `AgentSet` (including `model.agents`), making it the first experimental Mesa feature that is taken out of it's experimental phase.

Install this release with:
```
pip install --upgrade mesa
```
The Mesa 2.3.x-series supports Python 3.9 to 3.12. The next major release will require Python 3.10.

## What's Changed
### 🧪 Experimental features
* Add cell-centric discrete spaces (experimental) by @Corvince in https://github.com/mesa/mesa/pull/1994
### 🎉 New features added
* Add performance benchmarking scripts by @EwoutH in https://github.com/mesa/mesa/pull/1979
* feat: Implement Slider class for JupyterViz by @rht in https://github.com/mesa/mesa/pull/1972
* Stabilize AgentSet by @EwoutH in https://github.com/mesa/mesa/pull/2065
* Support discrete event scheduling by @quaquel in https://github.com/mesa/mesa/pull/2066
### 🛠 Enhancements made
* JupyterViz: Automatically deduce display name from model class by @rht in https://github.com/mesa/mesa/pull/1975
* Add radius argument to NetworkGrid.get_neighbors() by @EwoutH in https://github.com/mesa/mesa/pull/1973
* Speedup of Agentset.shuffle by @quaquel in https://github.com/mesa/mesa/pull/2010
* feat: Let mesa runserver detect server.py as fallback by @rht in https://github.com/mesa/mesa/pull/2015
* JupyterViz: {Convert make_plot & prepare ColorCard} to become Solara component by @rht in https://github.com/mesa/mesa/pull/2020
* new feature: AgentSet.get can retrieve one or more then one attribute by @quaquel in https://github.com/mesa/mesa/pull/2044
* Update CODE_OF_CONDUCT.md to version 2+ of contrib covenant by @jackiekazil in https://github.com/mesa/mesa/pull/2052
* Improve flocking benchmark  by @coderbeta1 in https://github.com/mesa/mesa/pull/2054
* Remove JupyterViz Altair marker overlap for huge grid size by @rht in https://github.com/mesa/mesa/pull/2062
* Add tooltip option to Altair chart by @FoFFolo in https://github.com/mesa/mesa/pull/2082
* feat: Display model seed & allow user to specify it in JupyterViz by @rht in https://github.com/mesa/mesa/pull/2069
* warn if placing already placed agent by @puer-robustus in https://github.com/mesa/mesa/pull/2083
### 🐛 Bugs fixed
* fix: Apply default value to slider by @rht in https://github.com/mesa/mesa/pull/2016
* fix: Initialize model _steps and _time during __new__ by @rht in https://github.com/mesa/mesa/pull/2026
* fix: Use model.schedule only when it is not None by @rht in https://github.com/mesa/mesa/pull/2050
* fix: Remove JupyterViz grid marker overlap for huge grid size by @rht in https://github.com/mesa/mesa/pull/2049
### 📜 Documentation improvements
* Improve readability of badges by @rht in https://github.com/mesa/mesa/pull/2009
* More pythonic implementation of wolf sheep by @quaquel in https://github.com/mesa/mesa/pull/2011
* Adding super().__init__() to MoneyModel tutorial by @sw23 in https://github.com/mesa/mesa/pull/2025
* docs: Convert howto.rst -> howto.md via rst2myst by @rht in https://github.com/mesa/mesa/pull/2033
* docs: Convert best-practices,overview,packages,mesa,index to .md via rst2myst by @rht in https://github.com/mesa/mesa/pull/2034
* docs: Convert api/*.rst -> api/*.md via rst2myst by @rht in https://github.com/mesa/mesa/pull/2035
* docs: Rewrite howto.md using ChatGPT for clarity and conciseness by @rht in https://github.com/mesa/mesa/pull/2037
* docs: Corrected Contributing Guide Link to Ensure Accessibility by @sahusiddharth in https://github.com/mesa/mesa/pull/2057
* Rename links to internal .rst files to .md by @rht in https://github.com/mesa/mesa/pull/2058
* docs: improve introductory tutorial by @puer-robustus in https://github.com/mesa/mesa/pull/2087
### 🔧 Maintenance
* Quality of Life: Make codecov less meticulous by @Corvince in https://github.com/mesa/mesa/pull/1966
* Add CI workflow for performance benchmarks by @EwoutH in https://github.com/mesa/mesa/pull/1983
* tests: Resolve warnings by defining PropertyLayer dtypes by @EwoutH in https://github.com/mesa/mesa/pull/1987
* benchmarks.yml: Fix PR branch checkout when triggered by comment by @EwoutH in https://github.com/mesa/mesa/pull/1998
* Quality of life: automatically fix ruff errors by @Corvince in https://github.com/mesa/mesa/pull/2004
* benchmarks.yml: Run on addition of label instead of comment by @EwoutH in https://github.com/mesa/mesa/pull/2002
* ci: Move codespell to pre-commit by @rht in https://github.com/mesa/mesa/pull/2040
* Schelling by @coderbeta1 in https://github.com/mesa/mesa/pull/2053
* Move ruff lint settings into dedicated section by @Corvince in https://github.com/mesa/mesa/pull/2073
* ci: Use uv pip for faster build by @rht in https://github.com/mesa/mesa/pull/2038
* test: Remove place_agent duplicate warnings by @rht in https://github.com/mesa/mesa/pull/2086
### Other changes
* Minor edits to benchmarking code by @quaquel in https://github.com/mesa/mesa/pull/1985
* build(deps): bump codecov/codecov-action from 3 to 4 by @dependabot in https://github.com/mesa/mesa/pull/2030
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/mesa/mesa/pull/2029
* tests: Speed up test_batch_run by @rht in https://github.com/mesa/mesa/pull/2039
* Update benchmarks.yml by @Corvince in https://github.com/mesa/mesa/pull/2043
* docs: Convert visualization .rst -> .md via rst2myst by @rht in https://github.com/mesa/mesa/pull/2036
* docs: Convert CONTRIBUTING .rst -> .md via rst2myst by @rht in https://github.com/mesa/mesa/pull/2041
* Correct wolf energy gained from eating sheep by @JackAtOmenApps in https://github.com/mesa/mesa/pull/2048
* feat: Implement Altair version of grid visualization by @rht in https://github.com/mesa/mesa/pull/1991

## New Contributors
* @sw23 made their first contribution in https://github.com/mesa/mesa/pull/2025
* @JackAtOmenApps made their first contribution in https://github.com/mesa/mesa/pull/2048
* @coderbeta1 made their first contribution in https://github.com/mesa/mesa/pull/2054
* @sahusiddharth made their first contribution in https://github.com/mesa/mesa/pull/2057
* @FoFFolo made their first contribution in https://github.com/mesa/mesa/pull/2082
* @puer-robustus made their first contribution in https://github.com/mesa/mesa/pull/2083

**Full Changelog**: https://github.com/mesa/mesa/compare/v2.2.4...2.3.0

# 2.3.0-rc1 (2024-04-18)
Mesa 2.3.0-rc1 is pre-release in preparation for 2.3.0 stable. It had the same release notes as 2.3.0.

# 2.2.4 (2024-01-26)
## Highlights
Mesa v2.2.4 is a small but important bugfix release for the 2.2 release series. It fixes an essential bug in where agents weren't shuffled in the `BaseScheduler`, affecting mainly the `RandomActivation` scheduler (effectively making it sequential activation)([#2007](https://github.com/mesa/mesa/pull/2007)). It also fixes a small behaviour change in `RandomActivationByType.agents_by_type()` ([#1996](https://github.com/mesa/mesa/pull/1996)). Furthermore, this release adds an internal clock to the `Model`, which allows to use a Mesa model without a scheduler (using the `AgentSet` API)([#1942](https://github.com/mesa/mesa/pull/1942)).

Updating from previous 2.2 releases is highly recommended, especially when using the `RandomActivation` scheduler.

## What's Changed
### 🛠 Enhancements made
* refactor: Remove dependence on model.schedule, add clock to Model by @rht in https://github.com/mesa/mesa/pull/1942
### 🐛 Bugs fixed
* Fix AgentSet inplace shuffle (and thus RandomActivation), add tests by @EwoutH and @quaquel in https://github.com/mesa/mesa/pull/2007
* fix: Reverse dict key and value for agents_by_type by @rht in https://github.com/mesa/mesa/pull/1996

**Full Changelog**: https://github.com/mesa/mesa/compare/v2.2.3...v2.2.4

# 2.2.3 (2024-01-22)
## Highlights
Mesa 2.2.3 is a small release with two improvements to the experimental Solara visualisation, on request of one of our contributors. No stable features have changed.

## What's Changed
### 🧪 Experimental features
* solara_viz: Add borders around ContinuousSpace by @EwoutH in https://github.com/mesa/mesa/pull/1988
### 🐛 Bugs fixed
* fix: Explicitly specify JupyterViz space view limits by @rht in https://github.com/mesa/mesa/pull/1984

**Full Changelog**: https://github.com/mesa/mesa/compare/v2.2.2...v2.2.3

# 2.2.2 (2024-01-22)

## Highlights

Mesa 2.2.2 is a small bugfix release, for models in which users had defined  `Model.agents` (`self.agents` in a Model (sub)class). This is deprecated, but for now allowed. See [#1919 (comment)](https://github.com/mesa/mesa/discussions/1919#discussioncomment-8141844).

## What's Changed
### 🐛 Bugs fixed
* Allow user models to assign `Model.agents` for now, but add warning by @quaquel in [#1976](https://github.com/mesa/mesa/pull/1976)

**Full Changelog**: https://github.com/mesa/mesa/compare/v2.2.1...v2.2.2


# 2.2.1 (2024-01-16)

## Highlights
After the huge 2.2.0 release we are releasing 2.2.1 which addresses a few bugs, unintended behaviors and a performance regression. #1960 makes sure agent addition and removal is handled correct, #1965 fixed an unintended behavior change in `RandomActivationByType.agents_by_type` and #1964 makes sure we're at least as fast as before 2.2.0, if not faster. The [introduction tutorial](https://mesa.readthedocs.io/en/stable/tutorials/intro_tutorial.html) is also extended with #1955.

We highly recommend updating to 2.2.1 if you're using 2.2.0.

## What's Changed
### 🧪 Experimental features
* jupyter_viz: Implement multiline plot by @rht in https://github.com/mesa/mesa/pull/1941
### 🛠 Enhancements made
* make mesa runnable without some dependencies by @Corvince in https://github.com/mesa/mesa/pull/1950
* Improve performance of AgentSet and iter_cell_list_contents by @Corvince in https://github.com/mesa/mesa/pull/1964
### 🐛 Bugs fixed
* Bugfix in agentset to handle addition and removal correctly by @quaquel in https://github.com/mesa/mesa/pull/1960
* Make RandomActivationByType.agents_by_type backward compatible by @quaquel in https://github.com/mesa/mesa/pull/1965
### 📜 Documentation improvements
* Refer to just Python instead of Python 3 by @rht in https://github.com/mesa/mesa/pull/1957
* intro tutorial: Analysing model reporters by @EwoutH in https://github.com/mesa/mesa/pull/1955
### 🔧 Maintenance
* Migrate from setuptools to hatch by @rht in https://github.com/mesa/mesa/pull/1882
* ci: Add tests for mesa-examples by @rht in https://github.com/mesa/mesa/pull/1956
* refactor: Move Matplotlib-specific Solara components to separate file by @rht in https://github.com/mesa/mesa/pull/1943

**Full Changelog**: https://github.com/mesa/mesa/compare/v2.2.0...v2.2.1


# 2.2.0 (2024-01-09)

## Highlights
The 2.2.0 release of the Mesa library introduces several updates and new features for managing and scheduling agents and modelling the environment, along with an experimental release policy aimed at enhancing development speed and community feedback. Below are key highlights of the new (experimental) features in this release. Mesa 2.2.0 supports Python 3.9+.

Despite the minor version number, this is one of our biggest releases yet.

### Experimental feature policy (discussion #1909)
This release introduces an experimental feature policy aimed at accelerating development and gathering community feedback. Features like #1890, #1898, and #1916 are marked as experimental under this policy.

**Policy overview:**
- Experimental features can be added or changed in any release, even patch releases.
- They don’t need a diligent review for every change, allowing for quicker development cycles.
- Community feedback is encouraged through discussion threads.

### Native support for multiple Agent types (PR #1894)
This update introduces a `agents` variable to the Mesa `Model` class, offering a first step in supporting multiple agent types as first class citizens. Each `Model` is now initialized with an `self.agents` variable (an `AgentSet`) in which all the agents are tracked. You can now always ask which agents are in the model with `model.agents`. It's the foundation which will allow us to solve problems with scheduling, data collection and visualisation of multiple agent types in the future.

### 🧪 AgentSet class (PR #1916)
The new `AgentSet` class encapsulates and manages collections of agents, streamlining the process of selecting, sorting, and applying actions to groups of agents.

**Key features:**
- Flexible and efficient agent management and manipulation.
- Methods like `select`, `shuffle`, `sort`, and `do` for intuitive operations.

**Example:**
```Python
# Applying a method to each agent
model.agents.do('step')

# Filtering and shuffling agents
shuffled_agents = model.agents.select(lambda agent: agent.attribute > threshold).shuffle()
```
_The AgentSet is an experimental feature. We would love feedback on it in #1919._

### 🧪 PropertyLayer and _PropertyGrid (PR #1898)
The introduction of `PropertyLayer` and the extension of `SingleGrid` and `MultiGrid` classes to support cell properties mark a significant enhancement in Mesa's environmental modeling capabilities. It allows to add different layers of variables to grids, that can be used to represent spatial environmental properties, such as elevation, pollution, flood levels or foliage.

**Key features:**
- Efficient management of environmental properties like terrain types and resources.
- Dynamic interaction between agents and their environment.
- Fast modification and selection of cells based on one or multiple properties

**Example:**
```Python
from mesa.space import SingleGrid, PropertyLayer

grid = SingleGrid(10, 10, False)
property_layer = PropertyLayer("elevation", 10, 10, default_value=0)
grid.add_property_layer(property_layer)

# Modify multiple cells values
grid.properties["elevation"].modify_cells(np.multiply, 2)

# Select cells that have an elevation of at least 50
high_elevation_cells = grid.properties["elevation"].select_cells(condition=lambda x: x > 50)
```
_The PropertyLayer is an experimental feature. We would love feedback on it in #1932._

### 🧪 DiscreteEventScheduler (PR #1890)
The `DiscreteEventScheduler` is an innovative addition to the Mesa time module, tailored for discrete event simulations. This scheduler advances simulations based on specific event timings rather than regular intervals, providing more flexibility in modeling complex systems.

**Key Features:**
- Efficient handling of events scheduled for specific simulation times.
- Randomized execution order for events scheduled at the same time.

_The DiscreteEventScheduler is an experimental feature. We would love feedback on it in #1923._

## What's Changed

### 🧪 Experimental features
* Native support for multiple agent types by @EwoutH in https://github.com/mesa/mesa/pull/1894
* space: Implement PropertyLayer and _PropertyGrid by @EwoutH in https://github.com/mesa/mesa/pull/1898
* Add DiscreteEventScheduler by @EwoutH in https://github.com/mesa/mesa/pull/1890

### 🎉 New features added
* Introduce AgentSet class by @EwoutH in https://github.com/mesa/mesa/pull/1916

### 🛠 Enhancements made
* Reimplement schedulers to use AgentSet by @quaquel in https://github.com/mesa/mesa/pull/1926
* space: Let move_agent choose from multiple positions by @EwoutH in https://github.com/mesa/mesa/pull/1920

### 🐛 Bugs fixed
* Work around for initializing model._agent by @quaquel in https://github.com/mesa/mesa/pull/1928
* Honor disabled space drawer option when rendering in the browser by @rlskoeser in https://github.com/mesa/mesa/pull/1907

### 📜 Documentation improvements
* Document empties property by @EwoutH in https://github.com/mesa/mesa/pull/1888
* docs: Fix README.md inline code formatting by @rht in https://github.com/mesa/mesa/pull/1887

### 🔧 Maintenance
* ci: Speed up pip install by caching deps install output by @rht in https://github.com/mesa/mesa/pull/1885
* Create release.yml file for automatic release notes generation by @EwoutH in https://github.com/mesa/mesa/pull/1925
* Drop support for Python 3.8 by @rht in https://github.com/mesa/mesa/pull/1756

### New Contributors
* @quaquel made their first contribution in https://github.com/mesa/mesa/pull/1928

**Full Changelog**: https://github.com/mesa/mesa/compare/v2.1.5...v2.2.0


# 2.1.5 (2023-11-26)

This release has some critical fixes to JupyterViz/Solara frontend to prevent
flickering and improve the display of the jupyter plots. It also has
improvements to datacollection and the documentation.

**Improvements**

- datacollection: check if model reporters is a partial function (#1872)

**Docs and Tutorial**

- docs: convert README from `.rst` to `.md` (#1881)
- docs: convert HISTORY from `.rst` to `.md` (#1873)
- docs: enhance docstrings for scheduler classes in `mesa.time` (#1866)

**CI and WorkFlows**

- ci: Remove redundant Ruff workflow from GitHub Actions (#1880)
- Replace Black with ruff format (#1880)
- Migrate setup from `setup.py` to `pyproject.toml` (#1870)

**Solara/JupyterViz**

- fix: Do render_in_jupyter on Colab env (#1884)
- Convert make_space into Solara component (#1877)
- remove controls for dragging and resizing (#1878)
- Improve ColorCard layout (#1876)
- refactor: Define current_step as reactive (#1875)
- fix: optimize controller and plots to fill screen in jupyter (#1868)
- fix: ensure space and plot subcomponent are not rendered on step (#1867)

# 2.1.4 (2023-11-7)

This release updates mesa-viz-tornado dependency v0.1.3. This removes
the external JavaScript templates and
[prevents 404 errors](https://github.com/mesa/mesa-viz-tornado/issues/40)

-   bugfix: ensure mesa_viz_tornado\>=0.1.3 #1862

# 2.1.3 (2023-11-5)

This release contains several improvements, fixes, and new features to
the JupyterViz/Solara frontend. It\'s a patch release instead of a minor
release because the JupyterViz frontend is still considered
experimental.

**Improvements**

-   model: ensure model is initialized with random seed based #1814
-   space: check if position values are tuples #1831
-   datacollection: add agent collection by type, documentation, and
    tests #1838

**Docs and Tutorial**

-   tutorial: explain how to set up reporter for multiple agents #1717
-   docs: rename useful snippets to how to guide #1839

**CI and WorkFlows**

-   Release CI: update to run workflows on releases #1479
-   CI: Update GHA workflows with Python 3.12 #1840
-   update ruff version #1824, #1841
-   Ensure mesa_viz_tornado\>=0.1.2 #1860

**Solara/JupyterViz**

-   perf: increase speed of Solara render #1819
-   implement drawer for continuous space and refactor code #1830
-   fix: configure change handler for checkbox input #1844
-   fix: ensure playing starts after model param change #1851

# 2.1.2 (2023-09-23)

This release contains fixes, and several improvements and new features
to the JupyterViz/Solara frontend. It\'s a patch release instead of a
minor release because the JupyterViz frontend is still considered
experimental.

**Improvements**

-   perf: Access grid only once #1751
-   docs: compile notebooks at build time #1753
-   docs: Remove nbsphinx and explicit .ipynb suffix #1754
-   rtd: Use gruvbox-dark as style #1719
-   build(deps): bump actions/checkout from 3 to 4 #1790

**Solara/JupyterViz**

-   solara: Implement visualization for network grid #1767
-   Add support for select input type #1779
-   Add step count display to JupyterViz #1775
-   Simplify solara code #1786
-   Add docstring for jupyterviz make_user_input that documents
    supported inputs #1784
-   Revise, test, & document JupyterViz options for drawing agent space
    #1783
-   Add UserInputs component #1788
-   Fix: Remove dict merge operator, python 3.8 compat #1793
-   feat: Add reset button to JupyterViz #1795
-   Add support for solara.Checkbox user input #1798
-   viz tutorial: Update custom plot to reflect new code #1799
-   fix: Don\'t continue playing when a model is reset #1796
-   Docker: Update to use Solara viz #1757

**Refactors**

-   Move viz stuff to mesa-viz-tornado Git repo #1746
-   simplify get neighborhood #1760
-   remove attrgetter performance optimization #1809

**Fixes**

-   fix: Add Matplotlib as dependency #1747
-   fix install for visualization tutorial in colab #1752
-   fix: Allow multiple connections in Solara #1759
-   Revert \"Ensure sphinx\>=7\" #1762
-   fix README pic to remove line on left side #1763
-   space: Ensure get_neighborhood output & cache are immutable #1780
-   fix: Use .pytemplate for name for cookiecutter #1785
-   HISTORY.rst: Correct neighbor_iter() replacement in 2.0.0 #1807
-   docs: Always link to stable version #1810
-   Remove exclude_none_values #1813

# 2.1.1 (2023-08-02)

This release improves the introductory and visualization tutorial.
Ensures both are Google Colab compatible with working badges.

Changes:

:   -   Update `intro_tutorial` to warn users to ensure up to date
        version, and make colab compatible #1739, #1744
    -   Improve new/experimental Solara based visualization to ensure
        pause button works #1745
    -   Fix bug in `space.py` -\> `get_heading()` #1739

# 2.1.0 (2023-07-22) Youngtown

This release creates `mesa.experimental` namespace, this solves the
issue that PyPI release will not allow git-based install.

**Users should read the Mesa 2.0.0 release note (directly below this),
as this contains the details about the breaking changes and other major
changes that were part of Mesa 2.0 release.**

Changes:

:   -   Creates `mesa.experimental` namespace #1736
    -   Fix Ruff lint error #1737
    -   Update permissions for PyPI #1732

# 2.0.0 (2023-07-15) Wellton

**Special notes**

Mesa 2.0 includes:

:   -   **an experimental pure python user interface/ visualization that
        is also jupyter compatible please see the**
        `visualization tutorial`\_
    -   an improved `datacollector` that allows collection by agent type
    -   several breaking changes that provide significant improvements
        to Mesa.

**Breaking Changes:**

-   space: change `coord_iter` to return `(content,(x,y))` instead of
    `(content, x,y)`; this reduces known errors of scheduler to grid
    mismatch #1566, #1723

-   space: change NetworkGrid `get_neighbors` to `get_neighborhood`;
    improves performance #1542

-   space: raise exception when pos is out of bounds in
    `Grid.get_neighborhood` #1524

-

    space: remove deprecations (#1520, #1687, #1688):

    :   -   `find_empty()`: convert this to `move_to_empty()`
        -   `num_agents`: removed parameter from `move_to_empty()`
        -   `position_agent()`: convert this to `place_agent`
        -   `neighbor_iter()`: convert this to `iter_neighbors()`

-

    batchrunner: remove deprecations #1627

    :   -   `class BatchRunner` and `class BatchRunnerMP`: convert these
            to `batch_run()`
        -   Please see this `batch_run() example`\_ if you would like to
            see an an implementation.

-

    visualization: easier visualization creation #1693

    :   -   `UserSettableParameter(['number', 'slider','checkbox', 'choice', 'StaticText'])`:
            convert to `NumberInput` , `Slider`, `CheckBox`, `Choice`,
            `StaticText`
        -   Please see this `visualization example`\_ if you would like
            to see an implementation.

**New Features:**

-   datacollector: can now handle data collection by agent type #1419,
    #1702
-   time: allows for model level `StageActivation` #1709
-   visualization: `ChartModule` can have dynamically named properties
    #1685
-   visualization: improved stop server to end visualizations #1646
-   *experimental* python front end option: integrated the initial
    prototype of the pure python front end option #1698, #1726

**Improvements**

-   update HexGrid and create HexSingleGrid and HexMultiGrid #1581

-   correct `get_heading` for toroidal space #1686

-   update slider to start at 1FPS #1674

-   update links to examples repo due to creation of mesa_examples
    #1636, #1637

-

    ** CI Improvements**

    :   -   update Ruff #1724
        -   remove Pipfile and Pipfile.lock #1692
        -   enable Codespell in Jupyter #1695
        -   improve regex for better build #1669, #1671
        -   exclude notebooks form linter #1670
        -   updated pip for zsh #1644
        -   CLI quality of life improvements #1640

-

    **Docs Improvements**

    :   -   update to PyData theme #1699
        -   remove .rst to create simpler build #1363, #1624
        -   use seaborn in tutorials #1718
        -   fix types and errors in docs #1624, #1705, #1706, #1720
        -   improve tutorials #1636, #1637, #1639, #1641, #1647, #1648,
            #1650, #1656, #1658, #1659, #1695, #1697,
        -   add nbsphinx to adv_tutorial #1694
        -   replace `const chart` for `var chart` in advanced tutorial
            #1679

-   update LICENSE to 2023 #1683

# 1.2.1 (2023-03-18)

This release fixes <https://github.com/mesa/mesa/issues/1606>,
where `mesa startproject` doesn\'t work.

Changes:

-   fix: Include cookiecutter folders in install content #1611
-   Fix Ruff errors and pin Ruff version #1609
-   datacollector: Add warning when returning empty dataframe with no
    reporters defined #1614

# 1.2.0 (2023-03-09) Taylor

**Special notes**

New features:

-   Implement radius for NetworkGrid.get_neighbors #1564

Some highlights for the perf improvements:

-   Use getattr for attribute strings in model data collection #1590
    this is a 2x speedup over the relevant line
-   Faster is_integer function for common cases #1597 is for 1.3x
    speedup for grid access (grid\[x, y\])
-   Refactor iter/get_cell_list_contents methods #1570 at least 1.3x
    speedup for iter/get_cell_list_contents
-   Evaluate empties set more lazily #1546 (comment) \~1.3x speedup for
    place_agent, remove_agent, and move_agent

**Improvements**

-   ci: Add testing on Python 3.11 #1519

-   Remove auto-update GH Actions for Pipfile.lock #1558

-

    ruff

    :   -   ruff: Add isort #1594
        -   ci: Replace flake8 with Ruff #1587
        -   ruff: Add more rules based on Zulip\'s config #1596

-   perf: faster is_integer function for common cases #1597

-   Remove \_reporter_decorator #1591

-   Change index at DataFrame creation in get_agent_vars_dataframe #1586

-   Make Grid class private #1575

-   Make the internal grid and empties_built in Grid class private #1568

-   Simplify code in ContinuousSpace #1536

-   Improve docstrings of ContinuousSpace #1535

-   Simplify accept_tuple_argument decorator in space.py #1531

-   Enhance schedulers to support intra-step removal of agents #1523

-   perf: Refactor iter_cell_list_contents Performance #1527

-   Replace two loops with dictionary comprehension, list- with
    generator comprehension #1458

-   Make MultiGrid.place_agent faster #1508

-   Update space module-level docstring summary #1518

-   Update NetworkGrid.\_\_init\_\_ docstring #1514

-   Deprecate SingleGrid.position_agent #1512

-   Make swap_pos part of Grid instead of SingleGrid #1507

-   Refactor NetworkGrid docstrings and iter/get_cell_list_contents
    #1498

-   Hexgrid: use get_neighborhood in iter_neighbors #1504

-   Auto update year for copyright in docs #1503

-   Refactor Grid.move_to_empty #1482

-   Put \"Mesa\" instead of \"it\" in README #1490

-   Batchrunner: Remove unnecessary dict transformation, .keys() in
    len() #1460

-   Add Dependabot configuration for GitHub Actions update check #1480

-   Use list transformation only when shuffled is True #1478

-   Implement swap_pos #1474

-   Clean up DataCollector #1475

**Fixes**

-   Update resources in README #1605

-   Fix accident from <https://github.com/mesa/mesa/pull/1488>
    #1489

-   pre-commit autoupdate #1598, #1576, #1548, #1494

-   Fix docstring of DataCollector #1592

-   Update Pipfile.lock (dependencies) #1495 #1487

-

    build(deps):

    :   -   build(deps): bump codecov/codecov-action from 2 to 3
            dependencies Pull requests that update a dependency file
            #1486
        -   build(deps): bump actions/upload-artifact from 2 to 3
            dependencies Pull requests that update a dependency file
            #1485
        -   build(deps): bump peter-evans/create-pull-request from 3 to
            4 dependencies Pull requests that update a dependency file
            #1484
        -   build(deps): bump actions/setup-python from 3 to 4
            dependencies Pull requests that update a dependency file
            #1483

-   Establish reproducibility for NetworkGrid.get_neighbors when radius
    \> 1 #1569

-   Format js code #1554

-   Add some missing const declarations #1549

-   fix tutorial url in examples #1538

-   Update cookiecutter to flat import style. #1525

-   Fix bug in Grid.get_neighborhood #1517

-   Revert changes of #1478 and #1456 #1516

-   Fix return types of some NetworkGrid methods #1505

-   Update year for copyright #1501

-   Add default_value function to NetworkGrid #1497

-   Remove extraneous spaces from docstrings in modules 2 #1496

-   Remove extraneous spaces from docstrings in modules #1493

-   SingleGrid: Remove extraneous attribute declaration (empties) #1491

# 1.1.1 (2022-10-21)

This release fixes <https://github.com/mesa/mesa/issues/1461>
where custom user-specified portrayal images don\'t load in the
visualization server.

# 1.1.0 (2022-10-10) Safford

**Special notes**

-   Perf: ContinuousSpace: speed-up add/remove agents #1376. This is a
    \~6x performance improvement for add/remove.
-   fix: time: Recompute agent_keys between stages #1391. This is a
    correctness fix for `SimultaneousActivation` and `StagedActivation`
    when agents are being removed during simulation.
-   ModularServer: Always set model.running = True on reset #1399. With
    this change, specifying `self.running = True` in your model
    `__init__` is now optional. Mesa\'s visualization server will
    automatically sets it to `True` in the beginning of a simulation.
-   feat: Allow user-specified local dir to be served by Tornado #1435.
    This simplifies the usage of `ModularServer` in Mesa-Geo.
-   Allow batch_run to take arbitrary parameters #1413. With this
    change, you can finally use any arbitrary Python objects as
    `batch_run` parameters, where previously they are restricted to
    hashable objects only.
-   Prevent seed and random from being shared between instances #1439.
    With this fix, a model instance has their own isolated RNG.

**Improvements**

-

    CI Updates

    :   -   ci: Cancel previous obsolete runs #1378
        -   ci: update black to prevent click error #1382
        -   Add \"falsy\" to .codespellignore #1412
        -   Upgrade pre-commit CI (with pyupgrade and syntax checks)
            #1422

-

    Tests

    :   -   test: RandomActivationByType: Test adding agents with
            duplicate ID #1392

-

    Dependency updates

    :   -   Update Pipfile.lock (dependencies) #1398
        -   Update Pipfile.lock (dependencies) #1408
        -   Update Pipfile.lock (dependencies) #1434

-

    Docs

    :   -   docs: Add Tim Pope\'s guideline for proper Git commit msg
            #1379
        -   readme: Improve the pip install for Git repo instruction
            #1416
        -   Docs: Remove trailing whitespaces #1421
        -   Fixes #1423 - fixes build badge in docs #1424

-

    Refactors

    :   -   refactor: Apply pyupgrade \--py37-plus #1429
        -   refactor ModularServer (moving code into \_\_init\_\_) #1403

-   Perf: ContinuousSpace: speed-up add/remove agents #1376

-   Remove monospace formatting for hyperlinks #1388

-   ModularServer: Always set model.running = True on reset #1399

-   Allow batch_run to take arbitrary parameters #1413

-   ModularServer: Put new optional arg port last #1432

-   feat: Allow user-specified local dir to be served by Tornado #1435

-   Improve and measure speed of clamp function #1440

**Fixes**

-   Fix stray \" in modular_template.html #1380
-   Fix zoom on network visualisation #1381
-   Fix broken monospace links #1387
-   fix: Ensure agent id is unique in RandomActivationByType.add #1386
-   fix: time: Recompute agent_keys between stages #1391
-   Fix batchrunner progress bar #1395
-   Fix stray \" in visualisation dropdown labels #1409
-   space: Fix type error for Python \< 3.9 #1430
-   Prevent seed and random from being shared between instances #1439

# 1.0.0 (2022-07-06) Quartzsite

**Special notes**

-   BREAKING: Rename mesa.visualizations.TextVisualization.TextElement
    to ASCIIElement
-   POTENTIALLY BREAKING: Default batch_run to 1 CPU #1300
-   Simplified namespace implements - see Improvements section.

**Improvements**

-

    Implement simplified namespace

    :   -   docs: simplified namespace tutorial update #1361
        -   examples: Convert shape_example, sugarscape_cg,
            virus_on_network, wolf_sheep to simple namespace #1339
        -   Convert hex_snowflake, pd_grid, schelling to simple
            namespace; \[BREAKING\] Remove class name collision #1333
        -   examples: Convert color_patches, conways_game_of_life,
            epstein_civil_violence, forest_fire to simple namespace
            #1331
        -   Examples: Convert boltzmann_wealth_model_network and chart
            to simple namespace #1322
        -   examples: Convert boid_flockers, boltzmann_wealth_model to
            simple namespace #1321
        -   examples: Convert bank_reserves to simple namespace #1317
        -   add batch_run to simple namespace #1316
        -   Implement simpler Mesa namespace #1294

-

    mypy

    :   -   mypy: Use \"\|\" operator instead of Union/Optional #1345
        -   mypy: Improve space.py annotation, part 2 #1219
        -   mypy: Improve annotations #1212

-

    Userparam class updates

    :   -   feat: Implement NumberInput UserParam class #1343

        -   feat: Implement StaticText UserParam #1342

        -   feat: Implement Choice UserParam class #1338

        -   feat: Implement Checkbox UserParam class #1332

        -

            feat: Implement Slider UserParam class #1272

            :   -   examples: Convert to using Slider UserParam class
                    #1340

-

    Front-end updates

    :   -   frontend: Add alignment options to agent portrayals in
            CanvasGridVisualization #1349

        -   frontend: Update Bootstrap 4.6.1 -\> 5.1.3 #1325

        -   ChartModule.js: Use more semantic HTML element creation
            method #1319

        -   Issue #1232; Replaced usage of var to const/let in some
            files #1248

        -   \[Issue 1232\] Refactor NetworkModuleSigma PieChartModule
            TextModule JS #1246

        -   js: Update D3 from 4.13.0 to 7.4.3 #1270

        -   support package and local css includes #1283

        -   Upgrade to Bootstrap 4! #1282

        -   refactor: update var to const/let in InteractionHandler.js
            #1273

        -   Change remaining vendored JS dependencies to be downloaded
            during install #1268

        -   Download jQuery and D3 during install #1260

        -   CSS support for custom visualization element #1267

        -   style: prettify js files #1266

        -   refactor: Change var to const/let for remaining js files
            #1265

        -   Remove NetworkModule_sigma and its dependencies #1262

        -   js: Download bootstrap-slider during install #1257

        -   js deps: Move Bootstrap to be inside external folder #1236

        -   Apply prettier to NetworkModule_d3.js #1225

        -   js: Download Bootstrap on-the-fly during install instead
            #1220

        -   Install JS dependencies using Fanstatic #1195

        -

            JQuery updates

            :   -   examples: Remove all usage of jQuery #1356
                -   Remove jQuery dependency completely #1355
                -   refactor: frontend: Remove remaining usage of jQuery
                    #1351
                -   refactor: frontend: Remove usage of jQuery for most
                    of the JS code #1348
                -   refactor: frontend: Remove jQuery usage in
                    CanvasHexModule.js & CanvasModule.js #1347
                -   refactor: frontend: Remove jQuery usage in
                    BarChartModule.js #1326
                -   visualization: Specify tooltip without jQuery #1308

-

    CI Updates

    :   -   ci: Ensure wheel is install before pip install step #1312
        -   Fix contributing (increasing black version) #1303
        -   ci: Disable PyPy for now #1254
        -   CI: Update checkout, setup-python and cache actions to v3
            #1217
        -   CI: Split off codespell job, don\'t run build on doc changes
            #1170
        -   ci: Add 6 min timeout for the test jobs #1194
        -   CI: test flake: batch runner sometimes takes 6 hours then
            gets killed by GitHub Actions #1166
        -   ci: Enable cache for all Python versions 🇺🇦 #1177
        -   CI: Create Action to publish to PyPI on release #1169
        -   CI: Python 3.6 should be removed because it has reached EOL
            #1165
        -   Update Black formatting (no spaces for power operator) #1160
        -   Improve code quality with static analysis #1328
        -   CI test: Increase timeout to 10 minutes #1250

-

    Dependency updates

    :   -   build(deps): bump cookiecutter from 2.1.0 to 2.1.1
            dependencies #1360
        -   Update Pipfile.lock (dependencies) #1374, #1350, #1301,
            #1224, #1203, #1135 by github-actions bot
        -   Migrate D3 from v4 to v7 #1088

-

    Other Improvements

    :   -   feat: Implement auto-conversion of function to TextElement
            #1346
        -   Readme: Add Matrix badge and description #1164
        -   examples: Convert nodes to list when drawing random
            sample#1330
        -   examples: Use nicer color for bank_reserves #1324
        -   examples: Use nicer color for chart #1323
        -   model: Implement initialize_data_collector #1287
        -   CONTRIBUTING: Add instruction to enable git pull autorebase
            #1298
        -   Improve MANIFEST.in #1281
        -   refactor: Merge \_remove_agent into remove_agent #1245
        -   examples: Remove usage of internal method \_remove_agent
            #1241
        -   refactor: Make \_place_agent args consistent with
            place_agent #1240
        -   Redirect user to GH discussions for seeking help #1237
        -   setup.py: Update setup classifiers and add python_requires
            for Python\>=3.7 #1215
        -   The tutorial.rst doesn\'t mention that the Pandas DataFrame
            output can be in CSV #1148
        -   Deprecate neighbor_iter in favor of iter_neighbors #1184
        -   Add snippet about using numpy\'s random #1204
        -   docs: make windows multiprocessing code appear #1201
        -   Capitalize CSV whenever applicable #1200
        -   update intro tutorial for pandas and CSV and batch_run and
            windows #1196
        -   docker-compose.yml: Make it consistent with Dockerfile #1197
        -   Improve Dockerfile #1193
        -   update to include Matrix and GitHub discussion links #1179
        -   Update docs to remove old discussion forums #1171
        -   Add \"Grass\" curve to wolf_sheep example chart #1178
        -   feat: Implement random activation by type #1162

**Fixes**

-   Git tags out of sync with conda and PyPi (0.8.8 and 0.8.9 missing on
    git) #1076

-   fix: Remove mesa.visualization.Number #1352

-   CI: the \"install dependencies\" step is slow #1163

-

    Readme related

    :   -   readme: Clarify/Update Docker instruction #1222, #1214
        -   Readme: Fix links to docs #1205

-   Add mesa/visualization/templates/js/external to gitignore #1320

-   fix: sugarscape_cg: Use better way to check if a cell is occupied by
    SsAgent #1313

-   fix double multiply of iterations in singleprocess #1310

-   pre-commit: fix required python version, correct example commit
    message... #1302

-   fix: Make bank_reserves batch_run example work #1293

-   Fixes #498. Replaces canvas_width with grid_rows to fill out color
    patches 3 - Accept easy task!!! #989

-   update pre-commit to include jupyter; fix warning #1190

-   fix: Grid.\_\_getitem\_\_: Handle Numpy integers #1181

-   fix: Make argument order in example models consistent #1176

-   issue template: Linkify discussions url #1239

-   batch_run: Do not iterate values when it is a single string #1289

-   examples: Clarify install instruction in wolf_sheep #1275

-   test: Disable batchrunnerMP (CI: test flake: batch runner sometimes
    takes 6 hours then gets killed by GitHub Actions #1166) #1256

-   examples: correcting comment in examples/pd_grid/pd_grid/agent.py
    #1247

-   space: Clarify the return object of get_cell_list_contents #1242

-   width and height were changed up #1149

-   fix typo in best-practices.rst #1368

-   fix: examples: Make space x, y order consistent #1366

# 0.9.0 (2022-01-31) Page

**Improvements**

-   Update number_processes and associated docs #1141

-   \[PERF\] Improve move_to_empty performance #1116

-   Adding logic to check whether there is agent data #1115

-   Convert all text strings to f-strings #1099

-   Format Python and Jupyter Notebook files with Black #1078

-   README: Add info on how to cite Mesa #1046

-   Re-Implementation of BatchRunner #924

-

    CI Related

    :   -   CI: Add workflow to update Pipfile.lock every month #1018
        -   CI: Lint typos with Codespell #1098
        -   CI: Only run Codecov on Ubuntu jobs and update to v2 #1083
        -   CI: Maintenance: Update to Python 3.10, split of lint jobs
            #1074

-

    Dependency updates

    :   -   Updates to Pipfile.lock (dependencies) #1114, #1086, #1080
        -   Update Pipfile to use Python 3.9 #1075
        -   Update Chart.js to 3.6.1 (v3) #1087
        -   Update Chart.js to version 2.9.4 #1084
        -   Pyupgrade 3.6: Update syntax with Python 3.6+ features #1105
        -   Bump urllib3 from 1.26.2 to 1.26.5 #1043
        -   Update packages.rst #1068

-

    Docs

    :   -   Update docs/README.md #1118
        -   Update number_processes and associated docs #1141
        -   Update section \'Batch Run\' of introductory tutorial #1119
        -   Readme: Add command to install specific branch #1111
        -   Docs: Add back some comments in space.py #1091
        -   Docs: Remove trailing white spaces #1106
        -   Update intro_tutorial.rst #1097, #1095
        -   Tweaking and improving the documentation #1072

**Fixes**

-   Rename i_steps -\> data_collection_period and add docstring #1120

-   bank_reserves: Say that the commented out legacy code is for
    comparison #1110

-   Fix broken image on PyPI #1071

-

    Docs

    :   -   Fix numbering typos in docs/README.md #1117
        -   Readme: Fix command for installing custom branch on fork
            #1144
        -   Docs: space.py: Fix single case of neighbor spelled as
            neighbour #1090

# 0.8.9 (2020-05-24) Oro Valley

*Note: Master branch was renamed to Main on 03/13/2021*

**Improvements**

-

    Master to Main change:

    :   -   Docs/examples: Update links to use main instead of master as
            branch #1012
        -   CI: Run on pushed to main and release branches #1011

-

    Github Actions

    :   -   GitHub Actions: run black only on ubuntu 3.8 #996
        -   GA: Only run CI when pushed to master #974
        -   GA: Add pypy3 #972
        -   rename github action to \"build\", remove redundant flake8
            check #971
        -   GA: Run on Windows and macOS #970
        -   Add GitHub Action for continuous integration testing #966

-   \[PERF\] Add neighborhood cache to grids and improve
    iter_cell_list_contents #823

-   forest_fire: Remove unnecessary code #981

-   Migrate away from type comments #984

-   Update License #985

-   Public remove_agent function for NetworkGrid #1001

-   Date update to release #962

-   Advanced indexing of grid #820

**Fixes**

-   Correct spelling #999
-   Update Pipfile.lock #983
-   Fix order of variable_params in model and agent vars data frames
    #979
-   Fix asyncio on windows with python 3.6 #973

# 0.8.8 (2020-11-27) Nogales

*Note: This is the last version to support Python 3.5.*

**Improvements**

-   Added pre-commit to automatically maintain a standard formatting
    with black #732

**Fixes**

-   MultiGrid: Set to using list for cell content #783

-

    Docs

    :   -   Fixed broken link to templates list in advanced tutorial.
            #833
        -   Fixed image links in rst #838
        -   Cleaned html to attempt correct build #839
        -   Fixed links on Boltzmann model #843
        -   Documentation update - batchrunner & data collector #870
        -   Deleted readthedocs.yml #946
        -   Doc builds #837, #840, #844, #941, #942

-   Fixed bulleted list in contribution read me #836

-   Updated test_examples.py, changed unused generator expression to
    actually run the models. #829

-   Fixed #827 issue (example Epstein Civil Violence Jupyter Notebook
    typos) #828

-   Eliminated Ipython3 references #841

-   Fixed cookie cutter Fixes #850. #853

-   Removed relative imports \-- fix #855. #863

-   Updated pytest requirement to fix issues on travis #864

-   Made linux compatible - travis #886

-   Fixed python 3.5 fails, boid failure #889, #898

-   Travis: Removed python 3.5 #899

-   Fixed example testing issues close multiprocess pools #890

-   Used ordered dict to make compatible with python 3.5 #892

-   Increased number of test to fix codecov patch #916

-   Fixed for #919, adding an exception for duplicate ids. #920

-

    Batchrunner

    :   -   Batch runner redux #917
        -   Fixed empty/None `variable_parameters` argument to
            BatchRunner (#861) #862
        -   Added ordereddict to BatchrunerMP for python 3.5 #893
        -   Fixed python 3.5 fails bathrunnerMP (multiple tries) #897,
            #896, #895
        -   Batchrunner_redux fixes #928

-   Fixed variables names, mp function locations, datacollector #933

-   ModularServer updated: Fix EventLoop and changes to default port
    #936

-   Ran black 20.8b1, which formats docstrings #951

# 0.8.7 (2020-05-05) Mammoth

**Improvements**

-   Enable BatchRunner to run specified set of parameter combinations
    #651 (#607)

-   Restructured runcontrol.js #661

-   Add pipenv support for mesa #678

-   Increase test coverage and change to codecov #692

-   Updates Travis to explicitly set the dist to be Xenial #699

-   time: Remove resolved TODO on random seed of random scheduler #708

-   hex_snowflake: Update description to be more informative #712

-   Added Coverall to Codecov in Contributing file #734

-   Makes opening the browser optional when launching the server #755
    #754

-   NetworkGrid: Update to networkx 2.4 API #763

-   Apply black to mesa/ directory #775

-   Updated travis to 3.8 and updated gitignore #777

-   Add information (to docstring) on image as agent portrayal shape
    #791

-   Change grid empties from list to set #649 (improves speed)

-

    Adding mypy annotation

    :   -   space: Add type annotation to Grid class #779
        -   add Mypy annotation to time, agent, and model #792
        -   space: Add mypy annotation to the remaining
            methods/functions #796

-

    Docs related

    :   -   Bulk merge of docs from \'docs\' to \'master\' #684

        -

            Created useful snippets code section in the docs #668 #669

            :   -   Updating index.rst #672
                -   Clarify runserver snippet in index.rst #682

        -   Add documentation for feature (pipenv) added in #678 #683

        -

            Add docs for BatchRunner to support Variable and Fixed Parameter Contribution #679 #683

            :   -   Resources #651 in docs branch #691. This preps for
                    #683 to be merged.

        -   intro tutorial: Clarify a function that is not defined in
            the class #705

        -   Updates formatting the readme Docs markdown #737

-

    Examples related

    :   -   Schelling: Separate text-only viz into run_ascii.py #706
        -   examples/Readme.md: Update description to be consistent with
            the folder names #707

**Fixes**

-   Fixes link to update code coverage module - Updates Removing last
    link to coveralls and replacing to codecoverage #748

-   Fixes D3 Network Visualization to update (rather than overwrite)
    #765 #767

-   Fix parameter order in initializing SingleGrid object #770 #769

-   Updating pipenv link #773

-   Fixed pip install from github by specifying egg #802

-

    Compatibility fixes

    :   -   Fixes VisualizationServer to be compatible with recent
            versions of Tornado #655
        -   Fixes #749 networkx incompatibility #750

-

    Fixing typos

    :   -   Fixes documentation typos in example code #695 #696
        -   Fixes typo in ModularServer\'s last parameter #711
        -   Fixed typo in BarChartModule line 100 #747
        -   Fix typo in documentation #809

-

    Doc fixes (not relating to typos)

    :   -   Update tutorial to point to correct repo location #671 #670
        -   Updating sphinx and reverting issues #674 #675 #677 #681
        -   Fixes code blocks that weren\'t showing up in the tutorial
            #686
        -   Remove figure from advanced tutorial showing the empty
            visualization #729
        -   Removes git clone from tutorial - Update intro_tutorial.rst
            #730
        -   Fixes citation links in docs tutorial section #736
        -   Fix histogram in advanced tutorial #794 #610
        -   Fixes Advanced Tutorial #elements #804 #803

-

    Fixes to examples

    :   -   Fixing test_random_walk bug - wolf sheep. #821
        -   Fixes shape_example server launch #762 #756
        -   Fixing broken table in pd_grid example #824

# 0.8.6 (2019-05-02) Lake Havasu City

**Improvements**

-   add docker-compose + Dockerfile support #593
-   install: Remove jupyter requirement #614
-   Add Bar and Pie Chart visualization #594 #490
-   Make models pickleable #582

**Fixes**

-   Year update. Happy New Year! #613
-   Fixed problem with grid and chart visualization javascript #612 #615
-   removed extra\" .random\" on line 178. #654
-   updated requirement for networkx #644 #646
-   Fix VisualizationServer to be compatible with recent versions of
    Tornado #655

# 0.8.5 (2018-11-26) Kearny

**Improvements**

-   Added mouse interactionHandler to close #457, fixed hexgrid
    drawLines #465
-   Run examples as part of the tests #529, #564
-   Add a github issue template. #560
-   Changes nose to pytest #561
-   Update and clean up cookiecutter layout #563
-   Updating setup to move requirements to setup.py. #566
-   Fixes #570 removed and updated stale comments in space.py #571
-   Adding model random number generator with \_\_new\_\_ #572
-   Faster agent attribute collection #576
-   Update install command to be edible #578
-   Adding read the docs yml. #579
-   agents can be removed and added during Scheduler.step() #584
-   Adding a description to bank_reserves. #587
-   F8 cleanup #600

**Fixes**

-   Fixes #543 (User Settable Parameters fail to work for non-string
    datatype #543) #544
-   Adding missing requirements files to examples. #550
-   Fixes issue #548, flockers visualization not showing up #548
-   updated BatchRunner (throwing error when passing in agent reporters)
    #556
-   Removing version numbers and fixing flake8 issues. #562
-   Fix issue #548 (Flockers visualization is not working) #566
-   Fixes code formatting in readmes. #577
-   Batchrunner.fix (BatchRunner\'s \"variable parameters\" is not
    strictly optional) #596

# 0.8.4 (2018-06-17) Jerome

**Improvements**

-   Mesa Packages docs created (#464, #480, #484, #503, #504)
-   Change size and tooltip text of nodes in D3 network visualization
    #468
-   Multiprocessing BatchRunner with pathos #506
-   Schedule.agent.dict - Implement tracking the agents in the scheduler
    via OrderedDict #510
-   Use click and add `mesa run` #522
-   Add a code of conduct #530

**Fixes**

-   iter_neighborhood() now gives correct neighborhoods for both von
    Neumann and Moore #459
-   fix typo #461
-   Flockers update & subsequent \"F\" versus \"f\" fix on Unix/Mac -
    #477, #518, #525, #500
-   Fixing date on release. #453
-   Batchrunner fixes: properly initialize models with correct
    parameters during subsequent runs. #486
-   Tornado Version Bug Fixes (upgrading #489, downgrading #497, adding
    to setup.py #527)
-   fix minor flake8 issues #519
-   align required dependencies between setup.py and requirements.txt
    #523, #528, #535
-   Fixes #499 grid size issue. #539

# 0.8.3 (2018-01-14) Hayden

**Improvements**

-   Datacollector fix #445
-   A first network grid model with visualization, using NetworkX and
    sigma.js #388
-   Cache pip packages for Travis setup #427
-   Remove localhost hardcoding + allow secure sockets #421
-   Update Chart.js to version 2.7.1 #401
-   Bank reserves example #432
-   Extended Grid to support hexagonal grids #409

**Fixes**

-   Faster ContinuousSpace neighbor search #439
-   Updating license year to 2018 #450
-   Updating language on license in contributing file #446
-   Updating license year to 2018 #450
-   Removed mutable defaults from DataCollector constructor #434
-   \[BUGFIX\] Torus adjustment in Grid class #429
-   Batchrunfixedparameters #423
-   \[BUGFIX\] Fix sidebar visibility in Edge #436
-   Updating Travis svg to target #master, not branches. #343
-   Email list language updates and link updates #399
-   Fix math problems in flockers; use numpy in space #378
-   Only start tornado ioloop if necessary #339
-   ContinuousSpace: Fix get_distance calculation on toroidal boundary
    condition #430

# 0.8.2 (2017-11-01) Gila Bend

**Improvements**

-   Split parameter_values into fixed & variable parameters in
    batchrunner #393

**Fixes**

-   Updating License year to 2017 \-- very minor update #391

-   Flockers: fix param naming #398

-   Remove unused class parameters. #400

-   \[hotfix!\] Disable e2e viz test for now. #414

-

    Fixing bug in release process. \[6a8ecb6\]

    :   -   See <https://github.com/pypa/pypi-legacy/issues/670>.

# 0.8.1 (2017-07-03) Flagstaff (PyCon Sprints & then some)

**Improvements**

-   Bootstrap UI starter #383

-   Add Sugarscape Constant Growback example #385

-   Add best-practices document and describe models. #371

-

    Refactored & model standards related:

    :   -   Prisoner\'s Dilemma refactor to meet new model standard
            format. #377
        -   refactored boltzmann wealth model to new layout #376
        -   Update tutorial to follow new model standards #370
        -   Moving wolf sheep pngs to sub-folder for better organization
            #372
        -   Add best-practices document and describe models. #371

-   Modified loop over agents in schedule step method #356

-   Added function to use local images as shapes in GridDraw #355

**Fixes**

-   Fix math problems in flockers; use numpy in space #378
-   Seed both global random number generators #373, #368
-   Dictionary parameters fix #309
-   Downgrade setuptools to fix #353
-   Minor forest fire fix #338, #346
-   Allow fixed seed for replication #107
-   Fix tutorial and example readme for port change 8b57aa

# 0.8.0 (2017-01-29) - Edgar

**Improvements**

-   Updating contribution file to prevent future travis breaks #336
-   Updating Travis svg to target #master, not branches. #343
-   implement \"end\" message in visualization #346
-   Move empty-cell functions to baseclass Grid #349

**Fixes**

-   Only start tornado ioloop if necessary #339
-   fix boundaries of ContinousSpace #345

# 0.7.8.1 (2016-11-02) Duncan

**Improvements**

-   Fixes #324 \-- renames all examples to be the pythonic format of
    naming #328
-   Changing to port 8521, fixes #320. #321
-   Opens a browser window when launching the server #323
-   Ticket #314 - added progress bar to BatchRunner #316
-   Auto update year for copyright. #329

**Fixes**

-   Minor bug fixes - Update ForestFire example notebook to new API, and
    rename Basic to Shape Example. #318
-   On-demand model stepping rather than an endless buffer #310
-   Updating contribution to prevent future travis breaks #330

# 0.7.7 (2016-08-18)

**Improvements**

-   Fixes - variable name heading0/1 in ArrowHead shape is not
    intuitive. #295 #301
-   Fixes - ArrowHead shape is not reflecting in the docs of api #300
    #301
-   Fixes - Documentation is not reflecting latest changes wrt
    width-height argument order in Grid() #296 #301

# 0.7.6 (2016-08-13)

Theme: Scipy Sprints 2016 ( '-')人(ﾟ_ﾟ ) & Then some.

**Feature adds**

-   Add new shapes & direction indication in CanvasGrid #285
-   Provides support for text overlay on Circle and Rectangle shapes.
    #265

**Improvements**

-   Fixes Parameters of CanvasGrid(): row, col, height, width inverted
    #285
-   Fixes \'coordinates on grid are used inconsistently throughout the
    code\' #285
-   Moves Agent and Model class outside of \_\_init\_\_.py #285
-   Minor pep updates to boltzmann. #269
-   Fix link to intro tutorial. #267
-   Updating template text visualization/ModularVisualization.md #273
-   Update intro_notebook and documents to include self.running = True
    in MoneyModel #275
-   Update .rst file location to make sure ReadTheDocs works correctly
    #276
-   Remove Mock code causing recursion and preventing build of docs.
    #281
-   MultiGrid docstring missing methods #282
-   No Docstring for model.grid.get_cell_list_contents #282
-   Refactor forest fire example #223 #288
-   Updating kernel version on forest fire model. #290
-   Making examples pep complaint. fixes #270 #291
-   Fixed pep8 examples and #292 #294
-   Fixes #283 - Fixes formatting on viz readme #299
-   Have Agent use self.model instead of passing it around #297

# 0.7.5 (2016-06-20)

**Pre-sprints**

-   Update of tutorial files and docs #176, #172
-   Adds np.int64() functions around some variables to get rid error
    caused by numpy update #188
-   Made examples Readme.md more readable #189

**From PyCon Sprints**

-   Updating model example readmes #207
-   Added nose to requirements #208
-   Updated link on style google style guide #209
-   Reset visualization when websocket connection is opened #210
-   Remove unused scipy dependency #211
-   Introduce a requirements.txt for the tutorial. #212
-   Remove references to running in tutorial #213
-   Simplify travis.yml; add python versions #215
-   Update Flocker Readme.md #216
-   Syntax error in .rst was swallowing a code block #217
-   Fixup HistogramModule in the tutorial. #218
-   add more test coverage to time #221
-   add a requirements.txt for WolfSheep. #222
-   add a requirements.txt for Schelling. #224
-   Refactor color patches example #227
-   Ignored \_build sphinx docs still in repo #228
-   Intro Tut completely in ipynb #230
-   pass optional port parameter to ModularServer.launch #231
-   open vis immediately when running color patches #232
-   Adds .DS_store to .gitignore #237
-   Documentation Update #240
-   Small fix for reading links #241
-   Test batchrunner #243
-   clean up TextVisualization #245
-   Documentation Update #250
-   Update Game of Life example to new format #253
-   Update Flockers example to new format #254
-   Update Epstein model to new layout #255
-   Subclassing object is unnecessary in Python 3 #258

**Post PyCon Sprints**

-   Adds a copy of jquery directly into the code. #261

# 0.7.0 (2016-03-06)

-   #184 Adding terminal echo for server launch to signal person running
    the model
-   #183 Adding Conway\'s Game of Life simulation to the examples.

# 0.6.9 (2016-02-16)

-   #170 Adding multi-stage activation
-   #169 Wolf-Sheep Cleanup
-   Updates requirements to latest libraries

# 0.6.7 (2015-07-11)

**Improvements**

-   Allow cell_list_content methods in Grids to accept single tuples in
    addition to lists

# 0.6.6 (2015-07-11)

Theme: Scipy Sprints ( '-')人(ﾟ_ﾟ )

**Improvements**

-   Standardizes the arguments passed to spatial functions to only
    tuples, not separate x and y coordinates. (Breaks backwards
    compatibility)

0.6.5.1 (2015-07-11) ++++++++++++++++++

Theme: Scipy Sprints ( '-')人(ﾟ_ﾟ )

**Improvements**

-   Adding version, license, copyright, title to \_\_init\_\_.py
-   Auto updating version in setup.py

**Fixes**

-   Updating MANIFEST.in to include visualization templates that were
    missing.

# 0.6.5 (2015-07-11)

Theme: Scipy Sprints ( '-')人(ﾟ_ﾟ )

**Edits**

-   Additions to tutorial doc
-   Minor edits to README & Intro
-   Minor edits / clean up to setup.py
-   Removing .ipynb_checkpoints
-   Removing out-of-date planning documentation.

**Fixes**

-   Use setuptools\' find_packages function to get the list of packages
    to install, fixes #141

**Improvements**

-   Use package_data for include the web files
-   Use a MANIFEST.in file to include the LICENSE file in source
    distributions
-   Using conda on Travis allows much faster builds and test runs

# 0.6.2 (2015-07-09)

-   Improvement: Adding continuous space.

-   Improvement: Adding a simultaneous activation scheduler.

-

    New models:

    :   -   Flockers
        -   Spatial Demographic Prisoner\'s Dilemma (PD_Grid)

# 0.6.1 (2015-06-27)

-   Fixes: Order of operations reversed: agent is removed first and then
    it is placed.
-   Improvement: `LICENSE`\_ was updates from MIT to Apache 2.0.

# 0.6.0 (2015-06-21)

-   Improvement: Add modular server feature, which breaks up a model
    into a .py file and a .js file. This breaks backwards compatibility.

# Pre 0.6.0

Code that is pre-0.6.0 is very unstable.

Our initial release was 0.5.0 (2014-11).

It included code for placing agents on a grid; a data collector and
batch runner; and a front-end visualization using HTML 5 and JavaScript.

**General**

-   Objects create \-- Agent, Time, Space
-   Project moved to Python 3
-   Tornado server setup

**Front-end**

-   Front-end grid implemented
-   ASCII visualization implemented

**Examples models**

-   Forest Fire
-   Schelling
-   Wolf-Sheep Predation

**0.1.0 (2014-09-19)**

-   A conversation
-   Birth

<!-- vale on -->
