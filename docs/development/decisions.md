# Project decisions

Decisions that aren't obvious from the code or the other pages, and why we made them. Add an entry in the PR that implements a decision. If a decision changes, mark the old entry as superseded instead of deleting it.

## Scope and ecosystem

| Date | Decision | Refs | Status |
|---|---|---|---|
| 2026-01 | Mesa stays focused on agent-based modelling. Calibration, sensitivity analysis, and surrogate modelling belong in external libraries; examples show how to combine them. | [#3170](https://github.com/mesa/mesa/issues/3170), [#3207](https://github.com/mesa/mesa/discussions/3207) | current |
| 2026-01 | `mesa.space` is maintenance-only; new work goes to `mesa.discrete_space`. | [#3066](https://github.com/mesa/mesa/issues/3066), [#3111](https://github.com/mesa/mesa/issues/3111) | superseded: removed in Mesa 4.0 |
| 2024-10 | Core examples (a small set of canonical models, tested in CI and updated with each release) ship in `mesa.examples`; community models live in mesa-examples. A separate examples package caused inconsistent imports. | [#2387](https://github.com/mesa/mesa/issues/2387) | current |
| 2025-01 | SolaraViz stays experimental throughout Mesa 3.x: it is still evolving, and time alone doesn't make an API stable. | [#2636](https://github.com/mesa/mesa/issues/2636) | current |
| 2025-12 | Mesa-Geo gets a visualization design suited to geospatial data instead of copying Mesa's. | [#2927](https://github.com/mesa/mesa/discussions/2927) | current |

## Releases and documentation

| Date | Decision | Refs | Status |
|---|---|---|---|
| 2026-01 | Only deprecate something once a stable replacement exists; users need a path forward first. | [#3136](https://github.com/mesa/mesa/issues/3136) | current |
| 2026-02 | No deprecation warnings on the Mesa 4.0 development branch. Settle the scope of 4.0 first, then add warnings on the 3.x branch. | [#3330](https://github.com/mesa/mesa/issues/3330) | current |
| 2026-02 | Rewrite the migration guide and tutorials together for the Mesa 4 beta, once the accumulated changes are clear. | [#3337](https://github.com/mesa/mesa/issues/3337) | current |
| 2022-06 | Tutorials are Jupyter notebooks rendered into the docs, not maintained as a separate copy. | [#1363](https://github.com/mesa/mesa/issues/1363) | current |
| 2024-09 | Notebook execution errors fail the Read the Docs build, so broken tutorials don't go unnoticed. | [#2276](https://github.com/mesa/mesa/issues/2276) | current |

## Community and governance

| Date | Decision | Refs | Status |
|---|---|---|---|
| 2025-11 | AI use is welcome, but contributors must understand and own what they submit (Code of Conduct). | [#2898](https://github.com/mesa/mesa/issues/2898) | current |
| 2026-01 | Labeling and stale-marking stay manual maintainer triage. | [#3094](https://github.com/mesa/mesa/issues/3094) | partly superseded: area labels (docs, testing, ci, …) are now applied automatically from changed files ([#3813](https://github.com/mesa/mesa/pull/3813)); type labels and stale-marking are still manual |
| 2026-03 | Automated PR tooling should do triage, not code review, and be monitored and improved over time. | [#3485](https://github.com/mesa/mesa/discussions/3485) | under re-evaluation |

## Ideas we decided against

These come up regularly. If you want to propose one again, start from the linked discussion and explain what has changed.

- **Distribution-specific scheduling methods** (such as a Poisson method). One invites many more; show the pattern in an example instead ([#1890](https://github.com/mesa/mesa/issues/1890)).
- **New `AgentSet` methods that duplicate existing access**, such as tags, `count`, or a `skip` mode. Agent attributes, `select`, and `get` already cover these ([#2074](https://github.com/mesa/mesa/discussions/2074), [#3391](https://github.com/mesa/mesa/issues/3391), [#2279](https://github.com/mesa/mesa/issues/2279)).
- **Making cells agents** to get per-cell activation. Use `FixedAgent` ([#2562](https://github.com/mesa/mesa/issues/2562)).
- **Looking up agents by `unique_id` as the default way to find them.** IDs are internal; keep references to agents instead ([#3865](https://github.com/mesa/mesa/issues/3865)).
- **Coupling visualization to model data**, such as automatically using a cell's `color` attribute. Keep the model independent of how it's drawn ([#2558](https://github.com/mesa/mesa/issues/2558)).

## Open questions

- Which APIs change in Mesa 4, and which deprecations should land on the 3.x line? ([#3271](https://github.com/mesa/mesa/issues/3271), [#3330](https://github.com/mesa/mesa/issues/3330))
- What should the tutorial teach about activation, event scheduling, and time? ([#3270](https://github.com/mesa/mesa/issues/3270))
