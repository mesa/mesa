# Design principles and trade-offs

Mesa is for modellers with varying programming experience: researchers, educators, and students. Most design discussions come down to a handful of trade-offs. When your contribution touches one, name it, say which side you chose, and why. These are tendencies, not absolute rules.

## 1. Keep core focused on broadly useful building blocks

Everything in core is a commitment to every user: API surface, documentation, tests, and maintenance. Specialised workflows usually belong in examples, extensions, or the wider Python ecosystem.

- **We ask:** Is this a common modelling need? Could an example, extension, or existing library cover it? Would it stay useful beyond the model that motivated it?
- **In practice:** calibration, sensitivity analysis, and surrogate modelling stay outside core; examples show how to use external tools ([#1010](https://github.com/mesa/mesa/issues/1010), [#3170](https://github.com/mesa/mesa/issues/3170), [#3207](https://github.com/mesa/mesa/discussions/3207)).
- **Where it gets hard:** something specialised today can become a foundation later. Real models and community use are better evidence than generality alone.

## 2. One clear way, with visible behaviour

Prefer one obvious way to do a common task over overlapping methods or speculative options. Users should be able to tell what a method returns, what it changes, and roughly what it costs. Defaults are good when they remove real burden without hiding behaviour that changes what a model means.

- **We ask:** Is there already a way to do this? Can users infer the behaviour? Does this option serve a real need or a hypothetical one?
- **In practice:** no `count` method when `len(agentset.select(...))` already works ([#3391](https://github.com/mesa/mesa/issues/3391)); no `skip` mode for `AgentSet.get` without concrete use cases ([#2279](https://github.com/mesa/mesa/issues/2279)); explicit styling with sensible defaults instead of silently mapping a `color` attribute in visualization ([#2668](https://github.com/mesa/mesa/issues/2668), [#2558](https://github.com/mesa/mesa/issues/2558)).
- **Where it gets hard:** convenience versus explicitness. If hidden behaviour would affect what a model means, make it explicit.

## 3. Components compose; they don't depend on each other

Components should work together without one owning another's responsibilities, so a model can use, replace, or leave out each of them.

- **We ask:** Can a model use this without adopting the rest? Does it need to know how another component is set up?
- **In practice:** grids don't require a model ([#1116](https://github.com/mesa/mesa/issues/1116)); visualization doesn't assume how a model initialises or collects data ([#1213](https://github.com/mesa/mesa/issues/1213)); signals are an optional way to connect components ([#2972](https://github.com/mesa/mesa/discussions/2972)).
- **Where it gets hard:** independence can mean some extra integration code. We accept that over forcing assumptions on models that don't need a feature.

## 4. Teachable as well as flexible

The common path should be easy to learn and explain. Advanced users should be able to extend Mesa without Mesa prescribing their model. Core examples are part of the learning path: canonical, focused, and lean.

- **We ask:** Is the common task easy to explain? Does this flexibility make the first steps harder? Would an example help more people than a new feature?
- **In practice:** readable code over a modest speedup ([#1711](https://github.com/mesa/mesa/issues/1711)); simple canonical implementations in core examples, alternatives in mesa-examples ([#3286](https://github.com/mesa/mesa/issues/3286)); an example should demonstrate what it claims to ([#3040](https://github.com/mesa/mesa/issues/3040)).

## 5. Performance needs evidence, weighed against complexity

Performance matters, but a plausible optimisation or a faster microbenchmark isn't enough on its own. Measure realistic workloads, compare like with like, and weigh the absolute time saved against the maintenance cost.

- **We ask:** Is the benchmark representative? Were both versions measured under the same conditions and seed? Is the gain meaningful for a whole model, and worth the extra complexity?
- **In practice:** demonstrate gains on realistic usage ([#1476](https://github.com/mesa/mesa/issues/1476)); prefer small, separately measured changes over broad refactors with uncertain gains ([#3388](https://github.com/mesa/mesa/issues/3388), [#3619](https://github.com/mesa/mesa/issues/3619)); CI benchmarks are useful but noisy ([#1931](https://github.com/mesa/mesa/discussions/1931)).
- **Where it gets hard:** a small cost in a hot path can matter, while a big percentage gain may barely change a full run. Say which level your evidence measures.

## 6. Predictable and reproducible

Modellers need behaviour they can understand, control, and reproduce. Don't silently guess intent, hide errors, or let results depend on incidental ordering.

- **We ask:** Could a user misread what this does? Is the error visible? Could iteration order, randomness, or a hidden default change results?
- **In practice:** deterministic selection instead of relying on set order ([#3088](https://github.com/mesa/mesa/issues/3088)); a missing agent type raises an error instead of returning an empty `AgentSet` ([#3608](https://github.com/mesa/mesa/issues/3608)); explicit RNG for components that need randomness ([#2323](https://github.com/mesa/mesa/issues/2323)).
- **Where it gets hard:** validation costs speed and code too. Add checks where they prevent consequential mistakes, not where the operation already fails clearly ([#3545](https://github.com/mesa/mesa/issues/3545)).

## 7. Stable and experimental are different promises

Stable APIs get deliberate compatibility and migration care. Experimental features are for learning from real use. That label doesn't make an unclear design or unlimited breakage acceptable.

- **We ask:** Do users already rely on this? Is the boundary between stable and experimental clear? What's the migration path?
- **In practice:** experimental APIs may change, stable ones follow semantic versioning ([#2675](https://github.com/mesa/mesa/issues/2675)); minor releases shouldn't break models ([#1917](https://github.com/mesa/mesa/issues/1917)); a major release is not by itself a migration plan ([#2026](https://github.com/mesa/mesa/issues/2026)). See the [deprecation policy](standards.md).

## Still contested

- Where the boundaries lie between core, its modules, and the wider ecosystem ([#1295](https://github.com/mesa/mesa/discussions/1295), [#2330](https://github.com/mesa/mesa/discussions/2330), [#2954](https://github.com/mesa/mesa/discussions/2954)).
- How much complexity a performance gain justifies; the answer depends on the workload ([#3388](https://github.com/mesa/mesa/issues/3388), [#3932](https://github.com/mesa/mesa/issues/3932)).
