# Code, test, and documentation standards

Conventions that apply across Mesa. When in doubt, follow the surrounding code.

## Style and tooling

- PEP 8 and Google-style docstrings, enforced by Ruff through pre-commit (`pre-commit install`). Configuration lives in `pyproject.toml`.
- Type hints use `X | Y` unions. New generic code can use PEP 695 syntax (Mesa requires Python 3.12+).

## API design

- Prefer clear names over short ones, and one way to do a thing over overlapping methods ([#1953](https://github.com/mesa/mesa/issues/1953), [#3391](https://github.com/mesa/mesa/issues/3391)).
- Document public APIs; keep internals private with a leading underscore ([#1942](https://github.com/mesa/mesa/issues/1942)).
- Use keyword arguments where they make calls clearer, such as `fields=[...]`, but don't make everything keyword-only by default ([#3156](https://github.com/mesa/mesa/issues/3156)).

## Errors and exceptions

- Never raise a bare `Exception`. Use the most fitting built-in exception (`ValueError`, `TypeError`, …) for ordinary validation.
- Use a `MesaException` subclass only when it adds domain context or hides internals. Pick the most specific existing one, and check with maintainers before adding a new one.
- Wrap internal exceptions with `raise ... from ...` to keep the original cause.
- Make messages actionable: say what went wrong and how to fix it.
- When you change error behaviour, test the exception type and message.

## Determinism and randomness

- Use the model's RNG (`model.rng`, `model.random`). Pass it explicitly to components that don't have the model ([#2541](https://github.com/mesa/mesa/issues/2541), [#2978](https://github.com/mesa/mesa/issues/2978)).
- Never let results depend on `set` iteration or other unordered sources; sort or use an ordered structure ([#3088](https://github.com/mesa/mesa/issues/3088)).
- Models accept `rng` (a seed or a NumPy `Generator`, directly or through a `Scenario`) and pass it to `Model`, which seeds both `model.rng` and `model.random`. Examples and tutorials expose it so readers can reproduce a run.
- Test randomized behaviour with fixed seeds, and assert what matters rather than incidental output.

## Performance

- Profile first. Measure elapsed time on realistic workloads before and after, with the same seed ([#1476](https://github.com/mesa/mesa/issues/1476), [#3388](https://github.com/mesa/mesa/issues/3388)).
- Report absolute and relative changes, and treat benchmarks on shared CI runners as noisy.
- The benchmark suite lives in `benchmarks/`. Maintainers can run it on a PR with the `trigger-benchmarks` label.

## Dependencies

- Keep specialised dependencies optional (extras such as `network` and `viz`), and import them inside the code that needs them, like NetworkX in `Network` ([#3598](https://github.com/mesa/mesa/issues/3598)).
- Set minimum versions only when Mesa needs a newer feature; never pin exact versions ([#1852](https://github.com/mesa/mesa/issues/1852)).

## Tests

- Every behaviour change comes with tests, in the test module for the changed source module (`tests/` mirrors `mesa/`).
- Extend existing tests instead of duplicating coverage, and don't let coverage go down.
- Use pytest. Give tests a short docstring, and keep fixtures local unless other modules need them.

## Documentation

- Docstrings start with a one-line summary. Reference docs cover public APIs; tutorials explain how to use Mesa.
- Docs are Markdown (MyST). Build them and check the rendered result; don't commit build output.
- Clear notebook outputs before committing. When adding or renaming a tutorial, update listings, links, and Colab/Binder badges ([#3280](https://github.com/mesa/mesa/issues/3280)).
- Incompatible changes get a [migration guide](../migration_guide) entry; new features go in the release notes.

## Examples

Core examples in `mesa/examples` are classic, canonical models. They are tested in CI, updated with each release, and used in the docs and benchmarks. *Basic* examples use only stable features; *advanced* examples may use experimental ones. New core examples need agreement in a discussion first, and should demonstrate something the existing ones don't.

Community models go to [mesa-examples](https://github.com/mesa/mesa-examples), which accepts contributions liberally: include a README, an environment file, and code that runs on at least one Mesa major version. Improvements to existing examples are welcome in both repositories. Background: [#2330](https://github.com/mesa/mesa/discussions/2330), [#2349](https://github.com/mesa/mesa/issues/2349).

## Experimental code

Experimental features live in `mesa.experimental` and are clearly marked. They can change or be removed in any release, as long as the release notes say so ([#2675](https://github.com/mesa/mesa/issues/2675)). Experimental status is for learning from real use, not for skipping agreement on a coherent design.

## Deprecations and breaking changes

Mesa follows semantic versioning. To replace a stable feature:

1. **Introduce the alternative.** Optionally add a `PendingDeprecationWarning` to the old feature.
2. **Complete the prerequisites.** Update docs and tutorials, add a migration guide entry, and move all examples to the new API.
3. **Deprecate.** Once the alternative has been available for at least one minor release, add a `FutureWarning` that names the replacement and links the migration guide:
   ```python
   warnings.warn(
       "old_feature() is deprecated and will be removed in Mesa X.0. "
       "Use new_feature() instead. "
       "See: https://mesa.readthedocs.io/latest/migration_guide.html#section",
       FutureWarning,
       stacklevel=2,
   )
   ```
4. **Remove** the old feature in the next major release.

| Action | Allowed in |
|---|---|
| Alternative + `PendingDeprecationWarning` | Any release |
| `FutureWarning` | Minor release (patch only with a compelling reason) |
| Removal | Major release only |

For `mesa.experimental`, steps 1–3 can happen together and removal can happen in any release. Still mention the change in the release notes.

Migration guide entries go at the top (newest first), with a clear heading, what changed and why, before/after code, and a link to the PR.
