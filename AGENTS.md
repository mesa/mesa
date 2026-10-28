# AGENTS.md

Instructions for AI coding assistants working on Mesa, a Python library for agent-based modelling. The knowledge lives in the [developer guide](docs/development/index.md); this file tells you how to behave and where to look.

## How to help a contributor

You are working for a human contributor. The goal is a contribution they understand and that fits Mesa, not the fastest possible diff.

1. **Problem before code.** Mesa works in three stages: problem → concept → implementation (see [How we work](docs/development/index.md#how-we-work)).
   - For a **feature or enhancement**, check whether there is an issue or discussion where a maintainer agreed to it. If there isn't, don't write the implementation. Help the user sharpen the problem (who is it for, what's the use case, what exists already) and draft a discussion post instead.
   - For a **bug fix**, reproduce the bug first with a minimal example, and ask the user how they ran into it in a real model. If you can't reproduce it, say so. Don't invent a bug report or a fix for behaviour you haven't observed. If it's an edge case found only by reading the code, that no realistic model would hit, tell the user it's likely to be closed rather than merged.
2. **Keep it small.** One PR solves one agreed problem. Don't bundle refactors, drive-by cleanups, or formatting changes to unrelated code.
3. **Check what was decided before.** Read [Project decisions](docs/development/decisions.md) and search existing issues, discussions, and PRs. Many ideas have been discussed, and some were deliberately declined.
4. **Name the trade-off.** If the change touches one of the [design principles](docs/development/principles.md) (core vs. examples, performance vs. readability, stability vs. experimentation, and so on), tell the user which side they're choosing and why.
5. **Help them understand; don't speak for them.** Explain the code and the reasoning so the user can defend the change in review. Don't write the main text of a PR or issue for them, including the motivation, "what I learned", and reasoning sections. Those must be in their own words; you can help improve the language of text they wrote. Remind them to mention AI use in the PR description.
6. **Verify claims.** Run the tests. Measure performance claims on a realistic workload with the same seed, before and after. Never report results you didn't produce.

## Rules that are expensive to get wrong

- Simulation behaviour must be deterministic for a given seed. Use the model's RNG (`model.rng` / `model.random`), and never let results depend on `set` iteration order.
- Keep optional dependencies optional: import them inside the code that needs them (for example NetworkX in `Network`).
- Experimental features live in `mesa.experimental` and may change. Stable APIs follow the [deprecation policy](docs/development/standards.md#deprecations-and-breaking-changes): no breaking changes without a deprecation cycle and a migration guide entry.
- Use built-in exceptions (`ValueError`, `TypeError`, …) for ordinary validation; see [Errors and exceptions](docs/development/standards.md#errors-and-exceptions).
- Core examples (`mesa/examples`) are canonical and lean. New examples go to [mesa-examples](https://github.com/mesa/mesa-examples) unless maintainers agreed otherwise.

## Commands

```bash
pip install -e ".[dev]"     # development install
pre-commit install          # ruff lint + format on commit
ruff check . --fix && ruff format .
pytest tests/               # CI runs: pytest tests/ -Werror -Wdefault::PendingDeprecationWarning
pytest --cov=mesa tests/    # check coverage didn't go down
```

Docs: `pip install -e ".[docs]"`, then `make html` in `docs/` and inspect the rendered result.

## Where to look

| Task | Read |
|---|---|
| Designing or proposing a change | [Design principles](docs/development/principles.md), [Project decisions](docs/development/decisions.md) |
| Writing code, tests, docs, or examples | [Standards](docs/development/standards.md) |
| Deprecating or changing a stable API | [Deprecations](docs/development/standards.md#deprecations-and-breaking-changes), [migration guide](docs/migration_guide.md) |
| Opening a PR | `.github/PULL_REQUEST_TEMPLATE/` (bug or feature template, fill in every section) |
| Reviewing a PR | [Reviewing contributions](docs/development/review.md) |
