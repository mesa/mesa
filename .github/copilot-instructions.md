# Copilot instructions for Mesa

## Big picture (what to understand first)
- Mesa’s public entrypoints are re-exported from `mesa/__init__.py`; keep namespace compatibility (`mesa.Model`, `mesa.Agent`, `mesa.batch_run`, `mesa.DataCollector`).
- Core runtime is event-driven in `mesa/model.py` + `mesa/time/events.py`: `Model.step()` is wrapped into a default recurring event (`Schedule(interval=1.0)`), and time advances via `_advance_time`.
- Agent lifecycle is automatic: `Agent.__init__` registers with `Model.register_agent`, `Agent.remove()` deregisters (`mesa/agent.py`). Avoid manual mutation of `model.agents` / `model.agents_by_type` internals.
- Spatial APIs have two tracks: `mesa.space` is supported but maintenance-only; new discrete-space work belongs in `mesa/discrete_space/*`.
- Visualization is Solara-based (`mesa/visualization/solara_viz.py`) and considered experimental in 3.x (`mesa/visualization/__init__.py`).

## Key developer workflows
- Install for development: `pip install -e ".[dev]"` (or CI-style: `uv pip install --system .[dev]`).
- Main test command (matches CI intent): `pytest --durations=10 tests/ -Werror -Wdefault::PendingDeprecationWarning`.
- Coverage run used in CI: `python -Im pytest -p pytest_cov --cov tests/`.
- Lint/format uses Ruff (`pyproject.toml`): run `ruff check . --fix` (formatter/lint integrated via Ruff config).
- Visualization tests require Playwright browser setup in CI (`playwright install chromium-headless-shell`).
- Docs build from `docs/`: `make html`.

## Project-specific coding patterns
- Prefer `rng` over `seed`; `seed` paths are deprecated in `Model` and `batch_run`.
- For repeated runs, prefer explicit `rng` lists in `batch_run`; `iterations` is deprecated (`mesa/batchrunner.py`, `docs/migration_guide.md`).
- In event scheduling, do not use lambdas for callbacks: `Event` stores weakrefs and lambda callables can disappear (`mesa/time/events.py`).
- In `DataCollector`, lambdas work but avoid them if models must be pickleable (`mesa/datacollection.py`).
- If changing model stepping/time semantics, verify `run_for`, `run_until`, `schedule_event`, and `schedule_recurring` behavior together (same subsystem).
- Preserve reserved model/agent fields from migration guide (`agents`, `random`, `time`, `steps`, `running`, `unique_id`, etc.).

## Integration points and boundaries
- `batch_run` expects models to expose `datacollector`; it reads internal collection history (`_collection_steps`) for time-aware sampling.
- Optional dependency boundaries are explicit in extras: `network`, `viz`, `examples`, `docs`, `dev` (`pyproject.toml`). Keep imports lazy/guarded where extras may be absent.
- Core examples live under `mesa/examples` and are CI-validated; use them as behavior references before introducing API changes.

## When making changes
- Keep backward compatibility where practical, and document user-facing API shifts in `docs/migration_guide.md`.
- Add or update tests in `tests/` adjacent to the changed subsystem (core, discrete_space, visualization, experimental).
- Avoid broad refactors across `space` and `discrete_space` in one PR unless migration rationale is explicit.
