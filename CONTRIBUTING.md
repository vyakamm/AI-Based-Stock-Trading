# Contributing

Thanks for your interest in improving this project!

## Development setup

```bash
git clone https://github.com/vyakamm/AI-Based-Stock-Trading.git
cd AI-Based-Stock-Trading
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,tensorflow-cpu,xgboost,yfinance]"
```

## Before opening a pull request

```bash
ruff check src tests             # no lint errors
ruff format --check src tests    # formatting is enforced
pytest                           # all tests must pass (CI runs 3.10-3.12)
```

## Conventions

- **Code style**: ruff (line length 120, `E/W/F/I/B/C4/UP` rules).
- **Type hints** on every public function; Google-style docstrings with
  `Args:` / `Returns:` / `Raises:` sections.
- **No hardcoded paths**: artifacts, data and outputs flow through
  parameters with sane defaults.
- **Tests**: any bug fix or new feature comes with a regression test. Model
  tests use tiny configurations so the suite stays under ~15 s.
- **Artifacts**: never commit new `artifacts/` binaries without explaining
  how they were produced (command + dataset + seed).

## Adding a new strategy

1. Implement a `Strategy` subclass in
   `src/dl_trading/strategies/base.py` (or a new module).
2. Register it in `src/dl_trading/strategies/registry.py` with an Optuna
   sampler that always returns `past_history`.
3. If it needs a model, add a branch to `dl_trading.pipeline.build_model`.
4. Add unit tests (signals for edge cases + registry resolution).
5. Re-run `python scripts/generate_report.py` if you want it in the README
   benchmark.

## Reporting bugs

Open an issue with: the exact CLI command, the full traceback, your Python
version and OS. Reproducible reports get fixed faster.
