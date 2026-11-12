"""Command line interface of the ``dl-trading`` package.

Three subcommands are exposed:

- ``backtest``: replay a registered strategy on historical data and print a
  performance report (optionally saving plots and results),
- ``train``: fit a model for an ML/DL strategy and persist it with its
  hyperparameters,
- ``optimize``: search the hyperparameter space with Optuna and persist the
  best artifacts.

Examples::

    dl-trading backtest --symbol AMZN --strategy lstm
    dl-trading backtest --symbol AMZN --strategy ma --use-yfinance --start-date 2021-01-01
    dl-trading train --symbol AMZN --strategy randomforest --past-history 20 --n-estimators 200
    dl-trading optimize --symbol AMZN --strategy ma --trials 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dl_trading import __version__
from dl_trading.backtesting.backtester import Backtester, BacktesterParameters
from dl_trading.backtesting.comparison import compare_strategy_vs_buy_and_hold
from dl_trading.data.loader import download_and_split_data
from dl_trading.optimization.optimizer import StrategyOptimizer
from dl_trading.pipeline import (
    build_strategy,
    ensure_rsi_defaults,
    load_or_default_hyperparams,
    save_artifacts,
    train_model,
)
from dl_trading.strategies.registry import STRATEGY_REGISTRY, list_strategies

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------
def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the common data-loading arguments to a subcommand parser."""
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data",
        default=None,
        help="Path to a local OHLCV CSV file (default: data/<SYMBOL>.csv).",
    )
    data_group.add_argument(
        "--use-yfinance",
        action="store_true",
        help="Download data from Yahoo Finance instead of reading a local file.",
    )
    parser.add_argument("--start-date", default="2020-01-01", help="Download start date (YYYY-MM-DD).")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of the history used for training (default: 0.8).",
    )


def _load_data(args: argparse.Namespace) -> tuple:
    """Load and split market data according to the CLI arguments."""
    if args.use_yfinance:
        return download_and_split_data(
            symbol=args.symbol,
            start_date=args.start_date,
            train_ratio=args.train_ratio,
            use_yfinance=True,
        )
    local_path = args.data or f"data/{args.symbol}.csv"
    return download_and_split_data(
        symbol=args.symbol,
        train_ratio=args.train_ratio,
        use_yfinance=False,
        local_path=local_path,
    )


def _resolve_strategy_name(raw: str) -> str:
    """Case-insensitive lookup of a strategy name in the registry."""
    name = raw.lower()
    if name not in STRATEGY_REGISTRY:
        raise SystemExit(f"Unknown strategy '{raw}'. Available strategies: {', '.join(list_strategies())}")
    return name


def _setup_logging(verbose: bool) -> None:
    """Configure root logging once."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Subcommand: backtest
# ---------------------------------------------------------------------------
def cmd_backtest(args: argparse.Namespace) -> int:
    """Run a backtest and print the report."""
    strategy_name = _resolve_strategy_name(args.strategy)

    train_data, test_data = _load_data(args)
    hyperparams = load_or_default_hyperparams(args.symbol, strategy_name, args.artifacts_dir)
    if strategy_name == "rsi":
        hyperparams = ensure_rsi_defaults(hyperparams)

    strategy = build_strategy(strategy_name, args.symbol, hyperparams, args.artifacts_dir)

    backtest_params = BacktesterParameters(
        stop_loss=float(hyperparams["stop_loss"]),
        past_history=int(hyperparams["past_history"]),
        initial_capital=args.initial_capital,
        trade_units=args.trade_units,
    )
    backtester = Backtester(
        backtest_params,
        strategy_name=strategy_name,
        symbol=args.symbol,
        enable_max_drawdown=not args.no_drawdown_record,
        max_drawdown_dir=str(Path(args.output_dir) / "max_drawdown"),
    )

    start_date = test_data.index[hyperparams["past_history"]].strftime("%Y-%m-%d")
    end_date = test_data.index[-1].strftime("%Y-%m-%d")
    print(f"\nTest period: {start_date} -> {end_date} ({len(test_data) - hyperparams['past_history']} trading days)\n")

    results = backtester.run_backtest(test_data, strategy)
    metrics = backtester.print_report()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_plots:
        analyzer = backtester.performance()
        prefix = f"{args.symbol}_{strategy_name}"
        analyzer.plot_equity_curve(
            title=f"{args.symbol} - {strategy_name.upper()} equity curve",
            save_path=output_dir / f"{prefix}_equity_curve.png",
        )
        analyzer.plot_monthly_returns_heatmap(save_path=output_dir / f"{prefix}_monthly_heatmap.png")
        analyzer.plot_trades_distribution(save_path=output_dir / f"{prefix}_pnl_distribution.png")

        comparison = compare_strategy_vs_buy_and_hold(
            backtester.equity_curve,
            test_data,
            initial_capital=backtest_params.initial_capital,
            trade_units=backtest_params.trade_units,
            save_path=output_dir / f"{prefix}_vs_buy_and_hold.png",
        )
        final_edge = comparison["strategy_pnl"].iloc[-1] - comparison["buy_and_hold_pnl"].iloc[-1]
        print(f"PnL edge vs Buy & Hold : {final_edge:>11,.2f}")

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbol": args.symbol,
            "strategy": strategy_name,
            "hyperparameters": dict(hyperparams.items()),
            "metrics": metrics,
            "total_transaction_costs": results.total_transaction_costs,
        }
        json_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"Results written to     : {json_path}")

    print(f"Outputs saved under    : {output_dir}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    """Train a model for an ML/DL strategy and persist the artifacts."""
    strategy_name = _resolve_strategy_name(args.strategy)
    strategy_type = STRATEGY_REGISTRY[strategy_name]["type"]

    if strategy_type.name == "TECHNICAL_INDICATOR":
        raise SystemExit(
            f"Strategy '{strategy_name}' is rule-based and needs no training. "
            "Use `dl-trading optimize` to search its best parameters."
        )

    train_data, test_data = _load_data(args)

    overrides = {
        "past_history": args.past_history,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "stack_depth": args.stack_depth,
        "number_units_per_layer": args.number_units_per_layer,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "stop_loss": args.stop_loss,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    # Start from existing hyperparameters when available, then apply overrides.
    try:
        hyperparams = load_or_default_hyperparams(args.symbol, strategy_name, args.artifacts_dir)
    except FileNotFoundError:
        defaults = {
            "lstm": {
                "past_history": 60,
                "stack_depth": 2,
                "number_units_per_layer": 64,
                "epochs": 30,
                "batch_size": 32,
            },
            "randomforest": {"past_history": 20, "n_estimators": 200, "max_depth": 8},
            "xgboost": {"past_history": 20, "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
            "svm": {"past_history": 20, "C": 1.0, "kernel": "rbf"},
            "knn": {"past_history": 20, "n_neighbors": 5, "weights": "uniform"},
        }[strategy_name]
        hyperparams = dict(defaults)

    hyperparams.update(overrides)
    hyperparams.setdefault("stop_loss", 0.03)

    print(f"\nTraining {strategy_name} for {args.symbol} with: {hyperparams}\n")
    model, hyperparams = train_model(strategy_name, hyperparams, train_data, test_data)
    paths = save_artifacts(args.symbol, strategy_name, hyperparams, model, args.artifacts_dir)

    print(f"\nArtifacts saved:\n  - {paths.hyperparams}\n  - {paths.model}")
    print("\nNext step: dl-trading backtest --symbol", args.symbol, "--strategy", strategy_name)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: optimize
# ---------------------------------------------------------------------------
def cmd_optimize(args: argparse.Namespace) -> int:
    """Run the Optuna hyperparameter search and persist the best artifacts."""
    strategy_name = _resolve_strategy_name(args.strategy)
    train_data, test_data = _load_data(args)

    optimizer = StrategyOptimizer(
        strategy_name=strategy_name,
        symbol=args.symbol,
        train_data=train_data,
        test_data=test_data,
        artifacts_dir=args.artifacts_dir,
    )
    optimizer.optimize(n_trials=args.trials, n_jobs=args.jobs, seed=args.seed)
    return 0


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="dl-trading",
        description="Backtesting of deep learning based stock trading strategies.",
    )
    parser.add_argument("--version", action="version", version=f"dl-trading {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    def common_arguments(sub: argparse.ArgumentParser, with_backtest_options: bool) -> None:
        sub.add_argument("--symbol", default="AMZN", help="Ticker symbol (default: AMZN).")
        sub.add_argument("--strategy", required=True, help=f"One of: {', '.join(list_strategies())}.")
        sub.add_argument("--artifacts-dir", default="artifacts", help="Root of the artifacts directory.")
        sub.add_argument("--verbose", action="store_true", help="Enable debug logging.")
        _add_data_arguments(sub)
        if with_backtest_options:
            sub.add_argument("--initial-capital", type=float, default=15_000.0, help="Starting capital.")
            sub.add_argument("--trade-units", type=int, default=50, help="Units traded per position.")
            sub.add_argument("--output-dir", default="outputs", help="Directory for plots and records.")
            sub.add_argument("--no-plots", action="store_true", help="Skip chart generation.")
            sub.add_argument("--no-drawdown-record", action="store_true", help="Do not persist max drawdown records.")
            sub.add_argument("--output-json", default=None, help="Write the metrics to this JSON file.")

    backtest_parser = subparsers.add_parser("backtest", help="Run a backtest and print a report.")
    common_arguments(backtest_parser, with_backtest_options=True)
    backtest_parser.set_defaults(func=cmd_backtest)

    train_parser = subparsers.add_parser("train", help="Train and persist a model for an ML/DL strategy.")
    common_arguments(train_parser, with_backtest_options=False)
    train_parser.add_argument("--past-history", type=int, default=None)
    train_parser.add_argument("--epochs", type=int, default=None, help="[lstm]")
    train_parser.add_argument("--batch-size", type=int, default=None, help="[lstm]")
    train_parser.add_argument("--stack-depth", type=int, default=None, help="[lstm]")
    train_parser.add_argument("--number-units-per-layer", type=int, default=None, help="[lstm]")
    train_parser.add_argument("--n-estimators", type=int, default=None, help="[randomforest, xgboost]")
    train_parser.add_argument("--max-depth", type=int, default=None, help="[randomforest, xgboost]")
    train_parser.add_argument("--stop-loss", type=float, default=None)
    train_parser.set_defaults(func=cmd_train)

    optimize_parser = subparsers.add_parser("optimize", help="Optuna hyperparameter search.")
    common_arguments(optimize_parser, with_backtest_options=False)
    optimize_parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials.")
    optimize_parser.add_argument("--jobs", type=int, default=1, help="Parallel trials (use 1 for LSTM).")
    optimize_parser.add_argument("--seed", type=int, default=42, help="Sampler seed.")
    optimize_parser.set_defaults(func=cmd_optimize)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
