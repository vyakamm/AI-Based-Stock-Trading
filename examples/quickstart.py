"""Quickstart: backtest the pre-trained LSTM and the MA crossover on AMZN.

Run from the repository root::

    python examples/quickstart.py

This script demonstrates the programmatic API (no CLI required):
load data -> build strategy -> backtest -> read metrics.
"""

from __future__ import annotations

from pathlib import Path

from dl_trading.backtesting.backtester import Backtester, BacktesterParameters
from dl_trading.backtesting.comparison import compare_strategy_vs_buy_and_hold
from dl_trading.data.loader import download_and_split_data
from dl_trading.pipeline import build_strategy, load_or_default_hyperparams

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # 1. Load the bundled AMZN dataset with a chronological 80/20 split.
    train_data, test_data = download_and_split_data(
        symbol="AMZN", train_ratio=0.8, local_path=REPO_ROOT / "data" / "AMZN.csv"
    )
    print(f"Train: {len(train_data)} days | Test: {len(test_data)} days")

    for strategy_name in ("lstm", "ma"):
        # 2. Load the saved hyperparameters for this (symbol, strategy) pair.
        hyperparams = load_or_default_hyperparams("AMZN", strategy_name, artifacts_dir=REPO_ROOT / "artifacts")

        # 3. Build the strategy (loads the trained model for ML/DL strategies).
        strategy = build_strategy(strategy_name, "AMZN", hyperparams, artifacts_dir=REPO_ROOT / "artifacts")

        # 4. Backtest it on the strictly out-of-sample window.
        backtester = Backtester(
            BacktesterParameters(
                stop_loss=float(hyperparams["stop_loss"]),
                past_history=int(hyperparams["past_history"]),
                initial_capital=15_000.0,
                trade_units=50,
            ),
            strategy_name=strategy_name,
            symbol="AMZN",
            enable_max_drawdown=False,
        )
        backtester.run_backtest(test_data, strategy)
        metrics = backtester.results().metrics

        print(
            f"\n[{strategy_name.upper()}] "
            f"return={metrics['total_return_pct']:.2f}% "
            f"sharpe={metrics['sharpe_ratio']:.2f} "
            f"maxDD={metrics['max_drawdown_pct']:.2f}% "
            f"win_rate={metrics['win_rate_pct']:.1f}% "
            f"trades={metrics['n_trades']:.0f}"
        )

        # 5. Optional: compare against Buy & Hold and save the chart.
        comparison = compare_strategy_vs_buy_and_hold(
            backtester.equity_curve,
            test_data,
            initial_capital=15_000.0,
            trade_units=50,
            save_path=REPO_ROOT / "outputs" / f"quickstart_{strategy_name}_vs_buy_and_hold.png",
        )
        edge = comparison["strategy_pnl"].iloc[-1] - comparison["buy_and_hold_pnl"].iloc[-1]
        print(f"[{strategy_name.upper()}] PnL edge vs Buy & Hold: {edge:,.2f}")


if __name__ == "__main__":
    main()
