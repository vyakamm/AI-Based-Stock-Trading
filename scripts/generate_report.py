"""Regenerate the README benchmark table and charts.

Runs every strategy shipped with pre-trained artifacts on the same AMZN test
window, computes the Buy & Hold benchmark, and writes:

- ``outputs/metrics_summary.csv``: one row per strategy with the main metrics,
- ``outputs/results.json``: full metric dictionaries,
- ``docs/assets/equity_comparison.png``: equity curves vs Buy & Hold (hero image),
- ``docs/assets/lstm_equity_curve.png``, ``lstm_monthly_heatmap.png``,
  ``lstm_pnl_distribution.png``: detailed LSTM analytics.

Usage (from the repository root)::

    python scripts/generate_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from dl_trading.backtesting.backtester import Backtester, BacktesterParameters
from dl_trading.data.loader import load_local_csv
from dl_trading.pipeline import build_strategy, ensure_rsi_defaults, load_or_default_hyperparams

REPO_ROOT = Path(__file__).resolve().parents[1]
SYMBOL = "AMZN"
TRAIN_RATIO = 0.8
INITIAL_CAPITAL = 15_000.0
TRADE_UNITS = 50

# Strategies shipped with pre-trained artifacts, in display order.
STRATEGIES = ["lstm", "randomforest", "xgboost", "svm", "knn", "ma", "rsi"]

DISPLAY_NAMES = {
    "lstm": "LSTM (deep learning)",
    "randomforest": "Random Forest",
    "xgboost": "XGBoost",
    "svm": "SVM",
    "knn": "KNN",
    "ma": "MA crossover",
    "rsi": "RSI",
}


def run_strategy(name: str, test_data: pd.DataFrame, output_dir: Path) -> tuple[dict, pd.Series]:
    """Backtest one strategy on ``test_data`` and return (metrics, equity curve)."""
    try:
        hyperparams = load_or_default_hyperparams(SYMBOL, name, REPO_ROOT / "artifacts")
    except FileNotFoundError:
        # RSI ships without tuned artifacts: use sensible defaults.
        hyperparams = {}
    if name == "rsi":
        hyperparams = ensure_rsi_defaults(hyperparams)
        hyperparams.setdefault("stop_loss", 0.03)

    strategy = build_strategy(name, SYMBOL, hyperparams, REPO_ROOT / "artifacts")

    params = BacktesterParameters(
        stop_loss=float(hyperparams["stop_loss"]),
        past_history=int(hyperparams["past_history"]),
        initial_capital=INITIAL_CAPITAL,
        trade_units=TRADE_UNITS,
    )
    backtester = Backtester(params, strategy_name=name, symbol=SYMBOL, enable_max_drawdown=False)
    backtester.run_backtest(test_data, strategy)
    results = backtester.results()
    analyzer = backtester.performance()
    metrics = results.metrics
    equity = results.equity_curve["equity"]

    prefix = f"{SYMBOL}_{name}"
    analyzer.plot_equity_curve(
        title=f"{SYMBOL} - {DISPLAY_NAMES[name]} equity curve",
        save_path=output_dir / f"{prefix}_equity_curve.png",
    )
    if name == "lstm":
        analyzer.plot_monthly_returns_heatmap(save_path=REPO_ROOT / "docs/assets/lstm_monthly_heatmap.png")
        analyzer.plot_trades_distribution(save_path=REPO_ROOT / "docs/assets/lstm_pnl_distribution.png")
        analyzer.plot_equity_curve(
            title=f"{SYMBOL} - LSTM equity curve and drawdown",
            save_path=REPO_ROOT / "docs/assets/lstm_equity_curve.png",
        )
    return metrics, equity


def buy_and_hold_equity(test_data: pd.DataFrame) -> pd.Series:
    """Equity curve of buying TRADE_UNITS at the first test close and holding."""
    price = test_data["Adj.Close"]
    initial_investment = price.iloc[0] * TRADE_UNITS
    return price * TRADE_UNITS - initial_investment + INITIAL_CAPITAL


def main() -> int:
    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    (REPO_ROOT / "docs/assets").mkdir(parents=True, exist_ok=True)

    full_data = load_local_csv(REPO_ROOT / "data" / "AMZN.csv")
    split_index = int(len(full_data) * TRAIN_RATIO)
    test_data = full_data.iloc[split_index:]

    all_metrics: dict[str, dict] = {}
    equity_curves: dict[str, pd.Series] = {}

    for name in STRATEGIES:
        print(f"Backtesting {name} ...", flush=True)
        metrics, equity = run_strategy(name, test_data, output_dir)
        all_metrics[name] = metrics
        equity_curves[name] = equity

    # Buy & Hold benchmark
    bh_equity = buy_and_hold_equity(test_data)
    bh_return = (bh_equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    bh_daily = bh_equity.pct_change().dropna()
    all_metrics["buy_and_hold"] = {
        "total_return_pct": round(bh_return, 4),
        "annualized_return_pct": round(((bh_equity.iloc[-1] / INITIAL_CAPITAL) ** (252 / len(bh_equity)) - 1) * 100, 4),
        "volatility_pct": round(bh_daily.std() * (252**0.5) * 100, 4),
        "sharpe_ratio": round(bh_daily.mean() / bh_daily.std() * (252**0.5), 4),
        "n_trades": 1,
    }

    # ----- Summary CSV -----
    rows = []
    for name, metrics in all_metrics.items():
        rows.append(
            {
                "strategy": DISPLAY_NAMES.get(name, "Buy & Hold"),
                "total_return_%": metrics.get("total_return_pct"),
                "annualized_return_%": metrics.get("annualized_return_pct"),
                "volatility_%": metrics.get("volatility_pct"),
                "sharpe": metrics.get("sharpe_ratio"),
                "sortino": metrics.get("sortino_ratio"),
                "max_drawdown_%": metrics.get("max_drawdown_pct"),
                "win_rate_%": metrics.get("win_rate_pct"),
                "profit_factor": metrics.get("profit_factor"),
                "trades": int(metrics.get("n_trades", 0)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    (output_dir / "results.json").write_text(json.dumps(all_metrics, indent=2, default=str))

    # ----- Hero chart -----
    plt.figure(figsize=(12, 6))
    colors = {"lstm": "#d62728", "ma": "#1f77b4", "rsi": "#2ca02c"}
    for name in ("lstm", "ma", "rsi"):
        if name in equity_curves:
            plt.plot(equity_curves[name].index, equity_curves[name], label=DISPLAY_NAMES[name], color=colors.get(name))
    plt.plot(bh_equity.index, bh_equity, label="Buy & Hold", color="grey", linestyle="--", alpha=0.8)
    plt.axhline(INITIAL_CAPITAL, color="black", linewidth=0.7, alpha=0.4)
    plt.title(f"{SYMBOL} - out-of-sample equity curves ({test_data.index[0]:%Y-%m} to {test_data.index[-1]:%Y-%m})")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(REPO_ROOT / "docs/assets/equity_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(summary.to_string(index=False))
    print("\nReport written to outputs/ and docs/assets/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
