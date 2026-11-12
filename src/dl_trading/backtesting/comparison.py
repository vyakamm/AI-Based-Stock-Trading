"""Benchmark comparison: strategy PnL versus Buy & Hold."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def compare_strategy_vs_buy_and_hold(
    equity_curve: list[tuple[pd.Timestamp, float]],
    price_data: pd.DataFrame,
    initial_capital: float,
    trade_units: int = 50,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute (and optionally plot) the strategy PnL against a Buy & Hold.

    The Buy & Hold benchmark buys ``trade_units`` at the first available
    price of ``price_data`` and holds them until the end.

    Args:
        equity_curve: List of ``(timestamp, portfolio_value)`` tuples produced
            by the backtester.
        price_data: DataFrame with an ``Adj.Close`` column, indexed by date.
        initial_capital: Starting capital of the strategy.
        trade_units: Number of units held by the Buy & Hold benchmark.
        save_path: If given, save the comparison chart to this path.

    Returns:
        DataFrame indexed by timestamp with columns ``strategy_pnl`` and
        ``buy_and_hold_pnl``.

    Raises:
        ValueError: If ``Adj.Close`` is missing from ``price_data``.
    """
    if "Adj.Close" not in price_data.columns:
        raise ValueError("Column 'Adj.Close' is required in price data.")

    dates = [t[0] for t in equity_curve]
    strategy_pnl = pd.Series([value - initial_capital for _, value in equity_curve], index=pd.DatetimeIndex(dates))

    adj_close = price_data["Adj.Close"].reindex(dates).ffill()
    initial_investment = adj_close.iloc[0] * trade_units
    buy_and_hold_pnl = adj_close * trade_units - initial_investment

    comparison = pd.DataFrame({"strategy_pnl": strategy_pnl, "buy_and_hold_pnl": buy_and_hold_pnl})

    if save_path is not None:
        import matplotlib.pyplot as plt

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(comparison.index, comparison["strategy_pnl"], label="Strategy")
        plt.plot(comparison.index, comparison["buy_and_hold_pnl"], label="Buy & Hold", alpha=0.8)
        plt.axhline(0, color="grey", linewidth=0.8)
        plt.title("PnL: strategy vs Buy & Hold")
        plt.xlabel("Date")
        plt.ylabel("PnL")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Strategy vs Buy & Hold chart saved to %s", save_path)

    return comparison
