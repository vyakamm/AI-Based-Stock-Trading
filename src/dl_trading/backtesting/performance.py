"""Performance analytics for backtest results.

:class:`PerformanceAnalyzer` consumes the trade list and the equity curve
produced by the :class:`~dl_trading.backtesting.backtester.Backtester` and
exposes the standard performance metrics of systematic strategies:

- returns: total, annualized, volatility,
- risk-adjusted ratios: Sharpe, Sortino, Calmar, recovery factor,
- trade statistics: win rate, profit factor, payoff ratio, streaks,
- risk measures: max drawdown, VaR (daily and monthly), risk of ruin,
- calendar analytics: monthly returns, share of positive months.

All figures are also available at once through :meth:`PerformanceAnalyzer.summary`,
which is what the CLI and the tests rely on.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EquityInput = list[tuple[pd.Timestamp, float]] | pd.DataFrame


def save_max_drawdown(
    strategy_name: str,
    symbol: str,
    new_drawdown: float,
    directory: str | Path = "outputs/max_drawdown",
    verbose: bool = True,
) -> bool:
    """Persist the deepest drawdown ever observed for a (symbol, strategy) pair.

    The value is stored as plain text in
    ``{directory}/maxdrawdown_{symbol}_{strategy}.txt`` and only overwritten
    when the new drawdown is deeper (in absolute value).

    Args:
        strategy_name: Name of the strategy.
        symbol: Ticker symbol.
        new_drawdown: Drawdown to compare (monetary value, <= 0).
        directory: Output directory (created if missing).
        verbose: Log the outcome at INFO level.

    Returns:
        True if ``new_drawdown`` is a new record and was written, else False.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / f"maxdrawdown_{symbol}_{strategy_name}.txt"

    old_drawdown: float | None = None
    if filepath.is_file():
        try:
            content = filepath.read_text().strip()
            if content:
                old_drawdown = float(content)
        except ValueError:
            if verbose:
                logger.warning(
                    "Could not parse previous max drawdown file %s; it may be corrupted.",
                    filepath,
                )

    if old_drawdown is None or abs(new_drawdown) > abs(old_drawdown):
        filepath.write_text(f"{new_drawdown}")
        if verbose:
            logger.info(
                "New max drawdown record for %s/%s: %.2f (previous: %s)",
                symbol,
                strategy_name,
                new_drawdown,
                old_drawdown,
            )
        return True

    if verbose:
        logger.info(
            "No new max drawdown record for %s/%s: current %.2f, record %.2f",
            symbol,
            strategy_name,
            new_drawdown,
            old_drawdown,
        )
    return False


class PerformanceAnalyzer:
    """Compute and visualize performance metrics of a backtest.

    Args:
        trades: List of trade dictionaries as produced by the backtester
            (each closed trade carries a ``pnl`` key).
        equity_curve: Either a list of ``(timestamp, portfolio_value)`` tuples
            or an already built DataFrame with an ``equity`` column.
        initial_capital: Starting capital of the backtest.
    """

    def __init__(self, trades: list[dict], equity_curve: EquityInput, initial_capital: float) -> None:
        self.trades = trades
        self.equity_curve = self._prepare_equity_curve(equity_curve)
        self.initial_capital = initial_capital

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_equity_curve(equity_curve: EquityInput) -> pd.DataFrame:
        """Normalize the equity curve input into a timestamp-indexed DataFrame.

        Args:
            equity_curve: List of ``(timestamp, value)`` tuples, list of plain
                values (a synthetic daily index is created), or DataFrame.

        Returns:
            DataFrame with a single ``equity`` column indexed by datetime.

        Raises:
            ValueError: If the input format is not supported.
        """
        if isinstance(equity_curve, pd.DataFrame):
            frame = equity_curve.copy()
            if not isinstance(frame.index, pd.DatetimeIndex):
                if "timestamp" in frame.columns:
                    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
                    frame.set_index("timestamp", inplace=True)
                else:
                    frame.index = pd.to_datetime(frame.index)
            return frame

        if isinstance(equity_curve, list):
            if len(equity_curve) == 0:
                return pd.DataFrame(columns=["equity"])

            first = equity_curve[0]
            if isinstance(first, tuple):
                frame = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
                frame["timestamp"] = pd.to_datetime(frame["timestamp"])
                frame.set_index("timestamp", inplace=True)
                return frame

            index = pd.date_range(start=pd.Timestamp.today().normalize(), periods=len(equity_curve), freq="D")
            return pd.DataFrame({"equity": equity_curve}, index=index)

        raise ValueError(f"Unsupported equity curve format: {type(equity_curve)!r}")

    # ------------------------------------------------------------------
    # 1. Core return metrics
    # ------------------------------------------------------------------
    def total_return(self) -> float:
        """Total return over the period, in percent.

        Raises:
            ValueError: If the equity curve is empty.
        """
        if self.equity_curve.empty:
            raise ValueError("Equity curve is empty; run a backtest first.")
        return (self.equity_curve["equity"].iloc[-1] - self.initial_capital) / self.initial_capital * 100

    def annualized_return(self) -> float:
        """Compound annual growth rate implied by the total return, in percent."""
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        total_return = self.total_return() / 100
        return ((1 + total_return) ** (365 / days) - 1) * 100 if days > 0 else 0.0

    def volatility(self) -> float:
        """Annualized volatility of daily portfolio returns, in percent."""
        returns = self.equity_curve["equity"].pct_change().dropna()
        return returns.std() * np.sqrt(252) * 100

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Annualized Sharpe ratio of daily portfolio returns."""
        returns = self.equity_curve["equity"].pct_change().dropna()
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Annualized Sortino ratio (downside volatility only)."""
        returns = self.equity_curve["equity"].pct_change().dropna()
        excess_returns = returns - risk_free_rate / 252
        downside = returns[returns < risk_free_rate / 252]
        if len(downside) == 0 or downside.std() == 0:
            return float("inf")
        return excess_returns.mean() / downside.std() * np.sqrt(252)

    def max_drawdown(
        self,
        strategy_name: str | None = None,
        symbol: str | None = None,
        save_history: bool = False,
    ) -> float:
        """Deepest peak-to-trough drop of the equity curve, in currency units.

        Args:
            strategy_name: Strategy name (required when ``save_history``).
            symbol: Ticker symbol (required when ``save_history``).
            save_history: Persist the record via
                :func:`save_max_drawdown`.
        """
        roll_max = self.equity_curve["equity"].cummax()
        drawdown_series = self.equity_curve["equity"] - roll_max
        max_dd = float(drawdown_series.min())

        if save_history and strategy_name and symbol:
            save_max_drawdown(strategy_name, symbol, max_dd)
        return max_dd

    def max_drawdown_pct(self) -> float:
        """Deepest peak-to-trough drop of the equity curve, in percent."""
        equity = self.equity_curve["equity"]
        peak = equity.cummax()
        return float(((equity - peak) / peak).min() * 100)

    def calmar_ratio(self, min_drawdown: float = 1.0) -> float:
        """Annualized return divided by the absolute max drawdown (in percent).

        The denominator is floored at ``min_drawdown`` percent and the ratio is
        capped at 100 to keep outliers readable.
        """
        annual_return = self.annualized_return()
        adjusted_drawdown = max(abs(self.max_drawdown_pct()), min_drawdown)
        return min(annual_return / adjusted_drawdown, 100.0)

    def recovery_factor(self) -> float:
        """Net profit divided by the absolute max drawdown."""
        max_dd = abs(self.max_drawdown())
        total_pnl = self.equity_curve["equity"].iloc[-1] - self.initial_capital
        return total_pnl / max_dd if max_dd > 0 else float("inf")

    # ------------------------------------------------------------------
    # 2. Trade statistics
    # ------------------------------------------------------------------
    def _closed_pnls(self) -> list[float]:
        """Return the PnL of every closed trade."""
        return [t["pnl"] for t in self.trades if "pnl" in t]

    def win_rate(self) -> float:
        """Share of winning closed trades, in percent."""
        wins = [pnl for pnl in self._closed_pnls() if pnl > 0]
        closes = [t for t in self.trades if t["type"] == "CLOSE"]
        return len(wins) / len(closes) * 100 if closes else 0.0

    def profit_factor(self) -> float:
        """Gross profit divided by gross loss (> 1 is profitable)."""
        gains = sum(pnl for pnl in self._closed_pnls() if pnl > 0)
        losses = sum(abs(pnl) for pnl in self._closed_pnls() if pnl < 0)
        return gains / losses if losses > 0 else float("inf")

    def average_win_loss(self) -> dict[str, float]:
        """Average win, average loss and their ratio."""
        wins = [pnl for pnl in self._closed_pnls() if pnl > 0]
        losses = [pnl for pnl in self._closed_pnls() if pnl < 0]
        return {
            "avg_win": float(np.mean(wins)) if wins else 0.0,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "win_loss_ratio": abs(np.mean(wins) / np.mean(losses)) if wins and losses else 0.0,
        }

    def payoff_ratio(self) -> float:
        """Average win divided by the absolute average loss."""
        stats = self.average_win_loss()
        return stats["avg_win"] / abs(stats["avg_loss"]) if stats["avg_loss"] else float("inf")

    def best_worst_trades(self) -> dict[str, float]:
        """Best and worst closed-trade PnL, in currency units."""
        pnls = self._closed_pnls()
        return {
            "best": float(max(pnls)) if pnls else 0.0,
            "worst": float(min(pnls)) if pnls else 0.0,
        }

    def average_trade_duration(self) -> float:
        """Average holding period of closed trades, in days."""
        durations: list[float] = []
        for i in range(1, len(self.trades)):
            if self.trades[i]["type"] == "CLOSE" and self.trades[i - 1]["type"] in ("BUY", "SELL"):
                durations.append((self.trades[i]["timestamp"] - self.trades[i - 1]["timestamp"]).days)
        return float(np.mean(durations)) if durations else 0.0

    def consecutive_wins_losses(self) -> dict[str, int]:
        """Longest winning and losing streaks across closed trades."""
        max_wins = max_losses = current_streak = 0
        last_sign = 0
        for trade in self.trades:
            if "pnl" not in trade:
                continue
            if trade["pnl"] > 0:
                current_streak = current_streak + 1 if last_sign == 1 else 1
                max_wins = max(max_wins, current_streak)
                last_sign = 1
            elif trade["pnl"] < 0:
                current_streak = current_streak + 1 if last_sign == -1 else 1
                max_losses = max(max_losses, current_streak)
                last_sign = -1
        return {"max_wins": max_wins, "max_losses": max_losses}

    # ------------------------------------------------------------------
    # 3. Calendar analytics
    # ------------------------------------------------------------------
    def monthly_returns(self) -> pd.Series:
        """Month-end to month-end portfolio returns, in percent."""
        monthly = self.equity_curve.resample("ME").last()
        return monthly["equity"].pct_change().dropna() * 100

    def positive_months_pct(self) -> float:
        """Share of months with a positive return, in percent."""
        monthly = self.monthly_returns()
        return float((monthly > 0).mean() * 100) if len(monthly) else 0.0

    # ------------------------------------------------------------------
    # 4. Risk measures
    # ------------------------------------------------------------------
    def risk_of_ruin(self, risk_per_trade: float = 1.0) -> float:
        """Long-term probability of ruin (gambler's ruin approximation).

        Args:
            risk_per_trade: Fraction of capital risked per trade
                (``0.01`` = 1%).

        Returns:
            Ruin probability in the ``[0, 1]`` range.
        """
        win_rate = self.win_rate() / 100
        payoff = self.payoff_ratio()

        if payoff == 0 or payoff == float("inf") or win_rate <= 0 or win_rate >= 1:
            return 0.0

        edge = win_rate - (1 - win_rate) / payoff
        if edge <= 0:
            return 1.0

        return min(math.exp(-2 * edge / risk_per_trade), 1.0)

    def value_at_risk_daily(self, confidence_level: float = 0.95) -> float:
        """Historical daily VaR, in currency units (negative = loss)."""
        returns = self.equity_curve["equity"].pct_change().dropna()
        if returns.empty:
            return 0.0
        var_percent = np.percentile(returns, (1 - confidence_level) * 100)
        return min(var_percent * self.equity_curve["equity"].iloc[-1], 0.0)

    def value_at_risk_monthly(self, confidence_level: float = 0.95) -> float:
        """Monthly VaR obtained by scaling the daily VaR by sqrt(21)."""
        returns = self.equity_curve["equity"].pct_change().dropna()
        if returns.empty:
            return 0.0
        daily_var = np.percentile(returns, (1 - confidence_level) * 100)
        monthly_var = daily_var * np.sqrt(21)
        return min(monthly_var * self.equity_curve["equity"].iloc[-1], 0.0)

    # ------------------------------------------------------------------
    # 5. Aggregated summary and plots
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, float]:
        """Return every metric as a flat dictionary (rounded to 4 decimals).

        Returns:
            Mapping of metric name to value; names carry a ``_pct`` suffix
            when expressed in percent.

        Raises:
            ValueError: If the equity curve is empty.
        """
        if self.equity_curve.empty:
            raise ValueError("Equity curve is empty; run a backtest first.")
        best_worst = self.best_worst_trades()
        streaks = self.consecutive_wins_losses()
        return {
            "total_return_pct": round(self.total_return(), 4),
            "annualized_return_pct": round(self.annualized_return(), 4),
            "volatility_pct": round(self.volatility(), 4),
            "sharpe_ratio": round(self.sharpe_ratio(), 4),
            "sortino_ratio": round(self.sortino_ratio(), 4),
            "max_drawdown": round(self.max_drawdown(), 4),
            "max_drawdown_pct": round(self.max_drawdown_pct(), 4),
            "calmar_ratio": round(self.calmar_ratio(), 4),
            "recovery_factor": round(self.recovery_factor(), 4),
            "win_rate_pct": round(self.win_rate(), 4),
            "profit_factor": round(self.profit_factor(), 4),
            "payoff_ratio": round(self.payoff_ratio(), 4),
            "best_trade": round(best_worst["best"], 4),
            "worst_trade": round(best_worst["worst"], 4),
            "max_consecutive_wins": streaks["max_wins"],
            "max_consecutive_losses": streaks["max_losses"],
            "avg_trade_duration_days": round(self.average_trade_duration(), 4),
            "positive_months_pct": round(self.positive_months_pct(), 4),
            "risk_of_ruin_pct": round(self.risk_of_ruin() * 100, 4),
            "var_daily_95": round(self.value_at_risk_daily(), 4),
            "var_monthly_95": round(self.value_at_risk_monthly(), 4),
            "n_trades": float(len(self.trades)),
        }

    def plot_equity_curve(self, title: str = "Equity curve", save_path: str | Path | None = None) -> None:
        """Plot the equity curve and the drawdown below it.

        Args:
            title: Title of the top panel.
            save_path: Save the figure to this path instead of showing it.
        """
        import matplotlib.pyplot as plt

        if self.equity_curve.empty:
            logger.warning("Equity curve is empty; nothing to plot.")
            return

        equity = self.equity_curve["equity"]

        plt.figure(figsize=(12, 7))

        plt.subplot(2, 1, 1)
        plt.plot(equity.index, equity, label="Equity", color="#1f77b4")
        plt.title(title)
        plt.ylabel("Portfolio value")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 1, 2)
        peak = equity.cummax()
        drawdown = (equity - peak) / peak * 100
        plt.plot(drawdown.index, drawdown, color="#d62728", label="Drawdown (%)")
        plt.fill_between(drawdown.index, drawdown, color="#d62728", alpha=0.3)
        plt.title("Drawdown (%)")
        plt.ylabel("Drawdown (%)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Equity curve plot saved to %s", save_path)
            plt.close()
        else:
            plt.show()

    def plot_monthly_returns_heatmap(self, save_path: str | Path | None = None) -> None:
        """Plot a year x month heatmap of portfolio returns (in percent)."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        monthly = self.monthly_returns()
        if monthly.empty:
            logger.warning("Not enough data for a monthly heatmap.")
            return

        data = pd.DataFrame(
            {
                "Year": monthly.index.year,
                "Month": monthly.index.month_name(),
                "Return": monthly.values,
            }
        ).pivot(index="Year", columns="Month", values="Return")

        month_order = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        data = data.reindex(columns=month_order)

        plt.figure(figsize=(12, 6))
        sns.heatmap(data, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar_kws={"label": "% Return"})
        plt.title("Monthly returns (%)")

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Monthly heatmap saved to %s", save_path)
            plt.close()
        else:
            plt.show()

    def plot_trades_distribution(self, save_path: str | Path | None = None) -> None:
        """Plot the histogram of closed-trade PnL."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        pnls = self._closed_pnls()
        if not pnls:
            logger.warning("No closed trades; nothing to plot.")
            return

        plt.figure(figsize=(10, 5))
        sns.histplot(pnls, kde=True, bins=30)
        plt.axvline(x=0, color="red", linestyle="--")
        plt.title("Distribution of trade PnL")
        plt.xlabel("Profit / Loss")
        plt.ylabel("Frequency")

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Trade distribution plot saved to %s", save_path)
            plt.close()
        else:
            plt.show()
