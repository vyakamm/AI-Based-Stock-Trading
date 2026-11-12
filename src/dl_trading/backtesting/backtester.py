"""Event-driven backtester with risk management.

The :class:`Backtester` replays a strategy over historical data day by day:

1. the strategy observes the ``past_history`` days preceding day ``i`` and
   emits a :class:`~dl_trading.strategies.base.TradingSignal`,
2. open positions are checked against the stop loss first (a ``CLOSE`` signal
   is synthesized when the adverse move exceeds the threshold),
3. the resulting signal is executed at the ``Adj.Close`` price of day ``i``
   with transaction costs,
4. the portfolio value is marked to market and appended to the equity curve.

Accounting model: a fixed number of ``trade_units`` is traded per position;
going long debits ``price * units + cost`` from the cash balance, going short
credits it. Unrealized PnL is included in the reported portfolio value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from dl_trading.backtesting.performance import PerformanceAnalyzer
from dl_trading.strategies.base import Strategy, TradingSignal

logger = logging.getLogger(__name__)


class PositionState(Enum):
    """Current exposure of the portfolio."""

    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class BacktesterParameters:
    """Configuration of a backtest run.

    Attributes:
        stop_loss: Relative adverse move (e.g. ``0.02`` = -2%) that closes a
            position.
        initial_capital: Starting cash balance.
        past_history: Number of past days the strategy needs (informational;
            the strategy object carries the authoritative value).
        transaction_cost: Cost per unit traded (e.g. ``0.0035``).
        min_transaction_cost: Minimum cost per transaction (e.g. ``0.35``).
        trade_units: Number of units traded for every position.
    """

    stop_loss: float = 0.02
    initial_capital: float = 15_000.0
    past_history: int = 60
    transaction_cost: float = 0.0035
    min_transaction_cost: float = 0.35
    trade_units: int = 50

    def __post_init__(self) -> None:
        if self.stop_loss <= 0:
            raise ValueError(f"stop_loss must be positive, got {self.stop_loss}")
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")
        if self.trade_units <= 0:
            raise ValueError(f"trade_units must be positive, got {self.trade_units}")


@dataclass
class BacktestResults:
    """Structured outcome of a backtest run.

    Attributes:
        metrics: Performance metrics produced by ``PerformanceAnalyzer.summary``.
        trades: Raw list of executed trades.
        equity_curve: DataFrame indexed by timestamp with an ``equity`` column.
        signals: List of ``(timestamp, TradingSignal)`` emitted by the strategy.
        total_transaction_costs: Sum of all transaction costs paid.
    """

    metrics: dict[str, float]
    trades: list[dict]
    equity_curve: pd.DataFrame
    signals: list[Any]
    total_transaction_costs: float


class Backtester:
    """Simulate a strategy on historical data with stop-loss protection."""

    def __init__(
        self,
        parameters: BacktesterParameters,
        strategy_name: str = "unnamed",
        symbol: str | None = None,
        enable_max_drawdown: bool = True,
        max_drawdown_dir: str = "outputs/max_drawdown",
    ) -> None:
        """Initialize the backtester.

        Args:
            parameters: Backtest configuration.
            strategy_name: Label used in reports and saved records.
            symbol: Ticker symbol, used in reports and saved records.
            enable_max_drawdown: Persist the running max drawdown record.
            max_drawdown_dir: Directory for max drawdown records.
        """
        self.params = parameters
        self.position = PositionState.FLAT
        self.entry_price = 0.0
        self.cash = parameters.initial_capital
        self.portfolio_value: list[float] = []
        self.trades: list[dict] = []
        self.signals: list[Any] = []
        self.equity_curve: list[Any] = []
        self.total_transaction_costs = 0.0
        self.symbol = symbol
        self.enable_max_drawdown = enable_max_drawdown
        self.max_drawdown_dir = max_drawdown_dir
        self.strategy_name = strategy_name

    # ------------------------------------------------------------------
    # Execution primitives
    # ------------------------------------------------------------------
    def calculate_transaction_cost(self, units: int) -> float:
        """Cost of trading ``units`` (percentage cost with a minimum fee)."""
        return max(abs(units * self.params.transaction_cost), self.params.min_transaction_cost)

    def execute_trade(self, signal: TradingSignal, current_price: float, timestamp: Any) -> None:
        """Execute one trading action at ``current_price``.

        Handles position closing (explicit CLOSE or signal reversal) and
        position opening. All state transitions update the cash balance and
        append to the trade log.

        Args:
            signal: Signal to act on.
            current_price: Execution price (``Adj.Close`` of the day).
            timestamp: Timestamp of the day.
        """
        units = self.params.trade_units

        # ----- Closing -----
        reversal = self.position != PositionState.FLAT and (
            (signal == TradingSignal.BUY and self.position == PositionState.SHORT)
            or (signal == TradingSignal.SELL and self.position == PositionState.LONG)
        )
        if signal == TradingSignal.CLOSE or reversal:
            if signal == TradingSignal.CLOSE:
                logger.info("Stop loss hit - closing position")
            elif signal == TradingSignal.BUY:
                logger.info("Closing short position")
            else:
                logger.info("Closing long position")

            cost = self.calculate_transaction_cost(units)
            if self.position == PositionState.LONG:
                self.cash += current_price * units - cost
                pnl = (current_price - self.entry_price) * units - cost
            else:  # SHORT
                self.cash -= current_price * units + cost
                pnl = (self.entry_price - current_price) * units - cost

            self.total_transaction_costs += cost
            self.position = PositionState.FLAT
            self.trades.append(
                {"timestamp": timestamp, "type": "CLOSE", "price": current_price, "units": units, "pnl": pnl}
            )
            return

        # ----- Opening -----
        if signal == TradingSignal.BUY and self.position == PositionState.FLAT:
            cost = self.calculate_transaction_cost(units)
            if self.cash >= current_price * units + cost:
                self.position = PositionState.LONG
                self.entry_price = current_price
                self.cash -= current_price * units + cost
                self.total_transaction_costs += cost
                self.trades.append(
                    {
                        "timestamp": timestamp,
                        "type": "BUY",
                        "price": current_price,
                        "units": units,
                        "cost": cost,
                    }
                )
                logger.info("Opening long position at %.2f", current_price)
            else:
                logger.warning("Insufficient funds to open long position (%.2f needed)", current_price * units + cost)

        elif signal == TradingSignal.SELL and self.position == PositionState.FLAT:
            # Short selling is assumed to be always available.
            cost = self.calculate_transaction_cost(units)
            self.position = PositionState.SHORT
            self.entry_price = current_price
            self.cash += current_price * units - cost
            self.total_transaction_costs += cost
            self.trades.append(
                {
                    "timestamp": timestamp,
                    "type": "SELL",
                    "price": current_price,
                    "units": units,
                    "cost": cost,
                }
            )
            logger.info("Opening short position at %.2f", current_price)

    def check_stop_conditions(self, current_price: float) -> TradingSignal | None:
        """Return CLOSE when the open position breaches the stop loss.

        Args:
            current_price: Current market price.

        Returns:
            ``TradingSignal.CLOSE`` when the adverse move from the entry price
            exceeds ``stop_loss``, otherwise ``None``.
        """
        if self.position == PositionState.FLAT or self.entry_price == 0:
            return None

        price_change = (current_price - self.entry_price) / self.entry_price

        if self.position == PositionState.LONG and price_change <= -self.params.stop_loss:
            return TradingSignal.CLOSE
        if self.position == PositionState.SHORT and price_change >= self.params.stop_loss:
            return TradingSignal.CLOSE
        return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_backtest(self, data: pd.DataFrame, strategy: Strategy) -> BacktestResults:
        """Replay ``strategy`` over ``data``.

        Args:
            data: OHLCV DataFrame with an ``Adj.Close`` column; the first
                ``strategy.past_history`` rows are used as warm-up only.
            strategy: A fitted strategy instance.

        Returns:
            A :class:`BacktestResults` with metrics, trades, equity curve and
            emitted signals.

        Raises:
            ValueError: If ``Adj.Close`` is missing from the data or if the
                warm-up window (``strategy.past_history``) is longer than the
                data itself.
        """
        if "Adj.Close" not in data.columns:
            raise ValueError("Column 'Adj.Close' is required to run a backtest.")
        if strategy.past_history >= len(data):
            raise ValueError(
                f"past_history ({strategy.past_history}) must be smaller than the "
                f"number of data rows ({len(data)}); not enough data to backtest."
            )

        for i in range(strategy.past_history, len(data)):
            current_data = data.iloc[i - strategy.past_history : i]
            current_price = data.iloc[i]["Adj.Close"]
            timestamp = data.index[i]

            signal = strategy.generate_signal(current_data)
            self.signals.append((timestamp, signal))

            stop_signal = self.check_stop_conditions(current_price)
            if stop_signal:
                self.execute_trade(stop_signal, current_price, timestamp)

            self.execute_trade(signal, current_price, timestamp)

            if self.position == PositionState.LONG:
                current_value = self.cash + current_price * self.params.trade_units
            elif self.position == PositionState.SHORT:
                current_value = self.cash - current_price * self.params.trade_units
            else:
                current_value = self.cash

            self.portfolio_value.append(current_value)
            self.equity_curve.append((timestamp, current_value))

        return self.results()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def results(self) -> BacktestResults:
        """Aggregate the run into a :class:`BacktestResults`."""
        analyzer = PerformanceAnalyzer(self.trades, self.equity_curve, self.params.initial_capital)
        return BacktestResults(
            metrics=analyzer.summary(),
            trades=self.trades,
            equity_curve=analyzer.equity_curve,
            signals=self.signals,
            total_transaction_costs=self.total_transaction_costs,
        )

    def performance(self) -> PerformanceAnalyzer:
        """Return a ``PerformanceAnalyzer`` over the current run."""
        return PerformanceAnalyzer(self.trades, self.equity_curve, self.params.initial_capital)

    def print_report(self) -> dict[str, float]:
        """Print a formatted performance report and return the metrics."""
        if not self.portfolio_value:
            logger.warning("No backtest has been run yet.")
            return {}

        metrics = self.performance().summary()
        initial_value = self.params.initial_capital
        final_value = self.portfolio_value[-1]

        line = "=" * 46
        print(line)
        print(f"{self.strategy_name.upper()} - BACKTEST REPORT" + (f" ({self.symbol})" if self.symbol else ""))
        print(line)
        print(f"Initial capital        : {initial_value:>12,.2f}")
        print(f"Final capital          : {final_value:>12,.2f}")
        print(f"Total return           : {metrics['total_return_pct']:>11.2f} %")
        print(f"Annualized return      : {metrics['annualized_return_pct']:>11.2f} %")
        print(f"Volatility (ann.)      : {metrics['volatility_pct']:>11.2f} %")
        print("-" * 46)
        print(f"Sharpe ratio           : {metrics['sharpe_ratio']:>11.2f}")
        print(f"Sortino ratio          : {metrics['sortino_ratio']:>11.2f}")
        print(f"Max drawdown           : {metrics['max_drawdown']:>11,.2f}")
        print(f"Max drawdown (%)       : {metrics['max_drawdown_pct']:>11.2f} %")
        print(f"Calmar ratio           : {metrics['calmar_ratio']:>11.2f}")
        print(f"Recovery factor        : {metrics['recovery_factor']:>11.2f}")
        print("-" * 46)
        print(f"Win rate               : {metrics['win_rate_pct']:>11.2f} %")
        print(f"Profit factor          : {metrics['profit_factor']:>11.2f}")
        print(f"Payoff ratio           : {metrics['payoff_ratio']:>11.2f}")
        print(f"Best / worst trade     : {metrics['best_trade']:>6,.0f} / {metrics['worst_trade']:,.0f}")
        print(f"Consec. wins / losses  : {metrics['max_consecutive_wins']:>4d} / {metrics['max_consecutive_losses']}")
        print(f"Avg trade duration     : {metrics['avg_trade_duration_days']:>11.2f} d")
        print(f"Transaction costs      : {self.total_transaction_costs:>11,.2f}")
        print("-" * 46)
        print(f"Positive months        : {metrics['positive_months_pct']:>11.2f} %")
        print(f"Risk of ruin           : {metrics['risk_of_ruin_pct']:>11.2f} %")
        print(f"VaR daily 95%          : {metrics['var_daily_95']:>11,.2f}")
        print(f"VaR monthly 95%        : {metrics['var_monthly_95']:>11,.2f}")
        print(f"Number of trades       : {metrics['n_trades']:>11.0f}")
        print(line)

        if self.enable_max_drawdown:
            self.performance().max_drawdown(
                strategy_name=self.strategy_name,
                symbol=self.symbol,
                save_history=True,
            )
        return metrics
