"""Tests for the backtester and the performance analyzer."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import make_ohlcv
from dl_trading.backtesting.backtester import (
    Backtester,
    BacktesterParameters,
    PositionState,
)
from dl_trading.backtesting.comparison import compare_strategy_vs_buy_and_hold
from dl_trading.backtesting.performance import PerformanceAnalyzer
from dl_trading.strategies.base import Strategy, TradingSignal


class ScriptedStrategy(Strategy):
    """Emit a fixed sequence of signals, then HOLD forever."""

    def __init__(self, signals, past_history=2):
        super().__init__(strategy_type=None, params={}, past_history=past_history)
        self._signals = list(signals)

    def generate_signal(self, data_window):
        return self._signals.pop(0) if self._signals else TradingSignal.HOLD


class AlwaysBuyStrategy(Strategy):
    def __init__(self, past_history=2):
        super().__init__(strategy_type=None, params={}, past_history=past_history)

    def generate_signal(self, data_window):
        return TradingSignal.BUY


# ---------------------------------------------------------------------------
# BacktesterParameters
# ---------------------------------------------------------------------------
def test_parameters_validation():
    with pytest.raises(ValueError, match="stop_loss"):
        BacktesterParameters(stop_loss=0.0)
    with pytest.raises(ValueError, match="initial_capital"):
        BacktesterParameters(initial_capital=-1)
    with pytest.raises(ValueError, match="trade_units"):
        BacktesterParameters(trade_units=0)


def test_transaction_cost_minimum():
    params = BacktesterParameters(transaction_cost=0.001, min_transaction_cost=10.0, trade_units=5)
    backtester = Backtester(params)
    assert backtester.calculate_transaction_cost(5) == 10.0  # min fee applies

    params2 = BacktesterParameters(transaction_cost=1.0, min_transaction_cost=0.35, trade_units=50)
    assert Backtester(params2).calculate_transaction_cost(50) == 50.0  # proportional


# ---------------------------------------------------------------------------
# Trade execution
# ---------------------------------------------------------------------------
def test_open_and_close_long():
    params = BacktesterParameters(
        initial_capital=10_000, trade_units=10, transaction_cost=0.0, min_transaction_cost=0.0
    )
    bt = Backtester(params)
    ts = pd.Timestamp("2024-01-02")

    bt.execute_trade(TradingSignal.BUY, current_price=100.0, timestamp=ts)
    assert bt.position == PositionState.LONG
    assert bt.entry_price == 100.0
    assert bt.cash == 10_000 - 100 * 10

    bt.execute_trade(TradingSignal.SELL, current_price=110.0, timestamp=ts)  # reversal closes
    assert bt.position == PositionState.FLAT
    assert bt.cash == 10_000 - 100 * 10 + 110 * 10

    closed = [t for t in bt.trades if t["type"] == "CLOSE"]
    assert len(closed) == 1
    assert closed[0]["pnl"] == pytest.approx(100.0)


def test_short_pnl():
    params = BacktesterParameters(
        initial_capital=10_000, trade_units=10, transaction_cost=0.0, min_transaction_cost=0.0
    )
    bt = Backtester(params)
    ts = pd.Timestamp("2024-01-02")

    bt.execute_trade(TradingSignal.SELL, current_price=100.0, timestamp=ts)
    assert bt.position == PositionState.SHORT
    bt.execute_trade(TradingSignal.BUY, current_price=90.0, timestamp=ts)
    closed = [t for t in bt.trades if t["type"] == "CLOSE"]
    assert closed[0]["pnl"] == pytest.approx(100.0)


def test_stop_loss_triggers_close():
    params = BacktesterParameters(initial_capital=10_000, trade_units=10, stop_loss=0.02)
    bt = Backtester(params)
    ts = pd.Timestamp("2024-01-02")

    bt.execute_trade(TradingSignal.BUY, current_price=100.0, timestamp=ts)
    assert bt.check_stop_conditions(current_price=99.0) is None
    assert bt.check_stop_conditions(current_price=97.9) == TradingSignal.CLOSE

    bt.execute_trade(TradingSignal.CLOSE, current_price=97.9, timestamp=ts)
    assert bt.position == PositionState.FLAT
    assert bt.check_stop_conditions(current_price=90.0) is None  # flat -> no stop


def test_run_backtest_produces_equity_and_trades():
    data = make_ohlcv(n=60, seed=7)
    strategy = AlwaysBuyStrategy(past_history=5)
    params = BacktesterParameters(initial_capital=10_000, trade_units=10, past_history=5)
    bt = Backtester(params, strategy_name="always_buy", symbol="TEST", enable_max_drawdown=False)

    results = bt.run_backtest(data, strategy)

    assert len(results.equity_curve) == len(data) - 5
    assert results.signals[0][1] == TradingSignal.BUY
    assert any(t["type"] == "BUY" for t in results.trades)
    assert results.metrics["n_trades"] >= 1


def test_run_backtest_requires_adj_close():
    data = pd.DataFrame({"Close": [1, 2, 3]})
    with pytest.raises(ValueError, match="Adj.Close"):
        Backtester(BacktesterParameters()).run_backtest(data, AlwaysBuyStrategy(past_history=1))


# ---------------------------------------------------------------------------
# PerformanceAnalyzer
# ---------------------------------------------------------------------------
def test_total_and_annualized_return():
    equity = pd.DataFrame(
        {"equity": [1000.0, 1050.0, 1100.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    analyzer = PerformanceAnalyzer([], equity, initial_capital=1000)
    assert analyzer.total_return() == pytest.approx(10.0)
    assert analyzer.annualized_return() > 0


def test_max_drawdown_pct():
    equity = pd.DataFrame(
        {"equity": [100.0, 120.0, 90.0, 110.0]},
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    analyzer = PerformanceAnalyzer([], equity, initial_capital=100)
    assert analyzer.max_drawdown_pct() == pytest.approx(-25.0)
    assert analyzer.max_drawdown() == pytest.approx(-30.0)  # monetary


def test_trade_statistics():
    ts = pd.Timestamp("2024-01-02")
    trades = [
        {"timestamp": ts, "type": "BUY", "price": 10, "units": 1},
        {"timestamp": ts, "type": "CLOSE", "price": 12, "units": 1, "pnl": 200.0},
        {"timestamp": ts, "type": "SELL", "price": 10, "units": 1},
        {"timestamp": ts, "type": "CLOSE", "price": 11, "units": 1, "pnl": -100.0},
    ]
    equity = pd.DataFrame({"equity": [1000, 1100]}, index=pd.date_range("2024-01-01", periods=2, freq="D"))
    analyzer = PerformanceAnalyzer(trades, equity, initial_capital=1000)

    assert analyzer.win_rate() == pytest.approx(50.0)
    assert analyzer.profit_factor() == pytest.approx(2.0)
    assert analyzer.payoff_ratio() == pytest.approx(2.0)
    assert analyzer.best_worst_trades() == {"best": 200.0, "worst": -100.0}
    assert analyzer.consecutive_wins_losses() == {"max_wins": 1, "max_losses": 1}


def test_var_is_a_loss_or_zero():
    equity = pd.DataFrame(
        {"equity": [100, 90, 105, 95, 100, 88, 102]},
        index=pd.date_range("2024-01-01", periods=7, freq="D"),
    )
    analyzer = PerformanceAnalyzer([], equity, initial_capital=100)
    assert analyzer.value_at_risk_daily() <= 0
    assert analyzer.value_at_risk_monthly() <= analyzer.value_at_risk_daily()


def test_summary_contains_all_metrics():
    data = make_ohlcv(n=80, seed=3)
    bt = Backtester(
        BacktesterParameters(initial_capital=10_000, trade_units=5, past_history=5), enable_max_drawdown=False
    )
    bt.run_backtest(data, AlwaysBuyStrategy(past_history=5))

    metrics = bt.performance().summary()
    expected_keys = {
        "total_return_pct",
        "annualized_return_pct",
        "volatility_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "max_drawdown_pct",
        "calmar_ratio",
        "recovery_factor",
        "win_rate_pct",
        "profit_factor",
        "payoff_ratio",
        "best_trade",
        "worst_trade",
        "max_consecutive_wins",
        "max_consecutive_losses",
        "avg_trade_duration_days",
        "positive_months_pct",
        "risk_of_ruin_pct",
        "var_daily_95",
        "var_monthly_95",
        "n_trades",
    }
    assert expected_keys.issubset(metrics.keys())


def test_unsupported_equity_format_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        PerformanceAnalyzer([], 42, initial_capital=100)


# ---------------------------------------------------------------------------
# Buy & Hold comparison
# ---------------------------------------------------------------------------
def test_comparison_dataframe():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    equity = [(ts, 1000 + 10 * i) for i, ts in enumerate(dates)]
    prices = pd.DataFrame({"Adj.Close": [100, 101, 102, 103, 104]}, index=dates)

    comparison = compare_strategy_vs_buy_and_hold(equity, prices, initial_capital=1000, trade_units=10)

    assert list(comparison.columns) == ["strategy_pnl", "buy_and_hold_pnl"]
    assert comparison["buy_and_hold_pnl"].iloc[-1] == pytest.approx(104 * 10 - 100 * 10)
    assert comparison["strategy_pnl"].iloc[-1] == pytest.approx(1040 - 1000)
