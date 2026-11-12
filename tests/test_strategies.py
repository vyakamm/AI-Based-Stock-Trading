"""Tests for strategies and the strategy registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dl_trading.strategies.base import (
    KNNStrategy,
    LSTMStrategy,
    MovingAverageCrossoverStrategy,
    RandomForestStrategy,
    RSIStrategy,
    StrategyType,
    SVMStrategy,
    TradingSignal,
    XGBoostStrategy,
)
from dl_trading.strategies.registry import (
    STRATEGY_REGISTRY,
    list_strategies,
    load_class_from_string,
)


def make_window(closes, opens=None):
    """Build a minimal OHLC window DataFrame from close prices."""
    closes = np.asarray(closes, dtype=float)
    opens = np.asarray(opens, dtype=float) if opens is not None else closes.copy()
    return pd.DataFrame({"Open": opens, "High": closes * 1.01, "Low": closes * 0.99, "Adj.Close": closes})


# ---------------------------------------------------------------------------
# Moving average crossover
# ---------------------------------------------------------------------------
def test_ma_crossover_bullish():
    strategy = MovingAverageCrossoverStrategy({"short_window": 3, "long_window": 5}, past_history=5)
    # Strong uptrend: short SMA > long SMA.
    window = make_window([100, 101, 102, 103, 104, 105])
    assert strategy.generate_signal(window) == TradingSignal.BUY


def test_ma_crossover_bearish():
    strategy = MovingAverageCrossoverStrategy({"short_window": 3, "long_window": 5}, past_history=5)
    window = make_window([105, 104, 103, 102, 101, 100])
    assert strategy.generate_signal(window) == TradingSignal.SELL


def test_ma_crossover_warmup_returns_hold():
    strategy = MovingAverageCrossoverStrategy({"short_window": 3, "long_window": 10}, past_history=5)
    window = make_window([100, 101, 102, 103, 104])  # fewer rows than long_window
    assert strategy.generate_signal(window) == TradingSignal.HOLD


# ---------------------------------------------------------------------------
# RSI (newly implemented strategy)
# ---------------------------------------------------------------------------
def test_rsi_all_gains_is_100():
    series = pd.Series(np.arange(1.0, 41.0))
    assert RSIStrategy.rsi(series, window=14) == 100.0


def test_rsi_all_losses_is_zero():
    series = pd.Series(np.arange(40.0, 0.0, -1.0))
    assert RSIStrategy.rsi(series, window=14) == 0.0


def test_rsi_strategy_signals():
    strategy = RSIStrategy({"window": 14, "overbought": 70, "oversold": 30}, past_history=15)

    rising = make_window(np.linspace(100, 160, 30))  # RSI = 100 -> overbought
    assert strategy.generate_signal(rising) == TradingSignal.SELL

    falling = make_window(np.linspace(160, 100, 30))  # RSI = 0 -> oversold
    assert strategy.generate_signal(falling) == TradingSignal.BUY

    flat = make_window(np.full(30, 100.0))  # RSI undefined-ish -> HOLD
    assert strategy.generate_signal(flat) == TradingSignal.HOLD


# ---------------------------------------------------------------------------
# ML strategies
# ---------------------------------------------------------------------------
def test_ml_strategy_buy_on_positive_prediction(dummy_ml_model):
    dummy_ml_model.value = 0.5
    strategy = RandomForestStrategy(dummy_ml_model, past_history=4)
    window = make_window([100, 101, 102, 103])
    assert strategy.generate_signal(window) == TradingSignal.BUY

    # The feature vector passed to the model must be (1, 2 * past_history).
    assert dummy_ml_model.last_input.shape == (1, 8)
    assert dummy_ml_model.last_input[0, 0] == pytest.approx(100.0 - 100.0)  # open - adj close (day 0)


def test_ml_strategy_sell_on_negative_prediction(dummy_ml_model):
    dummy_ml_model.value = -2.0
    strategy = SVMStrategy(dummy_ml_model, past_history=4)
    window = make_window([100, 101, 102, 103])
    assert strategy.generate_signal(window) == TradingSignal.SELL


def test_ml_strategy_hold_when_window_too_short(dummy_ml_model):
    strategy = XGBoostStrategy(dummy_ml_model, past_history=6)
    window = make_window([100, 101, 102])
    assert strategy.generate_signal(window) == TradingSignal.HOLD


def test_ml_strategy_never_crashes(dummy_ml_model):
    dummy_ml_model.fail = True
    strategy = KNNStrategy(dummy_ml_model, past_history=4)
    window = make_window([100, 101, 102, 103])
    assert strategy.generate_signal(window) == TradingSignal.HOLD


# ---------------------------------------------------------------------------
# LSTM strategy
# ---------------------------------------------------------------------------
def test_lstm_strategy_input_shape(dummy_lstm_model):
    dummy_lstm_model.value = 0.1
    strategy = LSTMStrategy(dummy_lstm_model, past_history=7)
    window = make_window(np.arange(107.0, 114.0))
    assert strategy.generate_signal(window) == TradingSignal.BUY
    assert dummy_lstm_model.last_shape == (2, 7)


def test_lstm_strategy_sell(dummy_lstm_model):
    dummy_lstm_model.value = -0.3
    strategy = LSTMStrategy(dummy_lstm_model, past_history=5)
    window = make_window(np.arange(100.0, 105.0))
    assert strategy.generate_signal(window) == TradingSignal.SELL


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class FakeTrial:
    """Deterministic Optuna trial double."""

    def suggest_int(self, name, low, high, **kwargs):
        return low

    def suggest_float(self, name, low, high, **kwargs):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


def test_registry_entries_resolve():
    for name, meta in STRATEGY_REGISTRY.items():
        klass = load_class_from_string(meta["class"])
        assert hasattr(klass, "generate_signal"), name


def test_registry_params_samplers():
    for name, meta in STRATEGY_REGISTRY.items():
        params = meta["suggest_params"](FakeTrial())
        assert "past_history" in params, name
        assert params["past_history"] >= 1


def test_list_strategies_sorted():
    assert list_strategies() == sorted(list_strategies())
    assert "lstm" in list_strategies()
    assert "rsi" in list_strategies()


def test_strategy_types():
    assert STRATEGY_REGISTRY["lstm"]["type"] == StrategyType.DEEP_LEARNING
    assert STRATEGY_REGISTRY["randomforest"]["type"] == StrategyType.MACHINE_LEARNING
    assert STRATEGY_REGISTRY["ma"]["type"] == StrategyType.TECHNICAL_INDICATOR
